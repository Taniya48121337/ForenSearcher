"""分叉式轨迹生成（2026-10-11 试点，见 claude/轨迹生成方法重设计_溯源事实表与分叉合成_20261011.md）。

要训练的是什么（每处都按这个约束写）：GIA / HI 两个**逐决策**策略——输入是 harness 渲染给学生的那一份状态，
输出是一个 JSON 决定。所以分叉产出的每一个"候选决定"都必须：
- 在同一个原始状态上由教师给出（教师看到的就是学生那份渲染，bridge.teacher_request）；
- 写回 run 目录时仍是普通的 request_NNNN（payload + response），导出不用改就能拆成训练行；
- 选哪一支只能看事后结果（事实表），不能把事实表写进任何模型输入。

三块：
1. 关键状态：在一条已跑完的轨迹里挑"做法会显著影响结果"的决定点（拿到候选列表、工具失败、找到同源视频 / 拿到
   compare_media 比对结果（10-12 起代替 DETECT 比对结果）、交报告、FINISH）。
2. 分叉执行：前缀按原轨迹的教师输出逐条强制回放（cli_generator 的 FOREN_FORCED_PLAN），分叉点换成候选决定，之后教师照常续跑。
   检索走录制回放、INSPECT 走工具缓存，前缀可复现；每条强制输出都核对状态指纹，对不上就判这一支作废（FORK_DIVERGED）。
3. 事实表与价值：从案卷自动导出"真相原子"（原件来源、结论标签、时间区间、实体），按 run 的 Store 与最终答案程序化地算
   每支建立了哪些原子；反事实变体（原件被屏蔽）里依赖原件的原子改为"不可知"，要求答案克制。
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .cases import Case, _normalize_url

_EID = re.compile(r"\bE\d+\b")


# ---------------------------------------------------------------- 读 run

@dataclass
class Decision:
    sequence: int
    request_dir: Path
    role: str                      # HI | GIA
    workspace_id: str | None
    payload: dict[str, Any]
    raw: str
    output: dict[str, Any] | None
    forced: str | None = None      # prefix | branch | None

    @property
    def view(self) -> dict[str, Any]:
        return (self.payload.get("model_view") or self.payload.get("global_view") or {}) if isinstance(self.payload, dict) else {}

    @property
    def is_repair(self) -> bool:
        return isinstance(self.payload, dict) and ("validation_error" in self.payload or "previous_invalid_output" in self.payload)


def _parse_json(text: str) -> dict[str, Any] | None:
    from ..local_models import extract_json_object

    try:
        value = extract_json_object(text)
    except Exception:  # noqa: BLE001 —— 坏输出在分叉里只当作"不可用"
        return None
    return value if isinstance(value, dict) else None


def load_decisions(run_dir: str | Path) -> list[Decision]:
    out: list[Decision] = []
    for request_dir in sorted(Path(run_dir, "requests").glob("request_*")):
        try:
            request = json.loads((request_dir / "request.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        response = request_dir / "response.json"
        raw = response.read_text(encoding="utf-8") if response.is_file() else ""
        transport = {}
        try:
            transport = json.loads((request_dir / "transport.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        payload = request.get("payload") or {}
        workspace = payload.get("workspace_id")
        out.append(Decision(sequence=int(request_dir.name.split("_")[1]), request_dir=request_dir,
                            role="HI" if workspace else "GIA", workspace_id=workspace, payload=payload, raw=raw,
                            output=_parse_json(raw) if raw else None, forced=transport.get("forced")))
    return out


def fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """前缀回放的一致性指纹：角色 / 工作区、出现过的 E 编号集合、历史条数。不含时间预算之类会变的字段。"""

    view = (payload.get("model_view") or payload.get("global_view") or {}) if isinstance(payload, dict) else {}
    text = json.dumps(view.get("evidence_view") if "evidence_view" in view else view.get("current_materials"), ensure_ascii=False)
    history = view.get("history") if isinstance(view.get("history"), list) else payload.get("memory_history") or []
    return {"workspace_id": payload.get("workspace_id"), "evidence": sorted(set(_EID.findall(text or ""))),
            "history": len(history) if isinstance(history, list) else 0}


# ---------------------------------------------------------------- 关键状态

@dataclass
class KeyState:
    sequence: int
    role: str
    workspace_id: str | None
    kind: str                      # source_visible | tool_failure | dead_end | source_match | compare_result | detect_result | candidates | report | finish
    note: str = ""


def _observation(decision: Decision) -> dict[str, Any]:
    value = decision.view.get("latest_observation")
    return value if isinstance(value, dict) else {}


def _results(details: dict[str, Any]) -> list[dict[str, Any]]:
    value = details.get("results")
    return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def _material_failed(result: dict[str, Any]) -> bool:
    """候选的材料全没取到，且至少有一份明确失败（403、下载失败…）；not_requested 不算失败。"""

    statuses = [str(m.get("status")) for m in result.get("materials") or [] if isinstance(m, dict)]
    return bool(statuses) and not any(s in ("ready", "partial") for s in statuses) and any(s in ("failed", "error") for s in statuses)


def _any_ready(result: dict[str, Any]) -> bool:
    return any(str(m.get("status")) in ("ready", "partial") for m in result.get("materials") or [] if isinstance(m, dict))


def _members(case: Case | None) -> list[Any]:
    return [r for r in (case.required_evidence if case else ()) if r.url_pattern]


def classify(decision: Decision, case: Case | None = None) -> str | None:
    """这一决定是否落在关键状态上；repair 回合不算（那是格式修复，不是调查决策）。

    tp1 起：材料取不到（结果里 materials 全失败，tp0 漏判了这种）算 tool_failure；搜索零结果算 dead_end；
    给了案卷时，观察里出现了必需证据的候选链接、但材料没取到，算 source_visible（事后看最该分叉的地方：学生看得到原件线索，接下来怎么做决定成败）。"""

    if decision.is_repair or decision.output is None:
        return None
    if decision.role == "GIA":
        return "finish" if decision.output.get("decision") == "FINISH" else None
    if "status" in decision.output and "findings" in decision.output:
        return "report"
    obs = _observation(decision)
    action, details = obs.get("action"), obs.get("details") if isinstance(obs.get("details"), dict) else {}
    status, results = details.get("status"), _results(details)
    members = _members(case)
    if action == "SEARCH" and members and any(_material_failed(r) or not _any_ready(r) for r in results
                                              if any(m.matches(source_url=str(r.get("source_url") or ""), acquired_by=None) for m in members)):
        return "source_visible"            # 必需证据的链接出现了、但还没取到（403、只列了链接…）：接下来怎么做最关键
    if action in ("INSPECT", "DETECT", "SEARCH") and (status in ("failed", "error") or (results and all(_material_failed(r) for r in results))):
        return "tool_failure"
    if action == "SEARCH" and isinstance(details.get("results"), list) and not results:
        return "dead_end"
    if action == "DETECT" and status in ("ready", "partial"):
        return "detect_result"             # 10-12 以前的 run（DETECT 已并入 INSPECT / SEARCH）
    if action == "INSPECT" and details.get("operation") == "compare_media" and status in ("ready", "partial"):
        return "compare_result"
    if action == "SEARCH" and any(isinstance(r.get("content_match"), dict) and r["content_match"].get("status") in ("source", "partial")
                                  for r in results):
        return "source_match"              # SEARCH 结果里有与材料视频同源的视频：接下来是否做完整比对决定成败
    if action == "SEARCH" and len(results) >= 2:
        return "candidates"
    return None


_PRIORITY = {"source_visible": 0, "tool_failure": 1, "dead_end": 2, "source_match": 3, "compare_result": 3, "detect_result": 3,
             "candidates": 4, "finish": 5, "report": 6}


def key_states(run_dir: str | Path, limit: int = 2, case: Case | None = None, *, per_actor: int = 1) -> list[KeyState]:
    """每条轨迹挑最多 limit 个关键状态：先"原件线索可见"/工具失败/零结果/同源视频与比对结果/候选列表（调查走向在这里分叉），
    再 FINISH / 报告。同一角色（工作区或 GIA）最多 per_actor 个，尽量分散。"""

    found = []
    for d in load_decisions(run_dir):
        if d.forced:                       # 分叉支自身的前缀不再当分叉点
            continue
        kind = classify(d, case)
        if kind:
            found.append(KeyState(d.sequence, d.role, d.workspace_id, kind))
    found.sort(key=lambda k: (_PRIORITY[k.kind], k.sequence))
    chosen: list[KeyState] = []
    count: dict[str, int] = {}
    for k in found:
        actor = k.workspace_id or "GIA"
        if count.get(actor, 0) >= per_actor:
            continue
        chosen.append(k)
        count[actor] = count.get(actor, 0) + 1
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda k: k.sequence)


# ---------------------------------------------------------------- 候选决定（同一状态，教师给几种做法）

PROPOSAL_PREFACE = (
    "You are a policy in a separate executable investigation environment. "
    "Use only the supplied state and attached images. Do not use any Codex tools, shell, filesystem, network or external memory. "
    "Do not inspect this directory. Do not invent observations, evidence, or future results.\n"
    "For the current state below, give {k} DIFFERENT reasonable next decisions that a careful investigator might take. "
    "They must differ in substance (a different action, a different target or source, or a different conclusion/stop decision), "
    "not just in wording. Each one must be a complete decision object exactly as the role's instructions require, "
    "self-contained, with its own rationale that does not mention the other options. "
    'Return exactly one JSON object: {{"candidates": [decision_1, ..., decision_{k}]}}.\n\n'
)


def normalized_action(output: dict[str, Any] | None) -> str:
    """比较两个决定是否"实质相同"：去掉 rationale 与 focus，只比动作与参数（或报告状态 / GIA 决定类型与任务数）。"""

    if not isinstance(output, dict):
        return ""
    value = {k: v for k, v in output.items() if k not in ("rationale", "focus_evidence_ids")}
    if "findings" in value:             # 报告：状态 + 引用集合
        cited = sorted({e for f in value.get("findings") or [] if isinstance(f, dict) for e in f.get("evidence_ids") or []})
        return json.dumps({"status": value.get("status"), "cited": cited})
    if value.get("decision") == "FINISH":
        return json.dumps({"decision": "FINISH"})
    if value.get("decision") == "INVESTIGATE":
        return json.dumps({"decision": "INVESTIGATE", "tasks": len(value.get("tasks") or [])})
    arguments = value.get("arguments")
    if isinstance(arguments, dict):            # 默认值不算差别：READ_RECORD offset=0 与不写 offset 是同一个动作（tp0 videosham s8）
        value["arguments"] = {k: v for k, v in arguments.items() if not (k == "offset" and v in (0, None, "0")) and v not in ({}, None)}
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def valid_for_role(output: dict[str, Any], role: str) -> bool:
    from ..contracts import parse_gia_decision, parse_hi_decision
    from ..local_models import strip_rationale

    try:
        clean = strip_rationale(output) if callable(strip_rationale) else output
        (parse_hi_decision if role == "HI" else parse_gia_decision)(clean)
    except Exception:  # noqa: BLE001
        return False
    return True


def proposal_prompt(request: dict[str, Any], k: int) -> tuple[str, list[str]]:
    """候选决定的提示：教师看到的状态与当时学生那份相同（bridge.teacher_request 按记录的档位渲染），附图相同。"""

    from .bridge import teacher_request

    shown, _info = teacher_request(request, None)
    return PROPOSAL_PREFACE.format(k=k) + json.dumps(shown, ensure_ascii=False), []


def propose(decision: Decision, k: int, *, codex: str = "codex", model: str | None = None, out_dir: Path) -> list[dict[str, Any]]:
    """调一次 Codex，拿 k 个候选；过滤掉不合法的和与原决定实质相同的。结果与原始回复都落盘（可复查）。"""

    request = json.loads((decision.request_dir / "request.json").read_text(encoding="utf-8"))
    prompt, _ = proposal_prompt(request, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / "proposal_response.json"
    if not result.is_file():
        images = sorted((decision.request_dir / "actor").glob("image_*.png"))
        command = [codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never",
                   "--json", "-C", str(out_dir), "-o", str(result)]
        if model:
            command += ["-m", model]
        if os.environ.get("FOREN_CODEX_EFFORT"):
            command += ["-c", f'model_reasoning_effort="{os.environ["FOREN_CODEX_EFFORT"]}"']
        for image in images:
            command += ["--image", str(image)]
        command += ["--", "-"]
        completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace",
                                   capture_output=True, timeout=900, check=False)
        (out_dir / "proposal_events.jsonl").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode or not result.is_file():
            raise RuntimeError(f"proposal failed for {decision.request_dir}: {completed.stderr[-300:]}")
    value = _parse_json(result.read_text(encoding="utf-8")) or {}
    original = normalized_action(decision.output)
    kept, seen = [], {original}
    for cand in value.get("candidates") or []:
        if not isinstance(cand, dict) or not valid_for_role(cand, decision.role):
            continue
        key = normalized_action(cand)
        if key in seen:
            continue
        seen.add(key)
        kept.append(cand)
    (out_dir / "candidates.json").write_text(json.dumps({"original": decision.output, "kept": kept,
                                                         "dropped": len(value.get("candidates") or []) - len(kept)},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    return kept


def forced_plan(run_dir: str | Path, sequence: int, alternative: dict[str, Any] | None) -> dict[str, Any]:
    """前缀强制回放计划：sequence 之前逐条用原教师输出，sequence 处换成候选（None = 原样，用于复现检查）。
    每条带 actor（工作区编号或 GIA）与状态指纹，cli_generator 按角色 + 指纹对号（不按全局序号，见那里的说明）。"""

    outputs = []
    for d in load_decisions(run_dir):
        if d.sequence > sequence:
            break
        kind = "prefix" if d.sequence < sequence else "branch"
        raw = d.raw if (kind == "prefix" or alternative is None) else json.dumps(alternative, ensure_ascii=False)
        outputs.append({"sequence": d.sequence, "actor": d.workspace_id or "GIA", "raw": raw, "kind": kind,
                        "expect": fingerprint(d.payload)})
    return {"source_run": str(Path(run_dir).name), "fork_sequence": sequence, "outputs": outputs}


# ---------------------------------------------------------------- 事实表（从案卷自动导出）与价值

@dataclass
class Atom:
    kind: str                      # EVIDENCE | VERDICT | INTERVAL | ENTITY
    key: str
    detail: dict[str, Any] = field(default_factory=dict)
    knowable: bool = True          # 反事实里依赖的来源全被屏蔽 / 不让检索时为 False
    required: bool = True          # 计入价值；必需证据组里全是 must_cite=False 的只记录不计分


def _hidden(pattern: str, hidden: set[str]) -> bool:
    key = _normalize_url(pattern) if pattern else ""
    return bool(key) and any(h and (h == key or h.rstrip("*") == key.rstrip("*")) for h in hidden)


def fact_table(case: Case, *, hidden_keys: Iterable[str] = (), no_retrieval: bool = False) -> list[Atom]:
    """案卷 → 真相原子。必需证据按 required_evidence_rule 分组（any_one_of 任取其一，tp0 当成各自必需，分数被压在 0.4 左右）；
    没有 url_pattern、靠工具产出的项（acquired_by，如 INSPECT/get_frames）按 Store 里证据的 acquired_by 判。"""

    hidden = {_normalize_url(h) for h in hidden_keys if h}
    by_key = {r.key: r for r in case.required_evidence}
    atoms: list[Atom] = []
    for mode, keys in case.evidence_groups():
        members = []
        for key in keys:
            r = by_key[key]
            unknown = no_retrieval and bool(r.url_pattern)
            members.append({"key": key, "url_pattern": r.url_pattern, "acquired_by": r.acquired_by, "must_cite": r.must_cite,
                            "description": r.description, "hidden": unknown or _hidden(r.url_pattern or "", hidden)})
        knowable = any(not m["hidden"] for m in members)
        atoms.append(Atom("EVIDENCE", "|".join(keys), {"mode": mode, "members": members}, knowable,
                          required=any(m["must_cite"] for m in members)))
    for check in case.checks:
        p = check.params
        if check.kind == "label_expected":
            atoms.append(Atom("VERDICT", str(p["label"]), {"forms": [p["label"], *p.get("aliases", [])]}, not no_retrieval))
        elif check.kind == "interval_iou":
            atoms.append(Atom("INTERVAL", f"{p['start']}-{p['end']}", dict(p)))
        elif check.kind == "entity_in_evidence":
            atoms.append(Atom("ENTITY", str(p["text"]), {"forms": [p["text"], *p.get("statement_aliases", []), *p.get("aliases", [])]},
                              not no_retrieval))
    return atoms


def _store_rows(run_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Store 里出现过的候选 URL（进池），与每条证据（E 编号、acquired_by、来源 URL；来源取自 candidate_evidence，
    派生证据（抽帧等）没有候选就只有 acquired_by）。只读打开，run 跑一半也能读。"""

    db = run_dir / "store.sqlite3"
    if not db.is_file():
        return [], []
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        pool = [str(u) for r in connection.execute("SELECT source_url, final_url FROM candidates") for u in r if u]
        rows: dict[int, dict[str, Any]] = {}
        try:
            for r in connection.execute("SELECT id, acquired_by, label FROM evidence"):
                rows[int(r["id"])] = {"eid": f"E{r['id']}", "acquired_by": str(r["acquired_by"] or ""), "label": str(r["label"] or ""), "urls": []}
            for r in connection.execute("""SELECT ce.evidence_id AS eid, ce.material_status AS st, c.source_url AS url, c.final_url AS final
                                           FROM candidate_evidence ce JOIN candidates c ON c.id = ce.candidate_id"""):
                row = rows.get(int(r["eid"]))
                if row is not None and r["st"] in ("ready", "partial"):
                    row["urls"].extend(str(u) for u in (r["url"], r["final"]) if u)
        except sqlite3.Error:
            pass
    finally:
        connection.close()
    return pool, list(rows.values())


_ARCHIVE = re.compile(r"^(https?://)?(www\.)?web\.archive\.org/web/[^/]+/", re.I)


def _match(url: str, pattern: str) -> bool:
    """链接是否落在 url_pattern 上；Wayback 存档（web.archive.org/web/<时间>/原址）按原址判——
    tp1 averitec 有一支从 Wayback 取到了原文，旧判法不认，偏好对被判反了。"""

    if not pattern:
        return False
    want = _normalize_url(pattern)
    return any(fnmatch.fnmatch(_normalize_url(u), want) for u in {url, _ARCHIVE.sub("", url.strip())})


def _member_hits(member: dict[str, Any], row: dict[str, Any]) -> bool:
    if member["url_pattern"] and any(_match(u, member["url_pattern"]) for u in row["urls"]):
        return True
    return bool(member["acquired_by"]) and row["acquired_by"].startswith(member["acquired_by"])


def final_texts(run_dir: Path, workspace: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """最终答案与报告 Finding（从请求回复里取，和导出看到的一样）。workspace 给了就只取那个工作区的报告、不取答案。"""

    answer, findings = "", []
    for d in load_decisions(run_dir):
        if d.output is None or d.is_repair:
            continue
        if workspace is None and d.role == "GIA" and d.output.get("decision") == "FINISH":
            answer = str(d.output.get("answer") or "")
        if d.role == "HI" and "findings" in d.output and (workspace is None or d.workspace_id == workspace):
            findings.extend(f for f in d.output.get("findings") or [] if isinstance(f, dict))
    return answer, findings


def evaluate(run_dir: str | Path, atoms: list[Atom], *, scope: str = "answer", workspace: str | None = None,
             verdict_judgment: dict[str, Any] | None = None) -> dict[str, Any]:
    """每个原子建立到哪一步：进池 / 取得 / 被引用。

    scope：answer = 整条轨迹（引用看最终答案 + 全部报告）；workspace = 只看某个工作区的报告（分叉跑到该工作区交报告为止）；
    progress = 不看引用，只看取得（分叉只跑一步时用：同一前缀上各支多拿到了什么）。
    VERDICT：verdict_judgment（核验模型对答案的语义判定 matches / contradicts / absent）优先，没有就退回子串匹配。
    可知时 matches=1、absent=0.25（没下结论）、contradicts=0；不可知时（不让检索）absent=1，下了结论=0。
    leaks：被屏蔽的来源仍被取得（V1 屏蔽没生效），这一支的 V1 结论不可信。"""

    from .verify.lang import HEDGE, LIMITATION, intervals_in, label_hit

    run_dir = Path(run_dir)
    pool, rows = _store_rows(run_dir)
    answer, findings = final_texts(run_dir, workspace if scope == "workspace" else None)
    text = answer + " " + " ".join(str(f.get("statement") or "") for f in findings)
    cited = set(_EID.findall(answer)) | {e for f in findings for e in f.get("evidence_ids") or []}
    out_rows, score, total, leaks = [], 0.0, 0, []
    for atom in atoms:
        row: dict[str, Any] = {"kind": atom.kind, "key": atom.key, "knowable": atom.knowable, "required": atom.required}
        if atom.kind == "EVIDENCE":
            members = []
            for m in atom.detail["members"]:
                eids = [r["eid"] for r in rows if _member_hits(m, r)]
                state = {"key": m["key"], "hidden": m["hidden"],
                         "in_pool": bool(m["url_pattern"]) and any(_match(u, m["url_pattern"]) for u in pool),
                         "acquired": bool(eids), "cited": bool(set(eids) & cited)}
                if scope == "progress":
                    state["value"] = 0.5 * (state["in_pool"] or state["acquired"]) + 0.5 * state["acquired"]
                elif m["url_pattern"]:
                    state["value"] = 0.25 * state["in_pool"] + 0.25 * state["acquired"] + 0.5 * state["cited"]
                else:
                    state["value"] = 0.5 * state["acquired"] + 0.5 * state["cited"]
                if m["hidden"] and state["acquired"]:
                    leaks.append(m["key"])
                members.append(state)
            row["members"] = members
            live = [s["value"] for s in members if not s["hidden"]]
            if not (live and atom.knowable):
                row["value"] = None
            elif scope == "progress":            # 只跑一步时组内取平均：前缀已经拿到一份时 max 恒为 1，看不出这一步多拿了什么
                row["value"] = round(sum(live) / len(live), 3)
            else:
                row["value"] = max(live)
        elif atom.kind == "VERDICT":
            if scope != "answer":
                row["value"] = None
            else:
                if verdict_judgment and verdict_judgment.get("status") in ("matches", "contradicts", "absent"):
                    status, method = verdict_judgment["status"], "judge"
                else:
                    hit = any(label_hit(str(f), answer) for f in atom.detail["forms"])
                    status, method = ("matches" if hit else "absent"), "substring"
                hedged = bool(HEDGE.search(answer) or LIMITATION.search(answer))
                row.update({"status": status, "method": method, "hedged": hedged, "finished": bool(answer)})
                if not answer:
                    row["value"] = 0.0
                elif atom.knowable:
                    row["value"] = {"matches": 1.0, "absent": 0.25, "contradicts": 0.0}[status]
                else:
                    row["value"] = 1.0 if status == "absent" else 0.0
        elif atom.kind == "INTERVAL":
            if scope == "progress":
                row["value"] = None
            else:
                want = (float(atom.detail["start"]), float(atom.detail["end"]))
                best = max((_iou(want, g) for g in intervals_in(text)), default=0.0)
                row.update({"best_iou": round(best, 3)})
                row["value"] = float(best >= float(atom.detail.get("min_iou", 0.5)))
        elif atom.kind == "ENTITY":
            if scope == "progress":
                row["value"] = None
            else:
                hit = any(f and str(f).lower() in text.lower() for f in atom.detail["forms"])
                row["value"] = float(hit) if atom.knowable else None
        if row.get("value") is not None and atom.required:
            score += row["value"]
            total += 1
        out_rows.append(row)
    return {"atoms": out_rows, "value": round(score / total, 3) if total else None, "answer_chars": len(answer),
            "finished": bool(answer), "n_findings": len(findings), "n_evidence": len(rows), "leaks": leaks}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- 反事实变体

def _write_variant(path: Path, data: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / path.name
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def hidden_variant(case_path: str | Path, out_dir: str | Path) -> tuple[Path, list[str]]:
    """V1 屏蔽原件：把案卷所有必需证据的 url_pattern 原样（含 * 通配）加进检索护栏的 excluded_keys：搜不到、打不开。
    tp0 把末尾 * 去掉当前缀用，而护栏只认"整段路径前缀"，结果 5 个里 4 个没屏蔽住（timesnow、prefpolice 照样取到）；
    现在 KeyExclusion 认通配。案卷其余部分不动（family_key 相同，划分相同）。考的是"原件拿不到时能否绕路、且如实说没拿到"，
    不是"必须不下结论"——同一件事往往还有别家核查（tp0 averitec V1 就从 NewsMobile / Global Times 得出正确结论）。"""

    path = Path(case_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = [_normalize_url(str(item.get("url_pattern"))) for item in data.get("required_evidence") or [] if item.get("url_pattern")]
    policy = dict(data.get("source_policy") or {})
    if keys:
        policy["mode"] = "exclude_original"
        policy["excluded_keys"] = list(dict.fromkeys([*(policy.get("excluded_keys") or []), *keys]))
    data["source_policy"] = policy
    data["notes"] = (data.get("notes") or "") + " [分叉试点 V1：必需证据来源全部屏蔽，考察能否绕路、是否如实说明原件未取得]"
    return _write_variant(path, data, out_dir), keys


def no_retrieval_variant(case_path: str | Path, out_dir: str | Path) -> Path:
    """V0 不让检索：来源策略改成 provided_media（没有 SEARCH），只剩给定材料。结论多半无从确认，答案应当不下结论、说明缺什么。
    给定材料本身就能定论的题（如视频比对）不该用这个变体。"""

    path = Path(case_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_policy"] = {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None}
    data["notes"] = (data.get("notes") or "") + " [分叉试点 V0：不让检索，考察答案是否克制]"
    return _write_variant(path, data, out_dir)


# ---------------------------------------------------------------- 可见性（防特权泄漏）

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}|\d{3,}")
_STOP = {"the", "and", "for", "with", "from", "that", "this", "video", "original", "source", "footage", "news", "fact",
         "check", "about", "who", "what", "when", "where", "was", "were", "are", "official", "full", "clip",
         "www", "com", "http", "https", "html", "org", "net", "watch", "status", "article", "story", "world"}


def privileged_terms(case: Case) -> set[str]:
    """只有总控知道、学生不该凭空想到的词：必需证据链接的站点名（域名各段、账号名）、链接里的长编号 / 视频号、
    source_hint、案卷断言里的实体写法；去掉任务文字里本来就有的。不收链接路径里的普通词和案卷说明文字
    （"satire""archive""police" 这类是正常的调查思路，不算泄漏）。"""

    words: set[str] = set()
    for r in case.required_evidence:
        pattern = (r.url_pattern or "").lower()
        host, _, path = pattern.partition("/")
        for label in host.split("."):                         # 站点名 / 账号名（"*prefpolice/status/…" 的第一段）；
            label = label.strip("*")                         # 中间带 * 的保留成通配（foolish*umor 同时认两种拼写）
            if len(label.replace("*", "")) >= 5 and re.fullmatch(r"[a-z0-9\-*]+", label):   # ≥5 字符，免得 nine.com.au 把 nine 也算上
                words.add(label)
        words.update(re.findall(r"\d{6,}", path))
        words.update(re.findall(r"v=([a-z0-9_\-]{8,})", path))
        if r.source_hint:
            words.update(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", r.source_hint))
    for check in case.checks:
        if check.kind == "entity_in_evidence":
            forms = [check.params.get("text"), *check.params.get("aliases", []), *check.params.get("statement_aliases", [])]
            words.update(w.lower() for f in forms if f for w in re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]{3,}", str(f)))
    public = " ".join([case.task_text or "", getattr(case, "task_text_en", "") or "", *[str(c) for c in case.claims]]).lower()
    generic = _STOP | {"archive", "web", "twitter", "youtube", "facebook", "instagram", "tiktok", "reddit", "fact-check", "factcheck", "observers", "bridge"}
    return {w for w in words if w not in generic and w not in public}


def leaked_terms(decision_payload: dict[str, Any], output: dict[str, Any], case: Case | None) -> list[str]:
    """分叉决定里（检索词、打开的链接、QUERY_MEMORY 词）出现了特权词、而学生那份渲染状态里没有的。
    tp0 的"不可见词"只看可见性，把法语翻译（tour / incendie）和 police 这类常识词都报了；现在只报特权词。"""

    from .teacher import render_for_model

    if not isinstance(output, dict) or case is None:
        return []
    arguments = output.get("arguments") or {}
    asked = " ".join(str(arguments.get(k) or "") for k in ("query", "source"))
    if not asked.strip():
        return []
    rendered, _ = render_for_model(decision_payload)
    visible = json.dumps(rendered, ensure_ascii=False).lower()
    secret = privileged_terms(case)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]{3,}|\d{3,}", asked)
    def is_secret(word: str) -> bool:
        return any(fnmatch.fnmatch(word, s) if "*" in s else word == s for s in secret)
    return sorted({w for w in words if is_secret(w.lower()) and w.lower() not in visible})


def invisible_query_terms(decision_payload: dict[str, Any], output: dict[str, Any]) -> list[str]:
    """（tp0 用，保留作参考）SEARCH 查询里有哪些词在学生可见的渲染状态里找不到。"""

    from .teacher import render_for_model

    if not isinstance(output, dict) or output.get("action") != "SEARCH":
        return []
    query = str((output.get("arguments") or {}).get("query") or "")
    rendered, _ = render_for_model(decision_payload)
    visible = json.dumps(rendered, ensure_ascii=False).lower()
    return [w for w in _WORD.findall(query) if w.lower() not in _STOP and w.lower() not in visible]


def to_dict(obj: Any) -> Any:
    return asdict(obj)


__all__ = ["Atom", "Decision", "KeyState", "classify", "evaluate", "fact_table", "fingerprint", "forced_plan",
           "hidden_variant", "invisible_query_terms", "key_states", "leaked_terms", "load_decisions", "no_retrieval_variant",
           "normalized_action", "privileged_terms", "propose", "proposal_prompt", "valid_for_role"]
