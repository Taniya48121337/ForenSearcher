"""核验后处理：充分性层判 completed_without_finishing（major / critical）的报告改为 INCOMPLETE（10-16 r0g 收尾，负责人定"核验器为准"）。

不改充分性提示、不新调教师。对每份这样的报告：
- status → INCOMPLETE，status_reason = 充分性判定的 missing（英文）里学生看得到的部分（含学生输入与报告里都没有的专名 / 数字
  的条目视为来自 GT 的说明，去掉，记在 dropped）；findings / limitations 不动。
- 同一 run 里凡是带这份报告的地方一并改：execution_links / 报告请求的 response.json / trajectory.jsonl / 之后各请求的
  request.json（payload 与 rendered_payload：GIA 的 latest_reports、history_index、memory_history）/ store.sqlite3
  （reports、rounds、trajectory_events）。原文件留 *.pre_status_rewrite（只留第一次的）。
- walker 自己记的可修缺口：按记录的 payload 重放 decide_hi，取交报告那一步 walker._must_read 的结果（未读命中页）与
  _open_gap_clause；B 族另取 walker 当时 status_reason 里 classify_gaps 判可修的段。可修且之后 GIA 的决定是 FINISH →
  这些 FINISH 请求记进 finish_after，导出时 held（finish_after_status_rewrite）。
- 每个 run 写 status_rewrite.json；改完由调用方整轮再核验一次（缓存按单元命中，只有改了的单元重判）。

不是 walker 改动，也不在核验器 code_paths（datagen/verify、errors.py）里。
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

REWRITE_FILE = "status_rewrite.json"
BACKUP_SUFFIX = ".pre_status_rewrite"
SEVERE = frozenset({"major", "critical"})
HELD_REASON = "finish_after_status_rewrite"
MAX_ITEMS = 5
MAX_CHARS = 700
_CJK = re.compile(r"[一-鿿]")


# ---------------------------------------------------------------------------- 选报告

def targets(verdict_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """核验文件（verdicts_*.jsonl）里充分性层判 completed_without_finishing 且 major / critical 的报告行。"""

    out = []
    for row in verdict_rows:
        if row.get("unit") != "report":
            continue
        errors = [e for e in row.get("errors") or [] if e.get("type") == "completed_without_finishing"
                  and e.get("severity") in SEVERE]
        if errors:
            subject = row.get("subject") or {}
            out.append({"run": str(subject.get("run")), "workspace": str(subject.get("workspace_id")), "errors": errors,
                        "sufficiency": row.get("sufficiency") or {}})
    return out


# ---------------------------------------------------------------------------- status_reason

def leaked_tokens(text: str, visible: str) -> list[str]:
    """条目里学生看不到的专名 / 数字（导出 hidden_context 同一套词规则）：充分性层拿着案卷的必需证据清单（含 GT 来源）写 missing，
    "The earliest source (Getty, 2013-12-12) … was not found" 这类说明不能进学生的输出。"""

    from .export import _COMMON_WORDS, _HIDDEN_TOKEN, _token_in

    lower = visible.lower()
    out = []
    for token in _HIDDEN_TOKEN.findall(text):
        if token.isalpha() and (token.lower() in _COMMON_WORDS or token.lower() in lower):
            continue
        if _token_in(token, visible):
            continue
        out.append(token)
    return out


def _english_requirement(key: str, requirements: list[dict[str, Any]]) -> str | None:
    """充分性层用中文写的要求短语（r0g 37 份里只有 vcsl_pair_5 一份）→ 案卷里字面最接近的那条要求的 text_en（按汉字重合）。"""

    chars = set(_CJK.findall(key))
    best, score = None, 0.0
    for r in requirements:
        other = set(_CJK.findall(str(r.get("text") or "")))
        if chars and other and r.get("text_en"):
            s = len(chars & other) / len(chars)
            if s > score:
                best, score = str(r["text_en"]), s
    return best if score >= 0.5 else None


def build_reason(sufficiency: dict[str, Any], errors: list[dict[str, Any]], visible: str,
                 requirements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = [" ".join(str(x).split()) for x in sufficiency.get("missing") or [] if str(x).strip()]
    kept, dropped = [], []
    for item in items:
        bad = leaked_tokens(item, visible)
        if bad or _CJK.search(item):
            dropped.append({"text": item, "tokens": bad, "cjk": bool(_CJK.search(item))})
        else:
            kept.append(item.rstrip(" ."))
    source = "missing"
    if not kept:
        # missing 全是 GT 说明（或为空）：退到没覆盖的要求的 key（任务文字归纳的短语 / 案卷 key），同样过一遍
        keys = []
        for r in sufficiency.get("requirements") or []:
            if not isinstance(r, dict) or r.get("covered"):
                continue
            key = str(r.get("key") or "").replace("_", " ").replace("|", " or ").strip()
            if _CJK.search(key):
                key = (_english_requirement(key, requirements or []) or "").rstrip(" .")
                key = key[:1].lower() + key[1:]
            if key and not leaked_tokens(key, visible) and key not in keys:
                keys.append(key)
        kept = [f"not established: {', '.join(keys)}"] if keys else ["the findings do not cover everything the task asks for"]
        source = "requirements" if keys else "generic"
    reason = ""
    for item in kept[:MAX_ITEMS]:
        piece = item if not reason else "; " + item
        if len(reason) + len(piece) > MAX_CHARS:
            break
        reason += piece
    return {"reason": reason, "source": source, "missing": items, "kept": kept[:MAX_ITEMS], "dropped": dropped,
            "detail": [e.get("detail") for e in errors]}


# ---------------------------------------------------------------------------- walker 自己记的可修缺口

def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _requests(run_dir: Path) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    out = []
    for rdir in sorted((run_dir / "requests").glob("request_*"), key=lambda p: int(p.name.split("_")[-1])):
        request = _load(rdir / "request.json")
        if isinstance(request, dict):
            out.append((rdir.name, request, _load(rdir / "transport.json") or {}))
    return out


def walker_gaps(run_dir: Path, workspace: str, report_request: str) -> dict[str, Any]:
    """重放这个工作区的 decide_hi（payload 用 *.pre_status_rewrite 之前的原样；HI 请求本来就不改），在交报告那一步截下
    _must_read 的结果。B 族没有 _must_read，取 walker 当时 status_reason 里 classify_gaps 判可修的段。"""

    from . import walker as w

    captured: list[list[dict[str, Any]]] = []
    original = w._must_read

    def spy(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    memory: dict[str, Any] = {}
    must: list[dict[str, Any]] = []
    error = None
    walker_status = walker_reason = None
    family = None
    w._must_read = spy
    try:
        for name, request, transport in _requests(run_dir):
            payload = request.get("payload") or {}
            if payload.get("workspace_id") != workspace:
                continue
            captured.clear()
            try:
                w.decide_hi(payload, memory, request.get("rendered_payload"))
            except Exception as exc:  # noqa: BLE001 —— 重放失败只记，不挡改写
                error = f"{name}: {type(exc).__name__}: {exc}"
            if name == report_request:
                must = captured[-1] if captured else []
                slot = (transport.get("oracle") or {}).get("slot") or {}
                walker_status, walker_reason = slot.get("status"), slot.get("status_reason")
                family = slot.get("family") or w.task_family(str((payload.get("model_view") or {}).get("task") or ""),
                                                             True)
                break
    finally:
        w._must_read = original
    pages = [{"evidence_id": p.get("evidence_id"), "url": p.get("url"), "why": p.get("why")} for p in must]
    clause = w._open_gap_clause(must)
    b_fixable: list[str] = []
    if walker_status == "INCOMPLETE" and walker_reason:
        reported = [{"report": {"task_id": workspace, "status": "INCOMPLETE", "status_reason": walker_reason, "findings": []}}]
        try:
            b_fixable = [g["reason"] for g in w.classify_gaps(reported, {}) if g.get("fixable")]
        except Exception:  # noqa: BLE001
            b_fixable = []
    return {"family": family, "walker_status": walker_status, "walker_reason": walker_reason, "unread_hit_pages": pages,
            "open_gap_clause": clause, "walker_fixable_segments": b_fixable,
            "fixable": bool(clause or pages or b_fixable), "replay_error": error}


# ---------------------------------------------------------------------------- 改写

def _rewrite(value: Any, workspace: str, reason: str) -> tuple[Any, int]:
    """把值里所有"这个工作区的报告记录"（task_id = 工作区、带 status 与 finding_ids / findings）改成 INCOMPLETE + reason。"""

    if isinstance(value, list):
        n = 0
        out = []
        for item in value:
            new, k = _rewrite(item, workspace, reason)
            out.append(new)
            n += k
        return out, n
    if not isinstance(value, dict):
        return value, 0
    n = 0
    out: dict[str, Any] = {}
    is_report = (value.get("task_id") == workspace and value.get("status") == "COMPLETED"
                 and ("finding_ids" in value or "findings" in value))
    for key, item in value.items():
        if is_report and key == "status":
            out[key] = "INCOMPLETE"
            if "status_reason" not in value:
                out["status_reason"] = reason
            continue
        if is_report and key == "status_reason":
            out[key] = reason
            continue
        new, k = _rewrite(item, workspace, reason)
        out[key] = new
        n += k
    return out, n + (1 if is_report else 0)


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def _rewrite_json_file(path: Path, workspace: str, reason: str) -> int:
    data = _load(path)
    if data is None:
        return 0
    new, n = _rewrite(data, workspace, reason)
    if n:
        _backup(path)
        path.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def _rewrite_jsonl(path: Path, workspace: str, reason: str) -> int:
    if not path.is_file():
        return 0
    lines, total = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            new, n = _rewrite(json.loads(line), workspace, reason)
        except ValueError:
            lines.append(line)
            continue
        total += n
        lines.append(json.dumps(new, ensure_ascii=False) if n else line)
    if total:
        _backup(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


def _rewrite_store(path: Path, workspace: str, reason: str) -> int:
    if not path.is_file():
        return 0
    _backup(path)
    total = 0
    con = sqlite3.connect(path)
    try:
        for table, key, cols in (("reports", "task_id", ("payload_json",)), ("rounds", "round_index", ("gia_input_json", "gia_decision_json")),
                                 ("trajectory_events", "id", ("payload_json",))):
            for row in con.execute(f"SELECT {key}, {', '.join(cols)} FROM {table}").fetchall():
                for col, raw in zip(cols, row[1:]):
                    try:
                        new, n = _rewrite(json.loads(raw), workspace, reason)
                    except (TypeError, ValueError):
                        continue
                    if n:
                        con.execute(f"UPDATE {table} SET {col} = ? WHERE {key} = ?", (json.dumps(new, ensure_ascii=False), row[0]))
                        total += n
        con.commit()
    finally:
        con.close()
    return total


def _report_request(run_dir: Path, workspace: str) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    links = _load(run_dir / "execution_links.json") or []
    for link in links:
        output = link.get("raw_output") or {}
        rid = link.get("request_id")
        if rid and isinstance(output, dict) and "status" in output and "findings" in output:
            request = _load(run_dir / "requests" / rid / "request.json") or {}
            if (request.get("payload") or {}).get("workspace_id") == workspace:
                return rid, request, output
    return None, None, None


def rewrite_report(run_dir: Path, target: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    workspace = target["workspace"]
    rid, request, output = _report_request(run_dir, workspace)
    if not rid or output is None or request is None:
        return {"workspace": workspace, "changed": False, "why": "report request not found"}
    if output.get("status") != "COMPLETED":
        return {"workspace": workspace, "request": rid, "changed": False, "why": f"status is {output.get('status')}"}
    visible = json.dumps(request.get("rendered_payload") or request.get("payload"), ensure_ascii=False) + "\n" + \
        json.dumps(output, ensure_ascii=False)
    config = _load(run_dir / "run_config.json") or {}
    case = _load(Path(str(config.get("case_file") or ""))) if config.get("case_file") else None
    reason = build_reason(target.get("sufficiency") or {}, target.get("errors") or [], visible,
                          (case or {}).get("requirements") or [])
    gaps = walker_gaps(run_dir, workspace, rid)
    counts = {"execution_links": _rewrite_json_file(run_dir / "execution_links.json", workspace, reason["reason"]),
              "response": _rewrite_json_file(run_dir / "requests" / rid / "response.json", workspace, reason["reason"]),
              "trajectory": _rewrite_jsonl(run_dir / "trajectory.jsonl", workspace, reason["reason"]),
              "store": _rewrite_store(run_dir / "store.sqlite3", workspace, reason["reason"]), "requests": {}}
    later_gia: list[dict[str, Any]] = []
    links = {l.get("request_id"): l for l in _load(run_dir / "execution_links.json") or [] if l.get("request_id")}
    for name, req, _ in _requests(run_dir):
        if int(name.split("_")[-1]) <= int(rid.split("_")[-1]):
            continue
        n = _rewrite_json_file(run_dir / "requests" / name / "request.json", workspace, reason["reason"])
        if n:
            counts["requests"][name] = n
        if not (req.get("payload") or {}).get("workspace_id"):
            decision = ((links.get(name) or {}).get("raw_output") or {})
            later_gia.append({"request": name, "decision": decision.get("decision") or decision.get("action"), "state_rewritten": bool(n)})
    finish_after = [g["request"] for g in later_gia if g["decision"] == "FINISH"] if gaps["fixable"] else []
    return {"workspace": workspace, "request": rid, "changed": True, "old_status": output.get("status"),
            "old_status_reason": output.get("status_reason"), "new_status_reason": reason["reason"], "reason_detail": reason,
            "verifier_errors": [{"severity": e.get("severity"), "detail": e.get("detail")} for e in target.get("errors") or []],
            "walker": gaps, "later_gia": later_gia, "finish_after": finish_after, "counts": counts}


def rewrite_round(verdict_rows: list[dict[str, Any]], runs_dir: Path, *, source: str = "") -> list[dict[str, Any]]:
    """对一轮的核验结果做改写；每个 run 的记录并进 run 目录的 status_rewrite.json（同一工作区只改一次）。返回本次改动。"""

    done: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for t in targets(verdict_rows):
        by_run.setdefault(t["run"], []).append(t)
    for run, items in sorted(by_run.items()):
        run_dir = Path(runs_dir) / run
        if not run_dir.is_dir():
            continue
        record_path = run_dir / REWRITE_FILE
        record = _load(record_path) or {"rewrites": []}
        seen = {r.get("workspace") for r in record["rewrites"] if r.get("changed")}
        for t in items:
            if t["workspace"] in seen:
                continue
            result = rewrite_report(run_dir, t)
            result.update({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "verdicts": source})
            if result.get("changed"):
                record["rewrites"].append(result)
                done.append({"run": run, **result})
        if record["rewrites"]:
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return done


def finish_after_marks(run_dir: Path) -> dict[str, dict[str, Any]]:
    """导出用：{FINISH 请求号: 改写记录摘要}——改写后的报告带可修缺口、GIA 当时仍 FINISH 的行。"""

    record = _load(Path(run_dir) / REWRITE_FILE) or {}
    out: dict[str, dict[str, Any]] = {}
    for r in record.get("rewrites") or []:
        for rid in r.get("finish_after") or []:
            out[rid] = {"workspace": r.get("workspace"), "report_request": r.get("request")}
    return out


def rewritten_requests(run_dir: Path) -> dict[str, str]:
    """导出用：{报告请求号: 工作区}——status 由核验后处理改写的报告行。"""

    record = _load(Path(run_dir) / REWRITE_FILE) or {}
    return {r["request"]: r["workspace"] for r in record.get("rewrites") or [] if r.get("changed") and r.get("request")}


__all__ = ["HELD_REASON", "REWRITE_FILE", "build_reason", "finish_after_marks", "leaked_tokens", "rewrite_report",
           "rewrite_round", "rewritten_requests", "targets", "walker_gaps"]
