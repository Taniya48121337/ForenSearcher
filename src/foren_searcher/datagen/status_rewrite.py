"""核验后处理：充分性层判 completed_without_finishing（major / critical）的报告改为 INCOMPLETE（10-16 r0g 收尾，负责人定"核验器为准"）。

不改充分性提示、不新调教师。对每份这样的报告：
- status → INCOMPLETE，status_reason = 充分性判定的 missing（英文）里学生看得到的部分（与案卷 GT 字段——标注来源的站名 / 网址、
  标注日期、标注地名——实际匹配且学生看不到的条目视为来自 GT 的说明，去掉，记在 dropped；10-16 r0+ 前是"学生看不到的专名 /
  数字"，句首大写词与中文条目被误去）；中文条目退到案卷要求的 text_en；findings / limitations 不动。build_reason 也给
  walker 循环内的充分性检查用（oracle_generator）。
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


def _english_requirement(key: str, requirements: list[dict[str, Any]], threshold: float = 0.5) -> str | None:
    """充分性层用中文写的要求短语 / missing 条目 → 案卷里字面最接近的那条要求的 text_en（按汉字重合，不调模型）。
    分数取 重合字数 / 两边较短的一边：条目常比要求长（"图片是否确实拍摄于美国国会大厦，未得到核验……" 对 "图片是否拍于声明所说的地点"）。"""

    chars = set(_CJK.findall(key))
    best, score = None, 0.0
    for r in requirements:
        other = set(_CJK.findall(str(r.get("text") or "")))
        if chars and other and r.get("text_en"):
            s = len(chars & other) / min(len(chars), len(other))
            if s > score:
                best, score = str(r["text_en"]), s
    return best if score >= threshold else None


# ---------------------------------------------------------------------------- GT 条目（10-16 r0+：只认与案卷 GT 字段实际匹配的）

_ISO_DATE = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_GT_URL = re.compile(r"(?:https?://)?(?:web\.archive\.org/web/\d+[a-z_]*/)?(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(/[^\s（()，,；;]*)?", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'&.-]*")
_TLD_LIKE = {"com", "org", "net", "gov", "edu", "co", "uk", "in", "news", "info", "tv", "de", "fr", "au", "ca", "es", "it", "ru",
             "cn", "jp", "br", "pk", "bd", "lk", "np", "ng", "za", "ke", "ph", "id", "my", "sg", "nz", "ie", "us", "io", "me", "ac"}
_HOST_SKIP = {"www", "m", "amp", "web", "en", "edition", "archive", "mobile"}


def _dates_in(text: str) -> set[str]:
    out: set[str] = set()
    for y, m, d in _ISO_DATE.findall(text):
        out.add(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
    for d, mon, y in _DMY.findall(text):
        if mon[:3].lower() in _MONTHS:
            out.add(f"{int(y):04d}-{_MONTHS[mon[:3].lower()]:02d}-{int(d):02d}")
    for mon, d, y in _MDY.findall(text):
        if mon[:3].lower() in _MONTHS:
            out.add(f"{int(y):04d}-{_MONTHS[mon[:3].lower()]:02d}-{int(d):02d}")
    return out


def _site_tokens(host: str) -> list[str]:
    """gettyimages.co.uk → ["gettyimages"]；www.tribuneindia.com → ["tribuneindia"]；edition.cnn.com → ["cnn"]。"""

    labels = [x for x in host.lower().split(".") if x and x not in _HOST_SKIP]
    while labels and labels[-1] in _TLD_LIKE:
        labels.pop()
    return [x.replace("-", "") for x in labels[-1:] if len(x) >= 3]


def gt_fields(case: dict[str, Any] | None) -> dict[str, Any]:
    """案卷里只有 GT 才知道的字段：标注来源的网址 / 站名、标注日期、标注地名（entity_in_evidence）。来源：notes 里的
    "标注来源 <url>"、required_evidence 里描述以"标注"开头的条目（与它的 url_pattern）、checks 的 date_in_range 注释
    （"标注：最早发布 YYYY-MM-DD"）与 entity_in_evidence 的 text。学生输入里本来就有的（声明、任务文字）不在这里排除，
    由 gt_matches 的可见性判断处理。"""

    case = case or {}
    texts: list[str] = []
    m = re.findall(r"标注(?:的最早)?来源[:：]?\s*(\S+)", str(case.get("notes") or ""))
    texts += m
    for r in case.get("required_evidence") or []:
        desc = str(r.get("description") or "")
        if desc.startswith("标注"):
            texts.append(desc)
            if r.get("url_pattern"):
                texts.append(str(r["url_pattern"]))
    dates: set[str] = set()
    entities: list[str] = []
    for check in case.get("checks") or []:
        kind, params = check.get("kind"), check.get("params") or {}
        if kind == "date_in_range":
            dates |= _dates_in(str(check.get("note") or ""))
        elif kind == "entity_in_evidence":
            for value in [params.get("text"), *(params.get("aliases") or [])]:
                if isinstance(value, str) and value.strip() and value.strip() not in entities:
                    entities.append(value.strip())
    hosts, sites, paths = [], [], []
    for text in texts:
        dates |= _dates_in(text)
        for host, path in _GT_URL.findall(text):
            host = host.lower().removeprefix("www.")
            if host.startswith("web.archive.org") or "." not in host:
                continue
            if host not in hosts:
                hosts.append(host)
            sites += [s for s in _site_tokens(host) if s not in sites]
            if path and len(path) > 1 and path not in paths:
                paths.append(path.lower().rstrip("*"))
    return {"hosts": hosts, "sites": sites, "paths": paths, "dates": sorted(dates), "entities": entities}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _site_hit(ngram: str, site: str) -> bool:
    """n 元词组（去空格小写）是否就是标注站名：相等，或是站名的前缀（Getty → gettyimages），或占站名一半以上
    （Guardian → theguardian）。"Times" 对 hindustantimes 只占 5/14、又不是前缀，不算。"""

    if len(ngram) < 4 or ngram not in site:
        return False
    return ngram == site or site.startswith(ngram) or len(ngram) * 2 >= len(site)


def gt_matches(item: str, gt: dict[str, Any], visible: str) -> list[str]:
    """条目里与 GT 字段实际匹配、且学生看不到的片段（站名 / 网址、日期、年份、地名）。空 = 不是 GT 条目。"""

    if not gt:
        return []
    vis_lower, vis_norm, vis_dates = visible.lower(), _norm(visible), _dates_in(visible)
    hits: list[str] = []
    words = _WORD.findall(item)
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            ng = _norm(phrase)
            if any(_site_hit(ng, s) for s in gt.get("sites") or []) and ng not in vis_norm \
                    and not any(phrase in h for h in hits):
                hits.append(phrase)
    for host in gt.get("hosts") or []:
        if host.removeprefix("www.") in item.lower() and host.removeprefix("www.") not in vis_lower and host not in hits:
            hits.append(host)
    gt_dates = set(gt.get("dates") or [])
    for d in _dates_in(item):
        if d in gt_dates and d not in vis_dates:
            hits.append(d)
    gt_years = {d[:4] for d in gt_dates}
    item_dates = _dates_in(item)
    for y in _YEAR.findall(item):
        if y in gt_years and y not in visible and not any(d.startswith(y) for d in item_dates if d in gt_dates):
            hits.append(y)
    for ent in gt.get("entities") or []:
        if re.search(rf"\b{re.escape(ent)}\b", item, re.I) and not re.search(rf"\b{re.escape(ent)}\b", visible, re.I):
            hits.append(ent)
    return hits


def build_reason(sufficiency: dict[str, Any], errors: list[dict[str, Any]], visible: str,
                 requirements: list[dict[str, Any]] | None = None, case: dict[str, Any] | None = None) -> dict[str, Any]:
    """status_reason = 充分性判定的 missing（英文）。10-16 r0+（负责人第 2 项）：只去掉与案卷 GT 字段实际匹配、学生看不到的条目
    （gt_matches：标注来源的站名 / 网址 / 日期 / 年份 / 地名）；句首大写词不再当专名。中文条目不去：退到字面最接近的那条
    案卷要求的 text_en（"not established: …"），对不上任何要求时才去掉。循环内（oracle_generator）与事后改写共用。"""

    items = [" ".join(str(x).split()) for x in sufficiency.get("missing") or [] if str(x).strip()]
    gt = gt_fields(case)
    kept, dropped, mapped = [], [], []
    for item in items:
        bad = gt_matches(item, gt, visible)
        if bad:
            dropped.append({"text": item, "tokens": bad, "cjk": bool(_CJK.search(item)), "why": "gt"})
            continue
        if _CJK.search(item):
            en = _english_requirement(item, requirements or [], threshold=0.4)
            if not en:
                dropped.append({"text": item, "tokens": [], "cjk": True, "why": "cjk_unmapped"})
                continue
            piece = "not established: " + en[:1].lower() + en[1:].rstrip(" .")
            mapped.append({"text": item, "text_en": en})
            if piece not in kept:
                kept.append(piece)
            continue
        if item.rstrip(" .") not in kept:
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
            if key and not gt_matches(key, gt, visible) and key not in keys:
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
            "mapped_cjk": mapped, "rule": "gt_fields@20261016", "detail": [e.get("detail") for e in errors]}


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
            ws = str(payload.get("workspace_id") or "")
            if ws.startswith("W1-") and "run_family" not in memory and not workspace.startswith("W1-"):
                # r1：第 2 轮沿用第 1 轮的族（oracle_generator 同样做法），重放时也要带上
                has_search = any(isinstance(a, dict) and a.get("name") == "SEARCH" for a in payload.get("available_actions") or [])
                memory["run_family"] = w.task_family(str((payload.get("model_view") or {}).get("task") or ""), has_search)
            if ws != workspace:
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
                          (case or {}).get("requirements") or [], case)
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


INLOOP_FILE = "inloop_sufficiency.json"       # = oracle_generator.INLOOP_FILE


def inloop_rewritten_requests(run_dir: Path) -> dict[str, str]:
    """导出用（10-16 r0+）：{报告请求号: 工作区}——循环内充分性检查把 status 改成 INCOMPLETE 的报告行（GIA 看到的就是改后的）。"""

    record = _load(Path(run_dir) / INLOOP_FILE) or {}
    return {c["request"]: c["workspace"] for c in record.get("checks") or [] if c.get("rewritten") and c.get("request")}


def rewritten_requests(run_dir: Path) -> dict[str, str]:
    """导出用：{报告请求号: 工作区}——status 由核验后处理改写的报告行。"""

    record = _load(Path(run_dir) / REWRITE_FILE) or {}
    return {r["request"]: r["workspace"] for r in record.get("rewrites") or [] if r.get("changed") and r.get("request")}


__all__ = ["HELD_REASON", "INLOOP_FILE", "REWRITE_FILE", "build_reason", "finish_after_marks", "gt_fields", "gt_matches",
           "inloop_rewritten_requests", "leaked_tokens", "rewrite_report", "rewrite_round", "rewritten_requests", "targets", "walker_gaps"]
