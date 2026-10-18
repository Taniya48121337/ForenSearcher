"""r0 训练集定稿：教学矩阵（这批 accepted 数据教什么）与三个 D2 坏习惯的程序计数（2026-10-19）。

    python scripts/teaching_matrix.py --pack r1 <r1 导出目录> <r1 runs 目录> qwen \
        --pack g1 <g1 导出目录> <g1 runs 目录> codex [--pack g2 ...] \
        --unanswerable experiments/sft_data/unanswerable_r1.json --out claude/_g2_work/teaching_matrix

只读导出的 *_accepted.jsonl（held / rejected 只用来判断某格"walker 会走、但一行都没进 accepted"）与 run 目录里对应请求的
request.json / transport.json（取证据卡片的来源、目标的模态、S4 的 required_findings 与 notes）；不改任何数据。

行 = 族（A / B / C / E / L）× 可回答 / 不可回答（只有 E / L 有探测口径；A / B / C 都记"可回答（未探测）"）。
列 = 要教的动作与习惯，见 COLUMNS。一行（训练样本）可以落在几列（例如一份报告同时是 COMPLETED、带比对结论、用了"候选"措辞）。
每格：accepted 行数、涉及案例数、按教师来源（S0–S3 = Qwen / Codex）分的行数；行数 < 15 或案例数 < 5 记为缺口；该族的 walker
从来不走这一列（三个桶里都没有）记"不适用"。

坏习惯（D2 通读点名、程序能查的三条；只报数）：
1. 理由模板化：同一 rationale 字符串（去首尾空白、合并空白）在 accepted 里出现 ≥ 3 次的行占比（全部 / 只算 S0 教师写的）。
2. 比对前就说匹配：rationale 含 match / same（词边界）而该步之前没有 compare_media 记录（证据卡片里没有 INSPECT/compare_media、
   历史里没有 compare_media 调用）。另给"断言式"口径：去掉 compare_media 本身那一步、去掉 whether / if / check / verify 一类问句。
3. GIA 答案出现可见输入里没有的实体：FINISH 答案里的专名（非句首的大写词串）与数字，不在该行学生可见的消息（system + user）里。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# (键, 组, 中文名)
COLUMNS: list[tuple[str, str, str]] = [
    ("S_text_text", "SEARCH", "文字搜 → 网页"),
    ("S_text_image", "SEARCH", "文字搜 → 图片（参考图）"),
    ("S_text_video", "SEARCH", "文字搜 → 视频"),
    ("S_image", "SEARCH", "以图搜"),
    ("S_text_image_in", "SEARCH", "图 + 文字搜（text_image）"),
    ("S_dates", "SEARCH", "查页面日期（include dates）"),
    ("S_open_page", "SEARCH", "打开页面（source → text）"),
    ("S_page_image", "SEARCH", "取页内主图（source → image）"),
    ("S_open_video", "SEARCH", "打开视频（source → video）"),
    ("R_page", "READ_RECORD", "读页面正文"),
    ("R_page_more", "READ_RECORD", "续读（offset > 0）"),
    ("R_record", "READ_RECORD", "读工具记录（OCR / 结构等）"),
    ("R_ref_image", "READ_RECORD", "看参考图（L）"),
    ("R_transcript", "READ_RECORD", "读整段转写"),
    ("R_compare_review", "READ_RECORD", "对照图回看"),
    ("R_frames_review", "READ_RECORD", "交报告前回看帧（C9v）"),
    ("I_ocr_image", "INSPECT", "OCR 图片"),
    ("I_ocr_video", "INSPECT", "OCR 视频"),
    ("I_asr", "INSPECT", "ASR"),
    ("I_frames_times", "INSPECT", "get_frames times_s"),
    ("I_frames_box", "INSPECT", "get_frames + box 放大"),
    ("I_frames_count", "INSPECT", "get_frames count（均匀）"),
    ("I_frames_range", "INSPECT", "get_frames range_s"),
    ("I_video_structure", "INSPECT", "video_structure"),
    ("I_compare_image", "INSPECT", "compare_media 图片模式"),
    ("I_compare_video", "INSPECT", "compare_media 视频模式"),
    ("I_compare_range", "INSPECT", "compare_media range_a_s / region"),
    ("I_locate", "INSPECT", "locate"),
    ("Q_memory", "QUERY_MEMORY", "QUERY_MEMORY"),
    ("P_completed", "REPORT", "报告 COMPLETED"),
    ("P_incomplete_walker", "REPORT", "报告 INCOMPLETE（walker / S4 自己判）"),
    ("P_inloop_rewritten", "REPORT", "报告循环内改写为 INCOMPLETE"),
    ("P_posthoc_rewritten", "REPORT", "报告事后 status 改写"),
    ("G_round1", "GIA", "GIA 第 1 轮派工"),
    ("G_round2", "GIA", "GIA 第 2 轮按缺口派工"),
    ("G_round2_scope_compare", "GIA", "第 2 轮 Scope 第一项是取图比对"),
    ("G_finish", "GIA", "GIA FINISH"),
    ("G_finish_after_round2", "GIA", "第 2 轮后 FINISH"),
    ("H_candidate", "习惯", "\"候选\"措辞（candidate）"),
    ("H_page_vs_image_date", "习惯", "页面日期与图片日期分开"),
    ("H_attribution", "习惯", "归因保留（according to / reports …）"),
    ("H_honest_incomplete", "习惯", "如实 INCOMPLETE（写明缺口）"),
    ("RF_compare", "S4 结论", "比对结论（compare_media / 参考图）"),
    ("RF_frames", "S4 结论", "帧结论（画面里看到什么）"),
    ("RF_transcript", "S4 结论", "整段转写概括"),
    ("RF_interval", "S4 结论", "区间（两侧时间对应）"),
]
COL_KEYS = [c[0] for c in COLUMNS]
GROUPS = [("A", "ans"), ("B", "ans"), ("C", "ans"), ("E", "ans"), ("E", "unans"), ("L", "ans"), ("L", "unans")]
GAP_ROWS, GAP_CASES = 15, 5

# 工具目录（hi@0.11.0 的 available_actions，request.json 里原样给学生的那份）里的操作与参数形态
CATALOG = [
    "SEARCH input_type=text output_type=text", "SEARCH input_type=text output_type=image", "SEARCH input_type=text output_type=video",
    "SEARCH input_type=image output_type=text", "SEARCH input_type=image output_type=image", "SEARCH input_type=image output_type=video",
    "SEARCH input_type=text_image output_type=text", "SEARCH input_type=text_image output_type=image",
    "SEARCH input_type=text_image output_type=video",
    "SEARCH source output_type=text", "SEARCH source output_type=image", "SEARCH source output_type=video",
    "SEARCH source output_type=text include=[dates]",
    "INSPECT get_frames", "INSPECT get_frames · times_s", "INSPECT get_frames · count", "INSPECT get_frames · range_s",
    "INSPECT get_frames · pages", "INSPECT get_frames · page_range", "INSPECT get_frames · box",
    "INSPECT ocr", "INSPECT ocr · range_s", "INSPECT asr", "INSPECT asr · language", "INSPECT asr · range_s",
    "INSPECT locate", "INSPECT locate · query", "INSPECT locate · range_s", "INSPECT locate · top_k",
    "INSPECT video_structure", "INSPECT video_structure · range_s",
    "INSPECT compare_media", "INSPECT compare_media · range_a_s", "INSPECT compare_media · region", "INSPECT compare_media · min_area",
    "QUERY_MEMORY", "QUERY_MEMORY · query", "QUERY_MEMORY · record_types", "QUERY_MEMORY · cursor",
    "READ_RECORD", "READ_RECORD · offset",
]

MATCH_RE = re.compile(r"\b(match(?:es|ed|ing)?|same)\b", re.I)
HEDGE_RE = re.compile(r"\b(whether|if|check(?:s|ing)?|verify|verifies|confirm(?:s)?|determine|test|compare|see)\b[^.;]*?\b(match(?:es|ed|ing)?|same)\b", re.I)
CANDIDATE_RE = re.compile(r"\bcandidates?\b", re.I)
PAGE_DATE_RE = re.compile(r"(page|article|post|site)(?:'s|’s)?\s+(?:\w+\s+){0,3}(?:declares?|declared|states?|gives|shows|lists)\b[^.]*\b(?:date|dated|publication|published)"
                          r"|declares? (?:its )?publication|publication date|dated \d|dated (?:January|February|March|April|May|June|July|August|September|October|November|December)", re.I)
IMAGE_DATE_RE = re.compile(r"not (?:necessarily |a confirmed |confirmed |the )?(?:when|date|first[- ]publication|first publication)"
                           r"|(?:image|picture|photo|photograph|video|footage)(?:'s|’s)? (?:own |actual |first )?(?:capture|creation|first (?:appearance|publication))"
                           r"|does not (?:establish|show|prove|date) (?:when|the (?:image|picture|photo|photograph|video))"
                           r"|no later than|online by|available by|first[- ]ever publication", re.I)
ATTRIB_RE = re.compile(r"\baccording to\b|\breports?\b|\breported\b|\bstates?\b|\bsays\b|\bdescribes?\b|\bwrites\b|\battribut", re.I)
GAP_RE = re.compile(r"\bnot (?:found|determined|established|confirmed|verified|identified|located|obtained)\b|\bcould not\b|\bcannot\b|\bunable\b|\bundetermined\b|\bunverified\b", re.I)
INTERVAL_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:–|-|to)\s*\d+(?:\.\d+)?\s*s\b")
STOP_CAPS = {"The", "This", "That", "These", "Those", "A", "An", "It", "Its", "In", "On", "At", "By", "No", "Not", "However", "Both",
             "Material", "Materials", "Finding", "Findings", "Evidence", "Image", "Video", "Frame", "Frames", "Report", "Reports",
             "COMPLETED", "INCOMPLETE", "FINISH", "OCR", "ASR", "URL", "January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday", "I", "If", "As", "And", "But", "Or", "With", "For", "From", "To", "Of", "Neither", "Nor",
             "One", "Two", "Three", "There", "They", "He", "She", "We", "Their", "His", "Her", "When", "While", "Where", "Which",
             "What", "Who", "Whether", "Also", "Any", "All", "Each", "Some", "Such", "Only", "Although", "Because", "Since", "Therefore",
             "Thus", "So", "Yet", "Earlier", "Later", "First", "Second", "Third", "Part", "Scope", "Task", "Tasks"}
CAPS_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z’'\-]{2,}(?:\s+[A-Z][a-zA-Z’'\-]{2,})*)")
NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
EID_RE = re.compile(r"\b[ERFWCI]\d+\b")


def family(case_id: str) -> str:
    ds, _, sid = str(case_id).partition(":")
    if ds in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if ds == "averimatec":
        return "E" if sid.startswith("e") else "L"
    if ds == "evid" or (ds == "fmnv" and sid.startswith("a_")):
        return "A"
    return "C"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def target(row: dict[str, Any]) -> dict[str, Any]:
    content = (row.get("messages") or [{}])[-1].get("content")
    try:
        return json.loads(content) if isinstance(content, str) else {}
    except ValueError:
        return {}


def visible_text(row: dict[str, Any]) -> str:
    out = []
    for m in (row.get("messages") or [])[:-1]:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [p.get("text") or "" for p in c if isinstance(p, dict)]
    return "\n".join(out)


class RunCache:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.cache: dict[tuple[str, str, str], Any] = {}

    def load(self, run: str, request: str, name: str) -> dict[str, Any]:
        key = (run, request, name)
        if key not in self.cache:
            path = self.runs_dir / run / "requests" / request / name
            try:
                self.cache[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.cache[key] = {}
        return self.cache[key]

    def cards(self, run: str, request: str) -> dict[str, dict[str, Any]]:
        payload = self.load(run, request, "request.json").get("payload") or {}
        mv = payload.get("model_view") or {}
        cards = {c.get("evidence_id"): c for c in mv.get("evidence_view") or [] if isinstance(c, dict)}
        for k, c in (mv.get("materials") or {}).items():
            if isinstance(c, dict) and c.get("evidence_id"):
                cards.setdefault(c["evidence_id"], c)
        return cards

    def history_ops(self, run: str, request: str) -> list[str]:
        payload = self.load(run, request, "request.json").get("payload") or {}
        ops = []
        for h in (payload.get("model_view") or {}).get("history") or []:
            req = (h.get("request") or {}).get("request") or {}
            ops.append(f"{h.get('action')}:{req.get('operation') or ''}")
        return ops


def modality(card: dict[str, Any] | None) -> str:
    mods = (card or {}).get("modalities") or []
    return "video" if "video" in mods else "image" if "image" in mods else "audio" if "audio" in mods else "text" if mods else "?"


def action_forms(tgt: dict[str, Any], cards: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """返回（教学列，工具目录形态）。"""

    action = tgt.get("action")
    args = tgt.get("arguments") or {}
    cols: list[str] = []
    form: list[str] = []
    if action == "SEARCH":
        out = args.get("output_type")
        if "source" in args:
            if args.get("include"):
                cols.append("S_dates")
                form = ["SEARCH source output_type=text include=[dates]"]
            else:
                cols.append({"image": "S_page_image", "video": "S_open_video"}.get(out, "S_open_page"))
                form = [f"SEARCH source output_type={out}"]
        else:
            it = args.get("input_type")
            cols.append({"image": "S_image", "text_image": "S_text_image_in"}.get(it) or f"S_text_{out}")
            form = [f"SEARCH input_type={it} output_type={out}"]
    elif action == "INSPECT":
        req = args.get("request") or {}
        op, params = req.get("operation"), req.get("params") or {}
        keys = sorted(k for k, v in params.items() if v not in (None, [], ""))
        form = [f"INSPECT {op}"] + [f"INSPECT {op} · {k}" for k in keys]
        mods = {modality(cards.get(t)) for t in args.get("targets") or []}
        if op == "ocr":
            cols.append("I_ocr_video" if "video" in mods else "I_ocr_image")
        elif op == "asr":
            cols.append("I_asr")
        elif op == "get_frames":
            if "box" in keys:
                cols.append("I_frames_box")
            elif "count" in keys:
                cols.append("I_frames_count")
            elif "times_s" in keys:
                cols.append("I_frames_times")
            else:
                cols.append("I_frames_range")
        elif op == "video_structure":
            cols.append("I_video_structure")
        elif op == "compare_media":
            cols.append("I_compare_video" if "video" in mods else "I_compare_image")
            if {"range_a_s", "region"} & set(keys):
                cols.append("I_compare_range")
        elif op == "locate":
            cols.append("I_locate")
    elif action == "QUERY_MEMORY":
        cols.append("Q_memory")
        form = ["QUERY_MEMORY"] + [f"QUERY_MEMORY · {k}" for k in sorted(k for k, v in args.items() if v not in (None, [], ""))]
    elif action == "READ_RECORD":
        form = ["READ_RECORD"] + (["READ_RECORD · offset"] if int(args.get("offset") or 0) > 0 else [])
    return cols, form


def read_record_col(tgt: dict[str, Any], md: dict[str, Any], cards: dict[str, dict[str, Any]]) -> str:
    args = tgt.get("arguments") or {}
    rule = str(md.get("walker_rule") or "")
    card = cards.get(args.get("record_id")) or {}
    acq = str(card.get("acquired_by") or "")
    if int(args.get("offset") or 0) > 0:
        return "R_page_more"
    if rule.startswith("L3"):
        return "R_ref_image"
    if rule.startswith("C9v"):
        return "R_frames_review"
    if acq.startswith("INSPECT/compare_media") or rule.startswith("E7") or rule == "B8_review_pairs":
        return "R_compare_review"
    if acq.startswith("INSPECT/asr"):
        return "R_transcript"
    if acq.startswith("SEARCH") or acq.startswith("HARNESS/search"):
        return "R_page"
    return "R_record"


def statements(tgt: dict[str, Any]) -> str:
    if "findings" in tgt:
        return "\n".join(str(f.get("statement") or "") for f in tgt.get("findings") or [])
    return str(tgt.get("answer") or "")


def habit_cols(text: str, tgt: dict[str, Any], fam: str) -> list[str]:
    cols = []
    if CANDIDATE_RE.search(text):
        cols.append("H_candidate")
    if fam in ("E", "L", "A") and PAGE_DATE_RE.search(text) and IMAGE_DATE_RE.search(text):
        cols.append("H_page_vs_image_date")
    if ATTRIB_RE.search(text):
        cols.append("H_attribution")
    if tgt.get("status") == "INCOMPLETE" and (tgt.get("status_reason") or GAP_RE.search(text)):
        cols.append("H_honest_incomplete")
    return cols


def report_rf_cols(tgt: dict[str, Any], fam: str, transport: dict[str, Any]) -> list[str]:
    slot = (transport.get("oracle") or {}).get("slot") or {}
    notes = " ".join(slot.get("notes") or [])
    msgs = " ".join(str(f.get("message") or "") for f in slot.get("required_findings") or [] if isinstance(f, dict))
    text = statements(tgt)
    blob = (notes + " " + msgs).lower()
    cols = []
    if ("compare_media record" in blob or "reference photos" in blob or "comparison" in blob) and re.search(
            r"same|match|differ|compar|identical|alter|edit|crop|consistent|inconsist", text, re.I):
        cols.append("RF_compare")
    elif fam in ("A", "B") and re.search(r"compare_media|comparison|aligned|correspond", text, re.I):
        cols.append("RF_compare")
    if "scene frames" in blob and re.search(r"show|visible|frame|appear|depict|footage|picture", text, re.I):
        cols.append("RF_frames")
    elif fam in ("A", "B") and re.search(r"\bframes?\b", text, re.I):
        cols.append("RF_frames")
    if "speech transcript" in blob and re.search(r"say|says|said|speak|speech|transcript|state|mention", text, re.I):
        cols.append("RF_transcript")
    if fam in ("A", "B") and len(INTERVAL_RE.findall(text)) >= 2:
        cols.append("RF_interval")
    return cols


def row_columns(row: dict[str, Any], rc: RunCache) -> tuple[list[str], list[str]]:
    md = row.get("metadata") or {}
    tgt = target(row)
    fam = family(md.get("case_id"))
    run, request = str(md.get("run")), str(md.get("request"))
    rule = str(md.get("walker_rule") or "")
    if md.get("role") == "gia":
        if md.get("unit") == "plan":
            if rule.startswith("G1"):
                return ["G_round1"], []
            cols = ["G_round2"]
            tasks = tgt.get("tasks") or []
            text = str((tasks[0] or {}).get("task") or "") if tasks else ""
            scope = text.split("Scope:", 1)[1] if "Scope:" in text else ""
            first = scope.split(";", 1)[0].lower()
            if "image" in first and "compar" in first:
                cols.append("G_round2_scope_compare")
            return cols, []
        cols = ["G_finish"] + (["G_finish_after_round2"] if rule == "G3_finish_after_round2" else [])
        return cols + habit_cols(statements(tgt), tgt, fam), []
    if md.get("unit") == "report":
        status = tgt.get("status")
        if status == "COMPLETED":
            cols = ["P_completed"]
        elif md.get("status_inloop_rewritten"):
            cols = ["P_inloop_rewritten"]
        elif md.get("status_rewritten"):
            cols = ["P_posthoc_rewritten"]
        else:
            cols = ["P_incomplete_walker"]
        transport = rc.load(run, request, "transport.json")
        return cols + habit_cols(statements(tgt), tgt, fam) + report_rf_cols(tgt, fam, transport), []
    cards = rc.cards(run, request)
    cols, form = action_forms(tgt, cards)
    if tgt.get("action") == "READ_RECORD":
        cols = [read_record_col(tgt, md, cards)]
    return cols, form


def norm_rationale(s: Any) -> str:
    return " ".join(str(s or "").split())


def entity_misses(answer: str, visible: str) -> list[str]:
    vis_nums = {n.replace(",", "").lstrip("0") or "0" for n in NUM_RE.findall(visible)}
    vis_nums |= {p.lstrip("0") or "0" for n in NUM_RE.findall(visible) for p in re.split(r"[.,]", n) if p}
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).split("](")[0][1:], answer)   # markdown 链接只留文字
    text = EID_RE.sub(" ", re.sub(r"https?://\S+", " ", text))
    misses = []
    for n in NUM_RE.findall(text):
        k = n.replace(",", "").lstrip("0") or "0"
        if k not in vis_nums and k.rstrip(".") not in vis_nums:
            misses.append(n)
    low = visible.lower()
    for sent in re.split(r"(?<=[.!?:;])\s+|\n+", text):
        for m in CAPS_RE.finditer(sent):
            if not sent[:m.start()].strip(" \"“‘'(—–-"):
                continue                                         # 句首（含引号、括号、破折号后）的大写词不算专名
            phrase = re.sub(r"(?:’s|'s|’|')$", "", m.group(1)).strip("’'-")
            parts = [w for p in phrase.split() for w in re.split(r"[-–]", re.sub(r"(?:’s|'s)$", "", p))
                     if w and w not in STOP_CAPS and len(w) >= 3]
            if not parts:
                continue
            if phrase.lower() not in low and any(w.lower() not in low for w in parts):
                misses.append(" ".join(parts))
    return misses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", nargs=4, action="append", metavar=("NAME", "EXPORT", "RUNS", "TEACHER"), required=True)
    ap.add_argument("--unanswerable", nargs="*", default=[])
    ap.add_argument("--out", required=True, help="输出前缀（写 .json 与 .md）")
    args = ap.parse_args(argv)

    unans: set[str] = set()
    for p in args.unanswerable:
        data = json.loads((ROOT / p).read_text(encoding="utf-8"))
        unans |= set(data.get("unanswerable_case_ids") or [])

    cells: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"rows": 0, "cases": set(), "teacher": Counter(), "packs": Counter()})
    produced: dict[tuple[str, str], Counter] = defaultdict(Counter)       # (族, 列) → 桶计数（含 held / rejected）
    forms_seen: Counter = Counter()
    forms_any: Counter = Counter()
    rationales: list[tuple[str, str, str, str]] = []                     # (pack, teacher, rationale, source)
    match_rows: list[dict[str, Any]] = []
    n_rationale = Counter()
    gia_answers: list[dict[str, Any]] = []
    totals = Counter()
    group_cases: dict[tuple[str, str], set] = defaultdict(set)
    group_rows: Counter = Counter()

    for name, export, runs, teacher in args.pack:
        export_dir, rc = ROOT / export, RunCache(ROOT / runs)
        for role in ("hi", "gia"):
            for bucket in ("accepted", "held", "rejected"):
                for row in jsonl(export_dir / f"{role}_{bucket}.jsonl"):
                    md = row.get("metadata") or {}
                    case = str(md.get("case_id"))
                    fam = family(case)
                    ans = "unans" if case in unans else "ans"
                    cols, form = row_columns(row, rc)
                    for c in cols:
                        produced[(fam, c)][bucket] += 1
                    forms_any.update(form)
                    if bucket != "accepted":
                        continue
                    totals[(name, role)] += 1
                    group_rows[(fam, ans)] += 1
                    group_cases[(fam, ans)].add(case)
                    forms_seen.update(form)
                    for c in cols:
                        cell = cells[(fam, ans, c)]
                        cell["rows"] += 1
                        cell["cases"].add(case)
                        cell["teacher"][teacher] += 1
                        cell["packs"][name] += 1
                    tgt = target(row)
                    rat = norm_rationale(tgt.get("rationale"))
                    if rat:
                        rationales.append((name, teacher, rat, str(md.get("rationale_source") or "")))
                    if role == "hi" and md.get("unit") == "step" and rat:
                        n_rationale[name] += 1
                        if MATCH_RE.search(rat):
                            cards = rc.cards(str(md.get("run")), str(md.get("request")))
                            had_compare = any(str(c.get("acquired_by") or "").startswith("INSPECT/compare_media") for c in cards.values()) \
                                or any(op == "INSPECT:compare_media" for op in rc.history_ops(str(md.get("run")), str(md.get("request"))))
                            if not had_compare:
                                is_cmp = (tgt.get("arguments") or {}).get("request", {}).get("operation") == "compare_media"
                                match_rows.append({"pack": name, "teacher": teacher, "run": md.get("run"), "request": md.get("request"),
                                                   "rule": md.get("walker_rule"), "rationale": rat,
                                                   "assertive": not is_cmp and not HEDGE_RE.search(rat)})
                    if role == "gia" and md.get("unit") == "answer":
                        miss = entity_misses(str(tgt.get("answer") or ""), visible_text(row))
                        gia_answers.append({"pack": name, "teacher": teacher, "run": md.get("run"), "family": fam, "misses": miss})

    # ---- 矩阵
    matrix = {}
    gaps = []
    for fam, ans in GROUPS:
        for key, grp, label in COLUMNS:
            cell = cells.get((fam, ans, key))
            prod = produced.get((fam, key)) or Counter()
            applicable = sum(prod.values()) > 0
            rows = cell["rows"] if cell else 0
            ncase = len(cell["cases"]) if cell else 0
            status = "n/a" if not applicable else ("gap" if rows < GAP_ROWS or ncase < GAP_CASES else "ok")
            matrix[f"{fam}|{ans}|{key}"] = {"rows": rows, "cases": ncase, "teacher": dict(cell["teacher"]) if cell else {},
                                             "packs": dict(cell["packs"]) if cell else {}, "produced_all_buckets": dict(prod),
                                             "status": status}
            if status == "gap":
                gaps.append({"family": fam, "answerable": ans, "column": key, "label": label, "rows": rows, "cases": ncase,
                             "held_or_rejected": prod.get("held", 0) + prod.get("rejected", 0)})

    # ---- 坏习惯
    rat_counts = Counter(r for _, _, r, _ in rationales)
    tmpl_rows = [x for x in rationales if rat_counts[x[2]] >= 3]
    s0_rows = [x for x in rationales if x[3] == "s0_teacher"]
    s0_tmpl = [x for x in s0_rows if rat_counts[x[2]] >= 3]
    by_teacher = {t: {"rows": sum(1 for x in rationales if x[1] == t), "templated": sum(1 for x in tmpl_rows if x[1] == t),
                      "s0_rows": sum(1 for x in s0_rows if x[1] == t), "s0_templated": sum(1 for x in s0_tmpl if x[1] == t)}
                  for t in sorted({x[1] for x in rationales})}
    # 另一口径：E / R / F 编号与数字换成占位符后再数（"Read E9 to …" 与 "Read E12 to …" 算同一句）
    def skeleton(r: str) -> str:
        return re.sub(r"\d+(?:\.\d+)?", "N", EID_RE.sub("#", r))
    sk_counts = Counter(skeleton(r) for _, _, r, _ in rationales)
    sk_rows = [x for x in rationales if sk_counts[skeleton(x[2])] >= 3]
    habits = {
        "templated_rationale": {"rows_with_rationale": len(rationales), "templated_rows": len(tmpl_rows),
                                "skeleton_templated_rows": len(sk_rows), "skeleton_share": round(len(sk_rows) / max(1, len(rationales)), 4),
                                "skeleton_by_teacher": {t: sum(1 for x in sk_rows if x[1] == t) for t in sorted({x[1] for x in rationales})},
                                "skeleton_top": [{"n": n, "rationale": r[:200]} for r, n in sk_counts.most_common(8) if n >= 3],
                                "share": round(len(tmpl_rows) / max(1, len(rationales)), 4),
                                "s0_rows": len(s0_rows), "s0_templated": len(s0_tmpl), "s0_share": round(len(s0_tmpl) / max(1, len(s0_rows)), 4),
                                "by_teacher": by_teacher,
                                "top": [{"n": n, "rationale": r[:200]} for r, n in rat_counts.most_common(12) if n >= 3]},
        "match_before_compare": {"hi_step_rows_with_rationale": dict(n_rationale), "flagged": len(match_rows),
                                 "share": round(len(match_rows) / max(1, sum(n_rationale.values())), 4),
                                 "assertive": sum(1 for m in match_rows if m["assertive"]),
                                 "assertive_share": round(sum(1 for m in match_rows if m["assertive"]) / max(1, sum(n_rationale.values())), 4),
                                 "by_pack": {p: {"flagged": sum(1 for m in match_rows if m["pack"] == p),
                                                 "assertive": sum(1 for m in match_rows if m["pack"] == p and m["assertive"])}
                                             for p in n_rationale},
                                 "examples": [m for m in match_rows if m["assertive"]][:12]},
        "gia_unseen_entities": {"answers": len(gia_answers), "with_miss": sum(1 for a in gia_answers if a["misses"]),
                                "share": round(sum(1 for a in gia_answers if a["misses"]) / max(1, len(gia_answers)), 4),
                                "by_pack": {p: {"answers": sum(1 for a in gia_answers if a["pack"] == p),
                                                "with_miss": sum(1 for a in gia_answers if a["pack"] == p and a["misses"])}
                                            for p in sorted({a["pack"] for a in gia_answers})},
                                "examples": [a for a in gia_answers if a["misses"]][:20]},
    }
    unseen_forms = [f for f in CATALOG if not forms_seen.get(f)]
    out = {"packs": [{"name": n, "export": e, "runs": r, "teacher": t} for n, e, r, t in args.pack],
           "totals": {f"{k[0]}:{k[1]}": v for k, v in totals.items()},
           "groups": {f"{f}|{a}": {"rows": group_rows[(f, a)], "cases": len(group_cases[(f, a)])} for f, a in GROUPS},
           "columns": [{"key": k, "group": g, "label": lab} for k, g, lab in COLUMNS], "matrix": matrix, "gaps": gaps,
           "catalog_forms_seen": dict(forms_seen), "catalog_forms_any_bucket": dict(forms_any),
           "catalog_never_in_accepted": unseen_forms,
           "catalog_never_produced": [f for f in CATALOG if not forms_any.get(f)],
           "noncatalog_forms_seen": {f: n for f, n in forms_seen.items() if f not in CATALOG},
           "habits": habits, "gap_rule": {"rows_lt": GAP_ROWS, "cases_lt": GAP_CASES}}
    prefix = ROOT / args.out
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=sorted), encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_md(out), encoding="utf-8")
    print(f"写出 {prefix.with_suffix('.json')} / .md；缺口格 {len(gaps)}；目录里 accepted 没出现的形态 {len(unseen_forms)}")
    return 0


def render_md(out: dict[str, Any]) -> str:
    head = "| 组 | 列 | " + " | ".join(f"{f}{'' if a == 'ans' else '（不可回答）'}" for f, a in GROUPS) + " |"
    lines = ["# 教学矩阵（accepted 行 / 案例；Q = Qwen 版 S0–S3 行，X = Codex 版行；⚠ 缺口：行 < 15 或案例 < 5；— 该族 walker 不走）", "",
             "| 族 | accepted 行 | 案例 |", "|---|---|---|"]
    for f, a in GROUPS:
        g = out["groups"][f"{f}|{a}"]
        lines.append(f"| {f}{'' if a == 'ans' else '（不可回答）'} | {g['rows']} | {g['cases']} |")
    lines += ["", head, "|" + "---|" * (2 + len(GROUPS))]
    for col in out["columns"]:
        cells = []
        for f, a in GROUPS:
            c = out["matrix"][f"{f}|{a}|{col['key']}"]
            if c["status"] == "n/a":
                cells.append("—")
                continue
            t = c["teacher"]
            mark = "⚠ " if c["status"] == "gap" else ""
            cells.append(f"{mark}{c['rows']}/{c['cases']}（Q{t.get('qwen', 0)} X{t.get('codex', 0)}）")
        lines.append(f"| {col['group']} | {col['label']} | " + " | ".join(cells) + " |")
    lines += ["", f"工具目录里有、accepted 一次都没出现的形态（{len(out['catalog_never_in_accepted'])}）："]
    for f in out["catalog_never_in_accepted"]:
        n = out["catalog_forms_any_bucket"].get(f, 0)
        lines.append(f"- `{f}`" + (f"（held / rejected 里有 {n} 行）" if n else "（三个桶里都没有：walker 从来不走）"))
    h = out["habits"]
    lines += ["", "## 坏习惯（程序口径）", "",
              f"- 理由模板化：{h['templated_rationale']['templated_rows']}/{h['templated_rationale']['rows_with_rationale']} = "
              f"{h['templated_rationale']['share']:.1%}；只算 S0 教师写的 {h['templated_rationale']['s0_templated']}/{h['templated_rationale']['s0_rows']} = "
              f"{h['templated_rationale']['s0_share']:.1%}；分教师 {json.dumps(h['templated_rationale']['by_teacher'], ensure_ascii=False)}；"
              f"编号 / 数字换占位符后 {h['templated_rationale']['skeleton_templated_rows']} 行 = {h['templated_rationale']['skeleton_share']:.1%}"
              f"（{json.dumps(h['templated_rationale']['skeleton_by_teacher'], ensure_ascii=False)}）",
              f"- 比对前就说匹配：{h['match_before_compare']['flagged']} 行（{h['match_before_compare']['share']:.1%}）；断言式 "
              f"{h['match_before_compare']['assertive']}（{h['match_before_compare']['assertive_share']:.1%}）；分包 "
              f"{json.dumps(h['match_before_compare']['by_pack'], ensure_ascii=False)}",
              f"- GIA 答案有可见输入里没有的实体：{h['gia_unseen_entities']['with_miss']}/{h['gia_unseen_entities']['answers']} = "
              f"{h['gia_unseen_entities']['share']:.1%}；分包 {json.dumps(h['gia_unseen_entities']['by_pack'], ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
