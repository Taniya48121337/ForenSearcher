"""walker 工具覆盖表（2026-10-13）：一轮里每个工具 / 参数变体实际用了几次，对照"该在哪里用、触发条件、最少几次"。

    python scripts\\walker_coverage.py --round wk8            # → experiments/sft_data/walker_wk8_coverage.md / .json

方法（见 claude/walker_wk8_日期查询与工具覆盖_20261013.md）：
1. 每个工具变体先写清它解决的证据需求、学生视图里的触发条件、自然出现在哪类案例；
2. 按这张表挑案例（每个变体至少有一类案例会自然触发），walker 只在触发条件成立时用它；
3. 跑完按实际动作计数（读 response.json，不读 walker 的规则名），低于最少次数的列出来；
4. 补法只能是加案例或加环境变体，不能让 walker 在没有触发条件时硬用（那会教出 wrong_tool）；
5. 最少次数为 0 的是"条件触发"的变体（触发条件取决于数据：PDF 是扫描件、页面给出视频链接、第 2 轮……），本轮没遇到写
   "条件未出现"，不算缺；没有"有意不用"的变体——在当前工具组合下没有合理用途的参数已并入默认（hi@0.10.0
   compact_tool_params：compare_media 的 mode=audio、SEARCH 的 include=subtitles / subtitle_language）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# 变体 → (合理用途 / 触发条件, 自然出现的案例族, 本轮最少次数；0 = 条件触发，取决于数据)
MATRIX: list[tuple[str, str, str, int]] = [
    ("SEARCH text→video", "找原片：用 OCR / 语音里的可识别线索搜视频", "A", 3),
    ("SEARCH text_image→video", "找原片：文字查询没对上时，文字 + 一帧同时搜（文字搜索与以图搜图合并）", "A", 1),
    ("SEARCH image→video", "找原片：文字与图文都没对上时只用帧搜", "A", 0),
    # 10-18 负责人定：r1 起 E / L 的反向图片搜索走 image→image（E2_reverse_image / L4_reverse_image），没有规则再发 image→text，
    # 这一行是口径遗留——改成条件触发（最少 0），不算缺口（《长尾分支补量方案》五节 ⑥）
    ("SEARCH image→text", "图片最早出处 / 照片拍摄地：反向图片搜索找载有同一张图的网页（r1 起 E/L 走 image→image；条件触发）", "E L", 0),
    ("SEARCH text→text", "核查声明 / 图片出处：搜事件本身的报道", "C E L", 5),
    ("SEARCH text→image", "照片拍摄地：声明点名了地点 / 地标时，取该地点的参考照片与给定照片当面比对", "L", 3),
    ("SEARCH open dates", "要回答谁先发布 / 某来源是否早于声明：查页面声明日期、上传日期、最早存档", "E A C", 5),
    ("SEARCH open video", "读到的页面给出声明所说视频的链接时打开（hi@0.10.0 起自动带原语言字幕，原 include=subtitles 并入）", "C", 0),
    ("SEARCH open URL/R", "打开一个已知网页来源（非视频、非日期）", "—", 0),
    ("INSPECT get_frames count", "看不到原片对应时两侧均匀取帧；以图找帧前取样", "B", 1),
    ("INSPECT get_frames times_s", "在比对报出的边界 / 对应时刻两侧取帧核看", "B A", 8),
    ("INSPECT get_frames times_s+box", "同一时刻画面不同的小区域：两侧放大", "B A", 2),
    ("INSPECT get_frames pages", "读到的 PDF 没有文字层（扫描件）：渲染页面再看", "C E", 0),
    ("INSPECT ocr(image)", "放大后的横条区域（字幕、标题、标牌）读字；给定图片读水印 / 字幕 / 标牌", "B E L", 6),
    ("INSPECT ocr(video)", "找原片 / 核查声明：整段材料的画面文字（台标、字幕、名牌）", "A C", 3),
    ("INSPECT asr", "材料有音轨且任务涉及说了什么，或需要语音里的线索", "A B C", 5),
    ("INSPECT locate(text)", "按描述在单段视频里找片段", "B(qv)", 1),
    ("INSPECT locate(image)", "对应类任务整段比对没对上：拿 A 的帧去 B 里找", "B(vcsl)", 1),
    ("INSPECT video_structure", "单段视频按描述定位时先看镜头表，按切点取边界帧", "B(qv)", 1),
    ("INSPECT compare_media", "两段视频（给定的 B / 找到的原片 / 页面链接的视频）整段比对；hi@0.10.0 起画面对不上且两边都有声音时"
                              "自动只比声音（原 mode=audio 并入）", "A B C", 6),
    ("INSPECT compare_media range", "等长短视频要逐帧精确边界时限定 A 全长再比", "B", 1),
    ("READ_RECORD text", "检索自动打开的页面、打开视频带的字幕：读正文", "C E L", 5),
    ("READ_RECORD offset>0", "正文没读完（next_offset 非空）且要求项还没覆盖", "C", 0),
    ("READ_RECORD focus", "比对图 / 帧没附上时带 focus 回看；参考照片与给定照片并排看", "B L", 3),
    ("QUERY_MEMORY (HI)", "交报告前要引用的 E 已掉出视图（只剩编号）：按标签查回来；V9 变体（max_evidence_cards=6 + max_history_entries=4）里自然出现", "B(v9)", 1),
    ("GIA INVESTIGATE round1", "每个案例第 1 轮", "全部", 1),
    ("GIA INVESTIGATE round2", "报告留下能靠更多查看补上的缺口（B 族时间窗 / 音轨）", "B", 1),
    ("GIA READ_RECORD", "报告引用的图不在本次附图里", "全部", 2),
    ("GIA QUERY_MEMORY", "第 2 轮之后第 1 轮报告不在视图里", "B(第 2 轮)", 0),
    ("GIA FINISH", "每个案例结束", "全部", 1),
]


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def variant(action: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """一步实际动作 → 变体名。10-15 起与 export 写进每行 metadata.branch 的是同一个函数（datagen.export.decision_branch），
    报告行另记 report_completed / report_incomplete（这里覆盖表只数动作，报告不计）。"""

    from foren_searcher.datagen.export import decision_branch

    if "findings" in action and "status" in action:
        return None
    return decision_branch(action, payload)


# --------------------------------------------------------------------------- 10-15：按导出行出 by_branch 表（清单 C4）

def family_of(case_id: str | None) -> str:
    """案例 → walker 族：A 找原片（fmnv a_）、B 给定媒体比对（videosham / vcsl / qvhighlights）、C 视频声明核查（fmnv c_、
    GroundLie360）、E 图片出处（AVerImaTeC e*）、L 拍摄地（AVerImaTeC l*）；其余（r0e 的 true_* / averitec_* 等）记 other。"""

    cid = str(case_id or "").lower()
    dataset, _, source = cid.partition(":")
    if dataset == "fmnv":
        return "A" if source.startswith("a_") else "C"
    if dataset in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if dataset == "groundlie360":
        return "C"
    if dataset == "averimatec" and source[:1] in ("e", "l") and source[1:2].isdigit():
        return source[0].upper()
    return "other"


# 纳入族的核心链路最低行数（《第一次SFT开训清单_20261015》C4）；名字 → (族, 说明, 最少行数)
CHAIN_MINIMUMS: dict[str, tuple[str, str, int]] = {
    "B compare→读差分→取帧": ("B", "工作区里 compare_media 及其后的 READ_RECORD focus / get_frames 行", 15),
    "C READ_RECORD": ("C", "C 族全部 READ_RECORD 行", 40),
    "C SEARCH page": ("C", "C 族 SEARCH 输出为网页（→text）的行", 20),
    "E 日期查询+读最早页": ("E", "E 族 SEARCH open dates 行 + 紧随其后的 READ_RECORD 行", 10),
}
# export 规则本身就不收的分支（不是缺口）：C3 照抄总任务的第 1 轮派工默认不进；GIA 记忆动作按设计 held
BY_DESIGN_ZERO: dict[str, str] = {
    "GIA INVESTIGATE round1": "C3：walker 第 1 轮照抄总任务，r0 默认不进（--plan-copy downweight 才以 ×0.3 进）",
    "GIA READ_RECORD": "GIA 记忆动作按设计 held（memory_action_follows_next_decision）",
    "GIA QUERY_MEMORY": "同上",
}
# 首批允许为零的分支（清单 C4）：出现 0 行不算不过
ALLOWED_ZERO: dict[str, str] = {
    "A 族链路": "A 族本轮不进（wk13 前 A 族 0 正例，S3 修复后再放量）",
    "INSPECT get_frames pages": "get_pages：扫描件 PDF 条件触发",
    "SEARCH image→video": "以图搜视频：文字与图文都没对上时才用",
    "INSPECT compare_media audio": "音轨比对：等 compare_media 逐段音轨（M 族），本轮不排",
}


def _accepted_rows(export_dir: Path) -> list[dict[str, Any]]:
    """accepted 行的 metadata；旧导出没有 metadata.branch 时，从行里的教师输出补算一个（存进 _branch_fallback）。"""

    rows: list[dict[str, Any]] = []
    for name in ("hi_accepted.jsonl", "gia_accepted.jsonl"):
        path = export_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta = dict(row["metadata"])
            if not meta.get("branch"):
                try:
                    output = json.loads(row["messages"][-1]["content"])
                except (ValueError, KeyError, TypeError):
                    output = None
                if isinstance(output, dict):
                    # 渲染后的行里没有原始 payload：GIA 轮次取 metadata.round，OCR 的模态判不出（记 ocr(image)）
                    stub = ({"workspace_id": meta.get("workspace"), "model_view": {}} if meta.get("role") == "hi"
                            else {"global_view": {"runtime_constraints": {"round_index": meta.get("round") or 1}}})
                    meta["_branch_fallback"] = variant(output, stub) or (
                        "report_" + str(output.get("status")).lower() if "findings" in output else None)
            rows.append(meta)
    return rows


def _row_branch(m: dict[str, Any]) -> str | None:
    return m.get("branch") or m.get("_branch_fallback") or m.get("walker_branch")


def export_branch_table(export_dir: Path) -> dict[str, Any]:
    """按导出目录的 accepted 行出 by_branch：每族 × 分支行数、核心链路行数与是否达最低、允许为零的分支。

    分支名优先用 export 写的 metadata.branch（与覆盖表同一套名字）；旧导出没有这个字段时退回 walker_branch，再没有记 unknown。"""

    metas = _accepted_rows(export_dir)
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for m in metas:
        table[family_of(m.get("case_id"))][_row_branch(m) or f"unknown:{m.get('action')}"] += 1
    chains: dict[str, int] = {name: 0 for name in CHAIN_MINIMUMS}
    by_ws: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in metas:
        if m.get("role") == "hi" and m.get("unit") == "step":
            by_ws[(str(m.get("run")), str(m.get("workspace")))].append(m)
    for items in by_ws.values():
        items.sort(key=lambda r: int(r.get("turn") or 0))
        fam = family_of(items[0].get("case_id"))
        branches = [(_row_branch(m) or "") for m in items]
        if fam == "B":
            started = False
            for b in branches:
                if b.startswith("INSPECT compare_media"):
                    started = True
                if started and (b.startswith("INSPECT compare_media") or b == "READ_RECORD focus" or b.startswith("INSPECT get_frames")):
                    chains["B compare→读差分→取帧"] += 1
        if fam == "C":
            chains["C READ_RECORD"] += sum(1 for b in branches if b.startswith("READ_RECORD"))
            chains["C SEARCH page"] += sum(1 for b in branches if b.startswith("SEARCH ") and b.endswith("→text"))
        if fam == "E":
            for i, b in enumerate(branches):
                if b == "SEARCH open dates":
                    chains["E 日期查询+读最早页"] += 1
                    if i + 1 < len(branches) and branches[i + 1].startswith("READ_RECORD"):
                        chains["E 日期查询+读最早页"] += 1
    chain_rows = [{"chain": name, "family": fam, "what": what, "rows": chains[name], "minimum": minimum,
                   "ok": chains[name] >= minimum} for name, (fam, what, minimum) in CHAIN_MINIMUMS.items()]
    all_branches = Counter(b for fam in table.values() for b in fam.elements())
    zero = []
    for name, why in ALLOWED_ZERO.items():
        n = sum(table["A"].values()) if name == "A 族链路" else all_branches.get(name, 0)
        zero.append({"branch": name, "rows": n, "why": why})
    expected = [m[0] for m in MATRIX if m[3] > 0]
    missing = [b for b in expected if all_branches.get(b, 0) == 0 and b not in ALLOWED_ZERO and b not in BY_DESIGN_ZERO]
    return {"families": {fam: dict(c.most_common()) for fam, c in sorted(table.items())}, "chains": chain_rows,
            "allowed_zero": zero, "expected_but_zero": missing, "rows": len(metas)}


def branch_table_markdown(result: dict[str, Any]) -> list[str]:
    lines = ["| 族 | 分支 | accepted 行 |", "|---|---|---|"]
    for fam, counts in result["families"].items():
        for branch, n in counts.items():
            lines.append(f"| {fam} | {branch} | {n} |")
    lines += ["", "| 核心链路 | 说明 | 行数 | 最低 | 通过 |", "|---|---|---|---|---|"]
    for c in result["chains"]:
        lines.append(f"| {c['chain']} | {c['what']} | {c['rows']} | {c['minimum']} | {'是' if c['ok'] else '否'} |")
    lines += ["", "| 允许为零的分支 | 本批行数 | 理由 |", "|---|---|---|"]
    for z in result["allowed_zero"]:
        lines.append(f"| {z['branch']} | {z['rows']} | {z['why']} |")
    if result["expected_but_zero"]:
        lines += ["", "覆盖表要求 ≥1 次、本批 accepted 为 0 且不在允许为零清单里的分支：" + "、".join(result["expected_but_zero"])]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", help="按 run 目录数一轮的动作（原用法）")
    parser.add_argument("--export", help="10-15：按导出目录的 accepted 行出 by_branch 表（清单 C4），写 <export>/by_branch.md/.json")
    args = parser.parse_args()
    if args.export:
        export_dir = Path(args.export)
        result = export_branch_table(export_dir)
        lines = [f"# by_branch：{export_dir.name}", "", f"accepted {result['rows']} 行。", "", *branch_table_markdown(result)]
        (export_dir / "by_branch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (export_dir / "by_branch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n".join(lines))
        return 0
    if not args.round:
        parser.error("需要 --round 或 --export")
    counts: Counter[str] = Counter()
    where: dict[str, set[str]] = defaultdict(set)
    runs = sorted(p for p in RUNS.glob(f"*_{args.round}_1") if p.is_dir())
    for run in runs:
        case = run.name[: -len(f"_{args.round}_1")]
        for req in sorted((run / "requests").glob("request_*")):
            payload = (_load(req / "request.json") or {}).get("payload") or {}
            output = _load(req / "response.json") or {}
            if not isinstance(output, dict) or "findings" in output:
                continue
            name = variant(output, payload)
            if name:
                counts[name] += 1
                where[name].add(case)
    known = {m[0] for m in MATRIX}
    lines = [f"# walker {args.round} 工具覆盖", "", f"{len(runs)} 个 run。计数按实际动作（response.json）。", "",
             "| 工具 / 变体 | 合理用途（触发条件） | 自然出现的案例族 | 本轮次数 | 最少 | 状态 | 用到的案例（前 4 个） |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for name, why, family, minimum in MATRIX:
        n = counts.get(name, 0)
        state = ("条件未出现" if minimum == 0 and n == 0 else "够" if n >= max(minimum, 1) else "不够" if minimum else "用了")
        cases = ", ".join(sorted(where.get(name, ()))[:4])
        lines.append(f"| {name} | {why} | {family} | {n} | {minimum} | {state} | {cases} |")
        rows.append({"variant": name, "why": why, "family": family, "count": n, "minimum": minimum, "state": state,
                     "cases": sorted(where.get(name, ()))})
    extra = {k: v for k, v in counts.items() if k not in known}
    if extra:
        lines += ["", "表外的变体：" + "；".join(f"{k} ×{v}" for k, v in sorted(extra.items()))]
    short = [r["variant"] for r in rows if r["state"] == "不够"]
    lines += ["", f"不够的：{'、'.join(short) or '无'}。补法只能是加案例或环境变体，不改 walker 去硬用。"]
    out = SFT / f"walker_{args.round}_coverage"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "extra": extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
