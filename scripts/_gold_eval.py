"""人工金标集第二步：拿人工标注对照核验器判定，算一致率、精确率、召回和 κ。

    python scripts/_gold_eval.py --batch experiments/sft_data/gold_set/v1 --labels labels_zya.json [labels_b.json ...]
        [--report-verdicts experiments/sft_data/verdicts_r0_judged.jsonl]
        [--unit-verdicts experiments/sft_data/verdicts_units_judged.jsonl]

可以换任何一份判定文件（例如本地 gemma 核验的 verdicts_*.local-*.jsonl），看不同核验器对同一批金标的表现。
结果写到 <batch>/对照_<判定文件名>.md，并打印。

口径：
- Finding：人工"部分支持 / 不支持"= 有问题；核验器对这条 Finding 记了 major/critical 错误 = 标错。"无法判断"不计。
  报告"核验器标错"的精确率（标错里人也认为有问题的比例）和召回（人认为有问题的里核验器标出的比例）。
- 答案 / 计划：人工"可接受、有小问题仍可接受 / 合理、基本合理"= positive，"不可接受 / 不合理"= negative，对照核验器 disposition。
- 任务级（v2 --plan-tasks 批次，kind=plan_task）：主判定"该不该派"对照该计划单元 disposition；另按错误类型逐任务对照——
  人工 A 题"有夹带"↔ 核验器对这个 T 记了 premise_smuggling，人工 B 题"不针对缺口"↔ 记了 task_not_gap_driven（"不确定"不计）。
- 多个标注人时另算两两 κ（主判定）。
- 每类条目的人工用时中位数（分钟），用于估每份金标的人工成本。

10-15（r0a 批次，清单 D1）另出 major/critical 分层：
- Finding 分"原始判定"与"校准后"（verify/calibrate.recalibrate，export 实际用的）两套标错集合，再各加一行"含 minor"，
  精确率 / 召回带 Wilson 95% 区间——r0a 标错只有 3 条，精确率样本太少，要和召回、"未标错里人工认为有错"的条数一起看。
- 答案 / 计划的"标错"= 有 major/critical 错误（与 export B1 的 held 口径一致），和 disposition 那一行并列。
- 分错误类型：核验器每类标错里人工认同几条；人工勾的每类错误核验器标出几条。漏报 / 误报逐条列出（带人工备注）。
- 被校准规则撤销了 major/critical 的 Finding 单列（如 videosham_1028 W1-1 F1 被 R4 撤销的 sloppy_numeric）。
- 判定文件默认取批次 items.json 的 meta（report_verdicts / unit_verdicts，相对 experiments/sft_data）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from foren_searcher.datagen.verify.calibrate import recalibrate
    from foren_searcher.datagen.verify.verdict import Verdict
    from foren_searcher.datagen.errors import SPECS, ErrorType
except Exception:  # noqa: BLE001 —— 没有 src 时只出原始判定那一套
    recalibrate = Verdict = SPECS = ErrorType = None

SEVERE = ("major", "critical")
D1_PRECISION = 0.85
POS ={"finding": {"full"}, "answer": {"acceptable", "minor"}, "plan": {"good", "minor"}, "plan_task": {"good", "minor"}}
NEG = {"finding": {"partial", "none"}, "answer": {"unacceptable"}, "plan": {"bad"}, "plan_task": {"bad"}}
TASK_LEVEL = {"premise_smuggling": ("premise", "smuggled", "clean"), "task_not_gap_driven": ("gap", "no_gap", "gap")}


def task_types(unit: dict[str, Any], label: str) -> set[str]:
    return {e["type"] for e in unit.get("errors") or [] if (e.get("extra") or {}).get("task") == label}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def pr(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    """rows: (人工认为有问题, 核验器认为有问题)。"""

    tp = sum(h and v for h, v in rows)
    fp = sum((not h) and v for h, v in rows)
    fn = sum(h and (not v) for h, v in rows)
    tn = sum((not h) and (not v) for h, v in rows)
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "agreement": round((tp + tn) / len(rows), 3) if rows else None, "kappa": kappa(rows)}


def wilson(k: int, n: int) -> str:
    """二项比例的 Wilson 95% 区间，样本少时比正态近似靠谱。"""

    if not n:
        return "—"
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"[{max(0.0, centre - half):.2f}, {min(1.0, centre + half):.2f}]"


def pr_row(name: str, rows: list[tuple[bool, bool]]) -> str:
    m = pr(rows)
    return (f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} {wilson(m['tp'], m['tp'] + m['fp'])} "
            f"| {m['recall']} {wilson(m['tp'], m['tp'] + m['fn'])} | {m['kappa']} | {m['fn']} / {m['fn'] + m['tn']} |")


PR_HEAD = ["| 口径 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 [95%] | 召回 [95%] | κ | 未标错里人工认为有错 |",
           "|---|---|---|---|---|---|---|---|---|---|"]


def finding_errors(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for err in row.get("errors") or []:
        fid = (err.get("extra") or {}).get("finding")
        if fid:
            out[fid].append(err)
    return out


def calibrated(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(校准后的判定 dict, 规则撤销 / 改写记录)。没有 src 时原样返回。"""

    if recalibrate is None or row.get("unit") not in ("report", "answer"):
        return row, []
    v = recalibrate(Verdict.from_dict(row))
    return v.to_dict(), list((v.extra.get("calibration") or {}).get("applied") or [])


def severity_of(err: dict[str, Any]) -> str | None:
    """从核验日志恢复的 plan 错误只有 type，按错误类型的默认严重度补（结果表里注明）。"""

    if err.get("severity") or SPECS is None:
        return err.get("severity")
    try:
        return SPECS[ErrorType(err["type"])].default_severity.value
    except (KeyError, ValueError):
        return None


def severe_types(errors: list[dict[str, Any]], minor: bool = False) -> list[str]:
    return [e["type"] for e in errors if severity_of(e) in SEVERE or (minor and severity_of(e) == "minor")]


def layered_section(ann: dict[str, Any], items: dict[str, Any], f_layers: dict[str, dict[tuple[str, str], list[str]]],
                    f_ws: dict[tuple[str, str], str], revoked: dict[tuple[str, str], list[dict[str, Any]]],
                    unit_disp: dict[tuple, dict[str, Any]], unit_severe: dict[tuple, list[str]]) -> list[str]:
    """10-15 清单 D1：major/critical 分层、分错误类型、漏报 / 误报清单、校准撤销单列、D1 判定。"""

    labels = {l["id"]: l for l in ann["labels"] if l.get("id") in items and l.get("main")}
    fkey = lambda it: (it["run"], it["finding_id"])  # noqa: E731
    fl = [(items[i], l) for i, l in labels.items() if items[i]["kind"] == "finding"
          and l["main"] in POS["finding"] | NEG["finding"] and fkey(items[i]) in f_layers["raw"]]
    bad = lambda l: l["main"] in NEG[items[l["id"]]["kind"]]  # noqa: E731
    note = lambda l: (l.get("note") or "").replace("|", "/").replace("\n", " ")  # noqa: E731
    out: list[str] = []
    if fl:
        names = {"raw": "原始判定 · major/critical", "cal": "校准后（export 用）· major/critical",
                 "raw+minor": "原始判定 · 含 minor", "cal+minor": "校准后 · 含 minor"}
        out += ["### Finding · major/critical 分层（清单 D1）", "", *PR_HEAD]
        for layer in ("raw", "cal", "raw+minor", "cal+minor"):
            out.append(pr_row(names[layer], [(bad(l), bool(f_layers[layer][fkey(it)])) for it, l in fl]))
        out += ["", "\"含 minor\" 两行是假设 export 把 minor 也 held 时的效果，供 D1 收紧参考。标错条数少时精确率区间很宽，要和召回、最后一列一起看。", ""]

        by_vt: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for it, l in fl:
            for t in sorted(set(f_layers["raw"][fkey(it)])):
                by_vt[t].append((it, l))
        if by_vt:
            out += ["| 核验器标错类型（原始 major/critical） | 标错 | 人工认为有问题 | 人工认为没问题 | 条目（✗ = 被校准撤销） |", "|---|---|---|---|---|"]
            for t, xs in sorted(by_vt.items()):
                ids = "、".join(it["id"] + ("" if t in f_layers["cal"][fkey(it)] else " ✗") + f"（{l['main']}）" for it, l in xs)
                out.append(f"| {t} | {len(xs)} | {sum(bad(l) for _, l in xs)} | {sum(not bad(l) for _, l in xs)} | {ids} |")
            out.append("")
        by_ht: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for it, l in fl:
            if bad(l):
                for t in (l.get("errors") or ["（未勾类型）"]):
                    by_ht[t].append((it, l))
        if by_ht:
            out += ["| 人工勾选的错误类型 | 人工有问题 | 核验器标出（校准后 major/critical） | 只在 minor 层标出 | 完全未标 | 没标出的条目 |", "|---|---|---|---|---|---|"]
            for t, xs in sorted(by_ht.items(), key=lambda kv: -len(kv[1])):
                hit = [it for it, _ in xs if f_layers["cal"][fkey(it)]]
                minor_only = [it for it, _ in xs if not f_layers["cal"][fkey(it)] and f_layers["cal+minor"][fkey(it)]]
                miss = [it["id"] for it, _ in xs if not f_layers["cal"][fkey(it)]]
                out.append(f"| {t} | {len(xs)} | {len(hit)} | {len(minor_only)} | {len(xs) - len(hit) - len(minor_only)} | {'、'.join(miss) or '—'} |")
            out.append("")
        fn = [(it, l) for it, l in fl if bad(l) and not f_layers["cal"][fkey(it)]]
        fp = [(it, l) for it, l in fl if not bad(l) and (f_layers["cal"][fkey(it)] or f_layers["raw"][fkey(it)])]
        if fn:
            out += [f"漏报（人工认为有问题、校准后无 major/critical）{len(fn)} 条：", "",
                    "| 条目 | run | 工作区 / Finding | 人工 | 人工错误类型 | 核验器 minor | 备注 |", "|---|---|---|---|---|---|---|"]
            for it, l in fn:
                out.append(f"| {it['id']} | {it['run']} | {f_ws.get(fkey(it), '')} {it['finding_id']} | {l['main']} | {'、'.join(l.get('errors') or []) or '—'} "
                           f"| {'、'.join(sorted(set(f_layers['cal+minor'][fkey(it)]))) or '—'} | {note(l)} |")
            out.append("")
        if fp:
            out += [f"误报（人工认为没问题、核验器标了 major/critical）{len(fp)} 条：", "",
                    "| 条目 | run | 工作区 / Finding | 原始类型 | 校准后类型 | 人工 | 备注 |", "|---|---|---|---|---|---|---|"]
            for it, l in fp:
                out.append(f"| {it['id']} | {it['run']} | {f_ws.get(fkey(it), '')} {it['finding_id']} | {'、'.join(f_layers['raw'][fkey(it)]) or '—'} "
                           f"| {'、'.join(f_layers['cal'][fkey(it)]) or '—'} | {l['main']} | {note(l)} |")
            out.append("")

    rv = [it for it in items.values() if it["kind"] == "finding" and fkey(it) in revoked]
    if rv:
        out += ["### 被校准规则撤销了 major/critical 的 Finding", ""]
        for it in rv:
            l = labels.get(it["id"])
            for a in revoked[fkey(it)]:
                gone = a.get("dismissed") or a.get("from") or {}
                out.append(f"- **{it['id']}** {it['run']} {f_ws.get(fkey(it), '')} {it['finding_id']}：{a['rule']} 撤销 `{gone.get('type')}`（{gone.get('severity')}，"
                           f"{gone.get('method')}）——{a.get('reason', '')}")
                out.append(f"  - 被撤销的片段：{gone.get('span')!r}；核验器原话：{(gone.get('detail') or '')[:200]}")
            if not l:
                out.append("  - 人工：未标")
            else:
                h = "有问题 → 这次撤销是校准规则误撤（export 漏放了一条该 held 的）" if bad(l) else (
                    "没问题 → 撤销正确" if l["main"] in POS["finding"] else "无法判断")
                out.append(f"  - 人工：{l['main']}（{'、'.join(l.get('errors') or []) or '未勾类型'}）→ {h}" + (f"；备注：{note(l)}" if note(l) else ""))
        out.append("")

    ul = [(items[i], l) for i, l in labels.items() if items[i]["kind"] in ("answer", "plan")]
    ukey = lambda it: (it["run"], it.get("request"), it["kind"])  # noqa: E731
    ul = [(it, l) for it, l in ul if ukey(it) in unit_disp]
    if ul:
        out += ["### 答案 / 计划 · major/critical 层（与 export B1 的 held 口径一致）", "", *PR_HEAD]
        for kind in ("answer", "plan"):
            rows = [(bad(l), bool(unit_severe[ukey(it)])) for it, l in ul if it["kind"] == kind and l["main"] in POS[kind] | NEG[kind]]
            if rows:
                out.append(pr_row(kind, rows))
        out += ["", "答案按校准后的错误；计划的错误是从核验日志恢复的，日志里没有严重度，按错误类型默认值算"
                "（task_not_gap_driven / premise_smuggling = major，unnecessary_round = minor）。", "",
                "| 条目 | run | 人工 | 人工错误类型 | 核验器 disposition | 核验器 major/critical | 备注 |", "|---|---|---|---|---|---|---|"]
        for it, l in sorted(ul, key=lambda x: x[0]["id"]):
            out.append(f"| {it['id']} | {it['run']} | {l['main']} | {'、'.join(l.get('errors') or []) or '—'} | {unit_disp[ukey(it)].get('disposition')} "
                       f"| {'、'.join(unit_severe[ukey(it)]) or '—'} | {note(l)} |")
        out.append("")

    if fl:
        out += ["### D1 判定（Finding，major/critical）", ""]
        for layer, name in (("raw", "核验器原始判定"), ("cal", "校准后（export 实际用的）")):
            m = pr([(bad(l), bool(f_layers[layer][fkey(it)])) for it, l in fl])
            if m["precision"] is None:
                out.append(f"- {name}：没有标错，精确率无定义；召回 {m['recall']}，未标错里人工认为有错 {m['fn']} / {m['fn'] + m['tn']}")
                continue
            go = "≥ 0.85 → 放行" if m["precision"] >= D1_PRECISION else "< 0.85 → 收紧 export"
            out.append(f"- {name}：精确率 {m['precision']}（{m['tp']} / {m['tp'] + m['fp']}，95% {wilson(m['tp'], m['tp'] + m['fp'])}）{go}；"
                       f"召回 {m['recall']}（{m['tp']} / {m['tp'] + m['fn']}）；未标错里人工认为有错 {m['fn']} / {m['fn'] + m['tn']}")
        out.append("")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--report-verdicts", default=None, help="默认取 items.json meta.report_verdicts，再退回 verdicts_r0_judged.jsonl")
    parser.add_argument("--unit-verdicts", default=None, help="默认取 items.json meta.unit_verdicts，再退回 verdicts_units_judged.jsonl")
    args = parser.parse_args()
    batch = Path(args.batch)
    items_doc = json.loads((batch / "items.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in items_doc["items"]}
    sft = Path("experiments/sft_data")
    meta = items_doc.get("meta") or {}
    args.report_verdicts = args.report_verdicts or str(sft / (meta.get("report_verdicts") or "verdicts_r0_judged.jsonl"))
    args.unit_verdicts = args.unit_verdicts or str(sft / (meta.get("unit_verdicts") or "verdicts_units_judged.jsonl"))
    if not Path(args.unit_verdicts).exists() and (sft / "gold_set" / Path(args.unit_verdicts).name).exists():
        args.unit_verdicts = str(sft / "gold_set" / Path(args.unit_verdicts).name)
    files = [Path(p) if Path(p).is_absolute() or Path(p).exists() else batch / p for p in args.labels]
    annotations = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    # (run, finding) → 原始判定的 major/critical 类型；另存校准后、含 minor、被校准撤销的记录（10-15 分层）
    flagged: dict[tuple[str, str], list[str]] = {}
    f_layers: dict[str, dict[tuple[str, str], list[str]]] = {"raw": {}, "raw+minor": {}, "cal": {}, "cal+minor": {}}
    f_ws: dict[tuple[str, str], str] = {}
    revoked: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(Path(args.report_verdicts)):
        run = row["subject"]["run"]
        fids = {c["finding_id"] for c in row.get("coverage") or []} | {j["finding_id"] for j in row.get("judgments") or []}
        for fid in fids:
            flagged.setdefault((run, fid), [])
        for err in row.get("errors") or []:
            fid = (err.get("extra") or {}).get("finding")
            if fid and err.get("severity") in ("major", "critical"):
                flagged.setdefault((run, fid), []).append(err["type"])
        cal_row, applied = calibrated(row)
        raw_e, cal_e = finding_errors(row), finding_errors(cal_row)
        for fid in fids | raw_e.keys():
            key = (run, fid)
            f_ws[key] = row["subject"].get("workspace_id") or ""
            f_layers["raw"][key] = severe_types(raw_e.get(fid, []))
            f_layers["raw+minor"][key] = severe_types(raw_e.get(fid, []), minor=True)
            f_layers["cal"][key] = severe_types(cal_e.get(fid, []))
            f_layers["cal+minor"][key] = severe_types(cal_e.get(fid, []), minor=True)
        for a in applied:
            gone = a.get("dismissed") or a.get("from") or {}
            if gone.get("severity") in SEVERE:
                revoked[(run, a["finding"])].append(a)
    unit_rows = load_jsonl(Path(args.unit_verdicts))
    unit_disp = {(u["subject"]["run"], u["subject"].get("request"), u.get("unit")): u for u in unit_rows}
    # 答案 / 计划的 major/critical（答案按校准后，与 export 一致；计划不过校准）
    unit_severe = {k: severe_types(calibrated(u)[0].get("errors") or []) for k, u in unit_disp.items()}
    # --from-hard 批次（v1'）：条目里有答案句、替换后的证据，核验器结论按抽样时记进 strata.json 的为准
    strata_path = batch / "strata.json"
    strata_doc = json.loads(strata_path.read_text(encoding="utf-8")) if strata_path.is_file() else {}
    from_hard = (strata_doc.get("meta") or {}).get("mode") == "from_hard"
    strata = {s["id"]: s for s in strata_doc.get("strata") or []}

    lines = [f"# 金标对照：{batch.name} × {Path(args.report_verdicts).name} / {Path(args.unit_verdicts).name}", ""]
    for ann in annotations:
        who = ann.get("annotator", "?")
        rows: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        type_rows: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        type_hits, missing, skipped = Counter(), 0, Counter()
        by_stratum: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        by_reason: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        by_version: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        by_version_ctx: dict[str, list[tuple[bool, bool]]] = defaultdict(list)   # 宽口径：允许依据同单元已核实句子（10-04）
        minutes: dict[str, list[float]] = defaultdict(list)
        for lab in ann["labels"]:
            it = items.get(lab["id"])
            if not it or not lab.get("main"):
                continue
            kind = it["kind"]
            minutes[kind].append((lab.get("seconds") or 0) / 60)
            if kind == "plan_task":
                u = unit_disp.get((it["run"], it.get("request"), "plan"))
                if u:
                    got = task_types(u, it["task_label"])
                    for etype, (field, bad, good) in TASK_LEVEL.items():
                        if lab.get(field) in (bad, good):
                            type_rows[etype].append((lab[field] == bad, etype in got))
            if lab["main"] not in POS[kind] | NEG[kind]:
                skipped[kind] += 1
                continue
            human_bad = lab["main"] in NEG[kind]
            if kind == "finding" and from_hard and lab["id"] in strata:
                st = strata[lab["id"]]
                ver_types = list(st.get("verifier_errors") or [])
                pair = (human_bad, bool(st.get("verifier_flagged")))
                rows[kind].append(pair)
                by_stratum[st.get("stratum", "?")].append(pair)
                for reason in st.get("reasons") or ["(对照)"]:
                    by_reason[reason].append(pair)
                if st.get("video_frames"):
                    by_reason["含视频帧"].append(pair)
                ctx_label = lab.get("context_label")
                human_bad_ctx = (ctx_label == "bad") if ctx_label in ("ok", "bad") else human_bad
                for tag, f in (st.get("verifier_flags") or {}).items():
                    by_version[tag].append((human_bad, bool(f.get("flagged"))))
                    if ctx_label != "cannot_tell":
                        by_version_ctx[tag].append((human_bad_ctx, bool(f.get("flagged"))))
                if human_bad and ver_types:
                    for t in lab.get("errors") or []:
                        type_hits["同类" if t in ver_types else "异类"] += 1
            elif kind == "finding":
                key = (it["run"], it["finding_id"])
                if key not in flagged:
                    missing += 1
                    continue
                ver_types = flagged[key]
                rows[kind].append((human_bad, bool(ver_types)))
                if human_bad and ver_types:
                    for t in lab.get("errors") or []:
                        type_hits["同类" if t in ver_types else "异类"] += 1
            else:
                u = unit_disp.get((it["run"], it.get("request"), "plan" if kind == "plan_task" else kind))
                if not u:
                    missing += 1
                    continue
                rows[kind].append((human_bad, u.get("disposition") != "positive"))
        lines.append(f"## 标注人 {who}")
        lines.append("")
        lines.append("| 单元 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 | 召回 | 一致率 | κ | 无法判断 | 用时中位数（分钟） |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for kind in ("finding", "answer", "plan", "plan_task"):
            if kind == "plan_task" and not rows[kind] and not minutes[kind]:
                continue
            m = pr(rows[kind])
            med = round(statistics.median(minutes[kind]), 1) if minutes[kind] else "—"
            lines.append(f"| {kind} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} | {m['recall']} | {m['agreement']} | {m['kappa']} | {skipped[kind]} | {med} |")
        lines.append("")
        lines.append("（上表答案 / 计划按 disposition；Finding 按原始判定的 major/critical。下面分层见各小节。）")
        lines.append("")
        if not from_hard:
            lines += layered_section(ann, items, f_layers, f_ws, revoked, unit_disp, unit_severe)
        if by_version:
            lines.append("| 核验器版本（同一批人工标签） | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 | 召回 | κ |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for tag, prs in sorted(by_version.items()):
                m = pr(prs)
                lines.append(f"| {tag} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} | {m['recall']} | {m['kappa']} |")
            lines.append("")
            n_ctx = sum(1 for x in ann["labels"] if x.get("context_label")) + len(ann.get("context_only") or [])
            if n_ctx:
                lines.append(f"宽口径（{n_ctx} 条答了宽口径题，其余沿用主判定）：回指 / 综合句允许依据同一报告或答案里已核实的句子。")
                lines.append("")
                lines.append("| 核验器版本 · 宽口径 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 | 召回 | κ |")
                lines.append("|---|---|---|---|---|---|---|---|---|")
                for tag, prs in sorted(by_version_ctx.items()):
                    m = pr(prs)
                    lines.append(f"| {tag} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} | {m['recall']} | {m['kappa']} |")
                lines.append("")
        if by_stratum:
            lines.append("| v1' 分层 / 原因 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 | 召回 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for name, prs in [*sorted(by_stratum.items()), *sorted(by_reason.items())]:
                m = pr(prs)
                lines.append(f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} | {m['recall']} |")
            lines.append("")
            lines.append("swap 层：核验器在替换后的无关证据上判了支持；人工判\"不支持\" = 盖章属实（记在核验器漏报一栏）。")
            lines.append("")
        if type_rows:
            lines.append("| 任务级错误类型 | n | 双方都认为有 | 核验器误报 | 核验器漏报 | 都认为没有 | 精确率 | 召回 | κ |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for etype, trs in type_rows.items():
                m = pr(trs)
                lines.append(f"| {etype} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']} | {m['recall']} | {m['kappa']} |")
            lines.append("")
        extra_labels = ann.get("extra_labels") or []
        sem = [x for x in extra_labels if x.get("kind") == "semcov" and x.get("choice") in ("yes", "no")]
        if sem:
            agree = sum((x["choice"] == "yes") == bool(x.get("candidate_covered")) for x in sem)
            fp = [x for x in sem if x.get("candidate_covered") and x["choice"] == "no"]
            fn = [x for x in sem if not x.get("candidate_covered") and x["choice"] == "yes"]
            lines.append(f"语义覆盖抽查（F 组）：{len(sem)} 条，与核验器一致 {agree} 条；核验器判已建立而人判未建立 {len(fp)} 条"
                         + (f"（{'、'.join(x['id'] for x in fp)}）" if fp else "") + f"；核验器判未建立而人判已建立 {len(fn)} 条"
                         + (f"（{'、'.join(x['id'] for x in fn)}）" if fn else "") + "。")
            lines.append("")
        for x in extra_labels:
            if x.get("kind") == "label":
                lines.append(f"标签口径 {x['id']}（{x.get('run')}）：{ {'accept': '算一致', 'reject': '不算一致', 'cannot_tell': '无法判断'}.get(x.get('choice'), x.get('choice')) }"
                             + (f"；备注：{x['note']}" if x.get("note") else ""))
                lines.append("")
        if type_hits:
            lines.append(f"双方都认为有问题的 Finding 里，人工勾选的错误类型与核验器同类 {type_hits['同类']} 次、异类 {type_hits['异类']} 次。")
        if missing:
            lines.append(f"有 {missing} 条在这份判定文件里找不到对应记录（判定文件与抽样批次不是同一批 run）。")
        lines.append("")
    if len(annotations) >= 2:
        lines.append("## 标注人之间（主判定，二值化）")
        lines.append("")
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                a = {x["id"]: x for x in annotations[i]["labels"] if x.get("main")}
                b = {x["id"]: x for x in annotations[j]["labels"] if x.get("main")}
                pairs = []
                for iid in a.keys() & b.keys():
                    kind = items[iid]["kind"]
                    va, vb = a[iid]["main"], b[iid]["main"]
                    if va in POS[kind] | NEG[kind] and vb in POS[kind] | NEG[kind]:
                        pairs.append((va in NEG[kind], vb in NEG[kind]))
                lines.append(f"- {annotations[i].get('annotator')} × {annotations[j].get('annotator')}：共同 {len(pairs)} 条，"
                             f"一致率 {round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else '—'}，κ {kappa(pairs)}")
        lines.append("")
    text = "\n".join(lines)
    out = batch / f"对照_{Path(args.report_verdicts).stem}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
