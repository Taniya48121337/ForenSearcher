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
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POS = {"finding": {"full"}, "answer": {"acceptable", "minor"}, "plan": {"good", "minor"}, "plan_task": {"good", "minor"}}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--report-verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl")
    parser.add_argument("--unit-verdicts", default="experiments/sft_data/verdicts_units_judged.jsonl")
    args = parser.parse_args()
    batch = Path(args.batch)
    items = {it["id"]: it for it in json.loads((batch / "items.json").read_text(encoding="utf-8"))["items"]}
    files = [Path(p) if Path(p).is_absolute() or Path(p).exists() else batch / p for p in args.labels]
    annotations = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    flagged: dict[tuple[str, str], list[str]] = {}
    for row in load_jsonl(Path(args.report_verdicts)):
        run = row["subject"]["run"]
        fids = {c["finding_id"] for c in row.get("coverage") or []} | {j["finding_id"] for j in row.get("judgments") or []}
        for fid in fids:
            flagged.setdefault((run, fid), [])
        for err in row.get("errors") or []:
            fid = (err.get("extra") or {}).get("finding")
            if fid and err.get("severity") in ("major", "critical"):
                flagged.setdefault((run, fid), []).append(err["type"])
    unit_disp = {(u["subject"]["run"], u["subject"].get("request"), u.get("unit")): u for u in load_jsonl(Path(args.unit_verdicts))}
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
