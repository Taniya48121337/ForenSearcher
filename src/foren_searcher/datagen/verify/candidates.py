"""在已有核验结果上叠加候选改动，不重判第二层（2026-10-04）。

大白话：程序规则、推理核对、答案语义覆盖都只改"已经判出来的错误怎么算"，第二层每对 (句, E) 的判定不变。
所以不必像 reanchor 那样回查证据原文（那要 run 的 artifacts）；只要 verdict 行（judgments 已在里面）+ run 的 store
（取报告 Finding 原文与工作区任务）+ 案卷，就能离线重算。新增的模型调用只有：推理核对每个候选片段一次、
语义覆盖每个答案一次，都进缓存。

    python -m foren_searcher.datagen.verify.candidates --report-verdicts R.jsonl --unit-verdicts U.jsonl \
        --runs experiments/sft_data/runs --cache verify_cache.precision.sonnet.jsonl --out-suffix v036c \
        [--rules on] [--infer on] [--semcov on] [--joint on]

输入应是 0.3.4 / 0.3.5 的结果（r0c 用 .v034.jsonl；r0cn 原文件即是）。产物另存，原结果不动；末尾打印哪些单元的处置变了、
由哪种机制改的。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..cases import load_cases
from . import anchored
from .calibrate_precision import cli_judge, load_jsonl, open_run
from .inference import INFERENCE_CHECK, apply_inference_check
from .joint_sources import JOINT_SOURCES, apply_joint_check
from .semantic_coverage import SEMANTIC_COVERAGE, apply_report_semantic_coverage, apply_semantic_coverage
from .verdict import Verdict, decide, tainted


def _report_context(ctx: Any, workspace: str) -> tuple[dict[str, str], str]:
    statements: dict[str, str] = {}
    task = ""
    for rep in ctx.store.workspace_report_records():
        if rep.task_id == workspace:
            statements = {str(f.get("finding_id")): str(f.get("statement") or "") for f in rep.to_dict().get("findings") or []}
            try:
                task = str(ctx.store.workspace_task_record(workspace).get("task") or "")
            except KeyError:
                task = ""
    return statements, task


async def apply_candidates(row: dict[str, Any], ctx: Any, judge: Any, *, case: Any = None,
                           report_rows: list[dict[str, Any]] | None = None) -> Verdict:
    v = Verdict.from_dict(row)
    if v.unit == "report":
        statements, task = _report_context(ctx, str(v.subject.get("workspace_id")))
        key = "finding"
    else:
        statements = {s["id"]: s["text"] for s in ((v.extra.get("anchored") or {}).get("sentences") or [])}
        task = str(ctx.config.get("task") or "")
        key = "sentence"
    errors = anchored.apply_program_rules(list(v.errors), statements, task or None, answer=v.unit == "answer", key=key)
    if INFERENCE_CHECK["enabled"]:
        errors = await apply_inference_check(errors, statements, list(v.judgments), judge, key=key)
    if JOINT_SOURCES["enabled"]:
        errors = await apply_joint_check(errors, statements, ctx.lookup, judge, key=key)
    v.errors = errors
    v.tainted_findings = tainted(v.errors)
    v.disposition = decide(v.errors, v.pending)
    if v.unit == "answer":
        await apply_semantic_coverage(v, case, report_rows or [], judge)
    else:
        await apply_report_semantic_coverage(v, case, judge)
    v.verifier = anchored.feature_version()
    return v


def _mechanisms(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    out = [f"{e['extra'].get('program_rule')}：{str(e.get('span') or '')[:60]}" for e in after.get("errors") or []
           if (e.get("extra") or {}).get("program_rule") not in (None, "semantic_coverage") and (e.get("extra") or {}).get("original_severity")
           and not any((b.get("extra") or {}).get("program_rule") == e["extra"]["program_rule"] and b.get("span") == e.get("span")
                       for b in before.get("errors") or [])]
    sc = (after.get("extra") or {}).get("semantic_coverage")
    if sc and any((e.get("extra") or {}).get("program_rule") == "semantic_coverage" for e in after.get("errors") or []):
        out.append("report_semantic_coverage：" + "；".join(f"{r} 由 {','.join(sc['by'].get(r) or [])} 建立" for r in sc.get("covered") or [])
                   + (f"（未建立：{', '.join(r for r in sc['asked'] if r not in sc['covered'])}）" if set(sc["asked"]) - set(sc["covered"]) else ""))
    had = any(e.get("type") == "premature_finish" for e in before.get("errors") or [])
    has = any(e.get("type") == "premature_finish" for e in after.get("errors") or [])
    if had and not has:
        sem = ((after.get("extra") or {}).get("sufficiency") or {}).get("semantic") or {}
        out.append("semantic_coverage：" + "；".join(f"{k} 由 {','.join(x.get('by') or [])} 建立" for k, x in sem.items() if x.get("covered")))
    return out


async def main_async(args: argparse.Namespace) -> int:
    if args.rules == "on":
        anchored.PROGRAM_RULES.update({k: True for k in anchored.PROGRAM_RULES})
    INFERENCE_CHECK["enabled"] = args.infer == "on"
    SEMANTIC_COVERAGE["enabled"] = args.semcov == "on"
    JOINT_SOURCES["enabled"] = args.joint == "on"
    cases = load_cases(args.cases)
    judge = cli_judge(args.judge_model, args.cache)
    runs: dict[str, Any] = {}
    reports_by_run: dict[str, list[dict[str, Any]]] = {}
    lines = [f"# 候选改动对照（{anchored.feature_version()}）", ""]
    try:
        for src in (args.report_verdicts, args.unit_verdicts):
            if not src:
                continue
            rows = load_jsonl(src)

            async def one(row: dict[str, Any]) -> dict[str, Any]:
                run = row["subject"]["run"]
                if row.get("unit") not in ("report", "answer"):
                    return row
                if run not in runs:
                    runs[run] = open_run(Path(args.runs) / run)
                v = await apply_candidates(row, runs[run], judge, case=cases.get(str(row["subject"].get("case_id") or "")),
                                           report_rows=reports_by_run.get(run))
                return v.to_dict()

            for row in rows:                                  # 先把 run 都打开（顺序打开，避免并发建临时库）
                if row.get("unit") in ("report", "answer") and row["subject"]["run"] not in runs:
                    runs[row["subject"]["run"]] = open_run(Path(args.runs) / row["subject"]["run"])
            out_rows = list(await asyncio.gather(*(one(r) for r in rows)))
            for r in out_rows:
                if r.get("unit") == "report":
                    reports_by_run.setdefault(r["subject"]["run"], []).append(r)
            target = Path(src).with_name(Path(src).name.replace(".jsonl", f".{args.out_suffix}.jsonl"))
            target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
            moves = Counter((r.get("unit"), a.get("disposition"), r.get("disposition")) for a, r in zip(rows, out_rows)
                            if r.get("unit") in ("report", "answer"))
            lines += [f"## {target.name}", "", "| 单元 | 旧 → 新 | 个数 |", "|---|---|---|"]
            lines += [f"| {u} | {a} → {b}{'' if a == b else ' ★'} | {n} |" for (u, a, b), n in sorted(moves.items())]
            lines += ["", "被改动的错误（★ = 处置变了）：", ""]
            for a, r in zip(rows, out_rows):
                mech = _mechanisms(a, r)
                if mech:
                    s = r["subject"]
                    star = " ★" if a.get("disposition") != r.get("disposition") else ""
                    lines.append(f"- {s.get('run')} {s.get('workspace_id') or s.get('request')}（{r.get('unit')}，{a.get('disposition')} → "
                                 f"{r.get('disposition')}{star}）")
                    lines += [f"  - {m}" for m in mech]
            lines.append("")
        lines.append(judge.summary())
    finally:
        for c in runs.values():
            c.close()
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="在已有核验结果上叠加候选改动（程序规则 / 推理核对 / 答案语义覆盖），不重判第二层")
    p.add_argument("--report-verdicts", required=True)
    p.add_argument("--unit-verdicts")
    p.add_argument("--runs", required=True)
    p.add_argument("--cases", default="experiments/sft_data/cases")
    p.add_argument("--cache", required=True)
    p.add_argument("--judge-model", default="sonnet")
    p.add_argument("--out-suffix", default="v036c")
    p.add_argument("--out", default=None, help="对照写成 markdown")
    p.add_argument("--rules", default="on", choices=("on", "off"))
    p.add_argument("--infer", default="on", choices=("on", "off"))
    p.add_argument("--semcov", default="on", choices=("on", "off"))
    p.add_argument("--joint", default="off", choices=("on", "off"), help="多来源合并复核（候选，10-05）")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
