"""只重算第二层（证据锚定）：拿已有的 verdict 行，按当前 anchored 逻辑重判每条 Finding / 答案句，其余层原样保留。

大白话：0.3.3 → 0.3.4 只改了第二层（限制句规则、实锤复核）。重跑整套 verify_report 要 trajectory、requests、案卷、
第三层充分性全部重来；这里只替换 verdict 里"属于某条 Finding / 某句答案、method 为 model / pixels"的错误和对应的
judgments / coverage，再按原规则重算处置与污染。第一层（exec）和第三层（sufficiency）的错误一条不动。
(句, E) 对的提示词与 payload 没变，全部命中原缓存；只有限制句的覆盖汇总与实锤复核是新调用。

    python -m foren_searcher.datagen.verify.reanchor --report-verdicts R.jsonl --unit-verdicts U.jsonl --runs RUNS \
        --cache verify_cache.claude-cli-claude-sonnet-5-5.jsonl --out-suffix v034 [--features off]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..cases import load_cases
from . import anchored
from .anchored import _verify_finding
from .inference import INFERENCE_CHECK, apply_inference_check
from .semantic_coverage import SEMANTIC_COVERAGE, apply_semantic_coverage
from .calibrate_precision import cli_judge, load_jsonl, open_run
from .verdict import Verdict, decide, tainted


def _owned(e: Any, fid: str, key: str) -> bool:
    return e.method in ("model", "pixels") and e.extra.get(key) == fid and not e.extra.get("check")


async def reanchor_row(row: dict[str, Any], ctx: Any, judge: Any, *, case: Any = None,
                       report_rows: list[dict[str, Any]] | None = None) -> Verdict:
    v = Verdict.from_dict(row)
    if v.unit == "report":
        ws = str(v.subject.get("workspace_id"))
        findings: dict[str, dict[str, Any]] = {}
        task = ""
        for rep in ctx.store.workspace_report_records():
            if rep.task_id == ws:
                findings = {str(f.get("finding_id")): f for f in rep.to_dict().get("findings") or []}
                try:
                    task = str(ctx.store.workspace_task_record(ws).get("task") or "")
                except KeyError:
                    task = ""
        statements = {fid: str(f.get("statement") or "") for fid, f in findings.items()}
        key = "finding"
    else:
        sentences = ((v.extra.get("anchored") or {}).get("sentences") or [])
        statements = {s["id"]: s["text"] for s in sentences}
        task = str(ctx.config.get("task") or "")
        key = "sentence"
    order: list[str] = []
    eids: dict[str, list[str]] = {}
    for j in v.judgments:
        fid = j["finding_id"]
        if fid not in eids:
            order.append(fid)
            eids[fid] = []
        eids[fid].append(j["evidence_id"])
    fids = [f for f in order if f in statements]
    outcomes = await asyncio.gather(*(_verify_finding(f, statements[f], [ctx.lookup.get(e) for e in eids[f]], judge, task=task or None)
                                      for f in fids))
    new_j, new_cov, new_err = [], [], []
    redone = set(fids)
    for f, out in zip(fids, outcomes):
        new_j += [j.to_dict() for j in out.judgments]
        if out.coverage is not None:
            new_cov.append(out.coverage)
        new_err += [e if key == "finding" else replace(e, extra={**e.extra, "sentence": e.extra.get("finding")}) for e in out.errors]
    new_err = anchored.apply_program_rules(new_err, statements, task or None, answer=v.unit == "answer", key=key)
    if INFERENCE_CHECK["enabled"]:
        new_err = await apply_inference_check(new_err, statements, new_j, judge, key=key)
    v.judgments = new_j + [j for j in v.judgments if j["finding_id"] not in redone]
    v.coverage = new_cov + [c for c in v.coverage if c.get("finding_id") not in redone]
    v.errors = [e for e in v.errors if not any(_owned(e, f, key) for f in redone)] + new_err
    v.tainted_findings = tainted(v.errors)
    v.disposition = decide(v.errors, v.pending)
    v.verifier = anchored.feature_version()
    if v.unit == "answer" and isinstance(v.extra.get("anchored"), dict):
        firm = [e for e in new_err if e.severity.value in ("major", "critical")]
        v.extra["anchored"]["unsupported_sentences"] = sorted({str(e.extra.get("sentence")) for e in firm if e.type.value == "unsupported_assertion"})
        v.extra["anchored"]["contradicted_sentences"] = sorted({str(e.extra.get("sentence")) for e in firm if e.type.value in
                                                                ("wrong_citation", "misread_fine_text", "sloppy_numeric")})
    if v.unit == "answer" and SEMANTIC_COVERAGE["enabled"]:
        await apply_semantic_coverage(v, case, report_rows or [], judge)
    return v


async def main_async(args: argparse.Namespace) -> int:
    if args.features == "off":
        anchored.FEATURES.update(limitation_rule=False, confirm_firm=False)
    # 0.3.6 起三项候选默认打开；reanchor 按参数显式设置（默认全关 = 复现 0.3.4 的第二层），不受默认值影响
    anchored.PROGRAM_RULES.update({k: args.rules == "on" for k in anchored.PROGRAM_RULES})
    INFERENCE_CHECK["enabled"] = args.infer == "on"
    SEMANTIC_COVERAGE["enabled"] = args.semcov == "on"
    cases = load_cases(args.cases) if args.semcov == "on" else {}
    judge = cli_judge(args.judge_model, args.cache)
    runs: dict[str, Any] = {}
    reports_by_run: dict[str, list[dict[str, Any]]] = {}
    try:
        for src in (args.report_verdicts, args.unit_verdicts):
            if not src:
                continue
            rows = load_jsonl(src)
            out_rows = []
            for row in rows:
                run = row["subject"]["run"]
                if run not in runs:
                    runs[run] = open_run(Path(args.runs) / run)
                if row.get("unit") not in ("report", "answer") or not row.get("judgments"):
                    out_rows.append(row)
                    continue
                v = await reanchor_row(row, runs[run], judge, case=cases.get(str(row["subject"].get("case_id") or "")),
                                       report_rows=reports_by_run.get(run))
                out_rows.append(v.to_dict())
                if v.unit == "report":
                    reports_by_run.setdefault(run, []).append(out_rows[-1])
            target = Path(src).with_name(Path(src).name.replace(".jsonl", f".{args.out_suffix}.jsonl"))
            target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
            before = Counter(r.get("disposition") for r in rows)
            after = Counter(r.get("disposition") for r in out_rows)
            firm = lambda rs: sum(1 for r in rs for e in r.get("errors") or [] if e.get("severity") in ("major", "critical")  # noqa: E731
                                  and e.get("method") in ("model", "pixels"))
            print(f"{target.name}：处置 {dict(before)} → {dict(after)}；模型层实锤错误 {firm(rows)} → {firm(out_rows)}", flush=True)
        print(judge.summary())
    finally:
        for c in runs.values():
            c.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="只重算第二层（证据锚定），其余层保留")
    p.add_argument("--report-verdicts", required=True)
    p.add_argument("--unit-verdicts")
    p.add_argument("--runs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--judge-model", default="sonnet")
    p.add_argument("--out-suffix", default="v034")
    p.add_argument("--features", default="on", choices=("on", "off"), help="off = 0.3.3 行为（用于核对重算是否逐字复现原结果）")
    p.add_argument("--rules", default="off", choices=("on", "off"), help="on = 打开候选程序规则（task_given / back_reference / verdict_sentence），产物标 verifier@0.3.6-rules")
    p.add_argument("--infer", default="off", choices=("on", "off"), help="on = 推理核对（inference.py）：实锤无支持片段能否由同单元已核实句子推出")
    p.add_argument("--semcov", default="off", choices=("on", "off"), help="on = 答案 premature_finish 按语义覆盖再判（semantic_coverage.py）；需要 --cases")
    p.add_argument("--cases", default="experiments/sft_data/cases")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
