"""补全 plan / answer 单元的模型判定（接核验模型后跑一次，产物给 export --units 叠加）。

大白话：export 里的 verify_units 只跑程序层（hindsight、代理锚定）；这里把 GIA 的两个单元再交给核验模型：
- answer：逐句锚定（answer_anchored.py），替掉"报告级代理"；
- plan：读任务文字判对准哪些要求、有没有前提走私（plan_judged.py），与 hindsight 对照。
写出 verdicts_units_judged.jsonl（只含 plan / answer 单元）；export 用 --units 叠加到程序层结果上。

用法（本机；默认连本地核验服务 --judge local，不花钱。--judge openrouter 走付费 API）：
    python -m foren_searcher.datagen.verify.complete_units --runs experiments/sft_data/runs/true_4124903_r1 \
        --verdicts experiments/sft_data/verdicts_r0_judged.jsonl --judge local \
        --out experiments/sft_data/verdicts_units_judged.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from ..cases import Case, load_cases
from ..runlog import load_run
from .answer_anchored import _WithRequestImages, anchor_answer, apply_answer_anchoring, as_report, request_images, split_answer, summarize_answer
from .anchored import BUDGET_EXIT_CODE, JsonJudge, JudgeBudgetExhausted, budget_message, count_pair_calls
from .calibrate import recalibrate
from .judges import add_judge_arguments, judge_from_args, judge_summary
from .plan_judged import apply_plan_judgment, judge_plan, summarize_plan
from .units import verify_units
from .verdict import Verdict


async def complete_run(run_dir: Path, case: Case | None, report_verdicts: list[Verdict], judge: JsonJudge, *,
                       judge_name: str | None, say: Callable[[str], None] = print,
                       sink: list[Verdict] | None = None) -> list[Verdict]:
    """sink：给了就把每个判完的单元即时追加进去（预算用尽中途抛 JudgeBudgetExhausted 时，已判完的单元仍在 sink 里）。"""

    log = load_run(run_dir)
    try:
        reports = [recalibrate(v) for v in report_verdicts if v.subject.get("run") == log.name and v.unit == "report"]
        units, useful = verify_units(log, case, reports)
        by_request = {d.request_id: d for d in log.decisions}
        out: list[Verdict] = []
        for unit in units:
            decision = by_request.get(str(unit.subject.get("request")))
            if decision is None:
                continue
            if unit.unit == "answer":
                sentences, result = await anchor_answer(decision, log, useful, judge)
                apply_answer_anchoring(unit, sentences, result, judge_model=judge_name)
                say(f"[{log.name}] answer {decision.request_id}: {json.dumps(summarize_answer(unit), ensure_ascii=False)}")
                out.append(unit)
                if sink is not None:
                    sink.append(unit)
            elif unit.unit == "plan":
                covered_before = set((unit.extra.get("gap") or {}).get("covered_before") or [])
                raw = await judge_plan(decision, case, covered_before, judge, log)
                apply_plan_judgment(unit, raw, case=case, judge_model=judge_name)
                say(f"[{log.name}] plan {decision.request_id}: {json.dumps(summarize_plan(unit), ensure_ascii=False)}")
                out.append(unit)
        return out
    finally:
        log.close()


def estimate_run_calls(run_dir: Path, case: Case | None, report_verdicts: list[Verdict]) -> dict[str, int]:
    """不调模型，估算 complete_run 要发的核验调用数：answer 每句 × 所引可判 E 各一次 + 每句一次汇总；plan 每单元一次。
    （含会命中缓存的调用；坏 JSON 重试不计。）"""

    log = load_run(run_dir)
    try:
        reports = [recalibrate(v) for v in report_verdicts if v.subject.get("run") == log.name and v.unit == "report"]
        units, useful = verify_units(log, case, reports)
        by_request = {d.request_id: d for d in log.decisions}
        answer_calls = plan_calls = 0
        for unit in units:
            decision = by_request.get(str(unit.subject.get("request")))
            if decision is None:
                continue
            if unit.unit == "plan":
                plan_calls += 1
            elif unit.unit == "answer":
                finding_evidence = {fid: list(row["evidence_ids"]) for fid, row in useful.clean_findings.items()}
                for rec in log.store.workspace_report_records():
                    for finding in rec.to_dict().get("findings") or []:
                        finding_evidence.setdefault(str(finding.get("finding_id")),
                                                    [x for x in finding.get("evidence_ids") or [] if isinstance(x, str)])
                sentences = split_answer(str(decision.output.get("answer") or ""), finding_evidence)
                skip = {e for s in sentences for e in s.evidence_ids if e not in decision.visible_ids}
                lookup = _WithRequestImages(log.lookup, request_images(log))
                answer_calls += count_pair_calls(as_report(sentences), lookup, skip_evidence_ids=skip)
        return {"answer": answer_calls, "plan": plan_calls, "total": answer_calls + plan_calls}
    finally:
        log.close()


def merge_units(units: list[Verdict], judged: list[Verdict]) -> list[Verdict]:
    """把模型判定过的 plan / answer 单元叠加到程序层结果上（按 run + request + unit 匹配），叠加时过 recalibrate。"""

    key = lambda v: (v.subject.get("run"), v.subject.get("request"), v.unit)  # noqa: E731
    table = {key(v): recalibrate(v) for v in judged}          # 模型判定文件存的是原判；规则校准在叠加时做，可审计可回滚
    return [table.get(key(v), v) for v in units]


def load_units(path: str | Path | None) -> list[Verdict]:
    if not path or not Path(path).is_file():
        return []
    return [Verdict.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补全 plan / answer 单元的模型判定")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cases", default="experiments/sft_data/cases")
    parser.add_argument("--verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl")
    add_judge_arguments(parser, default="local", choices=("local", "openrouter", "claude-cli"))
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl")
    parser.add_argument("--out", default="experiments/sft_data/verdicts_units_judged.jsonl")
    parser.add_argument("--estimate", action="store_true", help="只估算要发多少次核验调用（不调模型、不写产物）")
    args = parser.parse_args(argv)
    if args.estimate:
        cases = load_cases(args.cases)
        base = load_units(args.verdicts)
        grand = 0
        for run in args.runs:
            run_dir = Path(run)
            log = load_run(run_dir)
            case = cases.get(log.case_id or "")
            name = log.name
            log.close()
            counts = estimate_run_calls(run_dir, case, base)
            grand += counts["total"]
            print(f"  {name}: ~{counts['total']} 次（answer {counts['answer']}，plan {counts['plan']}）", flush=True)
        print(f"ESTIMATE {grand}", flush=True)
        return 0
    judge = judge_from_args(args, cache_path=args.cache or None)
    assert judge is not None
    judge_name = getattr(judge, "name", None)
    cases = load_cases(args.cases)
    base = load_units(args.verdicts)
    results: list[Verdict] = []
    exhausted: JudgeBudgetExhausted | None = None
    for run in args.runs:
        run_dir = Path(run)
        log = load_run(run_dir)
        case = cases.get(log.case_id or "")
        log.close()
        try:
            asyncio.run(complete_run(run_dir, case, base, judge, judge_name=judge_name, sink=results))
        except JudgeBudgetExhausted as exc:     # 没钱了：已判完的单元照常写盘，退出码 3
            exhausted = exc
            break
    out = Path(args.out)
    if exhausted is not None:      # 中途停止的部分结果不覆盖上一次完整产物，另存 .partial.jsonl
        out = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for v in results:
            handle.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
    print(f"{len(results)} 个单元 → {out}" + ("（预算用尽，中途停止；只含已判完的单元）" if exhausted else ""))
    line = judge_summary(judge)
    if line:
        print(line, flush=True)
    if exhausted is not None:
        print(budget_message(exhausted), flush=True)
        return BUDGET_EXIT_CODE
    return 0


__all__ = ["complete_run", "estimate_run_calls", "load_units", "main", "merge_units"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
