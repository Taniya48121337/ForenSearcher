"""Compare two student-context evaluations row by row.

The report separates genuine repairs from regressions.  This is deliberately
stricter than comparing only aggregate rates, which can hide offsetting errors.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOOLEAN_METRICS = (
    "tool_call_parseable",
    "tool_name_exact",
    "argument_shape_exact",
    "arguments_validate_visible_schema",
    "arguments_exact",
    "arguments_semantically_equal",
    "generated_tool_prestate_eligible",
    "reasoning_within_budget",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _generated_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["row_id"]): row
        for row in report.get("rows", [])
        if row.get("generated") is True
    }


def _transition(before: Any, after: Any) -> str:
    if before is True and after is False:
        return "regressed"
    if before is False and after is True:
        return "fixed"
    if before is True and after is True:
        return "passed_both"
    if before is False and after is False:
        return "failed_both"
    return "not_comparable"


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before_rows = _generated_rows(baseline)
    after_rows = _generated_rows(candidate)
    if set(before_rows) != set(after_rows):
        missing = sorted(set(before_rows) - set(after_rows))
        added = sorted(set(after_rows) - set(before_rows))
        raise ValueError(f"generation row mismatch: missing={missing}, added={added}")

    by_metric: dict[str, Counter[str]] = {metric: Counter() for metric in BOOLEAN_METRICS}
    by_dataset: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {metric: Counter() for metric in BOOLEAN_METRICS}
    )
    row_changes: list[dict[str, Any]] = []
    for row_id in sorted(before_rows):
        before = before_rows[row_id]
        after = after_rows[row_id]
        dataset = str(before.get("dataset") or "unknown")
        changes: dict[str, str] = {}
        for metric in BOOLEAN_METRICS:
            state = _transition(before.get(metric), after.get(metric))
            by_metric[metric][state] += 1
            by_dataset[dataset][metric][state] += 1
            if state in {"fixed", "regressed"}:
                changes[metric] = state
        before_limit = before.get("generation_hit_token_limit") is True
        after_limit = after.get("generation_hit_token_limit") is True
        if before_limit and not after_limit:
            limit_change = "fixed"
        elif not before_limit and after_limit:
            limit_change = "regressed"
        elif before_limit and after_limit:
            limit_change = "hit_both"
        else:
            limit_change = "clear_both"
        if changes or limit_change in {"fixed", "regressed"}:
            row_changes.append(
                {
                    "row_id": row_id,
                    "dataset": dataset,
                    "reference_tool": before.get("reference_tool"),
                    "baseline_generated_tool": before.get("generated_tool"),
                    "candidate_generated_tool": after.get("generated_tool"),
                    "metric_changes": changes,
                    "token_limit_change": limit_change,
                }
            )

    def pack(counter: Counter[str]) -> dict[str, int]:
        return {
            key: int(counter.get(key, 0))
            for key in ("fixed", "regressed", "passed_both", "failed_both", "not_comparable")
        }

    metric_report = {metric: pack(counts) for metric, counts in by_metric.items()}
    dataset_report = {
        dataset: {metric: pack(counts) for metric, counts in metrics.items()}
        for dataset, metrics in sorted(by_dataset.items())
    }
    baseline_summary = baseline.get("summary") or {}
    candidate_summary = candidate.get("summary") or {}
    nll_before = float(baseline_summary.get("target_nll"))
    nll_after = float(candidate_summary.get("target_nll"))
    limit_fixed = sum(1 for item in row_changes if item["token_limit_change"] == "fixed")
    limit_regressed = sum(
        1 for item in row_changes if item["token_limit_change"] == "regressed"
    )
    protected = (
        "tool_call_parseable",
        "tool_name_exact",
        "argument_shape_exact",
        "arguments_validate_visible_schema",
        "generated_tool_prestate_eligible",
    )
    no_protected_regression = all(metric_report[m]["regressed"] == 0 for m in protected)
    behavioral_net_gain = any(
        metric_report[m]["fixed"] > metric_report[m]["regressed"]
        for m in (
            "tool_name_exact",
            "arguments_validate_visible_schema",
            "arguments_semantically_equal",
            "generated_tool_prestate_eligible",
        )
    ) or limit_fixed > limit_regressed
    nll_not_worse = nll_after <= nll_before + 1e-9

    return {
        "artifact": "student_context_evaluation_comparison_v1",
        "generation_rows": len(before_rows),
        "target_likelihood": {
            "baseline_nll": nll_before,
            "candidate_nll": nll_after,
            "nll_delta": round(nll_after - nll_before, 6),
            "baseline_perplexity": baseline_summary.get("target_perplexity"),
            "candidate_perplexity": candidate_summary.get("target_perplexity"),
            "nll_not_worse": nll_not_worse,
        },
        "metrics": metric_report,
        "token_limit": {
            "fixed": limit_fixed,
            "regressed": limit_regressed,
            "baseline_rate": baseline_summary.get("generation_token_limit_hit_rate"),
            "candidate_rate": candidate_summary.get("generation_token_limit_hit_rate"),
        },
        "by_dataset": dataset_report,
        "changed_rows": row_changes,
        "pilot_verdict": {
            "no_protected_behavior_regression": no_protected_regression,
            "behavioral_net_gain": behavioral_net_gain,
            "target_nll_not_worse": nll_not_worse,
            "passed": no_protected_regression and behavioral_net_gain and nll_not_worse,
            "scope": "offline held-out next-action pilot; not yet closed-loop SFT proof",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(_load(args.baseline), _load(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
