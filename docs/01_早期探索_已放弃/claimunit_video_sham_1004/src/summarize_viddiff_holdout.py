"""Summarize frozen VidDiff holdout routes without exporting private labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    selection_audits: list[Path],
    route_reports: list[Path],
    *,
    authorized_total: int,
) -> dict[str, Any]:
    audits = [_read(path) for path in selection_audits]
    reports = [_read(path) for path in route_reports]

    accepted_case_ids: list[str] = []
    rejected_case_ids: list[str] = []
    private_leaks: set[str] = set()
    adaptive_sets = 0
    for audit in audits:
        adaptive_sets += int(audit.get("summary", {}).get("adaptive_candidate_sets", 0))
        private_leaks.update(audit.get("summary", {}).get("student_export_oracle_leak_tokens", []))
        private_leaks.update(audit.get("summary", {}).get("student_export_private_leak_tokens", []))
        for case in audit.get("cases", []):
            target = accepted_case_ids if case.get("accepted") is True else rejected_case_ids
            target.append(str(case.get("sample_key")))

    route_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "cases": 0,
            "direction_correct": 0,
            "interval_format_valid": 0,
            "execution_failed": 0,
            "order_consistent": 0,
            "order_consistency_observed_cases": 0,
        }
    )
    for report in reports:
        method = str(report.get("method") or "uniform_timeline")
        summary = report.get("summary", {})
        total = route_totals[method]
        total["cases"] += int(report.get("sample_size", len(report.get("cases", []))))
        total["direction_correct"] += int(summary.get("direction_correct", 0))
        total["interval_format_valid"] += int(summary.get("interval_format_valid", 0))
        total["execution_failed"] += int(summary.get("execution_failed", 0))
        if "order_consistent" in summary:
            total["order_consistent"] += int(summary.get("order_consistent", 0))
            total["order_consistency_observed_cases"] += int(
                report.get("sample_size", len(report.get("cases", [])))
            )

    tested_case_ids = sorted(set(accepted_case_ids + rejected_case_ids))
    accepted_case_ids = sorted(set(accepted_case_ids))
    rejected_case_ids = sorted(set(rejected_case_ids))
    tested = len(tested_case_ids)
    accepted = len(accepted_case_ids)
    return {
        "artifact": "viddiff_frozen_external_holdout_diagnostic_v1",
        "scope": {
            "authorized_cases": authorized_total,
            "externally_tested_cases": tested,
            "remaining_cases_not_run": max(authorized_total - tested, 0),
            "frozen_selection_required": True,
        },
        "strict_admission": {
            "accepted": accepted,
            "rejected": len(rejected_case_ids),
            "acceptance_rate": accepted / tested if tested else 0.0,
            "accepted_case_ids": accepted_case_ids,
            "rejected_case_ids": rejected_case_ids,
            "safe_for_batch_sft": False,
            "reason": "The strict yield is too low for batch production; only individually admitted cases may proceed.",
        },
        "adaptive_harness": {
            "student_visible_candidate_sets": adaptive_sets,
            "candidate_sets_are_final_sft_targets": False,
            "intended_use": "tool-routing and evidence-sufficiency decisions before any final claim",
        },
        "route_diagnostics": dict(route_totals),
        "privacy": {
            "student_export_leak_tokens": sorted(private_leaks),
            "student_export_passed": not private_leaks,
        },
        "tool_decisions": {
            "uniform_timeline": "retain: weak as a universal direction predictor but selected on an admitted case and useful for broad chronology",
            "global_motion_symmetric": "retain as an interval-proposal/corroboration tool, not as a standalone direction oracle",
            "action_refined_symmetric": "retain behind order-consistency and strict admission; strongest observed direct route but unstable across shards",
            "predicate_protocol": "retain as an observability/evidence-sufficiency critic; do not register as a direction tool",
        },
        "recommended_next_step": "Build closed-loop trajectories only for the strictly admitted cases; keep rejected cases as diagnostic negatives and do not run the remaining holdout until predicate-specific evidence improves.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-audit", action="append", type=Path, required=True)
    parser.add_argument("--route-report", action="append", type=Path, required=True)
    parser.add_argument("--authorized-total", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        args.selection_audit,
        args.route_report,
        authorized_total=args.authorized_total,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
