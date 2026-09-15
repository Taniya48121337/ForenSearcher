"""Join label-blind VidDiff reachability with current trajectory semantics.

The measurement selection audit may contain post-hoc verifier fields. They are used only
to define the already-frozen reachable set and are never copied into student SFT rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def build_report(selection: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    cases = list(selection.get("cases") or [])
    semantic_reports = list(semantics.get("episode_reports") or [])
    semantic_by_case: dict[str, list[dict[str, Any]]] = {}
    for report in semantic_reports:
        semantic_by_case.setdefault(str(report.get("case_id") or ""), []).append(report)

    reachable_rows: list[dict[str, Any]] = []
    quarantined_rows: list[dict[str, Any]] = []
    selected_tools: Counter[str] = Counter()
    for case in cases:
        sample_key = str(case.get("sample_key") or "")
        case_id = f"viddiff-{sample_key}"
        if not case.get("accepted"):
            quarantined_rows.append(
                {
                    "case_id": case_id,
                    "reason": "no_frozen_label_blind_reachable_measurement",
                }
            )
            continue
        selected_tool = str(case.get("selected_tool") or "")
        selected_tools[selected_tool] += 1
        matches = semantic_by_case.get(case_id, [])
        passed = [item for item in matches if item.get("semantic_passed")]
        reachable_rows.append(
            {
                "case_id": case_id,
                "selected_tool": selected_tool,
                "trajectory_found": bool(matches),
                "semantic_passed": bool(passed),
                "passing_modes": sorted({str(item.get("mode")) for item in passed}),
            }
        )

    total = len(cases)
    reachable = len(reachable_rows)
    trajectory_passes = sum(bool(row["semantic_passed"]) for row in reachable_rows)
    return {
        "audit": "viddiff_external_holdout_trajectory_coverage_v1",
        "protocol": {
            "measurement_selection": "frozen_label_blind_three_route_with_ab_swap",
            "trajectory_semantics": str(semantics.get("audit") or "unknown"),
            "oracle_or_verifier_used_during_trajectory_generation": False,
            "student_rows_include_oracle_or_verifier_fields": False,
        },
        "summary": {
            "holdout_cases": total,
            "measurement_reachable_cases": reachable,
            "measurement_reachability": reachable / total if total else 0.0,
            "reachable_cases_with_semantic_pass_trajectory": trajectory_passes,
            "trajectory_success_given_reachable": (
                trajectory_passes / reachable if reachable else 0.0
            ),
            "end_to_end_safe_coverage": trajectory_passes / total if total else 0.0,
            "measurement_unreachable_quarantined": len(quarantined_rows),
            "all_reachable_semantic_passed": trajectory_passes == reachable,
            "selected_tools": dict(sorted(selected_tools.items())),
        },
        "reachable_cases": reachable_rows,
        "quarantined_cases": quarantined_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.selection_audit.read_text(encoding="utf-8")),
        json.loads(args.semantic_audit.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
