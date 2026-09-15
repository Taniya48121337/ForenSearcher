from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_viddiff_holdout_trajectory_coverage import build_report  # noqa: E402


def test_report_separates_measurement_reachability_from_trajectory_quality() -> None:
    selection = {
        "cases": [
            {"sample_key": "surgery_1", "accepted": True, "selected_tool": "tool_a"},
            {"sample_key": "surgery_2", "accepted": True, "selected_tool": "tool_b"},
            {"sample_key": "surgery_3", "accepted": False, "selected_tool": None},
        ]
    }
    semantics = {
        "audit": "saved_episode_current_semantic_contract_v1",
        "episode_reports": [
            {"case_id": "viddiff-surgery_1", "mode": "clean", "semantic_passed": True},
            {"case_id": "viddiff-surgery_2", "mode": "clean", "semantic_passed": False},
        ],
    }
    report = build_report(selection, semantics)
    summary = report["summary"]
    assert summary["holdout_cases"] == 3
    assert summary["measurement_reachable_cases"] == 2
    assert summary["reachable_cases_with_semantic_pass_trajectory"] == 1
    assert summary["trajectory_success_given_reachable"] == 0.5
    assert summary["end_to_end_safe_coverage"] == 1 / 3
    assert summary["measurement_unreachable_quarantined"] == 1
    assert not summary["all_reachable_semantic_passed"]
    assert report["protocol"]["student_rows_include_oracle_or_verifier_fields"] is False
