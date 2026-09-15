from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_viddiff_predicate_coverage import build_report  # noqa: E402


def test_coverage_report_is_label_free_and_counts_families() -> None:
    report = build_report(
        {
            "seed": 7,
            "cases": [
                {
                    "sample_key": "surgery_1",
                    "candidate_difference": "More errors are corrected",
                    "differences_gt": {"1": "a"},
                },
                {
                    "sample_key": "surgery_2",
                    "candidate_difference": "The needle is grasped closer to the tip",
                    "oracle_direction": "b",
                },
            ],
        }
    )
    assert report["cases"] == 2
    assert report["family_counts"] == {
        "event_count": 1,
        "geometric_relation": 1,
    }
    assert report["all_cases_have_typed_protocol"] is True
    rendered = str(report).lower()
    assert "oracle_direction" not in rendered
    assert "differences_gt" not in rendered
    assert "'a'" not in rendered
