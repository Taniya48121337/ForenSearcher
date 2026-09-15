from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_viddiff_holdout import build_summary


def test_summary_keeps_strict_admission_separate_from_route_accuracy(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    route = tmp_path / "route.json"
    audit.write_text(
        json.dumps(
            {
                "summary": {
                    "adaptive_candidate_sets": 2,
                    "student_export_oracle_leak_tokens": [],
                    "student_export_private_leak_tokens": [],
                },
                "cases": [
                    {"sample_key": "one", "accepted": True},
                    {"sample_key": "two", "accepted": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    route.write_text(
        json.dumps(
            {
                "sample_size": 2,
                "method": "targeted",
                "summary": {
                    "direction_correct": 2,
                    "interval_format_valid": 2,
                    "execution_failed": 0,
                    "order_consistent": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_summary([audit], [route], authorized_total=5)

    assert summary["strict_admission"]["accepted"] == 1
    assert summary["strict_admission"]["safe_for_batch_sft"] is False
    assert summary["route_diagnostics"]["targeted"]["direction_correct"] == 2
    assert summary["scope"]["remaining_cases_not_run"] == 3
