from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compile_vcsl_selection_from_audit import compile_selection  # noqa: E402


def test_hidden_verifier_selection_strips_metrics_from_student_rows() -> None:
    pair_id = "a" * 32 + "-" + "b" * 32
    observation = {
        "relationship": "partial_copy",
        "copied_segments_frames": [[1, 2, 3, 4]],
        "localization_method": "DP",
    }
    audit = {
        "cases": [
            {
                "case_index": 11,
                "pair_id": pair_id,
                "variants": {
                    "dp": {
                        "status": "executed",
                        "student_visible_observation": observation,
                        "verifier_only_metrics": {"precision": 0.91, "recall": 0.92, "f1": 0.915},
                    },
                    "tn_stability": {
                        "status": "executed",
                        "student_visible_observation": {**observation, "localization_method": "TN-STABILITY"},
                        "verifier_only_metrics": {"precision": 0.99, "recall": 0.70, "f1": 0.82},
                    },
                },
            }
        ]
    }
    rows, report = compile_selection(audit)
    assert len(rows) == 1
    assert rows[0]["selected_localizer"] == "dp"
    assert report["summary"]["accepted"] == 1
    serialized = json.dumps(rows)
    assert "verifier_only_metrics" not in serialized
    assert "oracle" not in serialized
