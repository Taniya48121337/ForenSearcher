from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reverify_viddiff_typed_candidates import reverify  # noqa: E402


def test_posthoc_verifier_can_admit_typed_candidate_without_editing_observation() -> None:
    audit = {
        "summary": {},
        "cases": [
            {
                "sample_key": "surgery_1",
                "oracle_direction": "c",
                "accepted": False,
                "candidates": [],
            }
        ],
    }
    observation = {
        "preferred_video": "c",
        "video_a_interval_s": [0, 2],
        "video_b_interval_s": [0, 3],
        "confidence": 0.25,
        "order_consistent": True,
    }
    rows = [
        {
            "sample_key": "surgery_1",
            "candidate_tools": [
                {
                    "name": "compare_predicate_independent",
                    "tool_observation": observation,
                    "provenance": {
                        "artifact_id": "viddiff_local_typed_predicate_observer_canary_v1"
                    },
                }
            ],
        }
    ]
    result = reverify(audit, rows)
    assert result["cases"][0]["accepted"] is True
    assert result["cases"][0]["selected_tool"] == "compare_predicate_independent"
    assert result["summary"]["newly_accepted_by_typed_candidate"] == ["surgery_1"]
    assert rows[0]["candidate_tools"][0]["tool_observation"] == observation
