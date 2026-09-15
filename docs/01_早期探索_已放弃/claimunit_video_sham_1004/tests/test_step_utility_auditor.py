from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from step_utility_auditor import (  # noqa: E402
    _downstream_consumed,
    _external_observation_uptake,
)


EMPTY_DELTA = {
    "obligations_closed": [],
    "field_transitions": [],
    "claim_created": False,
    "terminal_changed": False,
}


def _turn(tool_name: str, params: dict) -> dict:
    return {
        "tool_name": tool_name,
        "params": params,
        "observation": {"created_evidence_ids": []},
    }


def test_external_observation_uptake_requires_id_detail_and_negative_reroute() -> None:
    turn = {
        "observation": {
            "created_evidence_ids": ["EA-NEG-001"],
            "evidence_atoms": [
                {
                    "evidence_id": "EA-NEG-001",
                    "observation": {"temporal_edit_detected": False},
                }
            ],
            "observation": {"goal_satisfied": False},
        }
    }
    next_turn = {
        "tool_name": "compare_spatial_regions",
        "reasoning": (
            "EA-NEG-001 reports that no temporal edit was detected, so the modality question remains unresolved. "
            "A spatial comparison is still needed to test the remaining explanation."
        ),
    }
    report = _external_observation_uptake(
        turn, next_turn, {"detect_temporal_edit", "compare_spatial_regions"}
    )
    assert report["passed"] is True
    assert "temporal" in report["matched_observation_detail_tokens"]

    next_turn["tool_name"] = "create_candidate_claim"
    report = _external_observation_uptake(
        turn, next_turn, {"detect_temporal_edit", "compare_spatial_regions"}
    )
    assert report["passed"] is False
    assert "negative_measurement_changes_route" in report["failures"]


def test_assessment_is_consumed_by_matching_adjudication() -> None:
    assessment = _turn(
        "assess_evidence_candidate",
        {"claim_id": "claim-1", "evidence_id": "evidence-2", "disposition": "reject"},
    )
    adjudication = _turn(
        "adjudicate_evidence",
        {
            "claim_id": "claim-1",
            "selected_evidence_ids": ["evidence-1"],
            "rejected_evidence": [{"evidence_id": "evidence-2", "defect_codes": ["conflict"]}],
        },
    )

    assert _downstream_consumed(assessment, [adjudication], EMPTY_DELTA, {})
    assert not _downstream_consumed(
        assessment,
        [_turn("adjudicate_evidence", {**adjudication["params"], "claim_id": "claim-2"})],
        EMPTY_DELTA,
        {},
    )


def test_adjudication_is_consumed_only_by_compatible_target_binding() -> None:
    adjudication = _turn(
        "adjudicate_evidence",
        {
            "claim_id": "claim-1",
            "target_fields": ["preferred_video", "measurement_method"],
            "selected_evidence_ids": ["evidence-1"],
            "rejected_evidence": [],
        },
    )
    compatible_binding = _turn(
        "bind_evidence",
        {
            "claim_id": "claim-1",
            "bindings": {
                "preferred_video": ["evidence-1"],
                "measurement_method": ["evidence-1"],
            },
        },
    )
    incompatible_binding = _turn(
        "bind_evidence",
        {
            "claim_id": "claim-1",
            "bindings": {"preferred_video": ["evidence-2"]},
        },
    )

    assert _downstream_consumed(adjudication, [compatible_binding], EMPTY_DELTA, {})
    assert not _downstream_consumed(adjudication, [incompatible_binding], EMPTY_DELTA, {})
