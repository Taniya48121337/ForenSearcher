from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unannounced_repair_interventions import INTERVENTIONS, score_episode  # noqa: E402


def test_matrix_has_eleven_unique_unannounced_interventions() -> None:
    assert len(INTERVENTIONS) == 11
    assert len({item["id"] for item in INTERVENTIONS}) == 11
    assert {item["dataset"] for item in INTERVENTIONS} == {"VideoSham", "VCSL", "VidDiffBench"}


def test_stop_without_targeted_action_is_not_discovery() -> None:
    spec = INTERVENTIONS[0]
    score = score_episode(
        {
            "terminal_success": True,
            "turn_count": 1,
            "rejection_count": 0,
            "turns": [{"generated_tool": "stop", "status": "accepted", "reasoning": "all done"}],
        },
        spec,
    )
    assert score["detected_before_stop"] is False
    assert score["stopped_without_targeted_action"] is True


def test_targeted_reopen_scores_localization_and_overinvalidation() -> None:
    spec = INTERVENTIONS[0]
    score = score_episode(
        {
            "terminal_success": False,
            "turn_count": 1,
            "rejection_count": 0,
            "turns": [
                {
                    "generated_tool": "set_field_decision",
                    "generated_arguments": {"fields": ["affected_source_interval_s", "direction"], "decision_status": "open"},
                    "status": "accepted",
                    "reasoning": "The interval is inconsistent.",
                }
            ],
        },
        spec,
    )
    assert score["detected_before_stop"] is True
    assert score["localized_reopen"] is True
    assert score["overinvalidated_fields"] == ["direction"]
