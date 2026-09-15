from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blind_trajectory_judge import (  # noqa: E402
    DIMENSIONS,
    blind_process_view,
    blind_view,
    judge_schema,
)


def test_blind_view_removes_identity_private_answers_and_other_judges() -> None:
    packet = {
        "dataset": "VidDiffBench",
        "case_id": "viddiff-surgery_20",
        "packet_id": "secret",
        "base": {"hidden_outcome_correct": True, "reasoning": "Visible motion changed."},
        "oracle": {"direction": "b"},
        "verifier_result": {"passed": True},
        "source_path": r"C:\private\video.mp4",
        "url": "https://example.test/private.mp4",
        "blind_judgments": [{"judge_id": "leak"}],
    }
    visible = blind_view(packet)
    text = json.dumps(visible)
    for forbidden in (
        "VidDiffBench",
        "surgery_20",
        "hidden_outcome_correct",
        "oracle",
        "verifier_result",
        "private\\\\video.mp4",
        "example.test",
        "judge_id",
    ):
        assert forbidden not in text
    assert visible["base"]["reasoning"] == "Visible motion changed."


def test_judge_schema_requires_all_dimensions_and_no_extras() -> None:
    schema = judge_schema()
    scores = schema["properties"]["dimension_scores"]
    evidence = schema["properties"]["dimension_evidence"]
    assert set(scores["required"]) == set(DIMENSIONS)
    assert set(evidence["required"]) == set(DIMENSIONS)
    assert scores["additionalProperties"] is False
    assert all(value["minimum"] == 1 and value["maximum"] == 5 for value in scores["properties"].values())


def test_blind_process_view_keeps_decisions_but_drops_repeated_chat_snapshots() -> None:
    packet = {
        "dataset": "VideoSham",
        "base": {"terminal_success": True, "hidden_outcome_correct": True},
        "probes": {"repair": {"executed": True, "reasoning": ["duplicate"]}},
        "raw_probe_episodes": {
            "repair": {
                "messages": [{"content": "very large repeated state"}],
                "api_trace": [{"private": "transport"}],
                "terminated": True,
                "turn_history": [
                    {
                        "turn_idx": 1,
                        "reasoning": "EA-1 conflicts with the visible interval, so reopen it.",
                        "tool_name": "set_field_decision",
                        "params": {"fields": ["interval"], "decision_status": "open"},
                        "observation": {
                            "status": "accepted",
                            "harness_state": {
                                "open_high_priority_obligations": ["interval"],
                                "eligible_tools": ["compare"],
                                "evidence_store": {},
                            },
                        },
                    }
                ],
            }
        },
    }
    visible = blind_process_view(packet)
    text = json.dumps(visible)
    assert "very large repeated state" not in text
    assert "api_trace" not in text
    assert "hidden_outcome_correct" not in text
    turn = visible["probe_process_episodes"]["probe_repair"]["turns"][0]
    assert turn["selected_tool"] == "set_field_decision"
    assert "EA-1 conflicts" in turn["reasoning"]
