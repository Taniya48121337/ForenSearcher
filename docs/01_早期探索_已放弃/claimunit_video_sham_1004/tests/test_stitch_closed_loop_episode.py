from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stitch_closed_loop_episode import (  # noqa: E402
    _exportable_sft_rows,
    _rebase_sft_histories,
)
from validate_student_batch_sft import validate_rows  # noqa: E402


def _turn(index: int, tool: str) -> dict:
    datum = {
        "messages": [
            {"role": "system", "content": "policy"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_harness_state": {"eligible_tools": [tool]},
                        "past_execution_history": [],
                    }
                ),
            },
            {
                "role": "assistant",
                "content": f"I select {tool} because this closes the visible gap.",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": tool, "arguments": {}},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool,
                    "description": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "metadata": {
            "dataset": "VidDiffBench",
            "target_tool": tool,
            "teacher_action_used": False,
            "oracle_or_verifier_used": False,
        },
    }
    return {
        "turn": index,
        "status": "accepted",
        "generated_tool": tool,
        "generated_arguments": {},
        "environment_outcome": {"status": "accepted", "authority": "test"},
        "sft_datum": datum,
    }


def test_stitched_sft_rebases_history_and_normalizes_acceptance_metadata() -> None:
    turns = [_turn(1, "inspect"), _turn(2, "stop")]
    _rebase_sft_histories(turns)
    first_payload = json.loads(turns[0]["sft_datum"]["messages"][1]["content"])
    second_payload = json.loads(turns[1]["sft_datum"]["messages"][1]["content"])
    assert first_payload["past_execution_history"] == []
    assert len(second_payload["past_execution_history"]) == 1
    assert second_payload["past_execution_history"][0]["past_action"] == {
        "executed_tool": "inspect",
        "past_input": {},
    }
    assert second_payload["latest_environment_observation"]["status"] == "accepted"

    artifact = {
        "dataset": "VidDiffBench",
        "case_id": "case-1",
        "mode": "clean",
        "quality": {"score": 100, "accepted_for_high_quality_sft": True},
        "episode": {
            "source_split": "test",
            "trajectory_type": "accepted_prefix_stitched_complete_episode",
            "turns": turns,
        },
    }
    rows = _exportable_sft_rows(artifact)
    assert [row["metadata"]["turn_index"] for row in rows] == [1, 2]
    assert all(row["metadata"]["strict_closed_loop_accepted"] for row in rows)
    assert validate_rows(rows)["all_passed"] is True
