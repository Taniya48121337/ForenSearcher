from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_control_plane_exposure import audit  # noqa: E402


def _row() -> dict:
    state = {
        "eligible_tools": ["set_field_decision"],
        "state_action_affordances": {
            "executable_decision_actions": [
                {"name": "set_field_decision", "arguments": {"claim_id": "C", "fields": ["x"]}}
            ],
            "must_reopen_before_rebinding": ["x"],
        },
        "claim_schema": {
            "target_fields": ["relationship"],
            "fields": {"relationship": {"enum": ["partial_copy"]}},
        },
    }
    payload = {"current_harness_state": state}
    return {
        "messages": [
            {"role": "system", "content": "Select the only eligible tool."},
            {"role": "user", "content": json.dumps(payload)},
            {
                "role": "assistant",
                "content": "decide",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "set_field_decision",
                            "arguments": {"claim_id": "C", "fields": ["x"]},
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "set_field_decision",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"const": "C"},
                            "fields": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "items": {"const": "x"},
                            },
                        },
                        "required": ["claim_id", "fields"],
                    },
                },
            }
        ],
        "metadata": {
            "row_id": "D:c:t:1",
            "case_id": "c",
            "dataset": "D",
            "trajectory_mode": "repair",
            "target_tool": "set_field_decision",
        },
    }


def test_audit_detects_control_plane_exposure() -> None:
    report, records = audit([_row()])
    row = records[0]
    assert row["eligible_one"] is True
    assert row["has_executable_decision_actions"] is True
    assert row["target_call_exactly_preexposed"] is True
    assert row["contains_only_eligible_tool_phrase"] is True
    assert row["has_singleton_target_field_enum"] is True
    assert report["repair_rollup"]["turns"] == 1
