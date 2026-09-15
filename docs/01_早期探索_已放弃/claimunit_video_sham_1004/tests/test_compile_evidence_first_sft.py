from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compile_evidence_first_sft import (  # noqa: E402
    _apply_exposure_weights,
    _deduplicate_reasoning,
    _strict_reasoning_core,
    compile_row,
    sanitize_payload,
)


def test_sanitize_payload_removes_compiled_answers_and_singleton_target_enum() -> None:
    payload = {
        "task": {
            "claim_schema": {
                "target_fields": ["relationship"],
                "fields": {"relationship": {"type": "string", "enum": ["partial_copy"]}},
            }
        },
        "current_harness_state": {
            "eligible_tools": ["bind_evidence"],
            "action_constraints": {"must_reopen_before_rebinding": ["relationship"]},
            "state_action_affordances": {
                "executable_binding_actions": [
                    {"tool_name": "bind_evidence", "arguments": {"claim_id": "C"}}
                ]
            },
        },
        "compression_contract": "copy these arguments exactly",
    }
    revised = sanitize_payload(payload)
    text = json.dumps(revised)
    assert "state_action_affordances" not in text
    assert "must_reopen_before_rebinding" not in text
    assert "copy these arguments exactly" not in text
    field = revised["task"]["claim_schema"]["fields"]["relationship"]
    assert "enum" not in field
    assert "evidence_first_contract" in revised


def test_compile_row_rebuilds_generic_bind_schema_without_target_const() -> None:
    payload = {
        "task": {
            "claim_schema": {
                "schema_id": "vcsl_copy_localization_claim_v1",
                "required_fields": [
                    "query_id",
                    "reference_id",
                    "relationship",
                    "copied_segments_frames",
                    "localization_method",
                ],
                "target_fields": ["relationship", "copied_segments_frames", "localization_method"],
                "fields": {
                    "query_id": {"types": ["str"]},
                    "reference_id": {"types": ["str"]},
                    "relationship": {"types": ["str"], "enum": ["partial_copy"]},
                    "copied_segments_frames": {"types": ["list"]},
                    "localization_method": {"types": ["str"], "enum": ["DP"]},
                },
            }
        },
        "current_harness_state": {
            "eligible_tools": ["bind_evidence", "set_field_decision"],
            "state_action_affordances": {
                "executable_binding_actions": [
                    {
                        "tool_name": "bind_evidence",
                        "arguments": {
                            "claim_id": "C-1",
                            "binding_items": [
                                {"field": "relationship", "support_ids": ["EA-1"]}
                            ],
                        },
                    }
                ]
            },
        },
    }
    row = {
        "messages": [
            {"role": "system", "content": "copy exact action"},
            {"role": "user", "content": json.dumps(payload)},
            {
                "role": "assistant",
                "content": "EA-1 reports the relationship and supports the corresponding field.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bind_evidence",
                            "arguments": {
                                "claim_id": "C-1",
                                "binding_items": [
                                    {"field": "relationship", "support_ids": ["EA-1"]}
                                ],
                            },
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bind_evidence",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"const": "C-1"},
                            "binding_items": {"const": []},
                        },
                        "required": ["claim_id", "binding_items"],
                    },
                },
            }
        ],
        "metadata": {
            "row_id": "VCSL:CLEAN:4",
            "case_id": "vcsl-case",
            "dataset": "VCSL",
            "target_tool": "bind_evidence",
        },
    }
    compiled = compile_row(copy.deepcopy(row))
    bind = next(tool for tool in compiled["tools"] if tool["function"]["name"] == "bind_evidence")
    properties = bind["function"]["parameters"]["properties"]
    assert "const" not in properties["claim_id"]
    assert "const" not in properties["binding_items"]
    assert compiled["metadata"]["training_weight"] == 0.8
    assert compiled["metadata"]["policy_view"] == "evidence_first_reduced_control_v1"


def test_static_interface_preexposure_is_retained_but_downweighted() -> None:
    rows = [
        {
            "metadata": {
                "row_id": "R-1",
                "action_family": "measurement",
                "supervision_class": "semantic_tool_decision",
                "training_weight": 1.0,
            }
        },
        {
            "metadata": {
                "row_id": "R-2",
                "action_family": "create_claim",
                "supervision_class": "evidence_interpretation",
                "training_weight": 1.0,
            }
        },
    ]
    counts = _apply_exposure_weights(
        rows,
        [
            {"row_id": "R-1", "target_call_exactly_preexposed": True},
            {"row_id": "R-2", "target_call_exactly_preexposed": False},
        ],
    )
    assert rows[0]["metadata"]["supervision_class"] == "interface_deterministic"
    assert rows[0]["metadata"]["training_weight"] == 0.1
    assert rows[1]["metadata"]["training_weight"] == 1.0
    assert counts == {"interface_deterministic": 1, "not_preexposed": 1}


def test_exact_reasoning_duplicates_are_excluded_without_rewriting() -> None:
    def row(row_id: str, weight: float) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "x"},
                {"role": "assistant", "content": "Same grounded note."},
            ],
            "metadata": {
                "row_id": row_id,
                "dataset": "VCSL",
                "case_id": row_id,
                "target_tool": "stop",
                "training_weight": weight,
            },
        }

    kept, excluded = _deduplicate_reasoning([row("R-low", 0.1), row("R-high", 1.0)])
    assert [item["metadata"]["row_id"] for item in kept] == ["R-high"]
    assert excluded[0]["row_id"] == "R-low"
    assert excluded[0]["reasoning"] == "Same grounded note."


def test_strict_core_projects_passes_without_rewriting_failed_notes() -> None:
    rows = [
        {"metadata": {"row_id": "pass"}, "messages": [{"content": "kept verbatim"}]},
        {"metadata": {"row_id": "fail"}, "messages": [{"content": "failed verbatim"}]},
    ]
    strict = _strict_reasoning_core(
        rows,
        [
            {"row_id": "pass", "passes_causal_reasoning": True},
            {"row_id": "fail", "passes_causal_reasoning": False},
        ],
    )
    assert strict == [rows[0]]
    assert rows[1]["messages"][0]["content"] == "failed verbatim"
