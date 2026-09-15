from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_scripted_control_baseline import choose_control_action  # noqa: E402


def _tool(name: str, value: str = "x") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {"arg": {"const": value}},
                "required": ["arg"],
            },
        },
    }


def test_script_prefers_compiled_action_without_evidence() -> None:
    state = {
        "eligible_tools": ["bind_evidence", "stop"],
        "state_action_affordances": {
            "executable_binding_actions": [
                {"tool_name": "bind_evidence", "arguments": {"claim_id": "C", "binding_items": []}}
            ]
        },
    }
    decision = choose_control_action(state, [_tool("bind_evidence"), _tool("stop")])
    assert decision["tool_name"] == "bind_evidence"
    assert decision["selection_rule"] == "first_legal_compiled_action"


def test_script_uses_public_priority_when_control_is_ambiguous() -> None:
    state = {"eligible_tools": ["compare_uniform_timeline", "compare_action_refined_symmetric"]}
    tools = [_tool("compare_uniform_timeline"), _tool("compare_action_refined_symmetric")]
    decision = choose_control_action(state, tools)
    assert decision["tool_name"] == "compare_action_refined_symmetric"
    assert decision["arguments"] == {"arg": "x"}


def test_script_cannot_invent_unmaterialized_arguments() -> None:
    state = {"eligible_tools": ["create_candidate_claim"], "candidate_claims": []}
    tool = {
        "type": "function",
        "function": {
            "name": "create_candidate_claim",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
            },
        },
    }
    decision = choose_control_action(state, [tool])
    assert decision["arguments"] is None


def test_script_materializes_fixed_array_from_public_schema_order() -> None:
    state = {"eligible_tools": ["inspect_media"]}
    tool = {
        "type": "function",
        "function": {
            "name": "inspect_media",
            "parameters": {
                "type": "object",
                "properties": {
                    "media_refs": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["source", "suspect"]},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["media_refs"],
            },
        },
    }
    decision = choose_control_action(state, [tool])
    assert decision["arguments"] == {"media_refs": ["source", "suspect"]}
    assert decision["materialization_rule"] == "public_schema_order"
