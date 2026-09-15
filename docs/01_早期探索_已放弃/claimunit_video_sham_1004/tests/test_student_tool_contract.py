from __future__ import annotations

from jsonschema import Draft202012Validator

from student_tool_contract import (
    constrain_tools_with_visible_invariants,
    executable_argument_schema,
)


def test_executable_argument_schema_accepts_only_complete_visible_bind_call() -> None:
    arguments = {
        "claim_id": "claim_1",
        "binding_items": [
            {"field": "preferred_video", "support_ids": ["EA-ACTION-002"]},
            {"field": "video_a_interval_s", "support_ids": ["EA-ACTION-002"]},
        ],
    }
    schema = executable_argument_schema([arguments])
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(arguments))
    wrong = {
        "claim_id": "claim_1",
        "binding_items": [
            {"field": "preferred_video", "support_ids": ["EA-GLOBAL-002"]},
            {"field": "video_a_interval_s", "support_ids": ["EA-GLOBAL-002"]},
        ],
    }
    assert list(validator.iter_errors(wrong))


def test_visible_invariant_is_compiled_into_measurement_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compare_uniform_timeline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_a_ref": {"enum": ["video_a"]},
                        "comparison_statement": {"type": "string"},
                    },
                    "required": ["video_a_ref", "comparison_statement"],
                },
            },
        }
    ]
    state = {
        "evidence_store": {
            "EA-TASK-001": {
                "active": True,
                "supports_fields": ["sample_key", "comparison_statement"],
                "observation": {
                    "sample_key": "surgery_108",
                    "comparison_statement": "Exact visible action statement.",
                },
            }
        }
    }
    constrained = constrain_tools_with_visible_invariants(tools, state)
    statement = constrained[0]["function"]["parameters"]["properties"][
        "comparison_statement"
    ]
    assert statement == {"const": "Exact visible action statement."}
    assert tools[0]["function"]["parameters"]["properties"][
        "comparison_statement"
    ] == {"type": "string"}
