"""Convert executable teacher tools into the exact student chat-template contract."""

from __future__ import annotations

import copy
import json
from typing import Any


def _without_teacher_reasoning(parameters: dict[str, Any]) -> dict[str, Any]:
    """Teacher reasoning is assistant text in SFT, not a function argument."""
    result = copy.deepcopy(parameters)
    properties = result.get("properties")
    if isinstance(properties, dict):
        properties.pop("reasoning", None)
    required = result.get("required")
    if isinstance(required, list):
        result["required"] = [name for name in required if name != "reasoning"]
    return result


def executable_argument_schema(argument_options: list[dict[str, Any]]) -> dict[str, Any]:
    """Compile visible complete actions into an exact native argument schema."""
    unique: list[dict[str, Any]] = []
    for arguments in argument_options:
        if arguments not in unique:
            unique.append(copy.deepcopy(arguments))
    if not unique:
        raise ValueError("at least one executable argument option is required")
    options = [
        {
            "type": "object",
            "properties": {
                name: {"const": copy.deepcopy(value)}
                for name, value in arguments.items()
            },
            "required": list(arguments),
            "additionalProperties": False,
        }
        for arguments in unique
    ]
    return options[0] if len(options) == 1 else {"oneOf": options}


def build_student_tools(
    claim_schema: dict[str, Any],
    tool_registry: list[dict[str, Any]],
    eligible_tools: list[str],
    executable_actions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return Qwen chat schemas for exactly the executable per-turn actions."""
    from llm_teacher_rollout import teacher_tool_schemas

    registry_by_name = {str(item["name"]): item for item in tool_registry}
    missing_registry = [name for name in eligible_tools if name not in registry_by_name]
    if missing_registry:
        raise ValueError(f"eligible tools missing from source registry: {missing_registry}")
    eligible_registry = [registry_by_name[name] for name in eligible_tools]
    teacher_schemas = teacher_tool_schemas(claim_schema, eligible_registry)
    action_arguments: dict[str, list[dict[str, Any]]] = {}
    for action in executable_actions or []:
        name = str(action.get("tool_name"))
        arguments = action.get("arguments")
        if name in eligible_tools and isinstance(arguments, dict):
            action_arguments.setdefault(name, []).append(arguments)
    result: list[dict[str, Any]] = []
    for schema in teacher_schemas:
        name = str(schema["name"])
        parameters = _without_teacher_reasoning(schema["parameters"])
        if name in action_arguments:
            parameters = executable_argument_schema(action_arguments[name])
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    actual = [item["function"]["name"] for item in result]
    if actual != eligible_tools:
        raise ValueError(f"student tool order/parity mismatch: expected={eligible_tools}, actual={actual}")
    return result


def constrain_tools_with_visible_invariants(
    tools: list[dict[str, Any]], visible_state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compile unambiguous observed invariant values into later tool schemas."""
    observed: dict[str, list[Any]] = {}
    for atom in (visible_state.get("evidence_store") or {}).values():
        if atom.get("active") is not True:
            continue
        observation = atom.get("observation") or {}
        for field in atom.get("supports_fields") or []:
            if field in observation and observation[field] not in observed.get(field, []):
                observed.setdefault(str(field), []).append(copy.deepcopy(observation[field]))
    constants = {
        field: values[0]
        for field, values in observed.items()
        if len(values) == 1
    }
    result = copy.deepcopy(tools)
    for tool in result:
        parameters = (tool.get("function") or {}).get("parameters") or {}
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        for field, value in constants.items():
            if field in properties:
                properties[field] = {"const": copy.deepcopy(value)}
    return result


def student_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return [str((item.get("function") or {}).get("name")) for item in tools]


def canonical_student_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Map stored harness-internal arguments back to the model-visible API contract."""
    result = copy.deepcopy(arguments)
    result.pop("reasoning", None)
    if tool_name == "bind_evidence" and "bindings" in result:
        bindings = result.pop("bindings")
        if not isinstance(bindings, dict):
            raise ValueError("bind_evidence internal bindings must be an object")
        result["binding_items"] = [
            {"field": field, "support_ids": copy.deepcopy(support_ids)}
            for field, support_ids in bindings.items()
        ]
    return result


def canonicalize_student_target(message: dict[str, Any]) -> dict[str, Any]:
    """Preserve reasoning/tool choice while making arguments match the visible schema."""
    result = copy.deepcopy(message)
    calls = result.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError("student target must contain exactly one tool call")
    function = calls[0].get("function") or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("student target arguments must be an object")
    function["arguments"] = canonical_student_arguments(str(function.get("name")), arguments)
    calls[0]["function"] = function
    return result
