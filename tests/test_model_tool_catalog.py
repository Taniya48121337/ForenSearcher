"""工具目录保持所有操作及原始调用约束，运行时继续使用原 schema。"""

import copy
import json

import pytest

from foren_searcher.inspect.contracts import inspect_action_schema
from foren_searcher.local_models import _decision_output_schema, _generation_messages, _inspect_model_catalog


@pytest.mark.parametrize("enabled", [None, {"get_frames", "align_media"}, {"visual_qa"}])
def test_catalog_preserves_enabled_operations_and_complete_parameter_constraints(enabled):
    schema = inspect_action_schema()
    if enabled is not None:
        schema["oneOf"] = [branch for branch in schema["oneOf"]
                           if branch["properties"]["request"]["properties"]["operation"]["const"] in enabled]
    for branch in schema["oneOf"]:
        if branch["properties"]["request"]["properties"]["operation"]["const"] == "visual_qa":
            branch["properties"]["targets"]["maxItems"] = 8
    action = {"name": "INSPECT", "kind": "tool", "description": "Inspect evidence", "schema": schema}
    original = copy.deepcopy(action)
    projected = _inspect_model_catalog(action)
    catalog = projected["operation_catalog"]
    assert len(catalog) == len(schema["oneOf"])
    for branch, (name, entry) in zip(schema["oneOf"], catalog.items()):
        operation = branch["properties"]["request"]["properties"]
        assert name == operation["operation"]["const"]
        assert entry["description"] == operation["operation"]["description"]
        assert entry["targets_schema"] == branch["properties"]["targets"]
        assert entry["params_schema"] == operation["params"]
    payload = {"workspace_id": "W1-1", "model_view": {}, "available_actions": [action]}
    decoder = _decision_output_schema(payload, full_tool_schema=True)
    messages = _generation_messages("system", payload)
    displayed = json.loads(messages[1]["content"])["available_actions"][0]
    assert displayed == projected
    assert "arguments仅含targets和request" in messages[0]["content"]
    assert _decision_output_schema(payload, full_tool_schema=True) == decoder
    displayed["operation_catalog"].clear()
    projected["operation_catalog"].clear()
    assert action == original


@pytest.mark.parametrize("extension", ["root", "branch", "request", "operation", "duplicate", "empty"])
def test_unrecognized_inspect_schema_keeps_all_original_constraints(extension):
    action = {"name": "INSPECT", "schema": inspect_action_schema()}
    schema = action["schema"]
    branch = schema["oneOf"][0]
    if extension == "root":
        schema["description"] = "Special deployment constraint"
    elif extension == "branch":
        branch["not"] = {"required": ["blocked"]}
    elif extension == "request":
        branch["properties"]["request"]["maxProperties"] = 2
    elif extension == "operation":
        branch["properties"]["request"]["properties"]["operation"]["deprecated"] = True
    elif extension == "duplicate":
        schema["oneOf"].append(copy.deepcopy(branch))
    else:
        schema["oneOf"] = []
    assert _inspect_model_catalog(action) is action
    messages = _generation_messages("system", {"workspace_id": "W1-1", "available_actions": [action]})
    assert messages[0]["content"] == "system"
    assert json.loads(messages[1]["content"])["available_actions"] == [action]


def test_other_actions_and_gia_tools_are_not_rewritten():
    action = {"name": "SEARCH", "schema": inspect_action_schema()}
    assert _inspect_model_catalog(action) is action
    payload = {"global_view": {}, "available_actions": [{"name": "INSPECT", "schema": inspect_action_schema()}]}
    messages = _generation_messages("system", payload)
    assert messages[0]["content"] == "system"
    assert json.loads(messages[1]["content"]) == payload


@pytest.mark.parametrize("targets,operation,params,accepted", [
    (["E1"], "get_frames", {"count": 8}, True),
    (["E1", "E2"], "align_media", {"mode": "visual"}, True),
    (["E1", "E2"], "get_frames", {"count": 8}, False),
    (["E1", "E2"], "align_media", {}, False),
    (["E1"], "crop", {"count": 8}, False),
    (["E1"], "locate_frames", {"query": {"type": "text", "text": "cat"}}, True),
])
def test_local_generation_decoder_applies_operation_specific_arguments(monkeypatch, targets, operation, params, accepted):
    import threading
    from lmformatenforcer import TokenEnforcerTokenizerData
    from foren_searcher.local_models import LocalModelConfig, TransformersJsonGenerator, _create_output_enforcer

    payload = {"workspace_id": "W1-1", "model_view": {},
               "available_actions": [{"name": "INSPECT", "schema": inspect_action_schema()}]}
    chars = [chr(i) for i in range(32, 127)]
    data = TokenEnforcerTokenizerData([(i, char, False) for i, char in enumerate(chars)],
        lambda ids: "".join(chars[i] for i in ids), len(chars), False, len(chars) + 1)
    generator = TransformersJsonGenerator(LocalModelConfig("unused", constrain_json=True))
    captured = []

    def generate(messages, stop, schema):
        captured.append(schema)
        return '{"ok":true}', 1, 1, None

    monkeypatch.setattr(generator, "_generate_once", generate)
    generator._generate_json("system", payload, threading.Event())
    enforcer = _create_output_enforcer(data, captured[0])
    value = {"action": "INSPECT", "arguments": {"targets": targets,
             "request": {"operation": operation, "params": params}}, "focus_evidence_ids": []}
    prefix = [0]
    actual = True
    for char in json.dumps(value, separators=(",", ":")):
        token = chars.index(char)
        if token not in enforcer.get_allowed_tokens(prefix).allowed_tokens:
            actual = False
            break
        prefix.append(token)
    assert actual is accepted
