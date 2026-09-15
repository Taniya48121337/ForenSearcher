"""本地 HI 观察轮次、GIA 记忆会话和独立纠错的消息角色。"""

import copy
import json
import threading

import pytest

from foren_searcher.local_models import (
    LocalModelConfig, TransformersJsonGenerator, _generation_messages,
)


def request(actor):
    output = {"action": "READ_RECORD", "arguments": {"record_id": "E2"}, "focus_evidence_ids": []}
    observation = {"action": "READ_RECORD", "items": ["CURRENT_EVIDENCE"], "details": {"evidence_id": "E2"}}
    actions = [{"name": "READ_RECORD", "schema": {}}]
    if actor == "GIA":
        return {"global_view": {"task": "Read the evidence"}, "available_actions": actions,
                "memory_history": [{"output": output, "observation": observation, "reasoning": "PRIOR_REASONING"}]}
    return {"workspace_id": actor, "available_actions": actions, "model_view": {
        "task_id": actor, "task": "Read the evidence", "direct_visual_input": True,
        "runtime_constraints": {"turn_index": 2}, "latest_observation": observation,
        "history": [{"action": "READ_RECORD", "request": {"record_id": "E2"},
                     "focus_evidence_ids": [], "observation": observation, "reasoning": "PRIOR_REASONING"}]}}


@pytest.mark.parametrize("actor", ["W1-1", "GIA"])
def test_local_roles_preserve_each_sessions_current_context(actor):
    generator = TransformersJsonGenerator(LocalModelConfig("unused", enable_thinking=True))
    payload = request(actor)
    messages, count, _ = generator._generation_context("rules", payload)
    messages.append({"role": "user", "content": [{"type": "image", "image": "CURRENT_PIXELS"}]})
    before = copy.deepcopy(messages)

    class Template:
        def apply_chat_template(self, prepared, **kwargs):
            roles = [message["role"] for message in prepared]
            assert roles == ["system", "user", "assistant", "tool",
                             "user" if actor != "GIA" else "tool",
                             "user" if actor != "GIA" else "tool"]
            assert prepared[2]["reasoning_content"] == "PRIOR_REASONING"
            assert "CURRENT_EVIDENCE" in prepared[3]["content"]
            assert prepared[-1]["content"] == messages[-1]["content"]
            return "rendered"

    assert generator._render_prompt(Template(), messages) == "rendered"
    # 批量请求更新预算时只替换正文，仍保留各自角色策略及图片。
    messages[:count] = _generation_messages("rules", payload)
    assert generator._render_prompt(Template(), messages) == "rendered"
    assert messages == before


def test_hi_json_retry_keeps_latest_feedback_as_a_new_user_message(monkeypatch):
    generator = TransformersJsonGenerator(LocalModelConfig("unused", enable_thinking=True, json_attempts=2))
    observed = []

    class Template:
        def apply_chat_template(self, prepared, **kwargs):
            observed.append(copy.deepcopy(prepared))
            assert prepared[-1]["role"] == "user"
            return "rendered"

    def generate(messages, *args):
        generator._render_prompt(Template(), messages)
        return ("bad" if len(observed) == 1 else '{"ok":true}', 10, 4, None)

    monkeypatch.setattr(generator, "_generate_once", generate)
    result = generator._generate_json("rules", request("W1-1"), threading.Event())
    assert result.value == {"ok": True}
    assert len(observed) == 2
    assert observed[1][-2] == {"role": "assistant", "content": "bad"}
    assert "JSON" in observed[1][-1]["content"]
    assert "CURRENT_EVIDENCE" in json.dumps(observed[1])


def test_shared_generator_does_not_leak_role_policy_between_requests():
    generator = TransformersJsonGenerator(LocalModelConfig("unused", enable_thinking=True))
    sequences = [generator._generation_context("rules", request(actor))[0] for actor in ("W1-1", "GIA", "W1-2")]

    class Template:
        def apply_chat_template(self, prepared, **kwargs):
            return prepared[-1]["role"]

    assert [generator._render_prompt(Template(), sequence) for sequence in reversed(sequences)] == ["user", "tool", "user"]
