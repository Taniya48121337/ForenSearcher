import asyncio
import copy
import json
import threading
from types import SimpleNamespace

import pytest

import foren_searcher.local_models as local


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 5., 100.])
async def test_queued_generation_receives_remaining_budget_without_changing_caller_snapshot(monkeypatch, budget):
    now = [0.]
    monkeypatch.setattr(local, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    generator = local.TransformersJsonGenerator(local.LocalModelConfig("unused"))
    received = []
    def generate(prompt, payload, stop):
        received.append(copy.deepcopy(payload))
        return SimpleNamespace(value={"ok": True}, input_tokens=1, output_tokens=1)
    monkeypatch.setattr(generator, "_generate_json", generate)
    constraints = {"remaining_turns": 3, "last_model_call_seconds": 12.}
    if budget is not None:
        constraints["remaining_seconds"] = budget
    payload = {"workspace_id": "W1-1", "model_view": {"runtime_constraints": constraints}}
    original = copy.deepcopy(payload)
    await generator._inference_gate.acquire()
    task = asyncio.create_task(generator.generate_json(system_prompt="system", payload=payload))
    await asyncio.sleep(0)
    assert received == []
    now[0] = 42.
    generator._inference_gate.release()
    assert await task == {"ok": True}
    limits = received[0]["model_view"]["runtime_constraints"]
    assert limits["last_model_call_seconds"] == 12.
    if budget is None:
        assert "remaining_seconds" not in limits
    else:
        assert limits["remaining_seconds"] == max(0., budget - 42.)
    assert payload == original


def test_json_repair_prompt_accounts_for_completed_generation_time(monkeypatch):
    now = [0.]
    monkeypatch.setattr(local, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    generator = local.TransformersJsonGenerator(local.LocalModelConfig("unused", json_attempts=2))
    received = []
    def generate(messages, *args):
        value = json.loads(messages[1]["content"])
        received.append(value["model_view"]["runtime_constraints"]["remaining_seconds"])
        now[0] += 30.
        if len(received) == 1:
            return "invalid json", 1, 1, ""
        assert any("invalid json" == message.get("content") for message in messages)
        return '{"ok":true}', 1, 1, ""
    monkeypatch.setattr(generator, "_generate_once", generate)
    result = generator._generate_json("system", {"workspace_id": "W1-1",
        "model_view": {"runtime_constraints": {"remaining_seconds": 100.}}}, threading.Event())
    assert result.value == {"ok": True}
    assert received == [100., 70.]


@pytest.mark.asyncio
async def test_hi_role_repair_prompt_accounts_for_the_failed_attempt(monkeypatch):
    now = [0.]
    monkeypatch.setattr(local, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    received = []
    class Generator:
        async def generate_json(self, *, system_prompt, payload):
            received.append(payload["model_view"]["runtime_constraints"]["remaining_seconds"])
            now[0] += 30.
            return {"action": "UNKNOWN_ACTION" if len(received) == 1 else "READ_RECORD", "arguments": {"record_id": "E1"},
                    "focus_evidence_ids": []}
    result = await local.TransformersHiPolicy(Generator()).decide(workspace_id="W1-1",
        model_view={"task_id": "W1-1", "task": "Read E1", "evidence_view": [{"evidence_id": "E1"}],
                    "runtime_constraints": {"remaining_seconds": 100.}},
        actions=({"name": "READ_RECORD", "kind": "tool", "schema": {"type": "object"}},))
    assert result["arguments"] == {"record_id": "E1"}
    assert received == [100., 70.]
