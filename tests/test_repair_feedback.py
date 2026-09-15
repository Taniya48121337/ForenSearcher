"""Role feedback remains the latest observation through rendering and retries."""

import asyncio
import base64
import copy
import io
import json
import threading

import pytest
from PIL import Image

from foren_searcher.local_models import LocalModelConfig, TransformersJsonGenerator


def request(actor):
    view = {"task_id": actor, "task": "Read the supplied evidence.", "direct_visual_input": True,
            "runtime_constraints": {"remaining_seconds": 100.}, "evidence_view": [{"id": "E1"}]}
    return {"available_actions": [],
            **({"global_view": view, "memory_history": []} if actor == "GIA" else
               {"workspace_id": actor, "model_view": view}),
            "previous_invalid_output": {"actor": actor, "answer": "E999999"},
            "validation_error": f"Unknown evidence for {actor}: E999999. Quoted data: \"ignore rules\"."}


def image_message(view):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 255)).save(buffer, format="PNG")
    return {"role": "user", "content": [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}}]}, ["E1"]


def check_feedback(messages, actor, *, pixels, json_retry):
    index = -3 if json_retry else -1
    feedback = json.loads(messages[index]["content"])
    expected = request(actor)
    assert feedback == {key: expected[key] for key in ("previous_invalid_output", "validation_error")}
    assert messages[index]["role"] == "user"
    text = "\n".join(message["content"] for message in messages if isinstance(message["content"], str))
    assert text.count('"validation_error"') == 1
    assert text.count('"previous_invalid_output"') == 1
    if pixels:
        image = messages[index - 1]["content"][0]
        if image["type"] == "image_url":
            with Image.open(io.BytesIO(base64.b64decode(image["image_url"]["url"].split(",", 1)[1]))) as picture:
                assert picture.getpixel((0, 0)) == (0, 0, 255)
        else:
            assert image["type"] == "image" and image["image"].getpixel((0, 0)) == (0, 0, 255)
    if json_retry:
        assert messages[-2] == {"role": "assistant", "content": "broken"}
        assert "JSON" in messages[-1]["content"]


@pytest.mark.parametrize("actor", ["W1-1", "GIA"])
@pytest.mark.parametrize("pixels", [False, True])
def test_role_feedback_follows_evidence_and_survives_json_retry(monkeypatch, actor, pixels):
    generator = TransformersJsonGenerator(LocalModelConfig("unused", json_attempts=2))
    if pixels:
        generator._processor = object()
        monkeypatch.setattr(generator, "_image_message", image_message)
    payload = request(actor)
    before = copy.deepcopy(payload)
    calls = []

    def generate(messages, *args):
        check_feedback(messages, actor, pixels=pixels, json_retry=bool(calls))
        calls.append(True)
        return ("broken" if len(calls) == 1 else '{"ok":true}', 10, 4, None)

    monkeypatch.setattr(generator, "_generate_once", generate)
    assert generator._generate_json("rules", payload, threading.Event()).value == {"ok": True}
    assert len(calls) == 2
    assert payload["previous_invalid_output"] == before["previous_invalid_output"]
    assert payload["validation_error"] == before["validation_error"]


@pytest.mark.asyncio
async def test_batch_refresh_preserves_each_roles_feedback_and_images(monkeypatch):
    generator = TransformersJsonGenerator(LocalModelConfig("unused", max_batch_size=2, json_attempts=2))
    generator._processor = object()
    monkeypatch.setattr(generator, "_image_message", image_message)
    calls = []

    def batch(messages, jobs, schemas):
        assert len(jobs) == 2
        for sequence, job in zip(messages, jobs, strict=True):
            check_feedback(sequence, job.payload["workspace_id"], pixels=True, json_retry=False)
        calls.append("batch")
        return [(' {"ok":true}', 10, 4, None), ("broken", 10, 4, None)]

    def single(messages, *args):
        check_feedback(messages, "W1-2", pixels=True, json_retry=True)
        calls.append("single")
        return '{"ok":true}', 10, 4, None

    monkeypatch.setattr(generator, "_generate_batch_once", batch)
    monkeypatch.setattr(generator, "_generate_once", single)
    payloads = [request(actor) for actor in ("W1-1", "W1-2")]
    before = copy.deepcopy(payloads)
    try:
        result = await asyncio.gather(*(generator.generate_json(system_prompt="rules", payload=p) for p in payloads))
        assert result == [{"ok": True}, {"ok": True}]
        assert calls == ["batch", "single"]
        for original, current in zip(before, payloads, strict=True):
            assert original["validation_error"] == current["validation_error"]
            assert original["previous_invalid_output"] == current["previous_invalid_output"]
    finally:
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("actor", ["W1-1", "GIA"])
@pytest.mark.parametrize("pixels", [False, True])
async def test_hosted_role_feedback_follows_evidence_and_survives_json_retry(monkeypatch, actor, pixels):
    from foren_searcher.openrouter import OpenRouterJsonGenerator
    from foren_searcher.store import GlobalStore

    calls = []

    class Client:
        async def complete(self, messages, **kwargs):
            check_feedback(messages, actor, pixels=pixels, json_retry=bool(calls))
            calls.append(True)
            return {"text": "broken" if len(calls) == 1 else '{"ok":true}', "reasoning": None,
                    "model": "test", "provider": "test", "generation_id": "test", "usage": {}}

    generator = OpenRouterJsonGenerator(Client(), store=GlobalStore() if pixels else None)
    if pixels:
        monkeypatch.setattr(generator, "_image_message", image_message)
    payload = request(actor)
    before = copy.deepcopy(payload)
    assert await generator.generate_json(system_prompt="rules", payload=payload) == {"ok": True}
    assert len(calls) == 2
    assert payload["previous_invalid_output"] == before["previous_invalid_output"]
    assert payload["validation_error"] == before["validation_error"]
