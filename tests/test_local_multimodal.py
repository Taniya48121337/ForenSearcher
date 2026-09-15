"""本地图像张量传输、工作区来源和上下文预算。"""

import asyncio
from types import SimpleNamespace

import pytest

from foren_searcher.local_models import LocalModelConfig, LocalModelContextError, TransformersJsonGenerator
from foren_searcher.store import GlobalStore
from inspect_helpers import image_bytes, ingest


def runtime(store, *, context_limit=24_000, fail=False):
    torch = pytest.importorskip("torch")
    calls = []

    class Tokenizer:
        pad_token_id = eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.ones((1, 4), dtype=torch.long)}

        def decode(self, *args, **kwargs):
            return '{"ok":true}'

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert any(part.get("type") == "image" for message in messages
                       if isinstance(message["content"], list) for part in message["content"])
            return "prompt with image"

        def __call__(self, *, images, **kwargs):
            return {"input_ids": torch.ones((1, 20), dtype=torch.long),
                    "pixel_values": torch.tensor([picture.getpixel((0, 0)) for picture in images]),
                    "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images))}

    class Model:
        config = SimpleNamespace(max_position_embeddings=8192)
        generation_config = SimpleNamespace(eos_token_id=0, prefill_chunk_size=1024)

        def parameters(self):
            yield torch.tensor(0)

        def generate(self, **kwargs):
            assert "prefill_chunk_size" not in kwargs
            assert kwargs["image_grid_thw"].shape == (1, 3)
            calls.append(kwargs["pixel_values"].tolist())
            if fail:
                raise RuntimeError("inference failed")
            return torch.cat((kwargs["input_ids"], torch.zeros((1, 1), dtype=torch.long)), dim=1)

    generator = TransformersJsonGenerator(LocalModelConfig("unused", max_input_tokens=context_limit), store=store)
    generator._model, generator._tokenizer, generator._processor = Model(), Tokenizer(), Processor()
    return generator, calls


def payload(store, actor, color):
    evidence = ingest(store, image_bytes(color=color), source=actor, filename=actor + ".png",
                      media_type="image/png", modality="image")
    return {"workspace_id": actor, "model_view": {"task_id": actor,
            "materials": [{"evidence_id": evidence.evidence_id}], "history": []}}


@pytest.mark.asyncio
async def test_shared_local_weights_receive_each_workspaces_pixels_and_record_only_its_ids(tmp_path):
    store = GlobalStore(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    generator, calls = runtime(store)
    requests = [payload(store, "W1-1", "red"), payload(store, "W1-2", "blue")]
    recorded = []
    generator.generation_observer = recorded.append
    try:
        for request in requests:
            generator.prepare_model_view(request["model_view"])
        assert await asyncio.gather(*(generator.generate_json(system_prompt="Describe the image.", payload=request)
                                      for request in requests)) == [{"ok": True}, {"ok": True}]
        assert calls == [[[255, 0, 0]], [[0, 0, 255]]]
        for request, record in zip(requests, recorded):
            view = request["model_view"]
            ids = [view["materials"][0]["evidence_id"]]
            assert record["workspace_id"] == request["workspace_id"]
            assert record["image_evidence_ids"] == ids
            assert view["model_observed_image_ids"] == ids
            assert view["content_observed_evidence_ids"] == ids
            assert record["input_tokens"] == 20
    finally:
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["context", "inference"])
async def test_failed_local_image_request_does_not_mark_pixels_observed(tmp_path, failure):
    store = GlobalStore(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    generator, calls = runtime(store, context_limit=10 if failure == "context" else 24_000,
                               fail=failure == "inference")
    request = payload(store, "W1-1", "red")
    generator.prepare_model_view(request["model_view"])
    recorded = []
    generator.generation_observer = recorded.append
    try:
        with pytest.raises(LocalModelContextError if failure == "context" else RuntimeError):
            await generator.generate_json(system_prompt="Describe the image.", payload=request)
        assert "model_observed_image_ids" not in request["model_view"]
        assert not recorded
        assert len(calls) == (0 if failure == "context" else 1)
    finally:
        await generator.aclose()


def test_text_only_local_model_does_not_advertise_pixels(tmp_path):
    store = GlobalStore(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    generator = TransformersJsonGenerator(LocalModelConfig("unused"), store=store)
    view = payload(store, "W1-1", "red")["model_view"]
    generator.prepare_model_view(view)
    assert not view.get("direct_visual_input")
    assert "image_inputs" not in view
