import json

import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore
from inspect_helpers import ingest, make_video


@pytest.mark.asyncio
async def test_unconfigured_semantic_query_does_not_return_sharp_frames_as_matches(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    path = tmp_path / "source.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    video = ingest(store, path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    async with create_inspect_service(store=store) as service:
        result = await service.execute({"targets": [video.evidence_id], "request": {
            "operation": "locate_frames", "params": {"query": {"type": "text", "text": "a horse jumping"}}}}, workspace_id="W1-1")
        assert result.status == "failed"
        assert result.result_evidence_id is None
        assert "get_frames" in result.feedback[0].message
        assert not result.feedback[0].retryable
        frames = await service.execute({"targets": [video.evidence_id], "request": {
            "operation": "get_frames", "params": {"count": 3}}}, workspace_id="W1-1")
        assert frames.status == "ready" and len(frames.result.frames) == 3
    assert store.healthcheck().status == "ok"


@pytest.mark.asyncio
async def test_color_retrieval_exposes_its_measurement_scope_in_shared_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    path = tmp_path / "source.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    video = ingest(store, path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    async with create_inspect_service(store=store) as service:
        result = await service.execute({"targets": [video.evidence_id], "request": {
            "operation": "locate_frames", "params": {"query": {"type": "text", "text": "red rectangle"}, "top_k": 3}}}, workspace_id="W1-1")
    assert result.status == "ready"
    saved = json.loads(store.evidence(result.result_evidence_id).content)
    assert saved["limitations"]
    assert any("颜色" in item for item in saved["limitations"])
    assert any("概率" in item for item in saved["limitations"])
    assert all(candidate.time_s >= 1.4 for candidate in result.result.candidates)
