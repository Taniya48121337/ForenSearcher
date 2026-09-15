"""同一调查内复用已完成处理，保持 E 身份和 Artifact 完整性。"""

import asyncio
from dataclasses import replace
import json

import pytest

from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore
from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes


@pytest.mark.asyncio
async def test_analysis_producer_is_shared_with_evidence_and_reused_observations(tmp_path):
    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.inspect.backends.base import BackendOutcome
    from foren_searcher.workspace import WorkspaceHarness

    class VisualBackend(LocalInspectBackend):
        calls = 0
        async def visual_qa(self, media, params, policy):
            self.calls += 1
            return BackendOutcome({"image_evidence_ids": [item.evidence_id for item in media],
                "answer": "A red region is visible.", "limitations": ["Selected image only."]},
                details={"backend": "test-vision", "model": "vision-model", "provider": "supplier",
                         "question": "Private local question", "cost_usd": .1})

    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    evidence = ingest(store, image_bytes(), source="I0", filename="source.png", media_type="image/png", modality="image")
    call = {"targets": [evidence.evidence_id], "request": {"operation": "visual_qa", "params": {"question": "What is visible?"}}}
    backend = VisualBackend()
    result_ids = []
    try:
        async with create_inspect_service(store=store, backend=backend) as service:
            for workspace_id in ("W1-1", "W1-2"):
                harness = WorkspaceHarness(task=WorkspaceTask(workspace_id, "Describe the image", (evidence.evidence_id,)),
                    store=store, inspect_service=service)
                response = await harness.execute_inspect(call)
                assert response.status == "ready"
                result_ids.append(response.result_evidence_id)
                saved = json.loads(store.evidence(response.result_evidence_id).content)
                observed = json.loads(harness.memory.latest_observation["items"][1])
                for visible in (saved, observed):
                    assert {key: visible[key] for key in ("backend", "model", "provider")} == {
                        "backend": "test-vision", "model": "vision-model", "provider": "supplier"}
                    assert "question" not in visible and "cost_usd" not in visible
        assert result_ids[0] == result_ids[1] and backend.calls == 1
    finally:
        await backend.aclose()


class CountingBackend(LocalInspectBackend):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.both_started = asyncio.Event()

    async def crop(self, media, params, policy):
        self.calls += 1
        if self.calls == 2:
            self.both_started.set()
        await asyncio.wait_for(self.both_started.wait(), timeout=2)
        return await super().crop(media, params, policy)


@pytest.mark.asyncio
async def test_parallel_calls_execute_independently_and_completed_results_are_reused(tmp_path):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, image_bytes(), source="I0", filename="source.png", media_type="image/png", modality="image")
    backend = CountingBackend()
    call = {"targets":[original.evidence_id], "request":{"operation":"crop", "params":{"box":[0,0,0.5,0.5]}}}
    async with create_inspect_service(store=store, backend=backend) as service:
        first, second = await asyncio.gather(service.execute(call, workspace_id="W1-1"),
                                            service.execute(call, workspace_id="W1-2"))
        assert first.status == second.status == "ready"
        assert first.result.evidence_id != second.result.evidence_id
        assert backend.calls == 2
        repeated = await service.execute(call, workspace_id="W1-3")
        assert repeated.status == "ready"
        assert repeated.result.evidence_id in {first.result.evidence_id, second.result.evidence_id}
        assert backend.calls == 2
        changed = {"targets":[original.evidence_id], "request":{"operation":"crop", "params":{"box":[0.5,0.5,1,1]}}}
        third = await service.execute(changed, workspace_id="W1-2")
        assert third.result.evidence_id != first.result.evidence_id
        assert backend.calls == 3
        artifact = store.evidence(repeated.result.evidence_id).artifact_ref
        store.artifact_path(artifact).write_bytes(b"damaged")
        damaged = await service.execute(call, workspace_id="W1-1")
        assert damaged.status == "failed"
        assert backend.calls == 3


def frame_call(evidence_id, times):
    return {"targets": [evidence_id], "request": {"operation": "get_frames", "params": {"times_s": times}}}


def initial_frame_call(evidence_id, operation):
    if operation == "get_frames":
        return frame_call(evidence_id, [0.0])
    return {"targets": [evidence_id], "request": {"operation": "locate_frames",
            "params": {"query": {"type": "text", "text": "red"}, "top_k": 1}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("frames_first", [False, True])
async def test_frame_retrieval_and_extraction_share_images_but_keep_query_scores(tmp_path, monkeypatch, frames_first):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    video_path = tmp_path / "source.avi"
    make_video(video_path, frames=20)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, video_path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    frames = {"targets": [original.evidence_id], "request": {"operation": "get_frames", "params": {"count": 8}}}
    locate = {"targets": [original.evidence_id], "request": {"operation": "locate_frames",
              "params": {"query": {"type": "text", "text": "red"}, "top_k": 8}}}
    first_call, second_call = (frames, locate) if frames_first else (locate, frames)
    async with create_inspect_service(store=store) as service:
        first = await service.execute(first_call, workspace_id="W1-1")
    assert first.status == "ready"
    first_rows = first.result.frames if frames_first else first.result.candidates
    original_ids = {row.time_s: row.evidence_id for row in first_rows}
    assert len(original_ids) == 8
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    async with create_inspect_service(store=store) as service:
        second = await service.execute(second_call, workspace_id="W1-2")
        assert second.status == "ready"
        second_rows = second.result.candidates if frames_first else second.result.frames
        assert {row.time_s: row.evidence_id for row in second_rows} == original_ids
        red = second if frames_first else first
        blue = await service.execute({**locate, "request": {"operation": "locate_frames",
            "params": {"query": {"type": "text", "text": "blue"}, "top_k": 8}}}, workspace_id="W1-2")
        assert blue.status == "ready"
        assert {row.time_s: row.evidence_id for row in blue.result.candidates} == original_ids
        assert blue.result_evidence_id != red.result_evidence_id
        assert {row.evidence_id: row.score for row in blue.result.candidates} != {
            row.evidence_id: row.score for row in red.result.candidates}
        assert sum(e.modalities == ("image",) for e in store.evidence_records()) == 8
        assert all(store.evidence(eid).discovered_by == "W1-1" for eid in original_ids.values())
        assert set(store.inspect_execution(2)["details"]["reused_evidence_ids"]) == set(original_ids.values())
        assert store.healthcheck().status == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen", [False, True])
async def test_overlapping_frame_batches_keep_existing_evidence_ids(tmp_path, reopen):
    video_path = tmp_path / "source.avi"
    make_video(video_path, frames=5)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, video_path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    service = create_inspect_service(store=store)
    try:
        first = await service.execute(frame_call(original.evidence_id, [0.0, 0.1]), workspace_id="W1-1")
        assert first.status == "ready"
        first_ids = {f.time_s: f.evidence_id for f in first.result.frames}
        if reopen:
            await service.aclose()
            store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
            service = create_inspect_service(store=store)
        second = await service.execute(frame_call(original.evidence_id, [0.1, 0.2]), workspace_id="W1-2")
        assert second.status == "ready"
        second_ids = {f.time_s: f.evidence_id for f in second.result.frames}
        assert second_ids[0.1] == first_ids[0.1]
        assert second_ids[0.2] not in first_ids.values()
        assert store.evidence(second_ids[0.1]).discovered_by == "W1-1"
        frames = [e for e in store.evidence_records() if e.acquired_by == "INSPECT/get_frames"]
        assert len(frames) == 3
        details = store.inspect_execution(2)["details"]
        assert details["reused_evidence_ids"] == [first_ids[0.1]]
        assert details["created_evidence_ids"] == [second_ids[0.2]]
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get_frames", "locate_frames"])
async def test_equal_frames_from_different_sources_keep_their_provenance(tmp_path, monkeypatch, operation):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    video_path = tmp_path / "source.avi"
    make_video(video_path, frames=5)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    originals = [ingest(store, video_path.read_bytes(), source=f"I{i}", filename=f"source{i}.avi", media_type="video/avi", modality="video") for i in range(2)]
    async with create_inspect_service(store=store) as service:
        replies = [await service.execute(initial_frame_call(e.evidence_id, operation), workspace_id=f"W1-{i+1}") for i, e in enumerate(originals)]
    assert all(r.status == "ready" for r in replies)
    ids = [(r.result.frames if operation == "get_frames" else r.result.candidates)[0].evidence_id for r in replies]
    assert ids[0] != ids[1]
    assert [store.evidence(eid).derived_from for eid in ids] == [(e.evidence_id,) for e in originals]


class ChangedFrameBackend(LocalInspectBackend):
    changed = False

    async def get_frames(self, media, params, policy):
        outcome = await super().get_frames(media, params, policy)
        if self.changed:
            return replace(outcome, artifacts=tuple(replace(a, data=image_bytes(color="blue", red_box=False, size=(160, 96))) for a in outcome.artifacts))
        return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get_frames", "locate_frames"])
async def test_changed_decoded_frame_bytes_are_not_reused(tmp_path, monkeypatch, operation):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    video_path = tmp_path / "source.avi"
    make_video(video_path, frames=5)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, video_path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    backend = ChangedFrameBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        first = await service.execute(initial_frame_call(original.evidence_id, operation), workspace_id="W1-1")
        first_rows = first.result.frames if operation == "get_frames" else first.result.candidates
        time_s = first_rows[0].time_s
        other_time_s = 0.1 if time_s != 0.1 else 0.3
        backend.changed = True
        second = await service.execute(frame_call(original.evidence_id, [time_s, other_time_s]), workspace_id="W1-2")
    assert first.status == second.status == "ready"
    assert first_rows[0].time_s == second.result.frames[0].time_s
    assert first_rows[0].evidence_id != second.result.frames[0].evidence_id


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get_frames", "locate_frames"])
async def test_damaged_frame_in_overlapping_batch_is_reported(tmp_path, monkeypatch, operation):
    monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    video_path = tmp_path / "source.avi"
    make_video(video_path, frames=5)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, video_path.read_bytes(), source="I0", filename="source.avi", media_type="video/avi", modality="video")
    async with create_inspect_service(store=store) as service:
        first = await service.execute(initial_frame_call(original.evidence_id, operation), workspace_id="W1-1")
        first_rows = first.result.frames if operation == "get_frames" else first.result.candidates
        time_s = first_rows[0].time_s
        other_time_s = 0.1 if time_s != 0.1 else 0.3
        artifact = store.evidence(first_rows[0].evidence_id).artifact_ref
        store.artifact_path(artifact).write_bytes(b"damaged")
        second = await service.execute(frame_call(original.evidence_id, [time_s, other_time_s]), workspace_id="W1-2")
    assert second.status == "partial"
    assert any(f.code == "SAVE_FAILED" for f in second.feedback)
    assert [f.time_s for f in second.result.frames] == [other_time_s]


@pytest.mark.asyncio
@pytest.mark.parametrize("frames_first", [False, True])
async def test_alignment_keeps_numeric_anchors_and_reuses_full_original_frames(tmp_path, frames_first):
    import cv2
    import numpy as np
    from PIL import Image

    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (640, 480))
    assert writer.isOpened()
    rng = np.random.default_rng(381)
    for _ in range(20):
        writer.write(cv2.resize(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8), (640, 480)))
    writer.release()
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    originals = [ingest(store, path.read_bytes(), source=f"I{i}", filename=f"source{i}.avi",
                       media_type="video/avi", modality="video") for i in range(2)]
    call = {"targets": [e.evidence_id for e in originals],
            "request": {"operation": "align_media", "params": {"mode": "visual", "top_k": 1}}}
    async with create_inspect_service(store=store) as service:
        initial = []
        if frames_first:
            initial = [await service.execute(frame_call(e.evidence_id, [0.00001, 3.79999]), workspace_id="W1-1")
                       for e in originals]
        first = await service.execute(call, workspace_id="W1-2")
        assert first.status == "ready"
        anchors = first.result.matches[0].anchors
        assert len(anchors) == 4
        assert all(a.image_a_evidence_id is None and a.image_b_evidence_id is None for a in anchors[1:-1])
        endpoints = [anchors[0], anchors[-1]]
        for side, original in zip(("a", "b"), originals):
            ids = [getattr(a, f"image_{side}_evidence_id") for a in endpoints]
            for eid in ids:
                evidence = store.evidence(eid)
                assert evidence.derived_from == (original.evidence_id,)
                with Image.open(store.artifact_path(evidence.artifact_ref)) as image:
                    assert image.size == (640, 480)
            if frames_first:
                assert ids == [f.evidence_id for f in initial[0 if side == "a" else 1].result.frames]
            else:
                requested = await service.execute(frame_call(original.evidence_id,
                    [getattr(a, f"time_{side}_s") + .00001 for a in endpoints]), workspace_id="W1-3")
                assert [f.evidence_id for f in requested.result.frames] == ids
        repeated = await service.execute(call, workspace_id="W1-3")
        assert repeated.result_evidence_id == first.result_evidence_id
        changed = await service.execute({**call, "request": {"operation": "align_media",
            "params": {"mode": "visual", "top_k": 2}}}, workspace_id="W1-3")
        assert changed.status == "ready"
        assert changed.result.matches[0].anchors == anchors
        assert sum("image" in e.modalities for e in store.evidence_records()) == 4


@pytest.mark.asyncio
async def test_hi_receives_reuse_feedback_and_compacts_only_identical_successful_history(tmp_path):
    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.workspace import WorkspaceHarness
    path=tmp_path/'source.avi';make_video(path,frames=5)
    store=GlobalStore(tmp_path/'store.sqlite3',tmp_path/'artifacts')
    video=ingest(store,path.read_bytes(),source='I0',filename='source.avi',media_type='video/avi',modality='video')
    async with create_inspect_service(store=store) as service:
        harness=WorkspaceHarness(task=WorkspaceTask('W1-1','Compare frames',(video.evidence_id,)),store=store,inspect_service=service)
        first=await harness.execute_inspect(frame_call(video.evidence_id,[0,.1]))
        repeated=await harness.execute_inspect(frame_call(video.evidence_id,[0,.1]))
        assert first.execution_id != repeated.execution_id
        observation=harness.memory.latest_observation
        assert observation['details']['execution']['reused'] is True
        assert observation['details']['execution']['created_evidence_ids']==[]
        assert 'no new Evidence' in observation['items'][0]
        assert 'focus_evidence_ids' in observation['items'][0]
        assert f'Result Evidence: {video.evidence_id},' not in observation['items'][0]
        assert len(harness.memory.history)==2
        assert len(harness.memory.to_model_view()['history'])==1
        await harness.execute_inspect(frame_call(video.evidence_id,[.2]))
        assert len(harness.memory.to_model_view()['history'])==2
        harness.set_focus([first.result.frames[0].evidence_id])
        await harness.execute_inspect(frame_call(video.evidence_id,[0,.1]))
        assert len(harness.memory.to_model_view()['history'])==3
        for _ in range(2):
            failed=await harness.execute_inspect(frame_call(video.evidence_id,[99]))
            assert failed.status=='failed'
        assert len(harness.memory.to_model_view()['history'])==5
        other=WorkspaceHarness(task=WorkspaceTask('W1-2','Independent task',(video.evidence_id,)),store=store,inspect_service=service)
        assert other.model_view()['history']==[]


@pytest.mark.asyncio
async def test_pdf_pages_reuse_across_requests_and_keep_source_and_render_settings(tmp_path):
    from foren_searcher.inspect.policy import InspectPolicy

    store = GlobalStore(tmp_path / 'store.sqlite3', tmp_path / 'artifacts')
    data = pdf_bytes()
    document = ingest(store, data, source='I0', filename='pages.pdf', media_type='application/pdf', modality='text')
    sibling = ingest(store, data, source='I1', filename='copy.pdf', media_type='application/pdf', modality='text')
    def call(source, pages):
        return {'targets': [source], 'request': {'operation': 'get_pages', 'params': {'pages': pages}}}
    async with create_inspect_service(store=store) as service:
        initial = await service.execute(call(document.evidence_id, [1, 2, 99]), workspace_id='W1-1')
        assert initial.status == 'partial'
        identifiers = [page.evidence_id for page in initial.result.pages]
        overlapping = await service.execute(call(document.evidence_id, [2]), workspace_id='W1-2')
        assert overlapping.result.pages[0].evidence_id == identifiers[1]
        reordered = await service.execute(call(document.evidence_id, [2, 1]), workspace_id='W1-2')
        assert [page.evidence_id for page in reordered.result.pages] == identifiers[::-1]
        duplicate = await service.execute(call(sibling.evidence_id, [1]), workspace_id='W1-2')
        assert duplicate.result.pages[0].evidence_id not in identifiers
        assert store.evidence(duplicate.result.pages[0].evidence_id).derived_from == (sibling.evidence_id,)
    policy = replace(InspectPolicy(), pdf_dpi=InspectPolicy().pdf_dpi * 2)
    async with create_inspect_service(store=store, policy=policy) as service:
        larger = await service.execute(call(document.evidence_id, [1]), workspace_id='W1-3')
        page = larger.result.pages[0]
        assert page.evidence_id not in identifiers
        assert page.width > initial.result.pages[0].width
        assert json.loads(store.evidence(page.evidence_id).locator)['dpi'] == policy.pdf_dpi
