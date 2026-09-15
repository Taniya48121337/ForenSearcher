import copy
import json
from dataclasses import replace

import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore
from inspect_helpers import ingest, make_video, pdf_bytes


def material(tmp_path, kind):
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    if kind == "frames":
        path = tmp_path / "video.avi"
        make_video(path)
        evidence = store.commit_ingested_evidence(source_ref="I0", media_type="video/x-msvideo",
            modalities=("video",), label="video", original_filename=path.name, path=path)
    else:
        evidence = ingest(store, pdf_bytes(), source="I0", filename="pages.pdf", media_type="application/pdf", modality="text")
    return store, evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field,requested,valid,invalid", [
    ("frames", "times_s", [2.8, 30, .1, 3], [2.8, .1], [30., 3.]),
    ("pages", "pages", [2, 9, 1, 3], [2, 1], [9, 3]),
])
async def test_partial_explicit_selection_publishes_only_valid_media_and_keeps_request_audit(tmp_path, kind, field, requested, valid, invalid):
    store, evidence = material(tmp_path, kind)
    call = {"targets": [evidence.evidence_id], "request": {"operation": "get_" + kind, "params": {field: requested}}}
    original = copy.deepcopy(call)
    async with create_inspect_service(store=store) as service:
        result = await service.execute(call, workspace_id="W1-1")
        assert result.status == "partial"
        assert result.feedback[0].severity == "error" and result.feedback[0].code == "OUT_OF_RANGE"
        assert result.feedback[0].locator[field] == invalid
        assert result.feedback[0].target_evidence_id == evidence.evidence_id
        items = getattr(result.result, kind)
        assert [getattr(item, "time_s" if kind == "frames" else "page_number") for item in items] == valid
        for item in items:
            saved = store.evidence(item.evidence_id)
            assert saved.derived_from == (evidence.evidence_id,)
            assert store.artifact_path(saved.artifact_ref).is_file()
            locator = json.loads(saved.locator)
            assert locator["requested_time_s" if kind == "frames" else "page_number"] in valid
        audit = store.inspect_execution(result.execution_id)
        assert audit["request"] == original
        assert audit["details"]["effective_request"]["request"]["params"][field] == valid
        assert call == original


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,params", [
    ("frames", {"times_s": [3, 30]}),
    ("pages", {"pages": [3, 9]}),
    ("frames", {"range_s": [0, 4], "count": 2}),
    ("pages", {"page_range": [1, 3], "count": 2}),
    ("frames", {"times_s": [0, 1, 30]}),
    ("pages", {"pages": [1, 2, 9]}),
])
async def test_empty_valid_selection_ranges_and_oversized_requests_never_reach_backend(tmp_path, monkeypatch, kind, params):
    store, evidence = material(tmp_path, kind)
    policy = replace(InspectPolicy(), maximum_frames=2, default_frame_count=1, maximum_pages=2, default_page_count=1)
    async with create_inspect_service(store=store, policy=policy) as service:
        calls = []
        async def execute(*args, **kwargs):
            calls.append(args)
            raise AssertionError("invalid request reached backend")
        monkeypatch.setattr(service.dispatcher, "execute", execute)
        result = await service.execute({"targets": [evidence.evidence_id], "request": {"operation": "get_" + kind, "params": params}}, workspace_id="W1-1")
        assert result.status == "failed" and result.result is None
        assert result.feedback[0].code in {"OUT_OF_RANGE", "RESOURCE_LIMIT"}
        assert calls == []
