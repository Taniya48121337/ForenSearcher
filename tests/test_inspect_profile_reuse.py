"""已完成媒体探测的复用、完整性和请求范围校验。"""

import asyncio
from dataclasses import replace
import io
import threading

from PIL import Image, ImageOps
import pytest

from foren_searcher.inspect.backends.base import BackendFailure
from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore
from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes


class ProbeBackend(LocalInspectBackend):
    def __init__(self):
        super().__init__()
        self.probes = 0

    async def probe(self, path, media_kind):
        self.probes += 1
        return await super().probe(path, media_kind)


def call(evidence, operation, params):
    return {"targets": [evidence.evidence_id], "request": {"operation": operation, "params": params}}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "document", "video"])
async def test_profiles_shared_across_requests_and_workspaces(tmp_path, kind):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    if kind == "image":
        data, filename, media_type = image_bytes(), "source.png", "image/png"
        operation, selections = "crop", [{"box": [0, 0, .5, 1]}, {"box": [.5, 0, 1, 1]}]
    elif kind == "document":
        data, filename, media_type = pdf_bytes(), "source.pdf", "application/pdf"
        operation, selections = "get_pages", [{"pages": [1]}, {"pages": [2]}]
    else:
        path = tmp_path / "source.avi"
        make_video(path)
        data, filename, media_type = path.read_bytes(), path.name, "video/avi"
        operation, selections = "get_frames", [{"times_s": [.1]}, {"times_s": [2.1]}]
    original = ingest(store, data, source="I0", filename=filename, media_type=media_type,
                      modality="text" if kind == "document" else kind)
    backend = ProbeBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        first = await service.execute(call(original, operation, selections[0]), workspace_id="W1-1")
        second = await service.execute(call(original, operation, selections[1]), workspace_id="W1-2")
        repeated = await service.execute(call(original, operation, selections[0]), workspace_id="W1-3")
        assert first.status == second.status == repeated.status == "ready"
        assert backend.probes == 1
        assert first.result == repeated.result
        assert first.execution_id != repeated.execution_id
        assert first.result.referenced_evidence_ids() != second.result.referenced_evidence_ids()
        for result in (first, second):
            for evidence_id in result.result.referenced_evidence_ids():
                derived = store.evidence(evidence_id)
                assert derived.derived_from == (original.evidence_id,)
                with Image.open(store.artifact_path(derived.artifact_ref)) as rendered:
                    assert rendered.width > 0 and rendered.height > 0
        if kind == "image":
            assert (first.result.width, first.result.height) == (160, 180)
            assert second.result.box_px == (160, 0, 320, 180)
        elif kind == "document":
            assert [p.page_number for p in second.result.pages] == [2]
        else:
            assert [f.time_s for f in second.result.frames] == [2.1]
    async with create_inspect_service(store=store, backend=backend) as service:
        assert (await service.execute(call(original, operation, selections[0]), workspace_id="W2-1")).status == "ready"
        assert backend.probes == 2


@pytest.mark.asyncio
async def test_cached_profile_uses_oriented_pixels_and_distinguishes_sources(tmp_path):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    data = io.BytesIO()
    image = Image.open(io.BytesIO(image_bytes(size=(320, 180))))
    exif = Image.Exif()
    exif[274] = 6
    image.save(data, "JPEG", exif=exif)
    originals = [ingest(store, data.getvalue(), source=f"I{i}", filename="source.jpg",
                        media_type="image/jpeg", modality="image") for i in range(2)]
    backend = ProbeBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        for index, original in enumerate(originals):
            await service.execute(call(original, "crop", {"box": [0, 0, .5, 1]}), workspace_id="W1-1")
            result = await service.execute(call(original, "crop", {"box": [0, 0, 1, 1]}), workspace_id="W1-2")
            assert result.status == "ready"
            assert (result.result.width, result.result.height) == (180, 320)
            assert backend.probes == index + 1
            derived = store.evidence(result.result.evidence_id)
            assert derived.derived_from == (original.evidence_id,)
            with Image.open(io.BytesIO(data.getvalue())) as raw, Image.open(store.artifact_path(derived.artifact_ref)) as cropped:
                assert cropped.tobytes() == ImageOps.exif_transpose(raw).convert("RGB").tobytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["same_size", "missing"])
async def test_cached_profile_never_skips_source_integrity(tmp_path, damage):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, image_bytes(), source="I0", filename="source.png", media_type="image/png", modality="image")
    backend = ProbeBackend()
    request = call(original, "crop", {"box": [0, 0, 1, 1]})
    async with create_inspect_service(store=store, backend=backend) as service:
        assert (await service.execute(request, workspace_id="W1-1")).status == "ready"
        path = store.artifact_path(original.artifact_ref)
        if damage == "missing":
            path.unlink()
        else:
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
        result = await service.execute(request, workspace_id="W1-2")
        assert result.status == "failed"
        assert result.result is None
        assert backend.probes == 1


@pytest.mark.asyncio
async def test_unverified_mutable_media_is_probed_on_each_request(tmp_path):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts", verify_artifacts_on_read=False)
    original = ingest(store, image_bytes(), source="I0", filename="source.png", media_type="image/png", modality="image")
    backend = ProbeBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        first = await service.execute(call(original, "crop", {"box": [0, 0, 1, 1]}), workspace_id="W1-1")
        assert first.status == "ready"
        store.artifact_path(original.artifact_ref).write_bytes(image_bytes(size=(640, 360)))
        second = await service.execute(call(original, "crop", {"box": [0, 0, .5, 1]}), workspace_id="W1-2")
        assert second.status == "ready"
        assert (second.result.width, second.result.height) == (320, 360)
        assert backend.probes == 2


@pytest.mark.asyncio
async def test_cached_profile_still_validates_current_range_and_policy(tmp_path):
    path = tmp_path / "source.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, path.read_bytes(), source="I0", filename=path.name, media_type="video/avi", modality="video")
    backend = ProbeBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        request = call(original, "get_frames", {"times_s": [.1]})
        assert (await service.execute(request, workspace_id="W1-1")).status == "ready"
        invalid = await service.execute(call(original, "get_frames", {"times_s": [3.1]}), workspace_id="W1-2")
        assert invalid.status == "failed" and invalid.feedback[0].code == "OUT_OF_RANGE"
        service.policy = replace(service.policy, maximum_duration_s=1)
        limited = await service.execute(request, workspace_id="W1-3")
        assert limited.status == "failed" and limited.feedback[0].code == "RESOURCE_LIMIT"
        assert backend.probes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["failure", "cancel"])
async def test_incomplete_probe_is_not_reused(tmp_path, interruption):
    entered, release = threading.Event(), threading.Event()

    class InterruptedBackend(ProbeBackend):
        async def probe(self, path, media_kind):
            if self.probes == 0:
                self.probes += 1
                if interruption == "failure":
                    raise BackendFailure("DECODE_FAILED", "probe unavailable", stage="probe")
                entered.set()
                assert release.wait(5)
                return await LocalInspectBackend.probe(self, path, media_kind)
            return await super().probe(path, media_kind)

    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    original = ingest(store, image_bytes(), source="I0", filename="source.png", media_type="image/png", modality="image")
    backend = InterruptedBackend()
    async with create_inspect_service(store=store, backend=backend) as service:
        request = call(original, "crop", {"box": [0, 0, 1, 1]})
        if interruption == "failure":
            failed = await service.execute(request, workspace_id="W1-1")
            assert failed.status == "failed" and failed.feedback[0].code == "DECODE_FAILED"
        else:
            pending = asyncio.create_task(service.execute(request, workspace_id="W1-1"))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            finally:
                release.set()
        assert not service._media_profiles
        retried = await service.execute(request, workspace_id="W1-2")
        repeated = await service.execute(request, workspace_id="W1-3")
        assert retried.status == repeated.status == "ready"
        assert retried.result == repeated.result
        assert backend.probes == 2
        assert store.inspect_execution(1)["status"] == "failed"
    assert not service._media_profiles
