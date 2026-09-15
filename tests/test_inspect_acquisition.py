"""INSPECT 视频 acquisition 路径测试。

验证候选页面触发直链获取、缓存复用、临时文件清理、来源限制和失败映射。
"""

from __future__ import annotations

import json
import shutil
import asyncio
from pathlib import Path

import pytest

from foren_searcher.inspect.acquisition import AcquiredMedia, DirectMediaAcquirer
from foren_searcher.inspect.backends.base import BackendFailure
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore

from inspect_helpers import make_video


class FakeAcquirer:
    """测试辅助类 FakeAcquirer；用可控状态替代真实外部依赖。"""

    def __init__(self, source: Path, destination: Path) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.source = source
        self.destination = destination
        self.calls = 0
        self.urls: list[str] = []

    async def acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy) -> AcquiredMedia:
        """模拟 acquire 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        self.urls.append(source_url)
        self.destination = destination
        shutil.copyfile(self.source, destination)
        return AcquiredMedia(destination, "video/x-msvideo", ".avi", "candidate.avi", {"requested_url": source_url, "final_url": source_url, "byte_size": destination.stat().st_size})


class HangingAcquirer:
    """测试辅助类 HangingAcquirer；用可控状态替代真实外部依赖。"""

    async def acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy) -> AcquiredMedia:
        """模拟 acquire 接口，返回由当前用例预设的可控结果。"""

        destination.write_bytes(b"partial download")
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


def _candidate_page(store: GlobalStore, *, url: str = "https://video.example.test/watch/1"):
    """测试辅助函数 _candidate_page；为用例提供可重复的输入或可控行为。"""

    candidate_id, _ = store.get_or_create_candidate(identity_key=f"url:{url}", source_url=url, candidate_type="video")
    page = store.commit_material_evidence(
        candidate_id=candidate_id, material_kind="page", material_status="ready",
        artifact_bytes=b"<html>video page</html>", media_type="text/html", suffix=".html",
        modalities=("text",), label="video source page", content="[1] video page",
        locator="paragraph 1", acquired_by="SEARCH/open/http", discovered_by="W1-1",
    )
    return candidate_id, page


@pytest.mark.asyncio
async def test_source_page_acquires_video_once_then_reuses_candidate_cache(tmp_path: Path):
    """验证 source page acquires video once then reuses candidate cache 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    source = tmp_path / "source.avi"
    make_video(source)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    candidate_id, page = _candidate_page(store)
    acquirer = FakeAcquirer(source, tmp_path / "owned-download.avi")
    service = create_inspect_service(store=store, acquirer=acquirer)
    try:
        first = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-1")
        second = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-2")
        assert first.status == second.status == "ready"
        assert acquirer.calls == 1
        video_rows = store.candidate_evidence(candidate_id, "video")
        assert len(video_rows) == 1
        video = video_rows[0][2]
        assert video.acquired_by == "INSPECT/acquire" and video.derived_from == ()
        assert any(card.evidence_id == video.evidence_id for card in first.evidence)
        assert not acquirer.destination.exists()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_source_page_cache_skips_newer_missing_artifact(tmp_path: Path):
    """验证 source page cache skips newer missing artifact 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    source = tmp_path / "source.avi"
    make_video(source)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    candidate_id, page = _candidate_page(store)
    old_path = store.new_owned_temp_path(".avi")
    new_path = store.new_owned_temp_path(".avi")
    shutil.copyfile(source, old_path)
    old = store.commit_candidate_file_evidence(candidate_id=candidate_id, owned_temp_path=old_path, media_type="video/x-msvideo", suffix=".avi", label="old", content="{}", locator="{}", discovered_by="W1-1")
    shutil.copyfile(source, new_path)
    newest = store.commit_candidate_file_evidence(candidate_id=candidate_id, owned_temp_path=new_path, media_type="video/x-msvideo", suffix=".avi", label="new", content="{}", locator="{}", discovered_by="W1-1")
    store.artifact_path(newest.artifact_ref).unlink()
    acquirer = FakeAcquirer(source, tmp_path / "unexpected.avi")
    service = create_inspect_service(store=store, acquirer=acquirer)
    try:
        response = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 1}}}, workspace_id="W1-2")
        assert response.status == "ready", [item.to_dict() for item in response.feedback]
        assert acquirer.calls == 0
        assert any(card.evidence_id == old.evidence_id for card in response.evidence)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_non_page_material_cannot_trigger_candidate_video_acquisition(tmp_path: Path):
    """验证 non page material cannot trigger candidate video acquisition 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    source = tmp_path / "source.avi"
    make_video(source)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    candidate_id, _ = _candidate_page(store)
    cover = store.commit_material_evidence(
        candidate_id=candidate_id, material_kind="cover", material_status="ready",
        artifact_bytes=b"not an image", media_type="image/jpeg", suffix=".jpg", modalities=("image",),
        label="cover", content="{}", locator="complete", acquired_by="SEARCH/open", discovered_by="W1-1",
    )
    acquirer = FakeAcquirer(source, tmp_path / "unexpected.avi")
    service = create_inspect_service(store=store, acquirer=acquirer)
    try:
        response = await service.execute({"targets": [cover.evidence_id], "request": {"operation": "get_frames", "params": {"count": 1}}}, workspace_id="W1-1")
        assert response.status == "failed"
        assert acquirer.calls == 0
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_saved_youtube_metadata_selects_progressive_audio_video_format(tmp_path: Path):
    """验证 saved youtube metadata selects progressive audio video format 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    source = tmp_path / "source.avi"
    make_video(source)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    candidate_id, _ = store.get_or_create_candidate(identity_key="youtube:test", source_url="https://www.youtube.com/watch?v=test", candidate_type="video", platform="youtube")
    metadata = {
        "extractor_key": "Youtube",
        "formats": [
            {"url": "https://cdn.example.test/video-only", "protocol": "https", "ext": "mp4", "height": 1080, "vcodec": "avc1", "acodec": "none"},
            {"url": "https://cdn.example.test/progressive-360.mp4", "protocol": "https", "ext": "mp4", "height": 360, "vcodec": "avc1", "acodec": "mp4a"},
            {"url": "https://cdn.example.test/progressive-720.mp4", "protocol": "https", "ext": "mp4", "height": 720, "vcodec": "avc1", "acodec": "mp4a"},
        ],
    }
    page = store.commit_material_evidence(candidate_id=candidate_id, material_kind="page", material_status="ready", artifact_bytes=json.dumps(metadata).encode(), media_type="application/json", suffix=".json", modalities=("text",), label="YouTube metadata", content="{}", locator="fields", acquired_by="SEARCH/open/yt-dlp", discovered_by="W1-1")
    acquirer = FakeAcquirer(source, tmp_path / "owned.avi")
    service = create_inspect_service(store=store, acquirer=acquirer)
    try:
        response = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 1}}}, workspace_id="W1-1")
        assert response.status == "ready", response.to_dict()
        assert acquirer.urls == ["https://cdn.example.test/progressive-720.mp4"]
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_default_acquirer_rejects_private_network_before_download():
    """验证 default acquirer rejects private network before download 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    acquirer = DirectMediaAcquirer()
    try:
        with pytest.raises(BackendFailure) as caught:
            await acquirer.acquire("http://127.0.0.1/private.avi", destination=Path.cwd() / ".pytest-private-download.avi", policy=InspectPolicy(acquisition_timeout_s=2.0))
        assert caught.value.code == "ACCESS_RESTRICTED"
        assert caught.value.stage == "acquire"
    finally:
        await acquirer.aclose()


def test_store_never_takes_ownership_of_an_external_path(tmp_path: Path):
    """验证 store never takes ownership of an external path 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    source = tmp_path / "user-owned.avi"
    make_video(source)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    candidate_id, _ = store.get_or_create_candidate(identity_key="url:https://example.test/video", source_url="https://example.test/video", candidate_type="video")
    with pytest.raises(ValueError, match="outside"):
        store.commit_candidate_file_evidence(candidate_id=candidate_id, owned_temp_path=source, media_type="video/x-msvideo", suffix=".avi", label="video", content="{}", locator="{}", discovered_by="W1-1")
    assert source.exists()


@pytest.mark.asyncio
async def test_acquisition_timeout_removes_partial_owned_file(tmp_path: Path):
    """验证 acquisition timeout removes partial owned file 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    _, page = _candidate_page(store)
    service = create_inspect_service(store=store, acquirer=HangingAcquirer(), policy=InspectPolicy(execution_timeout_s=0.03))
    try:
        response = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 1}}}, workspace_id="W1-1")
        assert response.status == "failed"
        assert response.feedback[0].code == "TIMEOUT"
        assert list(store.incoming_root.iterdir()) == []
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.asyncio
async def test_public_direct_video_is_acquired_and_inspected(tmp_path: Path):
    """验证 public direct video is acquired and inspected 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    url = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/video/768x576.avi"
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    _, page = _candidate_page(store, url=url)
    service = create_inspect_service(store=store, policy=InspectPolicy(acquisition_timeout_s=60.0))
    try:
        response = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames", "params": {"count": 3}}}, workspace_id="W-live")
        assert response.status == "ready", [item.to_dict() for item in response.feedback]
        acquired = [card for card in response.evidence if card.acquired_by == "INSPECT/acquire"]
        assert len(acquired) == 1
        execution = store.inspect_execution(1)
        details = execution["details"]
        assert details["acquisition"][0]["result_evidence_id"] == acquired[0].evidence_id
    finally:
        await service.aclose()
