"""INSPECT 工具结果缓存：命中后证据记录形态与未命中一致、E 编号照常由 Service 分配、换编号也能命中、
refresh / off 语义、可重试问题不缓存、key 随模型与参数变化；外加重跑方差脚本用假 backend 跑通。"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from foren_searcher.inspect.backends.base import BackendFailure, BackendOutcome, MediaProfileData, PendingArtifact
from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.tool_cache import CachingProcessor, ToolCache, apply_tool_cache
from foren_searcher.store import GlobalStore

from inspect_helpers import image_bytes, ingest


def _png(color: tuple[int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(out, "PNG")
    return out.getvalue()


class FakeBackend(LocalInspectBackend):
    """OCR / 抽帧 / 问答都是假的，calls 记录真正"计算"的次数；text 可改，用来模拟不确定输出。"""

    def __init__(self, text: str = "HELLO 2026", *, flaky: bool = False) -> None:
        super().__init__()
        self.calls: dict[str, int] = {}
        self.text, self.flaky = text, flaky

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    async def probe(self, path, media_kind):
        if media_kind == "video":
            return MediaProfileData("video", width=32, height=24, duration_s=5.0, fps=10.0, frame_count=50, audio_tracks=1)
        return await super().probe(path, media_kind)

    async def ocr(self, media, params, policy):
        self._count("ocr")
        items = [{"text": self.text, "box": (0.1, 0.1, 0.5, 0.5), "confidence": 0.93}]
        return BackendOutcome({"coverage": None, "results": [{"image_handle": None, "image_evidence_id": media[0].evidence_id,
                                                              "time_s": None, "items": items}]},
                              details={"backend": "fake-ocr", "limitations": ["测试用"]})

    async def get_frames(self, media, params, policy):
        self._count("get_frames")
        evidence_id = media[0].evidence_id
        artifacts = tuple(PendingArtifact(f"frame:{i}", _png((10 * i, 20, 30)), "image/png", ".png", "video_frame", evidence_id,
                                          f"视频{evidence_id}在{t:.3f}秒的帧", {"time_s": t, "requested_time_s": t, "frame_duration_s": 0.1},
                                          "INSPECT/get_frames") for i, t in enumerate((1.0, 2.0)))
        issues = (BackendFailure("DECODE_FAILED", "one frame failed", stage="decode", retryable=True,
                                 locator={"requested_time_s": 3.0}),) if self.flaky else ()
        rows = [{"handle": f"frame:{i}", "time_s": t, "width": 32, "height": 24, "frame_duration_s": 0.1} for i, t in enumerate((1.0, 2.0))]
        return BackendOutcome({"frames": rows}, artifacts, issues, {"backend": "fake-av"})

    async def visual_qa(self, media, params, policy):
        self._count("visual_qa")
        return BackendOutcome({"image_evidence_ids": [m.evidence_id for m in media], "answer": f"{media[0].evidence_id}: {self.text}",
                               "limitations": []}, details={"backend": "fake-vlm"})


def _service(tmp_path: Path, name: str, backend: FakeBackend, cache: ToolCache | None, *, padding: int = 0):
    store = GlobalStore(tmp_path / name / "s.sqlite3", tmp_path / name / "artifacts")
    for index in range(padding):          # 让同一张图在这个 run 里拿到不同的 E 编号
        ingest(store, image_bytes(color="black", red_box=False), source=f"P{index}", filename=f"p{index}.png",
               media_type="image/png", modality="image")
    image = ingest(store, image_bytes(), source="I0", filename="a.png", media_type="image/png", modality="image")
    video = ingest(store, b"fixture-video", source="V0", filename="v.mp4", media_type="video/mp4", modality="video")
    service = create_inspect_service(store=store, backend=backend)
    apply_tool_cache(service, cache)
    return store, service, image.evidence_id, video.evidence_id


def _shape(store: GlobalStore, evidence_id: str, mapping: dict[str, str]) -> dict:
    """证据记录里与编号无关的部分；derived_from 按本 run 的编号映射成角色名再比。"""

    evidence = store.evidence(evidence_id)
    return {"modalities": evidence.modalities, "label": _swap(evidence.label, mapping), "content": _swap(evidence.content, mapping),
            "locator": _swap(evidence.locator, mapping), "acquired_by": evidence.acquired_by,
            "production_method": evidence.production_method, "derived_from": tuple(mapping.get(x, x) for x in evidence.derived_from),
            "bytes": store.artifact_path(evidence.artifact_ref).read_bytes() if evidence.artifact_ref else None}


def _swap(text: str, mapping: dict[str, str]) -> str:
    import re

    return re.sub(r"(?<![A-Za-z0-9])E\d+(?![0-9])", lambda m: mapping.get(m.group(0), m.group(0)), text)


@pytest.mark.asyncio
async def test_cache_hit_keeps_evidence_shape_and_normal_id_allocation(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first_backend = FakeBackend()
    store_a, service_a, image_a, video_a = _service(tmp_path, "a", first_backend, ToolCache(cache_root, "on"))
    ocr_a = await service_a.execute({"targets": [image_a], "request": {"operation": "ocr", "params": {}}}, workspace_id="W1-1")
    frames_a = await service_a.execute({"targets": [video_a], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-1")
    assert first_backend.calls == {"ocr": 1, "get_frames": 1}

    second_backend = FakeBackend(text="SHOULD NOT BE USED")
    cache_b = ToolCache(cache_root, "on")
    store_b, service_b, image_b, video_b = _service(tmp_path, "b", second_backend, cache_b, padding=2)
    assert image_b != image_a                                            # 同一张图，这个 run 编号不同
    ocr_b = await service_b.execute({"targets": [image_b], "request": {"operation": "ocr", "params": {}}}, workspace_id="W1-1")
    frames_b = await service_b.execute({"targets": [video_b], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-1")
    assert second_backend.calls == {}                                    # 全部命中，没有重算
    assert cache_b.summary()["counts"]["hit"] == 2 and cache_b.summary()["hit_rate"] == 1.0

    # OCR 结果引用的是本 run 的图片 E；结果 E 仍由 Store 正常分配，内容形态与未命中一致
    assert ocr_b.result.results[0].image_evidence_id == image_b and ocr_b.result.results[0].items[0].text == "HELLO 2026"
    assert ocr_b.to_dict()["status"] == ocr_a.to_dict()["status"] == "ready"
    map_a, map_b = {image_a: "IMG", video_a: "VID"}, {image_b: "IMG", video_b: "VID"}
    assert _shape(store_b, ocr_b.result_evidence_id, map_b) == _shape(store_a, ocr_a.result_evidence_id, map_a)
    # 抽帧：帧 E 新分配，标签里的视频编号换成了本 run 的，字节、locator、derived_from 与未命中时一致
    frame_ids_a = [f.evidence_id for f in frames_a.result.frames]
    frame_ids_b = [f.evidence_id for f in frames_b.result.frames]
    assert len(frame_ids_b) == 2 and store_b.evidence(frame_ids_b[0]).label == f"视频{video_b}在1.000秒的帧"
    assert [_shape(store_b, e, map_b) for e in frame_ids_b] == [_shape(store_a, e, map_a) for e in frame_ids_a]
    details = store_b.inspect_execution(ocr_b.execution_id)["details"]
    assert details["tool_cache"]["status"] == "hit" and details["created_evidence_ids"] == [ocr_b.result_evidence_id]


@pytest.mark.asyncio
async def test_off_is_a_noop_and_refresh_recomputes(tmp_path: Path) -> None:
    backend = FakeBackend()
    _, service, image, _ = _service(tmp_path, "off", backend, ToolCache(tmp_path / "c", "off"))
    assert not any(isinstance(p, CachingProcessor) for p in service.dispatcher.processors.values())
    _, service, image, _ = _service(tmp_path, "none", backend, None)
    assert not any(isinstance(p, CachingProcessor) for p in service.dispatcher.processors.values())

    root = tmp_path / "cache"
    _, service, image, _ = _service(tmp_path, "r1", FakeBackend("old"), ToolCache(root, "on"))
    await service.execute({"targets": [image], "request": {"operation": "ocr", "params": {}}}, workspace_id="W1-1")
    refreshed = FakeBackend("new")
    cache = ToolCache(root, "refresh")
    _, service, image, _ = _service(tmp_path, "r2", refreshed, cache)
    response = await service.execute({"targets": [image], "request": {"operation": "ocr", "params": {}}}, workspace_id="W1-1")
    assert refreshed.calls == {"ocr": 1} and response.result.results[0].items[0].text == "new"
    assert cache.summary()["counts"]["refreshed"] == 1
    reader = FakeBackend("unused")
    _, service, image, _ = _service(tmp_path, "r3", reader, ToolCache(root, "on"))
    response = await service.execute({"targets": [image], "request": {"operation": "ocr", "params": {}}}, workspace_id="W1-1")
    assert reader.calls == {} and response.result.results[0].items[0].text == "new"    # 覆盖后的新值


@pytest.mark.asyncio
async def test_retryable_issues_are_not_cached(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = ToolCache(root, "on")
    _, service, _, video = _service(tmp_path, "a", FakeBackend(flaky=True), cache)
    response = await service.execute({"targets": [video], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-1")
    assert response.status == "partial" and cache.summary()["counts"]["not_cacheable"] == 1
    again = FakeBackend()
    _, service, _, video = _service(tmp_path, "b", again, ToolCache(root, "on"))
    await service.execute({"targets": [video], "request": {"operation": "get_frames", "params": {"count": 2}}}, workspace_id="W1-1")
    assert again.calls == {"get_frames": 1}


@pytest.mark.asyncio
async def test_key_depends_on_params_model_and_context_sensitive_ids(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    backend = FakeBackend()
    _, service, _, video = _service(tmp_path, "a", backend, ToolCache(root, "on"))
    for count in (2, 2, 3):
        await service.execute({"targets": [video], "request": {"operation": "get_frames", "params": {"count": count}}}, workspace_id="W1-1")
    # Service 自己的同请求复用会挡住第二次；第三次参数不同 → 未命中
    assert backend.calls == {"get_frames": 2}

    cache = ToolCache(root, "on")
    processor = CachingProcessor(service.dispatcher.processors["ocr"].inner, cache)
    from foren_searcher.inspect.backends.base import PreparedMedia
    from foren_searcher.inspect.contracts import OcrParams, VisualQaParams
    from foren_searcher.inspect.policy import InspectPolicy

    path = tmp_path / "x.png"
    path.write_bytes(image_bytes())
    media = (PreparedMedia("E1", path, "image", MediaProfileData("image", width=320, height=180)),)
    renamed = (PreparedMedia("E9", path, "image", MediaProfileData("image", width=320, height=180)),)
    key = lambda op, m, params: cache.key(cache.key_material(processor.inner, op, m, params, InspectPolicy(), None))  # noqa: E731
    assert key("ocr", media, OcrParams()) == key("ocr", renamed, OcrParams())          # OCR 与编号无关
    assert key("visual_qa", media, VisualQaParams("q")) != key("visual_qa", renamed, VisualQaParams("q"))   # 问答提示词含编号
    before = key("asr", media, OcrParams())
    monkeypatch.setenv("FOREN_WHISPER_MODEL", "large-v3")
    assert key("asr", media, OcrParams()) != before                                   # 换模型换 key
    assert key("ocr", media, OcrParams()) == key("ocr", media, OcrParams())
    assert key("ocr", media, OcrParams()) != cache.key(cache.key_material(processor.inner, "ocr", media, OcrParams(),
                                                                          InspectPolicy(ocr_base_fps=4.0), None))
    assert key("ocr", media, OcrParams()) == cache.key(cache.key_material(processor.inner, "ocr", media, OcrParams(),
                                                                          InspectPolicy(execution_timeout_s=10.0), None))


@pytest.mark.asyncio
async def test_run_cases_inspect_service_for_wires_tool_cache(tmp_path: Path) -> None:
    from foren_searcher.datagen import run_cases

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "artifacts")
    service, cache, acquirer = run_cases.inspect_service_for(store)
    assert cache is None and acquirer is None
    assert not any(isinstance(p, CachingProcessor) for p in service.dispatcher.processors.values())
    await service.aclose()
    service, cache, _ = run_cases.inspect_service_for(store, tool_cache="on", tool_cache_root=tmp_path / "tc")
    assert cache.root == tmp_path / "tc" and cache.mode == "on"
    assert all(isinstance(p, CachingProcessor) for p in service.dispatcher.processors.values())
    await service.aclose()
    from foren_searcher.search.recording import RecordingAcquirer, SearchRecording

    recording = SearchRecording("replay", tmp_path / "snap", "true:1")
    service, _, acquirer = run_cases.inspect_service_for(store, recording=recording)
    assert isinstance(acquirer, RecordingAcquirer) and service.acquirer is acquirer
    await service.aclose()
    await acquirer.aclose()


# ---------------------------------------------------------------- 重跑方差脚本

def _variance_module():
    spec = importlib.util.spec_from_file_location("tool_cache_variance", Path(__file__).resolve().parents[1] / "scripts" / "tool_cache_variance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class JitterBackend(FakeBackend):
    """每次调用输出略有不同：OCR 文字、ASR 文本、对齐锚点都会抖。"""

    async def ocr(self, media, params, policy):
        self._count("ocr")
        text = ["HELLO 2026", "HELL0 2026", "HELLO 2026"][(self.calls["ocr"] - 1) % 3]
        return BackendOutcome({"coverage": None, "results": [{"image_handle": None, "image_evidence_id": media[0].evidence_id,
                                                              "time_s": None, "items": [{"text": text, "box": [0, 0, 1, 1], "confidence": 0.9}]}]})

    async def asr(self, media, params, policy):
        self._count("asr")
        text = ["the tower was hit by lightning", "the tower was hit by lightning", "the tower is hit by lightning"][(self.calls["asr"] - 1) % 3]
        return BackendOutcome({"coverage": {"range_s": [0, 5], "processed_ranges": [[0, 5]], "failed_ranges": []},
                               "language": "en", "audio_track": 0, "segments": [{"start_s": 0.0, "end_s": 2.0, "text": text}]})

    async def align_media(self, media, params, policy):
        self._count("align_media")
        shift = [0.0, 0.1, 0.5][(self.calls["align_media"] - 1) % 3]
        match = {"range_a_s": [1.0, 3.0], "range_b_s": [1.5 + shift, 3.5 + shift], "score": 0.8,
                 "anchors": [{"time_a_s": 1.0, "time_b_s": 1.5 + shift}, {"time_a_s": 2.0, "time_b_s": 2.5 + shift}]}
        side = {"range_s": [0, 5], "processed": {"unit": "seconds", "value": 5}, "failed_ranges": []}
        return BackendOutcome({"mode": "visual", "coverage": {"a": side, "b": side}, "matches": [match]})


def test_variance_script_runs_with_fake_backend(tmp_path: Path) -> None:
    module = _variance_module()
    import asyncio

    image = tmp_path / "a.png"
    image.write_bytes(image_bytes())
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16)
    backend = JitterBackend()

    outcomes, elapsed = asyncio.run(module.run_variance(lambda: backend, [image], "ocr", {}, runs=3))
    rows = module.compare("ocr", outcomes)
    assert [r["similarity"] for r in rows][0] == 1.0 and 0.8 < rows[1]["similarity"] < 1.0 and rows[2]["similarity"] == 1.0
    assert len(elapsed) == 3 and backend.calls["ocr"] == 3

    outcomes, _ = asyncio.run(module.run_variance(lambda: backend, [video], "asr", {}, runs=3))
    rows = module.compare("asr", outcomes)
    assert rows[1]["similarity"] == 1.0 and rows[2]["similarity"] < 1.0

    outcomes, elapsed = asyncio.run(module.run_variance(lambda: backend, [video, video], "align_media", {"mode": "visual"}, runs=3))
    rows = module.compare("align_media", outcomes)
    assert [r["anchor_diff_s"] for r in rows] == [0.0, 0.1, 0.5]
    assert rows[0]["iou"] == 1.0 and rows[2]["iou"] == pytest.approx((1.0 + 1.5 / 2.5) / 2, abs=1e-3)
    text = module.render_markdown("align_media", ["v.mp4", "v.mp4"], {"mode": "visual"}, rows, elapsed, "JitterBackend")
    assert "| 次 | 耗时(s) | 锚点偏移(s) | 锚点差(s) | 区间 IoU | 问题数 |" in text and "max_anchor_diff_s=0.5" in text

    outcomes, _ = asyncio.run(module.run_variance(lambda: backend, [video], "get_frames", {"count": 2}, runs=2))
    rows = module.compare("get_frames", outcomes)
    assert all(r["pixel_identical"] for r in rows) and rows[0]["images"] == 2


def test_variance_script_main_writes_markdown(tmp_path: Path, capsys) -> None:
    module = _variance_module()
    image = tmp_path / "a.png"
    image.write_bytes(image_bytes())
    out = tmp_path / "variance.md"
    code = module.main(["--media", str(image), "--operation", "ocr", "--runs", "2", "--out", str(out)],
                       backend_factory=JitterBackend)
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "### ocr 重跑方差（2 次，缓存关闭）" in text and "文本相似度" in text and "后端：JitterBackend" in text
    with pytest.raises(SystemExit):
        module.main(["--media", str(image), "--operation", "ocr", "--runs", "1"], backend_factory=JitterBackend)
