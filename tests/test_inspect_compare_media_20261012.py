"""10-12 DETECT 并入 INSPECT：compare_media / video_structure / asr 声音时间线 / 低清兜底（合并方案第 7 节）。

合成对照（答案已知）：从一段平移的纹理视频 B 做出删段、插入、重排、2 倍速、倒放、镜像、定格，逐项要报得出来，
这是删掉 align_media / diff_media 的前提（方案 7.6）。
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from foren_searcher.inspect.acquisition import AcquiredMedia
from foren_searcher.inspect.backends.base import BackendFailure
from foren_searcher.inspect.contracts import MERGED_OPERATIONS
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.source_compare import SourceComparer
from foren_searcher.store import GlobalStore

FPS, W, H = 25, 320, 180


def _texture(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur((rng.random((900, 2400, 3)) * 255).astype(np.uint8), (0, 0), 3)
    return cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)


def _frames(tex: np.ndarray, seconds: float, *, y0: int = 300, speed: float = 40.0) -> list[np.ndarray]:
    out = []
    for t in range(int(seconds * FPS)):
        s = t / FPS
        x = int(speed * s + 30 * np.sin(s * 0.7))
        y = int(y0 + 120 * np.sin(s * 0.31))
        out.append(tex[y:y + H, x:x + W].copy())
    return out


def _write(path: Path, frames: list[np.ndarray]) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        out.write(f)
    out.release()
    return path


@pytest.fixture(scope="module")
def edits(tmp_path_factory):
    root = tmp_path_factory.mktemp("edits")
    b = _frames(_texture(), 24)
    seg = lambda a, z: b[int(a * FPS):int(z * FPS)]  # noqa: E731
    rng = np.random.default_rng(1)
    noise = [(rng.random((H, W, 3)) * 60 + np.linspace(0, 180, W)[None, :, None]).astype(np.uint8) for _ in range(4 * FPS)]
    variants = {
        "cut": seg(0, 8) + seg(13, 24),
        "insert": seg(0, 8) + noise + seg(8, 20),
        "reorder": seg(12, 24) + seg(0, 12),
        "speed": b[::2],
        "reverse": seg(4, 16)[::-1],
        "mirror": [cv2.flip(f, 1) for f in seg(2, 20)],
        "freeze": seg(0, 6) + [b[6 * FPS]] * (3 * FPS) + seg(6, 18),
        "unrelated": _frames(_texture(9), 12, y0=200, speed=-25.0)[::-1],
        "slowmo": [f for f in seg(3, 4.2) for _ in range(8)],          # VideoSham 0069: 9.6 s of A show 1.2 s of B
    }
    paths = {"B": _write(root / "B.mp4", b)}
    for name, frames in variants.items():
        paths[name] = _write(root / f"{name}.mp4", frames)
    return paths


def _compare(paths, name):
    result, _images = SourceComparer().compare(paths[name], paths["B"])
    return result


def test_cut_reports_the_skipped_part_of_b(edits):
    r = _compare(edits, "cut")
    assert r["match"]["fraction_of_a_matched"] >= 0.95
    (cut,) = r["b_cuts"]
    assert abs(cut["b_missing_s"][0] - 8.0) <= 0.6 and abs(cut["b_missing_s"][1] - 13.0) <= 0.6
    assert any(line.startswith("Missing from A:") for line in r["summary_lines"])


def test_insert_is_footage_only_in_a_with_b_continuous(edits):
    r = _compare(edits, "insert")
    (segment,) = r["segments_not_in_b"]
    assert segment["context"] == "inserted"
    assert abs(segment["a_range_s"][0] - 8.0) <= 0.6 and abs(segment["a_range_s"][1] - 12.0) <= 0.6


def test_reorder_speed_reverse_mirror_freeze(edits):
    assert _compare(edits, "reorder")["reorders"]
    kinds = {name: [(c["kind"], c["mirrored"]) for c in _compare(edits, name)["correspondences"]]
             for name in ("speed", "reverse", "mirror", "freeze")}
    assert kinds["speed"] == [("sped_up", False)]
    assert kinds["reverse"] == [("reversed", False)]
    assert kinds["mirror"] == [("normal", True)]
    assert ("frozen", False) in kinds["freeze"]
    speed = _compare(edits, "speed")["correspondences"][0]["mapping"]["scale"]
    assert 1.9 <= speed <= 2.1


def test_strong_slow_motion_is_slowed_not_dropped(edits):
    (corr,) = _compare(edits, "slowmo")["correspondences"]
    assert corr["kind"] == "slowed" and corr["mapping"]["scale"] < 0.2
    assert 2.8 <= corr["b_range_s"][0] <= 3.4


def test_match_is_high_for_a_source_and_low_for_an_unrelated_video(edits):
    comparer = SourceComparer()
    assert comparer.match(edits["cut"], edits["B"])["fraction"] >= 0.9
    other = comparer.match(edits["unrelated"], edits["B"])
    assert other["fraction"] < 0.08 and set(other) >= {"fraction", "engine", "a_duration_s", "b_duration_s", "runs"}


class _Describer:
    def __init__(self):
        self.calls = []

    def describe(self, jpeg: bytes, question: str) -> str:
        self.calls.append(question)
        return "a grey striped card"

    def identity(self):
        return {"model": "fake", "endpoint": "none", "source": "test"}


@pytest.mark.asyncio
async def test_compare_media_through_the_service_returns_images_and_describer_lines(edits, tmp_path):
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    ids = [store.commit_ingested_evidence(source_ref=f"I{k}", media_type="video/mp4", modalities=("video",), label=n,
                                          original_filename=f"{n}.mp4", path=edits[n]).evidence_id
           for k, n in enumerate(("insert", "B"))]
    async with create_inspect_service(store=store, enabled_operations=MERGED_OPERATIONS) as service:
        describer = _Describer()
        service.dispatcher.probe_backend.describer = describer
        response = await service.execute({"targets": ids, "request": {"operation": "compare_media", "params": {}}},
                                         workspace_id="W1-1")
    assert response.status == "ready" and response.operation == "compare_media"
    result = response.result.to_dict()
    assert any("added to A" in line for line in result["summary_lines"])
    assert result["images"] and all(i["evidence_id"].startswith("E") for i in result["images"])
    # 10-13：visual check 接在它描述的那一行后面（Not in B: … Visual check of comparison image k: …）
    assert describer.calls and any(line.startswith("Not in B:") and "Visual check of comparison image" in line
                                   for line in result["summary_lines"])
    image = store.evidence(result["images"][0]["evidence_id"])
    assert image.acquired_by == "INSPECT/compare_media"


@pytest.mark.asyncio
async def test_video_structure_lists_ten_second_shot_freeze_and_black_end(tmp_path):
    def tint(frames, colour):
        return [np.clip(f.astype(np.int16) * 0.6 + np.array(colour, np.int16), 0, 255).astype(np.uint8) for f in frames]

    shots = [tint(_frames(_texture(3), 3, y0=150), (90, 0, 0)), tint(_frames(_texture(5), 10, y0=300), (0, 90, 0)),
             tint(_frames(_texture(7), 4, y0=450), (0, 0, 90))]
    frozen = [shots[2][-1]] * FPS
    black = [np.zeros((H, W, 3), np.uint8)] * (FPS // 2)
    path = _write(tmp_path / "s.mp4", shots[0] + shots[1] + shots[2] + frozen + shots[0] + black)
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    eid = store.commit_ingested_evidence(source_ref="I0", media_type="video/mp4", modalities=("video",), label="s",
                                         original_filename="s.mp4", path=path).evidence_id
    async with create_inspect_service(store=store, enabled_operations=MERGED_OPERATIONS) as service:
        response = await service.execute({"targets": [eid], "request": {"operation": "video_structure", "params": {}}},
                                         workspace_id="W1-1")
    assert response.status == "ready"
    result = response.result.to_dict()
    report = result["report"] if "report" in result else result
    flagged = report["generator_length_shots"]
    assert len(flagged) == 1 and abs(flagged[0]["duration_s"] - 10.0) <= 0.2
    assert report["frozen_spans"] and report["ends_on_black"] is True
    assert report["repeated_shots"] and result["images"]


def test_sound_timeline_separates_speech_other_sound_and_silence():
    from foren_searcher.inspect.backends.local import LocalInspectBackend

    rate = 16000
    t = np.arange(rate * 6) / rate
    samples = np.zeros_like(t, dtype=np.float32)
    samples[: rate * 2] = 0.3 * np.sin(2 * np.pi * 220 * t[: rate * 2])        # 0-2 s "speech"
    samples[rate * 3: rate * 5] = 0.2 * np.sin(2 * np.pi * 880 * t[rate * 3: rate * 5])   # 3-5 s other sound
    spans, method = LocalInspectBackend().sound_timeline(samples, rate, 0.0, [{"start_s": 0.0, "end_s": 2.0}])
    kinds = [(s["kind"], round(s["start_s"]), round(s["end_s"])) for s in spans]
    assert ("speech", 0, 2) in kinds and ("other_sound", 3, 5) in kinds
    assert any(k == "silence" for k, _, _ in kinds) and method.startswith("energy")


# ---------------------------------------------------------------- 低清纯画面兜底（原 DETECT _light_download）


class _LowResOnly:
    """整片下载一律 403；只给画面（video_only_height）时成功。"""

    def __init__(self, source: Path):
        self.source, self.calls = source, []

    async def acquire(self, source_url, *, destination, policy, headers=None, page_url=None, video_only_height=None):
        self.calls.append(video_only_height)
        if not video_only_height:
            raise BackendFailure("ACCESS_RESTRICTED", "HTTP Error 403: Forbidden", stage="acquire")
        shutil.copyfile(self.source, destination)
        return AcquiredMedia(destination, "video/mp4", ".mp4", "low.mp4",
                             {"requested_url": source_url, "byte_size": destination.stat().st_size,
                              "video_only_height": video_only_height})


def _video_page(store: GlobalStore, url: str = "https://www.youtube.com/watch?v=abcdefghijk"):
    candidate_id, _ = store.get_or_create_candidate(identity_key=f"url:{url}", source_url=url, candidate_type="video")
    page = store.commit_material_evidence(
        candidate_id=candidate_id, material_kind="page", material_status="ready",
        artifact_bytes=b"<html>video page</html>", media_type="text/html", suffix=".html",
        modalities=("text",), label="video source page", content="[1] video page",
        locator="paragraph 1", acquired_by="SEARCH/open/http", discovered_by="W1-1")
    return candidate_id, page


@pytest.mark.asyncio
async def test_picture_operations_fall_back_to_low_res_and_sound_operations_do_not(edits, tmp_path):
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    candidate_id, page = _video_page(store)
    acquirer = _LowResOnly(edits["cut"])
    service = create_inspect_service(store=store, acquirer=acquirer, enabled_operations=MERGED_OPERATIONS)
    service.LOW_RES_RETRY_PAUSE_S = 0.0
    try:
        frames = await service.execute({"targets": [page.evidence_id], "request": {"operation": "get_frames",
                                                                                 "params": {"count": 2}}}, workspace_id="W1-1")
        assert frames.status == "ready" and acquirer.calls == [None, 360]
        (video,) = [row[2] for row in store.candidate_evidence(candidate_id, "video")]
        assert json.loads(video.content)["acquisition"]["video_only_height"] == 360
        speech = await service.execute({"targets": [page.evidence_id], "request": {"operation": "asr", "params": {}}},
                                       workspace_id="W1-1")
        # 转写不复用无声的低清文件，重新整片下载（这里仍被拒），也不走低清兜底
        assert speech.status == "failed" and acquirer.calls == [None, 360, None]
    finally:
        await service.aclose()


def test_recording_key_separates_low_res_downloads():
    from foren_searcher.search.recording import acquire_request_view

    url = "https://www.youtube.com/watch?v=abcdefghijk"
    assert acquire_request_view(url) == {"kind": "acquire", "source_url": acquire_request_view(url)["source_url"], "acquirer": 3}
    assert acquire_request_view(url, 360)["video_only_height"] == 360 and acquire_request_view(url, 360) != acquire_request_view(url)


@pytest.mark.asyncio
async def test_fetch_picture_downloads_low_res_without_registering_evidence(edits, tmp_path):
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    candidate_id, page = _video_page(store)
    acquirer = _LowResOnly(edits["cut"])
    service = create_inspect_service(store=store, acquirer=acquirer, enabled_operations=MERGED_OPERATIONS)
    try:
        path = await service.fetch_picture(candidate_id, page.evidence_id, tmp_path / "m" / "R1.media")
        assert Path(path).is_file() and acquirer.calls == [360]
        assert store.candidate_evidence(candidate_id, "video") == []
    finally:
        await service.aclose()


def test_describer_resolution_prefers_the_fixed_endpoint(monkeypatch):
    from foren_searcher.inspect.describe import EndpointDescriber

    run_model = {"model_provider": "openai_compatible", "endpoint": "http://127.0.0.1:11434/v1", "model": "qwen"}
    monkeypatch.delenv("FOREN_DESCRIBE_ENDPOINT", raising=False)
    assert EndpointDescriber.resolve(run_model).identity()["source"] == "run_model"
    monkeypatch.setenv("FOREN_DESCRIBE_ENDPOINT", "http://10.0.0.2:8000/v1")
    monkeypatch.setenv("FOREN_DESCRIBE_MODEL", "Qwen3-VL-8B")
    fixed = EndpointDescriber.resolve(run_model)
    assert fixed.identity() == {"model": "Qwen3-VL-8B", "endpoint": "http://10.0.0.2:8000/v1", "source": "env"}
    assert EndpointDescriber.resolve(None) is not None
    monkeypatch.delenv("FOREN_DESCRIBE_ENDPOINT")
    assert EndpointDescriber.resolve({"model_provider": "codex_cli"}) is None


def test_describer_posts_one_image_and_strips_reasoning():
    import httpx

    from foren_searcher.inspect.describe import EndpointDescriber

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think> B: backpack; A: box"}}]})

    describer = EndpointDescriber(endpoint="http://127.0.0.1:11434/v1", model="m", transport=httpx.MockTransport(handler))
    assert describer.describe(b"\xff\xd8jpeg", "what?") == "B: backpack; A: box"
    assert seen[0]["reasoning_effort"] == "none" and seen[0]["messages"][0]["content"][1]["type"] == "image_url"


def test_no_asyncio_leak():
    assert asyncio.get_event_loop_policy() is not None
