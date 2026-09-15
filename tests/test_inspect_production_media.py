"""INSPECT 面向生产媒体边界的测试。

重点验证损坏容器、异常时间基、资源上限、音轨与大型数据路径不会泄漏半成品或崩溃。
"""

from __future__ import annotations

from fractions import Fraction
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest
from PIL import Image

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.memory import MemoryPolicy, RecordAccess, workspace_visual_observations
from foren_searcher.store import GlobalStore

from inspect_helpers import ingest, make_video
from test_inspect_robustness import signal_wav_bytes


async def run(service, evidence_ids, operation, params):
    """测试辅助函数 run；为用例提供可重复的输入或可控行为。"""

    return await service.execute(
        {"targets": list(evidence_ids), "request": {"operation": operation, "params": params}},
        workspace_id="W1-1",
    )


@pytest.mark.asyncio
async def test_media_profile_scope_with_present_exif_is_shared_and_reused(tmp_path: Path):
    image = Image.new("RGB", (64, 48), "orange")
    exif = Image.Exif()
    exif[274] = 6
    exif[306] = "2024:02:03 04:05:06"
    exif[315] = "Public synthetic fixture"
    data = io.BytesIO()
    image.save(data, "JPEG", exif=exif)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    original = ingest(store, data.getvalue(), source="I0", filename="source.jpg",
                      media_type="image/jpeg", modality="image")
    profile_ids = []
    for workspace_id in ("W1-1", "W1-2"):
        async with create_inspect_service(store=store) as service:
            result = await service.execute(
                {"targets": [original.evidence_id], "request": {
                    "operation": "crop", "params": {"box": [0, 0, 1, 1]}}},
                workspace_id=workspace_id,
            )
        assert result.status == "ready", result.to_dict()
        profile_card, = [card for card in result.evidence if card.acquired_by == "INSPECT/media_profile"]
        profile_ids.append(profile_card.evidence_id)
        assert len(profile_card.to_dict()) == 6
        assert "未核验日期、地点或作者" in profile_card.label
        for role, actor in (("HI", "W1-2"), ("GIA", "GIA")):
            record = RecordAccess(store=store, role=role, actor_id=actor,
                                  round_index=1, policy=MemoryPolicy()).read(profile_card.evidence_id)
            assert record["label"] == profile_card.label
            profile = json.loads(record["content"])
            assert (profile["width"], profile["height"], profile["format"]) == (48, 64, "JPEG")
            assert exif[306] not in record["content"] and exif[315] not in record["content"]
    assert profile_ids[0] == profile_ids[1]
    with Image.open(store.artifact_path(original.artifact_ref)) as saved:
        assert saved.size == (64, 48)
        assert dict(saved.getexif()) == dict(exif)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [0, 10])
async def test_video_range_uses_exact_relative_pts_with_nonzero_origin(tmp_path: Path, origin):
    path = tmp_path / "offset.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 20, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = origin * 10 + index, Fraction(1, 10)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/mp4", modalities=("video",),
        label="offset", original_filename=path.name, path=path)
    async with create_inspect_service(store=store) as service:
        for start, end in ((.1, .2), (.2, .3), (.3, .4), (.6, .7)):
            result = await run(service, [video.evidence_id], "get_frames", {"range_s": [start, end], "count": 2})
            assert result.status == "ready", result.to_dict()
            assert [frame.time_s for frame in result.result.frames] == [start]
            assert result.result.frames[0].frame_duration_s == pytest.approx(.1)


@pytest.mark.asyncio
@pytest.mark.parametrize('fps', [5.0, 30.0])
async def test_get_frames_retains_declared_duration_at_last_frame(tmp_path, fps):
    path = tmp_path / 'timing.avi'
    make_video(path, frames=15, fps=fps)
    store = GlobalStore(tmp_path / 'case.sqlite3', tmp_path / 'artifacts')
    video = ingest(store, path.read_bytes(), source='V0', filename=path.name,
                   media_type='video/avi', modality='video')
    async with create_inspect_service(store=store) as service:
        result = await run(service, [video.evidence_id], 'get_frames',
                           {'times_s': [0, 14 / fps]})
    assert result.status == 'ready'
    first, last = result.result.frames
    assert last.time_s == pytest.approx(14 / fps)
    assert last.frame_duration_s == pytest.approx(1 / fps)
    assert last.time_s + last.frame_duration_s == pytest.approx(15 / fps)
    for extracted in (first, last):
        evidence = store.evidence(extracted.evidence_id)
        assert json.loads(evidence.locator)['frame_duration_s'] == pytest.approx(1 / fps)
    assert result.to_dict()['result']['frames'][-1]['frame_duration_s'] == pytest.approx(1 / fps)
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
        payload={'image_evidence_ids': [first.evidence_id, last.evidence_id]})
    _, summary = workspace_visual_observations(store, 'W1-1')
    coverage = summary['videos'][0]
    assert coverage['coverage_span_s'] == pytest.approx([0, 15 / fps])
    assert len(coverage['observed_ranges_s']) == 2
    assert coverage['unconfirmed_ranges_s'][0] == pytest.approx([1 / fps, 14 / fps])
    assert coverage['unknown_duration_frame_times_s'] == []


@pytest.mark.parametrize("origin", [0, 10])
@pytest.mark.parametrize("span", [(.1, .2), (.2, .65)])
@pytest.mark.parametrize("with_video", [False, True])
def test_compressed_audio_range_preserves_samples_with_nonzero_origin(tmp_path: Path, origin, span, with_video):
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    path = tmp_path / ("offset.mp4" if with_video else "offset.m4a")
    rate = 16_000
    signal = (np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 12_000).astype(np.int16)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("alac", rate=rate)
        stream.layout = "mono"
        if with_video:
            video_stream = container.add_stream("mpeg4", rate=10)
            video_stream.width, video_stream.height, video_stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(10):
                picture = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 20, dtype=np.uint8), format="rgb24")
                picture.pts, picture.time_base = origin * 10 + index, Fraction(1, 10)
                for packet in video_stream.encode(picture): container.mux(packet)
            for packet in video_stream.encode(): container.mux(packet)
        frame = av.AudioFrame.from_ndarray(signal[None, :], format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = rate, origin * rate, Fraction(1, rate)
        for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    samples, actual_rate, actual_span = LocalInspectBackend._av_audio_samples(path, span)
    assert actual_rate == rate and actual_span == span
    expected = signal[round(span[0] * rate):round(span[1] * rate)] / 32768.0
    np.testing.assert_array_equal(samples, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("span", [[.3, .4], [.1, .6]])
@pytest.mark.parametrize("operation", ["audio_qa", "asr", "align_media"])
async def test_compressed_audio_timestamp_gap_does_not_publish_silence(tmp_path: Path, monkeypatch, span, operation):
    monkeypatch.setenv("FOREN_WHISPER_MODEL", "unused-model")
    path = tmp_path / "gap.m4a"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("alac", rate=16000)
        stream.layout = "mono"
        for pts in (0, 8192):
            frame = av.AudioFrame.from_ndarray(np.full((1, 4096), 12000, dtype=np.int16), format="s16", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 16000, pts, Fraction(1, 16000)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    audio = store.commit_ingested_evidence(source_ref="A0", media_type="audio/mp4", modalities=("audio",),
        label="audio", original_filename=path.name, path=path)
    targets = [audio.evidence_id]
    params = {"range_s": span}
    if operation == "audio_qa": params["question"] = "Is there sound?"
    if operation == "align_media":
        targets *= 2
        params = {"mode": "audio", "range_a_s": span, "range_b_s": span}
    async with create_inspect_service(store=store) as service:
        result = await run(service, targets, operation, params)
    assert result.status == "failed", result.to_dict()
    assert result.result is None and result.result_evidence_id is None
    assert any(item.code == "DECODE_FAILED" for item in result.feedback)
    assert not [e for e in store.evidence_records() if e.acquired_by == f"INSPECT/{operation}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [16_000, 44_100])
@pytest.mark.parametrize("requested", [None, [.6, 1.1]])
async def test_asr_uses_video_time_for_delayed_audio_and_restores_segment_positions(tmp_path: Path, monkeypatch, rate, requested):
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    path = tmp_path / "delayed.mp4"
    signal = (np.sin(np.arange(rate) * 2 * np.pi * 440 / rate) * 12_000).astype(np.int16)
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("alac", rate=rate)
        audio.layout = "mono"
        for index in range(20):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = 100 + index, Fraction(1, 10)
            for packet in video.encode(frame): container.mux(packet)
        for packet in video.encode(): container.mux(packet)
        sound = av.AudioFrame.from_ndarray(signal[None, :], format="s16", layout="mono")
        sound.sample_rate, sound.pts, sound.time_base = rate, rate * 21 // 2, Fraction(1, rate)
        for packet in audio.encode(sound): container.mux(packet)
        for packet in audio.encode(): container.mux(packet)
    seen = []
    def transcribe(samples, **kwargs):
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32 and len(samples) == (8_000 if requested else 16_000)
        dominant_hz = np.fft.rfftfreq(len(samples), 1 / 16_000)[np.argmax(abs(np.fft.rfft(samples)))]
        assert dominant_hz == pytest.approx(440, abs=2)
        assert kwargs == {"language": "en", "vad_filter": True}
        seen.append(True)
        return iter([SimpleNamespace(start=.1, end=.3, text=" sample ")]), SimpleNamespace(language="en")
    monkeypatch.setenv("FOREN_WHISPER_MODEL", "test-whisper")
    monkeypatch.setenv("FOREN_WHISPER_DEVICE", "cpu")
    backend = LocalInspectBackend()
    backend._whisper = (SimpleNamespace(transcribe=transcribe), "cpu", "test-whisper")
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = store.commit_ingested_evidence(source_ref="V0", media_type="video/mp4", modalities=("video", "audio"),
        label="delayed audio", original_filename=path.name, path=path)
    try:
        async with create_inspect_service(store=store, backend=backend) as service:
            result = await run(service, [evidence.evidence_id], "asr", {"range_s": requested, "language": "en"})
        assert seen == [True] and result.status == "ready", result.to_dict()
        row = result.result.segments[0]
        origin = .6 if requested else .5
        assert row.start_s == pytest.approx(origin + .1) and row.end_s == pytest.approx(origin + .3)
        assert result.result.coverage.range_s == pytest.approx(requested or [.5, 1.5])
        assert row.text == "sample"
    finally:
        await backend.aclose()


def test_default_audio_range_clips_priming_and_uses_stream_endpoint(tmp_path: Path):
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    from foren_searcher.inspect.backends.base import BackendFailure
    path = tmp_path / "early-audio.mp4"
    rate = 16_000
    signal = (np.sin(np.arange(rate * 2) * 2 * np.pi * 440 / rate) * 12_000).astype(np.int16)
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("alac", rate=rate)
        audio.layout = "mono"
        for index in range(30):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = 100 + index, Fraction(1, 10)
            for packet in video.encode(frame): container.mux(packet)
        for packet in video.encode(): container.mux(packet)
        sound = av.AudioFrame.from_ndarray(signal[None, :], format="s16", layout="mono")
        sound.sample_rate, sound.pts, sound.time_base = rate, rate * 39 // 4, Fraction(1, rate)
        for packet in audio.encode(sound): container.mux(packet)
        for packet in audio.encode(): container.mux(packet)
    samples, actual_rate, span = LocalInspectBackend._av_audio_samples(path)
    assert actual_rate == rate and span == (0, 1.75)
    np.testing.assert_array_equal(samples, signal[rate // 4:] / 32768.0)
    with pytest.raises(BackendFailure, match="do not cover"):
        LocalInspectBackend._av_audio_samples(path, (0, 2))


@pytest.mark.asyncio
async def test_asr_selects_marked_default_audio_track_and_records_index(tmp_path: Path, monkeypatch):
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    path = tmp_path / "tracks.m4a"
    rate = 16_000
    with av.open(str(path), "w") as container:
        streams = [container.add_stream("alac", rate=rate) for _ in range(2)]
        for i, stream in enumerate(streams):
            stream.layout = "mono"
            stream.disposition = type(stream.disposition).default.value if i else 0
        for i, stream in enumerate(streams):
            values = (np.sin(np.arange(rate) * 2 * np.pi * (440 if i else 220) / rate) * 12_000).astype(np.int16)
            sound = av.AudioFrame.from_ndarray(values[None, :], format="s16", layout="mono")
            sound.sample_rate, sound.pts, sound.time_base = rate, 0, Fraction(1, rate)
            for packet in stream.encode(sound): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    def transcribe(samples, **kwargs):
        dominant = np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(abs(np.fft.rfft(samples)))]
        assert dominant == pytest.approx(440, abs=1)
        return iter([]), SimpleNamespace(language="en")
    monkeypatch.setenv("FOREN_WHISPER_MODEL", "test-whisper")
    monkeypatch.setenv("FOREN_WHISPER_DEVICE", "cpu")
    backend = LocalInspectBackend()
    backend._whisper = (SimpleNamespace(transcribe=transcribe), "cpu", "test-whisper")
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = store.commit_ingested_evidence(source_ref="A0", media_type="audio/mp4", modalities=("audio",),
        label="two tracks", original_filename=path.name, path=path)
    async with create_inspect_service(store=store, backend=backend) as service:
        result = await run(service, [evidence.evidence_id], "asr", {})
    assert result.status == "ready" and result.result.audio_track == 1, result.to_dict()
    first, _, _ = LocalInspectBackend._av_audio_samples(path, audio_track=0)
    frequency = np.fft.rfftfreq(len(first), 1 / rate)[np.argmax(abs(np.fft.rfft(first)))]
    assert frequency == pytest.approx(220, abs=1)


@pytest.mark.parametrize("requested", [None, (.4, .9)])
def test_vorbis_block_transitions_preserve_audio_timeline(tmp_path: Path, requested):
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    path = tmp_path / "transients.ogg"
    rate = 44_100
    values = (np.sin(np.arange(rate * 3) * 2 * np.pi * 440 / rate) * .4).astype(np.float32)
    values[::11025] = .9
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libvorbis", rate=rate)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(values[None, :], format="fltp", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = rate, 0, Fraction(1, rate)
        for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    samples, actual_rate, span = LocalInspectBackend._av_audio_samples(path, requested)
    assert actual_rate == 16000 and span == pytest.approx(requested or (0, 3), abs=1 / 16000)
    assert len(samples) == pytest.approx(round((span[1] - span[0]) * actual_rate), abs=1)
    expected = np.sin((np.arange(len(samples)) / actual_rate + span[0]) * 2 * np.pi * 440) * .4
    assert np.corrcoef(samples, expected)[0, 1] > .99


@pytest.mark.parametrize("suffix,codec", [(".m4a", "alac"), (".mkv", "flac"), (".ogg", "libvorbis")])
def test_audio_containers_are_identified_from_streams_despite_video_mime(tmp_path: Path, suffix, codec):
    from foren_searcher.evidence_resolver import sniff_media_kind
    path = tmp_path / ("audio" + suffix)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=16000)
        stream.layout = "mono"
        fmt = "fltp" if codec == "libvorbis" else "s16"
        values = np.zeros((1,16000), dtype=np.float32 if fmt == "fltp" else np.int16)
        frame = av.AudioFrame.from_ndarray(values, format=fmt, layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 16000, 0, Fraction(1,16000)
        for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    assert sniff_media_kind(path, "video/mp4") == "audio"


def make_vfr_video(path: Path) -> tuple[float, ...]:
    """测试辅助函数 make_vfr_video；为用例提供可重复的输入或可控行为。"""

    times = (0.0, 0.1, 0.2, 1.0, 1.1, 2.0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for value, time_s in enumerate(times):
            image = np.full((48, 64, 3), value * 35, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = round(time_s * 10)
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    return times


def make_histogram_collision_video(path: Path, *, checker: bool) -> None:
    """测试辅助函数 make_histogram_collision_video；为用例提供可重复的输入或可控行为。"""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 96))
    assert writer.isOpened()
    for _ in range(20):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        if checker:
            for y in range(0, 96, 16):
                for x in range(0, 160, 16):
                    if (x // 16 + y // 16) % 2 == 0: image[y:y+16, x:x+16] = 255
        else:
            image[:, 80:] = 255
        writer.write(image)
    writer.release()


def make_flash_title_video(path: Path) -> None:
    """测试辅助函数 make_flash_title_video；为用例提供可重复的输入或可控行为。"""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 96))
    assert writer.isOpened()
    for index in range(30):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        if index in {10, 11}:
            image[:] = 255
            cv2.putText(image, "FLASH", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        writer.write(image)
    writer.release()


def make_compressed_audio(path: Path, codec: str) -> None:
    """测试辅助函数 make_compressed_audio；为用例提供可重复的输入或可控行为。"""

    rate = 16_000
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate, layout="mono")
        samples = np.rint(10_000 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.int16)[None, :]
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = rate
        frame.pts = 0
        frame.time_base = Fraction(1, rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class BrightFrameOcr:
    """测试辅助类 BrightFrameOcr；用可控状态替代真实外部依赖。"""

    def __call__(self, source):
        """测试辅助函数 __call__；为用例提供可重复的输入或可控行为。"""

        if isinstance(source, Path):
            with Image.open(source) as image: value = float(np.asarray(image.convert("L")).mean())
        else:
            source.seek(0)
            with Image.open(source) as image: value = float(np.asarray(image.convert("L")).mean())
            source.seek(0)
        if value < 80:
            return None, None
        return [([[20, 20], [140, 20], [140, 70], [20, 70]], "FLASH", 0.99)], None


@pytest.mark.asyncio
async def test_media_signature_overrides_wrong_avi_mime(tmp_path: Path):
    """验证 media signature overrides wrong avi mime 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    path = tmp_path / "actually-video.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(
        source_ref="V0", media_type="image/png", modalities=("video",),
        label="mislabeled AVI", original_filename="claimed.png", path=path,
    )
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [video.evidence_id], "get_frames", {"count": 2})
        assert response.status == "ready", response.to_dict()
        assert len(response.result.frames) == 2
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('extension', ['mkv', 'mp4'])
async def test_vfr_get_frames_returns_nearest_real_pts(tmp_path: Path, extension):
    """验证 vfr get frames returns nearest real pts 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    path = tmp_path / f"vfr.{extension}"
    real_times = make_vfr_video(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(
        source_ref="V0", media_type="video/x-matroska", modalities=("video",),
        label="variable frame rate", original_filename=path.name, path=path,
    )
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [video.evidence_id], "get_frames", {"times_s": [0.15, .2, 1.05, 1.1, 1.95]})
        assert response.status == "ready", response.to_dict()
        returned = tuple(frame.time_s for frame in response.result.frames)
        assert len(returned) == 5
        assert all(value in real_times for value in returned)
        assert returned == pytest.approx((0.1, .2, 1.0, 1.1, 2.0), abs=1e-6)
        durations = [.1, .8, .1, .9, .1] if extension == 'mp4' else [.1] * 5
        assert [frame.frame_duration_s for frame in response.result.frames] == pytest.approx(durations)
        _, summary = workspace_visual_observations(store, 'W1-1', [f.evidence_id for f in response.result.frames])
        coverage = summary['videos'][0]
        expected = [[0, .1]] if extension == 'mp4' else [[0, .1], [.3, 1.0], [1.2, 2.0]]
        assert len(coverage['unconfirmed_ranges_s']) == len(expected)
        for actual, gap in zip(coverage['unconfirmed_ranges_s'], expected):
            assert actual == pytest.approx(gap)
        assert coverage['coverage_span_s'] == pytest.approx([0, 2.1])
        bounded = await run(service, [video.evidence_id], "get_frames", {"range_s": [.3, 1.15], "count": 8})
        assert bounded.status == "ready", bounded.to_dict()
        assert [frame.time_s for frame in bounded.result.frames] == pytest.approx([1.0, 1.1])
        empty = await run(service, [video.evidence_id], "get_frames", {"range_s": [.3, .9], "count": 2})
        assert empty.status == "failed" and empty.result is None
        assert all(item.code == "DECODE_FAILED" for item in empty.feedback)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_visual_alignment_uses_spatial_content_and_unique_artifacts(tmp_path: Path):
    """验证 visual alignment uses spatial content and unique artifacts 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    split_path, checker_path = tmp_path / "split.avi", tmp_path / "checker.avi"
    make_histogram_collision_video(split_path, checker=False)
    make_histogram_collision_video(checker_path, checker=True)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    split = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="split", original_filename=split_path.name, path=split_path)
    checker = store.commit_ingested_evidence(source_ref="V1", media_type="video/x-msvideo", modalities=("video",), label="checker", original_filename=checker_path.name, path=checker_path)
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [split.evidence_id, checker.evidence_id], "align_media", {"mode": "visual", "top_k": 3})
        assert response.status == "ready", response.to_dict()
        assert not response.result.matches
        response = await run(service, [split.evidence_id, split.evidence_id], "align_media", {"mode": "visual", "top_k": 3})
        assert response.result.matches
        anchor_ids = [
            evidence_id
            for match in response.result.matches
            for anchor in match.anchors
            for evidence_id in (anchor.image_a_evidence_id, anchor.image_b_evidence_id)
            if evidence_id is not None
        ]
        assert len(set(anchor_ids)) == 2
        assert all(a.image_a_evidence_id == a.image_b_evidence_id for m in response.result.matches for a in m.anchors)
        images = [e.evidence_id for e in response.evidence if "image" in e.modalities]
        assert len(images) == len(set(images)) == 2
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_video_ocr_adds_short_scene_and_one_round_of_refinement(tmp_path: Path, monkeypatch):
    """验证 video ocr adds short scene and one round of refinement 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda: BrightFrameOcr()))
    path = tmp_path / "flash.avi"
    make_flash_title_video(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="flash title", original_filename=path.name, path=path)
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [video.evidence_id], "ocr", {})
        assert response.status == "ready", response.to_dict()
        sampling = response.result.coverage.sampling
        assert sampling.base_frames == 6
        assert sampling.scene_frames >= 1
        assert sampling.refinement_frames >= 2
        assert sampling.processed_frames == len(response.result.results)
        assert any(item.text == "FLASH" for result in response.result.results for item in result.items)
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("suffix", "codec"), [(".mp3", "mp3"), (".flac", "flac"), (".ogg", "libvorbis")])
async def test_compressed_audio_formats_are_probed_and_processed(suffix: str, codec: str, tmp_path: Path):
    """验证 compressed audio formats are probed and processed 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    path = tmp_path / f"tone{suffix}"
    make_compressed_audio(path, codec)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    audio = store.commit_ingested_evidence(source_ref="A0", media_type="application/octet-stream", modalities=("audio",), label=f"compressed tone {suffix}", original_filename=path.name, path=path)
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [audio.evidence_id], "audio_qa", {"question": "Is there an audible signal?"})
        assert response.status == "ready", response.to_dict()
        assert "可听信号" in response.result.answer
        assert response.result.range_s[1] >= 0.9
        assert store.inspect_execution(1)["details"]["backend"] == "PyAV/numpy"
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rate_a", "rate_b", "sample_width", "stereo", "delay_s"),
    [
        (8_000, 12_000, 2, False, 0.121691447),
        (11_025, 12_000, 3, False, 0.199731281),
        (8_000, 16_000, 3, True, 0.444783119),
        (16_000, 24_000, 2, True, 0.516661171),
        (22_050, 8_000, 3, True, 0.124637293),
        (11_025, 16_000, 2, False, 0.539842761),
    ],
)
async def test_audio_alignment_diverse_formats(rate_a, rate_b, sample_width, stereo, delay_s, tmp_path: Path):
    """验证 audio alignment diverse formats 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    first = ingest(store, signal_wav_bytes(rate=rate_a, sample_width=sample_width, stereo_antiphase=stereo), source="A0", filename="a.wav", media_type="audio/wav", modality="audio")
    second = ingest(store, signal_wav_bytes(rate=rate_b, sample_width=sample_width, stereo_antiphase=stereo, delay_s=delay_s), source="A1", filename="b.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [first.evidence_id, second.evidence_id], "align_media", {"mode": "audio"})
        assert response.status == "ready", response.to_dict()
        match = response.result.matches[0]
        estimated = match.range_b_s[0] - match.range_a_s[0]
        assert estimated == pytest.approx(delay_s, abs=0.03)
        assert match.score >= 0.70
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_audio_alignment_randomized_24_case_matrix(tmp_path: Path):
    """验证 audio alignment randomized 24 case matrix 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    rng = np.random.default_rng(20260903)
    rates = (8_000, 11_025, 12_000, 16_000, 22_050, 24_000)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    service = create_inspect_service(store=store)
    outcomes = []
    try:
        for index in range(24):
            rate_a, rate_b = int(rng.choice(rates)), int(rng.choice(rates))
            sample_width = int(rng.choice((2, 3)))
            stereo = bool(rng.integers(0, 2))
            delay_s = float(rng.uniform(0.08, 0.58))
            first = ingest(store, signal_wav_bytes(rate=rate_a, sample_width=sample_width, stereo_antiphase=stereo), source=f"A{index * 2}", filename=f"a-{index}.wav", media_type="audio/wav", modality="audio")
            second = ingest(store, signal_wav_bytes(rate=rate_b, sample_width=sample_width, stereo_antiphase=stereo, delay_s=delay_s), source=f"A{index * 2 + 1}", filename=f"b-{index}.wav", media_type="audio/wav", modality="audio")
            response = await run(service, [first.evidence_id, second.evidence_id], "align_media", {"mode": "audio"})
            assert response.status == "ready", response.to_dict()
            match = response.result.matches[0]
            estimated = match.range_b_s[0] - match.range_a_s[0]
            outcomes.append((index, abs(estimated - delay_s) <= 0.03 and match.score >= 0.70, estimated, delay_s, match.score))
        assert all(success for _, success, *_ in outcomes), outcomes
    finally:
        await service.aclose()
