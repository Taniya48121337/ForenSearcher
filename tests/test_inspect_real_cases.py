"""使用程序生成的真实图片、视频、音频和 PDF 跑十种 INSPECT 操作。

这些是比纯 mock 更接近用户数据的端到端案例，验证后端确实能解码并产出 E。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore

from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes, text_image_bytes, wav_bytes


async def run(service, evidence_ids, operation, params):
    """测试辅助函数 run；为用例提供可重复的输入或可控行为。"""

    return await service.execute({"targets": list(evidence_ids), "request": {"operation": operation, "params": params}}, workspace_id="W1-1")


@pytest.mark.asyncio
async def test_real_image_crop_region_location_and_multi_image_qa(tmp_path: Path):
    """验证 real image crop region location and multi image qa 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    first = ingest(store, image_bytes(), source="I0", filename="scene.png", media_type="image/png", modality="image")
    second = ingest(store, image_bytes(color="navy", red_box=False), source="I1", filename="night.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store)
    try:
        crop = await run(service, [first.evidence_id], "crop", {"box": [0.25, 0.2, 0.75, 0.8]})
        assert crop.status == "ready" and crop.result.width == 160 and crop.result.height == 108
        located = await run(service, [first.evidence_id], "locate_regions", {"query": "红色区域", "top_k": 3})
        assert located.status == "ready" and located.result.regions
        assert located.result.regions[0].box_px[0] <= 80 and located.result.regions[0].box_px[2] >= 240
        compared = await run(service, [first.evidence_id, second.evidence_id], "visual_qa", {"question": "两图整体颜色是否有明显差异？"})
        assert compared.status == "ready" and compared.result.image_evidence_ids == (first.evidence_id, second.evidence_id)
        assert "明显差异" in compared.result.answer
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_real_pdf_specific_uniform_and_out_of_range_pages(tmp_path: Path):
    """验证 real pdf specific uniform and out of range pages 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    document = ingest(store, pdf_bytes(), source="D0", filename="mixed.pdf", media_type="application/pdf", modality="text")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [document.evidence_id], "get_pages", {"page_range": [1,2], "count": 2})
        assert response.status == "ready"
        assert [x.page_number for x in response.result.pages] == [1,2]
        assert (response.result.pages[0].width, response.result.pages[0].height) != (response.result.pages[1].width, response.result.pages[1].height)
        invalid = await run(service, [document.evidence_id], "get_pages", {"pages": [1,3]})
        assert invalid.status == "partial" and invalid.feedback[0].code == "OUT_OF_RANGE"
        assert [page.page_number for page in invalid.result.pages] == [1]
        assert invalid.feedback[0].locator == {"pages": [3], "page_count": 2}
        reduced = await run(service, [document.evidence_id], "get_pages", {"count": 8})
        assert reduced.status == "ready" and len(reduced.result.pages) == 2
        assert reduced.feedback[0].code == "COUNT_REDUCED" and reduced.feedback[0].severity == "warning"
    finally: await service.aclose()


@pytest.mark.asyncio
async def test_real_video_frames_text_and_image_location_and_visual_alignment(tmp_path: Path):
    """验证 real video frames text and image location and visual alignment 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    first_path, shifted_path = tmp_path / "first.avi", tmp_path / "shifted.avi"
    make_video(first_path); make_video(shifted_path, shifted=True)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    first = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="first", original_filename=first_path.name, path=first_path)
    shifted = store.commit_ingested_evidence(source_ref="V1", media_type="video/x-msvideo", modalities=("video",), label="shifted", original_filename=shifted_path.name, path=shifted_path)
    service = create_inspect_service(store=store)
    try:
        frames = await run(service, [first.evidence_id], "get_frames", {"times_s": [0.1, 1.7, 2.8]})
        assert frames.status == "ready" and len(frames.result.frames) == 3
        assert frames.result.frames[0].time_s == pytest.approx(0.1, abs=.11)
        query_frame = frames.result.frames[1].evidence_id
        by_text = await run(service, [first.evidence_id], "locate_frames", {"query": {"type":"text", "text":"红色矩形"}, "top_k": 3})
        assert by_text.status == "ready" and by_text.result.candidates[0].time_s >= 1.4
        by_image = await run(service, [shifted.evidence_id], "locate_frames", {"query": {"type":"image", "evidence_id":query_frame}, "top_k": 2})
        assert by_image.status == "ready" and by_image.result.candidates
        aligned = await run(service, [first.evidence_id, shifted.evidence_id], "align_media", {"mode":"visual", "top_k":2})
        assert aligned.status == "ready" and aligned.result.matches
        assert aligned.result.matches[0].anchors[0].image_a_evidence_id
        assert aligned.result.matches[0].anchors[0].image_b_evidence_id
        reduced = await run(service, [first.evidence_id], "get_frames", {"times_s": [0.01, 0.02, 0.03]})
        assert reduced.status == "ready" and len(reduced.result.frames) == 1
        assert reduced.feedback[0].code == "COUNT_REDUCED"
    finally: await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interval,expected", [
    ((.11, .21), [.2]),
    ((.19, .29), [.2]),
    ((.1, .2), [.1]),
    ((.2, .3), [.2]),
    ((0., .1), [0.]),
    ((.11, .2), []),
    ((.11, .19), []),
    ((2.91, 2.99), []),
])
async def test_uniform_frames_stay_inside_requested_interval(tmp_path: Path, interval, expected):
    path = tmp_path / "frames.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo",
        modalities=("video",), label="frames", original_filename=path.name, path=path)
    async with create_inspect_service(store=store) as service:
        # 先保存区间外的邻近帧，验证复用不会绕过区间选择。
        await run(service, [video.evidence_id], "get_frames", {"times_s": [.1, .2, .3, 2.9]})
        result = await run(service, [video.evidence_id], "get_frames", {"range_s": list(interval), "count": 2})
        if not expected:
            assert result.status == "failed"
            assert result.result is None
            assert all(item.code == "DECODE_FAILED" for item in result.feedback)
        else:
            assert result.status == "ready"
            assert [frame.time_s for frame in result.result.frames] == pytest.approx(expected)
            for frame in result.result.frames:
                assert interval[0] <= frame.time_s < interval[1]
                saved = store.evidence(frame.evidence_id)
                assert saved.derived_from == (video.evidence_id,)
                assert store.artifact_path(saved.artifact_ref).is_file()
            if len(expected) < 2:
                assert any(item.code == "COUNT_REDUCED" for item in result.feedback)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["locate_frames", "align_media", "ocr"])
@pytest.mark.parametrize("interval", [(.11, .69), (.11, .6), (.11, .19), (.11, .2)])
async def test_video_analysis_only_uses_frames_within_range(tmp_path: Path, monkeypatch, operation, interval):
    path = tmp_path / "frames.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo",
        modalities=("video",), label="frames", original_filename=path.name, path=path)
    targets = [video.evidence_id]
    params = {"range_s": list(interval)}
    if operation == "align_media":
        other = store.commit_ingested_evidence(source_ref="V1", media_type="video/x-msvideo",
            modalities=("video",), label="other", original_filename=path.name, path=path)
        targets.append(other.evidence_id)
        params = {"mode": "visual", "range_a_s": list(interval), "range_b_s": list(interval)}
    elif operation == "locate_frames":
        params["query"] = {"type": "text", "text": "red"}
        if interval in ((.11, .19), (.11, .2)):
            # 没有区间内的帧时，不应尝试加载或执行语义模型。
            monkeypatch.setenv("FOREN_CLIP_MODEL", "unavailable-test-model")
        else:
            monkeypatch.delenv("FOREN_CLIP_MODEL", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime",
            SimpleNamespace(RapidOCR=lambda: lambda _image: ([], None)))
    async with create_inspect_service(store=store, policy=InspectPolicy(ocr_base_fps=20)) as service:
        result = await run(service, targets, operation, params)
    if interval in ((.11, .19), (.11, .2)):
        assert result.status == "failed", result.to_dict()
        assert result.result is None
        assert all(item.code == "DECODE_FAILED" for item in result.feedback)
        assert not [e for e in store.evidence_records() if e.acquired_by == "INSPECT/" + operation]
    else:
        assert result.status == "ready", result.to_dict()
        if operation == "locate_frames":
            times = [frame.time_s for frame in result.result.candidates]
        elif operation == "align_media":
            times = [t for match in result.result.matches for anchor in match.anchors
                     for t in (anchor.time_a_s, anchor.time_b_s)]
        else:
            times = [frame.time_s for frame in result.result.results]
        assert times
        assert all(interval[0] <= t < interval[1] for t in times)
        for item in store.evidence_records():
            if item.artifact_ref and item.acquired_by == 'INSPECT/' + operation:
                assert interval[0] <= json.loads(item.locator)['time_s'] < interval[1]


@pytest.mark.asyncio
async def test_real_audio_tone_silence_and_alignment(tmp_path: Path):
    """验证 real audio tone silence and alignment 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    tone = ingest(store, wav_bytes(frequency=523.25), source="A0", filename="tone.wav", media_type="audio/wav", modality="audio")
    silence = ingest(store, wav_bytes(silence=True), source="A1", filename="silence.wav", media_type="audio/wav", modality="audio")
    tone2 = ingest(store, wav_bytes(frequency=523.25), source="A2", filename="tone2.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        audible = await run(service, [tone.evidence_id], "audio_qa", {"question":"是否有持续音调？"})
        quiet = await run(service, [silence.evidence_id], "audio_qa", {"question":"是否静音？"})
        assert audible.status == quiet.status == "ready"
        assert "523" in audible.result.answer and "静音" in quiet.result.answer
        aligned = await run(service, [tone.evidence_id, tone2.evidence_id], "align_media", {"mode":"audio", "top_k":1})
        assert aligned.status == "ready" and aligned.result.matches[0].score > .9
    finally: await service.aclose()


@pytest.mark.asyncio
async def test_real_corrupt_input_empty_backend_and_static_failure_semantics(tmp_path: Path):
    """验证 real corrupt input empty backend and static failure semantics 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    corrupt = ingest(store, b"not an image", source="I0", filename="bad.png", media_type="image/png", modality="image")
    clean = ingest(store, image_bytes(red_box=False), source="I1", filename="clean.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store)
    try:
        broken = await run(service, [corrupt.evidence_id], "crop", {"box":[0,0,1,1]})
        assert broken.status == "failed" and broken.feedback[0].code == "DECODE_FAILED"
        empty = await run(service, [clean.evidence_id], "locate_regions", {"query":"红色对象"})
        assert empty.status == "ready" and empty.result.regions == () and empty.result_evidence_id
        invalid = await service.execute({"targets":["E01"],"request":{"operation":"ocr","params":{}}}, workspace_id="W1-1")
        assert invalid.status == "failed" and invalid.operation is None and invalid.evidence == ()
    finally: await service.aclose()


@pytest.mark.asyncio
async def test_real_rapidocr_text_and_blank_image_are_distinct_from_failure(tmp_path: Path):
    """验证 real rapidocr text and blank image are distinct from failure 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    pytest.importorskip("rapidocr_onnxruntime")
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    text = ingest(store, text_image_bytes(), source="I0", filename="headline.png", media_type="image/png", modality="image")
    blank = ingest(store, image_bytes(red_box=False), source="I1", filename="blank.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store)
    try:
        recognized = await run(service, [text.evidence_id], "ocr", {})
        empty = await run(service, [blank.evidence_id], "ocr", {})
        assert recognized.status == empty.status == "ready"
        combined = " ".join(item.text for item in recognized.result.results[0].items).upper()
        assert "FOREN" in combined and "2026" in combined
        assert empty.result.results[0].items == ()
        assert recognized.result_evidence_id and empty.result_evidence_id
    finally: await service.aclose()


@pytest.mark.asyncio
async def test_unconfigured_heavy_asr_is_an_explicit_failure_not_empty_result(tmp_path: Path, monkeypatch):
    """验证 unconfigured heavy asr is an explicit failure not empty result 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    monkeypatch.delenv("FOREN_WHISPER_MODEL", raising=False)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    audio = ingest(store, wav_bytes(), source="A0", filename="speechless-tone.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [audio.evidence_id], "asr", {})
        assert response.status == "failed" and response.feedback[-1].code == "PROCESS_FAILED"
        assert response.result is None and response.result_evidence_id is None
        assert any(card.acquired_by == "INSPECT/media_profile" for card in response.evidence)
    finally: await service.aclose()
