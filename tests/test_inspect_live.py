"""可选的本地模型/真实媒体 INSPECT 集成测试。

只有显式配置模型路径和运行开关时才执行；默认测试套件不会隐式下载权重。
"""

from __future__ import annotations

import os
from pathlib import Path
import wave

import cv2
import numpy as np
import pytest
from scipy import signal

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


@pytest.mark.live
@pytest.mark.asr
@pytest.mark.asyncio
async def test_public_speech_sample_with_local_whisper(tmp_path: Path):
    """验证 public speech sample with local whisper 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    sample_value, model_value = os.environ.get("FOREN_ASR_SAMPLE"), os.environ.get("FOREN_WHISPER_MODEL")
    if not sample_value or not model_value: pytest.skip("FOREN_ASR_SAMPLE and FOREN_WHISPER_MODEL are required")
    sample = Path(sample_value)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    audio = store.commit_ingested_evidence(source_ref="A0", media_type="audio/wav", modalities=("audio",), label="public speech sample", original_filename=sample.name, path=sample)
    service = create_inspect_service(store=store)
    try:
        response = await service.execute({"targets":[audio.evidence_id],"request":{"operation":"asr","params":{"language":"en"}}},workspace_id="W1-1")
        assert response.status == "ready" and response.result.segments
        transcript = " ".join(segment.text for segment in response.result.segments).lower()
        assert "country" in transcript and "ask" in transcript
    finally: await service.aclose()


@pytest.mark.live
@pytest.mark.asyncio
async def test_public_image_pdf_and_video_samples(tmp_path: Path):
    """验证 public image pdf and video samples 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    values = {name: os.environ.get(name) for name in ("FOREN_PUBLIC_IMAGE", "FOREN_PUBLIC_PDF", "FOREN_PUBLIC_VIDEO")}
    if not all(values.values()): pytest.skip("public image, PDF and video paths are required")
    store = GlobalStore(tmp_path / "media.sqlite3", tmp_path / "media-artifacts")
    image = store.commit_ingested_evidence(source_ref="I0", media_type="image/png", modalities=("image",), label="public image", original_filename="sample.png", path=values["FOREN_PUBLIC_IMAGE"])
    document = store.commit_ingested_evidence(source_ref="D0", media_type="application/pdf", modalities=("text",), label="public PDF", original_filename="sample.pdf", path=values["FOREN_PUBLIC_PDF"])
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="public video", original_filename="sample.avi", path=values["FOREN_PUBLIC_VIDEO"])
    service = create_inspect_service(store=store)
    try:
        regions = await service.execute({"targets":[image.evidence_id],"request":{"operation":"locate_regions","params":{"query":"红色糖果","top_k":5}}},workspace_id="W1-1")
        pages = await service.execute({"targets":[document.evidence_id],"request":{"operation":"get_pages","params":{}}},workspace_id="W1-1")
        frames = await service.execute({"targets":[video.evidence_id],"request":{"operation":"get_frames","params":{"count":8}}},workspace_id="W1-1")
        located = await service.execute({"targets":[video.evidence_id],"request":{"operation":"locate_frames","params":{"query":{"type":"text","text":"红色区域"},"top_k":3}}},workspace_id="W1-1")
        aligned = await service.execute({"targets":[video.evidence_id,video.evidence_id],"request":{"operation":"align_media","params":{"mode":"visual","top_k":1}}},workspace_id="W1-1")
        assert regions.status == "ready" and regions.result.regions
        assert pages.status == "ready" and pages.result.pages
        assert frames.status == "ready" and len(frames.result.frames) == 8
        assert located.status == "ready" and located.result.candidates, located.to_dict()
        assert aligned.status == "ready" and aligned.result.matches[0].score > .99
    finally: await service.aclose()


@pytest.mark.live
@pytest.mark.ocr
@pytest.mark.asyncio
async def test_rapidocr_official_chinese_english_numeric_sample(tmp_path: Path):
    """验证 rapidocr official chinese english numeric sample 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    pytest.importorskip("rapidocr_onnxruntime")
    sample_value = os.environ.get("FOREN_PUBLIC_OCR_IMAGE")
    if not sample_value:
        pytest.skip("FOREN_PUBLIC_OCR_IMAGE is required")
    sample = Path(sample_value)
    store = GlobalStore(tmp_path / "ocr.sqlite3", tmp_path / "ocr-artifacts")
    image = store.commit_ingested_evidence(
        source_ref="I0", media_type="image/jpeg", modalities=("image",),
        label="RapidOCR official Chinese sample", original_filename=sample.name, path=sample,
    )
    service = create_inspect_service(store=store)
    try:
        response = await service.execute(
            {"targets": [image.evidence_id], "request": {"operation": "ocr", "params": {}}},
            workspace_id="W1-1",
        )
        assert response.status == "ready", response.to_dict()
        assert len(response.result.results) == 1
        items = response.result.results[0].items
        combined = " ".join(item.text for item in items)
        assert len(items) >= 6
        assert "5.8" in combined or ("5" in combined and "8" in combined)
        assert any("发货" in item.text or "标准" in item.text for item in items)
        assert all(item.confidence is None or 0 <= item.confidence <= 1 for item in items)
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.ocr
@pytest.mark.asyncio
async def test_rapidocr_public_sample_survives_common_image_perturbations(tmp_path: Path):
    """验证 rapidocr public sample survives common image perturbations 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    pytest.importorskip("rapidocr_onnxruntime")
    sample_value = os.environ.get("FOREN_PUBLIC_OCR_IMAGE")
    if not sample_value:
        pytest.skip("FOREN_PUBLIC_OCR_IMAGE is required")
    original = cv2.imread(sample_value, cv2.IMREAD_COLOR)
    assert original is not None
    height, width = original.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), 3.0, 1.0)
    transformations = {
        "original": original,
        "jpeg35": cv2.imdecode(cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 35])[1], cv2.IMREAD_COLOR),
        "half_size": cv2.resize(original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        "rotate_3deg": cv2.warpAffine(original, rotation, (width, height), borderValue=(255, 255, 255)),
        "low_brightness": np.clip(original.astype(np.float32) * 0.55, 0, 255).astype(np.uint8),
        "gaussian_blur": cv2.GaussianBlur(original, (3, 3), 0),
        "sharpen": cv2.filter2D(original, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)),
    }
    store = GlobalStore(tmp_path / "ocr-robustness.sqlite3", tmp_path / "ocr-robustness-artifacts")
    evidence = []
    for index, (name, image) in enumerate(transformations.items()):
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        evidence.append((name, store.commit_ingested_evidence(source_ref=f"I{index}", media_type="image/png", modalities=("image",), label=name, original_filename=f"{name}.png", data=encoded.tobytes())))
    service = create_inspect_service(store=store)
    try:
        successes = []
        for name, item in evidence:
            response = await service.execute({"targets": [item.evidence_id], "request": {"operation": "ocr", "params": {}}}, workspace_id="W-ocr")
            assert response.status == "ready", response.to_dict()
            text = " ".join(entry.text for result in response.result.results for entry in result.items)
            successes.append((name, any(token in text for token in ("发货", "标准", "5.8"))))
        assert sum(success for _, success in successes) == len(successes), successes
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.asr
@pytest.mark.asyncio
async def test_whisper_public_speech_survives_rate_gain_and_noise_variants(tmp_path: Path):
    """验证 whisper public speech survives rate gain and noise variants 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    sample_value, model_value = os.environ.get("FOREN_ASR_SAMPLE"), os.environ.get("FOREN_WHISPER_MODEL")
    if not sample_value or not model_value:
        pytest.skip("FOREN_ASR_SAMPLE and FOREN_WHISPER_MODEL are required")
    with wave.open(sample_value, "rb") as source:
        rate = source.getframerate()
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    rng = np.random.default_rng(20260903)
    variants = {
        "original": (samples, rate),
        "8khz": (signal.resample_poly(samples, 8000, rate), 8000),
        "low_gain": (samples * 0.18, rate),
        "noise_18db": (samples + rng.normal(0, np.sqrt(np.mean(samples ** 2)) / (10 ** (18 / 20)), len(samples)), rate),
    }
    store = GlobalStore(tmp_path / "asr-robustness.sqlite3", tmp_path / "asr-robustness-artifacts")
    evidence = []
    for index, (name, (audio, sample_rate)) in enumerate(variants.items()):
        path = tmp_path / f"{name}.wav"
        with wave.open(str(path), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        evidence.append((name, store.commit_ingested_evidence(source_ref=f"A{index}", media_type="audio/wav", modalities=("audio",), label=name, original_filename=path.name, path=path)))
    service = create_inspect_service(store=store)
    try:
        outcomes = []
        for name, item in evidence:
            response = await service.execute({"targets": [item.evidence_id], "request": {"operation": "asr", "params": {"language": "en"}}}, workspace_id="W-asr")
            assert response.status == "ready", response.to_dict()
            transcript = " ".join(segment.text for segment in response.result.segments).lower()
            outcomes.append((name, "country" in transcript and "ask" in transcript))
        assert all(success for _, success in outcomes), outcomes
    finally:
        await service.aclose()
