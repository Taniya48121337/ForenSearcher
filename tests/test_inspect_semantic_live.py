"""可选的语义 INSPECT 模型测试。

在本地已配置 VLM、CLIP、Grounding 或音频模型时，验证开放语义能力而非轻量后备统计。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


def _semantic_images() -> list[tuple[str, np.ndarray]]:
    """测试辅助函数 _semantic_images；为用例提供可重复的输入或可控行为。"""

    skimage = pytest.importorskip("skimage.data")
    return [
        ("astronaut", skimage.astronaut()),
        ("coffee", skimage.coffee()),
        ("cat", skimage.chelsea()),
        ("rocket", skimage.rocket()),
    ]


def _write_semantic_video(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    """测试辅助函数 _write_semantic_video；为用例提供可重复的输入或可控行为。"""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert writer.isOpened()
    for _, rgb in images:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = min(320 / bgr.shape[1], 240 / bgr.shape[0])
        resized = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        y = (240 - resized.shape[0]) // 2
        x = (320 - resized.shape[1]) // 2
        frame[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        for _ in range(10):
            writer.write(frame)
    writer.release()


@pytest.mark.live
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_clip_locates_diverse_real_photographs_by_text_and_image(tmp_path: Path, monkeypatch):
    """验证 clip locates diverse real photographs by text and image 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    model_name = os.environ.get("FOREN_CLIP_MODEL")
    if not model_name:
        pytest.skip("FOREN_CLIP_MODEL is required")
    images = _semantic_images()
    video_path = tmp_path / "semantic.avi"
    _write_semantic_video(video_path, images)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="four public sample photographs", original_filename=video_path.name, path=video_path)
    query_evidence = []
    for index, (name, rgb) in enumerate(images):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        query_evidence.append(store.commit_ingested_evidence(source_ref=f"I{index}", media_type="image/png", modalities=("image",), label=name, original_filename=f"{name}.png", data=encoded.tobytes()))
    service = create_inspect_service(store=store)
    try:
        prompts = ["an astronaut in a space suit", "a cup of coffee", "a cat", "a rocket launching into space"]
        for expected_index, prompt in enumerate(prompts):
            response = await service.execute({"targets": [video.evidence_id], "request": {"operation": "locate_frames", "params": {"query": {"type": "text", "text": prompt}, "top_k": 1}}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            assert int(response.result.candidates[0].time_s) == expected_index, (prompt, response.to_dict())
        for expected_index, query in enumerate(query_evidence):
            response = await service.execute({"targets": [video.evidence_id], "request": {"operation": "locate_frames", "params": {"query": {"type": "image", "evidence_id": query.evidence_id}, "top_k": 1}}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            assert int(response.result.candidates[0].time_s) == expected_index, response.to_dict()
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_qwen_visual_qa_answers_different_questions_from_real_photos(tmp_path: Path):
    """验证 qwen visual qa answers different questions from real photos 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    if not os.environ.get("FOREN_VLM_MODEL"):
        pytest.skip("FOREN_VLM_MODEL is required")
    images = _semantic_images()
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = []
    for index, (name, rgb) in enumerate(images[:3]):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        evidence.append(store.commit_ingested_evidence(source_ref=f"I{index}", media_type="image/png", modalities=("image",), label=name, original_filename=f"{name}.png", data=encoded.tobytes()))
    service = create_inspect_service(store=store)
    try:
        cases = [
            (evidence[0], "图中的人物是什么职业或身份？只用一个名词回答。", ("宇航员", "航天员")),
            (evidence[1], "杯中最可能是什么饮品？只用一个名词回答。", ("咖啡",)),
            (evidence[2], "图中的动物是什么？只用一个名词回答。", ("猫",)),
        ]
        answers = []
        for item, question, expected in cases:
            response = await service.execute({"targets": [item.evidence_id], "request": {"operation": "visual_qa", "params": {"question": question}}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            answers.append(response.result.answer)
            assert any(word in response.result.answer for word in expected), response.to_dict()
        assert len(set(answers)) == len(answers)
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_grounding_dino_locates_objects_in_diverse_real_photos(tmp_path: Path):
    """验证 grounding dino locates objects in diverse real photos 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    if not os.environ.get("FOREN_GROUNDING_MODEL"):
        pytest.skip("FOREN_GROUNDING_MODEL is required")
    images = _semantic_images()
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    service = create_inspect_service(store=store)
    try:
        cases = [(images[0], "person"), (images[1], "cup"), (images[2], "cat")]
        for index, ((name, rgb), query) in enumerate(cases):
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            assert ok
            item = store.commit_ingested_evidence(source_ref=f"I{index}", media_type="image/png", modalities=("image",), label=name, original_filename=f"{name}.png", data=encoded.tobytes())
            response = await service.execute({"targets": [item.evidence_id], "request": {"operation": "locate_regions", "params": {"query": query, "top_k": 3}}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            assert response.result.regions, (query, response.to_dict())
            assert response.result.regions[0].score >= 0.15
    finally:
        await service.aclose()


@pytest.mark.live
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_qwen_audio_answers_distinct_real_audio_questions(tmp_path: Path):
    """验证 qwen audio answers distinct real audio questions 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    if not os.environ.get("FOREN_AUDIO_MODEL"):
        pytest.skip("FOREN_AUDIO_MODEL is required")
    event_path = os.environ.get("FOREN_AUDIO_EVENT_SAMPLE")
    speech_path = os.environ.get("FOREN_ASR_SAMPLE")
    if not event_path or not speech_path:
        pytest.skip("FOREN_AUDIO_EVENT_SAMPLE and FOREN_ASR_SAMPLE are required")
    event_expected = tuple(
        token.strip().lower()
        for token in os.environ.get(
            "FOREN_AUDIO_EVENT_EXPECTED", "throat,clearing,cough"
        ).split(",")
        if token.strip()
    )
    assert event_expected
    event_question = os.environ.get(
        "FOREN_AUDIO_EVENT_QUESTION",
        "What sound event is heard? Answer in a few words.",
    )
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    event = store.commit_ingested_evidence(source_ref="A0", media_type="audio/wav", modalities=("audio",), label="public throat-clearing sample", original_filename=Path(event_path).name, path=event_path)
    speech = store.commit_ingested_evidence(source_ref="A1", media_type="audio/wav", modalities=("audio",), label="public JFK speech sample", original_filename=Path(speech_path).name, path=speech_path)
    service = create_inspect_service(store=store)
    try:
        cases = [
            (event, event_question, event_expected),
            (speech, "Is this mainly human speech, music, or an environmental sound? Answer with the category.", ("speech", "voice")),
        ]
        answers = []
        for item, question, expected in cases:
            response = await service.execute({"targets": [item.evidence_id], "request": {"operation": "audio_qa", "params": {"question": question}}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            answer = response.result.answer.lower()
            answers.append(answer)
            assert any(word in answer for word in expected), response.to_dict()
        assert answers[0] != answers[1]
    finally:
        await service.aclose()
