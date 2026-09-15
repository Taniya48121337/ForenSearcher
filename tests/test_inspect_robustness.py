"""INSPECT 鲁棒性和算法数值测试。

覆盖取消/关闭、分段失败、24-bit WAV、PyAV 音频切片、FFT 对齐和 OCR 采样等难测分支。
"""

from __future__ import annotations

import io
import json
import math
import sys
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from foren_searcher.inspect.backends.base import BackendOutcome, MediaProfileData
from foren_searcher.inspect.backends.local import (
    LocalInspectBackend,
    _without_unverified_visual_text,
)
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore

from inspect_helpers import image_bytes, ingest, make_video


@pytest.mark.asyncio
async def test_mixed_visual_targets_identify_wrong_media_and_allow_local_frame_recovery(tmp_path):
    store = GlobalStore(tmp_path/'store.sqlite3', tmp_path/'artifacts')
    picture = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    path = tmp_path/'video.avi'
    make_video(path)
    video = store.commit_ingested_evidence(source_ref='V0', media_type='video/x-msvideo', modalities=('video',),
        label='video', original_filename=path.name, path=path)
    service = create_inspect_service(store=store, policy=InspectPolicy(maximum_download_bytes=1))
    try:
        response = await service.execute({'targets':[picture.evidence_id, video.evidence_id],
            'request':{'operation':'visual_qa','params':{'question':'描述图片'}}}, workspace_id='W1-1')
        assert response.status == 'failed'
        assert response.feedback[0].target_evidence_id == video.evidence_id
        assert 'get_frames' in response.feedback[0].message
        frames = await service.execute({'targets':[video.evidence_id],
            'request':{'operation':'get_frames','params':{'times_s':[1.0]}}}, workspace_id='W1-1')
        assert frames.status == 'ready'
        assert len(frames.result.frames) == 1
        assert store.healthcheck().status == 'ok'
    finally:
        await service.aclose()


def test_visual_qa_drops_unverified_image_text_claims() -> None:
    """visual_qa 不应把未经 OCR 核验的小字转写写入 Evidence。"""

    answer = (
        "图中人物穿着橙色宇航服。"
        "胸前徽章写着“COLLINS ASTRONAUT”。"
        "背景可见美国国旗和航天飞机。"
    )

    assert _without_unverified_visual_text(answer) == (
        "图中人物穿着橙色宇航服。背景可见美国国旗和航天飞机。"
    )


@pytest.mark.parametrize("answer,expected", [
    ("The full video context remains unknown.", "The full video context remains unknown."),
    ("胸牌written: CREW. The full video context remains unknown.", "The full video context remains unknown."),
    ("E8 shows a cat. E20 shows a person. The sign reads LAUNCH. These samples cannot establish the full video context.",
     "E8 shows a cat. E20 shows a person. These samples cannot establish the full video context."),
    ('A cat is visible. The sign reads "LAUNCH. CREW ONLY!". The full scene is unknown.',
     'A cat is visible. The full scene is unknown.'),
    ('The sign reads "LAUNCH. CREW ONLY! Unclosed transcription.', ''),
    ("E8 at 1.2 s: a cat.\nE20 at 2.3 s: a person.", "E8 at 1.2 s: a cat.\nE20 at 2.3 s: a person."),
    ("E8：猫。\n无法确认“全片没有删减”。\nE20：人物。", "E8：猫。\n\nE20：人物。"),
    ("猫旁边写着‘仅限船员。不得进入！’。视频范围未知。", "视频范围未知。"),
])
def test_visual_text_filter_preserves_observations_without_leaking_quoted_transcriptions(answer, expected):
    assert _without_unverified_visual_text(answer) == expected


@pytest.mark.parametrize("heading", ["E10:", "**E10**:", "**E10:**", "- E10:", "* **E10**:", "2. E10："])
def test_visual_text_filter_keeps_image_attribution_when_first_sentence_is_removed(heading):
    answer = 'E9: A person.\n\n' + heading + ' The label reads "CREW". Dark structures are visible.\n\nE11: A tower.'
    result = _without_unverified_visual_text(answer)
    assert result == 'E9: A person.\n\n' + heading + ' Dark structures are visible.\n\nE11: A tower.'
    assert 'CREW' not in result


def test_visual_text_filter_does_not_create_empty_observations_or_change_clean_format():
    assert _without_unverified_visual_text('**E10**: The label reads "CREW".') == ''
    clean = '- E1: A cat.\n- E2: A dog.\n\nThe context is unknown.'
    assert _without_unverified_visual_text(clean) == clean


@pytest.mark.parametrize("closing", ['".', ''])
def test_visual_text_filter_does_not_treat_quoted_evidence_headings_as_observations(closing):
    answer = 'E1: A person. The sign reads "CREW.\nE2: Private room.\nEntry prohibited.' + closing
    assert _without_unverified_visual_text(answer) == 'E1: A person.'


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("local_model", [False, True])
async def test_visual_observations_keep_each_frames_source_and_time(tmp_path: Path, monkeypatch, filtered, local_model):
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = ingest(store, b"video", source="V0", filename="input.avi", media_type="video/x-msvideo", modality="video")
    frames = [store.commit_derived_evidence(
        derived_from=(video.evidence_id,), modalities=("image",), label="frame",
        content=json.dumps({"time_s": time_s}), locator=json.dumps({"time_s": time_s}),
        acquired_by="INSPECT/get_frames", discovered_by="W1-1", production_method="derived_media",
        artifact_bytes=image_bytes(), media_type="image/png", suffix=".png", role="video_frame",
    ) for time_s in (3.4, 1.1)]

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[-1]['content']
            assert len([item for item in content if item['type']=='image']) == 2
            labels = [item['text'] for item in content if item['type']=='text']
            assert frames[0].evidence_id in labels[0] and '实际解码时间 3.4 秒' in labels[0]
            assert frames[1].evidence_id in labels[1] and '实际解码时间 1.1 秒' in labels[1]
            assert frames[0].evidence_id + '、' + frames[1].evidence_id + ' 的输入图像完全相同' in labels[-1]
            assert '不证明采样时刻之间没有变化' in labels[-1]
            return {"input_ids": np.zeros((1, 1), dtype=int)}

        def batch_decode(self, *args, **kwargs):
            return ["画面中有一个色块。" + ("胸牌写着 CREW。完整背景未知。" if filtered else "")]

    backend = LocalInspectBackend()
    model = SimpleNamespace(generate=lambda **kwargs: np.zeros((1, 2), dtype=int))
    model_name = str(tmp_path / 'private-user' / 'weights') if local_model else 'example/vision-model'
    monkeypatch.setattr(backend, "_configured_vlm", lambda: (model, Processor(), "cpu", model_name))
    monkeypatch.setenv("FOREN_VLM_MODEL", "test")
    service = create_inspect_service(store=store, backend=backend)
    try:
        response = await run(service, [frame.evidence_id for frame in frames], "visual_qa", {"question": "描述画面"})
        assert response.status == "ready", response.to_dict()
        answer = response.result.answer
        assert any("离散帧" in value and "未连续检查" in value for value in response.result.limitations)
        observations = answer.split("画面观察：")
        assert len(observations) == 2
        assert frames[0].evidence_id + '、' + frames[1].evidence_id + ' 的输入图像完全相同' in observations[0]
        assert '不证明原始文件细节一致' in observations[0]
        for frame, time_s in zip(frames, (3.4, 1.1), strict=True):
            line = next(line for line in answer.splitlines() if line.startswith(frame.evidence_id + " 来源定位："))
            provenance = json.loads(line.split(" 来源定位：", 1)[1])
            assert provenance["derived_from"] == [video.evidence_id]
            assert provenance["locator"]["time_s"] == time_s
        assert answer.index(frames[0].evidence_id + " 来源定位：") < answer.index(frames[1].evidence_id + " 来源定位：")
        assert json.loads(store.evidence(response.result_evidence_id).content)["answer"] == answer
        saved_limits = json.loads(store.evidence(response.result_evidence_id).content)["limitations"]
        saved_model = json.loads(store.evidence(response.result_evidence_id).content)["model"]
        assert saved_model == ('weights' if local_model else model_name)
        assert str(tmp_path) not in json.dumps(response.to_dict())
        assert any("相关句子未保留" in value for value in saved_limits) == filtered
        assert "CREW" not in answer
        if filtered:
            assert "完整背景未知" in answer
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("eos,tokenizer_eos,last_token,length,completed", [
    (9, 8, 1, 272, False),
    (9, 8, 9, 272, True),
    ([9, 8], 7, 8, 272, True),
    (None, 8, 8, 272, False),
    (None, None, 1, 272, False),
    (9, 8, 9, 2, True),
])
async def test_local_visual_generation_limit_does_not_publish_incomplete_evidence(
    tmp_path, monkeypatch, eos, tokenizer_eos, last_token, length, completed,
):
    store = GlobalStore(tmp_path / 'case.sqlite3', tmp_path / 'artifacts')
    picture = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=tokenizer_eos)

        def apply_chat_template(self, *args, **kwargs):
            return {'input_ids': np.zeros((1, 3), dtype=int)}

        def batch_decode(self, *args, **kwargs):
            assert completed, 'An incomplete answer must not become evidence.'
            return ['画面中有一个色块。']

    def generate(**kwargs):
        assert kwargs['max_new_tokens'] == 272
        tokens = np.ones((1, 3 + length), dtype=int)
        tokens[0, -1] = last_token
        return tokens

    backend = LocalInspectBackend()
    model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=eos))
    monkeypatch.setattr(backend, '_configured_vlm', lambda: (model, Processor(), 'cpu', 'test'))
    monkeypatch.setenv('FOREN_VLM_MODEL', 'test')
    async with create_inspect_service(store=store, backend=backend) as service:
        response = await run(service, [picture.evidence_id], 'visual_qa', {'question': '描述画面'})
        if completed:
            assert response.status == 'ready'
            assert response.result_evidence_id is not None
            assert '色块' in store.evidence(response.result_evidence_id).content
        else:
            assert response.status == 'failed'
            assert response.result_evidence_id is None
            assert response.feedback[0].code == 'RESOURCE_LIMIT'
            assert 'truncated answer was not stored' in response.feedback[0].message


async def run(service, evidence_ids, operation, params):
    """测试辅助函数 run；为用例提供可重复的输入或可控行为。"""

    return await service.execute(
        {"targets": list(evidence_ids), "request": {"operation": operation, "params": params}},
        workspace_id="W1-1",
    )


def signal_wav_bytes(*, delay_s: float = 0.0, seconds: float = 2.0, stereo_antiphase: bool = False, sample_width: int = 2, rate: int = 16_000) -> bytes:
    """测试辅助函数 signal_wav_bytes；为用例提供可重复的输入或可控行为。"""

    count = int(seconds * rate)
    time = np.arange(count, dtype=np.float64) / rate
    signal = np.zeros(count, dtype=np.float64)
    active = time >= delay_s
    local_time = time[active] - delay_s
    signal[active] = 0.45 * np.sin(2 * np.pi * (220 * local_time + 170 * local_time**2))
    signal[active] += 0.25 * np.sin(2 * np.pi * 619 * local_time)
    channels = np.column_stack((signal, -signal)) if stereo_antiphase else signal[:, None]
    if sample_width == 2:
        raw = np.rint(channels * 32767).astype("<i2").tobytes()
    elif sample_width == 3:
        values = np.rint(channels * 8_388_607).astype(np.int32).reshape(-1)
        unsigned = values.astype(np.uint32)
        packed = np.empty((len(values), 3), dtype=np.uint8)
        packed[:, 0] = unsigned & 0xFF
        packed[:, 1] = (unsigned >> 8) & 0xFF
        packed[:, 2] = (unsigned >> 16) & 0xFF
        raw = packed.tobytes()
    else:
        raise AssertionError("unsupported test sample width")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels.shape[1])
        audio.setsampwidth(sample_width)
        audio.setframerate(rate)
        audio.writeframes(raw)
    return output.getvalue()


def animated_gif_bytes() -> bytes:
    """测试辅助函数 animated_gif_bytes；为用例提供可重复的输入或可控行为。"""

    first = Image.new("RGB", (32, 24), "red")
    second = Image.new("RGB", (32, 24), "blue")
    output = io.BytesIO()
    first.save(output, "GIF", save_all=True, append_images=[second], duration=100, loop=0)
    return output.getvalue()


def make_video_with_audio(path: Path, *, seconds: float = 1.0, frequencies: tuple[float, ...] = (523.25,)) -> None:
    """测试辅助函数 make_video_with_audio；为用例提供可重复的输入或可控行为。"""

    import av

    fps, rate = 10, 16_000
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        audio_streams = [container.add_stream("pcm_s16le", rate=rate, layout="mono") for _ in frequencies]
        for index in range(int(seconds * fps)):
            image = np.zeros((96, 160, 3), dtype=np.uint8)
            image[:] = (20 + index * 5, 40, 70)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame): container.mux(packet)
        for packet in video.encode(): container.mux(packet)
        for audio, frequency in zip(audio_streams, frequencies):
            samples = np.rint(10_000 * np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)).astype(np.int16)[None, :]
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = rate
            frame.pts = 0
            frame.time_base = Fraction(1, rate)
            for packet in audio.encode(frame): container.mux(packet)
            for packet in audio.encode(): container.mux(packet)


class EmptyOcr:
    """测试辅助类 EmptyOcr；用可控状态替代真实外部依赖。"""

    def __call__(self, source):
        """测试辅助函数 __call__；为用例提供可重复的输入或可控行为。"""

        return None, None


class OneFrameFailsOcr:
    """测试辅助类 OneFrameFailsOcr；用可控状态替代真实外部依赖。"""

    def __init__(self):
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    def __call__(self, source):
        """测试辅助函数 __call__；为用例提供可重复的输入或可控行为。"""

        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated per-frame OCR failure")
        return None, None


@pytest.mark.asyncio
async def test_file_size_policy_rejects_before_processing(tmp_path: Path):
    """验证 file size policy rejects before processing 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    image = ingest(store, image_bytes(), source="I0", filename="input.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store, policy=InspectPolicy(maximum_file_bytes=32))
    try:
        response = await run(service, [image.evidence_id], "crop", {"box": [0, 0, 1, 1]})
        assert response.status == "failed"
        assert response.feedback[0].code == "RESOURCE_LIMIT"
        assert response.feedback[0].stage == "validate"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_video_ocr_frame_budget_returns_partial_with_unprocessed_range(tmp_path: Path, monkeypatch):
    """验证 video ocr frame budget returns partial with unprocessed range 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=lambda: EmptyOcr()))
    path = tmp_path / "input.avi"
    make_video(path, frames=30, fps=10.0)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="video", original_filename=path.name, path=path)
    service = create_inspect_service(
        store=store,
        policy=InspectPolicy(ocr_base_fps=2.0, maximum_frames=4),
    )
    try:
        response = await run(service, [video.evidence_id], "ocr", {})
        assert response.status == "partial", response.to_dict()
        assert response.feedback[0].code == "RESOURCE_LIMIT"
        assert response.result.coverage.sampling.processed_frames == 4
        assert response.result.coverage.failed_ranges
        assert len(response.result.results) == 4
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_video_ocr_keeps_successful_frames_when_one_recognition_fails(tmp_path: Path, monkeypatch):
    """验证 video ocr keeps successful frames when one recognition fails 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=OneFrameFailsOcr))
    path = tmp_path / "input.avi"
    make_video(path, frames=15, fps=10.0)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",), label="video", original_filename=path.name, path=path)
    service = create_inspect_service(store=store, policy=InspectPolicy(ocr_base_fps=2.0))
    try:
        response = await run(service, [video.evidence_id], "ocr", {})
        assert response.status == "partial", response.to_dict()
        assert any(item.code == "PROCESS_FAILED" for item in response.feedback)
        assert len(response.result.results) == 2
        assert response.result.coverage.sampling.processed_frames == 2
        assert response.result.coverage.failed_ranges
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_animated_image_is_not_silently_treated_as_complete_single_image(tmp_path: Path):
    """验证 animated image is not silently treated as complete single image 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    image = ingest(store, animated_gif_bytes(), source="I0", filename="animated.gif", media_type="image/gif", modality="image")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [image.evidence_id], "visual_qa", {"question": "主要颜色是什么？"})
        assert response.status == "failed"
        assert response.feedback[0].code == "UNSUPPORTED_MEDIA"
        assert response.feedback[0].stage == "probe"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_semantic_model_input_budgets_reject_oom_shaped_requests(tmp_path: Path):
    """验证 semantic model input budgets reject oom shaped requests 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    images = [ingest(store, image_bytes(), source=f"I{index}", filename=f"input-{index}.png", media_type="image/png", modality="image") for index in range(9)]
    audio = ingest(store, signal_wav_bytes(seconds=31.0), source="A0", filename="long.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store, policy=InspectPolicy(maximum_visual_qa_images=8, maximum_audio_qa_duration_s=30.0))
    try:
        visual = await run(service, [item.evidence_id for item in images], "visual_qa", {"question": "Compare the images."})
        sound = await run(service, [audio.evidence_id], "audio_qa", {"question": "Describe the audio."})
        assert visual.status == sound.status == "failed"
        assert visual.feedback[0].code == sound.feedback[0].code == "RESOURCE_LIMIT"
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_width", [2, 3])
async def test_audible_antiphase_stereo_is_not_misclassified_as_silence(tmp_path: Path, sample_width: int):
    """验证 audible antiphase stereo is not misclassified as silence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    audio = ingest(
        store,
        signal_wav_bytes(stereo_antiphase=True, sample_width=sample_width),
        source=f"A{sample_width}",
        filename=f"antiphase-{sample_width * 8}bit.wav",
        media_type="audio/wav",
        modality="audio",
    )
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [audio.evidence_id], "audio_qa", {"question": "是否有可听信号？"})
        assert response.status == "ready", response.to_dict()
        assert "存在可听信号" in response.result.answer
        assert "静音" not in response.result.answer
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_audio_alignment_maps_known_delay_without_leaving_coverage(tmp_path: Path):
    """验证 audio alignment maps known delay without leaving coverage 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    original = ingest(store, signal_wav_bytes(), source="A0", filename="original.wav", media_type="audio/wav", modality="audio")
    delayed = ingest(store, signal_wav_bytes(delay_s=0.35), source="A1", filename="delayed.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [original.evidence_id, delayed.evidence_id], "align_media", {"mode": "audio", "top_k": 1})
        assert response.status == "ready", response.to_dict()
        match = response.result.matches[0]
        assert match.range_b_s[0] - match.range_a_s[0] == pytest.approx(0.35, abs=0.025)
        assert 0 <= match.range_a_s[0] < match.range_a_s[1] <= 2.0
        assert 0 <= match.range_b_s[0] < match.range_b_s[1] <= 2.0
        assert match.score > 0.75
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_audio_alignment_adapts_different_sample_rates(tmp_path: Path):
    """验证 audio alignment adapts different sample rates 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    high_rate = ingest(store, signal_wav_bytes(rate=16_000), source="A0", filename="16k.wav", media_type="audio/wav", modality="audio")
    low_rate = ingest(store, signal_wav_bytes(rate=8_000), source="A1", filename="8k.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [high_rate.evidence_id, low_rate.evidence_id], "align_media", {"mode": "audio"})
        assert response.status == "ready", response.to_dict()
        assert response.result.matches and response.result.matches[0].score > .95
        assert response.result.matches[0].range_a_s == pytest.approx((0.0, 2.0), abs=.01)
        assert response.result.matches[0].range_b_s == pytest.approx((0.0, 2.0), abs=.01)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_silence_does_not_create_false_audio_alignment(tmp_path: Path):
    """验证 silence does not create false audio alignment 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    silence = signal_wav_bytes()
    silence = bytearray(silence)
    silence[44:] = b"\x00" * (len(silence) - 44)
    first = ingest(store, bytes(silence), source="A0", filename="silence-a.wav", media_type="audio/wav", modality="audio")
    second = ingest(store, bytes(silence), source="A1", filename="silence-b.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [first.evidence_id, second.evidence_id], "align_media", {"mode": "audio"})
        assert response.status == "ready"
        assert response.result.matches == ()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_empty_wav_is_rejected_instead_of_returning_zero_length_ready_range(tmp_path: Path):
    """验证 empty wav is rejected instead of returning zero length ready range 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16_000); audio.writeframes(b"")
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    empty = ingest(store, output.getvalue(), source="A0", filename="empty.wav", media_type="audio/wav", modality="audio")
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [empty.evidence_id], "audio_qa", {"question": "有声音吗？"})
        assert response.status == "failed"
        assert response.feedback[0].code == "DECODE_FAILED"
        assert response.feedback[0].stage == "probe"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_video_audio_track_is_probed_and_available_to_audio_qa(tmp_path: Path):
    """验证 video audio track is probed and available to audio qa 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    path = tmp_path / "with-audio.mkv"
    make_video_with_audio(path)
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(
        source_ref="V0", media_type="video/x-matroska", modalities=("video",),
        label="video with audio", original_filename=path.name, path=path,
    )
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [video.evidence_id], "audio_qa", {"question": "是否有持续音调？"})
        assert response.status == "ready", response.to_dict()
        assert "存在可听信号" in response.result.answer
        assert "523" in response.result.answer
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_second_video_audio_track_can_be_selected(tmp_path: Path):
    """验证 second video audio track can be selected 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    path = tmp_path / "two-audio-tracks.mkv"
    make_video_with_audio(path, frequencies=(440.0, 880.0))
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(
        source_ref="V0", media_type="video/x-matroska", modalities=("video",),
        label="video with two audio tracks", original_filename=path.name, path=path,
    )
    service = create_inspect_service(store=store)
    try:
        response = await run(service, [video.evidence_id], "audio_qa", {"question": "主频是多少？", "audio_track": 1})
        assert response.status == "ready", response.to_dict()
        assert response.result.audio_track == 1
        assert "880" in response.result.answer
    finally:
        await service.aclose()
