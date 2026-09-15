"""默认 INSPECT 使用的 CPU/本地后端。

基础媒体操作依赖 Pillow、PyAV、OpenCV 和 PyMuPDF；OCR、ASR、VLM、CLIP、Grounding
等模型按环境变量启用并只从本地缓存加载。本层只产生 ``BackendOutcome`` 和内存中的
``PendingArtifact``，真正的 E ID 分配与持久化由 Service/Store 负责。
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import math
import os
import re
import threading
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import signal

from ..contracts import (
    AlignMediaParams, AsrParams, AudioQaParams, CropParams, GetFramesParams,
    GetPagesParams, ImageFrameQuery, LocateFramesParams, LocateRegionsParams,
    OcrParams, TextFrameQuery, TextImageFrameQuery, VisualQaParams, _OCR_COVERAGE_NOTE,
    _FRAME_RETRIEVAL_NOTE,
)
from ..policy import InspectPolicy
from ..temporal import TemporalMatch, align_sequences, path_score, prune_dominated_paths
from ..audio_alignment import spectral_matches
from ..visual_alignment import AlignmentFrame, sample_alignment_frames, geometric_similarities
from .base import BackendFailure, BackendOutcome, MediaProfileData, PendingArtifact, PreparedMedia


_ALIGNMENT_COVERAGE_NOTE = (
    "coverage 是各输入的处理范围和采样量，failed_ranges 仅指处理失败的区间；"
    "它们不表示内容匹配或删减。对应关系见 matches，score 只表示本次调用当前模式内的候选排序，而非结论正确的概率。"
    "matches 列出候选对应关系，不是剪辑事件清单或两侧视频的唯一分段。"
    "候选可以重叠，不要求全部同时成立；候选数量或重叠本身不能证明发生了删减、重复插入或拼接。"
    "range_a_s/range_b_s 是包含采样边界误差的区间，端点之差不是精确时间偏移。"
    "每对锚点的 time_b_minus_a_s 是 time_b_s 减去 time_a_s；a/b 对应本次 targets 的先后顺序。"
)

_VISUAL_TEXT_OMISSION_NOTE = (
    "部分视觉回答含未经 OCR 核验的文字描述，相关句子未保留；"
    "需核实文字、数字或书面日期时，对相应图片调用 ocr。"
    "其余回答可能不完整，不能据此推断未提及的场景或差异不存在。"
)

_VISUAL_SOURCE_NOTE = (
    "每张图的描述用E编号标识；来源和时间已由工具附在回答前，不必在每段描述中重复抄写。"
    "需要回答时间或顺序问题时，只使用图注中的实际时间，并分别保持各来源的时间轴。"
    "不能把不同视频的相同时间戳视为内容对应关系。"
)


def _png(image: np.ndarray | Image.Image) -> bytes:
    """把 OpenCV BGR 数组或 Pillow 图片编码成 PNG 字节。"""

    pil = image if isinstance(image, Image.Image) else Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output = io.BytesIO()
    pil.save(output, format="PNG")
    return output.getvalue()


def _jsonable_box(box: tuple[float, ...]) -> list[float]:
    """把数值元组转成 JSON 可直接保存的 float 列表。"""

    return [float(x) for x in box]


def _resolve_local_model(model_name: str) -> str:
    """接受明确本地目录，或解析已经存在于 Hugging Face 缓存的模型。"""

    path = Path(model_name)
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, local_files_only=True)


def _model_reference(model_name: str) -> str:
    """分析元数据保留模型名称，不携带本机父目录。"""

    path = Path(model_name)
    return path.name if path.is_absolute() or path.is_dir() else model_name


_VISUAL_QUOTED_TEXT = (
    r'"(?:\\.|[^"\\])*(?:"|$)|“[^”]*(?:”|$)|‘[^’]*(?:’|$)|'
    r"(?<!\w)'[^'\n]*(?:'|$)"
)


def _without_unverified_visual_text(answer: str) -> str:
    """移除 visual_qa 越权生成的图片文字转写句，文字内容应由 OCR 核验。"""

    answer = answer.strip()
    filtered = _filter_visual_sentences(answer)
    if filtered == answer:
        return answer
    # 图片编号属于段落归属，不能随首句过滤而丢失后续观察的来源。
    quoted = list(re.finditer(_VISUAL_QUOTED_TEXT, answer))
    headings = [heading for heading in re.finditer(
        r"(?m)^[ \t]*(?:(?:[-*]|\d+[.)])[ \t]+)?(?:\*\*)?E[1-9][0-9]*"
        r"(?:\*\*)?[ \t]*[:：](?:\*\*)?[ \t]*", answer
    ) if not any(quote.start() <= heading.start() < quote.end() for quote in quoted)]
    if headings:
        sections = [_filter_visual_sentences(answer[:headings[0].start()])]
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(answer)
            body = _filter_visual_sentences(answer[heading.end():end])
            if body:
                sections.append(heading.group() + body)
        return "\n\n".join(section for section in sections if section)
    return filtered


def _filter_visual_sentences(answer: str) -> str:
    # 引号内的句号属于待核验文字，不能切开后泄漏转写内容的后半句。
    boundaries = re.compile(
        _VISUAL_QUOTED_TEXT + r"|(?P<end>[。！？!?\n]|\.(?=\s|$))"
    )
    sentences: list[str] = []
    start = 0
    for match in boundaries.finditer(answer):
        if match.group("end") is not None:
            sentences.append(answer[start:match.end()])
            start = match.end()
    sentences.append(answer[start:])
    text_claim = re.compile(
        r"[\"“”‘’]|写着|字样|文字(?:是|为)|"
        r"(?<![A-Za-z0-9_])(?:reads?|says?|written|label(?:ed)?|text)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    kept = [sentence for sentence in sentences if sentence and not text_claim.search(sentence)]
    return "".join(kept).strip()


def _identical_visual_inputs(inputs: list[tuple[str, bytes]]) -> str:
    """只比较本次模型输入字节，不跨请求缓存或合并 Evidence。"""
    groups: list[tuple[bytes, list[str]]] = []
    for evidence_id, value in inputs:
        for previous, ids in groups:
            if value == previous:
                ids.append(evidence_id)
                break
        else:
            groups.append((value, [evidence_id]))
    identical = ["、".join(ids) for _, ids in groups if len(ids) > 1]
    if not identical:
        return ""
    return ("本次视觉输入一致性检查：" + "；".join(group + " 的输入图像完全相同" for group in identical)
            + "。检查仅适用于本次缩放和转换后的模型输入，不证明原始文件细节一致，"
              "也不证明采样时刻之间没有变化。")


class LocalInspectBackend:
    """实现十种操作的默认本地媒体后端。"""

    def __init__(self) -> None:
        """创建模型和 OCR 锁；大型模型都延迟到首次使用才加载。"""

        # 多个工作线程可能同时进入后端，模型推理和首次加载必须串行。
        self._model_lock = threading.RLock()
        self._ocr_lock = threading.RLock()
        self._ocr_engine: Any | None = None
        self._vlm: tuple[Any, Any, str, str] | None = None
        self._clip: tuple[Any, Any, str, str] | None = None
        self._grounding: tuple[Any, Any, str, str] | None = None
        self._audio_model: tuple[Any, Any, str, str, str] | None = None
        self._whisper: tuple[Any, str, str] | None = None

    def uses_gpu_model(self, operation: str) -> bool:
        """本地代理共用 GPU 时，仅协调会加载模型的操作。"""
        variables = {"locate_frames": "FOREN_CLIP_MODEL", "locate_regions": "FOREN_GROUNDING_MODEL",
                     "visual_qa": "FOREN_VLM_MODEL", "audio_qa": "FOREN_AUDIO_MODEL"}
        if operation == "asr":
            device = os.environ.get("FOREN_WHISPER_DEVICE", "cpu")
            if device == "auto" and self._whisper is not None:
                device = str(getattr(getattr(self._whisper[0], "model", None), "device", "auto"))
            return bool(os.environ.get("FOREN_WHISPER_MODEL")) and (device.startswith("cuda") or device == "auto")
        return bool(os.environ.get(variables[operation])) if operation in variables else False

    def release_gpu_models(self) -> None:
        """调用结束后释放闲置 CUDA 模型，保留 CPU 模型和非模型缓存。"""
        released = False
        with self._model_lock:
            for name, device_index in (("_clip", 2), ("_grounding", 2), ("_vlm", 2), ("_audio_model", 2), ("_whisper", 1)):
                cached = getattr(self, name)
                device = str(cached[device_index]) if cached is not None else "cpu"
                if name == "_whisper" and device == "auto":
                    device = str(getattr(getattr(cached[0], "model", None), "device", "auto"))
                if device.startswith("cuda") or device == "auto":
                    setattr(self, name, None)
                    released = True
            # 循环局部变量不能继续持有最后一个 GPU 模型。
            cached = None
            if released:
                gc.collect()
                import sys
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    async def aclose(self) -> None:
        """等待在用模型退出，再释放本次服务持有的模型与显存。"""

        def release() -> None:
            with self._model_lock, self._ocr_lock:
                self._vlm = self._clip = self._grounding = None
                self._audio_model = self._whisper = self._ocr_engine = None
                gc.collect()
                import sys
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        await asyncio.to_thread(release)

    async def probe(self, path: Path, media_kind: str) -> MediaProfileData:
        """按媒体种类读取尺寸、时长、帧率、页数或音频属性。"""

        try:
            if media_kind == "image":
                with Image.open(path) as raw:
                    if getattr(raw, "n_frames", 1) > 1:
                        raise BackendFailure("UNSUPPORTED_MEDIA", "multi-frame images require an explicit frame selection backend", stage="probe")
                    image = ImageOps.exif_transpose(raw); width, height = image.size
                    return MediaProfileData("image", width=width, height=height, format=raw.format, backend="Pillow", version=Image.__version__)
            if media_kind == "document":
                import fitz
                with fitz.open(path) as document:
                    return MediaProfileData("document", page_count=document.page_count, format="PDF", backend="PyMuPDF", version=fitz.VersionBind)
            if media_kind == "audio":
                # 音频和视频容器都用 PyAV，属性以第一条相应流为准。
                with av.open(str(path)) as container:
                    if not container.streams.audio:
                        raise BackendFailure("DECODE_FAILED", "container has no decodable audio stream", stage="probe")
                    audio_stream = container.streams.audio[0]
                    duration = float(audio_stream.duration * audio_stream.time_base) if audio_stream.duration is not None and audio_stream.time_base is not None else float(container.duration or 0) / av.time_base
                    sample_rate = int(audio_stream.codec_context.sample_rate or 0)
                    channels = int(audio_stream.codec_context.channels or 0)
                    audio_tracks = len(container.streams.audio)
                    container_format = container.format.name
                if duration <= 0 or sample_rate <= 0 or channels <= 0:
                    raise BackendFailure("DECODE_FAILED", "audio timing or stream properties are unavailable", stage="probe")
                return MediaProfileData("audio", duration_s=duration, sample_rate=sample_rate, channels=channels, audio_tracks=audio_tracks, format=container_format, backend="PyAV", version=av.__version__)
            if media_kind == "video":
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise BackendFailure("DECODE_FAILED", "container has no decodable video stream", stage="probe")
                    video_stream = container.streams.video[0]
                    duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration is not None and video_stream.time_base is not None else float(container.duration or 0) / av.time_base
                    fps = float(video_stream.average_rate or video_stream.base_rate or 0)
                    audio_tracks = len(container.streams.audio)
                    container_format = container.format.name
                    frame_count = int(video_stream.frames) if video_stream.frames else None
                    width, height = int(video_stream.codec_context.width), int(video_stream.codec_context.height)
                if duration <= 0 or width <= 0 or height <= 0:
                    raise BackendFailure("DECODE_FAILED", "video timing or dimensions are unavailable", stage="probe")
                return MediaProfileData("video", width=width, height=height, duration_s=duration, fps=fps or None, frame_count=frame_count, audio_tracks=audio_tracks, format=container_format, backend="PyAV", version=av.__version__)
        except BackendFailure:
            raise
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError, av.FFmpegError) as exc:
            raise BackendFailure("DECODE_FAILED", f"media probe failed: {type(exc).__name__}", stage="probe") from exc
        raise BackendFailure("UNSUPPORTED_MEDIA", "media type is not supported", stage="probe")

    @staticmethod
    def _frame_times(profile: MediaProfileData, params: GetFramesParams, policy: InspectPolicy) -> tuple[float, ...]:
        """将明确时间点或范围/count 转成实际请求时间元组。"""

        assert profile.duration_s is not None
        if params.times_s is not None: return params.times_s
        start, end = params.range_s or (0.0, profile.duration_s)
        count = params.count or policy.default_frame_count
        return tuple(start + (index + 0.5) * (end - start) / count for index in range(count))

    @staticmethod
    def _complete_frame_times(media: PreparedMedia, count: int, range_s: tuple[float, float] | None) -> tuple[float, ...] | None:
        """帧预算覆盖整份视频时读取实际 PTS；声明不符则回退到抽样。"""
        if not media.profile.frame_count or not 0 < media.profile.frame_count <= count:
            return None
        start, end = range_s or (0.0, math.inf)
        times = []
        try:
            with av.open(str(media.path)) as container:
                if not container.streams.video:
                    return None
                stream = container.streams.video[0]
                if stream.time_base is None:
                    return None
                origin = stream.start_time * stream.time_base if stream.start_time is not None else Fraction(0)
                for index, frame in enumerate(container.decode(stream)):
                    # 不信任容器帧数声明，不保留图像，也不扫描超过预算的长视频。
                    if index >= count:
                        return None
                    if frame.pts is None or frame.time_base is None:
                        return None
                    actual = float(frame.pts * frame.time_base - origin)
                    if not math.isfinite(actual):
                        return None
                    if start <= actual < end:
                        times.append(actual)
        except (OSError, ValueError, av.FFmpegError):
            return None
        return tuple(times) if times else None

    def _decode(self, media: PreparedMedia, times: tuple[float, ...], acquired_by: str,
                *, range_s: tuple[float, float] | None = None) -> BackendOutcome:
        """在各请求时间附近解码最近帧，并逐帧隔离可恢复错误。"""

        artifacts, rows, issues, seen = [], [], [], set()
        try:
            with av.open(str(media.path)) as container:
                if not container.streams.video:
                    raise BackendFailure("DECODE_FAILED", "container has no video stream", stage="decode")
                stream = container.streams.video[0]
                if stream.time_base is None:
                    raise BackendFailure("DECODE_FAILED", "video stream has no time base", stage="decode")
                origin = stream.start_time * stream.time_base if stream.start_time is not None else Fraction(0)
                window: tuple[float, float, tuple[float, av.VideoFrame] | None, tuple[float, av.VideoFrame] | None] | None = None
                for index, requested in enumerate(times):
                    try:
                        if window is not None and window[0] <= requested <= window[1]:
                            previous, following = window[2], window[3]
                        else:
                            window = None
                            # 先向前 seek 到关键帧，再顺序解码请求点两侧的相邻帧。
                            container.seek(max(0, int((Fraction(str(requested)) + origin) / stream.time_base)), stream=stream, backward=True, any_frame=False)
                            previous = following = None
                            for decoded in container.decode(stream):
                                if decoded.pts is None or decoded.time_base is None:
                                    continue
                                actual = float(decoded.pts * decoded.time_base - origin)
                                if range_s is not None:
                                    if actual < range_s[0]:
                                        continue
                                    if actual >= range_s[1]:
                                        break
                                if actual >= requested:
                                    following = (actual, decoded)
                                    break
                                previous = (actual, decoded)
                            if previous is not None or following is not None:
                                # 只复用已解码的相邻帧窗口，不猜测未知帧或额外向前解码。
                                window = (previous[0] if previous is not None else requested,
                                          following[0] if following is not None else math.inf, previous, following)
                        selected = (previous if following is None or (previous is not None
                                    and requested - previous[0] <= following[0] - requested) else following)
                        if selected is None:
                            raise BackendFailure("DECODE_FAILED", "no frame decoded near requested time", stage="decode")
                        actual, decoded = selected
                        pts_key = round(actual, 9)
                        # 多个请求时间可能落到同一真实帧，只发布一份 Artifact。
                        if pts_key in seen:
                            continue
                        frame = decoded.to_ndarray(format="bgr24")
                        handle = f"frame:{index}"; height, width = frame.shape[:2]
                        duration = getattr(decoded, "duration", 0)
                        frame_duration = float(duration * decoded.time_base) if duration and duration > 0 and decoded.time_base else None
                        artifacts.append(PendingArtifact(handle, _png(frame), "image/png", ".png", "video_frame", media.evidence_id, f"视频{media.evidence_id}在{actual:.3f}秒的帧", {"time_s": actual, "requested_time_s": requested, "frame_duration_s": frame_duration}, acquired_by))
                        rows.append({"handle": handle, "time_s": actual, "width": width, "height": height, "frame_duration_s": frame_duration})
                        seen.add(pts_key)
                    except BackendFailure as issue:
                        window = None
                        issues.append(BackendFailure(issue.code, str(issue), stage=issue.stage, retryable=True, locator={"requested_time_s": requested}))
                    except (OSError, ValueError, av.FFmpegError) as exc:
                        window = None
                        issues.append(BackendFailure("DECODE_FAILED", f"requested frame could not be decoded: {type(exc).__name__}", stage="decode", retryable=True, locator={"requested_time_s": requested}))
        except BackendFailure:
            raise
        except (OSError, ValueError, av.FFmpegError) as exc:
            raise BackendFailure("DECODE_FAILED", f"video decode failed: {type(exc).__name__}", stage="decode") from exc
        return BackendOutcome({"frames": rows}, tuple(artifacts), tuple(issues), {"requested_times_s": list(times), "backend": "PyAV", "version": av.__version__})

    async def get_frames(self, media: tuple[PreparedMedia, ...], params: GetFramesParams, policy: InspectPolicy) -> BackendOutcome:
        """执行 GET_FRAMES，并报告实际返回的不同帧数。"""

        times = self._frame_times(media[0].profile, params, policy)
        requested_count = len(times)
        if params.times_s is None:
            times = self._complete_frame_times(media[0], requested_count, params.range_s) or times
        outcome = self._decode(media[0], times, "INSPECT/get_frames", range_s=params.range_s)
        if len(outcome.result["frames"]) < requested_count and not outcome.issues:
            warning = BackendFailure("COUNT_REDUCED", f"requested {requested_count} frames; these selections returned {len(outcome.result['frames'])} distinct frames", stage="decode", severity="warning")
            outcome = BackendOutcome(outcome.result, outcome.artifacts, (warning,), outcome.details)
        return outcome

    async def get_pages(self, media: tuple[PreparedMedia, ...], params: GetPagesParams, policy: InspectPolicy) -> BackendOutcome:
        """均匀选择或按页码选择 PDF 页面，并以固定 DPI 渲染 PNG。"""

        import fitz
        document = fitz.open(media[0].path); total = document.page_count
        # 明确页码保序去重；范围请求则尽量均匀覆盖首尾。
        if params.pages is not None: pages = tuple(dict.fromkeys(params.pages))
        else:
            start, end = params.page_range or (1, total); count = min(params.count or policy.default_page_count, end - start + 1)
            if count == 1: pages = ((start + end) // 2,)
            elif count >= end - start + 1: pages = tuple(range(start, end + 1))
            else: pages = tuple(dict.fromkeys(round(start + i * (end - start) / (count - 1)) for i in range(count)))
        artifacts, rows, issues = [], [], []
        scale = policy.pdf_dpi / 72.0
        for index, page_number in enumerate(pages):
            try:
                page = document.load_page(page_number - 1); pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                # 渲染后的像素数可能远大于 PDF 文件大小，需独立设上限。
                if pixmap.width * pixmap.height > policy.maximum_page_pixels: raise BackendFailure("RESOURCE_LIMIT", "rendered page exceeds pixel limit", locator={"page_number": page_number})
                handle = f"page:{index}"; data = pixmap.tobytes("png")
                artifacts.append(PendingArtifact(handle, data, "image/png", ".png", "document_page", media[0].evidence_id, f"文档{media[0].evidence_id}第{page_number}页", {"page_number": page_number, "dpi": policy.pdf_dpi}, "INSPECT/get_pages"))
                rows.append({"handle": handle, "page_number": page_number, "width": pixmap.width, "height": pixmap.height})
            except BackendFailure as issue: issues.append(issue)
            except Exception as exc: issues.append(BackendFailure("DECODE_FAILED", f"page {page_number} failed: {type(exc).__name__}", stage="decode", locator={"page_number": page_number}))
        document.close()
        if params.count and params.count > len(pages):
            issues.append(BackendFailure("COUNT_REDUCED", f"requested {params.count} pages but only {len(pages)} pages were available", stage="decode", severity="warning"))
        return BackendOutcome({"pages": rows}, tuple(artifacts), tuple(issues), {"backend": "PyMuPDF", "dpi": policy.pdf_dpi})

    async def crop(self, media: tuple[PreparedMedia, ...], params: CropParams, policy: InspectPolicy) -> BackendOutcome:
        """把归一化裁剪框换算为覆盖完整区域的整数像素框。"""

        with Image.open(media[0].path) as raw: image = ImageOps.exif_transpose(raw).convert("RGBA" if "A" in raw.getbands() else "RGB")
        width, height = image.size; x1, y1, x2, y2 = params.box
        # 左上向下取整、右下向上取整，避免丢掉边缘像素。
        box_px = (math.floor(x1 * width), math.floor(y1 * height), math.ceil(x2 * width), math.ceil(y2 * height)); cropped = image.crop(box_px)
        artifact = PendingArtifact("crop:0", _png(cropped), "image/png", ".png", "image_crop", media[0].evidence_id, f"图片{media[0].evidence_id}裁剪区域", {"box": _jsonable_box(params.box), "box_px": list(box_px)}, "INSPECT/crop")
        return BackendOutcome({"handle": "crop:0", "width": cropped.width, "height": cropped.height, "box_px": list(box_px)}, (artifact,), details={"backend": "Pillow"})

    def _configured_grounding(self) -> tuple[Any, Any, str, str] | None:
        """按环境变量延迟加载本地 Grounding DINO 类模型。"""

        model_name = os.environ.get("FOREN_GROUNDING_MODEL")
        if not model_name:
            return None
        with self._model_lock:
            if self._grounding is None:
                try:
                    import torch
                    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # local_files_only 保证运行期不意外联网下载模型。
                    model_path = _resolve_local_model(model_name)
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        local_files_only=True,
                        use_fast=False,
                    )
                    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True).to(device).eval()
                except Exception as exc:
                    raise BackendFailure("PROCESS_FAILED", f"configured grounding model could not be loaded: {type(exc).__name__}") from exc
                self._grounding = (model, processor, device, model_name)
            return self._grounding

    def _locate_regions_with_grounding(self, media: PreparedMedia, params: LocateRegionsParams, policy: InspectPolicy) -> BackendOutcome:
        """用开放词汇目标检测模型定位任意文本描述区域。"""

        configured = self._configured_grounding()
        assert configured is not None
        model, processor, device, model_name = configured
        import torch
        try:
            with Image.open(media.path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB").copy()
            # Grounding DINO 把句点当作开放词汇短语的边界；缺少边界时，
            # 单词查询（如 ``person``）可能被处理器解析为空标签。
            query_prompt = params.query.strip()
            if query_prompt[-1] not in ".!?。！？":
                query_prompt += "."
            inputs = processor(images=image, text=query_prompt, return_tensors="pt")
            device_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._model_lock, torch.inference_mode():
                outputs = model(**device_inputs)
            processed = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=0.15,
                text_threshold=0.15,
                target_sizes=[(image.height, image.width)],
            )[0]
        except Exception as exc:
            raise BackendFailure("PROCESS_FAILED", f"region grounding failed: {type(exc).__name__}", retryable=True) from exc
        limit = params.top_k or policy.default_top_k
        labels = processed["text_labels"] if "text_labels" in processed else processed["labels"]
        # 后处理给出像素框和置信度，按分数取 top_k。
        candidates = sorted(
            zip(processed["scores"].detach().cpu().tolist(), processed["boxes"].detach().cpu().tolist(), labels),
            key=lambda item: item[0], reverse=True,
        )[:limit]
        artifacts, rows = [], []
        rgb = np.asarray(image)
        for index, (score, raw_box, label) in enumerate(candidates):
            x1, y1, x2, y2 = raw_box
            left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
            right, bottom = min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))
            if right <= left or bottom <= top:
                continue
            handle = f"region:{index}"
            box = (left / image.width, top / image.height, right / image.width, bottom / image.height)
            artifacts.append(PendingArtifact(handle, _png(Image.fromarray(rgb[top:bottom, left:right])), "image/png", ".png", "region_crop", media.evidence_id, f"{params.query}候选区域", {"box": list(box), "box_px": [left, top, right, bottom]}, "INSPECT/locate_regions"))
            rows.append({"handle": handle, "box": list(box), "box_px": [left, top, right, bottom], "matched_text": str(label), "score": float(score)})
        return BackendOutcome(
            {"image_evidence_id": media.evidence_id, "query": params.query, "regions": rows},
            tuple(artifacts), details={"backend": "Grounding DINO", "model": _model_reference(model_name)},
        )

    async def locate_regions(self, media: tuple[PreparedMedia, ...], params: LocateRegionsParams, policy: InspectPolicy) -> BackendOutcome:
        """颜色词走快速 HSV 分割，其余查询可选走 Grounding 模型。"""

        image = cv2.imread(str(media[0].path), cv2.IMREAD_COLOR)
        if image is None: raise BackendFailure("DECODE_FAILED", "image could not be read", stage="decode")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); query = params.query.lower(); masks = []
        # 红色跨越 HSV 色相首尾，因此需要两个掩码再合并。
        if "red" in query or "红" in query: masks = [cv2.inRange(hsv, np.array((0, 70, 50), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8)), cv2.inRange(hsv, np.array((168, 70, 50), dtype=np.uint8), np.array((180, 255, 255), dtype=np.uint8))]
        elif "green" in query or "绿" in query: masks = [cv2.inRange(hsv, np.array((35, 50, 40), dtype=np.uint8), np.array((90, 255, 255), dtype=np.uint8))]
        elif "blue" in query or "蓝" in query: masks = [cv2.inRange(hsv, np.array((90, 50, 40), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))]
        elif os.environ.get("FOREN_GROUNDING_MODEL"):
            return await asyncio.to_thread(self._locate_regions_with_grounding, media[0], params, policy)
        else: return BackendOutcome({"image_evidence_id": media[0].evidence_id, "query": params.query, "regions": []}, details={"backend": "OpenCV color locator", "capability": "color terms"})
        mask = np.maximum.reduce(masks); contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); h, w = image.shape[:2]
        candidates = sorted((cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 16), key=lambda box: box[2] * box[3], reverse=True)[: params.top_k or policy.default_top_k]
        artifacts, rows = [], []
        for index, (x, y, cw, ch) in enumerate(candidates):
            handle = f"region:{index}"; box = (x / w, y / h, (x + cw) / w, (y + ch) / h); score = min(1.0, cw * ch / (w * h))
            artifacts.append(PendingArtifact(handle, _png(image[y:y+ch, x:x+cw]), "image/png", ".png", "region_crop", media[0].evidence_id, f"{params.query}候选区域", {"box": list(box), "box_px": [x, y, x+cw, y+ch]}, "INSPECT/locate_regions"))
            rows.append({"handle": handle, "box": list(box), "box_px": [x, y, x+cw, y+ch], "matched_text": params.query, "score": score})
        return BackendOutcome({"image_evidence_id": media[0].evidence_id, "query": params.query, "regions": rows}, tuple(artifacts), details={"backend": "OpenCV color locator"})

    @staticmethod
    def _ocr_scene_times(media: PreparedMedia, start: float, end: float, base_times: tuple[float, ...], base_fps: float) -> tuple[float, ...]:
        """寻找均匀 OCR 采样遗漏的短场景，并返回其中心时间。"""
        cuts: list[float] = []
        previous: np.ndarray | None = None
        try:
            with av.open(str(media.path)) as container:
                stream = container.streams.video[0]
                if stream.time_base is None:
                    return ()
                origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
                if start > 0:
                    container.seek(max(0, int((start + origin) / float(stream.time_base))), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    if frame.time_base is None:
                        continue
                    time_s = float(frame.pts * frame.time_base) - origin
                    if time_s < start:
                        continue
                    if time_s >= end:
                        break
                    gray = frame.to_ndarray(format="gray")
                    thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                    # 小灰度缩略图平均差足够大时近似视为一次镜头切换。
                    if previous is not None and float(np.mean(np.abs(thumb - previous))) >= 0.18:
                        cuts.append(time_s)
                    previous = thumb
        except (OSError, ValueError, av.FFmpegError, IndexError):
            return ()

        boundaries = [start, *cuts, end]
        maximum_short_scene = 1.0 / max(base_fps, 1e-6)
        candidates: list[float] = []
        for scene_start, scene_end in zip(boundaries, boundaries[1:]):
            if scene_end <= scene_start or scene_end - scene_start > maximum_short_scene + 1e-6:
                continue
            if any(scene_start <= time_s < scene_end for time_s in base_times):
                continue
            candidates.append((scene_start + scene_end) / 2.0)
        return tuple(dict.fromkeys(round(value, 9) for value in candidates))

    async def ocr(self, media: tuple[PreparedMedia, ...], params: OcrParams, policy: InspectPolicy) -> BackendOutcome:
        """对图片直接 OCR；对视频执行基础、短场景和文字邻域三阶段采样。"""

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise BackendFailure("PROCESS_FAILED", "RapidOCR backend is not installed") from exc
        with self._ocr_lock:
            if self._ocr_engine is None:
                self._ocr_engine = RapidOCR()
            engine = self._ocr_engine
        artifacts: list[PendingArtifact] = []
        results: list[dict[str, Any]] = []
        issues: list[BackendFailure] = []
        if media[0].media_kind == "image":
            # 静态图片不需要 coverage，结果直接引用输入 Evidence。
            images = [(None, media[0].path, media[0].evidence_id)]
            for time_s, source, ref in images:
                with self._ocr_lock:
                    output, _ = engine(source)
                items = []
                with Image.open(source) as image:
                    width, height = image.size
                for item in output or []:
                    polygon, text, confidence = item; xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
                    items.append({"text": text, "box": [min(xs)/width, min(ys)/height, max(xs)/width, max(ys)/height], "confidence": float(confidence)})
                results.append({"image_handle": None, "image_evidence_id": ref, "time_s": time_s, "items": items})
            coverage = None
            return BackendOutcome({"coverage": coverage, "results": results}, tuple(artifacts), tuple(issues),
                                  {"backend": "RapidOCR", "limitations": [_OCR_COVERAGE_NOTE]})

        start, end = params.range_s or (0.0, media[0].profile.duration_s or 0.0)
        # 第一阶段按 ocr_base_fps 均匀覆盖完整范围。
        base_count = max(1, math.ceil((end - start) * policy.ocr_base_fps))
        base_times = tuple(start + (index + 0.5) * (end - start) / base_count for index in range(base_count))
        scene_times = self._ocr_scene_times(media[0], start, end, base_times, policy.ocr_base_fps)
        phase_counts = {"base": 0, "scene": 0, "refinement": 0}
        failed_ranges: list[tuple[float, float]] = []
        seen_pts: set[float] = set()
        text_times: list[float] = []
        artifact_index = 0

        def add_failure_range(requested: float) -> None:
            """把单帧失败近似映射回一小段时间范围。"""

            radius = min(0.25, 0.5 / max(policy.ocr_base_fps, 1e-6))
            failed_ranges.append((max(start, requested - radius), min(end, requested + radius)))

        def process_times(times: tuple[float, ...], phase: str) -> int:
            """解码一组时间点、执行 OCR，并累计 Artifact、结果和问题。"""

            nonlocal artifact_index
            if not times:
                return 0
            remaining = max(0, policy.maximum_frames - len(results))
            # 所有阶段共享 maximum_frames 总预算。
            selected_times = times[:remaining]
            if len(selected_times) < len(times):
                first_unprocessed = times[len(selected_times)]
                failed_ranges.append((max(start, first_unprocessed - 0.5 / max(policy.ocr_base_fps, 1e-6)), end))
                issues.append(BackendFailure(
                    "RESOURCE_LIMIT", "OCR frame budget exhausted before the requested range was fully processed",
                    stage="process", locator={"range_s": list(failed_ranges[-1]), "maximum_frames": policy.maximum_frames},
                ))
            decoded = self._decode(media[0], selected_times, "INSPECT/ocr", range_s=(start, end))
            issues.extend(decoded.issues)
            for issue in decoded.issues:
                requested = issue.locator.get("requested_time_s") if issue.locator else None
                if isinstance(requested, (int, float)):
                    add_failure_range(float(requested))
            artifact_by_handle = {artifact.handle: artifact for artifact in decoded.artifacts}
            added = 0
            for row in decoded.result["frames"]:
                actual = float(row["time_s"])
                pts_key = round(actual, 9)
                if pts_key in seen_pts:
                    continue
                source_artifact = artifact_by_handle.get(row["handle"])
                if source_artifact is None:
                    continue
                seen_pts.add(pts_key)
                handle = f"ocr-frame:{artifact_index}"
                artifact_index += 1
                artifact = replace(source_artifact, handle=handle)
                source = io.BytesIO(artifact.data)
                try:
                    with self._ocr_lock:
                        output, _ = engine(source)
                except Exception as exc:
                    add_failure_range(actual)
                    issues.append(BackendFailure(
                        "PROCESS_FAILED", f"OCR failed for one frame: {type(exc).__name__}",
                        stage="process", retryable=True, locator={"time_s": actual},
                    ))
                    continue
                source.seek(0)
                with Image.open(source) as image:
                    width, height = image.size
                items = []
                for item in output or []:
                    polygon, text, confidence = item; xs, ys = [point[0] for point in polygon], [point[1] for point in polygon]
                    items.append({"text": text, "box": [min(xs)/width, min(ys)/height, max(xs)/width, max(ys)/height], "confidence": float(confidence)})
                if items:
                    text_times.append(actual)
                artifacts.append(artifact)
                results.append({"image_handle": handle, "image_evidence_id": None, "time_s": actual, "items": items})
                phase_counts[phase] += 1
                added += 1
            return added

        process_times(base_times, "base")
        # 第二阶段补短场景，第三阶段在已检测文字附近加密采样。
        if len(results) < policy.maximum_frames and not any(issue.code == "RESOURCE_LIMIT" for issue in issues):
            process_times(scene_times, "scene")
        if len(results) < policy.maximum_frames and not any(issue.code == "RESOURCE_LIMIT" for issue in issues):
            refinement_times = tuple(sorted({
                round(candidate, 9)
                for detected in text_times
                for candidate in (detected - policy.ocr_refinement_s, detected + policy.ocr_refinement_s)
                if start <= candidate < end
            }))
            process_times(refinement_times, "refinement")

        sampling = {
            "base_fps": policy.ocr_base_fps,
            "base_frames": phase_counts["base"],
            "scene_frames": phase_counts["scene"],
            "refinement_frames": phase_counts["refinement"],
            "processed_frames": len(results),
        }
        coverage = {"range_s": [start, end], "sampling": sampling, "failed_ranges": [list(value) for value in failed_ranges]}
        return BackendOutcome({"coverage": coverage, "results": results}, tuple(artifacts), tuple(issues), {"backend": "RapidOCR"})

    async def asr(self, media: tuple[PreparedMedia, ...], params: AsrParams, policy: InspectPolicy) -> BackendOutcome:
        """使用显式配置的本地 faster-whisper 模型转写指定范围。"""

        model_name = os.environ.get("FOREN_WHISPER_MODEL")
        if not model_name: raise BackendFailure("PROCESS_FAILED", "ASR requires FOREN_WHISPER_MODEL pointing to a local faster-whisper model")
        with av.open(str(media[0].path)) as container:
            audio_track = self._select_audio_track(container.streams.audio)
        samples, _, (start, end) = self._av_audio_samples(media[0].path, params.range_s, audio_track)
        device = os.environ.get("FOREN_WHISPER_DEVICE", "cpu")
        with self._model_lock:
            # 首次使用才加载；后续若环境配置变化则拒绝混用旧实例。
            if self._whisper is None:
                try:
                    from faster_whisper import WhisperModel
                    compute_type = "float16" if device == "cuda" else "int8"
                    model = WhisperModel(model_name, local_files_only=True, device=device, compute_type=compute_type)
                except Exception as exc:
                    raise BackendFailure("PROCESS_FAILED", f"configured ASR model could not be loaded: {type(exc).__name__}") from exc
                self._whisper = (model, device, model_name)
            model, loaded_device, loaded_name = self._whisper
            if loaded_device != device or loaded_name != model_name:
                raise BackendFailure("PROCESS_FAILED", "ASR model configuration changed after backend initialization")
        with self._model_lock:
            # 输入已经按媒体时间裁切，模型时间相对该数组；不再按文件解码后的位置裁切。
            segments, info = model.transcribe(samples, language=params.language, vad_filter=True)
            rows = [{"start_s": max(start, start + float(s.start)), "end_s": min(end, start + float(s.end)), "text": s.text.strip()} for s in segments if s.end > 0 and s.start < end - start]
        return BackendOutcome({"coverage": {"range_s": [start,end], "processed_ranges": [[start,end]], "failed_ranges": []}, "language": getattr(info, "language", params.language), "audio_track": audio_track, "segments": rows},
            details={"backend": "faster-whisper", "model": _model_reference(model_name), "device": device, "vad_filter": True,
                     "speech_duration_s": getattr(info, "duration_after_vad", None),
                     "limitations": ["已启用讲话检测以过滤静音和非讲话片段；未检出可靠语音时 segments 可为空。检测可能漏掉较弱讲话，空转写不证明绝对无人声。",
                                     "时间戳对应原音频位置；语音识别不核验说话人数量、身份、音乐、掌声或其他非语音声音，不能由转写未提及来判断这些内容不存在。"]})

    @staticmethod
    def _wav_samples(path: Path, range_s: tuple[float, float] | None = None) -> tuple[np.ndarray, int, tuple[float,float]]:
        """读取 PCM WAV，选择能量最大声道并归一化到约 -1..1。"""

        with wave.open(str(path), "rb") as audio:
            rate, channels, width, frames = audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes(); duration = frames/rate
            start, end = range_s or (0.0, duration); audio.setpos(int(start*rate)); raw = audio.readframes(int((end-start)*rate))
        if width == 3:
            # NumPy 没有原生 24-bit dtype，需要手工拼接并做符号扩展。
            octets = np.frombuffer(raw, dtype=np.uint8)
            if len(octets) % 3: raise BackendFailure("DECODE_FAILED", "truncated 24-bit WAV sample", stage="decode")
            packed = octets.reshape(-1, 3).astype(np.int32)
            samples = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            samples = np.where(samples & 0x800000, samples - 0x1000000, samples).astype(np.float32)
        else:
            if width == 1: samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            elif width == 2: samples = np.frombuffer(raw, dtype=np.dtype("<i2")).astype(np.float32)
            elif width == 4: samples = np.frombuffer(raw, dtype=np.dtype("<i4")).astype(np.float32)
            else: raise BackendFailure("UNSUPPORTED_MEDIA", "unsupported WAV sample width")
            if width == 1: samples -= 128
        if channels > 1:
            # 多声道 QA/对齐选 RMS 最大声道，避免平均造成相位抵消。
            per_channel = samples.reshape(-1, channels)
            channel_rms = np.sqrt(np.mean(per_channel * per_channel, axis=0)) if len(per_channel) else np.zeros(channels)
            samples = per_channel[:, int(np.argmax(channel_rms))]
        peak = float(2 ** (8*width-1)); return samples/peak, rate, (start,end)

    @staticmethod
    def _select_audio_track(streams: Any, audio_track: int | None = None) -> int:
        if audio_track is None:
            audio_track = next((i for i, stream in enumerate(streams)
                                if stream.disposition & type(stream.disposition).default), 0)
        if not 0 <= audio_track < len(streams):
            raise BackendFailure("NO_AUDIO", "requested audio track does not exist", stage="probe")
        return audio_track

    @classmethod
    def _av_audio_samples(cls, path: Path, range_s: tuple[float, float] | None = None, audio_track: int | None = None) -> tuple[np.ndarray, int, tuple[float, float]]:
        """通过 PyAV 解码任意音视频音轨为 16 kHz 单声道浮点样本。"""

        rate = 16_000
        with av.open(str(path)) as container:
            stream = container.streams.audio[cls._select_audio_track(container.streams.audio, audio_track)]
            # 音视频共用主视频的零点；纯音频采用第一条音轨的零点。
            reference = container.streams.video[0] if container.streams.video else container.streams.audio[0]
            origin = reference.start_time * reference.time_base if reference.start_time is not None and reference.time_base is not None else Fraction(0)
            absolute_start = Fraction(str(range_s[0])) + origin if range_s else origin
            absolute_end: Fraction | None
            if range_s:
                absolute_end = Fraction(str(range_s[1])) + origin
            else:
                # 默认范围取媒体内实际解码的音轨覆盖；流时长不是视频坐标下的终点。
                limit_stream = reference if container.streams.video else stream
                limit_origin = (limit_stream.start_time * limit_stream.time_base
                                if limit_stream.start_time is not None and limit_stream.time_base is not None else origin)
                absolute_end = (limit_origin + limit_stream.duration * limit_stream.time_base
                                if limit_stream.duration is not None and limit_stream.time_base is not None else None)
            if range_s and range_s[0] > 0:
                if stream.time_base is not None:
                    container.seek(int(absolute_start / stream.time_base), stream=stream, backward=True)
                else:
                    container.seek(int(absolute_start * av.time_base), backward=True)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            chunks: list[np.ndarray] = []
            cursor = origin
            covered_until: Fraction | None = absolute_start if range_s else None
            first_sample: Fraction | None = None
            timestamp_resolution = max(Fraction(1, rate), stream.time_base or Fraction(0))

            def append_output(output: av.AudioFrame, fallback: Fraction) -> None:
                nonlocal covered_until, first_sample
                output_start = output.pts * output.time_base if output.pts is not None and output.time_base is not None else fallback
                values = output.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                first = max(0, math.ceil((absolute_start - output_start) * rate))
                last = min(len(values), math.ceil((absolute_end - output_start) * rate)) if absolute_end is not None else len(values)
                if first < last:
                    actual_start = output_start + Fraction(first, rate)
                    if covered_until is not None and abs(actual_start - covered_until) > timestamp_resolution:
                        raise BackendFailure("DECODE_FAILED", "audio timestamps are discontinuous in requested range; select a continuous subrange", stage="decode")
                    if first_sample is None:
                        first_sample = actual_start
                    chunks.append(values[first:last])
                    covered_until = output_start + Fraction(last, rate)

            resampled = False
            for frame in container.decode(stream):
                frame_start = frame.pts * frame.time_base if frame.pts is not None and frame.time_base is not None else cursor
                # Vorbis 长短块切换时部分 FFmpeg 版本保留包终点，却给出偏移的帧起点。
                # 仅在包终点和实际样本数恰好衔接前帧时修正，真实时间缺口仍会失败。
                if stream.codec_context.name == "vorbis" and frame.duration and frame.time_base is not None:
                    corrected = frame_start + frame.duration * frame.time_base - Fraction(frame.samples, frame.sample_rate)
                    if corrected != frame_start and abs(corrected - cursor) <= Fraction(1, frame.sample_rate):
                        frame.pts = round(corrected / frame.time_base)
                        frame_start = frame.pts * frame.time_base
                frame_end = frame_start + Fraction(frame.samples, frame.sample_rate)
                cursor = frame_end
                if frame_end <= absolute_start: continue
                if absolute_end is not None and frame_start >= absolute_end: break
                resampled = True
                for output in resampler.resample(frame):
                    if output is None: continue
                    append_output(output, frame_start)
            if resampled:
                for output in resampler.resample(None):
                    append_output(output, cursor)
            if not chunks:
                raise BackendFailure("DECODE_FAILED", "no audio samples decoded in requested range", stage="decode")
            assert covered_until is not None and first_sample is not None
            if range_s and absolute_end is not None and absolute_end - covered_until > timestamp_resolution:
                raise BackendFailure("DECODE_FAILED", "audio samples do not cover the requested range; select a covered subrange", stage="decode")
            samples = np.concatenate(chunks)
            actual_range = range_s or (float(first_sample - origin), float(min(covered_until, absolute_end) - origin) if absolute_end is not None else float(covered_until - origin))
        return samples, rate, actual_range

    @classmethod
    def _audio_samples(cls, media: PreparedMedia, range_s: tuple[float, float] | None = None, audio_track: int | None = None) -> tuple[np.ndarray, int, tuple[float, float]]:
        """WAV 走轻量标准库快路，其余音视频走 PyAV。"""

        if media.media_kind == "audio" and media.profile.format and "wav" in media.profile.format.lower() and audio_track in (None, 0):
            return cls._wav_samples(media.path, range_s)
        if media.media_kind in {"audio", "video"}:
            return cls._av_audio_samples(media.path, range_s, audio_track)
        raise BackendFailure("NO_AUDIO", "media has no supported audio stream", stage="probe")

    @staticmethod
    def _fft_correlate(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """以 O(n log n) 计算与 np.correlate 相同 lag 顺序的完整互相关。"""
        output_size = len(left) + len(right) - 1
        fft_size = 1 << (output_size - 1).bit_length()
        circular = np.fft.irfft(
            np.fft.rfft(left, fft_size) * np.conj(np.fft.rfft(right, fft_size)),
            fft_size,
        )
        negative_lags = circular[-(len(right) - 1):] if len(right) > 1 else np.empty(0)
        return np.concatenate((negative_lags, circular[:len(left)]))[:output_size]

    @staticmethod
    def _resample_for_correlation(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """用多相滤波把样本率精确变换到相关分析目标值。"""

        if source_rate == target_rate:
            return samples
        divisor = math.gcd(source_rate, target_rate)
        return signal.resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)

    def _configured_audio_model(self) -> tuple[Any, Any, str, str, str] | None:
        """按 FOREN_AUDIO_MODEL 延迟加载本地 Qwen 音频理解模型。"""

        model_name = os.environ.get("FOREN_AUDIO_MODEL")
        if not model_name:
            return None
        with self._model_lock:
            if self._audio_model is None:
                try:
                    import torch
                    from transformers import (
                        AutoConfig,
                        AutoProcessor,
                        Qwen2AudioForConditionalGeneration,
                        Qwen2_5OmniForConditionalGeneration,
                    )
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16 if device == "cuda" else torch.float32
                    model_path = _resolve_local_model(model_name)
                    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                    if config.model_type == "qwen2_5_omni":
                        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                            model_path,
                            local_files_only=True,
                            dtype=dtype,
                        )
                        # Demo 只需要文本答案；移除语音合成部分可显著减少显存占用。
                        model.disable_talker()
                        family = "qwen2_5_omni"
                    elif config.model_type == "qwen2_audio":
                        model = Qwen2AudioForConditionalGeneration.from_pretrained(
                            model_path,
                            local_files_only=True,
                            dtype=dtype,
                        )
                        family = "qwen2_audio"
                    else:
                        raise ValueError(
                            f"unsupported audio model type: {config.model_type}"
                        )
                    model = model.to(device).eval()
                except Exception as exc:
                    raise BackendFailure("PROCESS_FAILED", f"configured audio model could not be loaded: {type(exc).__name__}") from exc
                self._audio_model = (model, processor, device, model_name, family)
            return self._audio_model

    def _audio_qa_with_model(self, media: PreparedMedia, params: AudioQaParams) -> BackendOutcome:
        """提取目标音频范围并交给已配置音频语言模型回答。"""

        configured = self._configured_audio_model()
        assert configured is not None
        model, processor, device, model_name, family = configured
        with av.open(str(media.path)) as container:
            audio_track = self._select_audio_track(container.streams.audio, params.audio_track)
        samples, rate, actual = self._audio_samples(media, params.range_s, audio_track)
        target_rate = int(processor.feature_extractor.sampling_rate)
        samples = self._resample_for_correlation(samples, rate, target_rate)
        # 用处理器规定的多模态 chat template 同时传音频与问题。
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "local-audio"},
            {"type": "text", "text": params.question},
        ]}]
        try:
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=prompt, audio=[samples], sampling_rate=target_rate, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with self._model_lock:
                if family == "qwen2_5_omni":
                    generated = model.generate(
                        **inputs,
                        generation_mode="text",
                        thinker_max_new_tokens=160,
                        thinker_do_sample=False,
                    )
                else:
                    generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            answer = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            if not answer:
                raise BackendFailure("PROCESS_FAILED", "audio model returned an empty answer")
        except BackendFailure:
            raise
        except Exception as exc:
            raise BackendFailure("PROCESS_FAILED", f"audio question answering failed: {type(exc).__name__}", retryable=True) from exc
        return BackendOutcome(
            {"range_s": list(actual), "audio_track": audio_track, "answer": answer, "limitations": ["回答仅基于所选音轨和时间范围。"]},
            details={"backend": family, "model": _model_reference(model_name), "question": params.question},
        )

    async def audio_qa(self, media: tuple[PreparedMedia, ...], params: AudioQaParams, policy: InspectPolicy) -> BackendOutcome:
        """有模型时做语义问答，否则明确只返回声学统计。"""

        if os.environ.get("FOREN_AUDIO_MODEL"):
            return await asyncio.to_thread(self._audio_qa_with_model, media[0], params)
        with av.open(str(media[0].path)) as container:
            audio_track = self._select_audio_track(container.streams.audio, params.audio_track)
        samples, rate, actual = self._audio_samples(media[0], params.range_s, audio_track); rms = float(np.sqrt(np.mean(samples*samples))) if len(samples) else 0.0
        # 轻量后备只做可测量结论，不假装理解声音语义。
        if rms < 0.003: answer = "该范围基本静音。"
        else:
            spectrum = np.abs(np.fft.rfft(samples)); frequency = float(np.fft.rfftfreq(len(samples), 1/rate)[int(np.argmax(spectrum))]) if len(samples) else 0.0
            answer = f"该范围存在可听信号；均方根电平约{rms:.3f}，最强频率约{frequency:.1f} Hz。"
        wav_path = (media[0].media_kind == "audio" and media[0].profile.format
                    and "wav" in media[0].profile.format.lower() and audio_track == 0)
        backend = "wave/numpy" if wav_path else "PyAV/numpy"
        return BackendOutcome({"range_s": list(actual), "audio_track": audio_track, "answer": answer, "limitations": ["本地声学适配器只测量静音、能量和主频，不能识别开放式声音事件。"]}, details={"backend": backend})

    def _configured_vlm(self) -> tuple[Any, Any, str, str] | None:
        """按 FOREN_VLM_MODEL 延迟加载本地视觉语言模型。"""

        model_name = os.environ.get("FOREN_VLM_MODEL")
        if not model_name:
            return None
        with self._model_lock:
            if self._vlm is None:
                try:
                    import torch
                    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16 if device == "cuda" else torch.float32
                    model_path = _resolve_local_model(model_name)
                    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True, dtype=dtype).to(device).eval()  # type: ignore[arg-type]
                except Exception as exc:
                    raise BackendFailure("PROCESS_FAILED", f"configured visual model could not be loaded: {type(exc).__name__}") from exc
                self._vlm = (model, processor, device, model_name)
            return self._vlm

    def _visual_qa_with_vlm(self, media: tuple[PreparedMedia, ...], question: str, maximum_pixels: int) -> BackendOutcome:
        """按像素预算缩放图片，再执行多图视觉问答。"""

        configured = self._configured_vlm()
        assert configured is not None
        model, processor, device, model_name = configured
        images: list[Image.Image] = []
        try:
            for item in media:
                with Image.open(item.path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    if image.width * image.height > maximum_pixels:
                        # 等比例缩小而不裁剪，保留整图语义。
                        scale = math.sqrt(maximum_pixels / (image.width * image.height))
                        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
                    images.append(image.copy())
            identity_note = _identical_visual_inputs([
                (item.evidence_id, f"RGB/{image.width}/{image.height}/".encode() + image.tobytes())
                for item, image in zip(media, images)
            ])
            content: list[dict[str, Any]] = []
            sampled_video_frames = False
            for index, image in enumerate(images):
                item = media[index]
                caption = f"图{index + 1}，Evidence {item.evidence_id}"
                if item.source_context:
                    try:
                        context = json.loads(item.source_context)
                    except (ValueError,TypeError):
                        context = {}
                    locator = context.get("locator") if isinstance(context,dict) else None
                    if isinstance(locator,dict) and isinstance(locator.get("time_s"), (int,float)):
                        caption += f"，实际解码时间 {locator['time_s']} 秒"
                        sampled_video_frames = True
                    if isinstance(context,dict) and context.get("derived_from"):
                        caption += "，来自 " + ", ".join(context["derived_from"])
                content.append({"type": "text", "text": caption})
                content.append({"type": "image", "image": image})
            instruction = question + (
                "\n本次实际提供的图片仅为 " + ", ".join(item.evidence_id for item in media)
                + f"，共 {len(images)} 张。问题提到的其他图片若不在此列表中，就没有提供给你。"
                "不能把问题对其他图片的描述当作已观察事实；跨图比较仅限本次同时提供的图片。"
            )
            sampling_limit = (
                "视频相关观察仅覆盖本次提供的离散帧，未连续检查视频；"
                "最早和最晚的命中帧不能证明片段的起止边界。"
            )
            if sampled_video_frames:
                instruction += (
                    "\n" + sampling_limit + "判断大致范围时，在同一视频的已提供采样中，"
                    "找首个命中时刻 f、最后命中时刻 l、之前最近的未命中时刻 p 和之后最近的未命中时刻 n。"
                    "起点可限定在 (p, f]，终点可限定在 (l, n]。这些是采样中的相邻时刻，"
                    "不要求源视频的连续两帧；只在缺少某一侧未命中样本时说明那一侧界限未知。"
                )
            if len(images) > 1:
                instruction += (
                    f"\n本次输入有 {len(images)} 张图。请先按 Evidence 编号逐图回答，每图最多两句；"
                    "随后完成问题要求的跨图比较。某张图中存在目标，不代表其他图中也有。"
                    "时间关系只使用图像标签中的实际解码时间，不使用图序号推断先后。"
                )
            content.append({"type": "text", "text": instruction})
            if identity_note:
                content[-1]["text"] += "\n" + identity_note
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是取证式图像内容分析器。只依据当前输入图像中可直接观察到的像素回答，"
                                "把观察与推断明确区分。除非姓名、地点、组织或品牌在图中清晰可读，"
                                "否则不得给出具体专名，也不得依靠人脸记忆、文件名或外部常识识别人。"
                                "图像不能确认具体身份时必须明确说无法确认，但可以谨慎描述服装、物体、"
                                "场景和可能的职业。如果服装、装备与环境共同清晰指向一种职业，应以"
                                "‘很可能是’给出该职业，同时保持具体姓名未知。不得虚构徽章文字或"
                                "不可辨认的细节。visual_qa 不负责转写图片文字，因此不得引用或猜测"
                                "任何姓名牌、徽章或小字的具体内容；需要文字内容时应由 OCR 单独核验。"
                                "如果输入多张图片，必须按图序号和 Evidence 编号逐图分别回答问题，"
                                "每图简要描述后再比较，避免重复句子；不能把其中一张的观察套给其他图片。"
                                "图序号只是输入顺序；时间判断使用标签中的实际解码时间。视频帧的前后场景"
                                "指时间顺序中前一个和后一个场景，不是物体正面与侧面；未提供的场景不能猜测。"
                                + _VISUAL_SOURCE_NOTE
                            ),
                        }
                    ],
                },
                {"role": "user", "content": content},
            ]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            maximum_tokens = min(1024, 160 + 112 * len(images))
            with self._model_lock:
                generated = model.generate(**inputs, max_new_tokens=maximum_tokens, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            answer_tokens = generated[:, prompt_length:]
            eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            if answer_tokens.shape[1] >= maximum_tokens and not any(token in eos_ids for token in answer_tokens[0].tolist()):
                raise BackendFailure(
                    "RESOURCE_LIMIT",
                    "visual model reached its output token limit without completing the answer; "
                    "the truncated answer was not stored. Ask a more focused visual question.",
                )
            raw_answer = processor.batch_decode(answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            answer = _without_unverified_visual_text(raw_answer)
            if not answer:
                raise BackendFailure(
                    "PROCESS_FAILED",
                    "visual model returned only unverified image-text claims; use OCR for text",
                )
        except BackendFailure:
            raise
        except Exception as exc:
            raise BackendFailure("PROCESS_FAILED", f"visual question answering failed: {type(exc).__name__}", retryable=True) from exc
        return BackendOutcome(
            {
                "image_evidence_ids": [item.evidence_id for item in media],
                "answer": "\n".join(f"{item.evidence_id} 来源定位：{item.source_context or '未提供'}" for item in media)
                          + ("\n" + identity_note if identity_note else "") + f"\n画面观察：{answer}",
                "limitations": [
                    "回答仅基于本次提供的图像内容；具体人物身份需要可读标识或外部来源另行核验。",
                    *([sampling_limit] if sampled_video_frames else []),
                    *([_VISUAL_TEXT_OMISSION_NOTE] if answer != raw_answer else []),
                ],
            },
            details={"backend": "Qwen3-VL", "model": _model_reference(model_name), "question": question},
        )

    async def visual_qa(self, media: tuple[PreparedMedia, ...], params: VisualQaParams, policy: InspectPolicy) -> BackendOutcome:
        """有 VLM 时回答开放问题，否则返回诚实的颜色与尺寸统计。"""

        if os.environ.get("FOREN_VLM_MODEL"):
            return await asyncio.to_thread(self._visual_qa_with_vlm, media, params.question, policy.maximum_model_image_pixels)
        summaries = []
        for item in media:
            with Image.open(item.path) as raw: image = np.asarray(ImageOps.exif_transpose(raw).convert("RGB").resize((64,64)), dtype=np.float32)
            mean = image.mean(axis=(0,1)); summaries.append(f"{item.evidence_id}尺寸{item.profile.width}×{item.profile.height}，平均RGB约({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")
        answer = "；".join(summaries)
        if len(media) > 1:
            answer += "。输入图像的整体颜色统计" + ("接近。" if max(np.linalg.norm(np.asarray(Image.open(x.path).convert('RGB').resize((1,1)),dtype=float) - np.asarray(Image.open(media[0].path).convert('RGB').resize((1,1)),dtype=float)) for x in media[1:]) < 40 else "存在明显差异。")
        return BackendOutcome({"image_evidence_ids": [x.evidence_id for x in media], "answer": answer, "limitations": ["本地视觉适配器只提供尺寸和整体颜色统计；开放式语义问答需注入VLM后端。"]}, details={"backend": "Pillow statistics", "question": params.question})

    def _configured_clip(self) -> tuple[Any, Any, str, str] | None:
        """按 FOREN_CLIP_MODEL 延迟加载本地图文嵌入模型。"""

        model_name = os.environ.get("FOREN_CLIP_MODEL")
        if not model_name:
            return None
        with self._model_lock:
            if self._clip is None:
                try:
                    import torch
                    from transformers import AutoProcessor, CLIPModel
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_path = _resolve_local_model(model_name)
                    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                    # 两个不同文本若编码完全相同，说明下载的是不完整/退化 tokenizer。
                    tokenizer_probe = processor(text=["a cat", "an airplane"], return_tensors="pt", padding=True)["input_ids"]
                    if torch.equal(tokenizer_probe[0], tokenizer_probe[1]):
                        raise ValueError("embedding model tokenizer is incomplete or degenerate")
                    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()  # type: ignore[arg-type]
                except Exception as exc:
                    raise BackendFailure("PROCESS_FAILED", f"configured embedding model could not be loaded: {type(exc).__name__}") from exc
                self._clip = (model, processor, device, model_name)
            return self._clip

    def _clip_similarity_scores(self, frame_data: list[bytes], text: str | None, query_image: Path | None) -> np.ndarray:
        """计算各帧与可选文字/图片查询的平均 CLIP 相似度。"""

        configured = self._configured_clip()
        assert configured is not None
        model, processor, device, _ = configured
        import torch
        frame_images = []
        for data in frame_data:
            with Image.open(io.BytesIO(data)) as raw:
                frame_images.append(raw.convert("RGB").copy())
        with self._model_lock, torch.inference_mode():
            frame_inputs = processor(images=frame_images, return_tensors="pt")
            frame_inputs = {key: value.to(device) for key, value in frame_inputs.items()}
            frame_features = model.get_image_features(**frame_inputs)
            frame_features = getattr(frame_features, "pooler_output", frame_features)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
            scores: list[Any] = []
            if text:
                text_inputs = processor(text=[text], return_tensors="pt", padding=True)
                text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
                text_features = model.get_text_features(**text_inputs)
                text_features = getattr(text_features, "pooler_output", text_features)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                scores.append(frame_features @ text_features.T)
            if query_image:
                with Image.open(query_image) as raw:
                    query = ImageOps.exif_transpose(raw).convert("RGB")
                query_inputs = processor(images=[query], return_tensors="pt")
                query_inputs = {key: value.to(device) for key, value in query_inputs.items()}
                query_features = model.get_image_features(**query_inputs)
                query_features = getattr(query_features, "pooler_output", query_features)
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                scores.append(frame_features @ query_features.T)
            # 图文同时存在时等权平均，再把余弦相似度从 -1..1 映射到 0..1。
            combined = torch.stack(scores).mean(dim=0).squeeze(-1)
            return ((combined + 1.0) / 2.0).detach().cpu().numpy()

    @staticmethod
    def _histogram(image: np.ndarray) -> np.ndarray:
        """计算归一化的 8×8×8 BGR 颜色直方图。"""

        hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256]*3).flatten(); norm = np.linalg.norm(hist)
        return hist / norm if norm else hist

    @classmethod
    def _visual_descriptor(cls, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造颜色、低分辨率空间布局和 DCT 纹理三部分描述子。"""

        histogram = cls._histogram(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        spatial = np.asarray(cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA), dtype=np.float32).flatten()
        spatial = spatial - spatial.mean()
        spatial_norm = np.linalg.norm(spatial)
        if spatial_norm: spatial = spatial / spatial_norm
        dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA))[:8, :8].flatten()[1:]
        dct_norm = np.linalg.norm(dct)
        if dct_norm: dct = dct / dct_norm
        return histogram, spatial, dct

    @staticmethod
    def _visual_similarity(left: tuple[np.ndarray, np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        """按固定权重合并三类描述子相似度并裁剪到 0..1。"""

        histogram = float(np.dot(left[0], right[0]))
        spatial = (float(np.dot(left[1], right[1])) + 1.0) / 2.0
        dct = (float(np.dot(left[2], right[2])) + 1.0) / 2.0
        return min(1.0, max(0.0, 0.2 * histogram + 0.55 * spatial + 0.25 * dct))

    async def locate_frames(self, media: tuple[PreparedMedia, ...], params: LocateFramesParams, policy: InspectPolicy, query_media: PreparedMedia | None = None) -> BackendOutcome:
        """粗采样视频帧，并用 CLIP 或本地颜色/空间特征排序。"""

        text_query = params.query.text if isinstance(params.query, (TextFrameQuery, TextImageFrameQuery)) else None
        color = None
        if text_query:
            color = "red" if re.search(r"\bred\b|红", text_query, re.I) else "blue" if re.search(r"\bblue\b|蓝", text_query, re.I) else None
        if text_query and not os.environ.get("FOREN_CLIP_MODEL") and color is None:
            raise BackendFailure("UNSUPPORTED_QUERY",
                "Semantic frame retrieval requires a configured CLIP model. "
                "Use get_frames to inspect the video directly, or provide an image query; "
                "sharpness does not measure this text query.")
        # 粗采样数随时长增长，但始终限制在 8..64 和全局帧预算内。
        start, end = params.range_s or (0.0, media[0].profile.duration_s or 0.0); count = min(policy.maximum_frames, max(8, min(64, int(end-start)*2)))
        decoded = self._decode(media[0], tuple(start+(i+0.5)*(end-start)/count for i in range(count)),
                               "INSPECT/locate_frames", range_s=(start, end))
        query_descriptor = None
        if query_media:
            # 没有 CLIP 时，图片查询使用手工视觉描述子。
            image = cv2.imread(str(query_media.path)); query_descriptor = self._visual_descriptor(image) if image is not None else None
        clip_scores: np.ndarray | None = None
        if decoded.artifacts and os.environ.get("FOREN_CLIP_MODEL"):
            # 配置 CLIP 后，文本和图片查询都进入同一语义嵌入空间。
            try:
                clip_scores = await asyncio.to_thread(
                    self._clip_similarity_scores,
                    [artifact.data for artifact in decoded.artifacts],
                    text_query,
                    query_media.path if query_media else None,
                )
            except BackendFailure:
                raise
            except Exception as exc:
                raise BackendFailure("PROCESS_FAILED", f"semantic frame location failed: {type(exc).__name__}", retryable=True) from exc
        scored = []
        for frame_index, (row, artifact) in enumerate(zip(decoded.result["frames"], decoded.artifacts)):
            frame = cv2.imdecode(np.frombuffer(artifact.data, np.uint8), cv2.IMREAD_COLOR); descriptor = self._visual_descriptor(frame)
            if clip_scores is not None:
                scored.append((float(clip_scores[frame_index]), row, artifact))
                continue
            text_score: float | None = None
            if color:
                # 无语义模型时只测量颜色，适用范围随结果保存。
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                if color == "red": text_score = float((cv2.inRange(hsv,np.array((0,70,50),dtype=np.uint8),np.array((12,255,255),dtype=np.uint8))>0).mean())
                elif color == "blue": text_score = float((cv2.inRange(hsv,np.array((90,50,40),dtype=np.uint8),np.array((135,255,255),dtype=np.uint8))>0).mean())
            image_score = self._visual_similarity(descriptor, query_descriptor) if query_descriptor is not None else None
            score = (image_score + text_score) / 2 if image_score is not None and text_score is not None else image_score if image_score is not None else text_score or 0.0
            scored.append((score,row,artifact))
        selected = sorted(scored, key=lambda x:x[0], reverse=True)[: params.top_k or policy.default_top_k]
        artifacts = tuple(x[2] for x in selected); rows = [{"handle": x[2].handle, "time_s": x[1]["time_s"], "score": x[0]} for x in selected]
        backend_name = "CLIP semantic retrieval" if clip_scores is not None else "OpenCV spatial/color retrieval"
        details: dict[str, Any] = {"backend": backend_name, "query_type": params.query.type,
            "sampling_interval_s": (end - start) / count,
            "limitations": [_FRAME_RETRIEVAL_NOTE,
                "coverage 是工具内部的采样范围和数量，不表示 HI 已查看所有采样帧；候选时间不构成连续场景边界。"]}
        if clip_scores is None and text_query:
            details["limitations"].append("当前未配置语义模型，文本部分仅测量颜色覆盖比例，未识别文字中的对象、动作或其他语义。")
        if clip_scores is not None and self._clip is not None:
            details["model"] = self._clip[3]
        radius = (end - start) / count / 2
        failed_ranges = [[max(start, float(issue.locator["requested_time_s"]) - radius),
                          min(end, float(issue.locator["requested_time_s"]) + radius)]
                         for issue in decoded.issues if issue.locator and "requested_time_s" in issue.locator]
        return BackendOutcome({"coverage": {"range_s":[start,end], "coarse_frames":len(scored),
            "refined_ranges":[], "failed_ranges":failed_ranges}, "candidates":rows}, artifacts, decoded.issues, details)

    async def align_media(self, media: tuple[PreparedMedia, ...], params: AlignMediaParams, policy: InspectPolicy) -> BackendOutcome:
        """音频模式结合波形与频谱路径；视觉模式提取连续帧对应。"""

        if params.mode == "audio":
            # 统一到不高于 8 kHz 可显著降低长音频互相关的内存和计算量。
            a, rate_a, range_a = self._audio_samples(media[0], params.range_a_s); b, rate_b, range_b = self._audio_samples(media[1], params.range_b_s)
            sample_rate = min(8000, rate_a, rate_b)
            a_small = self._resample_for_correlation(a, rate_a, sample_rate)
            b_small = self._resample_for_correlation(b, rate_b, sample_rate)
            corr = self._fft_correlate(a_small - a_small.mean(), b_small - b_small.mean())
            # 最大相关位置换算为 B 相对 A 的时间偏移。
            index = int(np.argmax(corr)); lag_s = (index - (len(b_small) - 1)) / sample_rate
            time_b_minus_a = range_b[0] - range_a[0] - lag_s
            overlap_a_start = max(range_a[0], range_b[0] - time_b_minus_a)
            overlap_a_end = min(range_a[1], range_b[1] - time_b_minus_a)
            lag_samples = index - (len(b_small) - 1)
            if lag_samples >= 0:
                overlap_a, overlap_b = a_small[lag_samples:], b_small[:len(a_small) - lag_samples]
            else:
                overlap_a, overlap_b = a_small[:len(b_small) + lag_samples], b_small[-lag_samples:]
            overlap_size = min(len(overlap_a), len(overlap_b))
            overlap_a, overlap_b = overlap_a[:overlap_size], overlap_b[:overlap_size]
            if overlap_size:
                overlap_a = overlap_a - overlap_a.mean(); overlap_b = overlap_b - overlap_b.mean()
            denominator = np.linalg.norm(overlap_a) * np.linalg.norm(overlap_b)
            score = min(1.0, max(0.0, float(np.dot(overlap_a, overlap_b) / denominator))) if denominator else 0.0
            matches = []
            if denominator > 1e-9 and score >= 0.7 and overlap_a_end - overlap_a_start >= 0.25:
                overlap_b_start = overlap_a_start + time_b_minus_a
                overlap_b_end = overlap_a_end + time_b_minus_a
                matches.append({
                    "range_a_s": [overlap_a_start, overlap_a_end],
                    "range_b_s": [overlap_b_start, overlap_b_end],
                    "score": score,
                    "anchors": [{"time_a_s": overlap_a_start, "time_b_s": overlap_b_start, "image_a_handle": None, "image_b_handle": None}],
                })
            details: dict[str, Any] = {"backend":"numpy FFT correlation", "limitations":[_ALIGNMENT_COVERAGE_NOTE]}
            if not matches:
                matches, step = spectral_matches(a_small, b_small, sample_rate, range_a, range_b,
                                                  params.top_k or policy.default_top_k)
                details = {"backend":"local spectral sequence alignment", "sampling_interval_s":step,
                    "boundary_uncertainty_s":None,
                    "limitations":[_ALIGNMENT_COVERAGE_NOTE,
                                   "sampling_interval_s 是测量间隔，未估计边界误差上限（boundary_uncertainty_s=null）；不能换算成 ±误差或可排除的最短编辑长度。",
                                   "序列分数按匹配量、相似度和时间步长惩罚排序，并按本次采样规模归一化；短片段分数较低不表示没有对应关系。",
                                   "Approximate 0.5–2x tempo correspondence; pitch shifts and heavily altered audio may not match."]}
            return BackendOutcome({"mode":"audio","coverage":{"a":{"range_s":list(range_a),"processed":{"unit":"seconds","value":range_a[1]-range_a[0]},"failed_ranges":[]},"b":{"range_s":list(range_b),"processed":{"unit":"seconds","value":range_b[1]-range_b[0]},"failed_ranges":[]}},"matches":matches}, details=details)
        ranges = [params.range_a_s or (0.0, media[0].profile.duration_s or 0.0),
                  params.range_b_s or (0.0, media[1].profile.duration_s or 0.0)]
        decoded: list[list[AlignmentFrame]] = []
        steps = []
        frame_rates = [item.profile.fps for item in media]
        visual_sample_rate = min(max(4.0, fps) if fps is not None and math.isfinite(fps) and fps > 0 else 4.0
                          for fps in frame_rates)
        durations = [end - start for start, end in ranges]
        # 总帧数至多512，比较矩阵至多65536个元素；两侧共享实际时间步长。
        frame_budget = min(policy.maximum_frames, 512)
        step = 1 / visual_sample_rate
        counts = [max(1, math.ceil(duration / step)) for duration in durations]
        resource_issues: tuple[BackendFailure, ...] = ()
        if sum(counts) > frame_budget:
            step = sum(durations) / max(1, frame_budget - 2)
            counts = [max(1, math.ceil(duration / step)) for duration in durations]
            if sum(counts) > frame_budget or min(counts) < 4:
                raise BackendFailure("RESOURCE_LIMIT", "frame budget cannot cover both ranges with enough temporal samples; narrow the source ranges")
            resource_issues = (BackendFailure("COUNT_REDUCED",
                f"visual alignment used a {step:.6g}-second common sampling interval within {frame_budget} frames; narrow the source ranges only if finer boundaries are needed", severity="warning"),)
        decode_issues = []
        for item, interval, count in zip(media, ranges, counts):
            steps.append(step)
            sampled, issues = sample_alignment_frames(item, tuple(interval[0] + i * step for i in range(count)), interval)
            decoded.append(sampled)
            decode_issues.append(issues)
        if any(len(group) < 4 for group in decoded):
            raise BackendFailure("DECODE_FAILED", "too few distinct frames for temporal alignment", stage="decode")

        def descriptors(frames: list[AlignmentFrame]) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            variants: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(6)]
            for sampled in frames:
                frame = sampled.image
                height, width = frame.shape[:2]
                boxes = [(0, 0, width, height)]
                for x, y in ((0, 0), (0.15, 0), (0, 0.15), (0.15, 0.15), (0.075, 0.075)):
                    boxes.append((int(x * width), int(y * height), int((x + 0.85) * width), int((y + 0.85) * height)))
                for variant, (x1, y1, x2, y2) in zip(variants, boxes):
                    variant.append(self._visual_descriptor(frame[y1:y2, x1:x2]))
            return variants

        left, right = descriptors(decoded[0]), descriptors(decoded[1])
        matrix = np.zeros((len(left[0]), len(right[0])), dtype=np.float32)
        structural_matrix = np.zeros_like(matrix)
        if matrix.size:
            for desc_a, desc_b in [(left[0], variant) for variant in right] + [(variant, right[0]) for variant in left[1:]]:
                components = [np.stack([item[k] for item in desc_a]) @ np.stack([item[k] for item in desc_b]).T for k in range(3)]
                similarity = 0.2 * components[0] + 0.55 * (components[1] + 1) / 2 + 0.25 * (components[2] + 1) / 2
                matrix = np.maximum(matrix, np.clip(similarity, 0, 1))
                # 空间与纹理用于区分重叠候选，避免亮度变化掩盖更好的时间对应。
                structural = (0.55 * (components[1] + 1) / 2 + 0.25 * (components[2] + 1) / 2) / 0.8
                structural_matrix = np.maximum(structural_matrix, np.clip(structural, 0, 1))
        # 低纹理、近乎相同的工作图保留直接对应；其余由局部特征检索和几何验证筛选。
        geometric = geometric_similarities(decoded[0], decoded[1])
        matrix = np.maximum(np.where((matrix >= .985) & (structural_matrix >= .99), matrix, 0), geometric)
        paths = align_sequences(matrix, top_k=params.top_k or policy.default_top_k, relative_similarity=True)

        continuous_paths: list[TemporalMatch] = []
        for path in paths:
            segments: list[list[tuple[int, int]]] = [[]]
            for pair in path.pairs:
                if segments[-1] and any(
                    decoded[side][pair[side]].time_s
                    - decoded[side][segments[-1][-1][side]].time_s > steps[side] * 2.1
                    for side in (0, 1)):
                    segments.append([])
                segments[-1].append(pair)
            continuous_paths.extend(TemporalMatch(tuple(segment), path_score(matrix, tuple(segment), relative_similarity=True))
                                    for segment in segments if len(segment) >= 4)
        continuous_paths.sort(key=lambda match: match.score, reverse=True)
        continuous_paths = list(prune_dominated_paths(tuple(continuous_paths), matrix))
        # 按候选顺序保留完整的首末帧对，覆盖不同片段且不超过视觉输入容量。
        selected_pairs: list[tuple[int, int]] = []
        retained_paths = continuous_paths[:params.top_k or policy.default_top_k]
        pair_limit = min(8, policy.maximum_visual_qa_images) // 2
        for path in retained_paths:
            for pair in (path.pairs[0], path.pairs[-1]):
                if pair not in selected_pairs and len(selected_pairs) < pair_limit:
                    selected_pairs.append(pair)
        artifacts = []
        frame_handles = {}
        preview_issues: list[BackendFailure] = []
        for side in (0, 1):
            selected_indices = list(dict.fromkeys(pair[side] for pair in selected_pairs))
            if not selected_indices:
                continue
            original = self._decode(media[side], tuple(decoded[side][i].time_s for i in selected_indices),
                                    "INSPECT/align_media", range_s=ranges[side])
            preview_issues.extend(original.issues)
            for artifact in original.artifacts:
                handle = f"{side}:{artifact.handle}"
                artifacts.append(replace(artifact, handle=handle))
                frame_handles[(side, round(float(artifact.locator["time_s"]), 9))] = handle
        matches = []
        for path in retained_paths:
            first_a, first_b = path.pairs[0]
            last_a, last_b = path.pairs[-1]
            bounds = []
            for side_index, first, last in ((0, first_a, last_a), (1, first_b, last_b)):
                frames = decoded[side_index]
                bounds.append([round(max(ranges[side_index][0], frames[first].time_s - steps[side_index]), 9),
                               round(min(ranges[side_index][1], frames[last].time_s + steps[side_index]), 9)])
            anchors = []
            indices = sorted(set(int(i) for i in np.linspace(0, len(path.pairs) - 1, min(4, len(path.pairs)))))
            for index in indices:
                frame_a, frame_b = path.pairs[index]
                times = [decoded[0][frame_a].time_s, decoded[1][frame_b].time_s]
                anchors.append({"time_a_s": times[0], "time_b_s": times[1],
                    "image_a_handle": frame_handles.get((0, round(times[0], 9))),
                    "image_b_handle": frame_handles.get((1, round(times[1], 9)))})
            matches.append({"range_a_s": bounds[0], "range_b_s": bounds[1], "score": path.score, "anchors": anchors})
        coverage = {side: {"range_s": list(ranges[index]), "processed": {"unit": "frames", "value": len(decoded[index])},
            "failed_ranges": [[max(ranges[index][0], float(issue.locator["requested_time_s"]) - steps[index] / 2),
                               min(ranges[index][1], float(issue.locator["requested_time_s"]) + steps[index] / 2)]
                              for issue in decode_issues[index] if issue.locator and "requested_time_s" in issue.locator]}
                    for index, side in enumerate(("a", "b"))}
        return BackendOutcome({"mode": "visual", "coverage": coverage, "matches": matches}, tuple(artifacts),
            resource_issues + tuple(issue for group in decode_issues for issue in group) + tuple(preview_issues),
            details={"backend": "SIFT/FLANN + RANSAC + local temporal alignment", "sampling_interval_s": steps,
                     "boundary_uncertainty_s": None, "speed_ratio_range": [0.5, 2.0],
                     "limitations": [_ALIGNMENT_COVERAGE_NOTE,
                                     "序列分数按匹配量、相似度和时间步长惩罚排序，并按本次采样规模归一化；短片段分数较低不表示没有对应关系。",
                                     "按候选顺序在视觉输入容量内附首末锚点原帧，每对含A/B两帧；其余锚点只记录时间，可按对应源E和时间调用get_frames。帧对是待核对的对应候选，不能用同类场景任意重配时间。",
                                     "sampling_interval_s 是取样间隔，候选范围在首末命中外扩一个间隔；未估计真实剪辑边界的误差上限（boundary_uncertainty_s=null），不能换算成 ±误差或可排除的最短编辑长度。低纹理、镜像、倒放或大幅编辑可能无法对齐。",
                                     "静态或重复画面可能有多个对应位置；未匹配的空档不等于完整删减区间。"]})
