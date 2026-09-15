"""INSPECT 测试媒体生成器。

这里用代码生成很小的图片、视频、音频和 PDF，使测试输入可重复、无需下载外部样本。
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

from foren_searcher.store import GlobalStore


def image_bytes(*, color: str = "white", red_box: bool = True, size: tuple[int, int] = (320, 180)) -> bytes:
    """生成可选红框的 PNG 图片字节。"""

    image = Image.new("RGB", size, color)
    if red_box:
        ImageDraw.Draw(image).rectangle((80, 40, 240, 140), fill=(230, 20, 20))
    output = io.BytesIO(); image.save(output, "PNG"); return output.getvalue()


def text_image_bytes(text: str = "FOREN SEARCH 2026") -> bytes:
    """生成带清晰黑字的 PNG，供 OCR 测试使用。"""

    image = np.full((180, 720, 3), 255, dtype=np.uint8)
    cv2.putText(image, text, (24, 108), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image); assert ok
    return encoded.tobytes()


def pdf_bytes() -> bytes:
    """生成两页尺寸不同且含文字/色块的 PDF。"""

    document = fitz.open()
    page = document.new_page(width=300, height=200); page.insert_text((30, 70), "ForenSearcher page one")
    page = document.new_page(width=200, height=300); page.draw_rect(fitz.Rect(30, 40, 170, 240), color=(0, 0, 1), fill=(0, 0, 1))
    data = document.tobytes(); document.close(); return data


def wav_bytes(*, frequency: float = 440.0, seconds: float = 1.0, silence: bool = False) -> bytes:
    """生成指定频率、时长或静音的单声道 PCM WAV。"""

    rate = 16000; output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(rate)
        samples = [0 if silence else int(12000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(int(rate * seconds))]
        audio.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return output.getvalue()


def make_video(path: Path, *, shifted: bool = False, frames: int = 30, fps: float = 10.0) -> None:
    """写入小型 MJPG 视频；shifted 用于构造已知时间偏移。"""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 96))
    assert writer.isOpened()
    for index in range(frames):
        image = np.zeros((96, 160, 3), dtype=np.uint8)
        logical = max(0, index - 5) if shifted else index
        image[:] = (20 + logical * 3 % 100, 30, 40)
        if logical >= 15: cv2.rectangle(image, (55, 25), (120, 75), (0, 0, 255), -1)
        cv2.putText(image, str(logical), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        writer.write(image)
    writer.release()


def ingest(store: GlobalStore, data: bytes, *, source: str, filename: str, media_type: str, modality: str):
    """把生成的媒体字节提交到测试 Store，并返回 Evidence。"""

    return store.commit_ingested_evidence(source_ref=source, media_type=media_type, modalities=(modality,), label=filename, original_filename=filename, data=data)
