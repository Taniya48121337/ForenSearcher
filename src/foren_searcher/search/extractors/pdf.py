"""把 PDF 字节抽取成带页码标记的纯文本。"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    """PDF 抽取结果和两项质量统计。"""

    text: str
    page_count: int
    pages_with_text: int


def extract_pdf(raw: bytes) -> ExtractedPdf:
    """读取 PDF；处理空密码加密，并逐页保留可提取文本。"""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf") from exc
    reader = PdfReader(io.BytesIO(raw), strict=False)
    if reader.is_encrypted:
        # 部分 PDF 标记为加密但允许空密码打开，所以先尝试空密码。
        try:
            if reader.decrypt("") == 0:
                raise ValueError("encrypted PDF cannot be opened")
        except Exception as exc:
            raise ValueError("encrypted PDF cannot be opened") from exc
    pages: list[str] = []
    for number, page in enumerate(reader.pages, 1):
        # 没有文字层的扫描页不伪造内容，但会体现在 page_count 差值中。
        value = (page.extract_text() or "").strip()
        if value:
            pages.append(f"[page {number}]\n{value}")
    return ExtractedPdf("\n\n".join(pages), len(reader.pages), len(pages))
