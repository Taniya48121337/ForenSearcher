"""网页、纯文本和 PDF 候选的打开与正文抽取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..contracts import CandidateSource, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchRequest
from ..extractors import extract_html, extract_pdf
from ..policy import SearchPolicy
from .http import SafeHttpClient


class DocumentSourceOpener:
    """把文档型 Candidate 转换为可持久化的 OpenedMaterial。"""

    def __init__(self, http: SafeHttpClient, policy: SearchPolicy | None = None, renderer: Any | None = None) -> None:
        """注入安全 HTTP 客户端和可选动态页面渲染器。"""

        self.http, self.policy, self.renderer = http, policy or http.policy, renderer

    def supports(self, candidate: CandidateSource, output_type: str) -> bool:
        """文字输出总是支持；其他输出只接受文档型候选。"""

        return output_type == "text" or candidate.candidate_type in {"page", "document", "unknown"}

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        """下载来源，按 PDF 或 HTML 路径抽取，并保留原始字节。"""

        resource = await self.http.fetch(candidate.source_url, maximum_bytes=self.policy.maximum_document_bytes)
        media_type = resource.media_type
        # 同时检查 MIME 和文件魔数，兼容服务器错误标注 Content-Type。
        is_pdf = media_type == "application/pdf" or resource.content.startswith(b"%PDF-")
        if is_pdf:
            try:
                pdf_extracted = extract_pdf(resource.content)
            except Exception as exc:
                # 文件已经成功取得，即使抽取失败也以 partial 保存原 PDF。
                material = OpenedMaterial("document", "partial", f"PDF acquired; text extraction failed: {type(exc).__name__}", resource.content,
                    "application/pdf", ".pdf", ("text",), candidate.search_title or Path(resource.final_url).name or "PDF document",
                    "PDF file; no extracted page locator", "SEARCH/open/pdf", "source_metadata",
                    SearchError("extract", "EXTRACT_FAILED", "PDF was saved but its text could not be extracted"))
                return OpenedSource(resource.final_url, (material,))
            if not pdf_extracted.text:
                # 扫描型 PDF 没有文字层；这里明确说明尚未做 OCR。
                material = OpenedMaterial("document", "partial", f"PDF acquired; pages={pdf_extracted.page_count}; pages_with_text=0", resource.content,
                    "application/pdf", ".pdf", ("text",), candidate.search_title or Path(resource.final_url).name or "PDF document",
                    "PDF file; no existing text layer", "SEARCH/open/pdf", "source_metadata",
                    SearchError("extract", "EXTRACT_FAILED", "PDF has no extractable text layer; OCR was not run"))
            else:
                # 只有每页都有可提取文字时才算 ready。
                status: Literal["ready", "partial"] = "ready" if pdf_extracted.pages_with_text == pdf_extracted.page_count else "partial"
                error = None if status == "ready" else SearchError("extract", "EXTRACT_FAILED", "some PDF pages have no extractable text")
                material = OpenedMaterial("document", status, pdf_extracted.text, resource.content, "application/pdf", ".pdf", ("text",),
                    candidate.search_title or Path(resource.final_url).name or "PDF document", "page numbers embedded in extracted text",
                    "SEARCH/open/pdf+pypdf", "source_excerpt", error)
            return OpenedSource(resource.final_url, (material,))
        if media_type == "text/plain":
            text = resource.content.decode(resource.encoding or "utf-8", errors="replace")
            if not text.strip():
                raise SearchFailure(SearchError("extract", "EXTRACT_FAILED", "source contains no text"))
            # 纯文本中的尖括号、实体及空白是原文，不作为 HTML 标签解析。
            content = "\n".join(f"[{index}] {line}" for index, line in enumerate(text.splitlines(), 1))
            return OpenedSource(resource.final_url, (OpenedMaterial(
                "page", "ready", content, resource.content, "text/plain", ".txt", ("text",),
                candidate.search_title or resource.final_url, "line numbers embedded in extracted text",
                "SEARCH/open/text", "source_excerpt"),))
        if media_type not in {"text/html", "application/xhtml+xml", "text/plain"} and not resource.content.lstrip().startswith((b"<!DOCTYPE html", b"<html", b"<HTML")):
            raise SearchFailure(SearchError("extract", "UNSUPPORTED_FORMAT", f"source returned unsupported media type {media_type}"))
        html_extracted = extract_html(resource.content, base_url=resource.final_url, encoding=resource.encoding)
        if html_extracted.soft_error:
            raise SearchFailure(SearchError("open", html_extracted.soft_error, "page content indicates that the source is unavailable or restricted"))  # type: ignore[arg-type]
        static_is_probably_shell = not html_extracted.text or (
            len(html_extracted.text) < 80 and b"<script" in resource.content.lower()
        )
        # 静态 HTML 像空壳时才启动昂贵的浏览器渲染。
        if static_is_probably_shell and self.renderer is not None:
            rendered = await self.renderer.render(resource.final_url)
            rendered_extracted = extract_html(rendered, base_url=resource.final_url)
            if rendered_extracted.soft_error:
                raise SearchFailure(SearchError("open", rendered_extracted.soft_error, "rendered page indicates that the source is unavailable or restricted"))  # type: ignore[arg-type]
            if len(rendered_extracted.text) > len(html_extracted.text):
                # 动态版本只有确实提供更多正文时才替换静态版本。
                html_extracted = rendered_extracted
                resource = type(resource)(resource.requested_url, resource.final_url, resource.status_code, resource.headers, rendered, "text/html", "utf-8")
        if not html_extracted.text:
            raise SearchFailure(SearchError("extract", "EXTRACT_FAILED", "page contains no extractable text"))
        title = html_extracted.title or candidate.search_title or resource.final_url
        return OpenedSource(resource.final_url, (OpenedMaterial("page", "ready", html_extracted.text, resource.content,
            "text/html", ".html", ("text",), title, "paragraph numbers embedded in extracted text",
            "SEARCH/open/http+trafilatura", "source_excerpt"),), discovered_links=html_extracted.links)
