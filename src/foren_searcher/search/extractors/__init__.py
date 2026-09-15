"""网页和文档正文抽取器包；具体实现由 SourceOpener 调用。"""

from .html import ExtractedHtml, extract_html
from .pdf import ExtractedPdf, extract_pdf

__all__ = ["ExtractedHtml", "ExtractedPdf", "extract_html", "extract_pdf"]
