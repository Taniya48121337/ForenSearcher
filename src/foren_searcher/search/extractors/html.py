"""HTML 正文抽取。

原始 HTML 会由上层另存为 Artifact；这里仅负责从字节中得到标题、段落和链接。
优先使用 trafilatura 提升正文质量，不可用时仍有标准库解析器作为确定性后备。
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Any
from urllib.parse import urljoin

from lxml import etree, html


@dataclass(frozen=True, slots=True)
class ExtractedHtml:
    """HTML 的规范化文本结果；soft_error 表示页面内容暗示的软失败。"""

    title: str | None
    text: str
    paragraphs: tuple[str, ...]
    links: tuple[str, ...]
    soft_error: str | None = None


class _FallbackParser(HTMLParser):
    """无第三方依赖的最小 HTML 解析器。"""

    def __init__(self, base_url: str) -> None:
        """准备标题、正文和链接容器，并记住相对链接基址。"""

        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """进入忽略区、标题区，或收集 a 标签链接。"""

        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        """离开忽略区或标题区。"""

        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        """压缩空白后把文本归入标题或正文。"""

        value = " ".join(data.split())
        if not value or self._ignored:
            return
        if self._in_title:
            self.title_parts.append(value)
        else:
            self.text_parts.append(value)


def _visible_html(text: str, *, base_url: str) -> str:
    """按静态 HTML 整理可见文本；保留原始 Artifact 供渲染器读取。"""

    try:
        tree = html.document_fromstring(text, parser=html.HTMLParser(no_network=True, remove_comments=True))
    except (etree.ParserError, ValueError):
        return text
    bases = tree.xpath('(//base[@href])[1]/@href')
    link_base = urljoin(base_url, bases[0].strip()) if bases else base_url
    # 正文抽取器会把相对链接接到域名根目录；先按最终页面地址和首个 base 解析。
    # 这里只规范化引用，实际访问仍由 SafeHttpClient 检查协议、地址和跳转。
    for anchor in tree.iter("a"):
        href = anchor.get("href")
        if href is not None:
            anchor.set("href", urljoin(link_base, href.strip()))
    for node in tuple(tree.iter()):
        if not isinstance(node.tag, str):
            continue
        declarations = re.findall(r"(?:^|;)\s*(display|visibility)\s*:\s*([^;]+)",
                                  node.get("style", ""), flags=re.IGNORECASE)
        styles = {key.lower(): value.split("!", 1)[0].strip().lower()
                  for key, value in declarations}
        hidden = ("hidden" in node.attrib or styles.get("display") == "none"
                  or styles.get("visibility") in {"hidden", "collapse"})
        if hidden or node.tag.lower() == "template":
            if node is tree:
                return ""
            if node.getparent() is not None:
                node.drop_tree()
    # 普通段落的源码折行不是段落边界；预格式化内容的空白则必须保留。
    def collapse_wrapping(node: Any, *, in_block: bool = False, preserve: bool = False) -> None:
        if not isinstance(node.tag, str):
            return
        tag = node.tag.lower()
        in_block = in_block or tag in {"p", "li", "dt", "dd", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
        preserve = preserve or tag in {"pre", "textarea", "code", "samp", "kbd", "script", "style"} or bool(
            re.search(r"(?:^|;)\s*white-space\s*:", node.get("style", ""), flags=re.IGNORECASE))
        if in_block and not preserve and node.text:
            node.text = re.sub(r"[ \t\r\n\f]+", " ", node.text)
        for child in node:
            collapse_wrapping(child, in_block=in_block, preserve=preserve)
            if in_block and not preserve and child.tail:
                child.tail = re.sub(r"[ \t\r\n\f]+", " ", child.tail)

    collapse_wrapping(tree)
    return html.tostring(tree, encoding="unicode")


def extract_html(raw: bytes, *, base_url: str, encoding: str | None = None) -> ExtractedHtml:
    """解码并抽取 HTML，最后识别常见“404/登录验证”软错误页。"""

    text = _visible_html(raw.decode(encoding or "utf-8", errors="replace"), base_url=base_url)
    parser = _FallbackParser(base_url)
    parser.feed(text)
    fallback_parts = tuple(part for part in parser.text_parts if part)
    extracted = ""
    try:
        # trafilatura 能过滤导航栏等模板噪声；没安装时回退不算系统失败。
        import trafilatura
        extracted = trafilatura.extract(text, include_links=True, include_formatting=False, url=base_url) or ""
        primary = trafilatura.extract(text, include_links=True, include_formatting=False,
                                      url=base_url, fast=True) or ""
        extracted = _merge_article_sections(text, extracted, primary)
    except (ImportError, ValueError):
        extracted = ""
    paragraphs = tuple(part.strip() for part in extracted.splitlines() if part.strip()) or fallback_parts
    # 段落编号会进入 Evidence locator，便于 Finding 精确引用。
    body = "\n\n".join(f"[{index}] {part}" for index, part in enumerate(paragraphs, 1))
    sample = " ".join(fallback_parts[:30]).casefold()
    human_challenge = bool(
        fallback_parts and fallback_parts[0].casefold() == "verify to continue"
        and re.search(r"\b(?:confirm|verify)\s+(?:that\s+)?you(?:\s+are|['’]re)\s+(?:a\s+)?human\b", sample)
    )
    soft_error = None
    # HTTP 200 也可能实际展示错误页，所以从可见文字再做一次软错误判断。
    if any(marker in sample for marker in ("page not found", "404 not found", "content is no longer available", "页面不存在", "内容已删除")):
        soft_error = "NOT_FOUND"
    elif human_challenge or any(marker in sample for marker in ("sign in to continue", "please log in", "verify you are human", "captcha", "访问验证", "登录后查看")):
        soft_error = "ACCESS_RESTRICTED"
    return ExtractedHtml(" ".join(parser.title_parts) or None, body, paragraphs, tuple(dict.fromkeys(parser.links)), soft_error)


def _merge_article_sections(source: str, extracted: str, primary: str) -> str:
    """保留后备算法替换正文时遗漏的段落，仅合入能定位到原页面的文字。"""

    if not extracted or not primary or extracted == primary:
        return extracted or primary
    try:
        tree = html.document_fromstring(source, parser=html.HTMLParser(no_network=True))
    except (etree.ParserError, ValueError):
        return extracted
    body = tree.find("body")
    if body is None:
        return extracted
    for node in body.xpath(".//script|.//style|.//noscript|.//svg"):
        node.drop_tree()

    def normalize(value: str, *, markdown: bool = False) -> str:
        if markdown:
            value = re.sub(r"\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)", r"\1", value)
        return "".join(value.split())

    source_text = normalize(body.text_content())
    lines = [line.strip() for line in extracted.splitlines() if line.strip()]
    keys = [normalize(line, markdown=True) for line in lines]
    positions = [source_text.find(key) if key else -1 for key in keys]
    for line in primary.splitlines():
        line = line.strip()
        key = normalize(line, markdown=True)
        if not key or key in "".join(keys):
            continue
        position = source_text.find(key)
        if position < 0:
            continue
        index = next((i for i, other in enumerate(positions) if other > position), len(lines))
        lines.insert(index, line)
        keys.insert(index, key)
        positions.insert(index, position)
    return "\n".join(lines)
