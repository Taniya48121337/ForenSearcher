"""图片候选的原图、来源页面与缩略图获取。"""

from __future__ import annotations

import io
import json

from PIL import Image, UnidentifiedImageError

from ..contracts import CandidateSource, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchRequest
from ..policy import SearchPolicy
from .document import DocumentSourceOpener
from .http import SafeHttpClient


class ImageSourceOpener:
    """按固定三个材料槽位打开图片 Candidate。"""

    def __init__(self, http: SafeHttpClient, document_opener: DocumentSourceOpener, policy: SearchPolicy | None = None) -> None:
        """共享安全 HTTP 和文档打开器。"""

        self.http, self.document_opener, self.policy = http, document_opener, policy or http.policy

    def supports(self, candidate: CandidateSource, output_type: str) -> bool:
        """本打开器只服务 image 输出。"""

        return output_type == "image"

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        """依次取得原图、页面和缩略图；单项失败不会抹掉其他成功材料。"""

        values: list[OpenedMaterial] = []
        final_url = candidate.source_url
        # Service 把已有缓存种类写入内部 extra，避免再次下载相同材料。
        cached_materials = set(candidate.extra.get("_search_cached_materials", ()))
        image_url = candidate.image_url or (candidate.source_url if candidate.candidate_type == "image" else None)
        source_was_not_an_image = False
        if "image" in cached_materials:
            values.append(OpenedMaterial("image", "not_requested"))
        elif not image_url and candidate.candidate_type in {"page", "document", "unknown"}:
            # r1（10-17，负责人定 (a)）：文字搜到的网页没有引擎报的原图——取页内主图（og:image → twitter:image → 正文第一张大图）
            values.append(await self._page_main_image(candidate))
        elif image_url:
            try:
                resource = await self.http.fetch(image_url, maximum_bytes=self.policy.maximum_image_bytes)
                try:
                    # Pillow 完整解码用于确认下载不是伪装图片或截断文件。
                    with Image.open(io.BytesIO(resource.content)) as image:
                        image.load()
                        width, height, image_format = image.width, image.height, image.format
                except (UnidentifiedImageError, OSError) as exc:
                    raise SearchFailure(SearchError("extract", "UNSUPPORTED_FORMAT", "downloaded source is not a complete supported image")) from exc
                media_type = Image.MIME.get(image_format or "", resource.media_type)
                suffix = "." + (image_format or "bin").lower().replace("jpeg", "jpg")
                content = json.dumps({"width": width, "height": height, "format": image_format, "source_url": resource.final_url}, ensure_ascii=False)
                values.append(OpenedMaterial("image", "ready", content, resource.content, media_type, suffix,
                    ("image",), candidate.search_title or resource.final_url, "complete image",
                    "SEARCH/open/image+Pillow", "source_material"))
            except SearchFailure as exc:
                source_was_not_an_image = image_url == candidate.source_url and exc.error.code == "UNSUPPORTED_FORMAT"
                if source_was_not_an_image:
                    # r1（10-17 负责人定 (ii)）：第 2 轮按网址 SEARCH {"source": <页面网址>, "output_type": "image"} 取页内主图——
                    # 网址其实是网页，就按网页取它的主图
                    values.append(await self._page_main_image(candidate))
                else:
                    values.append(OpenedMaterial("image", "failed", error=exc.error))
        else:
            values.append(OpenedMaterial("image", "absent"))
        if "page" in cached_materials:
            values.append(OpenedMaterial("page", "not_requested"))
        elif candidate.source_url != image_url or source_was_not_an_image:
            # 原图 URL 与来源页不同，或直接 URL 其实不是图片时，再尝试抽取页面。
            try:
                page = await self.document_opener.open(candidate, request)
                final_url = page.final_url
                values.extend(page.materials)
            except SearchFailure as exc:
                values.append(OpenedMaterial("page", "failed", error=exc.error))
        else:
            values.append(OpenedMaterial("page", "not_requested"))
        if "thumbnail" in cached_materials:
            values.append(OpenedMaterial("thumbnail", "not_requested"))
        elif candidate.thumbnail_url and candidate.thumbnail_url != image_url:
            try:
                resource = await self.http.fetch(candidate.thumbnail_url, maximum_bytes=self.policy.maximum_image_bytes)
                with Image.open(io.BytesIO(resource.content)) as image:
                    image.load()
                    width, height, image_format = image.width, image.height, image.format
                values.append(OpenedMaterial("thumbnail", "ready", json.dumps({"width": width, "height": height, "format": image_format}),
                    resource.content, Image.MIME.get(image_format or "", resource.media_type), "." + (image_format or "bin").lower(),
                    ("image",), "search result thumbnail", "complete thumbnail", "SEARCH/open/image+Pillow", "source_material"))
            except Exception as exc:
                error = exc.error if isinstance(exc, SearchFailure) else SearchError("extract", "UNSUPPORTED_FORMAT", "thumbnail is not a supported image")
                values.append(OpenedMaterial("thumbnail", "failed", error=error))
        else:
            values.append(OpenedMaterial("thumbnail", "not_requested"))
        return OpenedSource(final_url, tuple(values))

    async def _page_main_image(self, candidate: CandidateSource) -> OpenedMaterial:
        """页内主图：按 og:image、twitter:image、link rel=image_src、正文第一张大图（≥ 200×150 像素）的顺序取第一张能下载、
        能完整解码的图。材料 content 记图片网址、所在页面、抽取方式；取不到记 absent / failed。"""

        try:
            page = await self.http.fetch(candidate.source_url, maximum_bytes=self.policy.maximum_document_bytes)
        except SearchFailure as exc:
            return OpenedMaterial("image", "failed", error=exc.error)
        if not (page.media_type in {"text/html", "application/xhtml+xml"}
                or page.content.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html"))):
            return OpenedMaterial("image", "absent")
        tried = 0
        last_error: SearchError | None = None
        attempts: list[tuple[str, str]] = []
        for url, method in page_image_candidates(page.content, page.final_url)[:4]:
            attempts.append((url, method))
            # r1：图片网址取不到（r1s2 l623：BuzzFeed News 停刊后图片主机解析不了）再试互联网档案馆的原样存档
            attempts.append((ARCHIVE_RAW + url, method + " (archived)"))
        archived_for: set[str] = set()
        for url, method in attempts:
            if tried >= 8:
                break
            if url.startswith(ARCHIVE_RAW) and url[len(ARCHIVE_RAW):] not in archived_for:
                continue                      # 原网址取到了（或图太小），不再取存档
            tried += 1
            try:
                resource = await self._fetch_with_retry(url)
                with Image.open(io.BytesIO(resource.content)) as image:
                    image.load()
                    width, height, image_format = image.width, image.height, image.format
            except SearchFailure as exc:
                last_error = exc.error
                if not url.startswith(ARCHIVE_RAW):
                    archived_for.add(url)     # 原网址取不到：下一个就试它的存档
                continue
            except (UnidentifiedImageError, OSError):
                last_error = SearchError("extract", "UNSUPPORTED_FORMAT", "page image is not a complete supported image")
                continue
            if min(width, height) < 150 or max(width, height) < 200:
                last_error = SearchError("extract", "EXTRACT_FAILED", "page image is too small to be the main picture")
                continue
            content = json.dumps({"width": width, "height": height, "format": image_format, "source_url": resource.final_url,
                                  "page_url": page.final_url, "extraction": method}, ensure_ascii=False)
            label = f"main image of the page ({method}): {candidate.search_title or page.final_url}"[:300]
            return OpenedMaterial("image", "ready", content, resource.content, Image.MIME.get(image_format or "", resource.media_type),
                                  "." + (image_format or "bin").lower().replace("jpeg", "jpg"), ("image",), label,
                                  f"page main image ({method})",
                                  f"SEARCH/open/page_main_image:{method.replace(' (archived)', '+archive')}+Pillow", "source_material")
        if last_error is not None:
            return OpenedMaterial("image", "failed", error=last_error)
        return OpenedMaterial("image", "absent")

    async def _fetch_with_retry(self, url: str):
        """页内主图下载：网络 / 超时 / 限流类失败按 1、2、4 秒退避，共试 3 次（负责人定：服务器网络和本机不同，重试无害）。
        内容类失败（404、不是图片）不重试。"""

        import asyncio

        last: SearchFailure | None = None
        for attempt in range(PAGE_IMAGE_ATTEMPTS):
            try:
                return await self.http.fetch(url, maximum_bytes=self.policy.maximum_image_bytes)
            except SearchFailure as exc:
                last = exc
                client_error = str(exc.error.message or "").startswith("source returned HTTP 4")
                if exc.error.code not in _RETRYABLE_FETCH or client_error or attempt == PAGE_IMAGE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        assert last is not None
        raise last


PAGE_IMAGE_ATTEMPTS = 3
_RETRYABLE_FETCH = {"NETWORK_ERROR", "TIMEOUT", "RATE_LIMITED", "PROVIDER_ERROR", "HTTP_ERROR"}
ARCHIVE_RAW = "https://web.archive.org/web/2id_/"     # 档案馆原样返回最近一次存档的字节（id_ = 不改写）
_SKIP_IMAGE = ("logo", "icon", "sprite", "avatar", "pixel", "spacer", "badge", "/ads/", "doubleclick", "gravatar", "emoji",
               "placeholder", "blank.", "loading")


def page_image_candidates(html: bytes, base_url: str) -> list[tuple[str, str]]:
    """页内主图的候选网址（按优先级，去重）：og:image（含 og:image:secure_url / og:image:url）→ twitter:image → link rel=image_src
    → 正文（article / main / body）里第一批 <img>（跳过 data:、svg、logo / icon / 头像之类；声明了宽高且小于 200 的跳过）。"""

    from urllib.parse import urljoin

    import lxml.html

    try:
        doc = lxml.html.fromstring(html)
    except (ValueError, TypeError, lxml.etree.ParserError):  # type: ignore[attr-defined]
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(raw: str | None, method: str) -> None:
        raw = (raw or "").strip()
        if not raw or raw.startswith("data:"):
            return
        url = urljoin(base_url, raw)
        low = url.lower()
        if not low.startswith(("http://", "https://")) or low.split("?", 1)[0].endswith(".svg") or url in seen:
            return
        if method == "first_large_img" and any(s in low for s in _SKIP_IMAGE):
            return
        seen.add(url)
        out.append((url, method))

    for prop in ("og:image:secure_url", "og:image:url", "og:image"):
        for node in doc.xpath(f'//meta[@property="{prop}" or @name="{prop}"]'):
            add(node.get("content"), "og:image")
    for node in doc.xpath('//meta[@name="twitter:image" or @property="twitter:image" or @name="twitter:image:src"]'):
        add(node.get("content"), "twitter:image")
    for node in doc.xpath('//link[@rel="image_src"]'):
        add(node.get("href"), "image_src")
    scope = (doc.xpath("//article") or doc.xpath("//main") or doc.xpath("//body") or [doc])[0]
    for node in scope.iter("img"):
        dims = []
        for attr in ("width", "height"):
            value = str(node.get(attr) or "").strip().rstrip("px")
            dims.append(int(value) if value.isdigit() else None)
        if any(d is not None and d < 200 for d in dims):
            continue
        add(node.get("src") or node.get("data-src") or node.get("data-original") or node.get("data-lazy-src"), "first_large_img")
        if sum(1 for _, m in out if m == "first_large_img") >= 4:
            break
    return out
