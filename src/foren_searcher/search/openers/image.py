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
                values.append(OpenedMaterial("image", "failed", error=exc.error))
                source_was_not_an_image = image_url == candidate.source_url and exc.error.code == "UNSUPPORTED_FORMAT"
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
