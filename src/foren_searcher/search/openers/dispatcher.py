"""根据目标输出类型选择合适的 SourceOpener。"""

from __future__ import annotations

from typing import Any

from ..contracts import CandidateSource, OpenedSource, SearchError, SearchFailure, SearchRequest
from .document import DocumentSourceOpener
from .image import ImageSourceOpener
from .video import VideoSourceOpener


class SourceDispatcher:
    """能力路由器，使 Service 编排层不包含具体站点或媒体规则。"""

    def __init__(self, document: DocumentSourceOpener, image: ImageSourceOpener, video: VideoSourceOpener,
                 dates: Any | None = None) -> None:
        """注入三类打开器；dates（10-13，openers/dates.py）处理 include=["dates"] 的日期查询。"""

        self.document, self.image, self.video, self.dates = document, image, video, dates

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        """图片走图片链、视频优先视频链，其余走文档链；日期查询走日期链。"""

        if getattr(request, "include", ()) == ("dates",):
            if self.dates is None:
                raise SearchFailure(SearchError("validate", "INVALID_INPUT", "date lookup is not available in this deployment"))
            return await self.dates.open(candidate, request)
        if request.output_type == "image":
            return await self.image.open(candidate, request)
        if request.output_type == "video":
            try:
                return await self.video.open(candidate, request)
            except SearchFailure:
                # 视频搜索也可能返回普通介绍页；此时保存页面但不冒充视频成功。
                if candidate.candidate_type == "video":
                    raise
                return await self.document.open(candidate, request)
        return await self.document.open(candidate, request)
