"""根据目标输出类型选择合适的 SourceOpener。"""

from __future__ import annotations

from ..contracts import CandidateSource, OpenedSource, SearchFailure, SearchRequest
from .document import DocumentSourceOpener
from .image import ImageSourceOpener
from .video import VideoSourceOpener


class SourceDispatcher:
    """能力路由器，使 Service 编排层不包含具体站点或媒体规则。"""

    def __init__(self, document: DocumentSourceOpener, image: ImageSourceOpener, video: VideoSourceOpener) -> None:
        """注入三类打开器。"""

        self.document, self.image, self.video = document, image, video

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        """图片走图片链、视频优先视频链，其余走文档链。"""

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
