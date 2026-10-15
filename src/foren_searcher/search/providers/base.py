"""SEARCH Provider 的最小接口契约。"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Protocol

from ..contracts import CandidateBatch, SearchQueryRequest


# 10-12：以图搜图时，查询图片 Evidence 自己的公开地址（SEARCH 找到的网络图片才有；本机材料的帧没有）。
# 只能按 URL 搜的引擎（Serper 的 Google Lens）靠它；SearchService 在调用 Provider 前设置，调用后复位。
QUERY_IMAGE_URL: ContextVar[str | None] = ContextVar("query_image_url", default=None)


class SearchProvider(Protocol):
    """任何文本/反向图片搜索适配器都需实现的异步接口。"""

    async def search(
        self, request: SearchQueryRequest, *, offset: int, limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """从 offset 开始最多返回 limit 个规范化候选。"""

        ...
