"""SEARCH Provider 的最小接口契约。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..contracts import CandidateBatch, SearchQueryRequest


class SearchProvider(Protocol):
    """任何文本/反向图片搜索适配器都需实现的异步接口。"""

    async def search(
        self, request: SearchQueryRequest, *, offset: int, limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """从 offset 开始最多返回 limit 个规范化候选。"""

        ...
