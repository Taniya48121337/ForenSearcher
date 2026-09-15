"""依据输入能力把稳定 SEARCH 请求分发给不同 Provider。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import CandidateBatch, SearchQueryRequest
from .base import SearchProvider


class InputRoutingSearchProvider:
    """组合文本 Provider 与反向图片 Provider，向 Service 暴露单一接口。"""

    def __init__(self, *, text_provider: SearchProvider, image_provider: SearchProvider) -> None:
        """保存两条路由；它们可以指向同一个复合 Provider。"""

        self._text_provider = text_provider
        self._image_provider = image_provider

    async def search(
        self,
        request: SearchQueryRequest,
        *,
        offset: int,
        limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """纯文本走 text_provider，其余含图输入走 image_provider。"""

        provider = self._text_provider if request.input_type == "text" else self._image_provider
        return await provider.search(request, offset=offset, limit=limit, image_path=image_path)

    async def aclose(self) -> None:
        """关闭底层 Provider，并避免同一实例被关闭两次。"""

        closed: set[int] = set()
        for provider in (self._text_provider, self._image_provider):
            if id(provider) in closed:
                continue
            closed.add(id(provider))
            close: Any = getattr(provider, "aclose", None)
            if callable(close):
                await close()
