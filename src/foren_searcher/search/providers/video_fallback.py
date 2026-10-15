"""文字搜视频优先走 yt-dlp 的 YouTube 搜索，失败或空结果时退回原 provider（DDGS）（2026-10-05）。

为什么：EVID 冒烟 evidb2 里 HI 的视频查询已经写对（字幕原文、事件关键词），但 DDGS 的视频搜索 21 次全部
PROVIDER_ERROR（DDGSException），原片因此一次都没搜到。DDGS 视频只有 duckduckgo 一个后端、没有备用；
yt-dlp 的 ytsearch 直接查 YouTube（旧 plan-harness-1 流水线就是这样找原片的），不需要密钥。

只改"文字 → 视频"这一种请求；文字搜网页 / 图片、以图搜图仍走原 provider。每次走了哪条路记在 stats 里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import SearchFailure


class VideoSearchFallbackProvider:
    def __init__(self, video_provider: Any, fallback: Any, *, explain_empty: bool = False) -> None:
        self.video_provider, self.fallback = video_provider, fallback
        # 10-06 找原片实验 guardlen：YouTube 零结果、DDGS 又失败或也空时，不再把 DDGS 的 PROVIDER_ERROR 交给 HI
        # （它看不出是查询太长），改报 NOT_FOUND 并说明怎么缩短。默认关，与以前完全相同。
        self.explain_empty = explain_empty
        self.stats = {"video_primary_ok": 0, "video_primary_empty": 0, "video_primary_failed": 0, "video_fallback_used": 0}
        if explain_empty:
            self.stats["video_empty_explained"] = 0
        self.last_primary_error: str | None = None

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> Any:
        if getattr(request, "input_type", None) != "text" or getattr(request, "output_type", None) != "video":
            return await self.fallback.search(request, offset=offset, limit=limit, image_path=image_path)
        try:
            batch = await self.video_provider.search(request, offset=offset, limit=limit, image_path=image_path)
        except SearchFailure as exc:
            self.stats["video_primary_failed"] += 1
            self.last_primary_error = f"{exc.error.code}: {exc.error.message}"
        else:
            if batch.candidates:
                self.stats["video_primary_ok"] += 1
                return batch
            self.stats["video_primary_empty"] += 1
            if self.explain_empty:
                return await self._fallback_or_explain(request, offset=offset, limit=limit, image_path=image_path)
        self.stats["video_fallback_used"] += 1
        return await self.fallback.search(request, offset=offset, limit=limit, image_path=image_path)

    async def _fallback_or_explain(self, request: Any, *, offset: int, limit: int, image_path: Path | None) -> Any:
        from ..arms import NO_RESULTS
        from ..contracts import SearchError

        self.stats["video_fallback_used"] += 1
        try:
            batch = await self.fallback.search(request, offset=offset, limit=limit, image_path=image_path)
        except SearchFailure:
            batch = None
        if batch is not None and batch.candidates:
            return batch
        self.stats["video_empty_explained"] += 1
        raise SearchFailure(SearchError("search", "NOT_FOUND", NO_RESULTS))

    async def aclose(self) -> None:
        for provider in (self.video_provider, self.fallback):
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()

    def summary(self) -> dict[str, Any]:
        return {"mode": "ytdlp_then_ddgs", **self.stats, "last_primary_error": self.last_primary_error}


__all__ = ["VideoSearchFallbackProvider"]
