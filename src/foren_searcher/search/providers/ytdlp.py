"""使用 yt-dlp 公共搜索的可选“文字 → YouTube 视频”Provider。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ..openers.http import SafeHttpClient
from ..openers.ytdlp_http import extract_metadata
from ..policy import SearchPolicy


class YtDlpSearchProvider:
    """把 yt-dlp 的同步搜索结果适配成异步 CandidateBatch。"""

    def __init__(self, youtube_dl_factory: Callable[[Mapping[str, Any]], Any] | None = None, timeout_s: float = 20.0) -> None:
        """允许注入工厂方便测试；生产环境默认导入 yt_dlp。"""

        self._factory = youtube_dl_factory
        self._timeout_s = timeout_s

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int, image_path: Path | None = None) -> CandidateBatch:
        """只接受文字搜视频，并在线程中执行阻塞的 yt-dlp。"""

        if request.input_type != "text" or request.output_type != "video":
            raise SearchFailure(SearchError("search", "INVALID_INPUT", "yt-dlp provider supports only text-to-video queries"))
        policy = SearchPolicy(source_timeout_s=self._timeout_s)
        http = SafeHttpClient(policy)
        try:
            options = {"extract_flat": "in_playlist", "ignoreconfig": True, "playlistend": offset + limit,
                       "quiet": True, "skip_download": True, "socket_timeout": self._timeout_s, "no_warnings": True}
            raw = await extract_metadata(http, policy, self._factory, options, f"ytsearch{offset + limit}:{request.query}")
        except SearchFailure:
            raise
        except Exception as exc:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"video search failed ({type(exc).__name__})")) from exc
        finally:
            await http.aclose()
        entries = raw.get("entries", []) if isinstance(raw, Mapping) else []
        candidates: list[CandidateSource] = []
        seen: set[str] = set()
        for rank, item in enumerate(entries[offset:offset + limit], start=offset + 1):
            # Provider 返回可能含 None 或缺字段条目，逐条跳过而不是让整批失败。
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not isinstance(item.get("title"), str):
                continue
            video_id = item["id"]
            if video_id in seen:
                continue
            seen.add(video_id)
            candidates.append(CandidateSource(
                f"https://www.youtube.com/watch?v={video_id}", item["title"], None,
                "yt-dlp", "youtube_search", rank, "video", thumbnail_url=item.get("thumbnail"),
                platform="youtube", platform_video_id=video_id,
                extra={"channel": item.get("channel") or item.get("uploader"), "duration": item.get("duration")},
            ))
        return CandidateBatch("yt-dlp", "youtube_search", tuple(candidates), dict(raw) if isinstance(raw, Mapping) else {})
