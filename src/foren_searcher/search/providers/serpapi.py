"""SerpAPI SDK 适配器；Provider 特有 JSON 到本层为止。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest


class SerpApiProvider:
    """把 SerpAPI 的多个搜索引擎统一成 SearchProvider 接口。"""

    def __init__(self, api_key: str, timeout_s: float = 30.0, client: Any | None = None) -> None:
        """校验密钥；未注入测试客户端时延迟导入可选 SDK。"""

        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if client is None:
            try:
                import serpapi
            except ImportError as exc:
                raise RuntimeError("SerpApi support requires the 'serpapi' extra") from exc
            client_type = getattr(serpapi, "Client", None)
            if client_type is None:
                raise RuntimeError("installed serpapi package does not expose Client")
            client = client_type(api_key=api_key, timeout=timeout_s)
        self._client = client

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int, image_path: Path | None = None) -> CandidateBatch:
        """选择引擎，必要时上传图片，再把结果列表规范化。"""

        engine, params, result_key = self._route(request, offset, limit)
        request_metadata: dict[str, Any] = {}
        try:
            if request.input_type in {"image", "text_image"}:
                # Google Lens 路径先上传本地 Artifact，搜索参数只携带 image_id。
                if image_path is None:
                    raise SearchFailure(SearchError("upload", "INVALID_INPUT", "image query has no readable image artifact"))
                upload = await asyncio.to_thread(self._client.upload_image, str(image_path))
                if not isinstance(upload, Mapping) or not upload.get("image_id"):
                    raise SearchFailure(SearchError("upload", "PROVIDER_ERROR", "image upload returned no image_id"))
                params["image_id"] = str(upload["image_id"])
                request_metadata["image_id"] = str(upload["image_id"])
            # SDK 是同步接口，放入工作线程避免阻塞 asyncio 事件循环。
            raw = await asyncio.to_thread(self._client.search, params)
        except SearchFailure:
            raise
        except Exception as exc:
            raise self._map_exception(exc) from exc
        if not isinstance(raw, Mapping):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "provider returned a non-object response"))
        if raw.get("error"):
            raise self._map_error_message(str(raw["error"]))
        items = raw.get(result_key, [])
        if not isinstance(items, list):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"provider field {result_key} is not a list"))
        candidates = tuple(item for rank, value in enumerate(items[:limit], start=offset + 1) if (item := self._normalize(value, request.output_type, rank)))
        return CandidateBatch("serpapi", engine, candidates, dict(raw), request_metadata)

    @staticmethod
    def _route(request: SearchQueryRequest, offset: int, limit: int) -> tuple[str, dict[str, object], str]:
        """把输入/输出组合映射到 SerpAPI engine、参数和结果字段名。"""

        if request.input_type in {"image", "text_image"}:
            params: dict[str, object] = {"engine": "google_lens", "num": limit, "start": offset}
            if request.input_type == "text_image":
                params["q"] = request.query["text"]  # type: ignore[index]
            return "google_lens", params, "visual_matches"
        query = str(request.query)
        if request.output_type == "image":
            return "google_images", {"engine": "google_images", "q": query, "ijn": offset // max(limit, 1)}, "images_results"
        if request.output_type == "video":
            return "google_videos", {"engine": "google_videos", "q": query, "start": offset, "num": limit}, "video_results"
        # 首屏文本用 Google；后续页用 offset 表达更直接的 Bing。
        if offset:
            return "bing", {"engine": "bing", "q": query, "first": offset + 1, "count": limit}, "organic_results"
        return "google", {"engine": "google", "q": query, "start": offset, "num": limit}, "organic_results"

    @staticmethod
    def _normalize(value: Any, output_type: str, rank: int) -> CandidateSource | None:
        """丢弃无效条目，并统一图片、网页和视频候选字段。"""

        if not isinstance(value, Mapping):
            return None
        if output_type == "image":
            image_url = _http_url(value.get("original") or value.get("image") or value.get("link"))
            page_url = _http_url(value.get("link") or value.get("source")) or image_url
            if image_url is None or page_url is None:
                return None
            return CandidateSource(page_url, _text(value.get("title")), _text(value.get("snippet")),
                "serpapi", "google_images", rank, "image", image_url,
                _http_url(value.get("thumbnail")), extra=dict(value))
        url = _http_url(value.get("link") or value.get("url") or value.get("source"))
        if url is None:
            return None
        has_video_identity = any(value.get(key) is not None for key in ("video_id", "duration", "channel", "platform"))
        candidate_type: Literal["video", "page"] = "video" if _looks_like_video(url) or (output_type == "video" and has_video_identity) else "page"
        image_url = _http_url(value.get("image") or value.get("thumbnail")) if output_type != "text" else None
        return CandidateSource(url, _text(value.get("title")), _text(value.get("snippet")),
            "serpapi", "google_videos" if output_type == "video" else "google_lens" if value.get("source") else "google",
            rank, candidate_type, image_url, _http_url(value.get("thumbnail")),
            platform=_platform(url), platform_video_id=_video_id(value, url), extra=dict(value))

    @staticmethod
    def _map_exception(exc: Exception) -> SearchFailure:
        """把 SDK 异常的状态码和消息映射成稳定错误类别。"""

        message = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        if "timeout" in message:
            return SearchFailure(SearchError("search", "TIMEOUT", "search provider timed out"))
        if status_code == 429 and ("quota" in message or "credit" in message):
            return SearchFailure(SearchError("search", "QUOTA_EXCEEDED", "search provider quota is exhausted"))
        if status_code == 429 or "429" in message or "rate" in message:
            return SearchFailure(SearchError("search", "RATE_LIMITED", "search provider rate limited the request"))
        if status_code in {401, 403} or "401" in message or "403" in message or "api key" in message:
            return SearchFailure(SearchError("search", "AUTH_ERROR", "search provider rejected authentication"))
        return SearchFailure(SearchError("search", "PROVIDER_ERROR", f"search provider failed ({type(exc).__name__})"))

    @staticmethod
    def _map_error_message(message: str) -> SearchFailure:
        """处理 SerpAPI 在成功响应 JSON 中返回的 error 消息。"""

        lowered = message.lower()
        code = "QUOTA_EXCEEDED" if "quota" in lowered or "credit" in lowered else "AUTH_ERROR" if "key" in lowered or "auth" in lowered else "RATE_LIMITED" if "rate" in lowered else "PROVIDER_ERROR"
        return SearchFailure(SearchError("search", code, "search provider rejected the request"))  # type: ignore[arg-type]


def _text(value: Any) -> str | None:
    """返回清理后的非空文本。"""

    return value.strip() if isinstance(value, str) and value.strip() else None


def _http_url(value: Any) -> str | None:
    """只保留语法有效且不内嵌凭据的 HTTP(S) URL。"""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    try:
        parsed = urlsplit(cleaned)
        _ = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    return cleaned


def _looks_like_video(url: str) -> bool:
    """按常见平台 URL 判断是否是视频候选。"""

    return any(host in url.lower() for host in ("youtube.com/", "youtu.be/", "vimeo.com/", "bilibili.com/video/", "tiktok.com/"))


def _platform(url: str) -> str | None:
    """从 URL 提取受支持的视频平台标识。"""

    lowered = url.lower()
    for key in ("youtube", "vimeo", "bilibili", "tiktok"):
        if key in lowered or (key == "youtube" and "youtu.be" in lowered):
            return key
    return None


def _video_id(value: Mapping[str, Any], url: str) -> str | None:
    """读取 Provider 给出的稳定视频 ID；url 参数保留用于统一调用形状。"""

    candidate = value.get("video_id") or value.get("id")
    return str(candidate) if candidate else None
