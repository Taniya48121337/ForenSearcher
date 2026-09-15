"""免费 SearXNG Provider；实例特有 JSON 在这一层被转换掉。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded


class SearXngProvider:
    """使用指定 SearXNG JSON API 完成由文本发起的搜索。"""

    _PAGE_SIZE = 10

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 30.0,
        *,
        language: str = "all",
        safe_search: int = 1,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        """验证实例 URL、安全搜索等级和响应大小，并准备客户端。"""

        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        if safe_search not in {0, 1, 2}:
            raise ValueError("safe_search must be 0, 1, or 2")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self._base_url = base_url.rstrip("/") + "/"
        self._language = language
        self._safe_search = safe_search
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "ForenSearcher/0.1", **dict(headers or {})},
        )

    async def aclose(self) -> None:
        """只关闭本 Provider 自己创建的 HTTP 客户端。"""

        if self._owns_client:
            await self._client.aclose()

    async def search(
        self,
        request: SearchQueryRequest,
        *,
        offset: int,
        limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """把稳定请求映射到 SearXNG 分类和页码，然后规范化结果。"""

        del image_path
        if request.input_type != "text":
            raise SearchFailure(SearchError(
                "search", "INVALID_INPUT",
                "SearXNG supports text queries only; it cannot upload an image for reverse search",
            ))
        category = {"text": "general", "image": "images", "video": "videos"}[request.output_type]
        # SearXNG 使用页码分页；Service 使用 offset/limit，所以需要换算并切片。
        page_number = offset // self._PAGE_SIZE + 1
        page_offset = offset % self._PAGE_SIZE
        params: dict[str, str | int] = {
            "q": str(request.query),
            "format": "json",
            "categories": category,
            "language": self._language,
            "safesearch": self._safe_search,
            "pageno": page_number,
        }
        endpoint = urljoin(self._base_url, "search")
        raw = await self._request(endpoint, params)
        values = raw.get("results", [])
        if not isinstance(values, list):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "SearXNG field results is not a list"))
        selected = values[page_offset:page_offset + limit]
        candidates = tuple(
            item
            for rank, value in enumerate(selected, start=offset + 1)
            if (
                item := self._normalize(
                    value,
                    request.output_type,
                    rank,
                    query=str(request.query),
                )
            ) is not None
        )
        return CandidateBatch(
            "searxng",
            category,
            candidates,
            raw,
            {"instance": self._base_url, "category": category, "pageno": page_number},
        )

    async def _request(self, endpoint: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        """执行有界请求，并把 HTTP/JSON 失败映射成稳定 SearchError。"""

        try:
            response = await request_bounded(
                self._client, "GET", endpoint, params=params,
                maximum_bytes=self._max_response_bytes, provider_name="SearXNG",
            )
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("search", "TIMEOUT", "SearXNG request timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("search", "NETWORK_ERROR", "SearXNG request failed at the network boundary")) from exc

        # Retry-After 会随 RATE_LIMITED 错误返回，供上层决定等待时间。
        retry_after = _retry_after(response.headers.get("Retry-After"))
        if response.status_code in {401, 407}:
            raise SearchFailure(SearchError("search", "AUTH_ERROR", "SearXNG rejected authentication"))
        if response.status_code == 403:
            raise SearchFailure(SearchError(
                "search", "ACCESS_RESTRICTED", "SearXNG instance denied JSON API access",
            ))
        if response.status_code == 429:
            raise SearchFailure(
                SearchError("search", "RATE_LIMITED", "SearXNG instance rate limited the request"),
                retry_after_s=retry_after,
            )
        if response.status_code >= 400:
            raise SearchFailure(SearchError(
                "search", "PROVIDER_ERROR", f"SearXNG returned HTTP {response.status_code}",
            ))
        try:
            value = response.json()
        except ValueError as exc:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "SearXNG returned invalid JSON")) from exc
        if not isinstance(value, dict):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "SearXNG returned a non-object response"))
        return value

    @staticmethod
    def _normalize(
        value: Any, output_type: str, rank: int, *, query: str
    ) -> CandidateSource | None:
        """过滤无效 Provider 条目，并按目标类型构造 CandidateSource。"""

        if not isinstance(value, Mapping):
            return None
        source_url = _http_url(value.get("url"))
        if source_url is None:
            return None
        engine = _engine(value)
        title = _text(value.get("title"))
        snippet = _text(value.get("content"))
        if output_type == "text" and _looks_like_search_navigation(
            source_url, title, query
        ):
            return None
        thumbnail = _http_url(value.get("thumbnail_src")) or _http_url(value.get("thumbnail"))
        extra = {key: value[key] for key in (
            "publishedDate", "score", "category", "template", "source", "iframe_src",
        ) if key in value}

        # 图片结果的 source_url 是页面，image_url 才是原图地址。
        if output_type == "image":
            image_url = _http_url(value.get("img_src")) or _http_url(value.get("image_src"))
            if image_url is None:
                return None
            return CandidateSource(
                source_url, title, snippet, "searxng", engine, rank, "image",
                image_url, thumbnail, extra=extra,
            )

        candidate_type: Literal["page", "document", "video"] = "page"
        platform = None
        platform_video_id = None
        if output_type == "video" or _looks_like_video(source_url, value):
            candidate_type = "video"
            platform = _platform(source_url)
            platform_video_id = _video_id(source_url, value)
        elif urlparse(source_url).path.lower().endswith(".pdf"):
            candidate_type = "document"
        return CandidateSource(
            source_url, title, snippet, "searxng", engine, rank, candidate_type,
            thumbnail if output_type == "video" else None, thumbnail,
            platform=platform, platform_video_id=platform_video_id, extra=extra,
        )


def _text(value: Any) -> str | None:
    """把非空字符串清理后返回，其余值统一视为空。"""

    return value.strip() if isinstance(value, str) and value.strip() else None


def _http_url(value: Any) -> str | None:
    """清理并验证无凭据的 HTTP(S) URL。"""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    try:
        parsed = urlparse(cleaned)
        _ = parsed.port
    except ValueError:
        return None
    return cleaned if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else None


def _engine(value: Mapping[str, Any]) -> str:
    """读取单个 engine；没有时合并 engines 列表作为来源说明。"""

    engine = _text(value.get("engine"))
    if engine:
        return engine
    engines = value.get("engines")
    if isinstance(engines, list):
        names = [item.strip() for item in engines if isinstance(item, str) and item.strip()]
        if names:
            return ",".join(names)
    return "searxng"


def _looks_like_video(url: str, value: Mapping[str, Any]) -> bool:
    """依据模板名和常见视频站点判断结果是否为视频。"""

    template = str(value.get("template", "")).lower()
    lowered = url.lower()
    return "video" in template or any(host in lowered for host in (
        "youtube.com/", "youtu.be/", "vimeo.com/", "bilibili.com/video/",
        "tiktok.com/", "dailymotion.com/video/",
    ))


def _looks_like_search_navigation(
    url: str, title: str | None, query: str
) -> bool:
    """过滤搜索引擎首页、登录页和地图入口，除非任务确实在查该品牌。"""

    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/").casefold()
    normalized_query = query.casefold()
    normalized_title = (title or "").strip().casefold()
    if "google" in host and "google" not in normalized_query:
        return (
            host.startswith(("accounts.", "maps."))
            or path in {"", "/index.html", "/webhp", "/maps"}
            or normalized_title in {"google", "google maps", "sign in - google accounts"}
        )
    if "bing.com" in host and "bing" not in normalized_query:
        return path in {"", "/search"} and normalized_title in {"", "bing"}
    return False


def _platform(url: str) -> str | None:
    """从 URL 识别少量支持的平台名称。"""

    lowered = url.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    for name in ("vimeo", "bilibili", "tiktok", "dailymotion"):
        if name in lowered:
            return name
    return None


def _video_id(url: str, value: Mapping[str, Any]) -> str | None:
    """优先取显式视频 ID，否则尝试解析 YouTube URL。"""

    explicit = value.get("video_id") or value.get("id")
    if explicit is not None:
        return str(explicit)
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc.lower():
        return parsed.path.strip("/").split("/")[0] or None
    if "youtube.com" in parsed.netloc.lower():
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


def _retry_after(value: str | None) -> float | None:
    """把 Retry-After 秒数解析成非负浮点数。"""

    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
