"""适配 Yandex 公共图片搜索流程的无密钥反向图片 Provider。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
from lxml import html

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded


class YandexReverseImageProvider:
    """上传查询图片，并规范化匹配的来源页面与原图。"""

    def __init__(
        self,
        timeout_s: float = 30.0,
        *,
        base_url: str = "https://yandex.com",
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        """验证 HTTPS 基址、响应上限，并准备受控 HTTP 客户端。"""

        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base_url must be an HTTPS URL without embedded credentials")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self._base_url = base_url.rstrip("/")
        self._search_url = f"{self._base_url}/images/search"
        # 手动跟随跳转时只允许回到这个主机，避免上传响应把客户端引向任意站点。
        self._allowed_host = parsed.hostname
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "ForenSearcher/0.1"},
        )
        # 一次上传结果可被后续 offset 翻页复用。
        self._cache_lock = asyncio.Lock()
        self._cached_key: tuple[str, str] | None = None
        self._cached_result: tuple[list[dict[str, Any]], str] | None = None

    async def aclose(self) -> None:
        """仅关闭本实例创建的客户端。"""

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
        """验证输入、复用图片加文本缓存，并返回指定候选切片。"""

        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError(
                "upload", "INVALID_INPUT", "Yandex reverse search requires one uploaded image Evidence",
            ))
        image_bytes = image_path.read_bytes()
        query_text = request.query["text"].strip() if request.input_type == "text_image" else ""  # type: ignore[index]
        cache_key = (str(image_path.resolve()), query_text)
        async with self._cache_lock:
            # 锁防止同一查询的并发翻页触发重复上传。
            reused = cache_key == self._cached_key and self._cached_result is not None
            if reused:
                sites, result_url = self._cached_result  # type: ignore[misc]
            else:
                sites, result_url = await self._upload(image_bytes, query_text)
                self._cached_key = cache_key
                self._cached_result = (sites, result_url)

        selected = sites[offset:offset + limit]
        candidates = tuple(
            candidate
            for rank, item in enumerate(selected, start=offset + 1)
            if (candidate := _normalize_yandex(item, request.output_type, rank)) is not None
        )
        return CandidateBatch(
            "yandex-reverse",
            "yandex-images",
            candidates,
            {"result_url": result_url, "result_count": len(sites), "results": selected},
            {
                "instance": self._base_url,
                "uploaded": not reused,
                "reused_provider_result": reused,
                "text_filter": bool(query_text),
            },
        )

    async def _upload(self, image_bytes: bytes, query_text: str) -> tuple[list[dict[str, Any]], str]:
        """执行 multipart 上传，并解析页面 data-state 中的站点列表。"""

        params: dict[str, str] = {"rpt": "imageview", "cbir_page": "sites"}
        if query_text:
            params["text"] = query_text
        try:
            response = await request_bounded(
                self._client, "POST", self._search_url,
                maximum_bytes=self._max_response_bytes, provider_name="Yandex", stage="upload",
                params=params,
                data={"prg": "1"},
                files={"upfile": ("query.jpg", image_bytes, "image/jpeg")},
            )
            response = await self._follow_same_host(response)
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("upload", "TIMEOUT", "Yandex reverse-image upload timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "Yandex reverse-image upload failed")) from exc
        _check_response(response, self._max_response_bytes, "Yandex")
        try:
            document = html.fromstring(response.content)
            # Yandex 将初始化数据作为 JSON 放在 ImagesApp 节点属性中。
            states = document.xpath('//div[starts-with(@id, "ImagesApp-")]/@data-state')
            state = json.loads(states[0])
            sites = state["initialState"]["cbirSites"]["sites"]
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SearchFailure(SearchError(
                "search", "PROVIDER_ERROR", "Yandex reverse-image result structure was not recognized",
            )) from exc
        if not isinstance(sites, list):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "Yandex returned an invalid result list"))
        return [dict(item) for item in sites if isinstance(item, Mapping)], str(response.url)

    async def _follow_same_host(self, response: httpx.Response) -> httpx.Response:
        """最多手动跟随三次同 HTTPS 主机跳转。"""

        for _ in range(3):
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("Location")
            target = urljoin(str(response.url), location or "")
            parsed = urlparse(target)
            if parsed.scheme != "https" or parsed.hostname != self._allowed_host:
                raise SearchFailure(SearchError("upload", "ACCESS_RESTRICTED", "Yandex redirected outside its configured host"))
            response = await request_bounded(
                self._client, "GET", target,
                maximum_bytes=self._max_response_bytes, provider_name="Yandex", stage="upload",
            )
        raise SearchFailure(SearchError("upload", "PROVIDER_ERROR", "Yandex exceeded the redirect limit"))


def _normalize_yandex(value: Mapping[str, Any], output_type: str, rank: int) -> CandidateSource | None:
    """把 Yandex 匹配记录转换成统一 CandidateSource。"""

    source_url = _http_url(value.get("url"))
    original = value.get("originalImage")
    thumb = value.get("thumb")
    image_url = _http_url(original.get("url")) if isinstance(original, Mapping) else None
    thumbnail_url = _http_url(thumb.get("url"), allow_protocol_relative=True) if isinstance(thumb, Mapping) else None
    if source_url is None or (output_type == "image" and image_url is None):
        return None
    candidate_type: Literal["image", "video", "page"] = "image" if output_type == "image" else "video" if _looks_like_video(source_url) else "page"
    extra: dict[str, Any] = {"domain": value.get("domain")}
    if isinstance(original, Mapping):
        extra["image_width"] = original.get("width")
        extra["image_height"] = original.get("height")
    return CandidateSource(
        source_url=source_url,
        search_title=_text(value.get("title")),
        search_snippet=_text(value.get("description")),
        provider="yandex-reverse",
        engine="yandex-images",
        provider_rank=rank,
        candidate_type=candidate_type,
        image_url=image_url if output_type == "image" else None,
        thumbnail_url=thumbnail_url,
        platform=_platform(source_url),
        extra=extra,
    )


def _check_response(response: httpx.Response, max_bytes: int, name: str) -> None:
    """把上传 HTTP 状态和兜底大小检查转换为 SearchFailure。"""

    status = response.status_code
    if status in {401, 407}:
        raise SearchFailure(SearchError("upload", "AUTH_ERROR", f"{name} rejected authentication"))
    if status == 403:
        raise SearchFailure(SearchError("upload", "ACCESS_RESTRICTED", f"{name} denied image upload"))
    if status == 429:
        raise SearchFailure(
            SearchError("upload", "RATE_LIMITED", f"{name} rate limited image upload"),
            retry_after_s=_retry_after(response.headers.get("Retry-After")),
        )
    if status >= 400:
        raise SearchFailure(SearchError("upload", "PROVIDER_ERROR", f"{name} returned HTTP {status}"))
    if len(response.content) > max_bytes:
        raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"{name} response exceeded the size limit"))


def _http_url(value: Any, *, allow_protocol_relative: bool = False) -> str | None:
    """验证 HTTP(S) URL，并可把 ``//host/path`` 补成 HTTPS。"""

    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip()
    if allow_protocol_relative and cleaned.startswith("//"):
        cleaned = "https:" + cleaned
    try:
        parsed = urlparse(cleaned)
        _ = parsed.port
    except ValueError:
        return None
    return cleaned if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else None


def _text(value: Any) -> str | None:
    """将 Provider 任意值收敛为非空文本或 None。"""

    return value.strip() if isinstance(value, str) and value.strip() else None


def _looks_like_video(url: str) -> bool:
    """通过常见视频平台主机判断候选类型。"""

    lowered = url.lower()
    return any(host in lowered for host in (
        "youtube.com/", "youtu.be/", "vimeo.com/", "bilibili.com/video/", "tiktok.com/",
    ))


def _platform(url: str) -> str | None:
    """从 URL 提取平台名称，供 Candidate 身份去重使用。"""

    lowered = url.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    for name in ("vimeo", "bilibili", "tiktok"):
        if name in lowered:
            return name
    return None


def _retry_after(value: str | None) -> float | None:
    """解析服务端建议等待秒数；非法值不影响主错误。"""

    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
