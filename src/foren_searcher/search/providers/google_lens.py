"""Google Lens 反向图片 Provider（10-12）。

AVerImaTeC 评测第 3 名（AIC CTU）在官方的 Google Cloud Vision 证据之外加了 Google Lens，结论是多个以图搜图引擎才能
覆盖够（单个引擎常常一条相关网页都没有）。两条接入方式：

- SerpAPI（`SERPAPI_API_KEY`）：先把查询图片上传到 https://serpapi.com/image（≤500 KB，SearchService 已经压成
  ≤500 KiB 的 JPEG），拿到 image_id，再 GET https://serpapi.com/search.json?engine=google_lens&type=exact_matches；
  一条完全匹配都没有时才再要 visual_matches（省额度：免费套餐每月 250 次，上传不计次）。本机材料的帧也能搜。
- 额度用完（QUOTA_EXCEEDED）或 key 无效（AUTH_ERROR）后，本 run 里不再调用，直接报同一个错误（多引擎合并时其他引擎照常）。
- Serper（`SERPER_API_KEY`）：POST https://google.serper.dev/lens {"url": 图片地址}，只能按公开 URL 搜，
  所以只用于 SEARCH 找到的网络图片（providers.base.QUERY_IMAGE_URL）；本机材料的帧会得到 INVALID_INPUT。

两个 key 都有时优先 SerpAPI。这个 Provider 会把调查图片发给 SerpAPI / Google。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded
from .base import QUERY_IMAGE_URL
from .google_vision import youtube_id
from .yandex_reverse import _http_url, _looks_like_video, _platform

SERPAPI_UPLOAD = "https://serpapi.com/image"
SERPAPI_SEARCH = "https://serpapi.com/search.json"
SERPER_LENS = "https://google.serper.dev/lens"


class GoogleLensProvider:
    def __init__(self, *, serpapi_key: str | None = None, serper_key: str | None = None, timeout_s: float = 45.0,
                 client: httpx.AsyncClient | None = None, max_response_bytes: int = 6 * 1024 * 1024,
                 language: str = "en") -> None:
        if not (serpapi_key or "").strip() and not (serper_key or "").strip():
            raise ValueError("Google Lens needs a SerpAPI or a Serper API key")
        self._serpapi = (serpapi_key or "").strip() or None
        self._serper = (serper_key or "").strip() or None
        self._language = language
        self._max = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)
        self._lock = asyncio.Lock()
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self.backend = "serpapi" if self._serpapi else "serper"
        self.stats: dict[str, Any] = {"provider": f"google-lens/{self.backend}", "calls": 0, "reused": 0, "errors": 0,
                                      "candidates": 0, "skipped_no_public_url": 0, "billed_searches": 0}
        self._disabled: SearchError | None = None     # 额度用完 / key 无效后不再请求

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats)

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "Google Lens search needs one image Evidence"))
        image = image_path.read_bytes()
        public_url = QUERY_IMAGE_URL.get()
        key = hashlib.sha256(image).hexdigest()
        async with self._lock:
            items = self._cache.get(key)
            reused = items is not None
            if items is None and self._disabled is not None:
                raise SearchFailure(self._disabled)
            if items is None:
                try:
                    if self._serpapi:
                        items = await self._serpapi_search(image)
                    elif public_url:
                        items = await self._serper_search(public_url)
                    else:
                        self.stats["skipped_no_public_url"] += 1
                        raise SearchFailure(SearchError(
                            "upload", "INVALID_INPUT",
                            "Google Lens via Serper can only search images that have a public URL"))
                except SearchFailure as exc:
                    self.stats["errors"] += 1
                    if exc.error.code in {"QUOTA_EXCEEDED", "AUTH_ERROR"}:
                        self._disabled = exc.error
                        self.stats["disabled"] = f"{exc.error.code}: {exc.error.message}"[:200]
                    raise
                self._cache[key] = items
            self.stats["reused" if reused else "calls"] += 1
        selected = items[offset:offset + limit]
        candidates = tuple(c for c in (_candidate(item, rank, request.output_type)
                                       for rank, item in enumerate(selected, start=offset + 1)) if c is not None)
        if request.output_type == "video":          # same order as the other engines: video pages first
            candidates = tuple(sorted(candidates, key=lambda c: c.candidate_type != "video"))
        self.stats["candidates"] += len(candidates)
        return CandidateBatch("google-lens", self.backend, candidates,
                              {"result_count": len(items), "results": selected},
                              {"uploaded": not reused, "backend": self.backend})

    async def _serpapi_search(self, image: bytes) -> list[dict[str, Any]]:
        upload = await self._request("POST", SERPAPI_UPLOAD, "SerpAPI image upload", "upload",
                                     data={"api_key": self._serpapi}, files={"image": ("query.jpg", image, "image/jpeg")})
        image_id = upload.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise SearchFailure(SearchError("upload", "PROVIDER_ERROR", "SerpAPI image upload returned no image_id"))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for kind in ("exact_matches", "visual_matches"):
            data = await self._request("GET", SERPAPI_SEARCH, "SerpAPI Google Lens", "search",
                                       params={"engine": "google_lens", "image_id": image_id, "type": kind,
                                               "hl": self._language, "api_key": self._serpapi})
            self.stats["billed_searches"] += 1
            for value in data.get(kind) or []:
                item = _normalize(value, "exact" if kind == "exact_matches" else "visual")
                if item and item["url"] not in seen:
                    seen.add(item["url"])
                    out.append(item)
            if out:                   # 有完全匹配就不再花一次查相似图
                break
        return out

    async def _serper_search(self, image_url: str) -> list[dict[str, Any]]:
        data = await self._request("POST", SERPER_LENS, "Serper Google Lens", "search",
                                   headers={"X-API-KEY": self._serper, "Content-Type": "application/json"},
                                   json={"url": image_url, "hl": self._language})
        self.stats["billed_searches"] += 1
        out, seen = [], set()
        for value in data.get("organic") or data.get("visual_matches") or []:
            item = _normalize(value, "visual")
            if item and item["url"] not in seen:
                seen.add(item["url"])
                out.append(item)
        return out

    async def _request(self, method: str, url: str, name: str, stage: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await request_bounded(self._client, method, url, maximum_bytes=self._max, provider_name=name,
                                             stage=stage, **kwargs)  # type: ignore[arg-type]
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError(stage, "TIMEOUT", f"{name} timed out")) from exc  # type: ignore[arg-type]
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError(stage, "NETWORK_ERROR", f"{name} failed")) from exc  # type: ignore[arg-type]
        status = response.status_code
        if status != 200:
            code = ("AUTH_ERROR" if status in (401, 403) else "RATE_LIMITED" if status == 429
                    else "QUOTA_EXCEEDED" if status == 402 else "PROVIDER_ERROR")
            raise SearchFailure(SearchError(stage, code, f"{name} returned HTTP {status}"))  # type: ignore[arg-type]
        try:
            data = json.loads(response.content)
        except ValueError as exc:
            raise SearchFailure(SearchError(stage, "PROVIDER_ERROR", f"{name} response was not JSON")) from exc  # type: ignore[arg-type]
        if not isinstance(data, dict):
            raise SearchFailure(SearchError(stage, "PROVIDER_ERROR", f"{name} response was not an object"))  # type: ignore[arg-type]
        if data.get("error") and not any(k in data for k in ("exact_matches", "visual_matches", "organic", "image_id")):
            message = str(data["error"])[:160]
            code = "QUOTA_EXCEEDED" if "run out" in message.lower() or "plan" in message.lower() else "PROVIDER_ERROR"
            if "hasn't returned any results" in message.lower() or "no results" in message.lower():
                return {}
            raise SearchFailure(SearchError(stage, code, f"{name} error: {message}"))  # type: ignore[arg-type]
        return data


def _normalize(value: Any, match: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    url = _http_url(value.get("link"))
    if url is None:
        return None
    return {"url": url, "title": value.get("title") if isinstance(value.get("title"), str) else None,
            "source": value.get("source") if isinstance(value.get("source"), str) else None,
            "image_url": _http_url(value.get("image") or value.get("imageUrl")),
            "thumbnail_url": _http_url(value.get("thumbnail") or value.get("thumbnailUrl")),
            "date": value.get("date") if isinstance(value.get("date"), str) else None, "match": match,
            "video_id": youtube_id(url)}


def _candidate(item: Mapping[str, Any], rank: int, output_type: str) -> CandidateSource | None:
    url = item["url"]
    is_video = _looks_like_video(url)
    if output_type == "image" and not item.get("image_url"):
        return None
    snippet = ("Reverse image search (Google Lens): this page contains the same image."
               if item["match"] == "exact" else "Reverse image search (Google Lens): this page shows a visually similar image.")
    if item.get("source"):
        snippet += f" Source: {item['source']}."
    if item.get("date"):
        snippet += f" Google Lens date: {item['date']}."
    try:
        return CandidateSource(
            source_url=url, search_title=item.get("title"), search_snippet=snippet, provider="google-lens",
            engine="google-lens", provider_rank=rank,
            candidate_type="image" if output_type == "image" else "video" if is_video else "page",
            image_url=item.get("image_url") if output_type == "image" else None,
            thumbnail_url=item.get("thumbnail_url"), platform=_platform(url),
            platform_video_id=item.get("video_id") if is_video else None,
            extra={"match": item["match"], **({"lens_date": item["date"]} if item.get("date") else {})},
        )
    except ValueError:
        return None


__all__ = ["GoogleLensProvider"]
