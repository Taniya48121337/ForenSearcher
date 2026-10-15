"""TinEye 反向图片 Provider（10-12）：找"谁最早发布了这张图"。

T²Agent 用 TinEye 追溯视觉内容最早在网上出现的时间；AVerImaTeC 一类题要求找最早发布者。TinEye 只认几乎相同的
副本（裁剪、缩放、调色后也认），每个副本附带它在哪些网页上出现（backlinks）及 TinEye 第一次爬到的日期（crawl_date）。

接口（与官方 pytineye 3.0 一致）：POST https://api.tineye.com/rest/search/，请求头 x-api-key，multipart 字段
image_upload，参数 offset / limit / sort / order。这里按 crawl_date 升序取（最早出现的在前），把每个副本的每个
backlink 展开成一个候选，摘要里写明爬取日期；同时给出所有结果里最早的日期。

注意：crawl_date 是 TinEye 第一次看到该网页上这张图的日期，不是发布日期，只能作为"不晚于"的线索。
这个 Provider 会把调查图片发给 TinEye（API 付费，按次计）。
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
from .google_vision import youtube_id
from .yandex_reverse import _http_url, _looks_like_video, _platform

ENDPOINT = "https://api.tineye.com/rest/search/"


class TinEyeProvider:
    def __init__(self, api_key: str, timeout_s: float = 45.0, *, max_matches: int = 50, backlinks_per_match: int = 10,
                 client: httpx.AsyncClient | None = None, max_response_bytes: int = 6 * 1024 * 1024) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        self._key = api_key.strip()
        self._max_matches = max_matches
        self._per_match = backlinks_per_match
        self._max = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        self.stats: dict[str, Any] = {"provider": "tineye", "calls": 0, "reused": 0, "errors": 0, "candidates": 0,
                                      "matches": 0}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats)

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "TinEye search needs one image Evidence"))
        image = image_path.read_bytes()
        key = hashlib.sha256(image).hexdigest()
        async with self._lock:
            cached = self._cache.get(key)
            reused = cached is not None
            if cached is None:
                try:
                    data = await self._search(image)
                except SearchFailure:
                    self.stats["errors"] += 1
                    raise
                cached = normalize_tineye(data, self._per_match)
                self._cache[key] = cached
                self.stats["matches"] += cached[1]["matches"]
            self.stats["reused" if reused else "calls"] += 1
        items, info = cached
        if request.output_type == "image":
            pool = [i for i in items if i.get("image_url")]
        elif request.output_type == "video":
            pool = sorted(items, key=lambda i: not _looks_like_video(i["url"]))     # video pages first, then by date
        else:
            pool = items
        selected = pool[offset:offset + limit]
        candidates = tuple(c for c in (_candidate(item, rank, request.output_type, info.get("earliest_crawl_date"))
                                       for rank, item in enumerate(selected, start=offset + 1)) if c is not None)
        self.stats["candidates"] += len(candidates)
        return CandidateBatch("tineye", "tineye-api", candidates, {**info, "results": selected},
                              {"uploaded": not reused, "sort": "crawl_date asc"})

    async def _search(self, image: bytes) -> dict[str, Any]:
        try:
            response = await request_bounded(
                self._client, "POST", ENDPOINT, maximum_bytes=self._max, provider_name="TinEye", stage="upload",
                headers={"x-api-key": self._key},
                data={"offset": "0", "limit": str(self._max_matches), "sort": "crawl_date", "order": "asc",
                      "backlink_limit": str(self._per_match)},
                files={"image_upload": ("image.jpg", image, "image/jpeg")})
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("upload", "TIMEOUT", "TinEye search timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "TinEye search failed")) from exc
        try:
            data = json.loads(response.content)
        except ValueError as exc:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"TinEye returned HTTP {response.status_code}, not JSON")) from exc
        code = data.get("code") if isinstance(data, dict) else None
        if response.status_code != 200 or code not in (200, None):
            status = int(code or response.status_code)
            messages = "; ".join(str(m) for m in (data.get("messages") or [])[:3]) if isinstance(data, dict) else ""
            err = ("AUTH_ERROR" if status in (401, 403) else "RATE_LIMITED" if status == 429
                   else "QUOTA_EXCEEDED" if status == 402 or "remaining" in messages.lower() else "PROVIDER_ERROR")
            raise SearchFailure(SearchError("search", err, f"TinEye error {status}: {messages[:160] or 'no message'}"))  # type: ignore[arg-type]
        return data if isinstance(data, dict) else {}


def normalize_tineye(data: Mapping[str, Any], per_match: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """matches[].backlinks[] → 每个网页一项，按 crawl_date 升序（无日期的排最后）。"""

    results = data.get("results") if isinstance(data.get("results"), Mapping) else {}
    matches = [m for m in (results.get("matches") or []) if isinstance(m, Mapping)]
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m_rank, match in enumerate(matches, start=1):
        backlinks = [b for b in (match.get("backlinks") or []) if isinstance(b, Mapping)][:per_match]
        for back in backlinks:
            page = _http_url(back.get("backlink"))
            if page is None or page in seen:
                continue
            seen.add(page)
            date = back.get("crawl_date") if isinstance(back.get("crawl_date"), str) else None
            items.append({"url": page, "image_url": _http_url(back.get("url")) or _http_url(match.get("image_url")),
                          "match_image_url": _http_url(match.get("image_url")), "crawl_date": date,
                          "domain": match.get("domain"), "score": match.get("score"), "match_rank": m_rank,
                          "size": [match.get("width"), match.get("height")], "video_id": youtube_id(page)})
    items.sort(key=lambda i: (i["crawl_date"] is None, i["crawl_date"] or "", i["match_rank"]))
    dated = [i["crawl_date"] for i in items if i["crawl_date"]]
    stats = data.get("stats") if isinstance(data.get("stats"), Mapping) else {}
    info = {"matches": len(matches), "pages": len(items), "earliest_crawl_date": min(dated) if dated else None,
            "total_results": stats.get("total_results") if isinstance(stats, Mapping) else None}
    return items, info


def _candidate(item: Mapping[str, Any], rank: int, output_type: str, earliest: str | None) -> CandidateSource | None:
    url = item["url"]
    is_video = _looks_like_video(url)
    date = item.get("crawl_date")
    snippet = "Reverse image search (TinEye): this page has a copy of the image"
    snippet += f", first crawled by TinEye on {date}." if date else "."
    if earliest:
        snippet += f" Earliest TinEye crawl date among all copies: {earliest} (a crawl date, not a publication date)."
    try:
        return CandidateSource(
            source_url=url, search_title=None, search_snippet=snippet, provider="tineye", engine="tineye-api",
            provider_rank=rank, candidate_type="image" if output_type == "image" else "video" if is_video else "page",
            image_url=item.get("image_url") if output_type == "image" else None,
            thumbnail_url=item.get("match_image_url"), platform=_platform(url),
            platform_video_id=item.get("video_id") if is_video else None,
            extra={"crawl_date": date, "earliest_crawl_date": earliest, "domain": item.get("domain"),
                   "tineye_score": item.get("score")},
        )
    except ValueError:
        return None


__all__ = ["TinEyeProvider", "normalize_tineye"]
