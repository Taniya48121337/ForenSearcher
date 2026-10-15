"""无需密钥的公开网页、图片与视频搜索适配器。"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from ..contracts import CandidateBatch, SearchError, SearchFailure, SearchQueryRequest
from .searxng import SearXngProvider


class DdgsProvider:
    """使用 DDGS 取得候选；正文和媒体仍由 SEARCH 的受限打开器获取。"""

    def __init__(self, timeout_s: float = 12.0, *, text_backend: str = "brave",
                 client_factory: Any = None) -> None:
        self.timeout_s = timeout_s
        self.text_backend = text_backend
        self._client_factory = client_factory

    def _fetch(self, request: SearchQueryRequest, offset: int, limit: int) -> tuple[list[Any], str, str]:
        if self._client_factory is None:
            try:
                from ddgs import DDGS
            except ImportError as exc:
                raise SearchFailure(SearchError("search", "PROVIDER_ERROR",
                    "Free search requires the 'free-search' optional dependency")) from exc
            factory = DDGS
        else:
            factory = self._client_factory
        method, backend = {
            "text": ("text", self.text_backend),
            "image": ("images", "bing"),
            "video": ("videos", "duckduckgo"),
        }[request.output_type]
        region = "cn-zh" if re.search(r"[\u3400-\u9fff]", str(request.query)) else "us-en"
        engines = ("brave", "bing", "yandex") if method == "text" and backend == "brave" else (backend,)
        deadline = time.monotonic() + self.timeout_s
        last_error: Exception | None = None
        for engine in engines:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                proxy = os.environ.get("FOREN_HTTP_PROXY") if self._client_factory is None else None
                client = factory(timeout=max(1, math.ceil(min(6, remaining))), **({"proxy": proxy} if proxy else {}))
                values = getattr(client, method)(str(request.query), region=region, safesearch="moderate",
                                                backend=engine, max_results=offset + limit)
                if not isinstance(values, list):
                    raise ValueError("search response must be a list")
                if values:
                    return values, engine, region
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("No search candidates were returned within the provider time budget")

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        del image_path
        if request.input_type != "text":
            raise SearchFailure(SearchError("search", "INVALID_INPUT", "DDGS accepts text queries only"))
        if offset < 0 or limit < 1:
            raise ValueError("offset must be non-negative and limit positive")
        try:
            values, backend, region = await asyncio.to_thread(self._fetch, request, offset, limit)
        except SearchFailure:
            raise
        except Exception as exc:
            name = type(exc).__name__
            code: Literal["RATE_LIMITED", "TIMEOUT", "PROVIDER_ERROR"] = "RATE_LIMITED" if "Ratelimit" in name else "TIMEOUT" if "Timeout" in name else "PROVIDER_ERROR"
            raise SearchFailure(SearchError("search", code, f"Free search failed ({name}); try another query or configured provider")) from exc
        candidates = []
        for rank, value in enumerate(values[offset:offset + limit], offset + 1):
            if not isinstance(value, dict):
                continue
            thumbnails = value.get("images")
            thumbnail = thumbnails.get("large") if isinstance(thumbnails, dict) else value.get("thumbnail")
            normalized = {
                "url": value.get("href") if request.output_type == "text" else value.get("content") if request.output_type == "video" else value.get("url"),
                "title": value.get("title"), "content": value.get("body") or value.get("description"),
                "img_src": value.get("image"), "thumbnail_src": thumbnail, "engine": backend,
                "publishedDate": value.get("published"), "source": value.get("publisher") or value.get("source"),
            }
            candidate = SearXngProvider._normalize(normalized, request.output_type, rank, query=str(request.query))
            if candidate is not None:
                candidates.append(replace(candidate, provider="ddgs"))
        return CandidateBatch("ddgs", backend, tuple(candidates), {"results": values}, {"region": region})
