"""Google Lens 反向图片 Provider（10-12；10-16 加 SerpAPI→Serper 自动切换、硬盘缓存）。

AVerImaTeC 评测第 3 名（AIC CTU）在官方的 Google Cloud Vision 证据之外加了 Google Lens，结论是多个以图搜图引擎才能
覆盖够（单个引擎常常一条相关网页都没有）。两条接入方式：

- SerpAPI（`SERPAPI_API_KEY`）：先把查询图片上传到 https://serpapi.com/image（≤500 KB，SearchService 已经压成
  ≤500 KiB 的 JPEG），拿到 image_id，再 GET https://serpapi.com/search.json?engine=google_lens&type=exact_matches；
  一条完全匹配都没有时才再要 visual_matches（省额度：免费套餐每月 250 次，上传不计次）。本机材料的帧也能搜。
- Serper（`SERPER_API_KEY`）：POST https://google.serper.dev/lens {"url": 图片地址}，只能按公开 URL 搜。
  SEARCH 找到的网络图片直接用它自己的地址（providers.base.QUERY_IMAGE_URL）。本机材料的帧默认得到 INVALID_INPUT；
  打开 imgbb 图床（image_host.py，FOREN_LENS_IMAGE_HOST=imgbb，默认关）后，先传到 imgbb 换公开网址
  （10 分钟后自动删除），再交给 Serper；imgbb 失败时改传备用图床 litterbox（1 小时后删除）。
  summary 的 image_host_used / host_uploads_by_host 写实际用的图床。

两个 key 都有时先用 SerpAPI；SerpAPI 额度用完（QUOTA_EXCEEDED）或 key 无效（AUTH_ERROR）后，本 run 里改用 Serper，
同一张图当场重试。能用的后端都用完以后才整个停用（直接报同一个错误；多引擎合并时其他引擎照常）。

硬盘缓存（cache_dir）：按图片 sha256 + 语言存结果，重跑同一张图不再计费。只存成功的结果（包括"没有结果"）。

这个 Provider 会把调查图片发给 SerpAPI / Google；打开图床时，本机帧还会短时公开在 imgbb（或 litterbox）上。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded
from .base import QUERY_IMAGE_URL
from .google_vision import youtube_id
from .image_host import DEFAULT_TTL_S, ImageHostChain, ImgbbHost, LitterboxHost
from .yandex_reverse import _http_url, _looks_like_video, _platform

SERPAPI_UPLOAD = "https://serpapi.com/image"
SERPAPI_SEARCH = "https://serpapi.com/search.json"
SERPER_LENS = "https://google.serper.dev/lens"
CACHE_VERSION = 1
_STOP_CODES = {"QUOTA_EXCEEDED", "AUTH_ERROR"}


class GoogleLensProvider:
    def __init__(self, *, serpapi_key: str | None = None, serper_key: str | None = None, timeout_s: float = 45.0,
                 client: httpx.AsyncClient | None = None, max_response_bytes: int = 6 * 1024 * 1024,
                 language: str = "en", cache_dir: str | Path | None = None,
                 imgbb_key: str | None = None, image_ttl_s: int = DEFAULT_TTL_S,
                 image_fallback: str | None = None, max_searches: int | None = None) -> None:
        if not (serpapi_key or "").strip() and not (serper_key or "").strip():
            raise ValueError("Google Lens needs a SerpAPI or a Serper API key")
        self._serpapi = (serpapi_key or "").strip() or None
        self._serper = (serper_key or "").strip() or None
        # 10-16：硬上限——本 provider 最多发 max_searches 次 Lens 搜索请求（按发出算，不论成败；缓存命中不算），
        # 到了就报 QUOTA_EXCEEDED、不再上传也不再请求。None = 不限。
        self._max_searches = None if max_searches is None else max(0, int(max_searches))
        self._language = language
        self._max = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)
        # imgbb 图床：只有给了 key（即负责人用 FOREN_LENS_IMAGE_HOST=imgbb 打开）且有 Serper 时才建；
        # image_fallback="litterbox" 时 imgbb 失败改传 litterbox
        self._host: Any = None
        if (imgbb_key or "").strip() and self._serper:
            self._host = ImgbbHost(imgbb_key, client=self._client, expiration_s=image_ttl_s)
            if image_fallback == "litterbox":
                self._host = ImageHostChain([self._host, LitterboxHost(client=self._client)])
            elif image_fallback:
                raise ValueError(f"unknown image host fallback {image_fallback!r}")
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        self._fetched_on: dict[str, str] = {}   # 10-16：结果是哪天取到的（硬盘缓存按 saved_at），换算相对日期用
        self.backend = "+".join(b for b, k in (("serpapi", self._serpapi), ("serper", self._serper)) if k)
        self.stats: dict[str, Any] = {"provider": f"google-lens/{self.backend}", "calls": 0, "reused": 0, "errors": 0,
                                      "candidates": 0, "skipped_no_public_url": 0, "billed_searches": 0,
                                      "billed_by_backend": {}, "disk_hits": 0, "host_uploads": 0,
                                      "search_requests": 0, "max_searches": max_searches,
                                      "image_host": self._host.label if self._host else "off",
                                      "image_host_used": None, "host_uploads_by_host": {},
                                      "cache_dir": str(self._cache_dir) if self._cache_dir else None}
        self._serpapi_off: SearchError | None = None  # SerpAPI 额度用完 / key 无效后改走 Serper
        self._disabled: SearchError | None = None     # 能用的后端都停了以后不再请求

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def summary(self) -> dict[str, Any]:
        out = dict(self.stats)
        if self._host is not None:
            out["image_host_stats"] = json.loads(json.dumps(self._host.stats))
        return out

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "Google Lens search needs one image Evidence"))
        image = image_path.read_bytes()
        public_url = QUERY_IMAGE_URL.get()
        key = hashlib.sha256(image).hexdigest()
        async with self._lock:
            cached = self._cache.get(key)
            reused = cached is not None
            if cached is None:
                cached = self._disk_get(key)
                if cached is not None:
                    self.stats["disk_hits"] += 1
                    reused = True
            if cached is None and self._disabled is not None:
                raise SearchFailure(self._disabled)
            if cached is None:
                try:
                    cached = await self._fetch(image, public_url)
                    self._fetched_on[key] = time.strftime("%Y-%m-%d")
                except SearchFailure as exc:
                    self.stats["errors"] += 1
                    if exc.error.code in _STOP_CODES and not self._usable_backends():
                        self._disabled = exc.error
                        self.stats["disabled"] = f"{exc.error.code}: {exc.error.message}"[:200]
                    raise
                self._disk_put(key, *cached)
            self._cache[key] = cached
            self.stats["reused" if reused else "calls"] += 1
        backend, items = cached
        fetched_on = self._fetched_on.get(key)
        selected = items[offset:offset + limit]
        candidates = tuple(c for c in (_candidate(item, rank, request.output_type, backend, fetched_on)
                                       for rank, item in enumerate(selected, start=offset + 1)) if c is not None)
        if request.output_type == "video":          # same order as the other engines: video pages first
            candidates = tuple(sorted(candidates, key=lambda c: c.candidate_type != "video"))
        self.stats["candidates"] += len(candidates)
        return CandidateBatch("google-lens", backend, candidates,
                              {"result_count": len(items), "results": selected},
                              {"uploaded": not reused, "backend": backend})

    def _usable_backends(self) -> list[str]:
        out = []
        if self._serpapi and self._serpapi_off is None:
            out.append("serpapi")
        if self._serper and self._disabled is None:
            out.append("serper")
        return out

    def _cap_check(self) -> None:
        if self._max_searches is not None and self.stats["search_requests"] >= self._max_searches:
            raise SearchFailure(SearchError("search", "QUOTA_EXCEEDED",
                                            f"local Google Lens search cap reached ({self._max_searches})"))

    def _reserve(self) -> None:
        self._cap_check()
        self.stats["search_requests"] += 1

    async def _fetch(self, image: bytes, public_url: str | None) -> tuple[str, list[dict[str, Any]]]:
        self._cap_check()                             # 到上限：不上传、不请求
        if self._serpapi and self._serpapi_off is None:
            try:
                return "serpapi", await self._serpapi_search(image)
            except SearchFailure as exc:
                if exc.error.code not in _STOP_CODES or not self._serper:
                    if exc.error.code in _STOP_CODES:
                        self._serpapi_off = exc.error
                    raise
                self._serpapi_off = exc.error         # 额度用完：本 run 余下都走 Serper，这张图当场重试
                self.stats["serpapi_disabled"] = f"{exc.error.code}: {exc.error.message}"[:200]
        if not self._serper:
            raise SearchFailure(self._serpapi_off or SearchError("search", "PROVIDER_ERROR", "no Google Lens backend left"))
        url = public_url
        if not url:
            if self._host is None:
                self.stats["skipped_no_public_url"] += 1
                raise SearchFailure(SearchError(
                    "upload", "INVALID_INPUT",
                    "Google Lens via Serper can only search images that have a public URL"))
            try:
                url = await self._host.upload(image)   # 本机帧：imgbb（或备用 litterbox）临时公开，到期自动删除
            except SearchFailure as exc:
                if exc.error.code == "AUTH_ERROR":       # imgbb key 无效且没有备用：本 run 不再上传
                    self.stats["image_host_disabled"] = exc.error.message[:200]
                    self._host = None
                raise
            used = getattr(self._host, "last_used", None) or self._host.name
            self.stats["host_uploads"] += 1
            self.stats["image_host_used"] = used
            per = self.stats["host_uploads_by_host"]
            per[used] = per.get(used, 0) + 1
        try:
            return "serper", await self._serper_search(url)
        except SearchFailure as exc:
            if exc.error.code in _STOP_CODES:
                self._disabled = exc.error
            raise

    # ------------------------------------------------------------ 硬盘缓存

    def _disk_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / key[:2] / f"{key}_{self._language}.json"

    def _disk_get(self, key: str) -> tuple[str, list[dict[str, Any]]] | None:
        path = self._disk_path(key)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION or not isinstance(data.get("items"), list):
            return None
        items = [i for i in data["items"] if isinstance(i, dict) and isinstance(i.get("url"), str)]
        if isinstance(data.get("saved_at"), str) and len(data["saved_at"]) >= 10:
            self._fetched_on[key] = data["saved_at"][:10]
        return str(data.get("backend") or "cache"), items

    def _disk_put(self, key: str, backend: str, items: list[dict[str, Any]]) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({"version": CACHE_VERSION, "backend": backend, "language": self._language,
                                       "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": items},
                                      ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            self.stats["cache_write_errors"] = self.stats.get("cache_write_errors", 0) + 1

    # ------------------------------------------------------------ 后端

    def _billed(self, backend: str) -> None:
        self.stats["billed_searches"] += 1
        per = self.stats["billed_by_backend"]
        per[backend] = per.get(backend, 0) + 1

    async def _serpapi_search(self, image: bytes) -> list[dict[str, Any]]:
        upload = await self._request("POST", SERPAPI_UPLOAD, "SerpAPI image upload", "upload",
                                     data={"api_key": self._serpapi}, files={"image": ("query.jpg", image, "image/jpeg")})
        image_id = upload.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise SearchFailure(SearchError("upload", "PROVIDER_ERROR", "SerpAPI image upload returned no image_id"))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for kind in ("exact_matches", "visual_matches"):
            self._reserve()
            data = await self._request("GET", SERPAPI_SEARCH, "SerpAPI Google Lens", "search",
                                       params={"engine": "google_lens", "image_id": image_id, "type": kind,
                                               "hl": self._language, "api_key": self._serpapi})
            self._billed("serpapi")
            for value in data.get(kind) or []:
                item = _normalize(value, "exact" if kind == "exact_matches" else "visual")
                if item and item["url"] not in seen:
                    seen.add(item["url"])
                    out.append(item)
            if out:                   # 有完全匹配就不再花一次查相似图
                break
        return out

    async def _serper_search(self, image_url: str) -> list[dict[str, Any]]:
        self._reserve()
        data = await self._request("POST", SERPER_LENS, "Serper Google Lens", "search",
                                   headers={"X-API-KEY": self._serper, "Content-Type": "application/json"},
                                   json={"url": image_url, "hl": self._language})
        self._billed("serper")
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
            body = response.content[:400].decode("utf-8", "replace").lower()
            # 10-16：SerpAPI 没额度时返回 429 + "run out of searches"，要先看正文再按 429 算限速
            code = ("AUTH_ERROR" if status in (401, 403)
                    else "QUOTA_EXCEEDED" if status == 402 or "credit" in body or "run out" in body
                    else "RATE_LIMITED" if status == 429 else "PROVIDER_ERROR")
            detail = f": {body[:120]}" if body and code != "AUTH_ERROR" else ""
            raise SearchFailure(SearchError(stage, code, f"{name} returned HTTP {status}{detail}"))  # type: ignore[arg-type]
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


def _candidate(item: Mapping[str, Any], rank: int, output_type: str, backend: str | None = None,
               fetched_on: str | None = None) -> CandidateSource | None:
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
            # 10-16：lens_backend / fetched_on / source 供 reverse_normalize 换算（只进 extra，摘要不变）
            extra={"match": item["match"], **({"lens_date": item["date"]} if item.get("date") else {}),
                   **({"lens_backend": backend} if backend else {}), **({"fetched_on": fetched_on} if fetched_on else {}),
                   **({"source": item["source"]} if item.get("source") else {})},
        )
    except ValueError:
        return None


__all__ = ["GoogleLensProvider"]
