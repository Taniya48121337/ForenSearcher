"""Google Cloud Vision Web Detection 反向图片 Provider（10-12）。

和 DEFAME 用的是同一个接口：把一张图片发给 Google，返回含有同一图片的网页（pagesWithMatchingImages）、
完全 / 部分匹配的图片 URL、相似图片，以及 Google 对这张图的猜测标签（bestGuessLabels）和网页实体。

注意：这个 Provider 会把调查图片原样发给 Google。只在负责人同意的实验里打开（run_cases --search-arm gcvimage）。
text_image 输入里的文字这个接口用不上，只用图片；文字记在 request_metadata 里。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded
from .yandex_reverse import _http_url, _looks_like_video, _platform

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
KEY_ENV = "GOOGLE_VISION_API_KEY"
KEY_FILE = Path(".local") / "google_vision_key.txt"
NOTE = ("Image-input search is a reverse image search (Google): it returns web pages and videos that contain the same"
        " or a very similar image, so searching with a frame of the material can find where that footage was"
        " published. The text part of text_image is ignored.")
_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/|/vi/|/vi_webp/)([A-Za-z0-9_-]{11})")
_TAG = re.compile(r"<[^>]+>")


def resolve_api_key(root: str | Path | None = None) -> str:
    """环境变量优先，其次 <root>/.local/google_vision_key.txt；都没有就报错（不打印 key）。"""

    key = os.environ.get(KEY_ENV, "").strip()
    if not key and root is not None:
        path = Path(root) / KEY_FILE
        if path.is_file():
            from .keys import first_key_line

            key = first_key_line(path.read_text(encoding="utf-8-sig"))
    if not key:
        raise RuntimeError(f"Google Cloud Vision 需要 API key：设环境变量 {KEY_ENV}，或写进 .local/google_vision_key.txt")
    return key


def youtube_id(url: str) -> str | None:
    match = _YT_ID.search(url or "")
    return match.group(1) if match else None


class GoogleVisionWebProvider:
    """image / text_image 输入 → Web Detection；结果规范成 CandidateSource。"""

    NOTE = NOTE

    def __init__(self, api_key: str, timeout_s: float = 30.0, *, max_results: int = 50,
                 client: httpx.AsyncClient | None = None, max_response_bytes: int = 4 * 1024 * 1024) -> None:
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        self._key = api_key.strip()
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, Any] = {"provider": "google-vision-web", "calls": 0, "reused": 0, "errors": 0,
                                      "candidates": 0, "video_candidates": 0}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats)

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "Google reverse image search needs one image Evidence"))
        image = image_path.read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        async with self._lock:
            web = self._cache.get(digest)
            reused = web is not None
            if web is None:
                web = await self._annotate(image)
                self._cache[digest] = web
            self.stats["reused" if reused else "calls"] += 1
        ranked = normalize_web_detection(web, request.output_type)
        selected = ranked[offset:offset + limit]
        guess = _best_guess(web)
        candidates = tuple(c for c in (
            _candidate(item, rank, guess) for rank, item in enumerate(selected, start=offset + 1)) if c is not None)
        self.stats["candidates"] += len(candidates)
        self.stats["video_candidates"] += sum(c.candidate_type == "video" for c in candidates)
        text = request.query.get("text") if isinstance(request.query, dict) else None
        return CandidateBatch(
            "google-vision", "web-detection", candidates,
            {"best_guess": _best_guess(web), "entities": _entities(web), "result_count": len(ranked),
             "results": selected},
            {"uploaded": not reused, "reused_provider_result": reused, "ignored_text": bool(text)},
        )

    async def _annotate(self, image: bytes) -> dict[str, Any]:
        body = {"requests": [{"image": {"content": base64.b64encode(image).decode()},
                              "features": [{"type": "WEB_DETECTION", "maxResults": self._max_results}],
                              "imageContext": {"webDetectionParams": {"includeGeoResults": False}}}]}
        try:
            response = await request_bounded(self._client, "POST", ENDPOINT, maximum_bytes=self._max_response_bytes,
                                             provider_name="Google Vision", stage="upload",
                                             params={"key": self._key}, json=body)
        except httpx.TimeoutException as exc:
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("upload", "TIMEOUT", "Google reverse image search timed out")) from exc
        except httpx.HTTPError as exc:
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "Google reverse image search failed")) from exc
        status = response.status_code
        if status != 200:
            self.stats["errors"] += 1
            code = ("AUTH_ERROR" if status in (400, 401, 403) else "RATE_LIMITED" if status == 429 else "PROVIDER_ERROR")
            raise SearchFailure(SearchError("upload", code, f"Google Vision returned HTTP {status}"))  # type: ignore[arg-type]
        try:
            data = json.loads(response.content)
            result = (data.get("responses") or [{}])[0]
        except (ValueError, AttributeError) as exc:
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "Google Vision response was not JSON")) from exc
        if "error" in result:
            self.stats["errors"] += 1
            message = str((result.get("error") or {}).get("message", ""))[:160]
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"Google Vision error: {message}"))
        web = result.get("webDetection")
        return dict(web) if isinstance(web, Mapping) else {}


def normalize_web_detection(web: Mapping[str, Any], output_type: str) -> list[dict[str, Any]]:
    """按输出类型排好序的结果列表（每项 url / title / kind / image_url / video_id）。

    video：先是含同一图片的视频页面，再是相似图里的 YouTube 缩略图（i.ytimg.com/vi/<编号>/… → 对应视频），最后是其他页面。
    image：完全匹配图、部分匹配图。其他：匹配页面。
    """

    pages, seen = [], set()
    for page in web.get("pagesWithMatchingImages") or []:
        url = _http_url(page.get("url")) if isinstance(page, Mapping) else None
        if not url or url in seen:
            continue
        seen.add(url)
        kind = "full" if page.get("fullMatchingImages") else "partial" if page.get("partialMatchingImages") else "page"
        imgs = [i.get("url") for i in (page.get("fullMatchingImages") or []) + (page.get("partialMatchingImages") or [])
                if isinstance(i, Mapping)]
        pages.append({"url": url, "title": _clean(page.get("pageTitle")), "kind": kind,
                      "image_url": _http_url(imgs[0]) if imgs else None, "video_id": youtube_id(url)})
    if output_type == "image":
        out = []
        for kind, key in (("full", "fullMatchingImages"), ("partial", "partialMatchingImages")):
            for item in web.get(key) or []:
                url = _http_url(item.get("url")) if isinstance(item, Mapping) else None
                if url and url not in seen:
                    seen.add(url)
                    out.append({"url": url, "title": None, "kind": kind, "image_url": url, "video_id": youtube_id(url)})
        return out
    if output_type != "video":
        return pages
    videos = [p for p in pages if _looks_like_video(p["url"])]
    have = {p["video_id"] for p in videos if p["video_id"]}
    thumbs = []
    for key in ("fullMatchingImages", "partialMatchingImages", "visuallySimilarImages"):
        for item in web.get(key) or []:
            url = item.get("url") if isinstance(item, Mapping) else None
            vid = youtube_id(url or "") if url and "ytimg.com" in url else None
            if vid and vid not in have:
                have.add(vid)
                thumbs.append({"url": f"https://www.youtube.com/watch?v={vid}", "title": None,
                               "kind": "similar_thumbnail" if key == "visuallySimilarImages" else "thumbnail",
                               "image_url": _http_url(url), "video_id": vid,
                               "thumb_match": {"fullMatchingImages": "full", "partialMatchingImages": "partial"}.get(key, "similar")})
    others = [p for p in pages if not _looks_like_video(p["url"])]
    return videos + thumbs + others


def _candidate(item: Mapping[str, Any], rank: int, guess: list[str] | None = None) -> CandidateSource | None:
    url = item["url"]
    is_video = _looks_like_video(url)
    kind = item.get("kind")
    snippet = {"full": "Reverse image search: this page contains the same image.",
               "partial": "Reverse image search: this page contains a cropped or edited copy of the image.",
               "page": "Reverse image search: this page contains a matching image.",
               "thumbnail": "Reverse image search: this video's thumbnail matches the image.",
               "similar_thumbnail": "Reverse image search: this video's thumbnail looks similar to the image."}.get(kind)
    if snippet and guess:
        snippet += f" Google's best guess for the image: {guess[0]}."
    try:
        return CandidateSource(
            source_url=url, search_title=item.get("title"), search_snippet=snippet, provider="google-vision",
            engine="web-detection", provider_rank=rank,
            candidate_type="image" if kind in ("full", "partial") and item.get("image_url") == url else "video" if is_video else "page",
            image_url=item.get("image_url") if item.get("image_url") == url else None,
            thumbnail_url=item.get("image_url") if item.get("image_url") != url else None,
            platform=_platform(url), platform_video_id=item.get("video_id") if is_video else None,
            # 10-16：thumb_match（缩略图来自 full / partial / similar 哪一栏）、best_guess 供 reverse_normalize 用，只进 extra
            extra={"match": kind, **({"thumb_match": item["thumb_match"]} if item.get("thumb_match") else {}),
                   **({"best_guess": guess[0]} if guess else {})},
        )
    except ValueError:
        return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = _TAG.sub("", value).replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'").strip()
    return text or None


def _best_guess(web: Mapping[str, Any]) -> list[str]:
    return [str(g.get("label")) for g in web.get("bestGuessLabels") or [] if isinstance(g, Mapping) and g.get("label")]


def _entities(web: Mapping[str, Any]) -> list[str]:
    return [str(e.get("description")) for e in (web.get("webEntities") or [])[:10]
            if isinstance(e, Mapping) and e.get("description")]


__all__ = ["GoogleVisionWebProvider", "NOTE", "normalize_web_detection", "resolve_api_key", "youtube_id"]
