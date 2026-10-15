"""SEARCH 的日期查询（10-13）：一个来源"什么时候发布 / 最早什么时候已经在网上"。

请求：``{"source": "R3" 或 URL, "output_type": "text", "include": ["dates"]}``。全部免费、不要 key：

- 页面自己声明的发布 / 修改日期：HTML 的 article:published_time、datePublished（含 JSON-LD）、<time datetime> 等
  （与 retrieval_guard 的日期截断同一套解析）；
- 平台上传日期：视频平台页面（YouTube 等）经 yt-dlp 元数据的 upload_date / release_date / timestamp；
- 互联网档案馆（Wayback Machine）最早存档：CDX 接口 ``web.archive.org/cdx/search/cdx``，对页面地址查一次；
  候选带原图地址（以图搜图 / 图片结果）时，对图片文件地址再查一次。

每个日期都写明它说明什么：页面声明的日期只是"这一页"的发布日期，不等于图片 / 视频第一次出现的日期；存档日期只说明
"不晚于这一天已经在网上"；查不到不说明不存在。结果是一份 kind="dates" 的文字材料（acquired_by=SEARCH/dates），
可以像别的 E 一样引用。网络请求经 SafeHttpClient（公网校验、大小上限、FOREN_HTTP_PROXY），录制回放随 opener 一起记。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlsplit

from ..contracts import CandidateSource, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchRequest
from ..policy import SearchPolicy
from .http import SafeHttpClient

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
_MODIFIED = re.compile(
    r"""<meta[^>]+(?:property|name|itemprop)\s*=\s*["'](?:article:modified_time|og:updated_time|dateModified|"""
    r"""last-modified)["'][^>]+content\s*=\s*["']([^"']+)["']""", re.I)
_JSONLD_MODIFIED = re.compile(r""""dateModified"\s*:\s*"([^"]+)\"""", re.I)
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "tiktok.com", "bilibili.com", "rumble.com")


def _iso(value: Any) -> str | None:
    from ..retrieval_guard import parse_date

    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def page_dates(raw: bytes | str | None) -> dict[str, str | None]:
    """页面 HTML → {"published": ..., "modified": ...}（YYYY-MM-DD；抽不到为 None）。"""

    from ..retrieval_guard import page_published_date

    published = page_published_date(raw)
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
    head = text[:400_000]
    modified = None
    for pattern in (_MODIFIED, _JSONLD_MODIFIED):
        m = pattern.search(head)
        if m:
            modified = _iso(m.group(1))
            if modified:
                break
    return {"published": published.isoformat() if published else None, "modified": modified}


def cdx_url(target: str) -> str:
    return (f"{CDX_ENDPOINT}?url={quote(target, safe='')}&output=json&limit=1&fl=timestamp,original,statuscode"
            "&filter=statuscode:200")


def parse_cdx(content: bytes) -> dict[str, str] | None:
    """CDX 的 JSON 输出：第一行是表头，第二行是最早一条存档。"""

    try:
        rows = json.loads(content.decode("utf-8", errors="replace") or "[]")
    except ValueError:
        return None
    if not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[1], list) or not rows[1]:
        return None
    stamp = str(rows[1][0])
    if not re.fullmatch(r"\d{8,14}", stamp):
        return None
    day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return {"date": day, "timestamp": stamp, "snapshot": f"https://web.archive.org/web/{stamp}/{rows[1][1] if len(rows[1]) > 1 else ''}"}


def _upload_from_metadata(info: Any) -> dict[str, str | None]:
    if not isinstance(info, dict):
        return {}
    upload = _iso(info.get("upload_date"))
    if not upload and isinstance(info.get("timestamp"), (int, float)):
        upload = datetime.fromtimestamp(float(info["timestamp"]), tz=timezone.utc).date().isoformat()
    return {"upload": upload, "release": _iso(info.get("release_date")),
            "channel": info.get("channel") or info.get("uploader"), "title": info.get("title")}


def _is_video_page(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _VIDEO_HOSTS)


class DateSourceOpener:
    """一个来源的日期证据：页面声明、平台上传、最早存档（页面与原图）。"""

    def __init__(self, http: SafeHttpClient, policy: SearchPolicy | None = None, video: Any | None = None) -> None:
        self.http, self.policy, self.video = http, policy or http.policy, video

    async def _archive(self, target: str | None) -> tuple[dict[str, str] | None, str | None]:
        if not target:
            return None, None
        try:
            resource = await self.http.fetch(cdx_url(target), maximum_bytes=256 * 1024)
        except SearchFailure as exc:
            return None, exc.error.message
        return parse_cdx(resource.content), None

    async def _platform(self, url: str) -> tuple[dict[str, Any], str | None]:
        if self.video is None or not _is_video_page(url):
            return {}, None
        from .ytdlp_http import extract_metadata

        try:
            info = await extract_metadata(self.video.http, self.video.policy, getattr(self.video, "_factory", None),
                                          {"skip_download": True, "quiet": True, "noplaylist": True, "ignoreconfig": True,
                                           "socket_timeout": self.policy.source_timeout_s, "no_warnings": True}, url)
        except SearchFailure as exc:
            return {}, exc.error.message
        except Exception as exc:  # noqa: BLE001 —— yt-dlp 各站点报错不统一
            return {}, f"video metadata failed ({type(exc).__name__})"
        return _upload_from_metadata(info), None

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        url = candidate.source_url
        notes: list[str] = []
        final_url = url
        declared: dict[str, str | None] = {"published": None, "modified": None}
        platform: dict[str, Any] = {}
        if _is_video_page(url):
            platform, error = await self._platform(url)
            if error:
                notes.append(f"platform metadata not available: {error}")
        else:
            try:
                resource = await self.http.fetch(url, maximum_bytes=self.policy.maximum_html_bytes)
                final_url = resource.final_url
                if resource.media_type in ("text/html", "application/xhtml+xml") or resource.content[:200].lstrip().startswith(b"<"):
                    declared = page_dates(resource.content)
            except SearchFailure as exc:
                notes.append(f"page could not be fetched: {exc.error.message}")
        page_archive, page_err = await self._archive(url)
        if page_err:
            notes.append(f"Internet Archive lookup for the page failed: {page_err}")
        image_url = candidate.image_url if candidate.image_url and candidate.image_url != url else None
        image_archive, image_err = await self._archive(image_url)
        if image_err:
            notes.append(f"Internet Archive lookup for the image file failed: {image_err}")

        lines = [f"Dates for {url}" + (f" (resolved to {final_url})" if final_url != url else "") + ":"]
        if declared.get("published"):
            lines.append(f"- published date declared by the page: {declared['published']} (when this page says it was "
                         f"published; the picture or video in it may have appeared earlier elsewhere)")
        if declared.get("modified"):
            lines.append(f"- last-modified date declared by the page: {declared['modified']}")
        if platform.get("upload"):
            lines.append(f"- platform upload date: {platform['upload']}"
                         + (f" (channel: {platform['channel']})" if platform.get("channel") else "")
                         + " — when this upload was posted; an earlier copy may exist on another account or site")
        if platform.get("release") and platform.get("release") != platform.get("upload"):
            lines.append(f"- platform release date: {platform['release']}")
        if page_archive:
            lines.append(f"- earliest Internet Archive capture of this page: {page_archive['date']} "
                         f"({page_archive['snapshot']}); the page existed no later than this date, which is not a publication date")
        if image_url:
            if image_archive:
                lines.append(f"- earliest Internet Archive capture of the image file {image_url}: {image_archive['date']}; "
                             f"the image was online no later than this date")
            else:
                lines.append(f"- the image file {image_url} has no Internet Archive capture")
        found = any(declared.values()) or platform.get("upload") or page_archive or image_archive
        if not found:
            lines.append("- no date was found: the page declares none and the Internet Archive has no capture; this does "
                         "not show when the page or its picture first appeared")
        lines += [f"- note: {n}" for n in notes]
        report = {"source_url": url, "final_url": final_url, "page_declared": declared, "platform": platform,
                  "archive_page": page_archive, "image_url": image_url, "archive_image": image_archive, "notes": notes}
        content = "\n".join(lines)
        material = OpenedMaterial(
            "dates", "ready", content, json.dumps(report, ensure_ascii=False, default=str).encode("utf-8"),
            "application/json", ".json", ("text",), f"dates of {url}"[:300],
            "page metadata, platform metadata and Internet Archive CDX", "SEARCH/dates", "source_dates")
        return OpenedSource(final_url, (material,))


def date_summary(content: str) -> str:
    """dates 材料的文字 → 观察里的一行摘要（只保留各日期，不带解释）。"""

    parts = []
    for line in content.splitlines():
        m = re.match(r"- (published date declared by the page|last-modified date declared by the page|platform upload date|"
                     r"earliest Internet Archive capture of this page|earliest Internet Archive capture of the image file)"
                     r"(?: \S+)?: (\d{4}-\d{2}-\d{2})", line)
        if m:
            label = {"published date declared by the page": "page says published",
                     "last-modified date declared by the page": "page says modified",
                     "platform upload date": "uploaded",
                     "earliest Internet Archive capture of this page": "first archived",
                     "earliest Internet Archive capture of the image file": "image first archived"}[m.group(1)]
            parts.append(f"{label} {m.group(2)}")
    return "; ".join(parts) or "no date found"


__all__ = ["CDX_ENDPOINT", "DateSourceOpener", "cdx_url", "date_summary", "page_dates", "parse_cdx"]
