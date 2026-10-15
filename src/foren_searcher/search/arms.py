"""找原片实验的检索臂（2026-10-06，claude/EVID_找原片实验计划_20261006.md）。只在 run_cases --search-arm 给了时生效。

    results10  视频搜索每条查询给满 10 个 R（默认打开 5 个就 target_reached）                       A1
    guardlen   视频查询 > 12 个词或含日期 / 时刻戳时拒收并提示"3–8 个词"；YouTube 零结果时说清楚是零结果    A2
               （默认是退回 DDGS、DDGS 再报 PROVIDER_ERROR，模型看不出是查询太长）
    imageclip  以图搜视频在本机做：文字部分查 YouTube 前 30 条，按缩略图帧与查询图的 CLIP 相似度重排；     A3
               纯图片只重排本 run 已有的候选视频。图片不发给任何搜索引擎（providers/thumbnail_clip.py）
    fanout5    一条文字视频查询由程序改写成至多 5 条（去日期时刻、去内部编号、取前 8 词、专名优先、去泛词），   A4
               结果轮流合并去重。替模型做事，只作消融

都不改学生提示；results10 / guardlen / fanout5 只改工具行为与工具反馈，imageclip 让动作目录里的图片输入从"不可用"变可用。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest

SEARCH_ARM_FEATURES = ("results10", "guardlen", "imageclip", "fanout5", "gcvimage", "yandeximage")
VIDEO_RESULTS = 10                 # results10：每条视频查询的 R 数
FANOUT_QUERIES = 5
FANOUT_RESULTS = 15                # fanout5：5 条改写各取前几名轮流合并后的 R 数（每条约前 3 名）
MAX_VIDEO_QUERY_WORDS = 12

_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]+")
_STAMP = re.compile(
    r"(?<!\d)(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)(?!\d)")
_TIMEZONE = re.compile(r"\b(?:UTC|GMT|EDT|EST|CDT|CST|PDT|PST|JST|KST|CET|BST)\b")
_INTERNAL = re.compile(r"(?<![A-Za-z0-9_])[ERFWC]\d{1,5}(?![A-Za-z0-9_])|(?<![A-Za-z])material\s+[A-H](?![A-Za-z])", re.IGNORECASE)
_GENERIC = {
    "video", "videos", "footage", "clip", "clips", "original", "source", "full", "official", "youtube", "with", "and",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "showing", "shows", "scene", "text", "caption",
    "subtitle", "subtitles", "screen", "frame", "frames", "man", "woman", "person", "people",
}

TOO_LONG = ("Video search query is too long ({words} words). YouTube search needs most words of the query to match a "
            "video, so long descriptive queries return nothing. Use 3-8 distinctive words: an exact on-screen caption "
            "fragment, a channel name or watermark, or named people, places or events.")
HAS_STAMP = ("Video search query contains a date or time stamp ({found}). Recording timestamps shown on screen are "
             "rarely in video titles, so such queries return nothing. Leave them out and use 3-8 distinctive words: "
             "an exact on-screen caption fragment, a channel name or watermark, or named people, places or events.")
NO_RESULTS = ("No videos matched this query. YouTube search needs most words of the query to match: shorten it to "
              "3-8 distinctive words (drop dates, times and generic descriptions) or search for the on-screen caption, "
              "channel name or event name instead.")


def parse_search_arm(value: str | None) -> frozenset[str]:
    if not value or value.strip().lower() in {"", "none", "off", "a0"}:
        return frozenset()
    features = frozenset(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(features - set(SEARCH_ARM_FEATURES))
    if unknown:
        raise ValueError(f"--search-arm 只能由 {SEARCH_ARM_FEATURES} 组成：{unknown}")
    return features


def query_words(text: str) -> int:
    """拉丁词按空白切；连续汉字 / 假名每 2 个字算 1 个词。"""

    cjk = sum((len(run) + 1) // 2 for run in _CJK_RUN.findall(text))
    return cjk + len(_CJK_RUN.sub(" ", text).split())


def video_query_problem(text: str) -> str | None:
    """guardlen：视频查询太长或含日期 / 时刻戳时返回给 HI 的提示，否则 None。"""

    stamps = [m.group(0) for m in _STAMP.finditer(text)] + _TIMEZONE.findall(text)
    if stamps:
        return HAS_STAMP.format(found=", ".join(dict.fromkeys(stamps)))
    words = query_words(text)
    if words > MAX_VIDEO_QUERY_WORDS:
        return TOO_LONG.format(words=words)
    return None


def _clean(text: str) -> str:
    text = _STAMP.sub(" ", text)
    text = _TIMEZONE.sub(" ", text)
    text = _INTERNAL.sub(" ", text)
    text = re.sub(r"[\"“”«»「」『』()\[\]{}<>|,;:!?]+|(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])", " ", text)
    return " ".join(text.split())


def _first_words(text: str, n: int) -> str:
    """前 n 个"词"（与 query_words 同一口径：一长串汉字按每 2 字 1 词截断）。"""

    out: list[str] = []
    used = 0
    for token in text.split():
        size = query_words(token)
        if used + size > n:
            if _CJK_RUN.fullmatch(token) and n - used >= 2:
                out.append(token[:2 * (n - used)])
            break
        out.append(token)
        used += size
    return " ".join(out)


def fanout_queries(query: str, k: int = FANOUT_QUERIES) -> list[str]:
    """一条文字查询的程序改写（不调模型）：原句、去日期时刻与内部编号、取前 8 词、专名 / 引文优先、去泛词。"""

    out: list[str] = []

    def add(text: str) -> None:
        text = " ".join(text.split())
        if text and query_words(text) >= 1 and text.lower() not in {q.lower() for q in out}:
            out.append(text)

    add(query)
    cleaned = _clean(query)
    add(cleaned)
    add(_first_words(cleaned, 8))
    quoted = [_clean(q) for q in re.findall(r"[\"“「『]([^\"”」』]{3,80})[\"”」』]", query)]
    names = [t for t in cleaned.split()
             if (len(t) > 1 and (t[:1].isupper() or t[:1].isdigit())) or _CJK_RUN.search(t)]
    entity = " ".join(dict.fromkeys([*quoted, *names]))
    if query_words(entity) >= 2:
        add(_first_words(entity, 8))
    add(_first_words(" ".join(t for t in cleaned.split() if t.lower() not in _GENERIC), 8))
    return out[:k]


class FanoutVideoProvider:
    """fanout5：文字 → 视频查询扇出成至多 5 条改写，各自查一次，结果轮流合并去重后给前 limit 个。"""

    def __init__(self, inner: Any, *, k: int = FANOUT_QUERIES) -> None:
        self.inner, self.k = inner, k
        self.stats = {"queries": 0, "rewrites": 0, "rewrite_failures": 0}

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> Any:
        if getattr(request, "input_type", None) != "text" or getattr(request, "output_type", None) != "video":
            return await self.inner.search(request, offset=offset, limit=limit, image_path=image_path)
        rewrites = fanout_queries(str(request.query), self.k)
        self.stats["queries"] += 1
        self.stats["rewrites"] += len(rewrites)
        per_query = max(3, -(-(offset + limit) // max(1, len(rewrites))) + 2)

        async def one(text: str) -> tuple[str, Any]:
            try:
                return text, await self.inner.search(SearchQueryRequest("text", text, "video"), offset=0,
                                                     limit=per_query, image_path=None)
            except SearchFailure:
                self.stats["rewrite_failures"] += 1
                return text, None

        batches = await asyncio.gather(*(one(text) for text in rewrites))
        lists = [(text, list(batch.candidates)) for text, batch in batches if batch is not None]
        merged: list[CandidateSource] = []
        seen: set[str] = set()
        for position in range(max((len(c) for _, c in lists), default=0)):
            for index, (text, candidates) in enumerate(lists):
                if position < len(candidates):
                    item = candidates[position]
                    key = item.platform_video_id or item.source_url
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(replace(item, engine="youtube_search_fanout", provider_rank=len(merged) + 1,
                                          extra={**dict(item.extra or {}), "fanout_query": text,
                                                 "fanout_index": index, "fanout_rank": item.provider_rank}))
        if not lists and batches:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "video search failed for every rewrite"))
        page = merged[offset:offset + limit]
        return CandidateBatch("yt-dlp", "youtube_search_fanout", tuple(page), {"rewrites": rewrites})

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if callable(close):
            await close()


def video_policy_for(policy: Any, arm: frozenset[str]) -> Any | None:
    """视频查询用的 SearchPolicy（其他输出类型不变）；results10 / fanout5 / imageclip 都没开返回 None。

    results10 → 每条 10 个 R；fanout5 → 15 个；imageclip / fanout5 的单次 provider 调用要查多次 YouTube、下缩略图、
    首次还要装 CLIP，单次上限 30 → 90 秒（整条 SEARCH 的总上限 180 秒不变）。
    """

    if not arm & {"results10", "fanout5", "imageclip"}:
        return None
    size = FANOUT_RESULTS if "fanout5" in arm else VIDEO_RESULTS if "results10" in arm else policy.initial_candidates
    changes: dict[str, Any] = {"initial_candidates": size, "maximum_candidates": max(size, policy.maximum_candidates),
                               "target_sources": size}
    if arm & {"fanout5", "imageclip"}:
        changes["provider_timeout_s"] = max(policy.provider_timeout_s, 90.0)
    return replace(policy, **changes)


__all__ = [
    "FANOUT_QUERIES", "FANOUT_RESULTS", "FanoutVideoProvider", "HAS_STAMP", "MAX_VIDEO_QUERY_WORDS", "NO_RESULTS",
    "SEARCH_ARM_FEATURES", "TOO_LONG", "VIDEO_RESULTS", "fanout_queries", "parse_search_arm", "query_words",
    "video_policy_for", "video_query_problem",
]
