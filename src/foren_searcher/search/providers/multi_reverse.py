"""多引擎以图搜图（10-12）：Google Cloud Vision、Google Lens、TinEye、Yandex 同时搜，结果合并。

依据：AVerImaTeC 评测总结"单个引擎常常一条相关网页都没有，多个引擎才能覆盖够"；文献里各家用的引擎不同
（DEFAME / CCN / AVerImaTeC 证据库用 Google Cloud Vision，AIC CTU / RAGAR 用 Google Lens，T²Agent 用 TinEye
找最早出现时间，多智能体核查用 Yandex）。

- 每个引擎对同一张图只请求一次（第一页时一起发），之后翻页从合并结果里切。
- 合并：按各引擎名次轮流取（第 1 名们、第 2 名们……），同一网页（同一 URL 或同一 YouTube 编号）只留一条，
  记下哪些引擎找到了它（extra.found_by），摘要写明；TinEye 的爬取日期保留在摘要与 extra 里。
- 某个引擎失败（无额度、超时、Serper 没有公开地址）不影响其他引擎；全部失败才报错。各引擎的调用与失败计数在 summary()。
- 10-16 统一写法（normalize=True 或 FOREN_REVERSE_NORMALIZE=1；默认关，关时与以前逐字相同）：见 reverse_normalize.py
  （去跟踪参数、engine_match、日期类型）与 thumb_verify.py（前 N 条下载缩略图比感知哈希，verified_match=same 的排最前）。
  核对在这里做（录制回放之内），回放时直接读录下的结果，不重新下载。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from . import reverse_normalize as rn
from .google_vision import youtube_id
from .thumb_verify import thumb_url_of

LABELS = {"gcv": "Google Cloud Vision", "lens": "Google Lens", "tineye": "TinEye", "yandex": "Yandex"}


def _identity(url: str) -> str:
    vid = youtube_id(url)
    if vid:
        return f"youtube:{vid}"
    return url.split("#", 1)[0].rstrip("/").replace("://www.", "://").lower()


class MultiReverseImageProvider:
    def __init__(self, engines: Sequence[tuple[str, Any]], *, per_engine: int = 30, engine_timeout_s: float = 60.0,
                 normalize: bool | None = None, verifier: Any | None = None, time_budget_s: float | None = None) -> None:
        if not engines:
            raise ValueError("at least one reverse image engine is required")
        self.engines = list(engines)
        self.per_engine = per_engine
        self.engine_timeout_s = engine_timeout_s
        self.normalize = rn.enabled(normalize)
        self.verifier = verifier if self.normalize else None   # 自核只在统一写法下做
        # SearchService 给整次 provider 调用的上限（provider_timeout_s）：自核只用引擎返回后剩下的时间，留 2 秒余量
        self.time_budget_s = time_budget_s
        self._lock = asyncio.Lock()
        self._cache: dict[tuple[str, str], tuple[list[CandidateSource], dict[str, Any]]] = {}
        self.stats: dict[str, Any] = {"provider": "multi-reverse-image", "engines": [n for n, _ in self.engines],
                                      "searches": 0, "engine_errors": {}, "engine_candidates": {}, "found_by_count": {}}
        if self.normalize:
            self.stats.update(normalize=True, url_tracking_removed=0, date_kinds={},
                              verify={"enabled": False} if self.verifier is None else self.verifier.stats)

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.engines]

    def note(self) -> str:
        labels = ", ".join(LABELS.get(n, n) for n in self.names)
        text = (f"Image-input search is a reverse image search on {labels}: it returns web pages, images and videos that "
                "contain the same or a very similar image, so searching with a frame or image of the material can find "
                "where it was published. Each result says which engines found it.")
        if "tineye" in self.names:
            text += (" TinEye results are listed oldest first with the date TinEye first crawled that page with the image;"
                     " use them to find the earliest known copy, remembering that a crawl date is not a publication date.")
        return text + " The text part of text_image is ignored."

    async def aclose(self) -> None:
        if self.verifier is not None:
            await self.verifier.aclose()
        for _, provider in self.engines:
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()

    def summary(self) -> dict[str, Any]:
        per = {}
        for name, provider in self.engines:
            getter = getattr(provider, "summary", None)
            if callable(getter):
                try:
                    per[name] = getter()
                except Exception:  # noqa: BLE001
                    per[name] = {}
        return {**self.stats, "per_engine": per}

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int,
                     image_path: Path | None = None) -> CandidateBatch:
        if request.input_type not in {"image", "text_image"} or image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "reverse image search needs one image Evidence"))
        key = (hashlib.sha256(image_path.read_bytes()).hexdigest(), request.output_type)
        async with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                cached = await self._search_all(request, image_path)
                self._cache[key] = cached
        merged, info = cached
        selected = tuple(replace(c, provider_rank=rank) for rank, c in enumerate(merged[offset:offset + limit], start=offset + 1))
        return CandidateBatch("reverse-image", "+".join(self.names), selected, info,
                              {"engines": self.names, "errors": info.get("errors", {})})

    async def _search_all(self, request: SearchQueryRequest, image_path: Path) -> tuple[list[CandidateSource], dict[str, Any]]:
        self.stats["searches"] += 1
        started = time.monotonic()

        async def one(name: str, provider: Any):
            return await asyncio.wait_for(provider.search(request, offset=0, limit=self.per_engine, image_path=image_path),
                                          self.engine_timeout_s)

        outcomes = await asyncio.gather(*(one(n, p) for n, p in self.engines), return_exceptions=True)
        lists: dict[str, list[CandidateSource]] = {}
        errors: dict[str, str] = {}
        failures: list[SearchFailure] = []
        for (name, _), outcome in zip(self.engines, outcomes):
            if isinstance(outcome, CandidateBatch):
                lists[name] = list(outcome.candidates)
                self.stats["engine_candidates"][name] = self.stats["engine_candidates"].get(name, 0) + len(outcome.candidates)
                continue
            if isinstance(outcome, SearchFailure):
                failures.append(outcome)
                errors[name] = f"{outcome.error.code}: {outcome.error.message}"[:200]
            elif isinstance(outcome, asyncio.TimeoutError):
                errors[name] = "TIMEOUT"
            elif isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                errors[name] = f"{type(outcome).__name__}: {outcome}"[:200]
            self.stats["engine_errors"][name] = self.stats["engine_errors"].get(name, 0) + 1
        if not lists:
            if failures:
                first = failures[0]
                raise SearchFailure(SearchError(first.error.stage, first.error.code,
                                                "every reverse image engine failed: " + "; ".join(f"{k} {v}" for k, v in errors.items())[:400]))
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "every reverse image engine failed: "
                                            + "; ".join(f"{k} {v}" for k, v in errors.items())[:400]))
        merged = (await self._normalized(lists, image_path, started) if self.normalize
                  else merge_candidates(lists, self.names))
        for c in merged:
            n = len(c.extra.get("found_by", ()))
            self.stats["found_by_count"][str(n)] = self.stats["found_by_count"].get(str(n), 0) + 1
        info = {"per_engine": {k: len(v) for k, v in lists.items()}, "errors": errors, "merged": len(merged)}
        if self.normalize:
            info["verified"] = {v: sum((c.extra or {}).get("verified_match") == v for c in merged)
                                for v in ("same", "similar", "different", "unchecked")}
        return merged, info

    async def _normalized(self, lists: dict[str, list[CandidateSource]], image_path: Path,
                          started: float) -> list[CandidateSource]:
        """统一写法：按去跟踪参数后的网址合并；前 N 条自核缩略图；same 排最前。"""

        slots = rn.merge_slots(lists, self.names)
        checks: list[dict[str, Any] | None] = [None] * len(slots)
        if self.verifier is not None and self.verifier.top_n > 0 and slots:
            top = slots[:self.verifier.top_n]
            urls = [thumb_url_of(s["base"]) or next((thumb_url_of(c) for c in s["per_engine"].values() if thumb_url_of(c)), None)
                    for s in top]
            budget = None
            if self.time_budget_s is not None:
                budget = self.time_budget_s - (time.monotonic() - started) - 2.0
            if budget is not None and budget < 1.0:
                checks[:len(top)] = [self.verifier.skipped("no_time_left") for _ in top]
            else:
                checks[:len(top)] = await self.verifier.verify(image_path.read_bytes(), urls, total_timeout_s=budget)
                await self._second_engine_thumbs(top, urls, checks, image_path, started)
        out = [rn.build_candidate(slot, check) for slot, check in zip(slots, checks)]
        for c in out:
            extra = c.extra or {}
            self.stats["url_tracking_removed"] += int("url_tracking_removed_from" in extra)
            if extra.get("date_kind"):
                kinds = self.stats["date_kinds"]
                kinds[extra["date_kind"]] = kinds.get(extra["date_kind"], 0) + 1
        return rn.same_first(out)


    async def _second_engine_thumbs(self, top: list[dict[str, Any]], urls: list[str | None], checks: list[dict[str, Any] | None],
                                    image_path: Path, started: float) -> None:
        """r1（10-17）：第一家引擎的缩略图没判 near-identical 时，再试同一网页其他引擎给的缩略图，取更好的那个判定
        （10-16 实测 e260：Yandex 缩略图 near-identical、Lens 缩略图是同一事件的另一张照片，合并按 Lens 判 no match，名次 2–3 → 5）。
        记 verified_engine_fallback（用了哪个引擎的缩略图）。"""

        rank = {"same": 3, "similar": 2, "different": 1}
        retry: list[tuple[int, str]] = []
        for i, slot in enumerate(top):
            if (checks[i] or {}).get("verified_match") == "same":
                continue
            for name, cand in slot["per_engine"].items():
                alt = thumb_url_of(cand)
                if alt and alt != urls[i]:
                    retry.append((i, alt))
                    break
        if not retry:
            return
        budget = None
        if self.time_budget_s is not None:
            budget = self.time_budget_s - (time.monotonic() - started) - 2.0
            if budget < 1.0:
                return
        again = await self.verifier.verify(image_path.read_bytes(), [u for _, u in retry], total_timeout_s=budget)
        for (i, alt), check in zip(retry, again):
            old = checks[i] or {}
            if rank.get(str(check.get("verified_match")), 0) > rank.get(str(old.get("verified_match")), 0):
                checks[i] = {**check, "verified_engine_fallback": alt}


def merge_candidates(lists: dict[str, list[CandidateSource]], order: Sequence[str]) -> list[CandidateSource]:
    """各引擎名次轮流合并、按网页去重；找到同一网页的引擎记进 extra.found_by，摘要里写明。"""

    slots: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    depth = max((len(v) for v in lists.values()), default=0)
    for i in range(depth):
        for name in order:
            items = lists.get(name) or []
            if i >= len(items):
                continue
            c = items[i]
            ident = _identity(c.source_url)
            slot = slots.get(ident)
            if slot is None:
                slots[ident] = {"base": c, "found_by": [name], "extra": {name: dict(c.extra)}}
                ordered.append(ident)
            elif name not in slot["found_by"]:
                slot["found_by"].append(name)
                slot["extra"][name] = dict(c.extra)
                base = slot["base"]
                if base.candidate_type != "video" and c.candidate_type == "video":
                    slot["base"] = replace(c, search_snippet=c.search_snippet or base.search_snippet)
                elif not base.search_title and c.search_title:
                    slot["base"] = replace(base, search_title=c.search_title)
    out = []
    for ident in ordered:
        slot = slots[ident]
        base: CandidateSource = slot["base"]
        found = slot["found_by"]
        snippet = base.search_snippet or "Reverse image search result."
        if len(found) > 1:
            snippet += " Found by " + ", ".join(LABELS.get(n, n) for n in found) + "."
        crawl = (slot["extra"].get("tineye") or {}).get("crawl_date")
        if crawl and found[0] != "tineye":
            snippet += f" TinEye first crawled this page with the image on {crawl} (a crawl date, not a publication date)."
        extra = {**base.extra, "found_by": found}
        if crawl:
            extra["crawl_date"] = crawl
            extra["earliest_crawl_date"] = (slot["extra"].get("tineye") or {}).get("earliest_crawl_date")
        out.append(replace(base, search_snippet=snippet, extra=extra))
    return out


__all__ = ["LABELS", "MultiReverseImageProvider", "merge_candidates"]
