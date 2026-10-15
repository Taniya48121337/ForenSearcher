"""文字搜视频优先走 yt-dlp（YouTube），失败或空结果退回 DDGS（10-05：evidb2 里 DDGS 视频 21 次全部失败）。"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from foren_searcher.search.contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.video_fallback import VideoSearchFallbackProvider


def _batch(name: str, n: int) -> CandidateBatch:
    cands = tuple(CandidateSource(f"https://www.youtube.com/watch?v={name}{i}", f"{name} {i}", None, name, name, i + 1, "video")
                  for i in range(n))
    return CandidateBatch(name, name, cands, {})


class _Fake:
    def __init__(self, name: str, n: int = 2, fail: bool = False) -> None:
        self.name, self.n, self.fail, self.calls = name, n, fail, []

    async def search(self, request, *, offset, limit, image_path=None):
        self.calls.append(request)
        if self.fail:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "Free search failed (DDGSException)"))
        return _batch(self.name, self.n)


def _run(provider, request):
    return asyncio.run(provider.search(request, offset=0, limit=5))


VIDEO = SearchQueryRequest("text", "8.5 magnitude earthquake self rescue tutorial", "video")


def test_text_to_video_prefers_youtube() -> None:
    yt, dd = _Fake("yt"), _Fake("ddgs")
    router = VideoSearchFallbackProvider(yt, dd)
    assert _run(router, VIDEO).provider == "yt" and dd.calls == []
    assert router.summary()["video_primary_ok"] == 1


@pytest.mark.parametrize("primary", [_Fake("yt", fail=True), _Fake("yt", n=0)])
def test_failure_or_empty_falls_back_to_ddgs(primary) -> None:
    dd = _Fake("ddgs")
    router = VideoSearchFallbackProvider(primary, dd)
    assert _run(router, VIDEO).provider == "ddgs"
    assert router.summary()["video_fallback_used"] == 1


@pytest.mark.parametrize("request_", [SearchQueryRequest("text", "earthquake", "text"),
                                      SearchQueryRequest("text", "earthquake", "image")])
def test_other_requests_bypass_youtube(request_) -> None:
    yt, dd = _Fake("yt"), _Fake("ddgs")
    assert _run(VideoSearchFallbackProvider(yt, dd), request_).provider == "ddgs" and yt.calls == []


def test_run_cases_finds_router_through_wrappers_and_passes_flag(tmp_path: Path) -> None:
    from foren_searcher.datagen import run_cases
    from foren_searcher.search.query_guard import QueryGuard, QueryGuardedProvider

    router = VideoSearchFallbackProvider(_Fake("yt"), _Fake("ddgs"))

    class _Wrap:                      # 录制 / 黑名单护栏同样用 .provider 串起来
        def __init__(self, provider):
            self.provider = provider

    service = argparse.Namespace(provider=QueryGuardedProvider(_Wrap(_Wrap(router)), QueryGuard()))
    assert run_cases._find_video_router(service) is router
    assert run_cases.VIDEO_SEARCH_MODES == ("ytdlp", "ddgs")
    item = run_cases.Planned("c.json", "evid:video_007", "test", 1, "r", "todo")
    base = dict(round="t0", runs_dir="runs", teacher="endpoint", specs="specs", hi_spec="hi@0.5.0", gia_spec="gia@0.6.0",
                no_think=False, blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s",
                tool_cache="off", tool_cache_root="t", harness_params="apply", inspect_ops="auto", harness_override=None,
                instruction_mode="teacher", query_guard="on")
    assert "--video-search" not in run_cases._worker_command(argparse.Namespace(**base, video_search="ytdlp"), item, "e", "m")
    assert run_cases._worker_command(argparse.Namespace(**base, video_search="ddgs"), item, "e", "m")[-2:] == ["--video-search", "ddgs"]
