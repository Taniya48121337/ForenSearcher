"""检索护栏：核查站点黑名单（exclude / mark）与按声明日期截断，套在 provider / opener 外面。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from foren_searcher.search.contracts import CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource, SearchFailure, SearchQueryRequest
from foren_searcher.search.retrieval_guard import (FACT_CHECK_DOMAINS, GuardedOpener, GuardedProvider, RetrievalGuard,
                                                   apply_retrieval_guard, context_mentions_fact_check, fact_check_match,
                                                   page_published_date, parse_date)


def _cand(url: str, **extra) -> CandidateSource:
    return CandidateSource(url, "t", "s", "p", "e", 1, "page", extra=extra)


class FakeProvider:
    def __init__(self, urls, **extra):
        self.batch = CandidateBatch("p", "e", tuple(_cand(u, **extra) for u in urls))

    async def search(self, request, *, offset, limit, image_path=None):
        return self.batch


class FakeOpener:
    def __init__(self, html: bytes = b"<html></html>", final_url: str | None = None, video=None):
        self.html, self.final_url, self.video = html, final_url, video
        self.opened = []

    async def open(self, candidate, request):
        self.opened.append(candidate.source_url)
        material = OpenedMaterial("page", "ready", "body", self.html)
        return OpenedSource(self.final_url or candidate.source_url, (material,), video_material=self.video)


REQ = SearchQueryRequest("text", "q", "text")


def test_fact_check_match_by_domain_suffix_and_path_prefix() -> None:
    assert fact_check_match("https://www.snopes.com/fact-check/x") == "snopes.com"
    assert fact_check_match("https://es.snopes.com/y") == "snopes.com"
    assert fact_check_match("http://web.archive.org/web/2020/https://politifact.com/a") == "politifact.com"
    assert fact_check_match("https://www.reuters.com/fact-check/foo") == "reuters.com/fact-check"
    assert fact_check_match("https://www.reuters.com/world/story") is None       # 普通新闻不拦
    assert fact_check_match("https://nydailynews.com/2016/x") is None
    assert fact_check_match(None) is None
    assert all("/" not in d or d.count("/") >= 1 for d in FACT_CHECK_DOMAINS)


def test_exclude_mode_drops_candidates_and_refuses_open(tmp_path: Path) -> None:
    guard = RetrievalGuard(blocklist="exclude", log_path=tmp_path / "g.jsonl")
    provider = GuardedProvider(FakeProvider(["https://www.snopes.com/a", "https://nydailynews.com/b"]), guard)
    batch = asyncio.run(provider.search(REQ, offset=0, limit=10))
    assert [c.source_url for c in batch.candidates] == ["https://nydailynews.com/b"]
    opener = GuardedOpener(FakeOpener(), guard)
    with pytest.raises(SearchFailure) as info:
        asyncio.run(opener.open(_cand("https://www.politifact.com/x"), REQ))
    assert info.value.error.code == "ACCESS_RESTRICTED"
    # 重定向到核查站点也拒
    redirected = GuardedOpener(FakeOpener(final_url="https://fullfact.org/z"), guard)
    with pytest.raises(SearchFailure):
        asyncio.run(redirected.open(_cand("https://short.link/1"), REQ))
    rows = [json.loads(l) for l in (tmp_path / "g.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["reason"] for r in rows] == ["fact_check_excluded"] * 3
    assert guard.summary()["counts"]["fact_check_excluded"] == 3


def test_mark_mode_keeps_candidates_but_tags_them(tmp_path: Path) -> None:
    guard = RetrievalGuard(blocklist="mark", log_path=tmp_path / "g.jsonl")
    provider = GuardedProvider(FakeProvider(["https://www.snopes.com/a", "https://nydailynews.com/b"]), guard)
    batch = asyncio.run(provider.search(REQ, offset=0, limit=10))
    assert len(batch.candidates) == 2
    assert batch.candidates[0].extra["fact_check_site"] == "snopes.com" and "fact_check_site" not in batch.candidates[1].extra
    opener = GuardedOpener(FakeOpener(), guard)
    asyncio.run(opener.open(batch.candidates[0], REQ))          # mark 模式不拒绝打开
    assert guard.summary()["counts"]["fact_check_marked"] == 1


def test_date_cutoff_uses_candidate_page_and_video_dates(tmp_path: Path) -> None:
    guard = RetrievalGuard(blocklist="off", date_cutoff="2016-07-20", log_path=tmp_path / "g.jsonl")
    provider = GuardedProvider(FakeProvider(["https://a.com/late"], publishedDate="2016-08-01"), guard)
    assert asyncio.run(provider.search(REQ, offset=0, limit=10)).candidates == ()
    kept = GuardedProvider(FakeProvider(["https://a.com/early"], publishedDate="2016-07-14"), guard)
    assert len(asyncio.run(kept.search(REQ, offset=0, limit=10)).candidates) == 1
    unknown = GuardedProvider(FakeProvider(["https://a.com/nodate"]), guard)
    cand = asyncio.run(unknown.search(REQ, offset=0, limit=10)).candidates[0]
    assert cand.extra.get("date_unknown") is True                # 抽不到日期：放行并打标
    late_page = GuardedOpener(FakeOpener(b'<meta property="article:published_time" content="2017-01-02T00:00:00Z">'), guard)
    with pytest.raises(SearchFailure) as info:
        asyncio.run(late_page.open(_cand("https://a.com/p"), REQ))
    assert "after the claim date" in info.value.error.message
    ok_page = GuardedOpener(FakeOpener(b'<script type="application/ld+json">{"datePublished":"2016-07-15"}</script>'), guard)
    asyncio.run(ok_page.open(_cand("https://a.com/q"), REQ))
    late_video = GuardedOpener(FakeOpener(video={"upload_date": "20190101"}), guard)
    with pytest.raises(SearchFailure):
        asyncio.run(late_video.open(_cand("https://youtube.com/watch?v=1"), REQ))
    counts = guard.summary()["counts"]
    assert counts["after_cutoff"] == 3 and counts["date_unknown"] >= 1


def test_parse_dates_and_page_extraction() -> None:
    assert parse_date("2016-07-14T10:00:00+02:00") == date(2016, 7, 14)
    assert parse_date("20160714") == date(2016, 7, 14)
    assert parse_date("July 14, 2016") == date(2016, 7, 14)
    assert parse_date("no date here") is None and parse_date(None) is None
    assert page_published_date(b'<time datetime="2016-07-14">') == date(2016, 7, 14)
    assert page_published_date(b"<html>nothing</html>") is None
    assert context_mentions_fact_check("see https://www.snopes.com/x and https://a.com") == ["snopes.com"]


def test_apply_guard_is_noop_when_off() -> None:
    class Service:
        provider, opener = object(), object()
    service = Service()
    apply_retrieval_guard(service, RetrievalGuard(blocklist="off"))
    assert not isinstance(service.provider, GuardedProvider)
    apply_retrieval_guard(service, RetrievalGuard(blocklist="exclude"))
    assert isinstance(service.provider, GuardedProvider) and isinstance(service.opener, GuardedOpener)
    with pytest.raises(ValueError):
        RetrievalGuard(blocklist="bogus")


def test_run_cases_wraps_search_service_with_guard(tmp_path: Path, monkeypatch) -> None:
    from foren_searcher.datagen import run_cases
    from foren_searcher.datagen.cases import Case, SourcePolicy

    class Service:
        def __init__(self):
            self.provider, self.opener = object(), object()

    import foren_searcher.search.factory as factory
    monkeypatch.setattr(factory, "create_free_search_service", lambda **kw: Service())
    case = Case(case_id="true:1", dataset="true", source_id="1", split="dev_train", task_kind="claim_check", task_text="t",
                family_key="true:1", media=(), source_policy=SourcePolicy(mode="unrestricted", date_cutoff="2016-07-20"))
    service, desc, guard = run_cases.search_service_for(case, store=None, output=tmp_path, blocklist="exclude")
    assert desc == "unrestricted_retrospective_web"
    assert isinstance(service.provider, GuardedProvider) and isinstance(service.opener, GuardedOpener)
    assert guard.date_cutoff == date(2016, 7, 20) and guard.log_path == tmp_path / "retrieval_guard.jsonl"
    provided = Case(case_id="videosham:1", dataset="videosham", source_id="1", split="dev_train", task_kind="video_diff",
                    task_text="t", family_key="videosham:1", media=(), source_policy=SourcePolicy(mode="provided_media"))
    assert run_cases.search_service_for(provided, store=None, output=tmp_path) == (None, "provided_media", None)


def test_fact_check_match_tolerates_malformed_urls():
    """导出从上下文文本里抓 URL，残缺地址（方括号、截断）不能让整行导出报错。"""

    from foren_searcher.search.retrieval_guard import fact_check_match

    assert fact_check_match("https://[E3]") is None
    assert fact_check_match("https://snopes.com/fact-check/x") == "snopes.com"


def test_blocklist_covers_secondary_fact_check_sections():
    """09-29 案卷对照：同一声明被其它核查机构 / 媒体核查栏目复核的页面也拦；栏目外的普通报道不拦。"""
    from foren_searcher.search.retrieval_guard import fact_check_match

    assert fact_check_match("https://newsmobile.in/nm-fact-checker/fact-check-image-of-loha-stambh-x") == "newsmobile.in/nm-fact-checker"
    assert fact_check_match("https://www.telugupost.com/english-factcheck/iron-pillar-in-viral-image") == "telugupost.com/english-factcheck"
    assert fact_check_match("https://timesofindia.indiatimes.com/times-fact-check/news/fake-alert-x/articleshow/1.cms")
    assert fact_check_match("https://haqcheck.org/false-the-image-doesnt-show-weapons/") == "haqcheck.org"
    assert fact_check_match("https://en.youturn.in/factcheck/inscription.html") == "youturn.in"
    assert fact_check_match("https://www.indiatoday.in/fact-check/story/x-2019-09-11") == "indiatoday.in/fact-check"
    # 栏目外：普通新闻、同域其它栏目
    assert fact_check_match("https://www.ndtv.com/india-news/viral-picture-of-padlock-on-grave") is None
    assert fact_check_match("https://thelogicalindian.com/pakistan-put-locks-daughters-graves") is None
    assert fact_check_match("https://timesofindia.indiatimes.com/city/delhi/x.cms") is None
    assert fact_check_match("https://www.newsmobile.in/articles/x") is None
