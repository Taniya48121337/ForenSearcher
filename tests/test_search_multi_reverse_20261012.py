"""10-12 多引擎以图搜图：Google Cloud Vision + Google Lens（SerpAPI / Serper）+ TinEye + Yandex，合并结果。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from foren_searcher.search.contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.base import QUERY_IMAGE_URL
from foren_searcher.search.providers.google_lens import GoogleLensProvider
from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider, merge_candidates
from foren_searcher.search.providers.tineye import TinEyeProvider, normalize_tineye

KEY_ENVS = ("GOOGLE_VISION_API_KEY", "SERPAPI_API_KEY", "SERPER_API_KEY", "TINEYE_API_KEY")


def _req(output_type: str = "text") -> SearchQueryRequest:
    return SearchQueryRequest("image", "E3", output_type)


@pytest.fixture
def image(tmp_path) -> Path:
    path = tmp_path / "query.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return path


# ---------------------------------------------------------------- TinEye

TINEYE_JSON = {"code": 200, "messages": [], "stats": {"total_results": 2},
               "results": {"matches": [
                   {"image_url": "https://img.tineye.com/a.jpg", "domain": "news.example", "score": 88.1, "width": 800,
                    "height": 600, "backlinks": [
                        {"url": "https://news.example/a.jpg", "backlink": "https://news.example/story-2019", "crawl_date": "2019-04-02"},
                        {"url": "https://news.example/a.jpg", "backlink": "https://news.example/story-2012", "crawl_date": "2012-01-15"}]},
                   {"image_url": "https://img.tineye.com/b.jpg", "domain": "youtube.com", "score": 70.0, "width": 480,
                    "height": 360, "backlinks": [
                        {"url": "https://i.ytimg.com/vi/abcdefghijk/hq.jpg",
                         "backlink": "https://www.youtube.com/watch?v=abcdefghijk", "crawl_date": "2016-08-30"},
                        {"url": "https://x.example/b.jpg", "backlink": "https://x.example/undated"}]}]}}


def test_tineye_backlinks_are_sorted_oldest_first_with_crawl_dates():
    items, info = normalize_tineye(TINEYE_JSON)
    assert [i["crawl_date"] for i in items] == ["2012-01-15", "2016-08-30", "2019-04-02", None]
    assert info["earliest_crawl_date"] == "2012-01-15" and info["matches"] == 2


def test_tineye_provider_request_and_candidates(image):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TINEYE_JSON)

    provider = TinEyeProvider("secret", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    batch = asyncio.run(provider.search(_req("text"), offset=0, limit=10, image_path=image))
    req = seen[0]
    body = req.content.decode("latin-1")
    assert str(req.url) == "https://api.tineye.com/rest/search/" and req.headers["x-api-key"] == "secret"
    assert 'name="image_upload"' in body and 'name="sort"' in body and "crawl_date" in body and 'name="order"' in body
    first = batch.candidates[0]
    assert first.source_url == "https://news.example/story-2012" and first.extra["crawl_date"] == "2012-01-15"
    assert "first crawled by TinEye on 2012-01-15" in first.search_snippet and "not a publication date" in first.search_snippet
    video = asyncio.run(provider.search(_req("video"), offset=0, limit=10, image_path=image))
    assert video.candidates[0].candidate_type == "video" and len(seen) == 1          # same image: cached


def test_tineye_errors_are_classified(image):
    def handler(request):
        return httpx.Response(403, json={"code": 403, "messages": ["Invalid API key"]})

    provider = TinEyeProvider("bad", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(provider.search(_req(), offset=0, limit=5, image_path=image))
    assert exc.value.error.code == "AUTH_ERROR"


# ---------------------------------------------------------------- Google Lens


def _serpapi_handler(calls, exact):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/image":
            assert b'name="image"' in request.content and b'name="api_key"' in request.content
            return httpx.Response(200, json={"message": "Image uploaded successfully.", "image_id": "IMG1"})
        assert request.url.params["engine"] == "google_lens" and request.url.params["image_id"] == "IMG1"
        if request.url.params["type"] == "exact_matches":
            return httpx.Response(200, json={"exact_matches": exact})
        return httpx.Response(200, json={"visual_matches": [
            {"position": 1, "title": "Similar", "link": "https://www.youtube.com/watch?v=abcdefghijk", "source": "YouTube",
             "thumbnail": "https://t.example/2.jpg", "image": "https://i.example/2.jpg"}]})
    return handler


def test_lens_via_serpapi_stops_after_exact_matches(image):
    calls = []
    exact = [{"position": 1, "title": "Original post", "link": "https://site.example/original", "source": "site.example",
              "date": "3 years ago", "thumbnail": "https://t.example/1.jpg"}]
    provider = GoogleLensProvider(serpapi_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(_serpapi_handler(calls, exact))))
    batch = asyncio.run(provider.search(_req("video"), offset=0, limit=10, image_path=image))
    assert [c.source_url for c in batch.candidates] == ["https://site.example/original"]
    assert "same image" in batch.candidates[0].search_snippet and "3 years ago" in batch.candidates[0].search_snippet
    assert [r.url.path for r in calls] == ["/image", "/search.json"] and provider.summary()["billed_searches"] == 1


def test_lens_via_serpapi_falls_back_to_visual_matches(image):
    calls = []
    provider = GoogleLensProvider(serpapi_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(_serpapi_handler(calls, []))))
    batch = asyncio.run(provider.search(_req("video"), offset=0, limit=10, image_path=image))
    assert batch.candidates[0].candidate_type == "video" and "visually similar" in batch.candidates[0].search_snippet
    assert [r.url.params.get("type") for r in calls[1:]] == ["exact_matches", "visual_matches"]


def test_lens_stops_calling_after_quota_runs_out(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/image":
            return httpx.Response(200, json={"image_id": "IMG1"})
        return httpx.Response(200, json={"error": "Your account has run out of searches."})

    provider = GoogleLensProvider(serpapi_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    for k in range(2):
        path = tmp_path / f"q{k}.jpg"
        path.write_bytes(b"\xff\xd8" + bytes([k]))
        with pytest.raises(SearchFailure) as exc:
            asyncio.run(provider.search(_req(), offset=0, limit=5, image_path=path))
        assert exc.value.error.code == "QUOTA_EXCEEDED"
    assert len(calls) == 2 and "disabled" in provider.summary()          # second image: no request at all


def test_lens_via_serper_needs_a_public_image_url(image):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["X-API-KEY"] == "s" and payload["url"] == "https://cdn.example/photo.jpg"
        return httpx.Response(200, json={"organic": [{"title": "Found", "link": "https://found.example/page",
                                                      "imageUrl": "https://found.example/p.jpg"}]})

    provider = GoogleLensProvider(serper_key="s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(provider.search(_req(), offset=0, limit=5, image_path=image))
    assert exc.value.error.code == "INVALID_INPUT"
    token = QUERY_IMAGE_URL.set("https://cdn.example/photo.jpg")
    try:
        batch = asyncio.run(provider.search(_req(), offset=0, limit=5, image_path=image))
    finally:
        QUERY_IMAGE_URL.reset(token)
    assert batch.candidates[0].source_url == "https://found.example/page"


# ---------------------------------------------------------------- merging


def _c(url, provider, rank, **extra):
    return CandidateSource(url, f"title {url}", f"{provider} snippet.", provider=provider, engine=provider, provider_rank=rank,
                           candidate_type="video" if "youtube" in url else "page", extra=extra)


def test_merge_interleaves_dedupes_and_names_the_engines():
    lists = {"gcv": [_c("https://a.example/x", "gcv", 1), _c("https://www.youtube.com/watch?v=abcdefghijk", "gcv", 2)],
             "tineye": [_c("https://a.example/x/", "tineye", 1, crawl_date="2011-05-01", earliest_crawl_date="2011-05-01"),
                        _c("https://old.example/y", "tineye", 2, crawl_date="2013-01-01")],
             "yandex": [_c("https://youtu.be/abcdefghijk", "yandex", 1)]}
    merged = merge_candidates(lists, ["gcv", "lens", "tineye", "yandex"])
    urls = [c.source_url for c in merged]
    assert urls == ["https://a.example/x", "https://www.youtube.com/watch?v=abcdefghijk", "https://old.example/y"]
    first = merged[0]
    assert first.extra["found_by"] == ["gcv", "tineye"] and first.extra["crawl_date"] == "2011-05-01"
    assert "Found by Google Cloud Vision, TinEye." in first.search_snippet and "first crawled this page" in first.search_snippet
    assert merged[1].extra["found_by"] == ["yandex", "gcv"] or merged[1].extra["found_by"] == ["gcv", "yandex"]


class _Engine:
    def __init__(self, name, urls=(), fail=None):
        self.name, self.urls, self.fail, self.calls = name, urls, fail, 0

    async def search(self, request, *, offset, limit, image_path=None):
        self.calls += 1
        if self.fail:
            raise SearchFailure(SearchError("upload", self.fail, f"{self.name} failed"))
        return CandidateBatch(self.name, self.name, tuple(_c(u, self.name, k + 1) for k, u in enumerate(self.urls[:limit])))


def test_multi_provider_survives_engine_failures_and_pages_from_cache(image):
    gcv = _Engine("gcv", [f"https://g.example/{k}" for k in range(5)])
    tin = _Engine("tineye", fail="QUOTA_EXCEEDED")
    yan = _Engine("yandex", [f"https://y.example/{k}" for k in range(5)])
    multi = MultiReverseImageProvider([("gcv", gcv), ("tineye", tin), ("yandex", yan)])
    first = asyncio.run(multi.search(_req(), offset=0, limit=4, image_path=image))
    second = asyncio.run(multi.search(_req(), offset=4, limit=4, image_path=image))
    assert [c.source_url for c in first.candidates] == ["https://g.example/0", "https://y.example/0", "https://g.example/1", "https://y.example/1"]
    assert [c.provider_rank for c in second.candidates] == [5, 6, 7, 8]
    assert gcv.calls == yan.calls == tin.calls == 1
    assert "tineye" in first.request_metadata["errors"] and multi.summary()["engine_errors"] == {"tineye": 1}
    assert "TinEye results are listed oldest first" in multi.note() and "Yandex" in multi.note()


def test_multi_provider_raises_when_every_engine_fails(image):
    multi = MultiReverseImageProvider([("gcv", _Engine("gcv", fail="AUTH_ERROR")), ("yandex", _Engine("yandex", fail="RATE_LIMITED"))])
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(multi.search(_req(), offset=0, limit=4, image_path=image))
    assert exc.value.error.code == "AUTH_ERROR" and "every reverse image engine failed" in exc.value.error.message


# ---------------------------------------------------------------- run_cases


def test_engine_resolution_and_service_wiring(monkeypatch, tmp_path):
    from foren_searcher.datagen import run_cases as R
    from foren_searcher.store import GlobalStore
    from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider as Multi

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert R.resolve_image_engines("auto") == ["yandex"]
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local" / "tineye_key.txt").write_text("t\n", encoding="utf-8")
    monkeypatch.setenv("SERPAPI_API_KEY", "s")
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "g")
    assert R.resolve_image_engines("auto") == ["gcv", "lens", "tineye", "yandex"] == R.resolve_image_engines("all")
    assert R.resolve_image_engines("tineye,gcv") == ["gcv", "tineye"]
    monkeypatch.delenv("SERPAPI_API_KEY")
    with pytest.raises(ValueError, match="缺 API key"):
        R.resolve_image_engines("all")
    with pytest.raises(ValueError, match="不认识"):
        R.resolve_image_engines("bing")
    from types import SimpleNamespace

    case = SimpleNamespace(source_policy=SimpleNamespace(mode="unrestricted", excluded_keys=(), date_cutoff=None), case_id="t:1")
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    search, _d, _g = R.search_service_for(case, store, tmp_path, blocklist="off", video_search="ddgs")
    router = R._find_provider(search, "ImageQueryRouter")
    assert isinstance(router.image_provider, Multi) and router.image_provider.names == ["gcv", "tineye", "yandex"]
    assert search.image_search_engine == "gcv+tineye+yandex" and "TinEye" in search.image_search_note
    asyncio.run(search.aclose())


def test_service_passes_public_image_url_of_web_images(tmp_path):
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    rid, _ = store.get_or_create_candidate(identity_key="url:https://web.example/p", source_url="https://web.example/p",
                                           candidate_type="image", image_url="https://web.example/p.jpg")
    image = store.commit_material_evidence(candidate_id=rid, material_kind="image", material_status="ready",
                                           artifact_bytes=b"\xff\xd8x", media_type="image/jpeg", suffix=".jpg",
                                           modalities=("image",), label="img", content="img", locator="image",
                                           acquired_by="SEARCH/open/image", discovered_by="W1-1")
    local = store.commit_ingested_evidence(source_ref="I0", media_type="image/jpeg", modalities=("image",), label="a",
                                           original_filename="a.jpg", data=b"\xff\xd8y")
    service = SearchService.__new__(SearchService)
    service.store = store
    assert service._public_image_url(image.evidence_id) == "https://web.example/p.jpg"
    assert service._public_image_url(local.evidence_id) is None


def test_key_files_ignore_comment_lines(monkeypatch, tmp_path):
    from foren_searcher.search.providers.keys import find_key

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local" / "tineye_key.txt").write_text("# paste the key below\n\n", encoding="utf-8")
    (tmp_path / ".local" / "serpapi_key.txt").write_text("﻿# paste the key below\r\nabc123\r\n", encoding="utf-8")
    assert find_key("tineye", tmp_path) is None and find_key("serpapi", tmp_path) == "abc123"
