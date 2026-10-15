"""SEARCH 以图搜图：Google Cloud Vision Web Detection Provider（10-12）。不联网，用 httpx.MockTransport。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.google_vision import (
    GoogleVisionWebProvider, normalize_web_detection, resolve_api_key, youtube_id,
)

WEB = {
    "bestGuessLabels": [{"label": "flood rescue"}],
    "webEntities": [{"description": "Flood"}, {"description": "Rescue"}],
    "pagesWithMatchingImages": [
        {"url": "https://news.example.com/story", "pageTitle": "<b>Flood</b> &amp; rescue",
         "fullMatchingImages": [{"url": "https://news.example.com/a.jpg"}]},
        {"url": "https://www.youtube.com/watch?v=WBDWyHIDpjA", "pageTitle": "Original upload",
         "partialMatchingImages": [{"url": "https://i.ytimg.com/vi/WBDWyHIDpjA/hqdefault.jpg"}]},
    ],
    "fullMatchingImages": [{"url": "https://news.example.com/a.jpg"}],
    "visuallySimilarImages": [{"url": "https://i.ytimg.com/vi/abcdefghijk/hq2.jpg"},
                              {"url": "https://i.ytimg.com/vi/WBDWyHIDpjA/hq1.jpg"}],
}


def _provider(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GoogleVisionWebProvider("k", client=client)


def test_video_ranking_puts_video_pages_then_thumbnails_then_pages():
    items = normalize_web_detection(WEB, "video")
    assert [i["url"] for i in items] == [
        "https://www.youtube.com/watch?v=WBDWyHIDpjA",
        "https://www.youtube.com/watch?v=abcdefghijk",
        "https://news.example.com/story",
    ]
    assert items[0]["title"] == "Original upload" and items[1]["kind"] == "similar_thumbnail"
    assert normalize_web_detection(WEB, "text")[0]["title"] == "Flood & rescue"
    assert [i["url"] for i in normalize_web_detection(WEB, "image")] == ["https://news.example.com/a.jpg"]
    assert youtube_id("https://youtu.be/WBDWyHIDpjA?t=3") == "WBDWyHIDpjA"


def test_search_returns_candidates_and_reuses_one_call_for_paging(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = json.loads(request.content)
        assert request.url.params["key"] == "k"
        assert body["requests"][0]["features"][0]["type"] == "WEB_DETECTION"
        return httpx.Response(200, json={"responses": [{"webDetection": WEB}]})

    image = tmp_path / "q.jpg"
    image.write_bytes(b"\xff\xd8fakejpeg")
    provider = _provider(handler)
    request = SearchQueryRequest("image", "E3", "video")
    first = asyncio.run(provider.search(request, offset=0, limit=2, image_path=image))
    second = asyncio.run(provider.search(request, offset=2, limit=2, image_path=image))
    assert len(calls) == 1
    assert [c.source_url for c in first.candidates] == ["https://www.youtube.com/watch?v=WBDWyHIDpjA",
                                                       "https://www.youtube.com/watch?v=abcdefghijk"]
    top = first.candidates[0]
    assert top.candidate_type == "video" and top.platform == "youtube" and top.platform_video_id == "WBDWyHIDpjA"
    assert "best guess for the image: flood rescue" in top.search_snippet
    assert [c.source_url for c in second.candidates] == ["https://news.example.com/story"]
    assert second.candidates[0].candidate_type == "page" and second.candidates[0].provider_rank == 3
    assert first.raw_response["best_guess"] == ["flood rescue"]
    assert provider.summary()["calls"] == 1 and provider.summary()["reused"] == 1


def test_text_image_ignores_text_and_errors_are_mapped(tmp_path):
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x")
    ok = _provider(lambda r: httpx.Response(200, json={"responses": [{}]}))
    batch = asyncio.run(ok.search(SearchQueryRequest("text_image", {"evidence_id": "E1", "text": "bridge"}, "video"),
                                  offset=0, limit=5, image_path=image))
    assert batch.candidates == () and batch.request_metadata["ignored_text"] is True
    bad = _provider(lambda r: httpx.Response(403, json={"error": {"message": "API not enabled"}}))
    with pytest.raises(SearchFailure) as info:
        asyncio.run(bad.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=image))
    assert info.value.error.code == "AUTH_ERROR"
    quota = _provider(lambda r: httpx.Response(429))
    with pytest.raises(SearchFailure) as info:
        asyncio.run(quota.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=image))
    assert info.value.error.code == "RATE_LIMITED"
    with pytest.raises(SearchFailure):
        asyncio.run(ok.search(SearchQueryRequest("text", "bridge", "video"), offset=0, limit=5, image_path=None))


def test_resolve_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        resolve_api_key(tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local" / "google_vision_key.txt").write_text("abc\n", encoding="utf-8")
    assert resolve_api_key(tmp_path) == "abc"
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "env")
    assert resolve_api_key(tmp_path) == "env"


def test_gcvimage_is_a_search_arm():
    from foren_searcher.search.arms import parse_search_arm

    assert parse_search_arm("gcvimage") == frozenset({"gcvimage"})
    assert parse_search_arm("yandeximage") == frozenset({"yandeximage"})
