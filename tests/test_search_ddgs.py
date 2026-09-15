"""免费候选适配、分页和失败处理。"""

import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.ddgs import DdgsProvider


class Client:
    def __init__(self, **kwargs):
        pass

    def text(self, query, **kwargs):
        assert kwargs["backend"] == "brave"
        assert kwargs["region"] == "cn-zh"
        return [{"title": f"资料{i}", "href": f"https://example.org/{i}", "body": "中文正文摘要"} for i in range(6)]

    def images(self, query, **kwargs):
        return [{"title": "Photo", "url": "https://example.org/page", "image": "https://example.org/photo.png"}]

    def videos(self, query, **kwargs):
        return [{"title": "Film", "content": "https://www.youtube.com/watch?v=abc", "description": "Original film", "images": {"large": "https://example.org/preview.jpg"}}]


@pytest.mark.asyncio
async def test_text_query_preserves_source_and_offset():
    result = await DdgsProvider(client_factory=Client).search(
        SearchQueryRequest("text", "中文事件日期", "text"), offset=2, limit=2)
    assert [x.provider_rank for x in result.candidates] == [3, 4]
    assert [x.source_url for x in result.candidates] == ["https://example.org/2", "https://example.org/3"]
    assert all(x.provider == "ddgs" and x.engine == "brave" for x in result.candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["image", "video"])
async def test_media_search_preserves_page_and_media_links(output):
    result = await DdgsProvider(client_factory=Client).search(
        SearchQueryRequest("text", "NASA astronaut", output), offset=0, limit=5)
    assert len(result.candidates) == 1
    assert result.candidates[0].candidate_type == output
    if output == "image":
        assert result.candidates[0].image_url == "https://example.org/photo.png"
    else:
        assert result.candidates[0].platform_video_id == "abc"


@pytest.mark.asyncio
async def test_provider_failure_is_explicit_and_does_not_use_fixture():
    class Broken(Client):
        def text(self, *args, **kwargs):
            raise TimeoutError("private debug details")
    with pytest.raises(SearchFailure) as caught:
        await DdgsProvider(client_factory=Broken).search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert caught.value.error.code == "TIMEOUT"
    assert "private debug" not in str(caught.value)


@pytest.mark.asyncio
async def test_empty_primary_search_uses_free_fallback_and_keeps_engine_provenance():
    calls = []
    class Fallback(Client):
        def text(self, query, **kwargs):
            calls.append(kwargs["backend"])
            if kwargs["backend"] == "brave":
                return []
            return [{"title":"Official source", "href":"https://example.org/source", "body":"Source excerpt"}]
    result = await DdgsProvider(client_factory=Fallback).search(
        SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert calls == ["brave", "bing"]
    assert result.candidates[0].engine == "bing"
    assert result.candidates[0].source_url == "https://example.org/source"
