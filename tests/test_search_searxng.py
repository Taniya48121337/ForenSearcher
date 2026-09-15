"""SearXNG Provider 和免费组合工厂测试。

验证分类/页码参数、各种结果类型、响应上限、HTTP 错误与输入能力路由。
"""

from __future__ import annotations

import json

import httpx
import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.factory import create_search_service
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers.searxng import SearXngProvider
from foren_searcher.store import GlobalStore
from search_helpers import LocalCaseServer


def _client(handler: object) -> httpx.AsyncClient:
    """测试辅助函数 _client；为用例提供可重复的输入或可控行为。"""

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_general_results_are_normalized_and_paged_without_skipping() -> None:
    """验证 general results are normalized and paged without skipping 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        requests.append(request)
        values = [
            {"url": f"https://example.org/{index}", "title": f"Title {index}",
             "content": f"Excerpt {index}", "engines": ["brave", "duckduckgo"], "score": index}
            for index in range(10)
        ]
        values.append({"url": "file:///blocked", "title": "invalid"})
        return httpx.Response(200, json={"results": values}, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example/base/", client=client, language="zh-CN", safe_search=2)
        first = await provider.search(SearchQueryRequest("text", "事故 通报", "text"), offset=0, limit=5)
        second = await provider.search(SearchQueryRequest("text", "事故 通报", "text"), offset=5, limit=5)

    assert [item.source_url for item in first.candidates] == [f"https://example.org/{index}" for index in range(5)]
    assert [item.source_url for item in second.candidates] == [f"https://example.org/{index}" for index in range(5, 10)]
    assert first.candidates[0].engine == "brave,duckduckgo"
    assert first.candidates[0].search_snippet == "Excerpt 0"
    assert first.candidates[0].extra == {"score": 0}
    assert requests[0].url.path == "/base/search"
    assert requests[0].url.params["format"] == "json"
    assert requests[0].url.params["categories"] == "general"
    assert requests[0].url.params["language"] == "zh-CN"
    assert requests[0].url.params["safesearch"] == "2"
    assert requests[1].url.params["pageno"] == "1"


@pytest.mark.asyncio
async def test_image_and_video_results_keep_source_identity() -> None:
    """验证 image and video results keep source identity 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    payloads = {
        "images": {"results": [{
            "url": "https://museum.example/object/4", "img_src": "https://cdn.example/full.jpg",
            "thumbnail_src": "https://cdn.example/thumb.jpg", "title": "Object", "engine": "wikimedia",
        }]},
        "videos": {"results": [{
            "url": "https://www.youtube.com/watch?v=abc123", "thumbnail": "https://img.example/abc.jpg",
            "title": "Clip", "template": "videos.html", "engine": "youtube",
        }]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(200, json=payloads[request.url.params["categories"]], request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client)
        image = await provider.search(SearchQueryRequest("text", "artifact", "image"), offset=0, limit=5)
        video = await provider.search(SearchQueryRequest("text", "public clip", "video"), offset=0, limit=5)

    assert image.candidates[0].candidate_type == "image"
    assert image.candidates[0].source_url == "https://museum.example/object/4"
    assert image.candidates[0].image_url == "https://cdn.example/full.jpg"
    assert image.candidates[0].thumbnail_url == "https://cdn.example/thumb.jpg"
    assert video.candidates[0].candidate_type == "video"
    assert video.candidates[0].platform == "youtube"
    assert video.candidates[0].platform_video_id == "abc123"


@pytest.mark.asyncio
async def test_search_navigation_junk_is_filtered_unless_requested() -> None:
    """实例返回搜索引擎首页时不把它当成任务证据。"""

    payload = {"results": [
        {"url": "https://www.google.com/", "title": "Google", "content": "Search"},
        {
            "url": "https://www.archives.gov/example",
            "title": "Inaugural address",
            "content": "Official record",
        },
    ]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client)
        result = await provider.search(
            SearchQueryRequest("text", "JFK inaugural address", "text"),
            offset=0,
            limit=5,
        )

    assert [item.source_url for item in result.candidates] == [
        "https://www.archives.gov/example"
    ]


@pytest.mark.asyncio
async def test_non_text_query_is_rejected_without_network_request() -> None:
    """验证 non text query is rejected without network request 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        nonlocal called
        called = True
        return httpx.Response(500, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(SearchQueryRequest("image", "E1", "text"), offset=0, limit=5)
    assert caught.value.error.code == "INVALID_INPUT"
    assert not called


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected"), [
    (401, "AUTH_ERROR"), (403, "ACCESS_RESTRICTED"), (429, "RATE_LIMITED"), (503, "PROVIDER_ERROR"),
])
async def test_http_failures_are_mapped(status: int, expected: str) -> None:
    """验证 http failures are mapped 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(status, headers={"Retry-After": "1.5"}, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert caught.value.error.code == expected
    if status == 429:
        assert caught.value.retry_after_s == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize(("body", "expected_message"), [
    (b"not-json", "invalid JSON"),
    (json.dumps({"results": {}}).encode(), "not a list"),
])
async def test_malformed_provider_responses_are_rejected(body: bytes, expected_message: str) -> None:
    """验证 malformed provider responses are rejected 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(200, content=body, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert caught.value.error.code == "PROVIDER_ERROR"
    assert expected_message in caught.value.error.message


@pytest.mark.asyncio
async def test_timeout_and_response_size_are_bounded() -> None:
    """验证 timeout and response size are bounded 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    def timeout(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 timeout；为用例提供可重复的输入或可控行为。"""

        raise httpx.ReadTimeout("slow", request=request)

    async with _client(timeout) as client:
        provider = SearXngProvider("https://search.example", client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert caught.value.error.code == "TIMEOUT"

    def large(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 large；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(200, content=b"{} " * 20, request=request)

    async with _client(large) as client:
        provider = SearXngProvider("https://search.example", client=client, max_response_bytes=10)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert "size limit" in caught.value.error.message


@pytest.mark.asyncio
async def test_chunked_provider_body_is_stopped_at_size_limit() -> None:
    """验证 chunked provider body is stopped at size limit 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    class Body(httpx.AsyncByteStream):
        """测试辅助类 Body；用可控状态替代真实外部依赖。"""

        def __init__(self) -> None:
            """保存测试替身需要的预设输入、状态和调用记录。"""
            self.chunks_read = 0
        async def __aiter__(self):
            """测试辅助函数 __aiter__；为用例提供可重复的输入或可控行为。"""

            for chunk in (b'{"results":', b" " * 20, b"must-not-be-read"):
                self.chunks_read += 1
                yield chunk

    body = Body()

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(200, stream=body, request=request)

    async with _client(handler) as client:
        provider = SearXngProvider("https://search.example", client=client, max_response_bytes=15)
        with pytest.raises(SearchFailure, match="size limit"):
            await provider.search(SearchQueryRequest("text", "query", "text"), offset=0, limit=5)
    assert body.chunks_read == 2


@pytest.mark.asyncio
async def test_searxng_result_flows_through_search_service_to_evidence() -> None:
    """验证 searxng result flows through search service to evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as source:
        provider = SearXngProvider(source.base_url)
        policy = SearchPolicy(initial_candidates=5, maximum_candidates=5, target_sources=1, maximum_provider_requests=1)
        store = GlobalStore()
        service = create_search_service(
            store=store, provider=provider, policy=policy, allow_private_network=True,
        )
        try:
            response = await service.execute(
                {"input_type": "text", "query": "local", "output_type": "text"},
                workspace_id="W-searxng-1",
            )
        finally:
            await service.aclose()

    assert response.stop_reason == "target_reached", response.to_dict()
    assert response.results[0].search_title == "Local SearXNG article"
    evidence = response.results[0].materials[0].evidence[0]
    assert "English traceable paragraph" in store.evidence(evidence.evidence_id).content


def test_searxng_base_url_rejects_embedded_credentials() -> None:
    """验证 searxng base url rejects embedded credentials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="credentials"):
        SearXngProvider("https://user:secret@search.example.org/")
