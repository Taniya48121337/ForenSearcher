"""SEARCH Provider 适配器测试。

使用假 SDK 和响应检查路由参数、结果规范化、错误分类及 yt-dlp 分页去重。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.factory import create_search_service
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers.serpapi import SerpApiProvider
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider
from foren_searcher.store import GlobalStore

from search_helpers import LocalCaseServer


class _FakeSerpClient:
    """测试辅助类 _FakeSerpClient；用可控状态替代真实外部依赖。"""

    response: Mapping[str, Any] = {}
    uploads = 0
    calls: list[dict[str, object]] = []
    def search(self, params: dict[str, object]) -> Mapping[str, Any]:
        """模拟 search 接口，返回由当前用例预设的可控结果。"""

        self.calls.append(params); return self.response
    def upload_image(self, path: str) -> Mapping[str, Any]:
        """测试辅助函数 upload_image；为用例提供可重复的输入或可控行为。"""

        self.uploads += 1; return {"image_id": "uploaded-1"}


@pytest.mark.asyncio
async def test_serpapi_normalizes_web_image_video_and_lens_fixtures(tmp_path: Path) -> None:
    """验证 serpapi normalizes web image video and lens fixtures 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    client = _FakeSerpClient(); provider = SerpApiProvider("test", client=client)
    client.response = {"organic_results": [{"title": "Report", "link": "https://example.org/report", "snippet": "search-only"}]}
    web = await provider.search(SearchQueryRequest("text", "report", "text"), offset=0, limit=5)
    assert web.candidates[0].source_url == "https://example.org/report"
    client.response = {"images_results": [{"title": "Image", "original": "https://cdn.example.org/a.jpg", "link": "https://example.org/page", "thumbnail": "https://cdn.example.org/t.jpg"}]}
    images = await provider.search(SearchQueryRequest("text", "image", "image"), offset=0, limit=5)
    assert images.candidates[0].image_url.endswith("a.jpg") and images.candidates[0].source_url.endswith("/page")
    client.response = {"video_results": [{"title": "Video", "link": "https://youtube.com/watch?v=abc", "video_id": "abc", "channel": "Public"}]}
    videos = await provider.search(SearchQueryRequest("text", "video", "video"), offset=0, limit=5)
    assert videos.candidates[0].candidate_type == "video"
    query_image = tmp_path / "q.jpg"; query_image.write_bytes(b"image")
    client.response = {"visual_matches": [{"title": "Intermediate page", "link": "https://example.org/story"}]}
    lens = await provider.search(SearchQueryRequest("text_image", {"evidence_id": "E1", "text": "bridge"}, "video"), offset=0, limit=5, image_path=query_image)
    assert client.uploads == 1 and client.calls[-1]["q"] == "bridge"
    assert lens.request_metadata == {"image_id": "uploaded-1"}
    assert lens.candidates[0].candidate_type == "page"


@pytest.mark.asyncio
async def test_serpapi_empty_results_are_normal_and_auth_errors_are_mapped() -> None:
    """验证 serpapi empty results are normal and auth errors are mapped 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    client = _FakeSerpClient(); provider = SerpApiProvider("test", client=client)
    client.response = {"organic_results": []}
    batch = await provider.search(SearchQueryRequest("text", "nothing", "text"), offset=0, limit=5)
    assert batch.candidates == ()
    client.response = {"error": "Invalid API key: secret-value"}
    with pytest.raises(SearchFailure) as caught:
        await provider.search(SearchQueryRequest("text", "x", "text"), offset=0, limit=5)
    assert caught.value.error.code == "AUTH_ERROR" and "secret-value" not in caught.value.error.message


@pytest.mark.asyncio
async def test_serpapi_drops_provider_urls_with_embedded_credentials() -> None:
    """验证 serpapi drops provider urls with embedded credentials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    client = _FakeSerpClient(); provider = SerpApiProvider("test", client=client)
    client.response = {"organic_results": [
        {"title": "unsafe", "link": "https://user:secret@example.org/source"},
        {"title": "safe", "link": "https://example.org/source"},
    ]}
    batch = await provider.search(SearchQueryRequest("text", "source", "text"), offset=0, limit=5)
    assert [item.search_title for item in batch.candidates] == ["safe"]


@pytest.mark.asyncio
async def test_serpapi_sdk_adapter_flows_through_service_to_saved_evidence() -> None:
    """验证 serpapi sdk adapter flows through service to saved evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as source:
        client = _FakeSerpClient()
        client.response = {"organic_results": [{
            "title": "SerpApi fixture article", "link": source.base_url + "/article", "snippet": "candidate only",
        }]}
        store = GlobalStore()
        service = create_search_service(
            store=store, provider=SerpApiProvider("test", client=client), allow_private_network=True,
            policy=SearchPolicy(target_sources=1, maximum_provider_requests=1),
        )
        try:
            response = await service.execute(
                {"input_type": "text", "query": "fixture", "output_type": "text"}, workspace_id="W-serpapi-1",
            )
        finally:
            await service.aclose()
    assert response.stop_reason == "target_reached" and response.error is None
    card = response.results[0].materials[0].evidence[0]
    assert "English traceable paragraph" in store.evidence(card.evidence_id).content
    assert "candidate only" not in store.evidence(card.evidence_id).content


class _FakeYdlSearch:
    """测试辅助类 _FakeYdlSearch；用可控状态替代真实外部依赖。"""

    response: Mapping[str, Any] = {}
    last_options: Mapping[str, Any] = {}
    def __init__(self, options: Mapping[str, Any]) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""
        type(self).last_options = options
    def __enter__(self) -> Self:
        """测试辅助函数 __enter__；为用例提供可重复的输入或可控行为。"""
        return self
    def __exit__(self, *args: object) -> None:
        """模拟资源关闭，并记录生命周期状态供断言。"""
        return None
    def extract_info(self, query: str, *, download: bool) -> Mapping[str, Any]:
        """测试辅助函数 extract_info；为用例提供可重复的输入或可控行为。"""
        return self.response


@pytest.mark.asyncio
async def test_ytdlp_provider_filters_invalid_and_duplicate_video_entries() -> None:
    """验证 ytdlp provider filters invalid and duplicate video entries 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    _FakeYdlSearch.response = {"entries": [None, {"id": "abc1234", "title": "First", "channel": "C"},
        {"id": "abc1234", "title": "Duplicate"}, {"id": None, "title": "Bad"}]}
    provider = YtDlpSearchProvider(youtube_dl_factory=_FakeYdlSearch)
    batch = await provider.search(SearchQueryRequest("text", "public footage", "video"), offset=0, limit=5)
    assert [item.search_title for item in batch.candidates] == ["First"]
    assert _FakeYdlSearch.last_options["skip_download"] is True


@pytest.mark.asyncio
async def test_ytdlp_provider_accepts_empty_results() -> None:
    """验证 ytdlp provider accepts empty results 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    _FakeYdlSearch.response = {"entries": []}
    batch = await YtDlpSearchProvider(youtube_dl_factory=_FakeYdlSearch).search(
        SearchQueryRequest("text", "nothing", "video"), offset=0, limit=5)
    assert batch.candidates == ()
