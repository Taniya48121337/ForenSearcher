"""Yandex/IQDB 反向图片 Provider 测试。

用 MockTransport 构造上传、同主机跳转、HTML 结构、分页缓存、限流和响应体积边界。
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import httpx
import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.factory import create_search_service
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers.fixture import FixtureSearchProvider
from foren_searcher.search.providers.iqdb_reverse import IqdbReverseImageProvider
from foren_searcher.search.providers.routing import InputRoutingSearchProvider
from foren_searcher.search.providers.yandex_reverse import YandexReverseImageProvider
from foren_searcher.store import GlobalStore
from search_helpers import LocalCaseServer, png_bytes


def _image(tmp_path: Path) -> Path:
    """测试辅助函数 _image；为用例提供可重复的输入或可控行为。"""

    path = tmp_path / "query.jpg"
    path.write_bytes(png_bytes())
    return path


def _yandex_html(sites: list[dict[str, object]]) -> str:
    """测试辅助函数 _yandex_html；为用例提供可重复的输入或可控行为。"""

    state = {"initialState": {"cbirSites": {"sites": sites}}}
    return f'<html><div class="Root" id="ImagesApp-1" data-state="{escape(json.dumps(state), quote=True)}"></div></html>'


@pytest.mark.asyncio
async def test_yandex_normalizes_pages_images_text_filter_and_reuses_upload(tmp_path: Path) -> None:
    """验证 yandex normalizes pages images text filter and reuses upload 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    requests: list[httpx.Request] = []
    sites = [
        {
            "url": f"https://source.example/{index}", "title": f"Match {index}",
            "description": f"Context {index}", "domain": "source.example",
            "originalImage": {"url": f"https://cdn.example/{index}.jpg", "width": 800, "height": 600},
            "thumb": {"url": f"//thumb.example/{index}.jpg"},
        }
        for index in range(4)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        requests.append(request)
        return httpx.Response(200, text=_yandex_html(sites), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = YandexReverseImageProvider(client=client)
        request = SearchQueryRequest("text_image", {"evidence_id": "E1", "text": "bridge"}, "image")
        first = await provider.search(request, offset=0, limit=2, image_path=_image(tmp_path))
        second = await provider.search(request, offset=2, limit=2, image_path=_image(tmp_path))

    assert len(requests) == 1
    assert requests[0].url.params["text"] == "bridge"
    assert requests[0].headers["content-type"].startswith("multipart/form-data")
    assert [item.image_url for item in first.candidates] == ["https://cdn.example/0.jpg", "https://cdn.example/1.jpg"]
    assert first.candidates[0].source_url == "https://source.example/0"
    assert first.candidates[0].thumbnail_url == "https://thumb.example/0.jpg"
    assert [item.provider_rank for item in second.candidates] == [3, 4]
    assert first.request_metadata["uploaded"] is True
    assert second.request_metadata["reused_provider_result"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "code"), [
    (403, "ACCESS_RESTRICTED"), (429, "RATE_LIMITED"), (500, "PROVIDER_ERROR"),
])
async def test_yandex_maps_upload_failures(tmp_path: Path, status: int, code: str) -> None:
    """验证 yandex maps upload failures 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(status, headers={"Retry-After": "2"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = YandexReverseImageProvider(client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(
                SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=_image(tmp_path),
            )
    assert caught.value.error.code == code
    if status == 429:
        assert caught.value.retry_after_s == 2


@pytest.mark.asyncio
async def test_yandex_rejects_unrecognized_result_structure(tmp_path: Path) -> None:
    """验证 yandex rejects unrecognized result structure 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        return httpx.Response(200, text="<html>changed</html>", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = YandexReverseImageProvider(client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(
                SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=_image(tmp_path),
            )
    assert caught.value.error.code == "PROVIDER_ERROR"


IQDB_HTML = """
<html><div id="pages">
  <div><table><tr><th>Your image</th></tr><tr><td><img src="/thu/query.jpg"></td></tr></table></div>
  <div><table>
    <tr><th>Best match</th></tr>
    <tr><td><a href="//danbooru.example/posts/42"><img src="/thumb/match.jpg"></a></td></tr>
    <tr><td><img src="/icon.png"> Danbooru</td></tr>
    <tr><td>1200x900 [Safe]</td></tr>
    <tr><td>94.5% similarity</td></tr>
  </table></div>
  <div><table><tr><th>No relevant matches</th></tr><tr></tr><tr></tr><tr></tr><tr></tr></table></div>
</div></html>
"""


@pytest.mark.asyncio
async def test_iqdb_normalizes_similarity_and_reuses_upload(tmp_path: Path) -> None:
    """验证 iqdb normalizes similarity and reuses upload 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

        nonlocal calls
        calls += 1
        return httpx.Response(200, text=IQDB_HTML, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = IqdbReverseImageProvider(client=client)
        request = SearchQueryRequest("image", "E1", "image")
        first = await provider.search(request, offset=0, limit=1, image_path=_image(tmp_path))
        empty = await provider.search(request, offset=1, limit=1, image_path=_image(tmp_path))

    assert calls == 1 and empty.candidates == ()
    candidate = first.candidates[0]
    assert candidate.source_url == "https://danbooru.example/posts/42"
    assert candidate.image_url == "https://iqdb.org/thumb/match.jpg"
    assert candidate.search_snippet == "Best match; 94.5% similarity"
    assert candidate.extra["similarity"] == 94.5


@pytest.mark.asyncio
async def test_iqdb_rejects_combined_text_query_without_silent_degradation(tmp_path: Path) -> None:
    """验证 iqdb rejects combined text query without silent degradation 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        provider = IqdbReverseImageProvider(client=client)
        with pytest.raises(SearchFailure) as caught:
            await provider.search(
                SearchQueryRequest("text_image", {"evidence_id": "E1", "text": "character"}, "text"),
                offset=0, limit=5, image_path=_image(tmp_path),
            )
    assert caught.value.error.code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_input_router_preserves_one_search_facade(tmp_path: Path) -> None:
    """验证 input router preserves one search facade 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    text = FixtureSearchProvider(())
    image = FixtureSearchProvider(())
    provider = InputRoutingSearchProvider(text_provider=text, image_provider=image)
    await provider.search(SearchQueryRequest("text", "report", "text"), offset=0, limit=5)
    await provider.search(SearchQueryRequest("image", "E1", "image"), offset=0, limit=5, image_path=_image(tmp_path))
    assert len(text.calls) == 1 and len(image.calls) == 1


@pytest.mark.asyncio
async def test_yandex_reverse_result_flows_from_evidence_to_new_evidence() -> None:
    """验证 yandex reverse result flows from evidence to new evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as source:
        site = {
            "url": f"{source.base_url}/article", "title": "Matched source", "description": "Visual match",
            "domain": "127.0.0.1", "originalImage": {"url": f"{source.base_url}/image", "width": 32, "height": 24},
            "thumb": {"url": f"{source.base_url}/image"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            """测试辅助函数 handler；为用例提供可重复的输入或可控行为。"""

            return httpx.Response(200, text=_yandex_html([site]), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = YandexReverseImageProvider(client=client)
            store = GlobalStore()
            candidate_id, _ = store.get_or_create_candidate(
                identity_key="image:query", source_url=source.base_url + "/image", candidate_type="image",
            )
            query = store.commit_material_evidence(
                candidate_id=candidate_id, material_kind="image", material_status="ready",
                artifact_bytes=png_bytes(), media_type="image/png", suffix=".png", modalities=("image",),
                label="query image", content='{"width":32,"height":24}', locator="complete image",
                acquired_by="test", discovered_by="W0-1",
            )
            policy = SearchPolicy(initial_candidates=5, maximum_candidates=5, target_sources=1, maximum_provider_requests=1)
            service = create_search_service(
                store=store, provider=provider, policy=policy, allow_private_network=True,
            )
            try:
                response = await service.execute(
                    {"input_type": "image", "query": query.evidence_id, "output_type": "text"},
                    workspace_id="W-reverse-1",
                )
            finally:
                await service.aclose()

    assert response.stop_reason == "target_reached", response.to_dict()
    assert response.results[0].search_title == "Matched source"
    evidence = response.results[0].materials[0].evidence[0]
    assert "English traceable paragraph" in store.evidence(evidence.evidence_id).content
