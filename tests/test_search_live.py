"""显式开启才运行的真实公网 SEARCH 检查。

第三方可用性和排名会变化，所以这些测试只验证接口/安全边界，不进入默认确定性套件。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest

from foren_searcher.search.contracts import SearchQueryRequest
from foren_searcher.search.factory import (
    create_search_service,
    create_searxng_search_service,
    create_serpapi_search_service,
    create_yandex_reverse_image_search_service,
)
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers.fixture import FixtureSearchProvider
from foren_searcher.search.providers.iqdb_reverse import IqdbReverseImageProvider
from foren_searcher.search.providers.searxng import SearXngProvider
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider
from foren_searcher.store import GlobalStore


pytestmark = [pytest.mark.live, pytest.mark.skipif(os.getenv("RUN_LIVE_SEARCH_TESTS") != "1", reason="set RUN_LIVE_SEARCH_TESTS=1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("url", "output_type", "expected_kind"), [
    ("https://www.iana.org/help/example-domains", "text", "page"),
    ("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "text", "document"),
    ("https://www.w3.org/Icons/w3c_home.png", "image", "image"),
])
async def test_real_static_source_matrix(url: str, output_type: str, expected_kind: str) -> None:
    """验证 real static source matrix 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(); service = create_search_service(store=store, provider=FixtureSearchProvider(()))
    try:
        response = await service.execute({"source": url, "output_type": output_type}, workspace_id="W-live-1")
    finally:
        await service.aclose()
    assert response.stop_reason == "target_reached", response.to_dict()
    material = next(item for item in response.results[0].materials if item.kind == expected_kind)
    assert material.status in {"ready", "partial"} and material.evidence
    assert store.artifact_path(store.evidence(material.evidence[0].evidence_id).artifact_ref).is_file()


@pytest.mark.asyncio
async def test_real_public_video_source_reports_actual_environment_result() -> None:
    """验证 real public video source reports actual environment result 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    url = "https://vimeo.com/76979871"
    store = GlobalStore(); service = create_search_service(store=store, provider=FixtureSearchProvider(()))
    try:
        response = await service.execute({"source": url, "output_type": "video"}, workspace_id="W-live-2")
    finally:
        await service.aclose()
    assert response.results and response.results[0].video_material is not None
    if response.stop_reason == "target_reached":
        page = response.results[0].materials[0]
        assert page.status in {"ready", "partial"} and page.evidence
    else:
        assert any(item.error is not None for item in response.results[0].materials)


@pytest.mark.asyncio
async def test_real_javascript_page_uses_isolated_browser_fallback() -> None:
    """验证 real javascript page uses isolated browser fallback 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    url = "https://quotes.toscrape.com/js/"
    store = GlobalStore(); service = create_search_service(store=store, provider=FixtureSearchProvider(()), enable_dynamic_pages=True)
    try:
        response = await service.execute({"source": url, "output_type": "text"}, workspace_id="W-live-3")
    finally:
        await service.aclose()
    assert response.stop_reason == "target_reached", response.to_dict()
    evidence = store.evidence(response.results[0].materials[0].evidence[0].evidence_id)
    assert "Albert Einstein" in evidence.content


@pytest.mark.asyncio
async def test_real_text_to_video_search_through_replaceable_provider() -> None:
    """验证 real text to video search through replaceable provider 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    policy = SearchPolicy(initial_candidates=3, maximum_candidates=3, target_sources=1,
        maximum_provider_requests=1, search_total_s=60.0, single_source_total_s=40.0)
    store = GlobalStore(); service = create_search_service(store=store, provider=YtDlpSearchProvider(), policy=policy)
    try:
        response = await service.execute({"input_type": "text", "query": "yt-dlp test video", "output_type": "video"}, workspace_id="W-live-4")
    finally:
        await service.aclose()
    assert response.error is None and response.results, response.to_dict()
    assert any(result.materials[0].evidence for result in response.results)


@pytest.mark.searxng
@pytest.mark.skipif(not os.getenv("SEARXNG_BASE_URL"), reason="SEARXNG_BASE_URL is not configured")
@pytest.mark.parametrize(("query", "output_type", "expected_type"), [
    ("!github huggingface transformers", "text", "page"),
    ("red panda", "image", "image"),
    ("yt-dlp test video", "video", "video"),
])
@pytest.mark.asyncio
async def test_real_searxng_diverse_result_types(query: str, output_type: str, expected_type: str) -> None:
    """验证 real searxng diverse result types 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    provider = SearXngProvider(os.environ["SEARXNG_BASE_URL"], timeout_s=45.0)
    try:
        batch = await provider.search(SearchQueryRequest("text", query, output_type), offset=0, limit=5)  # type: ignore[arg-type]
    finally:
        await provider.aclose()
    assert batch.candidates, batch.raw_response
    assert all(candidate.candidate_type == expected_type for candidate in batch.candidates)
    if output_type == "image":
        assert all(candidate.image_url for candidate in batch.candidates)


@pytest.mark.searxng
@pytest.mark.skipif(not os.getenv("SEARXNG_BASE_URL"), reason="SEARXNG_BASE_URL is not configured")
@pytest.mark.asyncio
async def test_real_searxng_result_opens_and_commits_evidence() -> None:
    """验证 real searxng result opens and commits evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    policy = SearchPolicy(initial_candidates=3, maximum_candidates=3, target_sources=1,
        maximum_provider_requests=1, provider_timeout_s=45.0, search_total_s=90.0)
    store = GlobalStore()
    service = create_searxng_search_service(
        store=store, base_url=os.environ["SEARXNG_BASE_URL"], policy=policy,
    )
    try:
        response = await service.execute(
            {"input_type": "text", "query": "!github huggingface transformers", "output_type": "text"},
            workspace_id="W-live-searxng",
        )
    finally:
        await service.aclose()
    assert response.error is None and response.results, response.to_dict()
    assert any(any(material.evidence for material in result.materials) for result in response.results)


@pytest.mark.reverse_image
@pytest.mark.asyncio
async def test_real_yandex_reverse_image_upload_opens_a_source() -> None:
    """验证 real yandex reverse image upload opens a source 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, trust_env=False) as client:
        downloaded = await client.get("https://www.w3.org/Icons/w3c_home.png")
        downloaded.raise_for_status()
    candidate_id, _ = store.get_or_create_candidate(
        identity_key="image:w3c-public-fixture", source_url="https://www.w3.org/Icons/w3c_home.png",
        candidate_type="image",
    )
    query = store.commit_material_evidence(
        candidate_id=candidate_id, material_kind="image", material_status="ready",
        artifact_bytes=downloaded.content, media_type="image/png", suffix=".png", modalities=("image",),
        label="W3C public test image", content='{"source":"W3C public fixture"}', locator="complete image",
        acquired_by="SEARCH/open", discovered_by="W-live-reverse-source",
    )
    query_id = query.evidence_id
    policy = SearchPolicy(initial_candidates=3, maximum_candidates=3, target_sources=1,
        maximum_provider_requests=1, provider_timeout_s=45.0, search_total_s=90.0)
    reverse = create_yandex_reverse_image_search_service(store=store, policy=policy)
    try:
        response = await reverse.execute(
            {"input_type": "image", "query": query_id, "output_type": "text"},
            workspace_id="W-live-reverse-yandex",
        )
    finally:
        await reverse.aclose()
    assert response.error is None and response.results, response.to_dict()
    assert any(any(material.evidence for material in result.materials) for result in response.results)


@pytest.mark.reverse_image
@pytest.mark.asyncio
async def test_real_iqdb_artwork_reverse_image_upload() -> None:
    """验证 real iqdb artwork reverse image upload 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    image_url = "https://raw.githubusercontent.com/kitUIN/PicImageSearch/main/demo/images/test01.jpg"
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, trust_env=False) as client:
        response = await client.get(image_url)
        response.raise_for_status()
    with tempfile.TemporaryDirectory(prefix="foren-iqdb-live-") as directory:
        image_path = Path(directory) / "iqdb-public-fixture.jpg"
        image_path.write_bytes(response.content)
        provider = IqdbReverseImageProvider(timeout_s=45.0)
        try:
            batch = await provider.search(
                SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=image_path,
            )
        finally:
            await provider.aclose()
    assert batch.candidates, batch.raw_response
    assert batch.candidates[0].extra["similarity"] >= 80


@pytest.mark.asyncio
@pytest.mark.serpapi
@pytest.mark.skipif(not os.getenv("SERPAPI_API_KEY"), reason="SERPAPI_API_KEY is not configured")
async def test_real_serpapi_text_search_and_open() -> None:
    """验证 real serpapi text search and open 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(); service = create_serpapi_search_service(store=store, api_key=os.environ["SERPAPI_API_KEY"])
    try:
        response = await service.execute({"input_type": "text", "query": "site:iana.org example domains", "output_type": "text"}, workspace_id="W-live-3")
    finally:
        await service.aclose()
    assert response.error is None and response.results
    assert any(any(material.evidence for material in result.materials) for result in response.results)
