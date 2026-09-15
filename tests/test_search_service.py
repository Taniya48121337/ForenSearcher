"""SearchService 主编排测试。

通过假 Provider/Opener 覆盖候选去重、并发、缓存、重试、超时、取消、保存失败和审计计数。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import ssl
from contextlib import closing
import pytest

from foren_searcher.search.contracts import CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchQueryRequest
from foren_searcher.search.policy import SearchPolicy

from search_helpers import LocalCaseServer, make_service


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type,code", [(ssl.SSLCertVerificationError, "AUTH_ERROR"),
                                              (ValueError, "EXTRACT_FAILED")])
async def test_unclassified_source_failure_preserves_concurrent_successes(failure_type, code):
    with LocalCaseServer() as server:
        candidates = tuple(CandidateSource(server.base_url + path, candidate_type="page")
                           for path in ("/article?first", "/broken", "/article?last"))
        service, store, _provider = make_service(candidates,
            policy=SearchPolicy(target_sources=2, source_concurrency=2))
        original = service.opener.open
        first_opened = asyncio.Event()

        async def open_source(candidate, request):
            if candidate.source_url.endswith("/broken"):
                await first_opened.wait()
                raise failure_type("private exception detail")
            result = await original(candidate, request)
            first_opened.set()
            return result

        service.opener.open = open_source
        try:
            response = await service.execute({"input_type":"text", "query":"mixed sources",
                                              "output_type":"text"}, workspace_id="W1-1")
        finally:
            await service.aclose()
    assert response.stop_reason == "target_reached" and response.error is None
    assert len(response.results) == 3
    failed = response.results[1].materials[0]
    assert failed.status == "failed" and failed.error.code == code and not failed.evidence
    assert "private exception detail" not in failed.error.message
    for index in (0, 2):
        result = response.results[index]
        assert result.materials[0].status == "ready"
        assert store.candidate_evidence(result.candidate_id)
    attempts = store.source_open_attempts(1)
    assert len(attempts) == 3
    assert sorted(item["outcome"] for item in attempts) == ["error", "success", "success"]
    assert next(item for item in attempts if item["outcome"] == "error")["error"]["code"] == code


@pytest.mark.asyncio
async def test_text_query_runs_provider_open_artifact_evidence_and_uniform_cards() -> None:
    """验证 text query runs provider open artifact evidence and uniform cards 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        candidates = tuple(CandidateSource(f"{server.base_url}/article?case={index}", f"Case {index}", "provider snippet",
            "fixture", "fixture", index, "page") for index in range(1, 6))
        service, store, provider = make_service(candidates)
        try:
            response = await service.execute({"input_type": "text", "query": "diverse cases", "output_type": "text"}, workspace_id="W1-1")
        finally:
            await service.aclose()
    assert response.stop_reason == "target_reached" and response.error is None
    assert len(response.results) == 5 and len(provider.calls) == 1
    cards = [result.materials[0].evidence[0] for result in response.results]
    assert all(set(card.to_dict()) == {"evidence_id", "modalities", "label", "source_ref", "derived_from", "acquired_by"} for card in cards)
    assert all(store.artifact_path(store.evidence(card.evidence_id).artifact_ref).is_file() for card in cards)
    assert all("provider snippet" not in store.evidence(card.evidence_id).content for card in cards)


@pytest.mark.asyncio
async def test_failures_trigger_one_supplement_batch_without_losing_successes() -> None:
    """验证 failures trigger one supplement batch without losing successes 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        paths = ["/missing", "/restricted", "/article?a=3", "/soft404", "/article?a=5",
                 "/article?a=6", "/article?a=7", "/article?a=8", "/article?a=9", "/article?a=10"]
        candidates = tuple(CandidateSource(server.base_url + path, provider_rank=index, candidate_type="page") for index, path in enumerate(paths, 1))
        service, store, provider = make_service(candidates)
        try:
            response = await service.execute({"input_type": "text", "query": "mixed outcomes", "output_type": "text"}, workspace_id="W1-2")
        finally:
            await service.aclose()
    assert response.stop_reason == "target_reached"
    assert len(provider.calls) == 2 and len(response.results) == 10
    assert response.results[0].materials[0].status == "failed"
    assert response.results[2].materials[0].status == "ready"
    assert all(item.materials[0].status == "not_requested" for item in response.results[8:])
    assert len(store.candidate_evidence(response.results[2].candidate_id)) == 1


@pytest.mark.asyncio
async def test_known_url_and_r_reuse_same_ids_without_refetch() -> None:
    """验证 known url and r reuse same ids without refetch 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, store, _provider = make_service(())
        try:
            first = await service.execute({"source": server.base_url + "/article", "output_type": "text"}, workspace_id="W1-1")
            second = await service.execute({"source": first.results[0].candidate_id, "output_type": "text"}, workspace_id="W1-2")
        finally:
            await service.aclose()
    assert first.results[0].candidate_id == second.results[0].candidate_id
    assert first.results[0].materials[0].evidence[0].evidence_id == second.results[0].materials[0].evidence[0].evidence_id
    assert server.counts["/article"] == 1


@pytest.mark.asyncio
async def test_final_redirect_url_reuses_the_original_candidate_id() -> None:
    """验证 final redirect url reuses the original candidate id 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, _store, _provider = make_service(())
        try:
            redirected = await service.execute({"source": server.base_url + "/redirect", "output_type": "text"}, workspace_id="W1-1")
            final = await service.execute({"source": server.base_url + "/article", "output_type": "text"}, workspace_id="W1-2")
        finally:
            await service.aclose()
    assert redirected.results[0].candidate_id == final.results[0].candidate_id == "R1"
    assert server.counts["/article"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 3])
async def test_redirect_aliases_count_once_and_allow_later_distinct_sources(concurrency):
    with LocalCaseServer() as server:
        candidates = tuple(CandidateSource(server.base_url + path, candidate_type="page")
                           for path in ("/article", "/redirect?first", "/redirect?second", "/independent"))
        service, store, provider = make_service(candidates, policy=SearchPolicy(
            initial_candidates=3, maximum_candidates=4, target_sources=2,
            source_concurrency=concurrency))
        try:
            response = await service.execute({"input_type":"text", "query":"source documents",
                                              "output_type":"text"}, workspace_id="W1-1")
        finally:
            await service.aclose()
    assert response.stop_reason == "target_reached"
    assert len(provider.calls) == 2
    assert len(response.results) == 4
    assert all(result.materials[0].status == "ready" for result in response.results)
    assert len({result.candidate_id for result in response.results[:3]}) == 1
    assert response.results[3].candidate_id != response.results[0].candidate_id
    assert len({result.materials[0].evidence[0].evidence_id for result in response.results[:3]}) == 1
    with closing(store._connect()) as connection:
        count = connection.execute("SELECT successful_source_count FROM search_executions").fetchone()[0]
    assert count == 2
    assert store.healthcheck().to_dict()["status"] == "ok"


@pytest.mark.asyncio
async def test_direct_image_keeps_image_identity_and_does_not_invent_page_evidence() -> None:
    """验证 direct image keeps image identity and does not invent page evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, store, _provider = make_service(())
        try:
            response = await service.execute({"source": server.base_url + "/image", "output_type": "image"}, workspace_id="W2-1")
        finally:
            await service.aclose()
    materials = {item.kind: item for item in response.results[0].materials}
    assert response.stop_reason == "target_reached"
    assert materials["image"].status == "ready" and materials["page"].status == "not_requested"
    evidence = store.evidence(materials["image"].evidence[0].evidence_id)
    assert evidence.modalities == ("image",) and evidence.production_method == "source_material"


@pytest.mark.asyncio
async def test_scanned_pdf_is_retained_but_does_not_count_as_acquired_text() -> None:
    """验证 scanned pdf is retained but does not count as acquired text 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, store, _provider = make_service(())
        try:
            response = await service.execute({"source": server.base_url + "/scan-pdf", "output_type": "text"}, workspace_id="W2-2")
        finally:
            await service.aclose()
    document = next(item for item in response.results[0].materials if item.kind == "document")
    assert response.stop_reason == "candidates_exhausted"
    assert document.status == "partial" and document.evidence
    assert store.artifact_path(store.evidence(document.evidence[0].evidence_id).artifact_ref).is_file()


@pytest.mark.asyncio
async def test_empty_results_are_normal_candidates_exhausted() -> None:
    """验证 empty results are normal candidates exhausted 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    service, _store, _provider = make_service(())
    try:
        response = await service.execute({"input_type": "text", "query": "nothing", "output_type": "text"}, workspace_id="W3-1")
    finally:
        await service.aclose()
    assert response.stop_reason == "candidates_exhausted" and response.error is None and response.results == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"source": "https://example.org", "output_type": []},
    {"source": "https://example.org", "output_type": "video", "include": [["subtitles"]]},
    {"source": "https://example.org", "output_type": "video", "include": ["subtitles"], "subtitle_language": 7},
])
async def test_malformed_runtime_payloads_return_invalid_input_instead_of_raising(payload: dict[str, object]) -> None:
    """验证 malformed runtime payloads return invalid input instead of raising 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    service, _store, _provider = make_service(())
    try:
        response = await service.execute(payload, workspace_id="W3-2")
    finally:
        await service.aclose()
    assert response.stop_reason == "error" and response.error.code == "INVALID_INPUT"


class _AlwaysRateLimited:
    """测试辅助类 _AlwaysRateLimited；用可控状态替代真实外部依赖。"""

    calls = 0
    async def search(self, *args: object, **kwargs: object) -> object:
        """模拟 search 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        raise SearchFailure(SearchError("search", "RATE_LIMITED", "rate limited"), retry_after_s=None)


class _SupplementFails:
    """测试辅助类 _SupplementFails；用可控状态替代真实外部依赖。"""

    def __init__(self, candidate: CandidateSource) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.candidate = candidate
        self.calls = 0

    async def search(self, request: object, *, offset: int, limit: int, image_path: object = None) -> CandidateBatch:
        """模拟 search 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            return CandidateBatch("fixture", "web", (self.candidate,), {"batch": 1})
        raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "supplement unavailable"))


@pytest.mark.asyncio
async def test_provider_rate_limit_is_top_level_and_does_not_create_evidence() -> None:
    """验证 provider rate limit is top level and does not create evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.factory import create_search_service
    from foren_searcher.store import GlobalStore
    store = GlobalStore(); provider = _AlwaysRateLimited()
    service = create_search_service(store=store, provider=provider)
    try:
        response = await service.execute({"input_type": "text", "query": "x", "output_type": "text"}, workspace_id="W4-1")
    finally:
        await service.aclose()
    assert response.stop_reason == "error" and response.error.code == "RATE_LIMITED"
    assert provider.calls == 1 and not store.has_reference("E1")
    with closing(sqlite3.connect(store.db_path)) as connection:
        assert connection.execute("SELECT provider_request_count FROM search_executions").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_unknown_candidate_reference_is_invalid_input_not_save_failure() -> None:
    """验证 unknown candidate reference is invalid input not save failure 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    service, _store, _provider = make_service(())
    try:
        response = await service.execute({"source": "R999", "output_type": "text"}, workspace_id="W4-2")
    finally:
        await service.aclose()
    assert response.stop_reason == "error"
    assert response.error.stage == "validate" and response.error.code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_supplement_provider_failure_retains_committed_initial_results() -> None:
    """验证 supplement provider failure retains committed initial results 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.factory import create_search_service
    from foren_searcher.store import GlobalStore

    with LocalCaseServer() as server:
        store = GlobalStore(); provider = _SupplementFails(CandidateSource(server.base_url + "/article", candidate_type="page"))
        service = create_search_service(store=store, provider=provider, allow_private_network=True)
        try:
            response = await service.execute({"input_type": "text", "query": "x", "output_type": "text"}, workspace_id="W4-3")
        finally:
            await service.aclose()
    assert response.stop_reason == "candidates_exhausted" and response.error is None
    assert len(response.results) == 1 and response.results[0].materials[0].status == "ready"
    assert store.has_reference(response.results[0].materials[0].evidence[0].evidence_id)


@pytest.mark.asyncio
async def test_image_query_resolves_real_evidence_and_rejects_text_evidence() -> None:
    """验证 image query resolves real evidence and rejects text evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, store, provider = make_service(())
        candidate_id, _ = store.get_or_create_candidate(identity_key="image:test", source_url=server.base_url + "/image", candidate_type="image")
        image = store.commit_material_evidence(candidate_id=candidate_id, material_kind="image", material_status="ready",
            artifact_bytes=__import__("search_helpers").png_bytes(), media_type="image/png", suffix=".png", modalities=("image",),
            label="query frame", content='{"width":32,"height":24}', locator="complete image", acquired_by="INSPECT/get_frames", discovered_by="W0-1")
        text = store.record_evidence(content="text only", source_ref="V0", locator="complete", acquired_by="INSPECT/asr", discovered_by="W0-1")
        try:
            valid = await service.execute({"input_type": "image", "query": image.evidence_id, "output_type": "image"}, workspace_id="W1-1")
            invalid = await service.execute({"input_type": "image", "query": text.evidence_id, "output_type": "image"}, workspace_id="W1-1")
        finally:
            await service.aclose()
    assert valid.stop_reason == "candidates_exhausted" and provider.calls[0]["image_path"] is not None
    assert invalid.stop_reason == "error" and invalid.error.code == "INVALID_INPUT"
    with closing(sqlite3.connect(store.db_path)) as connection:
        row = connection.execute("SELECT evidence_id, artifact_id, adaptation_json FROM search_image_uploads").fetchone()
        assert row is not None and row[0] == int(image.evidence_id[1:]) and row[1] > 0 and image.evidence_id in row[2]


@pytest.mark.asyncio
async def test_deployment_can_forbid_image_upload_without_text_fallback() -> None:
    """验证 deployment can forbid image upload without text fallback 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.factory import create_search_service
    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.store import GlobalStore
    store = GlobalStore()
    store.record_evidence(content="image metadata", modalities=("image",), source_ref="V0",
        locator="frame 1", acquired_by="INSPECT/get_frames", discovered_by="W0-1")
    provider = FixtureSearchProvider(())
    service = create_search_service(store=store, provider=provider, allow_image_uploads=False)
    try:
        response = await service.execute({"input_type": "image", "query": "E1", "output_type": "video"}, workspace_id="W1-1")
    finally:
        await service.aclose()
    assert response.stop_reason == "error" and response.error.code == "ACCESS_RESTRICTED"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_multiple_workspaces_can_search_concurrently_without_id_collisions() -> None:
    """验证 multiple workspaces can search concurrently without id collisions 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        service, store, _provider = make_service(())
        try:
            responses = await asyncio.gather(*(
                service.execute({"source": f"{server.base_url}/article?workspace={index}", "output_type": "text"}, workspace_id=f"W5-{index}")
                for index in range(1, 9)))
        finally:
            await service.aclose()
    candidate_ids = [item.results[0].candidate_id for item in responses]
    evidence_ids = [item.results[0].materials[0].evidence[0].evidence_id for item in responses]
    assert len(set(candidate_ids)) == len(set(evidence_ids)) == 8


class _SlowOpener:
    """测试辅助类 _SlowOpener；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""
        self.calls = 0
    async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
        """模拟 open 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls > 1:
            await asyncio.sleep(0.7)
        material = OpenedMaterial("page", "ready", "[1] committed source", b"<html>committed source</html>",
            "text/html", ".html", ("text",), "source", "paragraph 1", "SEARCH/open/test", "source_excerpt")
        return OpenedSource(candidate.source_url, (material,))


@pytest.mark.asyncio
async def test_total_timeout_keeps_already_committed_materials(monkeypatch: pytest.MonkeyPatch) -> None:
    """总预算用尽后，已提交的正文和未完成来源的超时反馈均保留。"""

    from types import SimpleNamespace
    import foren_searcher.search.service as search_module
    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore
    candidates = tuple(CandidateSource(f"https://example.org/{index}", candidate_type="page") for index in range(5))
    store = GlobalStore(); opener = _SlowOpener()
    policy = SearchPolicy(search_total_s=0.45, single_source_total_s=0.25, provider_timeout_s=0.10)
    now = 0.0
    commit = store.commit_material_evidence
    def commit_then_expire(**kwargs):
        nonlocal now
        evidence = commit(**kwargs)
        now = policy.search_total_s + 1
        return evidence
    # 总预算在实际提交后耗尽；未完成来源仍走真实的 asyncio 超时。
    monkeypatch.setattr(search_module, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(store, "commit_material_evidence", commit_then_expire)
    service = SearchService(store=store, provider=FixtureSearchProvider(candidates), opener=opener, policy=policy)
    response = await service.execute({"input_type": "text", "query": "timeout", "output_type": "text"}, workspace_id="W6-1")
    assert response.stop_reason == "timeout" and response.error.code == "TIMEOUT"
    assert store.has_reference("E1") and store.artifact_path(store.evidence("E1").artifact_ref).is_file()
    assert len(response.results) == 5
    assert any(any(material.status == "failed" and material.error.code == "TIMEOUT"
                   for material in item.materials) for item in response.results)


@pytest.mark.asyncio
async def test_duplicate_candidates_keep_each_discovery_but_open_once() -> None:
    """验证 duplicate candidates keep each discovery but open once 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        candidates = (
            CandidateSource(server.base_url + "/article", provider="fixture-a", engine="web", provider_rank=1, candidate_type="page"),
            CandidateSource(server.base_url + "/article", provider="fixture-b", engine="news", provider_rank=2, candidate_type="page"),
        )
        service, store, _provider = make_service(candidates, policy=SearchPolicy(target_sources=1))
        try:
            response = await service.execute({"input_type": "text", "query": "duplicate", "output_type": "text"}, workspace_id="W7-1")
        finally:
            await service.aclose()
    assert len(response.results) == 1 and server.counts["/article"] == 1
    discoveries = store.candidate_discoveries(1)
    assert [(item["provider"], item["engine"], item["provider_rank"]) for item in discoveries] == [
        ("fixture-a", "web", 1), ("fixture-b", "news", 2),
    ]


@pytest.mark.asyncio
async def test_known_image_reuses_image_and_retries_only_failed_carrier_page() -> None:
    """验证 known image reuses image and retries only failed carrier page 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        candidate = CandidateSource(
            server.base_url + "/toggle-page", provider_rank=1, candidate_type="image",
            image_url=server.base_url + "/image",
        )
        service, _store, _provider = make_service((candidate,), policy=SearchPolicy(target_sources=1))
        try:
            first = await service.execute({"input_type": "text", "query": "image", "output_type": "image"}, workspace_id="W8-1")
            second = await service.execute({"source": first.results[0].candidate_id, "output_type": "image"}, workspace_id="W8-2")
        finally:
            await service.aclose()
    first_materials = {item.kind: item for item in first.results[0].materials}
    second_materials = {item.kind: item for item in second.results[0].materials}
    assert first.stop_reason == "target_reached" and first_materials["image"].status == "ready"
    assert first_materials["page"].status == "failed"
    assert second.stop_reason == "target_reached" and second_materials["page"].status == "ready"
    assert first_materials["image"].evidence == second_materials["image"].evidence
    assert server.counts["/image"] == 1 and server.counts["/toggle-page"] == 2


@pytest.mark.asyncio
async def test_image_search_can_count_real_carrier_page_but_known_image_request_requires_image() -> None:
    """验证 image search can count real carrier page but known image request requires image 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with LocalCaseServer() as server:
        candidate = CandidateSource(server.base_url + "/article", provider_rank=1, candidate_type="page")
        service, _store, _provider = make_service((candidate,), policy=SearchPolicy(target_sources=1))
        try:
            searched = await service.execute({"input_type": "text", "query": "carrier", "output_type": "image"}, workspace_id="W8-3")
            reopened = await service.execute({"source": searched.results[0].candidate_id, "output_type": "image"}, workspace_id="W8-4")
        finally:
            await service.aclose()
    assert searched.stop_reason == "target_reached"
    assert {item.kind: item.status for item in searched.results[0].materials}["page"] == "ready"
    assert reopened.stop_reason == "candidates_exhausted"


class _MatrixProvider:
    """测试辅助类 _MatrixProvider；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.requests: list[object] = []

    async def search(self, request: object, **kwargs: object) -> CandidateBatch:
        """模拟 search 接口，返回由当前用例预设的可控结果。"""

        self.requests.append(request)
        output_type = request.output_type
        candidate_type = "page" if output_type == "text" else output_type
        candidate = CandidateSource(
            f"https://example.org/{output_type}", candidate_type=candidate_type,
            image_url="https://example.org/source.png" if output_type == "image" else None,
        )
        return CandidateBatch("matrix", output_type, (candidate,), {"query": request.query})


class _MatrixOpener:
    """测试辅助类 _MatrixOpener；用可控状态替代真实外部依赖。"""

    async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
        """模拟 open 接口，返回由当前用例预设的可控结果。"""

        if request.output_type == "image":
            material = OpenedMaterial("image", "ready", '{"width":1,"height":1}', b"image", "image/png", ".png",
                ("image",), "source image", "complete image", "SEARCH/open/test", "source_material")
            return OpenedSource(candidate.source_url, (material,))
        material = OpenedMaterial("page", "ready", "[1] source", b"<p>source</p>", "text/html", ".html",
            ("text",), "source page", "paragraph 1", "SEARCH/open/test",
            "source_metadata" if request.output_type == "video" else "source_excerpt")
        if request.output_type == "video":
            return OpenedSource(candidate.source_url, (material,),
                {"status": "formats_found", "reason": None},
                {"status": "not_requested", "evidence_id": None, "error": None})
        return OpenedSource(candidate.source_url, (material,))


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["text", "image", "text_image"])
@pytest.mark.parametrize("output_type", ["text", "image", "video"])
async def test_all_nine_query_combinations_execute_with_uniform_results(input_type: str, output_type: str) -> None:
    """验证 all nine query combinations execute with uniform results 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore
    from search_helpers import png_bytes

    store = GlobalStore()
    image = store.record_evidence(content="query image", modalities=("image",), source_ref="V0", locator="complete",
        acquired_by="INSPECT/get_frames", discovered_by="W0-1",
        artifact_ref=store.save_artifact(png_bytes(), media_type="image/png", suffix=".png", role="source_material"))
    query: object = "query" if input_type == "text" else image.evidence_id if input_type == "image" else {
        "evidence_id": image.evidence_id, "text": "unchanged original constraint",
    }
    provider = _MatrixProvider()
    service = SearchService(store=store, provider=provider, opener=_MatrixOpener(),
        policy=SearchPolicy(target_sources=1, maximum_provider_requests=1))
    response = await service.execute({"input_type": input_type, "query": query, "output_type": output_type}, workspace_id="W9-1")
    assert response.stop_reason == "target_reached" and len(response.results) == 1
    assert response.results[0].candidate_id.startswith("R")
    assert any(item.status == "ready" and item.evidence for item in response.results[0].materials)
    if input_type == "text_image":
        assert provider.requests[0].query["text"] == "unchanged original constraint"
    if output_type == "video":
        assert response.results[0].download_check == {"status": "formats_found", "reason": None}
        assert response.results[0].video_material == {"status": "not_requested", "evidence_id": None, "error": None}


class _ConcurrentOpener:
    """测试辅助类 _ConcurrentOpener；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.active = 0
        self.maximum_active = 0

    async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
        """模拟 open 接口，返回由当前用例预设的可控结果。"""

        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0.02)
            index = candidate.source_url.rsplit("/", 1)[-1]
            material = OpenedMaterial("page", "ready", f"[1] source {index}", f"<p>{index}</p>".encode(),
                "text/html", ".html", ("text",), index, "paragraph 1", "SEARCH/open/test", "source_excerpt")
            return OpenedSource(candidate.source_url, (material,))
        finally:
            self.active -= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 3, 5])
async def test_source_opening_has_finite_concurrency_and_preserves_candidate_order(concurrency: int) -> None:
    """验证 source opening has finite concurrency and preserves candidate order 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    candidates = tuple(CandidateSource(f"https://example.org/{index}", candidate_type="page") for index in range(5))
    opener = _ConcurrentOpener()
    service = SearchService(store=GlobalStore(), provider=FixtureSearchProvider(candidates), opener=opener,
        policy=SearchPolicy(source_concurrency=concurrency))
    response = await service.execute({"input_type": "text", "query": "parallel", "output_type": "text"}, workspace_id="W10-1")
    assert response.stop_reason == "target_reached" and opener.maximum_active == concurrency
    assert [item.source_url for item in response.results] == [item.source_url for item in candidates]


class _LanguageVideoOpener:
    """测试辅助类 _LanguageVideoOpener；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
        """模拟 open 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        language = request.subtitle_language or "en"
        page = OpenedMaterial("page", "ready", '{"title":"video"}', b'{"formats":[]}', "application/json", ".json",
            ("text",), "video", "video source fields", "SEARCH/open/yt-dlp", "source_metadata")
        subtitle_text = '{"language":"%s","automatic":false,"format":"vtt"}\nWEBVTT' % language
        subtitle = OpenedMaterial("subtitle", "ready", subtitle_text, b"WEBVTT", "text/vtt", ".vtt",
            ("text",), language, "platform subtitle timestamps", "SEARCH/open/yt-dlp+http", "source_transcript")
        return OpenedSource(candidate.source_url, (page, OpenedMaterial("cover", "absent"), subtitle),
            {"status": "formats_found", "reason": None},
            {"status": "not_requested", "evidence_id": None, "error": None})


@pytest.mark.asyncio
async def test_subtitle_reuse_is_language_specific_and_keeps_each_acquired_track() -> None:
    """验证 subtitle reuse is language specific and keeps each acquired track 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    store = GlobalStore(); opener = _LanguageVideoOpener()
    service = SearchService(store=store, provider=FixtureSearchProvider(()), opener=opener)
    source = "https://example.org/video/one"
    english = await service.execute({"source": source, "output_type": "video", "include": ["subtitles"],
        "subtitle_language": "en"}, workspace_id="W11-1")
    chinese = await service.execute({"source": english.results[0].candidate_id, "output_type": "video", "include": ["subtitles"],
        "subtitle_language": "zh"}, workspace_id="W11-2")
    reused = await service.execute({"source": english.results[0].candidate_id, "output_type": "video", "include": ["subtitles"],
        "subtitle_language": "zh"}, workspace_id="W11-3")
    zh_material = next(item for item in chinese.results[0].materials if item.kind == "subtitle")
    assert opener.calls == 2 and len(zh_material.evidence) == 2
    assert reused.results[0].materials[2].evidence == zh_material.evidence
    assert reused.results[0].download_check == {"status": "formats_found", "reason": None}


class _BlockingProvider:
    """测试辅助类 _BlockingProvider；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.started = asyncio.Event()

    async def search(self, request: object, **kwargs: object) -> CandidateBatch:
        """模拟 search 接口，返回由当前用例预设的可控结果。"""

        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_caller_cancellation_finishes_audit_and_remains_cancellation() -> None:
    """验证 caller cancellation finishes audit and remains cancellation 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    store = GlobalStore(); provider = _BlockingProvider()
    service = SearchService(store=store, provider=provider, opener=_MatrixOpener())
    task = asyncio.create_task(service.execute(
        {"input_type": "text", "query": "cancel", "output_type": "text"}, workspace_id="W12-1",
    ))
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    execution = store.search_execution(1)
    assert execution["finished_at"] is not None
    assert execution["stop_reason"] == "error"
    assert execution["error"]["code"] == "TIMEOUT"
    assert store.provider_calls(1)[0]["outcome"] == "cancelled"
    await service.aclose()


@pytest.mark.asyncio
async def test_service_close_waits_for_inflight_work_and_is_idempotent() -> None:
    """验证 service close waits for inflight work and is idempotent 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    started = asyncio.Event(); release = asyncio.Event(); close_calls = 0

    class Opener:
        """测试辅助类 Opener；用可控状态替代真实外部依赖。"""

        async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
            """模拟 open 接口，返回由当前用例预设的可控结果。"""

            started.set()
            await release.wait()
            material = OpenedMaterial("page", "ready", "[1] done", b"<p>done</p>", "text/html", ".html",
                ("text",), "done", "paragraph 1", "SEARCH/open/test", "source_excerpt")
            return OpenedSource(candidate.source_url, (material,))

    async def closed() -> None:
        """测试辅助函数 closed；为用例提供可重复的输入或可控行为。"""

        nonlocal close_calls
        close_calls += 1

    service = SearchService(
        store=GlobalStore(),
        provider=FixtureSearchProvider((CandidateSource("https://example.org/inflight"),)),
        opener=Opener(), policy=SearchPolicy(target_sources=1, maximum_provider_requests=1),
        close_callback=closed,
    )
    execution = asyncio.create_task(service.execute(
        {"input_type": "text", "query": "inflight", "output_type": "text"}, workspace_id="W12-2",
    ))
    await started.wait()
    closing_task = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    assert not closing_task.done()
    release.set()
    assert (await execution).stop_reason == "target_reached"
    await closing_task
    await service.aclose()
    assert close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await service.execute(
            {"input_type": "text", "query": "later", "output_type": "text"}, workspace_id="W12-3",
        )


@pytest.mark.asyncio
async def test_source_retry_attempts_are_persisted() -> None:
    """验证 source retry attempts are persisted 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    class Opener:
        """测试辅助类 Opener；用可控状态替代真实外部依赖。"""

        def __init__(self) -> None:
            """保存测试替身需要的预设输入、状态和调用记录。"""
            self.calls = 0
        async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
            """模拟 open 接口，返回由当前用例预设的可控结果。"""

            self.calls += 1
            if self.calls == 1:
                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "transient"))
            material = OpenedMaterial("page", "ready", "[1] recovered", b"ok", "text/plain", ".txt",
                ("text",), "recovered", "complete", "SEARCH/open/test", "source_excerpt")
            return OpenedSource(candidate.source_url, (material,))

    store = GlobalStore(); opener = Opener()
    service = SearchService(store=store,
        provider=FixtureSearchProvider((CandidateSource("https://example.org/retry"),)), opener=opener,
        policy=SearchPolicy(target_sources=1, maximum_provider_requests=1))
    response = await service.execute(
        {"input_type": "text", "query": "retry", "output_type": "text"}, workspace_id="W12-4",
    )
    assert response.stop_reason == "target_reached"
    assert [item["outcome"] for item in store.source_open_attempts(1)] == ["error", "success"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_same_candidate_requests_run_independently_and_reuse_completed_evidence(cancel_first) -> None:
    """同一来源的在途请求独立执行，后续调用复用已提交结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    started = asyncio.Event(); both_started = asyncio.Event(); release = asyncio.Event()

    class Opener:
        """测试辅助类 Opener；用可控状态替代真实外部依赖。"""

        def __init__(self) -> None:
            """保存测试替身需要的预设输入、状态和调用记录。"""
            self.calls = 0
        async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
            """模拟 open 接口，返回由当前用例预设的可控结果。"""

            self.calls += 1
            started.set()
            if self.calls == 2:
                both_started.set()
            await release.wait()
            material = OpenedMaterial("page", "ready", "[1] shared", b"shared", "text/plain", ".txt",
                ("text",), "shared", "complete", "SEARCH/open/test", "source_excerpt")
            return OpenedSource(candidate.source_url, (material,))

    store = GlobalStore(); opener = Opener()
    service = SearchService(store=store,
        provider=FixtureSearchProvider((CandidateSource("https://example.org/shared"),)), opener=opener,
        policy=SearchPolicy(target_sources=1, maximum_provider_requests=1))
    first = asyncio.create_task(service.execute(
        {"input_type": "text", "query": "shared", "output_type": "text"}, workspace_id="W13-1",
    ))
    await started.wait()
    second = asyncio.create_task(service.execute(
        {"input_type": "text", "query": "shared", "output_type": "text"}, workspace_id="W13-2",
    ))
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1)
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not second.done()
        release.set()
        if cancel_first:
            responses = [await second]
        else:
            responses = await asyncio.gather(first, second)
        identifiers = [response.results[0].materials[0].evidence[0].evidence_id for response in responses]
        assert len(set(identifiers)) == len(identifiers)
        assert all(response.results[0].candidate_id == responses[0].results[0].candidate_id for response in responses)
        reused = await service.execute({"source": responses[0].results[0].candidate_id,
            "output_type": "text"}, workspace_id="W13-3")
        assert {card.evidence_id for card in reused.results[0].materials[0].evidence} == set(identifiers)
        assert opener.calls == 2
        assert [item["outcome"] for item in store.source_open_attempts(1)] == ["cancelled" if cancel_first else "success"]
        assert [item["outcome"] for item in store.source_open_attempts(2)] == ["success"]
        assert [item["outcome"] for item in store.source_open_attempts(3)] == ["reused"]
        assert store.healthcheck().status == "ok"
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await service.aclose()


@pytest.mark.asyncio
async def test_many_concurrent_workspaces_finish_with_unique_evidence_and_audits() -> None:
    """验证 many concurrent workspaces finish with unique evidence and audits 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    class Provider:
        """测试辅助类 Provider；用可控状态替代真实外部依赖。"""

        async def search(self, request: SearchQueryRequest, **kwargs: object) -> CandidateBatch:
            """模拟 search 接口，返回由当前用例预设的可控结果。"""

            return CandidateBatch("load", "fixture", (
                CandidateSource(f"https://example.org/load/{request.query}", candidate_type="page"),
            ))

    class Opener:
        """测试辅助类 Opener；用可控状态替代真实外部依赖。"""

        async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
            """模拟 open 接口，返回由当前用例预设的可控结果。"""

            await asyncio.sleep(0)
            content = candidate.source_url.rsplit("/", 1)[-1]
            material = OpenedMaterial("page", "ready", f"[1] {content}", content.encode(), "text/plain", ".txt",
                ("text",), content, "complete", "SEARCH/open/load", "source_excerpt")
            return OpenedSource(candidate.source_url, (material,))

    store = GlobalStore()
    service = SearchService(store=store, provider=Provider(), opener=Opener(),
        policy=SearchPolicy(target_sources=1, maximum_provider_requests=1))
    responses = await asyncio.gather(*(service.execute(
        {"input_type": "text", "query": str(index), "output_type": "text"},
        workspace_id=f"W14-{index + 1}",
    ) for index in range(32)))
    await service.aclose()
    evidence_ids = {
        response.results[0].materials[0].evidence[0].evidence_id for response in responses
    }
    assert all(response.stop_reason == "target_reached" for response in responses)
    assert len(evidence_ids) == 32
    assert all(store.search_execution(index)["finished_at"] for index in range(1, 33))


@pytest.mark.asyncio
async def test_operational_logs_exclude_query_content(caplog: pytest.LogCaptureFixture) -> None:
    """验证 operational logs exclude query content 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    from foren_searcher.search.providers.fixture import FixtureSearchProvider
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    service = SearchService(store=GlobalStore(), provider=FixtureSearchProvider(()), opener=_MatrixOpener(),
        policy=SearchPolicy(maximum_provider_requests=1))
    with caplog.at_level(logging.INFO, logger="foren_searcher.search.service"):
        response = await service.execute(
            {"input_type": "text", "query": "private-query-marker", "output_type": "text"},
            workspace_id="W15-1",
        )
    await service.aclose()
    assert response.stop_reason == "candidates_exhausted"
    assert "SEARCH execution started" in caplog.text and "SEARCH execution finished" in caplog.text
    assert "private-query-marker" not in caplog.text
