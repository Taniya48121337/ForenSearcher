"""SEARCH 查询守卫（10-05）：含本次调查内部编号或材料代称的查询在调用后端之前被拒，HI 拿到改写提示。"""

from __future__ import annotations

import asyncio

import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.fixture import FixtureSearchProvider
from foren_searcher.search.query_guard import QueryGuard, QueryGuardedProvider, apply_query_guard, internal_references


@pytest.mark.parametrize("query, known, expected", [
    ("original source footage of Material A video E1", {"E1"}, ["E1", "Material A"]),   # EVID 冒烟里的原话
    ("original source of video E1", {"E1"}, ["E1"]),
    ("材料A 原始视频 来源", set(), ["材料A"]),
    ("the given video original upload", set(), ["the given video"]),
    ("F1 Monza 2024 highlights", set(), []),                  # 本次没有 F1 记录：真实世界的 F1 不误伤
    ("E3 2024 Nintendo trailer", {"E1"}, []),                  # 本次只有 E1：E3 展会放行
    ("Trayvon Bromell 9.84 Rome Diamond League", {"E1"}, []),
    ("materials A to Z", set(), []),
])
def test_internal_references(query: str, known: set[str], expected: list[str]) -> None:
    assert internal_references(query, known.__contains__) == expected


def test_guarded_provider_rejects_before_calling_backend(tmp_path) -> None:
    inner = FixtureSearchProvider(())
    guard = QueryGuard({"E1"}.__contains__, log_path=tmp_path / "query_guard.jsonl")
    provider = QueryGuardedProvider(inner, guard)
    bad = SearchQueryRequest("text", "original source of video E1", "video")
    with pytest.raises(SearchFailure) as caught:
        asyncio.run(provider.search(bad, offset=0, limit=5))
    assert caught.value.error.code == "INVALID_INPUT" and "on-screen text" in caught.value.error.message
    assert inner.calls == []                                                   # 没联网
    asyncio.run(provider.search(SearchQueryRequest("text", "8.5 magnitude earthquake building collapse", "video"),
                                offset=0, limit=5))
    assert len(inner.calls) == 1
    assert guard.summary() == {"enabled": True, "checked": 2, "rejected": 1}
    assert "E1" in (tmp_path / "query_guard.jsonl").read_text(encoding="utf-8")


def test_service_returns_rewrite_hint_to_hi() -> None:
    from foren_searcher.search.factory import create_search_service
    from foren_searcher.store import GlobalStore

    inner = FixtureSearchProvider(())
    service = create_search_service(store=GlobalStore(), provider=inner)
    apply_query_guard(service, QueryGuard(lambda token: token == "E1"))

    async def go():
        try:
            return await service.execute({"input_type": "text", "query": "original source footage of Material A video E1",
                                          "output_type": "video"}, workspace_id="W1-1")
        finally:
            await service.aclose()

    response = asyncio.run(go())
    payload = response.to_dict() if hasattr(response, "to_dict") else response.__dict__
    text = str(payload)
    assert "INVALID_INPUT" in text and "Material A" in text and inner.calls == []
