"""SEARCH 公开请求、响应、错误和策略契约测试。

覆盖严格字段解析、URL/ID 格式、材料状态组合、视频专属字段和预算参数边界。
"""

from __future__ import annotations

import pytest

from foren_searcher.contracts import EvidenceCard
from foren_searcher.search.contracts import CandidateSource, MaterialResult, SearchError, SearchFailure, SearchResponse, SearchResult, parse_search_request
from foren_searcher.search.identity import normalize_url
from foren_searcher.search.policy import SearchPolicy


@pytest.mark.parametrize("payload", [
    {"input_type": "text", "query": "bridge report", "output_type": "text"},
    {"input_type": "image", "query": "E7", "output_type": "video"},
    {"input_type": "text_image", "query": {"evidence_id": "E7", "text": "bridge"}, "output_type": "image"},
    {"source": "https://example.org/report", "output_type": "text"},
    {"source": "R2", "output_type": "video", "include": ["subtitles"], "subtitle_language": "en"},
])
def test_confirmed_request_forms_parse(payload: dict[str, object]) -> None:
    """验证 confirmed request forms parse 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    assert parse_search_request(payload).output_type == payload["output_type"]


@pytest.mark.parametrize("input_type", ["text", "image", "text_image"])
@pytest.mark.parametrize("output_type", ["text", "image", "video"])
def test_all_nine_query_input_output_contracts_parse(input_type: str, output_type: str) -> None:
    """验证 all nine query input output contracts parse 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    query: object = "query" if input_type == "text" else "E1" if input_type == "image" else {
        "evidence_id": "E1", "text": "original text constraint",
    }
    request = parse_search_request({"input_type": input_type, "query": query, "output_type": output_type})
    assert request.input_type == input_type and request.output_type == output_type


@pytest.mark.parametrize("field", [
    "provider_timeout_s", "source_timeout_s", "dynamic_timeout_s",
    "single_source_total_s", "search_total_s", "source_concurrency",
    "dynamic_concurrency", "maximum_document_bytes", "maximum_image_bytes",
    "maximum_html_bytes",
])
def test_search_policy_rejects_non_positive_runtime_limits(field: str) -> None:
    """验证 search policy rejects non positive runtime limits 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match=field):
        SearchPolicy(**{field: 0})


def test_candidate_urls_reject_embedded_credentials() -> None:
    """验证 candidate urls reject embedded credentials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="credentials"):
        CandidateSource("https://user:secret@example.org/source")


def test_retry_after_rejects_negative_values() -> None:
    """验证 retry after rejects negative values 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="retry_after_s"):
        SearchFailure(SearchError("search", "RATE_LIMITED", "limited"), retry_after_s=-1)


def test_ipv6_url_identity_keeps_required_brackets() -> None:
    """验证 ipv6 url identity keeps required brackets 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    assert normalize_url("https://[2606:4700:4700::1111]:443/path#fragment") == "https://[2606:4700:4700::1111]/path"


@pytest.mark.parametrize("payload", [
    {"source": "https://example.org", "input_type": "text", "query": "x", "output_type": "text"},
    {"input_type": "image", "query": "a textual description", "output_type": "image"},
    {"input_type": "text_image", "query": {"evidence_id": "E1"}, "output_type": "video"},
    {"source": "E1", "output_type": "text"},
    {"source": "https://example.org", "output_type": "text", "include": ["subtitles"]},
    {"source": "https://example.org", "output_type": "video", "subtitle_language": "en"},
    {"source": "https://", "output_type": "text"},
    {"source": "https://user:secret@example.org/page", "output_type": "text"},
    {"source": "R01", "output_type": "text"},
    {"input_type": "image", "query": "E01", "output_type": "image"},
    {"source": "https://example.org:bad/page", "output_type": "text"},
    {"source": "https://example.org", "output_type": [], "include": []},
    {"source": "https://example.org", "output_type": "video", "include": [["subtitles"]]},
    {"source": "https://example.org", "output_type": "video", "include": ["subtitles"], "subtitle_language": 7},
])
def test_invalid_or_ambiguous_request_forms_fail(payload: dict[str, object]) -> None:
    """验证 invalid or ambiguous request forms fail 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError):
        parse_search_request(payload)


def test_material_status_invariants_are_enforced() -> None:
    """验证 material status invariants are enforced 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="Evidence"):
        MaterialResult("page", "ready")
    card = EvidenceCard("E1", ("text",), "page", "R1", (), "SEARCH/open")
    with pytest.raises(ValueError, match="error"):
        MaterialResult("page", "partial", (card,))
    with pytest.raises(ValueError, match="error"):
        MaterialResult("page", "failed")
    assert MaterialResult("page", "not_requested").to_dict()["error"] is None


def test_response_and_video_status_invariants_are_enforced() -> None:
    """验证 response and video status invariants are enforced 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="top-level error"):
        SearchResponse("text", "error", ())
    with pytest.raises(ValueError, match="cannot contain"):
        SearchResponse("text", "target_reached", (), SearchError("search", "PROVIDER_ERROR", "x"))
    with pytest.raises(ValueError, match="download_check"):
        SearchResult("R1", "https://example.org/video", None, None, (),
            {"status": "available_formats_reported", "reason": None},
            {"status": "not_requested", "evidence_id": None, "error": None})
    with pytest.raises(ValueError, match="evidence_id"):
        SearchResult("R1", "https://example.org/video", None, None, (),
            {"status": "formats_found", "reason": None},
            {"status": "ready", "evidence_id": None, "error": None})


def test_url_identity_is_conservative() -> None:
    """验证 url identity is conservative 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    assert normalize_url("HTTPS://Example.COM:443#part") == "https://example.com/"
    assert normalize_url("https://example.com/?id=1") != normalize_url("https://example.com/?id=2")
