"""Source-binding regression checks; artificial HTML is never SFT material."""
from pathlib import Path
import asyncio
import json
import pytest

from ddgs.engines.bing import Bing
from ddgs.engines.duckduckgo import Duckduckgo
from ddgs.engines.yahoo import Yahoo
from source_bound_search import SourceBoundTextProvider, parse_bound_results
from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest


def parser():
    # Parsing needs no HTTP client, certificate store, or network.
    return Bing.__new__(Bing)


def test_live_captured_result_fields_belong_to_separate_sources():
    text = (Path(__file__).parent / "search_binding_capture_bing_direct/bing.html").read_text(encoding="utf-8")
    results, records = parse_bound_results(parser(), text)
    assert len(results) == 7
    amazon = next(r for r in results if "amazon.in" in r["href"])
    rentokil = next(r for r in results if "rentokil.com" in r["href"])
    assert "Rentokil" not in amazon["title"] and "GoodRx" not in amazon["title"]
    assert "carbon dioxide" not in amazon["body"]
    assert "Lemongrass" in rentokil["title"]
    assert all(r["href_count"] == 1 and r["outcome"] == "accepted" for r in records)


def test_nested_result_blocks_do_not_concatenate():
    text = '''<div><li class="b_algo"><h2><a href="https://a.example/page">Alpha</a></h2><p>Alpha snippet</p>
    <div><li class="b_algo"><h2><a href="https://b.example/page">Beta</a></h2><p>Beta snippet</p></li></div>
    </li></div>'''
    results, records = parse_bound_results(parser(), text)
    assert results == [{"title": "Alpha", "href": "https://a.example/page", "body": "Alpha snippet"},
                       {"title": "Beta", "href": "https://b.example/page", "body": "Beta snippet"}]
    assert len(records) == 2


def test_ambiguous_source_block_is_not_bound_to_first_url():
    text = '''<li class="b_algo"><h2><a href="https://a.example">Alpha</a></h2>
              <h2><a href="https://b.example">Beta</a></h2><p>Mixed snippet</p></li>'''
    results, records = parse_bound_results(parser(), text)
    assert results == []
    assert records[0]["outcome"] == "no_unique_source_link"


def test_failed_acquisition_remains_failure_with_a_saved_record(tmp_path):
    provider = SourceBoundTextProvider(None, capture_dir=tmp_path)
    provider._fetch = lambda request: ([], "none", "us-en", [{"engine": "brave", "outcome": "no_html"}])
    request = SearchQueryRequest("text", "query", "text")
    with pytest.raises(SearchFailure, match="does not establish"):
        asyncio.run(provider.search(request, offset=0, limit=5))
    record = json.loads((tmp_path / "call_0001.json").read_text())
    assert record["captures"][0]["outcome"] == "no_html"


def test_live_captured_link_around_heading_is_readable():
    capture = Path(__file__).parent / "runs/averimatec_63_q0_source_bound_r1/search_captures/call_0001.json"
    html_text = json.loads(capture.read_text(encoding="utf-8"))["captures"][-1]["html"]
    results, records = parse_bound_results(parser(), html_text)
    assert len(results) == 5
    assert results[0]["href"] == "https://support.google.com/youtube/answer/161805?hl=en&co=GENIE.Platform=Desktop"
    assert results[0]["title"] == "Create an account on YouTube - Computer - YouTube Help"
    assert all(r["href_count"] == 1 for r in records)


@pytest.mark.parametrize("engine_class", [Yahoo, Duckduckgo])
def test_actual_alternative_engine_returns_keep_institution_source_separate(engine_class):
    capture = Path(__file__).parent / f"search_engine_probe_r1/query_01_{engine_class.name}.json"
    record = json.loads(capture.read_text(encoding="utf-8"))
    results, blocks = parse_bound_results(engine_class.__new__(engine_class), record["html"])
    institution = [r for r in results if "dengue.health.gov.lk" in r["href"]]
    assert institution
    assert all("National Environment Agency" not in r["title"] for r in institution)
    assert all("AFP" not in r["body"] for r in institution)
    assert all(r["href_count"] == 1 for r in blocks if r["outcome"] == "accepted")
