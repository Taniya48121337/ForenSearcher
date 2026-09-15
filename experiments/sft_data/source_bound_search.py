"""Text-search acquisition with one result block per candidate and saved HTML."""
from __future__ import annotations

import asyncio
import copy
import itertools
import json
import math
import re
import time
from dataclasses import replace
from urllib.parse import urlsplit

from ddgs.engines.bing import Bing
from ddgs.engines.brave import Brave
from ddgs.engines.duckduckgo import Duckduckgo
from ddgs.engines.yahoo import Yahoo
from ddgs.results import TextResult

from foren_searcher.search.contracts import CandidateBatch, SearchError, SearchFailure
from foren_searcher.search.providers.searxng import SearXngProvider

PARSER_VERSION = "source_bound_v3"


def parse_bound_results(engine, html_text):
    """Detach nested result blocks before extracting each block's own fields."""
    tree = engine.extract_tree(engine.pre_process_html(html_text))
    items = tree.xpath(engine.items_xpath)
    output, blocks = [], []
    for item in items:
        local = copy.deepcopy(item)
        # XPath is scoped to the copied block; matched descendants get their own row.
        descendants = local.xpath("." + engine.items_xpath)
        for child in descendants:
            if child is not local and child.getparent() is not None:
                child.getparent().remove(child)
        selectors = dict(engine.elements_xpath)
        if engine.name == "bing":
            selectors.update(href=".//h2/a/@href | .//a[h2]/@href", title=".//h2//text()")
        hrefs = list(dict.fromkeys(local.xpath(selectors["href"])))
        record = {"path": tree.getroottree().getpath(item), "href_count": len(hrefs)}
        if len(hrefs) != 1:
            record["outcome"] = "no_unique_source_link"
            blocks.append(record)
            continue
        result = TextResult()
        for field in ("title", "body"):
            setattr(result, field, " ".join("".join(local.xpath(selectors[field])).split()))
        result.href = hrefs[0]
        parsed = engine.post_extract_results([result])
        if not parsed or not parsed[0].title or urlsplit(parsed[0].href).scheme not in {"http", "https"}:
            record["outcome"] = "not_a_text_result"
        else:
            record.update(outcome="accepted", source_url=parsed[0].href)
            output.append(parsed[0].__dict__)
        blocks.append(record)
    return output, blocks


class SourceBoundTextProvider:
    """Use explicit text engines; retain the existing provider for other modalities."""
    def __init__(self, provider, *, capture_dir, timeout_s=12.0):
        self.provider, self.timeout_s = provider, timeout_s
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = itertools.count(1)

    def _fetch(self, request):
        query = str(request.query)
        region = "cn-zh" if re.search(r"[\u3400-\u9fff]", query) else "us-en"
        deadline = time.monotonic() + self.timeout_s
        captures = []
        for engine_class in (Brave, Yahoo, Duckduckgo, Bing):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            engine = engine_class(timeout=max(1, math.ceil(min(6, remaining))))
            capture = {"engine": engine.name, "query": query}
            captures.append(capture)
            try:
                payload = engine.build_payload(query, region, "moderate", None, page=1)
                arguments = {"params" if engine.search_method == "GET" else "data": payload}
                capture.update(request_url=engine.search_url, method=engine.search_method, request_payload=payload)
                response = engine.http_client.request(engine.search_method, engine.search_url, **arguments)
                capture.update(http_status=response.status_code, html=response.text)
                if response.status_code != 200:
                    capture["outcome"] = "http_error"
                    continue
                html_text = response.text
                if not html_text:
                    capture["outcome"] = "no_html"
                    continue
                capture["html"] = html_text
                values, blocks = parse_bound_results(engine, html_text)
                capture.update(blocks=blocks, outcome="parsed", result_count=len(values))
                if values:
                    return values, engine.name, region, captures
            except Exception as error:
                capture.update(outcome="error", error_type=type(error).__name__)
        # Empty results retain the actual capture for audit; no invented evidence.
        return [], "none", region, captures

    async def search(self, request, *, offset, limit, image_path=None):
        if request.output_type != "text":
            return await self.provider.search(request, offset=offset, limit=limit, image_path=image_path)
        if request.input_type != "text":
            raise SearchFailure(SearchError("search", "INVALID_INPUT", "This pilot text search accepts text queries only"))
        if offset < 0 or limit < 1:
            raise ValueError("offset must be non-negative and limit positive")
        values, engine, region, captures = await asyncio.to_thread(self._fetch, request)
        capture_path = self.capture_dir / f"call_{next(self.sequence):04d}.json"
        capture_path.write_text(json.dumps({"offset": offset, "limit": limit,
            "results": values, "captures": captures, "parser_version": PARSER_VERSION}, ensure_ascii=False), encoding="utf-8")
        if not values:
            states = ", ".join(c["engine"] + ":" + c["outcome"] for c in captures)
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR",
                "No usable source-bound search records were obtained (" + states + "); this does not establish that relevant sources are absent"))
        candidates = []
        for rank, value in enumerate(values[offset:offset + limit], offset + 1):
            candidate = SearXngProvider._normalize({"url": value["href"], "title": value["title"],
                "content": value["body"], "engine": engine}, "text", rank, query=str(request.query))
            if candidate is not None:
                candidates.append(replace(candidate, provider="pilot_source_bound"))
        return CandidateBatch("pilot_source_bound", engine, tuple(candidates),
            {"results": values, "captures": captures, "parser": PARSER_VERSION},
            {"region": region, "actual_engine": engine, "parser": PARSER_VERSION})
