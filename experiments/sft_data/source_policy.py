"""Apply the pilot's original-article exclusion before evidence is committed."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from foren_searcher.search.contracts import SearchError, SearchFailure


def canonical_url(value):
    value = re.sub(r"^https?://web\.archive\.org/web/[^/]+/", "", value or "")
    parts = urlsplit(value)
    if not parts.netloc:
        return value
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."),
                       parts.path.rstrip("/"), parts.query, ""))


def article_key(url):
    parts = urlsplit(canonical_url(url))
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


class ArticleExclusion:
    def __init__(self, article_url: str, log_path: Path):
        if not article_url:
            raise ValueError("Original article exclusion requires a known article URL")
        self.key = article_key(article_url)
        self.keys = {self.key}
        for group in json.loads(Path(__file__).with_name("source_article_aliases.json").read_text(encoding="utf-8")):
            keys = {article_key(group["reference_url"]), *(article_key(u) for u in group["excluded_aliases"])}
            if self.key in keys:
                self.keys.update(keys)
        self.log_path = log_path

    def matches(self, url):
        return bool(url) and article_key(url) in self.keys

    def record(self, stage, source_url, final_url=None):
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "source_url": source_url,
                "final_url": final_url, "reason": "dataset_original_fact_check_article"}, ensure_ascii=False) + "\n")


class FilteredProvider:
    def __init__(self, provider, exclusion):
        self.provider, self.exclusion = provider, exclusion

    async def search(self, request, *, offset, limit, image_path=None):
        batch = await self.provider.search(request, offset=offset, limit=limit, image_path=image_path)
        kept = []
        for candidate in batch.candidates:
            if self.exclusion.matches(candidate.source_url):
                self.exclusion.record("provider_candidate", candidate.source_url)
            else:
                kept.append(candidate)
        # The unmodified provider response remains a training-side audit artifact.
        return replace(batch, candidates=tuple(kept))


class FilteredOpener:
    def __init__(self, opener, exclusion):
        self.opener, self.exclusion = opener, exclusion

    async def open(self, candidate, request):
        if self.exclusion.matches(candidate.source_url):
            self.exclusion.record("source_open", candidate.source_url)
            raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED",
                "The original fact-check article is excluded from this investigation's evidence sources"))
        result = await self.opener.open(candidate, request)
        if self.exclusion.matches(result.final_url):
            self.exclusion.record("redirect_target", candidate.source_url, result.final_url)
            raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED",
                "The source redirects to an excluded original fact-check article; its body was not saved as evidence"))
        return result


def exclude_original_article(service, article_url, log_path):
    exclusion = ArticleExclusion(article_url, log_path)
    service.provider = FilteredProvider(service.provider, exclusion)
    service.opener = FilteredOpener(service.opener, exclusion)
    return service
