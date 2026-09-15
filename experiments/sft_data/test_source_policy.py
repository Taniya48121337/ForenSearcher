import asyncio
import json

from source_policy import ArticleExclusion, exclude_original_article
from foren_searcher import GlobalStore
from foren_searcher.search.contracts import CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.service import SearchService


ARTICLE = "https://factcheck.example.org/reports/case"
ARCHIVE = "https://web.archive.org/web/20230101000000/" + ARTICLE
DOCUMENT = "https://factcheck.example.org/files/original-budget.pdf"


def test_known_same_case_alias_is_excluded_without_general_suffix_rules(tmp_path):
    original = "https://factcheck.afp.com/doc.afp.com.32JR6NT-1"
    alias = "https://factcheck.afp.com/doc.afp.com.32JR6NT"
    exclusion = ArticleExclusion(original, tmp_path / "aliases.jsonl")
    assert exclusion.matches(alias)
    assert exclusion.matches("https://web.archive.org/web/20230101000000/" + alias + "?utm_source=test")
    assert not exclusion.matches("https://factcheck.afp.com/doc.afp.com.32JR6NT2")
    assert not exclusion.matches("https://factcheck.afp.com/files/budget.pdf")
    assert not ArticleExclusion("https://example.org/report-1", tmp_path / "other.jsonl").matches("https://example.org/report")


class Provider:
    async def search(self, request, *, offset, limit, image_path=None):
        candidates = (CandidateSource(ARTICLE, "Excluded answer", "TEST_ANSWER"),
                      CandidateSource(ARCHIVE + "?utm_source=x#part", "Archived answer", "TEST_ANSWER"),
                      CandidateSource(DOCUMENT, "Budget document", "Source table"))
        return CandidateBatch("fixture", "fixture", candidates[offset:offset + limit],
                              {"raw_test_answer": "TEST_ANSWER"})


class Opener:
    def __init__(self):
        self.calls = []

    async def open(self, candidate, request):
        self.calls.append(candidate.source_url)
        final = ARTICLE if candidate.source_url.endswith("/redirect") else candidate.source_url
        material = OpenedMaterial("document", "ready", content="ACTUAL_TEST_BODY",
            artifact_bytes=b"ACTUAL_TEST_BODY", media_type="text/plain", suffix=".txt")
        return OpenedSource(final, (material,))


def test_excluded_candidates_never_become_model_visible_records(tmp_path):
    async def scenario():
        store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
        opener = Opener()
        service = SearchService(store=store, provider=Provider(), opener=opener,
                                policy=SearchPolicy(target_sources=1))
        exclude_original_article(service, ARCHIVE, tmp_path / "exclusions.jsonl")
        async with service:
            response = await service.execute({"input_type": "text", "query": "budget", "output_type": "text"}, workspace_id="W1-1")
        assert len(response.results) == 1
        assert response.results[0].source_url == DOCUMENT
        assert "TEST_ANSWER" not in json.dumps(response.to_dict())
        assert len(store.candidate_records()) == 1
        assert len(store.evidence_records()) == 1
        assert opener.calls == [DOCUMENT]
        events = [json.loads(line) for line in (tmp_path / "exclusions.jsonl").read_text().splitlines()]
        assert len(events) == 2
    asyncio.run(scenario())


def test_direct_and_redirected_article_body_is_not_committed(tmp_path):
    async def scenario():
        store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
        opener = Opener()
        service = SearchService(store=store, provider=Provider(), opener=opener)
        exclude_original_article(service, ARTICLE, tmp_path / "exclusions.jsonl")
        async with service:
            for url in (ARCHIVE, "https://ordinary.example.org/redirect"):
                response = await service.execute({"source": url, "output_type": "text"}, workspace_id="W1-1")
                assert response.results[0].materials[0].status == "failed"
                assert "ACTUAL_TEST_BODY" not in json.dumps(response.to_dict())
                assert not store.evidence_records()
        assert opener.calls == ["https://ordinary.example.org/redirect"]
        events = [json.loads(line) for line in (tmp_path / "exclusions.jsonl").read_text().splitlines()]
        assert [event["stage"] for event in events] == ["source_open", "redirect_target"]
    asyncio.run(scenario())
