"""10-13 evidM27a 13/44 < evidC27e 23/44: the three causes found in the runs, each pinned by a test.

1. Reverse image search returned the article that embeds the original (video_194: inspiremore.com → YouTube
   AcMmLmvOYTo); SEARCH followed it to the YouTube page, but INSPECT refused to download a "page" Candidate.
2. text_image queries went to the image engines only, so the YouTube text search behind them disappeared.
3. compare_media's summary put short re-timed spans first and the describer's "Visual check" lines last; the HI sees
   the first 20 lines (history: 8) and wrote the timing noise up as forgery points (video_208, video_154).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from foren_searcher.inspect.acquisition import AcquiredMedia
from foren_searcher.inspect.backends.merged import prioritise_compare_lines
from foren_searcher.inspect.contracts import MERGED_OPERATIONS
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.platform_video import is_platform_video_url, is_video_candidate
from foren_searcher.search.contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.thumbnail_clip import IGNORED_TEXT_NOTE, ImageQueryRouter, text_also_note
from foren_searcher.store import GlobalStore


# ---------------------------------------------------------------- 1. page candidates resolved to a platform video

def test_video_candidate_includes_pages_resolved_to_youtube():
    assert is_video_candidate({"candidate_type": "video"})
    assert is_video_candidate({"candidate_type": "page", "final_url": "https://www.youtube.com/watch?v=AcMmLmvOYTo"})
    assert is_video_candidate({"candidate_type": "page", "final_url": "https://youtu.be/AcMmLmvOYTo"})
    assert not is_video_candidate({"candidate_type": "page", "final_url": "https://www.inspiremore.com/toronto-trio/"})
    assert not is_video_candidate({"candidate_type": "page", "final_url": "https://www.youtube.com/@TheEllenShow"})
    assert not is_video_candidate({"candidate_type": "image", "final_url": None})
    assert not is_platform_video_url("https://www.youtube.com.evil.example/watch?v=AcMmLmvOYTo")


class _Acquirer:
    def __init__(self, source: Path) -> None:
        self.source, self.urls = source, []

    async def acquire(self, source_url, *, destination, policy, headers=None, page_url=None, video_only_height=None):
        self.urls.append(source_url)
        shutil.copyfile(self.source, destination)
        return AcquiredMedia(destination, "video/mp4", ".mp4", "low.mp4", {"requested_url": source_url})


def _article_page(store: GlobalStore, final_url: str | None):
    url = "https://www.inspiremore.com/toronto-trio-dance-on-ellen/"
    candidate_id, _ = store.get_or_create_candidate(identity_key=f"url:{url}", source_url=url, candidate_type="page")
    if final_url:
        candidate_id = store.update_candidate_final_url(candidate_id, final_url, f"url:{final_url}")
    page = store.commit_material_evidence(
        candidate_id=candidate_id, material_kind="page", material_status="ready",
        artifact_bytes=b"<html>article</html>", media_type="text/html", suffix=".html",
        modalities=("text",), label="article", content="[1] article", locator="paragraph 1",
        acquired_by="SEARCH/open/http", discovered_by="W1-1")
    return candidate_id, page


@pytest.mark.asyncio
@pytest.mark.parametrize("final_url, allowed", [("https://www.youtube.com/watch?v=AcMmLmvOYTo", True), (None, False)])
async def test_fetch_picture_accepts_an_article_that_resolved_to_youtube(tmp_path, final_url, allowed):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 64)
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    candidate_id, page = _article_page(store, final_url)
    acquirer = _Acquirer(video)
    service = create_inspect_service(store=store, acquirer=acquirer, enabled_operations=MERGED_OPERATIONS)
    try:
        if allowed:
            path = await service.fetch_picture(candidate_id, page.evidence_id, tmp_path / "m" / "R1.media")
            assert Path(path).is_file() and acquirer.urls == [final_url]
        else:
            with pytest.raises(Exception, match="not a video candidate"):
                await service.fetch_picture(candidate_id, page.evidence_id, tmp_path / "m" / "R1.media")
            assert acquirer.urls == []
    finally:
        await service.aclose()


def test_workspace_gives_comparison_hints_for_an_article_that_resolved_to_youtube(tmp_path):
    from types import SimpleNamespace

    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    candidate_id, page = _article_page(store, "https://www.youtube.com/watch?v=AcMmLmvOYTo")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=store, inspect_service=None)
    item = SimpleNamespace(candidate_id=candidate_id, content_match=None, materials=(
        SimpleNamespace(kind="page", status="ready", evidence=(SimpleNamespace(evidence_id=page.evidence_id),)),))
    hints = harness._comparison_hints(item)
    assert hints and hints[0].startswith(f"open_as_video[{candidate_id}]")


# ---------------------------------------------------------------- 2. text_image = image engines + text search

def _cand(url: str, kind: str, provider: str, vid: str | None = None) -> CandidateSource:
    return CandidateSource(source_url=url, provider=provider, engine=provider, candidate_type=kind,
                           platform="youtube" if vid else None, platform_video_id=vid)


class _Fixed:
    def __init__(self, name: str, items: list[CandidateSource] | None, fail: bool = False) -> None:
        self.name, self.items, self.fail, self.requests = name, items or [], fail, []

    async def search(self, request, *, offset, limit, image_path=None):
        self.requests.append(request)
        if self.fail:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"{self.name} failed"))
        return CandidateBatch(self.name, self.name, tuple(self.items[offset:offset + limit]))


def _request() -> SearchQueryRequest:
    return SearchQueryRequest("text_image", {"evidence_id": "E8", "text": "three girls Ellen dance"}, "video")


def test_text_image_runs_both_searches_and_interleaves_text_first_for_video():
    text = _Fixed("yt", [_cand("https://www.youtube.com/watch?v=AcMmLmvOYTo", "video", "yt", "AcMmLmvOYTo"),
                         _cand("https://www.youtube.com/watch?v=bbbbbbbbbbb", "video", "yt", "bbbbbbbbbbb")])
    image = _Fixed("lens", [_cand("https://www.inspiremore.com/a/", "page", "lens"),
                            _cand("https://www.youtube.com/watch?v=AcMmLmvOYTo", "video", "lens", "AcMmLmvOYTo")])
    router = ImageQueryRouter(text, image, text_also=True)
    batch = asyncio.run(router.search(_request(), offset=0, limit=10, image_path=Path("x.jpg")))
    assert [c.source_url for c in batch.candidates] == [
        "https://www.youtube.com/watch?v=AcMmLmvOYTo", "https://www.inspiremore.com/a/",
        "https://www.youtube.com/watch?v=bbbbbbbbbbb"]
    assert text.requests[0].input_type == "text" and text.requests[0].query == "three girls Ellen dance"
    assert image.requests[0].input_type == "text_image" and batch.provider == "lens+yt"
    assert router.stats["text_image_both"] == 1


def test_text_image_survives_one_side_failing_and_plain_routing_is_unchanged():
    text = _Fixed("yt", [_cand("https://www.youtube.com/watch?v=AcMmLmvOYTo", "video", "yt", "AcMmLmvOYTo")])
    broken = _Fixed("lens", None, fail=True)
    batch = asyncio.run(ImageQueryRouter(text, broken, text_also=True).search(_request(), offset=0, limit=5, image_path=Path("x")))
    assert batch.provider == "yt" and len(batch.candidates) == 1
    with pytest.raises(SearchFailure):
        asyncio.run(ImageQueryRouter(_Fixed("yt", None, fail=True), broken, text_also=True).search(
            _request(), offset=0, limit=5, image_path=Path("x")))
    # without text_also (imageclip arm) text_image still goes to the image provider only
    text2, image2 = _Fixed("yt", []), _Fixed("clip", [_cand("https://www.youtube.com/watch?v=ccccccccccc", "video", "clip", "ccccccccccc")])
    asyncio.run(ImageQueryRouter(text2, image2).search(_request(), offset=0, limit=5, image_path=Path("x")))
    assert text2.requests == [] and len(image2.requests) == 1
    # an image-only query never runs a text search
    asyncio.run(ImageQueryRouter(text2, image2, text_also=True).search(
        SearchQueryRequest("image", "E8", "video"), offset=0, limit=5, image_path=Path("x")))
    assert text2.requests == []


def test_engine_note_no_longer_says_the_text_is_ignored():
    note = text_also_note("Image-input search is a reverse image search." + IGNORED_TEXT_NOTE)
    assert "ignored" not in note and "ordinary text search" in note


# ---------------------------------------------------------------- 3. compare_media summary order

# video_208 (evidM27a) compare_media summary, as the comparer wrote it
LINES_208 = [
    "B matches A: 94% of A's half-second samples are found in B (A 186 s, B 192 s; the matched footage lies in B 1-191 s).",
    "A 37.3-38.8 s plays B backwards (B 39.9-41.1 s).",
    "A 39.8-41.3 s plays B slower (0.3x of B's speed) (B 41.3-41.6 s).",
    "A 64.7-66.2 s plays B backwards (B 73.4-74.5 s).",
    "A 130.7-133.2 s plays B slower (0.52x of B's speed) (B 138.3-139.4 s).",
    "A 182.3-183.8 s plays B faster (1.4x of B's speed) (B 189.9-191.3 s).",
    "Not in B: A 41.3-46.3 s (5.0 s); B continues from 41.6 s to 51.9 s around it, so it replaces about 10.2 s of B.",
    "Missing from A: B 41.6-51.9 s (10.2 s) is skipped at A 43.8 s (A continues straight from before it to after it).",
    "Repeated: B 73.4-74.4 s appears in A twice (A 65.0-66.0 s and 66.5-68.0 s).",
    "Changed region: A 49.2-56.3 s, x 0.47-0.55 y 0.02-0.27 (1.3% of the frame) differs from B at the same moment in 8 samples while the rest of the frame matches.",
    "Changed region: A 0.2-9.2 s, x 0.31-0.34 y 0.26-0.33 (0.1% of the frame) differs from B at the same moment in 10 samples while the rest of the frame matches.",
    "2 other face track(s) match B (median similarity 0.93, 0.97).",
    "Visual check of comparison image 2 (changed region 1): B: Black plunger handle; A: Wooden plunger handle",
    "Visual check of comparison image 3 (changed region 2): B: Wrench; A: Hammer",
    "Visual check of comparison image 1 (added footage 1): A hand uses an adjustable wrench to tighten a metal clamp.",
]


def test_compare_lines_put_content_first_and_attach_visual_checks():
    out = prioritise_compare_lines(LINES_208)
    assert out[0].startswith("B matches A")
    assert out[1].startswith("Not in B: A 41.3-46.3 s") and out[1].endswith("tighten a metal clamp.")
    assert out[2].startswith("Changed region: A 49.2-56.3 s") and "Wooden plunger handle" in out[2]
    assert out[3].startswith("Changed region: A 0.2-9.2 s") and "B: Wrench; A: Hammer" in out[3]
    assert not any(x.startswith("Visual check") for x in out)
    # the 2.5 s slow-down stays its own line after the structure lines; the 1.5 s spans share the last line
    assert out.index("A 130.7-133.2 s plays B slower (0.52x of B's speed) (B 138.3-139.4 s).") > out.index(
        next(x for x in out if x.startswith("Repeated:")))
    assert out[-1].startswith("Short spans of 2 s or less") and "A 37.3-38.8 s backwards" in out[-1] \
        and "A 39.8-41.3 s 0.3x" in out[-1] and "A 182.3-183.8 s 1.4x" in out[-1]
    assert len(out) == len(LINES_208) - 3 - 4 + 1          # 3 checks merged, 4 short spans → 1 line


def test_compare_lines_keep_unmatched_checks_and_face_lines():
    lines = ["B matches A: 90% of A's half-second samples are found in B (A 60 s, B 60 s; the matched footage lies in B 0-60 s).",
             "Face identity differs: person 2 (centre of the frame; A 3.0-40.0 s, 9 comparisons) has median similarity 0.21 "
             "to the face at the same position in B (the same person is usually above 0.8).",
             "Visual check of comparison image 1 (person 2): A: a woman with short dark hair; B: a man with a beard",
             "Visual check of comparison image 2 (changed region 4): B: cup; A: bottle"]
    out = prioritise_compare_lines(lines)
    assert "a man with a beard" in out[1] and out[2] == "Visual check of comparison image 2 (changed region 4): B: cup; A: bottle"
    assert prioritise_compare_lines(["B is probably not the source of A: only 0% ..."]) == ["B is probably not the source of A: only 0% ..."]


# ---------------------------------------------------------------- 10-13 fix 2: GIA numbered-answer check (video_138)

TASK_138 = ("Investigate Material A. Final answer, in English: a numbered list of 3 to 5 forgery points "
            "(\"1. ...\", \"2. ...\").")
PARAGRAPH_138 = ("The video is a DIY woodworking tutorial titled 'Flood Survival' (E3). Note: the investigation round "
                 "budget is used up, so no original-source search could be performed.")
NUMBERED = ("1. The captions recast a planter-box build as a flood rescue crate (E3).\n"
            "2. The black liner is called a waterproof isolation layer (E99).\n"
            "3. Overall the re-captioning sells an ordinary build as disaster engineering (E90).")


def _gia(monkeypatch, answers):
    from foren_searcher import local_models

    seen = []

    class Gen:
        async def generate_json(self, *, system_prompt, payload):
            seen.append(payload.get("validation_error"))
            return {"decision": "FINISH", "answer": next(answers)}

    monkeypatch.setattr(local_models, "_validate_gia_output", lambda *a, **k: None)
    monkeypatch.setattr(local_models, "_gia_turn_prompt", lambda view, history=(): "prompt")
    return local_models.TransformersGiaPolicy(Gen()), seen


def test_unnumbered_finish_is_sent_back_once_with_the_findings_reminder(monkeypatch):
    from foren_searcher.local_models import UNNUMBERED_FINISH
    from foren_searcher.policies import EVAL_GUARDS, runtime_guard_scope

    view = {"direct_visual_input": True, "task": TASK_138, "runtime_constraints": {"can_create_round": True}}
    policy, seen = _gia(monkeypatch, iter([PARAGRAPH_138, NUMBERED]))
    with runtime_guard_scope(EVAL_GUARDS) as state:
        output = asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    assert output["answer"] == NUMBERED and seen == [None, UNNUMBERED_FINISH]
    assert state.counters["unnumbered_finish_rejected"] == 1 and "finish_not_numbered" not in state.contract_flags
    # a second unnumbered answer in the same run is accepted and flagged (no endless loop)
    policy, seen = _gia(monkeypatch, iter([PARAGRAPH_138] * 3))
    with runtime_guard_scope(EVAL_GUARDS) as state:
        first = asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    assert first["answer"] == PARAGRAPH_138 and state.contract_flags == ["finish_not_numbered"]


def test_numbered_check_needs_the_guard_and_a_task_that_asks_for_a_list(monkeypatch):
    from foren_searcher.policies import EVAL_GUARDS, runtime_guard_scope

    view = {"direct_visual_input": True, "task": TASK_138, "runtime_constraints": {"can_create_round": True}}
    policy, seen = _gia(monkeypatch, iter([PARAGRAPH_138]))
    assert asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))["answer"] == PARAGRAPH_138
    other = {**view, "task": "Is this photo from the 2019 flood? Answer in one paragraph."}
    policy, seen = _gia(monkeypatch, iter([PARAGRAPH_138]))
    with runtime_guard_scope(EVAL_GUARDS):
        assert asyncio.run(policy.decide(model_view=dict(other), history=(), actions=()))["answer"] == PARAGRAPH_138
    assert seen == [None]


# ---------------------------------------------------------------- 10-13 fix 2: wider budget for merged text_image (video_156)

def test_wide_text_image_only_for_merged_text_image_video_queries():
    from types import SimpleNamespace

    from foren_searcher.search.service import SearchService

    service = SimpleNamespace(text_image_wide=True)
    wide = lambda r: SearchService._wide_text_image(service, r)  # noqa: E731
    assert wide(_request())
    assert not wide(SearchQueryRequest("text_image", {"evidence_id": "E8", "text": "x"}, "image"))
    assert not wide(SearchQueryRequest("text", "three girls", "video"))
    assert not SearchService._wide_text_image(SimpleNamespace(text_image_wide=False), _request())


def test_matcher_per_search_override_tries_more_candidates(tmp_path):
    from types import SimpleNamespace

    from foren_searcher.search.source_match import SourceMatcher

    class Store:
        def __init__(self):
            self.rows = {}

        def investigation_meta(self):
            return {"original_material_ids": ["E1"]}

        def candidate_matches(self, material):
            return {}

        def candidate(self, rid):
            return {"provider_rank": int(rid[1:]), "source_url": f"https://www.youtube.com/watch?v={rid:0<11}", "extra": {}}

        def record_candidate_match(self, material, rid, *, status, fraction=None, details=None):
            return {"status": status, "fraction": fraction, "details": details or {}, "attempts": 1}

    material = tmp_path / "a.mp4"
    material.write_bytes(b"0")
    fetched = []

    async def fetch(rid, url, page):
        fetched.append(rid)
        return material

    def item(k):
        page = SimpleNamespace(kind="page", status="ready", evidence=(SimpleNamespace(evidence_id=f"E{10 + k}"),))
        return SimpleNamespace(candidate_id=f"R{k}", source_url="u", materials=(page,), video_material={"status": "x"},
                               content_match=None)

    response = SimpleNamespace(output_type="video", results=tuple(item(k) for k in range(1, 11)))
    matcher = SourceMatcher(Store(), fetch=fetch, align=lambda a, b: {"fraction": 0.0}, material_path=lambda e: material)
    matcher._duration = staticmethod(lambda path: None)
    asyncio.run(matcher._match(response, "E1", material))
    assert len(fetched) == 5
    fetched.clear()
    asyncio.run(matcher._match(response, "E1", material, 8))
    assert len(fetched) == 8
