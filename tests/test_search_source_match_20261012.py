"""10-12 DETECT find_source 并入 SEARCH：视频结果的 content_match（search/source_match.py）与以图搜图默认开。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from foren_searcher.contracts import EvidenceCard
from foren_searcher.search.contracts import MaterialResult, SearchResponse, SearchResult
from foren_searcher.search.source_match import SourceMatcher, content_match_line, duration_tier
from foren_searcher.store import GlobalStore


def _setup(tmp_path: Path, durations: list[float | None]):
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    material = store.commit_ingested_evidence(source_ref="I0", media_type="video/mp4", modalities=("video",),
                                              label="A", original_filename="a.mp4", data=b"not decoded in this test")
    store.ensure_investigation(task="t", original_material_ids=[material.evidence_id])
    results = []
    for k, duration in enumerate(durations, start=1):
        url = f"https://www.youtube.com/watch?v=vid{k:08d}"
        rid, _ = store.get_or_create_candidate(identity_key=f"url:{url}", source_url=url, candidate_type="video",
                                               provider_rank=k, extra={"duration": duration} if duration else {})
        page = store.commit_material_evidence(
            candidate_id=rid, material_kind="page", material_status="ready", artifact_bytes=b"{}",
            media_type="application/json", suffix=".json", modalities=("text",), label="video page", content="page",
            locator="metadata", acquired_by="SEARCH/open/video", discovered_by="W1-1")
        card = EvidenceCard(page.evidence_id, ("text",), "video page", rid, (), "SEARCH/open/video")
        materials = (MaterialResult("page", "ready", (card,)), MaterialResult("cover", "absent"),
                     MaterialResult("subtitle", "absent"))
        results.append(SearchResult(rid, url, f"title {k}", None, materials,
                                    download_check={"status": "not_checked", "reason": None},
                                    video_material={"status": "not_requested", "evidence_id": None, "error": None}))
    response = SearchResponse("video", "candidates_exhausted", tuple(results))
    return store, material.evidence_id, response


class _Fixed:
    """fetch / align 替身：R 编号 → 匹配比例；列在 fail 里的下载失败。"""

    def __init__(self, fractions: dict[str, float], fail: set[str] = frozenset()):
        self.fractions, self.fail, self.fetched = fractions, set(fail), []

    async def fetch(self, rid, url, page):
        self.fetched.append(rid)
        if rid in self.fail:
            raise RuntimeError("yt-dlp: HTTP Error 403: Forbidden")
        return Path(f"/tmp/{rid}.mp4")

    def align(self, path_a, path_b):
        rid = Path(path_b).stem
        frac = self.fractions.get(rid, 0.0)
        return {"fraction": frac, "engine": "near_duplicate", "b_duration_s": 60.0,
                "runs": [{"a_range_s": [0.0, 10.0], "b_range_s": [20.0, 30.0], "kind": "normal", "mirrored": False}] if frac else []}


_MATERIAL: dict[str, Path] = {}


def _material_video() -> Path:
    """50 s 的小视频（2 fps、32×32）：只用来量材料时长。"""

    if "path" not in _MATERIAL:
        import tempfile

        import cv2
        import numpy as np

        path = Path(tempfile.mkdtemp()) / "a.mp4"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (32, 32))
        for k in range(100):
            out.write(np.full((32, 32, 3), k, np.uint8))
        out.release()
        _MATERIAL["path"] = path
    return _MATERIAL["path"]


def _matcher(store, fixed, **kw):
    path = _material_video()
    return SourceMatcher(store, fetch=fixed.fetch, align=fixed.align, material_path=lambda eid: path, **kw)


def test_duration_tiers_follow_find_source_ranking():
    assert duration_tier(60, 50)[0] == 0 and duration_tier(None, 50)[0] == 1
    assert duration_tier(30, 50)[0] == 2 and duration_tier(600, 50)[0] == 3


def test_results_get_content_match_and_state_is_kept_in_the_store(tmp_path):
    store, material, response = _setup(tmp_path, [60, 60, 3000, None, 60, 60, 60])
    fixed = _Fixed({"R1": 0.92, "R2": 0.12}, fail={"R5"})
    matcher = _matcher(store, fixed, per_search=5)
    annotated = asyncio.run(matcher.annotate(response, "W1-1"))
    by_id = {r.candidate_id: r.content_match for r in annotated.results}
    # 时长合理的先试（R1 R2 R5 R6 tier 0，R4 未知时长 tier 1），超长的 R3 排在后面、本次名额已满
    assert fixed.fetched == ["R1", "R2", "R5", "R6", "R7"]
    assert by_id["R1"]["status"] == "source" and by_id["R1"]["fraction"] == 0.92
    assert by_id["R2"]["status"] == "partial" and by_id["R5"]["status"] == "download_failed"
    assert by_id["R6"]["status"] == "none" and by_id["R3"]["status"] == by_id["R4"]["status"] == "not_checked"
    assert annotated.results[0].to_dict()["content_match"]["material"] == material
    # 第二次同样的结果：比过的不再下载，下载失败的再试一次，没比过的补上
    fixed.fetched.clear()
    again = asyncio.run(matcher.annotate(response, "W1-2"))
    assert fixed.fetched == ["R5", "R4", "R3"]
    assert {r.candidate_id: r.content_match["status"] for r in again.results}["R1"] == "source"
    assert store.candidate_matches(material)["R5"]["attempts"] == 2
    # 第三次：失败两次的不再试
    fixed.fetched.clear()
    asyncio.run(matcher.annotate(response, "W1-3"))
    assert fixed.fetched == []


def test_per_material_attempt_limit(tmp_path):
    store, material, response = _setup(tmp_path, [60] * 6)
    fixed = _Fixed({})
    matcher = _matcher(store, fixed, per_search=5, max_attempts=3)
    annotated = asyncio.run(matcher.annotate(response, "W1-1"))
    statuses = [r.content_match["status"] for r in annotated.results]
    assert statuses == ["none", "none", "none", "not_checked", "not_checked", "not_checked"]
    assert "already been compared" in annotated.results[-1].content_match["note"]


def test_non_video_responses_and_missing_material_are_untouched(tmp_path):
    store, material, response = _setup(tmp_path, [60])
    fixed = _Fixed({"R1": 1.0})
    text = SearchResponse("text", "candidates_exhausted", ())
    assert asyncio.run(_matcher(store, fixed).annotate(text, "W1-1")) is text
    nothing = SourceMatcher(store, fetch=fixed.fetch, align=fixed.align, material_path=lambda eid: None)
    assert asyncio.run(nothing.annotate(response, "W1-1")) is response and fixed.fetched == []


def test_content_match_line_states_facts_only():
    line = content_match_line("R3", {"material": "E1", "status": "source", "fraction": 0.92,
                                     "runs": [{"a_range_s": [0.0, 10.0], "b_range_s": [20.0, 30.0], "kind": "normal"}]})
    assert line.startswith("content_match[R3]: 92% of E1's picture samples are found in this video")
    assert "E1 0.0-10.0 s = this video 20.0-30.0 s" in line
    for word in ("AI", "fake", "generated", "authentic"):
        assert word not in line


def test_search_result_validates_content_match():
    materials = (MaterialResult("page", "absent"), MaterialResult("cover", "absent"), MaterialResult("subtitle", "absent"))
    video = {"download_check": {"status": "not_checked", "reason": None},
             "video_material": {"status": "not_requested", "evidence_id": None, "error": None}}
    SearchResult("R1", "https://x.test/v", None, None, materials, **video, content_match={"material": "E1", "status": "none"})
    with pytest.raises(ValueError):
        SearchResult("R1", "https://x.test/v", None, None, materials, **video, content_match={"material": "E1", "status": "maybe"})
    with pytest.raises(ValueError):
        SearchResult("R1", "https://x.test/v", None, None, (MaterialResult("page", "absent"), MaterialResult("document", "absent")),
                     content_match={"material": "E1", "status": "none"})


def test_workspace_shows_content_match_and_the_full_comparison_call(tmp_path):
    import json

    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask

    store, material, response = _setup(tmp_path, [60])
    matched = asyncio.run(_matcher(store, _Fixed({"R1": 0.9})).annotate(response, "W1-1"))

    class _Search:
        allows_image_uploads = True

        async def execute(self, action, *, workspace_id):
            return matched

    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "find the original", (material,)), store=store,
                               search_service=_Search())
    asyncio.run(harness.execute_search({"input_type": "text", "query": "q", "output_type": "video"}))
    lines = harness.memory.latest_observation["items"]
    assert any(line.startswith("content_match[R1]: 90%") for line in lines)
    call = next(line for line in lines if line.startswith("full_comparison[R1]: "))
    payload = json.loads(call.split(": ", 1)[1])
    assert payload["arguments"]["targets"][0] == material and payload["arguments"]["request"]["operation"] == "compare_media"


# ---------------------------------------------------------------- 以图搜图默认开（run_cases）


KEY_ENVS = ("GOOGLE_VISION_API_KEY", "SERPAPI_API_KEY", "SERPER_API_KEY", "TINEYE_API_KEY")


def _no_keys(monkeypatch):
    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_image_search_defaults_on(monkeypatch, tmp_path):
    from foren_searcher.datagen import run_cases as R

    _no_keys(monkeypatch)
    assert R.resolve_image_search_engine("auto", tmp_path) == "yandex"
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
    assert R.resolve_image_search_engine("auto", tmp_path) == "gcv+yandex"
    assert R.resolve_image_search_engine("off") == "off"
    assert R.IMAGE_SEARCH_MODES[0] == "auto"


def test_search_service_for_turns_on_reverse_image_search(monkeypatch, tmp_path):
    from foren_searcher.datagen import run_cases as R

    _no_keys(monkeypatch)
    monkeypatch.chdir(tmp_path)
    case = _unrestricted_case()
    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    search, _desc, _guard = R.search_service_for(case, store, tmp_path, blocklist="off", video_search="ddgs")
    assert search.allows_image_uploads and search.image_search_engine == "yandex"
    assert "reverse image search on Yandex" in search.image_search_note
    off, _d, _g = R.search_service_for(case, store, tmp_path, blocklist="off", video_search="ddgs", image_search="off")
    assert not off.allows_image_uploads and off.image_search_engine == "off"
    asyncio.run(search.aclose())
    asyncio.run(off.aclose())


def _unrestricted_case():
    from types import SimpleNamespace

    policy = SimpleNamespace(mode="unrestricted", excluded_keys=(), date_cutoff=None)
    return SimpleNamespace(source_policy=policy, case_id="test:1")
