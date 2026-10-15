"""10-14 评测协议 --auto-tools on（合并目录选项 A）与换脸行改写（选项 B）。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from foren_searcher.policies import RuntimeGuards, active_guards, runtime_guard_scope

AUTO = RuntimeGuards(auto_compare_on_match=True, auto_video_structure=True)


class _Inspect:
    enabled_operations = frozenset({"get_frames", "ocr", "asr", "locate", "video_structure", "compare_media"})

    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = [], fail

    async def execute(self, request, *, workspace_id):
        self.calls.append((request, workspace_id))
        if self.fail:
            raise RuntimeError("download failed")
        lines = ("B matches A: 96% of A's half-second samples are found in B.",
                 "Face identity differs: person 1 (centre; A 0.0-900.0 s, 40 comparisons) has median similarity 0.3 ...")
        return SimpleNamespace(status="ready", result_evidence_id="E40", result=SimpleNamespace(summary_lines=lines),
                               evidence=(SimpleNamespace(evidence_id="E40", acquired_by="INSPECT/compare_media"),),
                               feedback=())


def _harness(tmp_path, inspect):
    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask
    from foren_searcher.store import GlobalStore

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=store, inspect_service=None)
    harness.inspect_service = inspect
    # the fake responses use made-up E ids; record what the workspace would mark instead of touching the Store
    harness.seen = []
    harness._record_evidence_access = lambda ids: None
    harness._mark_content_observed = harness.seen.append
    return harness


def test_auto_compare_runs_the_best_source_match_and_marks_the_result_seen(tmp_path):
    inspect = _Inspect()
    harness = _harness(tmp_path, inspect)
    out = asyncio.run(harness._auto_compare([(0.42, "R2", "E1", "E11"), (0.96, "R5", "E1", "E17")]))
    assert inspect.calls == [({"targets": ["E1", "E17"], "request": {"operation": "compare_media", "params": {}}}, "W1-1")]
    assert out["candidate_id"] == "R5" and out["result_evidence_id"] == "E40"
    assert out["lines"][0].startswith("auto_compare[R5]: content_match found E1's footage in R5 (96% of its samples)")
    assert "Result Evidence: E40" in out["lines"][0] and out["lines"][2].startswith("Face identity differs")
    assert "E40" in harness.seen


def test_auto_compare_failure_is_one_line_and_needs_the_merged_catalog(tmp_path):
    harness = _harness(tmp_path, _Inspect(fail=True))
    out = asyncio.run(harness._auto_compare([(0.9, "R1", "E1", "E11")]))
    assert out["status"] == "error" and "failed (RuntimeError)" in out["lines"][0]
    old = _Inspect()
    old.enabled_operations = frozenset({"get_frames", "align_media"})
    assert asyncio.run(_harness(tmp_path / "b", old)._auto_compare([(0.9, "R1", "E1", "E11")])) is None


def _search_result(status="source", fraction=0.96):
    page = SimpleNamespace(kind="page", status="ready", evidence=(SimpleNamespace(evidence_id="E11", modalities=("text",)),))
    item = SimpleNamespace(candidate_id="R1", source_url="https://www.youtube.com/watch?v=AcMmLmvOYTo", materials=(page,),
                           content_match={"material": "E1", "status": status, "fraction": fraction})
    return SimpleNamespace(results=(item,), error=None, stop_reason="target_reached",
                           to_dict=lambda: {"output_type": "video", "results": []})


@pytest.mark.parametrize("guards, status, ran", [(AUTO, "source", True), (AUTO, "partial", False),
                                                  (RuntimeGuards(), "source", False)])
def test_search_observation_carries_the_automatic_comparison_only_when_asked(tmp_path, guards, status, ran):
    inspect = _Inspect()
    harness = _harness(tmp_path, inspect)

    class Search:
        async def execute(self, action, *, workspace_id):
            return _search_result(status)

    harness.search_service = Search()
    with runtime_guard_scope(guards):
        asyncio.run(harness.execute_search({"input_type": "text", "query": "ellen dancers", "output_type": "video"}))
    items = harness.memory.latest_observation["items"]
    assert bool(inspect.calls) is ran
    assert any(x.startswith("auto_compare[R1]") for x in items) is ran
    assert any(x.startswith("full_comparison[R1]") for x in items)           # the call stays visible either way


def test_ingest_structure_and_gia_card(tmp_path):
    from foren_searcher.gia.runtime import InvestigationRuntime

    calls = []

    class Inspect:
        enabled_operations = _Inspect.enabled_operations

        async def execute(self, request, *, workspace_id):
            calls.append((request["request"]["operation"], workspace_id))
            return SimpleNamespace(status="ready", result_evidence_id="E9",
                                   result=SimpleNamespace(summary_lines=("12 shot(s) between hard cuts ...",
                                                                         "Shot 4 (A 15.7-25.7 s) lasts 10.0 s ...")))

    events = []
    runtime = SimpleNamespace(inspect_service=Inspect(), store=SimpleNamespace(
        evidence=lambda eid: SimpleNamespace(modalities=("video",))))
    import foren_searcher.gia.runtime as R

    original = R._record_runtime_event
    R._record_runtime_event = lambda store, **kw: events.append(kw)
    try:
        asyncio.run(InvestigationRuntime._ingest_structure(runtime, ("E1", "C0")))
    finally:
        R._record_runtime_event = original
    assert calls == [("video_structure", "HARNESS")]
    assert events[0]["event_type"] == "ingest_structure" and events[0]["payload"]["summary_lines"][1].startswith("Shot 4")


def test_global_view_shows_harness_structure_summary_only_with_the_guard(tmp_path):
    from foren_searcher.gia.global_view import GlobalViewBuilder
    from foren_searcher.hi.memory import MemoryPolicy
    from foren_searcher.store import GlobalStore

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    material = store.commit_ingested_evidence(source_ref="I0", media_type="video/mp4", modalities=("video",),
                                              label="Material A", original_filename="A.mp4", data=b"\x00" * 16)
    eid = material.evidence_id
    content = json.dumps({"summary_lines": ["3 shot(s) between hard cuts in 0-30 s (median shot 10.0 s).",
                                            "Shot 2 (A 10.0-20.0 s) lasts 10.0 s between two hard cuts, ..."]})
    store.record_evidence(derived_from=(eid,), acquired_by="INSPECT/video_structure", discovered_by="HARNESS",
                          content=content, label="INSPECT video_structure result", locator="{}",
                          production_method="structured_analysis")
    builder = GlobalViewBuilder(store, MemoryPolicy())
    plain = builder.build(task="t", original_material_ids=(eid,), latest_round_index=None).to_dict()
    assert not any("summary_lines" in c for c in plain["current_materials"])
    with runtime_guard_scope(AUTO):
        view = builder.build(task="t", original_material_ids=(eid,), latest_round_index=None).to_dict()
    (card,) = [c for c in view["current_materials"] if "summary_lines" in c]
    assert card["summary_lines"][1].startswith("Shot 2") and "automatically" in card["note"]


def test_face_line_says_the_face_was_replaced():
    # 10-14：换脸行恢复 DETECT v2.4 的写法（"this person's face was replaced (AI face swap)"）
    from foren_searcher.inspect import source_compare

    text = open(source_compare.__file__, encoding="utf-8").read()
    assert "this person's face was replaced (AI face swap)" in text and "describe the person (hair, clothing" in text


def test_run_one_rejects_auto_tools_without_the_merged_catalog():
    from foren_searcher.datagen import run_cases as R

    assert R.AUTO_TOOLS_MODES == ("off", "on", "compare")
    args = SimpleNamespace(teacher="endpoint", runs_dir="runs", round="r", specs="specs", hi_spec="hi@0.8.0", gia_spec="gia@0.9.0",
                           blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s", tool_cache="off",
                           tool_cache_root="t", harness_params="apply", inspect_ops="auto", detect="off",
                           detect_config="x", image_search="auto", no_think=True, auto_tools="on")
    item = SimpleNamespace(case_file="c.json", run_name="n", sample=1, case_id="evid:1")
    command = R._worker_command(args, item, None, None)
    assert command[command.index("--auto-tools") + 1] == "on"
    assert replace(AUTO).auto_compare_on_match and not active_guards().auto_compare_on_match
