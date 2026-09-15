import json

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import _gia_read_text_pages, _gia_visible_reference_ids
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy, RecordAccess, bound_global_view
from foren_searcher.search.contracts import MaterialResult, SearchError, SearchResponse, SearchResult
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, WorkspaceRunner
from inspect_helpers import ingest, make_video


@pytest.mark.asyncio
async def test_failed_workspace_exposes_completed_tool_work_without_a_report(tmp_path):
    path = tmp_path / "clip.avi"
    make_video(path)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    video = ingest(store, path.read_bytes(), source="I0", filename=path.name,
                   media_type="video/avi", modality="video")
    task = WorkspaceTask("W1-1", "Describe the first frame", (video.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))

    class Interrupted:
        calls = 0

        async def decide(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                targets, operation, params = [video.evidence_id], "get_frames", {"times_s": [0.0]}
            elif self.calls == 2:
                frame = next(e for e in store.evidence_records() if "image" in e.modalities)
                targets, operation, params = [frame.evidence_id], "visual_qa", {"question": "Image dimensions?"}
            elif self.calls == 3:
                targets, operation, params = [video.evidence_id], "visual_qa", {"question": "Describe this video"}
            else:
                raise OSError("interrupted after tool results")
            return {"action": "INSPECT", "arguments": {"targets": targets,
                    "request": {"operation": operation, "params": params}}, "focus_evidence_ids": []}

    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(store=store, task=task, round_index=1, inspect_service=service)
        with pytest.raises(OSError):
            await WorkspaceRunner(harness=harness, policy=Interrupted()).run()
    store.set_workspace_run_status(task.task_id, "runtime_error", error={"type": "OSError", "message": "interrupted"})
    # A proposal without an observation was not executed. Private reasoning is not progress.
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="model_output",
        payload={"turn_index": 5, "output": {"action": "SEARCH", "arguments": {"query": "UNEXECUTED"}}})
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="raw_model_output",
        payload={"reasoning": "PRIVATE_REASONING"})
    store.record_event(actor_role="HI", actor_id="W1-2", event_type="observation",
        payload={"turn_index": 1, "observation": {"action": "SEARCH", "details": {"status": "SIBLING"}}})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    progress = view["latest_reports"][0]
    assert "report" not in progress
    observations = progress["action_observations"]
    assert [row["operation"] for row in observations] == ["get_frames", "visual_qa", "visual_qa"]
    assert observations[-1]["status"] == "failed"
    assert observations[-1]["feedback"][0]["code"] == "UNSUPPORTED_MEDIA"
    assert observations[0]["arguments"]["request"]["params"] == {"times_s": [0.0]}
    result_id = observations[1]["evidence_ids"][0]
    assert _gia_read_text_pages((), view)[0]["content"] == store.evidence(result_id).content
    assert result_id in _gia_visible_reference_ids(view, ())
    rendered = json.dumps(progress)
    assert all(text not in rendered for text in ("UNEXECUTED", "PRIVATE_REASONING", "SIBLING"))
    assert not store.workspace_report_records()
    sibling = RecordAccess(store=store, role="HI", actor_id="W2-1", round_index=2, policy=MemoryPolicy())
    with pytest.raises(PermissionError):
        sibling.read(task.task_id)
    assert sibling.read(result_id)["evidence_id"] == result_id


def test_progress_is_bounded_and_full_action_history_can_be_read_by_workspace_id():
    store = GlobalStore()
    task = WorkspaceTask("W1-1", "Recover the missing source", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    for turn in range(1, 9):
        store.record_event(actor_role="HI", actor_id=task.task_id, event_type="model_output",
            payload={"turn_index": turn, "output": {"action": "SEARCH",
                     "arguments": {"query": f"query-{turn} " + "word " * 100}}})
        store.record_event(actor_role="HI", actor_id=task.task_id, event_type="observation",
            payload={"turn_index": turn, "observation": {"action": "SEARCH", "details": {
                "stop_reason": "timeout", "error": {"code": "TIMEOUT"}, "results": []}}})
    store.set_workspace_run_status(task.task_id, "timed_out", error={"type": "timeout", "message": "time limit"})
    # A resumed runner can reuse turn numbers; a later proposal cannot replace an earlier executed request.
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="model_output", payload={"turn_index": 1,
        "output": {"action": "SEARCH", "arguments": {"query": "unexecuted resumed query"}}})
    policy = MemoryPolicy(max_observation_chars=2000, max_read_chars=300, max_history_entries=3)
    view = GlobalViewBuilder(store, policy).build(task=task.task,
        original_material_ids=(), latest_round_index=1).to_dict()
    progress = view["latest_reports"][0]
    assert progress["action_observations_truncated"]
    assert 0 < len(progress["action_observations"]) <= 3
    small = bound_global_view(view, 400)
    assert len(json.dumps(small, ensure_ascii=False)) <= 400
    assert small["latest_reports"][0]["task_id"] == task.task_id
    assert small["latest_reports"][0]["run_status"] == "timed_out"
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=1, policy=policy)
    page = access.read(task.task_id)
    text = page["content"]
    while page["next_offset"] is not None:
        page = access.read(task.task_id, offset=page["next_offset"])
        text += page["content"]
    full = json.loads(text)
    assert len(full["action_observations"]) == 8
    assert full["action_observations"][0]["arguments"]["query"].startswith("query-1 ")


def test_search_partial_results_and_rejections_remain_available_after_failure():
    store = GlobalStore()
    source = store.record_evidence(content="A retrieved paragraph.", label="Retrieved source", source_ref="C0",
        locator="paragraph:1", acquired_by="SEARCH/page", discovered_by="W1-1")
    task = WorkspaceTask("W1-1", "Verify the source", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    response = SearchResponse("text", "timeout", (SearchResult("R1", "https://example.test/source", None, None,
        (MaterialResult("page", "ready", (source.card(),)), MaterialResult("document", "failed",
          error=SearchError("fetch", "TIMEOUT", "Document fetch timed out")))),),
        SearchError("search", "TIMEOUT", "Search timed out after one source"))
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="observation", payload={"turn_index": 1,
        "observation": {"action": "SEARCH", "details": response.to_dict()}})
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="observation", payload={"turn_index": 2,
        "observation": {"action": "HARNESS_REJECTION", "details": {"error": "invalid offset"}}})
    store.set_workspace_run_status(task.task_id, "runtime_error", error={"type": "output", "message": "failed"})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=(), latest_round_index=1).to_dict()
    progress = view["latest_reports"][0]
    search, rejection = progress["action_observations"]
    assert search["stop_reason"] == "timeout"
    assert search["error"]["code"] == "TIMEOUT"
    assert search["results"][0]["materials"][1]["error"]["message"] == "Document fetch timed out"
    assert rejection["error"] == "invalid offset"
    assert progress["supporting_records"][0]["content"] == source.content
