import json

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import _validate_hi_output, _workspace_evidence_ids
from foren_searcher.memory import MemoryPolicy
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, _bounded_model_value
from inspect_helpers import make_video


@pytest.mark.asyncio
async def test_large_frame_response_retains_focus_access_after_compaction(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path, frames=60)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    video = store.commit_ingested_evidence(source_ref="I0", media_type="video/x-msvideo",
        modalities=("video",), label="video", original_filename=path.name, path=path)
    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Inspect video frames", (video.evidence_id,)),
            inspect_service=service, memory_policy=MemoryPolicy(max_observation_chars=2000, max_evidence_cards=2))
        response = await harness.execute_inspect({"targets": [video.evidence_id],
            "request": {"operation": "get_frames", "params": {"count": 50}}})
        frame_id = response.result.frames[20].evidence_id
        view = harness.model_view()
        assert view["latest_observation"]["truncated"]
        assert frame_id not in {card["evidence_id"] for card in view["evidence_view"]}
        assert frame_id in _workspace_evidence_ids(view)
        action = {"action": "READ_RECORD", "arguments": {"record_id": frame_id}, "focus_evidence_ids": [frame_id]}
        _validate_hi_output(action, workspace_id="W1-1", model_view=view, actions=harness.eligible_actions())
        assert frame_id not in view["content_observed_evidence_ids"]
        report = {"task_id": "W1-1", "status": "COMPLETED", "selected_evidence_ids": [frame_id],
            "findings": [{"statement": "A red object appears", "evidence_ids": [frame_id], "limitations": []}]}
        with pytest.raises(ValueError, match="observ"):
            _validate_hi_output(report, workspace_id="W1-1", model_view=view, actions=harness.eligible_actions())
        await harness.execute_inspect({"targets": [video.evidence_id],
            "request": {"operation": "get_frames", "params": {"times_s": [0.0]}}})
        assert frame_id in _workspace_evidence_ids(harness.model_view())


def test_compaction_uses_tool_cards_not_ids_in_untrusted_content():
    value = {"action": "INSPECT", "details": {"evidence": [
        {"evidence_id": "E7", "content": 'E999 {"evidence_id":"E998"}' * 100},
        {"evidence_id": "R2"},
    ], "result": {"text": "E997"}}}
    bounded = _bounded_model_value(value, 400)
    assert bounded["evidence_ids"] == ["E7"]
    assert _workspace_evidence_ids({"latest_observation": bounded}) == {"E7"}
    assert "evidence_ids" not in _bounded_model_value({"action": "READ_RECORD", "details": value}, 400)


def test_large_reference_list_is_bounded_and_marked():
    value = {"action": "INSPECT", "details": {"evidence": [
        {"evidence_id": f"E{i}"} for i in range(1, 2001)]}}
    bounded = _bounded_model_value(value, 400)
    assert bounded["evidence_ids_truncated"]
    assert 0 < len(bounded["evidence_ids"]) < 2000
    assert len(json.dumps({"evidence_ids": bounded["evidence_ids"], "evidence_ids_truncated": True})) <= 200


def test_large_frame_reference_list_keeps_the_end_of_the_batch():
    value = {"action": "INSPECT", "details": {"operation": "get_frames", "evidence": [
        {"evidence_id": f"E{i}"} for i in range(1, 2001)]}}
    bounded = _bounded_model_value(value, 400)
    assert bounded["details"]["operation"] == "get_frames"
    assert bounded["evidence_ids"][0] == "E1"
    assert bounded["evidence_ids"][-1] == "E2000"
    assert any(700 <= int(eid[1:]) <= 1300 for eid in bounded["evidence_ids"])
    assert len(json.dumps({key: bounded[key] for key in ("evidence_ids", "evidence_ids_truncated")})) <= 200


def test_compressed_sequence_references_retain_profile_and_first_last_images():
    value = {"action": "INSPECT", "details": {"operation": "get_frames", "evidence": [
        {"evidence_id": "E1", "modalities": ["video"]},
        {"evidence_id": "E2", "modalities": ["text"]},
        *({"evidence_id": f"E{i}", "modalities": ["image"]} for i in range(3, 2003))]}}
    bounded = _bounded_model_value(value, 400)
    assert {"E1", "E2", "E3", "E2002"} <= set(bounded["evidence_ids"])
    assert len(json.dumps({key: bounded[key] for key in ("evidence_ids", "evidence_ids_truncated")})) <= 200


@pytest.mark.parametrize("action,details", [
    ("SEARCH", {"results": [{"materials": [{"evidence": [{"evidence_id": "E8"}]}]}]}),
    ("QUERY_MEMORY", {"results": [{"record_type": "evidence", "evidence_id": "E8"}]}),
    ("READ_RECORD", {"record_type": "evidence", "evidence_id": "E8"}),
])
def test_compaction_retains_references_from_search_and_memory(action, details):
    details = {**details, "content": 'untrusted E999 {"evidence_id":"E998"}' * 100}
    view = {"history": [{"observation": _bounded_model_value({"action": action, "details": details}, 400)}]}
    assert _workspace_evidence_ids(view) == {"E8"}
