import json

import pytest

from foren_searcher.contracts import FindingDraft, WorkspaceReportDraft, WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import _generation_messages, _gia_read_text_pages
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy, RecordAccess, bound_global_view
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes


def finish(store, task, evidence_ids):
    harness = WorkspaceHarness(store=store, task=task, round_index=1)
    harness.finish(WorkspaceReportDraft(task_id=task.task_id, status="COMPLETED",
        findings=(FindingDraft(statement="A local interpretation.", evidence_ids=evidence_ids,
                               limitations=("The interpretation may be mistaken.",)),),
        selected_evidence_ids=evidence_ids))


@pytest.mark.asyncio
async def test_real_alignment_scope_and_actual_images_reach_gia_separately(tmp_path):
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    path = tmp_path / "source.avi"
    make_video(path)
    a = ingest(store, path.read_bytes(), source="I0", filename="A.avi", media_type="video/avi", modality="video")
    b = ingest(store, path.read_bytes(), source="I1", filename="B.avi", media_type="video/avi", modality="video")
    task = WorkspaceTask("W1-1", "Compare both recordings", (a.evidence_id, b.evidence_id))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    async with create_inspect_service(store=store) as service:
        response = await service.execute({"targets": list(task.materials),
            "request": {"operation": "align_media", "params": {"mode": "visual"}}}, workspace_id=task.task_id)
    assert response.status == "ready"
    result_id = response.result_evidence_id
    frame_ids = [e.evidence_id for e in response.evidence if "image" in e.modalities]
    assert len(frame_ids) >= 2
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="raw_model_output",
        payload={"image_evidence_ids": frame_ids[:1], "reasoning": "PRIVATE_ANALYSIS"})
    store.record_event(actor_role="HI", actor_id="W1-2", event_type="raw_model_output",
        payload={"image_evidence_ids": frame_ids[1:], "reasoning": "SIBLING_ANALYSIS"})
    finish(store, task, (result_id,))
    # A newly opened store exercises the persisted provenance path used after recovery.
    reopened = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    view = GlobalViewBuilder(reopened, MemoryPolicy()).build(
        task=task.task, original_material_ids=task.materials, latest_round_index=1).to_dict()
    report = view["latest_reports"][0]
    raw = json.loads(store.evidence(result_id).content)
    supplied = json.loads(report["supporting_records"][0]["content"])
    assert supplied == raw
    assert supplied["coverage"]["a"]["processed"]["value"] > len(report["image_inputs"])
    assert [r["evidence_id"] for r in report["image_inputs"]] == frame_ids[:1]
    assert report["image_inputs"][0]["locator"] == store.evidence(frame_ids[0]).locator
    assert "PRIVATE_ANALYSIS" not in json.dumps(view)
    sibling = WorkspaceHarness(store=reopened, task=WorkspaceTask("W1-2", "Other task", ()))
    assert "image_inputs" not in json.dumps(sibling.model_view())
    assert "PRIVATE_ANALYSIS" not in json.dumps(sibling.model_view())


@pytest.mark.parametrize("operation,content", [
    ("asr", {"coverage": {"range_s": [2, 11], "failed_ranges": [[7, 8]]},
             "segments": [{"start_s": 3, "end_s": 4, "text": "A spoken sentence."}]}),
    ("ocr", {"coverage": {"range_s": [0, 100], "sampling": {"processed_frames": 40},
                          "failed_ranges": [[10, 12]]}, "results": []}),
    ("locate_frames", {"coverage": {"range_s": [0, 50], "coarse_frames": 25,
                                    "refined_ranges": [[4, 6]], "failed_ranges": []}, "frames": []}),
])
def test_tool_original_text_reaches_summary_without_hi_rewriting(operation, content):
    store = GlobalStore()
    source = store.record_evidence(content="Source", label="Input", source_ref="C0",
        locator="paragraph:1", acquired_by="TEST", discovered_by="W1-1")
    evidence = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=("text",),
        label="Analysis", content=json.dumps(content), locator="{}", acquired_by=f"INSPECT/{operation}",
        discovered_by="W1-1", production_method="structured_analysis")
    task = WorkspaceTask("W1-1", "Check the material", (source.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    finish(store, task, (evidence.evidence_id,))
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    page = _gia_read_text_pages((), view)[0]
    assert json.loads(page["content"]) == content
    messages = _generation_messages("GIA", {"global_view": view, "history": []})
    assert any(evidence.content.replace(' ', '') in message["content"].replace(' ', '').replace('\\"', '"')
               for message in messages)


def test_report_source_pages_keep_pagination_and_budget():
    store = GlobalStore()
    source = store.record_evidence(content="First page. " * 100 + "Final limitation.", label="Long source",
        source_ref="C0", locator="paragraph:1", acquired_by="TEST", discovered_by="W1-1")
    task = WorkspaceTask("W1-1", "Read the source", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    finish(store, task, (source.evidence_id,))
    policy = MemoryPolicy(max_read_chars=100)
    view = GlobalViewBuilder(store, policy).build(task=task.task,
        original_material_ids=(), latest_round_index=1).to_dict()
    supplied = view["latest_reports"][0]["supporting_records"][0]
    assert supplied["content"] == source.content[:100]
    assert supplied["next_offset"] == 100
    assert view["latest_reports"][0]["image_inputs"] == []
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=1, policy=policy)
    text = supplied["content"]
    while supplied["next_offset"] is not None:
        supplied = access.read(source.evidence_id, offset=supplied["next_offset"])
        text += supplied["content"]
    assert text == source.content
    view["current_materials"] = []
    view["history_index"] = []
    full_size = len(json.dumps(view, ensure_ascii=False))
    small = bound_global_view(view, full_size - 20)
    assert len(json.dumps(small, ensure_ascii=False)) <= full_size - 20
    assert small["latest_reports"][0]["supporting_records_truncated"]
    assert small["latest_reports"][0]["report"]["status"] == "COMPLETED"
    assert not _gia_read_text_pages((), small)


@pytest.mark.parametrize("filename,media_type,modality,data", [
    ("source.pdf", "application/pdf", "text", pdf_bytes()),
    ("source.png", "image/png", "image", image_bytes()),
])
def test_gia_report_attachments_do_not_treat_media_descriptions_as_content(filename, media_type, modality, data):
    store = GlobalStore()
    source = ingest(store, data, source="I0", filename=filename, media_type=media_type, modality=modality)
    task = WorkspaceTask("W1-1", "Inspect the attachment", (source.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    from foren_searcher.contracts import WorkspaceReport
    store.add_workspace_report(WorkspaceReport(task.task_id, "INCOMPLETE", (), (source.evidence_id,),
                                               status_reason="Content not yet observed."))
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    assert view["latest_reports"][0]["supporting_records"][0]["content_notice"]
    assert not _gia_read_text_pages((), view)
