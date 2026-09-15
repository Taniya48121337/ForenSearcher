"""WorkspaceHarness 与真实 INSPECT Service 的接口对齐测试。

该测试从 Workspace 动作入口调用 crop，确认观察值和 Evidence 都能正确返回。
"""

from __future__ import annotations

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from foren_searcher.local_models import _validate_hi_output

from inspect_helpers import image_bytes, ingest, make_video


@pytest.mark.asyncio
async def test_visual_batch_schema_matches_the_configured_execution_limit():
    from jsonschema import validate, ValidationError

    store = GlobalStore()
    images = [ingest(store, image_bytes(), source=f"I{i}", filename=f"{i}.png",
                    media_type="image/png", modality="image").evidence_id for i in range(3)]
    service = create_inspect_service(store=store, policy=InspectPolicy(maximum_visual_qa_images=2))
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Compare images", ()),
                               store=store, inspect_service=service)
    try:
        schema = next(a["schema"] for a in harness.eligible_actions() if a["name"] == "INSPECT")
        call = {"targets": images, "request": {"operation": "visual_qa", "params": {"question": "Compare"}}}
        with pytest.raises(ValidationError):
            validate(call, schema)
        response = await harness.execute_inspect(call)
        assert response.status == "failed"
        limit = next(f for f in response.feedback if f.code == "RESOURCE_LIMIT")
        assert limit.locator["maximum_images"] == 2
        call["targets"] = images[:2]
        validate(call, schema)
        assert (await harness.execute_inspect(call)).status == "ready"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_harness_records_structured_call_and_global_cross_workspace_evidence():
    """验证 Harness 会记录结构化调用，并允许另一个 Workspace 读取全局派生 Evidence。"""

    store = GlobalStore(); image = ingest(store, image_bytes(), source="I0", filename="input.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store)
    first = WorkspaceHarness(task=WorkspaceTask("W1-1", "裁剪", (image.evidence_id,)), store=store, inspect_service=service)
    second = WorkspaceHarness(task=WorkspaceTask("W1-2", "复查", (image.evidence_id,)), store=store, inspect_service=service)
    try:
        cropped = await first.execute_inspect({"targets":[image.evidence_id],"request":{"operation":"crop","params":{"box":[.25,.25,.75,.75]}}})
        crop_id = cropped.result.evidence_id
        assert crop_id not in first.memory.content_observed_evidence_ids
        inspected = await second.execute_inspect({"targets":[crop_id],"request":{"operation":"visual_qa","params":{"question":"描述图像统计"}}})
        assert inspected.status == "ready"
        assert second.memory.history[0].request["targets"] == [crop_id]
        assert second.memory.history[0].response["status"] == "ready"
        assert crop_id in second.memory.evidence_pool_ids
        assert crop_id in second.memory.content_observed_evidence_ids
        assert inspected.result_evidence_id in second.memory.content_observed_evidence_ids
    finally: await service.aclose()


@pytest.mark.asyncio
async def test_report_can_cite_displayed_probe_attributes_without_claiming_frames_were_seen(tmp_path):
    path = tmp_path/'input.avi'
    make_video(path, frames=30, fps=10)
    store = GlobalStore(tmp_path/'store.sqlite3', tmp_path/'artifacts')
    video = store.commit_ingested_evidence(source_ref='I0', media_type='video/x-msvideo', modalities=('video',),
        label='input.avi', original_filename=path.name, path=path)
    service = create_inspect_service(store=store)
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1','报告视频时长', (video.evidence_id,)),
        store=store, inspect_service=service)
    try:
        response = await harness.execute_inspect({'targets':[video.evidence_id],
            'request':{'operation':'get_frames','params':{'times_s':[1.0]}}})
        profile = next(card for card in response.evidence if card.acquired_by=='INSPECT/media_profile')
        assert '3.000秒' in profile.label and '10.000fps' in profile.label
        assert profile.evidence_id in harness.memory.content_observed_evidence_ids
        assert video.evidence_id not in harness.memory.content_observed_evidence_ids
        frame = response.result.frames[0].evidence_id
        assert frame not in harness.memory.content_observed_evidence_ids
        view = harness.model_view()
        report = {'task_id':'W1-1','status':'COMPLETED','findings':[
            {'statement':'视频时长为3秒。','evidence_ids':[profile.evidence_id],'limitations':[]}],
            'selected_evidence_ids':[profile.evidence_id]}
        _validate_hi_output(report,workspace_id='W1-1',model_view=view,actions=())
        report['findings'][0]['evidence_ids'] = [frame]
        report['selected_evidence_ids'] = [frame]
        with pytest.raises(ValueError,match='content was not observed'):
            _validate_hi_output(report,workspace_id='W1-1',model_view=view,actions=())
    finally:
        await service.aclose()
