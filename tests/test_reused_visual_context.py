from dataclasses import replace
import json

import fitz
import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.model_images import ModelImageInputs
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from inspect_helpers import ingest, make_video


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['video', 'pdf'])
@pytest.mark.parametrize('capacity', [4, 8])
async def test_reused_media_keeps_own_seen_context_within_capacity(tmp_path, kind, capacity):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    if kind == 'video':
        path = tmp_path / 'source.avi'
        make_video(path, frames=20)
        source = ingest(store, path.read_bytes(), source='I0', filename=path.name, media_type='video/avi', modality='video')
        operation = 'get_frames'
    else:
        document = fitz.open()
        for i in range(7):
            page = document.new_page(width=160, height=96)
            page.insert_text((20, 40), f'Page {i+1}')
        source = ingest(store, document.tobytes(), source='I0', filename='source.pdf', media_type='application/pdf', modality='text')
        document.close()
        operation = 'get_pages'
    images = ModelImageInputs(store=store, image_policy=replace(InspectPolicy(), maximum_visual_qa_images=capacity))
    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(task=WorkspaceTask('W1-1', '核对材料内容', (source.evidence_id,)), store=store, inspect_service=service)
        first = await harness.execute_inspect({'targets': [source.evidence_id], 'request': {'operation': operation, 'params': {'count': 6}}})
        assert first.status == 'ready'
        view = harness.memory.to_model_view()
        images.prepare_model_view(view)
        seen = [item['evidence_id'] for item in view['image_inputs']]
        assert len(seen) == min(6, capacity)
        store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output', payload={'image_evidence_ids': seen})
        locator = json.loads(store.evidence(seen[0]).locator)
        params = {'times_s': [locator['time_s'] + .000001]} if kind == 'video' else {'pages': [locator['page_number']]}
        narrow = {'targets': [source.evidence_id], 'request': {'operation': operation, 'params': params}}
        harness.set_focus(seen[:1])
        for _ in range(2):
            result = await harness.execute_inspect(narrow)
            assert result.status == 'ready'
            assert store.inspect_execution(result.execution_id)['details']['created_evidence_ids'] == []
            view = harness.memory.to_model_view()
            images.prepare_model_view(view)
            retained = [item['evidence_id'] for item in view['image_inputs']]
            assert retained == seen
            assert len(retained) <= capacity
            store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output', payload={'image_evidence_ids': retained})
        other = WorkspaceHarness(task=WorkspaceTask('W1-2', '独立核对一处内容', (source.evidence_id,)), store=store, inspect_service=service)
        await other.execute_inspect(narrow)
        other_view = other.memory.to_model_view()
        images.prepare_model_view(other_view)
        assert [item['evidence_id'] for item in other_view['image_inputs']] == seen[:1]
        new_params = {'times_s': [0.0]} if kind == 'video' else {'pages': [4]}
        result = await harness.execute_inspect({'targets': [source.evidence_id], 'request': {'operation': operation, 'params': new_params}})
        assert result.status == 'ready'
        created = store.inspect_execution(result.execution_id)['details']['created_evidence_ids']
        assert len(created) == 1 and created[0] not in seen
        view = harness.memory.to_model_view()
        images.prepare_model_view(view)
        assert [item['evidence_id'] for item in view['image_inputs']] == [seen[0], created[0]]
        assert store.healthcheck().status == 'ok'
