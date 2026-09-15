import asyncio
import json
import os

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.harness1 import harness1_workspace_type
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy
from foren_searcher.orchestration import InvestigationRuntime
from foren_searcher.search.contracts import MaterialResult, SearchResponse, SearchResult
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, WorkspaceRunner
from inspect_helpers import image_bytes, ingest
from test_framework_runtime import FinishAfterResumeGia, add_text_evidence


@pytest.mark.asyncio
@pytest.mark.parametrize('original_runtime', [False, True])
@pytest.mark.parametrize('legacy_events', [False, True])
async def test_open_round_restores_only_own_committed_memory(tmp_path, original_runtime, legacy_events):
    workspace_type = WorkspaceHarness
    if original_runtime:
        root = os.environ.get('FOREN_HARNESS_ROOT')
        if not root:
            pytest.skip('FOREN_HARNESS_ROOT required')
        workspace_type = harness1_workspace_type(root)
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = add_text_evidence(store, 'Verified source text')
    unseen = add_text_evidence(store, 'Unseen source text')
    task = WorkspaceTask('W1-1', 'Read the source', (evidence.evidence_id,))
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task='Investigation',
        original_material_ids=task.materials, latest_round_index=None)
    store.start_round(1, gia_input=view.to_dict(), gia_decision={'decision': 'INVESTIGATE',
        'tasks': [{'task': task.task, 'materials': list(task.materials)}]}, tasks=(task,))

    class Interrupted:
        calls = 0
        async def decide(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id},
                        'focus_evidence_ids': [evidence.evidence_id]}
            if self.calls == 2:
                return {'action': 'READ_RECORD', 'arguments': {'record_id': unseen.evidence_id, 'extra': True},
                        'focus_evidence_ids': [unseen.evidence_id]}
            raise asyncio.CancelledError

    harness = workspace_type(task=task, store=store, round_index=1)
    with pytest.raises(asyncio.CancelledError):
        await WorkspaceRunner(harness=harness, policy=Interrupted()).run()
    expected = harness.model_view()
    assert len(expected['history']) == 2
    assert expected['latest_observation']['action'] == 'HARNESS_REJECTION'
    assert expected['content_observed_evidence_ids'] == [evidence.evidence_id]
    # 尚未收到 Observation 的请求不能算已执行，也不能进入已读集合。
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='model_output',
        payload={'turn_index': 3, 'output': {'action': 'READ_RECORD',
            'arguments': {'record_id': unseen.evidence_id}, 'focus_evidence_ids': []}})
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='model_input',
        payload={'model_view': {'content_observed_evidence_ids': [unseen.evidence_id],
            'focus_evidence_ids': [unseen.evidence_id], 'task': 'Sibling private investigation'}})
    if legacy_events:
        with store._connect() as db:
            for event in store.trajectory_events(actor_role='HI', actor_id=task.task_id):
                payload = event['payload']
                payload.pop('working_memory', None)
                db.execute('UPDATE trajectory_events SET payload_json=? WHERE id=?',
                    (json.dumps(payload), event['id']))
    store.set_workspace_run_status(task.task_id, 'cancelled', error={'type': 'cancelled', 'message': 'shutdown'})

    class FinishFromMemory:
        calls = 0
        async def decide(self, *, workspace_id, model_view, actions):
            self.calls += 1
            assert model_view == expected
            assert 'Sibling private investigation' not in json.dumps(model_view)
            return {'task_id': workspace_id, 'status': 'COMPLETED', 'findings': [{
                'statement': evidence.content, 'evidence_ids': [evidence.evidence_id], 'limitations': []}],
                'selected_evidence_ids': [evidence.evidence_id]}
    policy = FinishFromMemory()
    runtime = InvestigationRuntime(store=store, gia_policy=FinishAfterResumeGia(evidence.evidence_id),
        hi_policy=policy, max_rounds=1, workspace_harness_type=workspace_type)
    async with runtime:
        result = await runtime.run(task='Investigation', original_material_ids=task.materials)
    assert result.cited_evidence == (evidence,)
    assert policy.calls == 1
    assert len(store.evidence_records()) == 2


@pytest.mark.asyncio
async def test_restore_after_observation_before_next_input_preserves_focus(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = add_text_evidence(store, 'Source')
    task = WorkspaceTask('W1-1', 'Read source', ())
    class StopAfterObservation:
        async def decide(self, **kwargs):
            return {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id, 'extra': True},
                    'focus_evidence_ids': [evidence.evidence_id]}
    harness = WorkspaceHarness(task=task, store=store)
    original = store.record_event
    def interrupt_after_observation(**kwargs):
        event_id = original(**kwargs)
        if kwargs['event_type'] == 'observation':
            raise OSError('process interrupted after committed observation')
        return event_id
    store.record_event = interrupt_after_observation
    with pytest.raises(OSError):
        await WorkspaceRunner(harness=harness, policy=StopAfterObservation()).run()
    store.record_event = original
    restored = WorkspaceHarness(task=task, store=store)
    restored.restore_working_memory()
    assert restored.model_view() == harness.model_view()
    assert restored.memory.focus_evidence_ids == [evidence.evidence_id]
    assert restored.memory.content_observed_evidence_ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_events', [False, True])
async def test_restore_retains_inspected_content_and_local_candidates(tmp_path, legacy_events):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    image = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    task = WorkspaceTask('W1-1', 'Inspect and find a source', (image.evidence_id,))
    candidate, _ = store.get_or_create_candidate(identity_key='url:https://example.test/local',
        source_url='https://example.test/local', candidate_type='page')
    private, _ = store.get_or_create_candidate(identity_key='url:https://example.test/sibling',
        source_url='https://example.test/sibling', candidate_type='page')
    class Search:
        async def execute(self, *args, **kwargs):
            return SearchResponse('text', 'target_reached', (SearchResult(candidate,
                'https://example.test/local', 'Source', None,
                (MaterialResult('page', 'absent'), MaterialResult('document', 'absent'))),))
    class Interrupted:
        calls = 0
        async def decide(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {'action': 'INSPECT', 'arguments': {'targets': [image.evidence_id],
                    'request': {'operation': 'visual_qa', 'params': {'question': 'Image dimensions?'}}},
                    'focus_evidence_ids': [image.evidence_id]}
            if self.calls == 2:
                return {'action': 'SEARCH', 'arguments': {'input_type': 'text', 'query': 'Source',
                    'output_type': 'text'}, 'focus_evidence_ids': []}
            raise asyncio.CancelledError
    service = create_inspect_service(store=store)
    harness = WorkspaceHarness(task=task, store=store, inspect_service=service, search_service=Search())
    try:
        with pytest.raises(asyncio.CancelledError):
            await WorkspaceRunner(harness=harness, policy=Interrupted()).run()
        expected = harness.model_view()
        assert image.evidence_id in expected['content_observed_evidence_ids']
        assert len(expected['content_observed_evidence_ids']) >= 2
        if legacy_events:
            with store._connect() as db:
                for event in store.trajectory_events(actor_role='HI', actor_id=task.task_id):
                    payload = event['payload']
                    payload.pop('working_memory', None)
                    db.execute('UPDATE trajectory_events SET payload_json=? WHERE id=?',
                        (json.dumps(payload), event['id']))
        restored = WorkspaceHarness(task=task, store=store, inspect_service=service, search_service=Search())
        restored.restore_working_memory()
        assert restored.model_view() == expected
        assert restored.record_access.read(candidate)['candidate_id'] == candidate
        with pytest.raises(PermissionError):
            restored.record_access.read(private)
    finally:
        await service.aclose()
