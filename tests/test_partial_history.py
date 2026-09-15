import copy

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, WorkspaceHistoryEntry, WorkspaceMemory
from inspect_helpers import ingest, make_video, pdf_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,field,selection,changed_selection', [
    ('frames', 'times_s', [0, .1, 99], [0, .2, 99]),
    ('pages', 'pages', [1, 2, 99], [1, 99]),
])
async def test_repeated_partial_media_keep_valid_results_errors_and_execution_count(tmp_path, kind, field, selection, changed_selection):
    store = GlobalStore(tmp_path / 'store.sqlite3', tmp_path / 'artifacts')
    if kind == 'frames':
        path = tmp_path / 'source.avi'
        make_video(path, frames=5)
        media = ingest(store, path.read_bytes(), source='I0', filename='source.avi', media_type='video/avi', modality='video')
    else:
        media = ingest(store, pdf_bytes(), source='I0', filename='pages.pdf', media_type='application/pdf', modality='text')
    call = {'targets': [media.evidence_id], 'request': {'operation': 'get_' + kind, 'params': {field: selection}}}
    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Compare images', (media.evidence_id,)), store=store, inspect_service=service)
        for _ in range(3):
            result = await harness.execute_inspect(call)
            assert result.status == 'partial'
        original = copy.deepcopy(harness.memory.history)
        view = harness.memory.to_model_view()
        assert len(view['history']) == 1
        assert view['history'][0]['completed_occurrences'] == 3
        assert view['latest_observation']['details']['status'] == 'partial'
        assert len(view['latest_observation']['details']['result'][kind]) == 2
        assert view['latest_observation']['details']['feedback'][0]['code'] == 'OUT_OF_RANGE'
        assert harness.memory.history == original
        harness.set_focus([getattr(result.result, kind)[0].evidence_id])
        await harness.execute_inspect(call)
        assert len(harness.memory.to_model_view()['history']) == 2
        changed = copy.deepcopy(call)
        changed['request']['params'][field] = changed_selection
        await harness.execute_inspect(changed)
        assert len(harness.memory.to_model_view()['history']) == 3


@pytest.mark.parametrize('feedback', [[{'retryable': True}], [{}], []])
def test_partial_history_with_transient_or_unspecified_feedback_is_retained(feedback):
    observation = {'action': 'INSPECT', 'items': ['partial'], 'details': {
        'status': 'partial', 'result': {'frames': []}, 'feedback': feedback}}
    entries = [WorkspaceHistoryEntry('INSPECT', {'targets': ['E1']}, {}, copy.deepcopy(observation), ()) for _ in range(3)]
    memory = WorkspaceMemory('W1-1', 'Inspect frames', {}, history=entries, latest_observation=observation)
    assert len(memory.to_model_view()['history']) == 3


def test_partial_history_preserves_changes_in_results_and_feedback():
    observation = {'action': 'INSPECT', 'items': ['partial'], 'details': {
        'status': 'partial', 'result': {'frames': [{'evidence_id': 'E2'}]},
        'feedback': [{'code': 'OUT_OF_RANGE', 'retryable': False, 'locator': {'times_s': [99]}}]}}
    changed_result = copy.deepcopy(observation)
    changed_result['details']['result']['frames'][0]['evidence_id'] = 'E3'
    changed_error = copy.deepcopy(observation)
    changed_error['details']['feedback'][0]['locator']['times_s'] = [98]
    entries = [WorkspaceHistoryEntry('INSPECT', {'targets': ['E1']}, {}, obs, ())
               for obs in (observation, changed_result, changed_error)]
    memory = WorkspaceMemory('W1-1', 'Inspect frames', {}, history=entries, latest_observation=changed_error)
    assert len(memory.to_model_view()['history']) == 3
