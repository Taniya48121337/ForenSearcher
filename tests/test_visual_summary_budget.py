import copy
import json

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.memory import (
    MemoryPolicy, bound_global_view, investigation_visual_observations, workspace_visual_observations,
)
from foren_searcher.model_images import ModelImageInputs
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness


def observed_video(store):
    video = store.record_evidence(content='Video', label='Video', modalities=('video',),
        source_ref='I0', locator='source', acquired_by='TEST', discovered_by='HARNESS')
    frames = [store.commit_derived_evidence(derived_from=(video.evidence_id,), modalities=('image',),
        label='Frame', content='frame', locator=json.dumps({'time_s': i * .5, 'frame_duration_s': .1}),
        acquired_by='INSPECT/get_frames', discovered_by='W1-1', production_method='derived_media')
        for i in range(40)]
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
        payload={'image_evidence_ids': [f.evidence_id for f in frames], 'reasoning': 'PRIVATE'})
    return video, workspace_visual_observations(store, 'W1-1')[1]


def test_global_view_trims_coverage_details_before_discarding_findings():
    store = GlobalStore()
    source, summary = observed_video(store)
    report = {'report': {'task_id': 'W1-1', 'status': 'COMPLETED', 'findings': [
        {'statement': 'Only sampled positions have been inspected.', 'evidence_ids': [source.evidence_id]}]},
        'run_status': 'reported', 'visual_observation_summary': summary}
    view = {'task': 'Check the video', 'original_materials': [], 'current_materials': [],
            'latest_reports': [report], 'history_index': []}
    original = copy.deepcopy(view)
    bounded = bound_global_view(view, 900)
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 900
    assert bounded['latest_reports'][0]['report'] == report['report']
    assert bounded['context_truncated']
    retained = bounded['latest_reports'][0]['visual_observation_summary']
    assert retained['image_evidence_count'] == 40
    video = retained['videos'][0]
    assert video['observed_frame_count'] == 40
    assert video['coverage_truncated']
    assert 'unconfirmed_ranges_s' not in video and 'observed_ranges_s' not in video
    assert view == original
    assert workspace_visual_observations(store, 'W1-1')[1] == summary
    assert 'PRIVATE' not in json.dumps(bounded)


def test_global_view_can_omit_summary_when_only_status_index_fits():
    store = GlobalStore()
    _, summary = observed_video(store)
    view = {'task': 'Check', 'original_materials': [], 'current_materials': [], 'history_index': [],
        'latest_reports': [{'report': {'task_id': 'W1-1', 'status': 'COMPLETED'},
                            'visual_observation_summary': summary}],
        'visual_observation_summary': summary}
    bounded = bound_global_view(view, 380)
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 380
    assert bounded['latest_reports'][0]['report']['status'] == 'COMPLETED'
    assert bounded['visual_observation_summary_truncated']
    assert bounded['latest_reports'][0]['visual_observation_summary_truncated']
    assert 'visual_observation_summary' not in bounded


def test_hi_view_summary_obeys_budget_without_changing_observed_history():
    store = GlobalStore()
    source, summary = observed_video(store)
    harness = WorkspaceHarness(store=store, task=WorkspaceTask('W1-1', 'Check', (source.evidence_id,)),
        memory_policy=MemoryPolicy(max_model_view_chars=1000))
    view = harness.model_view()
    view['visual_observation_summary'] = copy.deepcopy(summary)
    harness.bound_model_view(view)
    assert len(json.dumps(view, ensure_ascii=False)) <= 1000
    assert view['visual_observation_summary']['videos'][0]['observed_frame_count'] == 40
    assert view['visual_observation_summary']['videos'][0]['coverage_truncated']
    assert view['context_truncated']
    assert workspace_visual_observations(store, 'W1-1')[1] == summary


def test_global_counts_deduplicate_workspaces_and_rounds_without_counting_unreceived_images():
    store = GlobalStore()
    source, summary = observed_video(store)
    frames, _ = workspace_visual_observations(store, 'W1-1')
    extra = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('image',),
        label='Later frame', content='frame', locator='{"time_s":20,"frame_duration_s":0.1}',
        acquired_by='INSPECT/get_frames', discovered_by='W2-1', production_method='derived_media')
    crop = store.commit_derived_evidence(derived_from=(extra.evidence_id,), modalities=('image',),
        label='Crop', content='crop', locator='{"time_s":20,"box":[0,0,0.5,0.5]}',
        acquired_by='INSPECT/crop', discovered_by='W2-1', production_method='derived_media')
    unseen = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('image',),
        label='Unreceived frame', content='frame', locator='{"time_s":21,"frame_duration_s":0.1}',
        acquired_by='INSPECT/get_frames', discovered_by='W2-2', production_method='derived_media')
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
        payload={'image_evidence_ids': [item['evidence_id'] for item in frames], 'reasoning': 'SIBLING_PRIVATE'})
    store.record_event(actor_role='HI', actor_id='W2-1', event_type='raw_model_output',
        payload={'image_evidence_ids': [frames[0]['evidence_id'], extra.evidence_id, crop.evidence_id]})
    store.record_event(actor_role='HI', actor_id='W2-2', event_type='model_input',
        payload={'image_evidence_ids': [unseen.evidence_id]})
    store.record_event(actor_role='GIA', actor_id='GIA', event_type='raw_model_output',
        payload={'image_evidence_ids': [unseen.evidence_id], 'reasoning': 'GIA_PRIVATE'})
    combined = investigation_visual_observations(store)
    assert combined['image_evidence_count'] == 42
    assert combined['videos'][0]['observed_frame_count'] == 41
    assert combined['videos'][0]['coverage_span_s'] == [0., 20.1]
    assert combined['videos'][0]['observed_ranges_s'][-1] == [20., 20.1]
    assert workspace_visual_observations(store, 'W1-1')[1] == summary
    assert 'PRIVATE' not in json.dumps(combined)
    assert 'PRIVATE' not in json.dumps(WorkspaceHarness(store=store,
        task=WorkspaceTask('W3-1', 'Independent', ())).model_view())


def test_gia_global_summary_is_independent_of_report_image_list_truncation():
    store = GlobalStore()
    _, summary = observed_video(store)
    view = {'task': 'Inspect', 'latest_reports': [], 'history_index': []}
    # The generator can collect persisted HI observations even when a report index was omitted.
    ModelImageInputs(store=store).prepare_global_model_view(view)
    assert view['visual_observation_summary'] == summary
    assert view['image_inputs'] == []
    bounded = bound_global_view(view, 600)
    assert bounded['visual_observation_summary']['videos'][0]['observed_frame_count'] == 40
    assert bounded['visual_observation_summary']['videos'][0]['coverage_truncated']
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 600
