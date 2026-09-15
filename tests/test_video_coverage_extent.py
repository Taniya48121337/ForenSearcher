import json

import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.memory import workspace_visual_observations
from foren_searcher.store import GlobalStore
from inspect_helpers import ingest, make_video


@pytest.mark.asyncio
@pytest.mark.parametrize('fps', [5., 30.])
async def test_video_coverage_includes_the_unseen_tail_at_the_probed_duration(tmp_path, fps):
    path = tmp_path / 'clip.avi'
    make_video(path, frames=30, fps=fps)
    store = GlobalStore(tmp_path / 'case.sqlite3', tmp_path / 'artifacts')
    source = ingest(store, path.read_bytes(), source='I0', filename=path.name, media_type='video/avi', modality='video')
    async with create_inspect_service(store=store) as service:
        response = await service.execute({'targets': [source.evidence_id], 'request': {
            'operation': 'get_frames', 'params': {'times_s': [1 / fps, 4 / fps]}}}, workspace_id='W1-1')
    assert response.status == 'ready'
    frames = response.result.frames
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
        payload={'image_evidence_ids': [frames[0].evidence_id]})
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
        payload={'image_evidence_ids': [frames[1].evidence_id]})
    _, first = workspace_visual_observations(store, 'W1-1')
    coverage = first['videos'][0]
    assert coverage['observed_frame_count'] == 1
    assert coverage['coverage_span_s'] == pytest.approx([0, 30 / fps])
    assert coverage['media_duration_s'] == pytest.approx(30 / fps)
    assert coverage['unconfirmed_ranges_s'][-1] == pytest.approx([2 / fps, 30 / fps])
    _, current = workspace_visual_observations(store, 'W1-1', [frames[1].evidence_id])
    coverage = current['videos'][0]
    assert coverage['observed_frame_count'] == 2
    assert coverage['unconfirmed_ranges_s'][-1] == pytest.approx([5 / fps, 30 / fps])
    assert len(coverage['observed_ranges_s']) == 2


@pytest.mark.parametrize('length', [None, 0, -1, True, '3', float('inf'), float('nan')])
def test_invalid_profile_duration_does_not_invent_a_full_video_extent(tmp_path, length):
    store = GlobalStore(tmp_path / 'case.sqlite3', tmp_path / 'artifacts')
    source = store.record_evidence(content='Video source', label='Video', modalities=('video',),
        source_ref=None, locator='source', acquired_by='TEST/source', discovered_by='W1-1')
    frame = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('image',),
        label='Frame', content='Frame', locator=json.dumps({'time_s': 1., 'frame_duration_s': .1}),
        acquired_by='INSPECT/get_frames', discovered_by='W1-1', production_method='derived_media')
    store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('text',),
        label='Media profile', content=json.dumps({'media_kind': 'video', 'duration_s': length}), locator='probe',
        acquired_by='INSPECT/media_profile', discovered_by='W1-1', production_method='media_probe')
    _, summary = workspace_visual_observations(store, 'W1-1', [frame.evidence_id])
    coverage = summary['videos'][0]
    assert 'media_duration_s' not in coverage
    assert coverage['coverage_span_s'] == [0., 1.1]
    assert coverage['observed_ranges_s'] == [[1., 1.1]]

@pytest.mark.parametrize('frames,duration,observed,gaps,unknown', [
    ({0.: 2.4, 2.4: .05}, 2.45, [[0., 2.45]], [], []),
    ({0.: .2, .2: .1}, .3, [[0., .3]], [], []),
    ({0.: 2.4, 2.4: .04}, 2.45, [[0., 2.44]], [[2.44, 2.45]], []),
    ({0.: 2.4, 2.4: .04999999}, 2.45, [[0., 2.44999999]], [[2.44999999, 2.45]], []),
    ({0.: 2.4, 2.4: None}, 2.45, [[0., 2.4]], [[2.4, 2.45]], [2.4]),
    ({i / 30: 1 / 30 for i in range(30)}, 1., [[0., 1.]], [], []),
])
def test_video_coverage_preserves_real_gaps_without_decimal_rounding_gaps(
        tmp_path, frames, duration, observed, gaps, unknown):
    store = GlobalStore(tmp_path / 'case.sqlite3', tmp_path / 'artifacts')
    source = store.record_evidence(content='Video', label='Video', modalities=('video',),
        source_ref=None, locator='source', acquired_by='TEST/source', discovered_by='W1-1')
    ids = []
    for start, length in frames.items():
        frame = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('image',),
            label='Frame', content='Frame', locator=json.dumps({'time_s': start, 'frame_duration_s': length}),
            acquired_by='INSPECT/get_frames', discovered_by='W1-1', production_method='derived_media')
        ids.append(frame.evidence_id)
    store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('text',),
        label='Media profile', content=json.dumps({'media_kind': 'video', 'duration_s': duration}), locator='probe',
        acquired_by='INSPECT/media_profile', discovered_by='W1-1', production_method='media_probe')
    _, summary = workspace_visual_observations(store, 'W1-1', ids)
    video = summary['videos'][0]
    assert video['observed_ranges_s'] == observed
    assert video['unconfirmed_ranges_s'] == gaps
    assert video['coverage_span_s'] == [0., duration]
    assert video['unknown_duration_frame_times_s'] == unknown
