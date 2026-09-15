from dataclasses import replace
import json

import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.openrouter import OpenRouterJsonGenerator
from foren_searcher.store import GlobalStore
from test_temporal_alignment import scenes, write_video


@pytest.mark.asyncio
@pytest.mark.parametrize('capacity', [3, 4, 5, 8])
async def test_disjoint_video_matches_supply_complete_pairs_within_pixel_capacity(tmp_path, capacity):
    frames = scenes(72)
    paths = [tmp_path / 'a.avi', tmp_path / 'b.avi']
    write_video(paths[0], frames)
    write_video(paths[1], frames[4:18] + frames[62:70])
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = [store.commit_ingested_evidence(source_ref=f'I{i}', media_type='video/x-msvideo',
        modalities=('video',), label=p.name, original_filename=p.name, path=p) for i, p in enumerate(paths)]
    policy = replace(InspectPolicy(), maximum_visual_qa_images=capacity)
    async with create_inspect_service(store=store, policy=policy) as service:
        response = await service.execute({'targets': [e.evidence_id for e in evidence],
            'request': {'operation': 'align_media', 'params': {'mode': 'visual', 'top_k': 2}}}, workspace_id='W1-1')
    assert response.status == 'ready'
    matches = response.result.to_dict()['matches']
    assert len(matches) == 2
    visible_matches = matches[:capacity // 4]
    for match in visible_matches:
        for anchor in (match['anchors'][0], match['anchors'][-1]):
            assert anchor['image_a_evidence_id'] and anchor['image_b_evidence_id']
            for side in ('a', 'b'):
                image = store.evidence(anchor[f'image_{side}_evidence_id'])
                assert json.loads(image.locator)['time_s'] == pytest.approx(anchor[f'time_{side}_s'], abs=.001)
    view = {'task_id': 'W1-1', 'latest_observation': {'action': 'INSPECT', 'details': {
        'operation': 'align_media', 'result': response.result.to_dict(),
        'evidence': [e.to_dict() for e in response.evidence]}}}
    generator = OpenRouterJsonGenerator(None, store=store, image_policy=policy)
    generator.prepare_model_view(view)
    ids = [item['evidence_id'] for item in view['image_inputs']]
    assert len(ids) <= capacity
    for match in visible_matches:
        for anchor in (match['anchors'][0], match['anchors'][-1]):
            assert anchor['image_a_evidence_id'] in ids and anchor['image_b_evidence_id'] in ids
    assert len(view['alignment_image_pairs']) == capacity // 2
    assert len(ids) == 2 * len(view['alignment_image_pairs'])


@pytest.mark.asyncio
async def test_repeated_pixels_cannot_establish_a_unique_edit_boundary(tmp_path):
    frames = [scenes(18)[0]] * 100
    paths = [tmp_path / 'a.avi', tmp_path / 'b.avi']
    write_video(paths[0], frames)
    write_video(paths[1], frames[40:80])
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    media = [store.commit_ingested_evidence(source_ref=f'I{i}', media_type='video/x-msvideo',
        modalities=('video',), label=p.name, original_filename=p.name, path=p) for i, p in enumerate(paths)]
    async with create_inspect_service(store=store) as service:
        response = await service.execute({'targets': [e.evidence_id for e in media],
            'request': {'operation': 'align_media', 'params': {'mode': 'visual'}}}, workspace_id='W1-1')
    assert response.status == 'ready' and response.result.matches
    saved = json.loads(store.evidence(response.result_evidence_id).content)
    # 相同画面可对应多个位置；已知编辑起点不能由采样步长给出误差保证。
    assert any(abs(m.range_a_s[0] - 4) > saved['sampling_interval_s'][0] for m in response.result.matches)
    assert saved['boundary_uncertainty_s'] is None
