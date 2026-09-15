import base64
import copy
import io
import json

import pytest
from PIL import Image

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.model_images import ModelImageInputs, _frame_timing_caption
from foren_searcher.store import GlobalStore
from inspect_helpers import image_bytes, ingest, make_video


@pytest.mark.parametrize('encoded', [False, True])
def test_frame_interval_uses_declared_duration_and_preserves_requested_time(encoded):
    locator = {'time_s': .2, 'frame_duration_s': .1, 'requested_time_s': .27}
    result = _frame_timing_caption(json.dumps(locator) if encoded else locator)
    assert result['locator'] == locator
    assert result['frame_presentation_interval_s'] == {
        'start': .2, 'end': .3, 'start_inclusive': True, 'end_inclusive': False}


@pytest.mark.parametrize('locator', [
    {'time_s': 1., 'fps': 30, 'sample_interval_s': .5},
    {'time_s': 1., 'frame_duration_s': None},
    {'time_s': 1., 'frame_duration_s': 0.},
    {'time_s': 1., 'frame_duration_s': -.1},
    {'time_s': 1., 'frame_duration_s': True},
    {'time_s': -.1, 'frame_duration_s': .1},
    {'time_s': 1., 'frame_duration_s': '0.1'},
    {'time_s': 1., 'frame_duration_s': float('nan')},
    {'time_s': 1e308, 'frame_duration_s': 1e308},
    {'time_s': 10**400, 'frame_duration_s': 1.},
    {'preview': '{"time_s":1', 'truncated': True},
    '[not-json', '['*1000+'0'+']'*1000,
])
def test_unknown_or_invalid_timing_does_not_create_a_frame_interval(locator):
    assert _frame_timing_caption(locator) == {}


@pytest.mark.asyncio
async def test_real_frame_captions_preserve_pixels_and_do_not_fill_sampling_gaps(tmp_path):
    store = GlobalStore(tmp_path/'state.sqlite3', tmp_path/'artifacts')
    path = tmp_path/'video.avi'
    make_video(path)
    video = ingest(store, path.read_bytes(), source='I0', filename=path.name,
                   media_type='video/x-msvideo', modality='video')
    async with create_inspect_service(store=store, enabled_operations=frozenset({'get_frames'})) as service:
        response = await service.execute({'targets': [video.evidence_id], 'request': {
            'operation': 'get_frames', 'params': {'times_s': [.1, .3]}}}, workspace_id='W1-1')
    images = []
    for frame in response.result.frames:
        evidence = store.evidence(frame.evidence_id)
        images.append({'evidence_id': evidence.evidence_id, 'locator': evidence.locator,
                       'derived_from': list(evidence.derived_from), 'width': frame.width, 'height': frame.height})
    view = {'image_inputs': images}
    original = copy.deepcopy(view)
    renderer = ModelImageInputs(store=store, image_policy=InspectPolicy(maximum_model_image_pixels=160*96))
    message, ids = renderer._image_message(view)
    parts = message['content']
    intervals = []
    for i, part in enumerate(parts):
        if part['type'] != 'image_url':
            continue
        label = json.loads(parts[i-1]['text'])
        intervals.append(label['frame_presentation_interval_s'])
        evidence = store.evidence(label['evidence_id'])
        assert label['locator'] == json.loads(evidence.locator)
        with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))) as actual:
            with Image.open(store.artifact_path(evidence.artifact_ref)) as expected:
                assert actual.convert('RGB').tobytes() == expected.convert('RGB').tobytes()
    assert [(i['start'], i['end']) for i in intervals] == [(.1, .2), (.3, .4)]
    assert all(i['start_inclusive'] and not i['end_inclusive'] for i in intervals)
    assert ids == [image['evidence_id'] for image in images] and view == original


def test_still_image_caption_keeps_original_locator_and_has_no_video_interval(tmp_path):
    store = GlobalStore(tmp_path/'state.sqlite3', tmp_path/'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='image.png', media_type='image/png', modality='image')
    image = {'evidence_id': evidence.evidence_id, 'locator': '{"time_s":0,"frame_duration_s":1}',
             'derived_from': [], 'width': 320, 'height': 180}
    renderer = ModelImageInputs(store=store, image_policy=InspectPolicy(maximum_model_image_pixels=320*180))
    message, _ = renderer._image_message({'image_inputs': [image]})
    label = json.loads(message['content'][1]['text'])
    assert label == {**image, 'model_image_width': 320, 'model_image_height': 180}
