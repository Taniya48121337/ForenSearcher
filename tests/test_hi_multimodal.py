"""HI 直接像素输入、真实媒体处理与独立工作区的传输验证。"""

import asyncio
import base64
import io
import json
import math
import os
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.harness1 import harness1_workspace_type
from foren_searcher.inspect.contracts import inspect_action_schema
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.local_models import TransformersHiPolicy
from foren_searcher.memory import MemoryPolicy
from foren_searcher.openrouter import OpenRouterError, OpenRouterJsonGenerator, OpenRouterPolicyBundle
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, WorkspaceRunner, WorkspaceTurnLimit
from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes


OPERATIONS = frozenset(branch['properties']['request']['properties']['operation']['const']
    for branch in inspect_action_schema()['oneOf']) - {'visual_qa', 'audio_qa'}


class RecordingClient:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    async def complete(self, messages, *, retry_observer=None, **kwargs):
        self.calls.append({'messages': messages, **kwargs})
        await asyncio.sleep(0)
        value = self.responder(messages)
        return {'text': json.dumps(value), 'model': 'qwen/qwen3.8-27b',
                'provider': 'test', 'generation_id': 'test', 'usage': {}}


def attached(messages):
    images = []
    for message in messages:
        parts = message['content']
        if not isinstance(parts, list):
            continue
        for i, part in enumerate(parts):
            if part['type'] == 'image_url':
                label = json.loads(parts[i - 1]['text'])
                image = Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',', 1)[1])))
                images.append((label, image))
    return images


def completed(workspace, ids):
    return {'task_id': workspace, 'status': 'COMPLETED', 'findings': [
        {'statement': 'The supplied pixels contain a colored rectangle.', 'evidence_ids': ids,
         'limitations': ['Pixel transport test only.']}], 'selected_evidence_ids': ids}


@pytest.mark.asyncio
async def test_native_reasoning_stays_with_its_workspace_and_survives_restore(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color=color), source=f'I{i}', filename=f'{i}.png',
                     media_type='image/png', modality='image') for i, color in enumerate(['red', 'blue'])]
    calls = {}
    recorded = []

    class Client:
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(0)
            payload = json.loads(messages[1]['content'])
            if 'workspace_id' not in payload:
                assert 'private observation' not in json.dumps(messages)
                assert not any('reasoning' in m for m in messages)
                return {'text': '{"decision":"FINISH","answer":"Combined findings."}',
                        'model': 'qwen/qwen3.8-27b', 'provider': 'test', 'generation_id': 'test', 'usage': {}}
            actor = payload['workspace_id']
            index = int(actor[-1]) - 1
            calls[actor] = calls.get(actor, 0) + 1
            reasoning = [m['reasoning'] for m in messages if 'reasoning' in m]
            if calls[actor] == 1:
                assert reasoning == []
                output = {'action': 'READ_RECORD', 'arguments': {'record_id': sources[index].evidence_id},
                          'focus_evidence_ids': [sources[index].evidence_id]}
            else:
                assert reasoning == [f'private observation for {actor}']
                output = completed(actor, [sources[index].evidence_id])
            recorded.append((actor, reasoning))
            return {'text': json.dumps(output), 'reasoning': f'private observation for {actor}',
                    'model': 'qwen/qwen3.8-27b', 'provider': 'test', 'generation_id': 'test', 'usage': {}}

    generator = OpenRouterJsonGenerator(Client(), store=store)
    generator.generation_observer = lambda value: store.record_event(actor_role='HI' if value.get('workspace_id') else 'GIA',
        actor_id=value.get('workspace_id') or 'GIA', event_type='raw_model_output', payload=value)
    async def run(index):
        actor = f'W1-{index+1}'
        task = WorkspaceTask(actor, 'Describe the supplied image.', (sources[index].evidence_id,))
        harness = WorkspaceHarness(task=task, store=store)
        return await WorkspaceRunner(harness=harness, policy=TransformersHiPolicy(generator), max_turns=2).run()
    reports = await asyncio.gather(run(0), run(1))
    assert len(reports) == 2 and calls == {'W1-1': 2, 'W1-2': 2}
    assert len(store.evidence_records()) == 2
    for actor in calls:
        last = next(e['payload']['model_view'] for e in reversed(store.trajectory_events(actor_role='HI', actor_id=actor))
                    if e['event_type'] == 'model_input')
        OpenRouterJsonGenerator(Client(), store=store).prepare_model_view(last)
        assert [h['reasoning'] for h in last['history'] if 'reasoning' in h] == [f'private observation for {actor}']
    await generator.generate_json(system_prompt='Summarize the reports.', payload={
        'global_view': {'task': 'Combine the findings.', 'latest_reports': [r.to_dict() for r in reports]}})


@pytest.mark.parametrize('allow_reasoning', [False, True])
@pytest.mark.parametrize('backend', ['remote', 'local_text'])
def test_native_reasoning_uses_complete_blocks_within_view_budget(tmp_path, allow_reasoning, backend):
    from foren_searcher.local_models import LocalModelConfig, TransformersJsonGenerator
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    output = {'action': 'READ_RECORD', 'arguments': {'record_id': 'E1'}, 'focus_evidence_ids': []}
    for actor, text in [('W1-1', 'own block ' * 100), ('W1-2', 'sibling block')]:
        store.record_event(actor_role='HI', actor_id=actor, event_type='raw_model_output',
            payload={'raw_output': json.dumps(output), 'reasoning': text})
        store.record_event(actor_role='HI', actor_id=actor, event_type='model_output', payload={'output': output})
        store.record_event(actor_role='HI', actor_id=actor, event_type='observation', payload={})
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='model_input', payload={})
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
        payload={'raw_output': json.dumps(output), 'reasoning': 'unexecuted response'})
    view = {'task_id': 'W1-1', 'history': [{'action': 'READ_RECORD', 'request': {'record_id': 'E1'},
              'focus_evidence_ids': [], 'observation': {}}], 'runtime_constraints': {'max_model_view_chars': 4096 if allow_reasoning else 700}}
    generator = (OpenRouterJsonGenerator(None, store=store) if backend == 'remote' else
                 TransformersJsonGenerator(LocalModelConfig('unused'), store=store))
    generator.prepare_model_view(view)
    assert len(json.dumps(view, ensure_ascii=False)) <= view['runtime_constraints']['max_model_view_chars']
    assert view['history'][0].get('reasoning') == ('own block ' * 100 if allow_reasoning else None)
    assert view.get('reasoning_context_omitted', False) is (not allow_reasoning)


@pytest.mark.asyncio
@pytest.mark.parametrize('media_kind', ['pdf', 'video'])
@pytest.mark.parametrize('invalid_item', [False, True])
async def test_real_pages_and_frames_reach_hi_with_original_locators(tmp_path, media_kind, invalid_item):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    if media_kind == 'pdf':
        source = ingest(store, pdf_bytes(), source='I0', filename='source.pdf',
                        media_type='application/pdf', modality='text')
        operation, params = 'get_pages', {'pages': [2]}
    else:
        path = tmp_path / 'source.avi'
        make_video(path)
        source = ingest(store, path.read_bytes(), source='I0', filename='source.avi',
                        media_type='video/x-msvideo', modality='video')
        operation, params = 'get_frames', {'times_s': [2.0]}
    if invalid_item:
        params['pages' if media_kind == 'pdf' else 'times_s'].append(3)
    service = create_inspect_service(store=store, enabled_operations=OPERATIONS)
    task = WorkspaceTask('W1-1', 'Describe the selected page or frame.', (source.evidence_id,))
    calls = 0

    def respond(messages):
        nonlocal calls
        calls += 1
        images = attached(messages)
        if calls == 1:
            assert images == []
            return {'action': 'INSPECT', 'arguments': {'targets': [source.evidence_id],
                'request': {'operation': operation, 'params': params}}, 'focus_evidence_ids': []}
        assert len(images) == 1
        label, picture = images[0]
        evidence = store.evidence(label['evidence_id'])
        assert evidence.derived_from == (source.evidence_id,)
        assert label['derived_from'] == [source.evidence_id]
        location = json.loads(evidence.locator)
        if media_kind == 'pdf':
            assert label['locator'] == evidence.locator
            assert location['page_number'] == 2
            assert picture.getpixel((picture.width // 2, picture.height // 2))[2] > 240
        else:
            assert label['locator'] == location
            assert label['frame_presentation_interval_s'] == {
                'start': 2., 'end': 2.1, 'start_inclusive': True, 'end_inclusive': False}
            assert location['time_s'] == pytest.approx(2.0)
            assert picture.getpixel((round(80 * picture.width / label['width']),
                                     round(50 * picture.height / label['height'])))[0] > 240
        return completed(task.task_id, [evidence.evidence_id])

    client = RecordingClient(respond)
    bundle = OpenRouterPolicyBundle(client, store=store, image_policy=service.policy)
    assert bundle.gia_model is bundle.hi_model
    workspace_type = harness1_workspace_type(os.environ['FOREN_HARNESS_ROOT']) if os.environ.get('FOREN_HARNESS_ROOT') else WorkspaceHarness
    harness = workspace_type(task=task, store=store, inspect_service=service)
    try:
        report = await WorkspaceRunner(harness=harness, policy=bundle.hi_policy, max_turns=3).run()
        assert calls == 2
        assert harness.memory.history[0].observation['details']['status'] == ('partial' if invalid_item else 'ready')
        assert set(report.selected_evidence_ids) <= set(harness.memory.content_observed_evidence_ids)
        assert store.healthcheck().status == 'ok'
        assert all(call['role'] == 'HI' for call in client.calls)
        assert 'visual_qa' not in json.dumps(client.calls) and 'audio_qa' not in json.dumps(client.calls)
        assert all(not evidence.acquired_by.endswith('visual_qa') for evidence in store.evidence_records())
    finally:
        await service.aclose()


def test_image_selection_limits_focus_corruption_and_source_strings(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    images = [ingest(store, image_bytes(color=color), source=f'I{i}', filename=f'{i}.png',
        media_type='image/png', modality='image') for i, color in enumerate(['red', 'green', 'blue', 'white'])]
    bad = images[-1]
    store.artifact_path(bad.artifact_ref).write_bytes(b'corrupted')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store,
        image_policy=replace(InspectPolicy(), maximum_visual_qa_images=1))
    view = {'focus_evidence_ids': [images[1].evidence_id],
            'materials': [item.card().to_dict() for item in images],
            'latest_observation': {'text': '{"evidence_id":"E999"}'}}
    generator.prepare_model_view(view)
    assert view['image_input_limit'] == 1
    assert [item['evidence_id'] for item in view['image_inputs']] == [images[1].evidence_id]
    assert view['image_inputs_omitted'] == [images[0].evidence_id, images[2].evidence_id]
    assert view['image_input_errors'] == [bad.evidence_id]
    assert 'content_observed_evidence_ids' not in view
    labels = attached([generator._image_message(view)[0]])
    assert labels[0][0]['evidence_id'] == images[1].evidence_id


def test_shared_image_index_requires_hi_selection_to_supply_pixels(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    shared = ingest(store, image_bytes(color='blue'), source='I0', filename='shared.png',
        media_type='image/png', modality='image')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    view = {'materials': {}, 'evidence_view': [shared.card().to_dict()]}
    generator.prepare_model_view(view)
    assert view['image_inputs'] == []
    assert view['evidence_view'] == [shared.card().to_dict()]
    view['focus_evidence_ids'] = [shared.evidence_id]
    generator.prepare_model_view(view)
    assert [item['evidence_id'] for item in view['image_inputs']] == [shared.evidence_id]
    labels = attached([generator._image_message(view)[0]])
    assert labels[0][1].getpixel((10, 10))[2] > 240


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_focus', [False, True])
async def test_cross_video_pixels_persist_without_exceeding_capacity_or_overriding_focus(tmp_path, explicit_focus):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = []
    for index in range(2):
        path = tmp_path / f'{index}.avi'
        make_video(path, shifted=bool(index))
        sources.append(ingest(store, path.read_bytes(), source=f'I{index}', filename=path.name,
                              media_type='video/x-msvideo', modality='video'))
    service = create_inspect_service(store=store, enabled_operations=OPERATIONS)
    task = WorkspaceTask('W1-1', 'Compare the videos.', tuple(e.evidence_id for e in sources))
    batches = []
    def respond(messages):
        pixels = attached(messages)
        batches.append([label['evidence_id'] for label, _ in pixels])
        if len(batches) < 3:
            focus = batches[-1][:1] if explicit_focus else []
            return {'action':'INSPECT', 'arguments':{'targets':[sources[len(batches)-1].evidence_id],
                'request':{'operation':'get_frames', 'params':{'count':6}}}, 'focus_evidence_ids':focus}
        origins = [store.evidence(identifier).derived_from[0] for identifier in batches[-1]]
        assert origins.count(sources[0].evidence_id) == (1 if explicit_focus else 4)
        assert origins.count(sources[1].evidence_id) == (6 if explicit_focus else 4)
        assert len(pixels) <= 8
        return completed(task.task_id, batches[-1])
    bundle = OpenRouterPolicyBundle(RecordingClient(respond), store=store, image_policy=service.policy)
    bundle.hi_model.generation_observer = lambda value: store.record_event(actor_role='HI',
        actor_id=value['workspace_id'], event_type='raw_model_output', payload=value)
    try:
        await WorkspaceRunner(harness=WorkspaceHarness(task=task, store=store, inspect_service=service),
            policy=bundle.hi_policy, max_turns=3).run()
        a_frames = [store.evidence(identifier).card().to_dict() for identifier in batches[1]]
        fresh = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
        # A new workspace cannot inherit another HI's viewed pixels.
        view = {'task_id':'W2-1', 'latest_observation':{'frames':a_frames[:2]}}
        fresh.prepare_model_view(view)
        assert len(view['image_inputs']) == 2
        # More frames from the same source do not bring back that source's old batch.
        store.record_event(actor_role='HI', actor_id='W2-1', event_type='raw_model_output',
                           payload={'image_evidence_ids':batches[1]})
        fresh.prepare_model_view(view)
        assert len(view['image_inputs']) == 2
    finally:
        await service.aclose()


@pytest.mark.parametrize('pixel_limit', [1_500_000, 200_000])
def test_small_frame_is_resampled_within_model_pixel_budget(tmp_path, pixel_limit):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    data = io.BytesIO()
    Image.new('RGB', (320, 240), 'red').save(data, format='PNG')
    evidence = ingest(store, data.getvalue(), source='I0', filename='frame.png',
        media_type='image/png', modality='image')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store,
        image_policy=replace(InspectPolicy(), maximum_model_image_pixels=pixel_limit))
    view = {'materials': [evidence.card().to_dict()]}
    generator.prepare_model_view(view)
    label, picture = attached([generator._image_message(view)[0]])[0]
    assert label['width'] == 320 and label['height'] == 240
    assert picture.width > 320 and picture.height > 240
    assert picture.width * picture.height <= pixel_limit
    assert picture.width / picture.height == pytest.approx(4 / 3, abs=.005)
    if pixel_limit == 1_500_000:
        assert picture.size == (1414, 1060)
    assert label['model_image_width'] == picture.width
    assert label['model_image_height'] == picture.height
    assert store.artifact_path(evidence.artifact_ref).read_bytes() == data.getvalue()


@pytest.mark.parametrize('limit', ['file', 'pixels'])
def test_oversized_images_are_not_supplied(tmp_path, limit):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    policy = replace(InspectPolicy(), **({'maximum_file_bytes': 20} if limit == 'file' else {'maximum_image_pixels': 100}))
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store, image_policy=policy)
    view = {'materials': [evidence.card().to_dict()]}
    generator.prepare_model_view(view)
    assert view['image_inputs'] == [] and view['image_input_errors'] == [evidence.evidence_id]
    assert generator._image_message(view) == (None, [])


def test_full_focus_batch_still_receives_new_tool_pixels(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    groups = [[ingest(store, image_bytes(color=color), source=f'I{group}-{index}',
        filename=f'{group}-{index}.png', media_type='image/png', modality='image')
        for index in range(8)] for group, color in enumerate(('red', 'blue'))]
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    view = {'focus_evidence_ids': [e.evidence_id for e in groups[0]],
            'latest_observation': {'action': 'INSPECT', 'results': [e.card().to_dict() for e in groups[1]]}}
    generator.prepare_model_view(view)
    pixels = attached([generator._image_message(view)[0]])
    assert len(pixels) == 8
    supplied = [label['evidence_id'] for label, _ in pixels]
    assert [i for i in supplied if i in {e.evidence_id for e in groups[0]}] == [e.evidence_id for e in groups[0][:4]]
    assert [i for i in supplied if i in {e.evidence_id for e in groups[1]}] == [e.evidence_id for e in groups[1][:4]]
    assert sum(picture.getpixel((10, 10))[0] > 240 for _, picture in pixels) == 4
    assert sum(picture.getpixel((10, 10))[2] > 240 for _, picture in pixels) == 4
    assert len(view['image_inputs_omitted']) == 8


def test_harness_rejection_keeps_recent_pixels_in_spare_capacity(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    frames = [ingest(store, image_bytes(color='red' if i < 6 else 'blue'), source=f'I{i}',
        filename=f'frame{i}.png', media_type='image/png', modality='image') for i in range(8)]
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
                       payload={'image_evidence_ids': [e.evidence_id for e in frames[:6]]})
    view = {'task_id': 'W1-1', 'focus_evidence_ids': [e.evidence_id for e in frames[6:]],
            'latest_observation': {'action': 'HARNESS_REJECTION', 'details': {'error': 'invalid parameters'}}}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_model_view(view)
    expected = [e.evidence_id for e in frames[6:] + frames[:6]]
    assert [item['evidence_id'] for item in view['image_inputs']] == expected
    pixels = attached([generator._image_message(view)[0]])
    assert [label['evidence_id'] for label, _ in pixels] == expected
    assert len(pixels) == 8 and view['image_inputs_omitted'] == []
    assert all(p.getpixel((10, 10))[0] > 240 for _, p in pixels[2:])


@pytest.mark.parametrize('actor,role,visual', [('W1-2', 'HI', True), ('GIA', 'GIA', True),
                                            ('W1-1', 'HI', False)])
def test_recent_pixels_require_own_hi_and_current_visual_input(tmp_path, actor, role, visual):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    a, b = [ingest(store, image_bytes(color=color), source=f'I{i}', filename=f'{i}.png',
        media_type='image/png', modality='image') for i, color in enumerate(('red', 'blue'))]
    store.record_event(actor_role=role, actor_id=actor, event_type='raw_model_output',
                       payload={'image_evidence_ids': [a.evidence_id]})
    view = {'task_id': 'W1-1', 'focus_evidence_ids': [b.evidence_id] if visual else [],
            'latest_observation': {'action': 'HARNESS_REJECTION' if visual else 'READ_RECORD'}}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_model_view(view)
    assert [item['evidence_id'] for item in view['image_inputs']] == ([b.evidence_id] if visual else [])


def test_recent_pixels_cannot_displace_focus_or_latest_and_use_latest_batch(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    images = [ingest(store, image_bytes(), source=f'I{i}', filename=f'{i}.png',
        media_type='image/png', modality='image') for i in range(6)]
    for group in (images[:2], images[2:4]):
        store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
                           payload={'image_evidence_ids': [e.evidence_id for e in group]})
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store,
        image_policy=replace(InspectPolicy(), maximum_visual_qa_images=3))
    view = {'task_id': 'W1-1', 'focus_evidence_ids': [e.evidence_id for e in images[4:]],
            'latest_observation': {'action': 'HARNESS_REJECTION'}}
    generator.prepare_model_view(view)
    assert [item['evidence_id'] for item in view['image_inputs']] == [
        images[4].evidence_id, images[5].evidence_id, images[2].evidence_id]


@pytest.mark.asyncio
async def test_report_format_repair_keeps_pixels_without_another_extraction(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    path = tmp_path / 'video.avi'
    make_video(path)
    source = ingest(store, path.read_bytes(), source='I0', filename=path.name,
                    media_type='video/x-msvideo', modality='video')
    task = WorkspaceTask('W1-1', 'Describe the video frames.', (source.evidence_id,))
    calls = []
    def respond(messages):
        ids = [label['evidence_id'] for label, _ in attached(messages)]
        calls.append(ids)
        if len(calls) == 1:
            return {'action': 'INSPECT', 'arguments': {'targets': [source.evidence_id],
                'request': {'operation': 'get_frames', 'params': {'count': 2}}}, 'focus_evidence_ids': []}
        if len(calls) in {2, 3}:
            assert len(ids) == 2
            return 'incomplete report'
        assert ids == calls[1]
        return completed(task.task_id, ids)
    async with create_inspect_service(store=store, enabled_operations=OPERATIONS) as service:
        bundle = OpenRouterPolicyBundle(RecordingClient(respond), store=store, image_policy=service.policy)
        bundle.hi_model.generation_observer = lambda value: store.record_event(actor_role='HI',
            actor_id=value['workspace_id'], event_type='raw_model_output', payload=value)
        harness = WorkspaceHarness(task=task, store=store, inspect_service=service)
        await WorkspaceRunner(harness=harness, policy=bundle.hi_policy, max_turns=3).run()
    assert len(calls) == 4
    assert [entry.action for entry in harness.memory.history] == ['INSPECT', 'HARNESS_REJECTION']


@pytest.mark.asyncio
async def test_failed_inspection_keeps_pixels_for_the_next_decision(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    path = tmp_path / 'video.avi'
    make_video(path)
    source = ingest(store, path.read_bytes(), source='I0', filename=path.name,
                    media_type='video/x-msvideo', modality='video')
    task = WorkspaceTask('W1-1', 'Describe the video frames.', (source.evidence_id,))
    calls = []
    def respond(messages):
        ids = [label['evidence_id'] for label, _ in attached(messages)]
        calls.append(ids)
        if len(calls) < 3:
            return {'action': 'INSPECT', 'arguments': {'targets': [source.evidence_id],
                'request': {'operation': 'get_frames', 'params': {'count': 2} if len(calls) == 1
                            else {'times_s': [1000.0]}}}, 'focus_evidence_ids': ids[:1]}
        assert len(ids) == 2 and ids == calls[1]
        return completed(task.task_id, ids)
    async with create_inspect_service(store=store, enabled_operations=OPERATIONS) as service:
        bundle = OpenRouterPolicyBundle(RecordingClient(respond), store=store, image_policy=service.policy)
        bundle.hi_model.generation_observer = lambda value: store.record_event(actor_role='HI',
            actor_id=value['workspace_id'], event_type='raw_model_output', payload=value)
        harness = WorkspaceHarness(task=task, store=store, inspect_service=service)
        await WorkspaceRunner(harness=harness, policy=bundle.hi_policy, max_turns=3).run()
    assert len(calls) == 3
    assert [entry.action for entry in harness.memory.history] == ['INSPECT', 'INSPECT']
    feedback = harness.memory.history[-1].observation['details']
    assert feedback['status'] == 'failed'
    assert feedback['feedback'][0]['code'] == 'OUT_OF_RANGE'


@pytest.mark.asyncio
@pytest.mark.parametrize('compressed,keep_focus', [(False, False), (True, False), (True, True)])
async def test_frame_overflow_delivers_the_full_requested_span(tmp_path, compressed, keep_focus):
    from foren_searcher.workspace import _bounded_model_value

    path = tmp_path / 'span.avi'
    make_video(path, frames=60)
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    source = ingest(store, path.read_bytes(), source='I0', filename='span.avi',
                    media_type='video/x-msvideo', modality='video')
    focus = ingest(store, image_bytes(color='blue'), source='I1', filename='focus.png',
                   media_type='image/png', modality='image')
    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(store=store, task=WorkspaceTask('W1-1', 'Inspect the video', (source.evidence_id,)),
            inspect_service=service)
        response = await harness.execute_inspect({'targets': [source.evidence_id],
            'request': {'operation': 'get_frames', 'params': {'count': 50}}})
        original = harness.memory.latest_observation
        observation = _bounded_model_value(original, 2000 if compressed else 100000)
        assert observation.get('truncated', False) is compressed
        view = {'task_id': 'W1-1', 'latest_observation': observation,
                'focus_evidence_ids': [focus.evidence_id] if keep_focus else []}
        generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
        generator.prepare_model_view(view)
        supplied = view['image_inputs']
        frames = [image for image in supplied if source.evidence_id in image['derived_from']]
        times = [json.loads(image['locator'])['time_s'] for image in frames]
        assert len(supplied) == 8
        assert (focus.evidence_id in {image['evidence_id'] for image in supplied}) is keep_focus
        assert min(times) == response.result.frames[0].time_s
        assert max(times) == response.result.frames[-1].time_s
        assert any(2.0 <= time <= 4.0 for time in times)
        ids = {image['evidence_id'] for image in supplied}
        assert not ids.intersection(view['image_inputs_omitted'])
        assert {frame.evidence_id for frame in response.result.frames} <= ids.union(view['image_inputs_omitted'])
        assert view['visual_observation_summary']['videos'][0]['observed_frame_count'] == len(frames)
        assert len(attached([generator._image_message(view)[0]])) == 8
        assert not store.trajectory_events(actor_role='HI', actor_id='W1-1', event_type='raw_model_output')
        all_ids = {frame.evidence_id for frame in response.result.frames}
        # 其他 HI 的附图不能代替本工作区的实际输入；失败的请求也不能推进覆盖。
        store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
                           payload={'image_evidence_ids': list(all_ids)})
        generator.prepare_model_view(view)
        assert {image['evidence_id'] for image in view['image_inputs']} == ids
        seen = set()
        for _ in range(9):
            batch = {image['evidence_id'] for image in view['image_inputs']}
            assert len(batch) == 8
            assert (focus.evidence_id in batch) is keep_focus
            if all_ids - seen:
                assert batch.intersection(all_ids - seen)
            seen.update(batch.intersection(all_ids))
            store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
                               payload={'image_evidence_ids': list(batch)})
            generator.prepare_model_view(view)
        assert seen == all_ids


@pytest.mark.asyncio
async def test_page_overflow_delivers_the_end_of_the_document(tmp_path):
    import fitz
    from foren_searcher.workspace import _bounded_model_value

    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    with fitz.open() as document:
        for number in range(1, 21):
            page = document.new_page(width=120, height=100)
            page.insert_text((10, 30), f'Page {number}')
        source = ingest(store, document.tobytes(), source='I0', filename='pages.pdf',
                        media_type='application/pdf', modality='text')
    async with create_inspect_service(store=store) as service:
        harness = WorkspaceHarness(store=store, task=WorkspaceTask('W1-1', 'Read the document', (source.evidence_id,)),
            inspect_service=service)
        response = await harness.execute_inspect({'targets': [source.evidence_id],
            'request': {'operation': 'get_pages', 'params': {'count': 20}}})
        view = {'task_id': 'W1-1', 'latest_observation': _bounded_model_value(harness.memory.latest_observation, 2000)}
        generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
        generator.prepare_model_view(view)
        pages = [json.loads(image['locator'])['page_number'] for image in view['image_inputs']]
        assert len(pages) == 8
        assert pages[0] == 1 and pages[-1] == 20
        assert any(8 <= number <= 12 for number in pages)
        assert all(image['derived_from'] == [source.evidence_id] for image in view['image_inputs'])
        assert len(attached([generator._image_message(view)[0]])) == 8
        assert len(response.result.pages) == 20
        seen = set()
        for _ in range(3):
            batch = {image['evidence_id'] for image in view['image_inputs']}
            seen.update(batch)
            store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
                               payload={'image_evidence_ids': list(batch)})
            generator.prepare_model_view(view)
        assert seen == {page.evidence_id for page in response.result.pages}


@pytest.mark.parametrize('with_focus', [False, True])
def test_alignment_pixels_keep_pairs_and_fill_capacity_across_candidates(tmp_path, with_focus):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    groups = [[ingest(store, image_bytes(color=color, red_box=False), source=f'I{group}-{index}',
        filename=f'{group}-{index}.png', media_type='image/png', modality='image')
        for index in range(8)] for group, color in enumerate(('red', 'blue'))]
    prior = [ingest(store, image_bytes(color='white'), source=f'P{i}', filename=f'prior{i}.png',
        media_type='image/png', modality='image') for i in range(4)] if with_focus else []
    matches = [{'anchors': [{'time_a_s': i, 'time_b_s': i + 3,
        'image_a_evidence_id': items[i].evidence_id, 'image_b_evidence_id': items[i + 1].evidence_id}
        for i in range(0, 8, 2)]} for items in groups]
    view = {'focus_evidence_ids': [e.evidence_id for e in prior], 'latest_observation': {
        'action': 'INSPECT', 'details': {'operation': 'align_media', 'result': {'matches': matches},
        'evidence': [e.card().to_dict() for group in groups for e in group]}}}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_model_view(view)
    pixels = attached([generator._image_message(view)[0]])
    supplied = [label['evidence_id'] for label, _ in pixels]
    assert len(supplied) == len(set(supplied)) == 8
    assert {e.evidence_id for e in groups[0][:2]+groups[0][-2:]} <= set(supplied)
    secondary = {e.evidence_id for e in groups[1][:2]+groups[1][-2:]}
    assert (secondary <= set(supplied)) is (not with_focus)
    assert {e.evidence_id for e in prior} <= set(supplied)
    assert len(view['image_inputs_omitted']) == (12 if with_focus else 8)
    assert len(view['alignment_image_pairs']) == (2 if with_focus else 4)
    assert sum(p.getpixel((10, 10))[0] > 240 and p.getpixel((10, 10))[2] < 10 for _, p in pixels) == 4

    view['focus_evidence_ids'] = [e.evidence_id for e in groups[1][:2]]
    generator.prepare_model_view(view)
    supplied = [label['evidence_id'] for label, _ in attached([generator._image_message(view)[0]])]
    assert len(supplied) == 8
    assert {e.evidence_id for e in groups[1][:2]} <= set(supplied)
    for pair in view['alignment_image_pairs']:
        assert supplied.index(pair['image_b_evidence_id']) == supplied.index(pair['image_a_evidence_id']) + 1



@pytest.mark.parametrize('size', [(320, 180), (768, 480), (2000, 1200)])
def test_model_transport_preserves_resized_fine_line_pixels(tmp_path, size):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    original = Image.new('RGB', size, 'white')
    drawing = ImageDraw.Draw(original)
    for y in range(5, size[1], 13):
        drawing.line((0, y, size[0] - 1, max(0, y - 4)), fill=(219, 17, 63), width=1)
        drawing.text((7, y), '0.35%', fill=(11, 31, 241))
    encoded = io.BytesIO()
    original.save(encoded, format='PNG')
    source = ingest(store, encoded.getvalue(), source='I0', filename='lines.png',
                    media_type='image/png', modality='image')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    view = {'materials': [source.card().to_dict()]}
    generator.prepare_model_view(view)
    message, ids = generator._image_message(view)
    assert ids == [source.evidence_id]
    label, supplied = attached([message])[0]
    scale = min(max(1, 1536 / max(size)), math.sqrt(1_500_000 / (size[0] * size[1])))
    expected = original.resize((int(size[0] * scale), int(size[1] * scale)), Image.Resampling.LANCZOS)
    assert supplied.format == 'PNG'
    assert supplied.size == expected.size
    assert supplied.tobytes() == expected.tobytes()
    assert (label['model_image_width'], label['model_image_height']) == supplied.size
    assert supplied.width * supplied.height <= 1_500_000


def test_model_image_has_exif_orientation_and_bounded_pixels(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    picture = Image.new('RGB', (400, 200), 'blue')
    exif = picture.getexif()
    exif[274] = 6
    data = io.BytesIO()
    picture.save(data, format='JPEG', exif=exif)
    evidence = ingest(store, data.getvalue(), source='I0', filename='rotated.jpg', media_type='image/jpeg', modality='image')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store,
        image_policy=replace(InspectPolicy(), maximum_model_image_pixels=5000))
    view = {'materials': [evidence.card().to_dict()]}
    generator.prepare_model_view(view)
    label, supplied = attached([generator._image_message(view)[0]])[0]
    assert supplied.width * supplied.height <= 5000 and supplied.height == 2 * supplied.width
    assert label['model_image_width'] == supplied.width and label['model_image_height'] == supplied.height
    assert (label['width'], label['height']) == (400, 200)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['api', 'invalid_output', 'success'])
async def test_observed_pixels_restore_only_own_successful_inputs(tmp_path, failure):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    task = WorkspaceTask('W1-1', 'Inspect the picture', (evidence.evidence_id,))
    def respond(messages):
        assert len(attached(messages)) == 1
        if failure == 'api':
            raise OpenRouterError('Unavailable')
        if failure == 'invalid_output':
            return completed(task.task_id, ['E999'])
        return completed(task.task_id, [evidence.evidence_id])
    generator = OpenRouterJsonGenerator(RecordingClient(respond), store=store)
    generator.generation_observer = lambda value: store.record_event(actor_role='HI',
        actor_id=value['workspace_id'], event_type='raw_model_output', payload=value)
    policy = TransformersHiPolicy(generator, policy_attempts=1)
    harness = WorkspaceHarness(task=task, store=store)
    if failure == 'success':
        await WorkspaceRunner(harness=harness, policy=policy, max_turns=1).run()
    else:
        with pytest.raises((OpenRouterError, WorkspaceTurnLimit)):
            await WorkspaceRunner(harness=harness, policy=policy, max_turns=1).run()
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
        payload={'image_evidence_ids': ['E999'], 'raw_output': 'private sibling finding'})
    restored = WorkspaceHarness(task=task, store=store)
    restored.restore_working_memory()
    expected = [] if failure == 'api' else [evidence.evidence_id]
    assert restored.memory.content_observed_evidence_ids == harness.memory.content_observed_evidence_ids == expected
    assert 'private sibling finding' not in json.dumps(restored.model_view())


@pytest.mark.asyncio
async def test_parallel_hi_calls_share_model_without_sharing_task_or_history(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='shared.png', media_type='image/png', modality='image')
    def respond(messages):
        text = json.dumps(messages)
        assert ('private-alpha' in text) != ('private-beta' in text)
        assert 'global-only-secret' not in text
        assert len(attached(messages)) == 1
        return completed('W1-1' if 'private-alpha' in text else 'W1-2', [evidence.evidence_id])
    client = RecordingClient(respond)
    bundle = OpenRouterPolicyBundle(client, store=store)
    harnesses = [WorkspaceHarness(task=WorkspaceTask(f'W1-{i}', task, (evidence.evidence_id,)), store=store)
        for i, task in enumerate(['private-alpha', 'private-beta'], 1)]
    reports = await asyncio.gather(*(WorkspaceRunner(harness=h, policy=bundle.hi_policy).run() for h in harnesses))
    assert len(reports) == 2 and len(client.calls) == 2 and bundle.gia_model is bundle.hi_model
    for harness in harnesses:
        sibling = harnesses[1] if harness is harnesses[0] else harnesses[0]
        with pytest.raises(PermissionError):
            harness.record_access.read(sibling.memory.task_id)
        assert sibling.memory.task not in json.dumps(harness.model_view())


@pytest.mark.asyncio
async def test_unified_inspect_advertises_only_enabled_tools_and_rejects_visual_delegation(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    service = create_inspect_service(store=store, enabled_operations=OPERATIONS)
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Inspect', ()), store=store, inspect_service=service)
    try:
        actions = json.dumps(harness.eligible_actions())
        assert 'visual_qa' not in actions and 'audio_qa' not in actions and 'DETECT' not in actions
        assert 'ocr' in actions and 'asr' in actions
        result = await service.execute({'targets': ['E1'], 'request': {
            'operation': 'visual_qa', 'params': {'question': 'Describe'}}}, workspace_id='W1-1')
        assert result.status == 'failed' and not result.evidence
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_multimodal_metadata_stays_within_existing_view_budget(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Describe', (evidence.evidence_id,)), store=store)
    original = harness.model_view(runtime_constraints={'turn_index': 1, 'max_turns': 1, 'remaining_turns': 0})
    harness.memory_policy = replace(MemoryPolicy(), max_model_view_chars=len(json.dumps(original, ensure_ascii=False)) + 1)
    client = RecordingClient(lambda _: {})
    with pytest.raises(ValueError, match='max_model_view_chars'):
        await WorkspaceRunner(harness=harness, policy=TransformersHiPolicy(OpenRouterJsonGenerator(client, store=store)), max_turns=1).run()
    assert not client.calls


@pytest.mark.asyncio
async def test_image_metadata_trims_recoverable_history_before_model_call(tmp_path):
    from foren_searcher.workspace import WorkspaceHistoryEntry

    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    evidence = ingest(store, image_bytes(), source='I0', filename='input.png', media_type='image/png', modality='image')
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Describe the image', (evidence.evidence_id,)), store=store)
    old = WorkspaceHistoryEntry(action='QUERY_MEMORY', request={'query': 'previous records'},
        response={}, observation={'action': 'QUERY_MEMORY', 'details': {'results': [], 'notice': 'old observation ' * 250}},
        focus_evidence_ids=())
    harness.memory.history.append(old)
    constraints = {'turn_index': 1, 'max_turns': 1, 'remaining_turns': 0, 'max_model_view_chars': 10000}
    original = harness.model_view(runtime_constraints=constraints)
    limit = len(json.dumps(original, ensure_ascii=False)) + 100
    harness.memory_policy = replace(MemoryPolicy(), max_model_view_chars=limit)

    def respond(messages):
        payload = json.loads(messages[1]['content'])
        view = payload['model_view']
        assert view['context_truncated']
        assert view['task'] == 'Describe the image'
        assert attached(messages)[0][0]['evidence_id'] == evidence.evidence_id
        assert len(json.dumps(view, ensure_ascii=False)) <= limit
        return completed('W1-1', [evidence.evidence_id])

    client = RecordingClient(respond)
    policy = TransformersHiPolicy(OpenRouterJsonGenerator(client, store=store))
    report = await WorkspaceRunner(harness=harness, policy=policy, max_turns=1).run()
    assert report.status == 'COMPLETED'
    assert len(client.calls) == 1
    assert old in harness.memory.history


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['image', 'pdf', 'pdf_page'])
async def test_reading_media_metadata_does_not_make_unseen_content_citable(tmp_path, kind):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    if kind == 'image':
        evidence = ingest(store, image_bytes(), source='I0', filename='image.png',
                          media_type='image/png', modality='image')
    else:
        evidence = ingest(store, pdf_bytes(), source='I0', filename='document.pdf',
                          media_type='application/pdf', modality='text')
        if kind == 'pdf_page':
            service = create_inspect_service(store=store, enabled_operations=OPERATIONS)
            try:
                pages = await service.execute({'targets': [evidence.evidence_id], 'request': {
                    'operation': 'get_pages', 'params': {'pages': [2]}}}, workspace_id='W0')
                assert pages.status == 'ready'
                evidence = store.evidence(pages.result.pages[0].evidence_id)
            finally:
                await service.aclose()
    task = WorkspaceTask('W1-1', 'Describe the supplied content.', (evidence.evidence_id,))
    calls = 0
    def respond(messages):
        nonlocal calls
        calls += 1
        assert not attached(messages)
        if calls == 1:
            return {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id},
                    'focus_evidence_ids': [evidence.evidence_id]}
        return completed(task.task_id, [evidence.evidence_id])
    client = RecordingClient(respond)
    generator = OpenRouterJsonGenerator(client, store=store,
        image_policy=replace(InspectPolicy(), maximum_image_pixels=1))
    harness = WorkspaceHarness(task=task, store=store)
    with pytest.raises(WorkspaceTurnLimit):
        await WorkspaceRunner(harness=harness,
            policy=TransformersHiPolicy(generator, policy_attempts=1), max_turns=2).run()
    assert evidence.evidence_id not in harness.memory.content_observed_evidence_ids
    restored = WorkspaceHarness(task=task, store=store)
    restored.restore_working_memory()
    assert evidence.evidence_id not in restored.memory.content_observed_evidence_ids
    assert not store.workspace_report_records()
