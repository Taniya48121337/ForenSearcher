import json

import pytest

from foren_searcher.contracts import FindingDraft, WorkspaceReportDraft, WorkspaceTask
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import TransformersGiaPolicy
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy, bound_global_view, workspace_visual_observations
from foren_searcher.openrouter import OpenRouterJsonGenerator
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from inspect_helpers import image_bytes, ingest, make_video, pdf_bytes
from test_hi_multimodal import RecordingClient, attached


@pytest.mark.asyncio
async def test_gia_receives_cited_pixels_across_findings_without_hi_private_history(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color=color, red_box=False), source=f'I{i}',
               filename=f'{i}.png', media_type='image/png', modality='image')
               for i, color in enumerate(['red'] * 10 + ['blue', 'green'])]
    # 兄弟工作区私下看过的 sources[11] 不进任务材料：原件可见与私图不外泄是两条独立规则，
    # 同一个编号兼任两种角色会让断言分不清是哪一条在起作用。
    task = WorkspaceTask('W1-1', 'Compare supplied colors.', tuple(s.evidence_id for s in sources[:11]))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    WorkspaceHarness(store=store, task=task, round_index=1).finish(WorkspaceReportDraft(
        task_id=task.task_id, status='COMPLETED', findings=(
            FindingDraft('The first group is red.', tuple(s.evidence_id for s in sources[:10]), ()),
            FindingDraft('The last scene is orange.', (sources[10].evidence_id,), ()),
        ), selected_evidence_ids=tuple(s.evidence_id for s in sources[:11])))
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='raw_model_output',
        payload={'image_evidence_ids': [sources[10].evidence_id], 'reasoning': 'PRIVATE_HI_REASONING'})
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
        payload={'image_evidence_ids': [sources[11].evidence_id], 'reasoning': 'PRIVATE_SIBLING_REASONING'})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    def respond(messages):
        assert 'PRIVATE_HI_REASONING' not in json.dumps(messages)
        assert 'PRIVATE_SIBLING_REASONING' not in json.dumps(messages)
        pixels = attached(messages)
        ids = [label['evidence_id'] for label, _ in pixels]
        assert len(ids) == len(set(ids)) == 8
        assert sources[10].evidence_id in ids
        assert sources[11].evidence_id not in ids
        blue = next(picture for label, picture in pixels if label['evidence_id'] == sources[10].evidence_id)
        assert blue.getpixel((0, 0)) == (0, 0, 255)
        return {'decision': 'FINISH', 'answer': f'The last image is blue ({sources[10].evidence_id}).'}
    client = RecordingClient(respond)
    generator = OpenRouterJsonGenerator(client, store=store)
    observations = []
    generator.generation_observer = observations.append
    result = await TransformersGiaPolicy(generator).decide(model_view=view, history=(), actions=())
    assert result['decision'] == 'FINISH'
    assert len(client.calls) == 1 and client.calls[0]['role'] == 'GIA'
    assert len(observations[0]['image_evidence_ids']) == 8
    assert sources[10].evidence_id in view['model_observed_image_ids']
    assert len(view['image_inputs_omitted']) == 3


def test_gia_read_record_prioritizes_requested_image_and_respects_context_budget(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    source = ingest(store, image_bytes(), source='I0', filename='picture.png', media_type='image/png', modality='image')
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    view = {'task': 'Check the image.', 'latest_reports': [], 'current_materials': []}
    generator.prepare_global_model_view(view)
    assert view['image_inputs'] == []
    generator.prepare_global_model_view(view, history=({'observation': {'record': source.to_dict()}},))
    assert [item['evidence_id'] for item in view['image_inputs']] == [source.evidence_id]
    size = len(json.dumps(view, ensure_ascii=False))
    limited = bound_global_view(view, size - 1)
    assert limited['image_inputs'] == []
    assert limited['image_inputs_truncated']
    assert generator._image_message(limited) == (None, [])


@pytest.mark.parametrize('status', ['timed_out', 'runtime_error', 'cancelled'])
def test_gia_receives_committed_pixels_from_unfinished_workspace_support(tmp_path, status):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    source = ingest(store, image_bytes(color='red', red_box=False), source='I0',
        filename='frame.png', media_type='image/png', modality='image')
    task = WorkspaceTask('W1-1', 'Check the color.', (source.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='model_output',
        payload={'turn_index': 1, 'output': {'action': 'INSPECT', 'arguments': {
            'targets': [source.evidence_id], 'request': {'operation': 'crop', 'params': {}}}}})
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='observation',
        payload={'turn_index': 1, 'observation': {'action': 'INSPECT', 'details': {
            'status': 'ready', 'operation': 'crop', 'result_evidence_id': source.evidence_id,
            'evidence': [source.card().to_dict()],
            'result': {'evidence_id': source.evidence_id}}}})
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='raw_model_output',
        payload={'image_evidence_ids': [source.evidence_id], 'reasoning': 'PRIVATE_REASONING'})
    store.set_workspace_run_status(task.task_id, status, error={'type': status, 'message': 'Stopped'})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    assert 'report' not in view['latest_reports'][0]
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view)
    pixels = attached([generator._image_message(view)[0]])
    assert [label['evidence_id'] for label, _ in pixels] == [source.evidence_id]
    assert pixels[0][1].getpixel((0, 0)) == (255, 0, 0)
    assert 'PRIVATE_REASONING' not in json.dumps(view)


@pytest.mark.asyncio
async def test_gia_can_view_real_tool_frames_when_hi_stops_before_receiving_them(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    path = tmp_path / 'clip.avi'
    make_video(path, frames=4, fps=2.)
    source = ingest(store, path.read_bytes(), source='I0', filename=path.name,
                    media_type='video/avi', modality='video')
    task = WorkspaceTask('W1-1', 'Inspect the video', (source.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    async with create_inspect_service(store=store) as service:
        result = await service.execute({'targets': [source.evidence_id], 'request': {
            'operation': 'get_frames', 'params': {'times_s': [0., .5]}}}, workspace_id=task.task_id)
    assert result.status == 'ready' and result.result_evidence_id is None
    frame_ids = [frame.evidence_id for frame in result.result.frames]
    store.record_event(actor_role='HI', actor_id=task.task_id, event_type='observation',
        payload={'turn_index': 1, 'observation': {'action': 'INSPECT', 'details': result.to_dict()}})
    store.set_workspace_run_status(task.task_id, 'timed_out', error={'type': 'timeout', 'message': 'Stopped'})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task.task,
        original_material_ids=task.materials, latest_round_index=1).to_dict()
    context = view['latest_reports'][0]
    assert set(frame_ids) <= set(context['action_observations'][0]['evidence_ids'])
    assert 'image_inputs' not in context
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view)
    assert 'visual_observation_summary' not in view
    pixels = attached([generator._image_message(view)[0]])
    assert [label['evidence_id'] for label, _ in pixels] == frame_ids
    assert all(picture.width > 0 for _, picture in pixels)


def test_media_profile_remains_discoverable_after_failed_workspace_and_history_truncation(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    source = ingest(store, image_bytes(), source='I0', filename='image.png',
                    media_type='image/png', modality='image')
    policy = MemoryPolicy(max_history_entries=1)
    builder = GlobalViewBuilder(store, policy)
    original = builder.build(task='Inspect the media', original_material_ids=(source.evidence_id,),
                             latest_round_index=None).original_materials
    task = WorkspaceTask('W1-1', 'Inspect the media', (source.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    profile = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('text',),
        label='媒体属性：image，800×450', content='{"width":800,"height":450,"media_kind":"image"}',
        locator='{}', acquired_by='INSPECT/media_profile', discovered_by=task.task_id, production_method='media_probe')
    store.set_workspace_run_status(task.task_id, 'running')
    store.set_workspace_run_status(task.task_id, 'timed_out', error={'type': 'timeout', 'message': 'timeout'})
    store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=('text',),
        label='Later result', content='Later public result', locator='{}', acquired_by='INSPECT/ocr',
        discovered_by=task.task_id, production_method='structured_analysis')
    view = builder.build(task='Inspect the media', original_material_ids=(source.evidence_id,),
                         latest_round_index=2)
    assert view.original_materials == original
    cards = [card for card in view.current_materials if card.get('evidence_id') == profile.evidence_id]
    assert len(cards) == 1
    assert cards[0]['acquired_by'] == 'INSPECT/media_profile'
    assert 'content' not in cards[0]
    assert view.context_truncated


@pytest.mark.parametrize('previous_round', [False, True])
def test_gia_image_budget_represents_independent_reports(tmp_path, previous_round):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color=color, red_box=False), source=f'I{i}',
                     filename=f'{i}.png', media_type='image/png', modality='image')
               for i, color in enumerate(['red'] * 9 + ['blue'])]
    reports = [{'report': {'findings': [{'evidence_ids': [image.evidence_id]} for image in group]}}
               for group in [sources[:9], sources[9:]]]
    if previous_round:
        reports.pop()
        store.record_event(actor_role='GIA', actor_id='GIA', event_type='raw_model_output',
                           payload={'image_evidence_ids': [sources[-1].evidence_id], 'reasoning': 'PRIVATE_GIA_REASONING'})
    view = {'latest_reports': reports}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view)
    pixels = attached([generator._image_message(view)[0]])
    ids = [label['evidence_id'] for label, _ in pixels]
    assert len(ids) == 8
    assert ids[:2] == [sources[0].evidence_id, sources[-1].evidence_id]
    assert pixels[1][1].getpixel((0, 0)) == (0, 0, 255)
    assert 'PRIVATE_GIA_REASONING' not in json.dumps(view)


@pytest.mark.parametrize('status', ['timed_out', 'runtime_error', 'cancelled', 'INCOMPLETE', 'COMPLETED'])
def test_gia_overflow_prioritizes_evidence_needing_review(tmp_path, status):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color=color, red_box=False), source=f'I{i}',
        filename=f'{i}.png', media_type='image/png', modality='image')
        for i, color in enumerate(['green'] + ['red'] * 7 + ['blue'])]
    completed = {'run_status': 'reported', 'report': {'status': 'COMPLETED',
        'findings': [{'statement': 'The page is green.', 'evidence_ids': [sources[0].evidence_id]}]}}
    unfinished = {'run_status': 'reported' if status in {'INCOMPLETE', 'COMPLETED'} else status,
        'supporting_records': [s.card().to_dict() for s in sources[1:]]}
    if status == 'INCOMPLETE':
        unfinished['report'] = {'status': 'INCOMPLETE', 'findings': [], 'selected_evidence_ids': []}
    elif status == 'COMPLETED':
        unfinished['report'] = {'status': 'COMPLETED', 'findings': [{'statement': 'The last state needs confirmation.',
            'evidence_ids': [s.evidence_id for s in sources[1:]], 'limitations': ['State transition remains uncertain.']}]}
    view = {'latest_reports': [completed, unfinished]}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view)
    pixels = attached([generator._image_message(view)[0]])
    assert [label['evidence_id'] for label, _ in pixels] == [s.evidence_id for s in sources[1:]]
    assert pixels[-1][1].getpixel((0, 0)) == (0, 0, 255)
    assert view['latest_reports'][0] == completed
    assert view['image_inputs_omitted'] == [sources[0].evidence_id]


def test_gia_explicit_read_precedes_unfinished_groups_and_groups_share_capacity(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color='blue', red_box=False), source=f'I{i}',
        filename=f'{i}.png', media_type='image/png', modality='image') for i in range(17)]
    view = {'latest_reports': [{'run_status': 'timed_out',
        'supporting_records': [s.card().to_dict() for s in group]}
        for group in [sources[1:9], sources[9:]]]}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view, history=({'observation': {'record': sources[0].to_dict()}},))
    ids = [item['evidence_id'] for item in view['image_inputs']]
    expected = [sources[i].evidence_id for i in [0, 1, 9, 2, 10, 3, 11, 4]]
    assert ids == expected
    assert len(view['image_inputs_omitted']) == 9


@pytest.mark.parametrize('status', ['timed_out', 'COMPLETED'])
def test_gia_keeps_completed_pixels_when_all_reports_fit(tmp_path, status):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(), source=f'I{i}', filename=f'{i}.png',
        media_type='image/png', modality='image') for i in range(3)]
    second = {'run_status': 'timed_out', 'supporting_records': [s.card().to_dict() for s in sources[1:]]}
    if status == 'COMPLETED':
        second = {'run_status': 'reported', 'report': {'status': 'COMPLETED', 'findings': [
            {'evidence_ids': [s.evidence_id for s in sources[1:]], 'limitations': ['The comparison remains uncertain.']}]}}
    view = {'latest_reports': [{'run_status': 'reported', 'report': {'status': 'COMPLETED',
        'findings': [{'evidence_ids': [sources[0].evidence_id]}]}},
        second]}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view)
    assert [item['evidence_id'] for item in view['image_inputs']] == [s.evidence_id for s in sources]
    assert view['image_inputs_omitted'] == []


def test_gia_explicit_reads_and_unfinished_work_precede_limited_reports(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(), source=f'I{i}', filename=f'{i}.png',
        media_type='image/png', modality='image') for i in range(18)]
    limited = [{'run_status': 'reported', 'report': {'status': 'COMPLETED', 'findings': [
        {'evidence_ids': [source.evidence_id for source in group], 'limitations': ['Boundary remains uncertain.']}]}}
        for group in (sources[1:9], sources[10:])]
    view = {'latest_reports': [limited[0], {'run_status': 'timed_out',
        'supporting_records': [sources[9].card().to_dict()]}, limited[1]]}
    generator = OpenRouterJsonGenerator(RecordingClient(lambda _: {}), store=store)
    generator.prepare_global_model_view(view, history=({'observation': {'record': sources[0].to_dict()}},))
    assert [image['evidence_id'] for image in view['image_inputs']] == [sources[i].evidence_id for i in [0, 9, 1, 10, 2, 11, 3, 12]]
    assert len(view['image_inputs_omitted']) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['timed_out', 'reported'])
async def test_gia_video_handoff_preserves_frames_needed_for_review_beside_completed_pdf(tmp_path, status):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    pdf = ingest(store, pdf_bytes(), source='I0', filename='document.pdf', media_type='application/pdf', modality='text')
    path = tmp_path / 'video.avi'
    make_video(path, frames=8, fps=2.)
    video = ingest(store, path.read_bytes(), source='I1', filename=path.name, media_type='video/avi', modality='video')
    tasks = (WorkspaceTask('W1-1', 'Read the PDF title.', (pdf.evidence_id,)),
             WorkspaceTask('W1-2', 'Inspect the complete video.', (video.evidence_id,)))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, tasks)
    async with create_inspect_service(store=store) as service:
        page_result = await service.execute({'targets': [pdf.evidence_id], 'request': {
            'operation': 'get_pages', 'params': {'pages': [1]}}}, workspace_id=tasks[0].task_id)
        frame_result = await service.execute({'targets': [video.evidence_id], 'request': {
            'operation': 'get_frames', 'params': {'times_s': [i / 2 for i in range(8)]}}}, workspace_id=tasks[1].task_id)
    assert page_result.status == frame_result.status == 'ready'
    page = page_result.result.pages[0].evidence_id
    frames = [frame.evidence_id for frame in frame_result.result.frames]
    for task, result in zip(tasks, [page_result, frame_result]):
        store.record_event(actor_role='HI', actor_id=task.task_id, event_type='observation',
            payload={'turn_index': 1, 'observation': {'action': 'INSPECT', 'details': result.to_dict()}})
    WorkspaceHarness(store=store, task=tasks[0], round_index=1).finish(WorkspaceReportDraft(
        task_id=tasks[0].task_id, status='COMPLETED', findings=(
            FindingDraft('The title is ForenSearcher page one.', (page,), ()),), selected_evidence_ids=(page,)))
    if status == 'timed_out':
        store.set_workspace_run_status(tasks[1].task_id, 'timed_out', error={'type': 'timeout', 'message': 'Time limit'})
    else:
        WorkspaceHarness(store=store, task=tasks[1], round_index=1).finish(WorkspaceReportDraft(
            task_id=tasks[1].task_id, status='COMPLETED', findings=(FindingDraft(
                'The video lasts four seconds.', tuple(frames), ('Frame states require review.',)),),
            selected_evidence_ids=tuple(frames)))
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task='Read the title and inspect the complete video.',
        original_material_ids=(pdf.evidence_id, video.evidence_id), latest_round_index=1).to_dict()
    def respond(messages):
        pixels = attached(messages)
        # 原件派生的图片优先于报告引用：PDF 页面进入视野，预算把最后一帧挤出。
        assert [label['evidence_id'] for label, _ in pixels] == [page] + frames[:7]
        assert pixels[-1][0]['locator']['time_s'] == 3.
        assert pixels[-1][0]['frame_presentation_interval_s'] == {
            'start': 3., 'end': 3.5, 'start_inclusive': True, 'end_inclusive': False}
        assert 'ForenSearcher page one' in json.dumps(messages)
        return {'decision': 'FINISH', 'answer': f'Title: ForenSearcher page one ({page}). Last observed frame: 3.0s ({frames[6]}).'}
    generator = OpenRouterJsonGenerator(RecordingClient(respond), store=store)
    result = await TransformersGiaPolicy(generator).decide(model_view=view, history=(), actions=())
    assert result['decision'] == 'FINISH'
    assert view['model_observed_image_ids'] == [page] + frames[:7]
    assert view['image_inputs_omitted'] == [frames[-1]]


def test_observation_count_deduplicates_timestamps_and_excludes_crops_and_siblings(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    path = tmp_path / 'clip.avi'
    make_video(path)
    video = ingest(store, path.read_bytes(), source='I0', filename=path.name, media_type='video/avi', modality='video')
    frames = []
    for index, time_s in enumerate([.1, .2, .2, .3]):
        frames.append(store.commit_derived_evidence(derived_from=(video.evidence_id,), modalities=('image',),
            label=f'Frame {index}', content='frame', locator=json.dumps({'time_s': time_s}),
            acquired_by='INSPECT/get_frames', discovered_by='W1-1', production_method='derived_media'))
    crop = store.commit_derived_evidence(derived_from=(frames[0].evidence_id,), modalities=('image',),
        label='Crop', content='crop', locator=json.dumps({'time_s': .1, 'box': [0, 0, .5, .5]}),
        acquired_by='INSPECT/crop', discovered_by='W1-1', production_method='derived_media')
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
        payload={'image_evidence_ids': [f.evidence_id for f in frames[:3]] + [crop.evidence_id]})
    store.record_event(actor_role='HI', actor_id='W1-2', event_type='raw_model_output',
        payload={'image_evidence_ids': [frames[3].evidence_id]})
    _, summary = workspace_visual_observations(store, 'W1-1')
    assert summary['image_evidence_count'] == 4
    assert summary['videos'] == [{'evidence_id': video.evidence_id, 'observed_frame_count': 2,
        'coverage_span_s': [0.0, .2], 'observed_ranges_s': [], 'unconfirmed_ranges_s': [[0.0, .2]],
        'unknown_duration_frame_times_s': [.1, .2]}]
    _, current = workspace_visual_observations(store, 'W1-1', [frames[3].evidence_id])
    assert current['image_evidence_count'] == 5
    assert current['videos'][0]['observed_frame_count'] == 3


@pytest.mark.parametrize('duration', [None, 0, -1, True, float('nan'), float('inf')])
def test_unknown_frame_duration_does_not_fill_unobserved_interval(tmp_path, duration):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    path = tmp_path / 'clip.avi'
    make_video(path)
    video = ingest(store, path.read_bytes(), source='I0', filename=path.name, media_type='video/avi', modality='video')
    frames = [store.commit_derived_evidence(derived_from=(video.evidence_id,), modalities=('image',),
        label='Frame', content='frame', locator=json.dumps({'time_s': time_s, 'frame_duration_s': length}),
        acquired_by='INSPECT/get_frames', discovered_by='W1-1', production_method='derived_media')
        for time_s, length in [(1, .1), (.3, duration), (1, .2), (2, duration)]]
    _, summary = workspace_visual_observations(store, 'W1-1', [f.evidence_id for f in frames])
    assert summary['videos'][0] == {'evidence_id': video.evidence_id, 'observed_frame_count': 3,
        'coverage_span_s': [0, 2], 'observed_ranges_s': [[1, 1.1]],
        'unconfirmed_ranges_s': [[0, 1], [1.1, 2]], 'unknown_duration_frame_times_s': [.3, 2]}


@pytest.mark.asyncio
async def test_gia_sees_original_materials_before_any_workspace_reports(tmp_path):
    """第 1 轮既没有报告也没有读取历史，原始材料本身必须带像素送到 GIA。"""

    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    picture = ingest(store, image_bytes(color='blue', red_box=False), source='I0',
                     filename='scene.png', media_type='image/png', modality='image')
    document = ingest(store, pdf_bytes(), source='I1', filename='document.pdf',
                      media_type='application/pdf', modality='text')
    view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task='Judge whether the picture supports the claim.',
        original_material_ids=(picture.evidence_id, document.evidence_id),
        latest_round_index=None).to_dict()
    assert view['latest_reports'] == [] and view['history_index'] == []
    def respond(messages):
        pixels = attached(messages)
        # 只有图片模态的原件带像素；PDF 在第 1 轮还没有渲染页面，仍然只是文字卡片。
        assert [label['evidence_id'] for label, _ in pixels] == [picture.evidence_id]
        assert pixels[0][1].getpixel((0, 0)) == (0, 0, 255)
        return {'decision': 'INVESTIGATE', 'tasks': [
            {'task': 'Identify where the blue scene was first published.', 'materials': [picture.evidence_id]}]}
    generator = OpenRouterJsonGenerator(RecordingClient(respond), store=store)
    result = await TransformersGiaPolicy(generator).decide(model_view=view, history=(), actions=())
    assert result['decision'] == 'INVESTIGATE'
    assert view['model_observed_image_ids'] == [picture.evidence_id]
    assert view['image_inputs_omitted'] == []
