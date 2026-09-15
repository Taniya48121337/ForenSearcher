import asyncio
import threading
from types import SimpleNamespace

import pytest

from foren_searcher.local_models import LocalModelBusyError, LocalModelConfig, TransformersJsonGenerator, _GenerationStopped
from foren_searcher.policies import PolicyReviewDeferred


def review_payload():
    return {'global_view': {'runtime_constraints': {'round_in_progress': True}}, 'available_actions': []}


def response(value):
    return SimpleNamespace(value=value, input_tokens=2, output_tokens=1)


def test_stopped_generation_does_not_publish_output_or_observation_receipts():
    generator = TransformersJsonGenerator(LocalModelConfig('unused'))
    generator._processor = object()
    generator._image_message = lambda view: ({'role': 'user', 'content': []}, ['E1'])
    observed = []
    generator.generation_observer = observed.append
    stop = threading.Event()
    def generate_once(*args):
        stop.set()
        return '{"decision":"FINISH","answer":"Stale"}', 2, 1, None
    generator._generate_once = generate_once
    payload = {'global_view': {}}
    with pytest.raises(_GenerationStopped):
        generator._generate_json('test', payload, stop)
    assert observed == []
    assert 'model_observed_image_ids' not in payload['global_view']


@pytest.mark.asyncio
@pytest.mark.parametrize('external_cancel', [False, True])
async def test_foreground_waits_for_preempted_review_worker_to_stop(external_cancel):
    generator = TransformersJsonGenerator(LocalModelConfig('unused'))
    started, stopping, release = threading.Event(), threading.Event(), threading.Event()
    foreground_started = threading.Event()
    view = review_payload()

    def generate(prompt, payload, stop_event):
        if prompt == 'review':
            started.set()
            assert stop_event.wait(5)
            stopping.set()
            assert release.wait(5)
            payload['global_view']['model_observed_image_ids'] = ['E9']
            return response({'decision': 'FINISH', 'answer': 'Stale answer (E9).'})
        foreground_started.set()
        return response({'done': prompt})

    generator._generate_json = generate
    review = asyncio.create_task(generator.generate_json(system_prompt='review', payload=view))
    foreground = []
    try:
        assert await asyncio.to_thread(started.wait, 5)
        foreground = [asyncio.create_task(generator.generate_json(system_prompt=f'HI {i}',
            payload={'workspace_id': f'W1-{i}', 'model_view': {}})) for i in (1, 2)]
        assert await asyncio.to_thread(stopping.wait, 5)
        if external_cancel:
            review.cancel()
            await asyncio.sleep(0)
            review.cancel()
            await asyncio.sleep(0)
        assert not foreground_started.is_set()
        assert generator._inference_gate.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError if external_cancel else PolicyReviewDeferred):
            await asyncio.wait_for(review, 5)
        assert await asyncio.wait_for(asyncio.gather(*foreground), 5) == [{'done': 'HI 1'}, {'done': 'HI 2'}]
        assert 'content_observed_evidence_ids' not in view['global_view']
        assert 'model_observed_image_ids' not in view['global_view']
        metrics = generator.metrics()
        assert (metrics.accepted_requests, metrics.completed_requests, metrics.cancelled_requests) == (3, 2, 1)
        assert metrics.failed_requests == metrics.pending_requests == 0
    finally:
        release.set()
        for task in [review, *foreground]:
            task.cancel()
        await asyncio.gather(review, *foreground, return_exceptions=True)


@pytest.mark.asyncio
async def test_foreground_preempts_a_queued_review_without_model_work():
    generator = TransformersJsonGenerator(LocalModelConfig('unused'))
    calls = []
    def generate(prompt, payload, stop):
        calls.append(prompt)
        return response({'done': prompt})
    generator._generate_json = generate
    await generator._inference_gate.acquire()
    review = asyncio.create_task(generator.generate_json(system_prompt='review', payload=review_payload()))
    await asyncio.sleep(0)
    foreground = asyncio.create_task(generator.generate_json(system_prompt='HI', payload={'workspace_id': 'W1-1'}))
    try:
        with pytest.raises(PolicyReviewDeferred):
            await asyncio.wait_for(review, 5)
        assert calls == []
        generator._inference_gate.release()
        assert await asyncio.wait_for(foreground, 5) == {'done': 'HI'}
        assert calls == ['HI']
        assert generator.metrics().pending_requests == 0
    finally:
        if generator._inference_gate.locked() and not calls:
            generator._inference_gate.release()
        review.cancel()
        foreground.cancel()
        await asyncio.gather(review, foreground, return_exceptions=True)


@pytest.mark.asyncio
async def test_ordinary_gia_generation_is_not_preempted():
    generator = TransformersJsonGenerator(LocalModelConfig('unused'))
    started, release, interrupted = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def generate(prompt, payload, stop):
        calls.append(prompt)
        if prompt == 'GIA':
            started.set()
            assert release.wait(5)
            if stop.is_set():
                interrupted.set()
        return response({'done': prompt})
    generator._generate_json = generate
    gia = asyncio.create_task(generator.generate_json(system_prompt='GIA', payload={'global_view': {}}))
    hi = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        hi = asyncio.create_task(generator.generate_json(system_prompt='HI', payload={'workspace_id': 'W1-1'}))
        await asyncio.sleep(0)
        assert calls == ['GIA']
        release.set()
        assert await asyncio.wait_for(asyncio.gather(gia, hi), 5) == [{'done': 'GIA'}, {'done': 'HI'}]
        assert not interrupted.is_set()
    finally:
        release.set()
        gia.cancel()
        if hi is not None:
            hi.cancel()
        await asyncio.gather(gia, *([hi] if hi is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_rejected_request_does_not_cancel_an_uncontended_review():
    generator = TransformersJsonGenerator(LocalModelConfig('unused', max_pending_requests=1))
    started, release = threading.Event(), threading.Event()
    def generate(prompt, payload, stop):
        started.set()
        assert release.wait(5)
        assert not stop.is_set()
        return response({'decision': 'FINISH', 'answer': 'Observed answer (E1).'})
    generator._generate_json = generate
    review = asyncio.create_task(generator.generate_json(system_prompt='review', payload=review_payload()))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(LocalModelBusyError):
            await generator.generate_json(system_prompt='HI', payload={'workspace_id': 'W1-1'})
        release.set()
        assert (await asyncio.wait_for(review, 5))['decision'] == 'FINISH'
        assert generator.metrics().cancelled_requests == 0
        assert generator.metrics().busy_rejections == 1
    finally:
        release.set()
        review.cancel()
        await asyncio.gather(review, return_exceptions=True)


@pytest.mark.asyncio
async def test_deferred_review_preserves_reports_and_final_synthesis():
    from foren_searcher.orchestration import InvestigationRuntime
    from foren_searcher.store import GlobalStore
    from test_framework_runtime import add_text_evidence
    from test_round_review import report

    store = GlobalStore()
    sources = [add_text_evidence(store, f'Recorded value is {n}.') for n in (12, 13)]
    class Hi:
        async def decide(self, *, workspace_id, **kwargs):
            index = 0 if workspace_id == 'W1-1' else 1
            if index:
                while not store.trajectory_events(event_type='round_review_deferred'):
                    await asyncio.sleep(.001)
            return report(workspace_id, sources[index].evidence_id, 12 + index)
    class Gia:
        initial = True
        async def decide(self, *, model_view, **kwargs):
            if self.initial:
                self.initial = False
                return {'decision': 'INVESTIGATE', 'tasks': [
                    {'task': 'Read the value', 'materials': [s.evidence_id]} for s in sources]}
            if model_view['runtime_constraints'].get('round_in_progress'):
                raise PolicyReviewDeferred('Shared model needed by HI')
            assert all(r['run_status'] == 'reported' for r in model_view['latest_reports'])
            return {'decision': 'FINISH', 'answer': f'Values are 12 ({sources[0].evidence_id}) and 13 ({sources[1].evidence_id}).'}
    async with InvestigationRuntime(store=store, gia_policy=Gia(), hi_policy=Hi(),
            max_rounds=1, report_review_delay_s=0, workspace_timeout_s=5) as runtime:
        result = await asyncio.wait_for(runtime.run(task='Compare the values',
            original_material_ids=tuple(s.evidence_id for s in sources)), 5)
    assert '12' in result.decision.answer and '13' in result.decision.answer
    assert len(store.trajectory_events(event_type='round_review_deferred')) == 1
    assert not store.trajectory_events(event_type='round_review_failed')
    assert not store.trajectory_events(event_type='runtime_error')
    assert all(t['run_status'] == 'reported' for t in store.workspace_task_records())
