import asyncio
import base64
import copy
from contextvars import ContextVar
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import foren_searcher.local_models as local
from foren_searcher.policies import OutputRepairLimit, PolicyReviewDeferred, output_repair_scope

torch = pytest.importorskip('torch')


def request(actor):
    return {'workspace_id': actor, 'model_view': {'task_id': actor,
            'runtime_constraints': {'remaining_seconds': 100.}}}


def raw(actor):
    return json.dumps({'actor': actor}), 10, 3, 'reasoning'


@pytest.mark.asyncio
async def test_batch_coalesces_distinct_hi_requests_and_preserves_metrics():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    batches = []
    def generate(messages, jobs, schemas):
        batches.append([job.payload['workspace_id'] for job in jobs])
        return [raw(job.payload['workspace_id']) for job in jobs]
    generator._generate_batch_once = generate
    result = await asyncio.gather(*(generator.generate_json(system_prompt='system', payload=request(actor))
                                  for actor in ['W1-1', 'W1-2']))
    assert result == [{'actor': 'W1-1'}, {'actor': 'W1-2'}]
    assert batches == [['W1-1', 'W1-2']]
    metrics = generator.metrics()
    assert (metrics.accepted_requests, metrics.completed_requests, metrics.pending_requests) == (2, 2, 0)
    assert (metrics.input_tokens, metrics.output_tokens) == (20, 6)


@pytest.mark.asyncio
async def test_batch_includes_hi_arriving_while_worker_thread_is_unavailable():
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    entered, release = threading.Event(), threading.Event()
    def occupy_thread():
        entered.set()
        assert release.wait(5)
    blocker = asyncio.create_task(asyncio.to_thread(occupy_thread))
    while not entered.is_set():
        await asyncio.sleep(.001)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    batches = []
    def batch(messages, jobs, schemas):
        batches.append([job.payload['workspace_id'] for job in jobs])
        return [raw(job.payload['workspace_id']) for job in jobs]
    generator._generate_batch_once = batch
    generator._generate_once = lambda *args: raw('single')
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-1')))]
    try:
        for _ in range(4):
            await asyncio.sleep(0)
        assert generator._inference_gate.locked()
        tasks.append(asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-2'))))
        for _ in range(3):
            await asyncio.sleep(0)
        release.set()
        assert await asyncio.gather(*tasks) == [{'actor': 'W1-1'}, {'actor': 'W1-2'}]
        assert batches == [['W1-1', 'W1-2']]
    finally:
        release.set()
        await asyncio.gather(blocker, *tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_batch_keeps_pixel_payloads_and_observation_receipts_with_each_hi():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    generator._processor = object()
    colors = {'W1-1': (255, 0, 0), 'W1-2': (0, 0, 255)}
    def image_message(view):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), colors[view['task_id']]).save(buffer, format='PNG')
        return {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}}]}, [view['task_id'] + '-image']
    def generate(messages, jobs, schemas):
        for sequence, job in zip(messages, jobs):
            part = sequence[-1]['content'][0]
            assert part['type'] == 'image'
            assert part['image'].getpixel((0, 0)) == colors[job.payload['workspace_id']]
        return [raw(job.payload['workspace_id']) for job in jobs]
    generator._image_message = image_message
    generator._generate_batch_once = generate
    payloads = [request(actor) for actor in colors]
    await asyncio.gather(*(generator.generate_json(system_prompt='system', payload=p) for p in payloads))
    for payload in payloads:
        expected = [payload['workspace_id'] + '-image']
        assert payload['model_view']['model_observed_image_ids'] == expected
        assert payload['model_view']['content_observed_evidence_ids'] == expected


@pytest.mark.asyncio
async def test_batch_repair_and_observer_use_each_hi_context():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, json_attempts=2))
    marker = ContextVar('actor')
    observed, repaired = [], []
    generator.generation_observer = lambda event: observed.append((event['workspace_id'], marker.get(), event['attempt']))
    generator._generate_batch_once = lambda *args: [raw('W1-1'), ('broken', 11, 4, None)]
    def repair(messages, stop, schema):
        repaired.append((copy.deepcopy(messages), marker.get()))
        return raw('W1-2')
    generator._generate_once = repair
    async def run(actor):
        marker.set(actor)
        with output_repair_scope() as budget:
            result = await generator.generate_json(system_prompt='system', payload=request(actor))
            return result, budget.failures
    values = await asyncio.gather(run('W1-1'), run('W1-2'))
    assert values == [({'actor': 'W1-1'}, 0), ({'actor': 'W1-2'}, 1)]
    assert observed == [('W1-1', 'W1-1', 1), ('W1-2', 'W1-2', 1), ('W1-2', 'W1-2', 2)]
    assert len(repaired) == 1 and repaired[0][1] == 'W1-2'
    assert any(message.get('content') == 'broken' for message in repaired[0][0])


@pytest.mark.asyncio
async def test_one_hi_exhausting_repair_budget_does_not_fail_its_peer():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    generator._generate_batch_once = lambda *args: [('broken', 10, 3, None), raw('W1-2')]
    async def run(actor):
        with output_repair_scope() as budget:
            budget.maximum_failures = 1
            return await generator.generate_json(system_prompt='system', payload=request(actor))
    results = await asyncio.gather(run('W1-1'), run('W1-2'), return_exceptions=True)
    assert isinstance(results[0], OutputRepairLimit)
    assert results[1] == {'actor': 'W1-2'}
    assert generator.metrics().failed_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['memory', 'context'])
async def test_batch_resource_failure_falls_back_per_request_and_accounts_for_wait(monkeypatch, failure):
    now = [0.]
    monkeypatch.setattr(local, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    budgets = []
    def batch(*args):
        now[0] += 5.
        raise (torch.cuda.OutOfMemoryError('batch too large') if failure == 'memory' else local.LocalModelContextError('too long'))
    def single(messages, *args):
        payload = json.loads(messages[1]['content'])
        budgets.append(payload['model_view']['runtime_constraints']['remaining_seconds'])
        now[0] += 10.
        if failure == 'context' and len(budgets) == 1:
            raise local.LocalModelContextError('individual request too long')
        return raw(payload['workspace_id'])
    generator._generate_batch_once = batch
    generator._generate_once = single
    requests = [request(actor) for actor in ['W1-1', 'W1-2']]
    original = copy.deepcopy(requests)
    result = await asyncio.gather(*(generator.generate_json(system_prompt='system', payload=p) for p in requests),
                                  return_exceptions=True)
    assert budgets == [95., 85.]
    assert requests == original
    assert result[1] == {'actor': 'W1-2'}
    if failure == 'context':
        assert isinstance(result[0], local.LocalModelContextError)
    else:
        assert result[0] == {'actor': 'W1-1'}


@pytest.mark.asyncio
async def test_cancelled_row_exits_after_ack_while_peer_keeps_inference_lock():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, max_pending_requests=2))
    entered, ack_allowed, acknowledged, release = (threading.Event() for _ in range(4))
    calls, observed = [], []
    generator.generation_observer = observed.append
    def batch(messages, jobs, schemas):
        calls.append('batch')
        entered.set()
        assert jobs[0].stop_event.wait(5)
        assert ack_allowed.wait(5)
        jobs[0].acknowledge_cancel()
        acknowledged.set()
        assert release.wait(5)
        assert not jobs[1].stop_event.is_set()
        return [raw('W1-1'), raw('W1-2')]
    def single(*args):
        calls.append('single')
        return raw('W1-3')
    generator._generate_batch_once = batch
    generator._generate_once = single
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        tasks[0].cancel()
        await asyncio.sleep(0)
        tasks[0].cancel()
        await asyncio.sleep(0)
        assert not tasks[0].done()
        ack_allowed.set()
        assert await asyncio.to_thread(acknowledged.wait, 5)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[0], 5)
        assert generator._inference_gate.locked()
        tasks.append(asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-3'))))
        await asyncio.sleep(0)
        assert calls == ['batch']
        release.set()
        assert await asyncio.gather(*tasks[1:]) == [{'actor': 'W1-2'}, {'actor': 'W1-3'}]
        assert [event['workspace_id'] for event in observed] == ['W1-2', 'W1-3']
        assert (generator.metrics().cancelled_requests, generator.metrics().pending_requests) == (1, 0)
    finally:
        ack_allowed.set()
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_while_queued_does_not_start_and_close_rejects_pending():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    calls = []
    generator._generate_once = lambda *args: calls.append('generated')
    await generator._inference_gate.acquire()
    first = asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-1')))
    second = asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-2')))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    closing = asyncio.create_task(generator.aclose())
    await asyncio.sleep(0)
    generator._inference_gate.release()
    with pytest.raises(local.LocalModelClosedError):
        await second
    await closing
    assert calls == []
    assert generator.metrics().pending_requests == 0


@pytest.mark.asyncio
async def test_successful_row_returns_before_peer_repair_finishes():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, json_attempts=2))
    repair_started, release = threading.Event(), threading.Event()
    generator._generate_batch_once = lambda *args: [raw('W1-1'), ('broken', 10, 3, None)]
    def repair(*args):
        repair_started.set()
        assert release.wait(5)
        return raw('W1-2')
    generator._generate_once = repair
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.to_thread(repair_started.wait, 5)
        assert await asyncio.wait_for(tasks[0], 5) == {'actor': 'W1-1'}
        assert generator._inference_gate.locked() and not tasks[1].done()
        release.set()
        assert await tasks[1] == {'actor': 'W1-2'}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_hi_batch_waits_for_preempted_gia_review_to_release_model():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    started, stopping, release = threading.Event(), threading.Event(), threading.Event()
    batches = []
    def review(prompt, payload, stop):
        started.set()
        assert stop.wait(5)
        stopping.set()
        assert release.wait(5)
        raise local._GenerationStopped('yielded')
    def batch(messages, jobs, schemas):
        batches.append([job.payload['workspace_id'] for job in jobs])
        return [raw(job.payload['workspace_id']) for job in jobs]
    generator._generate_json = review
    generator._generate_batch_once = batch
    task = asyncio.create_task(generator.generate_json(system_prompt='review',
        payload={'global_view': {'runtime_constraints': {'round_in_progress': True}}}))
    his = []
    try:
        assert await asyncio.to_thread(started.wait, 5)
        his = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
               for actor in ['W1-1', 'W1-2']]
        assert await asyncio.to_thread(stopping.wait, 5)
        assert not batches and generator._inference_gate.locked()
        release.set()
        with pytest.raises(PolicyReviewDeferred):
            await task
        assert await asyncio.gather(*his) == [{'actor': 'W1-1'}, {'actor': 'W1-2'}]
        assert batches == [['W1-1', 'W1-2']]
    finally:
        release.set()
        await asyncio.gather(task, *his, return_exceptions=True)


@pytest.mark.parametrize('mode', ['normal', 'cancel', 'decode_failure', 'cancel_all'])
def test_batch_tensor_generation_uses_separate_constraints_padding_counts_and_cancel(monkeypatch, mode):
    class Tokenizer:
        eos_token_id, pad_token_id, padding_side = 1, 0, 'right'
        def __call__(self, prompts, **kwargs):
            assert self.padding_side == 'left' and len(prompts) == 2
            return {'input_ids': torch.tensor([[0, 9, 10], [11, 12, 13]]),
                    'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]])}
        def encode(self, *args, **kwargs):
            return [7]
        def decode(self, values, **kwargs):
            return ' '.join(str(int(v)) for v in values if int(v) not in [0, 1])
    acknowledged, delivered = [], []
    phase = {'finished': False}
    def deliver(row, value):
        assert not phase['finished']
        delivered.append((row, value))
    jobs = [SimpleNamespace(stop_event=threading.Event(), acknowledge_cancel=lambda i=i: acknowledged.append(i),
                            on_generated=lambda value, i=i: deliver(i, value))
            for i in range(2)]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.config = SimpleNamespace(max_position_embeddings=100)
        def generate(self, input_ids, logits_processor, stopping_criteria, **kwargs):
            done = torch.zeros(2, dtype=torch.bool)
            if mode in ['cancel', 'cancel_all']:
                jobs[0].stop_event.set()
            if mode == 'cancel_all':
                jobs[1].stop_event.set()
            for step in range(4):
                if step == 2 and mode in ['normal', 'decode_failure']:
                    assert [row for row, _ in delivered] == [0]
                self.blocks[0](torch.zeros(1))
                scores = torch.zeros(2, 16)
                if mode == 'cancel':
                    scores[0, 1] = float('-inf')
                for processor in logits_processor:
                    scores = processor(input_ids, scores)
                token = scores.argmax(-1).masked_fill(done, 0)
                input_ids = torch.cat([input_ids, token[:, None]], dim=1)
                done |= (token == 1) | stopping_criteria(input_ids, scores)
                if done.all():
                    phase['finished'] = True
                    return input_ids
            raise AssertionError('rows did not stop')
    class Enforcer:
        def __init__(self, tokens):
            self.tokens = tokens
            self.start = None
        def get_allowed_tokens(self, values):
            if self.start is None:
                self.start = len(values)
            return SimpleNamespace(allowed_tokens=[self.tokens[len(values) - self.start]])
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', constrain_json=True, max_new_tokens=8))
    tokenizer, model = Tokenizer(), Model()
    generator._load = lambda: (model, tokenizer)
    generator._render_prompt = lambda *args: 'prompt'
    generator._constraint_tokens = lambda *args: object()
    monkeypatch.setattr(local, '_create_output_enforcer', lambda data, schema: Enforcer(schema['tokens']))
    if mode == 'decode_failure':
        decode = local._decode_generation_parts
        def decode_row(tokenizer, ids, **kwargs):
            if ids.tolist()[0] == 4:
                raise local.LocalModelOutputError('first row has no final answer')
            return decode(tokenizer, ids, **kwargs)
        monkeypatch.setattr(local, '_decode_generation_parts', decode_row)
    if mode == 'cancel_all':
        with pytest.raises(local._GenerationStopped):
            generator._generate_batch_once([[{}], [{}]], jobs, [{'tokens': [4, 1]}, {'tokens': [5, 6, 1]}])
        assert not model.blocks[0]._forward_pre_hooks
        return
    output = generator._generate_batch_once([[{}], [{}]], jobs, [{'tokens': [4, 1]}, {'tokens': [5, 6, 1]}])
    assert output[1] == ('5 6', 3, 3, None)
    if mode == 'decode_failure':
        assert isinstance(output[0], local.LocalModelOutputError)
    else:
        assert output[0] == (('', 2, 1, None) if mode == 'cancel' else ('4', 2, 2, None))
    assert dict(delivered) == ({1: output[1]} if mode == 'cancel' else dict(enumerate(output)))
    assert bool(acknowledged) == (mode == 'cancel')
    assert tokenizer.padding_side == 'right'
    assert not model.blocks[0]._forward_pre_hooks


@pytest.mark.asyncio
async def test_cancelled_fallback_waiter_does_not_wait_for_another_row():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    entered, release = threading.Event(), threading.Event()
    def batch(*args):
        raise torch.cuda.OutOfMemoryError('too large')
    def single(messages, stop, schema):
        if stop.is_set():
            raise local._GenerationStopped('cancelled')
        entered.set()
        assert release.wait(5)
        return raw('W1-1')
    generator._generate_batch_once = batch
    generator._generate_once = single
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[1], 1)
        assert generator._inference_gate.locked()
        release.set()
        assert await tasks[0] == {'actor': 'W1-1'}
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
