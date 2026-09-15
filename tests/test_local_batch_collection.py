import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

import foren_searcher.local_generation_queue as queues


def queue_fixture(monkeypatch, duration=40.):
    offset = [0.]
    clock = lambda: time.perf_counter() + offset[0]
    monkeypatch.setattr(queues, 'time', SimpleNamespace(perf_counter=clock))
    batches = []
    def generate(jobs):
        batches.append([job.payload['actor'] for job in jobs])
        offset[0] += duration
        return [job.payload['actor'] for job in jobs]
    queue = queues.LocalGenerationQueue(asyncio.Lock(), lambda: None, generate, 2)
    def submit(actor, age=0.):
        return asyncio.create_task(queue.submit(system_prompt='system', payload={'actor': actor},
            stop_event=threading.Event(), queued_at=clock()-age, on_start=lambda: None))
    return queue, submit, batches


async def collecting(queue):
    for _ in range(100):
        if queue.gate.locked() and queue.pending:
            return
        await asyncio.sleep(.001)
    raise AssertionError('queue did not enter collection')


@pytest.mark.asyncio
async def test_slow_batch_collects_staggered_followups_and_full_batch_dispatches_immediately(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    assert await asyncio.gather(submit('A'), submit('B')) == ['A', 'B']
    first = submit('A-next')
    await collecting(queue)
    assert not first.done() and batches == [['A', 'B']]
    second = submit('B-next')
    assert await asyncio.wait_for(asyncio.gather(first, second), 1.) == ['A-next', 'B-next']
    assert batches == [['A', 'B'], ['A-next', 'B-next']]


@pytest.mark.asyncio
async def test_cold_and_repeated_single_hi_do_not_wait(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    for actor in ('A', 'B', 'C'):
        assert await asyncio.wait_for(submit(actor), 1.) == actor
    assert batches == [['A'], ['B'], ['C']]


@pytest.mark.asyncio
async def test_time_already_spent_queued_counts_and_singleton_resets_collection(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    await asyncio.gather(submit('A'), submit('B'))
    assert await asyncio.wait_for(submit('old', age=3.), 1.) == 'old'
    assert await asyncio.wait_for(submit('next'), 1.) == 'next'
    assert batches == [['A', 'B'], ['old'], ['next']]


@pytest.mark.asyncio
async def test_fast_model_does_not_inherit_a_two_second_wait(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch, duration=.01)
    await asyncio.gather(submit('A'), submit('B'))
    assert await asyncio.wait_for(submit('next'), 1.) == 'next'
    assert batches[-1] == ['next']


@pytest.mark.asyncio
async def test_cancelled_last_waiter_releases_gate_without_generating(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    await asyncio.gather(submit('A'), submit('B'))
    first = submit('cancelled')
    await collecting(queue)
    runner = queue.runner
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(asyncio.shield(runner), 1.)
    assert not queue.gate.locked() and not queue.pending
    assert batches == [['A', 'B']]


@pytest.mark.asyncio
async def test_external_request_ends_collection_without_discarding_pending_hi(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    await asyncio.gather(submit('A'), submit('B'))
    first = submit('pending')
    await collecting(queue)
    queue.stop_collecting()
    assert await asyncio.wait_for(first, 1.) == 'pending'
    assert batches[-1] == ['pending']


@pytest.mark.asyncio
async def test_close_wakes_collection_and_rejects_pending_before_worker_generation(monkeypatch):
    queue, submit, batches = queue_fixture(monkeypatch)
    await asyncio.gather(submit('A'), submit('B'))
    first = submit('pending')
    await collecting(queue)
    def closed():
        raise RuntimeError('closed')
    queue.ensure_open = closed
    queue.stop_collecting()
    with pytest.raises(RuntimeError, match='closed'):
        await asyncio.wait_for(first, 1.)
    assert batches == [['A', 'B']]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['raised', 'row-error', 'cancelled'])
async def test_unsuccessful_batch_does_not_delay_following_request(monkeypatch, kind):
    queue, submit, batches = queue_fixture(monkeypatch)
    original = queue.generate
    def failed(jobs):
        result = original(jobs)
        if kind == 'raised':
            raise ValueError('failed')
        if kind == 'row-error':
            result[0] = ValueError('failed')
        else:
            jobs[0].stop_event.set()
        return result
    queue.generate = failed
    await asyncio.gather(submit('A'), submit('B'), return_exceptions=True)
    queue.generate = original
    assert await asyncio.wait_for(submit('next'), 1.) == 'next'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['gia', 'close'])
async def test_generator_lifecycle_wakes_collection(monkeypatch, operation):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    import foren_searcher.local_models as local
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    generator._generate_once = lambda *args: ('{"ok":true}', 1, 1, None)
    generator._unload = lambda: None
    await generator.generate_json(system_prompt='system', payload={})
    generator._batch_queue._collection_seconds = 2.
    first = asyncio.create_task(generator.generate_json(system_prompt='system', payload={
        'workspace_id': 'W1-1', 'model_view': {'runtime_constraints': {'remaining_seconds': 100.}}}))
    await collecting(generator._batch_queue)
    if operation == 'close':
        closing = asyncio.create_task(generator.aclose())
        with pytest.raises(local.LocalModelClosedError):
            await asyncio.wait_for(first, 1.)
        await asyncio.wait_for(closing, 1.)
    else:
        other = asyncio.create_task(generator.generate_json(system_prompt='system', payload={}))
        assert await asyncio.wait_for(asyncio.gather(first, other), 1.) == [{'ok': True}, {'ok': True}]
    assert generator.metrics().pending_requests == 0


@pytest.mark.asyncio
async def test_collection_wait_is_deducted_from_hi_budget(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    import foren_searcher.local_models as local
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    generator._batch_queue._collection_seconds = .2
    received = []
    def generate(messages, *args):
        received.append(json.loads(messages[1]['content'])['model_view']['runtime_constraints']['remaining_seconds'])
        return '{"ok":true}', 1, 1, None
    generator._generate_once = generate
    payload = {'workspace_id': 'W1-1', 'model_view': {'runtime_constraints': {'remaining_seconds': 100.}}}
    assert await generator.generate_json(system_prompt='system', payload=payload) == {'ok': True}
    assert 95. < received[0] <= 99.82
    assert payload['model_view']['runtime_constraints']['remaining_seconds'] == 100.


@pytest.mark.asyncio
async def test_optional_gia_review_preserves_collection_and_yields_to_arriving_hi(monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    import foren_searcher.local_models as local
    from foren_searcher.policies import PolicyReviewDeferred
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    generator._generate_once = lambda *args: ('{"ok":true}', 1, 1, None)
    generator._generate_batch_once = lambda messages, jobs, schemas: [('{"ok":true}', 1, 1, None) for _ in jobs]
    await generator.generate_json(system_prompt='system', payload={})
    generator._batch_queue._collection_seconds = 2.
    first = asyncio.create_task(generator.generate_json(system_prompt='system', payload={'workspace_id': 'A'}))
    await collecting(generator._batch_queue)
    review = asyncio.create_task(generator.generate_json(system_prompt='system', payload={
        'global_view': {'runtime_constraints': {'round_in_progress': True}}}))
    await asyncio.sleep(.03)
    assert not first.done() and len(generator._batch_queue.pending) == 1
    second = asyncio.create_task(generator.generate_json(system_prompt='system', payload={'workspace_id': 'B'}))
    assert await asyncio.wait_for(asyncio.gather(first, second), 1.) == [{'ok': True}, {'ok': True}]
    with pytest.raises(PolicyReviewDeferred):
        await review
    assert generator.metrics().pending_requests == 0
