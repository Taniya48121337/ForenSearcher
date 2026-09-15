import asyncio
import json
import threading

import pytest

import foren_searcher.local_models as local
from foren_searcher.policies import output_repair_scope

torch = pytest.importorskip('torch')


def request(actor):
    return {'workspace_id': actor, 'model_view': {'task_id': actor}}


def raw(actor):
    return json.dumps({'actor': actor}), 10, 3, None


@pytest.mark.asyncio
async def test_finished_row_returns_while_peer_generates_and_gpu_gate_stays_owned(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    release = threading.Event()
    receipts, singles = [], []
    generator.generation_observer = lambda event: receipts.append(event['workspace_id'])
    def batch(messages, jobs, schemas):
        jobs[0].on_generated(raw('W1-1'))
        assert release.wait(5)
        return [raw('W1-1'), raw('W1-2')]
    def single(messages, *args):
        actor = json.loads(messages[1]['content'])['workspace_id']
        singles.append(actor)
        return raw(actor)
    generator._generate_batch_once = batch
    generator._generate_once = single
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.wait_for(asyncio.shield(tasks[0]), 2) == {'actor': 'W1-1'}
        assert not tasks[1].done() and generator._inference_gate.locked()
        assert receipts == ['W1-1']
        tasks.append(asyncio.create_task(generator.generate_json(system_prompt='system', payload=request('W1-3'))))
        await asyncio.sleep(.02)
        assert not singles and not tasks[2].done()
        release.set()
        assert await asyncio.gather(*tasks) == [{'actor': actor} for actor in ['W1-1', 'W1-2', 'W1-3']]
        assert receipts == ['W1-1', 'W1-2', 'W1-3']
        assert singles == ['W1-3']
        assert generator.metrics().output_tokens == 9
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['oom', 'exception'])
async def test_peer_failure_does_not_repeat_or_erase_an_already_delivered_row(monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    release = threading.Event()
    receipts, singles = [], []
    generator.generation_observer = lambda event: receipts.append(event['workspace_id'])
    def batch(messages, jobs, schemas):
        jobs[0].on_generated(raw('W1-1'))
        assert release.wait(5)
        raise torch.cuda.OutOfMemoryError('peer allocation') if failure == 'oom' else ValueError('peer failure')
    def single(messages, *args):
        actor = json.loads(messages[1]['content'])['workspace_id']
        singles.append(actor)
        return raw(actor)
    generator._generate_batch_once = batch
    generator._generate_once = single
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.wait_for(asyncio.shield(tasks[0]), 2) == {'actor': 'W1-1'}
        release.set()
        values = await asyncio.gather(*tasks, return_exceptions=True)
        assert values[0] == {'actor': 'W1-1'}
        if failure == 'oom':
            assert values[1] == {'actor': 'W1-2'} and singles == ['W1-2']
            assert receipts == ['W1-1', 'W1-2']
        else:
            assert isinstance(values[1], ValueError) and not singles
            assert receipts == ['W1-1']
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await generator.aclose()


@pytest.mark.asyncio
async def test_early_malformed_row_repairs_once_in_its_own_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, json_attempts=2))
    release = threading.Event()
    receipts = []
    broken = ('broken', 10, 3, None)
    generator.generation_observer = lambda event: receipts.append((event['workspace_id'], event['attempt']))
    def batch(messages, jobs, schemas):
        jobs[0].on_generated(broken)
        jobs[1].on_generated(raw('W1-2'))
        assert release.wait(5)
        return [broken, raw('W1-2')]
    generator._generate_batch_once = batch
    generator._generate_once = lambda *args: raw('W1-1')
    async def run(actor):
        with output_repair_scope() as budget:
            value = await generator.generate_json(system_prompt='system', payload=request(actor))
            return value, budget.failures
    tasks = [asyncio.create_task(run(actor)) for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.wait_for(asyncio.shield(tasks[1]), 2) == ({'actor': 'W1-2'}, 0)
        assert not tasks[0].done()
        release.set()
        assert await tasks[0] == ({'actor': 'W1-1'}, 1)
        assert receipts == [('W1-1', 1), ('W1-2', 1), ('W1-1', 2)]
        assert generator.metrics().input_tokens == 30
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await generator.aclose()


@pytest.mark.asyncio
async def test_cancelling_unfinished_peer_does_not_create_its_observation_receipt(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2))
    receipts = []
    generator.generation_observer = lambda event: receipts.append(event['workspace_id'])
    def batch(messages, jobs, schemas):
        jobs[0].on_generated(raw('W1-1'))
        assert jobs[1].stop_event.wait(5)
        jobs[1].acknowledge_cancel()
        return [raw('W1-1'), raw('W1-2')]
    generator._generate_batch_once = batch
    tasks = [asyncio.create_task(generator.generate_json(system_prompt='system', payload=request(actor)))
             for actor in ['W1-1', 'W1-2']]
    try:
        assert await asyncio.wait_for(asyncio.shield(tasks[0]), 2) == {'actor': 'W1-1'}
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError): await tasks[1]
        if generator._batch_queue.runner is not None:
            await generator._batch_queue.runner
        assert receipts == ['W1-1']
    finally:
        tasks[1].cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await generator.aclose()
