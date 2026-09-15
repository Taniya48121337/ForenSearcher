import asyncio
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

import foren_searcher.local_models as local
from foren_searcher.policies import OutputRepairLimit, output_repair_scope


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['json', 'repair-limit', 'observer', 'inference'])
async def test_failed_batch_releases_worker_tensors_with_error_still_retained(failure):
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, json_attempts=1))
    allocations, budgets = [], {}
    def batch(messages, jobs, schemas):
        scratch = torch.empty(1024)
        allocations.append(weakref.ref(scratch))
        outputs = [('broken' if failure in {'json', 'repair-limit'} else '{"ok":true}', 10, 3, None),
                   ('{"ok":true}', 10, 3, None)]
        jobs[1].on_generated(outputs[1])
        if failure == 'inference':
            try:
                raise ValueError('underlying failure')
            except ValueError as exc:
                raise RuntimeError('generation failed') from exc
        jobs[0].on_generated(outputs[0])
        return outputs
    def observe(event):
        if failure == 'observer' and event['workspace_id'] == 'W1-1':
            raise RuntimeError('observer failed')
    generator._generate_batch_once = batch
    generator.generation_observer = observe
    async def call(actor):
        with output_repair_scope() as budget:
            if failure == 'repair-limit':
                budget.maximum_failures = 1
            budgets[actor] = budget
            return await generator.generate_json(system_prompt='system', payload={
                'workspace_id': actor, 'model_view': {'task_id': actor}})
    try:
        results = await asyncio.gather(call('W1-1'), call('W1-2'), return_exceptions=True)
        if generator._batch_queue.runner:
            await generator._batch_queue.runner
        await asyncio.sleep(0)
        assert results[1] == {'ok': True}
        error = results[0]
        assert isinstance(error, {'json': local.LocalModelOutputError, 'repair-limit': OutputRepairLimit,
                                  'observer': RuntimeError, 'inference': RuntimeError}[failure])
        assert allocations and all(ref() is None for ref in allocations)
        assert any('Inference traceback:' in note for note in error.__notes__)
        if failure == 'inference':
            assert str(error) == 'generation failed'
            assert isinstance(error.__cause__, ValueError)
            assert str(error.__cause__) == 'underlying failure'
            assert any('batch' in note for note in error.__notes__)
        if failure == 'repair-limit':
            assert isinstance(error.__cause__, local.LocalModelOutputError)
        assert budgets['W1-1'].failures == (1 if failure in {'json', 'repair-limit'} else 0)
        assert budgets['W1-2'].failures == 0
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_single_inference_error_keeps_diagnostics_without_worker_tensor():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=1))
    allocations = []
    def generate(*args):
        scratch = torch.empty(1024)
        allocations.append(weakref.ref(scratch))
        try:
            raise ValueError('underlying failure')
        except ValueError as exc:
            error = RuntimeError('generation failed')
            error.add_note('original note')
            raise error from exc
    generator._generate_once = generate
    try:
        result, = await asyncio.gather(generator.generate_json(system_prompt='system', payload={}), return_exceptions=True)
        assert isinstance(result, RuntimeError) and str(result) == 'generation failed'
        assert isinstance(result.__cause__, ValueError)
        assert result.__notes__[0] == 'original note'
        assert any('generate' in note for note in result.__notes__[1:])
        assert all(ref() is None for ref in allocations)
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_batch_retry_starts_after_failed_generation_tensor_is_released():
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_batch_size=2, json_attempts=2))
    allocations, repairs, budgets = [], [], {}
    def batch(messages, jobs, schemas):
        scratch = torch.empty(1024)
        allocations.append(weakref.ref(scratch))
        outputs = [('broken', 10, 3, None), ('{"ok":true}', 10, 3, None)]
        for job, output in zip(jobs, outputs):
            job.on_generated(output)
        return outputs
    def repair(messages, *args):
        assert allocations and all(ref() is None for ref in allocations)
        repairs.append(messages)
        return '{"repaired":true}', 12, 4, None
    generator._generate_batch_once = batch
    generator._generate_once = repair
    async def call(actor):
        with output_repair_scope() as budget:
            budgets[actor] = budget
            return await generator.generate_json(system_prompt='system', payload={
                'workspace_id': actor, 'model_view': {'task_id': actor}})
    try:
        results = await asyncio.gather(call('W1-1'), call('W1-2'))
        assert results == [{'repaired': True}, {'ok': True}]
        assert len(repairs) == 1
        assert any(message.get('content') == 'broken' for message in repairs[0])
        assert budgets['W1-1'].failures == 1 and budgets['W1-2'].failures == 0
        metrics = generator.metrics()
        assert (metrics.input_tokens, metrics.output_tokens) == (32, 10)
    finally:
        await generator.aclose()


def test_batch_decode_failure_releases_generated_tensor(monkeypatch):
    allocations = []
    class Tokenizer:
        eos_token_id, pad_token_id, padding_side = 1, 0, 'right'
        def __call__(self, prompts, **kwargs):
            return {'input_ids': torch.full((2, 2), 3), 'attention_mask': torch.ones((2, 2), dtype=torch.long)}
        def encode(self, *args, **kwargs):
            return [2]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4096)
        def generate(self, input_ids, stopping_criteria, **kwargs):
            output = torch.cat([input_ids, torch.ones((2, 1), dtype=torch.long)], dim=1)
            allocations.append(weakref.ref(output))
            stopping_criteria(output, None)
            return output
    def decode(*args, **kwargs):
        raise local.LocalModelOutputError('decode failed')
    monkeypatch.setattr(local, '_decode_generation_parts', decode)
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_new_tokens=8, constrain_json=False))
    generator._load = lambda: (Model(), Tokenizer())
    generator._render_prompt = lambda *args: 'prompt'
    received = []
    jobs = [SimpleNamespace(stop_event=threading.Event(), acknowledge_cancel=lambda: None, on_generated=received.append) for _ in range(2)]
    results = generator._generate_batch_once([[{}], [{}]], jobs, [None, None])
    assert len(received) == 2
    assert results == received and all(isinstance(error, local.LocalModelOutputError) for error in results)
    assert all(str(error) == 'decode failed' for error in results)
    assert allocations and all(ref() is None for ref in allocations)
