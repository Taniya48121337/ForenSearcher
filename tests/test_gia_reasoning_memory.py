import json

import pytest

from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.local_models import LocalModelConfig, TransformersGiaPolicy, TransformersJsonGenerator
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy
from foren_searcher.openrouter import OpenRouterJsonGenerator
from foren_searcher.orchestration import GiaRunner
from foren_searcher.store import GlobalStore
from inspect_helpers import image_bytes, ingest
from test_hi_multimodal import attached


@pytest.mark.asyncio
async def test_gia_keeps_own_visual_memory_when_next_read_displaces_pixels(tmp_path):
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    sources = [ingest(store, image_bytes(color=color, red_box=False), source=f'I{i}', filename=f'{i}.png',
                     media_type='image/png', modality='image') for i, color in enumerate(('red', 'blue'))]
    first, second = (item.evidence_id for item in sources)
    store.record_event(actor_role='HI', actor_id='W1-1', event_type='raw_model_output',
                       payload={'reasoning': 'HI_PRIVATE', 'image_evidence_ids': [first]})
    store.record_event(actor_role='GIA', actor_id='GIA', event_type='raw_model_output',
                       payload={'reasoning': 'OTHER_SESSION_PRIVATE', 'image_evidence_ids': [first]})
    calls = []

    class Client:
        async def complete(self, messages, **kwargs):
            calls.append(messages)
            text = json.dumps(messages)
            assert 'HI_PRIVATE' not in text and 'OTHER_SESSION_PRIVATE' not in text
            thoughts = [message['reasoning'] for message in messages if message.get('reasoning')]
            pixels = attached(messages)
            assert len(pixels) == 1
            if len(calls) % 2:
                assert thoughts == []
                assert pixels[0][0]['evidence_id'] == first
                assert pixels[0][1].getpixel((0, 0)) == (255, 0, 0)
                output = {'action': 'READ_RECORD', 'arguments': {'record_id': second}}
                reasoning = f'{first} is red. Read the other image to compare.'
            else:
                assert thoughts == [f'{first} is red. Read the other image to compare.']
                assert pixels[0][0]['evidence_id'] == second
                assert pixels[0][1].getpixel((0, 0)) == (0, 0, 255)
                assert 'model_reasoning' not in text
                assert '重复读取记录不会取得媒体内容' not in text
                assert 'READ_RECORD' in text and 'direct_visual_input=true' in text
                output = {'decision': 'FINISH', 'answer': f'{first} is red; {second} is blue.'}
                reasoning = 'FINAL_PRIVATE'
            return {'text': json.dumps(output), 'reasoning': reasoning, 'model': 'test',
                    'provider': 'test', 'generation_id': 'test', 'usage': {}}

    generator = OpenRouterJsonGenerator(Client(), store=store, image_policy=InspectPolicy(maximum_visual_qa_images=1))
    memory = MemoryPolicy()
    view = GlobalViewBuilder(store, memory).build(task='Compare both image colors.', original_material_ids=(first, second), latest_round_index=None)
    runner = GiaRunner(store=store, policy=TransformersGiaPolicy(generator), memory_policy=memory, max_turns=2)
    for _ in range(2):
        answer = await runner.run(view, round_index=2, can_create_round=False)
        assert answer.answer == f'{first} is red; {second} is blue.'
    inputs = store.trajectory_events(actor_role='GIA', event_type='model_input')
    assert all('FINAL_PRIVATE' not in json.dumps(item) for item in inputs)
    assert all('reasoning' not in json.dumps(item) for item in (e.to_dict() for e in store.evidence_records()))


@pytest.mark.asyncio
@pytest.mark.parametrize('budget', [2000, 14000])
async def test_gia_reasoning_budget_preserves_original_read_result(budget):
    store = GlobalStore()
    evidence = store.record_evidence(content='Original source fact', label='Source', modalities=('text',),
                                    source_ref='I0', locator='source', acquired_by='TEST', discovered_by='HARNESS')
    memory = MemoryPolicy(max_model_view_chars=budget)
    calls = 0
    thought = 'Private tentative comparison. ' * 240

    class Policy:
        async def decide(self, *, model_view, history, actions):
            nonlocal calls
            calls += 1
            if calls == 1:
                model_view['model_reasoning'] = thought
                return {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id}}
            assert len(json.dumps(model_view, ensure_ascii=False)) + len(json.dumps(history, ensure_ascii=False)) <= budget
            assert len(history) == 1 and history[0]['observation']['record']['content'] == 'Original source fact'
            if budget == 2000:
                assert 'reasoning' not in history[0]
                assert model_view['reasoning_context_omitted'] is True
            else:
                assert history[0]['reasoning'] == thought
            return {'decision': 'FINISH', 'answer': 'Original source fact E1'}

    view = GlobalViewBuilder(store, memory).build(task='Read source', original_material_ids=(), latest_round_index=None)
    await GiaRunner(store=store, policy=Policy(), memory_policy=memory, max_turns=2).run(view, round_index=1)


@pytest.mark.asyncio
async def test_local_gia_reasoning_belongs_to_successful_request_and_not_hi(monkeypatch):
    generator = TransformersJsonGenerator(LocalModelConfig('unused'))
    values = iter([('{"action":"READ_RECORD","arguments":{"record_id":"E1"}}', 10, 20, 'first'),
                   ('{"decision":"WAIT"}', 10, 20, None), ('{"action":"READ_RECORD"}', 10, 20, 'HI_PRIVATE')])
    monkeypatch.setattr(generator, '_generate_once', lambda *args: next(values))
    view = {}
    try:
        await generator.generate_json(system_prompt='rules', payload={'global_view': view})
        assert view['model_reasoning'] == 'first'
        await generator.generate_json(system_prompt='rules', payload={'global_view': view})
        assert 'model_reasoning' not in view
        await generator.generate_json(system_prompt='rules', payload={'workspace_id': 'W1-1', 'model_view': view})
        assert 'model_reasoning' not in view
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_invalid_gia_memory_action_does_not_carry_reasoning_into_history():
    store = GlobalStore()
    memory = MemoryPolicy()
    calls = 0

    class Policy:
        async def decide(self, *, model_view, history, actions):
            nonlocal calls
            calls += 1
            if calls == 1:
                model_view['model_reasoning'] = 'REJECTED_PRIVATE'
                return {'action': 'READ_RECORD', 'arguments': {'record_id': 'E999'}}
            assert 'REJECTED_PRIVATE' not in json.dumps(history)
            assert history[-1]['observation']['action'] == 'HARNESS_REJECTION'
            return {'decision': 'FINISH', 'answer': 'Evidence is unavailable.'}

    view = GlobalViewBuilder(store, memory).build(task='Read source', original_material_ids=(), latest_round_index=None)
    await GiaRunner(store=store, policy=Policy(), memory_policy=memory, max_turns=2).run(view, round_index=1)


@pytest.mark.asyncio
async def test_policy_retry_does_not_reuse_rejected_or_stale_reasoning():
    store = GlobalStore()
    evidence = store.record_evidence(content='Source text', label='Source', modalities=('text',),
        source_ref='I0', locator='source', acquired_by='TEST', discovered_by='HARNESS')
    calls = 0

    class Client:
        async def complete(self, messages, **kwargs):
            nonlocal calls
            calls += 1
            assert 'REJECTED_PRIVATE' not in json.dumps(messages)
            if calls == 1:
                output = {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id, 'unexpected': 1}}
                reasoning = 'REJECTED_PRIVATE'
            elif calls == 2:
                output = {'action': 'READ_RECORD', 'arguments': {'record_id': evidence.evidence_id}}
                reasoning = None
            else:
                assert not any(message.get('reasoning') for message in messages)
                output = {'decision': 'FINISH', 'answer': 'Source text E1'}
                reasoning = None
            return {'text': json.dumps(output), 'reasoning': reasoning, 'model': 'test',
                    'provider': 'test', 'generation_id': 'test', 'usage': {}}

    generator = OpenRouterJsonGenerator(Client(), store=store)
    memory = MemoryPolicy()
    view = GlobalViewBuilder(store, memory).build(task='Read source', original_material_ids=(evidence.evidence_id,), latest_round_index=None)
    await GiaRunner(store=store, policy=TransformersGiaPolicy(generator), memory_policy=memory, max_turns=2).run(view, round_index=1, can_create_round=False)
    assert calls == 3
