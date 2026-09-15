import threading
from types import SimpleNamespace

import pytest
import torch

import foren_searcher.local_models as local


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('thinking', ['none', 'natural', 'forced'])
def test_constraints_ignore_long_prompt_and_keep_generated_token_offsets(monkeypatch, batched, thinking):
    width = 2048
    class Tokenizer:
        eos_token_id, pad_token_id, padding_side = 1, 0, 'right'
        def __call__(self, prompts, **kwargs):
            rows = len(prompts) if isinstance(prompts, list) else 1
            ids = torch.full((rows, width), 9, dtype=torch.long)
            ids[:, -3:] = torch.tensor([7, 1, 4])
            mask = torch.ones_like(ids)
            if rows > 1:
                assert self.padding_side == 'left'
                ids[0, 0], mask[0, 0] = 0, 0
            return {'input_ids': ids, 'attention_mask': mask}
        def encode(self, text, **kwargs):
            assert text == '</think>'
            return [7]
    observed = []
    class Enforcer:
        def __init__(self):
            self.prefixes = []
            observed.append(self.prefixes)
        def get_allowed_tokens(self, values):
            self.prefixes.append(list(values))
            assert values == [4, 5][:len(values)]
            return SimpleNamespace(allowed_tokens=[[4, 5, 1][len(values)]])
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.config = SimpleNamespace(max_position_embeddings=4096)
            self.generation_config = SimpleNamespace(eos_token_id=1)
        def generate(self, input_ids, logits_processor, stopping_criteria, **kwargs):
            for step in range(8):
                scores = torch.zeros(input_ids.shape[0], 16)
                preferred = 9
                if thinking != 'none' and step == 0:
                    preferred = 8
                if thinking == 'natural' and step == 1:
                    preferred = 7
                scores[:, preferred] = 10
                for processor in logits_processor:
                    scores = processor(input_ids, scores)
                token = scores.argmax(-1)
                input_ids = torch.cat([input_ids, token[:, None]], dim=1)
                stopping_criteria(input_ids, scores)
                if token.eq(1).all().item():
                    return input_ids
            pytest.fail('constrained output did not finish')
    generator = local.TransformersJsonGenerator(local.LocalModelConfig('unused', max_new_tokens=8,
        constrain_json=True, enable_thinking=thinking != 'none', max_reasoning_tokens=1 if thinking == 'forced' else 4))
    generator._load = lambda: (model, tokenizer)
    generator._render_prompt = lambda *args: 'prompt'
    generator._constraint_tokens = lambda *args: object()
    monkeypatch.setattr(local, '_create_output_enforcer', lambda *args: Enforcer())
    monkeypatch.setattr(local, '_decode_generation_parts', lambda tokenizer, ids, **kwargs: (str(ids.tolist()), None))
    original_tolist = torch.Tensor.tolist
    copied = []
    def checked_tolist(tensor):
        copied.append(tensor.numel())
        assert tensor.numel() <= 16, 'decoder copied the prompt prefix to Python'
        return original_tolist(tensor)
    monkeypatch.setattr(torch.Tensor, 'tolist', checked_tolist)
    tokenizer, model = Tokenizer(), Model()
    if batched:
        jobs = [SimpleNamespace(stop_event=threading.Event(), acknowledge_cancel=lambda: None) for _ in range(2)]
        outputs = generator._generate_batch_once([[{}], [{}]], jobs, [{}, {}])
        assert [r[1] for r in outputs] == [width-1, width]
    else:
        outputs = [generator._generate_once([{}], threading.Event(), {})]
        assert outputs[0][1] == width
    expected = [4, 5, 1] if thinking == 'none' else [8, 7, 4, 5, 1]
    assert all(row[0] == str(expected) and row[2] == len(expected) for row in outputs)
    assert observed == [[[], [4], [4, 5]]] * len(outputs)
    assert copied and max(copied) <= 16
    assert tokenizer.padding_side == 'right'
