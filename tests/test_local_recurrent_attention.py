from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
qwen = pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')

from foren_searcher.local_models import _cached_qwen35_delta_rule, _configure_cuda_recurrent_attention


def inputs(batch, length, dtype):
    return (
        torch.randn(batch, length, 3, 16, dtype=dtype)[..., ::2] * .1,
        torch.randn(batch, length, 3, 16, dtype=dtype)[..., ::2] * .1,
        torch.randn(batch, length, 3, 10, dtype=dtype)[..., ::2],
        -torch.rand(batch, length, 6)[..., ::2],
        torch.rand(batch, length, 6, dtype=dtype)[..., ::2],
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('normalize', [False, True])
def test_cached_single_token_matches_native_through_successive_updates(dtype, batch, normalize):
    torch.manual_seed(73)
    original_state = torch.randn(batch, 3, 8, 5)
    optimized_state = original_state.clone()
    with torch.inference_mode():
        for _ in range(24):
            tensors = inputs(batch, 1, dtype)
            before = [value.clone() for value in tensors]
            original_before = original_state.clone()
            optimized_before = optimized_state.clone()
            expected, next_original = qwen.torch_recurrent_gated_delta_rule(*tensors, original_state, True, normalize)
            actual, next_optimized = _cached_qwen35_delta_rule(*tensors, optimized_state, True, normalize)
            assert torch.equal(expected, actual)
            assert torch.equal(next_original, next_optimized)
            assert torch.equal(original_state, original_before)
            assert torch.equal(optimized_state, optimized_before)
            assert all(torch.equal(value, saved) for value, saved in zip(tensors, before))
            original_state, optimized_state = next_original, next_optimized


def test_cached_single_token_can_omit_final_state():
    torch.manual_seed(97)
    tensors = inputs(1, 1, torch.bfloat16)
    state = torch.randn(1, 3, 8, 5)
    expected = qwen.torch_recurrent_gated_delta_rule(*tensors, state, False, True)
    actual = _cached_qwen35_delta_rule(*tensors, state, False, True)
    assert actual[1] is None and expected[1] is None
    assert torch.equal(actual[0], expected[0])


@pytest.mark.parametrize('length,cached', [(1, False), (3, True), (3, False)])
def test_uncached_or_multiple_tokens_use_native_operation(monkeypatch, length, cached):
    native = qwen.torch_recurrent_gated_delta_rule
    calls = []
    def observe(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)
    monkeypatch.setattr(qwen, 'torch_recurrent_gated_delta_rule', observe)
    tensors = inputs(1, length, torch.float32)
    state = torch.randn(1, 3, 8, 5) if cached else None
    expected = native(*tensors, state, True, True)
    actual = _cached_qwen35_delta_rule(*tensors, state, True, True)
    assert calls == [True]
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_loader_selects_only_cuda_qwen35_python_recurrence(monkeypatch):
    fallback = qwen.torch_recurrent_gated_delta_rule
    fused = lambda *args: None
    class Layer:
        def __init__(self, device, operation):
            self.device = device
            self.recurrent_gated_delta_rule = operation
        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type=self.device))
    monkeypatch.setattr(qwen, 'Qwen3_5GatedDeltaNet', Layer)
    layers = [Layer('cuda', fallback), Layer('cpu', fallback), Layer('cuda', fused)]
    model = SimpleNamespace(config=SimpleNamespace(model_type='qwen3'), modules=lambda: iter(layers))
    _configure_cuda_recurrent_attention(model)
    assert layers[0].recurrent_gated_delta_rule is fallback
    model.config.model_type = 'qwen3_5'
    _configure_cuda_recurrent_attention(model)
    assert layers[0].recurrent_gated_delta_rule is _cached_qwen35_delta_rule
    assert layers[1].recurrent_gated_delta_rule is fallback
    assert layers[2].recurrent_gated_delta_rule is fused
    _configure_cuda_recurrent_attention(SimpleNamespace())
