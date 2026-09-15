"""Preserve the ranking of close output scores without converting model weights."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from foren_searcher.local_models import _configure_cuda_output_head


def test_cpu_and_custom_output_heads_keep_their_own_forward():
    for head in (torch.nn.Linear(8, 2, bias=False, dtype=torch.bfloat16), object(), None):
        original = getattr(head, "forward", None)
        _configure_cuda_output_head(SimpleNamespace(get_output_embeddings=lambda: head))
        assert getattr(head, "forward", None) == original
    _configure_cuda_output_head(object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output precision requires a GPU")
@pytest.mark.parametrize("shape", [(1, 8), (2, 1, 8), (2, 3, 8)])
def test_cuda_preserves_a_real_score_difference_lost_by_bfloat16_rounding(shape):
    head = torch.nn.Linear(8, 2, bias=False, dtype=torch.bfloat16, device="cuda").eval()
    with torch.no_grad():
        head.weight.fill_(1)
        head.weight[1, -1] = 1.0078125
    values = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        rounded = head(values)
        reference = values.double() @ head.weight.t().double()
        assert torch.all(rounded[..., 0] == rounded[..., 1])
        assert torch.all(reference[..., 1] > reference[..., 0])
        pointer = head.weight.data_ptr()
        _configure_cuda_output_head(SimpleNamespace(get_output_embeddings=lambda: head))
        scores = head(values)
    assert scores.shape == rounded.shape and scores.dtype == torch.float32
    assert torch.equal(scores.double(), reference)
    assert torch.all(scores.argmax(dim=-1) == 1)
    assert head.weight.dtype == torch.bfloat16 and head.weight.data_ptr() == pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output precision requires a GPU")
@pytest.mark.parametrize("exception", [TypeError, NotImplementedError, RuntimeError])
def test_unsupported_float32_output_leaves_the_native_head_available(monkeypatch, exception):
    head = torch.nn.Linear(8, 2, bias=False, dtype=torch.bfloat16, device="cuda").eval()
    original = head.forward

    def unsupported(*args, **kwargs):
        raise exception("unsupported output dtype")

    monkeypatch.setattr(torch, "mm", unsupported)
    _configure_cuda_output_head(SimpleNamespace(get_output_embeddings=lambda: head))
    assert head.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output precision requires a GPU")
def test_device_hooks_custom_forwards_and_other_head_formats_are_preserved():
    heads = [torch.nn.Linear(8, 2, bias=False, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    heads[0]._hf_hook = object()
    heads[1].forward = lambda values: values
    heads += [torch.nn.Linear(8, 2, bias=True, dtype=torch.bfloat16, device="cuda"),
              torch.nn.Linear(8, 2, bias=False, dtype=torch.float32, device="cuda")]
    for head in heads:
        original = head.forward
        _configure_cuda_output_head(SimpleNamespace(get_output_embeddings=lambda: head))
        assert head.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA output precision requires a GPU")
def test_reconfiguration_is_idempotent_and_training_keeps_native_gradients():
    head = torch.nn.Linear(8, 2, bias=False, dtype=torch.bfloat16, device="cuda").eval()
    model = SimpleNamespace(get_output_embeddings=lambda: head)
    _configure_cuda_output_head(model)
    configured = head.forward
    _configure_cuda_output_head(model)
    assert head.forward is configured
    head.train()
    values = torch.ones((2, 8), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    output = head(values)
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert head.weight.grad is not None and values.grad is not None
