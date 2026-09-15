"""Harness-1 已从运行时剥离（DESIGN.md 第 7 节）：不给 harness_root 也能配置和运行。"""

import warnings

import pytest

from foren_searcher.demo import capability_report


def test_capability_report_does_not_require_harness1(tmp_path, monkeypatch):
    model = tmp_path / "Qwen3-14B-bnb-4bit"
    model.mkdir()
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (model / filename).write_text("{}")
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    monkeypatch.delenv("FOREN_HARNESS_ROOT", raising=False)

    report = capability_report(model_root=tmp_path, search="serpapi", harness_root=None, model_backend="local",
                               student_model="Qwen3-14B-bnb-4bit")

    assert report["configured_for_text_demo"] is True
    assert report["harness1"] == {"configured": False, "required": False, "deprecated": True, "source_root": None}


@pytest.mark.asyncio
async def test_run_demo_defaults_to_workspace_harness_without_harness_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import foren_searcher.demo as demo
    from foren_searcher.hi.workspace import WorkspaceHarness
    from foren_searcher.inspect.policy import InspectPolicy

    class Service:
        policy = InspectPolicy()
        async def aclose(self):
            pass

    class Policies(Service):
        gia_model = SimpleNamespace()
        hi_model = SimpleNamespace()
        gia_policy = None
        hi_policy = None
        async def preload(self):
            pass

    seen: dict[str, object] = {}

    class Recording:
        def __init__(self, *args, **kwargs):
            seen["workspace_harness_type"] = kwargs.get("workspace_harness_type")
            raise RuntimeError("stop before any model work")

    monkeypatch.delenv("FOREN_HARNESS_ROOT", raising=False)
    monkeypatch.setattr(demo, "capability_report", lambda **kwargs: {"configured_for_text_demo": True})
    monkeypatch.setattr(demo, "harness1_workspace_type", lambda root: pytest.fail("Harness-1 must not be loaded"))
    monkeypatch.setattr(demo, "_create_search", lambda *args: Service())
    monkeypatch.setattr(demo, "create_inspect_service", lambda **kwargs: Service())
    monkeypatch.setattr(demo, "create_demo_policies", lambda root, **kwargs: Policies())
    monkeypatch.setattr(demo, "InvestigationRuntime", Recording)
    with pytest.raises(demo.DemoRunError):
        await demo.run_demo(task="Check supplied sources", output_dir=tmp_path, model_backend="local")
    assert seen.get("workspace_harness_type") is WorkspaceHarness


def test_harness1_adapter_warns_when_used(tmp_path):
    from foren_searcher.harness1 import harness1_workspace_type

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            harness1_workspace_type(tmp_path)  # 目录里没有 Harness-1，加载会失败，但弃用警告先发出
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
