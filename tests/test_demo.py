"""论文 Demo 入口的配置、媒体输入和输出格式测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from foren_searcher.demo import (
    DemoConfigurationError,
    _markdown_report,
    _media_description,
    _search_configuration,
    _demo_paths,
    capability_report,
)

from inspect_helpers import image_bytes


def test_existing_demo_output_cannot_return_an_old_investigation(tmp_path):
    original = tmp_path / "result.json"
    original.write_text('{"task":"previous task"}')
    with pytest.raises(DemoConfigurationError, match="empty"):
        _demo_paths(tmp_path)
    assert original.read_text() == '{"task":"previous task"}'


@pytest.mark.asyncio
async def test_failed_model_load_preserves_trace_and_closes_services(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import foren_searcher.demo as demo
    from foren_searcher.workspace import WorkspaceHarness
    from foren_searcher.inspect.policy import InspectPolicy
    class Service:
        closed = False
        policy = InspectPolicy()
        async def aclose(self):
            self.closed = True
    class Policies(Service):
        gia_model = SimpleNamespace()
        hi_model = SimpleNamespace()
        async def preload(self):
            raise RuntimeError("local model load failed")
    policies, search, inspect = Policies(), Service(), Service()
    monkeypatch.setattr(demo, "capability_report", lambda **kwargs: {"configured_for_text_demo":True})
    monkeypatch.setattr(demo, "harness1_workspace_type", lambda root: WorkspaceHarness)
    monkeypatch.setattr(demo, "_create_search", lambda *args: search)
    monkeypatch.setattr(demo, "create_inspect_service", lambda **kwargs: inspect)
    monkeypatch.setattr(demo, "create_demo_policies", lambda root, **kwargs: policies)
    with pytest.raises(demo.DemoRunError):
        await demo.run_demo(task="Check supplied sources", output_dir=tmp_path, harness_root=tmp_path, model_backend="local")
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["decision"] is None
    assert (tmp_path / "trajectory.jsonl").exists()
    assert "尚未形成最终结论" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert all(value.closed for value in (policies, search, inspect))


def test_demo_capability_report_never_exposes_search_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """预检只报告供应商和可用性，不回显 API 密钥。"""

    model = tmp_path / "Qwen3-14B-bnb-4bit"
    model.mkdir()
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (model / filename).write_text("{}")
    harness = tmp_path / "harness/runtime"
    harness.mkdir(parents=True)
    (harness / "tool_executor.py").write_text("")
    monkeypatch.setenv("SERPAPI_API_KEY", "top-secret-demo-key")
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    report = capability_report(model_root=tmp_path, search="serpapi", harness_root=tmp_path, model_backend="local",
                               student_model="Qwen3-14B-bnb-4bit")

    assert report["configured_for_text_demo"] is True
    assert report["search"]["provider"] == "serpapi"
    assert "top-secret-demo-key" not in str(report)


def test_demo_defaults_to_free_search_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未配置供应商时使用免费在线后端。"""

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    assert _search_configuration("auto") == ("ddgs", "")
    with pytest.raises(DemoConfigurationError, match="SEARXNG_BASE_URL"):
        _search_configuration("searxng")


def test_demo_media_input_is_registered_with_real_modality(tmp_path: Path) -> None:
    """本地图片保持真实 MIME、文件名和 image modality。"""

    path = tmp_path / "sample.png"
    path.write_bytes(image_bytes())

    value = _media_description(path, 2)

    assert value.source_ref == "I2"
    assert value.media_type == "image/png"
    assert value.modalities == ("image",)
    assert value.path == path.resolve()


def test_missing_media_has_a_configuration_error(tmp_path):
    with pytest.raises(DemoConfigurationError, match="media cannot be opened"):
        _media_description(tmp_path / "missing.pdf", 0)


def test_demo_markdown_contains_only_cited_evidence() -> None:
    """展示报告列出最终答案真正引用的 Evidence 和 provenance。"""

    rendered = _markdown_report(
        "核验日期",
        {
            "decision": {"decision": "FINISH", "answer": "日期为某日（E2）。"},
            "cited_evidence": [
                {
                    "evidence_id": "E2",
                    "label": "来源正文",
                    "source_ref": "R1",
                    "locator": "paragraph:3",
                    "acquired_by": "SEARCH/open/http",
                }
            ],
        },
    )

    assert "日期为某日（E2）" in rendered
    assert "E2 — 来源正文" in rendered
    assert "R1" in rendered
