import asyncio
import json
from types import SimpleNamespace

import pytest

from foren_searcher import local_smoke


@pytest.mark.asyncio
async def test_suite_records_failed_case_and_runs_remaining_cases(tmp_path, monkeypatch):
    calls = []
    closed = []
    async def preload():
        pass
    async def close():
        closed.append(True)
    model = SimpleNamespace(metrics=lambda: SimpleNamespace(to_dict=lambda: {"completed_requests": 3}))
    bundle = SimpleNamespace(preload=preload, aclose=close, gia_model=model, hi_model=model)
    monkeypatch.setattr(local_smoke, "create_qwen_local_policies", lambda *a, **kw: bundle)
    async def case(**kw):
        name = kw["output_dir"].name
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("missing SEARCH")
        return {"status": "passed", "scenario": name}
    monkeypatch.setattr(local_smoke, "_run_controlled_case", case)
    result = await local_smoke.run_local_model_smoke_suite(model_root="unused", output_dir=tmp_path)
    assert calls == ["multimodal-tool-chain", "text-search-read", "natural-language-search"]
    assert result["status"] == "failed"
    assert [item["status"] for item in result["scenarios"]] == ["failed", "passed", "passed"]
    assert result["scenarios"][0]["error"] == {"type": "RuntimeError", "message": "missing SEARCH"}
    assert result["model_metrics"]["hi"]["completed_requests"] == 3
    assert closed == [True]


@pytest.mark.asyncio
async def test_suite_propagates_cancellation_and_closes_models(tmp_path, monkeypatch):
    closed = []
    async def preload():
        pass
    async def close():
        closed.append(True)
    monkeypatch.setattr(local_smoke, "create_qwen_local_policies", lambda *a, **kw:
        SimpleNamespace(preload=preload, aclose=close))
    async def case(**kw):
        raise asyncio.CancelledError()
    monkeypatch.setattr(local_smoke, "_run_controlled_case", case)
    with pytest.raises(asyncio.CancelledError):
        await local_smoke.run_local_model_smoke_suite(model_root="unused", output_dir=tmp_path)
    assert closed == [True]


def test_cli_saves_failed_summary_and_exits_nonzero(tmp_path, monkeypatch):
    async def suite(**kw):
        return {"status": "failed", "output_dir": str(tmp_path), "scenarios": []}
    monkeypatch.setattr(local_smoke, "run_local_model_smoke_suite", suite)
    monkeypatch.setattr("sys.argv", ["local-smoke", "--suite", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        local_smoke.main()
    assert exc.value.code == 1
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["status"] == "failed"
