"""两个真实本地 Qwen 模型的可选端到端冒烟测试。

仅在显式开启环境变量时运行，避免日常单元测试加载数 GB 权重。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from foren_searcher.local_smoke import run_local_model_smoke_suite


@pytest.mark.live
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_two_local_models_complete_the_controlled_system_smoke(tmp_path: Path) -> None:
    """验证 two local models complete the controlled system smoke 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    if os.environ.get("RUN_LOCAL_MODEL_SMOKE") != "1":
        pytest.skip("set RUN_LOCAL_MODEL_SMOKE=1 to run local model inference")
    model_root = Path(os.environ.get("FOREN_LOCAL_MODEL_ROOT", ".models"))
    if not (model_root / "Qwen3-0.6B").is_dir() or not (
        model_root / "Qwen3-1.7B"
    ).is_dir():
        pytest.fail("download both local models before running this smoke test")
    result = await run_local_model_smoke_suite(
        model_root=model_root, output_dir=tmp_path / "investigation"
    )
    assert result["status"] == "passed"
    scenarios = {item["scenario"]: item for item in result["scenarios"]}
    assert {"INSPECT", "SEARCH", "READ_RECORD"} <= set(
        scenarios["multimodal-tool-chain"]["actions"]
    )
    assert {"SEARCH", "READ_RECORD"} <= set(
        scenarios["text-search-read"]["actions"]
    )
    assert "SEARCH" in scenarios["natural-language-search"]["actions"]
    assert all(item["cited_evidence_ids"] for item in scenarios.values())
    assert result["model_metrics"]["gia"]["completed_requests"] >= 4
    assert result["model_metrics"]["hi"]["completed_requests"] >= 7
