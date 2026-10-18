"""10-19 sft0 六节裁决 4：训练加载时把 sample_weight 归一到均值 1（只改内存）；pack_r0 的 train_configs 不许关它。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_to_mean_one() -> None:
    pytest.importorskip("torch")
    sft = _load("sft_lora_t", "training/sft_lora.py")
    rows = [{"metadata": {"sample_weight": w}} for w in (2.0, 3.0, 1.0, 9.0)]
    info = sft.normalize_sample_weights(rows, [0, 1, 2], True)          # 第 4 行不在 usable 里：不参与、不改
    assert info["mean_before"] == 2.0 and info["mean_after"] == 1.0 and info["factor"] == 0.5
    assert [r["metadata"]["sample_weight"] for r in rows] == [1.0, 1.5, 0.5, 9.0]
    off = sft.normalize_sample_weights([{"metadata": {"sample_weight": 2.0}}], [0], False)
    assert off["factor"] == 1.0 and off["mean_after"] == 2.0


def test_config_default_on() -> None:
    pytest.importorskip("torch")
    sft = _load("sft_lora_t2", "training/sft_lora.py")
    for role in ("hi", "gia"):
        assert sft.TrainConfig.load(ROOT / "training" / "configs" / f"{role}_sft0_27b.json").normalize_sample_weight is True
