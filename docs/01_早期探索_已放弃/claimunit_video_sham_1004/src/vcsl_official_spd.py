"""Lazy, label-blind adapter for VCSL's released ISC SPD localizer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline import ROOT


DEFAULT_VCSL_ROOT = ROOT.parent / "external" / "VCSL"
DEFAULT_SPD_MODEL = (
    ROOT / "outputs/student_external_contract_v1/vcsl_spd_isc_download"
)
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.3

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _official_modules(vcsl_root: Path) -> tuple[Any, Any]:
    for path in (vcsl_root, vcsl_root / "vcsl" / "yolov5"):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    from vcsl.vta import attempt_load, spd

    return attempt_load, spd


def load_official_spd_model(
    *,
    vcsl_root: Path = DEFAULT_VCSL_ROOT,
    model_path: Path = DEFAULT_SPD_MODEL,
    device: str = "cpu",
) -> Any:
    """Load the trusted official full-model checkpoint once per process."""

    cache_key = (str(model_path.resolve()), device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    if not model_path.is_file():
        raise FileNotFoundError(f"official VCSL SPD model is unavailable: {model_path}")

    attempt_load, _spd = _official_modules(vcsl_root)
    original_load = torch.load

    def trusted_full_model_load(*args: Any, **kwargs: Any) -> Any:
        # PyTorch 2.6 defaults to weights_only=True. The released VCSL artifact is
        # a full YOLO checkpoint, so this narrowly scoped trusted load is required.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_full_model_load
    try:
        model = attempt_load(str(model_path), map_location=device)
    finally:
        torch.load = original_load
    _MODEL_CACHE[cache_key] = model
    return model


def official_spd_localize(
    similarity: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    device: str = "cpu",
    vcsl_root: Path = DEFAULT_VCSL_ROOT,
    model_path: Path = DEFAULT_SPD_MODEL,
) -> list[list[int]]:
    """Infer copied segments from a similarity map without reading labels."""

    model = load_official_spd_model(
        vcsl_root=vcsl_root,
        model_path=model_path,
        device=device,
    )
    _attempt_load, spd = _official_modules(vcsl_root)
    boxes = spd(
        similarity,
        model,
        confidence,
        device,
        iou_thresh=iou_threshold,
    )
    return [[int(value) for value in box] for box in boxes]

