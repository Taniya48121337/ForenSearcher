from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_policy_choice_surface import audit  # noqa: E402


def test_all_future_default_measurement_surfaces_require_a_real_policy_choice() -> None:
    report = audit()
    assert report["all_passed"] is True
    assert {
        dataset: result["semantic_choice_count"]
        for dataset, result in report["datasets"].items()
    } == {"VideoSham": 3, "VCSL": 8, "VidDiffBench": 3}
    assert all(
        not result["router_answer_fields_visible"]
        for result in report["datasets"].values()
    )
