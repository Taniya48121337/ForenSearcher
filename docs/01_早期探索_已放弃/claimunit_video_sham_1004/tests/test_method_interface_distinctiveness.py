from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_method_interface_distinctiveness import audit  # noqa: E402


def test_every_current_evidence_tool_has_a_distinguishable_public_mechanism() -> None:
    report = audit()
    assert report["all_passed"] is True
    assert {
        dataset: row["tool_count"]
        for dataset, row in report["datasets"].items()
    } == {"VideoSham": 3, "VCSL": 8, "VidDiffBench": 5}
