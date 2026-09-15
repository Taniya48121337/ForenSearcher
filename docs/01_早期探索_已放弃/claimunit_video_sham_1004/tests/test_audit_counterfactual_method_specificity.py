import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_counterfactual_method_specificity import audit


def test_real_tool_interfaces_reject_keyword_stuffing_but_accept_directional_warrant() -> None:
    report = audit()
    assert report["all_passed"] is True
    assert set(report["datasets"]) == {"VideoSham", "VCSL", "VidDiffBench"}
