from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from test_utqr_v7_causal import high_quality_packet  # noqa: E402
from utqr_v8_causal import evaluate_packet  # noqa: E402


def _complete_recovery_packet() -> dict:
    packet = copy.deepcopy(high_quality_packet())
    repair = packet["probes"]["unannounced_repair"]
    repair.update(
        {
            "challenge_executed": True,
            "challenged_evidence_ids_valid": True,
            "defect_visibly_grounded": True,
            "requested_check_method_fit": True,
            "fresh_check_executed": True,
            "fresh_evidence_consumed": True,
            "clean_control_unnecessary_challenge": False,
        }
    )
    return packet


def test_complete_recovery_chain_can_pass_utqr_v8() -> None:
    report = evaluate_packet(_complete_recovery_packet())
    assert report["rubric_id"] == "UTQR-8"
    assert report["accepted_for_sft"] is True
    assert report["diagnosis"]["unannounced_causal_recovery"]["passed"] is True


def test_challenge_without_fresh_check_fails_h5() -> None:
    packet = _complete_recovery_packet()
    repair = packet["probes"]["unannounced_repair"]
    repair["fresh_check_executed"] = False
    repair["fresh_evidence_consumed"] = False
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H5" in report["failed_gates"]
    assert "fresh_check_executed" in report["diagnosis"]["unannounced_causal_recovery"]["missing_or_failed"]


def test_clean_control_challenge_is_rejected() -> None:
    packet = _complete_recovery_packet()
    packet["probes"]["unannounced_repair"][
        "clean_control_unnecessary_challenge"
    ] = True
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H5" in report["failed_gates"]
