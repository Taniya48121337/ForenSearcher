from __future__ import annotations

from audit_recovery_production_readiness_v12 import build_report


def test_v12_keeps_prepared_blind_packet_distinct_from_completed_review() -> None:
    report = build_report()
    blind = report["independent_blind_delta_review"]
    assert blind["sampled_development_variants"] == 105
    assert blind["covered_development_cases_by_dataset"] == {
        "VCSL": 18,
        "VidDiffBench": 21,
        "VideoSham": 16,
    }
    assert sum(blind["intervention_families_covered_by_dataset"].values()) == 11
    assert blind["all_state_deltas_reconstructable"] is True
    assert blind["public_privacy_audit_passed"] is True
    assert blind["human_review_completed"] is False
    assert blind["missing_reviews"] == 105
    assert blind["blind_review_passed"] is False
    assert report["training_qualified"] is False
