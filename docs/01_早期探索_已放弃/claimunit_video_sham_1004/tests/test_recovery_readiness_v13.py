from __future__ import annotations

from audit_recovery_production_readiness_v13 import build_report


def test_v13_keeps_sentinels_out_of_quality_and_sft_claims() -> None:
    report = build_report()
    blind = report["calibrated_independent_blind_review"]
    assert blind["total_public_items"] == 129
    assert blind["primary_quality_items"] == 105
    assert blind["calibration_clean_items"] == 8
    assert blind["calibration_negative_items"] == 16
    assert blind["calibration_mutation_families"] == 8
    assert blind["sentinels_included_in_quality_estimate"] is False
    assert blind["negative_sentinels_included_in_sft"] is False
    assert blind["sentinel_construction_audit_passed"] is True
    assert blind["all_forms_complete"] is False
    assert blind["reviewers_reliable"] is False
    assert blind["blind_review_passed"] is False
    assert report["training_qualified"] is False
