from __future__ import annotations

from audit_recovery_production_readiness_v14 import build_report


def test_v14_reports_trainer_schedule_without_claiming_training_gain() -> None:
    report = build_report()
    schedule = report["contrast_aware_trainer_schedule"]
    assert schedule["source_rows"] == 1098
    assert schedule["epochs"] == 8
    assert schedule["schedule_shards"] == 48
    assert schedule["contrast_groups"] == 135
    assert schedule["all_source_rows_have_base_exposure_every_epoch"] is True
    assert schedule["all_groups_co_located_every_epoch"] is True
    assert schedule["all_group_anchors_have_different_targets"] is True
    assert schedule["all_epochs_preserve_row_loss_mass"] is True
    assert schedule["case_loss_mass_identical_across_epochs"] is True
    assert schedule["stage_loss_mass_identical_across_epochs"] is True
    assert schedule["tool_loss_mass_identical_across_epochs"] is True
    assert schedule["materialized_messages_unchanged"] is True
    assert schedule["materialized_tools_unchanged"] is True
    assert schedule["calibration_rows_included"] is False
    assert schedule["training_executed"] is False
    assert report["training_qualified"] is False
