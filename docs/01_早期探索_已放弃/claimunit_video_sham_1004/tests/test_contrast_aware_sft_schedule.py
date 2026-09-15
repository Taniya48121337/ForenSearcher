from __future__ import annotations

import json

from build_contrast_aware_sft_schedule import (
    DEFAULT_OUTPUT,
    build_epoch_schedule,
    materialize_schedule,
)


def _row(index: int, target: str, weight: float) -> dict:
    return {
        "messages": [{"role": "assistant", "content": f"reasoning {index}"}],
        "tools": [{"type": "function", "function": {"name": target}}],
        "metadata": {
            "row_id": f"row-{index}",
            "dataset": "D",
            "case_id": f"case-{index % 2}",
            "causal_recovery_stage": "challenge",
            "target_tool": target,
            "sft_sampling_weight": weight,
        },
    }


def test_epoch_schedule_colocates_different_targets_without_changing_loss_mass() -> None:
    rows = [
        _row(1, "challenge_evidence", 1.5),
        _row(2, "stop", 0.5),
        _row(3, "challenge_evidence", 2.0),
        _row(4, "stop", 1.0),
    ]
    groups = [
        {
            "group_id": "g1",
            "family": "same_menu_different_tool",
            "row_ids": ["row-1", "row-2", "row-3", "row-4"],
            "targets": [
                "challenge_evidence",
                "stop",
                "challenge_evidence",
                "stop",
            ],
        }
    ]
    batches, audit = build_epoch_schedule(
        rows, groups, epoch=1, batch_size=2, seed=7
    )
    assert audit["passed"] is True
    assert audit["contrast_groups_co_located"] == 1
    assert audit["contrast_groups_with_different_targets"] == 1
    assert audit["row_loss_mass_preserved"] is True
    assert audit["every_source_row_has_one_base_occurrence"] is True
    materialized = materialize_schedule(rows, batches)
    assert all("trainer_schedule" in row["metadata"] for row in materialized)


def test_real_schedule_is_loss_preserving_and_materialization_is_semantic_noop() -> None:
    audit = json.loads(
        (DEFAULT_OUTPUT / "schedule_audit.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (DEFAULT_OUTPUT / "materialization_smoke_audit.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_rows"] == 1098
    assert manifest["epochs"] == 8
    assert manifest["contrast_groups"] == 135
    assert audit["all_epochs_passed"] is True
    assert audit["all_epochs_preserve_row_loss_mass"] is True
    assert audit["all_groups_co_located_every_epoch"] is True
    assert audit["all_group_anchors_have_different_targets"] is True
    assert audit["batches_per_epoch"] == [171] * 8
    assert audit["occurrences_per_epoch"] == [1368] * 8
    assert smoke["messages_unchanged"] is True
    assert smoke["tools_unchanged"] is True
    assert smoke["metadata_unchanged_except_schedule"] is True
    assert smoke["negative_or_calibration_rows_included"] is False
    assert smoke["passed"] is True
