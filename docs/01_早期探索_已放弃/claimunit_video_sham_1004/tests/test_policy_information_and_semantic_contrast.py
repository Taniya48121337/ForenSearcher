from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_recovery_policy_information import audit  # noqa: E402
from audit_recovery_semantic_contrast_integrity import audit as audit_contrasts  # noqa: E402
from build_recovery_semantic_contrast_view import build  # noqa: E402


BASE = ROOT / "outputs/student_external_contract_v1"
BATCH = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike"
MIX = BASE / "recovery_sft_training_mix_v4_edgeaware55"
PORTABLE_CONTRAST = BASE / "recovery_sft_semantic_contrast_portable_v1_edgeaware55"
CONTRAST_V5 = BASE / "recovery_sft_semantic_contrast_v5_target_collision55"
PORTABLE_V6 = BASE / "recovery_sft_semantic_contrast_portable_v6_target_collision55"
CONTRAST_V6 = BASE / "recovery_sft_semantic_contrast_v6_neighborhood55"
PORTABLE_V7 = BASE / "recovery_sft_semantic_contrast_portable_v7_neighborhood55"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_policy_information_audit_separates_tool_and_argument_determinism() -> None:
    rows = _jsonl(BATCH / "recovery_sft_candidate.jsonl")
    weighted = _jsonl(MIX / "recovery_sft_weighted.jsonl")
    report, records = audit(rows, weighted)
    assert report["overall"]["turns"] == 415
    assert report["overall"]["counts"]["arguments_unique_given_tool"] == 73
    assert report["overall"]["counts"]["whole_call_unique_from_control_plane"] == 0
    assert report["overall"]["counts"]["exact_compiled_target_visible"] == 0
    assert report["overall"]["counts"]["choice_bearing_supervision"] == 415
    assert report["weighted_training_mass"]["share_by_tier"][
        "semantic_policy_core"
    ] > 0.5
    terminal = [row for row in records if row["stage"] == "terminal_verification"]
    assert len(terminal) == 55
    assert all(row["arguments_unique_given_tool"] for row in terminal)
    assert all(not row["whole_call_unique_from_control_plane"] for row in terminal)


def test_semantic_contrast_view_preserves_targets_and_builds_structural_collisions() -> None:
    rows = _jsonl(MIX / "recovery_sft_weighted.jsonl")
    output, result = build(rows)
    report = result["report"]
    assert report["source_rows"] == report["output_rows"] == 415
    assert report["contrast_groups"] == 82
    assert report["same_menu_stop_vs_challenge_episodes"] == 55
    assert report["same_menu_retain_vs_reject_episodes"] == 12
    assert report["structure_matched_target_collision_groups"] == 15
    assert report["structure_matched_collision_implied_cross_target_pairs"] == 1413
    assert report["contrast_covered_rows"] == 330
    assert report["uses_teacher_targets_for_grouping"] is True
    assert report["target_grouping_visible_to_student"] is False
    assert report["construction_failures"] == []
    assert report["messages_unchanged"] is True
    assert report["tools_unchanged"] is True
    assert report["sampling_weights_unchanged"] is True
    assert all(
        left["messages"] == right["messages"]
        for left, right in zip(rows, output)
    )
    integrity = audit_contrasts(output, result["groups"])
    assert integrity["passed"] is True
    assert integrity["covered_row_fraction"] > 0.79
    corrupted_groups = copy.deepcopy(result["groups"])
    collision = next(
        group
        for group in corrupted_groups
        if group["family"] == "same_control_surface_different_semantic_target"
    )
    collision["abstract_targets"][0] = {"impossible_mutation": True}
    assert audit_contrasts(output, corrupted_groups)["passed"] is False


def test_portable_contrast_view_remains_clean_and_not_training_qualified() -> None:
    portability = json.loads(
        (PORTABLE_CONTRAST / "portability_audit.json").read_text(encoding="utf-8")
    )
    contrast = json.loads(
        (PORTABLE_CONTRAST / "semantic_contrast_report.json").read_text(
            encoding="utf-8"
        )
    )
    readiness = json.loads(
        (
            BASE
            / "production_readiness_audit_v4_semantic_contrast55"
            / "readiness_report.json"
        ).read_text(encoding="utf-8")
    )
    assert portability["all_semantic_equivalence_checks_passed"] is True
    assert portability["all_forbidden_identifier_hits_zero"] is True
    assert contrast["contrast_groups"] == 67
    assert readiness["semantic_contrast_view"]["strict_reasoning_passed"] is True
    assert readiness["training_qualified"] is False


def test_utqr11_contrast_identifiability_package_is_integrity_checked() -> None:
    contrast = json.loads(
        (CONTRAST_V5 / "contrast_report.json").read_text(encoding="utf-8")
    )
    canonical = json.loads(
        (CONTRAST_V5 / "contrast_integrity_audit.json").read_text(encoding="utf-8")
    )
    portable = json.loads(
        (PORTABLE_V6 / "contrast_integrity_audit.json").read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (
            BASE
            / "production_readiness_audit_v8_utqr11_contrast55"
            / "readiness_report.json"
        ).read_text(encoding="utf-8")
    )
    assert contrast["contrast_groups"] == 179
    assert contrast["structure_matched_target_collision_groups"] == 112
    assert canonical["covered_row_fraction"] == 0.787081
    assert canonical["passed"] is True
    assert portable["passed"] is True
    assert readiness["contrastive_identifiability"]["passed"] is True
    assert readiness["training_qualified"] is False


def test_nary_neighborhood_package_covers_all_legitimate_collision_members() -> None:
    contrast = json.loads(
        (CONTRAST_V6 / "contrast_report.json").read_text(encoding="utf-8")
    )
    canonical = json.loads(
        (CONTRAST_V6 / "contrast_integrity_audit.json").read_text(encoding="utf-8")
    )
    portable = json.loads(
        (PORTABLE_V7 / "contrast_integrity_audit.json").read_text(encoding="utf-8")
    )
    trainer = json.loads(
        (PORTABLE_V7 / "trainer_contract.json").read_text(encoding="utf-8")
    )
    readiness = json.loads(
        (
            BASE
            / "production_readiness_audit_v9_utqr11_neighborhood55"
            / "readiness_report.json"
        ).read_text(encoding="utf-8")
    )
    assert contrast["contrast_groups"] == 82
    assert contrast["structure_matched_target_collision_groups"] == 15
    assert contrast["structure_matched_collision_implied_cross_target_pairs"] == 1429
    assert canonical["covered_rows"] == 333
    assert canonical["covered_row_fraction"] == 0.796651
    assert canonical["passed"] is True
    assert portable["passed"] is True
    assert trainer["load_entire_group_required"] is False
    assert trainer["drop_unpaired_rows"] is False
    assert readiness["contrastive_identifiability"]["implied_cross_target_pairs"] == 1429
    assert readiness["contrastive_identifiability"]["passed"] is True
    assert readiness["training_qualified"] is False
