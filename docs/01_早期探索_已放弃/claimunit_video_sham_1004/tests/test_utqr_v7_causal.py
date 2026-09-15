from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utqr_v7_causal import (  # noqa: E402
    evaluate_batch,
    evaluate_packet,
    packets_from_viddiff_falsification,
)


def _judge(judge_id: str) -> dict:
    dimensions = ["prospective_reasoning", "belief_update", "method_fit", "naturalness"]
    return {
        "judge_id": judge_id,
        "blind_to_forbidden_fields": True,
        "dimension_scores": {name: 5 for name in dimensions},
        "dimension_evidence": {
            name: "The visible decision note cites the observation and updates the next action prospectively."
            for name in dimensions
        },
    }


def high_quality_packet(dataset: str = "VideoSham", case_id: str = "case-001") -> dict:
    complete = {"executed": True, "label_blind": True}
    return {
        "packet_id": f"PACKET-{dataset}-{case_id}",
        "dataset": dataset,
        "case_id": case_id,
        "mode": "clean_plus_unannounced_repair",
        "base": {
            "terminal_success": True,
            "hidden_outcome_correct": True,
            "all_actions_executed": True,
            "binding_accuracy": 1.0,
            "provenance_closed": True,
            "all_claim_fields_supported": True,
            "evidence_task_identity_match": True,
            "evidence_scope_match": True,
            "donor_identity_copied": False,
        },
        "control_plane": {
            "compiled_semantic_actions_visible": False,
            "singleton_target_values_visible": False,
            "repair_recipe_visible": False,
            "full_target_call_materializable": False,
            "semantic_measurement_turns": 1,
            "nontrivial_semantic_choice_turns": 1,
            "branch_search_expected_turns": 1,
            "branch_search_covered_turns": 1,
            "branch_replay_mismatch_turns": 0,
            "branch_selection_ambiguous_turns": 0,
            "prospective_method_fit_expected_turns": 1,
            "prospective_method_fit_passed_turns": 1,
            "counterfactual_tool_swap_expected_turns": 1,
            "counterfactual_tool_swap_passed_turns": 1,
            "preaction_frozen_selection_turns": 1,
            "hindsight_selected_turns": 0,
            "selected_branch_inadmissible_turns": 0,
            "proposal_retry_outcome_feedback_turns": 0,
            "categorical_router_answer_visible": False,
        },
        "probes": {
            "semantic_mask": {
                **complete,
                "requested_measurement": True,
                "rational_abstention": False,
                "unsupported_claim_repeated": False,
            },
            "schema_compatible_swap": {
                **complete,
                "dependent_fields_changed": True,
                "rational_abstention": False,
                "requested_verification": False,
                "claim_equal_to_base": False,
                "task_identity_preserved": True,
                "invariant_fields_preserved": True,
                "donor_identity_copied": False,
            },
            "identity_mismatch": {
                **complete,
                "mismatch_detected": True,
                "rational_abstention": True,
                "requested_verification": False,
                "donor_identity_copied": False,
            },
            "ab_role_swap": {
                **complete,
                "direction_or_relation_transformed": True,
                "intervals_or_roles_transformed": True,
                "invariant_fields_preserved": True,
            },
            "reduced_control": {
                **complete,
                "semantic_decision_correct": True,
                "action_legal": True,
                "binding_or_terminal_success": True,
            },
            "structure_only": {**complete, "safe_success": False},
            "unannounced_repair": {
                **complete,
                "detected_before_stop": True,
                "targeted_action_executed": True,
                "localized_reopen": True,
                "overinvalidated_fields": [],
                "explicit_repair_hint_visible": False,
                "stopped_without_targeted_action": False,
                "correct_behavior": "repair",
                "terminal_success_raw": True,
                "rejection_count": 0,
            },
            "step_intervention": {**complete, "fixed_suffix_replay_passed": True},
        },
        "step_intervention": {
            "all_exported_steps_causally_necessary": True,
            "all_external_observations_consumed": True,
            "unnecessary_tool_calls": 0,
            "repeated_identical_calls": 0,
        },
        "sft": {
            "assistant_action_only": True,
            "target_was_executed": True,
            "oracle_or_private_locator_visible": False,
            "authoritative_pre_action_state": True,
            "native_tool_schema_valid": True,
            "control_only_training_fraction": 0.10,
        },
        "blind_judgments": [_judge("judge-a"), _judge("judge-b")],
    }


def test_complete_causal_neighborhood_can_pass() -> None:
    report = evaluate_packet(high_quality_packet())
    assert report["accepted_for_sft"] is True
    assert report["total_score"] == 100.0
    assert report["failed_gates"] == []


def test_generic_preaction_prose_fails_control_autonomy_even_if_branch_is_unique() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["prospective_method_fit_passed_turns"] = 0
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_tool_swappable_preaction_rationale_fails_control_autonomy() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["counterfactual_tool_swap_passed_turns"] = 0
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_hindsight_selected_branch_fails_even_without_oracle() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["hindsight_selected_turns"] = 1
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_failed_preview_feedback_cannot_guide_same_checkpoint_retry() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["proposal_retry_outcome_feedback_turns"] = 1
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_single_successful_trace_cannot_pass_without_counterfactual_probes() -> None:
    packet = high_quality_packet()
    packet["probes"].pop("semantic_mask")
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H3" in report["failed_gates"]
    assert "semantic_mask" in report["diagnosis"]["missing_or_invalid_probes"]


def test_donor_copy_fails_even_when_base_answer_is_correct() -> None:
    packet = high_quality_packet()
    swap = packet["probes"]["schema_compatible_swap"]
    swap["task_identity_preserved"] = False
    swap["invariant_fields_preserved"] = False
    swap["donor_identity_copied"] = True
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H3" in report["failed_gates"]
    c4 = next(item for item in report["criteria"] if item["id"] == "C4")
    assert c4["normalized_score"] < 0.85


def test_compiled_control_answer_fails_autonomy_gate() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["compiled_semantic_actions_visible"] = True
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_singleton_semantic_measurement_surface_fails_existing_autonomy_gate() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["nontrivial_semantic_choice_turns"] = 0
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]
    assert report["diagnosis"]["policy_choice_surface"]["passed"] is False


def test_collapsed_branch_proposals_fail_existing_autonomy_gate() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["branch_search_covered_turns"] = 0
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_branch_preview_replay_mismatch_fails_existing_autonomy_gate() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["branch_replay_mismatch_turns"] = 1
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_ambiguous_branch_selection_fails_existing_autonomy_gate() -> None:
    packet = high_quality_packet()
    packet["control_plane"]["branch_selection_ambiguous_turns"] = 1
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H4" in report["failed_gates"]


def test_explicitly_announced_repair_is_not_accepted_as_detection() -> None:
    packet = high_quality_packet()
    repair = packet["probes"]["unannounced_repair"]
    repair["explicit_repair_hint_visible"] = True
    repair["detected_before_stop"] = False
    report = evaluate_packet(packet)
    assert report["accepted_for_sft"] is False
    assert "H5" in report["failed_gates"]


def test_historical_viddiff_four_arm_run_is_diagnostic_not_gold() -> None:
    source = (
        ROOT
        / "outputs"
        / "student_external_contract_v1"
        / "control_plane_falsification_v1"
        / "arms"
        / "llm_viddiff30_reachable13_four_arm_v1"
        / "report.json"
    )
    packets = packets_from_viddiff_falsification(json.loads(source.read_text(encoding="utf-8")))
    reports = [evaluate_packet(packet) for packet in packets]
    assert len(reports) == 13
    assert sum(report["accepted_for_sft"] for report in reports) == 0
    assert all("H3" in report["failed_gates"] for report in reports)
    assert all("H4" in report["failed_gates"] for report in reports)


def test_mass_production_requires_all_three_datasets_and_batch_evidence() -> None:
    packets = []
    metadata = {"datasets": {}}
    for dataset in ("VideoSham", "VCSL", "VidDiffBench"):
        packets.extend(high_quality_packet(dataset, f"case-{index:02d}") for index in range(30))
        metadata["datasets"][dataset] = {
            "unconditioned_safe_coverage": 0.90,
            "pass_at_2": 0.90,
            "unannounced_repair_cases": 10,
            "entity_disjoint": True,
            "oracle_bounded_case_admission": False,
            "student_free_run_gain_verified": True,
            "control_only_fraction": 0.10,
        }
    report = evaluate_batch(packets, batch_metadata=metadata)
    assert report["mass_production_ready"] is True

    incomplete = evaluate_batch(
        [high_quality_packet("VideoSham", f"case-{index:02d}") for index in range(30)],
        batch_metadata={"datasets": {"VideoSham": metadata["datasets"]["VideoSham"]}},
    )
    assert incomplete["mass_production_ready"] is False
    assert incomplete["readiness_checks"]["datasets_present"] is False
