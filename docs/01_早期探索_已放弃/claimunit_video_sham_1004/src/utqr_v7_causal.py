"""UTQR-7: causal-neighborhood qualification for executable tool trajectories.

The evaluation unit is deliberately larger than one successful transcript.  A
packet contains an executed base trajectory and label-blind interventions that
test whether evidence, task identity, A/B roles, control affordances, and hidden
repair hints causally change the policy in the expected way.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pipeline import write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRIC = ROOT / "rubric" / "unified_trajectory_quality_rubric_v7.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rate(values: Iterable[Any]) -> float:
    items = list(values)
    return sum(bool(value) for value in items) / len(items) if items else 0.0


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _p10(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)] if ordered else 0.0


def _probe(packet: dict[str, Any], name: str) -> dict[str, Any]:
    value = (packet.get("probes") or {}).get(name)
    return value if isinstance(value, dict) else {}


def _probe_complete(probe: dict[str, Any]) -> bool:
    return bool(probe) and probe.get("executed") is True and probe.get("label_blind") is True


def _blind_judge_score(
    packet: dict[str, Any], rubric: dict[str, Any]
) -> tuple[float, bool, list[str], dict[str, Any]]:
    contract = rubric["blind_judge_contract"]
    dimensions = list(contract["dimensions"])
    rows = list(packet.get("blind_judgments") or [])
    valid: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: set[str] = set()
    for row in rows:
        judge_id = str(row.get("judge_id") or "")
        scores = row.get("dimension_scores") or {}
        if not judge_id or judge_id in seen:
            failures.append("missing_or_duplicate_judge_id")
            continue
        if set(scores) != set(dimensions):
            failures.append(f"{judge_id}:dimension_set_mismatch")
            continue
        if any(
            not isinstance(scores[name], int)
            or isinstance(scores[name], bool)
            or not 1 <= scores[name] <= 5
            for name in dimensions
        ):
            failures.append(f"{judge_id}:invalid_dimension_score")
            continue
        if row.get("blind_to_forbidden_fields") is not True:
            failures.append(f"{judge_id}:not_blind")
            continue
        evidence = row.get("dimension_evidence") or {}
        if any(len(str(evidence.get(name) or "").strip()) < 16 for name in dimensions):
            failures.append(f"{judge_id}:insufficient_evidence_note")
            continue
        seen.add(judge_id)
        valid.append(row)

    means = {
        name: _mean(row["dimension_scores"][name] for row in valid)
        for name in dimensions
    }
    ranges = {
        name: (
            max(row["dimension_scores"][name] for row in valid)
            - min(row["dimension_scores"][name] for row in valid)
            if valid
            else 5
        )
        for name in dimensions
    }
    enough = len(valid) >= int(contract["minimum_independent_judges"])
    means_ok = enough and all(
        value >= float(contract["minimum_dimension_mean"])
        for value in means.values()
    )
    ranges_ok = enough and all(
        value <= int(contract["maximum_unadjudicated_range"])
        for value in ranges.values()
    )
    passed = enough and means_ok and ranges_ok and not failures
    score = _mean(means.values()) / 5.0 if valid else 0.0
    return _bounded(score), passed, failures, {
        "valid_judges": len(valid),
        "dimension_means": means,
        "dimension_ranges": ranges,
    }


def evaluate_packet(
    packet: dict[str, Any], *, rubric: dict[str, Any] | None = None
) -> dict[str, Any]:
    rubric = rubric or _json(DEFAULT_RUBRIC)
    base = packet.get("base") or {}
    control = packet.get("control_plane") or {}
    steps = packet.get("step_intervention") or {}
    sft = packet.get("sft") or {}

    semantic_mask = _probe(packet, "semantic_mask")
    schema_swap = _probe(packet, "schema_compatible_swap")
    identity_mismatch = _probe(packet, "identity_mismatch")
    ab_swap = _probe(packet, "ab_role_swap")
    reduced = _probe(packet, "reduced_control")
    structure = _probe(packet, "structure_only")
    repair = _probe(packet, "unannounced_repair")
    step_probe = _probe(packet, "step_intervention")

    required = list(rubric["required_machine_probes"])
    probe_completeness = {
        name: _probe_complete(_probe(packet, name))
        for name in required
    }

    c1 = _mean(
        [
            float(base.get("terminal_success") is True),
            float(base.get("hidden_outcome_correct") is True),
            float(base.get("all_actions_executed") is True),
            float(base.get("binding_accuracy", 0.0)),
        ]
    )
    c2 = _rate(
        [
            base.get("provenance_closed") is True,
            base.get("all_claim_fields_supported") is True,
            base.get("evidence_task_identity_match") is True,
            base.get("evidence_scope_match") is True,
            base.get("donor_identity_copied") is False,
        ]
    )
    mask_safe = bool(
        semantic_mask.get("requested_measurement") is True
        or semantic_mask.get("rational_abstention") is True
    ) and semantic_mask.get("unsupported_claim_repeated") is not True
    swap_sensitive = bool(
        schema_swap.get("dependent_fields_changed") is True
        or schema_swap.get("rational_abstention") is True
        or schema_swap.get("requested_verification") is True
    )
    c3 = _rate(
        [
            _probe_complete(semantic_mask),
            mask_safe,
            _probe_complete(schema_swap),
            swap_sensitive,
            (
                schema_swap.get("intervened_evidence_bound") is False
                if "intervened_evidence_bound" in schema_swap
                else schema_swap.get("claim_equal_to_base") is False
            ),
        ]
    )
    identity_safe = bool(
        identity_mismatch.get("mismatch_detected") is True
        and (
            identity_mismatch.get("rational_abstention") is True
            or identity_mismatch.get("requested_verification") is True
        )
        and identity_mismatch.get("donor_identity_copied") is False
    )
    c4 = _rate(
        [
            schema_swap.get("task_identity_preserved") is True,
            schema_swap.get("invariant_fields_preserved") is True,
            schema_swap.get("donor_identity_copied") is False,
            identity_safe,
            ab_swap.get("direction_or_relation_transformed") is True,
            ab_swap.get("intervals_or_roles_transformed") is True,
            ab_swap.get("invariant_fields_preserved") is True,
        ]
    )
    no_answer_control = all(
        control.get(name) is False
        for name in (
            "compiled_semantic_actions_visible",
            "singleton_target_values_visible",
            "repair_recipe_visible",
            "full_target_call_materializable",
        )
    )
    semantic_measurement_turns = int(control.get("semantic_measurement_turns", 0))
    nontrivial_choice_turns = int(
        control.get("nontrivial_semantic_choice_turns", 0)
    )
    branch_search_expected_turns = int(
        control.get("branch_search_expected_turns", 0)
    )
    branch_search_covered_turns = int(
        control.get("branch_search_covered_turns", 0)
    )
    branch_replay_mismatch_turns = int(
        control.get("branch_replay_mismatch_turns", 0)
    )
    branch_selection_ambiguous_turns = int(
        control.get("branch_selection_ambiguous_turns", 0)
    )
    prospective_method_fit_expected_turns = int(
        control.get("prospective_method_fit_expected_turns", 0)
    )
    prospective_method_fit_passed_turns = int(
        control.get("prospective_method_fit_passed_turns", 0)
    )
    counterfactual_tool_swap_expected_turns = int(
        control.get("counterfactual_tool_swap_expected_turns", 0)
    )
    counterfactual_tool_swap_passed_turns = int(
        control.get("counterfactual_tool_swap_passed_turns", 0)
    )
    preaction_frozen_selection_turns = int(
        control.get("preaction_frozen_selection_turns", 0)
    )
    hindsight_selected_turns = int(control.get("hindsight_selected_turns", 0))
    selected_branch_inadmissible_turns = int(
        control.get("selected_branch_inadmissible_turns", 0)
    )
    proposal_retry_outcome_feedback_turns = int(
        control.get("proposal_retry_outcome_feedback_turns", 0)
    )
    counterfactual_tool_swap_safe = bool(
        counterfactual_tool_swap_expected_turns == 0
        or (
            counterfactual_tool_swap_expected_turns == semantic_measurement_turns
            and counterfactual_tool_swap_passed_turns
            == counterfactual_tool_swap_expected_turns
        )
    )
    choice_surface_safe = bool(
        semantic_measurement_turns > 0
        and nontrivial_choice_turns == semantic_measurement_turns
        and branch_search_expected_turns == semantic_measurement_turns
        and branch_search_covered_turns == branch_search_expected_turns
        and branch_replay_mismatch_turns == 0
        and branch_selection_ambiguous_turns == 0
        and prospective_method_fit_expected_turns == semantic_measurement_turns
        and prospective_method_fit_passed_turns
        == prospective_method_fit_expected_turns
        and counterfactual_tool_swap_safe
        and preaction_frozen_selection_turns == semantic_measurement_turns
        and hindsight_selected_turns == 0
        and selected_branch_inadmissible_turns == 0
        and proposal_retry_outcome_feedback_turns == 0
        and control.get("categorical_router_answer_visible") is False
    )
    c5 = _rate(
        [
            no_answer_control,
            choice_surface_safe,
            _probe_complete(reduced),
            reduced.get("semantic_decision_correct") is True,
            reduced.get("action_legal") is True,
            reduced.get("binding_or_terminal_success") is True,
            structure.get("answer_without_new_evidence") is not True,
        ]
    )
    c6 = _rate(
        [
            steps.get("all_exported_steps_causally_necessary") is True,
            steps.get("all_external_observations_consumed") is True,
            int(steps.get("unnecessary_tool_calls", 1)) == 0,
            int(steps.get("repeated_identical_calls", 1)) == 0,
            step_probe.get("fixed_suffix_replay_passed") is True,
        ]
    )
    repair_safe = bool(
        repair.get("detected_before_stop") is True
        and repair.get("targeted_action_executed") is True
        and repair.get("localized_reopen") is True
        and not list(repair.get("overinvalidated_fields") or [])
    )
    if repair.get("correct_behavior") == "abstain":
        repair_safe = repair_safe and repair.get("rational_abstention") is True
    else:
        repair_safe = (
            repair_safe
            and repair.get("terminal_success_raw") is True
            and int(repair.get("rejection_count", 1)) == 0
        )
    c7 = _rate(
        [
            _probe_complete(repair),
            repair_safe,
            repair.get("explicit_repair_hint_visible") is False,
            repair.get("stopped_without_targeted_action") is False,
        ]
    )
    c8, blind_passed, blind_failures, blind_stats = _blind_judge_score(packet, rubric)
    c9 = _rate(
        [
            sft.get("assistant_action_only") is True,
            sft.get("target_was_executed") is True,
            sft.get("oracle_or_private_locator_visible") is False,
            sft.get("authoritative_pre_action_state") is True,
            sft.get("native_tool_schema_valid") is True,
            float(sft.get("control_only_training_fraction", 1.0))
            <= float(rubric["supervision_policy"]["maximum_control_only_fraction_per_batch"]),
        ]
    )

    values = {
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "C6": c6,
        "C7": c7,
        "C8": c8,
        "C9": c9,
    }
    specs = {item["id"]: item for item in rubric["criteria"]}
    criteria = [
        {
            "id": criterion_id,
            "name": specs[criterion_id]["name"],
            "normalized_score": round(_bounded(score), 4),
            "earned": round(float(specs[criterion_id]["weight"]) * _bounded(score), 3),
            "max_points": specs[criterion_id]["weight"],
            "minimum": specs[criterion_id]["minimum"],
            "passed": score >= float(specs[criterion_id]["minimum"]),
        }
        for criterion_id, score in values.items()
    ]

    probe_actions_legal = all(
        int(_probe(packet, name).get("rejected_calls", 0)) == 0
        for name in (
            "semantic_mask",
            "schema_compatible_swap",
            "identity_mismatch",
            "ab_role_swap",
            "structure_only",
        )
    ) and int(repair.get("rejection_count", 0)) == 0
    hard_values = {
        "H1": c1 == 1.0,
        "H2": c2 == 1.0,
        "H3": (
            all(probe_completeness.values())
            and probe_actions_legal
            and c3 >= 0.85
            and c4 >= 0.85
        ),
        "H4": no_answer_control and choice_surface_safe and c5 >= 0.85,
        "H5": c7 >= 0.85,
        "H6": c6 >= 0.90 and c9 >= 0.90,
    }
    hard_gates = [
        {
            "id": item["id"],
            "name": item["name"],
            "passed": hard_values[item["id"]],
            "failure": None if hard_values[item["id"]] else item["failure"],
        }
        for item in rubric["hard_gates"]
    ]
    total = round(sum(item["earned"] for item in criteria), 3)
    acceptance = rubric["episode_acceptance"]
    critical_ok = all(
        values[item["id"]] >= float(acceptance["all_critical_criteria_minimum"])
        for item in rubric["criteria"]
        if item.get("critical")
    )
    any_ok = all(
        score >= float(acceptance["any_criterion_minimum"])
        for score in values.values()
    )
    accepted = bool(
        all(item["passed"] for item in hard_gates)
        and total >= float(acceptance["total_score_minimum"])
        and critical_ok
        and any_ok
        and blind_passed
    )
    return {
        "rubric_id": rubric["rubric_id"],
        "packet_id": packet.get("packet_id"),
        "dataset": packet.get("dataset"),
        "case_id": packet.get("case_id"),
        "mode": packet.get("mode"),
        "total_score": total,
        "accepted_for_sft": accepted,
        "hard_gates": hard_gates,
        "failed_gates": [item["id"] for item in hard_gates if not item["passed"]],
        "criteria": criteria,
        "probe_completeness": probe_completeness,
        "blind_judging": {
            "passed": blind_passed,
            "failures": blind_failures,
            **blind_stats,
        },
        "diagnosis": {
            "policy_choice_surface": {
                "semantic_measurement_turns": semantic_measurement_turns,
                "nontrivial_semantic_choice_turns": nontrivial_choice_turns,
                "branch_search_expected_turns": branch_search_expected_turns,
                "branch_search_covered_turns": branch_search_covered_turns,
                "branch_replay_mismatch_turns": branch_replay_mismatch_turns,
                "branch_selection_ambiguous_turns": branch_selection_ambiguous_turns,
                "prospective_method_fit_expected_turns": prospective_method_fit_expected_turns,
                "prospective_method_fit_passed_turns": prospective_method_fit_passed_turns,
                "counterfactual_tool_swap_expected_turns": counterfactual_tool_swap_expected_turns,
                "counterfactual_tool_swap_passed_turns": counterfactual_tool_swap_passed_turns,
                "counterfactual_tool_swap_safe": counterfactual_tool_swap_safe,
                "counterfactual_tool_swap_contract_status": (
                    "reported_and_passed"
                    if counterfactual_tool_swap_expected_turns > 0
                    and counterfactual_tool_swap_safe
                    else "reported_and_failed"
                    if counterfactual_tool_swap_expected_turns > 0
                    else "legacy_not_reported"
                ),
                "preaction_frozen_selection_turns": preaction_frozen_selection_turns,
                "hindsight_selected_turns": hindsight_selected_turns,
                "selected_branch_inadmissible_turns": selected_branch_inadmissible_turns,
                "proposal_retry_outcome_feedback_turns": proposal_retry_outcome_feedback_turns,
                "categorical_router_answer_visible": control.get(
                    "categorical_router_answer_visible"
                ),
                "passed": choice_surface_safe,
            },
            "priority_repairs": [
                item["id"]
                for item in sorted(criteria, key=lambda row: row["normalized_score"])
                if not item["passed"]
            ],
            "missing_or_invalid_probes": [
                name for name, passed in probe_completeness.items() if not passed
            ],
        },
    }


def evaluate_batch(
    packets: list[dict[str, Any]],
    *,
    batch_metadata: dict[str, Any] | None = None,
    rubric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rubric = rubric or _json(DEFAULT_RUBRIC)
    reports = [evaluate_packet(packet, rubric=rubric) for packet in packets]
    metadata = batch_metadata or {}
    spec = rubric["batch_readiness"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report.get("dataset"))].append(report)
    datasets: dict[str, Any] = {}
    for dataset, rows in sorted(grouped.items()):
        cases = {str(row.get("case_id")) for row in rows}
        accepted = [row for row in rows if row["accepted_for_sft"]]
        scores = [float(row["total_score"]) for row in rows]
        dataset_meta = (metadata.get("datasets") or {}).get(dataset, {})
        probe_rates = {
            name: _rate(row["probe_completeness"].get(name) for row in rows)
            for name in rubric["required_machine_probes"]
        }
        datasets[dataset] = {
            "packets": len(rows),
            "independent_cases": len(cases),
            "accepted_packets": len(accepted),
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
            "score_p10": _p10(scores),
            "unconditioned_safe_coverage": float(dataset_meta.get("unconditioned_safe_coverage", 0.0)),
            "pass_at_2": float(dataset_meta.get("pass_at_2", 0.0)),
            "unannounced_repair_cases": int(dataset_meta.get("unannounced_repair_cases", 0)),
            "entity_disjoint": dataset_meta.get("entity_disjoint") is True,
            "oracle_bounded_case_admission": dataset_meta.get("oracle_bounded_case_admission") is True,
            "student_free_run_gain_verified": dataset_meta.get("student_free_run_gain_verified") is True,
            "control_only_fraction": float(dataset_meta.get("control_only_fraction", 1.0)),
            "probe_family_rates": probe_rates,
        }

    checks: dict[str, bool] = {}
    checks["datasets_present"] = set(datasets) == set(spec["required_datasets"])
    checks["entity_disjoint_cases"] = bool(datasets) and all(
        row["entity_disjoint"]
        and row["independent_cases"] >= int(spec["minimum_entity_disjoint_cases_per_dataset"])
        for row in datasets.values()
    )
    checks["unconditioned_safe_coverage"] = bool(datasets) and all(
        row["unconditioned_safe_coverage"]
        >= float(spec["minimum_unconditioned_safe_coverage_per_dataset"])
        for row in datasets.values()
    )
    checks["packet_acceptance_rate"] = bool(datasets) and all(
        row["acceptance_rate"] >= float(spec["minimum_packet_acceptance_rate_per_dataset"])
        for row in datasets.values()
    )
    checks["score_p10"] = bool(datasets) and all(
        row["score_p10"] >= float(spec["minimum_score_p10_per_dataset"])
        for row in datasets.values()
    )
    checks["pass_at_2"] = bool(datasets) and all(
        row["pass_at_2"] >= float(spec["minimum_pass_at_2_per_dataset"])
        for row in datasets.values()
    )
    checks["probe_family_rate"] = bool(datasets) and all(
        all(
            rate >= float(spec["minimum_each_probe_family_pass_rate"])
            for rate in row["probe_family_rates"].values()
        )
        for row in datasets.values()
    )
    checks["unannounced_repair_coverage"] = bool(datasets) and all(
        row["unannounced_repair_cases"]
        >= int(spec["minimum_unannounced_repair_cases_per_dataset"])
        for row in datasets.values()
    )
    checks["control_only_fraction"] = bool(datasets) and all(
        row["control_only_fraction"] <= float(spec["maximum_control_only_fraction"])
        for row in datasets.values()
    )
    checks["no_oracle_bounded_admission"] = bool(datasets) and all(
        not row["oracle_bounded_case_admission"] for row in datasets.values()
    )
    checks["student_free_run_gain"] = bool(datasets) and all(
        row["student_free_run_gain_verified"] for row in datasets.values()
    )
    return {
        "rubric_id": rubric["rubric_id"],
        "packets": reports,
        "datasets": datasets,
        "readiness_checks": checks,
        "mass_production_ready": all(checks.values()),
        "pending": [name for name, passed in checks.items() if not passed],
    }


def packets_from_viddiff_falsification(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the completed four-arm report into diagnostic UTQR-7 packets.

    Missing probes remain missing on purpose; this adapter must not manufacture
    evidence that the historical run never collected.
    """

    scores = report.get("scores") or []
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        by_case[str(row.get("case_id"))][str(row.get("arm"))] = row
    packets: list[dict[str, Any]] = []
    for case_id, arms in sorted(by_case.items()):
        full = arms.get("full", {})
        evidence = arms.get("evidence_only", {})
        structure = arms.get("structure_only", {})
        shuffled = arms.get("shuffled_evidence", {})
        packets.append(
            {
                "packet_id": f"UTQR7-DIAG-{case_id}",
                "dataset": "VidDiffBench",
                "case_id": case_id,
                "mode": "clean",
                "base": {
                    "terminal_success": full.get("terminal_success") is True,
                    "hidden_outcome_correct": full.get("oracle_direction_correct") is True,
                    "all_actions_executed": int(full.get("rejection_count", 0)) == 0,
                    "binding_accuracy": float(full.get("binding_accuracy", 0.0)),
                    "provenance_closed": float(full.get("binding_accuracy", 0.0)) == 1.0,
                    "all_claim_fields_supported": float(full.get("binding_accuracy", 0.0)) == 1.0,
                    "evidence_task_identity_match": True,
                    "evidence_scope_match": True,
                    "donor_identity_copied": False,
                },
                "control_plane": {
                    "compiled_semantic_actions_visible": True,
                    "singleton_target_values_visible": True,
                    "repair_recipe_visible": False,
                    "full_target_call_materializable": True,
                },
                "probes": {
                    "reduced_control": {
                        "executed": True,
                        "label_blind": True,
                        "semantic_decision_correct": evidence.get("oracle_direction_correct") is True,
                        "action_legal": int(evidence.get("rejection_count", 0)) == 0,
                        "binding_or_terminal_success": evidence.get("terminal_success") is True,
                    },
                    "structure_only": {
                        "executed": True,
                        "label_blind": True,
                        "safe_success": structure.get("terminal_and_oracle_correct") is True,
                    },
                    "schema_compatible_swap": {
                        "executed": True,
                        "label_blind": True,
                        "claim_equal_to_base": shuffled.get("final_claim") == full.get("final_claim"),
                        "dependent_fields_changed": shuffled.get("final_claim") != full.get("final_claim"),
                        "task_identity_preserved": False,
                        "invariant_fields_preserved": False,
                        "donor_identity_copied": True,
                        "rational_abstention": False,
                    },
                },
                "step_intervention": {
                    "all_exported_steps_causally_necessary": False,
                    "all_external_observations_consumed": False,
                    "unnecessary_tool_calls": max(0, int(full.get("turn_count", 0)) - 6),
                    "repeated_identical_calls": 0,
                },
                "sft": {
                    "assistant_action_only": True,
                    "target_was_executed": True,
                    "oracle_or_private_locator_visible": False,
                    "authoritative_pre_action_state": True,
                    "native_tool_schema_valid": True,
                    "control_only_training_fraction": 0.60,
                },
                "blind_judgments": [],
            }
        )
    return packets


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packet", type=Path)
    source.add_argument("--packets", type=Path)
    source.add_argument("--viddiff-falsification-report", type=Path)
    parser.add_argument("--batch-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.packet:
        result = evaluate_packet(_json(args.packet))
    else:
        if args.packets:
            packets = _jsonl(args.packets)
        else:
            packets = packets_from_viddiff_falsification(_json(args.viddiff_falsification_report))
        result = evaluate_batch(
            packets,
            batch_metadata=_json(args.batch_metadata) if args.batch_metadata else None,
        )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "rubric_id": result["rubric_id"],
                "accepted_for_sft": result.get("accepted_for_sft"),
                "mass_production_ready": result.get("mass_production_ready"),
                "pending": result.get("pending"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
