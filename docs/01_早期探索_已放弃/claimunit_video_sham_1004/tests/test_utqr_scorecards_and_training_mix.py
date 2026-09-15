from __future__ import annotations

import copy
import json
from pathlib import Path

from build_balanced_recovery_sft_mix import build_mix
from audit_recovery_step_value import audit_batch
from falsify_utqr_episode_scorecards import (
    _add_unused_executed_measurement,
    run_falsification,
)
from score_utqr_recovery_episodes import (
    _expected_evidence_first_selection,
    score_batch,
)
from shadow_execute_generated_actions import _jsonl


ROOT = Path(__file__).resolve().parents[1]
BATCH = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio"
)
RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
RUBRIC_V10 = ROOT / "rubric/unified_trajectory_quality_rubric_v10.json"
RUBRIC_V11 = ROOT / "rubric/unified_trajectory_quality_rubric_v11.json"
RUBRIC_V12 = ROOT / "rubric/unified_trajectory_quality_rubric_v12.json"
EVIDENCE_FIRST_BATCH = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "recovery_batch_candidate_v31_evidence_first55"
)
CLAIM_FIRST_BATCH = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "recovery_batch_candidate_v20_abstention_semantic_transparency"
)
STEP_VALUE_BATCH = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "recovery_batch_candidate_v32_step_value55"
)
STEP_VALUE_PORTABLE = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "recovery_sft_semantic_contrast_portable_v8_step_value_neighborhood55"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[list[dict], list[dict], dict, dict]:
    return (
        _jsonl(BATCH / "recovery_episodes.jsonl"),
        _jsonl(BATCH / "recovery_sft_candidate.jsonl"),
        _load(BATCH / "compile_report.json"),
        _load(RUBRIC),
    )


def test_every_expanded_episode_reaches_full_automated_core_but_not_human_gate() -> None:
    episodes, rows, compile_report, rubric = _inputs()
    report = score_batch(episodes, rows, compile_report, rubric)
    assert report["episodes"] == 46
    assert report["batch_automated_core_p10"] == 92.0
    assert report["all_H1_to_H7_automated"] is True
    assert report["H8_independent_blind_review"] == "pending"
    assert all(
        summary["automated_core_p10"] == 92.0
        and summary["all_actions_useful"] is True
        and summary["human_review_completed"] is False
        for summary in report["dataset_summary"].values()
    )


def test_general_negative_controls_lower_score_and_fail_automated_summary() -> None:
    episodes, rows, compile_report, rubric = _inputs()
    report = run_falsification(episodes, rows, compile_report, rubric)
    assert report["mutations"] == 7
    assert report["all_detected"] is True
    assert report["minimum_score_drop"] > 0
    assert all(
        item["mutated_score"] < item["baseline_score"]
        and item["automated_hard_gates_after"] is False
        for item in report["results"]
    )


def test_weighted_mix_equalizes_dataset_mass_without_changing_messages() -> None:
    _, rows, compile_report, _ = _inputs()
    mixed, report = build_mix(rows, compile_report)
    assert len(mixed) == len(rows) == 319
    assert report["messages_unchanged"] is True
    masses = list(report["weighted_mass_by_dataset"].values())
    assert max(masses) - min(masses) < 1e-5
    assert report["weight_distribution"]["mean"] == 1.0
    assert report["weight_distribution"]["effective_sample_fraction"] > 0.75
    assert all(
        item["messages"] == source["messages"]
        and item["metadata"]["sft_sampling_weight"] > 0
        for item, source in zip(mixed, rows)
    )


def _score_named_batch(batch: Path, rubric: Path) -> dict:
    return score_batch(
        _jsonl(batch / "recovery_episodes.jsonl"),
        _jsonl(batch / "recovery_sft_candidate.jsonl"),
        _load(batch / "compile_report.json"),
        _load(rubric),
    )


def test_utqr10_distinguishes_claim_first_from_evidence_first_adjudication() -> None:
    evidence_first = _score_named_batch(EVIDENCE_FIRST_BATCH, RUBRIC_V10)
    claim_first = _score_named_batch(CLAIM_FIRST_BATCH, RUBRIC_V10)
    assert evidence_first["batch_automated_core_p10"] == 92.0
    assert evidence_first["all_H1_to_H7_automated"] is True
    assert sum(
        item["evidence_first_adjudication"]["applicable"]
        for item in evidence_first["scorecards"]
    ) == 12
    assert all(item["criteria"]["C11"] == 1.0 for item in evidence_first["scorecards"])

    failed = [item for item in claim_first["scorecards"] if item["criteria"]["C11"] == 0.0]
    wrong_winner = [
        item
        for item in failed
        if any(
            not detail["checks"]["selected_matches_visible_recompute"]
            for detail in item["evidence_first_adjudication"]["details"]
        )
    ]
    assert len(failed) == 12
    assert len(wrong_winner) == 4
    assert claim_first["dataset_summary"]["VidDiffBench"]["automated_core_p10"] == 84.0
    assert claim_first["dataset_summary"]["VideoSham"]["automated_core_p10"] == 92.0
    assert claim_first["dataset_summary"]["VCSL"]["automated_core_p10"] == 92.0


def test_utqr10_counts_repeated_runs_of_one_method_as_one_vote() -> None:
    evidence_store = {}
    for index in range(5):
        evidence_store[f"A{index}"] = {
            "active": True,
            "supports_fields": ["preferred_video"],
            "observation": {
                "preferred_video": "a",
                "measurement_method": "same_method",
                "confidence": 0.80 + index / 100,
            },
            "provenance": {},
        }
    evidence_store["B0"] = {
        "active": True,
        "supports_fields": ["preferred_video"],
        "observation": {
            "preferred_video": "b",
            "measurement_method": "other_method",
            "confidence": 0.9,
        },
        "provenance": {},
    }
    evidence_store["C0"] = {
        "active": True,
        "supports_fields": ["preferred_video"],
        "observation": {
            "preferred_video": "c",
            "measurement_method": "abstention_method",
            "confidence": 0.7,
        },
        "provenance": {},
    }
    selected, diagnostics = _expected_evidence_first_selection(
        evidence_store,
        {
            "target_fields": ["preferred_video"],
            "competing_evidence_ids": list(evidence_store),
        },
    )
    assert selected == ["C0"]
    assert diagnostics["method_votes"] == {"a": 1, "b": 1, "c": 1}


def test_utqr10_negative_controls_include_rejected_evidence_reentry() -> None:
    episodes = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_episodes.jsonl")
    rows = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_sft_candidate.jsonl")
    report = run_falsification(
        episodes,
        rows,
        _load(EVIDENCE_FIRST_BATCH / "compile_report.json"),
        _load(RUBRIC_V10),
    )
    assert report["mutations"] == 8
    assert report["all_detected"] is True
    reentry = next(
        item
        for item in report["results"]
        if item["mutation"] == "rejected_evidence_reenters_final_binding"
    )
    assert reentry["expected_criterion"] == "C11"
    assert reentry["score_drop"] == 8.0


def test_utqr12_requires_every_executed_step_to_have_a_downstream_role() -> None:
    episodes = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_episodes.jsonl")
    rows = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_sft_candidate.jsonl")
    current = audit_batch(episodes, rows)
    assert current["episodes_passed"] == current["episodes"] == 55
    assert current["turns"] == 418
    assert current["failure_counts"] == {}

    source = next(item for item in episodes if item["dataset"] == "VideoSham")
    mutated_episode = copy.deepcopy(source)
    mutated_rows = [
        copy.deepcopy(row)
        for row in rows
        if row["metadata"]["episode_id"] == source["episode_id"]
    ]
    _add_unused_executed_measurement(mutated_episode, mutated_rows)
    old = score_batch(
        [copy.deepcopy(mutated_episode)],
        copy.deepcopy(mutated_rows),
        _load(EVIDENCE_FIRST_BATCH / "compile_report.json"),
        _load(RUBRIC_V11),
    )["scorecards"][0]
    strict = score_batch(
        [mutated_episode],
        mutated_rows,
        _load(EVIDENCE_FIRST_BATCH / "compile_report.json"),
        _load(RUBRIC_V12),
    )["scorecards"][0]
    assert old["criteria"]["C5"] == 1.0
    assert old["automated_core_score"] == 92.0
    assert strict["criteria"]["C5"] == 0.0
    assert strict["automated_core_score"] == 83.0
    assert strict["step_value"]["failed_turns"] == 1
    failure = next(
        detail for detail in strict["step_value"]["details"] if not detail["passed"]
    )
    assert failure["failures"] == ["all_created_evidence_consumed"]


def test_utqr12_falsification_includes_executed_but_unused_measurement() -> None:
    episodes = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_episodes.jsonl")
    rows = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_sft_candidate.jsonl")
    report = run_falsification(
        episodes,
        rows,
        _load(EVIDENCE_FIRST_BATCH / "compile_report.json"),
        _load(RUBRIC_V12),
    )
    assert report["mutations"] == 9
    assert report["all_detected"] is True
    unused = next(
        item
        for item in report["results"]
        if item["mutation"] == "executed_but_unused_external_measurement"
    )
    assert unused["expected_criterion"] == "C5"
    assert unused["expected_criterion_before"] == 1.0
    assert unused["expected_criterion_after"] == 0.0
    assert unused["score_drop"] == 9.0


def test_step_value_generation_package_is_audited_without_changing_messages() -> None:
    source_rows = _jsonl(EVIDENCE_FIRST_BATCH / "recovery_sft_candidate.jsonl")
    rows = _jsonl(STEP_VALUE_BATCH / "recovery_sft_candidate.jsonl")
    scores = _load(STEP_VALUE_BATCH / "utqr12_episode_scorecards.json")
    step_value = _load(STEP_VALUE_BATCH / "step_value_audit.json")
    readiness = _load(
        ROOT
        / "outputs/student_external_contract_v1"
        / "production_readiness_audit_v10_utqr12_step_value55"
        / "readiness_report.json"
    )
    portability = _load(STEP_VALUE_PORTABLE / "portability_audit.json")
    contrast = _load(STEP_VALUE_PORTABLE / "contrast_integrity_audit.json")
    assert len(rows) == len(source_rows) == 418
    assert all(left["messages"] == right["messages"] for left, right in zip(source_rows, rows))
    assert all(
        row["metadata"]["step_value_audited"] is True
        and row["metadata"]["step_value_passed"] is True
        for row in rows
    )
    assert scores["rubric_id"] == "UTQR-12"
    assert scores["batch_automated_core_p10"] == 92.0
    assert all(
        item["all_steps_have_downstream_decision_role"] is True
        for item in scores["dataset_summary"].values()
    )
    assert step_value["episodes_passed"] == 55
    assert step_value["turns"] == 418
    assert readiness["step_value"]["unused_measurement_mutation_detected"] is True
    assert readiness["training_qualified"] is False
    assert portability["all_semantic_equivalence_checks_passed"] is True
    assert contrast["passed"] is True
