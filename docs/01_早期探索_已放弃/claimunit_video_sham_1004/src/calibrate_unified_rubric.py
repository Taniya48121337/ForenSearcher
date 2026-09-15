"""Minimal-edit adversarial calibration for UTQR-3.

This is a meta-evaluation of the rubric, not trajectory generation.  Each hard
negative changes one local property of an otherwise verified bundle so a scorer
cannot pass merely by recognizing the dataset or final answer shape.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from pipeline import ROOT, write_json
from unified_trajectory_rubric import evaluate_bundle, load_saved_bundle


DEFAULT_VIDEO = (
    ROOT / "outputs" / "batch" / "videosham_seed_20260818_n10"
    / "cases" / "videosham-1007"
)
DEFAULT_DECISION = (
    ROOT / "outputs" / "utqr_v2_viddiff_canary"
    / "viddiff_adaptive_seed_20260818_n2_bon1" / "cases" / "surgery_80"
)
DEFAULT_VCSL = (
    ROOT / "outputs" / "utqr_v2_vcsl_10" / "vcsl_seed_20260818_n10_bon1"
    / "cases" / "000_218c6b03_5a155639" / "attempts" / "01_auto"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "rubric_v3_baseline" / "hard_negative_calibration.json"


Mutation = Callable[[dict[str, Any], dict[str, Any]], None]


def _wrong_outcome(_bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    checks = validation["checks"]
    recognized = False
    for name in (
        "oracle_label_match_post_execution",
        "oracle_window_match_post_execution",
        "oracle_direction_match",
        "localization_precision_at_least_half",
        "localization_recall_at_least_half",
        "localization_f1_at_least_point_six",
    ):
        if name in checks:
            checks[name] = False
            recognized = True
    metrics = validation.get("localization_metrics")
    if isinstance(metrics, dict):
        for name in ("precision", "recall", "f1"):
            if name in metrics:
                metrics[name] = 0.0
                recognized = True
    if not recognized:
        raise ValueError("wrong-outcome calibration found no dataset result check to perturb")


def _oracle_leak(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    bundle["clean_turn_sft"][0]["messages"].append({"role": "user", "content": "oracle_direction=a"})


def _fabricated_support(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    field = next(iter(bundle["clean_episode"]["final_state"]["candidate_claims"][0]["fields"].values()))
    field["support_ids"] = ["EA-FABRICATED-999"]


def _premature_terminal(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    bundle["repair_episode"]["terminated"] = False


def _noncausal_step(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    bundle["step_utility_report"]["passed"] = False
    first_episode = next(iter(bundle["step_utility_report"]["episodes"].values()))
    step = first_episode[0]
    step.update(
        {
            "causally_necessary": False,
            "downstream_consumed": False,
            "accepted_for_sft": False,
            "verdict": "redundant",
        }
    )


def _generic_reasoning(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    generic = (
        "The latest observation established something useful and a remaining gap exists. "
        "The next action is generally appropriate and should make progress. "
        "This statement intentionally omits concrete evidence and tool-specific justification."
    )
    for episode_name in ("clean_episode", "repair_episode"):
        for turn in bundle[episode_name]["turn_history"]:
            turn["reasoning"] = generic


def _off_target_change(bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    validation["checks"]["off_target_invariance"] = False
    bundle["repair_episode"]["invariant_fields_after"] = {"corrupted": True}


def _rejected_training_target(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    bundle["repair_turn_sft"][0]["metadata"]["target_execution_status"] = "rejected"


def _private_locator_in_target(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    target = bundle["clean_turn_sft"][0]["messages"][-1]
    target["content"] = str(target.get("content", "")) + (
        " The supporting file is C:\\Users\\researcher\\private\\video.mp4."
    )


def _ineligible_target_action(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    target = bundle["clean_turn_sft"][0]["messages"][-1]
    target["tool_calls"][0]["function"]["name"] = "stop"


def _nonvisible_evidence_reasoning(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    target = bundle["clean_turn_sft"][0]["messages"][-1]
    target["content"] = str(target.get("content", "")) + " This is supported by EA-FUTURE-999."


def _decision_overrepair(bundle: dict[str, Any], _validation: dict[str, Any]) -> None:
    bundle["repair_episode"]["turn_history"][0]["observation"]["authority"] = "external_tool_environment"


def _subgold_localization(_bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    validation["localization_metrics"]["recall"] = 0.84
    validation["localization_metrics"]["f1"] = 0.84


def _evaluate_negative(
    name: str,
    base_bundle: dict[str, Any],
    base_validation: dict[str, Any],
    mutation: Mutation,
) -> dict[str, Any]:
    bundle = copy.deepcopy(base_bundle)
    validation = copy.deepcopy(base_validation)
    mutation(bundle, validation)
    report = evaluate_bundle(bundle, validation)
    return {
        "family": name,
        "detected": not report["accepted_for_sft"],
        "score": report["total_score"],
        "failed_gates": report["diagnosis"]["failed_gates"],
        "failed_criteria": report["diagnosis"]["failed_criteria"],
    }


def run(
    video_dir: Path,
    decision_dir: Path,
    vcsl_dir: Path = DEFAULT_VCSL,
    output: Path | None = None,
) -> dict[str, Any]:
    video_bundle, video_validation = load_saved_bundle(video_dir)
    decision_bundle, decision_validation = load_saved_bundle(decision_dir)
    vcsl_bundle, vcsl_validation = load_saved_bundle(vcsl_dir)
    clean_baselines = [
        evaluate_bundle(copy.deepcopy(video_bundle), copy.deepcopy(video_validation)),
        evaluate_bundle(copy.deepcopy(decision_bundle), copy.deepcopy(decision_validation)),
        evaluate_bundle(copy.deepcopy(vcsl_bundle), copy.deepcopy(vcsl_validation)),
    ]
    cases = [
        _evaluate_negative("wrong_outcome", video_bundle, video_validation, _wrong_outcome),
        _evaluate_negative("oracle_leak", video_bundle, video_validation, _oracle_leak),
        _evaluate_negative("fabricated_support", video_bundle, video_validation, _fabricated_support),
        _evaluate_negative("premature_terminal", video_bundle, video_validation, _premature_terminal),
        _evaluate_negative("noncausal_redundant_step", video_bundle, video_validation, _noncausal_step),
        _evaluate_negative("generic_copy_pasted_reasoning", video_bundle, video_validation, _generic_reasoning),
        _evaluate_negative("off_target_repair_change", video_bundle, video_validation, _off_target_change),
        _evaluate_negative("rejected_training_target", video_bundle, video_validation, _rejected_training_target),
        _evaluate_negative("private_locator_in_target", video_bundle, video_validation, _private_locator_in_target),
        _evaluate_negative("ineligible_target_action", video_bundle, video_validation, _ineligible_target_action),
        _evaluate_negative(
            "nonvisible_evidence_reasoning",
            video_bundle,
            video_validation,
            _nonvisible_evidence_reasoning,
        ),
        _evaluate_negative("decision_reopen_overrepair", decision_bundle, decision_validation, _decision_overrepair),
        _evaluate_negative("vcsl_subgold_localization", vcsl_bundle, vcsl_validation, _subgold_localization),
    ]
    detected = sum(item["detected"] for item in cases)
    report = {
        "rubric_id": "UTQR-3",
        "method": "minimal_edit_hard_negative_calibration",
        "positive_controls_passed": all(item["accepted_for_sft"] for item in clean_baselines),
        "summary": {
            "negative_count": len(cases),
            "detected": detected,
            "detection_rate": round(detected / len(cases), 4),
            "calibration_canary_passed": (
                all(item["accepted_for_sft"] for item in clean_baselines) and detected / len(cases) >= 0.90
            ),
        },
        "hard_negatives": cases,
    }
    if output:
        write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--decision-dir", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--vcsl-dir", type=Path, default=DEFAULT_VCSL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.video_dir, args.decision_dir, args.vcsl_dir, args.output)["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
