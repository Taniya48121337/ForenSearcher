"""Run four-arm student episodes on the 13/30 reachable VidDiff panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from control_plane_falsification_arms import ARMS, build_evidence_donor_bank
from evaluate_student_context_arm import _load_model
from pipeline import ROOT, write_json
from run_scripted_control_baseline import run_episode as run_scripted_episode
from run_student_closed_loop_canary import _run_episode
from shadow_execute_generated_actions import _jsonl, _payload
from train_student_context_arm import DEFAULT_MODEL
from viddiff_batch import (
    VIDDIFF_SCHEMA,
    VIDDIFF_TOOL_REGISTRY,
    VidDiffAdaptiveMeasurementEnvironment,
)


DEFAULT_SEEDS = (
    ROOT
    / "outputs/student_external_contract_v1/viddiff_external_holdout30_symmetry_abstention_v1/seeds.jsonl"
)
DEFAULT_CANDIDATES = (
    ROOT
    / "outputs/selection/viddiff_external_holdout30_symmetry_abstention_v1"
    / "candidate_measurements.student.jsonl"
)
DEFAULT_AUDIT = DEFAULT_CANDIDATES.with_name("selection_audit.json")
DEFAULT_DONOR_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_current40/current40_sft_v5_diverse.jsonl"
)
DEFAULT_ADAPTER = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_sft_gain_experiment_v4_bind_compact"
    / "adapter54_bind_visibleid_lr1e5/adapter"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "arms/llm_viddiff30_reachable13_four_arm_v1"
)


def _values(episode: dict[str, Any]) -> dict[str, Any]:
    claims = (episode.get("final_state") or {}).get("candidate_claims") or []
    if not claims:
        return {}
    return {
        field: item.get("value")
        for field, item in (claims[0].get("fields") or {}).items()
        if isinstance(item, dict)
    }


def _binding_rate(episode: dict[str, Any]) -> float:
    state = episode.get("final_state") or {}
    claims = state.get("candidate_claims") or []
    if not claims:
        return 0.0
    claim_id = str(claims[0].get("claim_id"))
    bindings = (state.get("field_bindings") or {}).get(claim_id) or {}
    accepted = sum(
        isinstance(item, dict) and item.get("validation_status") == "accepted"
        for item in bindings.values()
    )
    return accepted / len(VIDDIFF_SCHEMA["required_fields"])


def score_episode(
    episode: dict[str, Any],
    item: dict[str, Any],
    oracle_direction: str,
) -> dict[str, Any]:
    values = _values(episode)
    evidence = (episode.get("final_state") or {}).get("evidence_store") or {}
    supported_measurements = [
        atom.get("observation") or {}
        for atom in evidence.values()
        if isinstance(atom, dict) and atom.get("kind") == "action_comparison_measurement"
    ]
    target_tuple = (
        values.get("preferred_video"),
        values.get("video_a_interval_s"),
        values.get("video_b_interval_s"),
        values.get("measurement_method"),
    )
    measurement_supported = any(
        (
            observation.get("preferred_video"),
            observation.get("video_a_interval_s"),
            observation.get("video_b_interval_s"),
            observation.get("measurement_method"),
        )
        == target_tuple
        for observation in supported_measurements
    )
    checks = {
        "sample_key": values.get("sample_key") == item["sample_key"],
        "comparison_statement": values.get("comparison_statement")
        == item["task"]["comparison_statement"],
        "preferred_video": values.get("preferred_video") == oracle_direction,
        "measurement_tuple_supported": measurement_supported,
    }
    return {
        "case_id": f"viddiff-{item['sample_key']}",
        "arm": episode.get("policy_view_arm"),
        "terminal_success": bool(episode.get("terminal_success")),
        "oracle_direction_correct": checks["preferred_video"],
        "terminal_and_oracle_correct": bool(episode.get("terminal_success"))
        and checks["preferred_video"],
        "claim_checks": checks,
        "claim_check_rate": sum(checks.values()) / len(checks),
        "binding_accuracy": _binding_rate(episode),
        "turn_count": episode.get("turn_count"),
        "rejection_count": episode.get("rejection_count"),
        "episode_seconds": episode.get("episode_seconds"),
        "tool_sequence": episode.get("tool_sequence") or [],
        "final_claim": values,
    }


def _summary(scores: list[dict[str, Any]], total_cases: int, unreachable: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in scores if row["arm"] == arm]
        correct = sum(row["terminal_and_oracle_correct"] for row in rows)
        result[arm] = {
            "reachable_episodes": len(rows),
            "terminal_successes": sum(row["terminal_success"] for row in rows),
            "oracle_direction_correct": sum(row["oracle_direction_correct"] for row in rows),
            "terminal_and_oracle_correct": correct,
            "conditional_terminal_oracle_rate": correct / len(rows) if rows else 0.0,
            "end_to_end_safe_coverage": correct / total_cases,
            "mean_binding_accuracy": (
                sum(row["binding_accuracy"] for row in rows) / len(rows) if rows else 0.0
            ),
            "unreachable_or_quarantined": unreachable,
            "total_cases": total_cases,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _jsonl(args.seeds)
    seed_by_case = {str(row["metadata"]["case_id"]): row for row in seeds}
    items = {str(item["sample_key"]): item for item in _jsonl(args.candidates)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    reachable = [case for case in audit["cases"] if case.get("accepted") is True]
    unreachable = [case for case in audit["cases"] if case.get("accepted") is not True]
    missing = [
        case["sample_key"]
        for case in reachable
        if f"viddiff-{case['sample_key']}" not in seed_by_case or case["sample_key"] not in items
    ]
    if missing:
        raise ValueError(f"reachable VidDiff cases lack seed or candidates: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_turns").mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "artifact": "viddiff30_reachable13_four_arm_run_v1",
        "status": "fixed_before_model_load",
        "development_exposed": True,
        "sealed": False,
        "total_cases": len(audit["cases"]),
        "reachable_cases": [case["sample_key"] for case in reachable],
        "unreachable_cases": [case["sample_key"] for case in unreachable],
        "reachability_definition": (
            "verifier-only post-measurement admission: at least one label-blind candidate "
            "matches oracle; not visible to student"
        ),
        "conditional_measurement_quality": "13/13 by admitted-candidate construction",
        "safe_candidate_coverage": f"{len(reachable)}/{len(audit['cases'])}",
        "unreachable_or_quarantined": f"{len(unreachable)}/{len(audit['cases'])}",
        "arms": list(ARMS),
        "routing_variant": "balanced",
        "model": str(args.model),
        "adapter": str(args.adapter),
        "decode": {
            "load_in_4bit": not args.bf16_inference,
            "greedy": True,
            "max_turns": args.max_turns,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "max_generation_attempts": args.max_generation_attempts,
            "utqr": "unchanged",
        },
        "oracle_or_verifier_visible_to_policy": False,
    }
    write_json(args.output_dir / "RUN_MANIFEST.json", run_manifest)

    baseline = []
    for case in reachable:
        sample_key = str(case["sample_key"])
        baseline.append(
            run_scripted_episode(
                seed_row=seed_by_case[f"viddiff-{sample_key}"],
                environment=VidDiffAdaptiveMeasurementEnvironment(items[sample_key]),
                tool_registry=VIDDIFF_TOOL_REGISTRY,
                max_turns=args.max_turns,
            )
        )
    write_json(
        args.output_dir / "scripted_control_baseline.json",
        {
            "episodes": baseline,
            "summary": {
                "episodes": len(baseline),
                "terminal_successes": sum(row["terminal_success"] for row in baseline),
                "generated_tokens": 0,
            },
        },
    )

    donor_bank = build_evidence_donor_bank(_jsonl(args.evidence_donor_sft))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.adapter, load_in_4bit=not args.bf16_inference)
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case_index, case in enumerate(reachable, start=1):
        sample_key = str(case["sample_key"])
        seed = seed_by_case[f"viddiff-{sample_key}"]
        for arm in ARMS:
            path = args.output_dir / "episodes" / f"case_{case_index:02d}_{arm}.json"
            if path.exists():
                episode = json.loads(path.read_text(encoding="utf-8"))
                episodes.append(episode)
                print(json.dumps({"sample_key": sample_key, "arm": arm, "status": "resumed"}), flush=True)
                continue

            def save_partial(checkpoint: dict[str, Any]) -> None:
                turn = int(checkpoint.get("turn_count") or 0)
                write_json(
                    args.output_dir / "partial_turns" / f"case_{case_index:02d}_{arm}_turn_{turn:03d}.json",
                    checkpoint,
                )

            episode_started = time.perf_counter()
            episode = _run_episode(
                tokenizer=tokenizer,
                model=model,
                seed_row=seed,
                environment=VidDiffAdaptiveMeasurementEnvironment(items[sample_key]),
                tool_registry=VIDDIFF_TOOL_REGISTRY,
                max_turns=args.max_turns,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                max_generation_attempts=args.max_generation_attempts,
                constrain_eligible_tool_names=True,
                require_evidence_adjudication=True,
                require_progressive_evidence_assessment=True,
                prefer_natural_reasoning=True,
                policy_view_arm=arm,
                evidence_donor_bank=donor_bank if arm == "shuffled_evidence" else None,
                turn_checkpoint_callback=save_partial,
            )
            episode.update(
                {
                    "sample_key": sample_key,
                    "episode_seconds": round(time.perf_counter() - episode_started, 3),
                }
            )
            write_json(path, episode)
            episodes.append(episode)
            write_json(
                args.output_dir / "report_running.json",
                {"status": "running", "episodes_completed": len(episodes), "episodes": episodes},
            )
            print(
                json.dumps(
                    {
                        "sample_key": sample_key,
                        "arm": arm,
                        "terminal": episode["terminal_success"],
                        "turns": episode["turn_count"],
                        "seconds": episode["episode_seconds"],
                    }
                ),
                flush=True,
            )

    oracle = {str(case["sample_key"]): str(case["oracle_direction"]) for case in reachable}
    scores = [score_episode(ep, items[ep["sample_key"]], oracle[ep["sample_key"]]) for ep in episodes]
    return {
        "artifact": "viddiff30_reachable13_four_arm_run_v1",
        "status": "complete",
        "scope_warning": "development-exposed; 13-case oracle-bounded reachability subset",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "measurement_coverage": {
            "conditional_trajectory_quality": f"{len(reachable)}/{len(reachable)}",
            "end_to_end_safe_candidate_coverage": f"{len(reachable)}/{len(audit['cases'])}",
            "unreachable_or_quarantined": f"{len(unreachable)}/{len(audit['cases'])}",
        },
        "summary": _summary(scores, len(audit["cases"]), len(unreachable)),
        "scores": scores,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--evidence-donor-sft", type=Path, default=DEFAULT_DONOR_SFT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-generation-attempts", type=int, default=3)
    parser.add_argument("--bf16-inference", action="store_true")
    args = parser.parse_args()
    report = run(args)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
