"""Run the pre-fixed three-case VCSL exact-pair holdout across four arms."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from control_plane_falsification_arms import ARMS, build_evidence_donor_bank
from evaluate_student_context_arm import _load_model
from pipeline import ROOT, write_json, write_jsonl
from run_control_plane_falsification_diagnostic import _config as diagnostic_config
from run_scripted_control_baseline import run_episode as run_scripted_episode
from run_student_closed_loop_canary import _run_episode
from shadow_execute_generated_actions import _jsonl, _payload
from train_student_context_arm import DEFAULT_MODEL
from vcsl_batch import VCSL_SCHEMA, VCSL_TOOL_REGISTRY, VCSLFeatureEnvironment
from vcsl_probe import DEFAULT_FEATURE_ROOT, segment_precision_recall


DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_sft_gain_experiment_v4_bind_compact"
    / "decision_sufficient_state_v6_executable_affordances/validation/sft_all.jsonl"
)
DEFAULT_ADAPTER = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_sft_gain_experiment_v4_bind_compact"
    / "adapter54_bind_visibleid_lr1e5/adapter"
)
DEFAULT_DONOR_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_current40/current40_sft_v5_diverse.jsonl"
)
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "manifests/sealed_cases.json"
)
DEFAULT_VERIFIER = DEFAULT_MANIFEST.with_name("sealed_cases_verifier.json")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "arms/llm_partial_holdout_vcsl_exactpair3_v1"
)


def _template(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        copy.deepcopy(row)
        for row in rows
        if row["metadata"]["dataset"] == "VCSL"
        and str(row["metadata"]["row_id"]).endswith(":clean:1")
    )


def build_seed_row(template: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(template)
    query_id = str(candidate["query_id"])
    reference_id = str(candidate["reference_id"])
    frame_counts = []
    for media_id in (query_id, reference_id):
        path = DEFAULT_FEATURE_ROOT / f"{media_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"sealed VCSL feature unavailable: {path}")
        frame_counts.append(int(np.load(path, mmap_mode="r").shape[0]))

    payload = _payload(row)
    contract = payload["task"]["task_contract"]
    contract.update(
        {
            "contract_id": f"TASK-VCSL-{query_id.upper()}-{reference_id.upper()}",
            "episode_mode": "paired_compare",
            "task": "Localize copied temporal segments between the visible query and reference features.",
            "visible_assets": {
                "source": "query",
                "suspect": "reference",
                "comparison_scope_s": [0.0, float(max(frame_counts))],
            },
            "claim_schema_id": VCSL_SCHEMA["schema_id"],
        }
    )
    payload["task"]["claim_schema"] = copy.deepcopy(VCSL_SCHEMA)
    state = payload["current_harness_state"]
    state.update(
        {
            "task_contract": copy.deepcopy(contract),
            "candidate_claims": [],
            "field_bindings": {},
            "evidence_store": {},
            "available_evidence_ids": [],
            "dropped_evidence_ids": [],
            "open_high_priority_obligations": ["G-CONTEXT", "G-DIFFERENCE", "G-REPORT"],
            "eligible_tools": ["inspect_pair_features"],
            "terminal": False,
            "stop_decision": None,
            "state_action_affordances": {
                "derivation": "Compiled from student-visible state.",
                "binding_candidates": {},
                "executable_binding_actions": [],
                "executable_evidence_assessment_actions": [],
                "executable_adjudication_actions": [],
                "repair_restoration_candidates": [],
                "meaningful_decision_transitions": [],
                "executable_decision_actions": [],
            },
        }
    )
    row["messages"] = [
        copy.deepcopy(row["messages"][0]),
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    case_id = str(candidate["case_id"])
    trajectory_id = f"SEALED-EXACT-PAIR-{query_id.upper()}-{reference_id.upper()}-CLEAN-001"
    row["metadata"] = {
        "dataset": "VCSL",
        "case_id": case_id,
        "trajectory_id": trajectory_id,
        "trajectory_type": "teacher_free_exact_pair_holdout_clean",
        "turn_index": 1,
        "row_id": f"VCSL:{case_id}:{trajectory_id}:clean:1",
        "data_split": "partial_sealed_exact_pair",
        "source_kind": "clean",
        "entity_group": f"VCSL:{case_id}",
        "routing_variant": "auto",
        "seed_contains_teacher_action": False,
        "seed_contains_oracle_or_verifier": False,
        "seed_source": "prefixed_manifest_pair_identity_and_local_feature_shape_only",
    }
    return row


def _environment(candidate: dict[str, Any]) -> VCSLFeatureEnvironment:
    return VCSLFeatureEnvironment(
        case={
            "pair_id": str(candidate["pair_id"]),
            "query_id": str(candidate["query_id"]),
            "reference_id": str(candidate["reference_id"]),
        },
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="auto",
    )


def _claim_values(episode: dict[str, Any]) -> dict[str, Any]:
    claims = (episode.get("final_state") or {}).get("candidate_claims") or []
    if not claims:
        return {}
    return {
        field: value.get("value")
        for field, value in (claims[0].get("fields") or {}).items()
        if isinstance(value, dict)
    }


def _binding_rate(episode: dict[str, Any]) -> float:
    state = episode.get("final_state") or {}
    claims = state.get("candidate_claims") or []
    if not claims:
        return 0.0
    claim_id = str(claims[0].get("claim_id"))
    bindings = (state.get("field_bindings") or {}).get(claim_id) or {}
    valid = sum(
        isinstance(value, dict) and value.get("validation_status") == "accepted"
        for value in bindings.values()
    )
    return valid / len(VCSL_SCHEMA["required_fields"])


def score_episode(
    episode: dict[str, Any],
    candidate: dict[str, Any],
    oracle_boxes: list[list[int]],
) -> dict[str, Any]:
    values = _claim_values(episode)
    predicted = values.get("copied_segments_frames")
    localization = (
        segment_precision_recall(predicted, oracle_boxes)
        if isinstance(predicted, list)
        else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )
    state = episode.get("final_state") or {}
    method_supported = any(
        isinstance(atom, dict)
        and atom.get("kind") == "copy_localization_measurement"
        and (atom.get("observation") or {}).get("localization_method")
        == values.get("localization_method")
        for atom in (state.get("evidence_store") or {}).values()
    )
    field_checks = {
        "query_id": values.get("query_id") == candidate["query_id"],
        "reference_id": values.get("reference_id") == candidate["reference_id"],
        "relationship": values.get("relationship") == "partial_copy",
        "copied_segments_frames": (
            localization["precision"] >= 0.5 and localization["recall"] >= 0.5
        ),
        "localization_method": method_supported,
    }
    return {
        "case_id": candidate["case_id"],
        "arm": episode.get("policy_view_arm"),
        "terminal_success": bool(episode.get("terminal_success")),
        "hidden_oracle_field_accuracy": sum(field_checks.values()) / len(field_checks),
        "field_checks": field_checks,
        "localization": localization,
        "binding_accuracy": _binding_rate(episode),
        "tool_sequence": episode.get("tool_sequence") or [],
        "turn_count": episode.get("turn_count"),
        "rejection_count": episode.get("rejection_count"),
        "episode_seconds": episode.get("episode_seconds"),
        "final_claim": values,
    }


def _summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in scores if row["arm"] == arm]
        by_arm[arm] = {
            "episodes": len(rows),
            "terminal_successes": sum(row["terminal_success"] for row in rows),
            "mean_hidden_oracle_field_accuracy": (
                sum(row["hidden_oracle_field_accuracy"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_binding_accuracy": (
                sum(row["binding_accuracy"] for row in rows) / len(rows) if rows else 0.0
            ),
        }
    return by_arm


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = list(manifest["selected_cases"]["VCSL"])
    if not candidates:
        raise ValueError("sealed manifest has no VCSL exact-pair cases")
    verifier_path = args.verifier
    rows = _jsonl(args.sft)
    template = _template(rows)
    seeds = [build_seed_row(template, candidate) for candidate in candidates]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_turns").mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "seeds.jsonl", seeds)

    route_preview = {}
    for candidate in candidates:
        environment = _environment(candidate)
        route_preview[candidate["case_id"]] = {
            "requested": "auto",
            "resolved": environment.routing_variant,
            "diagnostics": environment.routing_diagnostics,
        }
    run_manifest = {
        "artifact": "partial_sealed_vcsl_exact_pair_four_arm_run_v1",
        "status": "fixed_before_model_load",
        "source_manifest": str(args.manifest),
        "cases": candidates,
        "case_count": len(candidates),
        "entity_disjoint": False,
        "exact_pair_disjoint": True,
        "arms": list(ARMS),
        "routing": route_preview,
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
        "verifier_visible_to_policy": False,
    }
    write_json(args.output_dir / "RUN_MANIFEST.json", run_manifest)

    baseline_episodes = []
    for candidate, seed in zip(candidates, seeds):
        baseline_episodes.append(
            run_scripted_episode(
                seed_row=seed,
                environment=_environment(candidate),
                tool_registry=VCSL_TOOL_REGISTRY,
                max_turns=args.max_turns,
            )
        )
    write_json(
        args.output_dir / "scripted_control_baseline.json",
        {
            "artifact": "partial_holdout_scripted_control_baseline_v1",
            "episodes": baseline_episodes,
            "summary": {
                "episodes": len(baseline_episodes),
                "terminal_successes": sum(item["terminal_success"] for item in baseline_episodes),
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
    for case_index, (candidate, seed) in enumerate(zip(candidates, seeds), start=1):
        case_slug = f"case_{case_index:02d}"
        for arm in ARMS:
            episode_path = args.output_dir / "episodes" / f"{case_slug}_{arm}.json"
            if episode_path.exists():
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                episodes.append(episode)
                print(json.dumps({"case_id": candidate["case_id"], "arm": arm, "status": "resumed"}), flush=True)
                continue

            def save_partial(checkpoint: dict[str, Any]) -> None:
                turn = int(checkpoint.get("turn_count") or 0)
                write_json(
                    args.output_dir / "partial_turns" / f"{case_slug}_{arm}_turn_{turn:03d}.json",
                    checkpoint,
                )

            episode_started = time.perf_counter()
            episode = _run_episode(
                tokenizer=tokenizer,
                model=model,
                seed_row=seed,
                environment=_environment(candidate),
                tool_registry=VCSL_TOOL_REGISTRY,
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
                    "sealed_case_id": candidate["case_id"],
                    "mode": "clean",
                    "episode_seconds": round(time.perf_counter() - episode_started, 3),
                }
            )
            write_json(episode_path, episode)
            episodes.append(episode)
            write_json(
                args.output_dir / "report_running.json",
                {
                    "artifact": "partial_sealed_vcsl_exact_pair_four_arm_run_v1",
                    "status": "running",
                    "episodes": episodes,
                },
            )
            print(
                json.dumps(
                    {
                        "case_id": candidate["case_id"],
                        "arm": arm,
                        "terminal": episode["terminal_success"],
                        "turns": episode["turn_count"],
                        "seconds": episode["episode_seconds"],
                    }
                ),
                flush=True,
            )

    verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
    hidden = verifier["cases"]["VCSL"]
    candidate_map = {item["case_id"]: item for item in candidates}
    scores = [
        score_episode(
            episode,
            candidate_map[episode["sealed_case_id"]],
            hidden[episode["sealed_case_id"]]["oracle_boxes"],
        )
        for episode in episodes
    ]
    return {
        "artifact": "partial_sealed_vcsl_exact_pair_four_arm_run_v1",
        "status": "complete",
        "scope_warning": "exact-pair holdout only; not entity-disjoint or cross-dataset sealed",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": _summary(scores),
        "scores": scores,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
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
