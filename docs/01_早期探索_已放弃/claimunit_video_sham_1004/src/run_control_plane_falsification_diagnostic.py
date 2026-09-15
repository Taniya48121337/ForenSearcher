"""Run matched full/ablation closed-loop diagnostics with one local model load."""

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
from run_student_closed_loop_canary import (
    VCSL_FIRST,
    VCSL_REPAIR_FIRST,
    VIDEO_FIRST,
    VIDEO_REPAIR_FIRST,
    VIDDIFF_FIRST,
    VIDDIFF_REPAIR_FIRST,
    _run_episode,
    _vcsl_environment,
    _vcsl_repair_environment,
    _video_environment,
    _video_repair_environment,
    _viddiff_item,
    _viddiff_repair_environment,
)
from shadow_execute_generated_actions import _jsonl
from train_student_context_arm import DEFAULT_MODEL
from vcsl_batch import VCSL_TOOL_REGISTRY
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, VidDiffAdaptiveMeasurementEnvironment
from videosham_batch import MULTIFAMILY_TOOL_REGISTRY


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
    / "outputs/student_external_contract_v1/utqr_v6_current40"
    / "current40_sft_v5_diverse.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "arms/llm_diagnostic_v1"
)


def _builders(rows: dict[str, dict[str, Any]], mode: str) -> dict[str, tuple[str, Any, list[dict[str, Any]]]]:
    clean = {
        "VideoSham": (VIDEO_FIRST, lambda: _video_environment(rows), MULTIFAMILY_TOOL_REGISTRY),
        "VCSL": (VCSL_FIRST, _vcsl_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (
            VIDDIFF_FIRST,
            lambda: VidDiffAdaptiveMeasurementEnvironment(_viddiff_item("surgery_70")),
            VIDDIFF_TOOL_REGISTRY,
        ),
    }
    repair = {
        "VideoSham": (VIDEO_REPAIR_FIRST, lambda: _video_repair_environment(rows), MULTIFAMILY_TOOL_REGISTRY),
        "VCSL": (VCSL_REPAIR_FIRST, _vcsl_repair_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (VIDDIFF_REPAIR_FIRST, _viddiff_repair_environment, VIDDIFF_TOOL_REGISTRY),
    }
    return clean if mode == "clean" else repair


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "terminal_successes": sum(bool(item.get("terminal_success")) for item in episodes),
        "episodes_with_rejection": sum(int(item.get("rejection_count") or 0) > 0 for item in episodes),
        "turns": sum(int(item.get("turn_count") or 0) for item in episodes),
        "generated_tokens": sum(
            int(turn.get("generated_tokens") or 0)
            for item in episodes
            for turn in item.get("turns") or []
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows_list = _jsonl(args.sft)
    rows = {str(item["metadata"]["row_id"]): item for item in rows_list}
    donor_bank = build_evidence_donor_bank(_jsonl(args.evidence_donor_sft))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.adapter, load_in_4bit=not args.bf16_inference)
    selected_arms = args.arm or list(ARMS)
    selected_modes = args.mode or ["clean", "repair"]
    selected_datasets = args.dataset or ["VideoSham", "VCSL", "VidDiffBench"]
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()
    report_path = args.output_dir / "diagnostic_report.json"
    for mode in selected_modes:
        builders = _builders(rows, mode)
        for dataset in selected_datasets:
            row_id, environment_builder, registry = builders[dataset]
            for arm in selected_arms:
                episode_path = args.output_dir / f"{mode}_{dataset.lower()}_{arm}.json"
                if episode_path.exists():
                    episode = json.loads(episode_path.read_text(encoding="utf-8"))
                    episodes.append(episode)
                    print(
                        json.dumps(
                            {"mode": mode, "dataset": dataset, "arm": arm, "status": "resumed_complete"},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                episode_started = time.perf_counter()

                def save_partial(checkpoint: dict[str, Any]) -> None:
                    turn_count = int(checkpoint.get("turn_count") or 0)
                    write_json(
                        args.output_dir
                        / "partial_turns"
                        / f"{mode}_{dataset.lower()}_{arm}_turn_{turn_count:03d}.json",
                        checkpoint,
                    )

                episode = _run_episode(
                    tokenizer=tokenizer,
                    model=model,
                    seed_row=rows[row_id],
                    environment=environment_builder(),
                    tool_registry=registry,
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
                        "mode": mode,
                        "seed_row_id": row_id,
                        "episode_seconds": round(time.perf_counter() - episode_started, 3),
                    }
                )
                episodes.append(episode)
                write_json(episode_path, episode)
                write_json(
                    report_path,
                    {
                        "artifact": "control_plane_falsification_llm_diagnostic_v1",
                        "status": "running",
                        "config": _config(args, selected_arms, selected_modes, selected_datasets),
                        "summary": _summary(episodes),
                        "episodes": episodes,
                    },
                )
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "dataset": dataset,
                            "arm": arm,
                            "success": episode["terminal_success"],
                            "turns": episode["turn_count"],
                            "rejections": episode["rejection_count"],
                            "seconds": episode["episode_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return {
        "artifact": "control_plane_falsification_llm_diagnostic_v1",
        "status": "complete",
        "config": _config(args, selected_arms, selected_modes, selected_datasets),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": _summary(episodes),
        "episodes": episodes,
    }


def _config(
    args: argparse.Namespace,
    arms: list[str],
    modes: list[str],
    datasets: list[str],
) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "adapter": str(args.adapter),
        "sft": str(args.sft),
        "evidence_donor_sft": str(args.evidence_donor_sft),
        "arms": arms,
        "modes": modes,
        "datasets": datasets,
        "deterministic_greedy": True,
        "bf16_inference": args.bf16_inference,
        "max_turns": args.max_turns,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "max_generation_attempts": args.max_generation_attempts,
        "constrain_eligible_tool_names": True,
        "require_progressive_evidence_assessment": True,
        "prefer_natural_reasoning": True,
        "oracle_or_verifier_visible_to_policy": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--evidence-donor-sft", type=Path, default=DEFAULT_DONOR_SFT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--mode", action="append", choices=("clean", "repair"))
    parser.add_argument("--dataset", action="append", choices=("VideoSham", "VCSL", "VidDiffBench"))
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-generation-attempts", type=int, default=3)
    parser.add_argument("--bf16-inference", action="store_true")
    args = parser.parse_args()
    report = run(args)
    write_json(args.output_dir / "diagnostic_report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
