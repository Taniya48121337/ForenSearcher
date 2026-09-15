"""Run resumable target-hidden student closed loops and export only rubric-accepted SFT."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_student_closed_loop_quality import _audit_episode, _repair_profile
from evaluate_student_context_arm import _load_model
from pipeline import ROOT, decode, write_json, write_jsonl
from run_student_closed_loop_canary import (
    DEFAULT_MODEL,
    _repetitive_reasoning_style_problems,
    _run_episode,
)
from run_student_closed_loop_expanded_panel import _public_video_identity
from shadow_execute_generated_actions import _jsonl, _payload, _restore_harness
from vcsl_batch import VCSLFeatureEnvironment, VCSL_TOOL_REGISTRY
from vcsl_probe import DEFAULT_FEATURE_ROOT
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, VidDiffAdaptiveMeasurementEnvironment
from viddiff_rejection_sampler import _student_visible
from validate_student_batch_sft import diversity_balance_rows, validate_rows
from videosham_batch import (
    EDITED_ROOT,
    MULTIFAMILY_TOOL_REGISTRY,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
)


DEFAULT_SFT_ROOT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_ADAPTER = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "v10_bind_then_viddiff_route8_lr1e5"
    / "adapter"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_batch_test_v10_native"
)


def _first_rows(sft_root: Path, split: str) -> list[dict[str, Any]]:
    all_rows = _jsonl(sft_root / split / "sft_all.jsonl")
    rows = [
        row
        for row in all_rows
        if str(row["metadata"]["row_id"]).endswith(":clean:1")
        or str(row["metadata"]["row_id"]).endswith(":repair:1")
    ]
    for row in rows:
        if row["metadata"]["dataset"] != "VideoSham":
            continue
        identity = None
        for candidate in all_rows:
            if (
                candidate["metadata"]["dataset"] != "VideoSham"
                or candidate["metadata"]["case_id"] != row["metadata"]["case_id"]
                or _mode(candidate) != _mode(row)
            ):
                continue
            try:
                identity = _public_video_identity(candidate)
                break
            except ValueError:
                continue
        if identity is None:
            raise ValueError(
                "student-visible VideoSham trajectory lacks public source identity or scope: "
                f"{row['metadata']['row_id']}"
            )
        row["_batch_public_video_identity"] = identity
    return rows


def _mode(row: dict[str, Any]) -> str:
    return "repair" if ":repair:" in str(row["metadata"]["row_id"]) else "clean"


def _case_id(row: dict[str, Any]) -> str:
    return str(row["metadata"]["case_id"])


def _damaged_fields(row: dict[str, Any]) -> list[str]:
    state = _payload(row)["current_harness_state"]
    subtype, fields = _repair_profile(state)
    return sorted(fields) if subtype == "evidence_dropout" else []


def _video_environment(row: dict[str, Any], mode: str) -> MultiFamilyVideoEnvironment:
    case_id = _case_id(row)
    number = case_id.removeprefix("videosham-")
    identity = row.get("_batch_public_video_identity")
    source_id, scope = identity if identity is not None else _public_video_identity(row)
    source_path = SOURCE_ROOT / f"{number}_unedited.mp4"
    suspect_path = EDITED_ROOT / f"{number}_processed.mp4"
    environment = MultiFamilyVideoEnvironment(
        manifest={
            "case_id": case_id,
            "source_id": source_id,
            "comparison_scope_s": scope,
        },
        source_frames=decode(source_path),
        suspect_frames=decode(suspect_path),
        source_path=source_path,
        suspect_path=suspect_path,
    )
    if mode == "repair":
        environment.execute("inspect_media", {"media_refs": ["source", "suspect"]})
        selected = environment.eligible_tool_names()
        if len(selected) != 1:
            raise ValueError(f"VideoSham repair seed has ambiguous modality: {case_id}:{selected}")
        environment.execute(
            selected[0],
            {
                "source_ref": "source",
                "suspect_ref": "suspect",
                "comparison_scope_s": scope,
            },
        )
    return environment


def _vcsl_identity(row: dict[str, Any]) -> tuple[str, str]:
    pair = _case_id(row).removeprefix("vcsl-")
    query_id, reference_id = pair.split("-", 1)
    return query_id, reference_id


def _vcsl_environment(row: dict[str, Any], mode: str) -> VCSLFeatureEnvironment:
    query_id, reference_id = _vcsl_identity(row)
    metadata = row["metadata"]
    variant = str(
        metadata.get("resolved_routing_variant")
        or metadata.get("routing_variant")
        or "auto"
    )
    environment = VCSLFeatureEnvironment(
        case={
            "pair_id": f"{query_id}-{reference_id}",
            "query_id": query_id,
            "reference_id": reference_id,
        },
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant=variant,
    )
    if mode == "repair":
        environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        environment.execute(
            environment.localizer_spec["tool_name"],
            {
                "query_ref": "query",
                "reference_ref": "reference",
                "method": environment.localizer_spec["method"],
            },
        )
        damaged = _damaged_fields(row)
        if damaged:
            dropped = _payload(row)["current_harness_state"].get("dropped_evidence_ids") or [
                "binding-invalidated"
            ]
            environment.on_evidence_dropout(str(dropped[0]), damaged)
    return environment


def _all_viddiff_items(
    paths: list[Path] | None = None,
) -> dict[str, dict[str, Any]]:
    paths = (
        sorted(ROOT.glob("outputs/**/candidate_measurements.student.jsonl"))
        if paths is None
        else [Path(path) for path in paths]
    )
    items: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"VidDiff candidate file does not exist: {path}")
        for item in _jsonl(path):
            items[str(item["sample_key"])] = _student_visible(item)
    return items


def _viddiff_environment(
    row: dict[str, Any],
    mode: str,
    items: dict[str, dict[str, Any]],
) -> VidDiffAdaptiveMeasurementEnvironment:
    sample_key = _case_id(row).removeprefix("viddiff-")
    item = items[sample_key]
    requested = str(row["metadata"].get("routing_variant") or "balanced")
    variant = (
        requested
        if requested in VidDiffAdaptiveMeasurementEnvironment.ROUTING_VARIANTS
        else "balanced"
    )
    environment = VidDiffAdaptiveMeasurementEnvironment(item, routing_variant=variant)
    if mode == "repair":
        state = _payload(row)["current_harness_state"]
        arguments = item["candidate_tools"][0]["arguments"]
        environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        called_sequences = [
            atom.get("observation", {}).get("routing_state", {}).get("called_methods", [])
            for atom in (state.get("evidence_store") or {}).values()
            if atom.get("kind") == "action_comparison_measurement"
        ]
        clean_methods = list(max(called_sequences, key=len, default=[]))
        for evidence_id in state.get("dropped_evidence_ids") or []:
            upper = str(evidence_id).upper()
            for method in environment.available_methods:
                if method.upper() in upper and method not in clean_methods:
                    clean_methods.append(method)
        for method in clean_methods:
            environment.execute(method, arguments)
        damaged = _damaged_fields(row)
        if damaged:
            dropped = state.get("dropped_evidence_ids") or ["binding-invalidated"]
            environment.on_evidence_dropout(str(dropped[0]), damaged)
    return environment


def _environment(
    row: dict[str, Any],
    mode: str,
    viddiff_items: dict[str, dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    dataset = str(row["metadata"]["dataset"])
    if dataset == "VideoSham":
        return _video_environment(row, mode), MULTIFAMILY_TOOL_REGISTRY
    if dataset == "VCSL":
        return _vcsl_environment(row, mode), VCSL_TOOL_REGISTRY
    if dataset == "VidDiffBench":
        return _viddiff_environment(row, mode, viddiff_items), VIDDIFF_TOOL_REGISTRY
    raise ValueError(f"unsupported dataset: {dataset}")


def _select(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = set(args.dataset or ("VideoSham", "VCSL", "VidDiffBench"))
    modes = set(args.mode or ("clean", "repair"))
    requested_cases = set(args.case_id or [])
    candidates = [
        row
        for row in rows
        if row["metadata"]["dataset"] in datasets
        and _mode(row) in modes
        and (not requested_cases or _case_id(row) in requested_cases)
    ]
    if requested_cases:
        missing = requested_cases - {_case_id(row) for row in candidates}
        if missing:
            raise ValueError(f"requested cases are unavailable in split {args.split}: {sorted(missing)}")
    if args.limit_cases_per_dataset:
        allowed: set[tuple[str, str]] = set()
        for dataset in sorted(datasets):
            case_ids = sorted(
                {_case_id(row) for row in candidates if row["metadata"]["dataset"] == dataset}
            )[: args.limit_cases_per_dataset]
            allowed.update((dataset, case_id) for case_id in case_ids)
        candidates = [
            row
            for row in candidates
            if (str(row["metadata"]["dataset"]), _case_id(row)) in allowed
        ]
    return sorted(
        candidates,
        key=lambda row: (
            str(row["metadata"]["dataset"]),
            _case_id(row),
            0 if _mode(row) == "clean" else 1,
        ),
    )


def _checkpoint_path(output: Path, row: dict[str, Any]) -> Path:
    dataset = str(row["metadata"]["dataset"]).lower()
    case_id = _case_id(row)
    return output / "cases" / dataset / case_id / f"{_mode(row)}.json"


def _accepted_sft(
    episode: dict[str, Any],
    quality: dict[str, Any],
    row: dict[str, Any],
    split: str,
    rollout_id: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    for turn in episode.get("turns") or []:
        datum = copy.deepcopy(turn.get("sft_datum"))
        if not datum:
            continue
        metadata = datum["metadata"]
        metadata.update(
            {
                "row_id": ":".join(
                    part
                    for part in (
                        str(row["metadata"]["dataset"]),
                        _case_id(row),
                        "student-closed-loop",
                        _mode(row),
                        rollout_id,
                        str(turn["turn"]),
                    )
                    if part is not None
                ),
                "case_id": _case_id(row),
                "source_split": split,
                "trajectory_mode": _mode(row),
                "trajectory_type": row["metadata"].get("trajectory_type"),
                "strict_closed_loop_score": quality["score"],
                "strict_closed_loop_accepted": True,
                **({"rollout_id": rollout_id} if rollout_id is not None else {}),
            }
        )
        result.append(datum)
    return result


def _preflight(
    rows: list[dict[str, Any]],
    viddiff_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        environment, registry = _environment(row, _mode(row), viddiff_items)
        harness = _restore_harness(row)
        harness.environment = environment
        harness.tool_registry = copy.deepcopy(registry)
        expected = list(
            _payload(row)["current_harness_state"].get("eligible_tools") or []
        )
        actual = list(harness.render_state().get("eligible_tools") or [])
        details.append(
            {
                "dataset": row["metadata"]["dataset"],
                "case_id": _case_id(row),
                "mode": _mode(row),
                "expected_eligible_tools": expected,
                "restored_eligible_tools": actual,
                "eligible_tools_match": expected == actual,
            }
        )
    mismatches = [item for item in details if not item["eligible_tools_match"]]
    return {
        "episodes": len(details),
        "environment_builds_succeeded": len(details),
        "eligible_tool_matches": len(details) - len(mismatches),
        "all_passed": not mismatches,
        "mismatches": mismatches,
        "details": details,
    }


def _group_metrics(
    records: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in sorted({str(record[group_key]) for record in records}):
        selected = [record for record in records if str(record[group_key]) == value]
        actions = sum(len(record["episode"].get("turns") or []) for record in selected)
        seconds = sum(float(record["elapsed_seconds"]) for record in selected)
        prompt_tokens = sum(
            int(turn.get("prompt_tokens_kept", 0))
            for record in selected
            for turn in record["episode"].get("turns") or []
        )
        generated_tokens = sum(
            int(turn.get("generated_tokens", 0))
            for record in selected
            for turn in record["episode"].get("turns") or []
        )
        result[value] = {
            "episodes": len(selected),
            "accepted_episodes": sum(
                record["quality"]["accepted_for_high_quality_sft"]
                for record in selected
            ),
            "actions": actions,
            "measured_seconds": round(seconds, 3),
            "seconds_per_episode": round(seconds / len(selected), 3),
            "seconds_per_action": round(seconds / actions, 3) if actions else None,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "mean_score": round(
                sum(record["quality"]["score"] for record in selected)
                / len(selected),
                3,
            ),
        }
    return result


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(float(ordered[index]), 3)


def _episode_style_diagnostics(episode: dict[str, Any]) -> dict[str, Any]:
    prior: list[str] = []
    findings: list[dict[str, Any]] = []
    for turn in episode.get("turns") or []:
        reasoning = str(turn.get("reasoning") or "")
        if not reasoning:
            continue
        problems = _repetitive_reasoning_style_problems(reasoning, prior)
        if problems:
            findings.append(
                {
                    "turn": turn.get("turn"),
                    "tool": turn.get("generated_tool"),
                    "problems": problems,
                }
            )
        prior.append(reasoning)
    return {
        "repeated_scaffold_turns": len(findings),
        "repeated_scaffold_problem_count": sum(
            len(item["problems"]) for item in findings
        ),
        "findings": findings,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = (
        _jsonl(args.seed_jsonl)
        if args.seed_jsonl is not None
        else _first_rows(args.sft_root, args.split)
    )
    rows = _select(source_rows, args)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = [
        {
            "dataset": row["metadata"]["dataset"],
            "case_id": _case_id(row),
            "mode": _mode(row),
            "seed_row_id": row["metadata"]["row_id"],
            "trajectory_type": row["metadata"].get("trajectory_type"),
            "routing_variant": row["metadata"].get("resolved_routing_variant")
            or row["metadata"].get("routing_variant"),
        }
        for row in rows
    ]
    write_json(
        args.output / "selection.json",
        {
            "split": args.split,
            "episodes": len(selection),
            "teacher_target_used_for_selection": False,
            "selection": selection,
        },
    )
    if args.dry_run:
        return {"summary": {"selected_episodes": len(rows), "dry_run": True}, "selection": selection}

    viddiff_items = _all_viddiff_items(
        getattr(args, "viddiff_candidates", None)
    )
    if args.preflight:
        preflight = _preflight(rows, viddiff_items)
        write_json(args.output / "preflight.json", preflight)
        return {"summary": preflight, "selection": selection}

    pending = [
        row
        for row in rows
        if args.rerun or not _checkpoint_path(args.output, row).exists()
    ]
    tokenizer = None
    model = None
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _load_model(
            args.model,
            args.adapter,
            load_in_4bit=not bool(getattr(args, "bf16_inference", False)),
        )
    episode_records: list[dict[str, Any]] = []
    accepted_sft: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        checkpoint = _checkpoint_path(args.output, row)
        if checkpoint.exists() and not args.rerun:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            episode = saved["episode"]
            elapsed = float(saved.get("elapsed_seconds", 0.0))
            resumed = True
        else:
            episode_seed = int(args.seed) + index - 1
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            environment, registry = _environment(row, _mode(row), viddiff_items)
            episode_started = time.perf_counter()
            episode = _run_episode(
                tokenizer=tokenizer,
                model=model,
                seed_row=row,
                environment=environment,
                tool_registry=registry,
                max_turns=args.max_turns,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                include_sft_data=True,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_generation_attempts=args.max_generation_attempts,
                constrain_eligible_tool_names=args.constrain_eligible_tool_names,
                require_evidence_adjudication=getattr(
                    args, "require_evidence_adjudication", False
                ),
                require_progressive_evidence_assessment=getattr(
                    args, "require_progressive_evidence_assessment", False
                ),
                prefer_natural_reasoning=getattr(
                    args, "prefer_natural_reasoning", False
                ),
            )
            elapsed = time.perf_counter() - episode_started
            resumed = False
        episode.update(
            {
                "source_case_id": _case_id(row),
                "case_name": f"{_case_id(row)}-{_mode(row)}",
                "mode": _mode(row),
                "seed_row_id": row["metadata"]["row_id"],
                "source_split": args.split,
                "trajectory_type": row["metadata"].get("trajectory_type"),
            }
        )
        quality = _audit_episode(_mode(row), episode, row, args.max_new_tokens)
        record = {
            "dataset": row["metadata"]["dataset"],
            "case_id": _case_id(row),
            "mode": _mode(row),
            "seed_row_id": row["metadata"]["row_id"],
            "elapsed_seconds": round(elapsed, 3),
            "resumed": resumed,
            "quality": quality,
            "style_diagnostics": _episode_style_diagnostics(episode),
            "episode": episode,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_json(checkpoint, record)
        episode_records.append(record)
        if quality["accepted_for_high_quality_sft"]:
            accepted_sft.extend(
                _accepted_sft(
                    episode,
                    quality,
                    row,
                    args.split,
                    rollout_id=(
                        f"sample-seed-{int(args.seed) + index - 1}"
                        if args.do_sample
                        else None
                    ),
                )
            )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(rows)}",
                    "case_id": _case_id(row),
                    "mode": _mode(row),
                    "terminal": episode.get("terminal_success"),
                    "score": quality["score"],
                    "accepted": quality["accepted_for_high_quality_sft"],
                    "seconds": round(elapsed, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    wall = time.perf_counter() - started
    accepted = sum(
        record["quality"]["accepted_for_high_quality_sft"] for record in episode_records
    )
    action_count = sum(len(record["episode"].get("turns") or []) for record in episode_records)
    generated_tokens = sum(
        int(turn.get("generated_tokens", 0))
        for record in episode_records
        for turn in record["episode"].get("turns") or []
    )
    prompt_tokens = sum(
        int(turn.get("prompt_tokens_kept", 0))
        for record in episode_records
        for turn in record["episode"].get("turns") or []
    )
    failure_dimensions = Counter(
        dimension
        for record in episode_records
        for dimension in record["quality"].get("failures", {})
    )
    generation_rejection_problems = Counter(
        str(problem)
        for record in episode_records
        for turn in record["episode"].get("turns") or []
        for attempt in turn.get("generation_attempt_audit") or []
        for problem in attempt.get("problems") or []
    )
    summary = {
        "episodes": len(episode_records),
        "accepted_episodes": accepted,
        "rejected_episodes": len(episode_records) - accepted,
        "acceptance_rate": round(accepted / len(episode_records), 4) if episode_records else 0.0,
        "terminal_successes": sum(
            bool(record["episode"].get("terminal_success")) for record in episode_records
        ),
        "episodes_with_rejection": sum(
            int(record["episode"].get("rejection_count", 0)) > 0
            for record in episode_records
        ),
        "actions": action_count,
        "generation_attempts": sum(
            int(turn.get("generation_attempts", 1))
            for record in episode_records
            for turn in record["episode"].get("turns") or []
            if turn.get("generated_tool") or turn.get("status") == "generation_contract_failure"
        ),
        "rejected_generation_attempts": sum(
            max(int(turn.get("generation_attempts", 1)) - 1, 0)
            for record in episode_records
            for turn in record["episode"].get("turns") or []
        ),
        "episodes_with_generation_retry": sum(
            any(
                int(turn.get("generation_attempts", 1)) > 1
                for turn in record["episode"].get("turns") or []
            )
            for record in episode_records
        ),
        "episodes_with_repeated_reasoning_scaffold": sum(
            int(record["style_diagnostics"]["repeated_scaffold_turns"] > 0)
            for record in episode_records
        ),
        "repeated_reasoning_scaffold_problem_count": sum(
            int(record["style_diagnostics"]["repeated_scaffold_problem_count"])
            for record in episode_records
        ),
        "generation_rejection_problem_counts": dict(
            generation_rejection_problems.most_common()
        ),
        "accepted_sft_rows": len(accepted_sft),
        "mean_score": round(
            sum(record["quality"]["score"] for record in episode_records)
            / len(episode_records),
            3,
        )
        if episode_records
        else 0.0,
        "minimum_score": min(
            (record["quality"]["score"] for record in episode_records), default=0
        ),
        "measured_episode_seconds": round(
            sum(record["elapsed_seconds"] for record in episode_records), 3
        ),
        "episode_seconds_p50": _nearest_rank(
            [float(record["elapsed_seconds"]) for record in episode_records], 0.50
        ),
        "episode_seconds_p95": _nearest_rank(
            [float(record["elapsed_seconds"]) for record in episode_records], 0.95
        ),
        "episode_seconds_max": round(
            max((float(record["elapsed_seconds"]) for record in episode_records), default=0.0),
            3,
        ),
        "current_run_wall_seconds": round(wall, 3),
        "seconds_per_episode": round(
            sum(record["elapsed_seconds"] for record in episode_records)
            / len(episode_records),
            3,
        )
        if episode_records
        else None,
        "seconds_per_action": round(
            sum(record["elapsed_seconds"] for record in episode_records) / action_count,
            3,
        )
        if action_count
        else None,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_generated_tokens_per_action": max(
            (
                int(turn.get("generated_tokens", 0))
                for record in episode_records
                for turn in record["episode"].get("turns") or []
            ),
            default=0,
        ),
        "max_prompt_tokens_per_action": max(
            (
                int(turn.get("prompt_tokens_kept", 0))
                for record in episode_records
                for turn in record["episode"].get("turns") or []
            ),
            default=0,
        ),
        "generation_seconds_p95_per_action": _nearest_rank(
            [
                float(turn["generation_seconds"])
                for record in episode_records
                for turn in record["episode"].get("turns") or []
                if turn.get("generation_seconds") is not None
            ],
            0.95,
        ),
        "generation_seconds_max_per_action": round(
            max(
                (
                    float(turn["generation_seconds"])
                    for record in episode_records
                    for turn in record["episode"].get("turns") or []
                    if turn.get("generation_seconds") is not None
                ),
                default=0.0,
            ),
            3,
        ),
        "failure_dimensions": dict(failure_dimensions),
        "by_dataset": _group_metrics(episode_records, "dataset"),
        "by_mode": _group_metrics(episode_records, "mode"),
    }
    sft_validation = validate_rows(accepted_sft)
    balanced_sft, diversity_balance = diversity_balance_rows(accepted_sft)
    balanced_validation = validate_rows(balanced_sft)
    summary["sft_validation_passed"] = sft_validation["all_passed"]
    summary["sft_validation_failed_rows"] = sft_validation["failed_rows"]
    summary["diversity_balanced_sft_rows"] = len(balanced_sft)
    summary["diversity_rows_removed"] = diversity_balance["removed_rows"]
    summary["diversity_balanced_validation_passed"] = balanced_validation[
        "all_passed"
    ]
    report = {
        "audit": "resumable_student_closed_loop_batch_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "adapter": str(args.adapter),
        "generation_mode": (
            "stochastic_local_model" if args.do_sample else "deterministic_greedy_local_model"
        ),
        "generation_parameters": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "base_seed": args.seed,
            "max_generation_attempts": args.max_generation_attempts,
        },
        "teacher_actions_used_during_generation": False,
        "oracle_or_verifier_used_during_generation": False,
        "rubric_threshold": "all critical dimensions and >=98/100",
        "summary": summary,
        "episodes": [
            {
                key: value
                for key, value in record.items()
                if key != "episode"
            }
            | {"tool_sequence": record["episode"].get("tool_sequence")}
            for record in episode_records
        ],
    }
    write_json(args.output / "batch_report.json", report)
    write_json(args.output / "sft_validation.json", sft_validation)
    write_json(
        args.output / "diversity_balanced_sft_validation.json",
        {
            "balancing": diversity_balance,
            "validation": balanced_validation,
        },
    )
    if sft_validation["all_passed"]:
        write_jsonl(args.output / "accepted_student_sft.jsonl", accepted_sft)
        write_jsonl(
            args.output / "diversity_balanced_student_sft.jsonl",
            balanced_sft if balanced_validation["all_passed"] else [],
        )
        write_jsonl(args.output / "quarantined_student_sft.jsonl", [])
    else:
        write_jsonl(args.output / "accepted_student_sft.jsonl", [])
        write_jsonl(args.output / "diversity_balanced_student_sft.jsonl", [])
        write_jsonl(args.output / "quarantined_student_sft.jsonl", accepted_sft)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "external_holdout"),
        default="test",
    )
    parser.add_argument("--seed-jsonl", type=Path)
    parser.add_argument(
        "--viddiff-candidates",
        action="append",
        type=Path,
        help=(
            "Load VidDiff runtime evidence only from these explicit student-visible "
            "candidate JSONL files. Repeat for multiple files."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", choices=("VideoSham", "VCSL", "VidDiffBench"))
    parser.add_argument("--mode", action="append", choices=("clean", "repair"))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit-cases-per-dataset", type=int)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--bf16-inference",
        action="store_true",
        help="Load the local model in BF16 instead of 4-bit NF4 for faster generation when VRAM permits.",
    )
    # A turn can need separate retries for length, method justification, and
    # semantic evidence scope. Four bounded drafts cover that observed composition
    # without executing or exporting any rejected draft.
    parser.add_argument("--max-generation-attempts", type=int, default=4)
    parser.add_argument(
        "--constrain-eligible-tool-names",
        action="store_true",
        help=(
            "Constrain only the native tool-call name tokens to the current eligible_tools; "
            "reasoning and arguments remain model-generated."
        ),
    )
    parser.add_argument(
        "--require-evidence-adjudication",
        action="store_true",
        help=(
            "Require an explicit adjudicate_evidence action before binding when "
            "conflicting active EvidenceAtoms compete for the same open fields."
        ),
    )
    parser.add_argument(
        "--require-progressive-evidence-assessment",
        action="store_true",
        help=(
            "Before adjudicate_evidence, require one stateful visible assessment "
            "for every competing EvidenceAtom. This also enables evidence adjudication."
        ),
    )
    parser.add_argument(
        "--prefer-natural-reasoning",
        action="store_true",
        help=(
            "Keep the first contract-valid draft as a fallback, then make one bounded "
            "attempt to replace repeated checker-like phrasing with natural case-specific prose."
        ),
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
