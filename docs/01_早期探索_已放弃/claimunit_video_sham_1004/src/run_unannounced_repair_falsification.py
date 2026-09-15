"""Run ten unannounced evidence-integrity interventions with the local student."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from evaluate_student_context_arm import _load_model
from interactive_harness import with_evidence_integrity_tools
from pipeline import ROOT, write_json
from run_student_closed_loop_canary import (
    _run_episode,
    _vcsl_environment,
    _video_environment,
    _viddiff_item,
)
from shadow_execute_generated_actions import _jsonl, _restore_harness
from train_student_context_arm import DEFAULT_MODEL
from unannounced_repair_interventions import INTERVENTIONS, apply_intervention, score_episode
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
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "unannounced_repairs/current6_v1"
)

STOP_ROWS = {
    "VideoSham": "VideoSham:videosham-4189:LLM-TRJ-4189-CLEAN-001:clean:6",
    "VCSL": (
        "VCSL:vcsl-06feccaaee94443a8ddfcd59d7938bad-808d18d803c649e0badaf17caff18f5c:"
        "LLM-TRJ-VCSL-06FECCAAEE94443A8DDFCD59D7938BAD-808D18D803C649E0BADAF17CAFF18F5C-CLEAN-001:clean:6"
    ),
    "VidDiffBench": "VidDiffBench:viddiff-surgery_70:LLM-TRJ-VIDDIFF-SURGERY_70-CLEAN-001:clean:7",
}

INTEGRITY_INSTRUCTION = (
    "Before stopping, independently check whether the active evidence content, current "
    "bindings, and committed claim are mutually consistent. Use any ordinary tool needed. "
    "Do not assume that active status alone proves semantic validity; if support is insufficient, "
    "leave the affected decision open or gather evidence instead of hard-binding."
)


def _builder(dataset: str, rows: dict[str, dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    if dataset == "VideoSham":
        return _video_environment(rows), MULTIFAMILY_TOOL_REGISTRY
    if dataset == "VCSL":
        return _vcsl_environment(), VCSL_TOOL_REGISTRY
    return VidDiffAdaptiveMeasurementEnvironment(_viddiff_item("surgery_70")), VIDDIFF_TOOL_REGISTRY


def _active_registry(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return with_evidence_integrity_tools(registry)


def preflight(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results = []
    for spec in INTERVENTIONS:
        environment, registry = _builder(spec["dataset"], rows)
        harness = _restore_harness(rows[STOP_ROWS[spec["dataset"]]])
        harness.environment = environment
        harness.tool_registry = _active_registry(registry)
        metadata = apply_intervention(harness, spec)
        state = harness.render_state()
        visible_text = json.dumps(state, ensure_ascii=False)
        fields = [
            field
            for claim in state.get("candidate_claims") or []
            for field in (claim.get("fields") or {}).values()
        ]
        checks = {
            "multiple_tools_visible": len(state.get("eligible_tools") or []) > 1,
            "no_dropped_evidence_hint": not state.get("dropped_evidence_ids"),
            "no_must_reopen_hint": not ((state.get("action_constraints") or {}).get("must_reopen_before_rebinding")),
            "no_restoration_candidate": not ((state.get("state_action_affordances") or {}).get("repair_restoration_candidates")),
            "no_unsupported_or_conflicted_status": all(field.get("evidence_status") == "supported" for field in fields),
            "intervention_id_not_visible": spec["id"] not in visible_text,
            "operator_name_not_visible": spec["operator"] not in visible_text,
        }
        results.append(
            {
                "intervention_id": spec["id"],
                "dataset": spec["dataset"],
                "eligible_tools": state.get("eligible_tools"),
                "checks": checks,
                "all_checks": all(checks.values()),
                "verifier_metadata": metadata,
            }
        )
    return {
        "artifact": "unannounced_repair_preflight_v1",
        "interventions": len(results),
        "all_pass": all(item["all_checks"] for item in results),
        "results": results,
    }


def _manifest(args: argparse.Namespace, preflight_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": "unannounced_repair_manifest_v1",
        "created_before_model_results": True,
        "cases_are_diagnostic_not_sealed": True,
        "model": str(args.model),
        "adapter": str(args.adapter),
        "decoding": {
            "greedy": True,
            "load_in_4bit": True,
            "max_turns": args.max_turns,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "max_generation_attempts": args.max_generation_attempts,
        },
        "visible_policy_arm": "real_evidence_plus_reduced_control_open_audit_surface",
        "integrity_instruction": INTEGRITY_INSTRUCTION,
        "policy_never_sees": [
            "intervention_id",
            "operator",
            "affected_fields_verifier_only",
            "target_tools_verifier_only",
            "repair_restoration_candidates",
            "must_reopen_before_rebinding",
            "dropped_evidence_ids",
        ],
        "success_rule": (
            "Before stop, execute a verifier-registered semantically targeted remeasurement, "
            "assessment/adjudication, or field reopen. Immediate stop is failure. Reopen localization "
            "and over-invalidation are reported separately. For U10, nonterminal abstention is necessary."
        ),
        "interventions": copy.deepcopy(list(INTERVENTIONS)),
        "preflight": preflight_report,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    row_list = _jsonl(args.sft)
    rows = {str(item["metadata"]["row_id"]): item for item in row_list}
    preflight_report = preflight(rows)
    write_json(args.output_dir / "preflight.json", preflight_report)
    manifest = _manifest(args, preflight_report)
    write_json(args.output_dir / "MANIFEST.json", manifest)
    if args.dry_run:
        return {"status": "dry_run", "manifest": manifest}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.adapter, load_in_4bit=True)
    results = []
    for spec in INTERVENTIONS:
        episode_path = args.output_dir / "episodes" / f"{spec['id']}.json"
        if episode_path.exists():
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
        else:
            environment, registry = _builder(spec["dataset"], rows)

            def setup(harness: Any, current_spec: dict[str, Any] = spec) -> dict[str, Any]:
                return apply_intervention(harness, current_spec)

            def save_partial(checkpoint: dict[str, Any], current_spec: dict[str, Any] = spec) -> None:
                turn = int(checkpoint.get("turn_count") or 0)
                write_json(
                    args.output_dir / "partial_turns" / f"{current_spec['id']}_turn_{turn:03d}.json",
                    checkpoint,
                )

            started = time.perf_counter()
            episode = _run_episode(
                tokenizer=tokenizer,
                model=model,
                seed_row=rows[STOP_ROWS[spec["dataset"]]],
                environment=environment,
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
                policy_view_arm="evidence_only",
                harness_setup_callback=setup,
                task_integrity_instruction=INTEGRITY_INSTRUCTION,
                turn_checkpoint_callback=save_partial,
            )
            episode["episode_seconds"] = round(time.perf_counter() - started, 3)
            write_json(episode_path, episode)
        score = score_episode(episode, spec)
        results.append({"spec": copy.deepcopy(spec), "score": score, "episode": episode})
        write_json(args.output_dir / "scores" / f"{spec['id']}.json", score)
        print(
            json.dumps(
                {
                    "id": spec["id"],
                    "dataset": spec["dataset"],
                    "detected": score["detected_before_stop"],
                    "first_tool": score["first_tool"],
                    "terminal": score["terminal_success_raw"],
                    "turns": score["turn_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        write_json(args.output_dir / "report_running.json", _report(results, complete=False))
    return _report(results, complete=True)


def _report(results: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    scores = [item["score"] for item in results]
    return {
        "artifact": "unannounced_repair_falsification_v1",
        "status": "complete" if complete else "running",
        "summary": {
            "episodes": len(scores),
            "detected_before_stop": sum(item["detected_before_stop"] for item in scores),
            "semantic_issue_named": sum(item["semantic_issue_named"] for item in scores),
            "localized_reopen": sum(item["localized_reopen"] for item in scores),
            "stopped_without_targeted_action": sum(item["stopped_without_targeted_action"] for item in scores),
            "correct_nonterminal_abstention": sum(item["correct_nonterminal_abstention"] for item in scores),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-generation-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(args)
    output_name = "dry_run_report.json" if args.dry_run else "report.json"
    write_json(args.output_dir / output_name, report)
    print(json.dumps(report.get("summary") or {"status": report["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
