"""Manifest-driven, failure-isolated causal-recovery batch compiler."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_recovery_reasoning_quality_v2 import audit_rows
from compile_causal_recovery_sft import compile_recovery_episodes
from pipeline import ROOT, write_json, write_jsonl
from recovery_case_manifest import (
    DEFAULT_MANIFEST,
    build_job,
    compile_spec,
    preflight_job,
)
from unannounced_repair_interventions import INTERVENTIONS


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_candidate_v1"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_jobs(
    jobs: list[dict[str, Any]], per_dataset_limit: int | None
) -> list[dict[str, Any]]:
    if per_dataset_limit is None:
        return list(jobs)
    counts: Counter[str] = Counter()
    selected = []
    for job in jobs:
        dataset = str(job["dataset"])
        if counts[dataset] >= per_dataset_limit:
            continue
        selected.append(job)
        counts[dataset] += 1
    return selected


def compile_batch(
    manifest: dict[str, Any],
    *,
    per_dataset_limit: int | None = None,
    dry_run: bool = False,
    intervention_portfolio: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    jobs = select_jobs(list(manifest.get("jobs") or []), per_dataset_limit)
    preflight = [preflight_job(job) for job in jobs]
    executable = {
        item["episode_id"] for item in preflight if item["passed"] is True
    }
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if not dry_run:
        for job in jobs:
            episode_id = str(job["episode_id"])
            if episode_id not in executable:
                failed = next(
                    item for item in preflight if item["episode_id"] == episode_id
                )
                results.append(
                    {
                        "episode_id": episode_id,
                        "dataset": job["dataset"],
                        "case_id": job["case_id"],
                        "status": "preflight_failed",
                        "reasons": failed["reasons"],
                    }
                )
                continue
            candidates = [copy.deepcopy(job["intervention"])]
            if intervention_portfolio:
                candidates.extend(
                    copy.deepcopy(item)
                    for item in INTERVENTIONS
                    if item["dataset"] == job["dataset"]
                    and item["id"] != job["intervention"]["id"]
                )
            attempts: list[dict[str, Any]] = []
            compiled = None
            for intervention in candidates:
                attempt_job = copy.deepcopy(job)
                attempt_job["intervention"] = intervention
                try:
                    spec = compile_spec(attempt_job)
                    compiled = compile_recovery_episodes(
                        {},
                        intervention_specs=[spec],
                        job_builder=build_job,
                    )
                    attempts.append(
                        {"intervention_id": intervention["id"], "status": "compiled"}
                    )
                    break
                except Exception as exc:
                    attempts.append(
                        {
                            "intervention_id": intervention["id"],
                            "status": "incompatible",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            if compiled is not None:
                current_rows, current_episodes, current_sidecars, episode_report = compiled
                rows.extend(current_rows)
                episodes.extend(current_episodes)
                sidecars.extend(current_sidecars)
                results.append(
                    {
                        "episode_id": episode_id,
                        "dataset": job["dataset"],
                        "case_id": job["case_id"],
                        "status": "compiled",
                        "turns": len(current_rows),
                        "selected_intervention_id": attempts[-1]["intervention_id"],
                        "intervention_attempts": attempts,
                        "terminal_success": episode_report["terminal_success"] == 1,
                        "fresh_evidence_consumed": (
                            episode_report["fresh_evidence_consumed"] == 1
                        ),
                        "independent_fields_preserved": (
                            episode_report["independent_fields_preserved"] == 1
                        ),
                        "counterfactuals_passed": all(
                            episode_report[name] == 1
                            for name in (
                                "clean_control_suppresses_challenge",
                                "defect_target_moves_with_atom",
                                "fresh_removal_blocks_repair",
                            )
                        ),
                        "step_value_passed": bool(
                            episode_report["step_value_audit"][
                                "all_steps_have_downstream_decision_role"
                            ]
                        ),
                    }
                )
            else:  # failure isolation is the purpose of this runner
                final = attempts[-1]
                results.append(
                    {
                        "episode_id": episode_id,
                        "dataset": job["dataset"],
                        "case_id": job["case_id"],
                        "status": "compile_failed",
                        "error_type": final["error_type"],
                        "error": final["error"],
                        "intervention_attempts": attempts,
                    }
                )
    status_counts = Counter(item["status"] for item in results)
    reasoning_audit = audit_rows(rows) if not dry_run else None
    qualification_blockers = [
        "closed_loop_student_gain_not_evaluated",
        "manifest_cases_are_development_not_sealed",
        "independent_blind_human_review_not_completed",
    ]
    if reasoning_audit is not None and not reasoning_audit["passed"]:
        qualification_blockers.insert(
            0, "local_reasoning_naturalness_audit_not_passed"
        )
    if not dry_run and any(
        item.get("status") == "compiled" and item.get("step_value_passed") is not True
        for item in results
    ):
        qualification_blockers.insert(0, "executed_step_without_downstream_role")
    report = {
        "artifact": "manifest_driven_causal_recovery_batch_v1",
        "artifact_status": (
            "dry_run" if dry_run else "structural_candidate_not_training_qualified"
        ),
        "manifest_status": manifest.get("status"),
        "jobs_selected": len(jobs),
        "datasets_selected": dict(Counter(job["dataset"] for job in jobs)),
        "preflight_passed": sum(item["passed"] for item in preflight),
        "preflight_failed": sum(not item["passed"] for item in preflight),
        "compiled_episodes": len(episodes),
        "sft_rows": len(rows),
        "status_counts": dict(status_counts),
        "intervention_portfolio_enabled": intervention_portfolio,
        "selected_intervention_counts": dict(
            Counter(
                item["selected_intervention_id"]
                for item in results
                if item.get("selected_intervention_id")
            )
        ),
        "portfolio_recovered_episodes": sum(
            len(item.get("intervention_attempts") or []) > 1
            and item.get("status") == "compiled"
            for item in results
        ),
        "terminal_success": sum(item.get("terminal_success") is True for item in results),
        "fresh_evidence_consumed": sum(
            item.get("fresh_evidence_consumed") is True for item in results
        ),
        "independent_fields_preserved": sum(
            item.get("independent_fields_preserved") is True for item in results
        ),
        "counterfactuals_passed": sum(
            item.get("counterfactuals_passed") is True for item in results
        ),
        "step_value_passed": sum(
            item.get("step_value_passed") is True for item in results
        ),
        "student_visible_leakage_episodes": sum(
            bool(item.get("student_visible_leakage_hits")) for item in episodes
        ),
        "reasoning_quality_audit": (
            {
                "passed": reasoning_audit["passed"],
                "metrics": reasoning_audit["metrics"],
                "checks": reasoning_audit["checks"],
            }
            if reasoning_audit is not None
            else None
        ),
        "training_qualified": False,
        "qualification_blockers": qualification_blockers,
        "preflight": preflight,
        "results": results,
    }
    return rows, episodes, sidecars, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-dataset-limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--intervention-portfolio", action="store_true")
    args = parser.parse_args()
    manifest = _load(args.manifest)
    rows, episodes, sidecars, report = compile_batch(
        manifest,
        per_dataset_limit=args.per_dataset_limit,
        dry_run=args.dry_run,
        intervention_portfolio=args.intervention_portfolio,
    )
    write_json(args.output_root / "manifest_used.json", copy.deepcopy(manifest))
    write_json(args.output_root / "compile_report.json", report)
    if not args.dry_run:
        write_json(
            args.output_root / "reasoning_quality_audit.json", audit_rows(rows)
        )
        write_jsonl(args.output_root / "recovery_sft_candidate.jsonl", rows)
        write_jsonl(args.output_root / "recovery_episodes.jsonl", episodes)
        write_jsonl(
            args.output_root / "private_intervention_sidecar.jsonl", sidecars
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
