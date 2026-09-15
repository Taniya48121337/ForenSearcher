"""Summarize what the cross-dataset canary proves and what remains pending."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import write_json


FORBIDDEN_REVIEW_TERMS = (
    "oracle",
    "verifier",
    "local_path",
    "video_path",
    "file://",
    "http://",
    "https://",
    "differences_gt",
    "ground_truth",
    "hidden label",
    "prompt",
    "rubric",
    "checker",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _review_complete(review_form: dict[str, Any]) -> bool:
    scores = review_form.get("dimension_scores") or {}
    return (
        bool(scores)
        and all(isinstance(score, int) and 1 <= score <= 5 for score in scores.values())
        and isinstance(review_form.get("hard_fail"), bool)
        and isinstance(review_form.get("keep_for_sft"), bool)
    )


def audit(
    package_report_path: Path,
    semantic_audit_path: Path,
    blind_packet_path: Path,
    ablation_path: Path,
) -> dict[str, Any]:
    package = _load_json(package_report_path)
    semantic = _load_json(semantic_audit_path)
    packet = _load_jsonl(blind_packet_path)
    ablation = _load_json(ablation_path)

    validation = package.get("validation") or {}
    dataset_rows = validation.get("dataset_rows") or {}
    mode_rows = validation.get("mode_rows") or {}
    packet_text = "\n".join(json.dumps(row, ensure_ascii=False).lower() for row in packet)
    forbidden_hits = [term for term in FORBIDDEN_REVIEW_TERMS if term in packet_text]
    completed_reviews = sum(_review_complete(row.get("review_form") or {}) for row in packet)
    pending_reviews = len(packet) - completed_reviews

    required_datasets = {"VidDiffBench", "VideoSham", "VCSL"}
    required_modes = {"clean", "repair"}
    package_checks = {
        "package_ready_for_training": package.get("ready_for_training") is True,
        "row_validation_passed": validation.get("all_passed") is True,
        "privacy_leaks_zero": not package.get("privacy_leaks"),
        "duplicate_row_ids_zero": not validation.get("duplicate_row_ids"),
        "exact_reasoning_duplicates_zero": validation.get("exact_reasoning_duplicates") == 0,
        "three_datasets_present": required_datasets.issubset(dataset_rows),
        "clean_and_repair_present": required_modes.issubset(mode_rows),
        "semantic_audit_coverage_complete": not (
            (package.get("semantic_audit") or {}).get("coverage_missing")
        ),
    }
    semantic_checks = {
        "all_saved_turns_pass_current_contract": semantic.get("all_passed") is True,
        "episode_count_matches_packet": semantic.get("episodes") == len(packet),
        "turn_count_matches_package": semantic.get("turns") == package.get("rows"),
    }
    review_packet_checks = {
        "nonempty": bool(packet),
        "all_ids_blinded": all(str(row.get("blind_id", "")).startswith("BLIND-") for row in packet),
        "identity_fields_absent": all(
            "dataset" not in row and "case_id" not in row and "source_episode" not in row
            for row in packet
        ),
        "forbidden_term_hits": forbidden_hits,
        "privacy_scan_passed": not forbidden_hits,
    }
    generic = ablation.get("generic_routes_only") or {}
    typed = ablation.get("typed_proof_route") or {}
    ablation_checks = {
        "decisions_frozen_before_oracle_load": (
            ablation.get("decisions_frozen_before_oracle_load") is True
        ),
        "external_service_calls_zero": ablation.get("external_service_calls") == 0,
        "typed_route_changed_decision": generic.get("decision") != typed.get("decision"),
        "typed_route_improved_posthoc_correctness": (
            generic.get("posthoc_correct") is False and typed.get("posthoc_correct") is True
        ),
        "scope_is_mechanistic_not_population": "not a population" in str(
            ablation.get("interpretation_limit", "")
        ).lower(),
    }

    machine_canary_ready = all(package_checks.values()) and all(semantic_checks.values()) and all(
        value is True for key, value in review_packet_checks.items() if key != "forbidden_term_hits"
    ) and all(ablation_checks.values())
    return {
        "audit": "cross_dataset_canary_readiness_v1",
        "machine_canary_ready": machine_canary_ready,
        "ready_for_small_scale_training_canary": machine_canary_ready,
        "ready_to_claim_population_improvement": False,
        "summary": {
            "episodes": semantic.get("episodes"),
            "turns": package.get("rows"),
            "datasets": dataset_rows,
            "modes": mode_rows,
            "distinct_tool_actions": len(validation.get("tool_rows") or {}),
            "human_reviews_completed": completed_reviews,
            "human_reviews_pending": pending_reviews,
        },
        "package_checks": package_checks,
        "semantic_checks": semantic_checks,
        "review_packet_checks": review_packet_checks,
        "typed_ablation_checks": ablation_checks,
        "remaining_evidence_gaps": [
            "Complete the six blinded human review forms before calling the sample human-audited.",
            "Run the typed-versus-generic ablation on a larger untouched holdout before claiming a population-level gain.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--typed-ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.package_report,
        args.semantic_audit,
        args.blind_packet,
        args.typed_ablation,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["machine_canary_ready"] else 1)


if __name__ == "__main__":
    main()
