"""Consolidate VidDiff route portfolios into accepted SFT and route preferences."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from interactive_harness import EVIDENCE_INTEGRITY_TOOL_NAMES
from sft_v2_sidecar import compile_transition_sidecar
from unified_trajectory_rubric import load_saved_bundle
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, _restore_viddiff_harness_config


DEFAULT_RUNS = [
    ROOT / "outputs" / "utqr_v2_viddiff_canary" / "viddiff_adaptive_seed_20260818_n2_bon1",
    ROOT
    / "outputs"
    / "utqr_v2_viddiff_repair_coverage"
    / "viddiff_adaptive_seed_20260818_n1_bon1",
    ROOT
    / "outputs"
    / "utqr_v2_viddiff_reachable_remaining"
    / "viddiff_adaptive_seed_20260818_n5_bon4",
    ROOT / "outputs" / "utqr_v2_viddiff_unreachable" / "viddiff_adaptive_seed_20260818_n2_bon1",
    ROOT
    / "outputs"
    / "utqr_v2_viddiff_v4_hard_cases"
    / "viddiff_adaptive_seed_20260818_n2_bon1",
    ROOT
    / "outputs"
    / "utqr_v2_viddiff_v4_q7_causal"
    / "viddiff_adaptive_seed_20260818_n1_bon1",
]
DEFAULT_OUTPUT = ROOT / "outputs" / "utqr_v2_viddiff_10_consolidated"


def _prefer_new_case(existing: dict[str, Any], new: dict[str, Any]) -> bool:
    """Preserve a verified winner across iterative runs.

    A later diagnostic or failed retry must not overwrite an earlier accepted
    trajectory. If both runs passed, keep the higher-rubric artifact. When
    neither passed, the later row remains the most current diagnosis.
    """
    existing_passed = existing.get("status") == "passed"
    new_passed = new.get("status") == "passed"
    if existing_passed != new_passed:
        return new_passed
    if existing_passed:
        return float(new.get("rubric_score", 0.0)) > float(existing.get("rubric_score", 0.0))
    return True


def _resolve_artifact(attempt: dict[str, Any]) -> Path:
    recorded = Path(str(attempt["artifact_dir"]))
    for candidate in (recorded, ROOT / recorded):
        if (candidate / "llm_teacher_clean_episode.json").exists():
            return candidate
    raise FileNotFoundError(recorded)


def consolidate(run_roots: list[Path], output_root: Path) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for run_root in run_roots:
        run_root = Path(run_root)
        report = json.loads((run_root / "batch_report.json").read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            row = copy.deepcopy(case)
            row["source_run"] = str(run_root)
            case_id = str(case["case_id"])
            if case_id not in by_case or _prefer_new_case(by_case[case_id], row):
                by_case[case_id] = row

    cases = list(by_case.values())
    clean_sft: list[dict[str, Any]] = []
    repair_sft: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    preference_pairs: list[dict[str, Any]] = []
    selected: Counter[str] = Counter()
    indispensable: Counter[str] = Counter()
    for case in cases:
        attempts = case.get("attempts", [])
        if case.get("status") != "passed":
            continue
        selected_attempt = int(case.get("selected_attempt", 1))
        winner = next(item for item in attempts if int(item.get("attempt_index", 0)) == selected_attempt)
        artifact = _resolve_artifact(winner)
        bundle, _validation = load_saved_bundle(artifact)
        _restore_viddiff_harness_config(bundle)
        for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]:
            datum.setdefault("metadata", {})["consolidated_source_run"] = case["source_run"]
        clean_sft.extend(bundle["clean_turn_sft"])
        repair_sft.extend(bundle["repair_turn_sft"])
        sidecar.extend(compile_transition_sidecar(bundle))
        selected.update(case.get("selected_calls", {}))
        indispensable.update(case.get("indispensable_calls", {}))
        case["source_artifact_dir"] = str(artifact)
        for loser in attempts:
            if int(loser.get("attempt_index", 0)) == selected_attempt or loser.get("status") == "passed":
                continue
            preference_pairs.append(
                {
                    "case_id": case["case_id"],
                    "winner_route": winner.get("routing_variant"),
                    "winner_artifact_dir": str(artifact),
                    "winner_score": winner.get("rubric_score"),
                    "loser_route": loser.get("routing_variant"),
                    "loser_artifact_dir": str(_resolve_artifact(loser)),
                    "loser_score": loser.get("rubric_score"),
                    "loser_status": loser.get("status"),
                    "loser_failed_gates": loser.get("rubric_failed_gates", []),
                    "loser_failed_criteria": loser.get("rubric_failed_criteria", []),
                    "preference_basis": "same_case_executed_route; winner passed validator and UTQR; loser did not",
                }
            )

    passed = sum(case.get("status") == "passed" for case in cases)
    unreachable = sum(case.get("status") == "oracle_unreachable_candidate_set" for case in cases)
    attempted_cases = len(cases) - unreachable
    implemented = {item["name"] for item in VIDDIFF_TOOL_REGISTRY}
    report = {
        "run_id": "viddiff_utqr_v2_deduplicated_10_v4",
        "dataset": "VidDiffBench",
        "source_runs": [str(Path(path)) for path in run_roots],
        "summary": {
            "cases": len(cases),
            "passed": passed,
            "failed_after_cli_start": sum(
                case.get("status") in {"validation_failed", "rubric_rejected", "execution_failed"}
                for case in cases
            ),
            "oracle_unreachable_candidate_sets": unreachable,
            "overall_gold_rate": round(passed / len(cases), 4) if cases else 0.0,
            "attempted_case_gold_rate": round(passed / attempted_cases, 4) if attempted_cases else 0.0,
            "safe_to_compile_sft": passed == len(cases) and bool(cases),
            "safe_to_compile_accepted_sft": passed > 0,
        },
        "tool_telemetry": {
            "selected_calls": dict(selected),
            "indispensable_calls": dict(indispensable),
            "unused_implemented_tools": sorted(implemented - EVIDENCE_INTEGRITY_TOOL_NAMES - set(selected)),
            "conditional_integrity_tools": sorted(EVIDENCE_INTEGRITY_TOOL_NAMES),
            "selected_but_not_indispensable": {
                name: count - indispensable.get(name, 0)
                for name, count in selected.items()
                if count > indispensable.get(name, 0)
            },
        },
        "route_preference_pair_count": len(preference_pairs),
        "cases": cases,
    }
    output_root = Path(output_root)
    write_json(output_root / "batch_report.json", report)
    write_jsonl(output_root / "sft_clean_accepted.jsonl", clean_sft)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", repair_sft)
    write_jsonl(output_root / "transition_sidecar.jsonl", sidecar)
    write_jsonl(output_root / "route_preference_pairs.jsonl", preference_pairs)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.runs, args.output_root)["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
