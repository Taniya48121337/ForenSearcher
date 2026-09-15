"""Consolidate base and targeted VCSL runs into one deduplicated SFT export."""

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
from vcsl_batch import VCSL_TOOL_REGISTRY, _restore_vcsl_harness_config, _telemetry


DEFAULT_RUNS = [
    ROOT / "outputs" / "utqr_v2_vcsl_10" / "vcsl_seed_20260818_n10_bon1",
    ROOT
    / "outputs"
    / "utqr_v2_vcsl_router_repair"
    / "vcsl_seed_20260818_n3_idx4-7-9_bon1",
    ROOT
    / "outputs"
    / "vcsl_teacher_save_canary"
    / "vcsl_seed_20260818_n1_idx1_11to11_selected",
    ROOT
    / "outputs"
    / "scale30_vcsl_teacher_incremental_v2"
    / "vcsl_seed_20260818_n11_idx11_12to29_selected",
    ROOT / "o" / "vcsl_seed_20260818_n9_idx9_12to29_selected",
]
DEFAULT_OUTPUT = ROOT / "outputs" / "scale30_vcsl_teacher_consolidated"


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
            by_case[str(case["case_id"])] = row

    cases = list(by_case.values())
    clean_sft: list[dict[str, Any]] = []
    repair_sft: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    selected: Counter[str] = Counter()
    indispensable: Counter[str] = Counter()
    for case in cases:
        if case.get("status") != "passed":
            continue
        selected_attempt = int(case.get("selected_attempt", 1))
        attempt = next(
            item for item in case.get("attempts", []) if int(item.get("attempt_index", 0)) == selected_attempt
        )
        artifact = _resolve_artifact(attempt)
        bundle, _validation = load_saved_bundle(artifact)
        _restore_vcsl_harness_config(bundle)
        telemetry = _telemetry(bundle)
        for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]:
            datum.setdefault("metadata", {})["consolidated_source_run"] = case["source_run"]
        clean_sft.extend(bundle["clean_turn_sft"])
        repair_sft.extend(bundle["repair_turn_sft"])
        sidecar.extend(compile_transition_sidecar(bundle))
        selected.update(telemetry["selected"])
        indispensable.update(telemetry["indispensable"])
        case["source_artifact_dir"] = str(artifact)

    passed = sum(case.get("status") == "passed" for case in cases)
    implemented = {item["name"] for item in VCSL_TOOL_REGISTRY}
    report = {
        "run_id": "vcsl_utqr_v2_deduplicated_scale30",
        "dataset": "VCSL",
        "source_runs": [str(Path(path)) for path in run_roots],
        "summary": {
            "cases": len(cases),
            "passed": passed,
            "rubric_rejected": sum(case.get("status") == "rubric_rejected" for case in cases),
            "gold_rate": round(passed / len(cases), 4) if cases else 0.0,
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
        "cases": cases,
    }
    output_root = Path(output_root)
    write_json(output_root / "batch_report.json", report)
    write_jsonl(output_root / "sft_clean_accepted.jsonl", clean_sft)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", repair_sft)
    write_jsonl(output_root / "transition_sidecar.jsonl", sidecar)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = consolidate(args.runs, args.output_root)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
