"""Consolidate deduplicated VideoSham canaries into one accepted SFT export."""

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
from videosham_batch import MULTIFAMILY_SCHEMA, MULTIFAMILY_TOOL_REGISTRY


DEFAULT_RUNS = [
    ROOT / "outputs" / "utqr_v2_videosham_canary" / "videosham_seed_20260818_n3",
    ROOT / "outputs" / "utqr_v2_videosham_canary" / "videosham_cases_2004_n1",
    ROOT
    / "outputs"
    / "utqr_v2_videosham_10_remaining"
    / "videosham_cases_4112-2010-4189-4142-4224-4039_n6",
    ROOT
    / "outputs"
    / "batch"
    / "videosham_cases_4055-4094-4005-4144-1045-0069-4080_n7",
]
DEFAULT_OUTPUT = ROOT / "outputs" / "utqr_v2_videosham_17_consolidated"


def _restore_config(bundle: dict[str, Any]) -> None:
    bundle["harness_config"] = {
        "claim_schema": copy.deepcopy(MULTIFAMILY_SCHEMA),
        "tool_registry": copy.deepcopy(MULTIFAMILY_TOOL_REGISTRY),
        "target_fields": list(MULTIFAMILY_SCHEMA["target_fields"]),
        "invariant_fields": list(MULTIFAMILY_SCHEMA["invariant_fields"]),
    }


def consolidate(run_roots: list[Path], output_root: Path) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for run_root in run_roots:
        run_root = Path(run_root)
        report = json.loads((run_root / "batch_report.json").read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            if case.get("status") == "preflight_blocked":
                continue
            row = copy.deepcopy(case)
            row["source_run"] = str(run_root)
            row["source_artifact_dir"] = str(run_root / "cases" / case["case_id"] / "outputs")
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
        bundle, _validation = load_saved_bundle(Path(case["source_artifact_dir"]))
        _restore_config(bundle)
        for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]:
            datum.setdefault("metadata", {})["consolidated_source_run"] = case["source_run"]
        clean_sft.extend(bundle["clean_turn_sft"])
        repair_sft.extend(bundle["repair_turn_sft"])
        sidecar.extend(compile_transition_sidecar(bundle))
        telemetry = case.get("tool_telemetry", {})
        selected.update(telemetry.get("selected_calls", {}))
        indispensable.update(telemetry.get("indispensable_calls", {}))

    passed = sum(case.get("status") == "passed" for case in cases)
    implemented = {item["name"] for item in MULTIFAMILY_TOOL_REGISTRY}
    report = {
        "run_id": f"videosham_utqr_v2_deduplicated_{len(cases)}",
        "dataset": "VideoSham",
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
    print(json.dumps(consolidate(args.runs, args.output_root)["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
