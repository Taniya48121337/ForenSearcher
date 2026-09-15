"""Consolidate case artifacts from interrupted or iterative batch runs.

Later run roots override earlier attempts for the same case.  Only cases whose
saved validation report passes are copied into aggregate SFT files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def consolidate(run_roots: list[Path], output_root: Path) -> dict[str, Any]:
    latest: dict[str, tuple[dict[str, Any], Path]] = {}
    for run_root in run_roots:
        batch_path = run_root / "batch_report.json"
        if not batch_path.exists():
            continue
        for row in _json(batch_path).get("cases", []):
            sample_key = row["case_id"].removeprefix("viddiff-")
            latest[row["case_id"]] = (row, run_root / "cases" / sample_key)

    clean: list[dict[str, Any]] = []
    repair: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for case_id in sorted(latest):
        row, case_dir = latest[case_id]
        validation_path = case_dir / "llm_teacher_validation_report.json"
        validation = _json(validation_path) if validation_path.exists() else {"passed": False, "checks": {}}
        accepted = row.get("status") == "passed" and validation.get("passed") is True
        failed_checks = row.get("failed_checks", [])
        if accepted:
            clean.extend(_jsonl(case_dir / "llm_teacher_sft_clean.jsonl"))
            repair.extend(_jsonl(case_dir / "llm_teacher_sft_repair.jsonl"))
            sidecar.extend(_jsonl(case_dir / "llm_teacher_transition_sidecar.jsonl"))
            disposition = "accepted_for_sft"
        elif row.get("status") == "oracle_unreachable_candidate_set":
            disposition = "oracle_unreachable_candidate_set"
        elif failed_checks == ["oracle_direction_match"]:
            disposition = "oracle_rejected_after_valid_process"
        else:
            disposition = "process_or_multi_check_rejected"
        cases.append(
            {
                **row,
                "source_case_dir": str(case_dir),
                "consolidated_disposition": disposition,
            }
        )

    serialized = json.dumps({"clean": clean, "repair": repair}, ensure_ascii=False)
    leak_tokens = [
        token
        for token in ("oracle_direction", "oracle_match", "differences_gt", "retrieval_frames")
        if token in serialized
    ]
    accepted_cases = sum(row["consolidated_disposition"] == "accepted_for_sft" for row in cases)
    report = {
        "dataset": "VidDiffBench",
        "input_policy": "visible_observation_adaptive_branching_v2",
        "case_count": len(cases),
        "summary": {
            "accepted_cases": accepted_cases,
            "oracle_unreachable_candidate_sets": sum(
                row["consolidated_disposition"] == "oracle_unreachable_candidate_set" for row in cases
            ),
            "oracle_rejected_after_valid_process": sum(
                row["consolidated_disposition"] == "oracle_rejected_after_valid_process" for row in cases
            ),
            "process_or_multi_check_rejected": sum(
                row["consolidated_disposition"] == "process_or_multi_check_rejected" for row in cases
            ),
            "clean_sft_turns": len(clean),
            "repair_sft_turns": len(repair),
            "transition_sidecar_rows": len(sidecar),
            "oracle_leak_tokens": leak_tokens,
            "safe_to_compile_accepted_sft": accepted_cases > 0 and not leak_tokens,
            "all_cases_passed": accepted_cases == len(cases) and bool(cases),
        },
        "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "consolidated_report.json", report)
    write_jsonl(output_root / "sft_clean_accepted.jsonl", clean)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", repair)
    write_jsonl(output_root / "transition_sidecar.jsonl", sidecar)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = consolidate(args.run_roots, args.output_root)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
