"""Build the fixed, label-blind next-stage UTQR-7 batch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from vcsl_probe import select_cases


VIDEOSHAM_CASE_IDS = [
    "videosham-4055",
    "videosham-4094",
    "videosham-4005",
    "videosham-4144",
    "videosham-1045",
    "videosham-0069",
    "videosham-4080",
]
VCSL_CASE_INDICES = [*range(11, 20), 21]
VIDDIFF_CANDIDATES = (
    ROOT
    / "outputs"
    / "selection"
    / "viddiff_external_holdout30_full_v1"
    / "candidate_measurements.student.jsonl"
)
VIDDIFF_AUDIT = VIDDIFF_CANDIDATES.with_name("selection_audit.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "utqr_v7_authorized_batch_20260824" / "manifest.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest() -> dict[str, Any]:
    vcsl_cases = select_cases(20260818, 30)
    selected_vcsl = [vcsl_cases[index] for index in VCSL_CASE_INDICES]
    vcsl_entities = [
        entity_id
        for case in selected_vcsl
        for entity_id in (case["query_id"], case["reference_id"])
    ]
    if len(vcsl_entities) != len(set(vcsl_entities)):
        raise ValueError("VCSL authorized scale subset is not entity-disjoint")

    viddiff_rows = _jsonl(VIDDIFF_CANDIDATES)
    selected_viddiff = viddiff_rows[:10]
    if len(selected_viddiff) != 10:
        raise ValueError("VidDiff frozen holdout does not contain ten rows")
    if any(len(row.get("candidate_tools") or []) < 3 for row in selected_viddiff):
        raise ValueError("VidDiff batch must retain the full three-route candidate set")

    return {
        "artifact": "utqr_v7_authorized_label_blind_batch_manifest_v1",
        "created_date": "2026-08-24",
        "selection_policy": {
            "case_selection_uses_oracle": False,
            "oracle_reachability_prescreen": False,
            "retain_all_execution_failures": True,
            "teacher_receives_oracle": False,
            "teacher_receives_verifier_output": False,
        },
        "datasets": {
            "VideoSham": {
                "requested_cases": len(VIDEOSHAM_CASE_IDS),
                "case_ids": list(VIDEOSHAM_CASE_IDS),
                "shortfall_from_ten": 10 - len(VIDEOSHAM_CASE_IDS),
                "shortfall_reason": "Only seven VideoSham cases are authorized for teacher transfer.",
                "routing": "student-visible modality routing",
            },
            "VCSL": {
                "requested_cases": len(VCSL_CASE_INDICES),
                "seed": 20260818,
                "case_indices": list(VCSL_CASE_INDICES),
                "pair_ids": [case["pair_id"] for case in selected_vcsl],
                "entity_disjoint": True,
                "routing": "auto_label_blind",
                "selection_path": None,
            },
            "VidDiffBench": {
                "requested_cases": len(selected_viddiff),
                "frozen_source_rows": 30,
                "sample_keys": [row["sample_key"] for row in selected_viddiff],
                "candidate_tool_counts": [
                    len(row["candidate_tools"]) for row in selected_viddiff
                ],
                "adaptive_path": str(VIDDIFF_CANDIDATES.relative_to(ROOT)),
                "audit_path": str(VIDDIFF_AUDIT.relative_to(ROOT)),
                "routing_variants": ["uniform_first", "global_first", "action_first"],
                "skip_oracle_unreachable": False,
            },
        },
        "admission": {
            "base_trajectory_requires_local_validation": True,
            "causal_packet_requires_utqr_v7": True,
            "blind_judges": 2,
            "export_rejected_tool_calls_as_positive_sft": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest()
    write_json(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
