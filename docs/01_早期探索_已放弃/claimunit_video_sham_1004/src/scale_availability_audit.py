"""Advisory inventory for real, locally executable scale expansion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import ROOT, write_json
from vcsl_probe import DEFAULT_FEATURE_ROOT, auto_route_localize, select_cases as select_vcsl
from viddiff_probe import available_surgery_rows
from videosham_batch import _rows as videosham_rows
from videosham_batch import capability_family, preflight_case


DEFAULT_OUTPUT = ROOT / "outputs" / "scale_readiness" / "local_data_availability.json"
DEFAULT_VIDDIFF_EXPANSION = ROOT / "outputs" / "scale30_viddiff_consolidated" / "batch_report.json"
DEFAULT_VCSL_SELECTION = (
    ROOT
    / "outputs"
    / "selection"
    / "vcsl_best_of_tree_seed_20260818_n30"
    / "selection_audit.json"
)


def audit(source_sample_size: int = 30) -> dict[str, Any]:
    videosham = videosham_rows()
    video_family_total = Counter(capability_family(row) for row in videosham)
    video_family_executable: Counter[str] = Counter()
    video_blockers: Counter[str] = Counter()
    for row in videosham:
        preflight = preflight_case(row)
        if preflight["cli_eligible"]:
            video_family_executable[preflight["capability_family"]] += 1
        else:
            video_blockers.update(preflight["preflight_reasons"])

    vcsl_cases = select_vcsl(20260818, source_sample_size)
    vcsl_methods: Counter[str] = Counter()
    vcsl_missing = 0
    for case in vcsl_cases:
        query = DEFAULT_FEATURE_ROOT / f"{case['query_id']}.npy"
        reference = DEFAULT_FEATURE_ROOT / f"{case['reference_id']}.npy"
        if not query.exists() or not reference.exists():
            vcsl_missing += 1
            continue
        method, _boxes, _diagnostics = auto_route_localize(np.load(query) @ np.load(reference).T)
        vcsl_methods[method] += 1

    expansion = (
        json.loads(DEFAULT_VIDDIFF_EXPANSION.read_text(encoding="utf-8"))
        if DEFAULT_VIDDIFF_EXPANSION.exists()
        else {"summary": {}}
    )
    vcsl_selection = (
        json.loads(DEFAULT_VCSL_SELECTION.read_text(encoding="utf-8"))
        if DEFAULT_VCSL_SELECTION.exists()
        else {"summary": {}}
    )
    vcsl_selected = int(vcsl_selection.get("summary", {}).get("accepted", 0))
    vcsl_rejected = int(vcsl_selection.get("summary", {}).get("rejected", 0))
    return {
        "audit_type": "advisory_inventory_not_runtime_gate",
        "target": "at least 30 real executable cases per dataset before mass-production claims",
        "VideoSham": {
            "metadata_cases": len(videosham),
            "locally_executable_cases": sum(video_family_executable.values()),
            "family_metadata_counts": dict(video_family_total),
            "family_locally_executable_counts": dict(video_family_executable),
            "blocker_counts": dict(video_blockers),
            "next_action": (
                "30-case local execution target reached; obtain more media only when broader family coverage is needed"
                if sum(video_family_executable.values()) >= 30
                else "obtain missing source/edited media; metadata rows alone cannot execute tools"
            ),
        },
        "VCSL": {
            "seeded_target_cases": source_sample_size,
            "locally_executable_feature_pairs": source_sample_size - vcsl_missing,
            "missing_feature_pairs": vcsl_missing,
            "auto_route_methods_on_available_pairs": dict(vcsl_methods),
            "strictly_reachable_after_localizer_portfolio": vcsl_selected,
            "measurement_unreachable_candidate_sets": vcsl_rejected,
            "next_action": (
                "run teacher only for selected student-visible localizations covered by separately recorded authorization"
                if vcsl_missing == 0 and vcsl_selected > 0
                else "run the 30-case local localization portfolio before teacher rollout"
                if vcsl_missing == 0
                else "obtain or extract features for missing pairs before teacher rollout"
            ),
        },
        "VidDiffBench": {
            "available_surgery_rows": len(available_surgery_rows()),
            "evaluated_expansion_cases": expansion.get("summary", {}).get("cases", 0),
            "gold_expansion_cases": expansion.get("summary", {}).get("passed", 0),
            "oracle_unreachable_candidate_sets": expansion.get("summary", {}).get(
                "oracle_unreachable_candidate_sets", 0
            ),
            "next_action": (
                "retain the nine failed predicate probes as unreachable diagnostics; register a new tool only "
                "after an independently validated measurement exceeds the existing baselines"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sample-size", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.source_sample_size)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
