"""Build a larger development recovery manifest without reusing current6 diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from recovery_case_manifest import build_manifest, preflight_job
from unannounced_repair_interventions import INTERVENTIONS


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_CURRENT6 = BASE / "cross_dataset_blind_canary_6episodes_v3/semantic_audit.json"
DEFAULT_OUTPUT = BASE / "recovery_expanded_manifest_v5/manifest.json"
DEFAULT_SOURCES = {
    "VideoSham": ROOT
    / "outputs/utqr_v2_videosham_17_consolidated/sft_clean_accepted.jsonl",
    "VCSL": ROOT
    / "outputs/scale30_vcsl_teacher_consolidated/sft_clean_accepted.jsonl",
    "VidDiffBench": [
        ROOT
        / "outputs/batch/viddiff_verified_seed_20260818_accepted_8/sft_clean_accepted.jsonl",
        ROOT / "outputs/scale30_viddiff_consolidated/sft_clean_accepted.jsonl",
    ],
}


def current6_case_ids(path: Path) -> set[str]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["case_id"]).lower()
        for item in report.get("episode_reports", [])
        if item.get("case_id")
    }


def build_expanded_manifest(current6_path: Path = DEFAULT_CURRENT6) -> dict[str, Any]:
    source_manifest = build_manifest(
        DEFAULT_SOURCES,
        exclude_case_ids=current6_case_ids(current6_path),
        enforce_entity_disjoint=True,
    )
    source_preflight = [preflight_job(job) for job in source_manifest["jobs"]]
    executable = {
        item["episode_id"] for item in source_preflight if item["passed"] is True
    }
    dataset_indices: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []
    for source_job in source_manifest["jobs"]:
        if source_job["episode_id"] not in executable:
            continue
        job = copy.deepcopy(source_job)
        dataset = str(job["dataset"])
        family = [item for item in INTERVENTIONS if item["dataset"] == dataset]
        job["episode_id"] = f"expanded-recovery-{len(jobs) + 1:03d}"
        job["intervention"] = copy.deepcopy(
            family[dataset_indices[dataset] % len(family)]
        )
        dataset_indices[dataset] += 1
        jobs.append(job)
    manifest = {
        **source_manifest,
        "jobs": jobs,
        "datasets": dict(Counter(job["dataset"] for job in jobs)),
        "source_candidates_considered": len(source_preflight),
        "availability_excluded": [
            {
                "dataset": item["dataset"],
                "case_id": item["case_id"],
                "reasons": item["reasons"],
            }
            for item in source_preflight
            if not item["passed"]
        ],
    }
    manifest.update(
        {
            "artifact": "causal_recovery_expanded_development_manifest_v5",
            "selection_note": (
                "Uses consolidated locally accepted checkpoints, excludes every case "
                "in the current6 semantic audit, and maximizes entity-disjoint cases "
                "within each dataset. Declared source order prefers verified checkpoints "
                "without reading oracle labels or rubric scores. Cases without a "
                "reconstructable local student-visible environment are removed before "
                "intervention families are reassigned. VidDiff reconstruction uses the "
                "explicit seed-20260818 development measurement catalog only; external "
                "holdout shards are not searched."
            ),
            "current6_reused": False,
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current6-audit", type=Path, default=DEFAULT_CURRENT6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_expanded_manifest(args.current6_audit)
    preflight = [preflight_job(job) for job in manifest["jobs"]]
    report = {
        "artifact": "causal_recovery_expanded_preflight_v5",
        "source_candidates_considered": manifest["source_candidates_considered"],
        "availability_excluded": manifest["availability_excluded"],
        "jobs": len(preflight),
        "datasets": manifest["datasets"],
        "current6_reused": manifest["current6_reused"],
        "passed": sum(item["passed"] for item in preflight),
        "failed": sum(not item["passed"] for item in preflight),
        "results": preflight,
    }
    write_json(args.output, manifest)
    write_json(args.output.parent / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
