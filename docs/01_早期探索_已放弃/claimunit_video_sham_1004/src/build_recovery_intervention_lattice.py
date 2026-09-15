"""Expand one clean checkpoint per case into a causal intervention lattice.

This is training-data augmentation, not an evaluation split.  Every variant keeps
the same clean checkpoint and changes only the private, dataset-compatible
unannounced intervention.  The resulting same-case variants are useful hard
contrasts, while downstream sampling must normalize mass by case.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from recovery_case_manifest import preflight_job
from unannounced_repair_interventions import INTERVENTIONS


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_SOURCE = (
    BASE / "recovery_batch_candidate_v32_step_value55/manifest_used.json"
)
DEFAULT_OUTPUT = BASE / "recovery_intervention_lattice_v1/manifest.json"


def build_lattice(source: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for intervention in INTERVENTIONS:
        families[str(intervention["dataset"])].append(intervention)

    jobs: list[dict[str, Any]] = []
    source_jobs = list(source.get("jobs") or [])
    for source_index, source_job in enumerate(source_jobs, start=1):
        dataset = str(source_job["dataset"])
        variants = families[dataset]
        for variant_index, intervention in enumerate(variants, start=1):
            job = copy.deepcopy(source_job)
            job["episode_id"] = (
                f"lattice-recovery-{source_index:03d}-{variant_index:02d}"
            )
            job["intervention"] = copy.deepcopy(intervention)
            job["lattice"] = {
                "source_episode_id": str(source_job["episode_id"]),
                "variant_index": variant_index,
                "variant_count_for_case": len(variants),
                "case_mass_policy": "normalize_after_compile",
            }
            jobs.append(job)

    dataset_case_counts = Counter(str(job["dataset"]) for job in source_jobs)
    dataset_variant_counts = Counter(str(job["dataset"]) for job in jobs)
    intervention_counts = Counter(str(job["intervention"]["id"]) for job in jobs)
    return {
        **copy.deepcopy(source),
        "artifact": "causal_recovery_intervention_lattice_manifest_v1",
        "status": "development_augmentation_not_sealed",
        "jobs": jobs,
        "datasets": dict(dataset_variant_counts),
        "lattice": {
            "source_manifest_artifact": source.get("artifact"),
            "source_cases": len(source_jobs),
            "variants": len(jobs),
            "cases_by_dataset": dict(dataset_case_counts),
            "variants_by_dataset": dict(dataset_variant_counts),
            "variants_by_intervention": dict(intervention_counts),
            "construction": (
                "Cartesian product of each executable clean development checkpoint "
                "and every dataset-compatible unannounced intervention."
            ),
            "student_visible_intervention_labels": False,
            "oracle_or_verifier_used_for_assignment": False,
            "evaluation_claim": "none; same-case variants are SFT augmentation only",
            "required_sampling_policy": "exact_case_mass_normalization",
        },
        "selection_note": (
            "Preserves the source development cases and expands only private, "
            "dataset-compatible intervention operators. No new data, oracle label, "
            "verifier score, or model result is used for variant assignment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    manifest = build_lattice(source)
    preflight = [preflight_job(job) for job in manifest["jobs"]]
    report = {
        "artifact": "causal_recovery_intervention_lattice_preflight_v1",
        "manifest": str(args.output),
        "jobs": len(preflight),
        "datasets": manifest["datasets"],
        "passed": sum(bool(item["passed"]) for item in preflight),
        "failed": sum(not bool(item["passed"]) for item in preflight),
        "results": preflight,
    }
    write_json(args.output, manifest)
    write_json(args.output.parent / "preflight.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
