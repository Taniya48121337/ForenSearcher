"""Re-run only preserved lattice failures with richer compiler diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from compile_recovery_batch import compile_batch
from pipeline import ROOT, write_json, write_jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_MANIFEST = BASE / "recovery_intervention_lattice_v1/manifest.json"
DEFAULT_PRIOR = BASE / "recovery_intervention_lattice_batch_v1/compile_report.json"
DEFAULT_OUTPUT = BASE / "recovery_intervention_lattice_failure_diagnostics_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    failed_ids = {
        str(item["episode_id"])
        for item in prior["results"]
        if item["status"] != "compiled"
    }
    diagnostic_manifest = copy.deepcopy(manifest)
    diagnostic_manifest["artifact"] = "causal_recovery_failed_variant_diagnostic_manifest_v2"
    diagnostic_manifest["status"] = "development_failure_diagnostic_not_sealed"
    diagnostic_manifest["jobs"] = [
        item for item in manifest["jobs"] if str(item["episode_id"]) in failed_ids
    ]
    rows, episodes, sidecars, report = compile_batch(
        diagnostic_manifest, intervention_portfolio=False
    )
    write_json(args.output_root / "manifest_used.json", diagnostic_manifest)
    write_json(args.output_root / "compile_report.json", report)
    write_jsonl(args.output_root / "recovery_sft_candidate.jsonl", rows)
    write_jsonl(args.output_root / "recovery_episodes.jsonl", episodes)
    write_jsonl(args.output_root / "private_intervention_sidecar.jsonl", sidecars)
    print(
        json.dumps(
            {
                "failed_ids_selected": len(failed_ids),
                "compiled": report["compiled_episodes"],
                "status_counts": report["status_counts"],
                "results": report["results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
