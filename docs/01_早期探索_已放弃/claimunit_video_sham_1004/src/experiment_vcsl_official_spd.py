"""Evaluate VCSL's official ISC SPD model with verifier-only labels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from pipeline import ROOT, write_json
from vcsl_official_spd import DEFAULT_SPD_MODEL, DEFAULT_VCSL_ROOT, official_spd_localize
from vcsl_probe import DEFAULT_FEATURE_ROOT, segment_precision_recall


DEFAULT_SELECTION = (
    ROOT
    / "outputs/selection/vcsl_best_of_tree_seed_20260818_n30/selection_audit.json"
)
DEFAULT_MODEL = DEFAULT_SPD_MODEL
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/vcsl_official_spd_experiment_v1.json"
)


def _strict(metrics: dict[str, float]) -> bool:
    return min(float(metrics[name]) for name in ("precision", "recall", "f1")) >= 0.85


def run(
    selection_path: Path,
    feature_root: Path,
    vcsl_root: Path,
    model_path: Path,
    confidence: float,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    details = []
    for index, case in enumerate(selection["cases"], start=1):
        pair_id = str(case["pair_id"])
        query_id, reference_id = pair_id.split("-", 1)
        query = np.load(feature_root / f"{query_id}.npy")
        reference = np.load(feature_root / f"{reference_id}.npy")
        started = time.perf_counter()
        boxes = official_spd_localize(
            query @ reference.T,
            confidence=confidence,
            vcsl_root=vcsl_root,
            model_path=model_path,
        )
        elapsed = time.perf_counter() - started
        metrics = segment_precision_recall(boxes, case["oracle_boxes"])
        strict = _strict(metrics)
        details.append(
            {
                "pair_id": pair_id,
                "old_candidate_reachable": bool(case.get("accepted")),
                "official_spd_boxes": boxes,
                "official_spd_metrics": {
                    name: round(float(value), 6) for name, value in metrics.items()
                },
                "official_spd_strict_reachable": strict,
                "newly_recovered": strict and not bool(case.get("accepted")),
                "combined_reachable": strict or bool(case.get("accepted")),
                "inference_seconds": round(elapsed, 3),
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(selection['cases'])}",
                    "pair_id": pair_id,
                    "spd_strict": strict,
                    "newly_recovered": details[-1]["newly_recovered"],
                    "seconds": round(elapsed, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    old_reachable = sum(item["old_candidate_reachable"] for item in details)
    spd_reachable = sum(item["official_spd_strict_reachable"] for item in details)
    newly_recovered = sum(item["newly_recovered"] for item in details)
    combined = sum(item["combined_reachable"] for item in details)
    return {
        "experiment": "vcsl_official_isc_spd_v1",
        "source": {
            "repository": "official local VCSL repository",
            "model_reference": "VCSL data/spd_models.txt ISC release",
            "model_path": str(model_path),
            "official_default_confidence": confidence,
            "confidence_tuned_on_these_cases": False,
        },
        "label_boundary": {
            "localizer_inputs": "released ISC feature arrays only",
            "oracle_used_during_spd_execution": False,
            "oracle_used_after_execution_for_verifier_metrics": True,
            "student_or_teacher_artifact_contains_oracle": False,
        },
        "summary": {
            "cases": len(details),
            "old_reachable": old_reachable,
            "official_spd_strict_reachable": spd_reachable,
            "newly_recovered": newly_recovered,
            "combined_reachable": combined,
            "combined_reachability_rate": round(combined / len(details), 4),
            "mean_inference_seconds": round(
                statistics.mean(item["inference_seconds"] for item in details), 3
            ),
            "median_inference_seconds": round(
                statistics.median(item["inference_seconds"] for item in details), 3
            ),
        },
        "eligible_for_toolbox_integration": newly_recovered > 0,
        "cases": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--vcsl-root", type=Path, default=DEFAULT_VCSL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(
        args.selection,
        args.feature_root,
        args.vcsl_root,
        args.model,
        args.confidence,
    )
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
