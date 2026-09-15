"""Attach label-blind dataset/stage/intervention-balanced sampling weights."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio"
DEFAULT_OUTPUT = BASE / "recovery_sft_training_mix_v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 6)


def build_mix(
    rows: list[dict[str, Any]],
    compile_report: dict[str, Any],
    *,
    normalize_case_mass: bool = False,
    case_tempering_power: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < case_tempering_power <= 1:
        raise ValueError("case_tempering_power must be in (0, 1]")
    reported_intervention = {
        str(item["episode_id"]): str(item["selected_intervention_id"])
        for item in compile_report.get("results") or []
        if item.get("status") == "compiled" and item.get("selected_intervention_id")
    }
    stage_counts = Counter(
        (
            str(row["metadata"]["dataset"]),
            str(row["metadata"]["causal_recovery_stage"]),
        )
        for row in rows
    )
    episode_dataset = {
        str(row["metadata"]["episode_id"]): str(row["metadata"]["dataset"])
        for row in rows
    }
    selected_intervention = {
        episode_id: intervention
        for episode_id, intervention in reported_intervention.items()
        if episode_id in episode_dataset
    }
    missing_intervention = sorted(set(episode_dataset) - set(selected_intervention))
    if missing_intervention:
        raise ValueError(
            "selected rows are missing compiled intervention metadata: "
            + ", ".join(missing_intervention[:5])
        )
    intervention_episode_counts = Counter(
        (episode_dataset[episode_id], intervention)
        for episode_id, intervention in selected_intervention.items()
    )

    raw_weights: list[float] = []
    datasets: list[str] = []
    for row in rows:
        metadata = row["metadata"]
        dataset = str(metadata["dataset"])
        stage = str(metadata["causal_recovery_stage"])
        episode_id = str(metadata["episode_id"])
        intervention = selected_intervention[episode_id]
        learning_value = float(metadata["learning_value_weight"])
        # Square-root tempering reduces dominance without flattening rare stages
        # or interventions into an artificial uniform distribution.
        raw = (
            learning_value
            / math.sqrt(stage_counts[(dataset, stage)])
            / math.sqrt(intervention_episode_counts[(dataset, intervention)])
        )
        raw_weights.append(raw)
        datasets.append(dataset)

    case_scale: dict[tuple[str, str], float] = {}
    if normalize_case_mass:
        raw_weights = [weight**case_tempering_power for weight in raw_weights]
        raw_case_mass: dict[tuple[str, str], float] = defaultdict(float)
        for row, raw in zip(rows, raw_weights):
            metadata = row["metadata"]
            key = (str(metadata["dataset"]), str(metadata["case_id"]))
            raw_case_mass[key] += raw
        case_scale = {
            key: 1.0 / mass for key, mass in raw_case_mass.items()
        }
        raw_weights = [
            raw
            * case_scale[
                (str(row["metadata"]["dataset"]), str(row["metadata"]["case_id"]))
            ]
            for row, raw in zip(rows, raw_weights)
        ]

    target_mass = len(rows) / len(set(datasets))
    raw_mass: dict[str, float] = defaultdict(float)
    for dataset, weight in zip(datasets, raw_weights):
        raw_mass[dataset] += weight
    dataset_scale = {
        dataset: target_mass / mass for dataset, mass in raw_mass.items()
    }
    weights = [
        raw * dataset_scale[dataset]
        for raw, dataset in zip(raw_weights, datasets)
    ]

    output_rows: list[dict[str, Any]] = []
    for row, weight in zip(rows, weights):
        item = copy.deepcopy(row)
        item["metadata"]["sft_sampling_weight"] = round(weight, 6)
        item["metadata"]["sampling_policy_id"] = (
            "dataset_case_normalized_stage_intervention_tempered_v2"
            if normalize_case_mass
            else "dataset_stage_intervention_sqrt_tempered_v1"
        )
        output_rows.append(item)

    weighted_dataset: dict[str, float] = defaultdict(float)
    weighted_stage: dict[str, float] = defaultdict(float)
    weighted_intervention: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights):
        metadata = row["metadata"]
        dataset = str(metadata["dataset"])
        stage = str(metadata["causal_recovery_stage"])
        episode_id = str(metadata["episode_id"])
        weighted_dataset[dataset] += weight
        weighted_stage[f"{dataset}:{stage}"] += weight
        weighted_intervention[
            f"{dataset}:{selected_intervention[episode_id]}"
        ] += weight
    total = sum(weights)
    ess = total * total / sum(weight * weight for weight in weights)
    weighted_case: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights):
        metadata = row["metadata"]
        weighted_case[
            f"{metadata['dataset']}:{metadata['case_id']}"
        ] += weight
    per_dataset_case_mass: dict[str, list[float]] = defaultdict(list)
    for key, value in weighted_case.items():
        dataset = key.split(":", 1)[0]
        per_dataset_case_mass[dataset].append(value)
    report = {
        "artifact": "balanced_recovery_sft_mix_v1",
        "source_rows": len(rows),
        "output_rows": len(output_rows),
        "messages_unchanged": all(
            left["messages"] == right["messages"]
            for left, right in zip(rows, output_rows)
        ),
        "policy": {
            "dataset_mass": "exactly equal across the three datasets",
            "stage_and_intervention": (
                "square-root inverse-frequency tempering multiplied by the existing "
                "learning-value stage weight"
            ),
            "case_mass": (
                "exactly equal within each dataset before dataset scaling"
                if normalize_case_mass
                else "not explicitly normalized"
            ),
            "within_case_tempering_power": case_tempering_power,
            "oracle_or_verifier_used": False,
            "loss_scope": "last executed assistant action only",
        },
        "row_count_by_dataset": dict(Counter(datasets)),
        "row_count_by_dataset_stage": {
            f"{dataset}:{stage}": count
            for (dataset, stage), count in sorted(stage_counts.items())
        },
        "weighted_mass_by_dataset": {
            key: round(value, 6) for key, value in sorted(weighted_dataset.items())
        },
        "weighted_mass_by_dataset_stage": {
            key: round(value, 6) for key, value in sorted(weighted_stage.items())
        },
        "weighted_mass_by_dataset_intervention_private_audit": {
            key: round(value, 6)
            for key, value in sorted(weighted_intervention.items())
        },
        "weighted_case_mass_range_by_dataset": {
            dataset: {
                "minimum": round(min(values), 6),
                "maximum": round(max(values), 6),
                "cases": len(values),
            }
            for dataset, values in sorted(per_dataset_case_mass.items())
        },
        "weight_distribution": {
            "minimum": round(min(weights), 6),
            "p10": _percentile(weights, 0.10),
            "median": _percentile(weights, 0.50),
            "p90": _percentile(weights, 0.90),
            "maximum": round(max(weights), 6),
            "mean": round(total / len(weights), 6),
            "effective_sample_size": round(ess, 3),
            "effective_sample_fraction": round(ess / len(weights), 4),
        },
        "training_qualification_note": (
            "Sampling balance improves the candidate training mix but does not replace "
            "blind review, sealed evaluation, or student free-run gain."
        ),
    }
    return output_rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalize-case-mass", action="store_true")
    parser.add_argument("--case-tempering-power", type=float, default=1.0)
    args = parser.parse_args()
    rows, report = build_mix(
        _jsonl(args.batch_root / "recovery_sft_candidate.jsonl"),
        _load(args.batch_root / "compile_report.json"),
        normalize_case_mass=args.normalize_case_mass,
        case_tempering_power=args.case_tempering_power,
    )
    write_jsonl(args.output_root / "recovery_sft_weighted.jsonl", rows)
    write_json(args.output_root / "sampling_report.json", report)
    write_json(
        args.output_root / "trainer_contract.json",
        {
            "artifact": "recovery_sft_weighted_trainer_contract_v1",
            "input": "recovery_sft_weighted.jsonl",
            "supervised_tokens": "last assistant action only",
            "per_row_loss_multiplier": "metadata.sft_sampling_weight",
            "case_mass_normalized": args.normalize_case_mass,
            "do_not_supervise": ["system", "user", "tool", "private sidecar"],
            "shuffle_unit": "episode-aware; do not split turns across train/validation",
            "training_qualified": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
