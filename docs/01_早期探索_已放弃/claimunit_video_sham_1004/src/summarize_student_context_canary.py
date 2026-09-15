"""Paired analysis of the controlled three-arm student context canary."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs" / "student_context_ablation_v2"
ARMS = {
    "full_state_normalized_v1": {
        "base": BASE / "eval_base_full_4k_nll18.json",
        "adapter": BASE / "eval_seeded_balanced_full_4k_nll18.json",
        "train": BASE / "canary12_seeded_balanced_full_4k" / "run_report.json",
    },
    "naive_token_tail_v1": {
        "base": BASE / "eval_base_naive_4k_nll18.json",
        "adapter": BASE / "eval_seeded_balanced_naive_4k_nll18.json",
        "train": BASE / "canary12_seeded_balanced_naive_4k" / "run_report.json",
    },
    "decision_sufficient_state_v1": {
        "base": BASE / "eval_base_compact_4k_nll18.json",
        "adapter": BASE / "eval_seeded_balanced_compact_4k_nll18.json",
        "train": BASE / "canary12_seeded_balanced_compact_4k" / "run_report.json",
    },
}
DEFAULT_OUTPUT = BASE / "canary12_seeded_balanced_summary.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _weighted_nll(rows: list[dict[str, Any]]) -> float:
    tokens = sum(int(row["target_tokens"]) for row in rows)
    return sum(float(row["target_nll"]) * int(row["target_tokens"]) for row in rows) / max(tokens, 1)


def _turn_bin(turn: int) -> str:
    return "early_0_2" if turn <= 2 else "middle_3_5" if turn <= 5 else "late_6_plus"


def _paired_bootstrap(values: list[float], seed: int = 20260821, samples: int = 10000) -> dict[str, float]:
    rng = random.Random(seed)
    means = [
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    return {
        "mean": round(statistics.mean(values), 6),
        "ci95_low": round(means[int(samples * 0.025)], 6),
        "ci95_high": round(means[int(samples * 0.975)], 6),
    }


def _strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "dataset": lambda row: str(row["dataset"]),
        "trajectory_type": lambda row: str(row["trajectory_type"]),
        "turn_depth": lambda row: _turn_bin(int(row["turn_index"])),
    }
    result: dict[str, Any] = {}
    for dimension, getter in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[getter(row)].append(row)
        result[dimension] = {
            name: {
                "rows": len(group),
                "base_nll": round(_weighted_nll([{**row, "target_nll": row["base_nll"]} for row in group]), 6),
                "adapter_nll": round(
                    _weighted_nll([{**row, "target_nll": row["adapter_nll"]} for row in group]), 6
                ),
                "relative_reduction": round(
                    1
                    - _weighted_nll([{**row, "target_nll": row["adapter_nll"]} for row in group])
                    / max(_weighted_nll([{**row, "target_nll": row["base_nll"]} for row in group]), 1e-12),
                    6,
                ),
                "rows_improved": sum(row["adapter_nll"] < row["base_nll"] for row in group),
            }
            for name, group in sorted(groups.items())
        }
    return result


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    arm_reports: dict[str, Any] = {}
    reference_ids: list[str] | None = None
    per_arm_relative_by_id: dict[str, dict[str, float]] = {}
    exposure_ids: list[str] | None = None

    for arm, paths in ARMS.items():
        base = _load(paths["base"])
        adapter = _load(paths["adapter"])
        train = _load(paths["train"])
        base_by_id = {str(row["row_id"]): row for row in base["rows"]}
        adapter_by_id = {str(row["row_id"]): row for row in adapter["rows"]}
        ids = list(base_by_id)
        if set(ids) != set(adapter_by_id):
            raise ValueError(f"base/adapter row mismatch for {arm}")
        if reference_ids is None:
            reference_ids = ids
        elif set(reference_ids) != set(ids):
            raise ValueError("evaluation row IDs differ across arms")
        current_exposure = train["sampling"]["first_exposed_row_ids"]
        if exposure_ids is None:
            exposure_ids = current_exposure
        elif exposure_ids != current_exposure:
            raise ValueError("training exposure row IDs differ across arms")

        paired: list[dict[str, Any]] = []
        relative_by_id: dict[str, float] = {}
        for row_id in ids:
            before = base_by_id[row_id]
            after = adapter_by_id[row_id]
            relative = (float(before["target_nll"]) - float(after["target_nll"])) / max(
                float(before["target_nll"]), 1e-12
            )
            relative_by_id[row_id] = relative
            paired.append({
                "row_id": row_id,
                "dataset": before["dataset"],
                "trajectory_type": before["trajectory_type"],
                "turn_index": before["turn_index"],
                "reference_tool": before["reference_tool"],
                "target_tokens": before["target_tokens"],
                "base_nll": float(before["target_nll"]),
                "adapter_nll": float(after["target_nll"]),
                "relative_reduction": round(relative, 6),
            })
        per_arm_relative_by_id[arm] = relative_by_id
        base_nll = float(base["summary"]["target_nll"])
        adapter_nll = float(adapter["summary"]["target_nll"])
        relatives = [row["relative_reduction"] for row in paired]
        arm_reports[arm] = {
            "training": {
                "steps": train["max_steps"],
                "exposed_rows": train["sampling"]["expected_exposed_rows"],
                "dataset_counts": train["sampling"]["dataset_counts"],
                "target_tool_counts": train["sampling"]["target_tool_counts"],
                "runtime_seconds": train["train_metrics"]["train_runtime"],
                "peak_cuda_memory_gib": train["peak_cuda_memory_gib"],
                "total_flops": train["train_metrics"]["total_flos"],
            },
            "validation": {
                "rows": len(paired),
                "target_tokens": base["summary"]["target_tokens"],
                "base_nll": base_nll,
                "adapter_nll": adapter_nll,
                "absolute_reduction": round(base_nll - adapter_nll, 6),
                "relative_reduction": round((base_nll - adapter_nll) / base_nll, 6),
                "rows_improved": sum(row["adapter_nll"] < row["base_nll"] for row in paired),
                "per_row_relative_reduction_bootstrap": _paired_bootstrap(relatives),
            },
            "strata": _strata(paired),
            "paired_rows": paired,
        }

    compact = per_arm_relative_by_id["decision_sufficient_state_v1"]
    paired_advantage: dict[str, Any] = {}
    for comparator in ("full_state_normalized_v1", "naive_token_tail_v1"):
        differences = [compact[row_id] - per_arm_relative_by_id[comparator][row_id] for row_id in reference_ids or []]
        paired_advantage[f"compact_minus_{comparator}"] = {
            **_paired_bootstrap(differences),
            "rows_compact_larger_reduction": sum(value > 0 for value in differences),
            "rows": len(differences),
        }

    report = {
        "experiment": "student_context_ablation_v2_canary12_seeded_balanced",
        "status": "canary_evidence_not_mass_production_proof",
        "fairness": {
            "same_base_model": True,
            "same_lora_initialization_seed": True,
            "same_training_row_ids_and_order": True,
            "same_training_steps": True,
            "same_optimizer_and_schedule": True,
            "same_validation_row_ids": True,
            "training_exposure_row_ids": exposure_ids,
            "validation_row_ids": reference_ids,
        },
        "arms": arm_reports,
        "paired_compact_advantage": paired_advantage,
        "limitations": [
            "Only 12 optimization steps and 18 validation actions were measured.",
            "The canary measures teacher-forced target likelihood, not closed-loop task success.",
            "One seed is insufficient for a variance-aware training conclusion.",
            "Different context arms have different base NLL and compute cost; relative reduction is primary.",
            "A production claim still requires multi-seed full training, greedy action validity, closed-loop harness evaluation, and human blind review."
        ],
    }
    write_json(output, report)
    return report


if __name__ == "__main__":
    summary = run()
    print(json.dumps({
        "status": summary["status"],
        "arms": {
            name: values["validation"] for name, values in summary["arms"].items()
        },
        "paired_compact_advantage": summary["paired_compact_advantage"],
    }, ensure_ascii=False, indent=2))
