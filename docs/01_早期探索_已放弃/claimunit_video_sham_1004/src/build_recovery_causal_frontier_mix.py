"""Build a compact SFT mix from complete episodes plus novel lattice frontiers.

The complete one-intervention-per-case batch remains the trajectory backbone.  The
larger intervention lattice contributes only target texts that are not already
represented and at most two contexts for the same normalized rationale template
within a dataset.  This keeps one invariance repeat without multiplying converged
tails merely because a case has more compatible interventions.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_recovery_reasoning_quality import _normalize
from audit_recovery_reasoning_quality_v2 import audit_rows
from build_balanced_recovery_sft_mix import build_mix
from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BACKBONE = BASE / "recovery_batch_candidate_v32_step_value55"
DEFAULT_LATTICE = BASE / "recovery_intervention_lattice_batch_v1"
DEFAULT_OUTPUT = BASE / "recovery_causal_frontier_training_mix_v1"


def _assistant_text(row: dict[str, Any]) -> str:
    return str(row["messages"][-1].get("content") or "")


def select_frontier(
    backbone_rows: list[dict[str, Any]],
    lattice_rows: list[dict[str, Any]],
    *,
    maximum_contexts_per_dataset_template: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if maximum_contexts_per_dataset_template < 1:
        raise ValueError("maximum_contexts_per_dataset_template must be positive")

    output: list[dict[str, Any]] = []
    exact_text = set()
    template_counts: Counter[tuple[str, str]] = Counter()
    for row in backbone_rows:
        item = copy.deepcopy(row)
        item["metadata"]["augmentation_role"] = "complete_episode_backbone"
        output.append(item)
        dataset = str(item["metadata"]["dataset"])
        text = _assistant_text(item)
        exact_text.add((dataset, text))
        template_counts[(dataset, _normalize(text))] += 1

    exclusion_counts: Counter[str] = Counter()
    added_by_dataset: Counter[str] = Counter()
    added_by_stage: Counter[str] = Counter()
    added_cases: set[tuple[str, str]] = set()
    for row in lattice_rows:
        dataset = str(row["metadata"]["dataset"])
        text = _assistant_text(row)
        template_key = (dataset, _normalize(text))
        if (dataset, text) in exact_text:
            exclusion_counts["exact_text_already_represented"] += 1
            continue
        if (
            template_counts[template_key]
            >= maximum_contexts_per_dataset_template
        ):
            exclusion_counts["template_context_budget_reached"] += 1
            continue
        item = copy.deepcopy(row)
        item["metadata"]["augmentation_role"] = "novel_causal_frontier"
        output.append(item)
        exact_text.add((dataset, text))
        template_counts[template_key] += 1
        added_by_dataset[dataset] += 1
        added_by_stage[str(item["metadata"]["causal_recovery_stage"])] += 1
        added_cases.add((dataset, str(item["metadata"]["case_id"])))

    return output, {
        "artifact": "recovery_causal_frontier_selection_v1",
        "policy": {
            "backbone": "retain every row from one complete executed episode per case",
            "lattice": (
                "add only exact-text-novel supervision and cap each normalized "
                "rationale template at two contexts per dataset"
            ),
            "why_two_contexts": (
                "one example plus one context-distinct invariance repeat; further "
                "converged tails add no new target wording and can dominate SFT"
            ),
            "selection_uses_oracle_or_verifier": False,
            "student_visible_intervention_labels": False,
        },
        "backbone_rows": len(backbone_rows),
        "lattice_candidate_rows": len(lattice_rows),
        "lattice_rows_added": len(output) - len(backbone_rows),
        "output_rows": len(output),
        "excluded_lattice_rows": dict(exclusion_counts),
        "added_cases": len(added_cases),
        "added_by_dataset": dict(added_by_dataset),
        "added_by_stage": dict(added_by_stage),
        "messages_unchanged": all(
            item["messages"]
            == (
                backbone_rows[index]["messages"]
                if index < len(backbone_rows)
                else next(
                    row["messages"]
                    for row in lattice_rows
                    if row["metadata"]["row_id"]
                    == item["metadata"]["row_id"]
                )
            )
            for index, item in enumerate(output)
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-root", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--lattice-root", type=Path, default=DEFAULT_LATTICE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-contexts-per-dataset-template", type=int, default=2)
    args = parser.parse_args()

    backbone_rows = _jsonl(args.backbone_root / "recovery_sft_candidate.jsonl")
    lattice_rows = _jsonl(args.lattice_root / "recovery_sft_candidate.jsonl")
    selected, selection = select_frontier(
        backbone_rows,
        lattice_rows,
        maximum_contexts_per_dataset_template=(
            args.maximum_contexts_per_dataset_template
        ),
    )
    combined_compile_report = {
        "results": (
            list(_load(args.backbone_root / "compile_report.json").get("results") or [])
            + list(_load(args.lattice_root / "compile_report.json").get("results") or [])
        )
    }
    weighted, sampling = build_mix(
        selected,
        combined_compile_report,
        normalize_case_mass=True,
        case_tempering_power=0.5,
    )
    reasoning = audit_rows(selected)
    selection["reasoning_quality"] = {
        "passed": reasoning["passed"],
        "metrics": reasoning["metrics"],
        "checks": reasoning["checks"],
    }
    selection["source_execution"] = {
        "lattice_jobs": _load(args.lattice_root / "compile_report.json")[
            "jobs_selected"
        ],
        "lattice_compiled": _load(args.lattice_root / "compile_report.json")[
            "compiled_episodes"
        ],
        "lattice_failed_preserved": _load(
            args.lattice_root / "compile_report.json"
        )["status_counts"].get("compile_failed", 0),
    }

    write_jsonl(args.output_root / "recovery_sft_causal_frontier.jsonl", selected)
    write_jsonl(args.output_root / "recovery_sft_weighted.jsonl", weighted)
    write_json(args.output_root / "selection_report.json", selection)
    write_json(args.output_root / "sampling_report.json", sampling)
    write_json(args.output_root / "reasoning_quality_audit.json", reasoning)
    write_json(
        args.output_root / "canonical_execution_sources.json",
        {
            "artifact": "recovery_causal_frontier_execution_sources_v1",
            "rubric": "../../../rubric/unified_trajectory_quality_rubric_v12.json",
            "complete_backbone": {
                "episodes": "../recovery_batch_candidate_v32_step_value55/recovery_episodes.jsonl",
                "scorecards": "../recovery_batch_candidate_v32_step_value55/utqr12_episode_scorecards.json",
                "step_value": "../recovery_batch_candidate_v32_step_value55/step_value_audit.json",
            },
            "causal_lattice": {
                "episodes": "../recovery_intervention_lattice_batch_v1/recovery_episodes.jsonl",
                "compile_report": "../recovery_intervention_lattice_batch_v1/compile_report.json",
                "scorecards": "../recovery_intervention_lattice_batch_v1/utqr12_episode_scorecards.json",
                "step_value": "../recovery_intervention_lattice_batch_v1/step_value_audit.json",
            },
            "frontier_rows_are_projection_not_new_executions": True,
        },
    )
    write_json(
        args.output_root / "trainer_contract.json",
        {
            "artifact": "recovery_causal_frontier_trainer_contract_v1",
            "input": "recovery_sft_weighted.jsonl",
            "supervised_tokens": "last assistant action only",
            "per_row_loss_multiplier": "metadata.sft_sampling_weight",
            "case_mass_normalized": True,
            "within_case_tempering_power": 0.5,
            "backbone_complete_episodes_retained": True,
            "lattice_selection": "novel causal frontier only",
            "canonical_execution_sources": "canonical_execution_sources.json",
            "rubric_id": "UTQR-12",
            "reasoning_audit": "reasoning_quality_audit.json",
            "control_plane_audit": "control_plane_exposure_audit.json",
            "do_not_supervise": ["system", "user", "tool", "private sidecar"],
            "shuffle_unit": "episode-aware where complete; keep row source metadata",
            "training_qualified": False,
            "qualification_blockers": [
                "independent_blind_human_review_not_completed",
                "no_sealed_entity_disjoint_holdout",
                "closed_loop_student_gain_not_evaluated",
            ],
        },
    )
    print(
        json.dumps(
            {
                "selection": selection,
                "sampling": sampling,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
