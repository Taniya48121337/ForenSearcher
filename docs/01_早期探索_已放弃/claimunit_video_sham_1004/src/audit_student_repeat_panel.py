"""Audit and safely re-key independently sampled student rollout SFT rows."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl
from validate_student_batch_sft import diversity_balance_rows, validate_rows


def _reasoning(row: dict[str, Any]) -> str:
    assistant = next(
        message for message in reversed(row.get("messages", [])) if message.get("role") == "assistant"
    )
    return str(assistant.get("content") or "").strip()


def audit_repeat_panel(run_roots: list[Path], output: Path) -> dict[str, Any]:
    source_rows: list[tuple[int, dict[str, Any]]] = []
    batch_summaries = []
    sequence_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    seeds = []
    for root in run_roots:
        report = json.loads((root / "batch_report.json").read_text(encoding="utf-8"))
        seed = int(report["generation_parameters"]["base_seed"])
        seeds.append(seed)
        batch_summaries.append(report["summary"])
        for episode in report.get("episodes", []):
            sequence_sets[str(episode["case_id"])].add(tuple(episode.get("tool_sequence", [])))
        source_rows.extend((seed, row) for row in _jsonl(root / "accepted_student_sft.jsonl"))

    original_ids = [str(row["metadata"]["row_id"]) for _seed, row in source_rows]
    rekeyed = []
    for seed, source in source_rows:
        row = copy.deepcopy(source)
        metadata = row["metadata"]
        rollout_id = f"sample-seed-{seed}"
        if metadata.get("rollout_id") is None:
            old = str(metadata["row_id"])
            prefix, turn = old.rsplit(":", 1)
            metadata["row_id"] = f"{prefix}:{rollout_id}:{turn}"
            metadata["rollout_id"] = rollout_id
        rekeyed.append(row)

    variants: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    for row in rekeyed:
        metadata = row["metadata"]
        key = (
            str(metadata["dataset"]),
            str(metadata["case_id"]),
            int(metadata["turn_index"]),
            str(metadata["target_tool"]),
        )
        variants[key].add(_reasoning(row))
    counts = [len(values) for values in variants.values()]
    validation = validate_rows(rekeyed)
    balanced, balancing = diversity_balance_rows(rekeyed)
    balanced_validation = validate_rows(balanced)
    report = {
        "audit": "independent_student_repeat_panel_v1",
        "source_runs": [str(path) for path in run_roots],
        "generation_seeds": seeds,
        "distinct_generation_seeds": len(set(seeds)) == len(seeds),
        "batches": len(run_roots),
        "episodes": sum(int(summary.get("episodes", 0)) for summary in batch_summaries),
        "accepted_episodes": sum(int(summary.get("accepted_episodes", 0)) for summary in batch_summaries),
        "source_sft_rows": len(source_rows),
        "original_unique_row_ids": len(set(original_ids)),
        "original_row_id_collision_occurrences": len(original_ids) - len(set(original_ids)),
        "rekeyed_unique_row_ids": len({row["metadata"]["row_id"] for row in rekeyed}),
        "action_positions": len(variants),
        "reasoning_variant_count_distribution": dict(Counter(counts)),
        "action_positions_with_all_rollouts_textually_distinct": sum(
            count == len(run_roots) for count in counts
        ),
        "action_positions_with_at_least_two_reasoning_variants": sum(count >= 2 for count in counts),
        "tool_sequence_variants_by_case": {
            case_id: [list(sequence) for sequence in sorted(sequences)]
            for case_id, sequences in sequence_sets.items()
        },
        "tool_sequence_stable_for_every_case": all(len(values) == 1 for values in sequence_sets.values()),
        "rekeyed_validation": validation,
        "diversity_balancing": balancing,
        "balanced_validation": balanced_validation,
        "batch_summaries": batch_summaries,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "repeat_panel_audit.json", report)
    write_jsonl(output / "rekeyed_repeat_sft.jsonl", rekeyed if validation["all_passed"] else [])
    write_jsonl(output / "diversity_balanced_repeat_sft.jsonl", balanced if balanced_validation["all_passed"] else [])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_repeat_panel(args.runs, args.output)
    compact = {
        key: report[key]
        for key in (
            "episodes",
            "accepted_episodes",
            "source_sft_rows",
            "original_row_id_collision_occurrences",
            "rekeyed_unique_row_ids",
            "reasoning_variant_count_distribution",
            "tool_sequence_stable_for_every_case",
        )
    }
    compact["rekeyed_validation_passed"] = report["rekeyed_validation"]["all_passed"]
    compact["balanced_rows"] = report["diversity_balancing"]["output_rows"]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
