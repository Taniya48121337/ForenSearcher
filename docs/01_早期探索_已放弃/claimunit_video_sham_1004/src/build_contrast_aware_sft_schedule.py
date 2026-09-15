"""Build loss-preserving, contrast-aware SFT batch schedules.

Every source row appears once as a base occurrence in each epoch.  Each semantic
contrast group additionally contributes two members with different targets in
one batch.  A row's original loss multiplier is divided across all of its
occurrences, so co-location does not change dataset, case, stage, tool, or row
loss mass.  The schedule is training metadata and never enters student messages.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_ROWS = (
    BASE
    / "recovery_causal_frontier_portable_v3_case_equivalence"
    / "recovery_sft_portable_weighted.jsonl"
)
DEFAULT_GROUPS = (
    BASE
    / "recovery_causal_frontier_semantic_contrast_v1"
    / "contrast_groups.jsonl"
)
DEFAULT_OUTPUT = BASE / "recovery_causal_frontier_trainer_ready_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _target_key(value: Any) -> str:
    return (
        str(value)
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def group_members(
    group: dict[str, Any], row_by_id: dict[str, dict[str, Any]] | None = None
) -> list[tuple[str, str]]:
    row_ids = [str(item) for item in group["row_ids"]]
    targets = group.get("abstract_targets")
    if not isinstance(targets, list) or len(targets) != len(row_ids):
        targets = group.get("targets")
    if not isinstance(targets, list) or len(targets) != len(row_ids):
        if row_by_id is None:
            raise ValueError(f"Group target alignment failed: {group.get('group_id')}")
        targets = []
        for row_id in row_ids:
            row = row_by_id.get(row_id)
            if row is None:
                raise ValueError(f"Group row is absent: {row_id}")
            match = next(
                (
                    item.get("contrast_target")
                    for item in row["metadata"].get("semantic_contrast_groups") or []
                    if str(item.get("group_id")) == str(group.get("group_id"))
                ),
                None,
            )
            targets.append(match or row["metadata"]["target_tool"])
    members = [(row_id, _target_key(target)) for row_id, target in zip(row_ids, targets)]
    if len({target for _, target in members}) < 2:
        raise ValueError(f"Contrast group lacks two targets: {group.get('group_id')}")
    return members


def choose_anchor_pair(
    group: dict[str, Any],
    epoch_index: int,
    row_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    by_target: dict[str, list[str]] = defaultdict(list)
    for row_id, target in group_members(group, row_by_id):
        by_target[target].append(row_id)
    targets = sorted(by_target)
    first_index = epoch_index % len(targets)
    offset = 1 + ((epoch_index // len(targets)) % (len(targets) - 1))
    second_index = (first_index + offset) % len(targets)
    chosen: list[dict[str, str]] = []
    for local_index, target_index in enumerate((first_index, second_index)):
        target = targets[target_index]
        candidates = sorted(by_target[target])
        member_index = (epoch_index + local_index) % len(candidates)
        chosen.append({"row_id": candidates[member_index], "target": target})
    if chosen[0]["row_id"] == chosen[1]["row_id"]:
        raise AssertionError("Distinct contrast targets selected the same row")
    return chosen


def _round_robin_by_dataset(
    items: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        queues[str(item["dataset"])].append(item)
    for index, dataset in enumerate(sorted(queues)):
        random.Random(seed + 97 * (index + 1)).shuffle(queues[dataset])
    result: list[dict[str, Any]] = []
    datasets = sorted(queues)
    while any(queues.values()):
        for dataset in datasets:
            if queues[dataset]:
                result.append(queues[dataset].pop())
    return result


def build_epoch_schedule(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    epoch: int,
    batch_size: int = 8,
    seed: int = 20260824,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size < 2:
        raise ValueError("Contrast-aware batches require batch_size >= 2")
    row_by_id = {str(row["metadata"]["row_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Source row_id values must be unique")
    pair_units: list[dict[str, Any]] = []
    anchor_counts: Counter[str] = Counter()
    for group in groups:
        pair = choose_anchor_pair(group, epoch - 1, row_by_id)
        for member in pair:
            if member["row_id"] not in row_by_id:
                raise ValueError(
                    f"Contrast member missing from source rows: {member['row_id']}"
                )
            anchor_counts[member["row_id"]] += 1
        datasets = {
            str(row_by_id[member["row_id"]]["metadata"]["dataset"])
            for member in pair
        }
        if len(datasets) != 1:
            raise ValueError(f"Contrast group crosses datasets: {group['group_id']}")
        pair_units.append(
            {
                "group_id": str(group["group_id"]),
                "family": str(group["family"]),
                "dataset": next(iter(datasets)),
                "members": pair,
            }
        )

    occurrence_counts = {
        row_id: 1 + anchor_counts.get(row_id, 0) for row_id in row_by_id
    }

    def occurrence(
        row_id: str,
        source: str,
        group_id: str | None = None,
        contrast_target: str | None = None,
    ) -> dict[str, Any]:
        metadata = row_by_id[row_id]["metadata"]
        original = float(metadata["sft_sampling_weight"])
        return {
            "row_id": row_id,
            "source": source,
            "contrast_group_id": group_id,
            "contrast_target": contrast_target,
            "dataset": str(metadata["dataset"]),
            "case_id": str(metadata["case_id"]),
            "stage": str(metadata["causal_recovery_stage"]),
            "target_tool": str(metadata["target_tool"]),
            "original_loss_multiplier": original,
            "row_occurrences_this_epoch": occurrence_counts[row_id],
            "scheduled_loss_multiplier": original / occurrence_counts[row_id],
        }

    total_occurrences = len(rows) + 2 * len(pair_units)
    batch_count = math.ceil(total_occurrences / batch_size)
    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    pair_order = _round_robin_by_dataset(pair_units, seed + epoch * 10007)
    batch_slots = list(range(batch_count))
    random.Random(seed + epoch * 7919).shuffle(batch_slots)
    if len(pair_order) > len(batch_slots):
        raise ValueError("More contrast groups than batches; increase batch size or split groups")
    group_batch: dict[str, int] = {}
    for unit, batch_index in zip(pair_order, batch_slots):
        group_batch[unit["group_id"]] = batch_index
        batches[batch_index].extend(
            occurrence(
                member["row_id"],
                "contrast_anchor",
                unit["group_id"],
                member["target"],
            )
            for member in unit["members"]
        )

    base_items = [
        {
            "row_id": row_id,
            "dataset": str(row["metadata"]["dataset"]),
        }
        for row_id, row in row_by_id.items()
    ]
    base_order = _round_robin_by_dataset(base_items, seed + epoch * 1543)
    tie_break = list(range(batch_count))
    random.Random(seed + epoch * 3571).shuffle(tie_break)
    tie_rank = {batch_index: rank for rank, batch_index in enumerate(tie_break)}
    for item in base_order:
        row_id = item["row_id"]
        dataset = item["dataset"]
        candidates = [
            index
            for index, batch in enumerate(batches)
            if len(batch) < batch_size
            and all(existing["row_id"] != row_id for existing in batch)
        ]
        if not candidates:
            raise ValueError(f"Unable to place base occurrence without collision: {row_id}")
        chosen = min(
            candidates,
            key=lambda index: (
                sum(existing["dataset"] == dataset for existing in batches[index]),
                len(batches[index]),
                tie_rank[index],
            ),
        )
        batches[chosen].append(occurrence(row_id, "base"))

    scheduled_batches: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, start=1):
        random.Random(seed + epoch * 1000003 + batch_index).shuffle(batch)
        scheduled_batches.append(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "batch_size": len(batch),
                "occurrences": batch,
            }
        )

    row_loss: defaultdict[str, float] = defaultdict(float)
    dataset_loss: defaultdict[str, float] = defaultdict(float)
    case_loss: defaultdict[str, float] = defaultdict(float)
    stage_loss: defaultdict[str, float] = defaultdict(float)
    tool_loss: defaultdict[str, float] = defaultdict(float)
    seen_counts: Counter[str] = Counter()
    maximum_dataset_share = 0.0
    for batch in scheduled_batches:
        dataset_counts = Counter(item["dataset"] for item in batch["occurrences"])
        maximum_dataset_share = max(
            maximum_dataset_share,
            max(dataset_counts.values()) / len(batch["occurrences"]),
        )
        ids = [item["row_id"] for item in batch["occurrences"]]
        if len(ids) != len(set(ids)):
            raise AssertionError("A batch contains the same row more than once")
        for item in batch["occurrences"]:
            weight = float(item["scheduled_loss_multiplier"])
            row_loss[item["row_id"]] += weight
            dataset_loss[item["dataset"]] += weight
            case_loss[f"{item['dataset']}:{item['case_id']}"] += weight
            stage_loss[f"{item['dataset']}:{item['stage']}"] += weight
            tool_loss[f"{item['dataset']}:{item['target_tool']}"] += weight
            seen_counts[item["row_id"]] += 1

    original_row_loss = {
        row_id: float(row["metadata"]["sft_sampling_weight"])
        for row_id, row in row_by_id.items()
    }
    maximum_row_loss_error = max(
        abs(row_loss[row_id] - original_row_loss[row_id]) for row_id in row_by_id
    )
    group_checks = []
    for unit in pair_units:
        batch = scheduled_batches[group_batch[unit["group_id"]]]["occurrences"]
        anchors = [
            item
            for item in batch
            if item["contrast_group_id"] == unit["group_id"]
        ]
        group_checks.append(
            {
                "group_id": unit["group_id"],
                "batch_index": group_batch[unit["group_id"]] + 1,
                "members": [item["row_id"] for item in anchors],
                "targets": [item["contrast_target"] for item in anchors],
                "co_located": len(anchors) == 2,
                "different_targets": len({item["contrast_target"] for item in anchors}) == 2,
            }
        )
    audit = {
        "epoch": epoch,
        "source_rows": len(rows),
        "contrast_groups": len(groups),
        "occurrences": total_occurrences,
        "batches": len(scheduled_batches),
        "batch_size_target": batch_size,
        "full_batches": sum(item["batch_size"] == batch_size for item in scheduled_batches),
        "minimum_batch_size": min(item["batch_size"] for item in scheduled_batches),
        "every_source_row_has_one_base_occurrence": all(
            any(
                occurrence_item["row_id"] == row_id
                and occurrence_item["source"] == "base"
                for batch in scheduled_batches
                for occurrence_item in batch["occurrences"]
            )
            for row_id in row_by_id
        ),
        "row_occurrence_minimum": min(seen_counts.values()),
        "row_occurrence_maximum": max(seen_counts.values()),
        "row_loss_mass_preserved": maximum_row_loss_error < 1e-9,
        "maximum_row_loss_error": maximum_row_loss_error,
        "dataset_loss_mass": {key: round(value, 9) for key, value in sorted(dataset_loss.items())},
        "case_loss_mass": {key: round(value, 9) for key, value in sorted(case_loss.items())},
        "stage_loss_mass": {key: round(value, 9) for key, value in sorted(stage_loss.items())},
        "tool_loss_mass": {key: round(value, 9) for key, value in sorted(tool_loss.items())},
        "maximum_single_dataset_share_in_batch": round(maximum_dataset_share, 6),
        "all_batches_have_unique_rows": True,
        "contrast_groups_co_located": sum(item["co_located"] for item in group_checks),
        "contrast_groups_with_different_targets": sum(
            item["different_targets"] for item in group_checks
        ),
        "group_checks": group_checks,
    }
    audit["passed"] = bool(
        audit["every_source_row_has_one_base_occurrence"]
        and audit["row_loss_mass_preserved"]
        and audit["contrast_groups_co_located"] == len(groups)
        and audit["contrast_groups_with_different_targets"] == len(groups)
        and audit["all_batches_have_unique_rows"]
    )
    return scheduled_batches, audit


def materialize_schedule(
    source_rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_by_id = {str(row["metadata"]["row_id"]): row for row in source_rows}
    materialized: list[dict[str, Any]] = []
    for batch in batches:
        for position, occurrence_item in enumerate(batch["occurrences"], start=1):
            row = copy.deepcopy(row_by_id[occurrence_item["row_id"]])
            metadata = row["metadata"]
            metadata["source_sft_sampling_weight"] = metadata["sft_sampling_weight"]
            metadata["sft_sampling_weight"] = occurrence_item[
                "scheduled_loss_multiplier"
            ]
            metadata["trainer_schedule"] = {
                "epoch": batch["epoch"],
                "batch_index": batch["batch_index"],
                "position": position,
                "source": occurrence_item["source"],
                "contrast_group_id": occurrence_item["contrast_group_id"],
                "contrast_target": occurrence_item["contrast_target"],
                "row_occurrences_this_epoch": occurrence_item[
                    "row_occurrences_this_epoch"
                ],
            }
            materialized.append(row)
    return materialized


def build_package(
    rows_path: Path,
    groups_path: Path,
    output_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 8,
    batches_per_shard: int = 32,
    seed: int = 20260824,
) -> dict[str, Any]:
    rows = _read_jsonl(rows_path)
    groups = _read_jsonl(groups_path)
    all_audits: list[dict[str, Any]] = []
    schedules: list[list[dict[str, Any]]] = []
    for epoch in range(1, epochs + 1):
        batches, audit = build_epoch_schedule(
            rows, groups, epoch=epoch, batch_size=batch_size, seed=seed
        )
        schedules.append(batches)
        all_audits.append(audit)
        epoch_dir = output_dir / "schedules" / f"epoch_{epoch:03d}"
        for start in range(0, len(batches), batches_per_shard):
            shard = batches[start : start + batches_per_shard]
            _write_jsonl(
                epoch_dir / f"shard_{start // batches_per_shard + 1:03d}.jsonl",
                shard,
            )

    anchor_members: defaultdict[str, set[str]] = defaultdict(set)
    anchor_target_pairs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for audit in all_audits:
        for group in audit["group_checks"]:
            anchor_members[group["group_id"]].update(group["members"])
            anchor_target_pairs[group["group_id"]].add(
                tuple(sorted(str(item) for item in group["targets"]))
            )
    group_sizes = {str(group["group_id"]): len(group["row_ids"]) for group in groups}
    member_coverage = {
        group_id: round(len(members) / group_sizes[group_id], 6)
        for group_id, members in anchor_members.items()
    }
    target_pair_diversity = {
        group_id: len(pairs) for group_id, pairs in anchor_target_pairs.items()
    }
    package_audit = {
        "artifact": "contrast_aware_sft_schedule_audit_v1",
        "epochs": epochs,
        "source_rows": len(rows),
        "contrast_groups": len(groups),
        "batch_size": batch_size,
        "batches_per_epoch": [item["batches"] for item in all_audits],
        "occurrences_per_epoch": [item["occurrences"] for item in all_audits],
        "all_epochs_passed": all(item["passed"] for item in all_audits),
        "all_epochs_preserve_row_loss_mass": all(
            item["row_loss_mass_preserved"] for item in all_audits
        ),
        "all_source_rows_have_base_exposure_every_epoch": all(
            item["every_source_row_has_one_base_occurrence"] for item in all_audits
        ),
        "case_loss_mass_identical_across_epochs": all(
            item["case_loss_mass"] == all_audits[0]["case_loss_mass"]
            for item in all_audits[1:]
        ),
        "stage_loss_mass_identical_across_epochs": all(
            item["stage_loss_mass"] == all_audits[0]["stage_loss_mass"]
            for item in all_audits[1:]
        ),
        "tool_loss_mass_identical_across_epochs": all(
            item["tool_loss_mass"] == all_audits[0]["tool_loss_mass"]
            for item in all_audits[1:]
        ),
        "all_groups_co_located_every_epoch": all(
            item["contrast_groups_co_located"] == len(groups)
            for item in all_audits
        ),
        "all_group_anchors_have_different_targets": all(
            item["contrast_groups_with_different_targets"] == len(groups)
            for item in all_audits
        ),
        "dataset_loss_mass_by_epoch": [item["dataset_loss_mass"] for item in all_audits],
        "maximum_single_dataset_share_in_any_batch": max(
            item["maximum_single_dataset_share_in_batch"] for item in all_audits
        ),
        "anchor_member_coverage_over_epochs": member_coverage,
        "anchor_member_coverage_minimum": min(member_coverage.values()),
        "anchor_member_coverage_mean": round(
            sum(member_coverage.values()) / len(member_coverage), 6
        ),
        "distinct_anchor_target_pairs_over_epochs": target_pair_diversity,
        "distinct_anchor_target_pairs_minimum": min(target_pair_diversity.values()),
        "distinct_anchor_target_pairs_mean": round(
            sum(target_pair_diversity.values()) / len(target_pair_diversity), 6
        ),
        "maximum_row_occurrences_in_any_epoch": max(
            item["row_occurrence_maximum"] for item in all_audits
        ),
        "scope_note": (
            "All rows receive base exposure every epoch. Anchor-member coverage only "
            "describes which members are additionally co-located; it is not row coverage."
        ),
        "epoch_audits": all_audits,
    }
    write_json(output_dir / "schedule_audit.json", package_audit)

    first_shard_batches = schedules[0][:batches_per_shard]
    smoke_rows = materialize_schedule(rows, first_shard_batches)
    smoke_path = output_dir / "materialization_smoke_epoch001_shard001.jsonl"
    _write_jsonl(smoke_path, smoke_rows)
    source_index = {str(row["metadata"]["row_id"]): row for row in rows}
    metadata_unchanged_except_schedule = all(
        {
            key: value
            for key, value in row["metadata"].items()
            if key
            not in {
                "sft_sampling_weight",
                "source_sft_sampling_weight",
                "trainer_schedule",
            }
        }
        == {
            key: value
            for key, value in source_index[str(row["metadata"]["row_id"])][
                "metadata"
            ].items()
            if key != "sft_sampling_weight"
        }
        and row["metadata"]["source_sft_sampling_weight"]
        == source_index[str(row["metadata"]["row_id"])]["metadata"][
            "sft_sampling_weight"
        ]
        for row in smoke_rows
    )
    calibration_keys = {
        "review_role",
        "mutation",
        "expected_dimension",
        "calibration_item_passed",
    }
    calibration_rows_included = any(
        calibration_keys & set(row.get("metadata") or {}) for row in smoke_rows
    )
    materialization_audit = {
        "artifact": "contrast_aware_sft_materialization_smoke_v1",
        "rows": len(smoke_rows),
        "batches": len(first_shard_batches),
        "messages_unchanged": all(
            row["messages"]
            == source_index[str(row["metadata"]["row_id"])]["messages"]
            for row in smoke_rows
        ),
        "tools_unchanged": all(
            row["tools"] == source_index[str(row["metadata"]["row_id"])]["tools"]
            for row in smoke_rows
        ),
        "metadata_unchanged_except_schedule": metadata_unchanged_except_schedule,
        "only_training_metadata_added": metadata_unchanged_except_schedule,
        "negative_or_calibration_rows_included": calibration_rows_included,
    }
    materialization_audit["passed"] = bool(
        materialization_audit["messages_unchanged"]
        and materialization_audit["tools_unchanged"]
        and materialization_audit["metadata_unchanged_except_schedule"]
        and materialization_audit["negative_or_calibration_rows_included"] is False
    )
    write_json(output_dir / "materialization_smoke_audit.json", materialization_audit)

    trainer_contract = {
        "artifact": "contrast_aware_recovery_sft_trainer_contract_v1",
        "base_rows": str(rows_path),
        "schedule_glob": "schedules/epoch_*/shard_*.jsonl",
        "join_key": "metadata.row_id -> occurrences[].row_id",
        "supervised_tokens": "last assistant action only",
        "loss_multiplier": "occurrences[].scheduled_loss_multiplier",
        "batch_boundary": "one schedule JSON object is one optimizer microbatch",
        "messages_and_tools_are_immutable": True,
        "contrast_behavior": (
            "Every semantic group contributes one different-target anchor pair per epoch."
        ),
        "loss_preservation": (
            "Sum scheduled multipliers over all occurrences of a row equals its original "
            "metadata.sft_sampling_weight in every epoch."
        ),
        "ordinary_sft_limit": (
            "Co-location changes batch composition but is not a contrastive objective. "
            "Its causal value must be tested against ordinary shuffle in B5."
        ),
        "do_not_supervise": [
            "system",
            "user",
            "tool",
            "metadata",
            "schedule fields",
            "private sidecars",
        ],
        "training_qualified": False,
        "qualification_blockers": [
            "calibrated_independent_blind_review_not_completed",
            "no_sealed_entity_disjoint_holdout",
            "closed_loop_student_gain_not_evaluated",
        ],
    }
    write_json(output_dir / "trainer_contract.json", trainer_contract)
    manifest = {
        "artifact": "recovery_causal_frontier_trainer_ready_v1",
        "source_rows": len(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "contrast_groups": len(groups),
        "schedule_shards": sum(
            math.ceil(len(batches) / batches_per_shard) for batches in schedules
        ),
        "batches_per_shard": batches_per_shard,
        "schedule_audit_passed": package_audit["all_epochs_passed"],
        "materialization_smoke_passed": materialization_audit["passed"],
        "messages_and_tools_changed": False,
        "training_executed": False,
        "training_qualified": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches-per-shard", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    manifest = build_package(
        args.rows,
        args.groups,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batches_per_shard=args.batches_per_shard,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
