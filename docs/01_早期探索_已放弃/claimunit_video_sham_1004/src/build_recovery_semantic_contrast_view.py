"""Annotate structure-matched evidence contrasts without changing SFT targets.

The resulting view keeps every original row and sampling weight.  It only adds
group metadata so a trainer can co-locate examples where the visible control
surface is the same but evidence semantics require different actions or
arguments.  This makes the existing supervision harder to solve by control-plane
pattern matching while preserving complete executable trajectories.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_control_plane_exposure import _preaction_payload, _target_call
from pipeline import ROOT, write_json, write_jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_INPUT = (
    BASE
    / "recovery_sft_training_mix_v4_edgeaware55"
    / "recovery_sft_weighted.jsonl"
)
DEFAULT_OUTPUT = BASE / "recovery_sft_semantic_contrast_v1_edgeaware55"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _control_signature(row: dict[str, Any]) -> dict[str, Any]:
    payload, _ = _preaction_payload(row)
    state = payload.get("current_harness_state") or payload.get("harness_state") or {}
    eligible = sorted(str(name) for name in state.get("eligible_tools") or [])
    visible_schemas = []
    for tool in row.get("tools") or []:
        function = tool.get("function") or {}
        name = str(function.get("name") or "")
        if name in eligible:
            visible_schemas.append(
                {
                    "name": name,
                    "parameters": function.get("parameters") or {},
                }
            )
    visible_schemas.sort(key=lambda item: item["name"])
    return {"eligible_tools": eligible, "tool_schemas": visible_schemas}


def _add_group(
    row: dict[str, Any],
    *,
    group_id: str,
    family: str,
    role: str,
    contrast_target: str,
) -> None:
    metadata = row["metadata"]
    groups = metadata.setdefault("semantic_contrast_groups", [])
    groups.append(
        {
            "group_id": group_id,
            "family": family,
            "role": role,
            "contrast_target": contrast_target,
        }
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _abstract_target(row: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the semantic decision while removing case-local opaque identifiers."""

    stage = str(row["metadata"]["causal_recovery_stage"])
    tool, arguments = _target_call(row)
    if stage == "fresh_measurement":
        return {"tool": tool}
    if tool == "challenge_evidence":
        return {
            "tool": tool,
            "defect_codes": sorted(arguments.get("defect_codes") or []),
            "target_fields": sorted(arguments.get("target_fields") or []),
            "requested_check": arguments.get("requested_check"),
        }
    if tool == "set_field_decision":
        return {
            "tool": tool,
            "fields": sorted(arguments.get("fields") or []),
            "decision_status": arguments.get("decision_status"),
        }
    if tool == "revise_claim_fields":
        return {
            "tool": tool,
            "fields": sorted(
                str(item.get("field"))
                for item in arguments.get("field_updates") or []
            ),
        }
    if tool == "bind_evidence":
        return {
            "tool": tool,
            "fields": sorted(
                str(item.get("field"))
                for item in arguments.get("binding_items") or []
            ),
        }
    if tool == "adjudicate_evidence":
        payload, _ = _preaction_payload(row)
        state = payload.get("current_harness_state") or payload.get("harness_state") or {}
        store = state.get("evidence_store") or {}
        selected = list(arguments.get("selected_evidence_ids") or [])
        selected_observations = [
            (store.get(evidence_id) or {}).get("observation") or {}
            for evidence_id in selected
        ]
        return {
            "tool": tool,
            "target_fields": sorted(arguments.get("target_fields") or []),
            "selected_values": [
                {
                    key: observation.get(key)
                    for key in ("preferred_video", "measurement_method")
                    if key in observation
                }
                for observation in selected_observations
            ],
        }
    return None


def _structure_matched_collision_groups(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build n-ary target neighborhoods under an identical control surface."""

    clusters: dict[
        tuple[str, str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        abstract = _abstract_target(row)
        if abstract is None:
            continue
        dataset = str(row["metadata"]["dataset"])
        stage = str(row["metadata"]["causal_recovery_stage"])
        control = _canonical(_control_signature(row))
        clusters[(dataset, stage, control)][_canonical(abstract)].append(row)

    groups: list[dict[str, Any]] = []
    ordinal = 0
    for (dataset, stage, _control), buckets in sorted(clusters.items()):
        nonempty = {
            target: sorted(values, key=lambda item: str(item["metadata"]["row_id"]))
            for target, values in buckets.items()
            if values
        }
        if len(nonempty) < 2:
            continue
        ordinal += 1
        group_id = f"semantic_neighborhood_{ordinal:04d}"
        members: list[tuple[str, dict[str, Any]]] = []
        for target, values in sorted(nonempty.items()):
            members.extend((target, row) for row in values)
        for target, row in members:
            _add_group(
                row,
                group_id=group_id,
                family="same_control_surface_different_semantic_target",
                role="neighborhood_member",
                contrast_target=target,
            )
        target_counts = {target: len(values) for target, values in nonempty.items()}
        count_values = list(target_counts.values())
        implied_pairs = sum(
            count_values[left] * count_values[right]
            for left in range(len(count_values))
            for right in range(left + 1, len(count_values))
        )
        groups.append(
            {
                "group_id": group_id,
                "family": "same_control_surface_different_semantic_target",
                "dataset": dataset,
                "stage": stage,
                "episode_ids": [
                    str(row["metadata"]["episode_id"]) for _target, row in members
                ],
                "row_ids": [
                    row["metadata"]["row_id"] for _target, row in members
                ],
                "abstract_targets": [
                    json.loads(target) for target, _row in members
                ],
                "distinct_abstract_targets": len(nonempty),
                "target_distribution": {
                    target: target_counts[target] for target in sorted(target_counts)
                },
                "implied_cross_target_pairs": implied_pairs,
                "control_signature_equal": True,
            }
        )
    return groups


def _same_case_cross_variant_stop_challenge_groups(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair converged and conflicting states across lattice variants of one case."""

    clusters: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metadata"]["causal_recovery_stage"] not in {
            "challenge",
            "terminal_verification",
        }:
            continue
        dataset = str(row["metadata"]["dataset"])
        case_id = str(row["metadata"]["case_id"])
        control = _canonical(_control_signature(row))
        clusters[(dataset, case_id, control)].append(row)

    groups: list[dict[str, Any]] = []
    ordinal = 0
    for (_dataset, _case_id, _control), members in sorted(clusters.items()):
        episode_ids = {
            str(row["metadata"]["episode_id"]) for row in members
        }
        if len(episode_ids) < 2:
            continue
        by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            tool, _arguments = _target_call(row)
            by_tool[tool].append(row)
        if not {"challenge_evidence", "stop"}.issubset(by_tool):
            continue
        ordinal += 1
        group_id = f"causal_lattice_stop_challenge_{ordinal:04d}"
        ordered = sorted(members, key=lambda row: str(row["metadata"]["row_id"]))
        for row in ordered:
            tool, _arguments = _target_call(row)
            _add_group(
                row,
                group_id=group_id,
                family="same_case_cross_variant_different_tool",
                role=(
                    "evidence_conflict_present"
                    if tool == "challenge_evidence"
                    else "evidence_conflict_resolved"
                ),
                contrast_target=tool,
            )
        groups.append(
            {
                "group_id": group_id,
                "family": "same_case_cross_variant_different_tool",
                "row_ids": [row["metadata"]["row_id"] for row in ordered],
                "targets": sorted(by_tool),
                "distinct_targets": len(by_tool),
                "distinct_episodes": len(episode_ids),
                "control_signature_equal": True,
                "same_case": True,
            }
        )
    return groups

def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = copy.deepcopy(rows)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_episode[str(row["metadata"]["episode_id"])].append(row)

    groups: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    partial_frontier_episodes_skipped: set[str] = set()

    # Contrast 1: the exact same stop/challenge action surface appears before and
    # after repair.  The semantic evidence state, not the menu, must decide which
    # target is correct.
    for episode_id, episode_rows in sorted(by_episode.items()):
        if all(
            row["metadata"].get("augmentation_role") == "novel_causal_frontier"
            for row in episode_rows
        ):
            partial_frontier_episodes_skipped.add(episode_id)
            continue
        challenge = [
            row
            for row in episode_rows
            if row["metadata"]["causal_recovery_stage"] == "challenge"
        ]
        terminal = [
            row
            for row in episode_rows
            if row["metadata"]["causal_recovery_stage"] == "terminal_verification"
        ]
        if len(challenge) != 1 or len(terminal) != 1:
            failures.append(
                {
                    "episode_id": episode_id,
                    "family": "stop_vs_challenge",
                    "reason": "expected exactly one challenge and one terminal row",
                }
            )
            continue
        left, right = challenge[0], terminal[0]
        signature_equal = _control_signature(left) == _control_signature(right)
        left_target, _ = _target_call(left)
        right_target, _ = _target_call(right)
        targets_differ = left_target != right_target
        if not signature_equal or not targets_differ:
            failures.append(
                {
                    "episode_id": episode_id,
                    "family": "stop_vs_challenge",
                    "reason": "control signatures differ or targets do not differ",
                    "control_signature_equal": signature_equal,
                    "targets_differ": targets_differ,
                }
            )
            continue
        group_id = f"{episode_id}:semantic_stop_challenge"
        _add_group(
            left,
            group_id=group_id,
            family="same_menu_different_tool",
            role="evidence_conflict_present",
            contrast_target=left_target,
        )
        _add_group(
            right,
            group_id=group_id,
            family="same_menu_different_tool",
            role="evidence_conflict_resolved",
            contrast_target=right_target,
        )
        groups.append(
            {
                "group_id": group_id,
                "family": "same_menu_different_tool",
                "episode_id": episode_id,
                "row_ids": [left["metadata"]["row_id"], right["metadata"]["row_id"]],
                "targets": [left_target, right_target],
                "control_signature_equal": True,
            }
        )

    # Contrast 2: every multi-candidate VidDiff assessment queue contains both a
    # retained and rejected atom under the same tool surface.  The observation
    # determines disposition and defect arguments.
    for episode_id, episode_rows in sorted(by_episode.items()):
        if episode_id in partial_frontier_episodes_skipped:
            continue
        assessments = [
            row
            for row in episode_rows
            if row["metadata"]["causal_recovery_stage"] == "evidence_assessment"
        ]
        if not assessments:
            continue
        signatures = [_control_signature(row) for row in assessments]
        signatures_equal = all(item == signatures[0] for item in signatures[1:])
        dispositions = []
        for row in assessments:
            _, arguments = _target_call(row)
            dispositions.append(str(arguments.get("disposition")))
        has_both = {"retain", "reject"}.issubset(set(dispositions))
        if not signatures_equal or not has_both:
            failures.append(
                {
                    "episode_id": episode_id,
                    "family": "assessment_partition",
                    "reason": "assessment menu differs or retain/reject contrast is absent",
                    "control_signature_equal": signatures_equal,
                    "dispositions": sorted(set(dispositions)),
                }
            )
            continue
        group_id = f"{episode_id}:semantic_assessment_partition"
        row_ids = []
        for row, disposition in zip(assessments, dispositions):
            _add_group(
                row,
                group_id=group_id,
                family="same_menu_different_arguments",
                role=f"candidate_{disposition}",
                contrast_target=disposition,
            )
            row_ids.append(row["metadata"]["row_id"])
        groups.append(
            {
                "group_id": group_id,
                "family": "same_menu_different_arguments",
                "episode_id": episode_id,
                "row_ids": row_ids,
                "targets": dispositions,
                "control_signature_equal": True,
            }
        )

    cross_variant_groups = _same_case_cross_variant_stop_challenge_groups(output)
    groups.extend(cross_variant_groups)
    collision_groups = _structure_matched_collision_groups(output)
    groups.extend(collision_groups)

    original_by_id = {str(row["metadata"]["row_id"]): row for row in rows}
    messages_unchanged = all(
        original_by_id[str(row["metadata"]["row_id"])]["messages"] == row["messages"]
        for row in output
    )
    tools_unchanged = all(
        original_by_id[str(row["metadata"]["row_id"])].get("tools") == row.get("tools")
        for row in output
    )
    weights_unchanged = all(
        original_by_id[str(row["metadata"]["row_id"])]["metadata"].get(
            "sft_sampling_weight"
        )
        == row["metadata"].get("sft_sampling_weight")
        for row in output
    )
    covered_rows = [
        row for row in output if row["metadata"].get("semantic_contrast_groups")
    ]
    family_counts = Counter(group["family"] for group in groups)
    collision_group_sizes = [len(group["row_ids"]) for group in collision_groups]
    dataset_coverage = Counter(str(row["metadata"]["dataset"]) for row in covered_rows)
    report = {
        "artifact": "recovery_sft_semantic_contrast_view_v1",
        "source_rows": len(rows),
        "output_rows": len(output),
        "contrast_groups": len(groups),
        "contrast_groups_by_family": dict(sorted(family_counts.items())),
        "contrast_covered_rows": len(covered_rows),
        "contrast_covered_row_fraction": round(len(covered_rows) / max(len(output), 1), 6),
        "covered_rows_by_dataset": dict(sorted(dataset_coverage.items())),
        "same_menu_stop_vs_challenge_episodes": family_counts["same_menu_different_tool"],
        "same_menu_retain_vs_reject_episodes": family_counts[
            "same_menu_different_arguments"
        ],
        "same_case_cross_variant_stop_vs_challenge_groups": len(
            cross_variant_groups
        ),
        "structure_matched_target_collision_groups": len(collision_groups),
        "structure_matched_collision_implied_cross_target_pairs": sum(
            group["implied_cross_target_pairs"] for group in collision_groups
        ),
        "structure_matched_collision_group_size": {
            "minimum": min(collision_group_sizes) if collision_group_sizes else 0,
            "maximum": max(collision_group_sizes) if collision_group_sizes else 0,
            "mean": round(
                sum(collision_group_sizes) / max(len(collision_group_sizes), 1), 3
            ),
        },
        "structure_matched_collision_groups_by_stage": dict(
            sorted(Counter(group["stage"] for group in collision_groups).items())
        ),
        "partial_frontier_episodes_skipped_for_episode_level_pairs": len(
            partial_frontier_episodes_skipped
        ),
        "partial_frontier_rows_still_used_for_structure_matched_groups": True,
        "construction_failures": failures,
        "messages_unchanged": messages_unchanged,
        "tools_unchanged": tools_unchanged,
        "sampling_weights_unchanged": weights_unchanged,
        "uses_oracle_or_verifier": False,
        "uses_teacher_targets_for_grouping": True,
        "target_grouping_visible_to_student": False,
        "purpose": (
            "Expose structure-matched supervision neighborhoods so evidence semantics, "
            "rather than eligible-tool patterns, must explain the different teacher targets."
        ),
        "recommended_use": (
            "Keep complete episodes and ordinary last-action cross-entropy. For an n-ary "
            "neighborhood, sample at least two members with different contrast_target values "
            "rather than loading the entire group; do not drop unpaired trajectory rows."
        ),
        "training_qualified": False,
        "qualification_note": (
            "Neighborhood construction is locally verified, but blind human review, sealed holdout, "
            "and measured student free-run gain remain pending."
        ),
    }
    return output, {"report": report, "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = _jsonl(args.input)
    output, result = build(rows)
    write_jsonl(args.output_root / "recovery_sft_contrastive.jsonl", output)
    write_json(args.output_root / "contrast_report.json", result["report"])
    write_jsonl(args.output_root / "contrast_groups.jsonl", result["groups"])
    write_json(
        args.output_root / "trainer_contract.json",
        {
            "artifact": "recovery_semantic_contrast_trainer_contract_v1",
            "input": "recovery_sft_contrastive.jsonl",
            "supervised_tokens": "last assistant action only",
            "per_row_loss_multiplier": "metadata.sft_sampling_weight",
            "optional_batch_group": "metadata.semantic_contrast_groups[*].group_id",
            "grouping_is_advisory": True,
            "n_ary_group_sampling": (
                "For groups with more than two rows, sample at least two members whose "
                "contrast_target values differ; loading the entire group is not required."
            ),
            "load_entire_group_required": False,
            "drop_unpaired_rows": False,
            "messages_or_targets_modified": False,
            "training_qualified": False,
        },
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
