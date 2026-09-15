"""Compile verified multi-dataset trajectories into leakage-safe SFT splits.

The compiler never promotes failed trajectories.  It keeps all turns from the
same case—and all cases sharing a media entity—in one split, exports clean and
repair views separately, carries verifier transitions as sidecars, and emits an
optional sampler index rather than duplicating rare examples.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from audit_control_plane_exposure import audit as audit_control_plane
from compile_evidence_first_sft import (
    _reasoning_audit,
    _sensitive_key_hits,
    compile_row as compile_evidence_first_row,
)
from pipeline import ROOT, write_json, write_jsonl
from sft_context_compaction import CONTEXT_VIEW, compact_sft_row, validate_compact_sft_row
from sft_learning_value import learning_value_profile


DEFAULT_DATASET_ROOTS = {
    "VideoSham": ROOT / "outputs" / "utqr_v2_videosham_17_consolidated",
    "VCSL": ROOT / "outputs" / "scale30_vcsl_teacher_consolidated",
    "VidDiffBench": ROOT / "outputs" / "scale30_viddiff_consolidated",
}
DEFAULT_OUTPUT = ROOT / "outputs" / "production_sft_v11_hierarchical_sampler_candidate"
DEFAULT_SEED = 20260820
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
STATE_TOOLS = {
    "create_candidate_claim",
    "challenge_evidence",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
}
EVIDENCE_ID_PATTERN = re.compile(r"\bEA-[A-Za-z0-9][A-Za-z0-9_-]*\b")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gold(case: dict[str, Any]) -> bool:
    return (
        case.get("status") == "passed"
        and float(case.get("rubric_score", 0.0)) >= 92.0
        and not case.get("rubric_failed_gates", [])
        and not case.get("rubric_failed_criteria", [])
    )


def _entities(dataset: str, case: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("query_id", "reference_id", "sample_key", "video_name", "source_id"):
        value = case.get(field)
        if isinstance(value, str) and value:
            values.add(f"{dataset}:{value}")
    for field in ("source_path", "suspect_path"):
        value = case.get(field)
        if isinstance(value, str) and value:
            values.add(f"{dataset}:media:{Path(value).name}")
    if not values:
        values.add(f"{dataset}:case:{case['case_id']}")
    return values


def _component_map(dataset: str, cases: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, set[str]]]:
    case_ids = [str(case["case_id"]) for case in cases]
    union = _UnionFind(case_ids)
    owner: dict[str, str] = {}
    case_entities: dict[str, set[str]] = {}
    for case in cases:
        case_id = str(case["case_id"])
        entities = _entities(dataset, case)
        case_entities[case_id] = entities
        for entity in entities:
            if entity in owner:
                union.union(case_id, owner[entity])
            else:
                owner[entity] = case_id
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        grouped[union.find(case_id)].append(case_id)
    component_for_case: dict[str, str] = {}
    entities_for_component: dict[str, set[str]] = {}
    for members in grouped.values():
        component_id = sorted(members)[0]
        entities_for_component[component_id] = set().union(*(case_entities[case_id] for case_id in members))
        for case_id in members:
            component_for_case[case_id] = component_id
    return component_for_case, entities_for_component


def _target_counts(total: int) -> dict[str, int]:
    if total <= 0:
        return {split: 0 for split in SPLITS}
    raw = {split: total * RATIOS[split] for split in SPLITS}
    targets = {split: int(math.floor(raw[split])) for split in SPLITS}
    for split in sorted(SPLITS, key=lambda name: raw[name] - targets[name], reverse=True):
        if sum(targets.values()) >= total:
            break
        targets[split] += 1
    if total >= 3:
        for split in ("validation", "test"):
            if targets[split] == 0:
                targets[split] = 1
                targets["train"] -= 1
    return targets


def _assign_components(
    dataset: str,
    component_for_case: dict[str, str],
    *,
    seed: int,
) -> dict[str, str]:
    members: dict[str, list[str]] = defaultdict(list)
    for case_id, component_id in component_for_case.items():
        members[component_id].append(case_id)
    components = list(members)
    rng = random.Random(f"{seed}:{dataset}")
    rng.shuffle(components)
    components.sort(key=lambda component_id: len(members[component_id]), reverse=True)
    targets = _target_counts(sum(len(values) for values in members.values()))
    assigned_counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}

    # Preserve evaluability for small canaries without splitting a media group.
    if len(components) >= 3:
        smallest = sorted(components, key=lambda component_id: len(members[component_id]))
        reserved = {"validation": smallest[0], "test": smallest[1]}
        for split, component_id in reserved.items():
            assignments[component_id] = split
            assigned_counts[split] += len(members[component_id])

    for component_id in components:
        if component_id in assignments:
            continue
        size = len(members[component_id])
        split = max(
            SPLITS,
            key=lambda name: (
                targets[name] - assigned_counts[name],
                RATIOS[name],
                -SPLITS.index(name),
            ),
        )
        assignments[component_id] = split
        assigned_counts[split] += size
    return assignments


def _last_assistant(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return next((message for message in reversed(messages) if message.get("role") == "assistant"), {})


def _action_group(tool_name: str) -> str:
    if tool_name == "stop":
        return "terminal"
    if tool_name in STATE_TOOLS:
        return "harness_state"
    if tool_name.startswith("inspect_"):
        return "context_inspection"
    return "external_measurement"


def _evidence_choice_eligible_trajectory_keys(
    rows: list[dict[str, Any]],
) -> set[tuple[str, str, str, str]]:
    """Require every semantic measurement in a trajectory to carry search evidence."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = (
            str(metadata.get("dataset") or ""),
            str(metadata.get("case_id") or ""),
            str(metadata.get("trajectory_id") or ""),
            str(metadata.get("source_kind") or ""),
        )
        grouped[key].append(row)
    eligible: set[tuple[str, str, str, str]] = set()
    for key, trajectory_rows in grouped.items():
        measurement_rows = [
            row
            for row in trajectory_rows
            if (row.get("metadata") or {}).get("action_group")
            == "external_measurement"
        ]
        def contract_passed(row: dict[str, Any]) -> bool:
            metadata = row.get("metadata") or {}
            search = metadata.get("evidence_choice_search") or {}
            return bool(
                metadata.get("policy_choice_training_eligible") is True
                and metadata.get("semantic_measurement_turn") is True
                and int(metadata.get("visible_semantic_choice_count", 0)) >= 2
                and search.get("applicable") is True
                and search.get("proposal_coverage_passed") is True
                and search.get("selection_ambiguous") is False
                and float(search.get("selected_preaction_method_fit_score", 0.0))
                >= 1.0
                and search.get("selected_counterfactual_tool_swap_passed") is True
                and search.get("selection_frozen_before_branch_execution") is True
                and search.get("hindsight_outcome_used_for_selection") is False
                and search.get("selected_branch_execution_admissible") is True
                and search.get("proposal_retry_outcome_feedback_used") is False
                and search.get("committed_replay_checked") is True
                and search.get("deterministic_replay_match") is True
            )

        if measurement_rows and all(contract_passed(row) for row in measurement_rows):
            belief_updates_passed = all(
                not bool((row.get("metadata") or {}).get("latest_observation_uptake_expected"))
                or (row.get("metadata") or {}).get("latest_observation_uptake_passed") is True
                for row in trajectory_rows
            )
            if belief_updates_passed:
                eligible.add(key)
    return eligible


def _trajectory_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = row.get("metadata") or {}
    return (
        str(metadata.get("dataset") or ""),
        str(metadata.get("case_id") or ""),
        str(metadata.get("trajectory_id") or ""),
        str(metadata.get("source_kind") or ""),
    )


def _row_identity(dataset: str, kind: str, row: dict[str, Any]) -> str:
    metadata = row["metadata"]
    return ":".join(
        (
            dataset,
            str(metadata["case_id"]),
            str(metadata["trajectory_id"]),
            kind,
            str(metadata["turn_index"]),
        )
    )


def _validate_row(dataset: str, kind: str, row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    required = ("case_id", "trajectory_id", "turn_index", "target_tool")
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"SFT row is missing metadata fields {missing}")
    if metadata.get("step_utility_verdict") != "indispensable":
        raise ValueError(f"non-indispensable action reached accepted SFT: {_row_identity(dataset, kind, row)}")
    if metadata.get("counterfactual_removal_changes_outcome") is not True:
        raise ValueError(f"action lacks positive deletion counterfactual: {_row_identity(dataset, kind, row)}")
    if metadata.get("reasoning_grounded") is not True:
        raise ValueError(f"ungrounded reasoning reached accepted SFT: {_row_identity(dataset, kind, row)}")
    if metadata.get("target_execution_status") not in {"executed", "accepted"}:
        raise ValueError(f"unexecuted target reached accepted SFT: {_row_identity(dataset, kind, row)}")
    assistant = _last_assistant(row.get("messages") or [])
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"accepted SFT target must contain exactly one tool call: {_row_identity(dataset, kind, row)}")
    actual_tool = calls[0].get("function", {}).get("name")
    if actual_tool != metadata["target_tool"]:
        raise ValueError(f"target tool metadata mismatch: {actual_tool!r} != {metadata['target_tool']!r}")
    return _row_identity(dataset, kind, row), str(actual_tool)


def _portable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    for field in ("consolidated_source_run", "source_run", "source_artifact_dir", "artifact_dir"):
        result.pop(field, None)
    return result


def _portable_path(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


def _normalized_reasoning(row: dict[str, Any]) -> str:
    text = str(_last_assistant(row.get("messages") or []).get("content", "")).lower()
    text = re.sub(r"\b(?:ea|obs|exec|task|snapshot|llm-trj)[a-z0-9_-]*\b", "<id>", text)
    text = re.sub(r"\b[a-f0-9]{24,}\b", "<uuid>", text)
    text = re.sub(r"\b(?:surgery|videosham)[-_]?\d+\b", "<case>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", text)
    return " ".join(text.split())


def _target_params(row: dict[str, Any]) -> dict[str, Any]:
    calls = _last_assistant(row.get("messages") or []).get("tool_calls") or []
    if len(calls) != 1:
        return {}
    arguments = calls[0].get("function", {}).get("arguments", {})
    if isinstance(arguments, dict):
        return copy.deepcopy(arguments)
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _annotate_learning_value(row: dict[str, Any]) -> None:
    metadata = row["metadata"]
    reasoning = str(_last_assistant(row.get("messages") or []).get("content", ""))
    profile = learning_value_profile(
        str(metadata["target_tool"]),
        params=_target_params(row),
        external_measurement=metadata.get("action_group") == "external_measurement",
        evidence_ids_in_reasoning=len(EVIDENCE_ID_PATTERN.findall(reasoning)),
        branch_enabling_measurement=bool(metadata.get("branch_enabling_measurement")),
    )
    metadata.update(profile)


def _sampler_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    action_counts = Counter(row["metadata"]["action_group"] for row in rows)
    action_total = len(rows) / max(len(action_counts), 1)
    base: dict[str, float] = {}
    for row in rows:
        metadata = row["metadata"]
        action_factor = math.sqrt(action_total / action_counts[metadata["action_group"]])
        tier = metadata.get("learning_value_tier", "minimal_control")
        if tier == "minimal_control":
            action_factor = min(action_factor, 1.0)
        elif tier == "supporting":
            action_factor = min(action_factor, 1.25)
        learning_value = float(metadata.get("learning_value_weight", 0.1))
        base[metadata["row_id"]] = action_factor * learning_value

    def trajectory_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
        metadata = row["metadata"]
        return (
            str(metadata["dataset"]),
            str(metadata["entity_group"]),
            str(metadata["case_id"]),
            str(metadata["trajectory_id"]),
            str(metadata["source_kind"]),
        )

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[trajectory_key(row)].append(row)
    datasets = sorted({key[0] for key in grouped})
    entities_by_dataset: dict[str, set[str]] = defaultdict(set)
    cases_by_entity: dict[tuple[str, str], set[str]] = defaultdict(set)
    trajectories_by_case: dict[tuple[str, str, str], set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for key in grouped:
        dataset, entity, case_id, _trajectory_id, _kind = key
        entities_by_dataset[dataset].add(entity)
        cases_by_entity[(dataset, entity)].add(case_id)
        trajectories_by_case[(dataset, entity, case_id)].add(key)
    total_information_mass = sum(
        base[row["metadata"]["row_id"]]
        for row in rows
        if row["metadata"].get("learning_value_tier") != "minimal_control"
    )
    dataset_target = total_information_mass / max(len(datasets), 1)
    raw: dict[str, float] = {}
    for key, trajectory_rows in grouped.items():
        dataset, entity, case_id, _trajectory_id, _kind = key
        target = (
            dataset_target
            / max(len(entities_by_dataset[dataset]), 1)
            / max(len(cases_by_entity[(dataset, entity)]), 1)
            / max(len(trajectories_by_case[(dataset, entity, case_id)]), 1)
        )
        information_mass = sum(
            base[row["metadata"]["row_id"]]
            for row in trajectory_rows
            if row["metadata"].get("learning_value_tier") != "minimal_control"
        )
        # Square-root tempering reduces length/entity concentration without
        # pretending that a complex trajectory contains no additional signal.
        hierarchy_factor = math.sqrt(target / max(information_mass, 1e-9))
        for row in trajectory_rows:
            row_id = row["metadata"]["row_id"]
            raw[row_id] = base[row_id] * hierarchy_factor
    core_values = [
        raw[row["metadata"]["row_id"]]
        for row in rows
        if row["metadata"].get("learning_value_tier") == "core_reasoning"
    ]
    reference = core_values or list(raw.values())
    normalizer = len(reference) / max(sum(reference), 1e-9)
    return {
        row["metadata"]["row_id"]: round(
            min(
                2.0
                if row["metadata"].get("learning_value_tier") == "core_reasoning"
                else 1.0
                if row["metadata"].get("learning_value_tier") == "supporting"
                else 0.5,
                max(0.1, raw[row["metadata"]["row_id"]] * normalizer),
            ),
            6,
        )
        for row in rows
    }


def _weighted_group_dispersion(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], float] = defaultdict(float)
    for row in rows:
        metadata = row["metadata"]
        key = tuple(str(metadata[field]) for field in fields)
        grouped[key] += weights.get(str(metadata["row_id"]), 0.0)
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for key, mass in grouped.items():
        by_dataset[key[0]].append(mass)
    return {
        dataset: {
            "groups": len(values),
            "mass_min": round(min(values), 6),
            "mass_max": round(max(values), 6),
            "max_min_ratio": round(max(values) / max(min(values), 1e-9), 6),
        }
        for dataset, values in sorted(by_dataset.items())
    }


def compile_production_sft(
    dataset_roots: dict[str, Path],
    output_root: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    output_root = Path(output_root)
    case_records: dict[str, dict[str, Any]] = {}
    component_for_case: dict[str, str] = {}
    component_entities: dict[str, set[str]] = {}
    case_split: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    duplicate_row_ids = 0
    seen_rows: dict[str, str] = {}
    compaction_audits: list[dict[str, Any]] = []

    for dataset, root in dataset_roots.items():
        root = Path(root)
        report = json.loads((root / "batch_report.json").read_text(encoding="utf-8"))
        gold_cases = [case for case in report.get("cases", []) if _gold(case)]
        local_component, local_entities = _component_map(dataset, gold_cases)
        assignments = _assign_components(dataset, local_component, seed=seed)
        for case in gold_cases:
            case_id = str(case["case_id"])
            if case_id in case_records:
                raise ValueError(f"case_id appears in more than one dataset: {case_id}")
            component_id = f"{dataset}:{local_component[case_id]}"
            case_records[case_id] = {"dataset": dataset, "case": case}
            component_for_case[case_id] = component_id
            component_entities[component_id] = local_entities[local_component[case_id]]
            case_split[case_id] = assignments[local_component[case_id]]

        for kind in ("clean", "repair"):
            for source_row in _read_jsonl(root / f"sft_{kind}_accepted.jsonl"):
                case_id = str(source_row.get("metadata", {}).get("case_id"))
                if case_id not in case_records or case_records[case_id]["dataset"] != dataset:
                    raise ValueError(f"accepted SFT row does not map to a Gold case: {dataset}:{case_id}")
                row_id, tool_name = _validate_row(dataset, kind, source_row)
                canonical = json.dumps(source_row.get("messages"), sort_keys=True, ensure_ascii=False)
                if row_id in seen_rows:
                    if seen_rows[row_id] != canonical:
                        raise ValueError(f"duplicate row_id has different model-visible content: {row_id}")
                    duplicate_row_ids += 1
                    continue
                seen_rows[row_id] = canonical
                row = copy.deepcopy(source_row)
                row["metadata"] = _portable_metadata(row["metadata"])
                row["metadata"].update(
                    {
                        "row_id": row_id,
                        "dataset": dataset,
                        "data_split": case_split[case_id],
                        "source_kind": kind,
                        "trajectory_mode": kind,
                        "entity_group": component_for_case[case_id],
                        "action_group": _action_group(tool_name),
                        "production_source_root": _portable_path(root),
                        "generation_source": "accepted_llm_teacher_rollout",
                        "teacher_action_used": True,
                        "oracle_or_verifier_used_during_generation": False,
                        "verified_gold_bundle_accepted": True,
                    }
                )
                compact = compact_sft_row(row)
                compaction = validate_compact_sft_row(row, compact)
                if not compaction["passed"]:
                    failed = [name for name, passed in compaction["checks"].items() if not passed]
                    raise ValueError(f"decision-sufficient context audit failed for {row_id}: {failed}")
                # Compaction preserves sufficient state, but it historically
                # retained instance-fixed answer enums and target arguments.
                # Rebuild the model-visible tool interface after compaction so
                # the final export matches the reduced-control teacher view.
                evidence_first = compile_evidence_first_row(compact)
                _annotate_learning_value(evidence_first)
                rows.append(evidence_first)
                compaction_audits.append({"row_id": row_id, **compaction})

        for transition in _read_jsonl(root / "transition_sidecar.jsonl"):
            case_id = str(transition.get("case_id"))
            if case_id not in case_split:
                continue
            item = copy.deepcopy(transition)
            item.update({"dataset": dataset, "data_split": case_split[case_id], "entity_group": component_for_case[case_id]})
            transitions.append(item)

        for preference in _read_jsonl(root / "route_preference_pairs.jsonl"):
            case_id = str(preference.get("case_id"))
            if case_id not in case_split:
                continue
            item = copy.deepcopy(preference)
            item.update({"dataset": dataset, "data_split": case_split[case_id], "entity_group": component_for_case[case_id]})
            preferences.append(item)

    reasoning_audit = _reasoning_audit(rows)
    reasoning_by_id = {str(item["row_id"]): item for item in reasoning_audit}
    strict_row_ids = {
        row_id
        for row_id, item in reasoning_by_id.items()
        if bool(item["passes_causal_reasoning"])
    }
    strict_rows = [
        row for row in rows if str(row["metadata"]["row_id"]) in strict_row_ids
    ]
    evidence_choice_trajectory_keys = _evidence_choice_eligible_trajectory_keys(rows)
    evidence_choice_rows = [
        row for row in rows if _trajectory_key(row) in evidence_choice_trajectory_keys
    ]
    strict_evidence_choice_rows = [
        row
        for row in evidence_choice_rows
        if str(row["metadata"]["row_id"]) in strict_row_ids
    ]
    sensitive_hits = _sensitive_key_hits(rows)
    control_exposure, control_exposure_rows = audit_control_plane(rows)
    weights = _sampler_weights([row for row in rows if row["metadata"]["data_split"] == "train"])
    case_kind_counts = Counter(
        (row["metadata"]["case_id"], row["metadata"]["source_kind"])
        for row in rows
    )
    missing_case_views = [
        f"{case_id}:{kind}"
        for case_id in case_records
        for kind in ("clean", "repair")
        if case_kind_counts[(case_id, kind)] == 0
    ]
    if missing_case_views:
        raise ValueError(f"Gold cases are missing clean/repair SFT views: {missing_case_views}")

    sft_transition_ids = {
        row["metadata"]["row_id"]
        for row in rows
    }
    transition_ids: set[str] = set()
    for transition in transitions:
        transition_id = ":".join(
            (
                str(transition["dataset"]),
                str(transition["case_id"]),
                str(transition["trajectory_id"]),
                str(transition["episode_mode"]),
                str(transition["turn_index"]),
            )
        )
        if transition_id in transition_ids:
            raise ValueError(f"duplicate transition sidecar row: {transition_id}")
        transition_ids.add(transition_id)
    missing_transitions = sorted(sft_transition_ids - transition_ids)
    extra_transitions = sorted(transition_ids - sft_transition_ids)
    if missing_transitions or extra_transitions:
        raise ValueError(
            "SFT/transition one-to-one mismatch: "
            f"missing={missing_transitions[:5]}, extra={extra_transitions[:5]}"
        )

    sampler_index: list[dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        sampler_index.append(
            {
                "row_id": metadata["row_id"],
                "dataset": metadata["dataset"],
                "case_id": metadata["case_id"],
                "entity_group": metadata["entity_group"],
                "trajectory_id": metadata["trajectory_id"],
                "data_split": metadata["data_split"],
                "source_kind": metadata["source_kind"],
                "target_tool": metadata["target_tool"],
                "action_group": metadata["action_group"],
                "supervision_class": metadata["supervision_class"],
                "supervision_subtype": metadata["supervision_subtype"],
                "learning_value_tier": metadata["learning_value_tier"],
                "learning_value_weight": metadata["learning_value_weight"],
                "sampling_weight": weights.get(metadata["row_id"], 1.0),
            }
        )

    for split in SPLITS:
        split_rows = [row for row in rows if row["metadata"]["data_split"] == split]
        strict_split_rows = [
            row for row in strict_rows if row["metadata"]["data_split"] == split
        ]
        strict_choice_split_rows = [
            row
            for row in strict_evidence_choice_rows
            if row["metadata"]["data_split"] == split
        ]
        write_jsonl(
            output_root / split / "sft_candidate_all_not_for_training.jsonl",
            split_rows,
        )
        write_jsonl(
            output_root / split / "sft_strict_causal_core_not_production_ready.jsonl",
            strict_split_rows,
        )
        write_jsonl(
            output_root
            / split
            / "sft_strict_evidence_choice_core_not_production_ready.jsonl",
            strict_choice_split_rows,
        )
        for kind in ("clean", "repair"):
            write_jsonl(
                output_root / split / f"sft_candidate_{kind}_not_for_training.jsonl",
                [row for row in split_rows if row["metadata"]["source_kind"] == kind],
            )
        write_jsonl(
            output_root / split / "candidate_transition_sidecar_not_for_training.jsonl",
            [item for item in transitions if item["data_split"] == split],
        )
        write_jsonl(
            output_root / split / "strict_core_transition_sidecar_not_production_ready.jsonl",
            [
                item
                for item in transitions
                if item["data_split"] == split
                and ":".join(
                    (
                        str(item["dataset"]),
                        str(item["case_id"]),
                        str(item["trajectory_id"]),
                        str(item["episode_mode"]),
                        str(item["turn_index"]),
                    )
                )
                in strict_row_ids
            ],
        )
        write_jsonl(
            output_root / split / "route_preference_pairs.jsonl",
            [item for item in preferences if item["data_split"] == split],
        )
    write_jsonl(output_root / "candidate_sampler_index_not_for_training.jsonl", sampler_index)
    write_jsonl(output_root / "causal_reasoning_audit.jsonl", reasoning_audit)
    write_jsonl(output_root / "control_plane_exposure_rows.jsonl", control_exposure_rows)
    write_jsonl(output_root / "sensitive_key_hits.jsonl", sensitive_hits)
    write_jsonl(
        output_root / "case_split_assignments.jsonl",
        [
            {
                "case_id": case_id,
                "dataset": record["dataset"],
                "data_split": case_split[case_id],
                "entity_group": component_for_case[case_id],
                "entities": sorted(component_entities[component_for_case[case_id]]),
                "repair_operator": record["case"].get("repair_operator"),
                "rubric_score": record["case"].get("rubric_score"),
            }
            for case_id, record in sorted(case_records.items())
        ],
    )

    split_entities = {
        split: set().union(
            *(component_entities[component_for_case[case_id]] for case_id, value in case_split.items() if value == split),
            set(),
        )
        for split in SPLITS
    }
    entity_overlap = {
        f"{left}__{right}": sorted(split_entities[left].intersection(split_entities[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    message_sets = {
        split: {
            json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
            for row in rows
            if row["metadata"]["data_split"] == split
        }
        for split in SPLITS
    }
    message_overlap = {
        f"{left}__{right}": len(message_sets[left].intersection(message_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    normalized_reasoning_sets = {
        split: {
            _normalized_reasoning(row)
            for row in rows
            if row["metadata"]["data_split"] == split
        }
        for split in SPLITS
    }
    normalized_reasoning_overlap = {
        f"{left}__{right}": len(
            normalized_reasoning_sets[left].intersection(normalized_reasoning_sets[right])
        )
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    ratios = sorted(item["context_char_ratio"] for item in compaction_audits)

    def ratio_percentile(q: float) -> float:
        return ratios[round((len(ratios) - 1) * q)] if ratios else 0.0

    source_visible_chars = sum(item["source_visible_chars"] for item in compaction_audits)
    compact_visible_chars = sum(item["compact_visible_chars"] for item in compaction_audits)
    train_rows = [row for row in rows if row["metadata"]["data_split"] == "train"]
    train_tier_counts = Counter(
        row["metadata"]["learning_value_tier"] for row in train_rows
    )
    train_supervision_counts = Counter(
        row["metadata"]["supervision_subtype"] for row in train_rows
    )
    weighted_tier_mass: dict[str, float] = {}
    for tier in sorted(train_tier_counts):
        weighted_tier_mass[tier] = round(
            sum(
                weights[row["metadata"]["row_id"]]
                for row in train_rows
                if row["metadata"]["learning_value_tier"] == tier
            ),
            6,
        )
    total_weighted_mass = sum(weighted_tier_mass.values())
    hierarchy_case_dispersion = _weighted_group_dispersion(
        train_rows,
        weights,
        ("dataset", "entity_group", "case_id"),
    )
    hierarchy_entity_dispersion = _weighted_group_dispersion(
        train_rows,
        weights,
        ("dataset", "entity_group"),
    )
    hierarchy_trajectory_dispersion = _weighted_group_dispersion(
        train_rows,
        weights,
        ("dataset", "entity_group", "case_id", "trajectory_id", "source_kind"),
    )
    report = {
        "compiler": "production_sft_compiler_v8_evidence_choice_hierarchical_learning_value_candidate",
        "artifact_status": "structural_candidate_only",
        "training_qualified": False,
        "seed": seed,
        "split_ratios": RATIOS,
        "source_roots": {dataset: _portable_path(Path(root)) for dataset, root in dataset_roots.items()},
        "summary": {
            "gold_cases": len(case_records),
            "sft_rows": len(rows),
            "strict_causal_core_rows": len(strict_rows),
            "evidence_choice_eligible_trajectories": len(
                evidence_choice_trajectory_keys
            ),
            "evidence_choice_eligible_rows": len(evidence_choice_rows),
            "strict_evidence_choice_core_rows": len(strict_evidence_choice_rows),
            "causal_reasoning_failed_rows": len(rows) - len(strict_rows),
            "training_qualified_sft_rows": 0,
            "clean_rows": sum(row["metadata"]["source_kind"] == "clean" for row in rows),
            "repair_rows": sum(row["metadata"]["source_kind"] == "repair" for row in rows),
            "transition_rows": len(transitions),
            "route_preference_pairs": len(preferences),
            "deduplicated_row_ids": duplicate_row_ids,
            "sft_transition_pairs_verified": len(sft_transition_ids),
        },
        "split_case_counts": {
            split: Counter(
                record["dataset"]
                for case_id, record in case_records.items()
                if case_split[case_id] == split
            )
            for split in SPLITS
        },
        "split_row_counts": {
            split: Counter(
                row["metadata"]["dataset"]
                for row in rows
                if row["metadata"]["data_split"] == split
            )
            for split in SPLITS
        },
        "action_distribution": Counter(row["metadata"]["target_tool"] for row in rows),
        "action_group_distribution": Counter(row["metadata"]["action_group"] for row in rows),
        "repair_distribution": Counter(
            str(record["case"].get("repair_operator", "unknown")) for record in case_records.values()
        ),
        "leakage_audit": {
            "case_split_overlap": 0,
            "entity_overlap": entity_overlap,
            "exact_model_visible_row_overlap": message_overlap,
            "normalized_reasoning_overlap_for_diversity_review": normalized_reasoning_overlap,
            "passed": not any(entity_overlap.values()) and not any(message_overlap.values()),
            "semantic_target_preexposure_after_compile": control_exposure["overall"]["counts"][
                "has_singleton_answer_like_field"
            ],
            "sensitive_model_visible_key_hits": len(sensitive_hits),
        },
        "context_compaction": {
            "view": CONTEXT_VIEW,
            "rows_verified": len(compaction_audits),
            "rows_failed": sum(not item["passed"] for item in compaction_audits),
            "verified_state_reasoning_tool_choice_and_external_target_rate": round(
                sum(item["passed"] for item in compaction_audits) / max(len(compaction_audits), 1), 6
            ),
            "targets_with_internal_to_external_argument_normalization": sum(
                bool(row.get("metadata", {}).get("target_interface_normalized")) for row in rows
            ),
            "source_visible_chars": source_visible_chars,
            "compact_visible_chars": compact_visible_chars,
            "visible_char_reduction_rate": round(
                1.0 - compact_visible_chars / max(source_visible_chars, 1), 6
            ),
            "context_char_ratio_mean": round(sum(ratios) / max(len(ratios), 1), 6),
            "context_char_ratio_p50": ratio_percentile(0.50),
            "context_char_ratio_p90": ratio_percentile(0.90),
            "private_locator_or_oracle_hits": sum(
                sum(item["privacy_hits"].values()) for item in compaction_audits
            ),
            "full_transcripts_retained_in_source_artifacts": True,
        },
        "sampling": {
            "policy": "entity_case_trajectory_tempered_hierarchy_then_learning_value_and_bounded_action_balance_without_row_duplication",
            "hierarchy": ["dataset", "entity_group", "case_id", "trajectory_id", "source_kind", "turn"],
            "hierarchy_tempering_exponent": 0.5,
            "minimal_control_weight_ceiling": 0.5,
            "supporting_weight_ceiling": 1.0,
            "train_weight_min": min(weights.values(), default=1.0),
            "train_weight_max": max(weights.values(), default=1.0),
            "train_learning_value_tier_counts": train_tier_counts,
            "train_supervision_subtype_counts": train_supervision_counts,
            "weighted_loss_mass_by_tier": weighted_tier_mass,
            "effective_minimal_control_loss_mass_fraction": round(
                weighted_tier_mass.get("minimal_control", 0.0)
                / max(total_weighted_mass, 1e-9),
                6,
            ),
            "weighted_case_mass_dispersion_by_dataset": hierarchy_case_dispersion,
            "weighted_entity_mass_dispersion_by_dataset": hierarchy_entity_dispersion,
            "weighted_trajectory_mass_dispersion_by_dataset": hierarchy_trajectory_dispersion,
            "sampler_index": "candidate_sampler_index_not_for_training.jsonl",
        },
        "qualification_blockers": [
            "causal_reasoning_failures_present",
            "evidence_choice_search_not_verified"
            if not evidence_choice_trajectory_keys
            else "evidence_choice_search_requires_batch_causal_qualification",
            "closed_loop_evidence_sensitivity_not_established",
            "sealed_cross_dataset_evaluation_not_available",
        ],
        "scope_note": (
            "This compiles currently verified Gold data into evidence-first candidate views. The candidate files are "
            "not training-qualified, and the small strict core is not production-ready. Full executable transcripts "
            "remain in source artifacts; closed-loop and sealed-case evidence are still required."
        ),
    }
    write_json(output_root / "production_manifest.json", report)
    return report


def _parse_dataset_roots(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_DATASET_ROOTS)
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-root values must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        result[dataset] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = compile_production_sft(
        _parse_dataset_roots(args.dataset_root),
        args.output_root,
        seed=args.seed,
    )
    print(json.dumps({"summary": report["summary"], "leakage_audit": report["leakage_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
