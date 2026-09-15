"""Export compact, stratified, student-visible recovery packets for human review.

The packet is a delta view of an executed transcript.  It preserves the initial
student-visible state, every assistant rationale and tool call, every non-state
tool response, and every change to the projected Harness state.  Private
intervention metadata is used only to obtain coverage and is never put in the
public packet.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from export_blind_recovery_review_packets import Redactor, _read_jsonl, _write_jsonl
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_EPISODES = (
    BASE / "recovery_intervention_lattice_batch_v1/recovery_episodes.jsonl"
)
DEFAULT_SIDECAR = (
    BASE
    / "recovery_intervention_lattice_batch_v1/private_intervention_sidecar.jsonl"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v12.json"
DEFAULT_OUTPUT = BASE / "recovery_lattice_blind_delta_packet_v1"

SUPPLEMENTAL_DIMENSIONS = [
    "anti_shortcut_semantic_identifiability",
    "per_step_sft_pedagogical_value",
]
BLIND_PRIVATE_VALUE_KEYS = {
    "candidate_report",
    "selection_policy",
    "verifier_result",
    "oracle",
    "metrics",
    "local_path",
    "source_path",
    "file_path",
    "url",
}
ORACLE_TOKEN = re.compile(r"(?i)[A-Za-z0-9_-]*oracle[A-Za-z0-9_-]*")
URL_TOKEN = re.compile(r"(?i)https?://\S+")
LOCAL_REPORT_TOKEN = re.compile(
    r"(?i)(?:^|.*[\\/])[^\\/]+\.(?:json|jsonl|npz|npy|pt|pth|mp4|avi|mov)$"
)


def _json(value: str | dict[str, Any]) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object in transcript content")
    return parsed


def _project_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep state needed for semantic review while dropping repeated boilerplate."""

    task = state.get("task_contract") or {}
    return {
        "task": task.get("task"),
        "episode_mode": task.get("episode_mode"),
        "visible_assets": task.get("visible_assets"),
        "claim_schema_id": task.get("claim_schema_id"),
        "case_goals": state.get("case_goals", []),
        "candidate_claims": state.get("candidate_claims", []),
        "field_bindings": state.get("field_bindings", {}),
        "evidence_store": state.get("evidence_store", {}),
        "evidence_assessments": state.get("evidence_assessments", {}),
        "evidence_adjudications": state.get("evidence_adjudications", {}),
        "evidence_challenges": state.get("evidence_challenges", []),
        "open_evidence_challenges": state.get("open_evidence_challenges", []),
        "open_high_priority_obligations": state.get(
            "open_high_priority_obligations", []
        ),
        "eligible_tools": state.get("eligible_tools", []),
        "evidence_adjudication": state.get("evidence_adjudication", {}),
        "evidence_assessment": state.get("evidence_assessment", {}),
        "terminal": state.get("terminal", False),
        "stop_decision": state.get("stop_decision"),
    }


def _blind_privacy_redact(value: Any, key: str | None = None) -> Any:
    """Remove construction provenance that is irrelevant to a blind judge."""

    normalized_key = (key or "").lower()
    if normalized_key in BLIND_PRIVATE_VALUE_KEYS:
        return "<PRIVATE_PROVENANCE_REDACTED>"
    if isinstance(value, dict):
        return {
            name: _blind_privacy_redact(item, name) for name, item in value.items()
        }
    if isinstance(value, list):
        return [_blind_privacy_redact(item, key) for item in value]
    if isinstance(value, str):
        if URL_TOKEN.search(value):
            value = URL_TOKEN.sub("<URL_REDACTED>", value)
        if ORACLE_TOKEN.search(value):
            value = ORACLE_TOKEN.sub("<PRIVATE_SELECTION_POLICY>", value)
        if LOCAL_REPORT_TOKEN.fullmatch(value):
            return "<LOCAL_REPORT_REDACTED>"
    return value


def _path(parent: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def state_delta(before: Any, after: Any, parent: str = "") -> list[dict[str, Any]]:
    """Return a deterministic JSON-Patch-like change list with before/after values."""

    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) - set(after)):
            changes.append(
                {"change": "removed", "path": _path(parent, key), "before": before[key]}
            )
        for key in sorted(set(after) - set(before)):
            changes.append(
                {"change": "added", "path": _path(parent, key), "after": after[key]}
            )
        for key in sorted(set(before) & set(after)):
            changes.extend(state_delta(before[key], after[key], _path(parent, key)))
        return changes
    if before != after:
        return [{"change": "replaced", "path": parent or "/", "before": before, "after": after}]
    return []


def apply_state_delta(before: dict[str, Any], changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the emitted delta; used to prove the review view is reconstructable."""

    result = copy.deepcopy(before)
    for item in changes:
        parts = [
            token.replace("~1", "/").replace("~0", "~")
            for token in str(item["path"]).split("/")[1:]
        ]
        if not parts:
            raise ValueError("Root replacement is not supported in projected state")
        parent: dict[str, Any] = result
        for token in parts[:-1]:
            child = parent[token]
            if not isinstance(child, dict):
                raise ValueError(f"Delta parent is not an object at {item['path']}")
            parent = child
        leaf = parts[-1]
        if item["change"] == "removed":
            parent.pop(leaf)
        elif item["change"] in {"added", "replaced"}:
            parent[leaf] = copy.deepcopy(item["after"])
        else:
            raise ValueError(f"Unknown delta operation: {item['change']}")
    return result


def _episode_public_view(episode: dict[str, Any]) -> tuple[dict[str, Any], str]:
    messages = episode["messages"]
    systems = [item["content"] for item in messages if item.get("role") == "system"]
    users = [item for item in messages if item.get("role") == "user"]
    if len(systems) != 1 or len(users) != 1:
        raise ValueError(f"Unexpected transcript preamble in {episode.get('episode_id')}")
    user = _json(users[0]["content"])
    initial_state = _project_state(user["current_harness_state"])
    schema = (user.get("task") or {}).get("claim_schema") or {}

    turns: list[dict[str, Any]] = []
    current = initial_state
    message_index = 0
    while message_index < len(messages):
        assistant = messages[message_index]
        if assistant.get("role") != "assistant":
            message_index += 1
            continue
        if message_index + 1 >= len(messages) or messages[message_index + 1].get("role") != "tool":
            raise ValueError(f"Assistant turn lacks tool result in {episode.get('episode_id')}")
        tool = messages[message_index + 1]
        tool_content = _json(tool["content"])
        next_state = _project_state(tool_content["harness_state"])
        calls = assistant.get("tool_calls") or []
        if len(calls) != 1:
            raise ValueError(f"Expected one tool call in {episode.get('episode_id')}")
        function = calls[0]["function"]
        non_state_result = {
            key: value for key, value in tool_content.items() if key != "harness_state"
        }
        changes = state_delta(current, next_state)
        if apply_state_delta(current, changes) != next_state:
            raise AssertionError(
                f"State delta is not reconstructable in {episode.get('episode_id')}"
            )
        turns.append(
            {
                "assistant_turn": f"A{len(turns) + 1}",
                "reasoning": assistant.get("content", ""),
                "tool_call": {
                    "name": function["name"],
                    "arguments": _json(function["arguments"]),
                },
                "tool_result_without_repeated_state": non_state_result,
                "visible_state_changes": changes,
            }
        )
        current = next_state
        message_index += 2

    redactor = Redactor()
    packet = _blind_privacy_redact(redactor.value(
        {
            "instructions": (
                "Judge only the visible initial state and executed deltas. Do not infer "
                "a hidden answer, dataset, intervention, or machine result, and do not "
                "reward the trajectory merely because the final tool accepts stop."
            ),
            "representation": (
                "Each turn contains the original assistant text and tool arguments, the "
                "student-visible tool response excluding its repeated full state, and "
                "all changes to the review projection of that state."
            ),
            "task_instruction": (user.get("task") or {}).get("instruction"),
            "claim_schema": {
                "required_fields": schema.get("required_fields", []),
                "target_fields": schema.get("target_fields", []),
                "invariant_fields": schema.get("invariant_fields", []),
                "fields": schema.get("fields", {}),
            },
            "visible_initial_state": initial_state,
            "turns": turns,
        }
    ))
    return packet, systems[0]


def _allocate_quotas(names: list[str], total: int, seed: int) -> dict[str, int]:
    if total < len(names):
        raise ValueError("Sample size must cover every intervention family")
    quotient, remainder = divmod(total, len(names))
    rotated = sorted(names)
    random.Random(seed).shuffle(rotated)
    return {
        name: quotient + (1 if name in set(rotated[:remainder]) else 0)
        for name in names
    }


def select_stratified(
    episodes: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
    per_dataset: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_episode = {str(item["episode_id"]): item for item in episodes}
    sidecar_by_episode = {str(item["episode_id"]): item for item in sidecars}
    if set(by_episode) != set(sidecar_by_episode):
        raise ValueError("Episode and private-sidecar IDs differ")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_id, episode in by_episode.items():
        intervention = str(sidecar_by_episode[episode_id]["intervention_id"])
        grouped[str(episode["dataset"])][intervention].append(episode)

    selected: list[dict[str, Any]] = []
    private_report: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(sorted(grouped)):
        families = grouped[dataset]
        quotas = _allocate_quotas(
            sorted(families), per_dataset, seed + 101 * (dataset_index + 1)
        )
        selected_case_counts: Counter[str] = Counter()
        dataset_selected: list[dict[str, Any]] = []
        for family_index, family in enumerate(sorted(families)):
            candidates = list(families[family])
            rng = random.Random(seed + 1009 * (dataset_index + 1) + family_index)
            random_keys = {str(item["episode_id"]): rng.random() for item in candidates}
            candidates.sort(
                key=lambda item: (
                    selected_case_counts[str(item["case_id"])],
                    random_keys[str(item["episode_id"])],
                    str(item["episode_id"]),
                )
            )
            quota = quotas[family]
            if len(candidates) < quota:
                raise ValueError(f"Not enough {dataset}/{family} episodes for quota {quota}")
            chosen = candidates[:quota]
            dataset_selected.extend(chosen)
            selected_case_counts.update(str(item["case_id"]) for item in chosen)
        if len(dataset_selected) != per_dataset:
            raise AssertionError("Stratified selection produced the wrong sample size")
        selected.extend(dataset_selected)
        private_report[dataset] = {
            "available_episodes": sum(len(items) for items in families.values()),
            "selected_episodes": len(dataset_selected),
            "available_cases": len(
                {str(item["case_id"]) for items in families.values() for item in items}
            ),
            "selected_cases": len({str(item["case_id"]) for item in dataset_selected}),
            "intervention_quotas": quotas,
            "selected_case_multiplicity": dict(sorted(selected_case_counts.items())),
        }
    random.Random(seed + 7919).shuffle(selected)
    return selected, private_report


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1 + (z * z / trials)
    centre = proportion + (z * z / (2 * trials))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / trials) + (z * z / (4 * trials * trials))
    )
    return (centre - margin) / denominator


def _write_review_sheet(
    path: Path, review_ids: list[str], dimensions: list[str]
) -> None:
    fields = ["review_id", *dimensions]
    fields += [
        "visible_hard_failure",
        "accept_for_sft",
        "strongest_turn",
        "weakest_turn",
        "required_revision",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_id in review_ids:
            writer.writerow({"review_id": review_id})


def export_delta_packets(
    episodes_path: Path,
    sidecar_path: Path,
    rubric_path: Path,
    output_dir: Path,
    *,
    per_dataset: int = 35,
    seed: int = 20260824,
) -> dict[str, Any]:
    episodes = _read_jsonl(episodes_path)
    sidecars = _read_jsonl(sidecar_path)
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    blind = rubric["blind_judge_contract"]
    base_dimensions = list(blind["dimensions"])
    all_dimensions = [*base_dimensions, *SUPPLEMENTAL_DIMENSIONS]
    selected, sampling = select_stratified(episodes, sidecars, per_dataset, seed)

    public_packets: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    common_systems: set[str] = set()
    sidecar_index = {str(item["episode_id"]): item for item in sidecars}
    for index, episode in enumerate(selected, start=1):
        review_id = f"review-{index:03d}"
        view, system_text = _episode_public_view(episode)
        common_systems.add(system_text)
        public_packets.append({"review_id": review_id, **view})
        private = sidecar_index[str(episode["episode_id"])]
        private_key.append(
            {
                "review_id": review_id,
                "episode_id": episode["episode_id"],
                "dataset": episode["dataset"],
                "case_id": episode["case_id"],
                "intervention_id": private["intervention_id"],
                "operator": private["operator"],
            }
        )
    if len(common_systems) != 1:
        raise ValueError("Selected episodes do not share one common system policy")

    public = output_dir / "public_share_only"
    private = output_dir / "private_do_not_share_with_judges"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    _write_jsonl(public / "review_packets_delta.jsonl", public_packets)
    write_json(private / "review_key.json", {"items": private_key})
    write_json(private / "sampling_report.json", sampling)

    order_a = [item["review_id"] for item in public_packets]
    order_b = order_a.copy()
    random.Random(seed + 1).shuffle(order_a)
    random.Random(seed + 2).shuffle(order_b)
    _write_review_sheet(public / "reviewer_A.csv", order_a, all_dimensions)
    _write_review_sheet(public / "reviewer_B.csv", order_b, all_dimensions)

    with (private / "adjudication.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "dimension",
                "reviewer_A_score",
                "reviewer_B_score",
                "adjudicated_score",
                "adjudication_reason",
            ],
        ).writeheader()

    guide = f"""# Blind review of causal-recovery SFT trajectories

Judge only the supplied student-visible state and executed deltas. The delta
representation removes unchanged copies of the Harness state; it does not add a
hidden answer or machine verdict. Do not search for the case or infer the
dataset/intervention. Reviewer A and Reviewer B work independently.

## Score anchors

Use 1 (unusable), 2 (major defects), 3 (mixed), 4 (good), or 5 (excellent).
A 4 requires no material defect; a 5 requires unusually clear and efficient
reasoning. Do not use 4 as a default when evidence is ambiguous.

## UTQR-12 H8 dimensions

- `prospective_reasoning`: the note justifies the next action before seeing its result and does not presume success.
- `visible_evidence_grounding`: cited atoms, values, roles, intervals, and conflicts match the visible state.
- `belief_update`: later reasoning actually reacts to fresh evidence, rejection, conflict, or abstention.
- `method_fit`: the selected tool can discriminate the named uncertainty and is not busywork.
- `locality`: challenges and repairs touch only dependent fields while independent support is preserved.
- `naturalness`: the trajectory reads like a compact analyst notebook rather than a repeated template.

## Development addendum (not folded into UTQR C9)

- `anti_shortcut_semantic_identifiability`: the action is selected from a concrete semantic clue, not merely the tool menu, step position, or a generic repair pattern.
- `per_step_sft_pedagogical_value`: every turn teaches a distinct state-dependent decision and its output is used downstream.

For every item, name the strongest and weakest assistant turns as `A1`, `A2`,
and so on. Notes must cite visible turn evidence. Mark `visible_hard_failure=yes`
for fabricated evidence, a result-presuming measurement, wrong-atom challenge,
off-target mutation, stale/rejected evidence reuse, unused measurement,
unsupported revision/binding, or premature stop. `accept_for_sft=yes` requires
all eight dimensions to be at least 4 and no visible hard failure or required
revision.

Any dimension gap above one point requires later adjudication by a third view.
Passing this packet establishes only sampled development-variant quality; it
does not establish sealed-case generalization or trained-student gain.
"""
    (public / "REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    (public / "COMMON_SYSTEM_POLICY.txt").write_text(
        Redactor().text(next(iter(common_systems))), encoding="utf-8"
    )

    packet_path = public / "review_packets_delta.jsonl"
    packet_text = packet_path.read_text(encoding="utf-8")
    forbidden_patterns = {
        "dataset_name": r"(?i)VideoSham|VidDiffBench|\bVCSL\b",
        "case_id": (
            r"(?i)videosham-\d+|viddiff-surgery_\d+|"
            r"vcsl-[0-9a-f]{32}|TASK-[A-Z0-9_-]+"
        ),
        "windows_path": r"(?i)\b[A-Z]:\\",
        "url": r"(?i)https?://",
        "oracle_token": r"(?i)oracle",
        "private_intervention": r"(?i)private_intervention|intervention_id|\bU\d{2}_",
        "verifier_or_machine_score": r"(?i)verifier_result|automated_core|blind_dimension_proxy",
    }
    privacy_hits = {
        name: len(re.findall(pattern, packet_text))
        for name, pattern in forbidden_patterns.items()
    }
    privacy_passed = all(count == 0 for count in privacy_hits.values())
    selected_full_bytes = sum(
        len(
            json.dumps(
                item["messages"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        for item in selected
    )
    representation_audit = {
        "artifact": "blind_delta_representation_audit_v1",
        "packets": len(public_packets),
        "turns": sum(len(item["turns"]) for item in public_packets),
        "all_projected_state_deltas_reconstructable": True,
        "privacy_patterns": privacy_hits,
        "privacy_passed": privacy_passed,
        "selected_full_transcript_bytes": selected_full_bytes,
        "public_delta_packet_bytes": packet_path.stat().st_size,
        "byte_reduction_fraction": round(
            1.0 - (packet_path.stat().st_size / selected_full_bytes), 6
        ),
        "scope": (
            "Privacy is checked at the public packet sharing boundary; private key "
            "and sampling files are intentionally excluded."
        ),
    }
    write_json(private / "representation_and_privacy_audit.json", representation_audit)
    if not privacy_passed:
        raise ValueError(f"Blind public packet contains private tokens: {privacy_hits}")

    perfect_lb = wilson_lower_bound(per_dataset, per_dataset)
    one_failure_lb = wilson_lower_bound(per_dataset - 1, per_dataset)
    manifest = {
        "artifact": "recovery_lattice_blind_delta_packet_v1",
        "rubric_id": rubric["rubric_id"],
        "representation": "student_visible_initial_state_plus_executed_state_deltas",
        "packets": len(public_packets),
        "per_dataset_private_stratum": per_dataset,
        "selection_seed": seed,
        "selection_predeclared_and_nonadaptive": True,
        "selection_uses_private_intervention_only_for_coverage": True,
        "selection_uses_oracle_verifier_or_automated_score": False,
        "minimum_independent_judges": blind["minimum_independent_judges"],
        "utqr_h8_dimensions": base_dimensions,
        "development_addendum_dimensions": SUPPLEMENTAL_DIMENSIONS,
        "minimum_each_dimension": 4.0,
        "maximum_unadjudicated_range": blind["maximum_unadjudicated_range"],
        "acceptance_wilson95_lower_bound_minimum": 0.9,
        "perfect_sample_wilson95_lower_bound": round(perfect_lb, 6),
        "one_failure_wilson95_lower_bound": round(one_failure_lb, 6),
        "zero_failure_required_at_this_sample_size": one_failure_lb < 0.9,
        "all_projected_state_deltas_reconstructable": True,
        "public_packet_privacy_audit_passed": True,
        "public_packet_byte_reduction_fraction": representation_audit[
            "byte_reduction_fraction"
        ],
        "human_review_completed": False,
        "training_qualified": False,
        "qualification_blockers": [
            "two_independent_blind_reviews_not_completed",
            "sealed_entity_disjoint_holdout_not_completed",
            "student_closed_loop_gain_not_evaluated",
        ],
        "share_boundary": "Share only public_share_only/ with judges.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-dataset", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    manifest = export_delta_packets(
        args.episodes,
        args.sidecar,
        args.rubric,
        args.output_dir,
        per_dataset=args.per_dataset,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
