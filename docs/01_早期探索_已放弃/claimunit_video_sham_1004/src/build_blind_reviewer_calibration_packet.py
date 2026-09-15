"""Interleave hidden clean and single-defect sentinels into the blind review packet.

Sentinels are reviewer-calibration items only.  They never become SFT positives,
never alter UTQR scores, and are excluded from development quality estimates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from export_blind_recovery_delta_review_packets import (
    DEFAULT_EPISODES,
    DEFAULT_RUBRIC,
    SUPPLEMENTAL_DIMENSIONS,
    _episode_public_view,
)
from export_blind_recovery_review_packets import _read_jsonl, _write_jsonl
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_PRIMARY = BASE / "recovery_lattice_blind_delta_packet_v1"
DEFAULT_OUTPUT = BASE / "recovery_lattice_blind_delta_packet_v2_calibrated"

CONTROL_TOOLS = {
    "challenge_evidence",
    "set_field_decision",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "revise_claim_fields",
    "bind_evidence",
    "commit_decision",
    "stop",
}


def _measurement_index(packet: dict[str, Any]) -> int:
    return next(
        index
        for index, turn in enumerate(packet["turns"])
        if str(turn["tool_call"]["name"]) not in CONTROL_TOOLS
    )


def _created_ids(turn: dict[str, Any]) -> list[str]:
    result = turn.get("tool_result_without_repeated_state") or {}
    return [str(item) for item in result.get("created_evidence_ids") or []]


def _first_result_fact(value: Any, path: str = "") -> tuple[str, Any] | None:
    if isinstance(value, dict):
        preferred = (
            "preferred_video",
            "direction",
            "start_s",
            "end_s",
            "affected_source_interval_s",
            "affected_suspect_interval_s",
            "source_interval_s",
            "query_interval_s",
            "confidence",
        )
        for key in preferred:
            if key in value and isinstance(value[key], (str, int, float, bool, list)):
                return f"{path}.{key}".strip("."), value[key]
        for key, item in value.items():
            found = _first_result_fact(item, f"{path}.{key}".strip("."))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _first_result_fact(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _result_presumption(packet: dict[str, Any], severity: str) -> None:
    index = _measurement_index(packet)
    turn = packet["turns"][index]
    found = _first_result_fact(turn["tool_result_without_repeated_state"])
    fact = (
        f"{found[0]}={json.dumps(found[1], ensure_ascii=False)}"
        if found
        else "the expected directional result"
    )
    if severity == "subtle":
        turn["reasoning"] = (
            f"The comparison is {fact}; I will run {turn['tool_call']['name']} to "
            "confirm this interval before repairing the claim."
        )
    else:
        turn["reasoning"] = (
            f"Before observing the measurement, its result is already {fact}. "
            f"I will call {turn['tool_call']['name']} only to certify that conclusion."
        )


def _wrong_atom_citation(packet: dict[str, Any], _: str) -> None:
    index = _measurement_index(packet)
    fresh = _created_ids(packet["turns"][index])
    if not fresh or index + 1 >= len(packet["turns"]):
        raise ValueError("Wrong-atom sentinel requires fresh evidence and a later turn")
    existing = list((packet["visible_initial_state"].get("evidence_store") or {}).keys())
    wrong = next((item for item in existing if item not in fresh), None)
    if wrong is None:
        raise ValueError("Wrong-atom sentinel requires a distinct visible atom")
    next_turn = packet["turns"][index + 1]
    if fresh[0] not in next_turn["reasoning"]:
        raise ValueError("Fresh atom is not cited in the next rationale")
    next_turn["reasoning"] = next_turn["reasoning"].replace(fresh[0], wrong, 1)


def _ignore_belief_update(packet: dict[str, Any], severity: str) -> None:
    index = _measurement_index(packet)
    next_turn = packet["turns"][index + 1]
    if severity == "subtle":
        next_turn["reasoning"] = (
            "The fresh observation can be recorded, but it need not affect the earlier "
            f"belief. I will proceed with {next_turn['tool_call']['name']} on the same "
            "basis as before the measurement."
        )
    else:
        next_turn["reasoning"] = (
            "I will ignore the fresh result entirely and preserve the stale belief. "
            f"The next operation is {next_turn['tool_call']['name']} regardless of what "
            "the tool returned."
        )


def _menu_only_router(packet: dict[str, Any], severity: str) -> None:
    index = _measurement_index(packet)
    turn = packet["turns"][index]
    if severity == "subtle":
        turn["reasoning"] = (
            f"Among the currently available operations, {turn['tool_call']['name']} is "
            "the expected next transition, so I will select it."
        )
    else:
        turn["reasoning"] = (
            f"I choose {turn['tool_call']['name']} only because it appears in the "
            "eligible-tool menu; no evidence question or semantic distinction is needed."
        )


def _off_target_scope(packet: dict[str, Any], _: str) -> None:
    invariant = list(packet.get("claim_schema", {}).get("invariant_fields") or [])
    if not invariant:
        raise ValueError("Off-target sentinel requires an invariant field")
    turn = next(
        item
        for item in packet["turns"]
        if item["tool_call"]["name"] == "challenge_evidence"
    )
    arguments = turn["tool_call"]["arguments"]
    target_fields = list(arguments.get("target_fields") or [])
    if invariant[0] not in target_fields:
        target_fields.append(invariant[0])
    arguments["target_fields"] = target_fields


def _repeated_template(packet: dict[str, Any], severity: str) -> None:
    for turn in packet["turns"]:
        tool = turn["tool_call"]["name"]
        turn["reasoning"] = (
            f"The state has changed. I will call {tool} because it is available. "
            "This advances the workflow."
            if severity == "subtle"
            else f"Next legal step: {tool}. Execute the template and continue."
        )


def _no_state_consequence(packet: dict[str, Any], severity: str) -> None:
    index = min(_measurement_index(packet) + 1, len(packet["turns"]) - 2)
    turn = packet["turns"][index]
    turn["reasoning"] = (
        "The preceding observation has no downstream decision role and I will not "
        f"explain what it changes. I will call {turn['tool_call']['name']} anyway."
        if severity == "obvious"
        else (
            f"The observation is now in the record. I will call {turn['tool_call']['name']} "
            "next, without linking it to any field, belief, or later decision."
        )
    )


def _premature_stop_claim(packet: dict[str, Any], severity: str) -> None:
    turn = packet["turns"][-1]
    if turn["tool_call"]["name"] != "stop":
        raise ValueError("Premature-stop sentinel requires a terminal stop turn")
    turn["reasoning"] = (
        "A high-priority obligation and an unsupported field remain open, but I will "
        "request stop without resolving either one."
        if severity == "obvious"
        else (
            "One cited evidence conflict is still unresolved; nevertheless, the episode "
            "can stop now without another assessment or binding check."
        )
    )


Mutation = tuple[str, str, Callable[[dict[str, Any], str], None]]
MUTATIONS: list[Mutation] = [
    ("result_presumption_before_measurement", "prospective_reasoning", _result_presumption),
    ("wrong_visible_atom_citation", "visible_evidence_grounding", _wrong_atom_citation),
    ("fresh_result_ignored", "belief_update", _ignore_belief_update),
    ("menu_only_method_router", "method_fit", _menu_only_router),
    ("off_target_invariant_field", "locality", _off_target_scope),
    ("trajectory_wide_stock_template", "naturalness", _repeated_template),
    (
        "semantic_clue_replaced_by_menu_position",
        "anti_shortcut_semantic_identifiability",
        _menu_only_router,
    ),
    (
        "step_without_stated_state_consequence",
        "per_step_sft_pedagogical_value",
        _no_state_consequence,
    ),
]


def apply_mutation(
    packet: dict[str, Any], mutation_name: str, severity: str
) -> dict[str, Any]:
    mutated = copy.deepcopy(packet)
    match = next(item for item in MUTATIONS if item[0] == mutation_name)
    match[2](mutated, severity)
    return mutated


def _write_sheet(path: Path, ids: list[str], dimensions: list[str]) -> None:
    fields = ["review_id", *dimensions]
    fields.extend(
        [
            "visible_hard_failure",
            "accept_for_sft",
            "strongest_turn",
            "weakest_turn",
            "required_revision",
            "reviewer_notes",
        ]
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_id in ids:
            writer.writerow({"review_id": review_id})


def build_packet(
    primary_root: Path,
    episodes_path: Path,
    rubric_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260824,
) -> dict[str, Any]:
    primary_packets = _read_jsonl(
        primary_root / "public_share_only/review_packets_delta.jsonl"
    )
    primary_key_rows = json.loads(
        (
            primary_root / "private_do_not_share_with_judges/review_key.json"
        ).read_text(encoding="utf-8")
    )["items"]
    primary_key = {str(item["review_id"]): item for item in primary_key_rows}
    selected_episode_ids = {str(item["episode_id"]) for item in primary_key_rows}
    episodes = [
        item
        for item in _read_jsonl(episodes_path)
        if str(item["episode_id"]) not in selected_episode_ids
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_dataset[str(episode["dataset"])].append(episode)
    extra_episodes: list[dict[str, Any]] = []
    for index, dataset in enumerate(sorted(by_dataset)):
        candidates = list(by_dataset[dataset])
        random.Random(seed + 401 * (index + 1)).shuffle(candidates)
        if len(candidates) < 8:
            raise ValueError(f"Need eight unused episodes for {dataset}")
        extra_episodes.extend(candidates[:8])

    clean_quota = {dataset: count for dataset, count in zip(sorted(by_dataset), [3, 3, 2])}
    negative_schedule = [item for item in MUTATIONS for _ in range(2)]
    random.Random(seed + 909).shuffle(negative_schedule)
    negative_index = 0
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    construction_rows: list[dict[str, Any]] = []
    for packet in primary_packets:
        key = primary_key[str(packet["review_id"])]
        items.append(
            (
                {name: copy.deepcopy(value) for name, value in packet.items() if name != "review_id"},
                {
                    **key,
                    "review_role": "primary_quality",
                    "expected_disposition": "score_normally",
                },
            )
        )

    used_clean: defaultdict[str, int] = defaultdict(int)
    severity_cycle = ["obvious", "subtle"]
    for episode in extra_episodes:
        packet, _ = _episode_public_view(episode)
        dataset = str(episode["dataset"])
        if used_clean[dataset] < clean_quota[dataset]:
            used_clean[dataset] += 1
            items.append(
                (
                    packet,
                    {
                        "episode_id": episode["episode_id"],
                        "dataset": dataset,
                        "case_id": episode["case_id"],
                        "review_role": "calibration_clean",
                        "expected_disposition": "accept",
                    },
                )
            )
            construction_rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "review_role": "calibration_clean",
                    "content_changed_from_executed_delta_view": False,
                }
            )
            continue
        mutation_name, expected_dimension, _ = negative_schedule[negative_index]
        severity = severity_cycle[negative_index % 2]
        negative_index += 1
        mutated_packet = apply_mutation(packet, mutation_name, severity)
        reasoning_changed = [
            before["assistant_turn"]
            for before, after in zip(packet["turns"], mutated_packet["turns"])
            if before["reasoning"] != after["reasoning"]
        ]
        tool_call_changed = [
            before["assistant_turn"]
            for before, after in zip(packet["turns"], mutated_packet["turns"])
            if before["tool_call"] != after["tool_call"]
        ]
        tool_result_changed = [
            before["assistant_turn"]
            for before, after in zip(packet["turns"], mutated_packet["turns"])
            if before["tool_result_without_repeated_state"]
            != after["tool_result_without_repeated_state"]
        ]
        state_delta_changed = [
            before["assistant_turn"]
            for before, after in zip(packet["turns"], mutated_packet["turns"])
            if before["visible_state_changes"] != after["visible_state_changes"]
        ]
        static_context_unchanged = all(
            packet.get(name) == mutated_packet.get(name)
            for name in (
                "instructions",
                "representation",
                "task_instruction",
                "claim_schema",
                "visible_initial_state",
            )
        )
        expected_surface = (
            len(tool_call_changed) == 1 and not reasoning_changed
            if mutation_name == "off_target_invariant_field"
            else len(reasoning_changed) == len(packet["turns"])
            and not tool_call_changed
            if mutation_name == "trajectory_wide_stock_template"
            else len(reasoning_changed) == 1 and not tool_call_changed
        )
        construction_passed = bool(
            static_context_unchanged
            and not tool_result_changed
            and not state_delta_changed
            and expected_surface
        )
        if not construction_passed:
            raise AssertionError(
                f"Sentinel mutation changed an unexpected surface: {mutation_name}"
            )
        construction_rows.append(
            {
                "episode_id": episode["episode_id"],
                "review_role": "calibration_negative",
                "mutation": mutation_name,
                "expected_dimension": expected_dimension,
                "severity": severity,
                "reasoning_turns_changed": reasoning_changed,
                "tool_call_turns_changed": tool_call_changed,
                "tool_result_turns_changed": tool_result_changed,
                "state_delta_turns_changed": state_delta_changed,
                "static_context_unchanged": static_context_unchanged,
                "single_family_construction_passed": construction_passed,
            }
        )
        items.append(
            (
                mutated_packet,
                {
                    "episode_id": episode["episode_id"],
                    "dataset": dataset,
                    "case_id": episode["case_id"],
                    "review_role": "calibration_negative",
                    "expected_disposition": "reject",
                    "mutation": mutation_name,
                    "expected_dimension": expected_dimension,
                    "severity": severity,
                },
            )
        )
    if negative_index != 16:
        raise AssertionError("Expected sixteen negative sentinels")

    random.Random(seed + 1619).shuffle(items)
    public_packets: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for index, (packet, key) in enumerate(items, start=1):
        review_id = f"review-{index:03d}"
        public_packets.append({"review_id": review_id, **packet})
        private_rows.append({**key, "review_id": review_id})

    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    dimensions = [
        *rubric["blind_judge_contract"]["dimensions"],
        *SUPPLEMENTAL_DIMENSIONS,
    ]
    public = output_dir / "public_share_only"
    private = output_dir / "private_do_not_share_with_judges"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    packet_path = public / "review_packets_delta.jsonl"
    _write_jsonl(packet_path, public_packets)
    write_json(private / "review_key.json", {"items": private_rows})
    construction_audit = {
        "artifact": "blind_reviewer_sentinel_construction_audit_v1",
        "clean_controls": sum(
            item["review_role"] == "calibration_clean" for item in construction_rows
        ),
        "negative_sentinels": sum(
            item["review_role"] == "calibration_negative"
            for item in construction_rows
        ),
        "all_clean_controls_unmodified": all(
            item.get("content_changed_from_executed_delta_view") is False
            for item in construction_rows
            if item["review_role"] == "calibration_clean"
        ),
        "all_negative_tool_results_unchanged": all(
            not item.get("tool_result_turns_changed")
            for item in construction_rows
            if item["review_role"] == "calibration_negative"
        ),
        "all_negative_state_deltas_unchanged": all(
            not item.get("state_delta_turns_changed")
            for item in construction_rows
            if item["review_role"] == "calibration_negative"
        ),
        "all_negative_static_context_unchanged": all(
            item.get("static_context_unchanged")
            for item in construction_rows
            if item["review_role"] == "calibration_negative"
        ),
        "all_single_family_constructions_passed": all(
            item.get("single_family_construction_passed")
            for item in construction_rows
            if item["review_role"] == "calibration_negative"
        ),
        "items": construction_rows,
    }
    construction_audit["passed"] = all(
        construction_audit[name]
        for name in (
            "all_clean_controls_unmodified",
            "all_negative_tool_results_unchanged",
            "all_negative_state_deltas_unchanged",
            "all_negative_static_context_unchanged",
            "all_single_family_constructions_passed",
        )
    )
    write_json(private / "sentinel_construction_audit.json", construction_audit)

    ids = [item["review_id"] for item in public_packets]
    order_a = ids.copy()
    order_b = ids.copy()
    random.Random(seed + 1).shuffle(order_a)
    random.Random(seed + 2).shuffle(order_b)
    _write_sheet(public / "reviewer_A.csv", order_a, dimensions)
    _write_sheet(public / "reviewer_B.csv", order_b, dimensions)
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

    base_guide = (
        primary_root / "public_share_only/REVIEW_GUIDE.md"
    ).read_text(encoding="utf-8")
    calibration_guide = base_guide + """

## Reviewer calibration items

A small number of hidden clean controls and single-defect sentinels are mixed
into the packet. Their identities and expected dimensions are unavailable to
reviewers. They are not candidate SFT rows and are excluded from trajectory
quality estimates. They only test whether each reviewer distinguishes grounded
evidence reasoning from superficially legal action sequences. Score every item
normally; do not try to guess which items are sentinels.
"""
    (public / "REVIEW_GUIDE.md").write_text(calibration_guide, encoding="utf-8")
    (public / "COMMON_SYSTEM_POLICY.txt").write_text(
        (primary_root / "public_share_only/COMMON_SYSTEM_POLICY.txt").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    text = packet_path.read_text(encoding="utf-8")
    forbidden = {
        "dataset_name": r"(?i)VideoSham|VidDiffBench|\bVCSL\b",
        "case_id": r"(?i)videosham-\d+|viddiff-surgery_\d+|vcsl-[0-9a-f]{32}|TASK-[A-Z0-9_-]+",
        "path_or_url": r"(?i)\b[A-Z]:\\|https?://",
        "private_label": r"(?i)review_role|calibration_negative|expected_dimension|mutation|intervention_id|\bU\d{2}_",
        "oracle_or_verifier": r"(?i)oracle|verifier_result|automated_core|blind_dimension_proxy",
    }
    privacy_hits = {name: len(re.findall(pattern, text)) for name, pattern in forbidden.items()}
    if any(privacy_hits.values()):
        raise ValueError(f"Calibrated public packet leaks private labels: {privacy_hits}")

    role_counts: dict[str, int] = defaultdict(int)
    dataset_role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mutation_counts: dict[str, int] = defaultdict(int)
    for item in private_rows:
        role_counts[str(item["review_role"])] += 1
        dataset_role_counts[str(item["dataset"])][str(item["review_role"])] += 1
        if item.get("mutation"):
            mutation_counts[str(item["mutation"])] += 1
    manifest = {
        "artifact": "recovery_lattice_blind_delta_packet_v2_calibrated",
        "rubric_id": rubric["rubric_id"],
        "total_public_items": len(public_packets),
        "primary_quality_items": role_counts["primary_quality"],
        "calibration_clean_items": role_counts["calibration_clean"],
        "calibration_negative_items": role_counts["calibration_negative"],
        "calibration_mutation_families": len(mutation_counts),
        "each_mutation_family_count": dict(sorted(mutation_counts.items())),
        "private_dataset_role_counts": {
            dataset: dict(counts) for dataset, counts in sorted(dataset_role_counts.items())
        },
        "reviewers_informed_that_hidden_calibration_exists": True,
        "sentinel_identity_visible_to_reviewers": False,
        "sentinels_included_in_quality_estimate": False,
        "negative_sentinels_included_in_sft": False,
        "reviewer_reliability_rule": {
            "minimum_negative_detection": "15/16",
            "minimum_clean_acceptance": "7/8",
            "negative_detection_requires": (
                "reject or hard-failure flag AND expected dimension score <=3"
            ),
        },
        "public_privacy_hits": privacy_hits,
        "sentinel_construction_audit_passed": construction_audit["passed"],
        "human_review_completed": False,
        "training_qualified": False,
        "share_boundary": "Share only public_share_only/ with judges.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    manifest = build_packet(
        args.primary_root,
        args.episodes,
        args.rubric,
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
