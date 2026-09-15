"""Aggregate the fixed 24-episode current-six falsification diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from control_plane_falsification_arms import ARMS
from pipeline import ROOT, write_json


DEFAULT_INPUT = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "arms/llm_diagnostic_current6_v1_4bit_attempt3"
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _claim_values(episode: dict[str, Any]) -> dict[str, Any] | None:
    claims = (episode.get("final_state") or {}).get("candidate_claims") or []
    if claims:
        return {
            str(name): field.get("value")
            for name, field in (claims[0].get("fields") or {}).items()
        }
    accepted = [
        turn
        for turn in episode.get("turns") or []
        if turn.get("generated_tool") == "create_candidate_claim"
        and turn.get("status") == "accepted"
    ]
    if accepted:
        return (accepted[-1].get("generated_arguments") or {}).get("field_values")
    return None


def run(input_dir: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for mode in ("clean", "repair"):
        for dataset in ("VideoSham", "VCSL", "VidDiffBench"):
            for arm in ARMS:
                path = input_dir / f"{mode}_{dataset.lower()}_{arm}.json"
                episode = json.loads(path.read_text(encoding="utf-8"))
                episode["source_path"] = str(path)
                episodes.append(episode)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[(str(episode["mode"]), str(episode["policy_view_arm"]))].append(episode)
    grouped = []
    for (mode, arm), items in sorted(groups.items()):
        successes = sum(bool(item.get("terminal_success")) for item in items)
        grouped.append(
            {
                "mode": mode,
                "arm": arm,
                "episodes": len(items),
                "terminal_successes": successes,
                "terminal_rate": round(successes / len(items), 4),
                "terminal_rate_wilson95": wilson(successes, len(items)),
                "mean_turns": round(sum(item["turn_count"] for item in items) / len(items), 3),
                "environment_rejections": sum(item["rejection_count"] for item in items),
                "generation_attempts": sum(
                    int(turn.get("generation_attempts") or 0)
                    for item in items
                    for turn in item.get("turns") or []
                ),
                "accepted_output_tokens": sum(
                    int(turn.get("generated_tokens") or 0)
                    for item in items
                    for turn in item.get("turns") or []
                ),
                "seconds": round(sum(float(item.get("episode_seconds") or 0) for item in items), 3),
            }
        )
    repair_by_dataset = []
    for dataset in ("VideoSham", "VCSL", "VidDiffBench"):
        arm_items = {
            str(item["policy_view_arm"]): item
            for item in episodes
            if item["mode"] == "repair" and item["dataset"] == dataset
        }
        full = arm_items["full"]
        full_sequence = full["tool_sequence"]
        full_claim = _claim_values(full)
        repair_by_dataset.append(
            {
                "dataset": dataset,
                "full_terminal": full["terminal_success"],
                "structure_terminal": arm_items["structure_only"]["terminal_success"],
                "shuffled_terminal": arm_items["shuffled_evidence"]["terminal_success"],
                "evidence_only_terminal": arm_items["evidence_only"]["terminal_success"],
                "structure_sequence_equals_full": arm_items["structure_only"]["tool_sequence"] == full_sequence,
                "shuffled_sequence_equals_full": arm_items["shuffled_evidence"]["tool_sequence"] == full_sequence,
                "structure_claim_equals_full": _claim_values(arm_items["structure_only"]) == full_claim,
                "shuffled_claim_equals_full": _claim_values(arm_items["shuffled_evidence"]) == full_claim,
                "shuffled_max_visible_swaps": max(
                    [int(turn.get("evidence_semantic_swaps_visible") or 0) for turn in arm_items["shuffled_evidence"]["turns"]]
                    or [0]
                ),
            }
        )
    return {
        "artifact": "control_plane_falsification_current6_diagnostic_summary_v1",
        "diagnostic_only_not_generalization_evidence": True,
        "episodes": len(episodes),
        "grouped": grouped,
        "repair_matched_comparison": repair_by_dataset,
        "repair_exact_sequence_structure_vs_full": sum(item["structure_sequence_equals_full"] for item in repair_by_dataset),
        "repair_exact_sequence_shuffled_vs_full": sum(item["shuffled_sequence_equals_full"] for item in repair_by_dataset),
        "repair_exact_claim_structure_vs_full": sum(item["structure_claim_equals_full"] for item in repair_by_dataset),
        "repair_exact_claim_shuffled_vs_full": sum(item["shuffled_claim_equals_full"] for item in repair_by_dataset),
        "episodes_detail": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.input_dir)
    output = args.output or args.input_dir / "diagnostic_aggregate.json"
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes_detail"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
