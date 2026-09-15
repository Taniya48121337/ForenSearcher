"""Build a label-free review packet from accepted cross-dataset episodes."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl


REVIEW_RUBRIC = {
    "scale": "1=unacceptable, 3=usable with edits, 5=publication-grade",
    "dimensions": [
        "tool_step_necessity",
        "visible_fact_grounding",
        "method_task_fit",
        "conflict_or_uncertainty_handling",
        "repair_locality_when_applicable",
        "natural_human_reasoning",
        "final_claim_support",
        "privacy_and_label_blindness",
    ],
    "hard_fail_questions": [
        "Does any step mention an oracle, verifier, hidden label, local path, prompt, rubric, or checker?",
        "Does any tool call fail to close the gap named in its reasoning?",
        "Does a repair step alter an unaffected field or reuse dropped evidence?",
        "Does the final decision rely on majority, recency, or confidence without method fit?",
    ],
}


def _load(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return artifact if "episode" in artifact else {"episode": artifact}


def build(paths: list[Path], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = [_load(path) for path in paths]
    order = list(range(len(artifacts)))
    random.Random(seed).shuffle(order)
    packet: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for blind_index, source_index in enumerate(order, start=1):
        artifact = artifacts[source_index]
        episode = artifact["episode"]
        blind_id = f"BLIND-{blind_index:03d}"
        turns = []
        for turn in episode.get("turns") or []:
            turns.append(
                {
                    "turn": turn.get("turn"),
                    "tool": turn.get("generated_tool"),
                    "reasoning": turn.get("reasoning"),
                    "arguments": copy.deepcopy(turn.get("generated_arguments") or {}),
                    "status": turn.get("status"),
                }
            )
        mode = artifact.get("mode") or episode.get("mode")
        packet.append(
            {
                "blind_id": blind_id,
                "mode": mode,
                "terminal_success": bool(episode.get("terminal_success")),
                "tool_sequence": copy.deepcopy(episode.get("tool_sequence") or []),
                "turns": turns,
                "review_form": {
                    "dimension_scores": {
                        dimension: None for dimension in REVIEW_RUBRIC["dimensions"]
                    },
                    "hard_fail": None,
                    "hard_fail_reason": None,
                    "keep_for_sft": None,
                    "reviewer_notes": None,
                },
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "dataset": artifact.get("dataset") or episode.get("dataset"),
                "case_id": artifact.get("case_id") or episode.get("case_id"),
                "mode": mode,
                "source_episode": str(paths[source_index]),
            }
        )
    key = {
        "seed": seed,
        "label_information_in_review_packet": False,
        "oracle_or_verifier_information_in_review_packet": False,
        "rubric": REVIEW_RUBRIC,
        "mapping": key_rows,
    }
    return packet, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    paths = list(args.episode)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths.extend(Path(str(item["episode_path"])) for item in manifest.get("items") or [])
    if not paths:
        parser.error("provide at least one --episode or --manifest")
    packet, key = build(paths, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "blind_review_packet.jsonl", packet)
    write_json(args.output_dir / "review_key.keep_separate.json", key)
    write_json(args.output_dir / "review_rubric.json", REVIEW_RUBRIC)
    print(
        json.dumps(
            {
                "episodes": len(packet),
                "turns": sum(len(item["turns"]) for item in packet),
                "packet": str(args.output_dir / "blind_review_packet.jsonl"),
                "key": str(args.output_dir / "review_key.keep_separate.json"),
                "oracle_or_verifier_in_packet": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
