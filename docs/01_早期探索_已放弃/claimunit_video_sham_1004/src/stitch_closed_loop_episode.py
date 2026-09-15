"""Stitch accepted closed-loop suffixes and re-audit the complete episode."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from audit_student_closed_loop_quality import _audit_episode


ACCEPTED_STATUSES = {"accepted", "executed"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_entry(turn: dict[str, Any], step_index: int) -> dict[str, Any]:
    """Rebuild one truthful, compact execution-history item from an accepted turn."""

    tool = str(turn.get("generated_tool") or "")
    past_input = copy.deepcopy(turn.get("generated_arguments") or {})
    if tool.startswith("compare_"):
        past_input.pop("comparison_statement", None)
    elif tool == "create_candidate_claim":
        past_input = {"claim_id": past_input.get("claim_id")}
    elif tool == "bind_evidence":
        past_input = {
            "claim_id": past_input.get("claim_id"),
            "bound_fields": [
                item.get("field") for item in past_input.get("binding_items") or []
            ],
        }
    elif tool == "assess_evidence_candidate":
        past_input = {
            "evidence_id": past_input.get("evidence_id"),
            "disposition": past_input.get("disposition"),
            "defect_codes": list(past_input.get("defect_codes") or []),
        }
    return {
        "step_index": step_index,
        "past_action": {"executed_tool": tool, "past_input": past_input},
        "environment_outcome": copy.deepcopy(turn.get("environment_outcome") or {}),
    }


def _rebase_sft_histories(turns: list[dict[str, Any]]) -> None:
    """Make every stitched SFT datum describe its true full pre-action prefix."""

    history: list[dict[str, Any]] = []
    for turn in turns:
        datum = turn.get("sft_datum") or {}
        messages = datum.get("messages") or []
        if len(messages) != 3:
            raise ValueError(f"turn {turn.get('turn')} has no three-message SFT datum")
        payload = json.loads(str(messages[1].get("content") or ""))
        payload["past_execution_history"] = copy.deepcopy(history)
        payload["latest_environment_observation"] = (
            copy.deepcopy(history[-1]["environment_outcome"]) if history else {}
        )
        messages[1]["content"] = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        history.append(_history_entry(turn, len(history)))


def _exportable_sft_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact["quality"].get("accepted_for_high_quality_sft"):
        return []
    episode = artifact["episode"]
    rows: list[dict[str, Any]] = []
    for turn in episode.get("turns") or []:
        datum = copy.deepcopy(turn.get("sft_datum") or {})
        if not datum:
            continue
        metadata = datum.setdefault("metadata", {})
        metadata.update(
            {
                "row_id": ":".join(
                    (
                        str(artifact["dataset"]),
                        str(artifact["case_id"]),
                        "student-closed-loop",
                        str(artifact["mode"]),
                        "stitched",
                        str(turn["turn"]),
                    )
                ),
                "case_id": artifact["case_id"],
                "source_split": episode.get("source_split", "test"),
                "trajectory_mode": artifact["mode"],
                "trajectory_type": episode.get("trajectory_type"),
                "turn_index": turn["turn"],
                "strict_closed_loop_score": artifact["quality"]["score"],
                "strict_closed_loop_accepted": True,
                "stitched_episode": True,
            }
        )
        rows.append(datum)
    return rows


def stitch(
    paths: list[Path],
    through_turns: list[int],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    if len(paths) != len(through_turns):
        raise ValueError("each episode path needs one --through-turn value")
    artifacts = [_load(path) for path in paths]
    identity = {(item.get("dataset"), item.get("case_id"), item.get("mode")) for item in artifacts}
    if len(identity) != 1:
        raise ValueError("all segments must have the same dataset, case_id, and mode")

    stitched_turns: list[dict[str, Any]] = []
    elapsed_seconds = 0.0
    rejection_count = 0
    for artifact, limit in zip(artifacts, through_turns, strict=True):
        episode = artifact["episode"]
        selected = [
            copy.deepcopy(turn)
            for turn in episode.get("turns") or []
            if int(turn.get("turn") or 0) <= limit
        ]
        if len(selected) != limit:
            raise ValueError(f"segment does not contain turns 1..{limit}")
        for turn in selected:
            if turn.get("status") not in ACCEPTED_STATUSES:
                raise ValueError(
                    f"refusing non-accepted turn {turn.get('turn')}: {turn.get('status')}"
                )
            turn["turn"] = len(stitched_turns) + 1
            datum = turn.get("sft_datum") or {}
            metadata = datum.get("metadata") or {}
            if metadata:
                metadata["turn_index"] = turn["turn"]
                metadata["stitched_episode"] = True
            stitched_turns.append(turn)
        elapsed_seconds += float(artifact.get("elapsed_seconds") or 0.0)
        rejection_count += int(episode.get("rejection_count") or 0)

    first = artifacts[0]
    final_episode = artifacts[-1]["episode"]
    episode = copy.deepcopy(first["episode"])
    episode.update(
        {
            "turns": stitched_turns,
            "turn_count": len(stitched_turns),
            "tool_sequence": [str(turn.get("generated_tool")) for turn in stitched_turns],
            "rejection_count": rejection_count,
            "terminal_success": bool(final_episode.get("terminal_success")),
            "final_terminal": bool(final_episode.get("final_terminal")),
            "final_stop_decision": copy.deepcopy(final_episode.get("final_stop_decision")),
            "trajectory_type": "accepted_prefix_stitched_complete_episode",
        }
    )
    _rebase_sft_histories(stitched_turns)
    seed_row = copy.deepcopy(stitched_turns[0].get("sft_datum") or {})
    if not seed_row:
        raise ValueError("first accepted turn has no SFT pre-action state for re-audit")
    quality = _audit_episode(str(first["mode"]), episode, seed_row, max_new_tokens)
    return {
        "dataset": first["dataset"],
        "case_id": first["case_id"],
        "mode": first["mode"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "resumed": False,
        "stitched_from": [
            {"episode": str(path), "through_turn": limit}
            for path, limit in zip(paths, through_turns, strict=True)
        ],
        "quality": quality,
        "episode": episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--through-turn", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    artifact = stitch(
        args.episode,
        args.through_turn,
        max_new_tokens=args.max_new_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exported_rows = 0
    if args.sft_output is not None:
        args.sft_output.parent.mkdir(parents=True, exist_ok=True)
        rows = _exportable_sft_rows(artifact)
        args.sft_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        exported_rows = len(rows)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "turns": artifact["episode"]["turn_count"],
                "terminal": artifact["episode"]["terminal_success"],
                "score": artifact["quality"]["score"],
                "accepted_for_high_quality_sft": artifact["quality"][
                    "accepted_for_high_quality_sft"
                ],
                "exported_sft_rows": exported_rows,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
