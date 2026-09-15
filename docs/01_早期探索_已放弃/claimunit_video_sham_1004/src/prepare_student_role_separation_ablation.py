"""Create a target-identical v1-vs-v2 role-separation SFT ablation."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


OLD_ROOT = ROOT / "outputs" / "production_sft_v4_compact"
NEW_ROOT = ROOT / "outputs" / "production_sft_v5_role_separated"
OUTPUT = ROOT / "outputs" / "student_role_separation_ablation_v1"
MODEL = (
    Path(os.environ.get("USERPROFILE", ""))
    / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
SPLITS = ("train", "validation", "test")
RESERVED = {"tool_name", "arguments", "tool_calls", "function", "call_id"}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_nested_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(child) for child in value)) if value else set()
    return set()


def _target(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"][-1], ensure_ascii=False, sort_keys=True)


def build() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    arms = {
        "decision_sufficient_state_v1_collision_baseline": [],
        "decision_sufficient_state_v2_role_separated": [],
    }
    split_counts: dict[str, int] = {}
    target_mismatches: list[str] = []
    split_mismatches: list[str] = []

    for split in SPLITS:
        old = _read(OLD_ROOT / split / "sft_all.jsonl")
        new = _read(NEW_ROOT / split / "sft_all.jsonl")
        old_by_id = {row["metadata"]["row_id"]: row for row in old}
        new_by_id = {row["metadata"]["row_id"]: row for row in new}
        if set(old_by_id) != set(new_by_id):
            raise ValueError(f"row IDs differ in {split}")
        split_counts[split] = len(old)
        for row_id in old_by_id:
            pair = [old_by_id[row_id], new_by_id[row_id]]
            if len({_target(row) for row in pair}) != 1:
                target_mismatches.append(row_id)
            if len({row["metadata"]["data_split"] for row in pair}) != 1:
                split_mismatches.append(row_id)
            for name, row in zip(arms, pair):
                row["metadata"] = dict(row["metadata"])
                row["metadata"]["context_arm"] = name
                arms[name].append(row)
        for name in arms:
            split_rows = [row for row in arms[name] if row["metadata"]["data_split"] == split]
            _write(OUTPUT / name / split / "sft_all.jsonl", split_rows)

    def stats(rows: list[dict[str, Any]], history_field: str) -> dict[str, Any]:
        counts: list[int] = []
        collision_rows = 0
        collision_counts: dict[str, int] = {}
        for row in rows:
            text = json.dumps(row["messages"][:-1], ensure_ascii=False, separators=(",", ":"))
            counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
            payload = json.loads(row["messages"][1]["content"])
            collisions = _nested_keys(payload.get(history_field, [])).intersection(RESERVED)
            if collisions:
                collision_rows += 1
            for key in collisions:
                collision_counts[key] = collision_counts.get(key, 0) + 1
        return {
            "rows": len(rows),
            "input_tokens_total": sum(counts),
            "input_tokens_p50": round(statistics.median(counts), 2),
            "input_tokens_p90": sorted(counts)[int(len(counts) * 0.9) - 1],
            "input_tokens_max": max(counts),
            "history_target_envelope_collision_rows": collision_rows,
            "history_target_envelope_collision_counts": collision_counts,
        }

    report = {
        "experiment": "student_role_separation_ablation_v1",
        "hypothesis": (
            "Renaming historical action fields away from the assistant target envelope reduces plain-JSON "
            "history imitation and improves native single-tool-call generation without changing targets or state."
        ),
        "source_roots": {"old": str(OLD_ROOT), "new": str(NEW_ROOT)},
        "split_counts": split_counts,
        "invariants": {
            "same_rows": sum(split_counts.values()),
            "target_mismatches": target_mismatches,
            "split_mismatches": split_mismatches,
            "all_targets_identical": not target_mismatches,
            "all_splits_identical": not split_mismatches,
        },
        "arms": {
            "decision_sufficient_state_v1_collision_baseline": stats(
                arms["decision_sufficient_state_v1_collision_baseline"], "action_history"
            ),
            "decision_sufficient_state_v2_role_separated": stats(
                arms["decision_sufficient_state_v2_role_separated"], "past_execution_history"
            ),
        },
        "status": "prepared_not_yet_trained",
    }
    if target_mismatches or split_mismatches:
        raise ValueError("role-separation ablation invariants failed")
    if report["arms"]["decision_sufficient_state_v2_role_separated"][
        "history_target_envelope_collision_rows"
    ]:
        raise ValueError("role-separated arm still collides with target envelope")
    write_json(OUTPUT / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
