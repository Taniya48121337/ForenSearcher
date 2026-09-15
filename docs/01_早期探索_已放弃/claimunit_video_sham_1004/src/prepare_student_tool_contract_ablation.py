"""Create a target/message-identical ablation for native per-turn tool schemas."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


OLD_ROOT = ROOT / "outputs" / "production_sft_v5_role_separated"
NEW_ROOT = ROOT / "outputs" / "production_sft_v6_tool_contract"
OUTPUT = ROOT / "outputs" / "student_tool_contract_ablation_v1"
MODEL = (
    Path(os.environ.get("USERPROFILE", ""))
    / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
SPLITS = ("train", "validation", "test")
ARMS = (
    "decision_sufficient_state_v2_no_tool_schemas",
    "decision_sufficient_state_v3_native_tool_schemas",
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"][-1], ensure_ascii=False, sort_keys=True)


def _rendered_prompt_tokens(tokenizer: Any, row: dict[str, Any]) -> int:
    return len(
        tokenizer.apply_chat_template(
            row["messages"][:-1],
            tools=row.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
    )


def build() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    arm_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ARMS}
    split_counts: dict[str, int] = {}
    target_mismatches: list[str] = []
    message_mismatches: list[str] = []
    split_mismatches: list[str] = []

    for split in SPLITS:
        old_by_id = {row["metadata"]["row_id"]: row for row in _read(OLD_ROOT / split / "sft_all.jsonl")}
        new_by_id = {row["metadata"]["row_id"]: row for row in _read(NEW_ROOT / split / "sft_all.jsonl")}
        if set(old_by_id) != set(new_by_id):
            raise ValueError(f"row IDs differ in {split}")
        split_counts[split] = len(old_by_id)
        for row_id in old_by_id:
            pair = [old_by_id[row_id], new_by_id[row_id]]
            if len({_target(row) for row in pair}) != 1:
                target_mismatches.append(row_id)
            if pair[0]["messages"] != pair[1]["messages"]:
                message_mismatches.append(row_id)
            if len({row["metadata"]["data_split"] for row in pair}) != 1:
                split_mismatches.append(row_id)
            for name, row in zip(ARMS, pair):
                row["metadata"] = dict(row["metadata"])
                row["metadata"]["context_arm"] = name
                arm_rows[name].append(row)
        for name in ARMS:
            _write(
                OUTPUT / name / split / "sft_all.jsonl",
                [row for row in arm_rows[name] if row["metadata"]["data_split"] == split],
            )

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [_rendered_prompt_tokens(tokenizer, row) for row in rows]
        schema_counts = [len(row.get("tools") or []) for row in rows]
        return {
            "rows": len(rows),
            "rendered_prompt_tokens_total": sum(counts),
            "rendered_prompt_tokens_p50": round(statistics.median(counts), 2),
            "rendered_prompt_tokens_p90": sorted(counts)[int(len(counts) * 0.9) - 1],
            "rendered_prompt_tokens_max": max(counts),
            "rows_with_native_tool_schemas": sum(count > 0 for count in schema_counts),
            "eligible_schemas_per_row_min": min(schema_counts),
            "eligible_schemas_per_row_max": max(schema_counts),
        }

    report = {
        "experiment": "student_tool_contract_ablation_v1",
        "hypothesis": (
            "Providing the same eligible function schemas and native tool-call instruction seen at executable "
            "teacher inference converts grounded reasoning into parseable calls without changing messages, "
            "targets, row exposure, or splits."
        ),
        "source_roots": {"without_schemas": str(OLD_ROOT), "with_schemas": str(NEW_ROOT)},
        "split_counts": split_counts,
        "invariants": {
            "same_rows": sum(split_counts.values()),
            "target_mismatches": target_mismatches,
            "message_mismatches": message_mismatches,
            "split_mismatches": split_mismatches,
            "all_targets_identical": not target_mismatches,
            "all_messages_identical": not message_mismatches,
            "all_splits_identical": not split_mismatches,
        },
        "arms": {name: stats(rows) for name, rows in arm_rows.items()},
        "status": "prepared_not_yet_trained",
    }
    if target_mismatches or message_mismatches or split_mismatches:
        raise ValueError("tool-contract ablation invariants failed")
    if report["arms"][ARMS[0]]["rows_with_native_tool_schemas"] != 0:
        raise ValueError("baseline unexpectedly contains native tool schemas")
    if report["arms"][ARMS[1]]["rows_with_native_tool_schemas"] != sum(split_counts.values()):
        raise ValueError("schema arm is missing native tool schemas")
    write_json(OUTPUT / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
