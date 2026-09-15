"""Prepare the externally canonical, schema-valid student SFT arm."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from pipeline import ROOT, write_json


SOURCE = ROOT / "outputs" / "production_sft_v7_external_contract"
OUTPUT = ROOT / "outputs" / "student_external_contract_v1"
ARM = "decision_sufficient_state_v4_external_tool_contract"
MODEL = (
    Path(os.environ.get("USERPROFILE", ""))
    / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def build() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    invalid_targets: list[str] = []
    prompt_tokens: list[int] = []
    for split in ("train", "validation", "test"):
        split_rows = _read(SOURCE / split / "sft_all.jsonl")
        split_counts[split] = len(split_rows)
        for row in split_rows:
            row["metadata"] = dict(row["metadata"])
            row["metadata"]["context_arm"] = ARM
            call = row["messages"][-1]["tool_calls"][0]["function"]
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            matching = [tool for tool in row["tools"] if tool["function"]["name"] == call["name"]]
            if len(matching) != 1 or list(
                Draft202012Validator(matching[0]["function"]["parameters"]).iter_errors(arguments)
            ):
                invalid_targets.append(str(row["metadata"]["row_id"]))
            prompt_tokens.append(
                len(
                    tokenizer.apply_chat_template(
                        row["messages"][:-1],
                        tools=row["tools"],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                    )["input_ids"]
                )
            )
        _write(OUTPUT / ARM / split / "sft_all.jsonl", split_rows)
        rows.extend(split_rows)

    report = {
        "experiment": "student_external_contract_v1",
        "source": str(SOURCE),
        "arm": ARM,
        "split_counts": split_counts,
        "rows": len(rows),
        "targets_with_internal_to_external_argument_normalization": sum(
            bool(row["metadata"].get("target_interface_normalized")) for row in rows
        ),
        "targets_valid_against_visible_schema": len(rows) - len(invalid_targets),
        "invalid_target_row_ids": invalid_targets,
        "rendered_prompt_tokens": {
            "p50": round(statistics.median(prompt_tokens), 2),
            "p90": sorted(prompt_tokens)[int(len(prompt_tokens) * 0.9) - 1],
            "max": max(prompt_tokens),
        },
        "status": "prepared_not_yet_trained",
    }
    if invalid_targets:
        raise ValueError(f"external-contract targets fail their visible schemas: {invalid_targets[:5]}")
    write_json(OUTPUT / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
