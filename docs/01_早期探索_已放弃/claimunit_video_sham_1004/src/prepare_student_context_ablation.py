"""Build target-identical context arms for a controlled student SFT ablation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from sft_context_compaction import PATH_OR_LOCATOR_PATTERNS, _normalize_private_locators


DEFAULT_FULL_ROOT = ROOT / "outputs" / "production_sft_v3_scale"
DEFAULT_COMPACT_ROOT = ROOT / "outputs" / "production_sft_v4_compact"
DEFAULT_OUTPUT = ROOT / "outputs" / "student_context_ablation_v2"
DEFAULT_MODEL = (
    Path(os.environ.get("USERPROFILE", ""))
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-VL-8B-Instruct"
    / "snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target(row: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(row["messages"][-1])


def _token_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return len(tokenizer.encode(text, add_special_tokens=False))


def _privacy_hits(messages: list[dict[str, Any]]) -> dict[str, int]:
    text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return {
        name: len(pattern.findall(text))
        for name, pattern in PATH_OR_LOCATOR_PATTERNS.items()
        if pattern.search(text)
    }


def _normalized_full(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result["messages"] = _normalize_private_locators(result["messages"], row)
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"].update({
        "context_arm": "full_state_normalized_v1",
        "private_locator_normalized": True,
    })
    if _target(result) != _target(row):
        raise ValueError(f"locator normalization changed assistant target for {row['metadata']['row_id']}")
    return result


def _trim_message_suffix_to_budget(
    tokenizer: Any,
    prefix: list[dict[str, Any]],
    boundary: dict[str, Any],
    suffix: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any] | None:
    content = boundary.get("content")
    if not isinstance(content, str) or not content:
        return None
    low, high = 0, len(content)
    best: dict[str, Any] | None = None
    while low <= high:
        keep = (low + high) // 2
        candidate = copy.deepcopy(boundary)
        candidate["content"] = "[left-truncated]\n" + (content[-keep:] if keep else "")
        if _token_count(tokenizer, prefix + [candidate] + suffix) <= budget:
            best = candidate
            low = keep + 1
        else:
            high = keep - 1
    return best


def _recent_window(
    normalized_full: dict[str, Any],
    compact: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    prior = copy.deepcopy(normalized_full["messages"][:-1])
    target = _target(normalized_full)
    budget = _token_count(tokenizer, compact["messages"][:-1])
    system = [prior[0]] if prior and prior[0].get("role") == "system" else []
    remaining = prior[1:] if system else prior
    selected: list[dict[str, Any]] = []
    boundary_trimmed = False
    for message in reversed(remaining):
        candidate = system + [message] + selected
        if _token_count(tokenizer, candidate) <= budget:
            selected.insert(0, message)
            continue
        trimmed = _trim_message_suffix_to_budget(tokenizer, system, message, selected, budget)
        if trimmed is not None:
            selected.insert(0, trimmed)
            boundary_trimmed = True
        break
    input_messages = system + selected
    result = copy.deepcopy(normalized_full)
    result["messages"] = input_messages + [target]
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"].update({
        "context_arm": "naive_recent_window_v1",
        "input_token_budget_from": "decision_sufficient_state_v1",
        "input_token_budget": budget,
        "input_tokens": _token_count(tokenizer, input_messages),
        "boundary_message_left_truncated": boundary_trimmed,
    })
    return result


def _naive_tail_window(
    normalized_full: dict[str, Any],
    compact: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Keep a raw tokenizer tail under the same per-row budget as compact context."""
    prior = copy.deepcopy(normalized_full["messages"][:-1])
    target = _target(normalized_full)
    budget = _token_count(tokenizer, compact["messages"][:-1])
    system = [prior[0]] if prior and prior[0].get("role") == "system" else []
    history = prior[1:] if system else prior
    history_text = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    history_ids = tokenizer.encode(history_text, add_special_tokens=False)
    low, high = 0, len(history_ids)
    best = system
    kept = 0
    while low <= high:
        count = (low + high) // 2
        suffix = tokenizer.decode(history_ids[-count:]) if count else ""
        candidate = system + [{
            "role": "user",
            "content": "[naive left-truncated transcript tail]\n" + suffix,
        }]
        if _token_count(tokenizer, candidate) <= budget:
            best = candidate
            kept = count
            low = count + 1
        else:
            high = count - 1
    result = copy.deepcopy(normalized_full)
    result["messages"] = best + [target]
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"].update({
        "context_arm": "naive_token_tail_v1",
        "input_token_budget_from": "decision_sufficient_state_v1",
        "input_token_budget": budget,
        "input_tokens": _token_count(tokenizer, best),
        "history_tokens_retained": kept,
        "history_tokens_dropped": len(history_ids) - kept,
    })
    return result


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[max(0, int(len(values) * fraction) - 1)] if values else 0


def _arm_stats(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    counts = [_token_count(tokenizer, row["messages"][:-1]) for row in rows]
    hits: dict[str, int] = {}
    for row in rows:
        for name, count in _privacy_hits(row["messages"]).items():
            hits[name] = hits.get(name, 0) + count
    return {
        "rows": len(rows),
        "input_tokens_total": sum(counts),
        "input_tokens_mean": round(statistics.mean(counts), 2) if counts else 0.0,
        "input_tokens_p50": round(statistics.median(counts), 2) if counts else 0.0,
        "input_tokens_p90": _percentile(counts, 0.90),
        "input_tokens_max": max(counts, default=0),
        "rows_over_8192_input_tokens": sum(count > 8192 for count in counts),
        "private_locator_or_oracle_hits": hits,
    }


def build(
    full_root: Path = DEFAULT_FULL_ROOT,
    compact_root: Path = DEFAULT_COMPACT_ROOT,
    output_root: Path = DEFAULT_OUTPUT,
    model_path: Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    all_arms: dict[str, list[dict[str, Any]]] = {
        "full_state_normalized_v1": [],
        "naive_token_tail_v1": [],
        "decision_sufficient_state_v1": [],
    }
    split_counts: dict[str, int] = {}
    target_mismatches: list[str] = []
    split_mismatches: list[str] = []

    for split in SPLITS:
        full_rows = _read_jsonl(Path(full_root) / split / "sft_all.jsonl")
        compact_rows = _read_jsonl(Path(compact_root) / split / "sft_all.jsonl")
        compact_by_id = {row["metadata"]["row_id"]: row for row in compact_rows}
        split_counts[split] = len(full_rows)
        split_arms = {name: [] for name in all_arms}
        if set(compact_by_id) != {row["metadata"]["row_id"] for row in full_rows}:
            raise ValueError(f"row-id mismatch in {split}")
        for source in full_rows:
            row_id = source["metadata"]["row_id"]
            compact = copy.deepcopy(compact_by_id[row_id])
            full = _normalized_full(source)
            recent = _naive_tail_window(full, compact, tokenizer)
            compact["metadata"] = dict(compact.get("metadata", {}))
            compact["metadata"]["context_arm"] = "decision_sufficient_state_v1"
            trio = (full, recent, compact)
            if len({json.dumps(_target(row), sort_keys=True) for row in trio}) != 1:
                target_mismatches.append(row_id)
            if len({row["metadata"].get("data_split") for row in trio}) != 1:
                split_mismatches.append(row_id)
            for name, row in zip(all_arms, trio):
                split_arms[name].append(row)
                all_arms[name].append(row)
        for name, rows in split_arms.items():
            _write_jsonl(Path(output_root) / name / split / "sft_all.jsonl", rows)

    compact_tokens = [
        _token_count(tokenizer, row["messages"][:-1])
        for row in all_arms["decision_sufficient_state_v1"]
    ]
    recent_tokens = [
        _token_count(tokenizer, row["messages"][:-1])
        for row in all_arms["naive_token_tail_v1"]
    ]
    naive_rows = all_arms["naive_token_tail_v1"]
    truncated_indices = [
        index
        for index, row in enumerate(naive_rows)
        if int(row["metadata"].get("history_tokens_dropped", 0)) > 0
    ]
    manifest = {
        "experiment": "student_context_ablation_v2",
        "purpose": "isolate context representation while holding rows, splits, and assistant targets fixed",
        "model_tokenizer": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "source_roots": {"full": str(full_root), "compact": str(compact_root)},
        "split_counts": split_counts,
        "invariants": {
            "row_count": sum(split_counts.values()),
            "target_mismatches": target_mismatches,
            "split_mismatches": split_mismatches,
            "all_targets_identical": not target_mismatches,
            "all_splits_identical": not split_mismatches,
        },
        "arms": {name: _arm_stats(rows, tokenizer) for name, rows in all_arms.items()},
        "naive_tail_budget": {
            "source": "per-row decision-sufficient input-token count",
            "mean_utilization": round(
                statistics.mean(r / max(c, 1) for r, c in zip(recent_tokens, compact_tokens)), 6
            ),
            "rows_history_truncated": len(truncated_indices),
            "mean_utilization_when_truncated": round(
                statistics.mean(
                    recent_tokens[index] / max(compact_tokens[index], 1)
                    for index in truncated_indices
                ),
                6,
            ) if truncated_indices else None,
            "max_budget_excess_tokens": max(
                (r - c for r, c in zip(recent_tokens, compact_tokens)), default=0
            ),
        },
        "training_contract": {
            "same_base_model": True,
            "same_case_split": True,
            "same_rows_and_targets": True,
            "same_optimizer_and_update_count": True,
            "assistant_target_only_loss": True,
            "recommended_max_sequence_length": 8192,
            "report": [
                "held-out target NLL",
                "tool-name exact match",
                "argument schema validity",
                "closed-loop task success",
                "evidence grounding",
                "repair success",
                "stuck rate",
                "inference tokens per solved case",
            ],
        },
    }
    if not manifest["invariants"]["all_targets_identical"] or not manifest["invariants"]["all_splits_identical"]:
        raise ValueError("ablation invariant failed")
    if manifest["naive_tail_budget"]["max_budget_excess_tokens"] > 0:
        raise ValueError("naive-tail arm exceeded compact token budget")
    if any(item["private_locator_or_oracle_hits"] for item in manifest["arms"].values()):
        raise ValueError("privacy pattern found in ablation arm")
    write_json(Path(output_root) / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--compact-root", type=Path, default=DEFAULT_COMPACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    report = build(args.full_root, args.compact_root, args.output, args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
