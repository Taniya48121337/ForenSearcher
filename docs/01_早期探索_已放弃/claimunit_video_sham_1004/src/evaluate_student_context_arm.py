"""Held-out teacher-forced and greedy action evaluation for one student arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jsonschema import Draft202012Validator
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from pipeline import ROOT, write_json
from train_student_context_arm import ARMS, DEFAULT_DATA_ROOT, DEFAULT_MODEL, encode_target_only


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _generated_sections(text: str) -> tuple[str, str]:
    """Separate policy reasoning from serialized native-tool payload."""
    match = TOOL_CALL_PATTERN.search(text)
    if match is None:
        return text.split("<tool_call>", 1)[0].strip(), ""
    return text[: match.start()].strip(), match.group(0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stratified_subset(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if not limit or limit >= len(rows):
        return rows
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        metadata = row["metadata"]
        groups[str(metadata["dataset"])][str(metadata["target_tool"])].append(row)
    for tool_groups in groups.values():
        for values in tool_groups.values():
            values.sort(key=lambda row: str(row["metadata"]["row_id"]))
    datasets = sorted(groups)
    selected: list[dict[str, Any]] = []
    base, remainder = divmod(limit, len(datasets))
    for dataset_index, dataset in enumerate(datasets):
        quota = base + (1 if dataset_index < remainder else 0)
        tool_groups = groups[dataset]
        tools = sorted(tool_groups)
        depth = 0
        chosen = 0
        while chosen < quota:
            progressed = False
            for tool in tools:
                if depth < len(tool_groups[tool]):
                    selected.append(tool_groups[tool][depth])
                    chosen += 1
                    progressed = True
                    if chosen == quota:
                        break
            if not progressed:
                break
            depth += 1
    return selected


def _spread_subset_by_dataset(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if limit >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["metadata"]["dataset"])].append(row)
    datasets = sorted(groups)
    base, remainder = divmod(limit, len(datasets))
    selected: list[dict[str, Any]] = []
    used_tools: set[str] = set()
    for index, dataset in enumerate(datasets):
        values = sorted(groups[dataset], key=lambda row: str(row["metadata"]["row_id"]))
        quota = min(len(values), base + (1 if index < remainder else 0))
        local_tools: set[str] = set()
        remaining = list(values)
        for _ in range(quota):
            chosen = min(
                remaining,
                key=lambda row: (
                    str(row["metadata"]["target_tool"]) in used_tools,
                    str(row["metadata"]["target_tool"]) in local_tools,
                    str(row["metadata"]["row_id"]),
                ),
            )
            remaining.remove(chosen)
            selected.append(chosen)
            tool_name = str(chosen["metadata"]["target_tool"])
            used_tools.add(tool_name)
            local_tools.add(tool_name)
    return selected


def _reference_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    call = row["messages"][-1]["tool_calls"][0]["function"]
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return str(call["name"]), arguments


def _parse_generated_call(text: str) -> tuple[str | None, dict[str, Any] | None]:
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None, None
    try:
        payload = json.loads(match.group(1))
        name = payload.get("name")
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return (str(name) if name else None), (arguments if isinstance(arguments, dict) else None)
    except (json.JSONDecodeError, TypeError):
        return None, None


def _same_argument_shape(generated: dict[str, Any] | None, reference: dict[str, Any]) -> bool:
    if generated is None or set(generated) != set(reference):
        return False
    return all(type(generated[key]) is type(reference[key]) for key in reference)


def _arguments_semantically_equal(
    tool_name: str, generated: dict[str, Any] | None, reference: dict[str, Any]
) -> bool:
    if generated is None:
        return False
    if tool_name == "create_candidate_claim":
        # A new claim_id is an opaque, model-chosen handle. Requiring the
        # spelling of an ID absent from the pre-action state scores an
        # impossible guess, not the meaning of the action. Later turns must
        # refer to whichever handle the executed action actually created.
        return generated.get("field_values") == reference.get("field_values") and bool(
            generated.get("claim_id")
        )
    if tool_name == "bind_evidence":
        def binding_map(value: dict[str, Any]) -> dict[str, tuple[str, ...]] | None:
            items = value.get("binding_items")
            if not isinstance(items, list):
                return None
            result: dict[str, tuple[str, ...]] = {}
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("support_ids"), list):
                    return None
                field = str(item.get("field"))
                if field in result:
                    return None
                result[field] = tuple(sorted(str(name) for name in item["support_ids"]))
            return result

        return generated.get("claim_id") == reference.get("claim_id") and binding_map(generated) == binding_map(reference)
    if tool_name == "set_field_decision":
        generated_copy = dict(generated)
        reference_copy = dict(reference)
        if isinstance(generated_copy.get("fields"), list):
            generated_copy["fields"] = sorted(generated_copy["fields"])
        if isinstance(reference_copy.get("fields"), list):
            reference_copy["fields"] = sorted(reference_copy["fields"])
        return generated_copy == reference_copy
    return generated == reference


def _arguments_validate_visible_schema(
    row: dict[str, Any], generated_name: str | None, generated_arguments: dict[str, Any] | None
) -> bool:
    if generated_name is None or generated_arguments is None:
        return False
    for tool in row.get("tools") or []:
        function = tool.get("function") or {}
        if function.get("name") == generated_name:
            return not list(Draft202012Validator(function.get("parameters") or {}).iter_errors(generated_arguments))
    return False


def _eligible_tools(row: dict[str, Any]) -> set[str]:
    for message in reversed(row["messages"][:-1]):
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        state = payload.get("current_harness_state") or payload.get("harness_state")
        if isinstance(state, dict):
            return set(str(name) for name in state.get("eligible_tools", []))
    return set()


def _load_model(
    model_path: Path, adapter: Path | None, *, load_in_4bit: bool = True
) -> Any:
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.data_root / args.arm / args.split / "sft_all.jsonl")
    rows = _stratified_subset(rows, args.max_rows)
    requested_generation_ids = set(args.generation_row_id or [])
    excluded_generation_ids = set(getattr(args, "exclude_generation_row_id", None) or [])
    available_ids = {str(row["metadata"]["row_id"]) for row in rows}
    missing_requested = sorted(requested_generation_ids - available_ids)
    if missing_requested:
        raise ValueError(f"requested generation rows are outside the selected evaluation subset: {missing_requested}")
    missing_excluded = sorted(excluded_generation_ids - available_ids)
    if missing_excluded:
        raise ValueError(f"excluded generation rows are outside the selected evaluation subset: {missing_excluded}")
    if requested_generation_ids.intersection(excluded_generation_ids):
        raise ValueError("the same row cannot be both requested and excluded from generation")
    generation_candidates = [
        row
        for row in rows
        if str(row["metadata"]["row_id"]) not in excluded_generation_ids
    ]
    generation_ids = requested_generation_ids or (
        {
            str(row["metadata"]["row_id"])
            for row in _spread_subset_by_dataset(
                generation_candidates,
                min(args.generation_rows, len(generation_candidates)),
            )
        }
        if args.generation_rows > 0
        else set()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.adapter)
    device = next(model.parameters()).device
    per_row: list[dict[str, Any]] = []
    nll_sum = 0.0
    target_tokens_sum = 0

    with torch.inference_mode():
        for index, row in enumerate(rows):
            encoded = encode_target_only(tokenizer, row["messages"], args.max_length, row.get("tools"))
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            attention = torch.ones_like(input_ids)
            labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
            row_target_tokens = int(encoded["target_tokens"])
            row_nll: float | None = None
            if not getattr(args, "skip_target_nll", False):
                output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
                shift_logits = output.logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                valid = shift_labels.ne(-100)
                token_losses = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(shift_labels)
                row_nll_sum = float(token_losses[valid].sum().item())
                row_target_tokens = int(valid.sum().item())
                row_nll = row_nll_sum / max(row_target_tokens, 1)
                nll_sum += row_nll_sum
                target_tokens_sum += row_target_tokens

            generated_text = ""
            generated_name = None
            generated_arguments = None
            generated_tokens = 0
            should_generate = str(row["metadata"]["row_id"]) in generation_ids
            if should_generate:
                prompt_length = len(encoded["input_ids"]) - encoded["target_tokens"]
                prompt_ids = input_ids[:, :prompt_length]
                generated = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                generated_text = tokenizer.decode(
                    generated[0, prompt_length:], skip_special_tokens=False
                )
                generated_tokens = int(generated.shape[1] - prompt_length)
                generated_name, generated_arguments = _parse_generated_call(generated_text)

            reference_name, reference_arguments = _reference_call(row)
            eligible = _eligible_tools(row)
            parseable = generated_name is not None and generated_arguments is not None
            reasoning_text, tool_payload_text = _generated_sections(generated_text)
            reasoning_tokens = len(
                tokenizer.encode(reasoning_text, add_special_tokens=False)
            ) if should_generate else 0
            tool_payload_tokens = len(
                tokenizer.encode(tool_payload_text, add_special_tokens=False)
            ) if should_generate and tool_payload_text else 0
            per_row.append({
                "row_id": row["metadata"]["row_id"],
                "dataset": row["metadata"]["dataset"],
                "trajectory_type": row["metadata"]["trajectory_type"],
                "turn_index": row["metadata"]["turn_index"],
                "reference_tool": reference_name,
                "prompt_tokens_source": encoded["prompt_tokens_source"],
                "prompt_tokens_kept": encoded["prompt_tokens_kept"],
                "target_tokens": row_target_tokens,
                "target_nll": round(row_nll, 6) if row_nll is not None else None,
                "generated": should_generate,
                "generated_tool": generated_name,
                "tool_call_parseable": parseable,
                "tool_name_exact": generated_name == reference_name,
                "argument_shape_exact": _same_argument_shape(generated_arguments, reference_arguments),
                "arguments_validate_visible_schema": _arguments_validate_visible_schema(
                    row, generated_name, generated_arguments
                ),
                "arguments_exact": generated_arguments == reference_arguments,
                "arguments_semantically_equal": _arguments_semantically_equal(
                    reference_name, generated_arguments, reference_arguments
                ),
                "generated_tool_prestate_eligible": bool(generated_name and generated_name in eligible),
                "generated_tokens": generated_tokens,
                "reasoning_tokens": reasoning_tokens,
                "tool_payload_tokens": tool_payload_tokens,
                "reasoning_within_budget": bool(
                    should_generate and reasoning_tokens <= args.max_reasoning_tokens
                ),
                "generation_hit_token_limit": should_generate and generated_tokens >= args.max_new_tokens,
                "generated_text": generated_text,
            })

    generated_rows = [row for row in per_row if row["generated"]]
    mean_nll = (
        nll_sum / max(target_tokens_sum, 1)
        if not getattr(args, "skip_target_nll", False)
        else None
    )
    summary = {
        "rows": len(per_row),
        "generation_rows": len(generated_rows),
        "target_tokens": target_tokens_sum,
        "target_nll": round(mean_nll, 6) if mean_nll is not None else None,
        "target_perplexity": round(math.exp(min(mean_nll, 20.0)), 6) if mean_nll is not None else None,
        "rows_prompt_truncated": sum(
            row["prompt_tokens_kept"] < row["prompt_tokens_source"] for row in per_row
        ),
        "tool_call_parse_rate": round(
            sum(row["tool_call_parseable"] for row in generated_rows) / max(len(generated_rows), 1), 4
        ),
        "tool_name_exact_rate": round(
            sum(row["tool_name_exact"] for row in generated_rows) / max(len(generated_rows), 1), 4
        ),
        "argument_shape_exact_rate": round(
            sum(row["argument_shape_exact"] for row in generated_rows) / max(len(generated_rows), 1), 4
        ),
        "argument_schema_valid_rate": round(
            sum(row["arguments_validate_visible_schema"] for row in generated_rows)
            / max(len(generated_rows), 1),
            4,
        ),
        "arguments_exact_rate": round(
            sum(row["arguments_exact"] for row in generated_rows) / max(len(generated_rows), 1), 4
        ),
        "argument_semantic_equivalence_rate": round(
            sum(row["arguments_semantically_equal"] for row in generated_rows)
            / max(len(generated_rows), 1),
            4,
        ),
        "prestate_eligible_rate": round(
            sum(row["generated_tool_prestate_eligible"] for row in generated_rows)
            / max(len(generated_rows), 1),
            4,
        ),
        "reasoning_within_budget_rate": round(
            sum(row["reasoning_within_budget"] for row in generated_rows)
            / max(len(generated_rows), 1),
            4,
        ),
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "generation_token_limit_hit_rate": round(
            sum(row["generation_hit_token_limit"] for row in generated_rows) / max(len(generated_rows), 1),
            4,
        ),
        "dataset_counts": dict(Counter(str(row["dataset"]) for row in per_row)),
        "reference_tool_counts": dict(Counter(str(row["reference_tool"]) for row in per_row)),
    }
    report = {
        "experiment": "student_context_ablation_v2",
        "arm": args.arm,
        "split": args.split,
        "adapter": str(args.adapter) if args.adapter else None,
        "base_model": str(args.model),
        "max_length": args.max_length,
        "selection": "deterministic_round_robin_by_dataset_and_target_tool",
        "selected_row_ids": [str(row["metadata"]["row_id"]) for row in rows],
        "generation_row_ids": sorted(generation_ids),
        "excluded_generation_row_ids": sorted(excluded_generation_ids),
        "summary": summary,
        "rows": per_row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=18)
    parser.add_argument("--generation-rows", type=int, default=18)
    parser.add_argument("--generation-row-id", action="append")
    parser.add_argument("--exclude-generation-row-id", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--max-reasoning-tokens",
        type=int,
        default=128,
        help="Post-hoc budget for natural-language reasoning; native tool JSON is measured separately.",
    )
    parser.add_argument("--skip-target-nll", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
