"""Local QLoRA trainer for one context-ablation arm with assistant-only loss."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from pipeline import ROOT, write_json


DEFAULT_DATA_ROOT = ROOT / "outputs" / "student_context_ablation_v2"
DEFAULT_MODEL = (
    Path(os.environ.get("USERPROFILE", ""))
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-VL-8B-Instruct"
    / "snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
ARMS = (
    "full_state_normalized_v1",
    "naive_token_tail_v1",
    "decision_sufficient_state_v1",
    "decision_sufficient_state_v1_collision_baseline",
    "decision_sufficient_state_v2_role_separated",
    "decision_sufficient_state_v2_no_tool_schemas",
    "decision_sufficient_state_v3_native_tool_schemas",
    "decision_sufficient_state_v4_external_tool_contract",
    "decision_sufficient_state_v5_action_affordances",
    "decision_sufficient_state_v6_executable_affordances",
)


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def _balanced_deterministic_order(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Interleave datasets and rotate tool families within each dataset."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[str(row["metadata"]["dataset"])][str(row["metadata"]["target_tool"])].append(row)
    dataset_sequences: dict[str, list[dict[str, Any]]] = {}
    for dataset_offset, dataset in enumerate(sorted(groups)):
        tool_groups = groups[dataset]
        tools = sorted(tool_groups)
        random.Random(seed + dataset_offset).shuffle(tools)
        for tool_offset, tool in enumerate(tools):
            random.Random(seed + dataset_offset * 1000 + tool_offset).shuffle(tool_groups[tool])
        sequence: list[dict[str, Any]] = []
        depth = 0
        while True:
            progressed = False
            for tool in tools:
                values = tool_groups[tool]
                if depth < len(values):
                    sequence.append(values[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        dataset_sequences[dataset] = sequence
    ordered: list[dict[str, Any]] = []
    depth = 0
    while True:
        progressed = False
        for dataset in sorted(dataset_sequences):
            values = dataset_sequences[dataset]
            if depth < len(values):
                ordered.append(values[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return ordered


def encode_target_only(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_length: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        tools=tools,
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        tools=tools,
    )["input_ids"]
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template target is not a strict continuation of its prompt")
    target = full[len(prompt) :]
    if not target:
        raise ValueError("assistant target tokenization is empty")
    if len(target) >= max_length:
        raise ValueError(f"assistant target alone has {len(target)} tokens, exceeding max_length={max_length}")
    kept_prompt = prompt[-(max_length - len(target)) :]
    input_ids = kept_prompt + target
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(kept_prompt) + target,
        "prompt_tokens_source": len(prompt),
        "prompt_tokens_kept": len(kept_prompt),
        "target_tokens": len(target),
    }


class ArmDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items = [
            encode_target_only(tokenizer, row["messages"], max_length, row.get("tools"))
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.items[index]
        return {key: item[key] for key in ("input_ids", "attention_mask", "labels")}

    def stats(self) -> dict[str, Any]:
        return {
            "rows": len(self.items),
            "source_prompt_tokens": sum(item["prompt_tokens_source"] for item in self.items),
            "kept_prompt_tokens": sum(item["prompt_tokens_kept"] for item in self.items),
            "target_tokens": sum(item["target_tokens"] for item in self.items),
            "rows_prompt_truncated": sum(
                item["prompt_tokens_kept"] < item["prompt_tokens_source"] for item in self.items
            ),
            "max_sequence_tokens": max((len(item["input_ids"]) for item in self.items), default=0),
        }


class TargetOnlyCollator:
    def __init__(
        self,
        pad_token_id: int,
        multiple: int = 8,
        fixed_length: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.multiple = multiple
        self.fixed_length = fixed_length

    def __call__(self, items: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        observed_maximum = max(len(item["input_ids"]) for item in items)
        maximum = self.fixed_length or observed_maximum
        maximum = ((maximum + self.multiple - 1) // self.multiple) * self.multiple
        if observed_maximum > maximum:
            raise ValueError(
                f"batch sequence length {observed_maximum} exceeds fixed padding {maximum}"
            )
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in items:
            padding = maximum - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


class DeterministicSequentialTrainer(Trainer):
    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> SequentialSampler:
        return SequentialSampler(train_dataset or self.train_dataset)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm {args.arm!r}; choose one of {ARMS}")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = _read_jsonl(
        args.data_root / args.arm / "train" / "sft_all.jsonl",
        args.max_train_rows,
    )
    if args.focus_dataset:
        focus_datasets = set(args.focus_dataset)
        train_rows = [
            row
            for row in train_rows
            if str(row["metadata"]["dataset"]) in focus_datasets
        ]
        if not train_rows:
            raise ValueError(
                f"no training rows match --focus-dataset {sorted(focus_datasets)}"
            )
    if args.focus_tool:
        focus_tools = set(args.focus_tool)
        train_rows = [
            row for row in train_rows if str(row["metadata"]["target_tool"]) in focus_tools
        ]
        if not train_rows:
            raise ValueError(f"no training rows match --focus-tool {sorted(focus_tools)}")
    train_rows = _balanced_deterministic_order(train_rows, args.seed)
    train_data = ArmDataset(train_rows, tokenizer, args.max_length)
    exposure_count = min(
        len(train_rows), args.max_steps * args.gradient_accumulation_steps
    )
    exposed = train_rows[:exposure_count]
    report: dict[str, Any] = {
        "experiment": "student_context_ablation_v2",
        "arm": args.arm,
        "model": str(args.model),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataset": train_data.stats(),
        "assistant_target_only_loss": True,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "focus_datasets": list(args.focus_dataset or []),
        "focus_tools": list(args.focus_tool or []),
        "sampling": {
            "method": "seeded_dataset_interleave_within_dataset_tool_family_rotation_sequential",
            "seed": args.seed,
            "expected_exposed_rows": exposure_count,
            "first_exposed_row_ids": [row["metadata"]["row_id"] for row in exposed],
            "dataset_counts": dict(Counter(row["metadata"]["dataset"] for row in exposed)),
            "target_tool_counts": dict(Counter(row["metadata"]["target_tool"] for row in exposed)),
        },
        "dry_run": args.dry_run,
        "optimization": {
            "max_steps": args.max_steps,
            "max_train_rows": args.max_train_rows,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lr_scheduler_type": "cosine",
            "warmup_steps": max(1, round(args.max_steps * 0.03)),
        },
    }
    fixed_padding_length = None
    if args.fixed_pad_to_longest:
        fixed_padding_length = max(
            (len(item["input_ids"]) for item in train_data.items),
            default=0,
        )
        fixed_padding_length = (
            (fixed_padding_length + 7) // 8
        ) * 8
    report["batch_padding"] = {
        "mode": "fixed_dataset_longest" if fixed_padding_length else "per_batch_multiple_of_8",
        "fixed_length": fixed_padding_length,
        "padding_labels_ignored": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json(args.output / "run_report.json", report)
        return report

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    report["parameters"] = {
        "trainable": trainable,
        "total": total,
        "trainable_fraction": round(trainable / total, 8),
    }
    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(args.max_steps * 0.03)),
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        seed=args.seed,
    )
    trainer = DeterministicSequentialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=TargetOnlyCollator(
            tokenizer.pad_token_id,
            fixed_length=fixed_padding_length,
        ),
    )
    result = trainer.train()
    report["train_metrics"] = result.metrics
    report["peak_cuda_memory_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    if args.save_adapter:
        model.save_pretrained(args.output / "adapter")
        tokenizer.save_pretrained(args.output / "adapter")
    write_json(args.output / "run_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--focus-dataset", action="append")
    parser.add_argument("--focus-tool", action="append")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument(
        "--fixed-pad-to-longest",
        action="store_true",
        help="Pad every batch to this filtered dataset's longest row for stable CUDA shapes.",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
