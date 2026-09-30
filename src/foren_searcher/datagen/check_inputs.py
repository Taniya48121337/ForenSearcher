"""学生输入检查：用学生模型（Qwen3.5-9B）的 processor 把 export 出的每一行过一遍 chat template，核对
label mask 边界、图片占位符展开、token 数，并把精确 token 数写回 summary。

用法：
    python -m foren_searcher.datagen.check_inputs --export experiments/sft_data/export/<dir> --model .models/Qwen3.5-9B

检查项（任一失败该行记 failed，不中断）：
1. 只有最后一条 assistant 进 loss：prefix（到 assistant 生成提示为止）token 与 full 的前缀逐 token 相同；
2. 图片占位符数量 == images 数量，且展开后的 prefix 仍与 full 一致（图片 token 在 prefix 内）；
3. 监督文本原样：解码 labels 段得到的文本以 assistant 内容开头，尾巴只剩 eos；
4. 长度：input_tokens ≤ --max-tokens（默认 32768），超过的行记 too_long。
不训练、不加载权重，只用 processor（CPU 即可）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def encode_row(row: dict[str, Any], export_dir: Path, processor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = [dict(m) for m in row["messages"]]
    paths = [export_dir / p for p in row["images"]]
    placeholders = sum(1 for m in messages for p in (m["content"] if isinstance(m["content"], list) else []) if p.get("type") == "image")
    if placeholders != len(paths):
        raise ValueError(f"图片占位符 {placeholders} 个，images {len(paths)} 个")
    prefix_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full_text.startswith(prefix_text):
        raise ValueError("chat template 改变了 assistant 前缀边界")
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
    full = processor(text=[full_text], images=images or None, return_tensors="pt", padding=False)
    prefix = processor(text=[prefix_text], images=images or None, return_tensors="pt", padding=False)
    n_prefix = int(prefix["input_ids"].shape[1])
    if not torch.equal(full["input_ids"][:, :n_prefix], prefix["input_ids"]):
        raise ValueError("图片展开或分词改变了前缀 token")
    target = processor.tokenizer.decode(full["input_ids"][0, n_prefix:], skip_special_tokens=False)
    expected = messages[-1]["content"]
    if not target.startswith(expected):
        raise ValueError("模板改动了监督文本")
    tail = target[len(expected):].strip()
    eos = processor.tokenizer.eos_token
    if tail not in (eos, f"{eos}\n", ""):
        raise ValueError(f"监督段尾巴异常: {tail!r}")
    labels = full["input_ids"].clone()
    labels[:, :n_prefix] = -100
    stats = {"input_tokens": int(full["input_ids"].shape[1]), "prefix_tokens": n_prefix,
             "supervised_tokens": int((labels != -100).sum()), "images": len(paths),
             "image_tokens": int((full["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())}
    return {"input_ids": full["input_ids"], "labels": labels}, stats


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {name: ordered[round((len(ordered) - 1) * q)] for name, q in (("min", 0), ("p50", .5), ("p90", .9), ("max", 1))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--model", default=".models/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--files", nargs="*", default=None, help="只检查这些 jsonl（默认 accepted 两份）")
    args = parser.parse_args(argv)
    export_dir = Path(args.export)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    files = args.files or ["hi_accepted.jsonl", "gia_accepted.jsonl"]
    totals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    too_long: list[dict[str, Any]] = []
    with (export_dir / "check_inputs.rows.jsonl").open("w", encoding="utf-8") as log:
        for name in files:
            path = export_dir / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                m = row["metadata"]
                identity = {"file": name, "role": m["role"], "unit": m["unit"], "workspace": m.get("workspace"), "turn": m.get("turn"), "request": m.get("request")}
                try:
                    batch, stats = encode_row(row, export_dir, processor)
                    del batch
                    record = {**identity, **stats, "status": "ok"}
                    totals[name].append(stats)
                    if stats["input_tokens"] > args.max_tokens:
                        too_long.append(record)
                except Exception as exc:  # noqa: BLE001
                    record = {**identity, "status": "failed", "error": str(exc)}
                    failures.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
    summary = {"model": str(Path(args.model).resolve()), "processor": type(processor).__name__, "max_tokens": args.max_tokens,
               "files": {name: {"rows": len(rows), "input_tokens": quantiles([r["input_tokens"] for r in rows]),
                                "supervised_tokens": quantiles([r["supervised_tokens"] for r in rows]),
                                "image_tokens": quantiles([r["image_tokens"] for r in rows]),
                                "above": {str(n): sum(r["input_tokens"] > n for r in rows) for n in (8192, 16384, 32768)}}
                         for name, rows in totals.items()},
               "failed": failures, "too_long": too_long}
    (export_dir / "check_inputs.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
