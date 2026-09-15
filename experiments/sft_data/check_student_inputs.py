"""Process actual candidate rows with a local multimodal model's chat template."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent


def prepare_row(row, corpus, processor):
    paths = [(corpus / path).resolve() for path in row["images"]]
    messages = [dict(message) for message in row["messages"]]
    marker = "<image>\n" * len(paths)
    if not messages[1]["content"].startswith(marker):
        raise ValueError("Image markers are not in their declared input order")
    user_text = messages[1]["content"][len(marker):]
    messages[1]["content"] = [{"type": "image"} for _ in paths] + [{"type": "text", "text": user_text}]
    prefix = processor.apply_chat_template(messages[:-1], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prefix):
        raise ValueError("The model template changed the assistant-prefix boundary")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    inputs = processor(text=full, images=images or None, return_tensors="pt", padding=False)
    prefix_inputs = processor(text=prefix, images=images or None, return_tensors="pt", padding=False)
    prefix_length = prefix_inputs["input_ids"].shape[1]
    if not torch.equal(inputs["input_ids"][:, :prefix_length], prefix_inputs["input_ids"]):
        raise ValueError("Image expansion or tokenization changed the prefix tokens")
    target = processor.tokenizer.decode(inputs["input_ids"][0, prefix_length:], skip_special_tokens=False)
    expected = row["messages"][-1]["content"]
    if not target.startswith(expected):
        raise ValueError("The chat template altered the supervised assistant text")
    suffix = target[len(expected):]
    if suffix.strip() != processor.tokenizer.eos_token:
        raise ValueError(f"Unexpected trailing supervision: {suffix!r}")
    labels = inputs["input_ids"].clone()
    labels[:, :prefix_length] = -100
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    if not (labels != -100).any():
        raise ValueError("No assistant target tokens remain")
    inputs["labels"] = labels
    return inputs, {"input_tokens": int(inputs["input_ids"].shape[1]),
                    "prefix_tokens": prefix_length,
                    "supervised_tokens": int((labels != -100).sum()),
                    "images": len(paths), "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)).tolist(),
                    "target_text_exact": True, "source_tokens_masked": True}


def quantiles(values):
    ordered = sorted(values)
    return {name: ordered[round((len(ordered)-1)*q)] for name,q in
            [("min",0),("p50",.5),("p90",.9),("p99",.99),("max",1)]} if ordered else {}


def main(args):
    corpus = ROOT / "corpus"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    totals, failures, selected_rows = defaultdict(list), [], 0
    files = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))["files"]
    with (output / "rows.jsonl").open("w", encoding="utf-8") as log:
        for item in files:
            for line in (corpus / item["file"]).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                metadata = row["metadata"]
                if args.run and metadata.get("run", metadata.get("construction")) != args.run:
                    continue
                selected_rows += 1
                identity = {"file": item["file"], "role": metadata["role"], "case_family": metadata["case_family"],
                            "run": metadata.get("run", metadata.get("construction")),
                            "request": metadata.get("request", metadata.get("state"))}
                try:
                    batch, stats = prepare_row(row, corpus, processor)
                    del batch
                    record = {**identity, **stats, "status": "processed"}
                    totals[item["file"]].append(stats)
                except Exception as error:
                    record = {**identity, "status": "failed", "error": str(error)}
                    failures.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
            print(json.dumps({"file": item["file"], "processed": len(totals[item["file"]]),
                              "failures_so_far": len(failures)}), flush=True)
    if selected_rows == 0:
        raise ValueError("No corpus rows match the requested selection")
    summary = {"model": str(Path(args.model).resolve()), "processor": type(processor).__name__,
               "selection": {"run": args.run, "selected_rows": selected_rows},
               "weight_training": "not_run", "truncation": False, "failed_rows": failures,
               "files": [{"file": name, "processed": len(rows),
                          "input_tokens": quantiles([r["input_tokens"] for r in rows]),
                          "supervised_tokens": quantiles([r["supervised_tokens"] for r in rows]),
                          "above_context_tokens": {str(length): sum(r["input_tokens"] > length for r in rows)
                                                   for length in [16384,32768,65536,131072,262144]}}
                         for name,rows in totals.items()]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT.parents[1] / ".models/Qwen3.5-9B"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", help="Only process retained rows from this run or construction")
    main(parser.parse_args())
