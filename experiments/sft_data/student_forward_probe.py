"""Run complete real text/visual inputs through a local student without changing weights."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from check_student_inputs import ROOT, prepare_row


def main(args):
    corpus = ROOT / "corpus"
    folder = ROOT / args.output
    folder.mkdir(parents=True, exist_ok=False)
    profile = [json.loads(line) for line in (ROOT / args.profile / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = []
    for role in ("HI", "GIA"):
        for visual in (False, True):
            candidates = [r for r in profile if r["file"].endswith("_development.jsonl")
                          and r["role"] == role and bool(r.get("images")) == visual and r["status"] == "processed"]
            if candidates:
                selected.append(min(candidates, key=lambda r:r["input_tokens"]))
    (folder / "selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, device_map={"":"cuda:0"}, attn_implementation="sdpa")
    model.eval()
    results = []
    for selection in selected:
        rows = [json.loads(line) for line in (corpus / selection["file"]).read_text(encoding="utf-8").splitlines()]
        row = next(row for row in rows if
                   row["metadata"].get("run",row["metadata"].get("construction")) == selection["run"] and
                   row["metadata"].get("request",row["metadata"].get("state")) == selection["request"])
        batch, stats = prepare_row(row, corpus, processor)
        labels = batch.pop("labels").to("cuda:0")
        batch = {key: value.to("cuda:0", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                 for key,value in batch.items()}
        target_positions = torch.where(labels[0] != -100)[0]
        if (target_positions < 1).any():
            raise ValueError("A target token has no causal predecessor")
        targets = labels[:, target_positions]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            output = model(**batch, use_cache=False, logits_to_keep=target_positions-1)
            loss = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), targets.reshape(-1))
        torch.cuda.synchronize()
        result = {**selection, **stats, "loss": float(loss), "finite_loss": bool(torch.isfinite(loss)),
                  "elapsed_forward_seconds": time.monotonic()-started,
                  "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                  "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
                  "weight_updates": 0, "truncation": False}
        results.append(result)
        with (folder / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False)+"\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del output, batch, labels, targets, loss
        torch.cuda.empty_cache()
    summary = {"model": str(Path(args.model).resolve()), "gpu":torch.cuda.get_device_name(0),
               "dtype":"bfloat16", "attention":"sdpa", "probes":results,
               "weight_updates":0, "scope":"selected_complete_input_forward_and_masked_loss_only; not_training_or_generalization"}
    (folder / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if not all(result["finite_loss"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default=str(ROOT.parents[1]/".models/Qwen3.5-9B"))
    parser.add_argument("--profile",default="student_inputs_qwen35_9b_r1")
    parser.add_argument("--output",required=True)
    main(parser.parse_args())
