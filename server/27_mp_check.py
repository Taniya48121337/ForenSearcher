"""runtime.mixed_precision 的等价性与速度核对（单卡 Vast，2026-10-20）。不打断正在跑的训练：用 Qwen3.8-27B 前 N 层的真实权重
（另加真实的词嵌入、视觉塔、输出头；线性 : 全注意力 = 3 : 1），挂与训练相同的 LoRA / 梯度检查点 / Liger，在 sft0 的 K 行真实数据上
各跑 K 次更新（每行一次更新），同一种子分别以 mixed_precision 关 / 开跑两遍，比每步 loss 与每步耗时。负责人门槛：每步 |Δloss| < 1e-3。

    CUDA_VISIBLE_DEVICES=0 python server/27_mp_check.py --config training/configs/hi_sft0_27b.json --layers 8 --steps 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import sft_lora as s  # noqa: E402


def run(cfg: s.TrainConfig, layers: int, encoded: list[dict], mixed: bool, seed: int) -> dict:
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg.mixed_precision = mixed
    config = AutoConfig.from_pretrained(cfg.model, local_files_only=True)
    config.text_config.layer_types = list(config.text_config.layer_types)[:layers]
    config.text_config.num_hidden_layers = layers
    if cfg.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = AutoModelForImageTextToText.from_pretrained(cfg.model, config=config, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0})
    model = s.attach_lora(model, cfg)
    if mixed:
        s.apply_mixed_precision(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    losses, times = [], []
    for item in encoded:
        inputs = {k: v.cuda() for k, v in item["inputs"].items()}
        labels = item["labels"].cuda()
        torch.cuda.synchronize(); t = time.time()
        with s.autocast_ctx(cfg):
            out = model(**inputs, logits_to_keep=item["keep"])
            loss, _, _ = s.weighted_loss(out.logits, labels, item["weight"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t)
        losses.append(float(loss.detach()))
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return {"losses": losses, "times": times, "peak_gb": round(peak, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/hi_sft0_27b.json")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rows", default="/workspace/smoke/check_inputs_1019b.rows.jsonl")
    ap.add_argument("--out", default="/workspace/smoke/mp_check.json")
    args = ap.parse_args()
    cfg = s.TrainConfig.load(ROOT / args.config)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    export = ROOT / cfg.export_dir
    want = cfg.files[0] if cfg.files else f"{cfg.role}.jsonl"
    stats = sorted((json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()), key=lambda r: r["input_tokens"])
    stats = [r for r in stats if r["file"] == want]
    mid = len(stats) // 2
    chosen = {(r["run"], r["request"]) for r in stats[mid - args.steps // 2: mid - args.steps // 2 + args.steps]}
    rows = [json.loads(l) for l in (export / want).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if (r["metadata"].get("run"), r["metadata"].get("request")) in chosen][: args.steps]
    encoded = []
    for r in rows:
        b = s.encode(r, export, processor)
        n_prefix = b.pop("n_prefix"); weight = b.pop("sample_weight"); labels = b.pop("labels")
        encoded.append({"inputs": b, "labels": labels, "weight": weight, "keep": s.keep_count(int(b["input_ids"].shape[1]), n_prefix),
                        "tokens": int(b["input_ids"].shape[1])})
    base = run(cfg, args.layers, encoded, False, cfg.seed)
    mixed = run(cfg, args.layers, encoded, True, cfg.seed)
    diffs = [abs(a - b) for a, b in zip(base["losses"], mixed["losses"])]
    tok = sum(e["tokens"] for e in encoded)
    steady = lambda r: sum(r["times"][1:]) / max(1, len(r["times"]) - 1)  # noqa: E731
    result = {"layers": args.layers, "steps": len(encoded), "tokens": tok, "rows_tokens": [e["tokens"] for e in encoded],
              "base": base, "mixed": mixed, "max_abs_loss_diff": round(max(diffs), 6), "mean_abs_loss_diff": round(sum(diffs) / len(diffs), 6),
              "pass_lt_1e-3": max(diffs) < 1e-3,
              "steady_s_per_step": {"base": round(steady(base), 4), "mixed": round(steady(mixed), 4)},
              "speedup": round(steady(base) / steady(mixed), 3)}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("steps", "tokens", "max_abs_loss_diff", "mean_abs_loss_diff", "pass_lt_1e-3", "steady_s_per_step", "speedup")}, ensure_ascii=False))
    for i, (a, b) in enumerate(zip(base["losses"], mixed["losses"]), 1):
        print(f"  step {i:2d}  base {a:.5f}  mixed {b:.5f}  |Δ| {abs(a - b):.2e}")
    print(f"  peak GB base {base['peak_gb']} mixed {mixed['peak_gb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
