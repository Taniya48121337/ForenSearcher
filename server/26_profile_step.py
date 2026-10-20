"""训练一步的算子分层计时（torch profiler；单卡 Vast，2026-10-19）。不打断正在跑的训练：用同架构、层数缩到 N 层
（线性注意力 : 全注意力 = 3 : 1 不变，隐藏维度 / 视觉塔都是真尺寸，随机权重），挂与训练相同的 LoRA / 梯度检查点 / Liger，
取 sft0 的一行真实数据跑 1 步预热 + 1 步计时，按算子与类别汇总 CUDA 时间。

    CUDA_VISIBLE_DEVICES=0 python server/26_profile_step.py --config training/configs/hi_sft0_27b.json --layers 8 --target-tokens 16000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import sft_lora as s  # noqa: E402

CATEGORIES = [  # (类别, 正则)，按顺序匹配
    ("linear_attn (fla / tilelang / causal_conv1d)", r"chunk|gated_delta|fused_recurrent|causal_conv1d|tilelang|fla_|kkt|wy_|solve_tril|recompute_w_u|chunk_fwd|chunk_bwd|l2norm"),
    ("full attention (sdpa / flash)", r"scaled_dot_product|efficient_attention|flash_attn|_flash_|fmha|attention_backward|cudnn.*attention"),
    ("GEMM", r"gemm|matmul|\bmm\b|addmm|bmm|cutlass|xmma|sm90|ampere|cublas|linear"),
    ("dropout / RNG", r"dropout|bernoulli|philox|rand"),
    ("norm (Liger / RMSNorm)", r"rms|layer_norm|norm"),
    ("activation / SwiGLU", r"silu|swiglu|gelu|sigmoid|softplus"),
    ("vision conv / patch", r"conv|im2col|cudnn"),
    ("copy / cast / memset", r"copy|cast|memcpy|memset|to_copy|contiguous|cat|clone"),
    ("elementwise / reduce", r"elementwise|vectorized|reduce|mul|add|sum|div|exp|where|index"),
]


def category(name: str) -> str:
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/hi_sft0_27b.json")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--target-tokens", type=int, default=16000, help="选输入 token 数最接近它的一行（统计来自 --rows）")
    ap.add_argument("--rows", default="/workspace/smoke/check_inputs_1019b.rows.jsonl")
    ap.add_argument("--out", default="/workspace/smoke/profile_step.json")
    ap.add_argument("--lora-dropout", type=float, default=None, help="只为对照：覆盖 lora_dropout（配方不变，训练不受影响）")
    args = ap.parse_args()

    cfg = s.TrainConfig.load(ROOT / args.config)
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    config = AutoConfig.from_pretrained(cfg.model, local_files_only=True)
    tc = config.text_config
    tc.layer_types = list(tc.layer_types)[: args.layers]
    tc.num_hidden_layers = args.layers
    if cfg.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
    extra = {"attn_implementation": cfg.attn_implementation} if cfg.attn_implementation else {}
    t0 = time.time()
    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16, **extra)
    model = s.attach_lora(model, cfg)
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    kernels = s.linear_attn_kernels(model)

    want_file = cfg.files[0] if cfg.files else f"{cfg.role}.jsonl"
    stats = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    stats = [r for r in stats if r["file"] == want_file]
    pick = min(stats, key=lambda r: abs(r["input_tokens"] - args.target_tokens))
    export = ROOT / cfg.export_dir
    row = next(json.loads(l) for l in (export / want_file).read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l)["metadata"].get("run") == pick["run"] and json.loads(l)["metadata"].get("request") == pick["request"])
    batch = s.encode(row, export, processor)
    n_prefix = batch.pop("n_prefix")
    weight = batch.pop("sample_weight")
    labels = batch.pop("labels").cuda()
    inputs = {k: v.cuda() for k, v in batch.items()}
    length = int(inputs["input_ids"].shape[1])
    keep = s.keep_count(length, n_prefix)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    model.train()

    def step() -> None:
        out = model(**inputs, logits_to_keep=keep)
        loss, _, _ = s.weighted_loss(out.logits, labels, weight)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    step()                                            # 预热（triton / tilelang JIT）
    torch.cuda.synchronize()
    t1 = time.time(); step(); torch.cuda.synchronize(); plain_s = time.time() - t1
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
    total = sum(e.self_device_time_total for e in events)
    top = sorted(events, key=lambda e: -e.self_device_time_total)[:15]
    cats: dict[str, float] = defaultdict(float)
    for e in events:
        cats[category(e.key)] += e.self_device_time_total
    result = {"layers": args.layers, "layer_types": tc.layer_types, "row": {k: pick[k] for k in ("run", "request", "input_tokens", "images")},
              "tokens": length, "step_s_unprofiled": round(plain_s, 3), "ms_per_token_this_model": round(plain_s / length * 1000, 4),
              "cuda_time_s": round(total / 1e6, 3), "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
              "lora_dropout": cfg.lora_dropout, "liger": cfg.liger, "attn": getattr(model.config, "_attn_implementation", None),
              "linear_attn_fused": kernels["fused"],
              "top_ops": [{"op": e.key[:90], "cuda_ms": round(e.self_device_time_total / 1000, 1), "share": round(e.self_device_time_total / total, 4),
                           "calls": e.count} for e in top],
              "categories": {k: {"cuda_ms": round(v / 1000, 1), "share": round(v / total, 4)} for k, v in sorted(cats.items(), key=lambda x: -x[1])},
              "setup_s": round(t1 - t0, 1)}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tokens", "step_s_unprofiled", "ms_per_token_this_model", "cuda_time_s", "gpu_mem_peak_gb", "linear_attn_fused")}, ensure_ascii=False))
    for c, v in result["categories"].items():
        print(f"  {c:46s} {v['cuda_ms']:9.1f} ms  {v['share']:6.1%}")
    for t in result["top_ops"][:8]:
        print(f"  {t['share']:6.1%}  {t['cuda_ms']:8.1f} ms  x{t['calls']:<5d} {t['op']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
