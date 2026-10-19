"""阶段 2 显存探针：从 sft0 里取最长的 N 行（按 datagen.check_inputs 的精确 input_tokens），建一个子集导出目录和一份临时训练配置，
让真 27B 用 --max-steps 恰好过一遍这些行，量训练峰值显存与每步耗时（单卡 Vast，2026-10-19 负责人要求：最长 20 行，≤ 70 GB 再谈与 vLLM 同卡）。

    python server/22_mem_probe.py --pack dist/sft0-20261019-4b79e1ed432f --rows <check_inputs 的 .rows.jsonl> --out /workspace/smoke/long20 --n 20
    # 产出 <out>/<role>/{hi|gia}.jsonl、images -> 原包 images/、<out>/<role>_long<N>.json（原配置只改 export_dir / output_dir / max_steps / log_every）
    CUDA_VISIBLE_DEVICES=0 python training/sft_lora.py --config <out>/hi_long20.json --seed 1

不改包、不改仓库里的配置；子集目录与 adapter 都在 --out 下，量完可删。max_steps = ceil(N / grad_accum)：一个 epoch 恰好把 N 行各过一次。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"hi": ROOT / "training/configs/hi_sft0_27b.json", "gia": ROOT / "training/configs/gia_sft0_27b.json"}
VARIANTS = {                                           # 名字 → 相对原配置改的字段
    "base": {},                                        # 现状：梯度检查点开、sdpa
    "nogc": {"gradient_checkpointing": False},
    "nogc_fa2_liger": {"gradient_checkpointing": False, "attn_implementation": "flash_attention_2", "liger": True},
    "gc_fa2_liger": {"attn_implementation": "flash_attention_2", "liger": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="sft0 包目录（含 hi.jsonl、gia.jsonl、images/）")
    ap.add_argument("--rows", required=True, help="datagen.check_inputs 写的逐行 token 统计（*.rows.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    pack, out = (ROOT / args.pack).resolve(), Path(args.out).resolve()
    stats = [json.loads(line) for line in Path(args.rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    overall = sorted(stats, key=lambda r: -int(r["input_tokens"]))[: args.n]
    summary: dict = {"n": args.n, "overall_top": [{k: r[k] for k in ("file", "run", "request", "input_tokens", "images")} for r in overall]}
    for role, cfg_path in CONFIGS.items():
        name = f"{role}.jsonl"
        top = sorted((r for r in stats if r["file"] == name), key=lambda r: -int(r["input_tokens"]))[: args.n]
        want = {(r["run"], r["request"]): int(r["input_tokens"]) for r in top}
        picked = []
        for line in (pack / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            md = row.get("metadata") or {}
            if (md.get("run"), md.get("request") or md.get("turn")) in want:
                picked.append(line)
        if len(picked) != len(want):
            raise SystemExit(f"{name}：统计里取了 {len(want)} 行，包里只对上 {len(picked)} 行（run / request 键不一致？）")
        sub = out / role
        sub.mkdir(parents=True, exist_ok=True)
        (sub / name).write_text("\n".join(picked) + "\n", encoding="utf-8")
        link = sub / "images"
        if not link.exists():
            link.symlink_to(pack / "images", target_is_directory=True)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg.update({"export_dir": str(sub), "output_dir": str(out / f"{role}_adapter"),
                    "max_steps": math.ceil(len(picked) / int(cfg["grad_accum"])), "log_every": 1, "save_each_epoch": False})
        cfg["_what"] = f"显存探针（server/22_mem_probe.py）：{cfg_path.name} 只改 export_dir / output_dir / max_steps / log_every；最长 {len(picked)} 行"
        configs = {}
        for variant, change in VARIANTS.items():       # 10-19 负责人：最长 20 行上比几种配置的 step 时间与峰值显存，选最快且不 OOM 的
            v = {**cfg, **change, "output_dir": str(out / f"{role}_{variant}_adapter")}
            v["_what"] = cfg["_what"] + f"；变体 {variant}：{json.dumps(change, ensure_ascii=False)}"
            path = out / f"{role}_long{args.n}_{variant}.json"
            path.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
            configs[variant] = str(path)
        summary[role] = {"configs": configs, "rows": len(picked), "max_steps": cfg["max_steps"], "grad_accum": cfg["grad_accum"],
                         "input_tokens": sorted(want.values(), reverse=True)}
    (out / "probe_rows.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "overall_top" else v[:3]) for k, v in summary.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
