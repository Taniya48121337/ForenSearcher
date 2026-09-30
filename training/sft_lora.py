"""第 0 轮教师 SFT：按角色分别训一套 LoRA（DESIGN.md 第 7 节：同一 Qwen3.5-9B 底座，GIA / HI 各挂一套 LoRA）。

用法（服务器）：
    python training/sft_lora.py --config training/configs/hi_r0.json
    python training/sft_lora.py --config training/configs/gia_r0.json
本地链路冒烟（无权重、无 GPU：用同架构的极小随机模型跑"取数据 → 训一步 → 存 LoRA → 加载 → 前向"）：
    python training/sft_lora.py --config training/configs/hi_r0.json --smoke-tiny

数据：export 目录的 <role>_accepted.jsonl；每行 messages（与推理路径同形）+ images + metadata.sample_weight。
损失：只算最后一条 assistant 的 token（labels 其余为 -100），逐行按 sample_weight 加权后按有效 token 数归一。
不做序列打包、不截断（超过 max_tokens 的行跳过并记数）；图片经 processor 展开，视觉塔与 embedding 冻结。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


@dataclass
class TrainConfig:
    role: str                                   # hi | gia
    export_dir: str
    model: str = ".models/Qwen3.5-9B"
    output_dir: str = "outputs/lora"
    files: list[str] = field(default_factory=list)   # 默认 <role>_accepted.jsonl
    max_tokens: int = 32768
    epochs: float = 1.0
    max_steps: int = 0                          # >0 时覆盖 epochs
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 10
    grad_accum: int = 8
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    seed: int = 0
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    use_sample_weight: bool = True
    log_every: int = 5
    save_every: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


class DecisionRows(Dataset):
    """一行一个样本；编码在 collate 里做（processor 需要图片）。"""

    def __init__(self, export_dir: Path, files: list[str]) -> None:
        self.export_dir = export_dir
        self.rows: list[dict[str, Any]] = []
        for name in files:
            path = export_dir / name
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def encode(row: dict[str, Any], export_dir: Path, processor: Any, *, max_tokens: int) -> dict[str, Any] | None:
    messages = row["messages"]
    prefix_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    images = []
    for rel in row["images"]:
        with Image.open(export_dir / rel) as im:
            images.append(im.convert("RGB"))
    full = processor(text=[full_text], images=images or None, return_tensors="pt", padding=False)
    prefix = processor(text=[prefix_text], images=images or None, return_tensors="pt", padding=False)
    n_prefix = int(prefix["input_ids"].shape[1])
    if int(full["input_ids"].shape[1]) > max_tokens:
        return None
    labels = full["input_ids"].clone()
    labels[:, :n_prefix] = -100
    out = {k: v for k, v in full.items()}
    out["labels"] = labels
    out["sample_weight"] = float(row["metadata"].get("sample_weight") or 1.0)
    return out


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weight: float) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    n = int(mask.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss * weight, n


def build_model(cfg: TrainConfig, *, smoke_tiny: bool) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    if smoke_tiny:
        # 同一架构（Qwen3_5ForConditionalGeneration，含视觉塔、线性注意力与全注意力交错层），维度缩到极小，随机初始化。
        config = AutoConfig.from_pretrained(cfg.model, local_files_only=True)
        tc, vc = config.text_config, config.vision_config
        tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 4
        tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
        tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        tc.linear_num_value_heads, tc.linear_num_key_heads, tc.linear_key_head_dim, tc.linear_value_head_dim = 4, 2, 16, 16
        vc.depth, vc.hidden_size, vc.num_heads, vc.intermediate_size, vc.out_hidden_size = 2, 32, 2, 64, 64
        model = AutoModelForImageTextToText.from_config(config)
        model = model.float()
    else:
        model = AutoModelForImageTextToText.from_pretrained(cfg.model, local_files_only=True, dtype=getattr(torch, cfg.dtype), device_map=cfg.device_map)
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # 视觉塔一律冻结（LoRA 只挂在语言塔的投影层；target_modules 若命中视觉塔的同名模块，这里再关掉）
    for name, param in model.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad_(False)
    return model, processor


def train(cfg: TrainConfig, *, smoke_tiny: bool = False) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    export_dir = Path(cfg.export_dir)
    files = cfg.files or [f"{cfg.role}_accepted.jsonl"]
    dataset = DecisionRows(export_dir, files)
    if not len(dataset):
        raise SystemExit(f"{export_dir} 里没有 {files} 的行")
    model, processor = build_model(cfg, smoke_tiny=smoke_tiny)
    device = next(model.parameters()).device
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(dataset) / cfg.grad_accum)
    max_steps = cfg.max_steps or max(1, int(steps_per_epoch * cfg.epochs))
    if smoke_tiny:
        max_steps = min(max_steps, 1)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    order = list(range(len(dataset)))
    random.shuffle(order)
    loader = DataLoader(order, batch_size=1, shuffle=False)
    model.train()
    step, micro, skipped, seen_tokens = 0, 0, 0, 0
    running_loss, running_tokens = 0.0, 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for index in loader:
        row = dataset[int(index)]
        batch = encode(row, export_dir, processor, max_tokens=cfg.max_tokens)
        if batch is None:
            skipped += 1
            continue
        weight = batch.pop("sample_weight") if cfg.use_sample_weight else 1.0
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        if smoke_tiny:
            # 极小模型：只取前 512 token 做一次前向反向，验证的是链路不是数字
            inputs = {k: (v[:, :512] if k in ("input_ids", "attention_mask") else v) for k, v in inputs.items()}
            inputs.pop("pixel_values", None); inputs.pop("image_grid_thw", None)
            inputs["input_ids"] = torch.where(inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                                              torch.tensor(processor.tokenizer.pad_token_id, device=device), inputs["input_ids"])
            labels = labels[:, :512]
            if not (labels != -100).any():
                labels[:, -8:] = inputs["input_ids"][:, -8:]
        outputs = model(**inputs)
        loss, n_tokens = weighted_loss(outputs.logits, labels, weight)
        (loss / max(1, n_tokens) / cfg.grad_accum).backward()
        running_loss += float(loss.detach()) ; running_tokens += n_tokens ; seen_tokens += int(inputs["input_ids"].shape[1])
        micro += 1
        if micro % cfg.grad_accum == 0 or smoke_tiny:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or smoke_tiny:
                record = {"step": step, "loss_per_token": running_loss / max(1, running_tokens), "lr": lr_at(step), "seen_tokens": seen_tokens,
                          "elapsed_s": round(time.time() - started, 1)}
                history.append(record); print(json.dumps(record), flush=True)
                running_loss, running_tokens = 0.0, 0
            if cfg.save_every and step % cfg.save_every == 0:
                model.save_pretrained(Path(cfg.output_dir) / f"step_{step}")
            if step >= max_steps:
                break
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "final")
    summary = {"role": cfg.role, "export_dir": str(export_dir), "files": files, "rows": len(dataset), "skipped_too_long": skipped,
               "steps": step, "trainable_params": trainable, "total_params": total, "smoke_tiny": smoke_tiny, "history": history,
               "adapter": str(out_dir / "final"), "model": cfg.model, "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "targets": cfg.lora_targets}}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 加载回来做一次前向，证明 adapter 可复用
    if smoke_tiny:
        from peft import PeftModel
        base, _ = build_model(cfg, smoke_tiny=True)
        base = base.unload() if hasattr(base, "unload") else base
        reloaded = PeftModel.from_pretrained(base, out_dir / "final")
        reloaded.eval()
        with torch.no_grad():
            ids = processor.tokenizer("冒烟：加载 LoRA 后前向", return_tensors="pt")["input_ids"]
            logits = reloaded(input_ids=ids).logits
        summary["reload_forward_ok"] = bool(logits.shape[-1] == processor.tokenizer.vocab_size or logits.ndim == 3)
        (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-tiny", action="store_true", help="同架构极小随机模型，只验证链路")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export-dir", default=None, help="覆盖配置里的 export_dir（例如本地核验导出的目录）")
    args = parser.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.export_dir:
        cfg.export_dir = args.export_dir
    train(cfg, smoke_tiny=args.smoke_tiny)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
