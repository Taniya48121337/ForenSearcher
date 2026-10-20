"""按角色分别训一套 LoRA（DESIGN.md 第 7 节：同一 Qwen3.5 底座，GIA / HI 各挂一套 LoRA）。

用法：
    # 服务器 9B（一张卡一个任务，由 server/13_train_queue.sh 设 CUDA_VISIBLE_DEVICES）
    python training/sft_lora.py --config training/configs/hi_r0_9b.json --seed 1 --output-dir outputs/lora/r0/hi_s1
    # 本机 4B 冒烟（scripts\\_训练链路冒烟_4B.cmd 会先查显卡占用，再调这里，再调 training/smoke_infer.py）
    python training/sft_lora.py --config training/configs/hi_smoke_4b.json
    # 无权重、无 GPU 的极小随机模型（同架构）只验证代码路径
    python training/sft_lora.py --config training/configs/hi_r0.json --smoke-tiny

数据：export 目录（或 pack_sft / pack_r0 的产物）的 <role>_accepted.jsonl；每行 messages（与推理路径同形）+ images +
metadata.sample_weight。

损失（10-15 起）：
- 只算最后一条 assistant 的 token（其余 labels = -100）。
- 一行的损失 = sample_weight × 该行监督 token 的平均交叉熵；grad_accum 行求平均后反传一次。
  即 sample_weight 直接乘进 loss（《SFT数据构造方法v2》§6：权重只在 loss 上乘，不改采样概率）；按行平均而不是按 token
  平均，是为了不让一份 2K token 的报告在梯度里抵 10 条动作行——行与行的相对分量只由 sample_weight 决定。
- 10-19（sft0 六节裁决 4）：加载时先把进训练的行的 sample_weight 归一到均值 1（normalize_sample_weight，默认开；只改内存，不改包），
  不同包（r1 均值约 2、sft0 已是 1、r0 基线包）在同一 lr 下才是同一尺度；归一前后的均值 / 系数写进 prescan.sample_weight_normalization。
- logits 只在监督段上算（logits_to_keep）：Qwen3.5 词表 248K，32K 长度的整段 logits 在 bf16 下约 16GB、转 float
  再翻倍，9B 在 80GB 卡上会被这一项挤爆；监督段通常只有几百 token。数值与整段计算相同（test：--check-loss）。

epoch（10-15 修）：旧版只遍历一遍数据就退出，epochs=2 实际只训了 1 遍；现在每个 epoch 重新打乱，epoch 末尾不足
grad_accum 的余数也做一次更新。开训前预扫一遍全部行，算出精确 token 数：超过 max_tokens 的行跳过并记数（清单 B9），
steps_per_epoch 按实际可用行数算。

显存与卡：
- 默认开梯度检查点。不开时 9B 在 32K 长度上一张 80GB 卡放不下激活。
- 一个训练任务只用一张卡。device_map="auto" 而进程又看得到多张卡时直接退出——否则 HF 会把模型按层摊到所有卡上
  串行跑。确实要跨卡（例如 27B 学生）时在配置里写 "allow_multi_gpu": true，并只给它看需要的那几张卡。

配置文件里以 "_" 开头的键（"_why" 等）是给人看的说明，代码忽略。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RUNTIME_KEYS = frozenset({"gradient_checkpointing", "attn_implementation", "liger", "mixed_precision"})   # 配置 "runtime" 段允许的键（只影响速度 / 显存）


@dataclass
class TrainConfig:
    role: str                                   # hi | gia
    export_dir: str
    model: str = ".models/Qwen3.5-9B"
    output_dir: str = "outputs/lora"
    files: list[str] = field(default_factory=list)   # 默认 <role>_accepted.jsonl
    max_tokens: int = 32768                     # 一行（输入 + 监督）的 token 上限；超过的行跳过并记数
    epochs: float = 1.0
    max_steps: int = 0                          # >0 时覆盖 epochs
    lr: float = 1e-4
    min_lr_ratio: float = 0.0                   # 余弦退火的终点 = lr × min_lr_ratio
    weight_decay: float = 0.0
    warmup_steps: int = 10
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    seed: int = 0
    seeds: list[int] = field(default_factory=list)   # 登记的种子（清单 E4）；--seed 不在其中时只警告
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    use_sample_weight: bool = True
    # 10-19 负责人定（sft0 六节裁决 4）：加载时把进训练的行的 sample_weight 归一到均值 1（只改内存，不改包）。各包的权重尺度不同
    # （r1 包均值约 2.0、sft0 合包后已是 1、r0 基线包另有尺度），不归一的话同一 lr 下梯度尺度跟着包走，基线与 sft0 不可比
    normalize_sample_weight: bool = True
    logits_on_supervised_only: bool = True     # logits_to_keep：只在监督段算 logits（见文件头）
    row_select: dict[str, Any] | None = None   # 冒烟用：{"n": 20, "max_input_tokens_est": 16000, "order": "shortest", "exclude_requests": [...]}
    log_every: int = 5
    save_every: int = 0                         # >0：每 N 次更新存一个可续训的 checkpoint-<N>/（adapter + 优化器 + 数据位置 + RNG），见 save_checkpoint
    keep_checkpoints: int = 2                   # 只留最近几个 checkpoint-*（epoch_* / final 不受影响）
    save_each_epoch: bool = False               # 每个 epoch 末存一份 epoch_<n>/（门 B 可以直接比 epoch 1 与 epoch 2，不用重训）
    gradient_checkpointing: bool = True
    allow_multi_gpu: bool = False
    # 10-19（Vast 单卡显存 / 速度探针）：None = transformers 默认（Qwen3.5 全注意力层与视觉塔是 sdpa）；"flash_attention_2" 要装 flash-attn
    attn_implementation: str | None = None
    liger: bool = False                         # True：加载前打 Liger 的 Qwen3.5 内核（RMSNorm / SwiGLU；Liger 没有 Qwen3.5 的 RoPE；不碰 loss——这里不给模型传 labels）
    # 10-20：True = 前向与 loss 在 torch.autocast(bf16) 下算，并关掉 peft 把 bf16 激活转成 fp32 的输入转换（LoRA 权重仍是 fp32 主权重、梯度 fp32）；
    # 默认关。关时 peft 每个挂 LoRA 的层每次调用都把激活拷成 fp32（profiler 里 copy / cast 约 14%）
    mixed_precision: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 10-19：配置里的 "runtime" 只放吞吐 / 显存开关（不是配方，不进配方 hash、冻结标识与续训身份），覆盖同名顶层字段
        runtime = data.pop("runtime", None) or {}
        bad = sorted(set(runtime) - RUNTIME_KEYS)
        if bad:
            raise SystemExit(f"{path}: runtime 里只认 {sorted(RUNTIME_KEYS)}，不认识 {bad}")
        data.update(runtime)
        known = set(cls.__dataclass_fields__)
        unknown = [k for k in data if not k.startswith("_") and k not in known]
        if unknown:
            raise SystemExit(f"{path}: 不认识的配置项 {unknown}")
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- data

def load_rows(export_dir: Path, files: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in files:
        for line in (export_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_sample_weights(rows: list[dict[str, Any]], usable: list[int], enabled: bool) -> dict[str, Any]:
    """把 usable 行的 metadata.sample_weight 乘同一系数使均值 = 1（就地改内存里的行；包文件不动）。返回前后均值与系数。"""

    ws = [float((rows[i].get("metadata") or {}).get("sample_weight") or 1.0) for i in usable]
    mean = sum(ws) / len(ws) if ws else 1.0
    factor = 1.0 / mean if enabled and mean > 0 else 1.0
    if factor != 1.0:
        for i in usable:
            md = rows[i].setdefault("metadata", {})
            md["sample_weight"] = float(md.get("sample_weight") or 1.0) * factor
    after = [float((rows[i].get("metadata") or {}).get("sample_weight") or 1.0) for i in usable]
    return {"enabled": enabled, "rows": len(ws), "mean_before": round(mean, 6), "sum_before": round(sum(ws), 4), "factor": round(factor, 6),
            "mean_after": round(sum(after) / len(after), 6) if after else None, "sum_after": round(sum(after), 4)}


def row_key(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    return f"{md.get('run')}/{md.get('request') or md.get('turn')}"


def select_rows(rows: list[dict[str, Any]], spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """冒烟取子集：按估计输入长度筛、按 order 取前 n 行；动作类型尽量覆盖（每种动作先各取一行再按顺序补）。"""

    if not spec:
        return rows
    exclude = set(spec.get("exclude_requests") or [])
    limit = spec.get("max_input_tokens_est")
    pool = [r for r in rows if row_key(r) not in exclude
            and (limit is None or (r.get("metadata") or {}).get("input_tokens_est", 0) <= limit)]
    if spec.get("order", "shortest") == "shortest":
        pool.sort(key=lambda r: (r.get("metadata") or {}).get("input_tokens_est", 0))
    n = int(spec.get("n") or len(pool))
    picked: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    for r in pool:                                    # 第一遍：每种动作一行
        action = (r.get("metadata") or {}).get("action")
        if action not in seen_actions:
            seen_actions.add(action)
            picked.append(r)
    for r in pool:                                    # 第二遍：按顺序补满
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)
    return picked[:n]


def encode(row: dict[str, Any], export_dir: Path, processor: Any) -> dict[str, Any]:
    messages = row["messages"]
    prefix_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full_text.startswith(prefix_text):
        raise ValueError(f"{row_key(row)}：聊天模板改变了 assistant 前缀边界")
    images = []
    for rel in row["images"]:
        with Image.open(export_dir / rel) as im:
            images.append(im.convert("RGB"))
    full = processor(text=[full_text], images=images or None, return_tensors="pt", padding=False)
    prefix = processor(text=[prefix_text], images=images or None, return_tensors="pt", padding=False)
    n_prefix = int(prefix["input_ids"].shape[1])
    if not torch.equal(full["input_ids"][:, :n_prefix], prefix["input_ids"]):
        raise ValueError(f"{row_key(row)}：图片展开或分词改变了前缀 token")
    labels = full["input_ids"].clone()
    labels[:, :n_prefix] = -100
    out = {k: v for k, v in full.items()}
    out["labels"] = labels
    out["n_prefix"] = n_prefix
    out["sample_weight"] = float((row.get("metadata") or {}).get("sample_weight") or 1.0)
    return out


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {name: ordered[round((len(ordered) - 1) * q)] for name, q in
            [("min", 0), ("p50", .5), ("p90", .9), ("p95", .95), ("max", 1)]}


def prescan(rows: list[dict[str, Any]], export_dir: Path, processor: Any, max_tokens: int) -> tuple[list[int], dict[str, Any]]:
    """逐行编码一次，拿精确 token 数（不留张量）。返回可用行下标与统计。"""

    usable: list[int] = []
    total, supervised, too_long, failed = [], [], [], []
    for i, row in enumerate(rows):
        try:
            enc = encode(row, export_dir, processor)
        except Exception as exc:                      # 模板 / 图片问题：记下来，不静默
            failed.append({"row": row_key(row), "error": str(exc)[:300]})
            continue
        n = int(enc["input_ids"].shape[1])
        total.append(n)
        supervised.append(n - enc["n_prefix"])
        if n > max_tokens:
            too_long.append({"row": row_key(row), "tokens": n})
        else:
            usable.append(i)
    stats = {"rows": len(rows), "usable": len(usable), "skipped_too_long": len(too_long), "too_long": too_long[:50],
             "encode_failed": failed, "tokens": quantiles(total), "supervised_tokens": quantiles(supervised),
             "sample_weight_sum_usable": round(sum(float((rows[i].get("metadata") or {}).get("sample_weight") or 1.0) for i in usable), 3)}
    return usable, stats


# --------------------------------------------------------------------------- loss

def prescan_cached(rows: list[dict[str, Any]], export_dir: Path, files: list[str], processor: Any, model_path: str, max_tokens: int,
                   row_select: dict[str, Any] | None, cache_dir: Path) -> tuple[list[int], dict[str, Any]]:
    """10-20：prescan 的结果按"数据文件 + processor 文件 + max_tokens + row_select"缓存到 output_dir/prescan_cache.json。
    续训 / 重启时 key 相同就直接用（预扫要把每行编码一遍，HI 3,369 行约 13 分钟 GPU 空转）；key 不同就重扫并覆盖。"""

    proc_files = sorted(p for p in Path(model_path).glob("*") if p.name in
                        {"tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja", "video_preprocessor_config.json"})
    key_src = {"data": {name: sha256_file(export_dir / name) for name in files},
               "processor": {p.name: sha256_file(p) for p in proc_files}, "max_tokens": max_tokens, "row_select": row_select,
               "rows": [row_key(r) for r in rows]}
    key = hashlib.sha256(json.dumps(key_src, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    cache = cache_dir / "prescan_cache.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                print(f"prescan：用缓存 {cache}（key {key[:12]}）", flush=True)
                return list(data["usable"]), dict(data["stats"], cached=True)
        except (OSError, ValueError, KeyError):
            pass
    usable, stats = prescan(rows, export_dir, processor, max_tokens)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "key_src": {k: v for k, v in key_src.items() if k != "rows"}, "usable": usable, "stats": stats},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache)
    return usable, stats


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, weight: float) -> tuple[torch.Tensor, int, float]:
    """logits 与 labels 末端对齐（logits 可以只覆盖最后 K 个位置）。返回 (weight × 平均 CE, 监督 token 数, 未加权平均 CE)。"""

    k = logits.shape[1]
    labels = labels[:, -k:]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    n = int((shift_labels != -100).sum())
    if n == 0:
        zero = logits.sum() * 0.0
        return zero, 0, 0.0
    ce = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
                                           ignore_index=-100, reduction="sum") / n
    return ce * weight, n, float(ce.detach())


def keep_count(batch_len: int, n_prefix: int) -> int:
    """监督段 [n_prefix, L) 的预测来自位置 [n_prefix-1, L-1)，留最后 L-n_prefix+1 个位置的 logits 即可。"""

    return max(2, batch_len - n_prefix + 1)


# --------------------------------------------------------------------------- model

def check_visible_gpus(cfg: TrainConfig, count: int | None = None) -> None:
    n = torch.cuda.device_count() if count is None else count
    if cfg.device_map == "auto" and n > 1 and not cfg.allow_multi_gpu:
        raise SystemExit(f"这个进程能看到 {n} 张 GPU，而 device_map=auto 会把模型摊到全部卡上串行跑（不会更快，还占掉推理卡）。"
                         "请用 CUDA_VISIBLE_DEVICES=<一张卡> 启动（服务器用 server/13_train_queue.sh），"
                         "确实要跨卡时在配置里写 \"allow_multi_gpu\": true。")


def enable_checkpointing(model: Any) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except ValueError as exc:
        raise SystemExit(f"这个模型不支持梯度检查点：{exc}。可在配置里写 \"gradient_checkpointing\": false 并调小 max_tokens。") from exc
    model.enable_input_require_grads()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False


def tiny_model(cfg: TrainConfig) -> Any:
    """同一架构（Qwen3_5ForConditionalGeneration：视觉塔 + 线性注意力 / 全注意力交错层），维度缩到极小，随机初始化。"""

    config = AutoConfig.from_pretrained(cfg.model, local_files_only=True)
    tc, vc = config.text_config, config.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = 64, 128, 4
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    tc.linear_num_value_heads, tc.linear_num_key_heads, tc.linear_key_head_dim, tc.linear_value_head_dim = 4, 2, 16, 16
    vc.depth, vc.hidden_size, vc.num_heads, vc.intermediate_size, vc.out_hidden_size = 2, 32, 2, 64, 64
    return AutoModelForImageTextToText.from_config(config).float()


def load_base(cfg: TrainConfig, *, smoke_tiny: bool) -> Any:
    if smoke_tiny:
        return tiny_model(cfg)
    if torch.cuda.is_available():
        check_visible_gpus(cfg)
    if cfg.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        # rope=False：Liger 0.8.4 对 Qwen3_5 没有 RoPE 内核（NotImplementedError）
        apply_liger_kernel_to_qwen3_5(rope=False, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
    extra = {"attn_implementation": cfg.attn_implementation} if cfg.attn_implementation else {}
    return AutoModelForImageTextToText.from_pretrained(cfg.model, local_files_only=True, dtype=getattr(torch, cfg.dtype),
                                                       device_map=cfg.device_map, **extra)


def lora_coverage(model: Any) -> dict[str, Any]:
    """LoRA 实际挂到了哪些模块：按模块名末段与层类型计数。Qwen3.5 的线性注意力层投影叫 in_proj_qkv / in_proj_z / out_proj，
    默认 targets 不含它们（见 configs 的 _why）；这里如实记下，免得以为全挂上了。"""

    by_name: dict[str, int] = {}
    by_layer_kind: dict[str, int] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not len(getattr(module, "lora_A", {})):
            continue
        leaf = name.rsplit(".", 1)[-1]
        by_name[leaf] = by_name.get(leaf, 0) + 1
        kind = "linear_attn" if ".linear_attn." in name else "self_attn" if ".self_attn." in name else "mlp" if ".mlp." in name else "other"
        by_layer_kind[kind] = by_layer_kind.get(kind, 0) + 1
    return {"modules": by_name, "by_kind": by_layer_kind}


def linear_attn_kernels(model: Any) -> dict[str, Any]:
    """Qwen3.5 的线性注意力层（GatedDeltaNet）实际用的是哪套实现：fla / causal-conv1d 的融合内核，还是 transformers 的 torch 回退
    （缺包时 causal_conv1d_fn = None、chunk_gated_delta_rule = torch_chunk_gated_delta_rule 等）。按"函数所在模块.函数名"计数。"""

    found: dict[str, dict[str, int]] = {}
    layers = 0
    for module in model.modules():
        if "GatedDeltaNet" not in type(module).__name__:
            continue
        layers += 1
        for attr in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
            fn = getattr(module, attr, None)
            name = "None（torch 回退）" if fn is None else f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', type(fn).__name__)}"
            found.setdefault(attr, {})[name] = found.setdefault(attr, {}).get(name, 0) + 1
        norm = getattr(module, "norm", None)
        if norm is not None:
            key = f"{type(norm).__module__}.{type(norm).__name__}"
            found.setdefault("norm", {})[key] = found.setdefault("norm", {}).get(key, 0) + 1
    fused = layers > 0 and all(not any(k.startswith(("None", "transformers.")) for k in v) for v in found.values())
    return {"linear_attn_layers": layers, "fused": fused, "impl": found}


def attach_lora(model: Any, cfg: TrainConfig) -> Any:
    if cfg.gradient_checkpointing:
        enable_checkpointing(model)
    from peft import LoraConfig, get_peft_model
    # 不用 exclude_modules：adapter_config.json 保持最少的字段，vLLM 加载时少一个变数。Qwen3.5 视觉塔的投影叫
    # qkv / proj / linear_fc1 / linear_fc2，默认 targets 本来就碰不到；下面再核一遍，碰到了就停。
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    in_visual = [n for n, m in model.named_modules() if "visual" in n and hasattr(m, "lora_A") and len(getattr(m, "lora_A", {}))]
    if in_visual:
        raise SystemExit(f"LoRA 挂到了视觉塔上（{in_visual[:3]}…），检查 lora_targets")
    for name, param in model.named_parameters():          # 视觉塔一律冻结
        if "visual" in name or "vision" in name:
            param.requires_grad_(False)
    return model


def apply_mixed_precision(model: Any) -> int:
    """runtime.mixed_precision：关掉 peft 层的输入 dtype 转换（整个训练期间都关——梯度检查点的重算发生在 backward 里，
    只包住前向的 context manager 管不到）。返回改了多少层。"""

    from peft.tuners.tuners_utils import BaseTunerLayer

    n = 0
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.cast_input_dtype_enabled = False
            n += 1
    return n


def autocast_ctx(cfg: TrainConfig) -> Any:
    import contextlib
    return torch.autocast("cuda", dtype=torch.bfloat16) if cfg.mixed_precision and torch.cuda.is_available() else contextlib.nullcontext()


def build_model(cfg: TrainConfig, *, smoke_tiny: bool) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    model = attach_lora(load_base(cfg, smoke_tiny=smoke_tiny), cfg)
    return model, processor


# --------------------------------------------------------------------------- provenance

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def export_provenance(export_dir: Path, files: list[str]) -> dict[str, Any]:
    """训练用的是哪份数据：文件 hash + 导出 manifest 里的身份字段（r0 包还会有 r0_pack.json）。"""

    out: dict[str, Any] = {"files": {name: sha256_file(export_dir / name) for name in files}}
    for name, path in (("r0_pack.json", export_dir.parent / "r0_pack.json"), ("r0_pack.json", export_dir / "r0_pack.json"),
                       ("manifest.json", export_dir / "manifest.json")):
        if path.is_file() and name not in out:          # r0 包：r0_pack.json 在 data/ 的上一级
            data = json.loads(path.read_text(encoding="utf-8"))
            keep = ("pack_id", "status", "data_digest", "export_id", "freeze_id", "spec", "spec_hash", "state_renderer", "verifier", "verifier_code_hash",
                    "judge_model", "tool_layer_hash", "teacher_model", "counts")
            out[name] = {k: data[k] for k in keep if k in data}
    return out


def runtime_info() -> dict[str, Any]:
    import transformers
    info: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                            "platform": platform.platform()}
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


# --------------------------------------------------------------------------- train

def dry_run(cfg: TrainConfig) -> dict[str, Any]:
    """只读数据与分词器：预扫 token、算步数，不加载模型、不要 GPU。服务器开训前 / 打包后核对用。"""

    export_dir = Path(cfg.export_dir)
    files = cfg.files or [f"{cfg.role}_accepted.jsonl"]
    rows = select_rows(load_rows(export_dir, files), cfg.row_select)
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    usable, scan = prescan(rows, export_dir, processor, cfg.max_tokens)
    scan["sample_weight_normalization"] = normalize_sample_weights(rows, usable, cfg.normalize_sample_weight)
    steps_per_epoch = math.ceil(len(usable) / cfg.grad_accum) if usable else 0
    plan = {"role": cfg.role, "export_dir": str(export_dir), "rows": len(rows), "usable": len(usable),
            "steps_per_epoch": steps_per_epoch, "max_steps": cfg.max_steps or math.ceil(steps_per_epoch * cfg.epochs),
            "prescan": scan, "data": export_provenance(export_dir, files)}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


CHECKPOINT_STATE = "trainer_state.pt"
# 换了这些字段就不是同一次训练，续训时只比其余字段（输出位置、存盘频率、试跑用的停止步不影响训练轨迹）
_RESUME_IGNORED = frozenset({"output_dir", "save_every", "keep_checkpoints", "log_every"}) | RUNTIME_KEYS


def run_identity(cfg: TrainConfig, rows: list[dict[str, Any]], usable: list[int]) -> dict[str, Any]:
    """续训的身份：配置（去掉 _RESUME_IGNORED）+ 进训练的行（键与顺序）的 hash。不一致就不续。"""

    keys = "\n".join(row_key(rows[i]) for i in usable).encode("utf-8")
    return {"config": {k: v for k, v in asdict(cfg).items() if k not in _RESUME_IGNORED},
            "usable_rows": len(usable), "usable_sha256": hashlib.sha256(keys).hexdigest()}


def complete_checkpoints(out_dir: Path) -> list[Path]:
    found = [p for p in out_dir.glob("checkpoint-*") if (p / "COMPLETE").is_file() and p.name.split("-", 1)[1].isdigit()]
    return sorted(found, key=lambda p: int(p.name.split("-", 1)[1]))


def save_checkpoint(model: Any, optimizer: Any, out_dir: Path, state: dict[str, Any], keep: int) -> Path:
    """10-19（Vast 单卡、宿主机维护期间实例可能被停）：在一次更新之后（梯度累积窗口为空）存可续训的状态。
    先写到 .tmp-checkpoint-<step>/，写完 COMPLETE 再改名，断电 / 被杀时不会留下半个 checkpoint；只留最近 keep 个。"""

    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    target, tmp = out_dir / f"checkpoint-{state['step']}", out_dir / f".tmp-checkpoint-{state['step']}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save({**state, "optimizer": optimizer.state_dict(), "rng": rng}, tmp / CHECKPOINT_STATE)
    (tmp / "COMPLETE").write_text(json.dumps({"step": state["step"], "epoch": state["epoch"], "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                  encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    for old in complete_checkpoints(out_dir)[:-max(1, keep)]:
        shutil.rmtree(old, ignore_errors=True)
    print(json.dumps({"checkpoint": str(target), "step": state["step"], "epoch": state["epoch"]}, ensure_ascii=False), flush=True)
    return target


def load_checkpoint(model: Any, optimizer: Any, path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """把 adapter 权重、优化器与 RNG 恢复到 path 存盘时的样子；身份（配置 + 训练行）不一致就停。"""

    from peft import set_peft_model_state_dict

    state = torch.load(path / CHECKPOINT_STATE, map_location="cpu", weights_only=False)
    if state.get("identity") != identity:
        saved, now = state.get("identity") or {}, identity
        diff = sorted(k for k in set(saved.get("config", {})) | set(now["config"]) if saved.get("config", {}).get(k) != now["config"].get(k))
        raise SystemExit(f"{path} 与这次的配置 / 训练行不是同一次训练，不能续（不同的配置项 {diff}；"
                         f"训练行 {saved.get('usable_rows')} / {now['usable_rows']}，hash 相同 {saved.get('usable_sha256') == now['usable_sha256']}）")
    weights_file = path / "adapter_model.safetensors"
    if weights_file.is_file():
        from safetensors.torch import load_file
        weights = load_file(str(weights_file))
    else:
        weights = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
    result = set_peft_model_state_dict(model, weights)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise SystemExit(f"{path} 的 adapter 权重有 {len(unexpected)} 个对不上的键（{unexpected[:3]}…）")
    optimizer.load_state_dict(state["optimizer"])
    rng = state["rng"]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def check_pack_status(export_dir: Path, allow_trial: bool) -> None:
    """r0 包（data/ 的上一级有 r0_pack.json）只认 status=ok 的；试包要显式 --allow-trial-pack。"""

    path = export_dir.parent / "r0_pack.json"
    if not path.is_file():
        return
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    if status != "ok":
        msg = f"{path} 的 status={status}（试包，不合冻结口径），不能当正式 r0 训练数据"
        if not allow_trial:
            raise SystemExit(msg + "；确实要用加 --allow-trial-pack")
        print("!! " + msg)


def train(cfg: TrainConfig, *, smoke_tiny: bool = False, check_loss: bool = False, allow_trial_pack: bool = False,
          resume: bool = False, stop_at_step: int | None = None) -> dict[str, Any]:
    check_pack_status(Path(cfg.export_dir), allow_trial_pack)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.seeds and cfg.seed not in cfg.seeds:
        print(f"!! 种子 {cfg.seed} 不在配置登记的 seeds {cfg.seeds} 里（清单 E4）")
    t0 = time.time()
    export_dir = Path(cfg.export_dir)
    files = cfg.files or [f"{cfg.role}_accepted.jsonl"]
    all_rows = load_rows(export_dir, files)
    rows = select_rows(all_rows, cfg.row_select)
    if not rows:
        raise SystemExit(f"{export_dir} 里没有 {files} 的行")
    model, processor = build_model(cfg, smoke_tiny=smoke_tiny)
    device = next(model.parameters()).device
    load_s = time.time() - t0
    kernels = linear_attn_kernels(model)
    print(json.dumps({"linear_attn_kernels": kernels}, ensure_ascii=False), flush=True)
    if cfg.mixed_precision and not smoke_tiny:
        print(json.dumps({"mixed_precision": {"peft_layers_input_cast_off": apply_mixed_precision(model), "autocast": "bfloat16"}}), flush=True)
    if smoke_tiny:
        usable, scan = prescan(rows, export_dir, processor, 10 ** 9)
    else:
        usable, scan = prescan_cached(rows, export_dir, files, processor, cfg.model, cfg.max_tokens, cfg.row_select, Path(cfg.output_dir))
    scan["sample_weight_normalization"] = normalize_sample_weights(rows, usable, cfg.normalize_sample_weight)
    if scan["encode_failed"]:
        print(f"!! {len(scan['encode_failed'])} 行编码失败：{scan['encode_failed'][:3]}")
    if not usable:
        raise SystemExit(f"没有可用行（{scan['skipped_too_long']} 行超过 max_tokens={cfg.max_tokens}）")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(usable) / cfg.grad_accum)
    max_steps = cfg.max_steps or max(1, math.ceil(steps_per_epoch * cfg.epochs))
    if smoke_tiny:
        max_steps = min(max_steps, 2)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        floor = cfg.lr * cfg.min_lr_ratio
        return floor + (cfg.lr - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    step, micro_in_window, rows_seen, seen_tokens, epoch = 0, 0, 0, 0, 0
    win_loss_w, win_loss_raw, win_weight, win_rows, win_tokens = 0.0, 0.0, 0.0, 0, 0
    history: list[dict[str, Any]] = []
    loss_checks: list[dict[str, float]] = []
    out_dir = Path(cfg.output_dir)
    identity = run_identity(cfg, rows, usable)
    pending: tuple[list[int], int] | None = None        # 续训：本 epoch 的打乱顺序与已过的行数
    resumed_from, elapsed_before = None, 0.0
    if resume:
        found = complete_checkpoints(out_dir)
        if found:
            state = load_checkpoint(model, optimizer, found[-1], identity)
            step, epoch, rows_seen, seen_tokens = state["step"], state["epoch"], state["rows_seen"], state["seen_tokens"]
            history, elapsed_before = list(state.get("history") or []), float(state.get("elapsed_s") or 0.0)
            if state.get("order") is not None:
                pending = (list(state["order"]), int(state["pos"]))
            resumed_from = str(found[-1])
            print(json.dumps({"resumed_from": resumed_from, "step": step, "epoch": epoch, "max_steps": max_steps}, ensure_ascii=False), flush=True)
        else:
            print(f"--resume：{out_dir} 里没有完整的 checkpoint-*，从头训", flush=True)
    train_started = time.time() - elapsed_before
    stopped_early = False

    def checkpoint(order: list[int] | None, pos: int) -> None:
        """order = None：本 epoch 已走完（epoch 计数已加 1），续训时从下一次打乱开始。"""
        save_checkpoint(model, optimizer, out_dir, {"step": step, "epoch": epoch, "order": order, "pos": pos, "rows_seen": rows_seen,
                                                    "seen_tokens": seen_tokens, "history": history, "identity": identity,
                                                    "elapsed_s": round(time.time() - train_started, 1), "max_steps": max_steps},
                        cfg.keep_checkpoints)

    def optimizer_step() -> None:
        nonlocal step, micro_in_window, win_loss_w, win_loss_raw, win_weight, win_rows, win_tokens
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if micro_in_window != cfg.grad_accum:          # epoch 末尾的余数：按实际行数重新归一
            scale = cfg.grad_accum / micro_in_window
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.log_every == 0 or step == 1 or step >= max_steps:
            record = {"step": step, "epoch": epoch, "rows": win_rows, "loss_weighted": round(win_loss_w / max(1, win_rows), 5),
                      "loss_unweighted": round(win_loss_raw / max(1, win_rows), 5), "mean_weight": round(win_weight / max(1, win_rows), 3),
                      "supervised_tokens": win_tokens, "grad_norm": round(float(grad_norm), 4), "lr": lr, "seen_tokens": seen_tokens,
                      "elapsed_s": round(time.time() - train_started, 1)}
            if torch.cuda.is_available():
                record["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        micro_in_window = 0
        win_loss_w = win_loss_raw = win_weight = 0.0
        win_rows = win_tokens = 0

    while step < max_steps and not stopped_early:
        if pending is not None:                        # 续训：接着存盘时那个 epoch 的同一顺序往下走
            order, start = pending
            pending = None
        else:
            order = list(usable)
            random.shuffle(order)
            start = 0
        for pos in range(start, len(order)):
            index = order[pos]
            batch = encode(rows[index], export_dir, processor)
            n_prefix = batch.pop("n_prefix")
            weight = batch.pop("sample_weight") if cfg.use_sample_weight else 1.0
            batch.pop("sample_weight", None)
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            if smoke_tiny:                              # 极小模型：截到前 512 token、去掉像素，验证的是链路不是数字
                pad = processor.tokenizer.pad_token_id
                image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                inputs = {k: (v[:, :512] if k in ("input_ids", "attention_mask") else v) for k, v in inputs.items()
                          if k not in ("pixel_values", "image_grid_thw", "mm_token_type_ids")}
                inputs["input_ids"] = torch.where(inputs["input_ids"] == image_pad, torch.tensor(pad, device=device), inputs["input_ids"])
                labels = labels[:, :512]
                if not (labels != -100).any():
                    labels[:, -8:] = inputs["input_ids"][:, -8:]
                n_prefix = int((labels == -100).cumprod(dim=1).sum())
            length = int(inputs["input_ids"].shape[1])
            keep = keep_count(length, n_prefix) if cfg.logits_on_supervised_only else 0
            with autocast_ctx(cfg) if not smoke_tiny else contextlib.nullcontext():
                outputs = model(**inputs, logits_to_keep=keep)
                loss, n_tokens, raw = weighted_loss(outputs.logits, labels, weight)
            if check_loss:                               # 与整段 logits 的结果对比（只在小模型 / 少量行上用）
                with torch.no_grad():
                    full_logits = model(**inputs, logits_to_keep=0).logits
                    ref, _, _ = weighted_loss(full_logits, labels, weight)
                loss_checks.append({"kept": float(loss.detach()), "full": float(ref), "keep": keep, "length": length})
            (loss / cfg.grad_accum).backward()
            win_loss_w += float(loss.detach()); win_loss_raw += raw; win_weight += weight
            win_rows += 1; win_tokens += n_tokens; rows_seen += 1; seen_tokens += length
            micro_in_window += 1
            if micro_in_window == cfg.grad_accum:
                optimizer_step()
                if cfg.save_every and step % cfg.save_every == 0 and step < max_steps:
                    checkpoint(order, pos + 1)
                    if stop_at_step and step >= stop_at_step:
                        stopped_early = True
                        break
                if step >= max_steps:
                    break
        if stopped_early:
            break
        remainder_step = bool(micro_in_window) and step < max_steps
        if remainder_step:                              # epoch 末尾不足 grad_accum 的余数
            optimizer_step()
        epoch += 1
        if cfg.save_each_epoch and step < max_steps:    # 最后一个 epoch 存成 final，不重复存
            model.save_pretrained(Path(cfg.output_dir) / f"epoch_{epoch}")
        if remainder_step and cfg.save_every and step % cfg.save_every == 0 and step < max_steps:
            checkpoint(None, 0)
            if stop_at_step and step >= stop_at_step:
                stopped_early = True

    train_s = time.time() - train_started
    out_dir.mkdir(parents=True, exist_ok=True)
    if stopped_early:                                   # --stop-at-step（续训试跑）：只留 checkpoint，不写 final
        print(json.dumps({"stopped_at_step": step, "checkpoint": str(complete_checkpoints(out_dir)[-1])}, ensure_ascii=False), flush=True)
        return {"stopped_at_step": step, "history": history}
    model.save_pretrained(out_dir / "final")
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"role": cfg.role, "export_dir": str(export_dir), "files": files, "rows_in_files": len(all_rows),
               "rows_selected": len(rows), "selected": [row_key(rows[i]) for i in usable] if cfg.row_select else None,
               "prescan": scan, "steps": step, "epochs_done": epoch, "rows_seen": rows_seen, "max_steps": max_steps,
               "steps_per_epoch": steps_per_epoch, "trainable_params": trainable, "total_params": total,
               "smoke_tiny": smoke_tiny, "history": history, "adapter": str(out_dir / "final"), "model": cfg.model,
               "resumed_from": resumed_from, "save_every": cfg.save_every, "linear_attn_kernels": kernels,
               "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout, "targets": cfg.lora_targets,
                        "coverage": lora_coverage(model)},
               "hyper": {"lr": cfg.lr, "min_lr_ratio": cfg.min_lr_ratio, "epochs": cfg.epochs, "warmup_steps": cfg.warmup_steps,
                         "grad_accum": cfg.grad_accum, "max_tokens": cfg.max_tokens, "weight_decay": cfg.weight_decay,
                         "max_grad_norm": cfg.max_grad_norm, "use_sample_weight": cfg.use_sample_weight,
                         "normalize_sample_weight": cfg.normalize_sample_weight,
                         "logits_on_supervised_only": cfg.logits_on_supervised_only, "save_each_epoch": cfg.save_each_epoch,
                         "attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", cfg.attn_implementation),
                         "liger": cfg.liger, "mixed_precision": cfg.mixed_precision},
               "seed": cfg.seed, "seeds_registered": cfg.seeds, "gradient_checkpointing": cfg.gradient_checkpointing,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "time_s": {"load_and_prescan": round(load_s, 1), "train": round(train_s, 1),
                          "per_step": round(train_s / max(1, step), 2), "per_row": round(train_s / max(1, rows_seen), 2)},
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
               "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else None,
               "data": export_provenance(export_dir, files), "runtime": runtime_info()}
    if loss_checks:
        summary["loss_check"] = {"rows": len(loss_checks),
                                 "max_abs_diff": max(abs(c["kept"] - c["full"]) for c in loss_checks), "samples": loss_checks[:5]}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if smoke_tiny:                                      # 极小模型：加载回来做一次前向，证明 adapter 可复用
        from peft import PeftModel
        reloaded = PeftModel.from_pretrained(tiny_model(cfg), out_dir / "final")
        reloaded.eval()
        with torch.no_grad():
            ids = processor.tokenizer("冒烟：加载 LoRA 后前向", return_tensors="pt")["input_ids"]
            logits = reloaded(input_ids=ids).logits
        summary["reload_forward_ok"] = bool(logits.ndim == 3)
        (out_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("history", "selected", "data")}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-tiny", action="store_true", help="同架构极小随机模型，只验证链路")
    parser.add_argument("--check-loss", action="store_true", help="逐行对比只算监督段 logits 与整段 logits 的损失（小模型用）")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export-dir", default=None, help="覆盖配置里的 export_dir（例如 r0 数据包目录）")
    parser.add_argument("--model", default=None, help="覆盖配置里的 model（服务器上权重目录不同时用）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置里的 seed（同一配置训多个种子时用）")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="只预扫数据、算步数，不加载模型")
    parser.add_argument("--allow-trial-pack", action="store_true", help="允许用 status!=ok 的 r0 试包训练（只用于调试）")
    parser.add_argument("--save-every", type=int, default=None, help="每 N 次更新存一个可续训的 checkpoint-<N>/（覆盖配置的 save_every）")
    parser.add_argument("--resume", action="store_true", help="从 output_dir 里最近一个完整的 checkpoint-* 接着训（没有就从头训）")
    parser.add_argument("--stop-at-step", type=int, default=None, help="试跑续训：存完第 N 步之后的 checkpoint 就停（不写 final）")
    args = parser.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    for name in ("seed", "output_dir", "export_dir", "model", "max_steps", "save_every"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.dry_run:
        plan = dry_run(cfg)
        return 0 if plan["usable"] else 1
    if args.stop_at_step and not cfg.save_every:
        raise SystemExit("--stop-at-step 要配 --save-every（停在一个存过的 checkpoint 上）")
    train(cfg, smoke_tiny=args.smoke_tiny, check_loss=args.check_loss, allow_trial_pack=args.allow_trial_pack,
          resume=args.resume, stop_at_step=args.stop_at_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
