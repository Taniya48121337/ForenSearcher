#!/usr/bin/env bash
# 训练显存 / 速度探针（单卡 Vast，2026-10-19）：最长 20 行上跑四个变体 × HI / GIA，每个记稳态每步耗时（去掉第 1 步）、
# torch 峰值显存（allocated / reserved）、nvidia-smi 峰值、是否 OOM、线性注意力内核实况。要先停 vLLM（GPU 空）。
#   bash server/25_probe_run.sh [rows.jsonl]        # rows 默认 /workspace/smoke/check_inputs_1019b.rows.jsonl
#   PROBE_VARIANTS="gc_fa2_liger gc_fa2" PROBE_OUT=/workspace/smoke/long20b bash server/25_probe_run.sh   # 只跑部分变体、另放一处
# 结果：/workspace/smoke/long20/probe_results.json（+ 每个任务的日志）。选择规则（负责人 10-19）：最快且峰值 < 125 GB。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROWS=${1:-/workspace/smoke/check_inputs_1019b.rows.jsonl}
OUT=${PROBE_OUT:-/workspace/smoke/long20}
PY=/venv/main/bin/python
export PYTHONPATH=src PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
[ "${used:-0}" -gt 2000 ] && { echo "GPU 0 uses ${used} MiB: stop vLLM first"; exit 1; }
mkdir -p "$OUT"
$PY server/22_mem_probe.py --pack dist/sft0-20261019-4b79e1ed432f --rows "$ROWS" --out "$OUT" --n 20 > "$OUT/probe_rows.out" || { cat "$OUT/probe_rows.out"; exit 1; }
: > "$OUT/probe_runs.jsonl"
for variant in ${PROBE_VARIANTS:-base gc_fa2_liger gc_fa2 nogc_fa2_liger nogc}; do
  for role in hi gia; do
    cfg="$OUT/${role}_long20_${variant}.json"; log="$OUT/${role}_${variant}.log"; smi="$OUT/${role}_${variant}.smi"
    rm -rf "$OUT/${role}_${variant}_adapter"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 1000 > "$smi" 2>/dev/null & smipid=$!
    start=$(date +%s)
    timeout 1800 $PY training/sft_lora.py --config "$cfg" --seed 1 > "$log" 2>&1; rc=$?
    kill $smipid 2>/dev/null; wait $smipid 2>/dev/null
    $PY - "$variant" "$role" "$rc" "$log" "$smi" "$OUT/${role}_${variant}_adapter/train_summary.json" "$(( $(date +%s) - start ))" >> "$OUT/probe_runs.jsonl" <<'PY'
import json, sys, re
from pathlib import Path
variant, role, rc, log, smi, summary, wall = sys.argv[1:8]
text = Path(log).read_text(encoding="utf-8", errors="replace")
peak_smi = max((int(x) for x in Path(smi).read_text().split() if x.isdigit()), default=0)
rec = {"variant": variant, "role": role, "exit": int(rc), "wall_s": int(wall), "oom": "OutOfMemoryError" in text or "CUDA out of memory" in text,
       "nvidia_smi_peak_gb": round(peak_smi / 1024, 1)}
kern = re.search(r'\{"linear_attn_kernels": (\{.*\})\}', text)
if kern:
    k = json.loads(kern.group(1)); rec["linear_attn_fused"] = k.get("fused"); rec["linear_attn_impl"] = k.get("impl")
rec["fast_path_warning"] = "fast path is not available" in text
if Path(summary).is_file():
    s = json.loads(Path(summary).read_text(encoding="utf-8"))
    h = [x for x in s.get("history", []) if "elapsed_s" in x]
    rec.update({"steps": s.get("steps"), "rows_seen": s.get("rows_seen"), "peak_alloc_gb": s.get("peak_gpu_mem_gb"),
                "peak_reserved_gb": s.get("peak_gpu_reserved_gb"), "train_s": s["time_s"]["train"], "load_s": s["time_s"]["load_and_prescan"],
                "attn": s["hyper"].get("attn_implementation"), "liger": s["hyper"].get("liger"),
                "gradient_checkpointing": s.get("gradient_checkpointing"),
                "first_step_s": h[0]["elapsed_s"] if h else None,
                "steady_step_s": round((h[-1]["elapsed_s"] - h[0]["elapsed_s"]) / max(1, len(h) - 1), 1) if len(h) > 1 else None,
                "tokens_max": s["prescan"]["tokens"].get("max"), "losses": [x.get("loss_weighted") for x in h]})
else:
    rec["error_tail"] = text[-600:]
print(json.dumps(rec, ensure_ascii=False))
PY
    tail -1 "$OUT/probe_runs.jsonl"
    rm -rf "$OUT/${role}_${variant}_adapter/final" "$OUT/${role}_${variant}_adapter"/epoch_*
    if grep -q '"oom": true' <(tail -1 "$OUT/probe_runs.jsonl"); then echo "== $variant OOM on $role: skip the rest of $variant"; break; fi
  done
done
$PY - "$OUT/probe_runs.jsonl" > "$OUT/probe_results.json" <<'PY'
import json, sys
runs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
ok = {}
for r in runs:
    ok.setdefault(r["variant"], {})[r["role"]] = r
cands = []
for v, roles in ok.items():
    finite = lambda x: bool(x.get("losses")) and all(isinstance(v, (int, float)) and v == v for v in x["losses"])   # 10-19：fa2 变体 loss 全 nan
    if set(roles) == {"hi", "gia"} and all(not x["oom"] and x["exit"] == 0 and finite(x) for x in roles.values()):
        peak = max(max(x.get("peak_reserved_gb") or 0, x.get("nvidia_smi_peak_gb") or 0) for x in roles.values())
        if peak < 125:
            cands.append((roles["hi"].get("steady_step_s") or 1e9, v, peak))
cands.sort()
print(json.dumps({"runs": runs, "eligible": [{"variant": v, "hi_steady_step_s": s, "peak_gb": p} for s, v, p in cands],
                  "chosen": cands[0][1] if cands else None, "rule": "最快（HI 稳态每步）且两个角色都不 OOM、loss 全是有限数、峰值（reserved 与 nvidia-smi 取大）< 125 GB"},
                 ensure_ascii=False, indent=1))
PY
grep -E '"chosen"|"eligible"' -A3 "$OUT/probe_results.json" | head -12
