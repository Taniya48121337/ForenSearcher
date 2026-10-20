#!/usr/bin/env bash
# 新机器准入：10 分钟烧机（《训练与评测提速方案_20261020》六节 G0；负责人 10-20 定：迁 4×H200 时只收通过的机器）。
# 每张可见 GPU 各起一个 bf16 16384² 矩阵乘循环，同时每 2 s 采 SM 时钟 / 温度 / 功耗，前后各读一次 PERFORMANCE 计数器。
# 10-20 旧机器（Vast H200 NVL）训练中实测：SW Thermal Slowdown 90% 时间、SM 均值 858 MHz、86–91 °C —— 这台会不合格。
#   bash server/29_burn_in.sh            # 默认 600 s
#   bash server/29_burn_in.sh 120        # 试跑
# 合格线（每张卡都要过；可用环境变量改）：前 60 s 预热之后 SM 均值 ≥ MIN_SM_MHZ(1400)、最高温度 ≤ MAX_TEMP_C(83)、
# 全程 SW + HW Thermal Slowdown 计数器增量 < MAX_THERMAL_FRAC(0.05) 的时间。TFLOPS 只报告不判（H200 NVL 正常应在 550 以上）。
# 结果：$BURN_OUT/summary.json；退出码 0 = 全部合格，1 = 有卡不合格，2 = 没跑（GPU 被占 / 无 GPU）。
set -uo pipefail
SECS=${1:-600}
PY=${PY:-/venv/main/bin/python}
OUT=${BURN_OUT:-/workspace/logs/burn_in_$(date +%Y%m%d_%H%M%S)}
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi"; exit 2; }
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 > 2000 {printf "%s(%s MiB) ", $1, $2}')
[ -n "$busy" ] && { echo "GPU in use: ${busy}— stop training / vLLM first"; exit 2; }
mkdir -p "$OUT"

counters() {   # 每行：GPU 序号 SW热降频us HW热降频us SW功耗封顶us（按 nvidia-smi -q 里 GPU 出现的顺序）
  nvidia-smi -q -d PERFORMANCE | awk '/^GPU [0-9A-Fa-f]+:/{g++}
    /SW Thermal Slowdown +: [0-9]/{st[g]=$5} /HW Thermal Slowdown +: [0-9]/{ht[g]=$5} /SW Power Capping +: [0-9]/{pc[g]=$5}
    END{for(i=1;i<=g;i++) print i-1, st[i]+0, ht[i]+0, pc[i]+0}'
}

counters > "$OUT/counters_before.txt"
start=$(date +%s)
pids=()
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  CUDA_VISIBLE_DEVICES=$i "$PY" - "$SECS" > "$OUT/gemm_$i.txt" 2>&1 <<'EOF' &
import sys, time, torch
secs = float(sys.argv[1]); n = 16384
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
for _ in range(5): a @ b
torch.cuda.synchronize(); t = time.time(); k = 0
while time.time() - t < secs:
    for _ in range(20): a @ b
    torch.cuda.synchronize(); k += 20
print(f"tflops {k * 2 * n ** 3 / (time.time() - t) / 1e12:.1f}")
EOF
  pids+=($!)
done
echo "burn-in ${SECS}s on ${#pids[@]} GPU(s) -> $OUT"
: > "$OUT/samples.csv"
while [ $(( $(date +%s) - start )) -lt "$SECS" ]; do
  t=$(( $(date +%s) - start ))
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits | sed "s/^/$t, /" >> "$OUT/samples.csv"
  sleep 2
done
wait "${pids[@]}"
end=$(date +%s)
counters > "$OUT/counters_after.txt"
nvidia-smi --query-gpu=index,name,power.limit,clocks.max.sm --format=csv,noheader > "$OUT/gpus.csv"

"$PY" - "$OUT" "$(( end - start ))" <<'EOF'
import json, os, re, statistics as st, sys
out, dur = sys.argv[1], float(sys.argv[2])
min_sm, max_temp, max_th = float(os.environ.get("MIN_SM_MHZ", 1400)), float(os.environ.get("MAX_TEMP_C", 83)), float(os.environ.get("MAX_THERMAL_FRAC", 0.05))
def counters(name):
    rows = [l.split() for l in open(f"{out}/{name}") if l.strip()]
    return {int(r[0]): [int(x) for x in r[1:]] for r in rows}
before, after = counters("counters_before.txt"), counters("counters_after.txt")
names = {int(l.split(",")[0]): l.split(",")[1].strip() for l in open(f"{out}/gpus.csv") if l.strip()}
samples = {}
for line in open(f"{out}/samples.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 5 and all(re.fullmatch(r"[0-9.]+", x) for x in p):
        samples.setdefault(int(p[1]), []).append((float(p[0]), float(p[2]), float(p[3]), float(p[4])))
result, ok_all = {}, True
for g in sorted(samples):
    s = [x for x in samples[g] if x[0] >= 60] or samples[g]
    clk, temp, pw = [x[1] for x in s], [x[2] for x in samples[g]], [x[3] for x in s]
    b, a = before.get(g, [0, 0, 0]), after.get(g, [0, 0, 0])
    thermal = ((a[0] - b[0]) + (a[1] - b[1])) / 1e6 / dur
    m = re.search(r"tflops ([0-9.]+)", open(f"{out}/gemm_{g}.txt").read())
    checks = {"sm_mean_ok": st.mean(clk) >= min_sm, "temp_max_ok": max(temp) <= max_temp, "thermal_frac_ok": thermal < max_th, "gemm_ran": bool(m)}
    ok = all(checks.values()); ok_all &= ok
    result[g] = {"name": names.get(g), "pass": ok, **checks, "sm_mhz_mean": round(st.mean(clk)), "sm_mhz_p10": sorted(clk)[len(clk) // 10],
                 "temp_c_max": max(temp), "temp_c_mean": round(st.mean(temp), 1), "power_w_mean": round(st.mean(pw)),
                 "thermal_slowdown_frac": round(thermal, 3), "power_cap_frac": round((a[2] - b[2]) / 1e6 / dur, 3),
                 "tflops": float(m.group(1)) if m else None}
summary = {"pass": ok_all and bool(result), "seconds": dur, "thresholds": {"min_sm_mhz": min_sm, "max_temp_c": max_temp, "max_thermal_frac": max_th}, "gpus": result}
open(f"{out}/summary.json", "w").write(json.dumps(summary, indent=1))
for g, r in result.items():
    print(f"GPU {g} {r['name']}: {'PASS' if r['pass'] else 'FAIL'}  SM {r['sm_mhz_mean']} MHz (p10 {r['sm_mhz_p10']})  "
          f"temp max {r['temp_c_max']:.0f} C  thermal {r['thermal_slowdown_frac']:.0%}  power-cap {r['power_cap_frac']:.0%}  {r['tflops']} TFLOPS")
print("OVERALL", "PASS" if summary["pass"] else "FAIL", "->", f"{out}/summary.json")
sys.exit(0 if summary["pass"] else 1)
EOF
