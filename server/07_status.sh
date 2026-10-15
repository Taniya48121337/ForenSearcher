#!/usr/bin/env bash
# Any time, from another terminal: how far the baselines are.
source "$(dirname "$0")/env.sh"
cd "$FS_ROOT"
"$FS_ENV_AGENT/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, "server")
import run_baselines as R
up = R.vllm_up()
print(f"vLLM {R.ENDPOINT}: {'up' if up else 'DOWN'}")
for name, arm in R.ARMS.items():
    try:
        ids = R.evid_ids() if arm["dataset"] == "evid" else R.bench_ids(arm["dataset"])
    except Exception as exc:
        print(f"{name:12s} ? ({exc})"); continue
    if not ids:
        print(f"{name:12s} no samples"); continue
    done = sum(R.is_done(R.run_dir(arm["dataset"], i, arm["round"])) for i in ids)
    started = sum(R.run_dir(arm["dataset"], i, arm["round"]).is_dir() for i in ids)
    print(f"{name:12s} round {arm['round']:8s} done {done:4d}/{len(ids):<4d} started {started}")
PY
echo; tail -3 "$FS_LOG_DIR/06_run_baselines.log" 2>/dev/null
echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
