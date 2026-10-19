#!/usr/bin/env bash
# E2: sft0 evaluation (eval_config sft0_eval_v1) on the single H200 — training-free baseline, then after SFT. Run it in tmux.
#   bash server/21_e2_eval.sh baseline                     preflight → check → seed --yes → check → run (2 seeds at once) → lock → score → metrics
#   HI_MODEL=hi_sft0_s1 GIA_MODEL=gia_sft0_s1 bash server/21_e2_eval.sh after_sft
#                                                          preflight → check → run (2 seeds at once) → score → metrics (vs r0base.json)
#   bash server/21_e2_eval.sh baseline run score           only these steps (preflight always runs first)
# Settings (environment): E2_JOBS (default FS_JOBS = 6 in the vast1 profile: runs in parallel per seed), E2_SEEDS ("1 2"),
#   E2_PARALLEL_SEEDS (1 = both seeds at the same time, 0 = one after the other), HI_MODEL / GIA_MODEL (after_sft LoRA names
#   from server/12_add_lora.sh, default *_sft0_s1). Judges: Claude CLI subscription (claude-cli:sonnet; `claude` logged in).
# Logs: server_logs/e2_<phase>_<step>.log (run: one log per seed, e2_<phase>_run_s<seed>.log). Every step stops the chain on failure,
#   except score (e.g. the Claude CLI hit its usage limit: metrics then runs without scores; rerun score, its cache resumes).
# Needs: FS_PROFILE=vast1 (default here), the student up (FS_PROFILE=vast1 bash server/11_loop_layout.sh serve), the evaluation
#   world in place (experiments/eval/r0/*, experiments/evid/{cases,search_snapshots}, EVID GT, the 4 dev_eval cases + media).
#
# Concurrency (checked 10-19 in the code, not measured):
# - eval_r0.py run → one run_cases call per set (evid_ctrl20: 20 runs, then dev_eval4: 4 runs); run_cases runs ThreadPoolExecutor(
#   max_workers=--jobs) with one subprocess per run (case × sample), so at most min(jobs, runs in the set) processes per seed, each
#   with its own tool models on GPU 0. Two seeds at once: up to 12 processes.
# - The two seeds use different round names (r0base-s1 / r0base-s2): run dirs, run_cases logs and manifests (per round) differ.
# - Both seeds replay / record the same snapshot directory per case: index.jsonl appends hold an flock, blobs are written to a temp
#   file and os.replace'd, readers only take complete lines and re-read on a miss; the usage file is per run name. Two misses on
#   the same request at the same moment both go live and both append; the later line wins. Sequential seeds share the directory
#   too (seed 2 replays what seed 1 recorded), so running them together does not change what is shared, only its timing.
# - eval_r0.py's own files carry a seconds timestamp and no seed (check_<time>.json; run_<tag>_<time>.log, opened in append mode),
#   and every `run` starts with a `check` that reads the snapshot indexes: seed 2 is therefore started only after seed 1 has passed
#   its check and started its first set (log line "== 种子"), at least 5 s later.
set -uo pipefail
export FS_PROFILE="${FS_PROFILE:-vast1}"
source "$(dirname "$0")/loop_env.sh"
set -o pipefail
cd "$FS_ROOT" || exit 1
PY="$FS_ENV_AGENT/bin/python"
export PYTHONPATH="$FS_ROOT/src" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 NO_PROXY="127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES="$FS_TOOL_GPUS"           # agent processes (tools) see only the student's GPU
# 10-19 thread caps (eval_config harness.thread_caps; speed only, same values before and after SFT): 12 concurrent run_cases
# processes otherwise each open full-width onnxruntime / torch / BLAS pools on 128 cores. rapidocr's onnxruntime threads are set
# in its installed config.yaml (intra 2 / inter 1, see server/requirements.lock.md). Per-run 30 min wall limit: server/e2_watchdog.sh.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

PHASE=${1:-}; shift || true
case "$PHASE" in baseline|after_sft) ;; *) echo "usage: $0 baseline|after_sft [preflight check seed run lock score metrics ...]"; exit 2 ;; esac
if [ $# -gt 0 ]; then STEPS="$*"
elif [ "$PHASE" = baseline ]; then STEPS="check seed check run lock score metrics"
else STEPS="check run score metrics"; fi
JOBS=${E2_JOBS:-$FS_JOBS}
SEEDS=${E2_SEEDS:-1 2}
ENDPOINT=$FS_STUDENT_ENDPOINT
if [ "$PHASE" = baseline ]; then MODEL=$FS_STUDENT_NAME; NAMES=("$FS_STUDENT_NAME")
else
  HI_MODEL=${HI_MODEL:-hi_sft0_s1}; GIA_MODEL=${GIA_MODEL:-gia_sft0_s1}
  MODEL="HI=$HI_MODEL,GIA=$GIA_MODEL"; NAMES=("$FS_STUDENT_NAME" "$HI_MODEL" "$GIA_MODEL")
fi
EVAL=(scripts/eval_r0.py)
logf() { echo "$FS_LOG_DIR/e2_${PHASE}_$1.log"; }
banner() { printf '\n===== %s  %s %s =====\n' "$(date '+%F %T')" "$PHASE" "$1"; }

chat_check() {  # model -> 0 if a thinking-off, temperature-0.6 request comes back with content and no reasoning text
  local body resp
  body=$(printf '{"model": "%s", "messages": [{"role": "user", "content": "Reply with exactly one word: ready"}], "temperature": 0.6, "max_tokens": 16, "chat_template_kwargs": {"enable_thinking": false}}' "$1")
  resp=$(curl -s -m 180 -H 'Content-Type: application/json' -d "$body" "$ENDPOINT/chat/completions" 2>&1)
  "$PY" - "$1" "$resp" <<'PY'
import json, sys
name, raw = sys.argv[1], sys.argv[2]
try:
    message = json.loads(raw)["choices"][0]["message"]
except Exception:
    print(f"  FAIL {name}: no completion: {raw[:200]}")
    sys.exit(1)
text = (message.get("content") or "").strip()
reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
bad = bool(reasoning) or "<think>" in text or not text
print(f"  {'FAIL' if bad else 'ok  '} {name}: thinking off -> content {text[:30]!r}, reasoning {len(reasoning)} chars")
sys.exit(1 if bad else 0)
PY
}

preflight() {
  local rc=0 name temp
  say "preflight: $PHASE, endpoint $ENDPOINT, model $MODEL, jobs $JOBS per seed, seeds $SEEDS (parallel=${E2_PARALLEL_SEEDS:-1})"
  for name in "${NAMES[@]}"; do
    if served_on "$ENDPOINT" "$name"; then ok "served: $name"; else fail "not served on $ENDPOINT: $name (11_loop_layout.sh serve / 12_add_lora.sh)"; rc=1; fi
  done
  [ $rc -eq 0 ] || return 1
  for name in "${NAMES[@]}"; do chat_check "$name" || rc=1; done
  # temperature: EndpointTeacher's default is what run_cases uses (it has no flag); it must equal eval_config sampling.temperature
  temp=$("$PY" - <<'PY'
import inspect, json
from foren_searcher.datagen.teacher import EndpointTeacher
default = inspect.signature(EndpointTeacher.__init__).parameters["temperature"].default
cfg = json.load(open("experiments/eval/r0/eval_config.json", encoding="utf-8"))
print(f"{default} {cfg['sampling']['temperature']} {cfg['harness']['no_think']} {cfg['eval_id']}")
PY
)
  read -r t_code t_cfg no_think eval_id <<< "$temp"
  if [ "$t_code" = "$t_cfg" ] && [ "$t_cfg" = 0.6 ]; then ok "temperature $t_code (EndpointTeacher default = eval_config $eval_id)"
  else fail "temperature: EndpointTeacher default $t_code, eval_config $t_cfg (must both be 0.6)"; rc=1; fi
  # the planned run_cases commands: --no-think, frozen specs, describer fixed to the base model
  local plan; plan=$("$PY" "${EVAL[@]}" plan --phase "$PHASE" --endpoint "$ENDPOINT" --model "$MODEL" --jobs "$JOBS" 2>&1)
  if [ "$no_think" = True ] && grep -q -- "--no-think" <<< "$plan"; then ok "run_cases gets --no-think (thinking off)"; else fail "plan has no --no-think"; rc=1; fi
  grep -q "FOREN_DESCRIBE_MODEL=$FS_STUDENT_NAME" <<< "$plan" && ok "describer fixed: $FS_STUDENT_NAME @ $ENDPOINT" || { fail "plan does not fix the describer"; rc=1; }
  grep -q -- "--hi-spec hi@0.11.0" <<< "$plan" && ok "specs hi@0.11.0 / gia@0.9.0" || { fail "plan is not on hi@0.11.0"; rc=1; }
  if [ "$PHASE" = after_sft ] && [ ! -f experiments/eval/r0/search_snapshots/snapshot_lock.json ]; then
    fail "after_sft needs the baseline snapshot lock (bash $0 baseline lock)"; rc=1
  fi
  echo "  GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"
  return $rc
}

run_seeds() {
  local seed pids=() logs=() rc=0 i
  for seed in $SEEDS; do
    local log; log=$(logf "run_s$seed"); logs+=("$log")
    banner "run seed $seed" >> "$log"
    "$PY" "${EVAL[@]}" run --phase "$PHASE" --endpoint "$ENDPOINT" --model "$MODEL" --jobs "$JOBS" --seeds "$seed" >> "$log" 2>&1 &
    pids+=($!)
    say "seed $seed started (pid $!, log $log)"
    if [ "${E2_PARALLEL_SEEDS:-1}" = 1 ]; then
      sleep 5
      for _ in $(seq 1 360); do            # wait until this seed passed its check and started its first set
        grep -q "== 种子" "$log" 2>/dev/null && break
        kill -0 "${pids[-1]}" 2>/dev/null || break
        sleep 5
      done
    else
      wait "${pids[-1]}" || rc=1
    fi
  done
  say "waiting for ${#pids[@]} seed process(es); follow: tail -f ${logs[*]}"
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { rc=1; fail "seed process ${pids[$i]} exited non-zero: ${logs[$i]}"; }; done
  "$PY" - "$PHASE" $SEEDS <<'PY'
import json, sys
from pathlib import Path
phase, seeds = sys.argv[1], sys.argv[2:]
cfg = json.load(open("experiments/eval/r0/eval_config.json", encoding="utf-8"))
tag = cfg["round_tags"][phase]
cases = sum(len(s["cases"]) for s in json.load(open(cfg["eval_set"], encoding="utf-8"))["sets"].values())
for seed in seeds:
    runs = [p for p in Path(cfg["runs_dir"]).glob(f"*_{tag}-s{seed}_1") if p.is_dir()]
    done = [p for p in runs if (p / "result.json").is_file()]
    print(f"  {tag}-s{seed}: {len(done)}/{cases} runs with result.json" + ("" if len(done) == cases else
          "  <- incomplete: rerun this step (run_cases --retry-failed resumes)"))
PY
  return $rc
}

step() {  # name -> run one step with its log
  local name=$1 log; log=$(logf "$name")
  case "$name" in
    preflight) banner preflight >> "$log"; preflight 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}" ;;
    check)     banner check >> "$log"; "$PY" "${EVAL[@]}" check 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}" ;;
    seed)      [ "$PHASE" = baseline ] || { say "seed: baseline only, skipped"; return 0; }
               banner seed >> "$log"; "$PY" "${EVAL[@]}" seed --yes 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}" ;;
    run)       run_seeds ;;
    lock)      [ "$PHASE" = baseline ] || { say "lock: baseline only, skipped"; return 0; }
               banner lock >> "$log"; "$PY" "${EVAL[@]}" lock 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}" ;;
    score)     banner score >> "$log"
               "$PY" "${EVAL[@]}" score --phase "$PHASE" 2>&1 | tee -a "$log"
               return "${PIPESTATUS[0]}" ;;
    metrics)   banner metrics >> "$log"; "$PY" "${EVAL[@]}" metrics --phase "$PHASE" 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}" ;;
    *)         fail "unknown step $name"; return 2 ;;
  esac
}

say "E2 $PHASE: steps $STEPS; logs $FS_LOG_DIR/e2_${PHASE}_*.log"
step preflight || { fail "preflight failed: nothing started"; exit 1; }
for s in $STEPS; do
  [ "$s" = preflight ] && continue
  say "step $s"
  if step "$s"; then ok "step $s done"
  elif [ "$s" = score ]; then warn "score did not run or failed (claude CLI logged in? usage limit?): metrics without scores"
  else fail "step $s failed: stopping (log $(logf "$s"))"; exit 1; fi
done
ok "E2 $PHASE finished: experiments/eval/r0/reports/$( [ "$PHASE" = baseline ] && echo r0base || echo r0sft ).{json,md}"
