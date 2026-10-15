#!/usr/bin/env bash
# walker trajectory production on 8x H20 96GB (2026-10-14 night). One script for the whole production loop:
#
#   bash server/16_walker_prod.sh up [397b|122b|27b]    start the teacher with the H20 layout below (default 397b: wk14 beat 122B
#                                                       on every column; 122B as verifier agreed with Sonnet at kappa ~0)
#   bash server/16_walker_prod.sh status | down
#   bash server/16_walker_prod.sh cases ws1 [walker_d2_cases.py flags]   build cases at scale, list -> walker_ws1_cases.json
#        e.g.  bash server/16_walker_prod.sh cases ws1 --pool all --n-a 60 --n-c-fake 80 --n-c-real 40 --n-e 60 --n-g360-fake 60 --n-g360-real 30
#   bash server/16_walker_prod.sh smoke                 one given-media case end to end, no verification (2-5 minutes)
#   bash server/16_walker_prod.sh run ws1 [walker_run.py d1 flags]       trajectories only (--no-verify), cases from
#                                                       walker_ws1_cases.json, FS_PROD_JOBS in parallel. Verification is frozen to
#                                                       Sonnet via Claude CLI (specs/r0_freeze.json verifier), which the server does
#                                                       not have: bundle the runs and verify + rewrite + export on the local machine.
#                                                       FS_PROD_VERIFIER_ENDPOINT=<url> verifies on the server instead (not r0-valid)
#   bash server/16_walker_prod.sh bundle ws1            tar the round's runs + cases + manifest -> walker_ws1_bundle.tgz to copy home
#   bash server/16_walker_prod.sh pack p1 ws1 ws2 ...   merge the accepted rows of those rounds into experiments/sft_data/sft_pack/p1
#                                                       (training/sft_lora.py reads that directory as export_dir)
#
# Layouts (weights: Hugging Face repos, downloaded to $FS_MODEL_DIR by `up` if missing):
#   122b  Qwen3.5-122B-A10B-FP8   GPU 0-3 tensor parallel 4 (~125 GB weights, ~220 GB KV)   tools GPU 6-7   GPU 4-5 free
#   397b  Qwen3.5-397B-A17B-FP8   GPU 0-7 tensor parallel 8, 80% memory (~400 GB weights)  tools share GPU 7 (~18 GB left)
#   27b   Qwen3.8-27B (bf16)      GPU 0-5 data parallel 6                                    tools GPU 6-7
# H20 is Hopper: FP8 runs natively (no Marlin fallback as on A100).
# Network: the agent's search tools need the internet; set FS_PROXY in server/env.sh if the server needs a proxy.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$FS_ROOT"
VLLM="$FS_ENV_VLLM/bin/vllm"
PY="$FS_ENV_AGENT/bin/python"
PORT="${FS_PROD_PORT:-8000}"
ENDPOINT="http://127.0.0.1:$PORT/v1"
NAME=teacher
PIDF="$FS_LOG_DIR/prod_teacher.pid"
JOBS="${FS_PROD_JOBS:-8}"

preset() {  # 122b|397b|27b -> HF, GPUS, ARGS, TOOLS
  case "$1" in
    122b) HF="${FS_PROD_HF:-Qwen/Qwen3.5-122B-A10B-FP8}"; GPUS="${FS_PROD_GPUS:-0,1,2,3}"; TOOLS="${FS_PROD_TOOL_GPUS:-6,7}"
          ARGS=(--tensor-parallel-size 4 --gpu-memory-utilization 0.90 --max-num-seqs 48) ;;
    397b) HF="${FS_PROD_HF:-Qwen/Qwen3.5-397B-A17B-FP8}"; GPUS="${FS_PROD_GPUS:-0,1,2,3,4,5,6,7}"; TOOLS="${FS_PROD_TOOL_GPUS:-7}"
          ARGS=(--tensor-parallel-size 8 --gpu-memory-utilization 0.80 --max-num-seqs 32) ;;
    27b)  HF="${FS_PROD_HF:-Qwen/Qwen3.8-27B}"; GPUS="${FS_PROD_GPUS:-0,1,2,3,4,5}"; TOOLS="${FS_PROD_TOOL_GPUS:-6,7}"
          ARGS=(--data-parallel-size 6 --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-num-seqs 32) ;;
    *) fail "unknown preset '$1' (122b | 397b | 27b)"; exit 2 ;;
  esac
}

up_now() { curl -s -m 5 "$ENDPOINT/models" 2>/dev/null | grep -q "\"$NAME\""; }

cmd_up() {
  local which=${1:-397b}; preset "$which"
  if up_now; then ok "teacher already up at $ENDPOINT"; return 0; fi
  local dir="$FS_MODEL_DIR/$(basename "$HF")"
  if [ ! -f "$dir/config.json" ]; then
    say "downloading $HF to $dir (once)"
    "$FS_ENV_AGENT/bin/huggingface-cli" download "$HF" --local-dir "$dir" || { fail "download failed (FS_HF_ENDPOINT / FS_PROXY in server/env.sh?)"; exit 1; }
  fi
  echo "$which" > "$FS_LOG_DIR/prod_preset"
  say "starting $(basename "$dir") on GPU $GPUS (port $PORT): ${ARGS[*]}"
  CUDA_VISIBLE_DEVICES="$GPUS" NO_PROXY="*" nohup "$VLLM" serve "$dir" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
    --trust-remote-code --max-model-len "${FS_PROD_MAX_LEN:-49152}" --limit-mm-per-prompt '{"image": 32, "video": 0}' \
    --reasoning-parser qwen3 "${ARGS[@]}" > "$FS_LOG_DIR/prod_teacher.log" 2>&1 &
  echo $! > "$PIDF"
  for i in $(seq 1 240); do
    sleep 10
    kill -0 "$(cat "$PIDF")" 2>/dev/null || { fail "vLLM exited; last lines:"; tail -40 "$FS_LOG_DIR/prod_teacher.log"; exit 1; }
    if up_now; then ok "teacher up after $((i * 10)) s"; return 0; fi
    [ $((i % 6)) -eq 0 ] && say "loading ($((i * 10)) s) $(grep -Eo 'Loading.*|Capturing.*' "$FS_LOG_DIR/prod_teacher.log" | tail -1 | cut -c1-80)"
  done
  fail "not up after 40 minutes; see $FS_LOG_DIR/prod_teacher.log"; exit 1
}

cmd_down() {
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then kill "$(cat "$PIDF")"; sleep 10; kill -9 "$(cat "$PIDF")" 2>/dev/null; ok "teacher stopped"; fi
  rm -f "$PIDF"
}

agent_env() {
  local which; which=$(cat "$FS_LOG_DIR/prod_preset" 2>/dev/null || echo 397b); preset "$which"
  up_now || { fail "teacher not up at $ENDPOINT (bash server/16_walker_prod.sh up $which)"; exit 1; }
  export FOREN_ORACLE_SLOT_ENDPOINT="$ENDPOINT" FOREN_ORACLE_SLOT_MODEL="$NAME" FOREN_ORACLE_SLOT_THINKING="${FOREN_ORACLE_SLOT_THINKING:-1}"
  export FOREN_ORACLE_SLOT_ATTEMPTS="${FOREN_ORACLE_SLOT_ATTEMPTS:-5}"
  export FOREN_WALKER_JUDGE=local FOREN_JUDGE_ENDPOINT="${FS_PROD_VERIFIER_ENDPOINT:-$ENDPOINT}"
  export FOREN_WALKER_JUDGE_MODEL="${FS_PROD_VERIFIER_NAME:-$NAME}" FOREN_JUDGE_CONCURRENCY="${FOREN_JUDGE_CONCURRENCY:-16}"
  export CUDA_VISIBLE_DEVICES="$TOOLS"
  say "teacher $NAME @ $ENDPOINT ($which); verifier $FOREN_WALKER_JUDGE_MODEL @ $FOREN_JUDGE_ENDPOINT; tools on GPU $TOOLS; jobs $JOBS"
}

case "${1:-status}" in
  up) cmd_up "${2:-397b}" ;;
  down) cmd_down ;;
  status) up_now && ok "teacher up at $ENDPOINT ($(cat "$FS_LOG_DIR/prod_preset" 2>/dev/null))" || warn "teacher down"
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /' ;;
  cases) round=${2:?round}; shift 2
         "$PY" scripts/walker_d2_cases.py --list "walker_${round}_cases.json" "$@" ;;
  smoke) agent_env
         "$PY" scripts/walker_run.py d1 --round prodsmoke --cases videosham_4126 --no-verify ;;
  run) round=${2:?round}; shift 2; agent_env
       list="experiments/sft_data/walker_${round}_cases.json"
       extra=(); [ -f "$list" ] && extra=(--case-list "walker_${round}_cases.json")
       if [ -z "${FS_PROD_VERIFIER_ENDPOINT:-}" ]; then
         extra+=(--no-verify)
         say "trajectories only; verify on the local machine (Sonnet): bash server/16_walker_prod.sh bundle $round, then see README step 5"
       fi
       "$PY" scripts/walker_run.py d1 --round "$round" "${extra[@]}" --jobs "$JOBS" "$@" ;;
  bundle) round=${2:?round}
          out="walker_${round}_bundle.tgz"
          tar czf "$out" experiments/sft_data/runs/*_"${round}"_1 experiments/sft_data/cases \
            $(ls experiments/sft_data/walker_"${round}"_cases.json experiments/sft_data/run_manifests/"${round}".jsonl 2>/dev/null) \
            && ok "$out ($(du -h "$out" | cut -f1)); unpack in the local ForenSearcher root" ;;
  pack) name=${2:?name}; shift 2
        exports=(); for r in "$@"; do exports+=("experiments/sft_data/export/${r}_*"); done
        "$PY" scripts/pack_sft.py --exports "${exports[@]}" --out "experiments/sft_data/sft_pack/$name" ;;
  *) echo "usage: $0 up [397b|122b|27b] | down | status | cases ROUND [flags] | smoke | run ROUND [flags] | bundle ROUND | pack NAME ROUND..."; exit 2 ;;
esac
