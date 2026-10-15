#!/usr/bin/env bash
# Switch the 8 GPUs between the two loop layouts (see loop_env.sh / README_自进化循环_GPU分配.md).
#   bash server/11_loop_layout.sh A                 rollout layout: teacher 0-3, student 4-5, verifier 6, tools (+verifier_ref) 7
#   bash server/11_loop_layout.sh B                 training layout: teacher stopped, GPUs 0-3 free for 13_train_queue.sh
#   bash server/11_loop_layout.sh status            services, LoRAs, GPU memory
#   bash server/11_loop_layout.sh stop [service]    stop teacher|student|verifier|vref (default: all four)
#   bash server/11_loop_layout.sh restart-student   restart the student with every LoRA in server_logs/loras.tsv
# Logs: server_logs/loop_<service>.log. Starting is idempotent: a service that already answers is left alone.
set -uo pipefail
source "$(dirname "$0")/loop_env.sh"
VLLM="$FS_ENV_VLLM/bin/vllm"
MM='{"image": 32, "video": 0}'

pidf() { echo "$FS_LOG_DIR/loop_$1.pid"; }
port_of() { case "$1" in teacher) echo "$FS_TEACHER_PORT";; student) echo "$FS_STUDENT_PORT";; verifier) echo "$FS_VERIFIER_PORT";; vref) echo "$FS_VREF_PORT";; esac; }
name_of() { case "$1" in teacher) echo "$FS_TEACHER_NAME";; student) echo "$FS_STUDENT_NAME";; verifier) echo "$FS_VERIFIER_NAME";; vref) echo "$FS_VREF_NAME";; esac; }
gpus_of() { case "$1" in teacher) echo "$FS_TEACHER_GPUS";; student) echo "$FS_STUDENT_GPUS";; verifier) echo "$FS_VERIFIER_GPUS";; vref) echo "$FS_VREF_GPUS";; esac; }
hf_of() { case "$1" in teacher) echo "$FS_TEACHER_HF";; student) echo "$FS_STUDENT_HF";; verifier) echo "$FS_VERIFIER_HF";; vref) echo "$FS_VREF_HF";; esac; }
endpoint_of() { echo "http://127.0.0.1:$(port_of "$1")/v1"; }
is_up() { served_on "$(endpoint_of "$1")" "$(name_of "$1")"; }

foreign_on_port() {  # something answers on the port but does not serve our name (e.g. the baseline vLLM on 8000)
  local ep; ep=$(endpoint_of "$1")
  curl -s -m 5 "$ep/models" >/dev/null 2>&1 && ! is_up "$1"
}

lora_args() {  # --lora-modules name=dir ... from the registry (only adapters that still exist)
  [ -s "$FS_LORA_REGISTRY" ] || return 0
  local mods=()
  while IFS=$'\t' read -r name dir; do
    [ -n "$name" ] && [ -f "$dir/adapter_config.json" ] && mods+=("$name=$dir")
  done < "$FS_LORA_REGISTRY"
  [ ${#mods[@]} -gt 0 ] && printf '%s\n' --lora-modules "${mods[@]}"
}

start() {  # service
  local svc=$1 hf dir gpus n args=()
  hf=$(hf_of "$svc")
  if [ -z "$hf" ]; then
    [ "$svc" = vref ] && { say "verifier_ref not configured (FS_VREF_HF empty): skipped"; return 0; }
    fail "$svc: no model configured (set FS_${svc^^}_HF in server/loop_env.sh)"; return 1
  fi
  if is_up "$svc"; then ok "$svc already up at $(endpoint_of "$svc") on GPU $(gpus_of "$svc")"; return 0; fi
  if foreign_on_port "$svc"; then
    fail "port $(port_of "$svc") is taken by another server (baseline vLLM?). Stop it first: bash server/04_stop_vllm.sh"; return 1
  fi
  dir="$FS_MODEL_DIR/$(basename "$hf")"
  [ -f "$dir/config.json" ] || { fail "$svc: weights not found at $dir (run bash server/10_loop_setup.sh)"; return 1; }
  gpus=$(gpus_of "$svc"); n=$(echo "$gpus" | tr ',' '\n' | grep -c .)
  args=(serve "$dir" --served-model-name "$(name_of "$svc")" --host 127.0.0.1 --port "$(port_of "$svc")" --trust-remote-code)
  case "$svc" in
    teacher)  args+=(--tensor-parallel-size "$n" --max-model-len "$FS_TEACHER_MAX_LEN" --gpu-memory-utilization 0.90
                     --max-num-seqs 32 --limit-mm-per-prompt "$MM" --reasoning-parser qwen3) ;;
    student)  args+=(--data-parallel-size "$n" --tensor-parallel-size 1 --max-model-len "$FS_STUDENT_MAX_LEN"
                     --gpu-memory-utilization 0.90 --max-num-seqs 48 --limit-mm-per-prompt "$MM" --reasoning-parser qwen3
                     --enable-lora --max-lora-rank "$FS_LORA_RANK" --max-loras "$FS_MAX_LORAS" --max-cpu-loras "$FS_MAX_CPU_LORAS")
              mapfile -t extra < <(lora_args); args+=("${extra[@]}") ;;
    verifier) args+=(--tensor-parallel-size "$n" --max-model-len "$FS_VERIFIER_MAX_LEN" --gpu-memory-utilization 0.90)
              [ -n "$FS_VERIFIER_EXTRA" ] && read -r -a ve <<< "$FS_VERIFIER_EXTRA" && args+=("${ve[@]}") ;;
    vref)     args+=(--tensor-parallel-size "$n" --max-model-len "$FS_VERIFIER_MAX_LEN" --gpu-memory-utilization "$FS_VREF_MEM")
              [ -n "$FS_VERIFIER_EXTRA" ] && read -r -a ve <<< "$FS_VERIFIER_EXTRA" && args+=("${ve[@]}") ;;
  esac
  say "starting $svc: $(basename "$dir") on GPU $gpus, port $(port_of "$svc")"
  CUDA_VISIBLE_DEVICES="$gpus" NO_PROXY="*" VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    nohup "$VLLM" "${args[@]}" > "$FS_LOG_DIR/loop_$svc.log" 2>&1 &
  echo $! > "$(pidf "$svc")"
}

wait_up() {  # service, minutes
  local svc=$1 limit=$(( $2 * 6 )) i
  [ -f "$(pidf "$svc")" ] || return 0
  for i in $(seq 1 "$limit"); do
    if is_up "$svc"; then ok "$svc up at $(endpoint_of "$svc") after $((i * 10)) s"; return 0; fi
    if ! kill -0 "$(cat "$(pidf "$svc")")" 2>/dev/null; then
      fail "$svc exited while loading. Last log lines:"; tail -30 "$FS_LOG_DIR/loop_$svc.log"
      [ "$svc" = teacher ] && warn "A100 has no FP8 compute units: vLLM must run the FP8 checkpoint weight-only (Marlin). If it refuses, see README §5 (Int4 fallback)."
      return 1
    fi
    sleep 10
  done
  fail "$svc not up after $2 minutes; see $FS_LOG_DIR/loop_$svc.log"; return 1
}

stop() {  # service
  local svc=$1 p
  p=$(pidf "$svc")
  if [ -f "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null; then
    kill "$(cat "$p")"
    for _ in $(seq 1 30); do kill -0 "$(cat "$p")" 2>/dev/null || break; sleep 2; done
    kill -0 "$(cat "$p")" 2>/dev/null && kill -9 "$(cat "$p")"
    ok "$svc stopped"
  else
    say "$svc not running"
  fi
  rm -f "$p"
}

wait_gpus_free() {  # gpu list, max MiB still in use
  local g used
  for _ in $(seq 1 30); do
    local busy=""
    for g in ${1//,/ }; do used=$(gpu_used_mib "$g"); [ "${used:-0}" -gt "$2" ] && busy="$busy $g(${used}MiB)"; done
    [ -z "$busy" ] && { ok "GPU $1 free"; return 0; }
    sleep 4
  done
  warn "GPU still in use:$busy  (nvidia-smi shows who)"; return 1
}

status() {
  local svc
  for svc in teacher student verifier vref; do
    if is_up "$svc"; then printf '  %-9s UP    %-28s GPU %s\n' "$svc" "$(endpoint_of "$svc")" "$(gpus_of "$svc")"
    elif [ -z "$(hf_of "$svc")" ]; then printf '  %-9s (not configured)\n' "$svc"
    else printf '  %-9s down  %-28s GPU %s\n' "$svc" "$(endpoint_of "$svc")" "$(gpus_of "$svc")"; fi
  done
  if is_up student; then
    echo "  models on the student: $(curl -s -m 5 "$FS_STUDENT_ENDPOINT/models" | "$FS_ENV_AGENT/bin/python" -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)"
  fi
  [ -s "$FS_LORA_REGISTRY" ] && { echo "  registered LoRAs:"; sed 's/^/    /' "$FS_LORA_REGISTRY"; }
  echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
}

case "${1:-status}" in
  A|a)
    say "layout A: teacher $FS_TEACHER_GPUS | student $FS_STUDENT_GPUS | verifier $FS_VERIFIER_GPUS | tools $FS_TOOL_GPUS (+ verifier_ref $FS_VREF_GPUS)"
    rc=0
    start verifier || rc=1; start student || rc=1; start vref || rc=1
    if ! is_up teacher; then wait_gpus_free "$FS_TEACHER_GPUS" 2000 || { fail "training jobs still on the teacher GPUs (13_train_queue.sh?)"; exit 1; }; fi
    start teacher || rc=1
    wait_up verifier 15 || rc=1; wait_up student 15 || rc=1; wait_up vref 15 || rc=1; wait_up teacher 40 || rc=1
    status; [ $rc -eq 0 ] && ok "layout A ready" || fail "layout A incomplete"; exit $rc ;;
  B|b)
    say "layout B: stopping the teacher; training slots $FS_TRAIN_SLOTS"
    stop teacher
    wait_gpus_free "$FS_TEACHER_GPUS" 2000
    rc=0; start verifier || rc=1; start student || rc=1; start vref || rc=1
    wait_up verifier 15 || rc=1; wait_up student 15 || rc=1; wait_up vref 15 || rc=1
    status; [ $rc -eq 0 ] && ok "layout B ready: bash server/13_train_queue.sh <jobs file>" || fail "layout B incomplete"; exit $rc ;;
  stop)
    if [ -n "${2:-}" ]; then stop "$2"; else for s in teacher student verifier vref; do stop "$s"; done; fi ;;
  restart-student)
    stop student; sleep 5; start student && wait_up student 15 ;;
  status) status ;;
  *) echo "usage: $0 A | B | status | stop [teacher|student|verifier|vref] | restart-student"; exit 2 ;;
esac
