#!/usr/bin/env bash
# Run LoRA training jobs in parallel, one job per training slot (default GPU 0;1;2;3 = the teacher's GPUs in layout B).
#   bash server/11_loop_layout.sh B                 # free GPUs 0-3 first
#   bash server/13_train_queue.sh jobs/r1_hi.txt
# Jobs file: one job per line = the arguments of training/sft_lora.py; blank lines and '#' comments are ignored. Example:
#   --config training/configs/hi_r1.json --seed 1 --output-dir outputs/lora/r1/full/hi_s1
#   --config training/configs/hi_r1.json --seed 2 --output-dir outputs/lora/r1/full/hi_s2
#   --config training/configs/hi_r1_train_only.json --seed 1 --output-dir outputs/lora/r1/train_only/hi_s1
# More jobs than slots: the rest wait for the next free slot. Each job sees only its slot's GPUs (CUDA_VISIBLE_DEVICES),
# and sft_lora.py refuses device_map=auto over several visible GPUs unless the config says allow_multi_gpu.
# Logs: server_logs/train/<n>_<output dir>.log; summary at the end and in server_logs/train/queue_<time>.tsv.
# After a job finishes, load it for gate B: bash server/12_add_lora.sh <name> <output dir>/final
set -uo pipefail
source "$(dirname "$0")/loop_env.sh"
JOBS_FILE=${1:-}; [ -f "$JOBS_FILE" ] || { echo "usage: $0 <jobs file>"; exit 2; }
cd "$FS_ROOT"
PY="$FS_ENV_AGENT/bin/python"
"$PY" -c "import peft" 2>/dev/null || { fail "peft missing in $FS_ENV_AGENT: run bash server/10_loop_setup.sh"; exit 1; }
mkdir -p "$FS_LOG_DIR/train"

mapfile -t JOBS < <(grep -v -E '^\s*(#|$)' "$JOBS_FILE")
IFS=';' read -r -a SLOTS <<< "$FS_TRAIN_SLOTS"
[ ${#JOBS[@]} -gt 0 ] || { warn "no jobs in $JOBS_FILE"; exit 0; }

# Refuse slots that a running service occupies (teacher still up = layout A).
for slot in "${SLOTS[@]}"; do
  for g in ${slot//,/ }; do
    used=$(gpu_used_mib "$g")
    [ -z "$used" ] && { fail "GPU $g not found by nvidia-smi"; exit 1; }
    [ "$used" -gt 2000 ] && { fail "GPU $g already uses ${used} MiB ($([ "${FS_PROFILE:-}" = vast1 ] && echo "student still up? run: FS_PROFILE=vast1 bash server/11_loop_layout.sh train" || echo "teacher still up? run: bash server/11_loop_layout.sh B"))"; exit 1; }
  done
done

STAMP=$(date +%Y%m%d_%H%M%S); SUMMARY="$FS_LOG_DIR/train/queue_$STAMP.tsv"
printf 'job\tslot\texit\tminutes\tlog\targs\n' > "$SUMMARY"
say "${#JOBS[@]} jobs on ${#SLOTS[@]} slots ($FS_TRAIN_SLOTS); summary $SUMMARY"
declare -A PID START LOG JOBOF
next=0; running=0; failed=0

launch() {  # slot index
  local s=$1 args out log
  args=${JOBS[$next]}
  out=$(echo "$args" | grep -oP '(?<=--output-dir )\S+' | tr '/' '_'); out=${out:-job}
  log="$FS_LOG_DIR/train/$((next + 1))_${out}.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${SLOTS[$s]}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" training/sft_lora.py $args > "$log" 2>&1 &
  PID[$s]=$!; START[$s]=$(date +%s); LOG[$s]=$log; JOBOF[$s]=$next
  say "job $((next + 1))/${#JOBS[@]} -> GPU ${SLOTS[$s]}: $args"
  next=$((next + 1)); running=$((running + 1))
}

while [ $next -lt ${#JOBS[@]} ] || [ $running -gt 0 ]; do
  for s in "${!SLOTS[@]}"; do
    if [ -n "${PID[$s]:-}" ] && ! kill -0 "${PID[$s]}" 2>/dev/null; then
      wait "${PID[$s]}"; rc=$?
      mins=$(( ($(date +%s) - START[$s]) / 60 )); j=${JOBOF[$s]}
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$((j + 1))" "${SLOTS[$s]}" "$rc" "$mins" "${LOG[$s]}" "${JOBS[$j]}" >> "$SUMMARY"
      if [ $rc -eq 0 ]; then ok "job $((j + 1)) done in $mins min (GPU ${SLOTS[$s]})"
      else failed=$((failed + 1)); fail "job $((j + 1)) exit $rc after $mins min; last lines of ${LOG[$s]}:"; tail -8 "${LOG[$s]}"; fi
      unset "PID[$s]"; running=$((running - 1))
    fi
    if [ -z "${PID[$s]:-}" ] && [ $next -lt ${#JOBS[@]} ]; then launch "$s"; fi
  done
  sleep 15
done

echo; column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
[ $failed -eq 0 ] && ok "all ${#JOBS[@]} jobs finished" || fail "$failed of ${#JOBS[@]} jobs failed"
exit $(( failed > 0 ))
