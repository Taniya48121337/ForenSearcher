#!/usr/bin/env bash
# Run investigations in the loop (layout A), then verify with the local verifier on GPU 6.
#   ACTOR=teacher bash server/14_run_cases.sh --round r0 --limit 1              # teacher, one case first
#   HI_MODEL=hi_r1_full_s1 GIA_MODEL=gia_r1_full_s1 bash server/14_run_cases.sh --round r1s --samples 4
#   bash server/14_run_cases.sh --round r0base                                 # untrained student (base, no LoRA)
# Any other run_cases flag passes through ("$@"). Agent processes (and their tools) see only FS_TOOL_GPUS (GPU 7).
# Student runs use the student prompt only (--instruction-mode student) with thinking off, like the baselines.
set -uo pipefail
source "$(dirname "$0")/loop_env.sh"
cd "$FS_ROOT"
ACTOR=${ACTOR:-student}
case "$ACTOR" in
  teacher) EP=$FS_TEACHER_ENDPOINT; MODEL=$FS_TEACHER_NAME; EXTRA=() ;;
  student) EP=$FS_STUDENT_ENDPOINT; MODEL="HI=${HI_MODEL:-$FS_STUDENT_NAME},GIA=${GIA_MODEL:-$FS_STUDENT_NAME}"
           EXTRA=(--instruction-mode student --state-renderer compact_16k); [ "${STUDENT_THINK:-0}" = 1 ] || EXTRA+=(--no-think) ;;
  *) fail "ACTOR must be teacher or student"; exit 2 ;;
esac
served_on "$EP" "${MODEL##*=}" || { fail "$ACTOR server at $EP is not serving ${MODEL} (bash server/11_loop_layout.sh status)"; exit 1; }
served_on "$FS_VERIFIER_ENDPOINT" "$FS_VERIFIER_NAME" || { fail "verifier not up at $FS_VERIFIER_ENDPOINT"; exit 1; }
say "$ACTOR run: endpoint $EP model $MODEL; tools on GPU $FS_TOOL_GPUS; jobs $FS_JOBS"
CUDA_VISIBLE_DEVICES="$FS_TOOL_GPUS" "$FS_ENV_AGENT/bin/python" -m foren_searcher.datagen.run_cases \
  --endpoint "$EP" --model "$MODEL" --jobs "$FS_JOBS" --inspect-ops "${INSPECT_OPS:-auto}" "${EXTRA[@]}" \
  --then-verify --judge local --judge-endpoint "$FS_VERIFIER_ENDPOINT" --judge-model "$FS_VERIFIER_NAME" "$@"
