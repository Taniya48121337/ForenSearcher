# Single-GPU profile "vast1": Vast 1×H200 NVL 141 GB, 128 vCPU, 377 GB RAM (2026-10-19).
# It overrides what server/loop_env.sh sets for the 8-GPU loop; nothing else changes.
#
# How it is selected (pick one):
#   FS_PROFILE=vast1 bash server/11_loop_layout.sh serve      # per command
#   export FS_PROFILE=vast1                                   # per shell / tmux window (or put it in ~/.bashrc)
#   source server/vast1_env.sh                                # interactive shell: loads loop_env.sh + this file, exports FS_PROFILE
# loop_env.sh sources this file at its end when FS_PROFILE=vast1, so 11_loop_layout.sh, 12_add_lora.sh, 13_train_queue.sh and
# 21_e2_eval.sh all see the same values. FS_PROFILE unset = the 8-GPU layout, unchanged.
#
# One GPU, so training and the student server take turns (11_loop_layout.sh train | serve):
#   serve  GPU 0  student Qwen3.8-27B bf16, ONE replica (no data parallel), LoRAs from server_logs/loras.tsv, port 8002
#               + agent tools (faster-whisper-small, CLIP B/32, GroundingDINO-tiny, FaceNet/MTCNN) in every run process
#   train  GPU 0  student stopped; 13_train_queue.sh runs the jobs one after another (one slot "0")
# No local teacher, verifier or verifier_ref: the verifier / EVID judge go to the Anthropic API (no GPU).
#
# Student memory (10-19 decision: E2 runs both seeds at once, --jobs 6 each = up to 12 run processes; vLLM --max-num-seqs 16).
# FS_STUDENT_MEM = --gpu-memory-utilization 0.75, worked out for 143,771 MiB = 140.4 GiB:
#   vLLM 0.75 × 140.4 = 105.3 GiB = weights 55.6 GB (51.8 GiB) + profiling peak (32-image vision batch) and CUDA graphs ~8 GiB
#     + LoRA buffers (2 slots × rank 32) ~1 GiB + KV / linear-attention state ~44.5 GiB.
#   Needed for 16 × 40,960 tokens: KV per token = 16 full-attention layers × 2 (K,V) × 4 KV heads × 256 dims × 2 B = 64 KiB
#     → 16 × 40,960 × 64 KiB = 40.0 GiB; GatedDeltaNet state per sequence = 48 layers × 48 heads × 128 × 128 × 2 B ≈ 72 MiB
#     (+ conv state) → 16 × ~75 MiB ≈ 1.2 GiB (2.4 GiB if kept in fp32). 41–42.5 GiB ≤ 44.5: all 16 slots fit at full length.
#     In practice E2 prompts are 16–24K tokens and each run has at most one request in flight (eval_r0.toml
#     max_concurrency 1) plus the describer, so 12 processes stay within 16 sequences most of the time; extra requests queue.
#   Left for the tools: 140.4 − 105.3 = 35.1 GiB for 12 processes ≈ 2.9 GiB each. run_cases starts one process per run and each
#     loads its own tools on first use: CUDA context ~0.5 GiB + CLIP B/32 fp32 ~0.6 + whisper-small fp16 ~0.8 (≈ 2 GiB typical)
#     + GroundingDINO-tiny ~0.7 weights / ~1.5 activations + FaceNet/MTCNN ~0.4 (≈ 4 GiB if all are resident at once).
#     Typical load fits; the all-tools worst case (12 × 4 = 48 GiB) does not. Nothing here was measured: watch `nvidia-smi`
#     in the first 10 minutes of E2. A tool that runs out of memory fails inside that run (vLLM has its memory already) and
#     the run's observation differs from a clean run, so lower the jobs (E2_JOBS=4) rather than raising the vLLM share.
#   Earlier single-seed default was 0.80 / 32 seqs / 4 jobs (28 GiB for the tools); 0.85 would leave 21 GiB.
#
# Overrides: the values below replace loop_env.sh's 8-GPU defaults. A value you set yourself before loop_env.sh runs
# (environment, ~/.bashrc) is kept, unless it is identical to the 8-GPU default (then it cannot be told apart and is replaced).
# Teacher / verifier / verifier_ref are always emptied: in this profile they cannot be configured.

if [ "${FS_LOOP_ENV_SOURCED:-}" != 1 ]; then
  # sourced on its own (interactive shell): load the base settings first; loop_env.sh comes back here at its end
  export FS_PROFILE=vast1
  source "$(dirname "${BASH_SOURCE[0]}")/loop_env.sh"
  return 0 2>/dev/null || exit 0
fi
export FS_PROFILE=vast1

v1_set() {  # name, vast1 value, 8-GPU default(s): replace an empty value or a default, keep a value the user chose
  local name=$1 value=$2 cur="${!1:-}" d
  shift 2
  if [ -z "$cur" ]; then export "$name=$value"; return 0; fi
  for d in "$@"; do [ "$cur" = "$d" ] && { export "$name=$value"; return 0; }; done
  return 0
}

# ---- paths and environments ----------------------------------------------------------------------------------------
v1_set FS_MODEL_DIR /workspace/models "$FS_ROOT/.models/llm"          # Qwen3.8-27B/ (bf16); .models/llm links here too
v1_set FS_ENV_AGENT /venv/main "$FS_ROOT/.venv-agent"                 # agent / tools / scoring / training (peft)
v1_set FS_ENV_VLLM /workspace/venvs/vllm "$FS_ROOT/.venv-vllm"        # vLLM only
# env.sh did this for the old path; repeat for the new one. Written with if/fi so no step returns non-zero: env.sh's
# `[ -d "$d" ] && export …` as the file's last command makes `source env.sh` return 1 when a directory is missing, which
# kills a caller running under `set -e` (none of the loop scripts uses -e; 21_e2_eval.sh sets it only after sourcing).
# /venv/main has lib/python3.12 and lib/python3.1, both with nvidia/{cublas,cudnn,cuda_runtime}/lib: duplicates are skipped.
if [ -d "$FS_ENV_AGENT/lib" ]; then
  for d in "$FS_ENV_AGENT"/lib/python3*/site-packages/nvidia/{cublas,cudnn,cuda_runtime}/lib; do
    if [ -d "$d" ]; then
      d=$(cd "$d" && pwd -P)
      case ":${LD_LIBRARY_PATH:-}:" in *":$d:"*) ;; *) export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
    fi
  done
fi

# ---- models: student only ------------------------------------------------------------------------------------------
v1_set FS_STUDENT_HF Qwen/Qwen3.8-27B Qwen/Qwen3.5-9B                 # served from $FS_MODEL_DIR/Qwen3.8-27B as "student"
export FS_TEACHER_HF="" FS_VERIFIER_HF="" FS_VREF_HF=""             # not deployed: 11_loop_layout.sh shows "(not configured)"
# names and ports unchanged from loop_env.sh: student @ 127.0.0.1:8002 (FS_STUDENT_ENDPOINT)

# ---- GPU 0 for everything ------------------------------------------------------------------------------------------
v1_set FS_STUDENT_GPUS 0 4,5          # one GPU = one replica; 11_loop_layout.sh adds --data-parallel-size only for > 1 GPU
v1_set FS_TOOL_GPUS 0 7 6,7           # agent processes (Whisper, CLIP, GroundingDINO, FaceNet) share the student's GPU
v1_set FS_TRAIN_SLOTS 0 "0;1;2;3"     # one slot: 13_train_queue.sh runs the jobs serially (layout "train" frees GPU 0 first)
v1_set FS_VLLM_GPUS 0 0,1,2,3,4,5     # only for the baseline scripts (04_*), kept consistent

# ---- student server ------------------------------------------------------------------------------------------------
export FS_STUDENT_MEM="${FS_STUDENT_MEM:-0.75}"        # --gpu-memory-utilization (budget above)
export FS_STUDENT_MAX_SEQS="${FS_STUDENT_MAX_SEQS:-16}" # --max-num-seqs: 16 × 40,960-token windows fit in the KV budget above
v1_set FS_STUDENT_MAX_LEN 40960 40960                  # --max-model-len: 32K input + 8K output, same window as training rows
v1_set FS_LORA_RANK 32 32                              # = lora_r of training/configs/*_sft0_27b.json
v1_set FS_MAX_LORAS 2 4                                # LoRAs in one batch: the HI and GIA adapter of one arm
v1_set FS_MAX_CPU_LORAS 4 16                           # LoRAs kept loaded: sft0 HI/GIA × 2 seeds (= the serve started 10-19)
v1_set FS_JOBS 6 16                                    # parallel runs per run_cases call (E2: 6 per seed, 2 seeds at once)

# ---- agent runtime config ------------------------------------------------------------------------------------------
# configs/eval_r0.toml, not loop_server.toml: loop_server.toml points the teacher at a local 122B on :8000 (not deployed
# here) and uses max_gia_turns 8 / max_concurrency 4, while the sft0 training runs used GIA 6 turns and one workspace at a
# time (run_config.runtime_limits of the r1 runs), which is what eval_r0.toml has. E2 uses eval_r0.toml anyway
# (scripts/eval_r0.py sets FOREN_CONFIG itself), so loop sampling on this box sees the same runtime as E2.
# HI turns per workspace come from the spec (harness_params.max_workspace_turns = 12 in hi@0.10.0 / hi@0.11.0), not from
# the toml's 8: --harness-params apply (default) overrides [runtime].
export FOREN_CONFIG="$FS_ROOT/configs/eval_r0.toml"

unset -f v1_set
