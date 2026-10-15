# Self-improvement loop on 8×A100 80GB: GPU layout, ports, model names (2026-10-12).
# Every server/1x_*.sh script sources this file (it sources env.sh first, then overrides what differs from the baselines).
# Plan and reasons: server/README_自进化循环_GPU分配.md
#
#   Layout A (rollout / bridging / verification / gate A / gate B / dev_eval)
#     GPU 0-3  teacher  Qwen3.5-122B-A10B FP8, tensor parallel 4      port 8000
#     GPU 4-5  student  Qwen3.5-9B, 2 data-parallel replicas, LoRAs   port 8002  (HI / GIA LoRAs of every arm and round)
#     GPU 6    verifier (training-side, different family from the teacher)  port 8001
#     GPU 7    agent tools (Whisper, CLIP, GroundingDINO, FaceNet, DETECT) + verifier_ref if it is small  port 8003
#   Layout B (outer-loop training window)
#     GPU 0-3  teacher stopped; one LoRA training job per GPU (server/13_train_queue.sh)
#     GPU 4-7  unchanged: the student keeps its LoRAs loaded, so gate B starts as soon as a job finishes
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ---- weights (Hugging Face repos; 10_loop_setup.sh downloads them to $FS_MODEL_DIR/<repo basename>) -------------------
export FS_TEACHER_HF="${FS_TEACHER_HF:-Qwen/Qwen3.5-122B-A10B-FP8}"   # check the exact repo name on the hub before step 10
export FS_STUDENT_HF="${FS_STUDENT_HF:-Qwen/Qwen3.5-9B}"
export FS_VERIFIER_HF="${FS_VERIFIER_HF:-}"   # REQUIRED: a different family from the teacher; must fit one GPU (<=32B dense bf16)
export FS_VREF_HF="${FS_VREF_HF:-}"           # optional verifier_ref: a third family, frozen at the start of the experiment

# ---- served names and ports -------------------------------------------------------------------------------------------
export FS_TEACHER_NAME=teacher  FS_TEACHER_PORT="${FS_TEACHER_PORT:-8000}"
export FS_VERIFIER_NAME=verifier FS_VERIFIER_PORT="${FS_VERIFIER_PORT:-8001}"
export FS_STUDENT_NAME=student  FS_STUDENT_PORT="${FS_STUDENT_PORT:-8002}"
export FS_VREF_NAME=verifier_ref FS_VREF_PORT="${FS_VREF_PORT:-8003}"
export FS_TEACHER_ENDPOINT="http://127.0.0.1:${FS_TEACHER_PORT}/v1"
export FS_VERIFIER_ENDPOINT="http://127.0.0.1:${FS_VERIFIER_PORT}/v1"
export FS_STUDENT_ENDPOINT="http://127.0.0.1:${FS_STUDENT_PORT}/v1"
export FS_VREF_ENDPOINT="http://127.0.0.1:${FS_VREF_PORT}/v1"

# ---- GPUs ---------------------------------------------------------------------------------------------------------
export FS_TEACHER_GPUS="${FS_TEACHER_GPUS:-0,1,2,3}"
export FS_STUDENT_GPUS="${FS_STUDENT_GPUS:-4,5}"       # one replica per GPU
export FS_VERIFIER_GPUS="${FS_VERIFIER_GPUS:-6}"
export FS_VREF_GPUS="${FS_VREF_GPUS:-7}"
export FS_VREF_MEM="${FS_VREF_MEM:-0.45}"             # verifier_ref shares GPU 7 with the tools: keep ~40 GB for them
export FS_TOOL_GPUS="${FS_TOOL_GPUS:-7}"              # overrides env.sh (baselines use 6,7)
# Training slots for layout B, separated by ';'. One slot = the GPUs one job sees. 9B: one GPU per job.
# 27B student: "0,1;2,3" and "allow_multi_gpu": true in the training config.
export FS_TRAIN_SLOTS="${FS_TRAIN_SLOTS:-0;1;2;3}"

# ---- limits -------------------------------------------------------------------------------------------------------
export FS_TEACHER_MAX_LEN="${FS_TEACHER_MAX_LEN:-40960}"     # 32K input + 8K output, same window as the baselines
export FS_STUDENT_MAX_LEN="${FS_STUDENT_MAX_LEN:-40960}"
export FS_VERIFIER_MAX_LEN="${FS_VERIFIER_MAX_LEN:-32768}"
export FS_LORA_RANK="${FS_LORA_RANK:-32}"                    # = lora_r in training/configs
export FS_MAX_LORAS="${FS_MAX_LORAS:-4}"                     # LoRAs active in one batch (HI + GIA of two arms)
export FS_MAX_CPU_LORAS="${FS_MAX_CPU_LORAS:-16}"            # LoRAs kept loaded (arms × rounds × roles)
export FS_VERIFIER_EXTRA="${FS_VERIFIER_EXTRA:-}"            # extra vllm serve flags for the verifier, e.g. --limit-mm-per-prompt ...
export FS_LORA_REGISTRY="$FS_LOG_DIR/loras.tsv"              # name<TAB>adapter dir; the student is always started with all of them

# ---- agent ----------------------------------------------------------------------------------------------------------
export FOREN_CONFIG="$FS_ROOT/configs/loop_server.toml"

# ---- helpers ------------------------------------------------------------------------------------------------------
served_on() {  # endpoint name -> 0 if the server lists that model
  curl -s -m 5 "$1/models" 2>/dev/null | grep -q "\"$2\""
}
gpu_used_mib() {  # gpu index -> MiB in use
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
}
