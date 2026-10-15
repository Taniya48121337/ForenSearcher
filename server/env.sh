# ForenSearcher server settings. Every server/*.sh script sources this file.
# Edit the values below once, then run the numbered steps (see server/README_服务器跑baseline.md).

# ---- paths ---------------------------------------------------------------------------------------------------
export FS_ROOT="${FS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # unpacked package root
export FS_MODEL_DIR="${FS_MODEL_DIR:-$FS_ROOT/.models/llm}"                    # where Qwen3.8-27B is downloaded
export FS_LOG_DIR="${FS_LOG_DIR:-$FS_ROOT/server_logs}"
mkdir -p "$FS_LOG_DIR"

# ---- python environments (created by 01_setup_env.sh) ------------------------------------------------------------
export FS_ENV_AGENT="${FS_ENV_AGENT:-$FS_ROOT/.venv-agent}"     # agent, tools, scoring
export FS_ENV_VLLM="${FS_ENV_VLLM:-$FS_ROOT/.venv-vllm}"        # vLLM server only (kept separate: its torch pin)
export FS_PYTHON_BIN="${FS_PYTHON_BIN:-python3}"               # base interpreter for the venvs; must be >= 3.11

# ---- GPUs ------------------------------------------------------------------------------------------------------
export FS_VLLM_GPUS="${FS_VLLM_GPUS:-0,1,2,3,4,5}"   # vLLM: one 27B bf16 replica per GPU (data parallel)
export FS_TOOL_GPUS="${FS_TOOL_GPUS:-6,7}"           # agent tools (Whisper, CLIP, GroundingDINO, FaceNet) of all runs

# ---- model server -------------------------------------------------------------------------------------------------
export FS_HF_MODEL="${FS_HF_MODEL:-Qwen/Qwen3.8-27B}"   # Hugging Face repo (bf16)
export FS_SERVED_NAME="${FS_SERVED_NAME:-qwen38-27b}"   # name the agent asks for
export FS_VLLM_PORT="${FS_VLLM_PORT:-8000}"
export FS_ENDPOINT="http://127.0.0.1:${FS_VLLM_PORT}/v1"
export FS_MAX_MODEL_LEN="${FS_MAX_MODEL_LEN:-40960}"    # 32K input (same window as the local Ollama runs) + 8K output

# ---- runs -------------------------------------------------------------------------------------------------------
export FS_JOBS="${FS_JOBS:-16}"          # investigations run in parallel (local PC used 2)
export FS_BENCH_N="${FS_BENCH_N:-200}"   # GroundLie360 / FMNV samples per dataset (half Real, half Fake)

# ---- network ----------------------------------------------------------------------------------------------------
# If the server needs a proxy for the internet, set it here; it is used for pip, Hugging Face, yt-dlp, DuckDuckGo
# and web pages fetched by the agent. Leave empty for a direct connection.
export FS_PROXY="${FS_PROXY:-}"                  # e.g. http://10.0.0.5:7890
# Hugging Face mirror if huggingface.co is blocked (e.g. https://hf-mirror.com); empty = official site.
export FS_HF_ENDPOINT="${FS_HF_ENDPOINT:-}"
# Pip index (e.g. https://pypi.tuna.tsinghua.edu.cn/simple); empty = default PyPI.
export FS_PIP_INDEX="${FS_PIP_INDEX:-}"

# ---- derived (do not edit) -------------------------------------------------------------------------------------
if [ -n "$FS_PROXY" ]; then
  export FOREN_HTTP_PROXY="$FS_PROXY"
  export HTTP_PROXY="$FS_PROXY" HTTPS_PROXY="$FS_PROXY" http_proxy="$FS_PROXY" https_proxy="$FS_PROXY"
  export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
fi
[ -n "$FS_HF_ENDPOINT" ] && export HF_ENDPOINT="$FS_HF_ENDPOINT"
export PYTHONPATH="$FS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export FOREN_CONFIG="$FS_ROOT/configs/evid_server_27b.toml"

say()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; }

# cuBLAS / cuDNN shipped with torch (pip nvidia-* packages) for faster-whisper on GPU
if [ -d "$FS_ENV_AGENT/lib" ]; then
  for d in "$FS_ENV_AGENT"/lib/python3*/site-packages/nvidia/{cublas,cudnn,cuda_runtime}/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
fi
