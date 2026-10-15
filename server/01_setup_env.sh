#!/usr/bin/env bash
# Step 1: two Python environments.
#   .venv-agent : the agent (ForenSearcher), its tools (Whisper, CLIP, GroundingDINO, FaceNet) and scoring
#   .venv-vllm  : the vLLM model server (kept separate because vLLM pins its own torch)
# Safe to rerun: existing environments are reused and only missing packages are installed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG="$FS_LOG_DIR/01_setup_env.log"
exec > >(tee "$LOG") 2>&1
say "Step 1: Python environments  (log: $LOG)"
PIP_ARGS=(--disable-pip-version-check)
[ -n "$FS_PIP_INDEX" ] && PIP_ARGS+=(-i "$FS_PIP_INDEX")

make_env() {  # dir
  if [ ! -x "$1/bin/python" ]; then say "creating $1 with $FS_PYTHON_BIN"; "$FS_PYTHON_BIN" -m venv "$1"; fi
  "$1/bin/python" -m pip install "${PIP_ARGS[@]}" -q --upgrade pip wheel setuptools
  ok "$1 -> $("$1/bin/python" --version)"
}

say "--- 1/2 agent environment"
make_env "$FS_ENV_AGENT"
cd "$FS_ROOT"
say "pip install (about 5-10 minutes; details in $FS_LOG_DIR/pip_agent.log)"
"$FS_ENV_AGENT/bin/pip" install "${PIP_ARGS[@]}" -e ".[free-search,asr,semantic,ocr,test]" \
    torchvision "huggingface-hub>=0.34" "hf_transfer" "ddgs>=9.16" "yt-dlp>=2025.9.1" > "$FS_LOG_DIR/pip_agent.log" 2>&1 \
  || { fail "pip install for the agent failed; last lines:"; tail -25 "$FS_LOG_DIR/pip_agent.log"; exit 1; }
# faster-whisper (CTranslate2) on GPU needs cuBLAS 12 + cuDNN 9 shared libraries; torch already ships them as pip
# packages, env.sh puts them on LD_LIBRARY_PATH.
"$FS_ENV_AGENT/bin/python" - <<'PY'
import importlib, sys
mods = ["httpx", "av", "cv2", "numpy", "PIL", "fitz", "trafilatura", "yt_dlp", "ddgs", "faster_whisper", "torch",
        "torchvision", "transformers", "huggingface_hub", "pytest", "foren_searcher"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"        {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        bad.append(m); print(f"        {m:16s} IMPORT FAILED: {type(exc).__name__}: {exc}")
import torch
print(f"        torch CUDA build {torch.version.cuda}, GPUs visible {torch.cuda.device_count()}")
sys.exit(1 if bad else 0)
PY
ok "agent environment ready"

say "--- 2/2 vLLM environment (downloads ~5 GB)"
make_env "$FS_ENV_VLLM"
"$FS_ENV_VLLM/bin/pip" install "${PIP_ARGS[@]}" "vllm>=0.17" "hf_transfer" > "$FS_LOG_DIR/pip_vllm.log" 2>&1 \
  || { fail "pip install vllm failed; last lines:"; tail -25 "$FS_LOG_DIR/pip_vllm.log"; exit 1; }
"$FS_ENV_VLLM/bin/python" -c "import vllm, torch, transformers; print('        vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
ok "vLLM environment ready"
ok "Step 1 done. Next: bash server/02_download_models.sh"
