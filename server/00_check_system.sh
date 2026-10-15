#!/usr/bin/env bash
# Step 0: look at the server before installing anything. Read-only; prints what is there and what is missing.
set -u
source "$(dirname "$0")/env.sh"
LOG="$FS_LOG_DIR/00_check_system.log"
exec > >(tee "$LOG") 2>&1
say "Step 0: system check  (log: $LOG)"

need=0
say "--- OS / CPU / RAM / disk"
. /etc/os-release 2>/dev/null && ok "OS: ${PRETTY_NAME:-unknown}" || warn "cannot read /etc/os-release"
ok "CPU cores: $(nproc)   RAM: $(free -g | awk '/Mem:/{print $2}') GB"
free_gb=$(df -BG --output=avail "$FS_ROOT" | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -ge 400 ]; then ok "free disk at $FS_ROOT: ${free_gb} GB"
else warn "free disk at $FS_ROOT: ${free_gb} GB (need ~400 GB: model 55 GB, datasets ~60 GB, runs and downloaded originals ~200 GB)"; fi

say "--- GPUs"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader | sed 's/^/        /'
  n=$(nvidia-smi -L | wc -l); [ "$n" -ge 8 ] && ok "$n GPUs" || warn "$n GPUs (env.sh assumes 8: vLLM on $FS_VLLM_GPUS, tools on $FS_TOOL_GPUS)"
  cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
  ok "driver supports CUDA $cuda (vLLM wheels need >= 12.4)"
else fail "nvidia-smi not found: NVIDIA driver missing"; need=1; fi

say "--- tools"
for py in python3.12 python3.11 python3; do
  if command -v $py >/dev/null; then v=$($py -c 'import sys;print("%d.%d"%sys.version_info[:2])'); echo "        $py -> $v"; fi
done
pyv=$($FS_PYTHON_BIN -c 'import sys;print(sys.version_info>=(3,11))' 2>/dev/null)
[ "$pyv" = "True" ] && ok "FS_PYTHON_BIN=$FS_PYTHON_BIN is >= 3.11" || { fail "FS_PYTHON_BIN=$FS_PYTHON_BIN is missing or < 3.11: install python3.11/3.12 (or conda) and set FS_PYTHON_BIN in server/env.sh"; need=1; }
$FS_PYTHON_BIN -c 'import venv, ensurepip' 2>/dev/null && ok "python venv + ensurepip available" || { fail "python venv/ensurepip missing: sudo apt install python3.11-venv (or the venv package of your python)"; need=1; }
command -v ffmpeg >/dev/null && ok "ffmpeg: $(ffmpeg -version | head -1 | cut -c1-60)" || warn "ffmpeg not found (optional; sudo apt install ffmpeg makes YouTube downloads faster)"
command -v node >/dev/null && ok "node $(node -v)" || warn "node not found: needed only to score on the server with Claude (step 6); see README"
command -v claude >/dev/null && ok "claude CLI: $(claude --version 2>/dev/null | head -1)" || warn "claude CLI not found (scoring judge; see README step 6)"
command -v unzip >/dev/null && ok "unzip" || warn "unzip not found: sudo apt install unzip (needed to unpack datasets)"

say "--- network (proxy: ${FS_PROXY:-none}; HF endpoint: ${FS_HF_ENDPOINT:-https://huggingface.co})"
probe() {  # name url
  code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$2" 2>/dev/null); code=${code:-000}
  if [ "$code" != "000" ] && [ "$code" -lt 500 ]; then ok "$1 reachable (HTTP $code)"; else fail "$1 NOT reachable ($2)"; return 1; fi
}
probe "PyPI"            "${FS_PIP_INDEX:-https://pypi.org/simple/}"           || need=1
probe "Hugging Face"    "${FS_HF_ENDPOINT:-https://huggingface.co}/api/models/Qwen/Qwen3.8-27B" || need=1
probe "GitHub"          "https://github.com"                                   || warn "  GitHub is used once for FaceNet weights (the package carries a copy)"
probe "YouTube"         "https://www.youtube.com/results?search_query=test"    || { warn "  without YouTube the agent cannot search/download original videos: scores will drop. Set FS_PROXY."; need=1; }
probe "DuckDuckGo"      "https://duckduckgo.com"                               || warn "  DuckDuckGo is the fallback search engine"
probe "Wikipedia"       "https://en.wikipedia.org/wiki/Main_Page"              || warn "  general web pages may be blocked"

echo
if [ $need -eq 0 ]; then ok "Step 0 done: nothing blocking. Next: bash server/01_setup_env.sh"
else fail "Step 0 found blocking problems (FAIL lines above). Fix them (often: set FS_PROXY / FS_HF_ENDPOINT / FS_PIP_INDEX in server/env.sh) and rerun."; fi
