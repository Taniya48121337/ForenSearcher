#!/usr/bin/env bash
# Step 2: model weights.
#   Qwen/Qwen3.8-27B (bf16, ~55 GB)                      -> $FS_MODEL_DIR/Qwen3.8-27B
#   INSPECT tools (~1.8 GB): faster-whisper-small, CLIP ViT-B/32, GroundingDINO-tiny -> .models/inspect/
#   FaceNet VGGFace2 (~110 MB, DETECT face comparison)     -> torch hub cache
# Resumable: rerun after an interruption and it continues where it stopped.
set -euo pipefail
source "$(dirname "$0")/env.sh"
LOG="$FS_LOG_DIR/02_download_models.log"
exec > >(tee "$LOG") 2>&1
say "Step 2: model weights  (log: $LOG; HF endpoint: ${HF_ENDPOINT:-https://huggingface.co}; proxy: ${FS_PROXY:-none})"
[ -z "$FS_PROXY" ] && export HF_HUB_ENABLE_HF_TRANSFER=1     # faster parallel download (not through proxies)
PY="$FS_ENV_AGENT/bin/python"

"$PY" - "$FS_HF_MODEL" "$FS_MODEL_DIR" "$FS_ROOT/.models/inspect" <<'PY'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

llm_repo, llm_dir, inspect_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
jobs = [
    (llm_repo, llm_dir / llm_repo.split("/")[-1], None, "LLM"),
    ("Systran/faster-whisper-small", inspect_dir / "faster-whisper-small", None, "ASR"),
    ("openai/clip-vit-base-patch32", inspect_dir / "clip-vit-base-patch32", ["flax_model.msgpack", "tf_model.h5", "rust_model.ot"], "frame search"),
    ("IDEA-Research/grounding-dino-tiny", inspect_dir / "grounding-dino-tiny", ["pytorch_model.bin"], "region search"),
]
bad = 0
for repo, target, ignore, role in jobs:
    t = time.time()
    print(f"[....] {role}: {repo} -> {target}", flush=True)
    for attempt in range(1, 4):
        try:
            snapshot_download(repo_id=repo, local_dir=str(target), ignore_patterns=ignore, max_workers=8)
            break
        except Exception as exc:
            print(f"[WARN] attempt {attempt} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            time.sleep(10 * attempt)
    else:
        print(f"[FAIL] {repo} could not be downloaded (check FS_PROXY / FS_HF_ENDPOINT in server/env.sh)", flush=True)
        bad += 1
        continue
    files = [p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts]
    size = sum(p.stat().st_size for p in files) / 1e9
    weights = [p.name for p in files if p.suffix in (".safetensors", ".bin")]
    print(f"[ OK ] {role}: {len(files)} files, {size:.2f} GB, weight files: {len(weights)}  ({time.time() - t:.0f} s)", flush=True)
sys.exit(1 if bad else 0)
PY

say "--- FaceNet (VGGFace2) weights for DETECT"
HUB=$("$PY" -c "import torch; print(torch.hub.get_dir())")
mkdir -p "$HUB/checkpoints"
W="$HUB/checkpoints/20180402-114759-vggface2.pt"
if [ -s "$W" ]; then ok "already present: $W"
elif [ -s "$FS_ROOT/server/assets/20180402-114759-vggface2.pt" ]; then
  cp "$FS_ROOT/server/assets/20180402-114759-vggface2.pt" "$W"; ok "copied from the package: $W"
else
  curl -L --retry 3 -m 600 -o "$W" https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt \
    && ok "downloaded from GitHub: $W" || { fail "FaceNet weights missing; copy 20180402-114759-vggface2.pt to $W"; exit 1; }
fi
ls -l "$W" | awk '{print "        "$5" bytes"}'
ok "Step 2 done. Next: bash server/03_download_data.sh"
