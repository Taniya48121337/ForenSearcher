#!/usr/bin/env bash
# Step 10 (loop, after baseline steps 0-3): training package + loop model weights.
#   - peft into .venv-agent (training/sft_lora.py; torch / transformers are already there from step 1)
#   - weights to $FS_MODEL_DIR: teacher (Qwen3.5-122B-A10B FP8, ~125 GB), student (Qwen3.5-9B, ~19 GB),
#     verifier and verifier_ref if configured in loop_env.sh
#   - checks: transformers knows the student architecture; disk space; the GPU plan has no overlaps
# Resumable: rerun after an interruption.
set -euo pipefail
source "$(dirname "$0")/loop_env.sh"
LOG="$FS_LOG_DIR/10_loop_setup.log"
exec > >(tee "$LOG") 2>&1
say "Step 10: loop setup  (log: $LOG)"
PY="$FS_ENV_AGENT/bin/python"
PIP_ARGS=(--disable-pip-version-check -q); [ -n "$FS_PIP_INDEX" ] && PIP_ARGS+=(-i "$FS_PIP_INDEX")

say "--- 1/4 GPU plan"
"$PY" - <<'PY'
import os, sys
def ids(v): return {g for g in v.replace(";", ",").split(",") if g.strip()}
t, s, v, r, tools = (ids(os.environ[k]) for k in ("FS_TEACHER_GPUS", "FS_STUDENT_GPUS", "FS_VERIFIER_GPUS", "FS_VREF_GPUS", "FS_TOOL_GPUS"))
train = ids(os.environ["FS_TRAIN_SLOTS"])
bad = []
if t & s or t & v or s & v: bad.append("teacher / student / verifier GPUs overlap")
if not train <= t: bad.append(f"training slots {sorted(train)} must be teacher GPUs {sorted(t)} (freed in layout B)")
if os.environ.get("FS_VREF_HF") and (r & (t | s)): bad.append("verifier_ref must not sit on teacher or student GPUs")
print(f"        teacher {sorted(t)} | student {sorted(s)} | verifier {sorted(v)} | verifier_ref {sorted(r)} | tools {sorted(tools)} | training {os.environ['FS_TRAIN_SLOTS']}")
for b in bad: print(f"[FAIL] {b}")
sys.exit(1 if bad else 0)
PY
ok "GPU plan consistent"

say "--- 2/4 peft"
"$FS_ENV_AGENT/bin/pip" install "${PIP_ARGS[@]}" "peft>=0.17,<1" || { fail "pip install peft failed"; exit 1; }
"$PY" -c "import peft, transformers, torch; print('        peft', peft.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"

say "--- 3/4 weights -> $FS_MODEL_DIR"
[ -z "$FS_VERIFIER_HF" ] && warn "FS_VERIFIER_HF is empty: choose the verifier (different family from the teacher, fits one GPU) and rerun"
df -h "$FS_MODEL_DIR" | tail -1 | awk '{print "        free disk: "$4}'
[ -z "$FS_PROXY" ] && export HF_HUB_ENABLE_HF_TRANSFER=1
"$PY" - "$FS_MODEL_DIR" "$FS_TEACHER_HF" "$FS_STUDENT_HF" "$FS_VERIFIER_HF" "$FS_VREF_HF" <<'PY'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
root, repos = Path(sys.argv[1]), [r for r in sys.argv[2:] if r]
bad = 0
for repo in repos:
    target = root / repo.split("/")[-1]
    t = time.time(); print(f"[....] {repo} -> {target}", flush=True)
    for attempt in range(1, 4):
        try:
            snapshot_download(repo_id=repo, local_dir=str(target), max_workers=8,
                              ignore_patterns=["*.pth", "*.pt", "original/*", "*.gguf"])
            break
        except Exception as exc:
            print(f"[WARN] attempt {attempt}: {type(exc).__name__}: {str(exc)[:300]}", flush=True); time.sleep(10 * attempt)
    else:
        print(f"[FAIL] {repo} (repo name right? proxy / FS_HF_ENDPOINT?)", flush=True); bad += 1; continue
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
    print(f"[ OK ] {repo}: {size:.1f} GB ({time.time() - t:.0f} s)", flush=True)
sys.exit(1 if bad else 0)
PY

say "--- 4/4 student architecture known to transformers (needed by training/sft_lora.py)"
"$PY" - "$FS_MODEL_DIR/$(basename "$FS_STUDENT_HF")" <<'PY'
import sys
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
print(f"        {cfg.architectures} vision={'vision_config' in cfg.to_dict()}")
PY
ok "Step 10 done. Next: bash server/11_loop_layout.sh A"
