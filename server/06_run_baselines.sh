#!/usr/bin/env bash
# Step 6: the baselines. Run inside tmux/screen so it survives a dropped SSH session:
#   tmux new -s fs   ->   bash server/06_run_baselines.sh   ->   detach with Ctrl-b d, reattach: tmux attach -t fs
# Options: --arms evid_base evid_detect g360 fmnv [g360_base fmnv_base]   --no-score   --summary-only
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$FS_ROOT"
say "Step 6: baselines (progress every minute; log $FS_LOG_DIR/06_run_baselines.log)"
"$FS_ENV_AGENT/bin/python" server/run_baselines.py "$@"
