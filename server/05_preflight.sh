#!/usr/bin/env bash
# Step 5: full pre-flight test (every module once + one end-to-end run per dataset). Options: --quick, --only <section>
set -uo pipefail
source "$(dirname "$0")/env.sh"
LOG="$FS_LOG_DIR/05_preflight.log"
say "Step 5: pre-flight test  (log: $LOG, report: $FS_LOG_DIR/preflight_report.md)"
cd "$FS_ROOT"
"$FS_ENV_AGENT/bin/python" server/preflight.py "$@" 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
