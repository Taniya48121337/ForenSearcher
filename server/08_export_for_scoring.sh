#!/usr/bin/env bash
# Only if the server cannot run the Claude judge: pack what scoring needs (answers, configs, trajectories; no videos)
# and score on the PC. Output: server_logs/runs_for_scoring_<date>.tar.gz
source "$(dirname "$0")/env.sh"
cd "$FS_ROOT"
OUT="$FS_LOG_DIR/runs_for_scoring_$(date +%Y%m%d_%H%M).tar.gz"
find experiments/evid/runs experiments/bench/*/runs -maxdepth 4 \( -name result.json -o -name execution_links.json \
     -o -name run_config.json -o -name trajectory.jsonl -o -path '*/requests/*/response.json' \) 2>/dev/null \
  | grep -E "_(srvF27|srvG27|gsrv27|fsrv27|gsrvF27|fsrvF27)_1/" > "$FS_LOG_DIR/export_list.txt"
tar -czf "$OUT" -T "$FS_LOG_DIR/export_list.txt" experiments/bench/*/selection_n*.json experiments/bench/*/cases 2>/dev/null
ok "$(wc -l < "$FS_LOG_DIR/export_list.txt") files -> $OUT ($(du -h "$OUT" | cut -f1))"
echo "Copy it to the PC (e.g. scp server:$OUT .), unpack inside the ForenSearcher folder, then on the PC:"
echo "  python scripts/_evid_score.py --round srvF27 --judges claude-cli:sonnet --ids <the 91 ids>   (and srvG27)"
echo "  python scripts/_bench_score.py --dataset g360 --round gsrv27 --n $FS_BENCH_N ; python scripts/_bench_score.py --dataset fmnv --round fsrv27 --n $FS_BENCH_N"
