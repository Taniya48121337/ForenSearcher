#!/usr/bin/env bash
# 机器回来后一条命令接着干（Vast 单卡；宿主机 10-20–10-21 维护，实例可能被停，stop/start 时容器文件系统都在）。
#   bash server/resume.sh                 自动：有没训完的 LoRA 任务 → train；否则 → serve（再看 E2 有没有没跑完的阶段，有就接着跑）
#   bash server/resume.sh status          只看：vLLM、训练任务（每个的 final / 最近 checkpoint）、E2 各阶段完成数、看门狗
#   bash server/resume.sh serve           起学生 vLLM（11_loop_layout.sh serve，连同 loras.tsv 里登记的 LoRA）
#   bash server/resume.sh e2 baseline|after_sft   serve + 看门狗 + tmux e2_<阶段> 里接着跑 21_e2_eval.sh（done 的 run 跳过、半截的重跑；
#                                         run 跑两遍 = failed / 被看门狗杀的统一重排一轮）
#   bash server/resume.sh train [jobs]    停看门狗与 vLLM → 只排没写出 final 的任务 → tmux sft 里 13_train_queue.sh（任务带 --resume，从最近 checkpoint 接着训）
# jobs 默认 server/jobs/vast1_sft0_s1.txt（种子 1；负责人 10-19 定先只训种子 1）。日志：/workspace/logs、server_logs/train/。
set -uo pipefail
export FS_PROFILE=vast1 FS_LOG_DIR=${FS_LOG_DIR:-/workspace/logs}   # logs where server/backup_pull.sh picks them up
cd "$(dirname "$0")/.." || exit 1
source server/loop_env.sh
JOBS_DEFAULT=server/jobs/vast1_sft0_s1.txt

job_out() { grep -oP '(?<=--output-dir )\S+' <<< "$1"; }
job_done() { [ -f "$1/final/adapter_config.json" ] && [ -f "$1/train_summary.json" ]; }
last_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | while read -r d; do [ -f "$d/COMPLETE" ] && echo "$d"; done | sort -V | tail -1; }
pending_jobs() {  # jobs file -> lines whose output dir has no final yet
  grep -v -E '^\s*(#|$)' "$1" | while IFS= read -r line; do job_done "$(job_out "$line")" || echo "$line"; done
}
e2_counts() {  # phase -> "s1 done/24 s2 done/24"
  "$FS_ENV_AGENT/bin/python" - "$1" <<'PY'
import json, sys
from pathlib import Path
cfg = json.load(open("experiments/eval/r0/eval_config.json", encoding="utf-8"))
tag = cfg["round_tags"][sys.argv[1]]
cases = sum(len(s["cases"]) for s in json.load(open(cfg["eval_set"], encoding="utf-8"))["sets"].values())
out = []
for seed in (1, 2):
    done = [p for p in Path(cfg["runs_dir"]).glob(f"*_{tag}-s{seed}_1") if (p / "result.json").is_file() and (p / "execution_links.json").is_file()]
    out.append(f"{tag}-s{seed} {len(done)}/{cases}")
print("  ".join(out))
PY
}
e2_complete() { [ "$(e2_counts "$1" | grep -o '[0-9]*/[0-9]*' | awk -F/ '$1 != $2' | wc -l)" = 0 ]; }

status() {
  say "status $(date '+%F %T')"
  bash server/11_loop_layout.sh status
  echo "  training jobs ($JOBS):"
  grep -v -E '^\s*(#|$)' "$JOBS" | while IFS= read -r line; do
    out=$(job_out "$line"); c=$(last_ckpt "$out")
    if job_done "$out"; then echo "    done      $out"
    else echo "    pending   $out  (latest checkpoint: ${c:-none})"; fi
  done
  [ -f experiments/eval/r0/search_snapshots/snapshot_lock.json ] && echo "  E2 snapshot lock: present" || echo "  E2 snapshot lock: none"
  echo "  E2 baseline:  $(e2_counts baseline)"
  echo "  E2 after_sft: $(e2_counts after_sft)"
  pgrep -f "server/e2_watchdog[.]sh" >/dev/null && echo "  watchdog: running" || echo "  watchdog: not running"
  pgrep -f "training/sft_lora[.]py" >/dev/null && echo "  training process: running" || true
  df -h /workspace | tail -1 | awk '{print "  disk /workspace: "$4" free"}'
}

start_watchdog() {
  pgrep -f "server/e2_watchdog[.]sh" >/dev/null || tmux new-session -d -s e2_watchdog "bash server/e2_watchdog.sh 1800"
  ok "watchdog running (tmux e2_watchdog)"
}
stop_watchdog() { tmux kill-session -t e2_watchdog 2>/dev/null; pkill -f "server/e2_watchdog[.]sh" 2>/dev/null; return 0; }

serve() {
  if pgrep -f "training/sft_lora[.]py" >/dev/null; then fail "a training job is running: not starting vLLM (bash $0 status)"; return 1; fi
  bash server/11_loop_layout.sh serve
}

e2() {  # phase
  local phase=$1 steps
  serve || return 1
  start_watchdog
  if [ "$phase" = baseline ]; then
    steps="run run"                 # second run = the one requeue pass for failed / watchdog-killed runs
    [ -f experiments/eval/r0/search_snapshots/snapshot_lock.json ] || steps="$steps lock"
    steps="$steps score metrics"
  else steps="run run score metrics"; fi
  tmux has-session -t "e2_$phase" 2>/dev/null && { fail "tmux e2_$phase already exists (attach: tmux attach -t e2_$phase)"; return 1; }
  tmux new-session -d -s "e2_$phase" "bash server/21_e2_eval.sh $phase $steps 2>&1 | tee -a $FS_LOG_DIR/e2_${phase}_resume.log"
  ok "E2 $phase resumed in tmux e2_$phase (steps: $steps; done runs are skipped, half-finished ones rerun)"
}

train() {
  local jobs_file=$1 todo="$FS_LOG_DIR/resume_jobs_$(date +%Y%m%d_%H%M%S).txt"
  pending_jobs "$jobs_file" > "$todo"
  if [ ! -s "$todo" ]; then ok "all jobs in $jobs_file have a final adapter: nothing to train"; rm -f "$todo"; return 0; fi
  grep -q -- "--resume" "$todo" || warn "jobs have no --resume: they would start from scratch"
  if pgrep -f "training/sft_lora[.]py" >/dev/null; then fail "a training job is already running"; return 1; fi
  stop_watchdog
  if is_up student && [ ! -f "$(pidf student)" ]; then   # started by hand (e.g. /workspace/logs/serve_cmd.sh in tmux serve)
    tmux kill-session -t serve 2>/dev/null; sleep 10
  fi
  bash server/11_loop_layout.sh train || return 1
  while IFS= read -r line; do out=$(job_out "$line"); echo "  queued: $out (resume from: $(last_ckpt "$out" || true))"; done < "$todo"
  tmux new-session -d -s sft "bash server/13_train_queue.sh $todo 2>&1 | tee -a $FS_LOG_DIR/sft_queue.log"
  ok "training queue in tmux sft ($(wc -l < "$todo") jobs); follow: tail -f server_logs/train/*.log"
}

MODE=${1:-auto}; JOBS=${2:-$JOBS_DEFAULT}
case "$MODE" in
  status) status ;;
  serve) serve ;;
  e2) [ -n "${2:-}" ] || { echo "usage: $0 e2 baseline|after_sft"; exit 2; }; JOBS=$JOBS_DEFAULT; e2 "$2" ;;
  train) train "$JOBS" ;;
  auto)
    status
    if [ -n "$(pending_jobs "$JOBS")" ] && ls -d /workspace/outputs/lora/sft0_27b/*/checkpoint-* >/dev/null 2>&1; then
      say "auto: unfinished LoRA jobs with checkpoints -> train"; train "$JOBS"
    elif ! e2_complete baseline; then say "auto: E2 baseline incomplete -> resume it"; e2 baseline
    elif [ -n "$(pending_jobs "$JOBS")" ]; then say "auto: baseline complete, LoRA jobs not done -> train"; train "$JOBS"
    elif ! e2_complete after_sft; then say "auto: E2 after_sft incomplete -> resume it"; e2 after_sft
    else say "auto: nothing pending -> serve"; serve; fi ;;
  *) echo "usage: $0 [auto | status | serve | e2 baseline|after_sft | train [jobs file]]"; exit 2 ;;
esac
