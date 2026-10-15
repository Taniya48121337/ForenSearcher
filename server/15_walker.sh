#!/usr/bin/env bash
# walker trajectories on the server. Since 2026-10-15 (r0 cost cut) the teacher is the frozen slot table in specs/r0_freeze.json:
#   S0-S3 (rationale / query / task text / sufficiency) -> qwen backend = the vLLM teacher on $FS_TEACHER_ENDPOINT
#                                                          (FOREN_TEACHER_QWEN_ENDPOINT / _MODEL point the frozen qwen backend at it)
#   S4-S5 (report / answer, incl. the post-verification rewrite) -> strong backend = Codex CLI (ChatGPT subscription, gpt-6-astra),
#                                                          serialized across processes; backup = OpenRouter Sonnet (whole run)
#   verifier -> Anthropic API direct, claude-sonnet-5-5, Message Batches + prompt caching (--judge api-batch), ANTHROPIC_API_KEY
#
#   bash server/15_walker.sh --round ws1 --cases videosham_4126 --no-verify       # one case, smoke
#   bash server/15_walker.sh --round r0c5 --jobs 2 --cases ...                     # trajectories + batch verify + rewrite + export
#   bash server/15_walker.sh --codex-check                                         # only check Codex CLI install / login / one call
#   FS_TEACHER_BACKUP=strong bash server/15_walker.sh --round ...                   # Codex out of quota: S4/S5 -> OpenRouter Sonnet
#   FS_WALKER_JUDGE=local bash server/15_walker.sh ...                              # old path: local verifier on $FS_VERIFIER_PORT
#                                                                                  #   (not r0-valid; r0 is judged by the frozen API judge)
# Any walker_run.py d1 flag passes through ("$@"). --jobs 2 is the recommended value (Codex calls queue on one lock; while one run
# waits for Codex the other runs Qwen slots and tools). Needs layout A (bash server/11_loop_layout.sh A) for the Qwen teacher.
#
# Keys / settings this script reads (put them in server/env.sh or export before running; never commit them):
#   ANTHROPIC_API_KEY          verifier (Anthropic direct; batch + cache)            required unless --no-verify or FS_WALKER_JUDGE=local
#   FS_OPENROUTER_API_KEY      strong backup (OpenRouter Sonnet) -> FOREN_TEACHER_STRONG_API_KEY; only needed with FS_TEACHER_BACKUP=strong
#   FS_PROXY                   (env.sh) outbound proxy; Codex CLI, Anthropic and OpenRouter all go through HTTPS_PROXY
#   FOREN_CODEX_BIN            Codex CLI path if it is not on PATH (e.g. ~/.local/node/bin/codex)
#   FOREN_ORACLE_RETRY_MAX_S   Codex 429 / usage-limit backoff budget per call (default 900 s: 30, 60, 120, 240, 480 s)
#
# ---------------------------------------------------------------------------------------------------------------------------
# Installing and authenticating Codex CLI on the server (once; ~10 minutes)
#   1. Node.js >= 20 for the agent user (no root needed):
#        curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz -o /tmp/node.tar.xz   # via $FS_PROXY if needed
#        mkdir -p ~/.local/node && tar -xJf /tmp/node.tar.xz -C ~/.local/node --strip-components=1
#        export PATH="$HOME/.local/node/bin:$PATH"            # add this line to ~/.bashrc (or set FOREN_CODEX_BIN)
#      (npm registry through the proxy: npm config set proxy "$FS_PROXY"; npm config set https-proxy "$FS_PROXY")
#   2. npm install -g @openai/codex        then  codex --version   (use the same major version as on the PC; check with codex --version there)
#   3. Sign in with the ChatGPT account that has the subscription. The server has no browser, so either
#        a) copy the PC's login:  scp %USERPROFILE%\.codex\auth.json  <server>:~/.codex/auth.json   then  chmod 600 ~/.codex/auth.json
#           (same account on two machines shares one usage limit; do not run the PC batch and the server batch at the same time), or
#        b) codex login --device-auth      open the printed URL on any device, enter the code, sign in with that account
#      then  codex login status   must say "Logged in using ChatGPT" (not an API key: an API key bills per token).
#   4. bash server/15_walker.sh --codex-check   (walker_run's preflight: version, login status, one minimal `codex exec` with the
#      frozen model gpt-6-astra and effort medium; reports "usage limit" when the subscription is out of quota)
#   Codex reads HTTPS_PROXY; env.sh exports it from FS_PROXY. auth.json is a credential: keep it out of the repo and of bundles.
# ---------------------------------------------------------------------------------------------------------------------------
set -uo pipefail
source "$(dirname "$0")/loop_env.sh"
cd "$FS_ROOT"
export PYTHONPATH="$FS_ROOT/src" PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8

CODEX_ONLY=0
NO_VERIFY=0
for arg in "$@"; do
  [ "$arg" = "--no-verify" ] && NO_VERIFY=1
  [ "$arg" = "--codex-check" ] && CODEX_ONLY=1
done

# ---- teacher: frozen slot table; the qwen backend points at the local vLLM teacher
unset FOREN_ORACLE_SLOT_ENDPOINT FOREN_ORACLE_SLOT_MODEL
export FOREN_ORACLE_SLOT=slots
export FOREN_TEACHER_QWEN_ENDPOINT="$FS_TEACHER_ENDPOINT" FOREN_TEACHER_QWEN_MODEL="$FS_TEACHER_NAME"
export FOREN_ORACLE_RETRY_MAX_S="${FOREN_ORACLE_RETRY_MAX_S:-900}"
if [ -n "${FS_TEACHER_BACKUP:-}" ]; then
  export FOREN_TEACHER_BACKUP="$FS_TEACHER_BACKUP"
fi
if [ -n "${FS_OPENROUTER_API_KEY:-}" ]; then
  export FOREN_TEACHER_STRONG_API_KEY="$FS_OPENROUTER_API_KEY" FOREN_TEACHER_API_KEY="$FS_OPENROUTER_API_KEY"
fi
case ",${FOREN_TEACHER_BACKUP:-}," in
  *,strong,*|*,all,*)
    [ -n "${FOREN_TEACHER_STRONG_API_KEY:-}${FOREN_TEACHER_API_KEY:-}" ] || { fail "FS_TEACHER_BACKUP=strong needs FS_OPENROUTER_API_KEY (OpenRouter Sonnet)"; exit 1; } ;;
  *)
    CODEX="${FOREN_CODEX_BIN:-$(command -v codex || true)}"
    [ -n "$CODEX" ] || { fail "Codex CLI not found (strong backend = codex): see the install steps at the top of server/15_walker.sh"; exit 1; } ;;
esac

if [ "$CODEX_ONLY" = 1 ]; then
  "$FS_ENV_AGENT/bin/python" - <<'PY'
import sys
sys.path.insert(0, "scripts")
import walker_run as w
w.env_setup()
log = w.Log("_codex_check_log.txt", "Codex CLI check (strong backend)")
table = w.slot_table()
ok = all(w.codex_backend_preflight(log, cfg, label=f"{name} backend") for name, cfg in table["backends"].items()
         if (cfg.get("type") or "endpoint") == "codex")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

served_on "$FS_TEACHER_ENDPOINT" "$FS_TEACHER_NAME" || { fail "qwen teacher not serving at $FS_TEACHER_ENDPOINT (bash server/11_loop_layout.sh A)"; exit 1; }

# ---- verifier: frozen = Anthropic direct + batch (walker_run reads specs/r0_freeze.json); FS_WALKER_JUDGE=local keeps the old path
if [ "${FS_WALKER_JUDGE:-}" = "local" ]; then
  if [ "$NO_VERIFY" = 0 ]; then
    served_on "$FS_VERIFIER_ENDPOINT" "$FS_VERIFIER_NAME" || { fail "verifier not up at $FS_VERIFIER_ENDPOINT (or pass --no-verify)"; exit 1; }
  fi
  export FOREN_WALKER_JUDGE=local FOREN_JUDGE_ENDPOINT="$FS_VERIFIER_ENDPOINT" FOREN_WALKER_JUDGE_MODEL="$FS_VERIFIER_NAME"
  JUDGE_TEXT="local $FS_VERIFIER_NAME (not r0-valid)"
else
  unset FOREN_WALKER_JUDGE FOREN_JUDGE_API_ENDPOINT FOREN_JUDGE_API_MODEL
  if [ "$NO_VERIFY" = 0 ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    fail "ANTHROPIC_API_KEY is not set (frozen verifier = Anthropic API, batch); set it in server/env.sh or pass --no-verify"; exit 1
  fi
  JUDGE_TEXT="api-batch claude-sonnet-5-5 (Anthropic direct)"
fi

STRONG_TEXT="codex gpt-6-astra (serialized; --jobs 2)"
[ -n "${FOREN_TEACHER_BACKUP:-}" ] && STRONG_TEXT="backup switch FOREN_TEACHER_BACKUP=$FOREN_TEACHER_BACKUP (see specs/r0_freeze.json backup_backends)"
say "walker: S0-S3 $FS_TEACHER_NAME @ $FS_TEACHER_ENDPOINT; S4/S5 $STRONG_TEXT; verifier $JUDGE_TEXT; tools on GPU $FS_TOOL_GPUS"
CUDA_VISIBLE_DEVICES="$FS_TOOL_GPUS" "$FS_ENV_AGENT/bin/python" scripts/walker_run.py d1 "$@"
