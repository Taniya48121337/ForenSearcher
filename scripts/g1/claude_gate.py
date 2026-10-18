"""g1（10-18）：循环内充分性检查的 Claude CLI 闸门。g1 批处理把 FOREN_CLAUDE_CLI 指向它的启动器（.g1/bin/claude_gate.exe），
ClaudeCliJudge 的每次 `claude -p` 先过这里。流水线代码一行不改。

- 遇到限额 / 429 一次：写 state/pause.json 整批暂停 20 min（codex_gate 也认这个文件，新 Codex 调用一并等），**这次不重试**，
  原样把限额回包交回（judge 抛 ClaudeCliLimit，循环内检查记 limit_hit，run 照常往下走）；
- 暂停未到期时新调用最多等 G1_CLAUDE_PAUSE_WAIT_S（默认 480 s，低于 judge 的 600 s 超时）；还在暂停就直接返回失败
  （stderr 只写 "g1 gate: batch paused"，不含限额字样 → 普通 ClaudeCliError，检查失败不挡 run），不去抢对话 3 的额度；
- 每次调用记一行 state/claude_gate.jsonl。
非 -p 调用（auth status 等）直接转给真 claude。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_gate import STATE, FileLock, pause_until  # noqa: E402

PAUSE_S = 1200.0
PAUSE_WAIT_S = float(os.environ.get("G1_CLAUDE_PAUSE_WAIT_S") or 480)
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "limit reached", "limit will reset", "overloaded", "quota")


def real_claude() -> str:
    path = os.environ.get("G1_REAL_CLAUDE") or shutil.which("claude")
    if not path or Path(path).stem.lower() == "claude_gate":
        sys.stderr.write("claude_gate: real claude not found (set G1_REAL_CLAUDE)\n")
        sys.exit(127)
    return path


def set_pause(seconds: float, reason: str) -> None:
    lock = FileLock(STATE / "pause.lock")
    lock.acquire()
    try:
        until = max(pause_until(), time.time() + seconds)
        (STATE / "pause.json").write_text(json.dumps({"until": until, "reason": reason, "set_at": time.time(),
                                                      "by": "claude_gate"}), encoding="utf-8")
        with (STATE / "pauses.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "claude", "seconds": seconds,
                                "reason": reason[:300]}) + "\n")
    finally:
        lock.release()


def main() -> int:
    argv = sys.argv[1:]
    claude = real_claude()
    if "-p" not in argv:
        return subprocess.call([claude, *argv])
    STATE.mkdir(parents=True, exist_ok=True)
    data = sys.stdin.buffer.read()
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "model": argv[argv.index("--model") + 1]
           if "--model" in argv else None, "pause_wait_s": 0.0}
    started = time.time()
    while pause_until() > time.time() and time.time() - started < PAUSE_WAIT_S:
        time.sleep(min(15.0, pause_until() - time.time() + 0.1))
    row["pause_wait_s"] = round(time.time() - started, 1)
    if pause_until() > time.time():
        row.update({"result": "skipped_paused", "rc": 1})
        with (STATE / "claude_gate.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(row) + "\n")
        sys.stderr.write("g1 gate: batch paused\n")
        return 1
    t0 = time.monotonic()
    done = subprocess.run([claude, *argv], input=data, capture_output=True)
    out, err = done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")
    result = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and '"type":"result"' in line.replace(" ", ""):
            try:
                result = json.loads(line)
            except ValueError:
                pass
    ok = bool(result) and not result.get("is_error") and result.get("subtype") == "success"
    blob = (err + " " + (json.dumps(result) if result else out[-2000:])).lower()
    limited = not ok and ((result or {}).get("api_error_status") == 429 or any(w in blob for w in LIMIT_WORDS))
    row.update({"rc": done.returncode, "result": "ok" if ok else "limit" if limited else "error",
                "elapsed_s": round(time.monotonic() - t0, 1), "cost_usd": (result or {}).get("total_cost_usd")})
    if limited:
        set_pause(PAUSE_S, "claude limit: " + blob.strip()[:200])
        row["batch_pause_s"] = PAUSE_S
    with (STATE / "claude_gate.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(row) + "\n")
    sys.stdout.buffer.write(done.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(done.stderr)
    sys.stderr.buffer.flush()
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
