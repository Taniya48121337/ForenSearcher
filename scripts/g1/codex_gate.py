"""g1（10-18）：Codex CLI 全局闸门。specs/g1_teacher.json 的 codex_bin 指向它的启动器（.g1/bin/codex_gate.exe），
walker 的每次 `codex exec` 都先过这里，再调真正的 codex。流水线代码一行不改。

- 同时在飞 ≤ G1_CODEX_MAX_INFLIGHT（默认 2）：跨进程文件锁槽位（state/slot_<i>.lock）；
- 相邻启动间隔 ≥ G1_CODEX_SPACING_S（默认 2 s）：state/last_start.txt，持 state/start.lock 时读写；
- 429 / 限额 / 5xx：指数退避 10 s 起、上限 300 s、±20% 抖动；退避期间不占槽位；
- 连续 3 次退避到顶（全局计数 state/capped.txt，任一调用成功清零）→ 写 state/pause.json 整批暂停 15 min 再试；
- state/pause.json 未到期时新调用一律等（claude_gate 遇限额也写它，20 min）；
- 每次调用记一行 state/codex_gate.jsonl（等待、退避、尝试次数、返回码、耗时），收尾统计用。
非 exec 子命令（--version、login status）直接转给真 codex。
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("G1_STATE") or Path(__file__).resolve().parents[2] / ".g1" / "state")
MAX_INFLIGHT = int(os.environ.get("G1_CODEX_MAX_INFLIGHT") or 2)
SPACING_S = float(os.environ.get("G1_CODEX_SPACING_S") or 2.0)
ATTEMPT_TIMEOUT_S = float(os.environ.get("G1_CODEX_ATTEMPT_TIMEOUT_S") or 900)
BACKOFF_START, BACKOFF_CAP = 10.0, 300.0
CAPPED_LIMIT, PAUSE_S = 3, 900.0
GIVE_UP_S = float(os.environ.get("G1_CODEX_GIVE_UP_S") or 5400)      # 一次调用累计等待超过 90 min 就把失败原样交回 walker
LIMIT_WORDS = ("429", "too many requests", "rate limit", "rate_limit", "ratelimit", "usage limit", "usage_limit",
               "at capacity", "quota", "try again in")
SERVER_RE = re.compile(r"\b50[0234]\b|internal server error|bad gateway|service unavailable|gateway timeout|overloaded|"
                       r"server_error|stream disconnected|connection reset", re.I)


def real_codex() -> str:
    path = os.environ.get("G1_REAL_CODEX") or shutil.which("codex")
    if not path or Path(path).stem.lower() == "codex_gate":
        sys.stderr.write("codex_gate: real codex not found (set G1_REAL_CODEX)\n")
        sys.exit(127)
    return path


class FileLock:
    """msvcrt 字节锁（Windows）/ fcntl（其他）；blocking=False 拿不到返回 False。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def acquire(self, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+b")
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                if not blocking:
                    self.handle.close()
                    self.handle = None
                    return False
                time.sleep(0.2)

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _read_float(path: Path, default: float = 0.0) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip() or default)
    except (OSError, ValueError):
        return default


def pause_until() -> float:
    try:
        return float(json.loads((STATE / "pause.json").read_text(encoding="utf-8")).get("until") or 0)
    except (OSError, ValueError):
        return 0.0


def set_pause(seconds: float, reason: str) -> None:
    lock = FileLock(STATE / "pause.lock")
    lock.acquire()
    try:
        until = max(pause_until(), time.time() + seconds)
        (STATE / "pause.json").write_text(json.dumps({"until": until, "reason": reason, "set_at": time.time(),
                                                      "by": "codex_gate"}), encoding="utf-8")
        with (STATE / "pauses.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "codex", "seconds": seconds,
                                "reason": reason[:300]}) + "\n")
    finally:
        lock.release()


def bump_capped(reset: bool = False) -> int:
    lock = FileLock(STATE / "capped.lock")
    lock.acquire()
    try:
        value = 0 if reset else int(_read_float(STATE / "capped.txt")) + 1
        (STATE / "capped.txt").write_text(str(value), encoding="utf-8")
        return value
    finally:
        lock.release()


def wait_pause() -> float:
    waited = 0.0
    while True:
        left = pause_until() - time.time()
        if left <= 0:
            return waited
        step = min(left, 30.0)
        time.sleep(step)
        waited += step


def take_slot() -> tuple[FileLock, float]:
    started = time.monotonic()
    while True:
        for i in range(MAX_INFLIGHT):
            lock = FileLock(STATE / f"slot_{i}.lock")
            if lock.acquire(blocking=False):
                return lock, time.monotonic() - started
        time.sleep(0.5)


def respect_spacing() -> float:
    lock = FileLock(STATE / "start.lock")
    lock.acquire()
    try:
        gap = time.time() - _read_float(STATE / "last_start.txt")
        slept = 0.0
        if gap < SPACING_S:
            slept = SPACING_S - gap
            time.sleep(slept)
        (STATE / "last_start.txt").write_text(f"{time.time():.3f}", encoding="utf-8")
        return slept
    finally:
        lock.release()


def classify(rc: int, out: str, err: str) -> str:
    if rc == 0:
        return "ok"
    blob = (out[-6000:] + " " + err[-4000:]).lower()
    if any(w in blob for w in LIMIT_WORDS):
        return "limit"
    if SERVER_RE.search(blob):
        return "server"
    return "error"


def log_call(row: dict) -> None:
    with (STATE / "codex_gate.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False) + "\n")


def arg_after(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main() -> int:
    argv = sys.argv[1:]
    codex = real_codex()
    if not argv or argv[0] != "exec":
        return subprocess.call([codex, *argv])
    STATE.mkdir(parents=True, exist_ok=True)
    data = sys.stdin.buffer.read()
    started = time.time()
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "dir": arg_after(argv, "-C"),
           "model": arg_after(argv, "-m"), "effort": next((a for a in argv if a.startswith("model_reasoning_effort")), None),
           "images": argv.count("--image"), "pause_wait_s": 0.0, "slot_wait_s": 0.0, "spacing_s": 0.0, "backoffs": [],
           "attempts": 0}
    backoff_n = 0
    while True:
        row["pause_wait_s"] += wait_pause()
        slot, waited = take_slot()
        row["slot_wait_s"] += waited
        try:
            row["spacing_s"] += respect_spacing()
            row["attempts"] += 1
            t0 = time.monotonic()
            try:
                done = subprocess.run([codex, *argv], input=data, capture_output=True, timeout=ATTEMPT_TIMEOUT_S)
                rc, out, err = done.returncode, done.stdout, done.stderr
            except subprocess.TimeoutExpired:
                rc, out, err = 124, b"", f"codex exec timed out ({ATTEMPT_TIMEOUT_S:.0f}s)".encode()
            row.setdefault("attempt_s", []).append(round(time.monotonic() - t0, 1))
        finally:
            slot.release()
        kind = classify(rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"))
        if kind in ("ok", "error") or time.time() - started > GIVE_UP_S:
            if kind == "ok":
                bump_capped(reset=True)
            row.update({"rc": rc, "result": kind if time.time() - started <= GIVE_UP_S or kind == "ok" else f"gave_up_{kind}",
                        "elapsed_s": round(time.time() - started, 1)})
            log_call(row)
            sys.stdout.buffer.write(out)
            sys.stdout.buffer.flush()
            sys.stderr.buffer.write(err)
            sys.stderr.buffer.flush()
            return rc
        delay = min(BACKOFF_CAP, BACKOFF_START * 2 ** backoff_n)
        backoff_n += 1
        capped = delay >= BACKOFF_CAP
        delay *= random.uniform(0.8, 1.2)
        row["backoffs"].append({"kind": kind, "sleep_s": round(delay, 1), "capped": capped,
                                "tail": (err.decode("utf-8", "replace") or out.decode("utf-8", "replace"))[-200:]})
        if capped and bump_capped() >= CAPPED_LIMIT:
            bump_capped(reset=True)
            set_pause(PAUSE_S, f"3 consecutive capped backoffs ({kind})")
            row["backoffs"][-1]["batch_pause_s"] = PAUSE_S
            continue
        time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
