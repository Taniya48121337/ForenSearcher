"""g1（10-18）批处理：按 r1 流水线生成轨迹，S0–S5 全走 Codex（specs/g1_teacher.json），不核验（等同 d1 --no-verify）。

    python scripts\\g1\\g1_run.py --cases fmnv_c_xxx groundlie360_vyyy ...       # 或 --groups C_fmnv B_vcsl（walker_g1_cases.json）
        [--jobs 2] [--deadline 2026-10-18T14:35] [--usage-stop 92] [--disk-floor-gb 60]

每个 run 与 walker_run d1 一样起子进程 `walker_run.py one <案卷> <run> --round r1g1`（同一函数、同一 run 目录布局、同一
run_manifests/<轮次>.jsonl 行），只是派工由这里管，才能做 d1 没有的停派条件：
- Codex 用量（app-server account/rateLimits/read，每 30 min 读一次）≥ --usage-stop → 不再派新 run，在跑的跑完；
- 到 --deadline（窗口结束前 30 min）不再派；E: 剩余 < --disk-floor-gb + 2 GB 不再派；.g1/state/stop 文件存在不再派；
- .g1/state/pause.json 未到期（闸门写的整批暂停）时等到期再派。
失败（返回码非 0 / run 半截 / 槽失败 / 没交报告）只重试 1 次：半截目录改名 *.failed-<时间> 留档。
教师预检：分槽表每个后端发一条最小请求（经闸门）；冻结口径：除 teacher_model 一项（S0–S3 换成 Codex，必然不同）外全部要 ok，
teacher_model 里 S4 / S5 要是 r1_freeze 的主模型、S0–S3 要是 g1_teacher 的 codex_low 模型，否则不开跑。
run 跑完在 run_config.json 顶层补 teacher_override（g1_teacher.json 的路径与 sha256）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
STATE = ROOT / ".g1" / "state"
TEACHER = ROOT / "specs" / "g1_teacher.json"
ROUND = "r1g1"

os.environ["FOREN_TEACHER_FREEZE"] = str(TEACHER)
os.environ["FOREN_ORACLE_SLOT"] = "slots"
os.environ.pop("FOREN_ORACLE_SLOT_ENDPOINT", None)
os.environ.pop("FOREN_TEACHER_BACKUP", None)
os.environ.pop("FOREN_FREEZE_STRICT", None)          # 冻结口径在 preflight() 里按上面的规则自己核
os.environ["FOREN_INLOOP_SUFFICIENCY"] = "1"
os.environ["FOREN_LENS_MAX_SEARCHES"] = "0"           # 任务书：Lens 关；C 族按 r1 策略只 Yandex + 文字搜，B 族不检索
os.environ["FOREN_CLAUDE_CLI"] = str(ROOT / ".g1" / "bin" / "claude_gate.exe")
os.environ["G1_STATE"] = str(STATE)

import walker_run as wr  # noqa: E402
from foren_searcher.datagen.run_cases import run_stats, run_status  # noqa: E402  （d1 里同样是局部导入）

wr.run_stats, wr.run_status = run_stats, run_status


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def jlog(name: str, row: dict[str, Any]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / name).open("a", encoding="utf-8") as h:
        h.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}, ensure_ascii=False) + "\n")


def disk_free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def codex_usage() -> dict[str, Any] | None:
    """Codex app-server 的 account/rateLimits/read（不耗额度）。"""

    codex = shutil.which("codex")
    if not codex:
        return None
    try:
        proc = subprocess.Popen([codex, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    result: dict[str, Any] = {}

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == 2:
                result.update(msg.get("result") or {})
                return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        assert proc.stdin is not None
        for msg in ({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "g1", "version": "0.1"}}},
                    {"method": "initialized"}, {"id": 2, "method": "account/rateLimits/read"}):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            time.sleep(1.0)
        thread.join(20)
    finally:
        proc.kill()
    limits = result.get("rateLimits") or {}
    primary = limits.get("primary") or {}
    if not primary:
        return None
    return {"used_percent": primary.get("usedPercent"), "window_mins": primary.get("windowDurationMins"),
            "resets_at": primary.get("resetsAt"), "secondary": limits.get("secondary"),
            "rate_limit_reached": limits.get("rateLimitReachedType"), "plan": limits.get("planType")}


def pause_left() -> float:
    try:
        return float(json.loads((STATE / "pause.json").read_text(encoding="utf-8")).get("until") or 0) - time.time()
    except (OSError, ValueError):
        return 0.0


def preflight(log: Any) -> bool:
    status = wr.freeze_status(ROUND)
    (wr.SFT / "run_manifests").mkdir(parents=True, exist_ok=True)
    out = wr.SFT / "run_manifests" / f"{ROUND}.freeze.json"
    out.write_text(json.dumps({k: v for k, v in status.items() if k != "tool_layer"} | {"teacher_override": str(TEACHER)},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    if not status.get("applies"):
        log.say(f"[冻结] {ROUND} 没有对应的冻结文件，不开跑")
        return False
    teacher = status["checks"]["teacher_model"]
    others = [k for k in status["failed"] if k != "teacher_model"]
    slots = teacher.get("slots") or {}
    s45_ok = all((slots.get(k) or {}).get("ok") for k in ("S4", "S5"))
    s03_ok = all((slots.get(k) or {}).get("actual") == "gpt-6-astra" for k in ("S0", "S1", "S2", "S3"))
    log.say(f"[冻结] {status['freeze_id']}（{status['file']}）：失败项 {status['failed']}；S4/S5 照 r1_freeze {'ok' if s45_ok else '不对'}；"
            f"S0–S3 = gpt-6-astra（g1_teacher）{'ok' if s03_ok else '不对'}")
    if others or not s45_ok or not s03_ok:
        log.say("[冻结] 除 teacher_model（S0–S3 换 Codex）以外还有不一致，或 S4/S5 不是 r1_freeze 的主模型：不开跑")
        return False
    if not wr.codex_preflight(log):
        return False
    return wr.claude_judge_preflight(log)


class Runner:
    def __init__(self, args: argparse.Namespace, log: Any) -> None:
        self.args, self.log = args, log
        self.lock = threading.Lock()
        self.stop_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.env = wr.env_setup()
        self.manifest = wr.SFT / "run_manifests" / f"{ROUND}.jsonl"
        self.teacher_sha = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
        self.deadline = time.mktime(time.strptime(args.deadline, "%Y-%m-%dT%H:%M")) if args.deadline else None
        self.done = threading.Event()

    # -------------------------------------------------------------- 监控
    def monitor(self) -> None:
        last_usage = last_disk = 0.0
        while not self.done.is_set():
            t = time.time()
            if t - last_usage >= 1800:
                self.usage = codex_usage()
                jlog("usage.jsonl", {"codex": self.usage})
                self.log.say(f"[用量] Codex {(self.usage or {}).get('used_percent')}%（{now()}）")
                last_usage = t
            if t - last_disk >= 3600:
                free = disk_free_gb()
                jlog("disk.jsonl", {"free_gb": round(free, 1)})
                self.log.say(f"[磁盘] E: 剩余 {free:.1f} GB（{now()}）")
                last_disk = t
            self.done.wait(30)

    def should_stop(self) -> str | None:
        if self.stop_reason:
            return self.stop_reason
        reason = None
        used = (self.usage or {}).get("used_percent")
        if used is not None and float(used) >= self.args.usage_stop:
            reason = f"Codex 用量 {used}% ≥ {self.args.usage_stop}%"
        elif self.deadline and time.time() >= self.deadline:
            reason = f"到了停派时间 {self.args.deadline}"
        elif disk_free_gb() < self.args.disk_floor_gb + 2:
            reason = f"E: 剩余 {disk_free_gb():.1f} GB < {self.args.disk_floor_gb + 2} GB"
        elif (STATE / "stop").exists():
            reason = "有 .g1/state/stop"
        if reason:
            with self.lock:
                if not self.stop_reason:
                    self.stop_reason = reason
                    self.log.say(f"[停派] {reason}：不再派新 run，在跑的跑完（{now()}）")
        return self.stop_reason

    # -------------------------------------------------------------- 单个 run
    def needs_rerun(self, run_dir: Path) -> bool:
        status = wr.run_status(run_dir) if run_dir.exists() else "missing"
        return status != "done" or bool(wr.slot_failures(run_dir)) or wr.no_report_finish(run_dir)

    def launch(self, index: int, total: int, case: str, attempt: int) -> tuple[int, float]:
        run_name = f"{case}_{ROUND}_1"
        batch_log = wr.SFT / "batch_logs" / f"{run_name}{'' if attempt == 1 else '.retry'}.log"
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        batch_log.write_text("", encoding="utf-8")
        started = time.monotonic()
        code = self.log.run(f"[{index}/{total}] {run_name}{'（重试）' if attempt > 1 else ''}（日志 {batch_log}）",
                            [wr.PY, "scripts/walker_run.py", "one", str(wr.CASES_DIR / f"{case}.json"), run_name, "--round", ROUND],
                            env=self.env, log_file=batch_log)
        return code, round(time.monotonic() - started, 1)

    def one(self, index: int, total: int, case: str) -> None:
        try:
            self._one(index, total, case)
        except Exception:  # noqa: BLE001 —— 线程池会吞掉异常，这里写进日志
            import traceback

            self.log.say(f"[{index}/{total}] {case}：调度出错\n{traceback.format_exc()}")

    def recorded(self) -> set[str]:
        out = set()
        if self.manifest.is_file():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                try:
                    out.add(json.loads(line)["run"])
                except (ValueError, KeyError):
                    pass
        return out

    def _one(self, index: int, total: int, case: str) -> None:
        run_name = f"{case}_{ROUND}_1"
        run_dir = wr.RUNS / run_name
        if run_dir.exists() and not self.needs_rerun(run_dir):
            if run_name not in self.recorded():          # 10-18 01:29 调度器 bug（run_status 没导入）那几条：补 manifest 行
                self.record(case, run_dir, 0, None, 1, False)
            self.log.say(f"[{index}/{total}] {run_name} 已完成，跳过")
            return
        for attempt in (1, 2):
            while pause_left() > 0 and not self.should_stop():
                time.sleep(min(30.0, pause_left() + 1))
            if self.should_stop():
                return
            if run_dir.exists():
                old = run_dir.with_name(f"{run_name}.failed-{time.strftime('%Y%m%d-%H%M%S')}")
                run_dir.rename(old)
                self.log.say(f"[{index}/{total}] {run_name} 是半截，改名为 {old.name}")
            code, wall = self.launch(index, total, case, attempt)
            failed = code != 0 or self.needs_rerun(run_dir)
            self.record(case, run_dir, code, wall, attempt, failed)
            if not failed:
                return
            self.log.say(f"      {run_name}：第 {attempt} 次没成功（返回码 {code}，状态 {wr.run_status(run_dir) if run_dir.exists() else 'missing'}）"
                         + ("，重试 1 次" if attempt == 1 else "，不再重试"))

    def record(self, case: str, run_dir: Path, code: int, wall: float, attempt: int, failed: bool) -> None:
        cfg_path = run_dir / "run_config.json"
        walker_cfg: dict[str, Any] = {}
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                cfg["teacher_override"] = {"file": "specs/g1_teacher.json", "sha256": self.teacher_sha,
                                           "base_freeze": "specs/r1_freeze.json", "batch": "g1"}
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                walker_cfg = cfg.get("walker") or {}
            except (OSError, ValueError):
                pass
        row = {"run": run_dir.name, "case_id": case, "round": ROUND, "teacher": "oracle_walker", "returncode": code,
               "status": wr.run_status(run_dir) if run_dir.exists() else "missing", "wall_seconds": wall,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt, "failed": failed,
               "teacher_model": walker_cfg.get("teacher_model") or wr.teacher_model(),
               "slot_models": walker_cfg.get("slot_models"), "teacher_role": "primary", "teacher_backup": None,
               "teacher_role_slots": walker_cfg.get("teacher_role"), "teacher_override": "specs/g1_teacher.json",
               **(wr.run_stats(run_dir) if run_dir.exists() else {}), **(wr.walker_stats(run_dir) if run_dir.exists() else {})}
        with self.lock:
            with self.manifest.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log.say(f"      {run_dir.name}：{row['status']}；{wall or 0:.0f}s；教师调用 {row.get('llm_calls')} 次")

    def run(self, cases: list[str]) -> None:
        threading.Thread(target=self.monitor, daemon=True).start()
        time.sleep(3)
        total = len(cases)
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            list(pool.map(lambda item: self.one(item[0], total, item[1]), list(enumerate(cases, 1))))
        self.done.set()
        wr.inloop_summary(ROUND, self.log)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--deadline", default=None, help="本地时间 YYYY-MM-DDTHH:MM，之后不再派新 run")
    parser.add_argument("--usage-stop", type=float, default=92.0)
    parser.add_argument("--disk-floor-gb", type=float, default=60.0)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    cases = list(args.cases)
    if args.groups:
        groups = json.loads((wr.ROOT / wr.SFT / "walker_g1_cases.json").read_text(encoding="utf-8"))["groups"]
        for g in args.groups:
            cases += [c for c in groups.get(g, []) if c not in cases]
    if not cases:
        print("没有案例")
        return 2
    os.chdir(ROOT)
    log = wr.Log(f"_walker_{ROUND}日志.txt", f"g1 批 {ROUND}：{len(cases)} 个案例（S0–S5 Codex，g1_teacher.json；不核验）")
    log.say(f"开始 {now()}；E: 剩余 {disk_free_gb():.1f} GB；jobs {args.jobs}；deadline {args.deadline}；用量停派 {args.usage_stop}%")
    if not args.skip_preflight and not preflight(log):
        return 1
    Runner(args, log).run(cases)
    log.say(f"结束 {now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
