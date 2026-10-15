"""Run the EVID setup on GroundLie360 and FMNV (2026-10-12): double-click scripts\\_bench_run.cmd.

Qwen3.8-27B Q4 without thinking, merged tool catalog (10-12: DETECT folded into SEARCH / INSPECT; hi@0.8.0 / gia@0.9.0,
reverse image search on), 3 rounds, compact_16k, eval guards, fact-check sites blocked, 2 runs in parallel.
(Until 10-12 this was the evidC27G agent with DETECT v2.4; those rounds were g27a / f27a.) Searches are
recorded (replay_or_record) under experiments/bench/<dataset>/search_snapshots so a rerun replays them.

    FOREN_BENCH_N        samples per dataset (default 20 = pilot; 100 for the larger run)
    FOREN_BENCH_SETS     comma list, default "g360,fmnv"; datasets without videos on disk are skipped
    FOREN_BENCH_ROUND    round tag suffix (default "27m"; rounds are g<suffix> / f<suffix>)

Close the window any time and double-click again: finished runs are kept. Log: _BENCH日志.txt. Scores:
experiments/bench/<dataset>/scores/<round>.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _evid_night as N  # noqa: E402
import _launch_evid as L  # noqa: E402
from _bench_common import BENCH, DIRS, video_index  # noqa: E402
from _launch import PY, ROOT, env_setup  # noqa: E402

N_SAMPLES = int(os.environ.get("FOREN_BENCH_N") or 20)
SETS = tuple(s.strip() for s in (os.environ.get("FOREN_BENCH_SETS") or "g360,fmnv").split(",") if s.strip())
SUFFIX = os.environ.get("FOREN_BENCH_ROUND") or "27m"
MERGED_SPECS = ("hi@0.8.0", "gia@0.9.0")


def round_of(ds: str) -> str:
    return ("g" if ds == "g360" else "f") + SUFFIX


def run_dir(ds: str, sample_id: str) -> Path:
    from foren_searcher.datagen.run_cases import _safe

    return ROOT / BENCH.relative_to(ROOT) / ds / "runs" / f"{_safe(ds)}_{_safe(sample_id)}_{_safe(round_of(ds))}_1"


def done(ds: str, sample_id: str) -> bool:
    path = run_dir(ds, sample_id)
    return (path / "result.json").is_file() and (path / "execution_links.json").is_file()


def run_command(ds: str, ids: list[str]) -> list[str]:
    specs = MERGED_SPECS
    base = BENCH.relative_to(ROOT) / ds
    return [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(base / "cases"), "--runs-dir", str(base / "runs"),
            "--split", "test", "--round", round_of(ds), "--endpoint", L.OLLAMA, "--model", L.EVID27_MODEL,
            "--no-think",
            "--jobs", L.EVID8_JOBS, "--instruction-mode", "student", "--retry-failed", "--blocklist", "exclude",
            "--hi-spec", specs[0], "--gia-spec", specs[1],
            "--search-mode", "replay_or_record", "--snapshot-root", str(base / "search_snapshots"),
            "--runtime-guards", L.EVID_GUARDS, "--state-renderer", L.EVID_RENDER,
            "--ids", *(f"{ds}:{i}" for i in ids)]


def main() -> int:
    env_setup()
    log = N.NightLog("_BENCH日志.txt", f"GroundLie360 / FMNV：每个 {N_SAMPLES} 个（{', '.join(round_of(d) for d in SETS)}）")
    print(f"其他数据集测试：{', '.join(DIRS[d] for d in SETS)}，每个 {N_SAMPLES} 个样本。可以随时关窗口，再双击续跑。日志 {log.path.name}")
    if N.free_gb() < N.MIN_FREE_GB:
        log.finish(f"磁盘只剩 {N.free_gb():.1f}GB，少于 {N.MIN_FREE_GB}GB，停。")
        return 1
    code = log.run("[0] 接入单元测试", [PY, "-m", "pytest", "tests/test_bench_20261012.py", "-q", "-o", "addopts=",
                                     "-p", "no:cacheprovider", "-p", "no:asyncio"], show=True)
    if code != 0:
        log.finish("接入单元测试没过，停。")
        return 1
    if L.EVID27_MODEL not in (L.ollama_models() or []) or not L.evid27_probe(log):
        log.finish(f"27B（{L.EVID27_MODEL}）不可用，停。")
        return 1
    os.environ["FOREN_CONFIG"] = str(L.EVID27_CONFIG)
    import json

    for ds in SETS:
        if not video_index(ds):
            log.say(f"[{DIRS[ds]}] plan-harness-1\\datasets\\{DIRS[ds]}\\ 下没有视频文件，跳过（先下载视频）。")
            continue
        if log.run(f"[{DIRS[ds]}] 标注 → 案卷（{N_SAMPLES} 个）",
                   [PY, "scripts/_bench_build_cases.py", "--dataset", ds, "--n", str(N_SAMPLES)], show=True) != 0:
            continue
        ids = json.loads((BENCH / ds / f"selection_n{N_SAMPLES}.json").read_text(encoding="utf-8"))
        todo = [i for i in ids if not done(ds, i)]
        if todo:
            log.run(f"[{DIRS[ds]}] 跑 {len(todo)} 个（轮次 {round_of(ds)}）", run_command(ds, todo), show=True)
        failed = [i for i in ids if not done(ds, i)]
        if failed and len(failed) <= max(3, len(ids) // 5):
            log.run(f"[{DIRS[ds]}] 失败的 {len(failed)} 个重跑一次", run_command(ds, failed), show=True)
        log.run(f"[{DIRS[ds]}] 评分", [PY, "scripts/_bench_score.py", "--dataset", ds, "--round", round_of(ds),
                                       "--n", str(N_SAMPLES)], show=True)
    log.finish("完成。分数在 experiments\\bench\\<数据集>\\scores\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
