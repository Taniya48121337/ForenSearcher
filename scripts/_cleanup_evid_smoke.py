"""删除 EVID 冒烟 / 修链路轮次的 run 目录（10-06，负责人确认只删这一类）。永久删除，不进回收站。

    python scripts/_cleanup_evid_smoke.py            # 预览：列出要删的目录与大小，不动文件
    python scripts/_cleanup_evid_smoke.py --apply    # 执行删除

只删 experiments/evid/runs 下名字为 evid_video_<编号>_<轮次>_<序号> 且轮次属于 SMOKE_ROUNDS 的目录：
    evidb0–evidb3、evidb5、evidb7–evidb11（001 / 027 / 151 三个已降级样本上的冒烟与修下载、修 OCR 等链路的轮次）。
不动：evidb4、evidb6、evidb12、evidb12bf、evidq27*（基线与对照）；scores/、cases/、search_snapshots/；batch_logs。
轮次用整段精确匹配，evidb1 不会误中 evidb12。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "evid" / "runs"
LOG = ROOT / "_EVID冒烟清理日志.txt"
SMOKE_ROUNDS = ("evidb0", "evidb1", "evidb2", "evidb3", "evidb5", "evidb7", "evidb8", "evidb9", "evidb10", "evidb11")
PATTERN = re.compile(r"^evid_video_\d{3}_(?P<round>[A-Za-z0-9]+)_\d+$")


def targets() -> list[Path]:
    if not RUNS.is_dir():
        raise SystemExit(f"找不到 {RUNS}")
    found = []
    for path in sorted(RUNS.iterdir()):
        match = PATTERN.match(path.name)
        if path.is_dir() and not path.is_symlink() and match and match.group("round") in SMOKE_ROUNDS:
            found.append(path)
    return found


def size_of(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                pass
    return total


def _writable(func, target, _exc) -> None:     # Windows 只读文件：去掉只读再删
    os.chmod(target, stat.S_IWRITE)
    func(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真的删除；不加只预览")
    args = parser.parse_args()
    items = targets()
    lines = [f"==== EVID 冒烟轮次清理（{'执行' if args.apply else '预览'}） {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"轮次：{', '.join(SMOKE_ROUNDS)}", ""]
    total = 0
    by_round: dict[str, list[int]] = {}
    sized = []
    for path in items:
        size = size_of(path)
        total += size
        rnd = PATTERN.match(path.name).group("round")
        by_round.setdefault(rnd, []).append(size)
        sized.append((path, size))
    for rnd in SMOKE_ROUNDS:
        sizes = by_round.get(rnd, [])
        lines.append(f"  {rnd:8s} {len(sizes):3d} 个 run  {sum(sizes) / 2**30:6.2f} GB")
    lines.append(f"  合计     {len(items):3d} 个 run  {total / 2**30:6.2f} GB")
    lines.append("")
    deleted, failed = 0, []
    for path, size in sized:
        if not args.apply:
            lines.append(f"  [将删] {path.relative_to(ROOT).as_posix()}  {size / 2**20:.1f} MB")
            continue
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_writable)
            else:
                shutil.rmtree(path, onerror=_writable)
            deleted += size
            lines.append(f"  [已删] {path.relative_to(ROOT).as_posix()}  {size / 2**20:.1f} MB")
        except OSError as exc:
            failed.append(path.name)
            lines.append(f"  [失败] {path.relative_to(ROOT).as_posix()}：{exc}")
    lines.append("")
    if args.apply:
        lines.append(f"释放 {deleted / 2**30:.2f} GB；失败 {len(failed)} 个" + (f"：{', '.join(failed)}" if failed else ""))
    else:
        lines.append("这只是预览，没有删除任何文件。")
    text = "\n".join(lines) + "\n"
    LOG.write_text(text, encoding="utf-8")
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
