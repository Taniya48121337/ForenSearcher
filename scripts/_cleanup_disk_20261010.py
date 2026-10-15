"""E 盘清理（2026-10-10，负责人确认"删掉"）。永久删除，不进回收站。

    python scripts/_cleanup_disk_20261010.py              # 先扫描预览，再逐项问可选组，最后输入 y 才删
    python scripts/_cleanup_disk_20261010.py --dry-run    # 只扫描预览，不问、不删

固定删除（一次确认）：
  A  experiments/evid/runs 下每个 run 的 requests/request_*/actor/ 里的附图副本（与 artifacts/ 里的图重复；
     request.json / response.json / transport.json、trajectory、result 都不动，评分与行为面板照常可重算）
  B  已结束轮次的整个 run 目录：evidb4、evidb6、evidq27a、evidDsmoke、evidb12bf、evidsA4/A5/A8/A8G/A8T/A9，
     以及 experiments/evid/search_snapshots_arms（只给这几组对照臂回放用）
  C  旧回收站：_trash_20260915、_trash_20260924、_archive
  D  不再用的模型与数据：.models 下 Qwen3-0.6B、Qwen3-1.7B、Qwen3-14B-bnb-4bit、inspect/Qwen2.5-Omni-3B、
     inspect/Qwen3-VL-2B-Instruct；external/VCSL/vcsl_isc_features_download；VidDiffBench 中断拉取留下的 4.7GB LFS 对象；
     experiments/sft_data/_verifier035_backup、_verifier032_r0c

逐项询问（默认不删）：
  O1 evidb12 全部 run 目录（Q4 全量，002–145）
  O2 .models/Qwen3.5-9B
  O3 experiments/sft_data/export（SFT 导出数据，可用 export_sft 重导）
  O4 保留下来的 run 的 artifacts/ 里与 plan-harness-1 原视频逐字节相同的视频副本（按大小 + 头/中/尾各 1MB 比对）

不动：D27 / D8 / F27 / F8 正式 baseline 的 run（只删其附图副本）、scores/、cases/、search_snapshots/、
search_snapshots_final/、plan-harness-1 原视频、.models/inspect 里的 clip / faster-whisper / grounding-dino、.models/faceswap、
src/、tests/、docs/。不跟随符号链接或目录联接。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "evid" / "runs"
VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
LOG = ROOT / "_磁盘清理日志_20261010.txt"
PATTERN = re.compile(r"^evid_video_(?P<case>\d{3})_(?P<round>[A-Za-z0-9]+)_\d+$")

DONE_ROUNDS = ("evidb4", "evidb6", "evidq27a", "evidDsmoke", "evidb12bf",
               "evidsA4", "evidsA5", "evidsA8", "evidsA8G", "evidsA8T", "evidsA9")
OPTIONAL_ROUND = "evidb12"
KEEP_ROUNDS = ("evidD27", "evidD8", "evidF27", "evidF8")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
SAMPLE = 1 << 20

FIXED_B_EXTRA = ["experiments/evid/search_snapshots_arms"]
FIXED_C = ["_trash_20260915", "_trash_20260924", "_archive"]
FIXED_D = [
    ".models/Qwen3-0.6B",
    ".models/Qwen3-1.7B",
    ".models/Qwen3-14B-bnb-4bit",
    ".models/inspect/Qwen2.5-Omni-3B",
    ".models/inspect/Qwen3-VL-2B-Instruct",
    "external/VCSL/vcsl_isc_features_download",
    "external/VidDiffBench/.git/lfs/objects/24/84/248496186897846d3c1249ca87a13619f91217509f9b72e54f734d955dd34619",
    "experiments/sft_data/_verifier035_backup",
    "experiments/sft_data/_verifier032_r0c",
]
OPT_O2 = [".models/Qwen3.5-9B"]
OPT_O3 = ["experiments/sft_data/export"]

GB = 1e9   # 十进制 GB；资源管理器显示的是 GiB，会小约 7%


# --------------------------------------------------------------------------- 基础

def is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        junction = getattr(path, "is_junction", None)       # Python 3.12+
        return bool(junction and junction())
    except OSError:
        return True


def inside_root(path: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(ROOT)))
        return True
    except ValueError:
        return False


def tree_size(path: Path) -> int:
    if is_link(path):
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total, stack = 0, [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def _writable(func, target, _exc) -> None:     # Windows 只读文件：去掉只读再删
    os.chmod(target, stat.S_IWRITE)
    func(target)


def remove_tree(path: Path) -> None:
    if path.is_file():
        remove_file(path)
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_writable)
    else:
        shutil.rmtree(path, onerror=_writable)


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def same_samples(a: Path, b: Path, size: int) -> bool:
    offsets = sorted({0, max(0, size // 2 - SAMPLE // 2), max(0, size - SAMPLE)})
    try:
        with a.open("rb") as fa, b.open("rb") as fb:
            for offset in offsets:
                fa.seek(offset)
                fb.seek(offset)
                if fa.read(SAMPLE) != fb.read(SAMPLE):
                    return False
        return True
    except OSError:
        return False


def gb(n: int) -> str:
    return f"{n / GB:7.2f} GB"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- 扫描

@dataclass
class Run:
    path: Path
    case: str
    round: str
    size: int = 0
    images: list[Path] = field(default_factory=list)
    image_bytes: int = 0
    videos: list[Path] = field(default_factory=list)
    video_bytes: int = 0


def scan_run(run: Run) -> None:
    run.size = tree_size(run.path)
    requests = run.path / "requests"
    if requests.is_dir() and not is_link(requests):
        for request in sorted(requests.glob("request_*")):
            actor = request / "actor"
            if not actor.is_dir() or is_link(actor):
                continue
            for item in actor.iterdir():
                if item.suffix.lower() in IMAGE_EXT and item.is_file() and not is_link(item):
                    run.images.append(item)
                    run.image_bytes += item.stat().st_size
    source = VIDEOS / f"video_{run.case}.mp4"
    artifacts = run.path / "artifacts"
    if source.is_file() and artifacts.is_dir() and not is_link(artifacts):
        size = source.stat().st_size
        for item in artifacts.iterdir():
            try:
                if (item.suffix.lower() in VIDEO_EXT and item.is_file() and not is_link(item)
                        and item.stat().st_size == size and same_samples(item, source, size)):
                    run.videos.append(item)
                    run.video_bytes += size
            except OSError:
                pass


def scan_runs(say) -> list[Run]:
    runs = []
    for path in sorted(RUNS.iterdir()):
        match = PATTERN.match(path.name)
        if match and path.is_dir() and not is_link(path):
            runs.append(Run(path, match.group("case"), match.group("round")))
    for index, run in enumerate(runs, 1):
        scan_run(run)
        if index % 50 == 0 or index == len(runs):
            say(f"  已扫描 {index}/{len(runs)} 个 run")
    return runs


def sized_paths(names: list[str]) -> list[tuple[Path, int]]:
    out = []
    for name in names:
        path = ROOT / name
        if not path.exists() or is_link(path) or not inside_root(path):
            continue
        out.append((path, tree_size(path)))
    return out


def recent_activity(minutes: int = 20) -> list[str]:
    cutoff = time.time() - minutes * 60
    hits = []
    for base in (RUNS, ROOT / "experiments" / "evid" / "search_snapshots_final",
                 ROOT / "experiments" / "evid" / "search_snapshots"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            try:
                if child.stat().st_mtime > cutoff:
                    hits.append(rel(child))
            except OSError:
                pass
    return hits[:10]


# --------------------------------------------------------------------------- 主流程

def ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "是")
    except EOFError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只扫描预览，不删除")
    args = parser.parse_args()

    log_lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        log_lines.append(text)

    def flush_log() -> None:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines) + "\n\n")
        log_lines.clear()

    if not RUNS.is_dir():
        print(f"找不到 {RUNS}，停止。")
        return 2

    say(f"==== E 盘清理 {'预览' if args.dry_run else ''} {time.strftime('%Y-%m-%d %H:%M:%S')}  根目录 {ROOT}")
    busy = recent_activity()
    if busy:
        say("注意：下面这些目录 20 分钟内有改动，可能有实验在跑：")
        for item in busy:
            say(f"  {item}")
        if not args.dry_run and not ask("确定没有实验在跑、继续吗？(y/N) "):
            say("已取消。")
            flush_log()
            return 1
    if not VIDEOS.is_dir():
        say(f"注意：找不到原视频目录 {rel(VIDEOS)}，O4（原视频副本）将跳过。")

    say("扫描 run 目录（只读，约需几分钟）……")
    runs = scan_runs(say)
    by_round: dict[str, list[Run]] = {}
    for run in runs:
        by_round.setdefault(run.round, []).append(run)

    b_runs = [r for r in runs if r.round in DONE_ROUNDS]
    o1_runs = by_round.get(OPTIONAL_ROUND, [])
    b_extra = sized_paths(FIXED_B_EXTRA)
    c_items = sized_paths(FIXED_C)
    d_items = sized_paths(FIXED_D)
    o2_items = sized_paths(OPT_O2)
    o3_items = sized_paths(OPT_O3)

    say("")
    say("run 目录按轮次（整目录 / 其中请求附图副本 / 其中原视频副本）：")
    for name in sorted(by_round):
        rs = by_round[name]
        tag = "整轮删除" if name in DONE_ROUNDS else ("可选 O1" if name == OPTIONAL_ROUND else "保留目录")
        say(f"  {name:11s} {len(rs):4d} 个  {gb(sum(r.size for r in rs))}  附图 {gb(sum(r.image_bytes for r in rs))}"
            f"  原视频 {gb(sum(r.video_bytes for r in rs))}  [{tag}]")
    say("")

    def show(items: list[tuple[Path, int]]) -> None:
        for path, size in items:
            say(f"      {gb(size)}  {rel(path)}")

    say("固定删除：")
    say(f"  B  已结束轮次 {len(b_runs)} 个 run 目录 {gb(sum(r.size for r in b_runs))}，另加：")
    show(b_extra)
    say("  C  旧回收站：")
    show(c_items)
    say("  D  不再用的模型与数据：")
    show(d_items)
    say("  A  保留下来的 run 里的请求附图副本：数量取决于 O1，见最后汇总")
    say("")
    say("可选：")
    say(f"  O1 evidb12 {len(o1_runs)} 个 run 目录 {gb(sum(r.size for r in o1_runs))}")
    say("  O2 Qwen3.5-9B：")
    show(o2_items)
    say("  O3 SFT 导出数据：")
    show(o3_items)
    say("  O4 保留下来的 run 里的原视频副本：数量取决于 O1，见最后汇总")
    say("")

    if args.dry_run:
        kept = [r for r in runs if r.round not in DONE_ROUNDS and r.round != OPTIONAL_ROUND]
        say(f"若 O1 不删：A ≈ {gb(sum(r.image_bytes for r in kept + o1_runs))}，O4 ≈ {gb(sum(r.video_bytes for r in kept + o1_runs))}")
        say(f"若 O1 删：  A ≈ {gb(sum(r.image_bytes for r in kept))}，O4 ≈ {gb(sum(r.video_bytes for r in kept))}")
        say("这只是预览，没有删除任何文件。")
        flush_log()
        return 0

    o1 = ask(f"O1 删除 evidb12 全部 {len(o1_runs)} 个 run？(y/N) ")
    o2 = ask("O2 删除 .models/Qwen3.5-9B？(y/N) ")
    o3 = ask("O3 删除 experiments/sft_data/export（SFT 导出数据）？(y/N) ")
    o4 = ask("O4 删除保留下来的 run 里的原视频副本？(y/N) ")

    whole_runs = b_runs + (o1_runs if o1 else [])
    whole_set = {r.path for r in whole_runs}
    kept_runs = [r for r in runs if r.path not in whole_set]
    dirs: list[tuple[str, Path, int]] = [("B", r.path, r.size) for r in b_runs]
    dirs += [("B", p, s) for p, s in b_extra] + [("C", p, s) for p, s in c_items] + [("D", p, s) for p, s in d_items]
    if o1:
        dirs += [("O1", r.path, r.size) for r in o1_runs]
    if o2:
        dirs += [("O2", p, s) for p, s in o2_items]
    if o3:
        dirs += [("O3", p, s) for p, s in o3_items]
    a_bytes = sum(r.image_bytes for r in kept_runs)
    v_bytes = sum(r.video_bytes for r in kept_runs) if o4 else 0
    total = sum(s for _, _, s in dirs) + a_bytes + v_bytes

    say(f"选择：O1={'删' if o1 else '留'} O2={'删' if o2 else '留'} O3={'删' if o3 else '留'} O4={'删' if o4 else '留'}")
    say(f"整目录 / 单文件 {len(dirs)} 项 {gb(sum(s for _, _, s in dirs))}")
    say(f"A 请求附图副本（{len(kept_runs)} 个保留的 run）{gb(a_bytes)}")
    if o4:
        say(f"O4 原视频副本 {gb(v_bytes)}")
    say(f"合计将释放约 {gb(total)}（永久删除，不进回收站）")
    if not ask("输入 y 开始删除，其他任意键取消：") :
        say("已取消，没有删除任何文件。")
        flush_log()
        return 1

    freed, failed = 0, []
    say("")
    for group, path, size in dirs:
        if is_link(path) or not inside_root(path):
            say(f"  [跳过] {rel(path)}（链接或不在根目录内）")
            continue
        try:
            remove_tree(path)
            freed += size
            say(f"  [已删 {group}] {rel(path)}  {size / 1e6:.1f} MB")
        except OSError as exc:
            failed.append(rel(path))
            say(f"  [失败 {group}] {rel(path)}：{exc}")
    flush_log()

    for index, run in enumerate(kept_runs, 1):
        files = list(run.images) + (list(run.videos) if o4 else [])
        done = 0
        for item in files:
            try:
                size = item.stat().st_size
                remove_file(item)
                done += size
            except FileNotFoundError:
                pass
            except OSError as exc:
                failed.append(rel(item))
                say(f"  [失败] {rel(item)}：{exc}")
        freed += done
        if files:
            say(f"  [已删 A{'+O4' if o4 and run.videos else ''}] {rel(run.path)}  {len(files)} 个文件 {done / 1e6:.1f} MB")
        if index % 50 == 0:
            print(f"  进度 {index}/{len(kept_runs)}，已释放 {gb(freed)}", flush=True)
            flush_log()

    say("")
    say(f"完成：释放 {gb(freed)}；失败 {len(failed)} 项" + ("（见上面 [失败] 行）" if failed else ""))
    say(f"日志：{LOG}")
    flush_log()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
