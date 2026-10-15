"""项目清理第二轮（2026-10-14）。永久删除，不进回收站。

    python scripts/_cleanup_disk_20261014.py              # 先只读扫描预览，再逐项问可选组，最后输入 y 才删
    python scripts/_cleanup_disk_20261014.py --dry-run    # 只扫描预览，不问、不删

固定删除（一次确认）：
  E1 已被取代的 EVID 轮次整目录：evidC27、evidC27b、evidC27c、evidC27d（DETECT v2 开发小轮）、evidC27F（v2.1 全量，
     已被 C27G 取代）、evidD27、evidD8（DETECT v1）；以及 evid / sft_data 两处 runs 下的 *.failed-* 目录。
     这些轮次不在 _merge_ab.py 的 ALL_ROUNDS 里，分数文件留在 scores/。
  E2 其余 EVID run 里的重复大文件（目录、store.sqlite3、trajectory、request/response/transport.json 都不动）：
       - requests/*/actor/ 里与本 run artifacts/ 中某张图逐字节相同的附图副本；
       - artifacts/ 和 source_match/ 里与检索快照（search_snapshots_final / search_snapshots 同编号目录）的某个 blob、
         或 plan-harness-1 原视频逐字节相同的视频。
     跳过 evidM27g / evidM27h（修复 4 正在跑），以及 3 小时内有改动的 run。
  M1 不再用的检测器权重与数据：.models/lit 下除 Skyra-RL 以外的全部（dda、dinov2-main、trufor、TruFor-main、sanity、
     _zips、effort）；.models/aigi 全部（UnivFD / NPR / SAFE / CommFor / CLIP-L / dinov2）；.models/deepfake 下除 pylib
     以外的全部（dfdc、sbi、dfb、originals、DeepfakeBench）。pylib 里的 facenet-pytorch 是 compare_media 人脸比对在用的，保留。
  T1 测试临时目录：.pytest-detect-*、.pytest-tmp、outputs/pytest_tmp 下一天前的子目录。
  S1 轨迹分叉试点的分支目录：sft_data/runs 下名字含 __f<数字> / __tp1 / __tp2 的目录（tp2 后已改方向）。
     各自的自由跑主轨迹（*_tp0_1、*_tp2_1）和反事实变体（*_tp*n_1、*_tp*h_1）保留。

逐项询问（默认不删）：
  O1 .models/lit/Skyra-RL（唯一在无原片时有信号的 AI 视频检测器，但没有接进 agent）
  O2 walker 早期轮次 wk1–wk6 的 run 目录（D1 迭代，已被 wk8 之后的轮次取代；导出和核验文件不动）
  O3 walker GT 提示对照臂 wk11b/g/n、wk12b/g/n 的 run 目录（pack_sft 默认排除这些臂；导出和核验文件不动）

不动：evidF27 / evidF8 / evidC27G / evidC27e / evidM27a–h 的 run 目录（_merge_ab 读它们的 store.sqlite3）、
search_snapshots_final、scores、cases、plan-harness-1 的三个数据集、sft_data 的 r0* / q* / validation / lcost / wk8–wk13、
sft_data 的 export、tool_cache、search_snapshots、corpus、.models/inspect、.models/deepfake/pylib、src、tests、docs、server。
不跟随符号链接或目录联接。
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
EVID = ROOT / "experiments" / "evid"
EVID_RUNS = EVID / "runs"
SFT_RUNS = ROOT / "experiments" / "sft_data" / "runs"
SNAP_DIRS = (EVID / "search_snapshots_final", EVID / "search_snapshots")
VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
MODELS = ROOT / ".models"
LOG = ROOT / "_磁盘清理日志_20261014.txt"

EVID_PATTERN = re.compile(r"^evid_video_(?P<case>\d{3})_(?P<round>[A-Za-z0-9]+)_\d+$")
DEAD_EVID_ROUNDS = {"evidC27", "evidC27b", "evidC27c", "evidC27d", "evidC27F", "evidD27", "evidD8"}
SKIP_EVID_ROUNDS = {"evidM27g", "evidM27h"}
FORK_BRANCH = re.compile(r"__(f\d|tp1|tp2)")
WALKER_TAG = re.compile(r"_(?P<tag>wk\d+[a-z]?)_\d+$")
EARLY_WALKER = {f"wk{i}" for i in range(1, 7)}
ARM_WALKER = {"wk11b", "wk11g", "wk11n", "wk12b", "wk12g", "wk12n"}

KEEP_IN = {MODELS / "lit": {"Skyra-RL"}, MODELS / "aigi": set(), MODELS / "deepfake": {"pylib"}}
SKYRA = MODELS / "lit" / "Skyra-RL"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv"}
SAMPLE = 1 << 20
RECENT_HOURS = 3
GB = 1e9   # 十进制 GB；资源管理器显示 GiB，会小约 7%


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


def walk_files(path: Path):
    """不跟随链接 / 联接地遍历文件，产出 (Path, size)。"""
    if is_link(path):
        return
    if path.is_file():
        try:
            yield path, path.stat().st_size
        except OSError:
            pass
        return
    stack = [path]
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
                            yield Path(entry.path), entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass


def tree_size(path: Path) -> int:
    return sum(size for _, size in walk_files(path))


def newest_mtime(path: Path) -> float:
    newest = 0.0
    try:
        newest = path.stat().st_mtime
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    pass
    except OSError:
        pass
    return newest


def _writable(func, target, _exc) -> None:     # Windows 只读文件：去掉只读再删
    os.chmod(target, stat.S_IWRITE)
    func(target)


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def remove_tree(path: Path) -> None:
    if path.is_file():
        remove_file(path)
    elif sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_writable)
    else:
        shutil.rmtree(path, onerror=_writable)


def same_bytes(a: Path, b: Path, size: int) -> bool:
    """小文件整份比，大文件比头 / 中 / 尾各 1MB。"""
    try:
        with a.open("rb") as fa, b.open("rb") as fb:
            if size <= 4 * SAMPLE:
                return fa.read() == fb.read()
            for offset in (0, size // 2 - SAMPLE // 2, size - SAMPLE):
                fa.seek(offset)
                fb.seek(offset)
                if fa.read(SAMPLE) != fb.read(SAMPLE):
                    return False
        return True
    except OSError:
        return False


def find_twin(path: Path, size: int, pool: dict[int, list[Path]]) -> bool:
    return any(same_bytes(path, other, size) for other in pool.get(size, ()) if other != path)


def run_of(path: Path) -> Path:
    for parent in path.parents:
        if parent.parent == EVID_RUNS:
            return parent
    return path.parent


def gb(n: int) -> str:
    return f"{n / GB:7.2f} GB"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- 扫描

@dataclass
class Group:
    key: str
    title: str
    dirs: list[tuple[Path, int]] = field(default_factory=list)     # 整目录 / 单个文件
    files: list[tuple[Path, int]] = field(default_factory=list)    # 逐个文件（E2）

    @property
    def size(self) -> int:
        return sum(s for _, s in self.dirs) + sum(s for _, s in self.files)


def snapshot_pool(case: str, cache: dict[str, dict[int, list[Path]]]) -> dict[int, list[Path]]:
    if case not in cache:
        pool: dict[int, list[Path]] = {}
        for base in SNAP_DIRS:
            folder = base / f"evid_video_{case}"
            if folder.is_dir():
                for path, size in walk_files(folder):
                    if size >= SAMPLE:
                        pool.setdefault(size, []).append(path)
        source = VIDEOS / f"video_{case}.mp4"
        if source.is_file():
            pool.setdefault(source.stat().st_size, []).append(source)
        cache[case] = pool
    return cache[case]


def strip_candidates(run: Path, case: str, cache) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    artifacts = run / "artifacts"
    art_images: dict[int, list[Path]] = {}
    if artifacts.is_dir() and not is_link(artifacts):
        for path, size in walk_files(artifacts):
            if path.suffix.lower() in IMAGE_EXT:
                art_images.setdefault(size, []).append(path)
    requests = run / "requests"
    if requests.is_dir() and not is_link(requests) and art_images:
        for request in requests.glob("request_*"):
            actor = request / "actor"
            if not actor.is_dir() or is_link(actor):
                continue
            for path, size in walk_files(actor):
                if path.suffix.lower() in IMAGE_EXT and find_twin(path, size, art_images):
                    out.append((path, size))
    pool = snapshot_pool(case, cache)
    for sub in ("artifacts", "source_match"):
        folder = run / sub
        if not folder.is_dir() or is_link(folder):
            continue
        for path, size in walk_files(folder):
            if size >= SAMPLE and path.suffix.lower() in VIDEO_EXT and find_twin(path, size, pool):
                out.append((path, size))
    return out


def sized(path: Path) -> tuple[Path, int]:
    return path, tree_size(path)


def scan(say) -> dict[str, Group]:
    groups = {k: Group(k, t) for k, t in (
        ("E1", "已被取代的 EVID 轮次整目录 + 失败目录"),
        ("E2", "其余 EVID run 里的重复附图与重复视频"),
        ("M1", "不再用的检测器权重与数据"),
        ("T1", "测试临时目录"),
        ("S1", "轨迹分叉试点的分支目录"),
        ("O1", "Skyra-RL 权重"),
        ("O2", "walker 早期轮次 wk1–wk6"),
        ("O3", "walker 对照臂 wk11b/g/n、wk12b/g/n"),
    )}
    now = time.time()
    skipped_recent: list[str] = []

    # E1 / E2
    if EVID_RUNS.is_dir():
        runs = sorted(p for p in EVID_RUNS.iterdir() if p.is_dir() and not is_link(p))
        cache: dict[str, dict[int, list[Path]]] = {}
        for index, run in enumerate(runs, 1):
            if ".failed-" in run.name:
                groups["E1"].dirs.append(sized(run))
                continue
            match = EVID_PATTERN.match(run.name)
            if not match:
                continue
            rnd, case = match.group("round"), match.group("case")
            if rnd in DEAD_EVID_ROUNDS:
                groups["E1"].dirs.append(sized(run))
            elif rnd in SKIP_EVID_ROUNDS:
                continue
            elif now - newest_mtime(run) < RECENT_HOURS * 3600:
                skipped_recent.append(run.name)
            else:
                if case not in cache:          # run 按名字排序，同一编号挨在一起；只留当前编号的快照索引
                    cache.clear()
                groups["E2"].files.extend(strip_candidates(run, case, cache))
            if index % 50 == 0 or index == len(runs):
                say(f"  EVID run 已扫描 {index}/{len(runs)}")

    # S1 / O2 / O3 / sft 失败目录
    if SFT_RUNS.is_dir():
        for run in sorted(p for p in SFT_RUNS.iterdir() if p.is_dir() and not is_link(p)):
            if now - newest_mtime(run) < RECENT_HOURS * 3600:
                continue
            if ".failed-" in run.name:
                groups["E1"].dirs.append(sized(run))
            elif FORK_BRANCH.search(run.name):
                groups["S1"].dirs.append(sized(run))
            else:
                tag = WALKER_TAG.search(run.name)
                if tag and tag.group("tag") in EARLY_WALKER:
                    groups["O2"].dirs.append(sized(run))
                elif tag and tag.group("tag") in ARM_WALKER:
                    groups["O3"].dirs.append(sized(run))
        say("  sft_data run 已扫描")

    # M1 / O1
    for parent, keep in KEEP_IN.items():
        if parent.is_dir() and not is_link(parent):
            for child in sorted(parent.iterdir()):
                if child.name not in keep and not is_link(child):
                    groups["M1"].dirs.append(sized(child))
    if SKYRA.is_dir() and not is_link(SKYRA):
        groups["O1"].dirs.append(sized(SKYRA))

    # T1
    for child in sorted(ROOT.iterdir()):
        if child.is_dir() and not is_link(child) and (child.name.startswith(".pytest-detect") or child.name == ".pytest-tmp"):
            groups["T1"].dirs.append(sized(child))
    pytest_tmp = ROOT / "outputs" / "pytest_tmp"
    if pytest_tmp.is_dir() and not is_link(pytest_tmp):
        for child in sorted(pytest_tmp.iterdir()):
            if not is_link(child) and now - newest_mtime(child) > 24 * 3600:
                groups["T1"].dirs.append(sized(child))

    if skipped_recent:
        say(f"  {len(skipped_recent)} 个 EVID run {RECENT_HOURS} 小时内有改动，本次跳过：{', '.join(skipped_recent[:8])}"
            + (" …" if len(skipped_recent) > 8 else ""))
    return groups


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
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    def flush() -> None:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n\n")
        lines.clear()

    if not EVID_RUNS.is_dir():
        print(f"找不到 {EVID_RUNS}，停止。")
        return 2
    say(f"==== 项目清理第二轮 {'预览 ' if args.dry_run else ''}{time.strftime('%Y-%m-%d %H:%M:%S')}  根目录 {ROOT}")
    say("扫描中（只读；比对重复文件要读文件头尾，约需几分钟到十几分钟）……")
    groups = scan(say)

    say("")
    for key in ("E1", "E2", "M1", "T1", "S1", "O1", "O2", "O3"):
        group = groups[key]
        count = len(group.dirs) + len(group.files)
        tag = "可选" if key.startswith("O") else "固定"
        say(f"  [{tag}] {key} {group.title}：{count} 项 {gb(group.size)}")
        if key in ("M1", "T1", "O1"):
            for path, size in group.dirs:
                say(f"        {gb(size)}  {rel(path)}")
    rounds: dict[str, int] = {}
    for path, size in groups["E1"].dirs:
        match = EVID_PATTERN.match(path.name)
        name = match.group("round") if match else "失败目录"
        rounds[name] = rounds.get(name, 0) + size
    say("  E1 按轮次：" + "，".join(f"{k} {v / GB:.1f}" for k, v in sorted(rounds.items())) + "（GB）")
    fixed = sum(groups[k].size for k in ("E1", "E2", "M1", "T1", "S1"))
    say(f"  固定部分合计 {gb(fixed)}")
    say("")

    if args.dry_run:
        say("这只是预览，没有删除任何文件。")
        flush()
        return 0

    chosen = ["E1", "E2", "M1", "T1", "S1"]
    if groups["O1"].size and ask(f"O1 删除 Skyra-RL 权重（{gb(groups['O1'].size).strip()}，以后要用得重新下载）？(y/N) "):
        chosen.append("O1")
    if groups["O2"].size and ask(f"O2 删除 walker wk1–wk6 的 run 目录（{gb(groups['O2'].size).strip()}）？(y/N) "):
        chosen.append("O2")
    if groups["O3"].size and ask(f"O3 删除 walker wk11b/g/n、wk12b/g/n 的 run 目录（{gb(groups['O3'].size).strip()}）？(y/N) "):
        chosen.append("O3")
    total = sum(groups[k].size for k in chosen)
    say(f"选择：{' '.join(chosen)}；合计将释放约 {gb(total)}（永久删除，不进回收站）")
    if not ask("输入 y 开始删除，其他任意键取消："):
        say("已取消，没有删除任何文件。")
        flush()
        return 1

    freed, failed = 0, []
    for key in chosen:
        group = groups[key]
        for path, size in group.dirs:
            if is_link(path) or not inside_root(path) or not path.exists():
                continue
            try:
                remove_tree(path)
                freed += size
                say(f"  [已删 {key}] {rel(path)}  {size / 1e6:.1f} MB")
            except OSError as exc:
                failed.append(rel(path))
                say(f"  [失败 {key}] {rel(path)}：{exc}")
        by_run: dict[Path, list[int]] = {}
        for path, size in group.files:
            try:
                if path.exists() and not is_link(path):
                    remove_file(path)
                    freed += size
                    by_run.setdefault(run_of(path), []).append(size)
            except OSError as exc:
                failed.append(rel(path))
                say(f"  [失败 {key}] {rel(path)}：{exc}")
        for folder, sizes in sorted(by_run.items()):
            say(f"  [已删 {key}] {rel(folder)}  {len(sizes)} 个文件 {sum(sizes) / 1e6:.1f} MB")
        flush()
        print(f"  {key} 完成，累计释放 {gb(freed)}", flush=True)

    say("")
    say(f"完成：释放 {gb(freed)}；失败 {len(failed)} 项" + ("（见上面 [失败] 行）" if failed else ""))
    say(f"日志：{LOG}")
    flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
