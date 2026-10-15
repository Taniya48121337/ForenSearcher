"""其他数据集的视频准备（2026-10-13）。双击 scripts\\_datasets_prepare.cmd，或：

    python scripts\\datasets_prepare.py g360  [--zip E:\\Downloads\\vid.zip]     # 解压 GroundLie360 视频
    python scripts\\datasets_prepare.py fmnv  [--classes real,ft] [--limit N]   # 用 yt-dlp 下 FMNV 能下的那部分
    python scripts\\datasets_prepare.py status                                  # 只看现状

GroundLie360（g360）：
- 解压到 plan-harness-1/datasets/GroundLie360/videos/（保留压缩包里的子目录；已存在且大小相同的跳过，可断点续）。
- 先看磁盘：目标盘剩余空间要 ≥ 解压后总大小 + 5 GB，否则不解压、只报差多少。
- 压缩包里套压缩包的，也逐个解开到同名目录。
- 解完对照标注（data.csv.gz）：帖子视频（video_id）、原片（original_videoid）、佐证视频（evidence_videoid）、辟谣视频
  （debunking_videoid）各找到多少；写索引 GroundLie360/videos_index.json（编号 → 相对路径）。
- 不删压缩包（解压核对无误后要不要删由你定）。

FMNV（fmnv）：
- 只下 video_id 是 YouTube 编号的；默认两类：真新闻（label=true）与 CD 类假新闻（class=ft：真实视频配了不符的标题，
  视频本身没改，所以原片就是数据集里的那段）。放 FMNV/yt/<video_id>.<扩展名>，与 bench 脚本"文件名 = video_id"的查找一致。
- 另外三类假新闻（fv 剪掉关键段、fa 换配音、fc 荒诞夸大）的视频是作者加工过的，网上没有；它们的 video_id 是加工前的
  原片。--with-originals 时把这些原片下到 FMNV_originals/<class>/（不放进 FMNV/，免得 bench 把原片当成假视频），
  以后可以与网盘里的假视频配成 A 族（找原片）案例。
- 推特编号（纯数字）默认跳过；--twitter 时试 https://x.com/i/status/<id>（常要登录，失败照记）。
- 断点续：已下的跳过；下载结果逐条记 FMNV/yt_download_log.jsonl；明确下不了的（已删除、私享、地区限制）下次不再试，
  --retry-failed 才重试。
- 剩余空间低于 --min-free-gb（默认 30）就停。单条超过 --max-duration 秒（默认 900）不下。
日志：_数据集准备日志.txt
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATASETS = ROOT / "plan-harness-1" / "datasets"
G360 = DATASETS / "GroundLie360"
FMNV = DATASETS / "FMNV"
FMNV_ORIGINALS = DATASETS / "FMNV_originals"
LOG = ROOT / "_数据集准备日志.txt"
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi", ".flv")
YT_ID = re.compile(r"^[\w-]{11}$")
_TERMINAL = ("private", "removed", "unavailable", "terminated", "copyright", "not available in your country",
             "members-only", "has been removed", "does not exist", "longer than")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


# ---------------------------------------------------------------------------- GroundLie360

def _ids(value: Any) -> list[str]:
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) and value.strip() else []
    except (ValueError, SyntaxError):
        parsed = []
    return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else []


def g360_rows() -> list[dict[str, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(G360 / "data.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _extract(archive: zipfile.ZipFile, target: Path, label: str) -> tuple[int, int, list[Path]]:
    """解压一个 zip（流式写 .part 再改名）；返回 (新解出个数, 跳过个数, 解出的内层 zip)。"""

    done = skipped = 0
    nested: list[Path] = []
    members = [m for m in archive.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX/")
               and not Path(m.filename).name.startswith("._")]
    for index, member in enumerate(members, 1):
        parts = [p for p in Path(member.filename).parts if p not in ("..", "/", "\\")]
        dest = target.joinpath(*parts)
        if dest.exists() and dest.stat().st_size == member.file_size:
            skipped += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with archive.open(member) as src, part.open("wb") as out:
                shutil.copyfileobj(src, out, 4 * 1024 * 1024)
            part.replace(dest)
            done += 1
        if dest.suffix.lower() == ".zip":
            nested.append(dest)
        if index % 100 == 0 or index == len(members):
            log(f"  {label}：{index}/{len(members)}（新解出 {done}，已有跳过 {skipped}）")
    return done, skipped, nested


SEVEN_ZIP = [Path(r"C:\Program Files\7-Zip\7z.exe"), Path(r"C:\Program Files (x86)\7-Zip\7z.exe")]


def _seven_zip(zip_path: Path, target: Path) -> int:
    """10-14：Python zipfile 打不开（wk8 实测 vid.zip 一行没解就停了）时改用本机 7-Zip：-aos 已有文件跳过，可断点续。"""

    import subprocess

    exe = next((p for p in SEVEN_ZIP if p.is_file()), None) or (Path(shutil.which("7z")) if shutil.which("7z") else None)
    if exe is None:
        log("没有找到 7-Zip（C:\\Program Files\\7-Zip\\7z.exe），无法改用 7-Zip 解压")
        return 3
    existing = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    need = max(0.0, (zip_path.stat().st_size * 1.05 - existing) / 1024**3) + 5.0
    have = free_gb(target)
    log(f"改用 7-Zip：{exe}；压缩包 {zip_path.stat().st_size / 1024**3:.1f} GB，已解出 {existing / 1024**3:.1f} GB；"
        f"{target.anchor} 剩余 {have:.1f} GB，大约需要 {need:.1f} GB")
    if have < need:
        log(f"空间不够，差 {need - have:.1f} GB，没有解压。腾出空间后再运行一次。")
        return 2
    started = time.time()
    proc = subprocess.run([str(exe), "x", "-aos", "-y", f"-o{target}", str(zip_path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:] + (proc.stderr or "").strip().splitlines()[-6:])
    log(f"7-Zip 结束（返回码 {proc.returncode}，{(time.time() - started) / 60:.1f} 分钟）：\n{tail}")
    if proc.returncode in (0, 1):                         # 1 = 有警告（例如个别文件跳过）
        return 0
    if "Unexpected end of archive" in out:
        # 10-14：vid.zip 没下完整（文件在中途截断，没有结尾目录）。截断点之前的文件都是完整的，
        # 只有报错的最后一个文件是半截：删掉它，其余照常建索引
        import re as _re

        for name in _re.findall(r"Unexpected end of data : (.+)", out):
            partial = target / name.strip().replace("\\", "/")
            if partial.is_file():
                partial.unlink()
                log(f"删掉半截文件 {partial.name}")
        n = sum(1 for p in target.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        log(f"压缩包不完整（下载没下完，在中途截断）：截断点之前的 {n} 个视频可用，照常建索引。要全部视频需重新下载 vid.zip。")
        return 0
    return 4


def g360_unzip(zip_path: Path) -> int:
    log(f"==== GroundLie360 解压 {now()}：{zip_path}")
    target = G360 / "videos"
    if not zip_path.is_file():
        have = [x for x in target.rglob("*") if x.is_file() and x.suffix.lower() in VIDEO_SUFFIXES] if target.is_dir() else []
        if have:
            # 10-14：压缩包已删掉，但以前解出的视频还在（1314 个）：直接用它们建索引
            log(f"找不到 {zip_path}，但 {target} 下已有 {len(have)} 个视频，直接建索引")
            return g360_index()
        log(f"找不到 {zip_path}")
        return 1
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.infolist()
    except Exception as exc:  # noqa: BLE001 —— 记下原因再改用 7-Zip
        log(f"Python zipfile 打不开 {zip_path.name}：{type(exc).__name__}: {str(exc)[:200]}")
        code = _seven_zip(zip_path, target)
        if code:
            return code
        return _g360_nested(target, 0, 0, [p for p in target.rglob("*.zip")])
    try:
        return _g360_zipfile(zip_path, target)
    except Exception as exc:  # noqa: BLE001 —— 例如 Deflate64 等 zipfile 不支持的压缩方法
        log(f"Python zipfile 解压中途出错：{type(exc).__name__}: {str(exc)[:200]}；改用 7-Zip（已解出的文件跳过）")
        code = _seven_zip(zip_path, target)
        if code:
            return code
        return _g360_nested(target, 0, 0, [p for p in target.rglob("*.zip")])


def _g360_zipfile(zip_path: Path, target: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        total = sum(m.file_size for m in archive.infolist() if not m.is_dir())
        existing = sum(p.stat().st_size for p in target.rglob("*") if p.is_file() and not p.name.endswith(".part"))
        need = max(0.0, (total - existing) / 1024**3) + 5.0
        have = free_gb(target)
        log(f"压缩包 {zip_path.stat().st_size / 1024**3:.1f} GB，解压后 {total / 1024**3:.1f} GB（已解出 {existing / 1024**3:.1f} GB）；"
            f"{target.anchor} 剩余 {have:.1f} GB，需要 {need:.1f} GB")
        if have < need:
            log(f"空间不够，差 {need - have:.1f} GB，没有解压。腾出空间后再运行一次。")
            return 2
        done, skipped, nested = _extract(archive, target, "vid.zip")
    return _g360_nested(target, done, skipped, nested)


def _g360_nested(target: Path, done: int, skipped: int, nested: list[Path]) -> int:
    for inner in nested:
        sub = inner.with_suffix("")
        try:
            with zipfile.ZipFile(inner) as archive:
                d, s, _ = _extract(archive, sub, inner.name)
            log(f"  内层 {inner.name}：新解出 {d}，跳过 {s}")
            inner.unlink()          # 内层 zip 是刚从 vid.zip 解出来的副本，解开后删掉省空间（vid.zip 本身不动）
        except zipfile.BadZipFile:
            log(f"  内层 {inner.name} 不是有效 zip，跳过")
        except Exception as exc:  # noqa: BLE001
            log(f"  内层 {inner.name} 用 zipfile 解不开（{type(exc).__name__}），改用 7-Zip")
            if _seven_zip(inner, sub) == 0:
                inner.unlink()
    log(f"解压完成：新解出 {done}，已有跳过 {skipped}")
    return g360_index()


def g360_index() -> int:
    target = G360 / "videos"
    files = [p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    by_stem: dict[str, str] = {}
    for path in files:
        by_stem.setdefault(path.stem, str(path.relative_to(G360)).replace("\\", "/"))
    (G360 / "videos_index.json").write_text(json.dumps(by_stem, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = g360_rows()
    posts = {str(r["video_id"]).strip() for r in rows}
    groups = {"帖子视频 video_id": posts,
              "原片 original_videoid": {i for r in rows for i in _ids(r.get("original_videoid"))},
              "佐证视频 evidence_videoid": {i for r in rows for i in _ids(r.get("evidence_videoid"))},
              "辟谣视频 debunking_videoid": {i for r in rows for i in _ids(r.get("debunking_videoid"))}}
    log(f"索引：{len(files)} 个视频文件，{len(by_stem)} 个不同编号 → {G360 / 'videos_index.json'}")
    for name, ids in groups.items():
        hit = len(ids & set(by_stem))
        log(f"  {name}：标注 {len(ids)} 个，找到 {hit} 个")
    known = set().union(*groups.values())
    unknown = sorted(set(by_stem) - known)
    if unknown:
        log(f"  不在标注里的文件 {len(unknown)} 个，例如：{', '.join(unknown[:8])}")
    fakes = [r for r in rows if r.get("first_level_annotation") == "Fake" and str(r["video_id"]).strip() in by_stem]
    pairs = [r for r in fakes if any(i in by_stem for i in _ids(r.get("original_videoid")))]
    log(f"  可直接用的：假帖子有视频 {len(fakes)} 个，其中原片也在的 {len(pairs)} 个（可做两段比对 / 找原片）")
    return 0


# ---------------------------------------------------------------------------- FMNV

def fmnv_downloaded() -> set[str]:
    stems = set()
    for base in (FMNV, FMNV_ORIGINALS):
        if base.is_dir():
            stems |= {p.stem for p in base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES}
    return stems


def fmnv_log_rows() -> dict[str, dict[str, Any]]:
    path = FMNV / "yt_download_log.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            out[str(row.get("video_id"))] = row
    return out


def _download_youtube(video_id: str, folder: Path, max_duration: float) -> dict[str, Any]:
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{video_id}.download"
    media = ytdlp_download(f"https://www.youtube.com/watch?v={video_id}", tmp,
                           InspectPolicy(maximum_duration_s=max_duration))
    final = folder / f"{video_id}{media.suffix or '.mp4'}"
    Path(media.path).replace(final)
    return {"file": str(final.relative_to(DATASETS)).replace("\\", "/"), "bytes": final.stat().st_size,
            "duration_s": media.details.get("duration_s"), "method": media.details.get("method")}


def _download_twitter(video_id: str, folder: Path, max_duration: float) -> dict[str, Any]:
    from yt_dlp import YoutubeDL

    folder.mkdir(parents=True, exist_ok=True)
    options = {"format": "b[ext=mp4]/b", "outtmpl": str(folder / f"{video_id}.%(ext)s"), "noplaylist": True,
               "quiet": True, "no_warnings": True, "noprogress": True, "ignoreconfig": True, "socket_timeout": 30,
               "match_filter": lambda info, *, incomplete=False: (
                   "too long" if isinstance(info.get("duration"), (int, float)) and info["duration"] > max_duration else None)}
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://x.com/i/status/{video_id}", download=True)
    files = sorted(folder.glob(f"{video_id}.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp produced no file")
    return {"file": str(files[0].relative_to(DATASETS)).replace("\\", "/"), "bytes": files[0].stat().st_size,
            "duration_s": (info or {}).get("duration"), "method": "yt-dlp twitter"}


def fmnv_download(args: argparse.Namespace) -> int:
    log(f"==== FMNV 用 yt-dlp 下载 {now()}：classes={args.classes} limit={args.limit} with_originals={args.with_originals} "
        f"twitter={args.twitter}")
    rows = json.loads((FMNV / "data.json").read_text(encoding="utf-8"))
    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    have = fmnv_downloaded()
    logged = fmnv_log_rows()
    plan: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        cls = "real" if row.get("label") == "true" else str(row.get("class") or "")
        vid = str(row["video_id"])
        is_yt = bool(YT_ID.match(vid))
        if not is_yt and not (args.twitter and vid.isdigit()):
            continue
        if cls in wanted:
            folder = FMNV / "yt"
        elif args.with_originals and cls in ("fv", "fa", "fc"):
            folder = FMNV_ORIGINALS / cls
        else:
            continue
        if vid in have:
            continue
        prev = logged.get(vid)
        if prev and prev.get("status") == "failed" and prev.get("terminal") and not args.retry_failed:
            continue
        plan.append((row, folder))
    if args.limit:
        plan = plan[:args.limit]
    log(f"要下 {len(plan)} 个（已有 {len(have)} 个跳过）")
    stats = {"ok": 0, "failed": 0}
    started = time.monotonic()
    with (FMNV / "yt_download_log.jsonl").open("a", encoding="utf-8") as out:
        for index, (row, folder) in enumerate(plan, 1):
            if free_gb(FMNV) < args.min_free_gb:
                log(f"剩余空间低于 {args.min_free_gb} GB，停在第 {index} 个")
                break
            vid = str(row["video_id"])
            cls = "real" if row.get("label") == "true" else row.get("class")
            record: dict[str, Any] = {"video_id": vid, "class": cls, "at": now()}
            try:
                info = (_download_youtube if YT_ID.match(vid) else _download_twitter)(vid, folder, args.max_duration)
                record.update(status="ok", **info)
                stats["ok"] += 1
            except Exception as exc:  # noqa: BLE001 —— 一条失败不影响其余
                message = str(exc)[:300]
                record.update(status="failed", error=f"{type(exc).__name__}: {message}",
                              terminal=any(t in message.lower() for t in _TERMINAL))
                stats["failed"] += 1
                for leftover in folder.glob(f"{vid}.download*"):
                    leftover.unlink(missing_ok=True)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if index % 10 == 0 or index == len(plan):
                rate = (time.monotonic() - started) / index
                log(f"  {index}/{len(plan)}：成功 {stats['ok']}，失败 {stats['failed']}；平均 {rate:.0f} s/个；"
                    f"剩余空间 {free_gb(FMNV):.0f} GB")
            time.sleep(args.pause)
    log(f"FMNV 下载结束：成功 {stats['ok']}，失败 {stats['failed']}。明细 {FMNV / 'yt_download_log.jsonl'}")
    return 0


def status() -> int:
    log(f"==== 数据集现状 {now()}")
    g = [p for p in (G360 / "videos").rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES] if (G360 / "videos").is_dir() else []
    log(f"GroundLie360：{len(g)} 个视频文件")
    if g:
        g360_index()
    rows = json.loads((FMNV / "data.json").read_text(encoding="utf-8"))
    have = fmnv_downloaded()
    by: dict[str, list[int]] = {}
    for row in rows:
        cls = "real" if row.get("label") == "true" else str(row.get("class"))
        entry = by.setdefault(cls, [0, 0, 0])
        entry[0] += 1
        entry[1] += bool(YT_ID.match(str(row["video_id"])))
        entry[2] += str(row["video_id"]) in have
    for cls, (n, yt, got) in sorted(by.items()):
        log(f"FMNV {cls}：{n} 条，其中 YouTube 编号 {yt}，本地已有 {got}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    g = sub.add_parser("g360")
    g.add_argument("--zip", default=r"E:\Downloads\vid.zip")
    f = sub.add_parser("fmnv")
    f.add_argument("--classes", default="real,ft", help="逗号分隔：real（真新闻）、ft（CD 类，视频未改）")
    f.add_argument("--with-originals", action="store_true", help="另把 fv / fa / fc 的原片下到 FMNV_originals/")
    f.add_argument("--twitter", action="store_true", help="也试推特编号（常要登录）")
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--retry-failed", action="store_true")
    f.add_argument("--min-free-gb", type=float, default=30.0)
    f.add_argument("--max-duration", type=float, default=900.0)
    f.add_argument("--pause", type=float, default=1.0, help="每条之间停几秒")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.job == "g360":
            return g360_unzip(Path(args.zip))
        if args.job == "fmnv":
            return fmnv_download(args)
        return status()
    except Exception:  # noqa: BLE001 —— 10-14：以前出错只打在窗口里，日志里看不到（wk8 时 g360 一行没解就停了）
        import traceback

        log(f"出错（{args.job}）：\n{traceback.format_exc()[-3000:]}")
        return 9


if __name__ == "__main__":
    raise SystemExit(main())
