"""执行 _待删除清单_20260924.md（不做永久删除）。

    python scripts/_cleanup_20260924.py            # 预览：只列出要动的文件，写 _待删除清单执行记录.txt
    python scripts/_cleanup_20260924.py --apply    # 执行

执行内容（路径相对 experiments/sft_data）：
  [删]   二、export/r0_true4124903_hi@0.1.0_gia@0.1.0_compact_8k/；五、verdicts_r0*.jsonl(+summary)、verdicts_r0_judged_pilot*
         → 剪切到仓库根目录 _trash_20260924/<原相对路径>
  [归档] 三、各 run / gia_decisions / hi_terminal_repairs 下的 assessment_* 与 selected_assessment.json；
         四、construction_validation*.json、pilot_results.json、试验结果.md、batch_outcomes.jsonl、batch_logs/*_assessment.log
         → 打包 _archive/pilot_assessments_20260915.zip（zip 内保留相对路径），逐个比对 CRC 与大小后，
           原件剪切到 _trash_20260924/archived_originals/<原相对路径>
  一、corpus/ 按清单要求保留到 Demo 文案改完，本脚本不动。
  _trash_20260924 由用户在跑完 pytest 与 Demo 检查后自己清空。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
TRASH = ROOT / "_trash_20260924"
ARCHIVE = ROOT / "_archive" / "pilot_assessments_20260915.zip"
LOG = ROOT / "_待删除清单执行记录.txt"

DELETE = [
    "export/r0_true4124903_hi@0.1.0_gia@0.1.0_compact_8k",
    "verdicts_r0.jsonl", "verdicts_r0.jsonl.summary.json",
    "verdicts_r0_judged_pilot.jsonl", "verdicts_r0_judged_pilot.jsonl.summary.json",
]
ARCHIVE_TOP = [
    "construction_validation.json", "construction_validation_binding.json", "construction_validation_current_input.json",
    "construction_validation_r3_bridge.json", "construction_validation_r3_text.json", "construction_validation_report_r7.json",
    "construction_validation_source_policy.json", "pilot_results.json", "试验结果.md", "batch_outcomes.jsonl",
]
ASSESSMENT_PARENTS = ("runs", "gia_decisions", "hi_terminal_repairs")
KEEP_NEVER = ("corpus", "cases", "candidate_exclusions.json", "rubric_calibration", "verdicts_r0_judged.jsonl",
              "verdicts_units_judged.jsonl")


def archive_items() -> list[Path]:
    items: list[Path] = [SFT / name for name in ARCHIVE_TOP]
    for parent in ASSESSMENT_PARENTS:
        base = SFT / parent
        if not base.is_dir():
            continue
        for run in sorted(p for p in base.iterdir() if p.is_dir()):
            items += sorted(p for p in run.iterdir() if p.name.startswith("assessment_") and p.is_dir())
            items += [run / "selected_assessment.json"]
    logs = SFT / "batch_logs"
    if logs.is_dir():
        items += sorted(logs.glob("*_assessment.log"))
    return [p for p in items if p.exists()]


def files_under(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())


def rel(path: Path) -> str:
    return path.relative_to(SFT).as_posix()


def guard(path: Path) -> None:
    first = path.relative_to(SFT).parts[0]
    if first in KEEP_NEVER:
        raise SystemExit(f"拒绝移动保留项：{path}")


def move(path: Path, dest_root: Path, out: list[str]) -> None:
    dest = dest_root / "experiments" / "sft_data" / rel(path)
    if dest.exists():
        raise SystemExit(f"目标已存在，停止以免覆盖：{dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    out.append(f"  移动 {rel(path)} → {dest.relative_to(ROOT).as_posix()}")


def crc_of(path: Path) -> int:
    crc = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    lines: list[str] = [f"模式：{'执行' if args.apply else '预览（未改动任何文件）'}", ""]
    deletes = [SFT / p for p in DELETE if (SFT / p).exists()]
    missing = [p for p in DELETE if not (SFT / p).exists()]
    archives = archive_items()
    for p in deletes + archives:
        guard(p)
    del_files = [f for p in deletes for f in files_under(p)]
    arc_files = [f for p in archives for f in files_under(p)]
    size = lambda fs: sum(f.stat().st_size for f in fs) / 1e6  # noqa: E731
    lines.append(f"[删] {len(deletes)} 项，{len(del_files)} 个文件，{size(del_files):.1f} MB → {TRASH.name}\\")
    lines += [f"  {rel(p)}" for p in deletes]
    if missing:
        lines.append(f"  （清单里有、磁盘上已不存在：{', '.join(missing)}）")
    lines.append(f"[归档] {len(archives)} 项，{len(arc_files)} 个文件，{size(arc_files):.1f} MB → {ARCHIVE.relative_to(ROOT)}")
    lines += [f"  {rel(p)}" for p in archives]
    lines.append("[不动] corpus\\（清单要求等 Demo 文案改完再处理）")
    if not args.apply:
        LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    if ARCHIVE.exists():
        raise SystemExit(f"{ARCHIVE} 已存在，停止（不覆盖旧压缩包）")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in arc_files:
            zf.write(f, rel(f))
    # 校验：重新打开，逐个比对 CRC 与大小，并做 testzip
    with zipfile.ZipFile(ARCHIVE) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"压缩包校验失败：{bad}；原件未动")
        infos = {i.filename: i for i in zf.infolist()}
        for f in arc_files:
            info = infos.get(rel(f))
            if info is None or info.file_size != f.stat().st_size or info.CRC != crc_of(f):
                raise SystemExit(f"压缩包内 {rel(f)} 与原件不一致；原件未动")
    lines.append(f"压缩包校验通过：{len(arc_files)} 个文件，{ARCHIVE.stat().st_size / 1e6:.1f} MB")
    moved: list[str] = []
    for p in deletes:
        move(p, TRASH, moved)
    for p in archives:
        move(p, TRASH / "archived_originals", moved)
    lines += ["", "已移动："] + moved
    lines += ["", f"完成。请跑 scripts\\_本机回归测试.cmd 和 Demo 检查，没问题再手动清空 {TRASH.name}\\。"]
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
