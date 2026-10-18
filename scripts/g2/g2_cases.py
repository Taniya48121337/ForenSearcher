"""g2（10-18 / 19）覆盖补量建卷：VideoSham 全集（解压、索引、预留评测 60 对、B 族 20 条）+ 按教学矩阵缺口格定向补 FMNV / GroundLie360 / FMNV-A。

    python scripts\\g2\\g2_cases.py vs_index                 # 解压 videosham_full/_zips → media/{edited,source}，写 index.json
    python scripts\\g2\\g2_cases.py vs_reserve --n 60        # 预留 60 对 → experiments/sft_data/reserved_eval_ids.json
    python scripts\\g2\\g2_cases.py videosham --n 20         # B 族（模板同 r0g / g1 的 build_videosham）
    python scripts\\g2\\g2_cases.py fmnv_c --n-fake 3 --n-real 3 | g360 ... | fmnv_a --n 5   # 照 g1 模板，媒体放 cases/media/g2
    python scripts\\g2\\g2_cases.py list                     # 写 experiments/sft_data/walker_g2_cases.json

建卷逻辑全部复用 scripts/g1/g1_cases.py（g1 的声明规则、DROP、标签、需求、证据写法一字不改），只换：媒体目录 cases/media/g2、
建卷记录 g2_cases_made.json、频道表 g2_channels.json、日志 claude/_g2_work/g2_cases.log；VideoSham 数据目录换成全集。
去重同 g1：主目录 experiments/sft_data/cases 与本工作区（含 r1 / g1 入库的案卷）的 family_key / 文件名 / 媒体路径 / notes 里的编号；
EVID（222 条 test）全文里出现的编号；能取到频道的、频道在 EVID 131–222 频道表里的。另加：预留评测的 VideoSham 对（及其同一来源组）
不进训练。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "g1"))
import g1_cases as g1  # noqa: E402
import walker_d2_cases as d2  # noqa: E402

SFT = ROOT / "experiments" / "sft_data"
VS_FULL = ROOT / "plan-harness-1" / "datasets" / "videosham_full"
RESERVED = SFT / "reserved_eval_ids.json"
WORK = ROOT / "claude" / "_g2_work"

g1.MEDIA = g1.CASES / "media" / "g2"
g1.LOG = WORK / "g2_cases.log"
g1.MADE_FILE = SFT / "g2_cases_made.json"
g1.CHANNELS_FILE = SFT / "g2_channels.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def vs_id(name: str) -> str:
    """'061_unedited.mp4' / '0061.mp4' / '4061_processed.mp4' → '0061' / '4061'（CSV 的 Video-Name 是 4 位）。"""

    m = re.match(r"(\d+)", Path(name).name)
    return f"{int(m.group(1)):04d}" if m else ""


def vs_rows() -> dict[str, dict[str, str]]:
    with (VS_FULL / "VideoSourceandEdits.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {vs_id(r["Video-Name"]): r for r in csv.DictReader(stream)}


def vs_index() -> None:
    from _d2_media import probe_duration

    out = {"edited": VS_FULL / "media" / "edited", "source": VS_FULL / "media" / "source"}
    for folder in out.values():
        folder.mkdir(parents=True, exist_ok=True)
    for zname, kind, suffix in (("Unedited-Part1.zip", "source", "unedited"), ("Unedited-Part2.zip", "source", "unedited"),
                                ("Processed-Part1.zip", "edited", "processed"), ("Processed-Part2.zip", "edited", "processed")):
        with zipfile.ZipFile(VS_FULL / "_zips" / zname) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.startswith("__MACOSX") or info.is_dir() or not name.lower().endswith(".mp4"):
                    continue
                sid = vs_id(name)
                target = out[kind] / f"{sid}_{suffix}.mp4"
                if target.is_file() and target.stat().st_size == info.file_size:
                    continue
                with zf.open(info) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        g1.log(f"[vs_index] 解压 {zname}")
    rows = vs_rows()
    index = {}
    for sid, row in sorted(rows.items()):
        a, b = out["edited"] / f"{sid}_processed.mp4", out["source"] / f"{sid}_unedited.mp4"
        index[sid] = {"group": row["Name/Id in source dataset"].strip(), "manipulation": " ".join(row["Manipulation"].split()),
                      "task": " ".join(row["Task"].split()), "edited": a.relative_to(VS_FULL).as_posix() if a.is_file() else None,
                      "source": b.relative_to(VS_FULL).as_posix() if b.is_file() else None,
                      "edited_s": round(probe_duration(a) or 0, 2) if a.is_file() else None,
                      "source_s": round(probe_duration(b) or 0, 2) if b.is_file() else None}
    orphan = sorted({vs_id(p.name) for p in out["edited"].glob("*.mp4")} - set(rows))
    g1.save_json(VS_FULL / "index.json", {"made_at": now(), "made_by": "scripts/g2/g2_cases.py vs_index", "pairs": index,
                                          "csv_rows": len(rows), "media_without_csv_row": orphan,
                                          "complete_pairs": sum(1 for v in index.values() if v["edited"] and v["source"]),
                                          "license": "Adobe Research License v1.1（非商业研究），见 AdobeResearchLicensev1.1a.txt"})
    g1.log(f"[vs_index] CSV {len(rows)} 行，成对 {sum(1 for v in index.values() if v['edited'] and v['source'])}，无 CSV 行的媒体 {len(orphan)}")


def used_videosham() -> tuple[set[str], set[str]]:
    """训练侧已用（任何案卷）的 VideoSham 编号与来源组。"""

    rows = vs_rows()
    ex = g1.existing()
    ids = {n.split("_", 1)[1] for n in ex["names"] if n.startswith("videosham_")}
    ids |= set(re.findall(r"videosham[_:/](\d{4})", ex["blob"]))
    return ids, {rows[s]["Name/Id in source dataset"].strip() for s in ids if s in rows}


def reserved() -> dict[str, Any]:
    return g1.load_json(RESERVED, {})


def vs_reserve(n: int) -> None:
    if RESERVED.is_file():
        raise SystemExit(f"{RESERVED} 已存在（预留集只建一次，评测划分要稳定）")
    index = g1.load_json(VS_FULL / "index.json", {})["pairs"]
    used_ids, used_groups = used_videosham()
    evid = d2._evid_text()
    groups: dict[str, list[str]] = {}
    for sid, info in index.items():
        if info["edited"] and info["source"]:
            groups.setdefault(info["group"], []).append(sid)
    free = [g for g in groups if g not in used_groups and not set(groups[g]) & used_ids and g not in evid]
    picked: list[str] = []
    picked_groups: list[str] = []
    for g in g1.hash_order(free, "g2-videosham-reserve"):
        if len(picked) + len(groups[g]) > n:
            continue
        picked += sorted(groups[g])
        picked_groups.append(g)
        if len(picked) == n:
            break
    g1.save_json(RESERVED, {
        "made_at": now(), "made_by": "scripts/g2/g2_cases.py vs_reserve", "purpose": "评测划分预留（g2 任务书：本对话不碰）",
        "dataset": "videosham", "source": "plan-harness-1/datasets/videosham_full（Adobe Research VideoSham v1.0 全集）",
        "rule": "按来源组整组预留（同一来源视频的几个编辑版本都在预留集里）；组从训练侧没用过的组里按 sha1('g2-videosham-reserve'+组) 排序取，凑满 60 对为止；训练建卷跳过这些编号与组",
        "n_pairs": len(picked), "ids": picked, "groups": picked_groups,
        "manipulations": {k: sum(1 for s in picked if index[s]["manipulation"] == k) for k in sorted({index[s]["manipulation"] for s in picked})},
        "excluded_at_reserve_time": {"training_ids": len(used_ids), "training_groups": len(used_groups)}})
    g1.log(f"[vs_reserve] 预留 {len(picked)} 对（{len(picked_groups)} 组）→ {RESERVED}")


def build_videosham(n: int) -> None:
    res = reserved()
    if not res.get("ids"):
        raise SystemExit("先跑 vs_reserve（训练 B 族不能在预留之前建）")
    # g1.build_videosham 按 d2.VIDEOSHAM/media/{edited,source} 与 CSV 找对；按"已用来源组"去重——把预留对当成已用名字塞进 existing()
    d2.VIDEOSHAM = VS_FULL
    real_existing = g1.existing

    def existing_with_reserved() -> dict[str, Any]:
        ex = real_existing()
        ex["names"] = set(ex["names"]) | {f"videosham_{s}" for s in res["ids"]}
        return ex

    g1.existing = existing_with_reserved
    try:
        g1.build_videosham(n)
    finally:
        g1.existing = real_existing
    made = g1.load_json(g1.MADE_FILE, {})
    # g1.build_videosham 把案卷里的媒体 / reference 路径写死成 media/g1/…，文件却按 g1.MEDIA 复制到了 media/g2/…：改回 g2
    for name, info in made.items():
        path = g1.CASES / f"{name}.json"
        if info.get("dataset") == "videosham" and path.is_file():
            text = path.read_text(encoding="utf-8")
            if "media/g1/videosham_" in text:
                path.write_text(text.replace("media/g1/videosham_", "media/g2/videosham_"), encoding="utf-8")
    clash = [k for k in made if k.startswith("videosham_") and k.split("_", 1)[1] in set(res["ids"])]
    if clash:
        raise SystemExit(f"预留对进了训练：{clash}")


def write_list() -> None:
    made = g1.load_json(g1.MADE_FILE, {})
    groups: dict[str, list[str]] = {}
    for name, info in made.items():
        if (g1.CASES / f"{name}.json").is_file():
            groups.setdefault(f"{info['family']}_{info['dataset']}", []).append(name)
    out = {"round": "r1g2", "made_by": "scripts/g2/g2_cases.py", "written_at": now(),
           "groups": {k: sorted(v) for k, v in sorted(groups.items())}, "total": sum(len(v) for v in groups.values())}
    g1.save_json(SFT / "walker_g2_cases.json", out)
    print(json.dumps({k: len(v) for k, v in out["groups"].items()}, ensure_ascii=False), out["total"])


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("vs_index")
    p = sub.add_parser("vs_reserve"); p.add_argument("--n", type=int, default=60)
    p = sub.add_parser("videosham"); p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("fmnv_c"); p.add_argument("--n-fake", type=int, default=0); p.add_argument("--n-real", type=int, default=0); p.add_argument("--ids", nargs="*")
    p = sub.add_parser("g360"); p.add_argument("--n-fake", type=int, default=0); p.add_argument("--n-real", type=int, default=0); p.add_argument("--ids", nargs="*")
    p = sub.add_parser("fmnv_a"); p.add_argument("--n", type=int, default=0)
    sub.add_parser("list")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    d2.LOG = g1.LOG
    if args.cmd == "vs_index":
        vs_index()
        return 0
    if args.cmd == "vs_reserve":
        vs_reserve(args.n)
        return 0
    if args.cmd == "videosham":
        build_videosham(args.n)
    elif args.cmd == "fmnv_c":
        g1.build_fmnv_c(args.n_fake, args.n_real, set(args.ids) if args.ids else None)
    elif args.cmd == "g360":
        g1.build_g360(args.n_fake, args.n_real, set(args.ids) if args.ids else None)
    elif args.cmd == "fmnv_a":
        g1.build_fmnv_a(args.n)
    write_list()
    return 0


if __name__ == "__main__":
    sys.exit(main())
