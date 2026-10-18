"""把一个或几个导出目录拼成 / 按族筛成一个"子集导出目录"，再交给同一个 export_gate.py 跑门（门本身一行不改）（2026-10-18）。

    # g1 按族单独报：A / B / C 各一个子集目录，各跑一次门
    python scripts/gate_subset.py --export <g1 导出目录> --family A --out <导出目录>/_by_family/A
    # r1 + g1 + g2 合集再跑一遍门
    python scripts/gate_subset.py --export <r1 导出> --export <g1 导出> --export <g2 导出> --out experiments/sft_data/export/_r0train_union

做法：六个 jsonl（hi / gia × accepted / held / rejected）按行筛（--family）或原样拼接；verdicts_reports / verdicts_units 同样处理；
manifest.json 取第一个目录的（export_options、freeze、spec 等门要读的字段三包相同），另记 subset_of；images/ 下按 run 建目录联接
（Windows junction，不复制图）。只读源目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

FILES = ["hi_accepted", "hi_held", "hi_rejected", "gia_accepted", "gia_held", "gia_rejected"]
VERDICTS = ["verdicts_reports", "verdicts_units"]


def family(case_id: str) -> str:
    ds, _, sid = str(case_id).partition(":")
    if ds in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if ds == "averimatec":
        return "E" if sid.startswith("e") else "L"
    if ds == "evid" or (ds == "fmnv" and sid.startswith("a_")):
        return "A"
    return "C"


def row_case(row: dict) -> str:
    md = row.get("metadata") or row.get("subject") or {}
    return str(md.get("case_id") or row.get("case_id") or "")


def run_case_map(exports: list[Path]) -> dict[str, str]:
    """verdicts 行可能只有 run 没有 case_id：用导出行建 run → case_id。"""

    out = {}
    for e in exports:
        for name in FILES:
            p = e / f"{name}.jsonl"
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        md = json.loads(line).get("metadata") or {}
                        out[str(md.get("run"))] = str(md.get("case_id"))
    return out


def junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="append", required=True)
    ap.add_argument("--family", nargs="*", default=None, help="只留这些族（A B C E L）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    exports = [Path(e).resolve() for e in args.export]
    out = Path(args.out).resolve()
    if out.exists():
        for child in (out / "images").glob("*") if (out / "images").is_dir() else []:
            child.rmdir() if child.is_dir() else child.unlink()      # junction：只删联接本身
        shutil.rmtree(out)
    out.mkdir(parents=True)
    keep = set(args.family) if args.family else None
    runs = run_case_map(exports)
    counts = {}
    used_runs: dict[str, Path] = {}
    for name in FILES + VERDICTS:
        lines = []
        for e in exports:
            p = e / f"{name}.jsonl"
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                case = row_case(row) or runs.get(str(row.get("run") or (row.get("metadata") or {}).get("run")), "")
                if keep is not None and family(case) not in keep:
                    continue
                lines.append(line)
                if name in FILES:
                    run = str((row.get("metadata") or {}).get("run"))
                    used_runs.setdefault(run, e)
        (out / f"{name}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[name] = len(lines)
    manifest = json.loads((exports[0] / "manifest.json").read_text(encoding="utf-8"))
    manifest["subset_of"] = {"exports": [str(e) for e in exports], "families": sorted(keep) if keep else None, "rows": counts}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "images").mkdir()
    for run, e in sorted(used_runs.items()):
        src = e / "images" / run
        if src.is_dir():
            junction(out / "images" / run, src)
    print(json.dumps({"out": str(out), "rows": counts, "image_run_links": len(used_runs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
