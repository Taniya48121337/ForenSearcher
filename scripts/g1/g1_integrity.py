"""g1：材料完整性检查——store 引用的文件、导出要拷的教师附图、案卷输入媒体，一个都不能缺。

    python scripts\\g1\\g1_integrity.py [--round r1g1]      # 有缺失返回 1（清理后跑，不过就停）

每个 run（不含 *.failed-*）：
- store.sqlite3 拷到临时目录打开（同 runlog.load_run，不动原库、不碰 -wal），artifacts 表每行 relative_path 在
  artifacts 根（run_config.artifacts_from 或 run_dir/artifacts）下存在且大小一致；
- requests/*/request.json 存在；request.json 里点名的 actor/image_NN.png（导出 export.py 用 shutil.copy2 拷的那批）都存在；
- 案卷 media 路径（相对 cases/）存在。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "experiments" / "sft_data"


def check_run(run: Path) -> list[str]:
    bad: list[str] = []
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
    root = Path(cfg["artifacts_from"]) if cfg.get("artifacts_from") else run / "artifacts"
    db = run / "store.sqlite3"
    if not db.is_file():
        return [f"{run.name}: store.sqlite3 missing"]
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "store.sqlite3"
        shutil.copy2(db, copy)
        con = sqlite3.connect(copy)
        try:
            rows = con.execute("SELECT relative_path, byte_size FROM artifacts").fetchall()
        except sqlite3.Error as exc:
            rows = []
            bad.append(f"{run.name}: artifacts table unreadable ({exc})")
        finally:
            con.close()
    for rel, size in rows:
        path = root / rel
        if not path.is_file():
            bad.append(f"{run.name}: artifact missing {rel}")
        elif size is not None and path.stat().st_size != int(size):
            bad.append(f"{run.name}: artifact size mismatch {rel}")
    for req in sorted((run / "requests").glob("request_*")):
        if not (req / "request.json").is_file():
            bad.append(f"{run.name}: {req.name}/request.json missing")
            continue
        text = (req / "request.json").read_text(encoding="utf-8", errors="replace")
        for name in set(re.findall(r"image_\d+\.(?:png|jpg|jpeg|webp)", text)):
            if not (req / "actor" / name).is_file():
                bad.append(f"{run.name}: {req.name}/actor/{name} missing")
    case_file = Path(cfg.get("case_file") or "")
    if case_file.is_file():
        case = json.loads(case_file.read_text(encoding="utf-8"))
        for m in case.get("media") or []:
            if not (case_file.parent / m["path"]).resolve().is_file():
                bad.append(f"{run.name}: case media missing {m['path']}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="r1g1")
    args = parser.parse_args()
    runs = [r for r in sorted(SFT.glob(f"runs/*_{args.round}_1")) if r.is_dir()]
    bad: list[str] = []
    for run in runs:
        bad += check_run(run)
    out = {"round": args.round, "runs": len(runs), "missing": len(bad), "items": bad[:200]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    log = ROOT / "claude" / "_g1_work" / "integrity.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as h:
        import time
        h.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **out}, ensure_ascii=False) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
