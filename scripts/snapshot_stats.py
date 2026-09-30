"""检索快照统计：每个案例快照的条目数、大小、漂移，以及各 run 的命中率。

    python scripts/snapshot_stats.py                                   # 默认 experiments/sft_data/search_snapshots
    python scripts/snapshot_stats.py --root <快照根> --case true:4124903 --json

只读快照目录（index.jsonl、blobs/、usage/<run>.json，见 search/recording.py），不联网。
- 条目：按 key 去重后的有效条目（同 key 后写覆盖先写），分 search / open / acquire，另计失败与"瞬时失败"。
- 漂移：同一个 key 录过不止一种内容指纹（去掉时间戳等易变字段后仍不同）——说明录制期间网络世界变了。
- 各 run 命中率：run 结束时写的 usage/<run>.json 里的 hit / miss / recorded。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foren_searcher.search.recording import DEFAULT_SNAPSHOT_ROOT, case_key  # noqa: E402


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def case_stats(directory: Path) -> dict[str, Any]:
    """一个案例快照目录的统计。"""

    index = directory / "index.jsonl"
    lines = 0
    latest: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = defaultdict(set)
    if index.is_file():
        for raw in index.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines += 1
            latest[entry["key"]] = entry
            fingerprints[entry["key"]].add(str(entry.get("fingerprint")))
    by_kind: dict[str, int] = defaultdict(int)
    failures = transient = 0
    for entry in latest.values():
        by_kind[entry.get("kind", "?")] += 1
        failures += entry.get("outcome") == "failure"
        transient += bool(entry.get("transient"))
    runs = []
    for usage in sorted((directory / "usage").glob("*.json")) if (directory / "usage").is_dir() else []:
        try:
            value = json.loads(usage.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        counts = value.get("counts") or {}
        runs.append({"run": value.get("run") or usage.stem, "mode": value.get("search_mode"),
                     "hit": counts.get("hit", 0), "miss": counts.get("miss", 0), "recorded": counts.get("recorded", 0),
                     "hit_rate": value.get("hit_rate")})
    blobs = directory / "blobs"
    return {"case": directory.name, "entries": len(latest), "index_lines": lines,
            "search": by_kind.get("search", 0), "open": by_kind.get("open", 0), "acquire": by_kind.get("acquire", 0),
            "failures": failures, "transient": transient,
            "drifted_keys": sum(1 for values in fingerprints.values() if len(values) > 1),
            "blobs": sum(1 for p in blobs.rglob("*") if p.is_file() and not p.name.startswith(".tmp-")) if blobs.is_dir() else 0,
            "bytes": _size(directory), "runs": runs}


def collect(root: str | Path, cases: list[str] | None = None) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        return []
    wanted = {case_key(c) for c in cases or ()}
    return [case_stats(d) for d in sorted(root.iterdir())
            if d.is_dir() and (not wanted or d.name in wanted)]


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "（没有快照）\n"
    out = ["| 案例 | 条目 | search | open | acquire | 失败(瞬时) | 漂移 key | blob | 大小 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        out.append(f"| {row['case']} | {row['entries']} | {row['search']} | {row['open']} | {row['acquire']} | "
                   f"{row['failures']}({row['transient']}) | {row['drifted_keys']} | {row['blobs']} | {_human(row['bytes'])} |")
    run_rows = [(row["case"], run) for row in rows for run in row["runs"]]
    if run_rows:
        out += ["", "| 案例 | run | 模式 | 命中 | 未命中 | 新录 | 命中率 |", "|---|---|---|---:|---:|---:|---:|"]
        for case, run in run_rows:
            rate = "—" if run["hit_rate"] is None else f"{run['hit_rate'] * 100:.1f}%"
            out.append(f"| {case} | {run['run']} | {run['mode']} | {run['hit']} | {run['miss']} | {run['recorded']} | {rate} |")
    total = sum(r["bytes"] for r in rows)
    out += ["", f"合计 {len(rows)} 个案例，{sum(r['entries'] for r in rows)} 条，{_human(total)}。"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索快照统计：条目数、大小、漂移、各 run 命中率")
    parser.add_argument("--root", default=str(DEFAULT_SNAPSHOT_ROOT))
    parser.add_argument("--case", nargs="+", default=None, help="只看这些案例（case_id 或目录名）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是 markdown 表")
    args = parser.parse_args(argv)
    rows = collect(args.root, args.case)
    print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else render_markdown(rows), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
