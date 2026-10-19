"""E2 评测 run 的时间分解（只读；单卡 Vast，2026-10-19）。对每个已完成的 run 统计：
模型调用（requests/*/transport.json 的 elapsed_s 与 token）、INSPECT 各操作（store.sqlite3 inspect_executions 起止时间）、
搜索（search_executions.elapsed_ms，provider_calls 按引擎）、页面 / 来源打开（source_open_attempts.elapsed_ms）、
检索快照命中 / 新录（search_snapshot.jsonl），墙钟取 run_manifests/<轮次>.jsonl 的 wall_seconds。

    python server/24_run_time_breakdown.py experiments/eval/r0/runs --rounds r0base-s1 r0base-s2 [--split-at 2026-10-19T16:41:00] [--json out.json]

run 内是串行的（eval_r0.toml max_concurrency = 1），各项之和与墙钟可比；差额记"其他"（规格渲染、状态落盘、图片编码、子进程启动等）。
--split-at 把 run 按完成时间分成两组（例如线程上限前 / 后）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace(" ", "T"))


def run_breakdown(run_dir: Path) -> dict:
    out: dict = {"model_s": 0.0, "model_calls": 0, "in_tokens": 0, "out_tokens": 0, "inspect": defaultdict(lambda: [0.0, 0]),
                 "inspect_status": Counter(), "search_s": 0.0, "search_calls": 0, "provider": defaultdict(lambda: [0.0, 0]),
                 "open_s": 0.0, "open_calls": 0, "open_outcome": Counter(), "snapshot": Counter(), "acquire": [0.0, 0]}
    for transport in sorted((run_dir / "requests").glob("request_*/transport.json")):
        t = json.loads(transport.read_text(encoding="utf-8"))
        out["model_s"] += float(t.get("elapsed_s") or 0)
        out["model_calls"] += 1
        usage = t.get("usage") or {}
        out["in_tokens"] += int(usage.get("input_tokens") or 0)
        out["out_tokens"] += int(usage.get("output_tokens") or 0)
    db = sqlite3.connect(f"file:{run_dir / 'store.sqlite3'}?mode=ro", uri=True)
    for req, started, finished, status, details in db.execute(
            "select request_json, started_at, finished_at, status, details_json from inspect_executions"):
        op = (json.loads(req).get("request") or {}).get("operation", "?")
        secs = (ts(finished) - ts(started)).total_seconds() if started and finished else 0.0
        out["inspect"][op][0] += secs
        out["inspect"][op][1] += 1
        out["inspect_status"][f"{op}:{status}"] += 1
        acq = (json.loads(details or "{}") or {}).get("acquisition") or {}
        if isinstance(acq, dict) and acq:
            out["acquire"][1] += 1
            for key in ("elapsed_s", "download_s", "seconds"):
                if isinstance(acq.get(key), (int, float)):
                    out["acquire"][0] += float(acq[key])
                    break
    for elapsed, in db.execute("select elapsed_ms from search_executions"):
        out["search_s"] += (elapsed or 0) / 1000
        out["search_calls"] += 1
    for provider, engine, elapsed in db.execute("select provider, engine, elapsed_ms from provider_calls"):
        bucket = out["provider"][f"{provider}/{engine}"]
        bucket[0] += (elapsed or 0) / 1000
        bucket[1] += 1
    for elapsed, outcome in db.execute("select elapsed_ms, outcome from source_open_attempts"):
        out["open_s"] += (elapsed or 0) / 1000
        out["open_calls"] += 1
        out["open_outcome"][outcome] += 1
    db.close()
    snap = run_dir / "search_snapshot.jsonl"
    if snap.is_file():
        for line in snap.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out["snapshot"][f"{e.get('kind')}:{e.get('event')}"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--manifests", default="experiments/sft_data/run_manifests")
    ap.add_argument("--split-at", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    walls: dict[str, dict] = {}
    for rnd in args.rounds:
        path = Path(args.manifests) / f"{rnd}.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "done":
                        walls[row["run"]] = row
    split = datetime.fromisoformat(args.split_at) if args.split_at else None
    groups: dict[str, list] = defaultdict(list)
    per_run = []
    for name, row in sorted(walls.items()):
        run_dir = Path(args.runs_dir) / name
        if not (run_dir / "result.json").is_file():
            continue
        b = run_breakdown(run_dir)
        wall = float(row["wall_seconds"])
        inspect_s = sum(v[0] for v in b["inspect"].values())
        other = wall - b["model_s"] - inspect_s - b["search_s"] - b["open_s"]
        group = "all" if split is None else ("before" if datetime.fromisoformat(row["finished_at"]) < split else "after")
        rec = {"run": name, "group": group, "wall_s": wall, "model_s": round(b["model_s"], 1), "model_calls": b["model_calls"],
               "in_tokens": b["in_tokens"], "out_tokens": b["out_tokens"],
               "inspect": {k: [round(v[0], 1), v[1]] for k, v in sorted(b["inspect"].items(), key=lambda x: -x[1][0])},
               "inspect_status": dict(b["inspect_status"]), "search_s": round(b["search_s"], 1), "search_calls": b["search_calls"],
               "provider": {k: [round(v[0], 1), v[1]] for k, v in b["provider"].items()},
               "open_s": round(b["open_s"], 1), "open_calls": b["open_calls"], "open_outcome": dict(b["open_outcome"]),
               "snapshot": dict(b["snapshot"]), "other_s": round(other, 1)}
        per_run.append(rec)
        groups[group].append(rec)
    for group, recs in groups.items():
        total = defaultdict(float)
        for r in recs:
            total["墙钟"] += r["wall_s"]
            total["模型调用"] += r["model_s"]
            for op, (secs, _) in r["inspect"].items():
                total[f"INSPECT {op}"] += secs
            total["搜索"] += r["search_s"]
            total["页面 / 来源打开"] += r["open_s"]
            total["其他（差额）"] += r["other_s"]
        wall = total.pop("墙钟")
        print(f"\n== {group}：{len(recs)} 个 run，墙钟合计 {wall / 60:.1f} 分钟（平均 {wall / len(recs) / 60:.1f}）")
        for key, secs in sorted(total.items(), key=lambda x: -x[1]):
            print(f"  {key:28s} {secs / 60:7.1f} 分钟  {secs / wall * 100:5.1f}%")
        calls = sum(r["model_calls"] for r in recs)
        print(f"  模型：{calls} 次，平均 {sum(r['model_s'] for r in recs) / max(1, calls):.1f} s / 次；输入 {sum(r['in_tokens'] for r in recs):,}、输出 {sum(r['out_tokens'] for r in recs):,} token")
        snap = Counter()
        for r in recs:
            snap.update(r["snapshot"])
        print(f"  检索快照：{dict(snap)}")
        status = Counter()
        for r in recs:
            status.update(r["inspect_status"])
        print(f"  INSPECT 状态：{dict(status)}")
    print(f"\n{'run':30s} {'组':6s} {'墙钟s':>6s} {'调用':>4s} {'prefill tok':>11s} {'decode tok':>10s} {'模型等待s':>8s} {'工具s':>6s} {'其他s':>6s}  模型 / 工具 / 其他 占墙钟 | INSPECT 前 3")
    for r in per_run:
        tools = sum(v[0] for v in r["inspect"].values()) + r["search_s"] + r["open_s"]
        top = ", ".join(f"{k} {v[0]:.0f}s×{v[1]}" for k, v in list(r["inspect"].items())[:3])
        w = r["wall_s"] or 1
        print(f"{r['run']:30s} {r['group']:6s} {r['wall_s']:6.0f} {r['model_calls']:4d} {r['in_tokens']:11,d} {r['out_tokens']:10,d} {r['model_s']:8.0f} {tools:6.0f} {r['other_s']:6.0f}"
              f"  {r['model_s'] / w:4.0%} / {tools / w:4.0%} / {r['other_s'] / w:4.0%} | {top}")
    if args.json:
        Path(args.json).write_text(json.dumps(per_run, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
