"""g1：本批 run 清单与统计（报告五节用）。

    python scripts\\g1\\g1_status.py [--out claude\\_g1_work\\g1_summary.json] [--md claude\\_g1_work\\g1_runs.md]

每个 run：案例、族、数据集、状态、报告状态（W1 / W2）、循环内改写、第 2 轮、Codex 调用数（S0–S3 / S4–S5）、Claude 判官调用、
耗时；另汇总闸门日志（Codex 退避 / 暂停）、Claude 闸门（限额）、run_manifests 里的失败与重试。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "experiments" / "sft_data"
STATE = ROOT / ".g1" / "state"
ROUND = "r1g1"


def jl(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_row(run: Path, made: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case = run.name[: -len(f"_{ROUND}_1")]
    info = made.get(case, {})
    slots = collections.Counter()
    slot_s = collections.Counter()
    rules = collections.Counter()
    reports = []
    for t in sorted(run.glob("requests/*/transport.json")):
        try:
            d = json.loads(t.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for s in d.get("slots") or []:
            k = s.get("slot") or "?"
            slots[k] += 1
            slot_s[k] += float(s.get("elapsed_s") or 0)
        rule = (d.get("oracle") or {}).get("rule")
        if rule:
            rules[rule] += 1
    inloop = {}
    p = run / "inloop_sufficiency.json"
    if p.is_file():
        inloop = json.loads(p.read_text(encoding="utf-8"))
    checks = inloop.get("checks") or []
    for c in checks:
        reports.append(f"{c.get('workspace', '?').split('-')[0]}:{c.get('status_in')}→{c.get('status_out')}")
    m = manifest.get(run.name, {})
    return {"run": run.name, "case": case, "family": info.get("family"), "dataset": info.get("dataset"), "kind": info.get("kind"),
            "status": m.get("status"), "attempt": m.get("attempt"), "wall_s": m.get("wall_seconds"),
            "reports": reports, "inloop_rewritten": sum(1 for c in checks if c.get("rewritten")),
            "inloop_limit_hit": sum(1 for c in checks if c.get("limit_hit")),
            "inloop_errors": sum(1 for c in checks if c.get("error") and not c.get("limit_hit")),
            "judge_calls": sum(int(c.get("judge_new_calls") or 0) for c in checks),
            "round2": sum(v for r, v in rules.items() if "round2" in r or r.startswith("G2_")),
            "codex_s0_s3": sum(slots[k] for k in ("S0", "S1", "S2", "S3")), "codex_s4_s5": slots["S4"] + slots["S5"],
            "codex_s": round(sum(slot_s.values()), 1), "slots": dict(slots)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "claude" / "_g1_work" / "g1_summary.json"))
    parser.add_argument("--md", default=str(ROOT / "claude" / "_g1_work" / "g1_runs.md"))
    args = parser.parse_args()
    made = json.loads((SFT / "g1_cases_made.json").read_text(encoding="utf-8")) if (SFT / "g1_cases_made.json").is_file() else {}
    rows_m = jl(SFT / "run_manifests" / f"{ROUND}.jsonl")
    manifest = {r["run"]: r for r in rows_m}
    runs = [run_row(r, made, manifest) for r in sorted(SFT.glob(f"runs/*_{ROUND}_1")) if r.is_dir()]
    gate = jl(STATE / "codex_gate.jsonl")
    cgate = jl(STATE / "claude_gate.jsonl")
    summary = {
        "runs": len(runs), "by_status": collections.Counter(r["status"] for r in runs),
        "by_family_dataset": collections.Counter(f"{r['family']}_{r['dataset']}" for r in runs),
        "done_by_family_dataset": collections.Counter(f"{r['family']}_{r['dataset']}" for r in runs if r["status"] == "done"),
        "inloop_rewritten_runs": sum(1 for r in runs if r["inloop_rewritten"]), "round2_runs": sum(1 for r in runs if r["round2"]),
        "codex_calls_in_runs": sum(r["codex_s0_s3"] + r["codex_s4_s5"] for r in runs),
        "manifest_failed_attempts": sum(1 for r in rows_m if r.get("failed")),
        "retried_runs": sorted({r["run"] for r in rows_m if r.get("attempt") == 2}),
        "failed_dirs": sorted(p.name for p in SFT.glob(f"runs/*_{ROUND}_1.failed-*")),
        "codex_gate": {"calls": len(gate), "results": collections.Counter(g.get("result") for g in gate),
                       "backoffs": sum(len(g.get("backoffs") or []) for g in gate),
                       "backoff_kinds": collections.Counter(b.get("kind") for g in gate for b in g.get("backoffs") or []),
                       "slot_wait_s": round(sum(float(g.get("slot_wait_s") or 0) for g in gate), 1),
                       "pause_wait_s": round(sum(float(g.get("pause_wait_s") or 0) for g in gate), 1),
                       "attempt_s_mean": round(sum(sum(g.get("attempt_s") or [0]) for g in gate) / max(1, len(gate)), 1)},
        "claude_gate": {"calls": len(cgate), "results": collections.Counter(g.get("result") for g in cgate)},
        "pauses": jl(STATE / "pauses.jsonl"),
        "walls": {"mean_s": round(sum(float(r["wall_s"] or 0) for r in runs) / max(1, len(runs)), 1)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=1, default=dict),
                              encoding="utf-8")
    lines = ["| run | 族 | 数据集 | 状态 | 报告（循环内前→后） | 循环内改写 | 第 2 轮 | Codex S0–S3 / S4–S5 | 判官 | 耗时 s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['case']} | {r['family']} | {r['dataset']} | {r['status']}{'（重试）' if r['attempt'] == 2 else ''} | "
                     f"{'；'.join(r['reports']) or '—'} | {r['inloop_rewritten']} | {r['round2']} | {r['codex_s0_s3']} / {r['codex_s4_s5']} | "
                     f"{r['judge_calls']} | {r['wall_s']} |")
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=dict)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
