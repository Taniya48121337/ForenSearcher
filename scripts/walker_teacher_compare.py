"""教师对照（2026-10-14 晚）：同一批案例、同一套代码，换教师模型各跑一轮，比轨迹质量与可用数据量。

    python scripts\\walker_teacher_compare.py --rounds wk13 wk14 wk15 --cases <案例…>

每轮看：
  跑没跑完      done / 槽失败；墙钟中位数；教师调用次数、token、OpenRouter 记的费用
  格式          S4 报告第一次没引到可用 E 而重写的次数；S0 rationale 被采用的比例（不合格的退回模板）
  核验（Sonnet） 报告 positive、报告"严格入选"（positive 且无 major）、COMPLETED 比例；答案 positive（重写前 → 后）
  导出          accepted 行数（HI / GIA）、held 的主要原因（hidden_context、terminal_major_error、渲染超预算）
只比各轮都有 run 的案例。产物：experiments/sft_data/walker_teachers_<轮次们>.md / .json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"


def _latest(pattern: str, *, pre: bool = False) -> Path | None:
    files = [p for p in SFT.glob(pattern) if ".summary" not in p.name and p.name.endswith(".pre_revise") == pre]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _units(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not path:
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        subject = row.get("subject") or {}
        unit = str(row.get("unit") or "report")
        out[(unit, str(subject.get("run")) + "/" + str(subject.get("workspace_id") if unit == "report" else subject.get("request")))] = row
    return out


def _strict(row: dict[str, Any]) -> bool:
    return row.get("disposition") == "positive" and not any(e.get("severity") in ("major", "critical") for e in row.get("errors") or [])


def run_stats(run_dir: Path) -> dict[str, Any]:
    calls = tin = tout = 0
    cost = 0.0
    seconds = 0.0
    s4_retry = s0_used = s0_total = 0
    statuses: list[str] = []
    for transport in sorted((run_dir / "requests").glob("request_*/transport.json")):
        try:
            meta = json.loads(transport.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for slot in meta.get("slots") or []:
            calls += 1
            seconds += float(slot.get("elapsed_s") or 0)
            usage = slot.get("usage") or {}
            tin += int(usage.get("input_tokens") or 0)
            tout += int(usage.get("output_tokens") or 0)
            cost += float(slot.get("cost_usd") or 0)
            s4_retry += slot.get("name") == "S4_report_retry"
        r0 = transport.parent / "slot_S0_rationale.json"
        if r0.is_file():
            s0_total += 1
            try:
                s0_used += bool(json.loads(r0.read_text(encoding="utf-8")).get("used"))
            except ValueError:
                pass
        response = transport.parent / "response.json"
        if response.is_file():
            try:
                out = json.loads(response.read_text(encoding="utf-8"))
                if isinstance(out, dict) and out.get("status") in ("COMPLETED", "INCOMPLETE") and "findings" in out:
                    statuses.append(out["status"])
            except ValueError:
                pass
    return {"llm_calls": calls, "llm_seconds": seconds, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
            "s4_retry": s4_retry, "s0_used": s0_used, "s0_total": s0_total, "report_statuses": statuses}


def export_manifest(tag: str) -> dict[str, Any] | None:
    dirs = [p for p in (SFT / "export").glob(f"{tag}_*") if (p / "manifest.json").is_file() and ".claude-cli" in p.name]
    if not dirs:
        return None
    return json.loads((max(dirs, key=lambda p: p.stat().st_mtime) / "manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    have = {tag: {c for c in args.cases if (RUNS / f"{c}_{tag}_1" / "result.json").is_file()} for tag in args.rounds}
    skipped = [t for t in args.rounds if not have[t]]
    if skipped:                       # 10-15：没跑的轮次（_skip_<轮次>）不参加对照，不然共同案例是空集
        print(f"没有 run 的轮次不比：{', '.join(skipped)}")
        args.rounds = [t for t in args.rounds if have[t]]
        have = {t: have[t] for t in args.rounds}
    common = sorted(set.intersection(*have.values())) if have else []
    table: dict[str, dict[str, Any]] = {}
    for tag in args.rounds:
        reports = _units(_latest(f"verdicts_{tag}_judged.claude-cli-*.jsonl"))
        units = _units(_latest(f"verdicts_units_{tag}_judged.claude-cli-*.jsonl"))
        pre = _units(_latest(f"verdicts_units_{tag}_judged.claude-cli-*.jsonl*", pre=True)) or units
        runs = [f"{c}_{tag}_1" for c in common]
        stats = [run_stats(RUNS / r) for r in runs]
        manifest_rows = [json.loads(l) for l in (SFT / "run_manifests" / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()
                         if l.strip()] if (SFT / "run_manifests" / f"{tag}.jsonl").is_file() else []
        walls = [float(m.get("wall_seconds") or 0) for m in manifest_rows if m.get("run") in runs and m.get("status") == "done"]
        model = next((m.get("teacher_model") for m in manifest_rows if m.get("teacher_model")), None)
        if not model:
            cfg = RUNS / (runs[0] if runs else "") / "run_config.json"
            model = (json.loads(cfg.read_text(encoding="utf-8")).get("walker") or {}).get("teacher_model") if cfg.is_file() else "?"
        rep = [v for (u, k), v in reports.items() if u == "report" and k.split("/")[0] in runs]
        ans = [v for (u, k), v in units.items() if u == "answer" and k.split("/")[0] in runs]
        ans_pre = [v for (u, k), v in pre.items() if u == "answer" and k.split("/")[0] in runs]
        statuses = [s for st in stats for s in st["report_statuses"]]
        manifest = export_manifest(tag) or {}
        held = Counter(str(h.get("reason") or "").split(":")[0] for h in manifest.get("held") or []
                       if str(h.get("run", "")).rsplit("_", 2)[0] in common)
        table[tag] = {
            "teacher": model, "cases": len(runs),
            "wall_median_s": round(statistics.median(walls), 1) if walls else None,
            "llm_calls": sum(s["llm_calls"] for s in stats), "tokens_in": sum(s["tokens_in"] for s in stats),
            "tokens_out": sum(s["tokens_out"] for s in stats), "cost_usd": round(sum(s["cost_usd"] for s in stats), 3),
            "s4_retry": sum(s["s4_retry"] for s in stats),
            "s0_used": f"{sum(s['s0_used'] for s in stats)}/{sum(s['s0_total'] for s in stats)}",
            "reports": len(rep), "report_positive": sum(r.get("disposition") == "positive" for r in rep),
            "report_strict": sum(_strict(r) for r in rep), "report_completed": f"{statuses.count('COMPLETED')}/{len(statuses)}",
            "answers": len(ans), "answer_positive_pre": sum(r.get("disposition") == "positive" for r in ans_pre),
            "answer_positive": sum(r.get("disposition") == "positive" for r in ans), "answer_strict": sum(_strict(r) for r in ans),
            "accepted": manifest.get("counts"), "held": dict(held.most_common(6)),
            "render_overflow": len(manifest.get("render_overflow") or []),
        }
    lines = [f"# walker 教师对照：{' / '.join(args.rounds)}（共同案例 {len(common)} 个，核验 Sonnet）", "",
             "| | " + " | ".join(args.rounds) + " |", "|---|" + "---|" * len(args.rounds)]
    rows = [("教师", "teacher"), ("案例", "cases"), ("墙钟中位数 s", "wall_median_s"), ("教师调用", "llm_calls"),
            ("输入 token", "tokens_in"), ("输出 token", "tokens_out"), ("费用 $（OpenRouter 记账）", "cost_usd"),
            ("S4 重写（没引到可用 E）", "s4_retry"), ("S0 rationale 采用", "s0_used"),
            ("报告 positive", "report_positive"), ("报告严格入选", "report_strict"), ("报告 COMPLETED", "report_completed"),
            ("答案 positive 重写前", "answer_positive_pre"), ("答案 positive 重写后", "answer_positive"),
            ("答案严格入选", "answer_strict"), ("导出 accepted", "accepted"), ("held 主要原因", "held"),
            ("渲染超预算丢行", "render_overflow")]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(json.dumps(table[t].get(key), ensure_ascii=False) if isinstance(table[t].get(key), dict)
                                                    else str(table[t].get(key)) for t in args.rounds) + " |")
    lines += ["", "共同案例：" + ", ".join(common)]
    name = "walker_teachers_" + "_".join(args.rounds)
    (SFT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SFT / f"{name}.json").write_text(json.dumps({"rounds": args.rounds, "common": common, "table": table}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
