"""r0 成本压缩（10-15，对话 5b）实测报告：每案例教师 / 核验费用与 5a 冒烟并排、r0s1 判定前后对照、150 个案例预算表。

    python scripts\\r0_cost_report.py --round r0c5 --ab r0s1d          # 默认参照 5a 冒烟 r0s1
    python scripts\\r0_cost_report.py --round r0c5 --ab r0s1d --cases 150 --out experiments\\sft_data\\r0_cost_r0c5

读：
- runs/*_<轮次>_*/requests/*/transport.json 与 slot_S5_revise.json 的槽记录：每槽调用数、后端（codex / endpoint）、
  模型 / model_served、Codex 退避次数与等锁秒数、端点花费 cost_usd（Codex 订阅记 $0）；
- judge_usage/<轮次>.api-batch.jsonl（api_judge 每次真实 API 调用一行，含 run）：每 run 核验调用数、批量占比、
  未缓存 / 缓存读 / 缓存写 / 输出 token 与费用（重写后的复核也记在同一个文件里）；
- 5a 冒烟（r0s1）参照：教师费用从 r0s1 的槽记录现算；核验费用 5a 没有逐调用记账，用 5a 文档的实测总数
  （2 个案例 166 次调用 $1.71），按回放估计的两案例占比拆开（见 REF_5A）；
- A/B：verdicts[_units]_<r0s1>_judged.api-anthropic… / api-claude-sonnet-5.5（5a，OpenRouter）对
  verdicts[_units]_<ab>_judged.api-claude-sonnet-5-5（本对话，Anthropic 直连 + 批量 + 缓存），同一批 run 逐单元比 disposition。
写 <out>.md / <out>.json；A/B 有单元 positive → negative 或 positive 数下降时退出码 1（"不能变差"）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"

# 5a 冒烟（claude/教师分槽与核验器API化_20261015.md 第三节）：教师按槽记录现算（runs 在就重算）；核验只有总数
REF_5A = {
    "round": "r0s1",
    "verify_total_usd": 1.71, "verify_total_calls": 166,
    # 5a 的 ApiJudge 不逐调用记账：两案例按回放（同样的 169 个缓存键、OpenRouter 价）估计的占比拆分
    "verify_split": {"videosham_4126": {"calls": 126, "share": 0.55}, "fmnv_c_RXR0B1SxiU0": {"calls": 43, "share": 0.45}},
    "teacher_usd": {"videosham_4126": 1.08, "fmnv_c_RXR0B1SxiU0": 0.81},
}
FAMILY = (("videosham_", "B"), ("vcsl_", "B"), ("fmnv_c_", "C"), ("groundlie360_c_", "C"), ("averimatec_", "E"),
          ("averitec_", "E"), ("true_", "L"), ("fmnv_a_", "A"), ("groundlie360_v", "V"), ("qvhighlights_", "Q"))


def family(case: str) -> str:
    return next((f for p, f in FAMILY if case.startswith(p)), "?")


def case_of(run: str, round_tag: str) -> str:
    marker = f"_{round_tag}_"
    return run.split(marker)[0] if marker in run else run


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------- 教师（槽记录）

def teacher_costs(round_tag: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(RUNS.glob(f"*_{round_tag}_*")):
        if not run_dir.is_dir() or ".failed-" in run_dir.name:
            continue
        records: list[dict[str, Any]] = []
        for path in sorted((run_dir / "requests").glob("request_*/transport.json")):
            records += [s for s in (_load(path) or {}).get("slots") or [] if isinstance(s, dict)]
        for path in sorted((run_dir / "requests").glob("request_*/slot_S5_revise.json")):
            records += [s for s in (_load(path) or {}).get("slots") or [] if isinstance(s, dict)]
        by = Counter()
        cost: dict[str, float] = defaultdict(float)
        models: Counter[str] = Counter()
        served: Counter[str] = Counter()
        codex = {"calls": 0, "rate_limited_attempts": 0, "lock_wait_s": 0.0, "elapsed_s": 0.0}
        for rec in records:
            backend = str(rec.get("slot_backend") or rec.get("backend") or "?")
            by[f"{rec.get('slot') or str(rec.get('name', '?'))[:2]}:{backend}"] += 1
            cost[backend] += float(rec.get("cost_usd") or 0.0)
            models[f"{backend}={rec.get('model')}"] += 1
            if rec.get("backend") == "codex":
                codex["calls"] += 1
                codex["rate_limited_attempts"] += sum(1 for a in rec.get("attempts") or [] if a.get("rate_limited"))
                codex["lock_wait_s"] += float(rec.get("lock_wait_s") or 0.0)
                codex["elapsed_s"] += float(rec.get("elapsed_s") or 0.0)
                served[str(rec.get("model_served"))] += 1
        result = _load(run_dir / "result.json") or {}
        out[run_dir.name] = {"case": case_of(run_dir.name, round_tag), "status": result.get("status"), "slot_calls": dict(by),
                             "calls": len(records), "cost_usd": {k: round(v, 4) for k, v in cost.items()},
                             "teacher_usd": round(sum(cost.values()), 4), "models": dict(models), "codex": codex,
                             "codex_model_served": dict(served)}
    return out


# ---------------------------------------------------------------------------- 核验（usage 日志）

def judge_costs(round_tag: str, kind: str = "api-batch") -> dict[str, dict[str, Any]]:
    rows = _jsonl(SFT / "judge_usage" / f"{round_tag}.{kind}.jsonl")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        run = str(row.get("run") or "(none)")
        if run.startswith("("):
            continue
        item = out.setdefault(run, {"calls": 0, "batch_calls": 0, "uncached": 0, "cache_read": 0, "cache_write": 0, "output": 0,
                                    "cost_usd": 0.0, "kinds": Counter(), "models_served": Counter()})
        item["calls"] += 1
        item["batch_calls"] += row.get("mode") == "batch"
        item["uncached"] += int(row.get("uncached") or 0)
        item["cache_read"] += int(row.get("cache_read") or 0)
        item["cache_write"] += int(row.get("cache_write_5m") or 0) + int(row.get("cache_write_1h") or 0)
        item["output"] += int(row.get("output") or 0)
        item["cost_usd"] += float(row.get("cost_usd") or 0.0)
        item["kinds"][str(row.get("kind"))] += 1
        item["models_served"][str(row.get("model_served"))] += 1
    for item in out.values():
        item["cost_usd"] = round(item["cost_usd"], 4)
        item["kinds"] = dict(item["kinds"])
        item["models_served"] = dict(item["models_served"])
        total_in = item["uncached"] + item["cache_read"] + item["cache_write"]
        item["cache_read_share"] = round(item["cache_read"] / total_in, 3) if total_in else 0.0
    return out


# ---------------------------------------------------------------------------- A/B（r0s1 判定前后）

def _verdict_files(round_tag: str, units: bool) -> list[Path]:
    prefix = "verdicts_units_" if units else "verdicts_"
    return sorted(p for p in SFT.glob(f"{prefix}{round_tag}_judged.*.jsonl") if not p.name.endswith((".pre_revise",)))


def _unit_key(row: dict[str, Any]) -> str:
    s = row.get("subject") or {}
    return f"{s.get('run')}|{row.get('unit')}|{s.get('workspace_id') or s.get('request') or ''}"


def verdicts(round_tag: str, model_hint: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for units in (False, True):
        files = [p for p in _verdict_files(round_tag, units) if model_hint in p.name]
        if not files:
            continue
        latest = max(files, key=lambda p: p.stat().st_mtime)
        for row in _jsonl(latest):
            out[_unit_key(row)] = {"disposition": row.get("disposition"), "errors": len(row.get("errors") or []),
                                   "tally": dict(Counter(e.get("type") for e in row.get("errors") or [])),
                                   "judge_model": row.get("judge_model"), "file": latest.name}
    return out


def ab_compare(ref_round: str, ab_round: str) -> dict[str, Any]:
    before = verdicts(ref_round, "api-anthropic-claude-sonnet-5.5") or verdicts(ref_round, "api-claude-sonnet-5.5")
    after = verdicts(ab_round, "api-claude-sonnet-5-5")
    keys = sorted(set(before) | set(after))
    rows = []
    for key in keys:
        run, unit, where = key.split("|")
        a, b = before.get(key) or {}, after.get(key) or {}
        rows.append({"run": run, "unit": unit, "where": where, "before": a.get("disposition"), "after": b.get("disposition"),
                     "errors_before": a.get("errors"), "errors_after": b.get("errors"), "tally_after": b.get("tally"),
                     "judge_after": b.get("judge_model")})
    pos_a = sum(1 for r in rows if r["before"] == "positive")
    pos_b = sum(1 for r in rows if r["after"] == "positive")
    worse = [r for r in rows if r["before"] == "positive" and r["after"] != "positive"]
    better = [r for r in rows if r["before"] != "positive" and r["after"] == "positive"]
    missing = [r for r in rows if r["after"] is None]
    ok = bool(after) and not worse and pos_b >= pos_a and not missing
    return {"ref": ref_round, "ab": ab_round, "units": len(rows), "positive_before": pos_a, "positive_after": pos_b,
            "worse": worse, "better": better, "missing_after": missing, "rows": rows, "ok": ok,
            "files_before": sorted({v["file"] for v in before.values()}), "files_after": sorted({v["file"] for v in after.values()})}


# ---------------------------------------------------------------------------- 预算

def budget(per_family: dict[str, dict[str, float]], n_cases: int, mix: dict[str, float]) -> dict[str, Any]:
    """per_family：{族: {"verify": $/案例, "teacher": $/案例}}；mix：族 → 占比（缺测的族用已测各族的平均）。"""

    measured = [v for v in per_family.values()]
    mean_v = sum(v["verify"] for v in measured) / max(1, len(measured))
    mean_t = sum(v["teacher"] for v in measured) / max(1, len(measured))
    lines = []
    total_v = total_t = 0.0
    for fam, share in mix.items():
        n = round(n_cases * share)
        v = per_family.get(fam, {}).get("verify", mean_v)
        t = per_family.get(fam, {}).get("teacher", mean_t)
        lines.append({"family": fam, "cases": n, "verify_per_case": round(v, 3), "teacher_per_case": round(t, 3),
                      "verify_usd": round(n * v, 2), "teacher_usd": round(n * t, 2), "measured": fam in per_family})
        total_v += n * v
        total_t += n * t
    return {"cases": n_cases, "lines": lines, "verify_usd": round(total_v, 2), "teacher_usd": round(total_t, 2),
            "total_usd": round(total_v + total_t, 2), "verify_per_case": round(total_v / max(1, n_cases), 3)}


DEFAULT_MIX = {"B": 0.38, "C": 0.28, "E": 0.20, "L": 0.14}     # r0 纳入案例 B 9+2 / C 8 / E 6 / L 4（多对话执行计划对话 5）


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", required=True, help="5 个案例的实测轮次（例如 r0c5）")
    p.add_argument("--ref", default=REF_5A["round"], help="5a 冒烟轮次（默认 r0s1）")
    p.add_argument("--ab", default=None, help="用 api-batch 重核 r0s1 那批 run 的轮次标签（例如 r0s1d）")
    p.add_argument("--cases", type=int, default=150)
    p.add_argument("--qwen-local-usd", type=float, default=0.0, help="服务器自部署 Qwen 时 S0–S3 每案例按 $0 计；本机走 OpenRouter 时报实测")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    teacher = teacher_costs(args.round)
    judge = judge_costs(args.round)
    ref_teacher = teacher_costs(args.ref)
    ab_judge = judge_costs(args.ab) if args.ab else {}
    cases: dict[str, dict[str, Any]] = {}
    for run, item in teacher.items():
        cases[item["case"]] = {"run": run, "family": family(item["case"]), "teacher": item, "judge": judge.get(run) or {}}
    for run, item in judge.items():
        case = case_of(run, args.round)
        cases.setdefault(case, {"run": run, "family": family(case), "teacher": {}, "judge": item})
    per_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"verify": [], "teacher": [], "qwen": []})
    for case, item in cases.items():
        t = item["teacher"]
        qwen = float((t.get("cost_usd") or {}).get("qwen", 0.0))
        strong = float((t.get("cost_usd") or {}).get("strong", 0.0))
        per_family[item["family"]]["verify"].append(float(item["judge"].get("cost_usd") or 0.0))
        per_family[item["family"]]["teacher"].append(strong + args.qwen_local_usd)
        per_family[item["family"]]["qwen"].append(qwen)
    fam_avg = {f: {k: sum(v) / len(v) for k, v in d.items() if v} for f, d in per_family.items()}
    plan = budget({f: {"verify": v["verify"], "teacher": v["teacher"]} for f, v in fam_avg.items()}, args.cases, DEFAULT_MIX)
    ab = ab_compare(args.ref, args.ab) if args.ab else None
    n = max(1, len(cases))
    verify_total = sum(float(c["judge"].get("cost_usd") or 0.0) for c in cases.values())
    report = {"round": args.round, "cases": cases, "per_family": fam_avg, "verify_per_case": round(verify_total / n, 4),
              "verify_calls_per_case": round(sum(int(c["judge"].get("calls") or 0) for c in cases.values()) / n, 1),
              "ref_5a": {**REF_5A, "teacher_from_runs": {v["case"]: v["teacher_usd"] for v in ref_teacher.values()}},
              "ab_verify": ab_judge, "ab": ab, "budget": plan}

    out = Path(args.out) if args.out else SFT / f"r0_cost_{args.round}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [f"# r0 成本实测 {args.round}（Codex strong + Anthropic 批量核验）", ""]
    md += ["## 每案例（本轮）与 5a 冒烟并排", "",
           "| 案例 | 族 | 状态 | 槽调用（Codex 次数） | Codex 429 次 / 等锁 s | 教师 API $（strong / qwen） | 核验调用（批量） | 缓存读占比 | 核验 $ | 5a 核验调用 / $ | 5a 教师 $ |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case, item in sorted(cases.items(), key=lambda kv: (kv[1]["family"], kv[0])):
        t, j = item["teacher"], item["judge"]
        ref = REF_5A["verify_split"].get(case)
        ref_v = f"{ref['calls']} / ${REF_5A['verify_total_usd'] * ref['share']:.2f}（估）" if ref else "-"
        ref_t = REF_5A["teacher_usd"].get(case) or report["ref_5a"]["teacher_from_runs"].get(case)   # 文档数含核验后重写
        md.append(f"| {case} | {item['family']} | {t.get('status') or '-'} | {t.get('calls', 0)}（{(t.get('codex') or {}).get('calls', 0)}） | "
                  f"{(t.get('codex') or {}).get('rate_limited_attempts', 0)} / {(t.get('codex') or {}).get('lock_wait_s', 0):.0f} | "
                  f"${(t.get('cost_usd') or {}).get('strong', 0):.3f} / ${(t.get('cost_usd') or {}).get('qwen', 0):.3f} | "
                  f"{j.get('calls', 0)}（{j.get('batch_calls', 0)}） | {j.get('cache_read_share', 0):.0%} | ${float(j.get('cost_usd') or 0):.3f} | "
                  f"{ref_v} | {('$%.2f' % ref_t) if ref_t else '-'} |")
    md += ["", f"本轮每案例核验 **${report['verify_per_case']:.3f}**（{report['verify_calls_per_case']} 次调用）；目标 ≤ $0.15。"
           f"5a 冒烟每案例核验 ${REF_5A['verify_total_usd'] / 2:.3f}（{REF_5A['verify_total_calls'] / 2:.0f} 次）。", ""]
    served = Counter()
    for item in cases.values():
        served.update((item["teacher"].get("codex_model_served") or {}))
    md += [f"Codex 输出里的模型名：{dict(served) or '无'}（None = Codex 没在输出里给模型名，只记请求的 -m）", ""]
    if ab_judge:
        md += [f"## r0s1 两条用 api-batch 重核（{args.ab}）的费用", "", "| run | 调用（批量） | 缓存读占比 | $ |", "|---|---|---|---|"]
        for run, j in sorted(ab_judge.items()):
            md.append(f"| {run} | {j['calls']}（{j['batch_calls']}） | {j['cache_read_share']:.0%} | ${j['cost_usd']:.3f} |")
        total = sum(j["cost_usd"] for j in ab_judge.values())
        md += ["", f"合计 ${total:.3f}（5a 同两条 ${REF_5A['verify_total_usd']:.2f}，{REF_5A['verify_total_calls']} 次）", ""]
    if ab:
        md += [f"## 判定前后对照（{args.ref} 5a OpenRouter → {args.ab} Anthropic 直连批量）", "",
               f"单元 {ab['units']}；positive {ab['positive_before']} → {ab['positive_after']}；变差 {len(ab['worse'])}；变好 {len(ab['better'])}；"
               f"缺 {len(ab['missing_after'])}；**{'通过' if ab['ok'] else '不通过'}**", "",
               "| run | 单元 | 工作区 / 请求 | 5a | 本对话 | 错误数 5a → 本对话 |", "|---|---|---|---|---|---|"]
        for r in ab["rows"]:
            md.append(f"| {r['run']} | {r['unit']} | {r['where']} | {r['before']} | {r['after']} | {r['errors_before']} → {r['errors_after']} |")
        md.append("")
    md += [f"## {args.cases} 个案例预算（按 r0 纳入族比例 {DEFAULT_MIX}；没测到的族用已测族均值）", "",
           "| 族 | 案例 | 核验 $/案例 | strong 教师 $/案例 | 核验 $ | 教师 $ | 实测 |", "|---|---|---|---|---|---|---|"]
    for line in plan["lines"]:
        md.append(f"| {line['family']} | {line['cases']} | {line['verify_per_case']:.3f} | {line['teacher_per_case']:.3f} | "
                  f"{line['verify_usd']:.2f} | {line['teacher_usd']:.2f} | {'是' if line['measured'] else '均值'} |")
    md += ["", f"合计：核验 ${plan['verify_usd']:.2f} + 教师 ${plan['teacher_usd']:.2f} = **${plan['total_usd']:.2f}**（上限 $30）；"
           f"S0–S3 在服务器自部署 Qwen 记 $0（本机走 OpenRouter 时每案例另加实测 qwen 费用）。", ""]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n写了 {out.with_suffix('.md')} / .json")
    return 0 if (ab is None or ab["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
