"""对照两批 run（改进前 / 改进后），把《第 0 轮数据质量评估》里的 8 个问题各出一个数。

    python scripts/compare_rounds.py --before-runs <12 个旧 run> --after-round r0b \
        [--before-export experiments/sft_data/export/<旧导出>] [--after-export experiments/sft_data/export/<新导出>] \
        [--out experiments/sft_data/compare_r0_vs_r0b.md]

只读 run 目录（trajectory.jsonl / requests / retrieval_guard.jsonl / result.json）和导出目录的 manifest.json，不调模型。
量的东西：
  教师侧行为（不依赖核验器）：每 run 教师调用数；动作分布与 SEARCH 占比；每工作区最长同动作连击；报告 COMPLETED / INCOMPLETE；
      带 rationale 的输出占比；命中核查站点的次数（新 run 看护栏日志，旧 run 扫轨迹里的来源 URL）。
  数据侧（依赖导出，两边应用同一个核验器才可比）：accepted 行数与权重和、按动作的权重份额、监督 token 中位、沾染行数。
  声明类案例的取证（2026-09-30 新增，替代"SEARCH 占步数"）：每 run 的检索查询数、带核查字样的查询数、打开的页面数与域名数、
      实际读过（READ_RECORD）和报告引用的独立域名数，其中核查类页面（黑名单或路径含 fact-check / fact-brief 等）单列。
      一次 SEARCH 会自动打开几页、再由 READ_RECORD 逐页读，所以只数动作名会把"搜到并读了三页"算成搜索只占四分之一。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foren_searcher.search.retrieval_guard import fact_check_match  # noqa: E402

SFT = Path("experiments") / "sft_data"
_URL = re.compile(r"https?://[^\s\"'\\<>]+")


def _events(run: Path) -> list[dict[str, Any]]:
    path = run / "trajectory.jsonl"
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _json(text: str) -> dict[str, Any] | None:
    try:
        from foren_searcher.local_models import extract_json_object
        return extract_json_object(text)
    except Exception:
        return None


def run_metrics(run: Path) -> dict[str, Any]:
    events = _events(run)
    actions: Counter[str] = Counter()
    per_ws: dict[str, list[str]] = defaultdict(list)
    reports: Counter[str] = Counter()
    outputs = 0
    with_rationale = 0
    for e in events:
        kind = e.get("event_type")
        payload = e.get("payload") or {}
        if kind == "raw_model_output":
            outputs += 1
            obj = _json(str(payload.get("raw_output") or ""))
            if obj and isinstance(obj.get("rationale"), str) and obj["rationale"].strip():
                with_rationale += 1
        elif kind == "model_output":
            out = payload.get("output") or {}
            if e.get("actor_role") == "HI":
                name = out.get("action") or out.get("status")
                if name:
                    actions[name] += 1
                    if out.get("action"):
                        per_ws[str(e.get("actor_id"))].append(str(out["action"]))
                    else:
                        reports[str(out.get("status"))] += 1
            else:
                actions["GIA:" + str(out.get("decision") or out.get("action"))] += 1
    longest = 0
    for seq in per_ws.values():
        run_len = 1
        for a, b in zip(seq, seq[1:]):
            run_len = run_len + 1 if a == b else 1
            longest = max(longest, run_len)
    # 核查站点：新 run 有护栏日志；旧 run 扫轨迹里出现过的来源 URL（候选 + 已打开）
    guard = {}
    if (run / "guard_summary.json").is_file():
        guard = json.loads((run / "guard_summary.json").read_text(encoding="utf-8")).get("counts", {})
    fc_urls: set[str] = set()
    opened_fc = 0
    for e in events:
        if e.get("event_type") != "observation":
            continue
        text = json.dumps(e.get("payload"), ensure_ascii=False)
        for url in _URL.findall(text):
            if fact_check_match(url.rstrip("\\")):
                fc_urls.add(url)
                if "page=ready" in text[text.find(url): text.find(url) + 400]:
                    opened_fc += 1
    requests = sorted((run / "requests").glob("request_*")) if (run / "requests").is_dir() else []
    tokens_in = 0
    for r in requests:
        try:
            usage = json.loads((r / "transport.json").read_text(encoding="utf-8")).get("usage") or {}
            tokens_in += int(usage.get("input_tokens") or 0)
        except (OSError, ValueError):
            pass
    decision = None
    if (run / "result.json").is_file():
        try:
            decision = json.loads((run / "result.json").read_text(encoding="utf-8")).get("decision")
            if isinstance(decision, dict):          # result.json 里 decision 是完整对象 {"decision": "FINISH", "answer": ...}
                decision = decision.get("decision")
        except ValueError:
            pass
    hi_steps = sum(v for k, v in actions.items() if not k.startswith("GIA:") and k not in ("COMPLETED", "INCOMPLETE"))
    return {
        "run": run.name, "teacher_calls": len(requests), "input_tokens": tokens_in, "decision": decision,
        "hi_steps": hi_steps, "search": actions.get("SEARCH", 0), "read_record": actions.get("READ_RECORD", 0),
        "inspect": actions.get("INSPECT", 0), "query_memory": actions.get("QUERY_MEMORY", 0),
        "reports_completed": reports.get("COMPLETED", 0), "reports_incomplete": reports.get("INCOMPLETE", 0),
        "longest_same_action_run": longest, "outputs": outputs, "outputs_with_rationale": with_rationale,
        "fact_check_urls_seen": len(fc_urls), "fact_check_pages_opened": opened_fc,
        "guard_fact_check_excluded": guard.get("fact_check_excluded", 0), "guard_after_cutoff": guard.get("after_cutoff", 0),
        "claim_sources": claim_source_metrics(run, events),
    }


CLAIM_DATASETS = {"true", "averitec", "averimatec"}
_FACT_CHECK_QUERY = re.compile(r"fact[\s_-]?check|debunk|hoax|snopes|politifact|辟谣|谣言|真假|是否属实|核查", re.I)
_FACT_CHECK_PATH = re.compile(r"fact[-_]?check|fact[-_]?brief|factcheck|hoax|debunk|/fake[-_]?news|/viral[-_]check", re.I)
_OPENED = re.compile(r"^\[(R\d+)\]\s+(\S+);.*?page=(?:ready|partial)\((E\d+(?:,E\d+)*)\)")


def _domain(url: str) -> str:
    from urllib.parse import urlsplit

    try:
        host = urlsplit(url if "://" in url else "https://" + url).netloc.lower()
    except ValueError:
        return ""
    return host.split("@")[-1].split(":")[0].removeprefix("www.").removeprefix("m.")


def _fact_check_like(url: str) -> bool:
    return bool(fact_check_match(url) or _FACT_CHECK_PATH.search(url))


def claim_source_metrics(run: Path, events: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """声明类案例（TRUE / AVeriTeC / AVeriMaTeC）的取证量；其它数据集返回 None。只读轨迹。"""

    config = {}
    if (run / "run_config.json").is_file():
        try:
            config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        except ValueError:
            config = {}
    dataset = str(config.get("dataset") or run.name.split("_", 1)[0]).lower()
    if dataset not in CLAIM_DATASETS:
        return None
    events = _events(run) if events is None else events
    queries: list[str] = []
    opens = 0
    e_url: dict[str, str] = {}
    read_ids: set[str] = set()
    cited_ids: set[str] = set()
    for e in events:
        if e.get("actor_role") != "HI":
            continue
        kind = e.get("event_type")
        payload = e.get("payload") or {}
        if kind == "model_output":
            out = payload.get("output") or {}
            args = out.get("arguments") or {}
            if out.get("action") == "SEARCH":
                if args.get("query"):
                    queries.append(str(args["query"]))
                elif args.get("source"):
                    opens += 1
            elif out.get("action") == "READ_RECORD" and args.get("record_id"):
                read_ids.add(str(args["record_id"]))
        elif kind == "observation":
            obs = payload.get("observation") or {}
            if obs.get("action") == "SEARCH":
                for item in obs.get("items") or []:
                    m = _OPENED.match(str(item))
                    if m:
                        for eid in m.group(3).split(","):
                            e_url[eid] = m.group(2)
        elif kind == "workspace_report":
            report = payload.get("report") or payload
            for f in report.get("findings") or []:
                cited_ids.update(str(x) for x in f.get("evidence_ids") or [])

    def domains(ids: set[str] | None = None, *, fact_check: bool) -> set[str]:
        return {_domain(u) for eid, u in e_url.items() if (ids is None or eid in ids) and _fact_check_like(u) == fact_check and _domain(u)}

    return {
        "queries": len(queries), "fact_check_queries": sum(1 for q in queries if _FACT_CHECK_QUERY.search(q)),
        "direct_opens": opens, "pages_opened": len(e_url),
        "domains_opened": len(domains(fact_check=False)), "domains_read": len(domains(read_ids, fact_check=False)),
        "domains_cited": len(domains(cited_ids, fact_check=False)),
        "fact_check_pages_read_or_cited": len(domains(read_ids | cited_ids, fact_check=True)),
        "query_samples": queries[:6],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    total = lambda k: sum(int(r.get(k) or 0) for r in rows)   # noqa: E731
    steps = max(1, total("hi_steps"))
    return {
        "runs": len(rows), "teacher_calls_per_run": round(total("teacher_calls") / n, 1),
        "input_tokens_per_run": round(total("input_tokens") / n),
        "finish_rate": round(sum(1 for r in rows if r.get("decision") == "FINISH") / n, 2),
        "hi_steps": total("hi_steps"), "search_share": round(total("search") / steps, 3),
        "read_record_share": round(total("read_record") / steps, 3), "inspect_share": round(total("inspect") / steps, 3),
        "reports_completed": total("reports_completed"), "reports_incomplete": total("reports_incomplete"),
        "longest_same_action_run_max": max((int(r["longest_same_action_run"]) for r in rows), default=0),
        "longest_same_action_run_median": statistics.median([int(r["longest_same_action_run"]) for r in rows]) if rows else 0,
        "rationale_share": round(total("outputs_with_rationale") / max(1, total("outputs")), 3),
        "fact_check_urls_seen": total("fact_check_urls_seen"), "fact_check_pages_opened": total("fact_check_pages_opened"),
        "guard_fact_check_excluded": total("guard_fact_check_excluded"), "guard_after_cutoff": total("guard_after_cutoff"),
        **claim_aggregate([r["claim_sources"] for r in rows if r.get("claim_sources")]),
    }


def claim_aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"claim_runs": 0}
    n = len(items)
    mean = lambda k: round(sum(float(x.get(k) or 0) for x in items) / n, 2)   # noqa: E731
    queries = sum(int(x.get("queries") or 0) for x in items)
    return {"claim_runs": n, "claim_queries_per_run": mean("queries"),
            "claim_fact_check_query_share": round(sum(int(x.get("fact_check_queries") or 0) for x in items) / max(1, queries), 3),
            "claim_pages_opened_per_run": mean("pages_opened"), "claim_domains_opened_per_run": mean("domains_opened"),
            "claim_domains_read_per_run": mean("domains_read"), "claim_domains_cited_per_run": mean("domains_cited"),
            "claim_fact_check_pages_used": sum(int(x.get("fact_check_pages_read_or_cited") or 0) for x in items)}


def export_metrics(export_dir: Path | None) -> dict[str, Any]:
    if export_dir is None or not (export_dir / "manifest.json").is_file():
        return {}
    m = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    acc_hi = []
    if (export_dir / "hi_accepted.jsonl").is_file():
        acc_hi = [json.loads(l)["metadata"] for l in (export_dir / "hi_accepted.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    share: dict[str, float] = defaultdict(float)
    for meta in acc_hi:
        share[str(meta.get("action"))] += float(meta.get("sample_weight") or 0)
    total = sum(share.values()) or 1.0
    sup = sorted(int(x.get("supervised_tokens_est") or 0) for x in acc_hi)
    return {
        "export": export_dir.name,
        "verifier_tag": export_dir.name.split("compact_16k.", 1)[1] if "compact_16k." in export_dir.name else "openrouter(qwen3.8-27b)",
        "counts": m.get("counts"), "hi_weight_sum": (m.get("sample_weight_sum") or {}).get("hi"),
        "gia_weight_sum": (m.get("sample_weight_sum") or {}).get("gia"),
        "gia_accepted_answers": (m.get("by_unit") or {}).get("gia", {}).get("answer", 0),
        "hi_weight_share_by_action": {k: round(v / total, 3) for k, v in sorted(share.items(), key=lambda kv: -kv[1])},
        "supervised_tokens_p50_hi": sup[len(sup) // 2] if sup else None,
        "fact_check_exposure": m.get("fact_check_exposure"), "report_status": m.get("report_status"),
        "rationale_rows": m.get("rationale_rows"), "weight_adjustments": m.get("weight_adjustments"),
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, dict):
        return "; ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    return str(v)


def markdown(before: dict[str, Any], after: dict[str, Any], eb: dict[str, Any], ea: dict[str, Any], labels: tuple[str, str]) -> str:
    lines = [f"# 对照：{labels[0]} → {labels[1]}", "", "## 教师侧行为（只看 run，不依赖核验器）", "",
             f"| 指标 | {labels[0]} | {labels[1]} |", "|---|---|---|"]
    keys = [("runs", "run 数"), ("teacher_calls_per_run", "每 run 教师调用"), ("input_tokens_per_run", "每 run 输入 token"),
            ("finish_rate", "GIA 给出 FINISH 的比例"), ("hi_steps", "HI 动作步数"), ("search_share", "SEARCH 占步数"),
            ("read_record_share", "READ_RECORD 占步数"), ("inspect_share", "INSPECT 占步数"),
            ("reports_completed", "报告 COMPLETED"), ("reports_incomplete", "报告 INCOMPLETE"),
            ("longest_same_action_run_max", "最长同动作连击（最大）"), ("longest_same_action_run_median", "最长同动作连击（中位）"),
            ("rationale_share", "带 rationale 的输出占比"), ("fact_check_urls_seen", "轨迹里出现的核查站点 URL 数"),
            ("fact_check_pages_opened", "实际打开的核查站点页面数"), ("guard_fact_check_excluded", "护栏拦下的核查站点候选/打开"),
            ("guard_after_cutoff", "护栏按日期拦下"),
            ("claim_runs", "声明类 run 数"), ("claim_queries_per_run", "声明类：每 run 检索查询数"),
            ("claim_fact_check_query_share", "声明类：带核查字样的查询占比"),
            ("claim_pages_opened_per_run", "声明类：每 run 打开页面数"), ("claim_domains_opened_per_run", "声明类：每 run 打开的非核查域名数"),
            ("claim_domains_read_per_run", "声明类：每 run 实际读过的非核查域名数"),
            ("claim_domains_cited_per_run", "声明类：每 run 报告引用的非核查域名数"),
            ("claim_fact_check_pages_used", "声明类：读过或引用的核查类页面（域名）")]
    for key, label in keys:
        lines.append(f"| {label} | {_fmt(before.get(key, '—'))} | {_fmt(after.get(key, '—'))} |")
    if eb or ea:
        lines += ["", "## 数据侧（导出；两边核验器：" + f"{eb.get('verifier_tag', '—')} / {ea.get('verifier_tag', '—')}）", "",
                  f"| 指标 | {labels[0]} | {labels[1]} |", "|---|---|---|"]
        for key, label in [("counts", "行数"), ("hi_weight_sum", "HI 权重和"), ("gia_weight_sum", "GIA 权重和"),
                           ("gia_accepted_answers", "GIA accepted 答案数"), ("hi_weight_share_by_action", "HI 权重份额（按动作）"),
                           ("supervised_tokens_p50_hi", "HI 监督 token 中位"), ("report_status", "accepted 报告状态"),
                           ("fact_check_exposure", "核查站点沾染"), ("rationale_rows", "带 rationale 的 accepted 行"),
                           ("weight_adjustments", "权重调整次数")]:
            lines.append(f"| {label} | {_fmt(eb.get(key, '—'))} | {_fmt(ea.get(key, '—'))} |")
    lines += ["", "说明：教师侧指标不依赖核验器，可直接比；数据侧指标只有两边用同一个核验器时才可比。",
              "READ_RECORD 占比高本身不一定是坏事（读页面是取证的一部分）；声明类案例看\"每 run 查询数、实际读过和引用的独立域名数\"，",
              "不看 SEARCH 占步数。VideoSham 是给定媒体比对，本来不搜索。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="对照两批 run / 导出")
    parser.add_argument("--before-runs", nargs="+", required=True, help="改进前的 run 名或路径")
    parser.add_argument("--after-runs", nargs="*", default=None, help="改进后的 run 名或路径")
    parser.add_argument("--after-round", default=None, help="或者：按轮次标签在 runs/ 下找 *_<round>_*")
    parser.add_argument("--before-export", default=None)
    parser.add_argument("--after-export", default=None)
    parser.add_argument("--labels", nargs=2, default=("before", "after"))
    parser.add_argument("--out", default=None, help="写 markdown（同名 .json 一并写）")
    args = parser.parse_args()
    runs_dir = SFT / "runs"

    def resolve(names: list[str]) -> list[Path]:
        return [Path(n) if Path(n).is_dir() else runs_dir / n for n in names]

    before = resolve(args.before_runs)
    if args.after_runs:
        after = resolve(args.after_runs)
    elif args.after_round:
        after = sorted(p for p in runs_dir.glob(f"*_{args.after_round}_*") if p.is_dir() and ".failed-" not in p.name)
    else:
        raise SystemExit("要 --after-runs 或 --after-round")
    b_rows = [run_metrics(p) for p in before if p.is_dir()]
    a_rows = [run_metrics(p) for p in after if p.is_dir()]
    b, a = aggregate(b_rows), aggregate(a_rows)
    def export_dir(value: str | None) -> Path | None:
        if not value:
            return None
        if "*" in value:
            matches = sorted(Path(value).parent.glob(Path(value).name))
            return matches[-1] if matches else None
        return Path(value)

    eb = export_metrics(export_dir(args.before_export))
    ea = export_metrics(export_dir(args.after_export))
    text = markdown(b, a, eb, ea, tuple(args.labels))
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps({"before": {"aggregate": b, "runs": b_rows, "export": eb},
                                                        "after": {"aggregate": a, "runs": a_rows, "export": ea}},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写到 {out} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
