"""扩充案卷第一步（本机跑）：对已有试点 run 逐个起草案卷，并导出供改写要求用的"证据摘要"。

    python scripts/_draft_new_cases.py [--prefix averimatec_ averitec_ qvhighlights_ vcsl_] [--out experiments/sft_data/cases_draft/r2_20260928]

每个 run：
  1. load_run → 摘要 _briefs/<run>.json：run_config 要点、GIA 首个任务、最终答案、各工作区报告（状态 / Finding / 所引 E）、
     全部 E（编号、label、来源 URL、获取方式、正文前 600 字）。
  2. draft_case → <out>/<run>/<dataset>_<id>.json + .review.md（每个 run 各自一个子目录，避免同一案例多个 run 互相覆盖）。
任何一步出错只记进汇总，不中断其余 run。汇总写 <out>/_起草汇总.json。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.draft_case import draft_case  # noqa: E402
from foren_searcher.datagen.runlog import load_run  # noqa: E402

SFT = ROOT / "experiments" / "sft_data"
CONFIG_KEYS = ("dataset", "case", "question_index", "question", "condition", "evidence_source_policy", "source_policy",
               "search_provider", "search", "inspect_operations", "label", "max_rounds", "max_gia_turns", "task", "instruction")


def _short(value: Any, n: int = 600) -> Any:
    if isinstance(value, str):
        return value if len(value) <= n else value[:n] + f"…（共 {len(value)} 字）"
    return value


def brief(run_dir: Path) -> dict[str, Any]:
    log = load_run(run_dir)
    try:
        gia = [d for d in log.decisions if d.role == "GIA"]
        answers = [d for d in gia if d.kind == "answer"]
        first_plan = next((d for d in gia if d.action == "INVESTIGATE"), None)
        reports = []
        for ws in log.workspaces():
            d = log.report_decision(ws)
            if d is None:
                reports.append({"workspace": ws, "report": None})
                continue
            out = d.output or {}
            reports.append({"workspace": ws, "task": _short(str(d.view.get("task") or d.view.get("local_task") or ""), 400),
                            "status": out.get("status"), "status_reason": _short(out.get("status_reason"), 300),
                            "findings": [{k: _short(v, 400) for k, v in f.items()} for f in (out.get("findings") or [])]})
        evidence = []
        n = 1
        misses = 0
        while misses < 5:
            eid = f"E{n}"
            n += 1
            if not log.lookup.has(eid):
                misses += 1
                continue
            misses = 0
            t = log.lookup.get(eid)
            evidence.append({"id": eid, "label": t.label, "source_url": t.source_url, "acquired_by": t.acquired_by,
                             "modalities": list(t.modalities), "derived_from": list(t.derived_from),
                             "content": _short(t.content, 600)})
        return {
            "run": log.name,
            "config": {k: log.config.get(k) for k in CONFIG_KEYS if k in log.config},
            "gia_task": gia[0].view.get("task") if gia else None,
            "first_plan": first_plan.output if first_plan else None,
            "answer": answers[-1].output.get("answer") if answers else None,
            "n_decisions": len(log.decisions),
            "reports": reports,
            "evidence": evidence,
        }
    finally:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", nargs="+", default=["averimatec_", "averitec_", "qvhighlights_", "vcsl_"])
    parser.add_argument("--out", default=str(SFT / "cases_draft" / "r2_20260928"))
    args = parser.parse_args()
    out = Path(args.out)
    (out / "_briefs").mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in (SFT / "runs").iterdir() if p.is_dir() and p.name.startswith(tuple(args.prefix)))
    summary = []
    for run in runs:
        row: dict[str, Any] = {"run": run.name, "has_result": (run / "result.json").exists(),
                               "has_store": (run / "store.sqlite3").exists()}
        try:
            b = brief(run)
            (out / "_briefs" / f"{run.name}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
            row.update(brief_ok=True, has_answer=bool(b["answer"]), n_evidence=len(b["evidence"]),
                       n_reports=len(b["reports"]), dataset=b["config"].get("dataset"), case=b["config"].get("case"))
        except Exception as exc:  # noqa: BLE001
            row.update(brief_ok=False, brief_error=f"{type(exc).__name__}: {exc}", brief_tb=traceback.format_exc()[-1500:])
        try:
            path = draft_case(run, out_dir=out / run.name, cases_dir=SFT / "cases")
            row.update(draft_ok=True, draft=str(Path(path).relative_to(out)))
        except Exception as exc:  # noqa: BLE001
            row.update(draft_ok=False, draft_error=f"{type(exc).__name__}: {exc}", draft_tb=traceback.format_exc()[-1500:])
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_tb")}, ensure_ascii=False), flush=True)
        summary.append(row)
    (out / "_起草汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in summary if r.get("brief_ok"))
    drafted = sum(1 for r in summary if r.get("draft_ok"))
    print(f"共 {len(summary)} 个 run：摘要成功 {ok}，起草成功 {drafted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
