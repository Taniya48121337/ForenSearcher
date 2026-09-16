"""把试点 runs/ 里每个已发布的 Workspace Report 用三层核验器重核一遍，得到第 0 轮错误分布。

用法：
    python -m foren_searcher.datagen.verify.reverify --runs experiments/sft_data/runs --out experiments/sft_data/verdicts_r0.jsonl
    python -m foren_searcher.datagen.verify.reverify ... --judge openrouter --model qwen/qwen3.5-122b-a10b

--judge none（默认）只跑第一层，产物里所有处置都是 uncertain / negative，用来先看硬错分布；
接上核验模型后重跑，同一文件被覆盖，summary 里的 layers 会显示三层都跑了。
核验模型必须与教师模型不同（试点教师是 Codex CLI / gpt-6-astra）。

来源限制：run_config.json 没有排除名单，这里从 trajectory / corpus 的 known_source_keys 读不到；先不带 Case，
案卷化（cases/）落地后由 --cases 目录补上第一层的案卷断言。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...store import GlobalStore
from ..errors import tally
from .context import StoreLookup
from .judges import build_judge
from .verdict import Verdict, verify_report, write_verdicts


def _run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "store.sqlite3").is_file())


def _visible_ids(run_dir: Path, task_id: str) -> set[str] | None:
    """从 trajectory.jsonl 的最后一次 model_input 事件取该工作区提交报告时可见的 E。取不到返回 None（不查可见性）。"""

    path = run_dir / "trajectory.jsonl"
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event_type") == "model_input" and event.get("actor_id") == task_id:
            last = event
    if last is None:
        return None
    view = (last.get("payload") or {}).get("model_view") or {}
    ids: set[str] = set()
    for card in view.get("evidence_view") or []:
        if isinstance(card, dict) and isinstance(card.get("evidence_id"), str):
            ids.add(card["evidence_id"])
    materials = view.get("materials") or {}
    for key in (materials if isinstance(materials, dict) else {}):
        if isinstance(key, str) and key.startswith("E"):
            ids.add(key)
    for eid in view.get("content_observed_evidence_ids") or []:
        if isinstance(eid, str):
            ids.add(eid)
    return ids or None


async def reverify_run(run_dir: Path, judge: Any, judge_name: str | None, *, check_visibility: bool) -> list[Verdict]:
    store = GlobalStore(run_dir / "store.sqlite3", run_dir / "artifacts")
    lookup = StoreLookup(store)
    config = {}
    if (run_dir / "run_config.json").is_file():
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    verdicts = []
    for report in store.workspace_report_records():
        try:
            task = store.workspace_task_record(report.task_id)
            task_text = str(task.get("task") or "")
        except KeyError:
            task_text = ""
        subject = {"run": run_dir.name, "dataset": config.get("dataset"), "case": config.get("case"),
                   "case_id": f"{str(config.get('dataset') or '').lower()}:{config.get('case')}" if config.get("case") else None,
                   "workspace_id": report.task_id, "teacher": (config.get("configured_model") or {}).get("model")}
        visible = _visible_ids(run_dir, report.task_id) if check_visibility else None
        verdict = await verify_report(report.to_dict(), lookup, subject=subject, task_text=task_text or None,
                                      judge=judge, visible_ids=visible, judge_model=judge_name)
        verdicts.append(verdict)
    return verdicts


def summarize(verdicts: list[Verdict]) -> dict[str, Any]:
    all_errors = [e for v in verdicts for e in v.errors]
    hard = [e for e in all_errors if e.extra.get("confidence") != "flag"]
    return {
        "reports": len(verdicts),
        "disposition": dict(Counter(v.disposition for v in verdicts)),
        "layers": dict(Counter(",".join(v.layers_run) for v in verdicts)),
        "errors_total": {k: n for k, n in tally(all_errors).items() if n},
        "errors_firm": {k: n for k, n in tally(hard).items() if n},
        "reports_with_firm_error": sum(1 for v in verdicts if any(e.extra.get("confidence") != "flag" for e in v.errors)),
        "tainted_findings": sum(len(v.tainted_findings) for v in verdicts),
        "pending": dict(Counter(p.get("reason", "?") for v in verdicts for p in v.pending)),
    }


async def main_async(args: argparse.Namespace) -> int:
    root = Path(args.runs)
    judge = build_judge(args.judge, model=args.model, config_path=args.openrouter_config)
    judge_name = getattr(judge, "name", None) if judge else None
    verdicts: list[Verdict] = []
    dirs = _run_dirs(root)[: args.limit] if args.limit else _run_dirs(root)
    for run_dir in dirs:
        try:
            batch = await reverify_run(run_dir, judge, judge_name, check_visibility=not args.no_visibility)
        except Exception as exc:  # noqa: BLE001 —— 单个 run 坏了不应中断全部
            print(f"跳过 {run_dir.name}: {type(exc).__name__}: {exc}")
            continue
        verdicts.extend(batch)
        print(f"{run_dir.name}: {len(batch)} 份报告，" + ", ".join(f"{v.subject['workspace_id']}={v.disposition}" for v in batch))
    out = write_verdicts(args.out, verdicts)
    summary = summarize(verdicts)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用三层核验器重核试点 runs 的已发布报告")
    parser.add_argument("--runs", default="experiments/sft_data/runs")
    parser.add_argument("--out", default="experiments/sft_data/verdicts_r0.jsonl")
    parser.add_argument("--judge", default="none", choices=["none", "openrouter"])
    parser.add_argument("--model", default=None, help="核验模型名（必须与教师不同）")
    parser.add_argument("--openrouter-config", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-visibility", action="store_true", help="不从 trajectory 检查引用是否可见")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
