"""把试点 runs/ 里每个已发布的 Workspace Report 用三层核验器重核一遍，得到第 0 轮错误分布。

用法：
    python -m foren_searcher.datagen.verify.reverify --runs experiments/sft_data/runs --out experiments/sft_data/verdicts_r0.jsonl
    python -m foren_searcher.datagen.verify.reverify --runs runs/a runs/b ... --judge local      # 本地 OpenAI 兼容服务（不花钱）
    python -m foren_searcher.datagen.verify.reverify ... --judge openrouter --model qwen/qwen3.8-27b  # 付费 API

--judge none（默认）只跑第一层，产物里所有处置都是 uncertain / negative，用来先看硬错分布；
接上核验模型后重跑，同一文件被覆盖，summary 里的 layers 会显示三层都跑了。
核验模型必须与教师模型不同（试点教师是 Codex CLI / gpt-6-astra）。

案卷：--cases 目录里有对应 case_id 的 run 走案卷断言与必需证据检查；0.3.0 起必需证据按工作区范围给（scope.py：
plan 单元的 judge_targets → REQ → required_evidence.supports 反查，没有评判就按任务文字兜底），不再把整案清单压给单个工作区。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ...store import GlobalStore
from ..errors import tally
from .anchored import BUDGET_EXIT_CODE, JudgeBudgetExhausted, budget_message
from .context import StoreLookup
from . import judges as judges_module
from ..cases import Case, load_cases
from .judges import CachedJudge, add_judge_arguments, judge_from_args, judge_summary
from .scope import requirements_for_workspace
from .verdict import Verdict, verify_report, write_verdicts


def _run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "store.sqlite3").is_file())


def expand_runs(paths: list[str]) -> list[Path]:
    """--runs 可以给一个 runs 根目录（跑其下全部 run，旧用法），也可以直接列若干个 run 目录（只跑这些）。"""

    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "store.sqlite3").is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(_run_dirs(path))
        else:
            raise SystemExit(f"找不到 run 目录：{raw}")
    return out


_EVIDENCE_TOKEN = re.compile(r'"(E\d+)"')


def _visible_ids(run_dir: Path, task_id: str) -> set[str] | None:
    """该工作区在整个回合里被展示过的 E：trajectory.jsonl 中所有 model_input 事件的 model_view 出现过的编号并集。

    evidence_view 只保留最近 12 张卡，只看最后一次输入会把早先读过、后来被挤出卡片区的 E 误判为不可见，
    所以取并集；取不到事件返回 None（不查可见性）。
    """

    path = run_dir / "trajectory.jsonl"
    if not path.is_file():
        return None
    ids: set[str] = set()
    found = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if '"model_input"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event_type") != "model_input" or event.get("actor_id") != task_id:
                continue
            found = True
            view = (event.get("payload") or {}).get("model_view")
            if isinstance(view, dict):
                ids.update(_EVIDENCE_TOKEN.findall(json.dumps(view, ensure_ascii=False)))
    return ids if found else None


async def reverify_run(run_dir: Path, judge: Any, judge_name: str | None, *, check_visibility: bool,
                       cases: dict[str, Case] | None = None, units: list[Verdict] | None = None) -> list[Verdict]:
    # GlobalStore 打开时会执行建表脚本并切 WAL，会动原文件；核验只读，所以先复制一份再开（artifacts 仍指向原目录）。
    scratch = Path(tempfile.mkdtemp(prefix="reverify_"))
    shutil.copy2(run_dir / "store.sqlite3", scratch / "store.sqlite3")
    store = GlobalStore(scratch / "store.sqlite3", run_dir / "artifacts")
    lookup = StoreLookup(store)
    config = {}
    if (run_dir / "run_config.json").is_file():
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    case_id = f"{str(config.get('dataset') or '').lower()}:{config.get('case')}" if config.get("case") else ""
    case = (cases or {}).get(case_id)
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
        req_ids, scope_note = requirements_for_workspace(case, task_text, run=run_dir.name, workspace_id=report.task_id, units=units)
        verdict = await verify_report(report.to_dict(), lookup, subject=subject, task_text=task_text or None, case=case,
                                      requirement_ids=req_ids, judge=judge, visible_ids=visible, judge_model=judge_name)
        if case is not None:
            verdict.extra["scope_note"] = scope_note
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


def _say(args: argparse.Namespace, line: str) -> None:
    """同时打到屏幕和日志文件（Windows 下不用管道，双击 .cmd 也能看到实时进度）。"""

    print(line, flush=True)
    if args.log:
        with Path(args.log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


async def main_async(args: argparse.Namespace) -> int:
    if args.log:
        Path(args.log).write_text("", encoding="utf-8")
    judges_module.FAILURE_LOG = Path(args.out).with_name("verify_failures.jsonl")   # 模型坏输出原文留档，便于排查
    judge = judge_from_args(args, cache_path=args.cache or None)
    judge_name = getattr(judge, "name", None) if judge else None
    cases = load_cases(args.cases) if args.cases else {}
    units: list[Verdict] = []
    if args.units and Path(args.units).is_file():
        units = [Verdict.from_dict(json.loads(line)) for line in Path(args.units).read_text(encoding="utf-8").splitlines() if line.strip()]
    verdicts: list[Verdict] = []
    all_dirs = expand_runs(args.runs)
    dirs = all_dirs[: args.limit] if args.limit else all_dirs
    for index, run_dir in enumerate(dirs, 1):
        try:
            batch = await reverify_run(run_dir, judge, judge_name, check_visibility=not args.no_visibility, cases=cases, units=units)
        except JudgeBudgetExhausted as exc:     # 没钱了：已完成的 run 已落盘，醒目提示后退出码 3
            write_verdicts(args.out, verdicts)
            _say(args, f"[{index}/{len(dirs)}] {run_dir.name}: 中止（预算用尽），已完成 {len(verdicts)} 份报告 → {args.out}")
            _say(args, budget_message(exc))
            line = judge_summary(judge)
            if line:
                _say(args, line)
            return BUDGET_EXIT_CODE
        except Exception as exc:  # noqa: BLE001 —— 单个 run 坏了不应中断全部
            _say(args, f"[{index}/{len(dirs)}] 跳过 {run_dir.name}: {type(exc).__name__}: {exc}")
            continue
        verdicts.extend(batch)
        _say(args, f"[{index}/{len(dirs)}] {run_dir.name}: {len(batch)} 份报告，"
             + ", ".join(f"{v.subject['workspace_id']}={v.disposition}" for v in batch))
        write_verdicts(args.out, verdicts)      # 每个 run 完成就落盘，跑到一半也能看进度
    out = write_verdicts(args.out, verdicts)
    summary = summarize(verdicts)
    if isinstance(judge, CachedJudge):
        summary["judge_cache"] = {"hits": judge.hits, "misses": judge.misses}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _say(args, json.dumps(summary, ensure_ascii=False, indent=2))
    line = judge_summary(judge)
    if line:
        _say(args, line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用三层核验器重核试点 runs 的已发布报告")
    parser.add_argument("--runs", nargs="+", default=["experiments/sft_data/runs"],
                        help="runs 根目录（跑全部），或直接列出要跑的若干 run 目录")
    parser.add_argument("--out", default="experiments/sft_data/verdicts_r0.jsonl")
    add_judge_arguments(parser, default="none")     # 核验模型必须与教师不同
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl",
                        help="核验模型回答的缓存文件；同样的问题不再重复付费（空字符串关闭）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个 run（先小规模试）")
    parser.add_argument("--no-visibility", action="store_true", help="不从 trajectory 检查引用是否可见")
    parser.add_argument("--log", default=None, help="进度与汇总同时写到这个文件")
    parser.add_argument("--cases", default="experiments/sft_data/cases", help="案卷目录（有案卷的 run 走案卷断言 + 按工作区范围的必需证据检查）；传空字符串关闭")
    parser.add_argument("--units", default="experiments/sft_data/verdicts_units_judged.jsonl", help="plan 单元模型评判，用来定各工作区的要求范围")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
