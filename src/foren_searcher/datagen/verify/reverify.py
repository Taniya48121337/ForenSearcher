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
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ...store import GlobalStore
from ..errors import tally
from .context import StoreLookup
from .judges import CachedJudge, build_judge
from .verdict import Verdict, verify_report, write_verdicts


def _run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "store.sqlite3").is_file())


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


async def reverify_run(run_dir: Path, judge: Any, judge_name: str | None, *, check_visibility: bool) -> list[Verdict]:
    # GlobalStore 打开时会执行建表脚本并切 WAL，会动原文件；核验只读，所以先复制一份再开（artifacts 仍指向原目录）。
    scratch = Path(tempfile.mkdtemp(prefix="reverify_"))
    shutil.copy2(run_dir / "store.sqlite3", scratch / "store.sqlite3")
    store = GlobalStore(scratch / "store.sqlite3", run_dir / "artifacts")
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


def _say(args: argparse.Namespace, line: str) -> None:
    """同时打到屏幕和日志文件（Windows 下不用管道，双击 .cmd 也能看到实时进度）。"""

    print(line, flush=True)
    if args.log:
        with Path(args.log).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


async def main_async(args: argparse.Namespace) -> int:
    root = Path(args.runs)
    if args.log:
        Path(args.log).write_text("", encoding="utf-8")
    judge = build_judge(args.judge, model=args.model, config_path=args.openrouter_config,
                        reasoning_effort=args.reasoning, cache_path=args.cache)
    judge_name = getattr(judge, "name", None) if judge else None
    verdicts: list[Verdict] = []
    dirs = _run_dirs(root)[: args.limit] if args.limit else _run_dirs(root)
    for index, run_dir in enumerate(dirs, 1):
        try:
            batch = await reverify_run(run_dir, judge, judge_name, check_visibility=not args.no_visibility)
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
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用三层核验器重核试点 runs 的已发布报告")
    parser.add_argument("--runs", default="experiments/sft_data/runs")
    parser.add_argument("--out", default="experiments/sft_data/verdicts_r0.jsonl")
    parser.add_argument("--judge", default="none", choices=["none", "openrouter"])
    parser.add_argument("--model", default=None, help="核验模型名（必须与教师不同）")
    parser.add_argument("--openrouter-config", default=None)
    parser.add_argument("--reasoning", default="low", choices=["none", "minimal", "low", "medium", "high"],
                        help="核验模型的推理力度；核验是对照题，low 足够，输出 token 少八成")
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl",
                        help="核验模型回答的缓存文件；同样的问题不再重复付费（空字符串关闭）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个 run（先小规模试）")
    parser.add_argument("--no-visibility", action="store_true", help="不从 trajectory 检查引用是否可见")
    parser.add_argument("--log", default=None, help="进度与汇总同时写到这个文件")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
