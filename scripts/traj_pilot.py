"""分叉式轨迹生成试点。由 scripts/_launch.py traj_pilot 调用，也可单独跑某一步。

训练目标（每一步都对着它）：GIA / HI 的逐决策策略——同一份学生可见状态 → 一个 JSON 决定。试点要回答：
1. 在关键状态让教师给出别的做法，结果会不会真的不同（不同才有"选优"和偏好对的价值）；
2. 选出来的那支，凭学生看得到的东西能不能推出来（特权词检查、人工复读）；
3. 原件被屏蔽（V1）时能否绕路并如实说明；不让检索（V0）时是否克制；
4. 每多一支要花多少教师调用与时间；前缀强制回放能否复现原轨迹（FORK_DIVERGED 比例）。

tp0（10-11）的结论与 tp1 的改动见 claude/轨迹分叉试点tp0结果与tp1改动_20261011.md。tp1 的做法：
- 不重跑自由轨迹，直接在 tp0 的自由轨迹上分叉（--source-tag tp0）；
- 每个关键状态分两种远度：step（分叉动作执行完就停，不花教师调用，看这一步直接拿到了什么）和
  workspace（跑到分叉所在工作区交报告为止，教师用 --effort 档位续跑），各配一支"原样复现"量噪声；
- 打分按案卷 required_evidence_rule 分组、工具产出按 acquired_by 判、结论用核验模型做三分类判定（--judge）；
- 写出 pairs.jsonl：同一状态上价值差超过噪声的两个决定 = 一条偏好对候选（这就是要拿去训练的东西）。

步骤（可重复执行，做过的跳过）：
  free     教师自由跑（英文规格 + 合并工具目录 hi@0.8.0 / gia@0.9.0，10-12 起；以前是 + DETECT v2），轮次 <tag>
  cf       V1 屏蔽原件（轮次 <tag>h）+ V0 不让检索（轮次 <tag>n）
  fork     在 <source-tag> 的自由轨迹上挑关键状态 → 教师给候选 → 前缀强制回放 + 分叉（step / workspace 两种远度）
  analyze  打分，写 experiments/sft_data/traj_pilot/<tag>/summary.json、report.md、pairs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen import forks  # noqa: E402
from foren_searcher.datagen.cases import Case  # noqa: E402
from foren_searcher.datagen.run_cases import run_status  # noqa: E402

SFT = ROOT / "experiments" / "sft_data"
CASES = SFT / "cases"
RUNS = SFT / "runs"
DEFAULT_CASES = ["averitec_392", "true_4124903", "videosham_1028"]
DEFAULT_CF = ["averitec_392", "true_4124903"]          # 视频比对题不做 V1 / V0（材料本身就能定论）
COMMON = ["--teacher", "codex", "--hi-spec", "hi@0.8.0", "--gia-spec", "gia@0.9.0", "--blocklist", "exclude",
          "--search-mode", "replay_or_record", "--tool-cache", "on"]      # 10-12：DETECT 并入 SEARCH / INSPECT（合并目录）
STEP_MARGIN = 0.0          # step 远度：工具走缓存 / 检索走回放，同一动作结果确定，有差就算
WORKSPACE_MARGIN = 0.25    # workspace 远度：教师续跑有随机性，差距至少这么大、且大于原样复现与原轨迹之差
STEP_KINDS = {"source_visible", "tool_failure", "dead_end"}


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def run_batch(cases_dir: Path, ids: list[str], round_tag: str) -> int:
    argv = [sys.executable, "-m", "foren_searcher.datagen.run_cases", "--cases", str(cases_dir), "--ids", *ids,
            "--round", round_tag, "--retry-failed", *COMMON]
    say(" ".join(argv[2:]))
    return subprocess.call(argv, cwd=ROOT)


def stage_free(args: argparse.Namespace) -> None:
    run_batch(CASES, args.cases, args.tag)


def stage_cf(args: argparse.Namespace) -> None:
    if "v1" in args.cf_modes:
        out = SFT / f"cases_cf_{args.tag}"
        for name in args.cf:
            _, keys = forks.hidden_variant(CASES / f"{name}.json", out)
            say(f"V1 {name}: 屏蔽 {keys}")
        run_batch(out, args.cf, args.tag + "h")
    if "v0" in args.cf_modes:
        out0 = SFT / f"cases_v0_{args.tag}"
        for name in args.cf:
            forks.no_retrieval_variant(CASES / f"{name}.json", out0)
        say(f"V0 {args.cf}: 不让检索")
        run_batch(out0, args.cf, args.tag + "n")


# ---------------------------------------------------------------- 分叉

def source_run(name: str, args: argparse.Namespace) -> Path:
    """分叉用的自由轨迹：<name>_<source-tag>_1，没跑完就依次试 --fallback-source 里的轮次（tp2 复用 tp0 的两条）。"""

    for tag in [args.source_tag, *args.fallback_source]:
        run = RUNS / f"{name}_{tag}_1"
        if run_status(run) == "done":
            return run
    return RUNS / f"{name}_{args.source_tag}_1"


def snapshot_cutoff(run: Path) -> str:
    """源轨迹跑完的时刻（本地时间，与快照 recorded_at 同格式）。分叉回放只用这之前录下的检索结果。"""

    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime((run / "result.json").stat().st_mtime))


def branch_state(target: Path) -> str:
    if (target / "fork_diverged.json").is_file():
        return "diverged"
    if (target / "fork_stop.json").is_file():
        return "horizon"                                   # 按预定远度停下，算跑完
    return run_status(target)


def _branch(case_file: Path, source: Path, name: str, plan: dict, args: argparse.Namespace, horizon: str) -> str:
    target = RUNS / name
    state = branch_state(target)
    if state in ("done", "horizon", "diverged"):
        return state
    if target.exists():
        target.rename(target.with_name(f"{name}.failed-{time.strftime('%Y%m%d-%H%M%S')}"))
    plan_dir = SFT / "traj_pilot" / args.tag / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / f"{name}.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log_dir = SFT / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FOREN_FORCED_PLAN": str(plan_path), "FOREN_FORK_HORIZON": "" if horizon == "full" else horizon,
           "FOREN_SNAPSHOT_AS_OF": snapshot_cutoff(source)}
    if args.effort:
        env["FOREN_CODEX_EFFORT"] = args.effort
    argv = [sys.executable, "-m", "foren_searcher.datagen.run_cases", "--one", str(case_file), "--run-name", name,
            "--round", args.tag + "f", *COMMON]
    started = time.monotonic()
    with (log_dir / f"{name}.log").open("w", encoding="utf-8") as stream:
        code = subprocess.call(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    state = branch_state(target)
    say(f"    {name}: {'作废（前缀没能复现）' if state == 'diverged' else state}，{time.monotonic() - started:.0f}s，退出码 {code}")
    return state


def _branch_name(run: Path, tag: str, horizon: str, sequence: int, label: str) -> str:
    return f"{run.name}__{tag}{horizon[0]}{sequence:03d}{label}"


def stage_fork(args: argparse.Namespace) -> None:
    model = os.environ.get("FOREN_CODEX_MODEL")
    for name in args.cases:
        run = source_run(name, args)
        if run_status(run) != "done":
            say(f"{run.name} 还没跑完，跳过分叉")
            continue
        case_file = CASES / f"{name}.json"
        case = Case.load(case_file)
        states = forks.key_states(run, limit=args.states, case=case)
        say(f"{run.name}: 关键状态 {[(k.sequence, k.workspace_id or 'GIA', k.kind) for k in states]}")
        decisions = {d.sequence: d for d in forks.load_decisions(run)}
        deep_left = args.deep
        for state in states:
            out_dir = SFT / "traj_pilot" / args.tag / "proposals" / f"{run.name}_s{state.sequence:03d}"
            try:
                kept = forks.propose(decisions[state.sequence], args.k, model=model, out_dir=out_dir)
            except Exception as exc:  # noqa: BLE001
                say(f"  s{state.sequence} 候选失败：{exc}")
                continue
            if state.role == "GIA":            # GIA 只在 FINISH 上分叉，且只比别的 FINISH 写法：答案本身就是结果，不用续跑
                kept = [c for c in kept if c.get("decision") == "FINISH"]
            alts = kept[: args.max_alts]
            say(f"  s{state.sequence}（{state.workspace_id or 'GIA'} {state.kind}）: 候选 {len(kept)} 个实质不同，跑 {len(alts)} 个")
            # step 只在"取材料"类状态上有意义（tp1：读记录 / 比对之类的一步看不出差别，全是平局）；GIA FINISH 本身就是结果
            horizons = ["step"] if state.kind in STEP_KINDS or state.role == "GIA" else []
            if deep_left > 0 and state.role == "HI":
                horizons.append("workspace")
                deep_left -= 1
            for horizon in horizons:
                _branch(case_file, run, _branch_name(run, args.tag, horizon, state.sequence, "r"),
                        forks.forced_plan(run, state.sequence, None), args, horizon)
                for index, alt in enumerate(alts):
                    _branch(case_file, run, _branch_name(run, args.tag, horizon, state.sequence, f"a{index}"),
                            forks.forced_plan(run, state.sequence, alt), args, horizon)


# ---------------------------------------------------------------- 打分与偏好对

def summarize_decision(output: dict | None) -> str:
    if not isinstance(output, dict):
        return "?"
    if "findings" in output:
        return f"REPORT {output.get('status')} ({len(output.get('findings') or [])} findings)"
    if output.get("decision"):
        return f"GIA {output['decision']}" + (f" {len(output.get('tasks') or [])} tasks" if output.get("tasks") else "")
    arguments = output.get("arguments") or {}
    if output.get("action") == "SEARCH":
        return f"SEARCH {arguments.get('query') or arguments.get('source')!r} → {arguments.get('output_type')}"
    if output.get("action") == "QUERY_MEMORY":
        return f"QUERY_MEMORY {arguments.get('query')!r}"
    request = arguments.get("request") or {}
    return f"{output.get('action')} {request.get('operation', '')} {arguments.get('targets') or arguments.get('record_id') or ''}".strip()


def step_outcome(run: Path) -> str:
    """step 远度：分叉动作的直接结果（停下那次请求里的 latest_observation）。"""

    decisions = forks.load_decisions(run)
    stop = next((d for d in decisions if (d.request_dir / "transport.json").is_file()
                 and '"horizon_stop"' in (d.request_dir / "transport.json").read_text(encoding="utf-8")), None)
    if stop is None:
        return "—"
    obs = stop.view.get("latest_observation") if isinstance(stop.view.get("latest_observation"), dict) else {}
    details = obs.get("details") if isinstance(obs.get("details"), dict) else {}
    results = forks._results(details)
    if results:
        ready = sum(1 for r in results if not forks._material_failed(r))
        return f"{obs.get('action')}: {len(results)} 个候选，{len(results) - ready} 个取不到"
    status = details.get("status") or ("ok" if obs else "—")
    return f"{obs.get('action') or ''} {status}".strip()


def _judge(args: argparse.Namespace) -> Any:
    if args.judge == "none":
        return None
    from foren_searcher.datagen.verify.judges import build_judge

    try:
        return build_judge(args.judge, cache_path=SFT / "traj_pilot" / args.tag / "judge_cache.jsonl")
    except BaseException as exc:  # noqa: BLE001 —— 核验模型不可用就退回子串匹配，不挡打分
        say(f"核验模型不可用（{exc}），结论改用子串匹配")
        return None


def verdict_judgment(judge: Any, run: Path, case: Case) -> dict | None:
    if judge is None:
        return None
    from foren_searcher.datagen.verify.label_judged import judge_label, label_checks

    answer, _ = forks.final_texts(run)
    checks = label_checks(case)
    if not answer or not checks:
        return None
    label, aliases = checks[0]
    judged, failure = asyncio.run(judge_label(judge, answer, case.task_for("en") if hasattr(case, "task_for") else case.task_text,
                                              label, aliases))
    if judged is None or not (judged["status"] == "absent" or judged["quote_verified"]):
        return None
    return judged


def _row(run: Path, case: Case, atoms: list, scope: str, workspace: str | None, judge: Any, *, branch: bool) -> dict:
    state = branch_state(run)
    judged = verdict_judgment(judge, run, case) if scope == "answer" and state == "done" else None
    value = forks.evaluate(run, atoms, scope=scope, workspace=workspace, verdict_judgment=judged)
    decisions = forks.load_decisions(run)
    fork = next((d for d in decisions if d.forced == "branch"), None)
    return {"run": run.name, "status": state, **value,
            "live_teacher_calls": sum(1 for d in decisions if d.forced is None and d.output is not None),
            "fork": summarize_decision(fork.output) if fork else None,
            "fork_raw": fork.raw if fork else None,
            "leaked_terms": forks.leaked_terms(fork.payload, fork.output, case) if fork else [],
            "step_outcome": step_outcome(run) if branch and scope == "progress" else None}


def _atoms_text(row: dict) -> str:
    parts = []
    for a in row.get("atoms", []):
        if a.get("value") is None:
            continue
        name = a["key"] if len(a["key"]) < 28 else a["key"][:25] + "…"
        parts.append(f"{a['kind'][:3]}:{name}={a['value']:g}" + ("" if a.get("required", True) else "(选)"))
    return "; ".join(parts)


def stage_analyze(args: argparse.Namespace) -> None:
    out = SFT / "traj_pilot" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    judge = _judge(args)
    summary: dict = {"tag": args.tag, "source_tag": args.source_tag, "judge": args.judge, "cases": {}}
    pairs: list[dict] = []
    lines = [f"# 分叉试点 {args.tag}（源轨迹 {args.source_tag}）", ""]
    for name in args.cases:
        case = Case.load(CASES / f"{name}.json")
        atoms = forks.fact_table(case)
        free = source_run(name, args)
        entry: dict = {"atoms": [forks.to_dict(a) for a in atoms], "free": None, "states": {}, "variants": {}}
        lines += [f"## {name}", ""]
        if run_status(free) == "done":
            entry["free"] = _row(free, case, atoms, "answer", None, judge, branch=False)
            lines.append(f"自由轨迹 {free.name}：价值 {entry['free']['value']}，教师 {entry['free']['live_teacher_calls']} 次；{_atoms_text(entry['free'])}")
            lines.append("")
        decisions = {d.sequence: d for d in forks.load_decisions(free)} if free.is_dir() else {}
        branches = sorted(p for p in RUNS.glob(f"{free.name}__{args.tag}*") if ".failed-" not in p.name)
        by_state: dict[tuple[int, str], list[Path]] = {}
        for b in branches:
            suffix = b.name.split("__", 1)[1][len(args.tag):]          # s003a0 / w015r
            horizon = {"s": "step", "w": "workspace", "f": "full"}[suffix[0]]
            by_state.setdefault((int(suffix[1:4]), horizon), []).append(b)
        for (sequence, horizon), runs in sorted(by_state.items()):
            origin = decisions.get(sequence)
            workspace = origin.workspace_id if origin else None
            scope = "answer" if (origin is not None and origin.role == "GIA") else \
                {"step": "progress", "workspace": "workspace", "full": "answer"}[horizon]
            rows = {b.name.rsplit("__", 1)[1][len(args.tag) + 4:]: _row(b, case, atoms, scope, workspace, judge, branch=True) for b in runs}
            if horizon == "workspace" and run_status(free) == "done":     # 原轨迹在同一远度上的价值：与 r 之差 = 教师续跑噪声
                base = forks.evaluate(free, atoms, scope="workspace", workspace=workspace)
                rows["orig"] = {"run": free.name, "status": "reference", **base, "live_teacher_calls": None,
                                "fork": "（原轨迹）", "fork_raw": None, "leaked_terms": [], "step_outcome": None}
            key = f"s{sequence:03d}_{horizon}"
            entry["states"][key] = {"workspace": workspace, "kind": forks.classify(origin, case) if origin else None,
                                    "original": summarize_decision(origin.output) if origin else None, "branches": rows}
            lines += [f"### 状态 {sequence}（{workspace or 'GIA'}，{entry['states'][key]['kind']}）· 远度 {horizon}", "",
                      "| 支 | 分叉点决定 | 直接结果 | 状态 | 价值 | 原子 | 续跑教师调用 |", "|---|---|---|---|---|---|---|"]
            for label, row in rows.items():
                leak = f" ⚠特权词 {row['leaked_terms']}" if row["leaked_terms"] else ""
                lines.append(f"| {label} | {row['fork'] or '—'}{leak} | {row['step_outcome'] or '—'} | {row['status']} | "
                             f"{row['value']} | {_atoms_text(row)} | {row['live_teacher_calls']} |")
            lines.append("")
            pairs.extend(make_pairs(origin, rows, horizon, name, sequence))
        for variant, suffix, cases_dir in (("V1_屏蔽原件", "h", f"cases_cf_{args.tag}"), ("V0_不让检索", "n", f"cases_v0_{args.tag}")):
            run = RUNS / f"{name}_{args.tag}{suffix}_1"
            if run_status(run) != "done":
                continue
            variant_case = Case.load(SFT / cases_dir / f"{name}.json")
            v_atoms = forks.fact_table(case, hidden_keys=variant_case.source_policy.excluded_keys, no_retrieval=suffix == "n")
            row = _row(run, case, v_atoms, "answer", None, judge, branch=False)
            row["answer"] = forks.final_texts(run)[0][:1500]
            entry["variants"][variant] = row
            leak = f"；⚠屏蔽没生效：{row['leaks']}" if row["leaks"] else ""
            verdict = next((a for a in row["atoms"] if a["kind"] == "VERDICT"), {})
            lines.append(f"{variant} {run.name}：价值 {row['value']}，结论判定 {verdict.get('status')}（{verdict.get('method')}），"
                         f"教师 {row['live_teacher_calls']} 次{leak}；{_atoms_text(row)}")
        lines.append("")
        summary["cases"][name] = entry
    lines += ["## 偏好对候选", "", f"共 {len(pairs)} 条（pairs.jsonl）。step 远度差 > {STEP_MARGIN}；workspace 远度差 ≥ {WORKSPACE_MARGIN}，"
              "且大于原样复现（r）与原轨迹（orig）之差。前缀没能复现（diverged）的支不参与。", ""]
    for p in pairs:
        lines.append(f"- {p['case']} s{p['sequence']:03d} {p['horizon']}：选 {p['chosen_label']}（{p['chosen_value']}）"
                     f"> 弃 {p['rejected_label']}（{p['rejected_value']}）{' ⚠特权词' if p['leaked_terms'] else ''}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"写好 {out / 'report.md'}（偏好对候选 {len(pairs)} 条）")


def make_pairs(origin: forks.Decision | None, rows: dict[str, dict], horizon: str, case_name: str, sequence: int) -> list[dict]:
    """同一状态上的偏好对：价值最高的一支 vs 每个明显更差的一支。workspace 远度要求差距超过"原样复现 vs 原轨迹"的噪声。
    原样复现（r）本身也是一个候选：教师原决定可以是被选的，也可以是被弃的。只收 step/workspace 状态完整的支。"""

    if origin is None:
        return []
    usable = {k: v for k, v in rows.items() if v["value"] is not None and v["status"] in ("horizon", "done") and v.get("fork_raw")}
    if len(usable) < 2:
        return []
    margin = STEP_MARGIN if horizon == "step" else WORKSPACE_MARGIN
    if horizon == "workspace" and "r" in usable and rows.get("orig", {}).get("value") is not None:
        margin = max(margin, abs(usable["r"]["value"] - rows["orig"]["value"]))
    best_label, best = max(usable.items(), key=lambda kv: (kv[1]["value"], kv[0] == "r"))
    out = []
    for label, row in usable.items():
        gap = best["value"] - row["value"]
        if label == best_label or gap <= margin:
            continue
        out.append({"case": case_name, "sequence": sequence, "horizon": horizon, "state_request": str(origin.request_dir),
                    "role": origin.role, "workspace_id": origin.workspace_id,
                    "chosen_label": best_label, "chosen_value": best["value"], "chosen_raw": best["fork_raw"], "chosen_run": best["run"],
                    "rejected_label": label, "rejected_value": row["value"], "rejected_raw": row["fork_raw"], "rejected_run": row["run"],
                    "margin": margin, "leaked_terms": sorted(set(best["leaked_terms"]) | set(row["leaked_terms"]))})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stages", nargs="+", choices=["free", "cf", "fork", "analyze"])
    parser.add_argument("--tag", default="tp1")
    parser.add_argument("--source-tag", default=None, help="分叉用哪一轮的自由轨迹（默认同 --tag）")
    parser.add_argument("--fallback-source", nargs="*", default=[], help="--source-tag 的自由轨迹没有时依次试这些轮次")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--cf", nargs="+", default=DEFAULT_CF)
    parser.add_argument("--cf-modes", nargs="+", default=["v1", "v0"], choices=["v1", "v0"], help="反事实变体：v1 屏蔽原件 / v0 不让检索")
    parser.add_argument("--states", type=int, default=3, help="每条轨迹最多几个关键状态（每个角色最多一个）")
    parser.add_argument("--k", type=int, default=3, help="每个关键状态让教师给几个候选")
    parser.add_argument("--max-alts", type=int, default=2, help="每个关键状态最多跑几个候选分支")
    parser.add_argument("--deep", type=int, default=2, help="每条轨迹有几个 HI 关键状态再跑 workspace 远度（要花教师调用）")
    parser.add_argument("--effort", default="medium", help="分叉续跑的 Codex 推理档位（空串 = 用 config.toml 的设置）")
    parser.add_argument("--judge", default="claude-cli", choices=["claude-cli", "local", "openrouter", "none"],
                        help="答案结论的语义判定；不可用时退回子串匹配")
    args = parser.parse_args(argv)
    args.source_tag = args.source_tag or args.tag
    for stage in args.stages:
        say(f"==== {stage}")
        {"free": stage_free, "cf": stage_cf, "fork": stage_fork, "analyze": stage_analyze}[stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
