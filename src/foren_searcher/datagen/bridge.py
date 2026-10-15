"""出错点重答（桥接，2026-10-05）：在 GIA 答案没通过核验的那个时刻，让教师从**同一状态**再答一次。

大白话：一轮数据里，最终答案被核验器判 negative 的那一行不会进训练（rejected）。这里不重跑整条调查，
只把那次 FINISH 请求原样（同一 system、同一桌面 payload、同一批附图）再交给教师，最多答 --attempts 次：
- 新答案通过核验（positive）→ 多出一条可训练的答案（gia_bridge_accepted.jsonl），
  同时和原答案组成一对"同一状态下 对 vs 错"（gia_preference_pairs.jsonl），给 RSI 方案里的离线 DPO 备选用；
- 没通过 / 教师改成继续调查（不是 FINISH）→ 只记录，不产数据。

核验与正式流程完全相同：check_answer（程序层）→ 逐句锚定 → 案卷标签语义判定 → 语义覆盖，用同一个 run 的证据库与报告判定。
导出格式与 export.py 一致（system = 规格 student_prompt；messages 末条 assistant = 新答案）。

    python -m foren_searcher.datagen.bridge --round r0e --units <verdicts_units_*.jsonl> --verdicts <verdicts_*.jsonl> \
        --judge claude-cli --model claude-sonnet-5-5 --cache experiments/sft_data/verify_cache.jsonl --gia-spec gia@0.6.0

产物在 experiments/sft_data/bridges/<round>/：requests/<run>/<request>_b<k>/（请求、附图、教师原文）、bridge_log.jsonl、
bridge_verdicts.jsonl、gia_bridge_accepted.jsonl、gia_preference_pairs.jsonl、images/、summary.md。
可重复运行：已有 response.json 的尝试不再调教师，核验走缓存。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

from ..local_models import extract_json_object
from .cases import Case, load_cases
from .runlog import Decision, RunLog, load_run
from .specs import Spec
from .verify.calibrate import recalibrate
from .verify.verdict import Verdict, read_verdicts

SFT = Path("experiments") / "sft_data"

# 与试点 Codex 教师（experiments/sft_data/cli_generator.py）逐字相同的前言：重答时教师看到的输入与原请求一致
CODEX_PREFACE = (
    "You are a policy in a separate executable investigation environment. "
    "Use only the supplied state and attached images. Do not use any Codex tools, "
    "shell, filesystem, network or external memory. Do not inspect this directory. "
    "The host executes your chosen action and will request the subsequent decision. "
    "Return exactly one JSON object conforming to the role's instructions. "
    "Do not invent observations, evidence, or future results.\n\n"
)

# runner(request, image_paths, directory) -> (原文或 None, transport 元数据)；测试里换成假的
Runner = Callable[[dict[str, Any], list[Path], Path], tuple[str | None, dict[str, Any]]]


@dataclass
class Target:
    run: str
    request: str
    case_id: str | None
    errors: list[str]


def select_targets(unit_rows: list[dict[str, Any]], runs_root: Path, *, runs: set[str] | None = None) -> list[Target]:
    """negative 的 answer 单元；run 目录不在（例如已被移到 runs_superseded）的跳过。"""

    out: list[Target] = []
    for row in unit_rows:
        if row.get("unit") != "answer" or row.get("disposition") != "negative":
            continue
        subj = row.get("subject") or {}
        run = str(subj.get("run") or "")
        if not run or (runs and run not in runs) or not (runs_root / run / "requests").is_dir():
            continue
        firm = [e["type"] for e in row.get("errors") or [] if e.get("severity") in ("major", "critical")]
        out.append(Target(run, str(subj.get("request")), subj.get("case_id"), firm))
    return out


def codex_runner(model: str | None = None, *, timeout: int = 600) -> Runner:
    """与 cli_generator 相同的 codex exec 调法（只读沙箱、不许用工具、附图走 --image）。"""

    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("找不到 codex 命令")
    model = model or os.environ.get("FOREN_CODEX_MODEL") or None

    def run(request: dict[str, Any], images: list[Path], directory: Path) -> tuple[str | None, dict[str, Any]]:
        actor = directory / "actor"
        result = directory / "response.json"
        command = [codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--json",
                   "-C", str(actor), "-o", str(result)]
        if model:
            command += ["-m", model]
        for path in images:
            command += ["--image", str(path)]
        command += ["--", "-"]
        started = time.monotonic()
        attempts = []
        completed = None
        for attempt in range(3):
            completed = subprocess.run(command, input=CODEX_PREFACE + json.dumps(request, ensure_ascii=False), text=True,
                                       encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=False)
            attempts.append({"attempt": attempt + 1, "returncode": completed.returncode})
            (directory / f"attempt_{attempt + 1:02d}_events.jsonl").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode == 0 or "Selected model is at capacity" not in completed.stdout:
                break
            if attempt < 2:
                time.sleep(15)
        assert completed is not None
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        tools = [e.get("item", {}) for e in events
                 if e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}]
        usage = next((e.get("usage", {}) for e in reversed(events) if e.get("type") == "turn.completed"), {})
        meta = {"returncode": completed.returncode, "elapsed_s": round(time.monotonic() - started, 1), "usage": usage,
                "external_tool_items": tools, "transport": "codex_cli_chatgpt", "images": len(images), "attempts": attempts,
                "configured_model": {"model": model}}
        (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode or tools or not result.is_file():
            return None, meta
        return result.read_text(encoding="utf-8"), meta

    return run


def prepare_attempt(src: Path, dst: Path) -> tuple[dict[str, Any], list[Path]]:
    """把原请求（request.json + actor 附图）复制进本次尝试目录，返回 (request, 附图路径)。"""

    (dst / "actor").mkdir(parents=True, exist_ok=True)
    request = json.loads((src / "request.json").read_text(encoding="utf-8"))
    (dst / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    images = []
    for path in sorted((src / "actor").glob("image_*")):
        target = dst / "actor" / path.name
        if not target.exists():
            shutil.copy2(path, target)
        images.append(target)
    return request, images


def teacher_request(request: dict[str, Any], renderer: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """教师重答时实际看到的请求（10-05：教师输入 = 学生输入）。

    payload 按 renderer 渲染——与 export --renderer 给学生的同一档位（renderer=None 时按原请求记录的档位）。原请求的
    request.json 存原始 payload；以前重答直接把它给教师，教师看到的比学生多。返回 (给教师的请求, 渲染信息)。"""

    from .teacher import render_for_model

    payload = json.loads(json.dumps(request.get("payload") or {}))
    if renderer:
        view = payload.get("model_view") if payload.get("workspace_id") else payload.get("global_view")
        if isinstance(view, dict):
            constraints = view.setdefault("runtime_constraints", {})
            if isinstance(constraints, dict):
                constraints["state_renderer"] = renderer      # 与 export.build_messages 同样的写法
    shown, info = render_for_model(payload)
    out = {k: v for k, v in request.items()
           if k not in ("rendered_payload", "state_render", "student_system_prompt", "teacher_instruction")}
    out["payload"] = shown
    return out, info


def ask(src: Path, dst: Path, runner: Runner, *, renderer: str | None = None) -> tuple[dict[str, Any] | None, str]:
    """一次重答：已有 response.json 就直接读（可重复运行）。返回 (解析后的输出, 状态)。"""

    result = dst / "response.json"
    if not result.is_file():
        request, images = prepare_attempt(src, dst)
        shown, info = teacher_request(request, renderer)
        (dst / "teacher_view.json").write_text(json.dumps({"state_render": info}, ensure_ascii=False), encoding="utf-8")
        raw, meta = runner(shown, images, dst)
        (dst / "transport.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if raw is None:
            return None, "teacher_failed"
        if not result.is_file():
            result.write_text(raw, encoding="utf-8")
    try:
        output = extract_json_object(result.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 教师回了坏 JSON：记下，不让整批停
        return None, "bad_output"
    if not isinstance(output, dict):
        return None, "bad_output"
    if output.get("decision") != "FINISH":
        return output, "not_finish"
    return output, "ok"


async def verify_answer(decision: Decision, log: RunLog, case: Case | None, reports: list[Verdict], useful: Any, judge: Any,
                        *, subject: dict[str, Any], judge_name: str | None) -> Verdict:
    """与 complete_units.complete_run 对答案单元做的完全相同。"""

    from .verify.answer_anchored import anchor_answer, apply_answer_anchoring
    from .verify.label_judged import check_labels
    from .verify.semantic_coverage import apply_semantic_coverage
    from .verify.units import check_answer

    unit = check_answer(decision, log, case, useful, reports, subject=subject)
    sentences, result = await anchor_answer(decision, log, useful, judge)
    apply_answer_anchoring(unit, sentences, result, judge_model=judge_name)
    await check_labels(unit, str(decision.output.get("answer") or ""), case, judge)
    await apply_semantic_coverage(unit, case, [v.to_dict() for v in reports], judge)
    return unit


def sft_rows(old: Decision, new: Decision, unit: Verdict, log: RunLog, case: Case | None, spec: Spec, *, renderer: str,
             out_dir: Path, attempt_name: str, original_errors: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(新答案的导出行, 偏好对或 None)。行格式同 export.py；偏好对只在新答案 accepted 时给。"""

    from .export import _compact, build_messages, decide_row, estimate_tokens, spec_system_match
    from .verify.verdict import VERIFIER_VERSION

    system, matches, variant, repair = spec_system_match(spec, new)
    messages, images = build_messages(system, new, renderer)
    rel: list[str] = []
    for path in images:
        target = out_dir / "images" / log.name / attempt_name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        rel.append(str(target.relative_to(out_dir)).replace("\\", "/"))
    bucket, weight, reason = decide_row(new, unit, None)
    input_est, sup_est = estimate_tokens(messages, len(rel))
    meta = {"role": "gia", "unit": "answer", "origin": "bridge", "run": log.name, "workspace": None, "round": new.round_index,
            "turn": new.turn, "request": new.request_id, "bridge_attempt": attempt_name, "case_id": log.case_id,
            "dataset": log.config.get("dataset"), "case_family": case.family_key if case else log.family_case_id,
            "split": case.effective_split if case else None, "teacher_model": log.teacher_model, "action": "FINISH",
            "spec": spec.name, "spec_hash": spec.hash(), "system_source": "spec_student_prompt", "runtime_system_matches_spec": matches,
            "system_variant": variant, "repair_turn": repair, "state_renderer": renderer, "verifier": VERIFIER_VERSION,
            "judge_model": unit.judge_model, "verdict": unit.disposition, "error_types": [e.type.value for e in unit.errors],
            "bucket": bucket, "bucket_reason": reason, "sample_weight": weight,
            "has_rationale": isinstance(new.output.get("rationale"), str), "images": new.image_evidence_ids,
            "input_tokens_est": input_est, "supervised_tokens_est": sup_est, "loss_scope": "last_assistant_only",
            "exported_at": date.today().isoformat(), "replaces_rejected": original_errors}
    row = {"messages": messages, "images": rel, "metadata": meta}
    if bucket != "accepted":
        return row, None
    pair = {"messages": messages[:-1], "images": rel, "chosen": messages[-1]["content"], "rejected": _compact(old.output),
            "metadata": {**{k: meta[k] for k in ("run", "request", "bridge_attempt", "case_id", "dataset", "case_family", "split",
                                                  "teacher_model", "spec", "spec_hash", "state_renderer", "verifier", "judge_model")},
                         "pair_source": "bridge_resample_same_state", "rejected_errors": original_errors}}
    return row, pair


async def bridge_run(run_dir: Path, targets: list[Target], case: Case | None, report_verdicts: list[Verdict], judge: Any, runner: Runner,
                     spec: Spec, *, out_dir: Path, attempts: int, renderer: str, judge_name: str | None,
                     say: Callable[[str], None] = print) -> list[dict[str, Any]]:
    from .verify.units import verify_units

    log = load_run(run_dir)
    records: list[dict[str, Any]] = []
    try:
        reports = [recalibrate(v) for v in report_verdicts if v.subject.get("run") == log.name and v.unit == "report"]
        units, useful = verify_units(log, case, reports)
        by_request = {d.request_id: d for d in log.decisions}
        subjects = {str(u.subject.get("request")): u.subject for u in units if u.unit == "answer"}
        for t in targets:
            old = by_request.get(t.request)
            if old is None or old.kind != "answer":
                records.append({"run": t.run, "request": t.request, "status": "no_decision"})
                continue
            for k in range(1, attempts + 1):
                name = f"{t.request}_b{k}"
                dst = out_dir / "requests" / log.name / name
                dst.mkdir(parents=True, exist_ok=True)
                output, status = await asyncio.to_thread(ask, run_dir / "requests" / t.request, dst, runner, renderer=renderer)
                rec: dict[str, Any] = {"run": t.run, "case_id": t.case_id, "request": t.request, "attempt": k, "status": status,
                                       "original_errors": t.errors}
                if status == "ok" and output is not None:
                    new = replace(old, output=output)
                    subject = {**subjects.get(t.request, {"run": log.name, "case_id": log.case_id}), "bridge": {"of": t.request, "attempt": k}}
                    unit = await verify_answer(new, log, case, reports, useful, judge, subject=subject, judge_name=judge_name)
                    row, pair = sft_rows(old, new, unit, log, case, spec, renderer=renderer, out_dir=out_dir, attempt_name=name,
                                         original_errors=t.errors)
                    rec.update({"disposition": unit.disposition, "errors": [e.type.value for e in unit.errors
                                                                           if e.severity.value in ("major", "critical")],
                                "verdict": unit.to_dict(), "row": row, "pair": pair})
                say(f"[{log.name}] {name}: {status}" + (f" → {rec.get('disposition')}" if "disposition" in rec else "")
                    + (f"（{', '.join(rec['errors'])}）" if rec.get("errors") else ""))
                records.append(rec)
                if rec.get("disposition") == "positive":
                    break
    finally:
        log.close()
    return records


def write_outputs(records: list[dict[str, Any]], out_dir: Path, *, round_tag: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    dump("bridge_log.jsonl", [{k: v for k, v in r.items() if k not in ("verdict", "row", "pair")} for r in records])
    dump("bridge_verdicts.jsonl", [r["verdict"] for r in records if r.get("verdict")])
    dump("gia_bridge_accepted.jsonl", [r["row"] for r in records if r.get("row") and r["row"]["metadata"]["bucket"] == "accepted"])
    dump("gia_preference_pairs.jsonl", [r["pair"] for r in records if r.get("pair")])
    targets = {(r["run"], r["request"]) for r in records}
    fixed = {(r["run"], r["request"]) for r in records if r.get("disposition") == "positive"}
    status = Counter(r["status"] for r in records)
    lines = [f"# 出错点重答（{round_tag}，{date.today().isoformat()}）", "",
             f"- 目标：{len(targets)} 个 negative 答案；教师调用 {sum(1 for r in records if r['status'] != 'no_decision')} 次",
             f"- 重答后通过：{len(fixed)} 个（= 新增 accepted 答案行与偏好对各 {len(fixed)} 条）",
             f"- 尝试状态：{dict(status)}", "", "| run | 请求 | 原错误 | 尝试 | 结果 |", "|---|---|---|---|---|"]
    for r in records:
        result = r.get("disposition") or r["status"]
        if r.get("errors"):
            result += "（" + ", ".join(sorted(set(r["errors"]))) + "）"
        lines.append(f"| {r['run']} | {r['request']} | {', '.join(sorted(set(r.get('original_errors') or [])))} | {r.get('attempt', '-')} | {result} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    from .verify.judges import add_judge_arguments, judge_from_args, judge_summary

    p = argparse.ArgumentParser(description="出错点重答：GIA 答案没通过核验的状态下让教师再答，通过的作正例与偏好对")
    p.add_argument("--round", required=True)
    p.add_argument("--units", required=True, help="verdicts_units_*.jsonl（找 negative 答案）")
    p.add_argument("--verdicts", required=True, help="verdicts_*.jsonl（同一批 run 的报告判定）")
    p.add_argument("--runs-dir", default=str(SFT / "runs"))
    p.add_argument("--cases", default=str(SFT / "cases"))
    p.add_argument("--specs", default="specs")
    p.add_argument("--gia-spec", default="gia@0.6.0")
    p.add_argument("--renderer", default="compact_16k")
    p.add_argument("--attempts", type=int, default=2, help="每个出错点最多重答几次（通过即停）")
    p.add_argument("--teacher", default="codex", choices=["codex"])
    p.add_argument("--runs", nargs="*", default=None, help="只处理这些 run")
    p.add_argument("--out-dir", default=None, help="默认 experiments/sft_data/bridges/<round>")
    p.add_argument("--dry-run", action="store_true", help="只列目标，不调教师")
    add_judge_arguments(p, default="claude-cli", choices=("local", "openrouter", "claude-cli"))
    p.add_argument("--cache", default=str(SFT / "verify_cache.jsonl"))
    args = p.parse_args(argv)

    runs_root = Path(args.runs_dir)
    unit_rows = [json.loads(line) for line in Path(args.units).read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = select_targets(unit_rows, runs_root, runs=set(args.runs) if args.runs else None)
    print(f"目标：{len(targets)} 个 negative 答案" + "".join(f"\n  {t.run} {t.request}（{', '.join(sorted(set(t.errors)))}）" for t in targets),
          flush=True)
    if args.dry_run or not targets:
        return 0
    out_dir = Path(args.out_dir) if args.out_dir else SFT / "bridges" / args.round
    spec = Spec.load(Path(args.specs) / args.gia_spec)
    cases = load_cases(args.cases)
    reports = read_verdicts(args.verdicts)
    judge = judge_from_args(args, cache_path=args.cache or None)
    judge_name = getattr(judge, "name", None)
    runner = codex_runner()
    records: list[dict[str, Any]] = []
    by_run: dict[str, list[Target]] = {}
    for t in targets:
        by_run.setdefault(t.run, []).append(t)
    for run, items in by_run.items():
        case = cases.get(str(items[0].case_id or ""))
        records += asyncio.run(bridge_run(runs_root / run, items, case, reports, judge, runner, spec, out_dir=out_dir,
                                          attempts=args.attempts, renderer=args.renderer, judge_name=judge_name))
        write_outputs(records, out_dir, round_tag=args.round)          # 每个 run 落一次盘，中途关窗口不丢
    print(write_outputs(records, out_dir, round_tag=args.round), flush=True)
    line = judge_summary(judge)
    if line:
        print(line, flush=True)
    return 0


__all__ = ["CODEX_PREFACE", "Target", "ask", "teacher_request", "bridge_run", "codex_runner", "main", "prepare_attempt", "select_targets", "sft_rows",
           "verify_answer", "write_outputs"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
