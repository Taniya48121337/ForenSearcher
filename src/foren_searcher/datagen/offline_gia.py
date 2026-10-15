"""数据来源 D：GIA 离线换题。同一组已提交的真实工作区报告，配不同的总任务，让教师离线重做 GIA 的这一步决定。

它检验的命题是"相同的局部报告在不同的全局问题下，应对应不同的补查 / 停止决定"。所以每组变体必须同时有
expected_decision=FINISH 和 =INVESTIGATE 的题（程序强制），否则学生会学成"看到报告就结束"。

    python -m foren_searcher.datagen.offline_gia --variants experiments/sft_data/offline_gia --dry-run
    python -m foren_searcher.datagen.offline_gia --endpoint http://127.0.0.1:11434/v1 --model qwen3vl-8b-32k:latest

变体文件（experiments/sft_data/offline_gia/<源 run>.json，人写、人审）：
    {"source_run": "true_4124903_r1", "source_request": "request_0022",
     "variants": [{"id": "v1", "task": "……", "expected_decision": "FINISH", "rationale": "为什么这题现有报告够 / 不够"}, …]}

每个变体落成一个轻量 run 目录（runs_offline/<源 run>__D_<id>/），结构与真实 run 相同，核验和导出直接复用：
- store.sqlite3 复制源 run；图片不复制，run_config.artifacts_from 指向源 run 的 artifacts；
- requests/request_NNNN/ 由教师写（同 run_cases 的教师：服务器 / 本地 OpenAI 兼容服务，或 Codex CLI）；
- trajectory.jsonl 只有这一步的 raw_model_output / model_output 事件，execution_links.json 同 run_cases；
- run_config：origin=D、expected_decision、source_run / source_request、task；dataset/case 继承源案例（只用于划分与 family，
  runlog 对 D 的 case_id 返回 None，原案卷的要求与标签不套用到新题上）。
导出时 export.offline_gate 再加一道：决定类型 ≠ expected_decision → rejected；一致才看单元核验结果。

状态怎么改：取源 run 那一次 GIA 请求的 global_view，只换 task 与 C0 材料里的任务文字；报告、证据卡片、附图、轮次约束原样保留。
不改写报告（与 gia_text_states 雏形相同的原则：报告必须是真实 HI 产物）。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SFT = Path("experiments") / "sft_data"
DEFAULT_VARIANTS = SFT / "offline_gia"
DEFAULT_SOURCE_RUNS = SFT / "runs"
DEFAULT_OUT = SFT / "runs_offline"
DECISIONS = ("FINISH", "INVESTIGATE")


class VariantError(ValueError):
    pass


@dataclass
class Variant:
    source_run: str
    source_request: str
    id: str
    task: str
    expected_decision: str
    rationale: str
    file: str

    @property
    def run_name(self) -> str:
        return f"{self.source_run}__D_{self.id}"


def load_variants(directory: str | Path) -> list[Variant]:
    """读全部变体文件并校验：字段齐全、id 不重复、每个文件同时有 FINISH 与 INVESTIGATE。"""

    out: list[Variant] = []
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        raise VariantError(f"{directory} 下没有变体文件")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        run, request = data.get("source_run"), data.get("source_request")
        if not run or not request:
            raise VariantError(f"{path.name}: 缺 source_run / source_request")
        seen: set[str] = set()
        kinds: set[str] = set()
        for item in data.get("variants") or []:
            vid, task = str(item.get("id") or ""), str(item.get("task") or "").strip()
            expected = str(item.get("expected_decision") or "").upper()
            if not vid or not task:
                raise VariantError(f"{path.name}: 变体缺 id 或 task")
            if vid in seen:
                raise VariantError(f"{path.name}: 变体 id {vid} 重复")
            if expected not in DECISIONS:
                raise VariantError(f"{path.name}/{vid}: expected_decision 只能是 FINISH / INVESTIGATE")
            seen.add(vid)
            kinds.add(expected)
            out.append(Variant(run, request, vid, task, expected, str(item.get("rationale") or ""), path.name))
        if kinds != set(DECISIONS):
            raise VariantError(f"{path.name}: 每组必须同时有 FINISH 与 INVESTIGATE 的题（对照），现在只有 {sorted(kinds)}")
    return out


def build_state(source_request: dict[str, Any], task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """源 GIA 请求 → (新 global_view, available_actions)。只换总任务（view.task 与 C0 材料内容），其余原样。"""

    payload = source_request["payload"]
    view = copy.deepcopy(payload["global_view"])
    view["task"] = task
    for key in ("original_materials", "current_materials"):
        for item in view.get(key) or []:
            if isinstance(item, dict) and item.get("reference_id") == "C0" and item.get("record_type") == "material":
                item["content"] = json.dumps({"task": task}, ensure_ascii=False)
    view.pop("model_reasoning", None)
    return view, list(payload.get("available_actions") or [])


def check_source(variant: Variant, view: dict[str, Any]) -> None:
    constraints = view.get("runtime_constraints") or {}
    if variant.expected_decision == "INVESTIGATE" and constraints.get("can_create_round") is False:
        raise VariantError(f"{variant.run_name}: 源状态已不能开新轮，INVESTIGATE 题无效（换一个 can_create_round=true 的 GIA 请求）")
    if constraints.get("round_in_progress"):
        raise VariantError(f"{variant.run_name}: 源状态还有 HI 在跑（WAIT 分支），不适合离线换题")


def status_of(out_dir: Path, variant: "Variant | None" = None) -> str:
    """todo / done / incomplete / stale。stale = 目录里的 run 是按旧题面跑的（变体文件改过 task 或 expected_decision）。"""

    if not out_dir.exists():
        return "todo"
    if variant is not None and (out_dir / "run_config.json").is_file():
        config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
        if config.get("task") != variant.task or config.get("expected_decision") != variant.expected_decision:
            return "stale"
    return "done" if (out_dir / "execution_links.json").is_file() else "incomplete"


def retire(out_dir: Path) -> Path:
    """把旧题面的 run 挪到 <out>/_stale/<名>-<时间>/（不删，留作对照），腾出位置重跑。
    放进子目录是为了不让按目录扫 run 的工具（reverify 的 _run_dirs 等）把旧题面当成有效 run。"""

    target = out_dir.parent / "_stale" / f"{out_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    out_dir.rename(target)
    return target


async def run_variant(variant: Variant, *, source_runs: Path, out_root: Path, teacher: str, endpoint: str | None,
                      model: str | None, thinking: bool, specs_root: str | Path, hi_spec: str, gia_spec: str) -> dict[str, Any]:
    from .. import GlobalStore
    from ..local_models import TransformersGiaPolicy
    from .links import write_execution_links
    from .run_cases import make_teacher
    from .teacher import spec_instructions

    source = source_runs / variant.source_run
    request = json.loads((source / "requests" / variant.source_request / "request.json").read_text(encoding="utf-8"))
    if request["payload"].get("workspace_id") or "global_view" not in request["payload"]:
        raise VariantError(f"{variant.source_run}/{variant.source_request} 不是 GIA 请求")
    view, actions = build_state(request, variant.task)
    check_source(variant, view)
    source_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))

    out = out_root / variant.run_name
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "store.sqlite3", out / "store.sqlite3")
    artifacts = source.resolve() / "artifacts"
    instructions, spec_ids = spec_instructions(specs_root, hi_spec, gia_spec)
    store = GlobalStore(out / "store.sqlite3", artifacts)
    generator = make_teacher(teacher, store=store, output=out, instructions=instructions, endpoint=endpoint, model=model,
                             thinking=thinking)
    events: list[dict[str, Any]] = []
    constraints = view.get("runtime_constraints") or {}

    def observe(value: dict[str, Any]) -> None:
        events.append({"id": len(events) + 1, "event_type": "raw_model_output", "actor_id": "GIA", "actor_role": "GIA",
                       "payload": {**value, "turn_index": constraints.get("turn_index"), "round_index": constraints.get("round_index")}})

    generator.generation_observer = observe
    config = {
        "origin": "D", "mode": "offline_gia", "dataset": source_config.get("dataset"), "case": source_config.get("case"),
        "source_run": variant.source_run, "source_request": variant.source_request, "variant_id": variant.id,
        "variants_file": variant.file, "task": variant.task, "expected_decision": variant.expected_decision,
        "rationale": variant.rationale, "artifacts_from": os.path.relpath(artifacts, out.resolve()),
        "gia_output_source": "openai_compatible" if teacher == "endpoint" else "codex_cli",
        "configured_model": getattr(generator, "model_config", None), "teacher_specs": spec_ids, "instruction": instructions,
        "driver": "foren_searcher.datagen.offline_gia", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    error: str | None = None
    decision: dict[str, Any] | None = None
    try:
        decision = await TransformersGiaPolicy(generator).decide(model_view=view, history=(), actions=tuple(actions))
        events.append({"id": len(events) + 1, "event_type": "model_output", "actor_id": "GIA", "actor_role": "GIA",
                       "payload": {"output": decision}})
    except Exception as exc:  # noqa: BLE001 —— 教师两次都没给出合法决定：记下来，这个变体不出数据
        error = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    (out / "trajectory.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    result = {"run": variant.run_name, "expected": variant.expected_decision, "error": error,
              "decision": (decision or {}).get("decision"), "teacher_calls": len(events) - (1 if decision else 0)}
    if not events or (decision is None and error):
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    if decision is not None and not any(e["event_type"] == "raw_model_output" for e in events):
        result["error"] = "策略没调用教师就给出了决定（兜底分支），不能作为训练数据"
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    write_execution_links(out)
    result["matches_expected"] = result["decision"] == variant.expected_decision
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GIA 离线换题（数据来源 D）")
    parser.add_argument("--variants", default=str(DEFAULT_VARIANTS))
    parser.add_argument("--source-runs", default=str(DEFAULT_SOURCE_RUNS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--only", nargs="+", default=None, help="只跑这些 run 名（<源 run>__D_<id>）或源 run 名")
    parser.add_argument("--teacher", default="endpoint", choices=["endpoint", "codex"])
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--specs", default="specs")
    parser.add_argument("--hi-spec", default="hi@0.4.1")
    parser.add_argument("--gia-spec", default="gia@0.6.0")     # 10-05 晚：运行时 GIA 提示已同步到 0.6.0
    parser.add_argument("--dry-run", action="store_true", help="只校验变体、列计划")
    parser.add_argument("--then-verify", action="store_true", help="跑完后对全部已完成的 D run 调 scripts/complete_verify_local.py")
    parser.add_argument("--judge-endpoint", default=None)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args(argv)

    variants = load_variants(args.variants)
    if args.only:
        variants = [v for v in variants if v.run_name in args.only or v.source_run in args.only]
    out_root = Path(args.out)
    finish = sum(v.expected_decision == "FINISH" for v in variants)
    print(f"变体 {len(variants)} 个（FINISH {finish} / INVESTIGATE {len(variants) - finish}），来自 "
          f"{len({v.source_run for v in variants})} 个源 run", flush=True)
    for v in variants:
        print(f"  {status_of(out_root / v.run_name, v):10s} {v.run_name:50s} 期望 {v.expected_decision}", flush=True)
    if args.dry_run:
        return 0
    endpoint = model = None
    if args.teacher == "endpoint":
        from .run_cases import _teacher_target

        endpoint, model = _teacher_target(argparse.Namespace(teacher="endpoint", endpoint=args.endpoint, model=args.model))
    results = []
    for v in variants:
        state = status_of(out_root / v.run_name, v)
        if state == "stale":
            print(f"{v.run_name}: 题面已修订，旧 run 挪到 {retire(out_root / v.run_name)}，重跑", flush=True)
        elif state != "todo":
            continue
        try:
            result = asyncio.run(run_variant(v, source_runs=Path(args.source_runs), out_root=out_root, teacher=args.teacher,
                                             endpoint=endpoint, model=model, thinking=not args.no_think, specs_root=args.specs,
                                             hi_spec=args.hi_spec, gia_spec=args.gia_spec))
        except VariantError as exc:
            result = {"run": v.run_name, "error": str(exc)}
        results.append(result)
        mark = ("✓ 与期望一致" if result.get("matches_expected") else "✗ 与期望不一致") if not result.get("error") else "失败：" + result["error"]
        print(f"{v.run_name}: 期望 {v.expected_decision}，教师给 {result.get('decision')} — {mark}", flush=True)
    done = [r for r in results if not r.get("error")]
    if done:
        agree = sum(bool(r.get("matches_expected")) for r in done)
        print(f"\n本批 {len(results)} 个：完成 {len(done)}，与期望一致 {agree}/{len(done)}", flush=True)
    if args.then_verify:
        finished = [str(out_root / v.run_name) for v in load_variants(args.variants) if status_of(out_root / v.run_name, v) == "done"]
        if not finished:
            print("没有已完成的 D run 可核验。", flush=True)
            return 1
        command = [sys.executable, str(Path("scripts") / "complete_verify_local.py"), "--runs", *finished]
        if args.judge_endpoint:
            command += ["--endpoint", args.judge_endpoint]
        if args.judge_model:
            command += ["--model", args.judge_model]
        import subprocess

        print("\n接着核验与导出：python " + " ".join(command[1:]), flush=True)
        return subprocess.call(command)
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
