"""训练链路冒烟的后半段（清单 E1）：重新加载底座 + LoRA → 对一个真实状态推理一次 → 输出过程序核验。

    python training/smoke_infer.py --config training/configs/hi_smoke_4b.json
    （adapter 默认取 <output_dir>/final；状态默认取配置里 smoke_infer.row；都可以用参数覆盖）

做的事：
  1. 加载底座与 adapter（PeftModel.from_pretrained），确认 LoRA 层确实生效：同一输入在开 / 关 adapter 时
     最后一个位置的 logits 不同（训练过的 LoRA B 不再是 0）。
  2. 从导出行里取一个不在训练子集里的真实状态（messages[:-1] + 实际附图），按训练时同一聊天模板拼提示，贪心解码。
  3. 解析输出 JSON，按 HI 接口分两种：
       报告（status + findings）→ datagen.verify.exec_checks.exec_checks（第一层程序核验，查引用存在 / 可见、数字、
         案卷必需证据），lookup 用该 run 的 store，visible_ids 用该步视图里出现过的全部 E；
       动作（action + arguments）→ datagen.verify.units.check_step（合法动作、目标可见、重复 / 无视反馈）。
     同一状态下教师原输出也过一遍，作对照。
  4. 可选 --compare-base：关掉 adapter 再生成一次，看 LoRA 对格式有没有影响。
  5. 结果写 <output_dir>/smoke_infer.json：加载 / 生成耗时、显存峰值、原始输出、解析结果、核验错误。

通过条件（冒烟只验证链路，不验证质量）：adapter 生效、生成非空、核验跑完不抛异常。输出能否解析成 JSON、核验报了
几个错，如实记下来——几十步、20 行训出来的 4B 不指望写出合格报告。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_lora import TrainConfig, load_rows, row_key  # noqa: E402


def parse_output(text: str) -> tuple[dict[str, Any] | None, str]:
    """和 harness 一样只收一个 JSON 对象：先整段解析，再去掉 ``` 围栏，最后取第一个 {...}。"""

    body = text.strip()
    if body.startswith("<think>") and "</think>" in body:
        body = body.split("</think>", 1)[1].strip()
    for candidate in (body, re.sub(r"^```(?:json)?\s*|\s*```$", "", body)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value, "whole"
        except json.JSONDecodeError:
            pass
    start = body.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(body[start:])
            if isinstance(value, dict):
                return value, "first_object"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def kind_of(output: dict[str, Any]) -> str:
    if "status" in output and "findings" in output:
        return "report"
    if "action" in output:
        return "step"
    return "unknown"


def program_checks(output: dict[str, Any], *, run_dir: Path, request: str, cases_dir: Path) -> dict[str, Any]:
    """按输出类型走 exec_checks（报告）或 check_step（动作）。返回 {unit, errors[], ...}。"""

    from foren_searcher.datagen.cases import load_cases
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.verify.exec_checks import exec_checks
    from foren_searcher.datagen.verify.hindsight import build_useful_evidence
    from foren_searcher.datagen.verify.scope import requirements_for_workspace
    from foren_searcher.datagen.verify.units import check_step

    log = load_run(run_dir)
    try:
        decision = next(d for d in log.decisions if d.request_id == request)
        case = load_cases(cases_dir).get(log.case_id) if log.case_id else None
        unit = kind_of(output)
        result: dict[str, Any] = {"unit": unit, "workspace": decision.actor_id, "turn": decision.turn,
                                  "case": log.case_id, "visible_ids": len(decision.visible_ids)}
        allowed = [a.get("name") if isinstance(a, dict) else a for a in decision.payload.get("available_actions") or []]
        result["available_actions"] = allowed
        if unit == "report":
            task_text = str((decision.view or {}).get("task") or "")
            req_ids, scope_note = requirements_for_workspace(case, task_text, run=log.name, workspace_id=decision.actor_id)
            errors = exec_checks(output, log.lookup, case=case, visible_ids=decision.visible_ids, requirement_ids=req_ids)
            result["checker"] = "datagen.verify.exec_checks.exec_checks"
            result["scope"] = scope_note
        elif unit == "step":
            legal = output.get("action") in allowed and isinstance(output.get("arguments"), dict)
            candidate = dataclasses.replace(decision, output=output, kind="step", observation=None,
                                            status="executed" if legal else "rejected")
            earlier = [d for d in log.decisions if d.role == decision.role and d.actor_id == decision.actor_id and d.turn < decision.turn]
            useful = build_useful_evidence(log, case, [])
            verdict = check_step(candidate, earlier, useful, log.evidence_first_seen(),
                                 subject={"run": log.name, "request": request, "smoke": True})
            errors = verdict.errors
            result["checker"] = "datagen.verify.units.check_step"
            result["disposition"] = getattr(verdict, "disposition", None)
            result["checks"] = verdict.extra.get("checks")
        else:
            return {**result, "checker": None, "errors": [], "note": "输出既不是报告也不是动作，程序核验无从下手"}
        result["errors"] = [e.to_dict() for e in errors]
        result["by_severity"] = {s: sum(1 for e in errors if e.severity.value == s) for s in sorted({e.severity.value for e in errors})}
        result["n_errors"] = len(errors)
        return result
    finally:
        log.close()


def build_inputs(row: dict[str, Any], export_dir: Path, processor: Any) -> dict[str, Any]:
    prompt = processor.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images = []
    for rel in row["images"]:
        with Image.open(export_dir / rel) as im:
            images.append(im.convert("RGB"))
    return processor(text=[prompt], images=images or None, return_tensors="pt", padding=False)


def generate(model: Any, processor: Any, inputs: dict[str, Any], max_new_tokens: int) -> tuple[str, int, float]:
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in inputs.items()}
    started = time.time()
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
    new = out[0, batch["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0]), round(time.time() - started, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="默认 <output_dir>/final")
    parser.add_argument("--model", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--row", default=None, help="<run>/<request>，默认取配置 _smoke_infer.row")
    parser.add_argument("--runs-dir", default="experiments/sft_data/runs")
    parser.add_argument("--cases-dir", default="experiments/sft_data/cases")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--compare-base", action="store_true", help="关掉 adapter 再生成一次作对照")
    parser.add_argument("--out", default=None, help="默认 <output_dir>/smoke_infer.json")
    args = parser.parse_args(argv)

    raw_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = TrainConfig.load(args.config)
    model_dir = args.model or cfg.model
    export_dir = Path(args.export_dir or cfg.export_dir)
    adapter = Path(args.adapter or Path(cfg.output_dir) / "final")
    row_id = args.row or (raw_cfg.get("_smoke_infer") or {}).get("row")
    if not row_id:
        raise SystemExit("没有指定推理状态：--row <run>/<request> 或配置里 _smoke_infer.row")
    rows = {row_key(r): r for r in load_rows(export_dir, cfg.files or [f"{cfg.role}_accepted.jsonl"])}
    if row_id not in rows:
        raise SystemExit(f"{export_dir} 里没有 {row_id}")
    row = rows[row_id]
    summary_path = Path(cfg.output_dir) / "train_summary.json"
    if summary_path.is_file():
        trained_on = set(json.loads(summary_path.read_text(encoding="utf-8")).get("selected") or [])
        if row_id in trained_on:
            print(f"!! {row_id} 在训练子集里，推理结果不能说明泛化（冒烟仍可继续）")

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    result: dict[str, Any] = {"config": args.config, "model": model_dir, "adapter": str(adapter), "row": row_id,
                              "row_meta": {k: row["metadata"].get(k) for k in ("unit", "action", "case_id", "teacher_model", "input_tokens_est")}}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=getattr(torch, cfg.dtype),
                                                       device_map=cfg.device_map)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    result["load_s"] = round(time.time() - t0, 1)
    n_lora = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})))
    result["lora_modules_loaded"] = n_lora

    inputs = build_inputs(row, export_dir, processor)
    result["prompt_tokens"] = int(inputs["input_ids"].shape[1])
    result["images"] = len(row["images"])
    # adapter 是否生效：同一前缀，开 / 关 adapter 的最后位置 logits 差
    device = next(model.parameters()).device
    probe = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        on = model(**probe, logits_to_keep=1).logits[0, -1].float()
        with model.disable_adapter():
            off = model(**probe, logits_to_keep=1).logits[0, -1].float()
    result["adapter_effect_max_abs_logit_diff"] = round(float((on - off).abs().max()), 4)
    result["adapter_active"] = bool(result["adapter_effect_max_abs_logit_diff"] > 0)
    del probe, on, off

    text, n_new, gen_s = generate(model, processor, inputs, args.max_new_tokens)
    parsed, how = parse_output(text)
    result["generation"] = {"new_tokens": n_new, "seconds": gen_s, "tokens_per_s": round(n_new / max(gen_s, 1e-6), 1),
                            "text": text, "parse": how, "kind": kind_of(parsed) if parsed else None,
                            "hit_max_new_tokens": n_new >= args.max_new_tokens}
    if torch.cuda.is_available():
        result["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        result["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        result["gpu"] = torch.cuda.get_device_name(0)

    run_name, request = row_id.split("/", 1)
    run_dir = Path(args.runs_dir) / run_name
    checks_ok = False
    if not run_dir.is_dir():
        result["checks"] = {"error": f"找不到 run 目录 {run_dir}，程序核验需要该 run 的 store"}
    elif parsed is None:
        result["checks"] = {"error": "输出不是 JSON 对象，程序核验没跑（记为格式失败）"}
    else:
        try:
            result["checks"] = program_checks(parsed, run_dir=run_dir, request=request, cases_dir=Path(args.cases_dir))
            checks_ok = True
        except Exception as exc:
            result["checks"] = {"error": f"程序核验抛异常：{type(exc).__name__}: {exc}"}
    # 对照：教师在同一状态下的原输出过同一套核验
    teacher_out, _ = parse_output(row["messages"][-1]["content"])
    try:
        result["teacher_checks"] = program_checks(teacher_out, run_dir=run_dir, request=request, cases_dir=Path(args.cases_dir)) \
            if teacher_out and run_dir.is_dir() else None
    except Exception as exc:
        result["teacher_checks"] = {"error": f"{type(exc).__name__}: {exc}"}

    if args.compare_base:
        with model.disable_adapter():
            base_text, base_n, base_s = generate(model, processor, inputs, args.max_new_tokens)
        base_parsed, base_how = parse_output(base_text)
        result["base_generation"] = {"new_tokens": base_n, "seconds": base_s, "parse": base_how,
                                     "kind": kind_of(base_parsed) if base_parsed else None, "text": base_text}

    result["pass"] = {"adapter_active": result["adapter_active"], "generated": n_new > 0, "checks_ran": checks_ok,
                      "output_is_json": parsed is not None}
    result["smoke_ok"] = bool(result["adapter_active"] and n_new > 0 and checks_ok)
    out_path = Path(args.out or Path(cfg.output_dir) / "smoke_infer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in result.items() if k not in ("generation", "base_generation", "checks", "teacher_checks")}
    brief["generation"] = {k: v for k, v in result["generation"].items() if k != "text"}
    brief["checks"] = {k: result["checks"].get(k) for k in ("unit", "checker", "n_errors", "error") if k in result["checks"]}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"输出前 600 字：{text[:600]}")
    print(f"详细结果：{out_path}")
    return 0 if result["smoke_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
