"""export：把 run 目录 + 核验结果 + 案卷 + 规格，按 v2 §3 的入选规则打包成可训练的 jsonl。

用法：
    python -m foren_searcher.datagen.export --runs experiments/sft_data/runs/true_4124903_r1 \\
        --cases experiments/sft_data/cases --verdicts experiments/sft_data/verdicts_r0_judged.jsonl \\
        --out experiments/sft_data/export/r0_true4124903 --renderer compact_16k

一行 = 一次决策（v2 §5）：
    messages  与学生推理路径 local_models._generation_messages() **完全一致**的对话（system=规格 student_prompt、
              user=按档位渲染的桌面、历史动作按 assistant/tool 轮次展开、末尾的固定说明、最后一条 user 是附图消息），
              再加一条 assistant = 教师的真实输出（紧凑 JSON）。loss 只算最后一条 assistant。
    images    附图消息里 {"type":"image"} 占位符按顺序对应的文件（export 目录内相对路径）。
    metadata  可追溯字段（run/workspace/turn/request、spec 与 hash、渲染档、核验器版本、处置、错误类型、clue_gain、权重…）。

产物目录：
    hi_accepted.jsonl  hi_held.jsonl  hi_rejected.jsonl  gia_accepted.jsonl  gia_held.jsonl  gia_rejected.jsonl
    verdicts_reports.jsonl（校准后的 report 单元）  verdicts_units.jsonl（step/plan/answer 单元）
    images/<run>/<request>/image_NN.png   manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from ..local_models import _generation_messages
from ..search.retrieval_guard import FACT_CHECK_DOMAINS, fact_check_match
from ..state_render import TIERS, RenderError, payload_chars
from .cases import Case, load_cases, split_summary
from .runlog import Decision, RunLog, load_run
from .specs import Spec
from .verify.calibrate import recalibrate
from .verify.complete_units import load_units, merge_units
from .verify.units import verify_units
from .verify.verdict import VERIFIER_VERSION, Verdict, read_verdicts

GIA_VARIANT_MARKER = "<!-- ===== 首轮派工变体（运行时在没有历史报告时使用） ===== -->"
IMAGE_TOKENS_EST = 880           # 1280×720 一张图在 Qwen3.5-9B（patch 16 × merge 2 = 32px/token）实测 880 token；精确值由 check_inputs 用真实 processor 算
CHARS_PER_TOKEN = 1.7            # 试点语料实测
WEIGHT_CAP = 3.0
FAMILY_CAP = {"hi": 40, "gia_plan": 5, "gia_answer": 3}
# 09-28 权重规则（第 0 轮数据质量评估）：
MEMORY_GAIN0_FACTOR = 0.5        # READ_RECORD / QUERY_MEMORY 且 clue_gain=0：权重 ×0.5（r0 里 READ_RECORD 占 accepted step 的 48%）
REPEAT_RUN_MIN = 4               # 同一工作区连续 ≥4 步同一动作：只留首末满权重，中间 ×0.5（4126 两个工作区各 8 连 INSPECT）
REPEAT_RUN_FACTOR = 0.5
REPORT_COMPLETED_CLEAN = 3.0     # COMPLETED 且无错误
REPORT_COMPLETED_FLAGGED = 2.0   # COMPLETED 带 minor/flag
REPORT_INCOMPLETE_SPECIFIC = 2.0 # INCOMPLETE 且 status_reason 写明缺什么
REPORT_INCOMPLETE_VAGUE = 1.0    # INCOMPLETE 但 status_reason 空泛


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- 规格 system

def spec_system_prompt(spec: Spec, decision: Decision) -> tuple[str, bool]:
    """返回 (system 文本, 运行时 system 是否与规格一致)。细节见 spec_system_match。"""

    system, matches, _variant, _repair = spec_system_match(spec, decision)
    return system, matches


def spec_system_match(spec: Spec, decision: Decision) -> tuple[str, bool, str | None, bool]:
    """(训练用 system, 是否与规格逐字一致, 对上的变体名, 是否修复轮)。

    规格可以声明多个变体（gia@0.3.0 收编了运行时全部 8 个；0.1.0/0.2.0 的旧分隔符按 findings + dispatch 读）；
    比较前去掉教师指令（decision.student_system）和策略重试的修复附注。对上时 system 用运行时原文，对不上时用规格主变体。
    """

    from .prompt_variants import match_runtime

    return match_runtime(spec.student_prompt, decision.student_system, decision.role)


# --------------------------------------------------------------------------- 行

def build_messages(system: str, decision: Decision, renderer: str) -> tuple[list[dict[str, Any]], list[Path]]:
    payload = json.loads(json.dumps(decision.payload))      # 深拷贝
    view = payload.get("model_view") if decision.role == "HI" else payload.get("global_view")
    if isinstance(view, dict):
        constraints = view.setdefault("runtime_constraints", {})
        if isinstance(constraints, dict):
            constraints["state_renderer"] = renderer
    messages: list[dict[str, Any]] = list(_generation_messages(system, payload, defer_feedback=True))
    images: list[Path] = []
    if decision.image_evidence_ids:
        notes = list(decision.image_notes)
        intro = notes[0] if notes else "以下是本次决策实际提供的图片像素。"
        captions = notes[1:] if len(notes) > 1 else [""] * len(decision.image_evidence_ids)
        content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
        for caption, path in zip(captions, decision.image_paths):
            if caption:
                content.append({"type": "text", "text": caption})
            content.append({"type": "image"})
            images.append(path)
        messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": _compact(decision.output)})
    return messages, images


def estimate_tokens(messages: list[dict[str, Any]], n_images: int) -> tuple[int, int]:
    text = 0
    for m in messages[:-1]:
        c = m.get("content")
        if isinstance(c, str):
            text += len(c)
        elif isinstance(c, list):
            text += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    supervised = len(messages[-1]["content"])
    return int(text / CHARS_PER_TOKEN) + n_images * IMAGE_TOKENS_EST, int(supervised / CHARS_PER_TOKEN)


def decide_row(decision: Decision, unit: Verdict | None, report: Verdict | None) -> tuple[str, float, str | None]:
    """返回 (bucket, sample_weight, reason)。bucket ∈ accepted | held | rejected。"""

    if decision.role == "HI" and decision.kind == "step":
        assert unit is not None
        firm = [e for e in unit.errors if e.extra.get("confidence") != "flag"]
        if firm:
            return "rejected", 0.0, "step_errors:" + ",".join(e.type.value for e in firm)
        gain = int(unit.extra.get("clue_gain", {}).get("total", 0))
        ws = report.disposition if report else "uncertain"
        weight = min(WEIGHT_CAP, 1.0 + gain)
        if ws == "positive":
            return "accepted", weight, None
        if ws == "negative":
            if gain > 0:
                return "accepted", weight, "gain_in_negative_ws"
            return "held", 0.0, "gain0_in_negative_ws"
        return "held", 0.0, "workspace_uncertain"
    if decision.role == "HI" and decision.kind == "report":
        if report is None:
            return "held", 0.0, "no_report_verdict"
        if report.disposition == "positive":
            return "accepted", report_weight(decision, report), None
        if report.disposition == "negative":
            return "rejected", 0.0, "negative_report_until_recall_calibrated"
        return "held", 0.0, "report_uncertain"
    if decision.role == "GIA" and decision.kind == "plan":
        assert unit is not None
        if unit.disposition == "positive":
            return "accepted", 1.0, None
        if unit.disposition == "negative":
            return "rejected", 0.0, "plan_errors:" + ",".join(e.type.value for e in unit.errors)
        return "held", 0.0, "plan_uncertain"
    if decision.role == "GIA" and decision.kind == "answer":
        assert unit is not None
        if unit.extra.get("tainted_input"):
            return "held", 0.0, "tainted_input"
        if unit.disposition == "positive":
            return "accepted", (2.0 if unit.extra.get("recovered_from_tainted") else 1.0), None
        if unit.disposition == "negative":
            return "rejected", 0.0, "answer_errors:" + ",".join(e.type.value for e in unit.errors)
        return "held", 0.0, "answer_uncertain"
    # GIA 的 READ_RECORD / QUERY_MEMORY：跟随其后的决定
    return "held", 0.0, "memory_action_follows_next_decision"


_SPECIFIC_REASON = re.compile(r"E\d+|R\d+|来源|证据|页面|帧|转写|字幕|原文|档案|记录|source|evidence|frame|page|transcript|archive", re.I)


def report_weight(decision: Decision, report: Verdict) -> float:
    """报告权重：COMPLETED 无错误 3 / 带错误 2；INCOMPLETE 只有在 status_reason 写明缺哪份证据时才 2，空泛的 1。
    目的：r0 里 accepted 报告 INCOMPLETE 11 : COMPLETED 9，不让"诚实放弃"比"查完"更值钱。"""

    output = decision.output if isinstance(decision.output, dict) else {}
    if output.get("status") == "COMPLETED":
        return REPORT_COMPLETED_FLAGGED if report.errors else REPORT_COMPLETED_CLEAN
    reason = str(output.get("status_reason") or "")
    specific = len(reason) >= 20 and bool(_SPECIFIC_REASON.search(reason))
    return REPORT_INCOMPLETE_SPECIFIC if specific else REPORT_INCOMPLETE_VAGUE


_E_READY = re.compile(r"(https?://[^\s\"'\\;]+)[^\"\\]{0,120}?page=ready\((E\d+)\)")
_E_SOURCE = re.compile(r"\"evidence_id\":\s*\"(E\d+)\"[^{}]{0,600}?\"source_url\":\s*\"(https?://[^\"]+)\"")
_ANY_URL = re.compile(r"https?://[^\s\"'\\<>]+")
_EID = re.compile(r"\bE\d+\b")


def fact_check_exposure(messages: list[dict[str, Any]], output: Any) -> dict[str, Any]:
    """这一行有没有沾到核查站点：context 里出现过（候选卡、证据卡）算 in_context；
    教师输出本身打开 / 读取 / 引用了核查站点的来源（SEARCH 的 source URL、READ_RECORD 的 E、finding / answer 引用的 E）算 in_target。"""

    text = json.dumps(messages[:-1], ensure_ascii=False)
    e_to_domain: dict[str, str] = {}
    for url, eid in _E_READY.findall(text):
        matched = fact_check_match(url.rstrip("\\"))
        if matched:
            e_to_domain[eid] = matched
    for eid, url in _E_SOURCE.findall(text):
        matched = fact_check_match(url)
        if matched:
            e_to_domain[eid] = matched
    in_context = sorted({fact_check_match(u.rstrip("\\")) or "" for u in _ANY_URL.findall(text)} - {""})
    target = json.dumps(output, ensure_ascii=False) if not isinstance(output, str) else output
    in_target: set[str] = set()
    for url in _ANY_URL.findall(target):
        matched = fact_check_match(url)
        if matched:
            in_target.add(matched)
    for eid in _EID.findall(target):
        if eid in e_to_domain:
            in_target.add(e_to_domain[eid])
    return {"in_context": in_context, "in_target": sorted(in_target)}


def adjust_action_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """accepted 的 HI 动作行：gain=0 的记忆动作 ×0.5；同一工作区连续 ≥4 步同一动作只留首末满权重。"""

    by_ws: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        m = row["metadata"]
        if m["role"] == "hi" and m["unit"] == "step" and m["bucket"] == "accepted":
            by_ws[(m["run"], str(m["workspace"]))].append(row)
    for items in by_ws.values():
        items.sort(key=lambda r: int(r["metadata"]["turn"]))
        for row in items:
            m = row["metadata"]
            adjustments = m.setdefault("weight_adjustments", [])
            if m["action"] in ("READ_RECORD", "QUERY_MEMORY") and int(m.get("clue_gain") or 0) == 0:
                m["sample_weight"] = round(m["sample_weight"] * MEMORY_GAIN0_FACTOR, 3)
                adjustments.append("memory_action_gain0")
        # 连续同动作
        start = 0
        while start < len(items):
            end = start
            while end + 1 < len(items) and items[end + 1]["metadata"]["action"] == items[start]["metadata"]["action"]:
                end += 1
            if end - start + 1 >= REPEAT_RUN_MIN:
                for row in items[start + 1:end]:
                    m = row["metadata"]
                    m["sample_weight"] = round(m["sample_weight"] * REPEAT_RUN_FACTOR, 3)
                    m.setdefault("weight_adjustments", []).append("repetitive_run_middle")
            start = end + 1
    return rows


def parse_share_caps(values: list[str] | None) -> dict[str, float]:
    """["videosham=0.34", ...] → {"videosham": 0.34}；份额须在 (0, 1) 内。"""

    caps: dict[str, float] = {}
    for item in values or []:
        name, sep, raw = item.partition("=")
        try:
            share = float(raw)
        except ValueError:
            share = -1.0
        if not sep or not name.strip() or not 0 < share < 1:
            raise SystemExit(f"--max-dataset-share 应写成 数据集=份额（0–1 之间），得到 {item!r}")
        caps[name.strip().lower()] = share
    return caps


def apply_dataset_share_caps(rows: list[dict[str, Any]], caps: dict[str, float]) -> dict[str, Any]:
    """按角色把某数据集在 accepted 权重和里的份额压到上限以内（2026-09-30）：只缩权重、不删行，行上记 dataset_share_cap。

    例：r0b 的 HI accepted 里 videosham 占 83/141 行、约六成权重；上限 1/3 时把 videosham 行的权重统一乘 f，
    f = cap·(T−W)/((1−cap)·W)，缩完后该数据集份额正好等于上限。只有一个数据集时无从压缩，跳过。返回每个角色的缩放记录。
    """

    report: dict[str, Any] = {}
    for role in ("hi", "gia"):
        accepted = [r for r in rows if r["metadata"]["role"] == role and r["metadata"]["bucket"] == "accepted"]
        for dataset, cap in caps.items():
            mine = [r for r in accepted if str(r["metadata"].get("dataset") or "").lower() == dataset]
            total = sum(float(r["metadata"]["sample_weight"] or 0) for r in accepted)
            weight = sum(float(r["metadata"]["sample_weight"] or 0) for r in mine)
            if not mine or weight <= 0 or weight >= total or weight / total <= cap:
                continue
            factor = cap * (total - weight) / ((1 - cap) * weight)
            for r in mine:
                m = r["metadata"]
                m["sample_weight"] = round(float(m["sample_weight"]) * factor, 4)
                m.setdefault("weight_adjustments", []).append("dataset_share_cap")
            report.setdefault(role, {})[dataset] = {"share_before": round(weight / total, 3), "cap": cap, "factor": round(factor, 4),
                                                    "rows": len(mine)}
    return report


def offline_gate(log: RunLog, decision: Decision, bucket: str, weight: float, reason: str | None) -> tuple[str, float, str | None]:
    """来源 D（离线换题）的额外门槛：决定类型必须等于变体预先写好的 expected_decision（FINISH ↔ INVESTIGATE 对照），
    不一致一律 rejected；记忆动作（READ_RECORD / QUERY_MEMORY）离线无法执行下一步，held。一致时仍要过原有的单元核验。"""

    expected = str(log.config.get("expected_decision") or "").upper()
    if decision.role != "GIA":
        return bucket, weight, reason
    if decision.kind == "memory":
        return "held", 0.0, "offline_memory_action"
    actual = "INVESTIGATE" if decision.kind == "plan" else "FINISH" if decision.kind == "answer" else None
    if expected and actual != expected:
        return "rejected", 0.0, f"offline_expected_{expected.lower()}_got_{(actual or decision.kind).lower()}"
    return bucket, weight, reason


def export_run(log: RunLog, case: Case | None, report_verdicts: list[Verdict], specs: dict[str, Spec], *, renderer: str,
               out_dir: Path, image_root: Path, judged_units: list[Verdict] | None = None,
               render_overflow: list[dict[str, Any]] | None = None,
               family_case: Case | None = None) -> tuple[list[dict[str, Any]], list[Verdict]]:
    """render_overflow：传入列表时，渲染超预算（RenderError）而跳过的决策追加到这里，供 manifest 记录。
    family_case：离线换题（D）时源案例的案卷，只用来取 family / 划分（案卷检查不适用）。"""

    reports = [recalibrate(v) for v in report_verdicts]
    units, useful = verify_units(log, case, reports)
    if judged_units:
        units = merge_units(units, [v for v in judged_units if v.subject.get("run") == log.name])
    by_unit: dict[tuple[str, str], Verdict] = {}
    for v in units:
        by_unit[(v.unit, str(v.subject.get("request")))] = v
    report_by_ws = {v.subject.get("workspace_id"): v for v in reports}
    rows: list[dict[str, Any]] = []
    for decision in log.decisions:
        if decision.role == "GIA" and not log.gia_is_model():
            continue
        spec = specs[decision.role.lower()]
        system, matches, system_variant, repair_turn = spec_system_match(spec, decision)
        try:
            messages, image_paths = build_messages(system, decision, renderer)
        except RenderError as exc:
            # 连最后一级退化都压不进预算：跳过这一行，不让整次导出崩掉
            tier = TIERS.get(renderer)
            if render_overflow is not None:
                render_overflow.append({"run": log.name, "request": decision.request_id, "role": decision.role.lower(), "kind": decision.kind,
                                        "chars": exc.chars if exc.chars is not None else payload_chars(decision.payload),
                                        "budget": exc.budget if exc.budget is not None else (tier.budget_chars if tier else None),
                                        "error": str(exc)})
            continue
        # 图片复制到 export 目录
        rel_images: list[str] = []
        for path in image_paths:
            target = image_root / log.name / decision.request_id / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            rel_images.append(str(target.relative_to(out_dir)).replace("\\", "/"))
        unit_key = {"step": "step", "plan": "plan", "answer": "answer"}.get(decision.kind)
        unit = by_unit.get((unit_key, decision.request_id)) if unit_key else None
        report = report_by_ws.get(decision.actor_id) if decision.role == "HI" else None
        bucket, weight, reason = decide_row(decision, unit, report)
        if log.origin == "D":
            bucket, weight, reason = offline_gate(log, decision, bucket, weight, reason)
        exposure = fact_check_exposure(messages, decision.output)
        if exposure["in_target"] and bucket == "accepted":
            # 教师这一步自己去开 / 读 / 引核查文章：不教给学生（黑名单上线后新 run 不会再产生这种行）
            bucket, weight, reason = "held", 0.0, "fact_check_target:" + ",".join(exposure["in_target"])
        input_est, sup_est = estimate_tokens(messages, len(rel_images))
        verdict_src = unit if decision.kind in ("step", "plan", "answer") else report
        meta: dict[str, Any] = {
            "role": decision.role.lower(), "unit": decision.kind, "origin": log.origin,
            "run": log.name, "workspace": decision.actor_id if decision.role == "HI" else None,
            "round": decision.round_index, "turn": decision.turn, "request": decision.request_id,
            "case_id": log.case_id, "dataset": log.config.get("dataset"),
            "case_family": (case or family_case).family_key if (case or family_case) else log.family_case_id,
            "split": (case or family_case).effective_split if (case or family_case) else None,
            "teacher_model": log.teacher_model, "action": decision.action,
            "spec": spec.name, "spec_hash": spec.hash(), "system_source": "spec_student_prompt", "runtime_system_matches_spec": matches,
            "system_variant": system_variant, "repair_turn": repair_turn,
            "state_renderer": renderer, "verifier": VERIFIER_VERSION,
            "judge_model": (reports[0].judge_model if reports else None),
            "verdict": verdict_src.disposition if verdict_src else None,
            "error_types": [e.type.value for e in verdict_src.errors] if verdict_src else [],
            "workspace_verdict": report.disposition if report else None,
            "tainted": bool(report and decision.kind == "report" and report.tainted_findings),
            "bucket": bucket, "bucket_reason": reason, "sample_weight": weight,
            "fact_check_in_context": exposure["in_context"], "fact_check_in_target": exposure["in_target"],
            "has_rationale": isinstance(decision.output, dict) and isinstance(decision.output.get("rationale"), str),
            "images": decision.image_evidence_ids, "input_tokens_est": input_est, "supervised_tokens_est": sup_est,
            "loss_scope": "last_assistant_only", "exported_at": date.today().isoformat(),
        }
        if unit is not None and decision.kind == "step":
            meta["clue_gain"] = unit.extra.get("clue_gain", {}).get("total", 0)
            meta["clue_gain_detail"] = {k: unit.extra["clue_gain"][k] for k in ("acquired_ids", "observed_ids", "quote_page")}
            meta["env_errors"] = [x["type"] for x in unit.extra.get("env_errors", [])]
        if unit is not None and decision.kind == "plan":
            meta["round_yield"] = unit.extra.get("round_yield")
            meta["gap"] = unit.extra.get("gap")
            meta["premises"] = (unit.extra.get("checks") or {}).get("premises")
            meta["plan_judged"] = "judge_plan" in unit.layers_run
        if unit is not None and decision.kind == "answer":
            meta["tainted_input"] = unit.extra.get("tainted_input")
            meta["recovered_from_tainted"] = unit.extra.get("recovered_from_tainted")
            meta["anchored_method"] = unit.extra.get("anchored", {}).get("method")
            meta["unsupported_sentences"] = unit.extra.get("anchored", {}).get("unsupported_sentences")
        rows.append({"messages": messages, "images": rel_images, "metadata": meta})
    return rows, reports + units


def apply_family_caps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一 case_family 的 accepted 行封顶（HI ≤ 40；GIA plan ≤ 5、answer ≤ 3），超出的降为 held。"""

    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        m = row["metadata"]
        if m["bucket"] != "accepted":
            continue
        key = (m["case_family"], "hi" if m["role"] == "hi" else f"gia_{m['unit']}")
        cap = FAMILY_CAP.get(key[1], 10 ** 9)
        counts[key] += 1
        if counts[key] > cap:
            m["bucket"], m["bucket_reason"], m["sample_weight"] = "held", "family_cap", 0.0
    return rows


def write_outputs(rows: list[dict[str, Any]], verdicts: list[Verdict], *, out_dir: Path, specs: dict[str, Spec], renderer: str,
                  verdicts_source: str, cases_dir: str, units_source: str | None = None,
                  cases: dict[str, Case] | None = None, render_overflow: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        m = row["metadata"]
        files[f"{m['role']}_{m['bucket']}"].append(row)
    names = ["hi_accepted", "hi_held", "hi_rejected", "gia_accepted", "gia_held", "gia_rejected"]
    for name in names:
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in files.get(name, []):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "verdicts_reports.jsonl").open("w", encoding="utf-8") as handle:
        for v in verdicts:
            if v.unit == "report":
                handle.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
    with (out_dir / "verdicts_units.jsonl").open("w", encoding="utf-8") as handle:
        for v in verdicts:
            if v.unit != "report":
                handle.write(json.dumps(v.to_dict(), ensure_ascii=False) + "\n")
    metas = [r["metadata"] for r in rows]

    def dist(items: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(Counter(str(m.get(key)) for m in items))

    accepted = [m for m in metas if m["bucket"] == "accepted"]
    # 划分：对本次导出涉及的案卷按 effective_split 统计；cases 未传时从 cases_dir 读；没有案卷的行计入 no_case_rows
    case_index = cases if cases is not None else load_cases(cases_dir)
    involved = [case_index[cid] for cid in sorted({str(m.get("case_id")) for m in metas}) if cid in case_index]
    splits = {"by_split": split_summary(involved),
              "rows_by_split": dict(Counter(str(m.get("split")) for m in metas)),
              "no_case_rows": sum(1 for m in metas if m.get("split") is None)}
    manifest = {
        "export_id": out_dir.name, "exported_at": date.today().isoformat(),
        "spec": {k: v.name for k, v in specs.items()}, "spec_hash": {k: v.hash() for k, v in specs.items()},
        "state_renderer": renderer, "verifier": VERIFIER_VERSION, "verdicts_source": verdicts_source, "units_source": units_source, "cases_dir": cases_dir,
        "files": {name: f"{name}.jsonl" for name in names},
        "counts": {name: len(files.get(name, [])) for name in names},
        "by_unit": {role: dist([m for m in accepted if m["role"] == role], "unit") for role in ("hi", "gia")},
        "by_action": {role: dist([m for m in accepted if m["role"] == role], "action") for role in ("hi", "gia")},
        "by_case_family": {fam: dict(Counter(m["role"] for m in accepted if m["case_family"] == fam)) for fam in {m["case_family"] for m in accepted}},
        "splits": splits,
        "sample_weight_sum": {role: sum(m["sample_weight"] for m in accepted if m["role"] == role) for role in ("hi", "gia")},
        "fact_check_exposure": {"rows_in_context": sum(1 for m in metas if m.get("fact_check_in_context")),
                                "rows_target_held": sum(1 for m in metas if str(m.get("bucket_reason") or "").startswith("fact_check_target")),
                                "accepted_in_context": sum(1 for m in accepted if m.get("fact_check_in_context"))},
        "weight_adjustments": dict(Counter(a for m in accepted for a in m.get("weight_adjustments", []))),
        "report_status": {role: dist([m for m in accepted if m["role"] == role and m["unit"] == "report"], "action") for role in ("hi",)},
        "rationale_rows": {role: sum(1 for m in accepted if m["role"] == role and m.get("has_rationale")) for role in ("hi", "gia")},
        "supervised_tokens_est": {"p50": sorted(m["supervised_tokens_est"] for m in accepted)[len(accepted) // 2] if accepted else 0},
        "input_tokens_est": {"max": max((m["input_tokens_est"] for m in metas), default=0),
                             "p50": sorted(m["input_tokens_est"] for m in metas)[len(metas) // 2] if metas else 0},
        "held": [{"run": m["run"], "role": m["role"], "unit": m["unit"], "workspace": m["workspace"], "turn": m["turn"], "reason": m["bucket_reason"]}
                 for m in metas if m["bucket"] == "held"],
        "rejected": [{"run": m["run"], "role": m["role"], "unit": m["unit"], "workspace": m["workspace"], "turn": m["turn"], "reason": m["bucket_reason"]}
                     for m in metas if m["bucket"] == "rejected"],
        "render_overflow": list(render_overflow or []),   # 渲染超预算而跳过的决策；不计入 counts
        "runtime_system_matches_spec": dict(Counter(f"{m['role']}:{m['runtime_system_matches_spec']}" for m in metas)),
        "note": "system 用规格 student_prompt；与运行时 system 是否逐字一致逐行记在 runtime_system_matches_spec（比较前去掉教师指令与修复附注；gia@0.3.0 起收编运行时全部 8 个 GIA 变体，对上的变体名记在 system_variant，修复轮记 repair_turn）。"
                "token 估算按 1.7 字符/token 与每图 1200 token；精确值见 check_inputs 的 summary。",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 v2 规则导出第 0 轮教师 SFT 数据")
    parser.add_argument("--runs", nargs="+", required=True, help="run 目录（可多个）")
    parser.add_argument("--cases", default="experiments/sft_data/cases")
    parser.add_argument("--verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl", help="report 单元的三层判定文件")
    parser.add_argument("--units", default=None, help="complete_units 产出的 plan / answer 模型判定（可选，叠加到程序层结果上）")
    parser.add_argument("--specs", default="specs")
    parser.add_argument("--hi-spec", default="hi@0.4.0", help="须与 run 实际用的提示一致：r0 用 hi@0.2.0，r0b 用 hi@0.3.0，09-30 以后新跑的用 hi@0.4.0")
    parser.add_argument("--gia-spec", default="gia@0.5.0", help="r0 用 gia@0.3.0，r0b 用 gia@0.4.0，09-30 以后新跑的用 gia@0.5.0")
    parser.add_argument("--renderer", default="compact_16k", choices=["legacy", "compact_8k", "compact_16k"])
    parser.add_argument("--max-dataset-share", nargs="*", default=None, metavar="数据集=份额",
                        help="按角色把某数据集在 accepted 权重里的份额压到上限（只缩权重，不删行），例如 videosham=0.34")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    specs = {"hi": Spec.load(Path(args.specs) / args.hi_spec), "gia": Spec.load(Path(args.specs) / args.gia_spec)}
    cases = load_cases(args.cases)
    all_verdicts = read_verdicts(args.verdicts)
    judged_units = load_units(args.units)
    out_dir = Path(args.out)
    image_root = out_dir / "images"
    rows: list[dict[str, Any]] = []
    verdicts: list[Verdict] = []
    overflow: list[dict[str, Any]] = []
    for run_path in args.runs:
        log = load_run(run_path)
        try:
            case = cases.get(log.case_id or "")
            reports = [v for v in all_verdicts if v.subject.get("run") == log.name]
            run_rows, run_verdicts = export_run(log, case, reports, specs, renderer=args.renderer, out_dir=out_dir, image_root=image_root,
                                                judged_units=judged_units, render_overflow=overflow,
                                                family_case=cases.get(log.family_case_id or ""))
            rows.extend(run_rows)
            verdicts.extend(run_verdicts)
            print(f"{log.name}: {len(run_rows)} 行（case={'有' if case else '无'}，report verdicts={len(reports)}）", flush=True)
        finally:
            log.close()
    rows = adjust_action_weights(apply_family_caps(rows))
    share_caps = apply_dataset_share_caps(rows, parse_share_caps(args.max_dataset_share))
    if share_caps:
        print(f"数据集份额上限：{json.dumps(share_caps, ensure_ascii=False)}", flush=True)
    manifest = write_outputs(rows, verdicts, out_dir=out_dir, specs=specs, renderer=args.renderer, verdicts_source=args.verdicts, cases_dir=args.cases,
                             units_source=args.units, cases=cases, render_overflow=overflow)
    manifest["dataset_share_caps"] = share_caps
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if overflow:
        print(f"渲染超预算跳过 {len(overflow)} 行，见 manifest.render_overflow", flush=True)
    print(json.dumps({k: manifest[k] for k in ("counts", "by_unit", "by_action", "sample_weight_sum", "input_tokens_est",
                                             "fact_check_exposure", "weight_adjustments", "report_status", "rationale_rows",
                                             "supervised_tokens_est")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
