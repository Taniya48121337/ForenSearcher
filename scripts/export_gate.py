"""export 质量门（2026-10-15，《第一次SFT开训清单_20261015》B2–B10、C1–C6）：只读一个导出目录，逐项出数字与通过 / 不通过。

    python scripts/export_gate.py --export experiments/sft_data/export/<目录>
        [--cases experiments/sft_data/cases] [--evid-bench plan-harness-1/datasets/EVID-Bench/evid_bench.json]
        [--evid-range 131 222] [--evid-channels 频道表.json] [--train-channels 训练侧频道表.json] [--dictionary 词典.txt]
        [--phash --evid-videos <EVID 视频目录>]
        [--model .models/Qwen3.5-9B]   # 有学生 processor 时逐行做 <image> 展开与 label mask 精确核对（接 datagen.check_inputs）
    → <导出目录>/gate_report.md、gate_report.json、gate_b2_review.md（B2 人工抽查用的 20 条）

原则：
- 能从导出行里重算的项一律重算（引用可见、JSON、占位符、封顶、划分、去重），不只信 export 写的 metadata；
  只有 export 才算得出的项（证据放行出处、工具结果截断、INCOMPLETE 比对、S0 / S5 槽记录）读 metadata，旧导出没有这些字段时记"无数据"。
- 状态只有五种：通过 / 不通过 / 需人工（B2 的 20 条抽查）/ 无数据（导出里没有这项要的字段）/ 未检查（缺外部输入，如 EVID 频道表）。
- 阈值全部写在 THRESHOLDS，与清单一致；改阈值 = 改清单。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from walker_coverage import branch_table_markdown, export_branch_table, family_of  # noqa: E402

THRESHOLDS = {
    "B2_manual_sample": 20, "B2_max_wrong": 2,
    "B5_privileged_rate": 0.05,
    "B6_incomplete_rate": 0.30,
    "B7_s0_adoption": 0.90, "B7_template_max": 5,
    "B9_p95_tokens": 32768,
    "C1_hi_min": 500, "C1_gia_min": 60, "C1_finish_min": 30,
    "C2_hi_per_case": 40, "C2_gia_per_case": 8, "C2_read_record_share": 0.40, "C2_family_share": 0.50,
}
FILES = ["hi_accepted", "hi_held", "hi_rejected", "gia_accepted", "gia_held", "gia_rejected"]
REQUIRED_META = ["role", "unit", "origin", "run", "workspace", "turn", "request", "case_id", "dataset", "case_family", "split",
                 "teacher_model", "spec", "spec_hash", "state_renderer", "verifier", "judge_model", "verdict", "error_types",
                 "sample_weight", "input_tokens_est", "supervised_tokens_est", "loss_scope", "exported_at", "bucket"]
FREEZE_META = ["tool_layer_hash", "teacher_role"]
GT_ERROR_TYPES = {"omitted_required_fact", "premature_finish"}       # = walker_run._GT_TYPES：带 GT 的核验意见
_ID = re.compile(r"\b[ER][1-9][0-9]*\b")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_TRUNC_MARKS = ("[truncated ", "已截断", "more not shown", "未展示", "\"context_truncated\"", "\"reports_truncated\"")
UNIT_KEYS = {"step": ("action",), "memory": ("action",), "report": ("status", "findings"), "plan": ("decision", "tasks"),
             "answer": ("decision", "answer")}


# --------------------------------------------------------------------------- 读

def load_export(export_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in FILES:
        path = export_dir / f"{name}.jsonl"
        if not path.is_file():
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                row = json.loads(line)
                row["_file"], row["_index"] = name, index
                rows.append(row)
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    return rows, manifest


def student_text(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"][:-1], ensure_ascii=False)


def target_text(row: dict[str, Any]) -> str:
    content = row["messages"][-1].get("content")
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def parse_target(row: dict[str, Any]) -> Any:
    try:
        return json.loads(target_text(row))
    except ValueError:
        return None


def _values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _values(v)]
    if isinstance(value, list):
        return [s for v in value for s in _values(v)]
    return [value] if isinstance(value, str) else []


def zh_ratio(text: str) -> float:
    cjk, latin = len(_CJK.findall(text)), len(_LATIN.findall(text))
    return cjk / (cjk + latin) if cjk + latin else 0.0


def quantile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else None


def item(item_id: str, name: str, status: str, value: dict[str, Any], threshold: str, note: str = "") -> dict[str, Any]:
    return {"id": item_id, "name": name, "status": status, "value": value, "threshold": threshold, "note": note}


def ok(flag: bool) -> str:
    return "通过" if flag else "不通过"


IMAGE_TOKENS_EST = 880                # = datagen.export.IMAGE_TOKENS_EST
NON_CJK_CHARS_PER_TOKEN = 3.0         # = datagen.export.NON_CJK_CHARS_PER_TOKEN


def row_tokens_est(row: dict[str, Any]) -> int:
    """与 datagen.export.row_tokens_est 同一算法，直接从行里算（旧导出没有 row_tokens_est 字段也能比）。"""

    total = 0.0
    for m in row["messages"]:
        content = m.get("content")
        parts = [content] if isinstance(content, str) else [p.get("text", "") for p in content or [] if isinstance(p, dict)]
        for text in parts:
            cjk = len(_CJK.findall(text))
            total += cjk + (len(text) - cjk) / NON_CJK_CHARS_PER_TOKEN
    return int(total) + len(row.get("images") or []) * IMAGE_TOKENS_EST


# --------------------------------------------------------------------------- B

def check_b2(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    metas = [r["metadata"] for r in rows]
    if not any("hidden_context_released" in m for m in metas):
        flagged = [m for m in metas if m.get("hidden_context")]
        return item("B2", "hidden_context：证据里出现过的词不算泄漏", "无数据",
                    {"rows_with_hidden": len(flagged), "held": sum(1 for m in metas if str(m.get("bucket_reason") or "").startswith("hidden_context"))},
                    "抽 20 条 held 人工看，误扣 ≤ 2", "旧导出：export 还没有按证据放行（没有 hidden_context_released 字段），用 10-15 的 export.py 重导")
    flagged = [r for r in rows if r["metadata"].get("hidden_context")]
    released_src = Counter()
    for r in flagged:
        for src in (r["metadata"].get("hidden_context_released") or {}).values():
            released_src["证据 E" if str(src).startswith("E") else "更早的工具观察"] += 1
    held = [r for r in rows if str(r["metadata"].get("bucket_reason") or "").startswith("hidden_context")]
    # 人工抽查：先放仍被扣的，再补放行的，凑 20 条（两类都要看：误扣看 held，误放看 released）
    sample = held[: THRESHOLDS["B2_manual_sample"]]
    sample += [r for r in flagged if r not in sample][: THRESHOLDS["B2_manual_sample"] - len(sample)]
    want = THRESHOLDS["B2_manual_sample"]
    short = (f"**本次只有 {len(sample)} 条，不足 {want} 条**：带 hidden_context 词的行一共只有 {len(flagged)} 行，已全部列出；"
             f"误扣上限仍按清单 ≤ {THRESHOLDS['B2_max_wrong']} 条判。") if len(sample) < want else ""
    lines = ["# B2 人工抽查：hidden_context", "",
             f"共 {len(flagged)} 行带 hidden_context 词；其中仍扣（held）{len(held)} 行。下表先列仍扣的，再补按证据放行的，最多 {want} 条。",
             *([short] if short else []),
             "每条判一个字：**对**（判定正确）/ **错**（误扣：其实是学生看得到的证据；或误放：其实是特权词）。清单 B2：误扣 ≤ 2。", "",
             "| # | run | request | 单元 | 词 | 放行出处 | 桶 / 原因 | 教师输出（节选） | 判定 |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sample, 1):
        m = r["metadata"]
        out = target_text(r).replace("|", "\\|").replace("\n", " ")[:220]
        lines.append(f"| {i} | {m.get('run')} | {m.get('request')} | {m.get('unit')} | {', '.join(m.get('hidden_context') or [])} | "
                     f"{json.dumps(m.get('hidden_context_released') or {}, ensure_ascii=False)} | {m.get('bucket')} / {m.get('bucket_reason')} | {out} |  |")
    (out_dir / "gate_b2_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return item("B2", "hidden_context：证据里出现过的词不算泄漏", "需人工",
                {"rows_with_hidden": len(flagged), "released_rows": sum(1 for r in flagged if r["metadata"].get("hidden_context_released")),
                 "released_tokens_by_source": dict(released_src), "held": len(held), "review_sample": len(sample),
                 "review_sample_short": len(sample) < want},
                "抽 20 条 held 人工看，误扣 ≤ 2", "抽查表 gate_b2_review.md；held 不足 20 条时用放行行补齐（看有没有误放）"
                + (f"。抽查只有 {len(sample)} 条（不足 {want}）：带 hidden_context 的行总共 {len(flagged)} 行，全部列入" if short else ""))


def check_b3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = []
    for r in rows:
        if r["metadata"].get("bucket") != "accepted":
            continue
        seen = set(_ID.findall(student_text(r)))
        outside = sorted(set(_ID.findall(target_text(r))) - seen)
        if outside:
            bad.append({"run": r["metadata"].get("run"), "request": r["metadata"].get("request"), "ids": outside[:6]})
    return item("B3", "引用可见：accepted 行引用的 E / R ⊂ 该步视图", ok(not bad),
                {"accepted": sum(1 for r in rows if r["metadata"].get("bucket") == "accepted"), "violations": len(bad), "examples": bad[:5]},
                "违例 0", "从行里重算：教师输出里的 E/R 编号是否在学生这一步的输入里出现过")


def check_b4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [r for r in rows if r["metadata"].get("bucket") == "accepted"]
    marked = sum(1 for r in acc if any(mark in student_text(r) for mark in _TRUNC_MARKS))
    if not any("tool_result_truncation" in r["metadata"] for r in rows):
        return item("B4", "决策依赖的工具结果未被截断", "无数据", {"accepted_with_truncation_marks": marked},
                    "accepted 中依赖被截内容的行 0", "旧导出没有 tool_result_truncation 字段（要原始 payload 才算得出），用 10-15 的 export.py 重导")
    bad = [r["metadata"] for r in acc if r["metadata"].get("tool_result_truncation")]
    held = sum(1 for r in rows if str(r["metadata"].get("bucket_reason") or "").startswith("tool_result_"))
    kinds = Counter((r["metadata"].get("tool_result_truncation") or {}).get("kind") for r in rows if r["metadata"].get("tool_result_truncation"))
    return item("B4", "决策依赖的工具结果未被截断", ok(not bad),
                {"accepted_violations": len(bad), "held_for_truncation": held, "rows_flagged_by_kind": dict(kinds),
                 "accepted_with_truncation_marks": marked},
                "accepted 中依赖被截内容的行 0",
                "依赖 = 输出用到了最新观察（GIA：本轮报告）里的编号 / 数字 / 专名而学生输入里没有；带截断标记但没用到被截部分的行不算")


def check_b5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [r["metadata"] for r in rows]
    acc = [m for m in metas if m.get("bucket") == "accepted"]
    held_hidden = [m for m in metas if str(m.get("bucket_reason") or "").startswith("hidden_context")]
    denom = len(acc) + len(held_hidden)
    rate = len(held_hidden) / denom if denom else 0.0
    answers = [m for m in acc if m.get("unit") == "answer"]
    known = [m for m in answers if "s5_gt_review" in m]
    gt_used = [m for m in known if m.get("s5_gt_review")]
    gt_revise = [m for m in answers if set(m.get("s5_revise_error_types") or []) & GT_ERROR_TYPES]
    status = ok(rate <= THRESHOLDS["B5_privileged_rate"] and not gt_used and not gt_revise) if acc else "无数据"
    if answers and not known and not any(m.get("walker_rule") for m in acc):
        note = "非 walker 导出（没有 S5 槽），S5 一项不适用"
    elif answers and len(known) < len(answers):
        note = f"{len(answers) - len(known)} 条 accepted 答案没有 slot_S5_gt_check.json（不是 walker S5 槽写的或旧 run）"
    else:
        note = ""
    return item("B5", "特权词拦截率 ≤ 5%；S5 不用 GT", status,
                {"privileged_rate": round(rate, 4), "held_hidden": len(held_hidden), "accepted_plus_held_hidden": denom,
                 "accepted_answers": len(answers), "answers_with_s5_record": len(known), "answers_gt_review": len(gt_used),
                 "answers_revised_with_gt_error_types": len(gt_revise)},
                "拦截率 ≤ 5%；accepted 答案里按 GT 重写的 0", note)


def check_b6(rows: list[dict[str, Any]], unanswerable: set[str]) -> dict[str, Any]:
    reports = [r for r in rows if r["metadata"].get("unit") == "report"]
    acc_reports = [r for r in reports if r["metadata"].get("bucket") == "accepted"]
    empty = [r["metadata"].get("run") for r in acc_reports if not ((parse_target(r) or {}).get("findings"))]
    have_checks = any("incomplete_reason_checks" in r["metadata"] for r in reports)
    mismatch_acc = [r["metadata"] for r in acc_reports
                    if any(c.get("verdict") == "mismatch" for c in r["metadata"].get("incomplete_reason_checks") or [])]
    mismatch_all = Counter(c["claim"] for r in reports for c in r["metadata"].get("incomplete_reason_checks") or [] if c.get("verdict") == "mismatch")
    answerable_acc = [r for r in acc_reports if str(r["metadata"].get("case_id")) not in unanswerable]
    answerable_all = [r for r in reports if str(r["metadata"].get("case_id")) not in unanswerable]

    def inc_rate(items: list[dict[str, Any]]) -> float | None:
        return round(sum(1 for r in items if (parse_target(r) or {}).get("status") == "INCOMPLETE") / len(items), 3) if items else None

    # 10-16 负责人定：核验后按充分性判定改写成 INCOMPLETE 的报告（metadata.status_rewritten，datagen.status_rewrite）照常
    # 进训练集，但单列、不计入 30% 线——线只量 walker / S4 自己判的 INCOMPLETE。空 findings、比对不符仍对全部 accepted 报告查
    rewritten = [r for r in answerable_acc if r["metadata"].get("status_rewritten")]
    rated_acc = [r for r in answerable_acc if not r["metadata"].get("status_rewritten")]
    rate_acc, rate_all = inc_rate(rated_acc), inc_rate([r for r in answerable_all if not r["metadata"].get("status_rewritten")])

    def fam(r: dict[str, Any]) -> str:
        ds, _, sid = str(r["metadata"].get("case_id") or "").partition(":")
        return "B" if ds in ("videosham", "vcsl", "qvhighlights") else ("E" if sid.startswith("e") else "L") if ds == "averimatec" else "C"

    def frac(items: list[dict[str, Any]]) -> str:
        k = sum(1 for r in items if (parse_target(r) or {}).get("status") == "INCOMPLETE")
        return f"{k}/{len(items)}"

    by_family = {f: {"rated": frac([r for r in rated_acc if fam(r) == f]),
                     "verifier_rewrite": len([r for r in rewritten if fam(r) == f]),
                     "with_rewrite": frac([r for r in answerable_acc if fam(r) == f])}
                 for f in "BCEL"}
    passed = (not empty and (not mismatch_acc) and rate_acc is not None and rate_acc <= THRESHOLDS["B6_incomplete_rate"])
    status = "无数据" if (not have_checks and reports) or rate_acc is None else ok(passed)
    return item("B6", "INCOMPLETE：status_reason 与实际一致；空 findings 0；可回答案例 INCOMPLETE 率 ≤ 30%", status,
                {"accepted_reports": len(acc_reports), "accepted_empty_findings": len(empty),
                 "accepted_reason_mismatch": len(mismatch_acc), "reason_mismatch_all_buckets": dict(mismatch_all),
                 "incomplete_rate_accepted": rate_acc, "incomplete_rate_all_reports": rate_all,
                 "verifier_rewrite_accepted_answerable": len(rewritten),
                 "incomplete_rate_accepted_with_rewrite": inc_rate(answerable_acc), "by_family_accepted_answerable": by_family,
                 "unanswerable_cases": sorted(unanswerable)},
                "空 findings 0；比对不符 0；accepted 报告 INCOMPLETE 率 ≤ 30%（核验改写的单列，不计入）",
                "INCOMPLETE 率按 accepted 报告算（训进去的是这些）；全部报告的比率一并列出，看生成侧。核验后按充分性判定改写的报告"
                "（verifier_rewrite）照常进训练集，单列在 verifier_rewrite_accepted_answerable / incomplete_rate_accepted_with_rewrite，"
                "不计入 30% 线。"
                + ("" if have_checks else " 旧导出没有 incomplete_reason_checks 字段。"))


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def _norm_sentence(text: str) -> str:
    text = _ID.sub("#ID", text)
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    return " ".join(text.lower().split())


B7_TEACHER_SLOTS = ("s3_teacher", "s1_teacher")       # 10-15 对话 6：没跑 S0、理由由 S3_next / S1_query 槽写的动作行


def _b7_source(m: dict[str, Any]) -> str | None:
    """walker 程序定的动作行的理由出处：s0_teacher / s3_teacher / s1_teacher（教师槽写的）、s0_template_fallback / walker_template（模板）。
    不是 walker 动作行（报告 / 答案 / 派工的 walker_slot、非 walker 的 teacher）返回 None，不进 B7 分母。"""

    source = m.get("rationale_source")
    if source == "walker_slot" and m.get("rationale_slot") in B7_TEACHER_SLOTS:
        return str(m["rationale_slot"])
    return source if source in ("s0_teacher", "s0_template_fallback", "walker_template") else None


def check_b7(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [r for r in rows if r["metadata"].get("bucket") == "accepted"]
    steps = [r["metadata"] for r in acc if r["metadata"].get("role") == "hi" and r["metadata"].get("unit") == "step"]
    # 分母 = accepted 的 walker 动作行（S0 槽写的、S3 / S1 槽写的、模板）；分子 = 教师槽写的（S0 / S3 / S1）
    walker_steps = [m for m in steps if _b7_source(m)]
    by_source = Counter(_b7_source(m) for m in walker_steps)
    teacher_rows = sum(n for s, n in by_source.items() if s in ("s0_teacher", *B7_TEACHER_SLOTS))
    adoption = teacher_rows / len(walker_steps) if walker_steps else None
    s0_only = by_source["s0_teacher"] / len(walker_steps) if walker_steps else None
    sentences: Counter[str] = Counter()
    example: dict[str, str] = {}
    for r in acc:
        rationale = (parse_target(r) or {}).get("rationale") if isinstance(parse_target(r), dict) else None
        if not isinstance(rationale, str):
            continue
        for sentence in _SENT_SPLIT.split(rationale.strip()):
            if len(sentence) < 25:
                continue
            key = _norm_sentence(sentence)
            sentences[key] += 1
            example.setdefault(key, sentence)
    top = [{"count": n, "sentence": example[k][:160]} for k, n in sentences.most_common(8)]
    worst = top[0]["count"] if top else 0
    if not any("rationale_source" in r["metadata"] for r in rows):
        status, note = "无数据", "旧导出没有 rationale_source 字段，S0 采用率算不出；模板句频次照算"
    elif adoption is None:
        status, note = ok(worst <= THRESHOLDS["B7_template_max"]), "没有 walker 程序定的动作行（非 walker 导出），只看模板句"
    else:
        status, note = ok(adoption >= THRESHOLDS["B7_s0_adoption"] and worst <= THRESHOLDS["B7_template_max"]), ""
    old_labels = any(m.get("rationale_source") == "walker_template" for m in walker_steps) and not any(m.get("rationale_slot") for m in steps)
    return item("B7", "rationale：教师槽采用率 ≥ 90%；模板句 ≤ 5 次/句", status,
                {"walker_step_rows": len(walker_steps), "teacher_rationale_rows": teacher_rows,
                 "s0_adoption": round(adoption, 3) if adoption is not None else None,
                 "s0_slot_only": round(s0_only, 3) if s0_only is not None else None,
                 "by_source": dict(by_source),
                 "rationale_source": dict(Counter(str(m.get("rationale_source")) for m in steps)),
                 "max_repeat": worst, "top_sentences": top},
                "采用率 ≥ 90%（accepted 的 walker 动作行里理由由教师槽写的占比：S0_rationale，或没跑 S0 时 S3_next / S1_query 槽；"
                "分母 = 这些 + 模板）；同一句（编号、数字归一后）≤ 5 次",
                note + ("；s0_adoption 键名沿用，含 S3 / S1 槽写的理由（10-15 对话 6 起）；s0_slot_only 只算 S0 槽；"
                        "walker_template = 没有任何槽写理由、用了程序模板（如 C4_search_fallback）" if walker_steps else "")
                + ("；本导出没有 rationale_slot 字段（10-15 对话 6 之前的 export），S3 / S1 槽写的理由仍记成 walker_template，"
                   "用当前 export.py 重导" if old_labels else ""))


def _exact_mask(rows: list[dict[str, Any]], export_dir: Path, model: str | None) -> dict[str, Any]:
    """接 datagen.check_inputs：有 --model 时用学生 processor 逐行精确核对；否则读导出目录里已有的 check_inputs.rows.jsonl。"""

    if model:
        try:
            from transformers import AutoProcessor

            from foren_searcher.datagen.check_inputs import encode_row
        except Exception as exc:  # noqa: BLE001
            return {"source": "none", "error": f"加载 check_inputs / transformers 失败：{exc}"}
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        failed, tokens = [], []
        for r in rows:
            if r["metadata"].get("bucket") != "accepted":
                continue
            try:
                _, stats = encode_row(r, export_dir, processor)
                tokens.append(stats["input_tokens"])
            except Exception as exc:  # noqa: BLE001
                failed.append({"run": r["metadata"].get("run"), "request": r["metadata"].get("request"), "error": str(exc)[:200]})
        return {"source": "processor", "checked": len(tokens) + len(failed), "failed": failed, "input_tokens": tokens}
    path = export_dir / "check_inputs.rows.jsonl"
    if not path.is_file():
        return {"source": "none"}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok_rows = [x for x in records if x.get("status") == "ok" and str(x.get("file", "")).endswith("accepted.jsonl")]
    return {"source": "check_inputs.rows.jsonl", "checked": len(records), "failed": [x for x in records if x.get("status") == "failed"],
            "input_tokens": [int(x["input_tokens"]) for x in ok_rows]}


def check_b8(rows: list[dict[str, Any]], export_dir: Path, exact: dict[str, Any]) -> dict[str, Any]:
    bad_json, bad_keys, bad_images, missing_files, bad_mask, framework_zh = [], [], [], 0, [], []
    meta_missing: Counter[str] = Counter()
    freeze_missing: Counter[str] = Counter()
    datasets = Counter()
    for r in rows:
        m = r["metadata"]
        ident = {"run": m.get("run"), "request": m.get("request"), "file": r["_file"]}
        target = parse_target(r)
        if not isinstance(target, dict):
            bad_json.append(ident)
        else:
            need = UNIT_KEYS.get(str(m.get("unit")), ())
            if any(k not in target for k in need):
                bad_keys.append({**ident, "unit": m.get("unit"), "missing": [k for k in need if k not in target]})
        placeholders = sum(1 for msg in r["messages"] for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                           if isinstance(part, dict) and part.get("type") == "image")
        if placeholders != len(r.get("images") or []):
            bad_images.append({**ident, "placeholders": placeholders, "images": len(r.get("images") or [])})
        missing_files += sum(1 for p in r.get("images") or [] if not (export_dir / p).is_file())
        roles = [msg.get("role") for msg in r["messages"]]
        if roles[-1] != "assistant" or m.get("loss_scope") != "last_assistant_only" or roles[0] != "system":
            bad_mask.append({**ident, "roles_tail": roles[-2:], "loss_scope": m.get("loss_scope")})
        # 规格是英文时，框架写给模型的固定说明也必须是英文（10-15 发现：旧 export 没开语言开关，追加的说明是中文）
        system = r["messages"][0].get("content") if isinstance(r["messages"][0].get("content"), str) else ""
        if zh_ratio(system) < 0.1:
            for msg in r["messages"][2:-1]:
                content = msg.get("content")
                text = content if isinstance(content, str) else " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict))
                if msg.get("role") == "user" and len(_CJK.findall(text)) >= 20 and zh_ratio(text) > 0.5:
                    framework_zh.append(ident)
                    break
        for key in REQUIRED_META:
            if key not in m:
                meta_missing[key] += 1
        for key in FREEZE_META:
            if m.get(key) in (None, ""):
                freeze_missing[key] += 1
        datasets[str(m.get("dataset"))] += 1
    bad_dataset = [d for d in datasets if d in ("None", "") or d != d.lower()]
    passed = not (bad_json or bad_keys or bad_images or missing_files or bad_mask or framework_zh or meta_missing or bad_dataset
                  or exact.get("failed"))
    return item("B8", "assistant 合法 JSON；<image> 对齐与 label mask；dataset 字段统一", ok(passed),
                {"rows": len(rows), "invalid_json": len(bad_json), "missing_unit_keys": len(bad_keys), "image_placeholder_mismatch": len(bad_images),
                 "image_files_missing": missing_files, "mask_structure_bad": len(bad_mask), "framework_text_zh_rows": len(framework_zh),
                 "metadata_missing_keys": dict(meta_missing), "freeze_fields_missing": dict(freeze_missing), "dataset_values": dict(datasets),
                 "bad_dataset_values": bad_dataset, "exact_check": {k: (len(v) if isinstance(v, list) else v) for k, v in exact.items()},
                 "examples": (bad_json + bad_keys + bad_images + bad_mask + framework_zh)[:5]},
                "全部为 0；有学生 processor 时 check_inputs 逐行失败 0",
                ("精确的 label mask / 图片 token 展开要学生 processor：本次" +
                 ("用了 " + exact["source"] if exact.get("source") not in (None, "none") else "没有（加 --model 或先跑 datagen.check_inputs）")
                 + "；结构检查（最后一条是 assistant、loss_scope、占位符数 = 图片数、文件存在）已逐行做。"
                 + (" 冻结字段（tool_layer_hash / teacher_role）缺失只记录，r0 重跑后应齐。" if freeze_missing else "")))


def check_b9(rows: list[dict[str, Any]], exact: dict[str, Any]) -> dict[str, Any]:
    acc_rows = [r for r in rows if r["metadata"].get("bucket") == "accepted"]
    acc = [r["metadata"] for r in acc_rows]
    est = [row_tokens_est(r) for r in acc_rows]
    old_est = [int(m.get("input_tokens_est") or 0) + int(m.get("supervised_tokens_est") or 0) for m in acc]
    limit = THRESHOLDS["B9_p95_tokens"]
    exact_tokens = exact.get("input_tokens") or []
    basis = exact_tokens if exact_tokens else est
    p95 = quantile(basis, 0.95)
    too_long_held = sum(1 for r in rows if str(r["metadata"].get("bucket_reason") or "").startswith("too_long"))
    over = sum(1 for v in basis if v > limit)
    return item("B9", "整行 token p95 ≤ 32K；超长行丢弃并记数", "无数据" if p95 is None else ok(p95 <= limit and over == 0),
                {"basis": "精确（学生 processor）" if exact_tokens else "估计（row_tokens_est：CJK 1/字，其余 3.0 字符/token）",
                 "accepted": len(acc), "p50": quantile(basis, 0.5), "p95": p95, "max": max(basis) if basis else None,
                 "accepted_over_limit": over, "held_too_long": too_long_held,
                 "old_estimate_p95(1.7字符/token)": quantile(old_est, 0.95)},
                "p95 ≤ 32768 且 accepted 里没有超长行",
                "" if exact_tokens else "没有精确计数：在本机有 Qwen3.5-9B processor 时加 --model 重跑本门，或先跑 datagen.check_inputs")


def check_b10(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """B10（10-15 清单补）：同一条 compare_media / video_structure 记录在一条轨迹里被 READ_RECORD 的次数 ≤ 2。
    轨迹 = 同一 run 的全部 HI 行（不分桶，读几次是轨迹本身的事）；记录类型从学生输入里该 E 的卡片行认（取得方式写在卡片上）。"""

    reads: Counter[tuple[str, str]] = Counter()
    for r in rows:
        m = r["metadata"]
        if m.get("role") != "hi" or m.get("action") != "READ_RECORD":
            continue
        target = parse_target(r) or {}
        rid = str(((target.get("arguments") or {}) if isinstance(target, dict) else {}).get("record_id") or "")
        if not rid:
            continue
        if re.search(re.escape(rid) + r"\b[^\n]{0,240}?(?:compare_media|video_structure)", student_text(r)):
            reads[(str(m.get("run")), rid)] += 1
    worst = max(reads.values(), default=0)
    over = {f"{run}:{rid}": n for (run, rid), n in reads.items() if n > 2}
    return item("B10", "对 compare_media / video_structure 记录的 READ_RECORD ≤ 2 次", ok(worst <= 2),
                {"records_read": len(reads), "max_reads": worst, "over_2": over},
                "同一 run 同一记录 ≤ 2 次", "")


# --------------------------------------------------------------------------- C

def check_c1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [r["metadata"] for r in rows if r["metadata"].get("bucket") == "accepted"]
    hi = sum(1 for m in acc if m["role"] == "hi")
    gia = sum(1 for m in acc if m["role"] == "gia")
    finish = sum(1 for m in acc if m["role"] == "gia" and m["unit"] == "answer")
    return item("C1", "HI accepted ≥ 500；GIA ≥ 60（FINISH ≥ 30）",
                ok(hi >= THRESHOLDS["C1_hi_min"] and gia >= THRESHOLDS["C1_gia_min"] and finish >= THRESHOLDS["C1_finish_min"]),
                {"hi_accepted": hi, "gia_accepted": gia, "gia_finish": finish, "gia_by_unit": dict(Counter(m["unit"] for m in acc if m["role"] == "gia"))},
                "HI ≥ 500；GIA ≥ 60；FINISH ≥ 30")


def check_c2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [r["metadata"] for r in rows if r["metadata"].get("bucket") == "accepted"]
    hi = [m for m in acc if m["role"] == "hi"]
    per_case = defaultdict(Counter)
    for m in acc:
        per_case[str(m.get("case_id"))][m["role"]] += 1
    worst_hi = max((c["hi"] for c in per_case.values()), default=0)
    worst_gia = max((c["gia"] for c in per_case.values()), default=0)
    rr = sum(1 for m in hi if m.get("action") == "READ_RECORD") / len(hi) if hi else 0.0
    fam = Counter(family_of(m.get("case_id")) for m in hi)
    fam_share = max(fam.values()) / len(hi) if hi else 0.0
    passed = (worst_hi <= THRESHOLDS["C2_hi_per_case"] and worst_gia <= THRESHOLDS["C2_gia_per_case"]
              and rr <= THRESHOLDS["C2_read_record_share"] and fam_share <= THRESHOLDS["C2_family_share"])
    return item("C2", "每案例 HI ≤ 40、GIA ≤ 8；READ_RECORD ≤ 40%；单一族 ≤ 50%", ok(passed),
                {"max_hi_per_case": worst_hi, "max_gia_per_case": worst_gia, "read_record_share": round(rr, 3),
                 "family_rows": dict(fam), "max_family_share": round(fam_share, 3),
                 "hi_by_action": dict(Counter(str(m.get("action")) for m in hi))},
                "每案例 HI ≤ 40、GIA ≤ 8；HI accepted 中 READ_RECORD 行 ≤ 40%；单一族行数 ≤ 50%",
                "族按 walker 案例族（A/B/C/E/L）算；export 的家族封顶按 case_family 做，这里按 case_id 复核")


def check_c3(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    gia_acc = [r["metadata"] for r in rows if r["metadata"].get("bucket") == "accepted" and r["metadata"].get("role") == "gia"]
    mode = (manifest.get("export_options") or {}).get("plan_copy")
    copies = [m for m in gia_acc if m.get("plan_copies_task") or "plan_copies_task" in (m.get("weight_adjustments") or [])]
    wait = [m for m in gia_acc if m.get("system_variant") == "wait"]
    gia_total_wo = len(gia_acc) - len(copies)
    copies_ok = (not copies) or (mode == "downweight" and gia_total_wo < THRESHOLDS["C1_gia_min"]
                                 and all(float(m.get("sample_weight") or 0) <= 0.3 + 1e-9 for m in copies))
    walker = [m for m in gia_acc if m.get("walker_rule")]
    composition = Counter(("r0e" if m.get("origin") == "r0e" else "walker" if m.get("walker_rule") else str(m.get("origin")))
                          + ":" + m["unit"] for m in gia_acc)
    gap_plans = [m for m in walker if m["unit"] == "plan" and not m.get("plan_copies_task")]
    return item("C3", "GIA 口径：r0e + walker FINISH；抄题 INVESTIGATE 不进或 ×0.3；wait 行剔除", ok(copies_ok and not wait),
                {"plan_copy_mode": mode, "accepted_plan_copies": len(copies), "accepted_wait_rows": len(wait),
                 "excluded_review_rows": len(manifest.get("excluded_rows") or []), "composition": dict(composition),
                 "walker_gap_plans_accepted": len(gap_plans)},
                "抄题派工 accepted 0（downweight 模式下仅当 GIA 不足 60 行、权重 ≤ 0.3）；wait 行 0",
                ("walker 非抄题的派工（第 2 轮按缺口派）照常进：口径只排除抄题派工；若负责人要求 walker 只收 FINISH，"
                 "在 export 加一条规则即可" if gap_plans else ""))


def check_c4(export_dir: Path) -> tuple[dict[str, Any], list[str]]:
    result = export_branch_table(export_dir)
    passed = all(c["ok"] for c in result["chains"])
    return (item("C4", "by_branch 出表；纳入族核心分支达最低行数；允许为零清单", ok(passed),
                 {"chains": {c["chain"]: f"{c['rows']}/{c['minimum']}" for c in result["chains"]},
                  "allowed_zero": {z["branch"]: z["rows"] for z in result["allowed_zero"]},
                  "expected_but_zero": result["expected_but_zero"]},
                 "B 链 ≥ 15；C READ_RECORD ≥ 40、SEARCH page ≥ 20；E 日期 + 读最早页 ≥ 10",
                 "完整表见报告末尾；A 族本轮不进，A 族 accepted 行数列在允许为零一栏供核对"),
            branch_table_markdown(result))


def check_c5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r0e = [r["metadata"] for r in rows if r["metadata"].get("origin") == "r0e"]
    if not r0e:
        return item("C5", "r0e 按当前规格重导出，当前核验器仍 positive 才进；origin 区分", "无数据", {"r0e_rows": 0},
                    "r0e accepted 行：verifier = 当前、兼容问题 0、origin=r0e",
                    "本导出没有 origin=r0e 的行（r0e 单独用 export --origin-tag r0e 导出，再与 walker 导出一起打包）")
    from foren_searcher.datagen.verify.verdict import VERIFIER_VERSION

    acc = [m for m in r0e if m.get("bucket") == "accepted"]
    stale = [m for m in acc if m.get("verifier") != VERIFIER_VERSION]
    compat = [m for m in acc if m.get("compat_problems")]
    reasons = Counter(str(m.get("bucket_reason") or "").split(":")[0] + (":" + ",".join(m.get("compat_problems") or []) if m.get("compat_problems") else "")
                      for m in r0e if m.get("bucket") != "accepted")
    return item("C5", "r0e 按当前规格重导出，当前核验器仍 positive 才进；origin 区分", ok(not stale and not compat),
                {"r0e_rows": len(r0e), "r0e_accepted": {"hi": sum(1 for m in acc if m["role"] == "hi"), "gia": sum(1 for m in acc if m["role"] == "gia")},
                 "stale_verifier_accepted": len(stale), "compat_problem_accepted": len(compat), "not_accepted_reasons": dict(reasons.most_common(8))},
                "accepted 的 r0e 行 verifier = 当前、兼容问题 0", "")


def _evid_ids(bench: Path | None, evid_cases: Path | None, lo: int, hi: int) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = defaultdict(set)      # 外部 ID → EVID 编号
    if bench and bench.is_file():
        for entry in json.loads(bench.read_text(encoding="utf-8")):
            vid = str(entry.get("id") or "")
            num = int(vid.split("_")[-1]) if vid.split("_")[-1].isdigit() else -1
            if lo <= num <= hi:
                for src in entry.get("source") or []:
                    ids[str(src)].add(vid)
    if evid_cases and evid_cases.is_dir():
        for path in evid_cases.glob("evid_video_*.json"):
            num = int(path.stem.split("_")[-1])
            if lo <= num <= hi:
                case = json.loads(path.read_text(encoding="utf-8"))
                for part in re.split(r"[:+]", str(case.get("family_key") or ""))[1:]:
                    if len(part) >= 8:
                        ids[part].add(path.stem)
    return ids


def _dhash_frames(path: Path, n: int = 8) -> list[int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    hashes: list[int] = []
    for i in range(n):
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        okay, frame = cap.read()
        if not okay:
            continue
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(full.std()) < 8.0:          # 黑场 / 纯色帧：哪两段视频都像，不参与比对
            continue
        gray = cv2.resize(full, (9, 8))
        bits = (gray[:, 1:] > gray[:, :-1]).flatten()
        hashes.append(int("".join("1" if b else "0" for b in bits), 2))
    cap.release()
    return hashes


CHANNEL_KEYS = {"uploader", "uploader_id", "channel", "channel_id", "channel_name", "author", "creator"}   # 案卷里的上传者 / 频道字段
DICTIONARY_PATHS = ("/usr/share/dict/words", "/usr/dict/words")


def _norm_channel(name: Any) -> str:
    return " ".join(str(name or "").casefold().split())


def _evid_channel_ids(table: Path, evid: dict[str, set[str]]) -> set[str]:
    """频道表旁的 <名>.detail.json（{外部 ID: {channel_id, uploader_id, …}}）里 EVID 视频的频道 / 上传者 ID；没有就空。"""

    detail = table.with_name(table.stem + ".detail.json")
    if not detail.is_file():
        return set()
    out: set[str] = set()
    for key, info in json.loads(detail.read_text(encoding="utf-8")).items():
        if isinstance(info, dict) and (key in evid or info.get("evid")):
            out.update(_norm_channel(info.get(k)) for k in ("channel_id", "uploader_id") if info.get(k))
    return out


def _load_dictionary(path: str | None) -> tuple[set[str], str]:
    """英文词典（一行一词）：--dictionary 给的文件 > 系统 words 文件 > nltk words 语料（已下载时）；都没有返回空集与"无"。"""

    for candidate in ([path] if path else []) + list(DICTIONARY_PATHS):
        if candidate and Path(candidate).is_file():
            words = {w.strip().casefold() for w in Path(candidate).read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()}
            return words, f"{candidate}（{len(words)} 词）"
    try:
        from nltk.corpus import words as nltk_words

        words = {w.casefold() for w in nltk_words.words()}
        return words, f"nltk words（{len(words)} 词）"
    except Exception:  # noqa: BLE001 —— 没装 nltk 或语料没下载
        return set(), "无"


def _train_channel_meta(case_ids: list[str], case_dir: Path, table: Path | None) -> dict[str, dict[str, Any]]:
    """每个训练案卷的上传者 / 频道元数据：{case_id: {"fields": [名字或 ID], "notes": 来源说明}}。
    fields = 案卷 JSON 里 CHANNEL_KEYS 字段（任意层级）+ --train-channels 表（{case_id 或外部 ID: 频道名或名字列表}）里对上本案卷的项。"""

    extra = json.loads(table.read_text(encoding="utf-8")) if table and table.is_file() else {}
    out: dict[str, dict[str, Any]] = {}
    for cid in case_ids:
        dataset, _, source = cid.partition(":")
        path = case_dir / f"{dataset}_{source}.json"
        case = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        fields: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, sub in value.items():
                    if key in CHANNEL_KEYS and isinstance(sub, str) and sub.strip():
                        fields.append(sub)
                    else:
                        walk(sub)
            elif isinstance(value, list):
                for sub in value:
                    walk(sub)

        walk(case)
        keys = {cid, source, str(case.get("family_key") or "").partition(":")[2]}
        for key, value in extra.items():
            if key in keys:
                fields += [value] if isinstance(value, str) else [str(v) for v in value or []]
        out[cid] = {"fields": fields, "notes": str(case.get("notes") or "")}
    return out


def _channel_hits(names: set[str], ids: set[str], meta: dict[str, dict[str, Any]], dictionary: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """结构化字段：与 EVID 频道名或频道 ID 归一后相等算命中。notes：多词频道名、或不在词典里的单词频道名，整词出现算命中；
    单词且在词典里（或没有词典）的名字不在 notes 里比，列进第二个返回值。"""

    single = {n for n in names if len(n.split()) == 1}
    skipped = sorted(n for n in single if not dictionary or n in dictionary)
    free_text = [n for n in names if n not in skipped and len(n) >= 3]
    patterns = {n: re.compile(r"(?<!\w)" + re.escape(n).replace(r"\ ", r"\s+") + r"(?!\w)", re.IGNORECASE) for n in free_text}
    hits: list[dict[str, Any]] = []
    for cid, m in meta.items():
        for value in m["fields"]:
            norm = _norm_channel(value)
            if norm in names or norm in ids:
                hits.append({"case": cid, "where": "上传者 / 频道字段", "channel": value})
        for name, pattern in patterns.items():
            if m["notes"] and pattern.search(m["notes"]):
                hits.append({"case": cid, "where": "案卷 notes", "channel": name})
    return hits, skipped


def check_c6(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    from foren_searcher.datagen.cases import check_split_leakage, load_cases

    acc = [r for r in rows if r["metadata"].get("bucket") == "accepted"]
    not_train = Counter(str(r["metadata"].get("split")) for r in acc if r["metadata"].get("split") != "dev_train")
    try:
        cases = load_cases(args.cases) if Path(args.cases).is_dir() else {}
    except Exception as exc:  # noqa: BLE001 —— 案卷目录里有读不了的文件时，划分检查记"案卷读取失败"，其余照做
        cases, case_error = {}, str(exc)[:200]
    else:
        case_error = None
    train_families = {str(r["metadata"].get("case_family")) for r in acc}
    other_split = {c.family_key: c.effective_split for c in cases.values() if c.effective_split != "dev_train"}
    family_leak = sorted(f for f in train_families if f in other_split)
    leakage = check_split_leakage(list(cases.values())) if cases else []
    # EVID 三重去重之一：外部 ID（YouTube 编号等）
    evid = _evid_ids(Path(args.evid_bench) if args.evid_bench else None, Path(args.evid_cases) if args.evid_cases else None,
                     args.evid_range[0], args.evid_range[1])
    involved = {str(r["metadata"].get("case_id")) for r in acc}
    id_hits: list[dict[str, Any]] = []
    case_dir = Path(args.cases)
    for cid in sorted(involved):
        dataset, _, source = cid.partition(":")
        path = case_dir / f"{dataset}_{source}.json"
        text = path.read_text(encoding="utf-8") if path.is_file() else cid
        for ext, evids in evid.items():
            if ext in text:
                id_hits.append({"case": cid, "where": "案卷", "id": ext, "evid": sorted(evids)})
    by_case_rows: dict[str, list[str]] = defaultdict(list)
    for r in acc:
        by_case_rows[str(r["metadata"].get("case_id"))].append(json.dumps(r["messages"], ensure_ascii=False))
    for cid, texts in by_case_rows.items():
        blob = "\n".join(texts)
        for ext, evids in evid.items():
            if ext in blob and not any(h["case"] == cid and h["id"] == ext for h in id_hits):
                id_hits.append({"case": cid, "where": "训练行文本", "id": ext, "evid": sorted(evids)})
    fmnv_hits = [h for h in id_hits if h["case"].startswith("fmnv:")]
    # 之二：频道——10-15 对话 6：只比训练案卷的上传者 / 频道元数据，不再扫训练行全文（全文子串匹配把 "Young""Mana" 这类
    # 普通词频道名误报成泄漏）；结构化字段按名字 / ID 精确比，案卷 notes（来源说明）按整词比、跳过词典词
    channel_compare: dict[str, Any] = {}
    if args.evid_channels and Path(args.evid_channels).is_file():
        channels = json.loads(Path(args.evid_channels).read_text(encoding="utf-8"))      # {外部 ID 或 EVID 编号: 频道名}
        names = {_norm_channel(v) for k, v in channels.items() if (k in evid or str(k).startswith("video_")) and v}
        names.discard("")
        ids = _evid_channel_ids(Path(args.evid_channels), evid)
        dictionary, dict_source = _load_dictionary(getattr(args, "dictionary", None))
        train_table = getattr(args, "train_channels", None)
        meta = _train_channel_meta(sorted(involved), case_dir, Path(train_table) if train_table else None)
        channel_hits, skipped = _channel_hits(names, ids, meta, dictionary)
        channel_compare = {"train_cases": len(meta), "cases_with_channel_fields": sum(1 for m in meta.values() if m["fields"]),
                           "cases_with_notes": sum(1 for m in meta.values() if m["notes"]), "train_channels_table": train_table,
                           "evid_channel_names": len(names), "evid_channel_ids": len(ids), "dictionary": dict_source,
                           "notes_skipped_names": skipped}
        channel_state = (f"已检查（训练侧：上传者 / 频道字段 {channel_compare['cases_with_channel_fields']} 个案卷、"
                         f"notes {channel_compare['cases_with_notes']} 个案卷；EVID 频道 {len(names)} 个；词典 {dict_source}）")
    else:
        channel_hits, channel_state = [], "未检查（缺 EVID 频道表：--evid-channels）"
    # 之三：感知哈希
    phash_hits: list[dict[str, Any]] = []
    if args.phash and args.evid_videos and Path(args.evid_videos).is_dir():
        evid_hashes: dict[str, list[int]] = {}
        for num in range(args.evid_range[0], args.evid_range[1] + 1):
            path = Path(args.evid_videos) / f"video_{num:03d}.mp4"
            if path.is_file():
                evid_hashes[path.stem] = _dhash_frames(path)
        for cid in sorted(involved):
            dataset, _, source = cid.partition(":")
            case_path = case_dir / f"{dataset}_{source}.json"
            if not case_path.is_file():
                continue
            for media in json.loads(case_path.read_text(encoding="utf-8")).get("media") or []:
                mpath = (case_dir / str(media.get("path") or "")).resolve()
                if not mpath.is_file() or not str(media.get("media_type") or "").startswith("video"):
                    continue
                mine = _dhash_frames(mpath)
                for vid, theirs in evid_hashes.items():
                    # 至少 3 帧（帧少时全部）各自在对方里找得到近似帧才算重复，单帧碰巧相像不算
                    close = [min((bin(a ^ b).count("1") for b in theirs), default=64) for a in mine]
                    matched = sum(1 for d in close if d <= args.phash_max_distance)
                    if mine and theirs and matched >= min(3, len(mine)):
                        phash_hits.append({"case": cid, "evid": vid, "matched_frames": matched, "of": len(mine)})
        phash_state = (f"已检查（{len(evid_hashes)} 个 EVID 视频，各取 8 帧 dHash，去掉纯色帧；≥ 3 帧在对方找到汉明距离 ≤ "
                       f"{args.phash_max_distance} 的近似帧算重复）")
    else:
        phash_state = "未检查（加 --phash --evid-videos <EVID 视频目录>，在有视频的本机跑）"
    split_ok = not not_train and not family_leak and not leakage and case_error is None
    dedup_ok = not id_hits and not channel_hits and not phash_hits
    complete = channel_state.startswith("已检查") and phash_state.startswith("已检查")
    status = "不通过" if not (split_ok and dedup_ok) else ("通过" if complete else "未检查")
    return item("C6", "划分泄漏 0；与 EVID 131–222 三重去重；FMNV 排除 EVID 编号", status,
                {"accepted_not_dev_train": dict(not_train), "train_family_in_other_split": family_leak, "case_split_leakage": leakage[:5],
                 "case_load_error": case_error,
                 "evid_external_ids": len(evid), "id_hits": id_hits[:10], "fmnv_id_hits": len(fmnv_hits),
                 "channel": channel_state, "channel_hits": channel_hits[:10], "channel_compare": channel_compare,
                 "phash": phash_state, "phash_hits": phash_hits[:10]},
                "划分泄漏 0；ID / 频道 / 感知哈希三项都检查且命中 0",
                "ID 一项比对 EVID 的源视频编号（evid_bench.json 的 source + EVID 案卷 family_key）与训练案卷全文、训练行全文。"
                "频道一项只比训练案卷的上传者 / 频道元数据：案卷里的 uploader / channel 等字段与 --train-channels 表按名字（归一后）"
                "或频道 ID 精确比，案卷 notes 按整词比且跳过词典词（没有词典时跳过全部单词频道名）；不扫训练行全文")


# --------------------------------------------------------------------------- 汇总

def freeze_appendix(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    metas = [r["metadata"] for r in rows]
    return {"manifest_freeze": manifest.get("freeze"), "spec": manifest.get("spec"), "spec_hash": manifest.get("spec_hash"),
            "rows_tool_layer_hash": dict(Counter(str(m.get("tool_layer_hash")) for m in metas)),
            "rows_teacher_model": dict(Counter(str(m.get("teacher_model")) for m in metas)),
            "rows_freeze_ok": dict(Counter(str(m.get("freeze_ok")) for m in metas)),
            "skipped_runs": manifest.get("skipped_runs"), "export_options": manifest.get("export_options")}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    export_dir = Path(args.export)
    rows, manifest = load_export(export_dir)
    exact = _exact_mask(rows, export_dir, args.model)
    items = [check_b2(rows, export_dir), check_b3(rows), check_b4(rows), check_b5(rows), check_b6(rows, set(args.unanswerable or [])),
             check_b7(rows), check_b8(rows, export_dir, exact), check_b9(rows, exact), check_b10(rows),
             check_c1(rows), check_c2(rows), check_c3(rows, manifest)]
    c4, branch_lines = check_c4(export_dir)
    items += [c4, check_c5(rows), check_c6(rows, args)]
    report = {"export": export_dir.name, "counts": dict(Counter(r["_file"] for r in rows)), "items": items,
              "summary": dict(Counter(i["status"] for i in items)), "freeze": freeze_appendix(rows, manifest), "thresholds": THRESHOLDS}
    lines = [f"# export 质量门：{export_dir.name}", "",
             "行数：" + "、".join(f"{k} {v}" for k, v in report["counts"].items()), "",
             "状态汇总：" + "、".join(f"{k} {v}" for k, v in report["summary"].items()), "",
             "| 项 | 内容 | 状态 | 关键数字 | 通过条件 |", "|---|---|---|---|---|"]
    for i in items:
        key = {k: v for k, v in i["value"].items() if not isinstance(v, (list, dict)) or k in ("chains",)}
        lines.append(f"| {i['id']} | {i['name']} | **{i['status']}** | {json.dumps(key, ensure_ascii=False)} | {i['threshold']} |")
    lines += ["", "## 逐项说明", ""]
    for i in items:
        lines += [f"### {i['id']} {i['name']}：{i['status']}", "", "```json", json.dumps(i["value"], ensure_ascii=False, indent=1), "```"]
        if i["note"]:
            lines.append(i["note"])
        lines.append("")
    lines += ["## by_branch（C4）", "", *branch_lines, "", "## 附：冻结一致性", "", "```json",
              json.dumps(report["freeze"], ensure_ascii=False, indent=1), "```"]
    (export_dir / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (export_dir / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="export 目录（datagen.export 或 pack_sft 的产物）")
    parser.add_argument("--cases", default=str(ROOT / "experiments" / "sft_data" / "cases"))
    parser.add_argument("--evid-bench", default=str(ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "evid_bench.json"))
    parser.add_argument("--evid-cases", default=str(ROOT / "experiments" / "evid" / "cases"))
    parser.add_argument("--evid-range", nargs=2, type=int, default=[131, 222])
    parser.add_argument("--evid-channels", default=None, help="EVID 源视频频道表 JSON：{外部 ID 或 video_NNN: 频道名}")
    parser.add_argument("--train-channels", default=None,
                        help="训练侧频道表 JSON：{case_id 或外部 ID: 频道名或名字列表}（案卷里没有上传者字段时补；C6 频道一项按名字精确比）")
    parser.add_argument("--dictionary", default=None,
                        help="英文词典（一行一词）：C6 在案卷 notes 里比频道名时跳过词典词；不给时找系统 words 文件 / nltk 语料，都没有就跳过全部单词频道名")
    parser.add_argument("--phash", action="store_true", help="对训练案卷视频与 EVID 视频做感知哈希比对（要本机视频文件）")
    parser.add_argument("--evid-videos", default=str(ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"))
    parser.add_argument("--phash-max-distance", type=int, default=6)
    parser.add_argument("--model", default=None, help="学生 processor 路径（如 .models/Qwen3.5-9B）：给了就逐行精确核对 mask 与 token")
    parser.add_argument("--unanswerable", nargs="*", default=[], help="不可回答案例的 case_id（不计入 INCOMPLETE 率）")
    parser.add_argument("--strict", action="store_true", help="有任何一项不通过时返回 1")
    args = parser.parse_args(argv)
    report = run_gate(args)
    for i in report["items"]:
        print(f"{i['id']:<3} {i['status']:<4} {i['name']}")
    print(f"报告：{Path(args.export) / 'gate_report.md'}", flush=True)
    return 1 if args.strict and any(i["status"] == "不通过" for i in report["items"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
