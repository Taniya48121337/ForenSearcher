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

10-15 质量门（《第一次SFT开训清单_20261015》B2–B9、C3、C5，《export质量门_20261015》）：每行另记
    hidden_context_released（B2：本 run 学生够得着的证据里出现过的词放行）、cited_outside_view（B3）、tool_result_truncation（B4）、
    incomplete_reason_checks（B6）、compat_problems（C5）、branch / walker_rule / rationale_source / s5_gt_review（C4、B7、B5）、
    row_tokens_est（B9）、tool_layer_hash / teacher_role / freeze_ok（r0 冻结口径 §6）。GIA 定时复核行（wait 变体）不写进任何文件，
    记在 manifest.excluded_rows。scripts/export_gate.py 读导出目录逐项出数。

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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..local_models import _generation_messages
from ..search.retrieval_guard import FACT_CHECK_DOMAINS, fact_check_match
from ..state_render import TIERS, RenderError, payload_chars
from .cases import Case, load_cases, split_summary
from .runlog import Decision, RunLog, load_run
from .prompt_variants import spec_scope
from .specs import Spec
from .verify.calibrate import recalibrate
from .verify.complete_units import load_units, merge_units
from .verify.units import verify_units
from .errors import Severity
from .verify.verdict import VERIFIER_VERSION, Verdict, read_verdicts

GIA_VARIANT_MARKER = "<!-- ===== 首轮派工变体（运行时在没有历史报告时使用） ===== -->"
IMAGE_TOKENS_EST = 880           # 1280×720 一张图在 Qwen3.5-9B（patch 16 × merge 2 = 32px/token）实测 880 token；精确值由 check_inputs 用真实 processor 算
CHARS_PER_TOKEN = 1.7            # 试点语料实测
WEIGHT_CAP = 3.0
FAMILY_CAP = {"hi": 40, "gia_plan": 5, "gia_answer": 3}
# 09-28 权重规则（第 0 轮数据质量评估）：
PLAN_COPY_FACTOR = 0.3           # 10-14 晚：GIA 第 1 轮把总任务原样派给一个工作区（walker 每个案例都这样）：留着但降权
MEMORY_GAIN0_FACTOR = 0.5        # READ_RECORD / QUERY_MEMORY 且 clue_gain=0：权重 ×0.5（r0 里 READ_RECORD 占 accepted step 的 48%）
REPEAT_RUN_MIN = 4               # 同一工作区连续 ≥4 步同一动作：只留首末满权重，中间 ×0.5（4126 两个工作区各 8 连 INSPECT）
REPEAT_RUN_FACTOR = 0.5
REPORT_COMPLETED_CLEAN = 3.0     # COMPLETED 且无错误
REPORT_COMPLETED_FLAGGED = 2.0   # COMPLETED 带 minor/flag
REPORT_INCOMPLETE_SPECIFIC = 2.0 # INCOMPLETE 且 status_reason 写明缺什么
REPORT_INCOMPLETE_VAGUE = 1.0    # INCOMPLETE 但 status_reason 空泛
_CJK = re.compile(r"[\u4e00-\u9fff]")
MAX_ROW_TOKENS = 32768           # 10-15 清单 B9：整行（输入 + 监督）估计 token 超过它的行不进 accepted（记 too_long）


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


NON_CJK_CHARS_PER_TOKEN = 3.0    # 10-15：英文 / JSON 文本（Qwen 词表英文约 3.5–4 字符/token，取偏保守的 3.0）；未用学生 processor 校准


def row_tokens_est(messages: list[dict[str, Any]], n_images: int) -> int:
    """整行（输入 + 监督）token 估计，按语言分开算：CJK 字符 1 token / 字，其余 3.0 字符 / token，图片按 IMAGE_TOKENS_EST。

    estimate_tokens 的 1.7 字符/token 是 09 月中文试点语料实测，hi@0.7.0 起输入输出全英文后对英文行偏高约一倍（gia@0.9.0
    的 system 一段就约 4 万字符），B9 的超长判定用这个估计；精确值以 check_inputs（学生 processor）为准。"""

    total = 0.0
    for m in messages:
        c = m.get("content")
        parts = [c] if isinstance(c, str) else [p.get("text", "") for p in c if isinstance(p, dict)] if isinstance(c, list) else []
        for text in parts:
            cjk = len(_CJK.findall(text))
            total += cjk + (len(text) - cjk) / NON_CJK_CHARS_PER_TOKEN
    return int(total) + n_images * IMAGE_TOKENS_EST


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
            if _terminal_major(report):
                return "held", 0.0, "terminal_major_error:" + ",".join(_terminal_major(report))
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
            if _terminal_major(unit):
                return "held", 0.0, "terminal_major_error:" + ",".join(_terminal_major(unit))
            return "accepted", (2.0 if unit.extra.get("recovered_from_tainted") else 1.0), None
        if unit.disposition == "negative":
            return "rejected", 0.0, "answer_errors:" + ",".join(e.type.value for e in unit.errors)
        return "held", 0.0, "answer_uncertain"
    # GIA 的 READ_RECORD / QUERY_MEMORY：跟随其后的决定
    return "held", 0.0, "memory_action_follows_next_decision"


def _terminal_major(verdict: Verdict) -> list[str]:
    """终端行（报告 / 答案）的严格入选（10-14 晚）：disposition=positive 允许带 1 个 major 错，审读发现 wk8 accepted 的
    答案 12 条里 7 条带核验错误。报告与答案是"结论对不对"的样本，带 major 的一律不进 accepted（记 held，原因写错误类型）。
    flag 级（程序疑点）也算：例如"实体出现在结论里但不在所引 E 文本中"。minor（含"拿不准、仅备注"）不算。"""

    return sorted({e.type.value for e in verdict.errors if e.severity in (Severity.MAJOR, Severity.CRITICAL)})


_SPECIFIC_REASON = re.compile(r"E\d+|R\d+|来源|证据|页面|帧|转写|字幕|原文|档案|记录|source|evidence|frame|page|transcript|archive", re.I)


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def _plan_copies_task(decision: Decision) -> bool:
    """GIA INVESTIGATE 只派一个任务、任务文字就是总任务原文（审读：49 条 accepted plan 只有 15 种任务文本）。"""

    if decision.role != "GIA" or decision.kind != "plan" or not isinstance(decision.output, dict):
        return False
    tasks = decision.output.get("tasks") or []
    view = (decision.payload or {}).get("global_view") or {}
    task = view.get("task") if isinstance(view, dict) else None
    if isinstance(task, dict):
        task = task.get("text") or task.get("task")
    return len(tasks) == 1 and isinstance(tasks[0], dict) and bool(task) and _norm(tasks[0].get("task")) == _norm(task)


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


_HIDDEN_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])(?:[ERFW][1-9][0-9]*|\d+(?:[.:,]\d+)+|\d{3,}|[A-Z][A-Za-z][A-Za-z'\-]{2,})(?![A-Za-z0-9_])")


# 句首会大写的普通英文词：不是专名（wk6 报告 9 行因 "These" 被扣）
_COMMON_WORDS = frozenset("""these this those that there their they then than thus both each either neither every
another other such some many most more less few several all any none nothing only also however although though
because since while when where which what whose whom who how why with within without from into onto over under
after before during between across along around about against among beyond near frame frames face faces material
materials sound audio video videos picture pictures image images scene scenes person people text title sample samples
segment segments region regions comparison comparisons result results evidence report finding findings
correspondence correspondences difference differences boundary boundaries start end first last second third
neither nor not none yes no here""".split())


def _output_values(value: Any) -> list[str]:
    """输出里的字符串值（不含键名）。"""

    if isinstance(value, dict):
        return [s for v in value.values() for s in _output_values(v)]
    if isinstance(value, list):
        return [s for v in value for s in _output_values(v)]
    return [value] if isinstance(value, str) else [str(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


# 工具结果里的运行元数据（耗时、版本、后端名）：不是证据内容。10-15：videosham_0069 的取帧时刻 0.4 碰巧等于 "detail_s": 0.4，
# 被当成"学生看不到的数字"扣下；比对前先去掉这些键
_TOOL_META_KEYS = frozenset({"timings_s", "call_timings_ms", "elapsed_s", "elapsed_ms", "duration_ms", "backend", "version", "model",
                             "created_at", "cache", "cache_key"})


def strip_tool_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_tool_metadata(v) for k, v in value.items() if k not in _TOOL_META_KEYS}
    if isinstance(value, list):
        return [strip_tool_metadata(v) for v in value]
    if isinstance(value, str) and value.startswith("{") and len(value) < 4000 and any(f'"{k}"' in value for k in _TOOL_META_KEYS):
        try:
            return json.dumps(strip_tool_metadata(json.loads(value)), ensure_ascii=False)
        except ValueError:
            return value
    return value


def hidden_context(messages: list[dict[str, Any]], payload: Any, output: Any, limit: int = 8) -> list[str]:
    """教师输出里用到、学生输入（渲染后的 messages）里没有、但原始 payload 里有的数字 / 编号 / 专名（10-05）。

    审计：10-05 以前教师看原始全量视图，导出再按档位给学生渲染；r0e 的 35 条报告行里 17 条写了学生那份看不到的
    数字或专名——学生学到的是"凭空写出"。只算原始 payload 里确有的词：教师看图读出的、自己编的都不在这里算。
    10-05 起教师按同一档位渲染（teacher.render_for_model），新 run 这里应为空。
    10-15：数字按数值整段比（"129" 不因 payload 里有 129600 就算；3.500 = 学生看到的 3.5），比对前去掉工具运行元数据（耗时等）；
    这里只"找出"，放不放行由 release_by_evidence（B2）与 truncated_dependency（B4）决定。"""

    student = json.dumps(messages[:-1], ensure_ascii=False)
    student_lower = student.lower()
    raw = json.dumps(strip_tool_metadata(payload), ensure_ascii=False)
    found: list[str] = []
    for text in _output_values(output):
        for token in _HIDDEN_TOKEN.findall(text):
            # 10-15：数字按整数值比（"129" 不因原始 payload 里有 129600 就算"学生看不到的数字"）
            if token in found or _token_in(token, student) or not _token_in(token, raw):
                continue
            # 10-13：句首大写的普通词（These / Face / This）学生输入里有小写形式就不算专名（walker wk5 报告 10/11 条因此被扣）
            if token.isalpha() and (token.lower() in student_lower or token.lower() in _COMMON_WORDS):
                continue
            found.append(token)
    return found[:limit]


# =========================================================================== 10-15 质量门（清单 B2–B9、C3、C5）
# 下面这些检查逐行算、逐行写进 metadata（无论最后进哪个桶），export_gate.py 只读导出目录就能出数；会改桶的规则只作用于
# 本来要进 accepted 的行，原因写在 bucket_reason。

_ID = re.compile(r"\b[ER][1-9][0-9]*\b")
_TRUNC_MARKS = ("[truncated ", "已截断", "more not shown", "未展示", "\"context_truncated\"", "\"reports_truncated\"")


def student_text(messages: list[dict[str, Any]]) -> str:
    """学生这一步实际看到的全部文字（不含最后一条 assistant）。"""

    return json.dumps(messages[:-1], ensure_ascii=False)


def _ids(value: Any) -> set[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return set(_ID.findall(text))


def cited_ids_outside_view(messages: list[dict[str, Any]], output: Any) -> list[str]:
    """B3：教师输出里出现的 E / R 编号，学生这一步的输入里一次都没出现过的（引用了视图外的证据）。"""

    seen = _ids(student_text(messages))
    return sorted(_ids(output) - seen, key=lambda x: (x[0], int(x[1:])))


@dataclass
class RunEvidence:
    """B2 用：一个 run 里"可见证据文本"。records = 存储里每个 E 的全文（OCR / ASR / 页面正文 / 工具结果卡片）；
    observations = (event id, actor, 观察全文)，按发生顺序。"""

    records: dict[str, str] = field(default_factory=dict)
    observations: list[tuple[int, str, str]] = field(default_factory=list)


def build_run_evidence(log: RunLog) -> RunEvidence:
    corpus = RunEvidence()
    try:
        for record in log.store.evidence_records():
            parts = [record.label or "", record.content or ""]
            corpus.records[record.evidence_id] = "\n".join(p for p in parts if p)
    except Exception:  # noqa: BLE001 —— 旧 store 读不出时只用观察
        pass
    for event in log.events:
        if event.get("event_type") == "observation":
            payload = event.get("payload") or {}
            corpus.observations.append((int(event.get("id") or 0), str(event.get("actor_id")),
                                        json.dumps(payload.get("observation"), ensure_ascii=False, default=str)))
    return corpus


_NUMERIC_TOKEN = re.compile(r"\d+(?:[.:,]\d+)*")


def _is_numeric(token: str) -> bool:
    return bool(_NUMERIC_TOKEN.fullmatch(token))


def _token_in(token: str, text: str) -> bool:
    """词：原样子串（专名）。数字：按数值整段比，不认子串（129 不能靠 129600 放行）；"3.500,4.100" 逐个比，3.500 = 3.5；
    按 token 的小数位四舍五入后相等也算（112.6 ← 112.61）。"""

    if not _is_numeric(token):
        return token in text
    parts = [p for p in re.split(r"[,:]", token) if p]
    numbers = [float(x) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text)]

    def present(part: str) -> bool:
        # 教师常把 112.61 写成 112.6：按 part 的小数位四舍五入后相等也算同一个数
        value = float(part)
        decimals = len(part.split(".")[1]) if "." in part else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        return any(abs(x - value) < tolerance for x in numbers)

    return bool(parts) and all(present(p) for p in parts)


def release_by_evidence(tokens: list[str], decision: Decision, messages: list[dict[str, Any]],
                        corpus: RunEvidence | None) -> dict[str, str]:
    """B2：hidden_context 的词如果在本 run 学生够得着的证据里出现过，就不算泄漏。返回 {词: 出处}。

    够得着 = ① 某个 E 的全文里有，且这个 E 的编号在学生这一步的输入里出现过（学生能 READ_RECORD 它）；
            ② HI：同一工作区、本步之前某次工具观察的全文里有（学生在更早一步看到过）。
    原始 payload 里有、但不在任何证据 / 观察里的词（教师提示、案卷、规则说明带进来的）照旧扣。"""

    if not tokens or corpus is None:
        return {}
    visible = _ids(student_text(messages))
    cited = _ids(decision.output)
    released: dict[str, str] = {}
    for token in tokens:
        pool = visible
        if _is_numeric(token):
            # 数字太容易碰巧对上（5.7 能在工具耗时字段里找到）：只认这一步输出自己引用的 E；没引用就只看观察
            pool = visible & cited
        for eid in sorted((e for e in pool if e.startswith("E") and e in corpus.records), key=lambda x: int(x[1:])):
            if _token_in(token, corpus.records[eid]):
                released[token] = eid
                break
        if token in released or decision.role != "HI":
            continue
        before = decision.raw_event_id if decision.raw_event_id is not None else 10 ** 12
        for event_id, actor, text in corpus.observations:
            if event_id < before and actor == decision.actor_id and _token_in(token, text):
                released[token] = f"observation@{event_id}"
                break
    return released


def _dependency_source(decision: Decision) -> Any:
    """B4：这一步决策直接依赖的工具结果。HI = 最新观察（上一步动作的结果）；GIA = 本轮报告 + 记忆动作的最新观察。"""

    view = decision.view
    if decision.role == "HI":
        return view.get("latest_observation")
    history = (decision.payload or {}).get("memory_history")
    last = history[-1].get("observation") if isinstance(history, list) and history and isinstance(history[-1], dict) else None
    return {"latest_reports": view.get("latest_reports"), "memory_latest": last}


def truncated_dependency(decision: Decision, messages: list[dict[str, Any]], output: Any) -> dict[str, Any] | None:
    """B4：输出用到了"直接依赖的工具结果"里的编号 / 数字 / 专名，而这部分在学生输入里被截掉或删掉了。

    返回 None = 没问题；否则 {"kind": truncated|omitted, "items": [...]}。truncated = 学生输入里带截断标记（正文截断、列表
    "另有 N 条"、context_truncated / reports_truncated）；omitted = 没有标记但内容不在（渲染器压缩时去掉的字段）。"""

    source = _dependency_source(decision)
    if not source:
        return None
    raw = json.dumps(strip_tool_metadata(source), ensure_ascii=False, default=str)
    student = student_text(messages)
    student_lower = student.lower()
    missing: list[str] = []
    for eid in sorted(_ids(output)):
        if eid in raw and eid not in _ids(student):
            missing.append(eid)
    for text in _output_values(output):
        for token in _HIDDEN_TOKEN.findall(text):
            if token in missing or _ID.fullmatch(token) or _token_in(token, student) or not _token_in(token, raw):
                continue
            if token.isalpha() and (token.lower() in student_lower or token.lower() in _COMMON_WORDS):
                continue
            missing.append(token)
    if not missing:
        return None
    kind = "truncated" if any(mark in student for mark in _TRUNC_MARKS) else "omitted"
    return {"kind": kind, "items": missing[:8]}


# ---- B6：INCOMPLETE 的 status_reason 与实际失败比对

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a single": 1, "once": 1, "twice": 2}
_CLAIMS: list[tuple[str, re.Pattern[str]]] = [
    ("no_audio", re.compile(r"no audio track|has no audio|without (?:an )?audio|no sound track|没有音轨|无音轨", re.I)),
    ("searches", re.compile(r"(?:after|despite)\s+(\w+(?: \w+)?)\s+(?:\w+\s+){0,3}search|(\d+|one|two|three|four|five|six)\s+(?:\w+[/-]?\w*\s+){0,2}"
                            r"searches|search(?:es)?\s+(?:found|returned|yielded)\s+no|搜索.{0,6}(?:未|没有)找到", re.I)),
    ("original_not_found", re.compile(r"original (?:source )?(?:video|footage)?\s*(?:was )?not (?:found|identified|located|obtained)|"
                                      r"no (?:matching|original) (?:source )?video|(?:claimed )?video was not obtained|未找到原片", re.I)),
    ("no_correspondence", re.compile(r"no (?:consistent )?(?:frame-level )?correspondence|could not be aligned|no match(?:ing)? (?:segment|footage)", re.I)),
    ("boundaries_missing", re.compile(r"(?:start and end|exact start/end|exact start and end)(?: timestamps| boundaries)? (?:boundaries )?not "
                                      r"established|boundaries not established", re.I)),
    ("tool_failed", re.compile(r"\bfail(?:ed|ure)\b|\berror\b|timed? ?out|unavailable|could not be (?:opened|downloaded|accessed|fetched)|"
                               r"\b40[34]\b|\bblocked\b|inaccessible|打不开|失败", re.I)),
    ("budget", re.compile(r"\bbudget\b|turn limit|ran out of (?:turns|steps)|no (?:turns|steps) left|step limit|轮次用完|预算", re.I)),
]
_MATCH_PCT = re.compile(r"(\d+)% of A's half-second samples are found in B")


@dataclass
class WorkspaceFacts:
    searches: int = 0                 # SEARCH 查询（不含 source= 打开）
    failed_items: int = 0             # 观察里 failed / error 的条目数
    compare_runs: int = 0
    compare_best_pct: int | None = None
    compare_spans: bool = False       # compare_media 报了对应段（"matched footage lies in" / correspondences）
    audio_tracks: dict[str, int] = field(default_factory=dict)   # 视频 E → 音轨数（media_profile）
    turns_used: int = 0
    max_turns: int | None = None


def workspace_facts(log: RunLog, workspace: str) -> WorkspaceFacts:
    facts = WorkspaceFacts()
    for d in log.steps(workspace):
        facts.turns_used = max(facts.turns_used, int(d.turn or 0))
        rc = d.view.get("runtime_constraints") if isinstance(d.view, dict) else None
        if isinstance(rc, dict) and isinstance(rc.get("max_turns"), int):
            facts.max_turns = rc["max_turns"]
        out = d.output if isinstance(d.output, dict) else {}
        args = out.get("arguments") or {}
        if out.get("action") == "SEARCH" and "query" in args:
            facts.searches += 1
        if out.get("action") == "INSPECT" and ((args.get("request") or {}).get("operation") == "compare_media"):
            facts.compare_runs += 1
        obs = json.dumps(d.observation, ensure_ascii=False, default=str) if d.observation is not None else ""
        facts.failed_items += len(re.findall(r"=failed|\"status\": \"(?:failed|error)\"|\"error\":", obs))
        for pct in _MATCH_PCT.findall(obs):
            facts.compare_best_pct = max(int(pct), facts.compare_best_pct or 0)
        if "matched footage lies in" in obs or "\"correspondences\": [{" in obs:
            facts.compare_spans = True
    try:
        for record in log.store.evidence_records():
            if record.acquired_by == "INSPECT/media_profile" and len(record.derived_from) == 1:
                tracks = json.loads(record.content or "{}").get("audio_tracks")
                if isinstance(tracks, int):
                    facts.audio_tracks[record.derived_from[0]] = tracks
    except Exception:  # noqa: BLE001
        pass
    return facts


def check_incomplete_reason(output: dict[str, Any], facts: WorkspaceFacts) -> list[dict[str, str]]:
    """把 status_reason 里能程序核对的说法逐条对实际：返回 [{claim, verdict: ok|mismatch|unverifiable, why}]。"""

    reason = str(output.get("status_reason") or "")
    results: list[dict[str, str]] = []
    for claim, pattern in _CLAIMS:
        match = pattern.search(reason)
        if not match:
            continue
        verdict, why = "unverifiable", ""
        if claim == "no_audio":
            if facts.audio_tracks:
                silent = [e for e, n in facts.audio_tracks.items() if n == 0]
                verdict, why = ("ok", f"media_profile 无音轨：{','.join(silent)}") if silent else ("mismatch", "所有视频的 media_profile 都有音轨")
        elif claim == "searches":
            stated = next((g for g in match.groups() if g), None) if match.groups() else None
            n = int(stated) if stated and stated.isdigit() else _WORD_NUM.get((stated or "").lower().strip())
            if facts.searches == 0:
                verdict, why = "mismatch", "工作区没有发过 SEARCH 查询"
            elif n is not None and facts.searches < n:
                verdict, why = "mismatch", f"说搜了 {n} 次，实际 {facts.searches} 次"
            else:
                verdict, why = "ok", f"实际 {facts.searches} 次查询"
        elif claim == "original_not_found":
            if facts.compare_best_pct is not None and facts.compare_best_pct >= 60:
                verdict, why = "mismatch", f"compare_media 报 {facts.compare_best_pct}% 对上"
            elif facts.searches or facts.compare_runs:
                verdict, why = "ok", f"查询 {facts.searches} 次，比对 {facts.compare_runs} 次"
            else:
                verdict, why = "mismatch", "没搜也没比就说原片没找到"
        elif claim == "no_correspondence":
            if facts.compare_runs == 0:
                verdict, why = "mismatch", "工作区没跑过 compare_media"
            elif facts.compare_best_pct is not None and facts.compare_best_pct >= 60:
                verdict, why = "mismatch", f"compare_media 报 {facts.compare_best_pct}% 对上"
            else:
                verdict, why = "ok", f"compare_media 最高 {facts.compare_best_pct}%"
        elif claim == "boundaries_missing":
            if facts.compare_spans:
                verdict, why = "mismatch", "compare_media 已按工具精度给出对应段（S4：按精度即算回答）"
            else:
                verdict, why = "ok", "比对没有给出对应段"
        elif claim == "tool_failed":
            verdict, why = ("ok", f"{facts.failed_items} 处失败") if facts.failed_items else ("mismatch", "工作区观察里没有失败 / 报错")
        elif claim == "budget":
            if facts.max_turns:
                verdict, why = (("ok", f"用了 {facts.turns_used}/{facts.max_turns} 轮") if facts.turns_used >= facts.max_turns - 1
                                else ("mismatch", f"只用了 {facts.turns_used}/{facts.max_turns} 轮"))
        results.append({"claim": claim, "verdict": verdict, "why": why})
    return results


# ---- C5：旧轮（r0e）按当前规格重导出时的兼容检查

_LATIN = re.compile(r"[A-Za-z]")


def _zh_ratio(texts: list[str]) -> float:
    cjk = sum(len(_CJK.findall(t)) for t in texts)
    latin = sum(len(_LATIN.findall(t)) for t in texts)
    return cjk / (cjk + latin) if cjk + latin else 0.0


HI_ACTIONS = frozenset({"SEARCH", "INSPECT", "READ_RECORD", "QUERY_MEMORY"})


def compat_problems(decision: Decision, spec: Spec, messages: list[dict[str, Any]]) -> list[str]:
    """当前规格下这一行还能不能当训练样本：语言一致、动作与 INSPECT 操作在当前目录里。新 run 应全为空。"""

    problems: list[str] = []
    language = str(spec.harness_params.get("language") or "")
    if language == "en":
        # 按中文字占全部字母的比例判（英文输出里引用证据原文的几个汉字不算，例如 OCR 读出的中文字幕）
        # 逐个字段看：任何一段自然语言（rationale / statement / answer …）以中文为主就算（r0e 的英文查询 + 中文理由也算）
        if any(len(_CJK.findall(v)) >= 8 and _zh_ratio([v]) > 0.5 for v in _output_values(decision.output)):
            problems.append("language_mismatch:output_zh")
        if _zh_ratio([json.dumps(m.get("content"), ensure_ascii=False) for m in messages[1:-1]]) > 0.3:
            problems.append("language_mismatch:state_zh")
    out = decision.output if isinstance(decision.output, dict) else {}
    if decision.role == "HI" and decision.kind == "step":
        action = out.get("action")
        if action not in HI_ACTIONS:
            problems.append(f"action_not_in_current_spec:{action}")
        elif action == "INSPECT" and spec.harness_params.get("inspect_merged_catalog"):
            from ..inspect.contracts import MERGED_OPERATIONS

            op = ((out.get("arguments") or {}).get("request") or {}).get("operation")
            if op not in MERGED_OPERATIONS:
                problems.append(f"op_not_in_current_catalog:{op}")
    return problems


# ---- GIA 定时复核行（清单 A6 / 11）：r0 规格下不该有，有就剔除

def is_review_row(decision: Decision, system_variant: str | None) -> bool:
    if decision.role != "GIA":
        return False
    rc = decision.view.get("runtime_constraints") if isinstance(decision.view, dict) else None
    out = decision.output if isinstance(decision.output, dict) else {}
    return (system_variant == "wait" or (isinstance(rc, dict) and rc.get("round_in_progress") is True)
            or str(out.get("decision") or "").upper() == "WAIT")


# ---- 分支（by_branch，清单 C4）：一步实际动作 → 变体名（与 scripts/walker_coverage.py 的覆盖表同一套名字）

def decision_branch(action: dict[str, Any], payload: dict[str, Any]) -> str | None:
    name = action.get("action") or action.get("decision")
    args = action.get("arguments") or {}
    gia = "workspace_id" not in payload
    if "findings" in action and "status" in action:
        return "report_" + str(action.get("status")).lower()
    if gia:
        if action.get("decision") == "INVESTIGATE":
            rc = ((payload.get("global_view") or {}).get("runtime_constraints") or {})
            return "GIA INVESTIGATE round" + ("1" if int(rc.get("round_index") or 1) <= 1 else "2")
        if action.get("decision") == "FINISH":
            return "GIA FINISH"
        return f"GIA {name}" if name else None
    if name == "SEARCH":
        if "source" in args:
            inc = list(args.get("include") or [])
            if inc:
                return "SEARCH open " + inc[0]
            return "SEARCH open video" if args.get("output_type") == "video" else "SEARCH open URL/R"
        return f"SEARCH {args.get('input_type')}→{args.get('output_type')}"
    if name == "INSPECT":
        req = args.get("request") or {}
        op, params = req.get("operation"), req.get("params") or {}
        if op == "get_frames":
            kind = "pages" if ("pages" in params or "page_range" in params) else "times_s" if "times_s" in params else \
                "range_s" if "range_s" in params else "count"
            return f"INSPECT get_frames {kind}" + ("+box" if "box" in params else "")
        if op == "ocr":
            target = (args.get("targets") or [""])[0]
            view = payload.get("model_view") or {}
            card = (view.get("materials") or {}).get(target) if isinstance(view.get("materials"), dict) else None
            card = card or next((c for c in view.get("evidence_view") or [] if isinstance(c, dict) and c.get("evidence_id") == target), {})
            mods = card.get("modalities") or [] if isinstance(card, dict) else []
            return "INSPECT ocr(video)" if "video" in mods else "INSPECT ocr(image)"
        if op == "locate":
            return f"INSPECT locate({((params.get('query') or {}).get('type')) or 'text'})"
        if op == "compare_media":
            if params.get("mode") == "audio":
                return "INSPECT compare_media audio"
            return "INSPECT compare_media range" if params.get("range_a_s") else "INSPECT compare_media"
        return f"INSPECT {op}"
    if name == "READ_RECORD":
        if args.get("offset"):
            return "READ_RECORD offset>0"
        return "READ_RECORD focus" if action.get("focus_evidence_ids") else "READ_RECORD text"
    if name == "QUERY_MEMORY":
        return "QUERY_MEMORY (HI)"
    return None


def walker_slots(log: RunLog, decision: Decision) -> dict[str, Any]:
    """walker 的逐步记录：transport.json 的 oracle（规则、分支），slot_S0_rationale.json（S0 槽是否采用），
    slot_S5_gt_check.json（答案是否按 GT 提示重写，清单 B5"S5 不用 GT"），slot_S5_revise.json（核验后按证据重写的错误类型）。"""

    rdir = log.run_dir / "requests" / decision.request_id
    info: dict[str, Any] = {}
    try:
        transport = json.loads((rdir / "transport.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        transport = {}
    oracle = transport.get("oracle") if isinstance(transport.get("oracle"), dict) else {}
    if oracle:
        info["walker_rule"], info["walker_branch"], info["walker_kind"] = oracle.get("rule"), oracle.get("branch"), oracle.get("kind")
    # 10-15 对话 5a：这一步实际调过的槽与模型（分槽后 S0–S3 / S4–S5 是不同的模型）
    row_models = {str(s.get("name")): s.get("model") for s in transport.get("slots") or [] if isinstance(s, dict) and s.get("name")}
    if row_models:
        info["row_slot_models"] = row_models
    slot_file = rdir / "slot_S0_rationale.json"
    if slot_file.is_file():
        try:
            slot = json.loads(slot_file.read_text(encoding="utf-8"))
        except ValueError:
            slot = {}
        info["rationale_source"] = "s0_teacher" if slot.get("used") else "s0_template_fallback"
        info["rationale_template"] = slot.get("template")
    gt_file = rdir / "slot_S5_gt_check.json"
    if gt_file.is_file():
        try:
            gt = json.loads(gt_file.read_text(encoding="utf-8"))
        except ValueError:
            gt = {}
        info["s5_gt_review"] = bool(gt.get("gt_review"))          # True = 答案按 GT 提示重写过（FOREN_ORACLE_GT_REVIEW / ARMS）
    revise_file = rdir / "slot_S5_revise.json"
    if revise_file.is_file():
        try:
            revise = json.loads(revise_file.read_text(encoding="utf-8"))
        except ValueError:
            revise = {}
        info["s5_revised"] = bool(revise.get("changed"))
        revise_models = sorted({str(x.get("model")) for x in revise.get("slots") or [] if isinstance(x, dict) and x.get("model")})
        if revise_models:
            info["s5_revise_model"] = "+".join(revise_models)
        info["s5_revise_error_types"] = sorted({str(e.get("type")) for e in revise.get("errors") or [] if isinstance(e, dict)})
    if "rationale_source" in info:
        pass
    elif oracle.get("kind") == "action" and decision.kind == "step":
        info["rationale_source"] = "walker_template"      # walker 定的动作、没跑 S0 槽：rationale 是程序模板
    elif transport.get("transport") == "oracle_walker":
        info["rationale_source"] = "walker_slot"           # 报告 / 答案 / 派工：由 S1–S5 槽写
    else:
        info["rationale_source"] = "teacher"               # 教师整步输出（r0e 等非 walker run）
    return info


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


def _teacher_model(log: RunLog) -> str | None:
    """r0 冻结口径 §6.5：老 run 的 configured_model 里没有 model 时，用 run_config.walker.teacher_model 补上。"""

    return log.teacher_model or ((log.config.get("walker") or {}).get("teacher_model")) or log.config.get("teacher_model")


SLOT_TEACHER_KEYS = ("S0", "S1", "S2", "S3")        # 10-15 对话 5a：自部署 Qwen 写的槽
TERMINAL_TEACHER_KEYS = ("S4", "S5")                 # API 强模型写的槽（报告 / 答案，含核验后重写）


def slot_teachers(log: RunLog) -> dict[str, str | None]:
    """10-15 对话 5a（清单 A4）：每行记 slot_teacher（S0–S3 用的模型）与 terminal_teacher（S4 / S5 用的模型），取自
    run_config.walker.slot_models（walker_run 写）。分槽之前的 run 所有槽是同一个教师，两项都记 teacher_model。"""

    models = (log.config.get("walker") or {}).get("slot_models") or {}
    if not models:
        model = _teacher_model(log)
        return {"slot_teacher": model, "terminal_teacher": model}

    def join(keys: tuple[str, ...]) -> str | None:
        seen = [str(models[k]) for k in keys if models.get(k)]
        return "+".join(dict.fromkeys(seen)) or None

    return {"slot_teacher": join(SLOT_TEACHER_KEYS), "terminal_teacher": join(TERMINAL_TEACHER_KEYS)}


RUNTIME_GUARD_FLAGS = ("finish_without_citation",)     # forced_* 另按前缀认


def runtime_guard_markers(log: RunLog) -> list[str]:
    """10-05 修正优化计划 P3：评测兜底改写过结果的 run（result.json 的 contract_flags 里有 forced_* / finish_without_citation，
    或有工作区由运行时代写了 auto_report: 报告）。这类 run 的决定不能当教师样本——导出时整 run 记 held。"""

    markers: list[str] = []
    try:
        result = json.loads((log.run_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    for flag in result.get("contract_flags") or []:
        if str(flag).startswith("forced_") or flag in RUNTIME_GUARD_FLAGS:
            markers.append(str(flag))
    if any(event.get("event_type") == "auto_report" for event in log.events):
        markers.append("auto_report")
    return markers


def export_run(log: RunLog, case: Case | None, report_verdicts: list[Verdict], specs: dict[str, Spec], *, renderer: str,
               out_dir: Path, image_root: Path, judged_units: list[Verdict] | None = None,
               render_overflow: list[dict[str, Any]] | None = None,
               family_case: Case | None = None, hidden_context_mode: str = "hold",
               plan_copy_mode: str = "hold", origin_tag: str | None = None, compat_check: bool = True,
               max_row_tokens: int = MAX_ROW_TOKENS, excluded: list[dict[str, Any]] | None = None,
               ) -> tuple[list[dict[str, Any]], list[Verdict]]:
    """render_overflow：传入列表时，渲染超预算（RenderError）而跳过的决策追加到这里，供 manifest 记录。
    family_case：离线换题（D）时源案例的案卷，只用来取 family / 划分（案卷检查不适用）。
    10-15 质量门参数：plan_copy_mode（C3：照抄总任务的 INVESTIGATE，hold 不进 / downweight ×0.3）；origin_tag（C5：写进
    metadata.origin，例如 r0e；原 A/D 来源记 source_kind）；compat_check（C5：语言 / 动作目录与当前规格不符的不进）；
    max_row_tokens（B9）；excluded（GIA 定时复核行剔除，不写进任何文件，记在这里供 manifest）。"""

    reports = [recalibrate(v) for v in report_verdicts]
    units, useful = verify_units(log, case, reports)
    if judged_units:
        units = merge_units(units, [v for v in judged_units if v.subject.get("run") == log.name])
    by_unit: dict[tuple[str, str], Verdict] = {}
    for v in units:
        by_unit[(v.unit, str(v.subject.get("request")))] = v
    report_by_ws = {v.subject.get("workspace_id"): v for v in reports}
    guard_markers = runtime_guard_markers(log)
    corpus = build_run_evidence(log)
    facts_cache: dict[str, WorkspaceFacts] = {}
    rows: list[dict[str, Any]] = []
    for decision in log.decisions:
        if decision.role == "GIA" and not log.gia_is_model():
            continue
        spec = specs[decision.role.lower()]
        system, matches, system_variant, repair_turn = spec_system_match(spec, decision)
        if is_review_row(decision, system_variant):
            # 定时复核（wait 变体 / 本轮还有 HI 在跑时的 GIA 决策）：r0 规格关闭了复核，旧 run 里有就整行剔除
            if excluded is not None:
                excluded.append({"run": log.name, "request": decision.request_id, "reason": "gia_review_row",
                                 "system_variant": system_variant})
            continue
        try:
            # 10-15：按规格的语言 / 合并目录开关拼消息。以前导出没开这个 scope，英文规格（hi@0.7.0 起）的行里由
            # _generation_messages 追加的固定说明、INSPECT 目录附注、截断标记都是中文，与学生部署时看到的英文不一致
            with spec_scope(spec):
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
        plan_copy = bucket == "accepted" and _plan_copies_task(decision)
        if plan_copy:
            if plan_copy_mode == "hold":
                # C3（r0 口径）：walker 第 1 轮照抄总任务的派工默认不进；GIA 不足 60 行时 --plan-copy downweight 以 ×0.3 补
                bucket, weight, reason = "held", 0.0, "plan_copies_task"
            else:
                weight = round(weight * PLAN_COPY_FACTOR, 3)
        if log.origin == "D":
            bucket, weight, reason = offline_gate(log, decision, bucket, weight, reason)
        if guard_markers and bucket != "rejected":
            bucket, weight, reason = "held", 0.0, "runtime_guard:" + ",".join(guard_markers)
        exposure = fact_check_exposure(messages, decision.output)
        if exposure["in_target"] and bucket == "accepted":
            # 教师这一步自己去开 / 读 / 引核查文章：不教给学生（黑名单上线后新 run 不会再产生这种行）
            bucket, weight, reason = "held", 0.0, "fact_check_target:" + ",".join(exposure["in_target"])
        hidden = hidden_context(messages, decision.payload, decision.output)
        truncation = truncated_dependency(decision, messages, decision.output)          # B4
        trunc_items = set((truncation or {}).get("items") or [])
        released = release_by_evidence([t for t in hidden if t not in trunc_items], decision, messages, corpus)   # B2
        hidden_blocking = [t for t in hidden if t not in released and t not in trunc_items]   # 截断依赖的词记到 B4
        if hidden_blocking and bucket == "accepted" and hidden_context_mode == "hold":
            # 教师据以决策的内容学生输入里没有、也不在本 run 学生够得着的证据里：不当正例教
            bucket, weight, reason = "held", 0.0, "hidden_context:" + ",".join(hidden_blocking[:5])
        if truncation and bucket == "accepted":
            bucket, weight, reason = "held", 0.0, f"tool_result_{truncation['kind']}:" + ",".join(truncation["items"][:5])
        outside = cited_ids_outside_view(messages, decision.output)                       # B3
        if outside and bucket == "accepted":
            bucket, weight, reason = "held", 0.0, "citation_not_in_view:" + ",".join(outside[:5])
        report_checks: list[dict[str, str]] = []
        output_dict = decision.output if isinstance(decision.output, dict) else {}
        if decision.kind == "report":                                                       # B6
            if not output_dict.get("findings"):
                if bucket == "accepted":
                    bucket, weight, reason = "held", 0.0, "empty_findings"
            if output_dict.get("status") == "INCOMPLETE":
                if decision.actor_id not in facts_cache:
                    facts_cache[decision.actor_id] = workspace_facts(log, decision.actor_id)
                report_checks = check_incomplete_reason(output_dict, facts_cache[decision.actor_id])
                bad = [c["claim"] for c in report_checks if c["verdict"] == "mismatch"]
                if bad and bucket == "accepted":
                    bucket, weight, reason = "held", 0.0, "incomplete_reason_mismatch:" + ",".join(bad)
        compat = compat_problems(decision, spec, messages) if compat_check else []      # C5
        if compat and bucket == "accepted":
            bucket, weight, reason = "held", 0.0, "compat:" + ",".join(compat[:3])
        verdict_for_row = unit if decision.kind in ("step", "plan", "answer") else report
        if verdict_for_row is not None and verdict_for_row.verifier != VERIFIER_VERSION and bucket == "accepted":
            bucket, weight, reason = "held", 0.0, f"stale_verifier:{verdict_for_row.verifier}"
        input_est, sup_est = estimate_tokens(messages, len(rel_images))
        row_tokens = row_tokens_est(messages, len(rel_images))
        if max_row_tokens and row_tokens > max_row_tokens and bucket == "accepted":      # B9
            bucket, weight, reason = "held", 0.0, f"too_long:{row_tokens}"
        slots = walker_slots(log, decision)
        if slots.get("s5_gt_review") and bucket == "accepted":                            # B5：S5 不用 GT
            bucket, weight, reason = "held", 0.0, "s5_used_gt"
        verdict_src = unit if decision.kind in ("step", "plan", "answer") else report
        meta: dict[str, Any] = {
            "role": decision.role.lower(), "unit": decision.kind, "origin": origin_tag or log.origin, "source_kind": log.origin,
            "run": log.name, "workspace": decision.actor_id if decision.role == "HI" else None,
            "round": decision.round_index, "turn": decision.turn, "request": decision.request_id,
            "case_id": log.case_id, "dataset": log.config.get("dataset"),
            "case_family": (case or family_case).family_key if (case or family_case) else log.family_case_id,
            "split": (case or family_case).effective_split if (case or family_case) else None,
            "teacher_model": _teacher_model(log), "teacher_role": ((log.config.get("walker") or {}).get("teacher_role")),
            **slot_teachers(log),
            "tool_layer_hash": log.config.get("tool_layer_hash"),
            "freeze_ok": (log.config.get("freeze") or {}).get("ok") if isinstance(log.config.get("freeze"), dict) else None,
            "action": decision.action,
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
            "hidden_context": hidden, "hidden_context_released": released,
            "tool_result_truncation": truncation, "cited_outside_view": outside,
            "incomplete_reason_checks": report_checks, "compat_problems": compat,
            "branch": decision_branch(output_dict, decision.payload or {}) if output_dict else None,
            **slots,
            "has_rationale": isinstance(decision.output, dict) and isinstance(decision.output.get("rationale"), str),
            "images": decision.image_evidence_ids, "input_tokens_est": input_est, "supervised_tokens_est": sup_est,
            "row_tokens_est": row_tokens,
            "loss_scope": "last_assistant_only", "exported_at": date.today().isoformat(),
        }
        meta["plan_copies_task"] = bool(plan_copy)
        if plan_copy and bucket == "accepted":
            meta.setdefault("weight_adjustments", []).append("plan_copies_task")
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


def quality_summary(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """manifest 里质量门相关的计数（export_gate.py 会从行里重算一遍，这里方便直接看）。"""

    def reasons(prefix: str) -> int:
        return sum(1 for m in metas if str(m.get("bucket_reason") or "").startswith(prefix))

    return {
        "hidden_context": {"rows": sum(1 for m in metas if m.get("hidden_context")),
                           "released_rows": sum(1 for m in metas if m.get("hidden_context_released")),
                           "held": reasons("hidden_context")},
        "tool_result_truncation": {"rows": sum(1 for m in metas if m.get("tool_result_truncation")),
                                   "held": reasons("tool_result_")},
        "cited_outside_view": {"rows": sum(1 for m in metas if m.get("cited_outside_view")), "held": reasons("citation_not_in_view")},
        "empty_findings_held": reasons("empty_findings"),
        "incomplete_reason_mismatch_held": reasons("incomplete_reason_mismatch"),
        "plan_copies_task": {"rows": sum(1 for m in metas if m.get("plan_copies_task")), "held": reasons("plan_copies_task")},
        "compat_held": reasons("compat:"), "stale_verifier_held": reasons("stale_verifier"), "too_long_held": reasons("too_long"),
        "origin": dict(Counter(f"{m['role']}:{m.get('origin')}:{m['bucket']}" for m in metas)),
        "rationale_source": dict(Counter(str(m.get("rationale_source")) for m in metas if m["bucket"] == "accepted")),
    }


def freeze_summary(logs_config: list[tuple[str, dict[str, Any]]], freeze_file: Path | None) -> dict[str, Any]:
    """r0 冻结口径 §6.2：freeze_id、tool_layer_hash 集合（必须只有一个值）、spec_hash 集合、教师分布。"""

    freeze = {}
    if freeze_file and freeze_file.is_file():
        try:
            freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
        except ValueError:
            freeze = {}
    tool_hashes = Counter(str(cfg.get("tool_layer_hash")) for _, cfg in logs_config)
    spec_hashes = Counter(json.dumps(cfg.get("spec_hash"), sort_keys=True) for _, cfg in logs_config)
    teachers = Counter(str((cfg.get("configured_model") or {}).get("model") or (cfg.get("walker") or {}).get("teacher_model"))
                       for _, cfg in logs_config)
    roles = Counter(str((cfg.get("walker") or {}).get("teacher_role")) for _, cfg in logs_config)
    ok = Counter(str((cfg.get("freeze") or {}).get("ok")) if isinstance(cfg.get("freeze"), dict) else "missing" for _, cfg in logs_config)
    slot_models = Counter(json.dumps((cfg.get("walker") or {}).get("slot_models"), sort_keys=True) for _, cfg in logs_config)
    return {"freeze_id": freeze.get("freeze_id"), "freeze_tool_layer_hash": (freeze.get("tool_layer") or {}).get("hash"),
            "tool_layer_hash": dict(tool_hashes), "tool_layer_hash_unique": len([h for h in tool_hashes if h != "None"]) <= 1,
            "spec_hash": dict(spec_hashes), "teacher_model": dict(teachers), "teacher_role": dict(roles), "run_freeze_ok": dict(ok),
            "slot_models": dict(slot_models),
            "verifier": VERIFIER_VERSION}


def write_outputs(rows: list[dict[str, Any]], verdicts: list[Verdict], *, out_dir: Path, specs: dict[str, Spec], renderer: str,
                  verdicts_source: str, cases_dir: str, units_source: str | None = None,
                  cases: dict[str, Case] | None = None, render_overflow: list[dict[str, Any]] | None = None,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "hidden_context": {"rows": sum(1 for m in metas if m.get("hidden_context")),
                           "held": sum(1 for m in metas if str(m.get("bucket_reason") or "").startswith("hidden_context")),
                           "by_role_unit": dict(Counter(f"{m['role']}:{m['unit']}" for m in metas if m.get("hidden_context")))},
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
        "quality_gate": quality_summary(metas),
        **(extra or {}),
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
    parser.add_argument("--hi-spec", default="hi@0.5.0", help="须与 run 实际用的提示一致：r0 用 hi@0.2.0，r0b 用 hi@0.3.0，r0c 用 hi@0.4.0，09-30 晚到 10-04 的 r0cn 等用 hi@0.4.1，10-05 的 r0dv2 等用 hi@0.4.2，10-05 晚以后新跑的用 hi@0.5.0")
    parser.add_argument("--gia-spec", default="gia@0.6.0", help="r0 用 gia@0.3.0，r0b 用 gia@0.4.0，09-30 到 10-05 新跑的用 gia@0.5.0，10-05 晚以后新跑的用 gia@0.6.0")
    parser.add_argument("--renderer", default="compact_16k", choices=["legacy", "compact_8k", "compact_16k"])
    parser.add_argument("--max-dataset-share", nargs="*", default=None, metavar="数据集=份额",
                        help="按角色把某数据集在 accepted 权重里的份额压到上限（只缩权重，不删行），例如 videosham=0.34")
    parser.add_argument("--hidden-context", choices=["hold", "off"], default="hold",
                        help="教师输出用到学生输入里没有（原始 payload 里有）的数字 / 编号 / 专名时：hold 降为 held（默认），off 只记不改")
    parser.add_argument("--plan-copy", choices=["hold", "downweight"], default="hold",
                        help="GIA 第 1 轮照抄总任务的 INVESTIGATE：hold 不进（r0 默认，C3）；downweight 以 ×0.3 进（GIA 不足 60 行时补）")
    parser.add_argument("--origin-tag", default=None, help="写进每行 metadata.origin（例如 r0e），区分复习来源（C5）；原 A/D 记 source_kind")
    parser.add_argument("--no-compat", action="store_true", help="不做当前规格兼容检查（语言、动作 / INSPECT 操作目录）")
    parser.add_argument("--max-row-tokens", type=int, default=MAX_ROW_TOKENS, help="整行估计 token 上限（B9），超过的不进 accepted")
    parser.add_argument("--require-freeze", action="store_true",
                        help="只收 run_config.freeze.ok == true 的 run（r0 口径）；--freeze-exempt 里的 run 例外")
    parser.add_argument("--freeze-exempt", nargs="*", default=["*_r0e_*"], help="--require-freeze 时的白名单（glob，默认 r0e）")
    parser.add_argument("--freeze-file", default="specs/r0_freeze.json")
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
    excluded: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    configs: list[tuple[str, dict[str, Any]]] = []
    from fnmatch import fnmatch
    for run_path in args.runs:
        log = load_run(run_path)
        try:
            freeze = log.config.get("freeze") if isinstance(log.config.get("freeze"), dict) else {}
            if args.require_freeze and freeze.get("ok") is not True and not any(fnmatch(log.name, pat) for pat in args.freeze_exempt or []):
                skipped_runs.append({"run": log.name, "reason": "freeze_not_ok" if freeze else "freeze_missing",
                                     "failed": freeze.get("failed")})
                print(f"{log.name}: 跳过（run_config.freeze 不是 ok）", flush=True)
                continue
            configs.append((log.name, log.config))
            case = cases.get(log.case_id or "")
            reports = [v for v in all_verdicts if v.subject.get("run") == log.name]
            run_rows, run_verdicts = export_run(log, case, reports, specs, renderer=args.renderer, out_dir=out_dir, image_root=image_root,
                                                judged_units=judged_units, render_overflow=overflow,
                                                family_case=cases.get(log.family_case_id or ""),
                                                hidden_context_mode=args.hidden_context, plan_copy_mode=args.plan_copy,
                                                origin_tag=args.origin_tag, compat_check=not args.no_compat,
                                                max_row_tokens=args.max_row_tokens, excluded=excluded)
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
                             units_source=args.units, cases=cases, render_overflow=overflow,
                             extra={"excluded_rows": excluded, "skipped_runs": skipped_runs,
                                    "freeze": freeze_summary(configs, Path(args.freeze_file)),
                                    "export_options": {"plan_copy": args.plan_copy, "origin_tag": args.origin_tag,
                                                       "compat_check": not args.no_compat, "max_row_tokens": args.max_row_tokens,
                                                       "require_freeze": args.require_freeze, "hidden_context": args.hidden_context}})
    manifest["dataset_share_caps"] = share_caps
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if overflow:
        print(f"渲染超预算跳过 {len(overflow)} 行，见 manifest.render_overflow", flush=True)
    print(json.dumps({k: manifest[k] for k in ("counts", "quality_gate", "by_unit", "by_action", "sample_weight_sum", "input_tokens_est",
                                             "fact_check_exposure", "hidden_context", "weight_adjustments", "report_status", "rationale_rows",
                                             "supervised_tokens_est")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
