"""校准规则 R1 / R2 / R4 的反例注入：量"规则把真错误豁免掉"的假接受率，也量"该豁免的真情况"的真接受率。

大白话：calibrate.py 的规则会把某些模型报错撤销或降级（修误报）。S4 指出学生能反过来钻：堆引用触发 R1、
把假话写成短关系句触发 R2、让同句另一条 E 给一个短片段触发 R4。这里从真实报告出发，造两类样本：
- 反例（should_exempt=False）：满足规则的触发条件，但确实有错——规则一旦豁免就是**假接受**；
- 正例（should_exempt=True）：规则本来要照顾的真实情况——规则没豁免就是**假拒绝**（只看一边会把规则调得越来越死）。

每条规则的触发条件（读 calibrate.recalibrate 的原文）与反例针对的条件：

R1 multi_evidence_scope：scope_overreach（method=model）且该 Finding 所引不同 E ≥ 3、每对都 supported 且无反驳、coverage
   grounded 且无 unsupported_parts → 撤销。
   - r1_universal_overreach（反例）：针对"每对 supported + coverage 通过 ⇒ 合起来撑得起"。原本就引 ≥3 条 E 的 Finding 追加一句
     全称判断（"视频自始至终每一帧都是如此"）：每条帧确实支持自己那部分，覆盖层按提示词会把它当"对已支持内容的合理归纳"，
     但几帧撑不起全片——正是 scope_overreach 本身。
   - r1_citation_stacking（反例）：针对"≥3 条 E"只是计数门槛。原本引 1–2 条 E 的 Finding 追加全称判断，再从同报告其它 Finding
     借 E 堆到 3 条，并各附一句这些 E 确实支持的短句（否则逐对判不成 supported）。
   - r1_joint_support（正例）：原句不改，≥3 条 E 合起来支持、逐对给了 scope 提示 → 应豁免。
   - r1_joint_support_two（正例）：同上但只引 2 条 E → 应豁免，规则因门槛不豁免（假拒绝）。
   注意：anchored 层的 scope_overreach 是 minor，豁免与否不改变 disposition，只改错误计数 / 类型统计。

R2 relational_clause_citation：unsupported_assertion（method=model，report 单元）且 span 去空白后 0 < 长度 ≤ 20、含
   RELATIONAL_MARKERS 之一、同报告其它 Finding 有非图片 E 判 supported → 降级为 incomplete_citation_set（major，不传播）。
   - r2_false_relation（反例）：针对"不查关系本身是否成立"。追加一句错误的关系句（"与2015年11月巴黎袭击现场画面一致"，19 字）。
   - r2_fact_in_relational_wrapper（反例）：针对"关系词只是关键词匹配"。把编造的事实包进关系句（"现场37人受伤与通报相符"）。
   - r2_one_sided_relation（正例）：只引帧的 Finding 追加"这与报道描述相容"，支持在同报告其它 Finding 的文本 E 里 → 应降级。
   - r2_one_sided_relation_long（正例）：同上但关系句 > 20 字 → 应降级，规则因长度不降级（假拒绝）。
   critical → major 且不传播：若这是该报告唯一的 critical，disposition 会从 negative 翻成 positive。

R4 contradiction_covered_by_sibling：wrong_citation / misread_fine_text / sloppy_numeric（method=model|pixels），span 非空、
   evidence_ids 恰为 1 条；同一 Finding 另一条被引 E 判 supported，且其某个支持片段（verified 不为 False——图片片段 verified=None
   也算）与反驳 span 去空白后**互为子串**（任一方向，片段 ≥ 2 字）→ 撤销。
   - r4_substring_sibling（反例）：针对"互为子串、任一方向"。改掉句中一个数字（年份 +13），E_a 反驳"数字+上下文"那一段；另一条
     E 只核实了其中不含数字的一小段（如"7月14日"），这一小段是反驳 span 的子串 → 撤销。
   - r4_image_sibling（反例）：针对"图片来源的支持片段不做程序回查"。改数字后文本 E 反驳；同句所引的帧被判"支持"整段（像素判读
     幻觉，verified=None 无从回查）→ 撤销。
   - r4_earlier_state（正例）：真实的 E24 + E21 模式——一条 E 对某部分判反驳，另一条 E 逐字核实了同一部分 → 应撤销。
   - r4_paraphrased_sibling（正例）：同上但另一条 E 的支持片段是换了一个字的转述 → 应撤销，规则子串匹配不上（假拒绝）。

R4 样本里该 Finding 只保留"反驳者 + 兄弟"两条判定：其它被引 E 的真实支持片段可能恰好覆盖同一部分，混进来就分不清是谁撤销的。

两种模式：
- verdict（默认，不调模型）：在真实报告判定上**模拟一个如实标出错误的核验器**（该 Finding 的 judgments / coverage / 目标错误按
  上面的设定合成，其它 Finding 保持真实判定），直接 recalibrate。量的是"核验模型抓到了，规则会不会把它放掉"——条件假接受率。
- report（要 --judge）：把改过的报告按正式流程 verify_report（exec + anchored）再 recalibrate。目标错误没被核验模型标出记
  not_flagged（规则没被触发，不进分母）；这是端到端数字，只有接真实核验模型才有意义。

用法：
    python -m foren_searcher.datagen.verify.calibrate_rule_audit --runs experiments/sft_data/runs/true_4124903_r1 \
        --verdicts experiments/sft_data/verdicts_r0_judged.jsonl --mode verdict \
        --out experiments/sft_data/calibration/rule_audit_r0.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..errors import ErrorRecord, ErrorType, Severity, error
from ..runlog import load_run
from .anchored import BUDGET_EXIT_CODE, JsonJudge, JudgeBudgetExhausted, budget_message
from .calibrate import RELATIONAL_MARKERS, recalibrate
from .judges import add_judge_arguments, judge_from_args, judge_summary
from .reverify import _visible_ids
from .verdict import Verdict, decide, tainted, verify_report

if TYPE_CHECKING:
    from ..runlog import RunLog

UNIVERSAL_CLAUSE = "视频自始至终每一帧都是如此，全片没有任何其它画面"
R2_FALSE_RELATION = "与2015年11月巴黎袭击现场画面一致"
R2_WRAPPED_FACT = "现场37人受伤与通报相符"
R2_TRUE_RELATION = "这与报道描述相容"
R2_TRUE_RELATION_LONG = "这与本报告其它条目所引文字报道中描述的现场位置、时间和火情特征均相容"
_NUM = re.compile(r"(?<![\dEF])(\d{2,4})(?!\d)")
_CONTRADICTION_TYPES = (ErrorType.WRONG_CITATION, ErrorType.MISREAD_FINE_TEXT, ErrorType.SLOPPY_NUMERIC)


@dataclass
class RuleCase:
    kind: str
    rule: str                                   # R1 | R2 | R4
    should_exempt: bool
    condition: str                              # 针对规则的哪个条件（反例）/ 规则要照顾的哪种情况（正例）
    run: str
    workspace_id: str
    finding_id: str
    target_types: tuple[ErrorType, ...]
    target_span: str | None                     # 目标错误的 span 片段；None = 该 Finding 上任一目标类型错误
    detail: str
    report: dict[str, Any]                      # report 模式核验的报告（改过的，正例可能是原样）
    simulated: Verdict                          # verdict 模式：如实标出错误的原始判定（未校准）
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rule": self.rule, "should_exempt": self.should_exempt, "condition": self.condition, "run": self.run,
                "workspace_id": self.workspace_id, "finding_id": self.finding_id, "target_types": [t.value for t in self.target_types],
                "target_span": self.target_span, "detail": self.detail, **self.extra}


# --------------------------------------------------------------------------- 取材

@dataclass
class Source:
    log: RunLog
    workspace_id: str
    report: dict[str, Any]
    verdict: Verdict                            # 真实原始判定（未校准）

    def judgments(self, fid: str) -> list[dict[str, Any]]:
        return [j for j in self.verdict.judgments if j.get("finding_id") == fid]

    def finding(self, fid: str) -> dict[str, Any]:
        return next(f for f in self.report["findings"] if f.get("finding_id") == fid)

    def other_text_supported(self, fid: str) -> bool:
        return any(j.get("finding_id") != fid and j.get("verdict") == "supported" and not j.get("is_image") for j in self.verdict.judgments)

    def is_text(self, eid: str) -> bool:
        return self.log.lookup.has(eid) and self.log.lookup.get(eid).is_text

    def is_image(self, eid: str) -> bool:
        return self.log.lookup.has(eid) and self.log.lookup.get(eid).is_image


def _all_supported(js: list[dict[str, Any]]) -> bool:
    return bool(js) and all(j.get("verdict") == "supported" and not j.get("contradicted") for j in js)


def _append(statement: str, clause: str) -> str:
    s = statement.rstrip()
    s = s[:-1] if s.endswith(("。", ".", "；", ";")) else s
    return f"{s}，{clause}。"


def _with_statement(report: dict[str, Any], fid: str, statement: str | None = None, evidence_ids: list[str] | None = None) -> dict[str, Any]:
    out = copy.deepcopy(report)
    for f in out["findings"]:
        if f.get("finding_id") == fid:
            if statement is not None:
                f["statement"] = statement
            if evidence_ids is not None:
                f["evidence_ids"] = list(evidence_ids)
    return out


def _simulate(src: Source, fid: str, judgments: list[dict[str, Any]], coverage: dict[str, Any], errors: list[ErrorRecord]) -> Verdict:
    """真实判定的副本：目标 Finding 的 judgments / coverage / 错误换成合成的，其它 Finding 原样；处置按现行 decide 重算。"""

    v = Verdict.from_dict(copy.deepcopy(src.verdict.to_dict()))
    v.judgments = [j for j in v.judgments if j.get("finding_id") != fid] + judgments
    v.coverage = [c for c in v.coverage if c.get("finding_id") != fid] + [{"finding_id": fid, **coverage}]
    v.errors = [e for e in v.errors if e.extra.get("finding") != fid] + errors
    v.tainted_findings = tainted(v.errors)
    v.disposition = decide(v.errors, v.pending)
    return v


def _judgment(fid: str, eid: str, *, supported: list[dict[str, Any]] | None = None, contradicted: list[dict[str, Any]] | None = None,
              scope: bool = False, is_image: bool = False) -> dict[str, Any]:
    verdict = "contradicted" if contradicted else "supported" if supported else "not_found"
    return {"finding_id": fid, "evidence_id": eid, "verdict": verdict, "source": "", "supported": supported or [], "contradicted": contradicted or [],
            "rejected": [], "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": scope, "is_image": is_image, "notes": "simulated"}


def _part(part: str, quote: str = "", verified: bool | None = True) -> dict[str, Any]:
    return {"part": part, "quote": quote or part, "verified": verified, "hint": None}


def _copy_judgments(src: Source, fid: str, *, scope_first: bool = False) -> list[dict[str, Any]]:
    out = []
    for k, j in enumerate(src.judgments(fid)):
        j = copy.deepcopy(j)
        j["scope_overreach"] = bool(scope_first and k == 0)
        out.append(j)
    return out


def _scope_error(fid: str, statement: str, eids: list[str]) -> ErrorRecord:
    return error(ErrorType.SCOPE_OVERREACH, "核验模型提示：由局部/少数样本推断了整体", span=statement, severity=Severity.MINOR,
                 evidence_ids=eids[:1], method="model", extra={"finding": fid})


Builder = Callable[[Source, dict[str, Any], random.Random], "RuleCase | None"]


def _case(kind: str, rule: str, should: bool, condition: str, src: Source, fid: str, types: tuple[ErrorType, ...], span: str | None,
          detail: str, report: dict[str, Any], sim: Verdict, **extra: Any) -> RuleCase:
    return RuleCase(kind, rule, should, condition, src.log.name, src.workspace_id, fid, types, span, detail, report, sim, extra)


# --------------------------------------------------------------------------- R1

def r1_universal_overreach(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
    fid = str(f["finding_id"])
    js = src.judgments(fid)
    if len({j.get("evidence_id") for j in js}) < 3 or not _all_supported(js):
        return None
    statement = _append(str(f.get("statement") or ""), UNIVERSAL_CLAUSE)
    sim = _simulate(src, fid, _copy_judgments(src, fid, scope_first=True), {"grounded": True, "unsupported_parts": [], "notes": "simulated: 合理归纳"},
                    [_scope_error(fid, statement, [str(j["evidence_id"]) for j in js])])
    return _case("r1_universal_overreach", "R1", False, "每对 supported + coverage 通过 ⇒ 合起来撑得起（不看结论是不是全称判断）", src, fid,
                 (ErrorType.SCOPE_OVERREACH,), None, f"{fid} 追加全称判断 “{UNIVERSAL_CLAUSE}”", _with_statement(src.report, fid, statement), sim)


def r1_citation_stacking(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
    fid = str(f["finding_id"])
    js = src.judgments(fid)
    own = [str(j.get("evidence_id")) for j in js]
    if not 1 <= len(set(own)) <= 2 or not _all_supported(js):
        return None
    donors: list[tuple[str, dict[str, Any]]] = []          # (E, 它在别的 Finding 里的一个核实支持片段)
    for j in src.verdict.judgments:
        eid = str(j.get("evidence_id"))
        if j.get("finding_id") == fid or eid in own or eid in {d[0] for d in donors} or j.get("verdict") != "supported":
            continue
        parts = [p for p in j.get("supported") or [] if p.get("verified") is not False and len(str(p.get("part") or "")) >= 4]
        if parts:
            donors.append((eid, parts[0]))
    need = 3 - len(set(own))
    if len(donors) < need:
        return None
    picked = rng.sample(donors, k=need)
    extras = "；".join(str(p["part"])[:30] for _, p in picked)
    statement = _append(_append(str(f.get("statement") or ""), UNIVERSAL_CLAUSE).rstrip("。"), f"另见：{extras}")
    judgments = _copy_judgments(src, fid, scope_first=True) + [
        _judgment(fid, eid, supported=[_part(str(p["part"])[:30], str(p.get("quote") or ""), p.get("verified"))], is_image=src.is_image(eid))
        for eid, p in picked]
    sim = _simulate(src, fid, judgments, {"grounded": True, "unsupported_parts": [], "notes": "simulated"},
                    [_scope_error(fid, statement, own)])
    report = _with_statement(src.report, fid, statement, [*f.get("evidence_ids", []), *(e for e, _ in picked)])
    return _case("r1_citation_stacking", "R1", False, "“所引 E ≥ 3” 只是计数门槛（堆引用即可跨过）", src, fid, (ErrorType.SCOPE_OVERREACH,), None,
                 f"{fid} 追加全称判断，并从别的 Finding 借 {[e for e, _ in picked]} 把引用堆到 3 条", report, sim, stacked=[e for e, _ in picked])


def _r1_positive(kind: str, n_min: int, n_max: int, should_note: str) -> Builder:
    def build(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
        fid = str(f["finding_id"])
        js = src.judgments(fid)
        n = len({j.get("evidence_id") for j in js})
        if not n_min <= n <= n_max or not _all_supported(js):
            return None
        statement = str(f.get("statement") or "")
        sim = _simulate(src, fid, _copy_judgments(src, fid, scope_first=True), {"grounded": True, "unsupported_parts": [], "notes": "simulated"},
                        [_scope_error(fid, statement, [str(j["evidence_id"]) for j in js])])
        return _case(kind, "R1", True, should_note, src, fid, (ErrorType.SCOPE_OVERREACH,), None,
                     f"{fid} 原句（{n} 条 E 合起来支持），逐对给了 scope 提示", copy.deepcopy(src.report), sim)
    return build


r1_joint_support = _r1_positive("r1_joint_support", 3, 99, "≥3 条 E 合起来撑得起整句，逐对 scope 提示是逐对核验的盲区")
r1_joint_support_two = _r1_positive("r1_joint_support_two", 2, 2, "2 条 E 合起来撑得起整句（规则门槛 ≥3，会假拒绝）")


# --------------------------------------------------------------------------- R2

def _r2(kind: str, should: bool, clause: str, condition: str, *, frames_only: bool) -> Builder:
    def build(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
        fid = str(f["finding_id"])
        eids = [str(e) for e in f.get("evidence_ids") or []]
        js = src.judgments(fid)
        if not eids or not js or not src.other_text_supported(fid):
            return None
        if frames_only and not all(src.is_image(e) for e in eids):
            return None
        statement = _append(str(f.get("statement") or ""), clause)
        err = error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=clause, evidence_ids=eids, method="model", extra={"finding": fid})
        sim = _simulate(src, fid, _copy_judgments(src, fid), {"grounded": False, "unsupported_parts": [clause], "notes": "simulated"}, [err])
        return _case(kind, "R2", should, condition, src, fid, (ErrorType.UNSUPPORTED_ASSERTION,), clause, f"{fid} 追加 “{clause}”（{len(clause)} 字）",
                     _with_statement(src.report, fid, statement), sim)
    return build


r2_false_relation = _r2("r2_false_relation", False, R2_FALSE_RELATION, "只看 span≤20 + 关系词 + 别处有 supported 文本 E，不查关系本身是否成立",
                        frames_only=False)
r2_fact_in_relational_wrapper = _r2("r2_fact_in_relational_wrapper", False, R2_WRAPPED_FACT, "关系词只是关键词匹配：编造的事实包一层“与…相符”即可降级",
                                    frames_only=False)
r2_one_sided_relation = _r2("r2_one_sided_relation", True, R2_TRUE_RELATION, "比较句只引了帧一边，文本支持在同报告其它 Finding 里", frames_only=True)
r2_one_sided_relation_long = _r2("r2_one_sided_relation_long", True, R2_TRUE_RELATION_LONG, "同上但关系句 > 20 字（规则会假拒绝）", frames_only=True)


# --------------------------------------------------------------------------- R4

def _alter_number(src: Source, f: dict[str, Any], rng: random.Random) -> tuple[str, str, str, str] | None:
    """改掉句中一个在所引文本 E 里出现过的数字（+13）。返回 (新句, 反驳 span, 反驳它的文本 E, span 里不含数字的一段)。
    span = 数字前后到最近标点为止（各最多 6 字），像核验模型摘出的"部分"；不含被改数字的一段取数字之后（不够 2 字取之前）。"""

    statement = str(f.get("statement") or "")
    eids = [str(e) for e in f.get("evidence_ids") or []]
    texts = [e for e in eids if src.is_text(e)]
    hits = [(m, owner[0]) for m in _NUM.finditer(statement)
            if (owner := [e for e in texts if m.group(1) in (src.log.lookup.get(e).content or "")])
            and "http" not in statement[max(0, m.start() - 12):m.start()]]
    if not hits:
        return None
    m, owner = rng.choice(hits)
    new = str(int(m.group(1)) + 13)
    mutated = statement[:m.start()] + new + statement[m.end():]
    end = m.start() + len(new)
    b0 = max(0, m.start() - 6)
    while b0 > 0 and mutated[b0 - 1].isdigit():      # 窗口不切断前面的数字（"2016年7月14日"改 14 时带上完整的 2016）
        b0 -= 1
    before = re.split(r"[，。；,;：:\s]", mutated[b0:m.start()])[-1]
    after = re.split(r"[，。；,;：:\s]", mutated[end:end + 6])[0]
    span = before + new + after
    piece = after if len(after) >= 2 else before       # 必须是 span 的连续子串（“7月14日”这种核验模型真会单独列出的片段）
    if len(piece) < 2:
        return None
    return mutated, span, owner, piece


def r4_substring_sibling(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
    fid = str(f["finding_id"])
    eids = [str(e) for e in f.get("evidence_ids") or []]
    if len(set(eids)) < 2:
        return None
    altered = _alter_number(src, f, rng)
    if altered is None:
        return None
    statement, span, owner, piece = altered
    sibling = next((e for e in eids if e != owner and src.is_text(e)), None) or next((e for e in eids if e != owner), None)
    if sibling is None:
        return None
    judgments = [_judgment(fid, owner, contradicted=[_part(span, "（来源里的原数字）")]),
                 _judgment(fid, sibling, supported=[_part(piece, piece, True if src.is_text(sibling) else None)], is_image=src.is_image(sibling))]
    err = error(ErrorType.WRONG_CITATION, f"{owner} 与该部分相反", span=span, evidence_ids=[owner], method="model", extra={"finding": fid})
    sim = _simulate(src, fid, judgments, {"grounded": True, "unsupported_parts": [], "notes": "simulated"}, [err])
    return _case("r4_substring_sibling", "R4", False, "支持片段与反驳 span 任一方向互为子串即撤销（短片段“覆盖”带错数字的长 span）", src, fid,
                 _CONTRADICTION_TYPES, span, f"{fid} 改数字 → “{span}”，{owner} 反驳；{sibling} 只核实了 “{piece}”",
                 _with_statement(src.report, fid, statement), sim, sibling=sibling, sibling_part=piece)


def r4_image_sibling(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
    fid = str(f["finding_id"])
    eids = [str(e) for e in f.get("evidence_ids") or []]
    image = next((e for e in eids if src.is_image(e)), None)
    if image is None:
        return None
    altered = _alter_number(src, f, rng)
    if altered is None:
        return None
    statement, span, owner, _piece = altered
    judgments = [_judgment(fid, owner, contradicted=[_part(span, "（来源里的原数字）")]),
                 _judgment(fid, image, supported=[_part(span, f"画面：{span}", None)], is_image=True)]
    err = error(ErrorType.WRONG_CITATION, f"{owner} 与该部分相反", span=span, evidence_ids=[owner], method="model", extra={"finding": fid})
    sim = _simulate(src, fid, judgments, {"grounded": True, "unsupported_parts": [], "notes": "simulated"}, [err])
    return _case("r4_image_sibling", "R4", False, "图片来源的支持片段 verified=None，不做程序回查也被采信", src, fid, _CONTRADICTION_TYPES, span,
                 f"{fid} 改数字 → “{span}”，{owner} 反驳；帧 {image} 被判“支持”整段（像素幻觉无从回查）",
                 _with_statement(src.report, fid, statement), sim, sibling=image)


def _clause(statement: str) -> str | None:
    parts = [p.strip() for p in re.split(r"[，。；,;]", statement) if len(p.strip()) >= 6]
    return parts[-1][:24] if parts else None


def _r4_positive(kind: str, paraphrase: bool, condition: str) -> Builder:
    def build(src: Source, f: dict[str, Any], rng: random.Random) -> RuleCase | None:
        fid = str(f["finding_id"])
        eids = [str(e) for e in f.get("evidence_ids") or []]
        texts = [e for e in dict.fromkeys(eids) if src.is_text(e)]
        part = _clause(str(f.get("statement") or ""))
        if len(texts) < 2 or part is None:
            return None
        owner, sibling = texts[0], texts[1]
        half = len(part) // 2
        sib_part = part[:half] + "的" + part[half:] if paraphrase else part
        judgments = [_judgment(fid, owner, contradicted=[_part(part, "（较早状态的原文）")]),
                     _judgment(fid, sibling, supported=[_part(sib_part, sib_part, True)])]
        err = error(ErrorType.WRONG_CITATION, f"{owner} 与该部分相反", span=part, evidence_ids=[owner], method="model", extra={"finding": fid})
        sim = _simulate(src, fid, judgments, {"grounded": True, "unsupported_parts": [], "notes": "simulated"}, [err])
        return _case(kind, "R4", True, condition, src, fid, _CONTRADICTION_TYPES, part,
                     f"{fid} “{part}”：{owner} 判反驳，{sibling} {'以转述' if paraphrase else '逐字'}核实了同一部分",
                     copy.deepcopy(src.report), sim, sibling=sibling, sibling_part=sib_part)
    return build


r4_earlier_state = _r4_positive("r4_earlier_state", False, "引前后两条通报：较早那条对“后来状态”判反驳，另一条逐字核实了同一部分")
r4_paraphrased_sibling = _r4_positive("r4_paraphrased_sibling", True, "同上，但另一条 E 的支持片段是转述（规则子串匹配不上，会假拒绝）")


BUILDERS: dict[str, Builder] = {
    "r1_universal_overreach": r1_universal_overreach, "r1_citation_stacking": r1_citation_stacking,
    "r1_joint_support": r1_joint_support, "r1_joint_support_two": r1_joint_support_two,
    "r2_false_relation": r2_false_relation, "r2_fact_in_relational_wrapper": r2_fact_in_relational_wrapper,
    "r2_one_sided_relation": r2_one_sided_relation, "r2_one_sided_relation_long": r2_one_sided_relation_long,
    "r4_substring_sibling": r4_substring_sibling, "r4_image_sibling": r4_image_sibling,
    "r4_earlier_state": r4_earlier_state, "r4_paraphrased_sibling": r4_paraphrased_sibling,
}
assert all(len(c) <= 20 and any(m in c for m in RELATIONAL_MARKERS) for c in (R2_FALSE_RELATION, R2_WRAPPED_FACT, R2_TRUE_RELATION))
assert len(R2_TRUE_RELATION_LONG) > 20


# --------------------------------------------------------------------------- 生成与打分

def sources(log: RunLog, base: list[Verdict]) -> list[Source]:
    by_ws = {str(v.subject.get("workspace_id")): v for v in base if v.subject.get("run") == log.name and v.unit == "report"}
    out = []
    for rec in log.store.workspace_report_records():
        if rec.task_id in by_ws:
            out.append(Source(log, rec.task_id, rec.to_dict(), by_ws[rec.task_id]))
    return out


def build_cases(log: RunLog, base: list[Verdict], *, kinds: list[str], per_kind: int, seed: int) -> list[RuleCase]:
    rng = random.Random(seed)
    out: list[RuleCase] = []
    for src in sources(log, base):
        findings = [f for f in src.report.get("findings") or [] if isinstance(f, dict) and f.get("evidence_ids")]
        for kind in kinds:
            made = 0
            for f in rng.sample(findings, k=len(findings)):
                if made >= per_kind:
                    break
                case = BUILDERS[kind](src, f, rng)
                if case is not None:
                    out.append(case)
                    made += 1
    return out


def _matches(case: RuleCase, e: ErrorRecord) -> bool:
    if e.type not in case.target_types or e.extra.get("finding") != case.finding_id or e.extra.get("confidence") == "flag":
        return False
    if case.target_span is None:
        return True
    span = re.sub(r"\s", "", e.span or "")
    target = re.sub(r"\s", "", case.target_span)
    return bool(span) and (target in span or span in target)


def classify(case: RuleCase, before: Verdict, after: Verdict) -> dict[str, Any]:
    """before：未校准的原始判定；after：recalibrate(before)。目标错误在 before 里没标出 → not_flagged（规则没被考到）。"""

    flagged = [e for e in before.errors if _matches(case, e)]
    survived = [e for e in after.errors if _matches(case, e)]
    applied = [a for a in (after.extra.get("calibration") or {}).get("applied") or []
               if a.get("finding") == case.finding_id and str(a.get("rule", "")).startswith(case.rule)]
    if not flagged:
        outcome = "not_flagged"
    else:
        outcome = "exempted" if applied and len(survived) < len(flagged) else "kept"
    # 翻正只算"因为目标错误被豁免"：把目标错误放回（去掉它降级出来的那条）若处置不是 positive，才记翻正。
    # 同报告别的 Finding 被其它规则豁免导致的翻正不算在这条样本头上。
    counterfactual = [e for e in after.errors if not (e.extra.get("finding") == case.finding_id and e.extra.get("calibrated_from"))] + \
        [e for e in flagged if e not in survived]
    flipped = outcome == "exempted" and after.disposition == "positive" and decide(counterfactual, after.pending) != "positive"
    return {"outcome": outcome, "exempted": outcome == "exempted", "rules_applied": sorted({str(a["rule"]) for a in applied}),
            "disposition_before": decide(before.errors, before.pending), "disposition_after": after.disposition, "disposition_flipped": flipped}


async def audit(run_dirs: list[Path], base: list[Verdict], *, kinds: list[str], per_kind: int, seed: int, mode: str = "verdict",
                judge: JsonJudge | None = None, judge_name: str | None = None, real_verdicts: list[Verdict] | None = None,
                say: Callable[[str], None] = print) -> dict[str, Any]:
    if mode == "report" and judge is None:
        raise ValueError("report 模式需要核验模型（--judge local|openrouter）；不调模型请用 --mode verdict")
    rows: list[dict[str, Any]] = []
    aborted = ""
    try:
        for run_dir in run_dirs:
            log = load_run(run_dir)
            try:
                if log.origin == "D":
                    continue
                cases = build_cases(log, base, kinds=kinds, per_kind=per_kind, seed=seed)
                say(f"[{log.name}] {len(cases)} 个规则样本（{mode} 模式）")
                for case in cases:
                    if mode == "verdict":
                        before = case.simulated
                    else:
                        subject = {"run": log.name, "case_id": log.case_id, "workspace_id": case.workspace_id, "rule_audit": case.kind}
                        before = await verify_report(case.report, log.lookup, subject=subject, judge=judge, judge_model=judge_name,
                                                     visible_ids=_visible_ids(run_dir, case.workspace_id))
                    after = recalibrate(before)
                    result = classify(case, before, after)
                    rows.append({**case.to_dict(), **result, "mode": mode})
                    say(f"  {case.kind:32s} {case.workspace_id} {case.finding_id:4s} {result['outcome']:11s} {result['rules_applied']} — {case.detail[:80]}")
            finally:
                log.close()
    except JudgeBudgetExhausted as exc:
        aborted = str(exc)
        say(budget_message(exc))
    summary = summarize(rows, kinds=kinds, mode=mode, judge_name=judge_name)
    summary["observed_real_triggers"] = real_triggers(real_verdicts or [])
    if aborted:
        summary["aborted"] = "budget_exhausted"
        summary["aborted_message"] = aborted
    return summary


def _rate(sub: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [r for r in sub if r["outcome"] != "not_flagged"]
    exempted = sum(1 for r in flagged if r["exempted"])
    return {"n": len(sub), "flagged": len(flagged), "not_flagged": len(sub) - len(flagged), "exempted": exempted,
            "rate": round(exempted / len(flagged), 3) if flagged else None,
            "disposition_flipped": sum(1 for r in flagged if r["exempted"] and r["disposition_flipped"])}


def summarize(rows: list[dict[str, Any]], *, kinds: list[str], mode: str, judge_name: str | None) -> dict[str, Any]:
    by_kind = {k: {**_rate([r for r in rows if r["kind"] == k]), "rule": k[:2].upper(), "should_exempt": BUILDERS[k] in _POSITIVE}
               for k in kinds}
    by_rule: dict[str, dict[str, Any]] = {}
    for rule in ("R1", "R2", "R4"):
        neg = [r for r in rows if r["rule"] == rule and not r["should_exempt"]]
        pos = [r for r in rows if r["rule"] == rule and r["should_exempt"]]
        by_rule[rule] = {"false_accept": _rate(neg), "true_accept": _rate(pos)}
    return {"mode": mode, "judge_model": judge_name,
            "note": ("verdict 模式：目标 Finding 的核验结果是模拟“如实标出错误”的核验器，量的是规则本身在触发条件下的条件假/真接受率；"
                     if mode == "verdict" else "report 模式：端到端，not_flagged = 核验模型本来就没标出目标错误，不进分母；") +
                    "false_accept.rate = 反例被豁免 / 反例被标出；true_accept.rate = 正例被豁免 / 正例被标出；disposition_flipped = 豁免后处置翻成 positive",
            "n_cases": len(rows), "by_rule": by_rule, "by_kind": by_kind, "rows": rows}


_POSITIVE = {r1_joint_support, r1_joint_support_two, r2_one_sided_relation, r2_one_sided_relation_long, r4_earlier_state, r4_paraphrased_sibling}


def real_triggers(verdicts: list[Verdict]) -> dict[str, Any]:
    """真实判定上各规则实际触发了多少次（背景数：规则在真实数据里有多常用）。"""

    counts: dict[str, int] = {}
    units = 0
    for v in verdicts:
        if v.unit not in ("report", "answer"):
            continue
        units += 1
        for a in (recalibrate(v).extra.get("calibration") or {}).get("applied") or []:
            rule = str(a.get("rule"))
            counts[rule] = counts.get(rule, 0) + 1
    return {"units": units, "applied": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校准规则 R1/R2/R4 的反例注入：假接受率与真接受率")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl")
    parser.add_argument("--units", default="experiments/sft_data/verdicts_units_judged.jsonl", help="只用来统计真实数据上的规则触发次数")
    parser.add_argument("--mode", default="verdict", choices=["verdict", "report"])
    add_judge_arguments(parser, default="none")
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl")
    parser.add_argument("--kinds", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    parser.add_argument("--per-kind", type=int, default=2, help="每份报告每种样本最多几条")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    judge = judge_from_args(args, cache_path=args.cache or None) if args.mode == "report" else None
    if args.mode == "report" and judge is None:
        parser.error("--mode report 需要 --judge local 或 openrouter")
    judge_name = getattr(judge, "name", None) if judge is not None else None

    def read(path: str) -> list[Verdict]:
        p = Path(path)
        return [Verdict.from_dict(json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []

    base = read(args.verdicts)
    summary = asyncio.run(audit([Path(r) for r in args.runs], base, kinds=args.kinds, per_kind=args.per_kind, seed=args.seed, mode=args.mode,
                                judge=judge, judge_name=judge_name, real_verdicts=base + read(args.units)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{summary['n_cases']} 个规则样本（{args.mode}）→ {out}")
    for rule, row in summary["by_rule"].items():
        fa, ta = row["false_accept"], row["true_accept"]
        print(f"  {rule} 假接受 {fa['exempted']}/{fa['flagged']}（翻正 {fa['disposition_flipped']}）  真接受 {ta['exempted']}/{ta['flagged']}"
              f"  not_flagged {fa['not_flagged'] + ta['not_flagged']}")
    line = judge_summary(judge)
    if line:
        print(line, flush=True)
    return BUDGET_EXIT_CODE if summary.get("aborted") else 0


__all__ = ["BUILDERS", "RuleCase", "audit", "build_cases", "classify", "main", "real_triggers", "summarize"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
