"""核验器精确率校准（规则版，verifier@0.2.0）：对已存的 report 单元判定按规则重算，不重新调用核验模型。

来源：`docs/04_训练与数据/sft轨迹示例_true4124903_20260921.md` 第 6 节。第 0 轮抽样发现的两类疑似误报都能用
判定文件里已有的信息（judgments / coverage）程序化识别，所以先做规则校准，把模型层的召回率校准（perturb 注入）
留到训练之后。

规则（每条都写进 verdict.extra["calibration"]，可审计、可回滚）：

R1 multi_evidence_scope —— 逐对核验给出的 scope_overreach 提示，在该 Finding 引用 ≥ 3 条 E、所有对都 supported、
   coverage 通过时予以豁免。理由：单看一对当然撑不起整句，合起来才撑得起；这是逐对核验的固有盲区，不是教师的错。
R2 relational_clause_citation —— unsupported_assertion 的 span 很短（≤ 20 字）且含比较/关系词（相容、一致、相符、
   对应、吻合、印证），同一报告里有别的 Finding 引用了文本类 E 并被判 supported 时，降级为 incomplete_citation_set
   （major，不传播）。理由：比较句一边引了帧、一边没引报道，支持其实在本工作区里，错在引用集合不全，不该按
   "凭空断言"一票否决。
R3 minor_no_taint —— 由 verdict.tainted() 的新规则承担：minor 不再传播污染。这里只是重算 tainted_findings。
R4 contradiction_covered_by_sibling（calibrate@0.3.0，来自 answer 逐句锚定的第一次真实结果）—— 一句话引了多条 E，
   其中一条对某部分判 contradicted，但同一句另一条被引 E 对**同一部分**给了程序核实的支持片段（或图片来源的支持）时，
   撤销这条反驳。理由：教师引 E24（"卡车起火，消防员正在出动"）+ E21（"已扑灭"）本来就是在引前后两条通报，
   前一条不是反驳后一条；逐对核验看不到另一条 E，把"时间上更早的状态"当成了"相反"。

适用单元：report 与 answer（answer 的"Finding"就是拆出来的句子 A1…An，judgments / coverage 形态一致）。

重算后 disposition 用同一个 decide()；原判定原样保留在 extra["calibration"]["before"]。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..errors import ErrorRecord, ErrorType, Severity, error
from .verdict import Verdict, decide, tainted

RELATIONAL_MARKERS = ("相容", "一致", "相符", "对应", "吻合", "印证", "相同", "匹配")
CALIBRATION_VERSION = "calibrate@0.3.0"
_CONTRADICTION_TYPES = (ErrorType.WRONG_CITATION, ErrorType.MISREAD_FINE_TEXT, ErrorType.SLOPPY_NUMERIC)


def _squash(text: str) -> str:
    import re
    return re.sub(r"\s+", "", text).replace("“", '"').replace("”", '"').lower()


def _sibling_supports_part(verdict: Verdict, finding_id: str, part: str, contradicting_eid: str) -> str | None:
    """同一 Finding 的另一条被引 E 是否有核实的支持片段覆盖这一部分；返回那条 E 的编号。"""

    target = _squash(part)
    if len(target) < 2:
        return None
    for j in _judgments_for(verdict, finding_id):
        if j.get("evidence_id") == contradicting_eid or j.get("verdict") != "supported":
            continue
        for p in j.get("supported") or []:
            if p.get("verified") is False:
                continue
            text = _squash(str(p.get("part") or ""))
            if target in text or (len(text) >= 2 and text in target):
                return str(j.get("evidence_id"))
    return None


def _judgments_for(verdict: Verdict, finding_id: str) -> list[dict[str, Any]]:
    return [j for j in verdict.judgments if j.get("finding_id") == finding_id]


def _coverage_ok(verdict: Verdict, finding_id: str) -> bool:
    for c in verdict.coverage:
        if c.get("finding_id") == finding_id:
            return bool(c.get("grounded")) and not c.get("unsupported_parts")
    return False


def _has_other_supported_text_finding(verdict: Verdict, finding_id: str) -> bool:
    for j in verdict.judgments:
        if j.get("finding_id") == finding_id:
            continue
        if j.get("verdict") == "supported" and not j.get("is_image"):
            return True
    return False


def recalibrate(verdict: Verdict) -> Verdict:
    """返回一个新的 Verdict（不改原对象）。处理 report 与 answer 单元（answer 需已做逐句锚定）。"""

    if verdict.unit not in ("report", "answer"):
        return verdict
    if verdict.unit == "answer" and (verdict.extra.get("anchored") or {}).get("method") != "judge_sentences":
        return verdict
    before = {"disposition": verdict.disposition, "errors": [e.to_dict() for e in verdict.errors],
              "tainted_findings": list(verdict.tainted_findings), "verifier": verdict.verifier}
    applied: list[dict[str, Any]] = []
    kept: list[ErrorRecord] = []
    for e in verdict.errors:
        finding = e.extra.get("finding") if isinstance(e.extra, dict) else None
        # R1
        if e.type is ErrorType.SCOPE_OVERREACH and e.method == "model" and isinstance(finding, str):
            pairs = _judgments_for(verdict, finding)
            cited = {j.get("evidence_id") for j in pairs}
            all_supported = bool(pairs) and all(j.get("verdict") == "supported" and not j.get("contradicted") for j in pairs)
            if len(cited) >= 3 and all_supported and _coverage_ok(verdict, finding):
                applied.append({"rule": "R1_multi_evidence_scope", "finding": finding, "dismissed": e.to_dict(),
                                "reason": f"{finding} 引用 {len(cited)} 条 E 且全部 supported、coverage 通过；逐对提示的 scope 不成立"})
                continue
        # R4
        if e.type in _CONTRADICTION_TYPES and e.method in ("model", "pixels") and isinstance(finding, str) and isinstance(e.span, str) \
                and len(e.evidence_ids) == 1:
            sibling = _sibling_supports_part(verdict, finding, e.span, e.evidence_ids[0])
            if sibling:
                applied.append({"rule": "R4_contradiction_covered_by_sibling", "finding": finding, "dismissed": e.to_dict(),
                                "reason": f"{e.evidence_ids[0]} 对「{e.span[:40]}」判反驳，但同句所引 {sibling} 有核实的支持片段覆盖同一部分"})
                continue
        # R2
        if verdict.unit == "report" and e.type is ErrorType.UNSUPPORTED_ASSERTION and e.method == "model" and isinstance(e.span, str):
            span = e.span.strip()
            if 0 < len(span) <= 20 and any(m in span for m in RELATIONAL_MARKERS) and isinstance(finding, str) \
                    and _has_other_supported_text_finding(verdict, finding):
                downgraded = error(ErrorType.INCOMPLETE_CITATION_SET,
                                   f"关系句「{span}」只引了一边：支持它的文本 E 已被同报告其它 Finding 引用并核实（原判 unsupported_assertion）",
                                   span=span, evidence_ids=list(e.evidence_ids), method="model",
                                   extra={**e.extra, "calibrated_from": e.type.value})
                applied.append({"rule": "R2_relational_clause_citation", "finding": finding, "from": e.to_dict(), "to": downgraded.to_dict()})
                kept.append(downgraded)
                continue
        kept.append(e)
    new = replace(verdict, errors=kept)
    new.tainted_findings = tainted(kept)
    new.disposition = decide(kept, new.pending)
    new.extra = {**verdict.extra, "calibration": {"version": CALIBRATION_VERSION, "applied": applied, "before": before}}
    if new.verifier < "verifier@0.2.0":
        new.verifier = "verifier@0.2.0"
    if verdict.unit == "answer":
        new.extra["anchored"] = {**(verdict.extra.get("anchored") or {}),
                                 "contradicted_sentences": sorted({str(x.extra.get("sentence")) for x in kept if x.type in _CONTRADICTION_TYPES}),
                                 "unsupported_sentences": sorted({str(x.extra.get("sentence")) for x in kept if x.type is ErrorType.UNSUPPORTED_ASSERTION})}
    return new


__all__ = ["CALIBRATION_VERSION", "RELATIONAL_MARKERS", "recalibrate"]
