"""第三层：充分性。整份报告对照任务文字和必需证据清单：该拿的拿齐没有、状态报得对不对。

这一层看的是"缺了什么"，所以不能只看 Finding 本身；输入是任务全文 + 案卷的必需证据清单 + 报告。
它的错误都是非传播型（omitted_required_fact / completed_without_finishing / incomplete_though_answerable）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, error
from .anchored import JsonJudge

SUFFICIENCY_SYSTEM_PROMPT = """你是报告审核员。给你一个局部调查任务、任务要求取得的证据清单（可能为空）和调查员提交的报告。
只依据报告里写出来的内容判断，不补充外部知识，也不评价文笔。输出 JSON：
{"requirements":[{"key":"清单里的 key 或你从任务文字里归纳的要求（短语）","covered":true|false,"by":"支持它的 finding 编号或 null","note":"一句话"}],
 "answerable_from_report":true|false,   // 仅凭报告中的 Finding 是否已能回答该局部任务
 "status_should_be":"COMPLETED|INCOMPLETE",
 "missing":["报告没有回答、而任务明确要求的点"]}
规则：
- 清单里 must_cite=true 的项，报告必须有 Finding 引用了对应证据才算 covered。
- 任务文字里明确要求的每一点（例如"说明来源""给出时间区间""核验原站"）都要列成一条 requirement。
- 报告写了 limitations 说明为何取不到，不等于 covered，但可以让 status_should_be 是 INCOMPLETE。
- 保守：拿不准就 covered=false，并在 note 里说明。"""


@dataclass
class SufficiencyResult:
    errors: list[ErrorRecord] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors], "raw": self.raw}


def _report_payload(report: dict[str, Any]) -> dict[str, Any]:
    findings = []
    for index, finding in enumerate(report.get("findings") or []):
        if isinstance(finding, dict):
            findings.append({"finding_id": finding.get("finding_id") or f"finding[{index}]",
                             "statement": finding.get("statement"), "evidence_ids": list(finding.get("evidence_ids") or []),
                             "limitations": list(finding.get("limitations") or [])})
    return {"status": report.get("status"), "status_reason": report.get("status_reason"), "findings": findings}


async def verify_sufficiency(report: dict[str, Any], task_text: str, judge: JsonJudge, *, case: Case | None = None) -> SufficiencyResult:
    required = [{"key": r.key, "description": r.description, "must_cite": r.must_cite} for r in (case.required_evidence if case else ())]
    payload = {"task": task_text, "required_evidence": required, "report": _report_payload(report)}
    raw = await judge.judge(system_prompt=SUFFICIENCY_SYSTEM_PROMPT, payload=payload, images=[])
    result = SufficiencyResult(raw=dict(raw))
    status = report.get("status")
    requirements = [r for r in (raw.get("requirements") or []) if isinstance(r, dict)]
    must_keys = {r["key"] for r in required if r["must_cite"]}
    uncovered = [r for r in requirements if not r.get("covered")]
    for item in uncovered:
        key = str(item.get("key") or "?")
        result.errors.append(error(ErrorType.OMITTED_REQUIRED_FACT, f"任务要求 {key!r} 未被回答：{item.get('note') or ''}".strip("："),
                                   method="model", extra={"requirement": key, "from_case": key in {r['key'] for r in required}}))
    missing_must = [r for r in uncovered if str(r.get("key")) in must_keys]
    should = raw.get("status_should_be")
    if status == "COMPLETED" and (missing_must or should == "INCOMPLETE"):
        keys = ", ".join(str(r.get("key")) for r in (missing_must or uncovered))
        detail = f"报 COMPLETED 但任务要求未全部覆盖：{keys}" if keys else "报 COMPLETED 但审核认为应为 INCOMPLETE"
        result.errors.append(error(ErrorType.COMPLETED_WITHOUT_FINISHING, detail, method="model", extra={"status_should_be": should}))
    if status == "INCOMPLETE" and bool(raw.get("answerable_from_report")) and not uncovered and should == "COMPLETED":
        result.errors.append(error(ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, "报告内容已足以回答任务，却报 INCOMPLETE",
                                   method="model", extra={"status_should_be": should}))
    return result


__all__ = ["SUFFICIENCY_SYSTEM_PROMPT", "SufficiencyResult", "verify_sufficiency"]
