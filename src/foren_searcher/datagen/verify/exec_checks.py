"""第一层：纯程序核验。判得死的记 hard，拿不准的记 flag（extra["confidence"]）交给模型层。"""

from __future__ import annotations

import re
from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity, error
from .context import EvidenceLookup, EvidenceText, normalize_source_key

_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?")
_ISO_DATE = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})日?")
_INTERVAL = re.compile(r"(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*[–\-~至到]\s*(\d+(?:\.\d+)?)\s*(?:秒|s)")


def _norm_number(token: str) -> str | None:
    value = token.replace(",", "").rstrip("%")
    try:
        number = float(value)
    except ValueError:
        return None
    if abs(number) < 10 and "." not in value:
        return None            # 个位整数噪音太大（序号、"两次"里的 2），不查
    if number == int(number) and "e" not in value.lower():
        return str(int(number))          # 整数保留全部位数（format 'g' 只留 6 位有效数字会把 3735245 变成 3735240）
    return format(number, ".6f").rstrip("0").rstrip(".")


def numbers_in(text: str) -> set[str]:
    out = set()
    for match in _NUMBER.finditer(text):
        norm = _norm_number(match.group())
        if norm is not None:
            out.add(norm)
    return out


def _close(a: str, b: str) -> bool:
    """1.033333 与 1.033 算同一个数：绝对误差 ≤ 0.01 或相对误差 ≤ 0.1%。"""

    x, y = float(a), float(b)
    return abs(x - y) <= max(0.01, 1e-3 * max(abs(x), abs(y)))


def numbers_missing(statement: str, pool: str) -> list[str]:
    """结论里的数字在原文里找不到的那些。两条匹配途径：数值近似相等；或（≥4 位的）数字串出现在去掉分隔符的原文里
    （对付 PDF 表格里 "3 735 245" / "3,735,245" 这类分组写法）。"""

    have = numbers_in(pool)
    digits_only = re.sub(r"[^0-9]", "", pool)
    missing = []
    for n in sorted(numbers_in(statement)):
        if any(_close(n, h) for h in have):
            continue
        run = n.replace(".", "").lstrip("-")
        if len(run) >= 4 and run in digits_only:
            continue
        missing.append(n)
    return missing


def dates_in(text: str) -> list[str]:
    return [f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in _ISO_DATE.findall(text)]


def _finding_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, finding in enumerate(report.get("findings") or []):
        if not isinstance(finding, dict):
            continue
        rows.append({
            "index": index, "id": finding.get("finding_id") or f"finding[{index}]",
            "statement": str(finding.get("statement") or ""),
            "evidence_ids": [e for e in (finding.get("evidence_ids") or []) if isinstance(e, str)],
        })
    return rows


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def exec_checks(report: dict[str, Any], lookup: EvidenceLookup, *, case: Case | None = None,
                visible_ids: set[str] | None = None) -> list[ErrorRecord]:
    """report：{status, findings[{statement, evidence_ids, limitations}], selected_evidence_ids, status_reason}。"""

    errors: list[ErrorRecord] = []
    status = report.get("status")
    findings = _finding_rows(report)

    def hard(kind: ErrorType, detail: str, **kw: Any) -> None:
        errors.append(error(kind, detail, method="exec", extra={"confidence": "hard", **kw.pop("extra", {})}, **kw))

    def flag(kind: ErrorType, detail: str, *, severity: Severity | None = None, **kw: Any) -> None:
        if severity is not None:
            kw["severity"] = severity
        errors.append(error(kind, detail, method="exec", extra={"confidence": "flag", **kw.pop("extra", {})}, **kw))

    # ---- 报告级
    if status == "COMPLETED" and not findings:
        hard(ErrorType.EMPTY_REPORT, "COMPLETED 报告没有任何 Finding")
    if status == "INCOMPLETE" and not findings and not str(report.get("status_reason") or "").strip():
        hard(ErrorType.EMPTY_REPORT, "INCOMPLETE 报告既无 Finding 也无 status_reason")

    # ---- Finding 级：引用存在性、可见性、数字
    cited_texts: dict[str, EvidenceText] = {}
    for row in findings:
        if not row["evidence_ids"]:
            hard(ErrorType.UNSUPPORTED_ASSERTION, "Finding 没有引用任何 E", span=row["statement"], extra={"finding": row["id"]})
            continue
        texts: list[EvidenceText] = []
        for eid in row["evidence_ids"]:
            if not lookup.has(eid):
                hard(ErrorType.FABRICATED_EVIDENCE_ID, f"引用的 {eid} 在证据库中不存在", span=row["statement"],
                     evidence_ids=[eid], extra={"finding": row["id"]})
                continue
            if visible_ids is not None and eid not in visible_ids:
                hard(ErrorType.FABRICATED_EVIDENCE_ID, f"引用的 {eid} 在提交报告时不可见", span=row["statement"],
                     evidence_ids=[eid], extra={"finding": row["id"], "reason": "not_visible"})
                continue
            text = lookup.get(eid)
            cited_texts[eid] = text
            texts.append(text)
        if not texts:
            continue
        text_pool = "\n".join(t.content for t in texts if t.is_text)
        has_pixels = any(t.is_image for t in texts)
        missing = numbers_missing(row["statement"], text_pool)
        if missing:
            detail = f"数字 {', '.join(missing)} 未在所引 E 的文本中出现" + ("；所引含图片，可能来自像素" if has_pixels else "")
            flag(ErrorType.UNSUPPORTED_ASSERTION, detail, severity=Severity.MAJOR, span=row["statement"],
                 evidence_ids=[t.evidence_id for t in texts],
                 extra={"finding": row["id"], "check": "numeric_in_cited_text", "values": missing, "needs_pixels": has_pixels})

    if case is None:
        return errors

    # ---- 来源限制
    excluded = [k.lower() for k in case.source_policy.excluded_keys]
    for eid, text in cited_texts.items():
        if text.source_url and excluded:
            key = normalize_source_key(text.source_url)
            if any(key == ex or key.startswith(ex.rstrip("/") + "/") for ex in excluded):
                hard(ErrorType.SOURCE_RESTRICTION_MISS, f"{eid} 来自任务排除的来源 {text.source_url}", evidence_ids=[eid])

    # ---- 案卷断言
    statements = " \n ".join(row["statement"] for row in findings)
    cited_text_pool = "\n".join(t.content for t in cited_texts.values() if t.is_text)
    required = {r.key: r for r in case.required_evidence}
    for check in case.checks:
        params = check.params
        if check.kind == "status_expected":
            expected = params["status"]
            if status != expected and expected == "COMPLETED":
                hard(ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, "案卷标注可回答，报告却是 INCOMPLETE", extra={"check": check.kind})
            elif status != expected:
                hard(ErrorType.COMPLETED_WITHOUT_FINISHING, "案卷标注不可完成，报告却是 COMPLETED", extra={"check": check.kind})
        elif check.kind == "label_expected":
            label = str(params["label"])
            if label.lower() not in statements.lower():
                hard(ErrorType.OMITTED_REQUIRED_FACT, f"结论中没有出现应有的标签 {label!r}", extra={"check": check.kind, "reference": True})
        elif check.kind == "entity_in_evidence":
            needle = str(params["text"])
            if needle.lower() not in statements.lower():
                hard(ErrorType.OMITTED_REQUIRED_FACT, f"结论中没有出现应有的实体 {needle!r}", extra={"check": check.kind, "reference": True})
            elif needle.lower() not in cited_text_pool.lower():
                flag(ErrorType.UNSUPPORTED_ASSERTION, f"实体 {needle!r} 出现在结论里，但不在所引 E 的文本中", severity=Severity.MAJOR,
                     extra={"check": check.kind})
        elif check.kind == "numeric_in_evidence":
            target = float(params["value"])
            tol = float(params.get("tolerance", 0))
            found = [n for n in numbers_in(statements) if abs(float(n) - target) <= tol]
            if not found:
                hard(ErrorType.OMITTED_REQUIRED_FACT, f"结论中没有出现应有的数值 {params['value']}", extra={"check": check.kind, "reference": True})
            elif not any(abs(float(n) - target) <= tol for n in numbers_in(cited_text_pool)):
                flag(ErrorType.UNSUPPORTED_ASSERTION, f"数值 {found[0]} 出现在结论里，但不在所引 E 的文本中", severity=Severity.MAJOR,
                     extra={"check": check.kind})
        elif check.kind == "date_in_range":
            found = dates_in(statements)
            lo, hi = params.get("min"), params.get("max")
            if not found:
                hard(ErrorType.OMITTED_REQUIRED_FACT, "结论中没有给出日期", extra={"check": check.kind, "reference": True})
            else:
                bad = [d for d in found if (lo and d < lo) or (hi and d > hi)]
                if bad and len(bad) == len(found):
                    hard(ErrorType.UNSUPPORTED_ASSERTION, f"结论日期 {', '.join(bad)} 不在案卷范围 [{lo}, {hi}] 内",
                         extra={"check": check.kind, "reference": True})
        elif check.kind == "interval_iou":
            want = (float(params["start"]), float(params["end"]))
            got = [(float(a), float(b)) for a, b in _INTERVAL.findall(statements) if float(a) <= float(b)]
            if not got:
                hard(ErrorType.OMITTED_REQUIRED_FACT, "结论中没有给出时间区间", extra={"check": check.kind, "reference": True})
            elif max(_iou(want, g) for g in got) < float(params["min_iou"]):
                hard(ErrorType.UNSUPPORTED_ASSERTION, f"结论区间与案卷区间 {want} 重叠度低于 {params['min_iou']}",
                     extra={"check": check.kind, "reference": True, "best_iou": round(max(_iou(want, g) for g in got), 3)})
        elif check.kind == "evidence_key_required":
            req = required[params["key"]]
            hint = (req.source_hint or "").lower()
            if not hint:
                flag(ErrorType.OMITTED_REQUIRED_FACT, f"必需证据 {req.key} 没有 source_hint，留给充分性层判断",
                     extra={"check": check.kind, "pending_layer": 3, "key": req.key})
                continue
            matched = any(hint in ((t.source_url or "") + " " + t.label).lower() for t in cited_texts.values())
            if not matched:
                hard(ErrorType.OMITTED_REQUIRED_FACT, f"必需证据 {req.key}（{req.description}）未被引用",
                     extra={"check": check.kind, "key": req.key})
                if status == "COMPLETED" and req.must_cite:
                    hard(ErrorType.COMPLETED_WITHOUT_FINISHING, f"必需证据 {req.key} 未取得却报 COMPLETED",
                         extra={"check": check.kind, "key": req.key})
    return errors


__all__ = ["dates_in", "exec_checks", "numbers_in", "numbers_missing"]
