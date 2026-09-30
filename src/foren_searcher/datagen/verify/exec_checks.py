"""第一层：纯程序核验。判得死的记 hard，拿不准的记 flag（extra["confidence"]）交给模型层。"""

from __future__ import annotations

import re
from typing import Any

from ..cases import Case, Check
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


def fact_check_errors(check: Check, statements: str, cited_text_pool: str) -> list[ErrorRecord]:
    """entity_in_evidence / numeric_in_evidence 的共用判定（工作区报告与 answer 单元同口径）。

    statements：结论文字（报告全部 Finding 或 GIA 答案）；cited_text_pool：所引 E 的文本拼接。
    不出现 → hard omitted_required_fact；出现但所引 E 文本里没有 → flag unsupported_assertion（MAJOR）。其它 kind 返回 []。
    """

    params = check.params
    out: list[ErrorRecord] = []

    def hard(kind: ErrorType, detail: str, extra: dict[str, Any]) -> None:
        out.append(error(kind, detail, method="exec", extra={"confidence": "hard", **extra}))

    def flag(kind: ErrorType, detail: str, extra: dict[str, Any]) -> None:
        out.append(error(kind, detail, method="exec", severity=Severity.MAJOR, extra={"confidence": "flag", **extra}))

    scope = {"requirement": check.requirement} if check.requirement else {}
    if check.kind == "entity_in_evidence":
        needle = str(params["text"])
        # params.aliases（可选）：所引 E 原文里的等价写法（如 耶拿桥 ↔ pont d'Iéna），只用于"在所引 E 文本中"这一步
        forms = [needle, *(str(a) for a in params.get("aliases") or [])]
        if needle.lower() not in statements.lower():
            hard(ErrorType.OMITTED_REQUIRED_FACT, f"结论中没有出现应有的实体 {needle!r}", {"check": check.kind, "reference": True, **scope})
        elif not any(form.lower() in cited_text_pool.lower() for form in forms):
            flag(ErrorType.UNSUPPORTED_ASSERTION, f"实体 {needle!r} 出现在结论里，但不在所引 E 的文本中", {"check": check.kind, **scope})
    elif check.kind == "numeric_in_evidence":
        target = float(params["value"])
        tol = float(params.get("tolerance", 0))
        found = [n for n in numbers_in(statements) if abs(float(n) - target) <= tol]
        if not found:
            hard(ErrorType.OMITTED_REQUIRED_FACT, f"结论中没有出现应有的数值 {params['value']}", {"check": check.kind, "reference": True, **scope})
        elif not any(abs(float(n) - target) <= tol for n in numbers_in(cited_text_pool)):
            flag(ErrorType.UNSUPPORTED_ASSERTION, f"数值 {found[0]} 出现在结论里，但不在所引 E 的文本中", {"check": check.kind, **scope})
    return out


# interval_iou 同样按 check.requirement 收窄（2026-09-28 第二批案卷审稿发现：只做边界复核的工作区不该被整案区间检查判错）
_SCOPED_KINDS = frozenset({"entity_in_evidence", "numeric_in_evidence", "date_in_range", "status_expected", "interval_iou"})


def exec_checks(report: dict[str, Any], lookup: EvidenceLookup, *, case: Case | None = None,
                visible_ids: set[str] | None = None, requirement_ids: set[str] | None = None) -> list[ErrorRecord]:
    """report：{status, findings[{statement, evidence_ids, limitations}], selected_evidence_ids, status_reason}。

    requirement_ids：这份报告所属工作区对准的案卷要求（scope.requirements_for_workspace）；给了就只按这几项反查必需证据。

    案卷断言里跳过 label_expected（verifier@0.3.0，09-24 收窄测试发现）：标签是整案结论，只有 GIA 的 FINISH 答案该带，
    由 units.check_answer 在 answer 单元核对；工作区报告只查来源 / 语境等子任务，不该因为没写"不实"之类的标签被记
    omitted_required_fact。其它 check kinds 照旧在这里跑。

    check.requirement（09-24）：entity_in_evidence / numeric_in_evidence / date_in_range / status_expected 若标了 requirement，
    且给了 requirement_ids 而该 REQ 不在其中，就跳过——只查外部报道的实体不该扣只看视频画面的工作区。
    requirement_ids 为 None（不收窄）时全部照跑。answer 单元不分 requirement 全查（units.check_answer）。
    """

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

    # ---- 必需证据（verifier@0.3.0）：按工作区范围，用 url_pattern / acquired_by 程序匹配所引 E；any_one_of 分组任取其一
    scoped = {r.key: r for r in case.scoped_required_evidence(requirement_ids)}
    matchable = {k for k, r in scoped.items() if r.url_pattern or r.acquired_by or r.source_hint}
    matched_keys = {key for key, req in scoped.items()
                    if any(req.matches(source_url=t.source_url, acquired_by=t.acquired_by, label=t.label) for t in cited_texts.values())}
    for mode, keys in case.evidence_groups():
        keys = tuple(k for k in keys if k in scoped)
        if not keys:
            continue
        satisfied = all(k in matched_keys for k in keys) if mode == "all" else any(k in matched_keys for k in keys)
        if satisfied:
            continue
        must = any(scoped[k].must_cite for k in keys)
        label = " / ".join(keys) if mode == "any_one_of" else keys[0]
        if not must:
            # verifier@0.3.1：组内全部 must_cite=False = 可选证据（帮助性材料，如 media_profile / alignment / 旁证报道）。
            # 没引用只记 minor 备注：不计入处置（decide 只数 critical / major）、不转 uncertain、不污染 Finding。
            errors.append(error(ErrorType.OMITTED_REQUIRED_FACT, f"可选证据 {label} 未被引用（仅备注）", method="exec",
                                severity=Severity.MINOR,
                                extra={"confidence": "hard", "check": "required_evidence_cited", "optional": True, "key": keys[0],
                                       "keys": list(keys), "mode": mode,
                                       "scope": sorted(requirement_ids) if requirement_ids else None}))
            continue
        if not all(k in matchable for k in keys):
            # 没有 url_pattern / acquired_by / source_hint 的项程序判不了，留给第三层按描述判
            flag(ErrorType.OMITTED_REQUIRED_FACT, f"必需证据 {' / '.join(keys)} 没有程序可匹配的特征，留给充分性层判断",
                 extra={"check": "required_evidence_cited", "pending_layer": 3, "key": keys[0], "keys": list(keys)})
            continue
        hard(ErrorType.OMITTED_REQUIRED_FACT, f"必需证据 {label}（{scoped[keys[0]].description}）未被任何 Finding 引用",
             extra={"check": "required_evidence_cited", "key": keys[0], "keys": list(keys), "mode": mode, "scope": sorted(requirement_ids) if requirement_ids else None})
        if status == "COMPLETED" and must:
            hard(ErrorType.COMPLETED_WITHOUT_FINISHING, f"必需证据 {label} 未取得却报 COMPLETED",
                 extra={"check": "required_evidence_cited", "key": keys[0]})

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
        if check.kind in _SCOPED_KINDS and not check.applies_to(requirement_ids):
            continue
        if check.kind == "status_expected":
            expected = params["status"]
            if status != expected and expected == "COMPLETED":
                hard(ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, "案卷标注可回答，报告却是 INCOMPLETE", extra={"check": check.kind})
            elif status != expected:
                hard(ErrorType.COMPLETED_WITHOUT_FINISHING, "案卷标注不可完成，报告却是 COMPLETED", extra={"check": check.kind})
        elif check.kind == "label_expected":
            continue          # 整案标签只在 answer 单元核（units.check_answer），见 docstring
        elif check.kind in ("entity_in_evidence", "numeric_in_evidence"):
            errors.extend(fact_check_errors(check, statements, cited_text_pool))
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
            # evidence_key_required：0.3.0 起由上面的必需证据检查统一处理（按范围、按分组），这里不再重复
            continue
    return errors


__all__ = ["dates_in", "exec_checks", "fact_check_errors", "numbers_in", "numbers_missing"]
