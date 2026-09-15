"""Shared failure taxonomy and coverage accounting for trajectory batches."""

from __future__ import annotations

from collections import Counter
from typing import Any


def generation_failure_metadata(error: Exception) -> dict[str, Any]:
    artifact = getattr(error, "failure_artifact", None)
    artifact = artifact if isinstance(artifact, dict) else {}
    reason_counts = {
        str(key): int(value)
        for key, value in (artifact.get("reason_counts") or {}).items()
    }
    if artifact.get("failure_kind") == "bounded_checkpoint_proposal_exhaustion":
        category = (
            "selected_branch_execution_inadmissible"
            if reason_counts.get("selected_branch_execution_inadmissible", 0)
            else "preaction_proposal_quality_exhausted"
        )
    elif artifact.get("failure_kind") == "bounded_state_proposal_exhaustion":
        category = "state_reasoning_quality_exhausted"
    elif artifact.get("schema_version") == "teacher_rollout_failure_v1":
        category = "bounded_teacher_episode_incomplete"
    else:
        category = "unclassified_execution_error"
    return {
        "failure_category": category,
        "failure_reason_counts": reason_counts,
        "failure_schema_version": artifact.get("schema_version"),
    }


def generation_admission_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_statuses = Counter(str(case.get("status") or "unknown") for case in cases)
    attempt_rows: list[dict[str, Any]] = []
    recovered_cases = 0
    for case in cases:
        attempts = list(case.get("attempts") or [])
        if attempts:
            attempt_rows.extend(attempts)
            if case.get("status") == "passed" and any(
                attempt.get("status") != "passed" for attempt in attempts[:-1]
            ):
                recovered_cases += 1
        elif case.get("status") not in {"ready", "preflight_blocked"}:
            attempt_rows.append(case)
    attempt_statuses = Counter(
        str(attempt.get("status") or "unknown") for attempt in attempt_rows
    )
    failure_categories = Counter(
        str(attempt.get("failure_category") or "uncategorized_failure")
        for attempt in attempt_rows
        if attempt.get("status") != "passed"
    )
    reason_counts: Counter[str] = Counter()
    for attempt in attempt_rows:
        reason_counts.update(
            {
                str(key): int(value)
                for key, value in (attempt.get("failure_reason_counts") or {}).items()
            }
        )
    passed = case_statuses.get("passed", 0)
    return {
        "case_count": len(cases),
        "accepted_cases": passed,
        "accepted_case_fraction": round(passed / max(1, len(cases)), 6),
        "case_status_counts": dict(sorted(case_statuses.items())),
        "attempt_count": len(attempt_rows),
        "attempt_status_counts": dict(sorted(attempt_statuses.items())),
        "failed_attempt_category_counts": dict(sorted(failure_categories.items())),
        "proposal_rejection_reason_counts": dict(sorted(reason_counts.items())),
        "cases_recovered_by_later_attempt": recovered_cases,
    }
