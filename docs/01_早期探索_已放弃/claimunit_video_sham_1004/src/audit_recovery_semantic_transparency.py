"""Audit whether VidDiff abstention and adjudication prose states its semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_control_plane_exposure import _preaction_payload, _target_call
from compare_recovery_humanlike_quality import _action_signatures
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BEFORE = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike/recovery_sft_candidate.jsonl"
DEFAULT_AFTER = BASE / "recovery_batch_candidate_v20_abstention_semantic_transparency/recovery_sft_candidate.jsonl"
DEFAULT_OUTPUT = BASE / "recovery_batch_candidate_v20_abstention_semantic_transparency/semantic_transparency_audit.json"


def _text(row: dict[str, Any]) -> str:
    return str(row["messages"][-1].get("content") or "")


def _assessment_observation(row: dict[str, Any]) -> dict[str, Any]:
    payload, _ = _preaction_payload(row)
    state = payload.get("current_harness_state") or payload.get("harness_state") or {}
    _, arguments = _target_call(row)
    evidence_id = str(arguments.get("evidence_id") or "")
    return (((state.get("evidence_store") or {}).get(evidence_id) or {}).get("observation") or {})


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c_rows = [row for row in rows if 'preferred_video="c"' in _text(row)]
    c_transparent = [
        row
        for row in c_rows
        if "abstention" in _text(row).lower()
        and "neither/equal/insufficient evidence" in _text(row).lower()
    ]
    assessment_rows = [
        row
        for row in rows
        if row["metadata"].get("causal_recovery_stage") == "evidence_assessment"
    ]
    retained_c_order_false = []
    for row in assessment_rows:
        _, arguments = _target_call(row)
        observation = _assessment_observation(row)
        if (
            arguments.get("disposition") == "retain"
            and observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
        ):
            retained_c_order_false.append(row)
    retained_c_explained = [
        row
        for row in retained_c_order_false
        if all(
            phrase in _text(row).lower()
            for phrase in ("abstention", "order", "third video")
        )
    ]
    redundant_rejections = []
    conflict_rejections = []
    for row in assessment_rows:
        _, arguments = _target_call(row)
        codes = arguments.get("defect_codes") or []
        if "redundant_same_conclusion" in codes:
            redundant_rejections.append(row)
        if "claim_value_conflict" in codes:
            conflict_rejections.append(row)
    redundant_explained = [
        row
        for row in redundant_rejections
        if any(word in _text(row).lower() for word in ("duplicate", "redundant"))
        and "contradictory evidence" not in _text(row).lower()
    ]
    conflict_explained = [
        row
        for row in conflict_rejections
        if "does not match" in _text(row).lower()
        and "binding" in _text(row).lower()
    ]
    adjudications = [
        row
        for row in rows
        if row["metadata"].get("causal_recovery_stage") == "evidence_adjudication"
    ]
    adjudication_semantic = []
    for row in adjudications:
        text = _text(row)
        _, arguments = _target_call(row)
        ids = list(arguments.get("selected_evidence_ids") or []) + [
            item["evidence_id"] for item in arguments.get("rejected_evidence") or []
        ]
        if all(evidence_id in text for evidence_id in ids) and any(
            field in text
            for field in ("preferred_video", "measurement_method", "confidence", "order_consistent")
        ):
            adjudication_semantic.append(row)
    return {
        "preferred_c_rows": len(c_rows),
        "preferred_c_semantics_explicit": len(c_transparent),
        "retained_c_order_inconsistent_rows": len(retained_c_order_false),
        "retained_c_order_inconsistency_explained": len(retained_c_explained),
        "redundant_rejection_rows": len(redundant_rejections),
        "redundancy_explained_without_false_contradiction": len(redundant_explained),
        "claim_conflict_rejection_rows": len(conflict_rejections),
        "claim_conflict_value_and_binding_consequence_explicit": len(conflict_explained),
        "adjudication_rows": len(adjudications),
        "adjudication_partition_semantically_rendered": len(adjudication_semantic),
    }


def compare(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    action_change_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_metrics = _metrics(before)
    after_metrics = _metrics(after)
    actions_unchanged = _action_signatures(before) == _action_signatures(after)
    evidence_first_changes_verified = bool(
        action_change_audit
        and action_change_audit.get("passed") is True
        and (action_change_audit.get("checks") or {}).get(
            "all_adjudications_match_label_blind_rule"
        )
        is True
        and (action_change_audit.get("checks") or {}).get(
            "all_target_bindings_use_selected_only"
        )
        is True
        and (action_change_audit.get("checks") or {}).get(
            "all_selected_values_match_final_claim"
        )
        is True
    )
    passed = bool(
        (actions_unchanged or evidence_first_changes_verified)
        and after_metrics["preferred_c_rows"]
        == after_metrics["preferred_c_semantics_explicit"]
        and after_metrics["retained_c_order_inconsistent_rows"]
        == after_metrics["retained_c_order_inconsistency_explained"]
        and after_metrics["redundant_rejection_rows"]
        == after_metrics["redundancy_explained_without_false_contradiction"]
        and after_metrics["claim_conflict_rejection_rows"]
        == after_metrics["claim_conflict_value_and_binding_consequence_explicit"]
        and after_metrics["adjudication_rows"]
        == after_metrics["adjudication_partition_semantically_rendered"]
    )
    return {
        "artifact": "recovery_semantic_transparency_audit_v1",
        "before": before_metrics,
        "after": after_metrics,
        "executed_action_signatures_unchanged": actions_unchanged,
        "evidence_first_action_changes_verified": evidence_first_changes_verified,
        "passed": passed,
        "uses_oracle_or_verifier": False,
        "interpretation": (
            "preferred_video=c is rendered as abstention, order disagreement is explained "
            "as the reason not to assert A/B, duplicate evidence is not mislabeled as "
            "contradictory, and adjudication prose names the semantic partition."
        ),
        "scope_limit": (
            "This checks deterministic transparency against student-visible state; it does "
            "not replace blind human review or student closed-loop evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-change-audit", type=Path)
    args = parser.parse_args()
    report = compare(
        _jsonl(args.before),
        _jsonl(args.after),
        action_change_audit=(
            json.loads(args.action_change_audit.read_text(encoding="utf-8"))
            if args.action_change_audit
            else None
        ),
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
