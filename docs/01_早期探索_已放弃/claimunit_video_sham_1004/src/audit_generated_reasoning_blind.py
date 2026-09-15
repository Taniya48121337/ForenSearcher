"""Target-hidden audit of generated rationale and action quality.

The auditor reads only each public pre-action state, visible tool schemas, and
the student's generated rationale/call.  It never reads the teacher action.
This is a reproducible developer audit, not an independent human rating.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import (
    _arguments_validate_visible_schema,
    _parse_generated_call,
)
from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl, _payload


DEFAULT_REPORT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "eval_v6_repair16_novel_validation24.json"
)
DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "decision_sufficient_state_v6_executable_affordances"
    / "validation/sft_all.jsonl"
)
DEFAULT_BRANCH_REPLAY = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "viddiff_surgery155_generated_branch_checkpoint_replay.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "target_hidden_reasoning_quality_audit_novel24.json"
)
DEFAULT_PACKET = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "target_hidden_reasoning_review_packet_novel24.jsonl"
)

STATE_TOOLS = {
    "create_candidate_claim",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
    "stop",
}
EVIDENCE_PATTERN = re.compile(r"\bEA-[A-Z0-9_-]+-\d{3}\b", re.I)


def _reasoning(text: str) -> str:
    return text.split("<tool_call>", 1)[0].strip()


def _active_atoms(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(evidence_id): atom
        for evidence_id, atom in (state.get("evidence_store") or {}).items()
        if atom.get("active", True)
    }


def _claim_fields(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        field
        for claim in state.get("candidate_claims") or []
        for field in (claim.get("fields") or {}).values()
    ]


def _open_high_priority_goals(state: dict[str, Any]) -> list[str]:
    return [
        str(goal["goal_id"])
        for goal in state.get("case_goals") or []
        if goal.get("status") == "open" and goal.get("priority") == "high"
    ]


def _assertion_checks(
    reasoning: str,
    state: dict[str, Any],
    chosen_name: str | None,
) -> tuple[bool, list[str]]:
    lower = reasoning.lower()
    active = _active_atoms(state)
    eligible = list(state.get("eligible_tools") or [])
    claims = list(state.get("candidate_claims") or [])
    fields = _claim_fields(state)
    problems: list[str] = []
    if "evidence store is empty" in lower and active:
        problems.append("claims_empty_evidence_store_but_active_evidence_exists")
    if ("no available evidence ids" in lower or "no available evidenceatom ids" in lower) and active:
        problems.append("claims_no_available_evidence_but_active_evidence_exists")
    if "no candidate claim exists" in lower and claims:
        problems.append("claims_no_candidate_claim_but_claim_exists")
    if (
        "only eligible tool" in lower
        or "only eligible action" in lower
        or "only remaining eligible measurement" in lower
    ) and eligible != ([chosen_name] if chosen_name else []):
        problems.append(f"claims_unique_eligible_action_but_visible_eligible_tools_are:{eligible}")
    if "every required field" in lower and "committed" in lower:
        if not fields or any(field.get("decision_status") != "committed" for field in fields):
            problems.append("claims_every_required_field_committed_but_state_disagrees")
    if "all five claim fields" in lower and "committed" in lower:
        if len(fields) != 5 or any(field.get("decision_status") != "committed" for field in fields):
            problems.append("claims_all_five_fields_committed_but_state_disagrees")
    if "every supported field remains `decision_status: open`" in lower:
        supported = [field for field in fields if field.get("evidence_status") == "supported"]
        if not supported or any(field.get("decision_status") != "open" for field in supported):
            problems.append("claims_all_supported_fields_open_but_state_disagrees")
    if "no open high-priority obligations" in lower and _open_high_priority_goals(state):
        problems.append("claims_no_open_high_priority_goal_but_one_is_open")
    dropped = {str(item).upper() for item in state.get("dropped_evidence_ids") or []}
    for evidence_id in EVIDENCE_PATTERN.findall(reasoning):
        suffix = reasoning.lower().split(evidence_id.lower(), 1)[1][:24]
        if "was dropped" in suffix and evidence_id.upper() not in dropped:
            problems.append(f"claims_uncorroborated_dropout:{evidence_id}")
    return not problems, problems


def _citation_checks(reasoning: str, state: dict[str, Any]) -> tuple[bool, list[str]]:
    visible = {str(item).upper() for item in (state.get("evidence_store") or {})}
    visible.update(str(item).upper() for item in state.get("dropped_evidence_ids") or [])
    cited = sorted({item.upper() for item in EVIDENCE_PATTERN.findall(reasoning)})
    unknown = [item for item in cited if item not in visible]
    return not unknown, unknown


def _gap_quality(reasoning: str, state: dict[str, Any]) -> tuple[bool, list[str]]:
    lower = reasoning.lower()
    has_gap = "blocking gap" in lower or "remaining gap" in lower or "gap remains" in lower
    has_visible_anchor = bool(EVIDENCE_PATTERN.search(reasoning)) or any(
        token in lower
        for token in (
            "evidence store",
            "current state",
            "visible state",
            "latest observation",
            "claim field",
            "comparison context",
            "pair context",
        )
    )
    reasons = []
    if not has_gap:
        reasons.append("no_explicit_blocking_or_remaining_gap")
    if not has_visible_anchor:
        reasons.append("gap_not_anchored_to_visible_state")
    return not reasons, reasons


def _language_quality(reasoning: str, name: str | None) -> tuple[bool, list[str]]:
    words = re.findall(r"\b\w+[\w'-]*\b", reasoning)
    reasons = []
    if len(words) < 28:
        reasons.append("rationale_too_short_for_state_gap_action_explanation")
    if len(words) > 190:
        reasons.append("rationale_unnecessarily_long")
    if any(token in reasoning.lower() for token in ("oracle", "verifier", "teacher target")):
        reasons.append("mentions_hidden_or_verifier_information")
    return not reasons, reasons


def _minimality(
    row_id: str,
    state: dict[str, Any],
    chosen_name: str | None,
    branch_replay: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    eligible = list(state.get("eligible_tools") or [])
    if chosen_name not in eligible:
        return False, ["chosen_action_not_visible_as_eligible"]
    if branch_replay and row_id == branch_replay.get("row_id"):
        recovery = branch_replay.get("complete_recovery") or {}
        extra = int(recovery.get("extra_measurement_count") or 0)
        compatibility = (branch_replay.get("generated_branch") or {}).get(
            "claim_gap_compatibility"
        ) or {}
        compatible = compatibility.get(
            "all_open_claim_values_supported",
            compatibility.get("all_supported", True),
        )
        if extra > 0 and not compatible:
            return False, [
                f"executed_branch_was_recoverable_but_required_{extra}_extra_measurements"
            ]
    return True, []


def audit(
    report_path: Path,
    sft_path: Path,
    branch_replay_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation = json.loads(report_path.read_text(encoding="utf-8"))
    # Deliberately retain only generated fields.  reference_tool and all other
    # teacher comparison fields in the evaluation artifact are ignored.
    generated_rows = [
        {
            "row_id": str(item["row_id"]),
            "dataset": str(item["dataset"]),
            "generated_text": str(item.get("generated_text", "")),
        }
        for item in evaluation["rows"]
        if item.get("generated")
    ]
    sft_rows = {
        str(item["metadata"]["row_id"]): item
        for item in _jsonl(sft_path)
    }
    branch_replay = None
    if branch_replay_path and branch_replay_path.exists():
        branch_replay = json.loads(branch_replay_path.read_text(encoding="utf-8"))
    packet: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    dimensions = [
        "action_contract",
        "evidence_reference_integrity",
        "state_assertion_integrity",
        "explicit_gap_grounding",
        "action_utility",
        "route_minimality",
        "natural_language_quality",
    ]
    critical = {
        "action_contract",
        "evidence_reference_integrity",
        "state_assertion_integrity",
        "action_utility",
        "route_minimality",
    }
    for item in generated_rows:
        row_id = item["row_id"]
        sft_row = sft_rows[row_id]
        payload = _payload(sft_row)
        state = payload["current_harness_state"]
        name, arguments = _parse_generated_call(item["generated_text"])
        reasoning = _reasoning(item["generated_text"])
        eligible = list(state.get("eligible_tools") or [])
        parseable = name is not None and arguments is not None
        schema_valid = _arguments_validate_visible_schema(sft_row, name, arguments)
        action_contract = parseable and schema_valid and name in eligible
        citation_ok, citation_reasons = _citation_checks(reasoning, state)
        assertions_ok, assertion_reasons = _assertion_checks(reasoning, state, name)
        gap_ok, gap_reasons = _gap_quality(reasoning, state)
        # A visible-eligible, schema-valid action has a defined state transition
        # or measurement role.  Actual media/feature effects are audited in the
        # separate checkpoint replay artifacts.
        utility_ok = action_contract and (
            name in STATE_TOOLS
            or any(
                tool["function"]["name"] == name
                for tool in sft_row.get("tools") or []
            )
        )
        minimality_ok, minimality_reasons = _minimality(
            row_id, state, name, branch_replay
        )
        language_ok, language_reasons = _language_quality(reasoning, name)
        scores = {
            "action_contract": int(action_contract),
            "evidence_reference_integrity": int(citation_ok),
            "state_assertion_integrity": int(assertions_ok),
            "explicit_gap_grounding": int(gap_ok),
            "action_utility": int(utility_ok),
            "route_minimality": int(minimality_ok),
            "natural_language_quality": int(language_ok),
        }
        failures = {
            key: value
            for key, value in {
                "action_contract": ([] if action_contract else [
                    f"parseable={parseable},schema_valid={schema_valid},eligible={name in eligible if name else False}"
                ]),
                "evidence_reference_integrity": citation_reasons,
                "state_assertion_integrity": assertion_reasons,
                "explicit_gap_grounding": gap_reasons,
                "action_utility": ([] if utility_ok else ["no_defined_useful_transition"]),
                "route_minimality": minimality_reasons,
                "natural_language_quality": language_reasons,
            }.items()
            if value
        }
        critical_pass = all(scores[name] for name in critical)
        total = sum(scores.values())
        accepted = critical_pass and total >= 6
        details.append(
            {
                "row_id": row_id,
                "dataset": item["dataset"],
                "generated_tool": name,
                "visible_eligible_tools": eligible,
                "dimension_scores": scores,
                "score": total,
                "max_score": len(dimensions),
                "critical_dimensions_passed": critical_pass,
                "accepted_for_reasoning_sft": accepted,
                "failures": failures,
            }
        )
        packet.append(
            {
                "row_id": row_id,
                "dataset": item["dataset"],
                "public_pre_action_state": state,
                "generated_rationale": reasoning,
                "generated_tool_call": {"name": name, "arguments": arguments},
                "teacher_target_included": False,
            }
        )
    accepted_count = sum(item["accepted_for_reasoning_sft"] for item in details)
    by_dataset = {}
    for dataset in sorted({item["dataset"] for item in details}):
        subset = [item for item in details if item["dataset"] == dataset]
        by_dataset[dataset] = {
            "rows": len(subset),
            "accepted": sum(item["accepted_for_reasoning_sft"] for item in subset),
            "mean_score": round(sum(item["score"] for item in subset) / len(subset), 3),
        }
    result = {
        "audit": "target_hidden_generated_reasoning_quality",
        "reviewer_type": "reproducible_structural_semantic_developer_audit",
        "independent_human_review": False,
        "teacher_action_read_by_auditor": False,
        "rubric": {
            "dimensions": dimensions,
            "critical_dimensions": sorted(critical),
            "acceptance_rule": "all critical dimensions pass and total score >= 6/7",
        },
        "summary": {
            "rows": len(details),
            "accepted": accepted_count,
            "rejected": len(details) - accepted_count,
            "acceptance_rate": round(accepted_count / len(details), 4),
            "mean_score": round(sum(item["score"] for item in details) / len(details), 3),
            "by_dataset": by_dataset,
        },
        "details": details,
    }
    return result, packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--branch-replay", type=Path, default=DEFAULT_BRANCH_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    args = parser.parse_args()
    result, packet = audit(args.report, args.sft, args.branch_replay)
    write_json(args.output, result)
    write_jsonl(args.packet, packet)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
