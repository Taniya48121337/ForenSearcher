"""Execute generated harness-state actions from their public pre-action snapshots."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import (
    _arguments_validate_visible_schema,
    _parse_generated_call,
)
from interactive_harness import (
    CandidateClaim,
    CaseGoal,
    ClaimFieldState,
    EvidenceAdjudication,
    EvidenceAtom,
    EvidenceCandidateAssessment,
    EvidenceChallenge,
    FieldBinding,
    TaskContract,
    VideoForensicHarness,
)
from llm_teacher_rollout import _normalize_harness_params
from pipeline import ROOT, write_json


STATE_TOOLS = {
    "create_candidate_claim",
    "challenge_evidence",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
    "stop",
}
DEFAULT_REPORT = ROOT / "outputs/student_external_contract_v1/eval_adapter120_v4_4k_validation64_gen64_512.json"
DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v4_external_tool_contract"
    / "validation/sft_all.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/student_external_contract_v1/shadow_execution_validation64.json"


class NoExternalExecution:
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"external tool is outside state-only shadow execution: {name}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][1]["content"])


def _tool_registry(row: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in row.get("tools") or []:
        function = tool["function"]
        name = str(function["name"])
        result.append(
            {
                "name": name,
                "domain": (
                    "harness_state"
                    if name in {
                        "create_candidate_claim",
                        "challenge_evidence",
                        "assess_evidence_candidate",
                        "adjudicate_evidence",
                        "bind_evidence",
                        "set_field_decision",
                    }
                    else "harness_terminal"
                    if name == "stop"
                    else "external_media"
                ),
                "description": function.get("description", ""),
            }
        )
    return result


def _restore_harness(row: dict[str, Any]) -> VideoForensicHarness:
    payload = _payload(row)
    state = payload["current_harness_state"]
    task = payload["task"]
    harness = VideoForensicHarness(
        TaskContract(**state["task_contract"]),
        NoExternalExecution(),
        claim_schema=task["claim_schema"],
        tool_registry=_tool_registry(row),
        target_fields=task["claim_schema"].get("target_fields"),
        invariant_fields=task["claim_schema"].get("invariant_fields"),
    )
    harness.goals = {
        item["goal_id"]: CaseGoal(**copy.deepcopy(item)) for item in state.get("case_goals", [])
    }
    harness.evidence_store = {
        evidence_id: EvidenceAtom(**copy.deepcopy(atom))
        for evidence_id, atom in state.get("evidence_store", {}).items()
    }
    requirement = (state.get("evidence_adjudication") or {}).get("requirement") or {}
    default_target_fields = [str(item) for item in requirement.get("target_fields") or []]
    default_considered_ids = [
        str(item) for item in requirement.get("competing_evidence_ids") or []
    ]
    harness.evidence_assessments = {}
    for claim_id, assessments in state.get("evidence_assessments", {}).items():
        restored: dict[str, EvidenceCandidateAssessment] = {}
        for evidence_id, assessment in assessments.items():
            values = copy.deepcopy(assessment)
            values.setdefault("claim_id", str(claim_id))
            values.setdefault("target_fields", copy.deepcopy(default_target_fields))
            values.setdefault("evidence_id", str(evidence_id))
            values.setdefault(
                "considered_evidence_ids", copy.deepcopy(default_considered_ids)
            )
            restored[str(evidence_id)] = EvidenceCandidateAssessment(**values)
        harness.evidence_assessments[str(claim_id)] = restored
    harness.evidence_adjudications = {}
    for claim_id, adjudication in state.get("evidence_adjudications", {}).items():
        values = copy.deepcopy(adjudication)
        values.setdefault("claim_id", str(claim_id))
        values.setdefault("target_fields", copy.deepcopy(default_target_fields))
        values.setdefault(
            "considered_evidence_ids", copy.deepcopy(default_considered_ids)
        )
        harness.evidence_adjudications[str(claim_id)] = EvidenceAdjudication(**values)
    harness.evidence_challenges = []
    for challenge in state.get("evidence_challenges", []):
        values = copy.deepcopy(challenge)
        values.setdefault("target_fields", [])
        harness.evidence_challenges.append(EvidenceChallenge(**values))
    harness.claims = {}
    for claim in state.get("candidate_claims", []):
        harness.claims[str(claim["claim_id"])] = CandidateClaim(
            claim_id=str(claim["claim_id"]),
            schema_id=str(
                claim.get("schema_id")
                or state["task_contract"].get("claim_schema_id")
                or task["claim_schema"].get("schema_id")
            ),
            fields={
                name: ClaimFieldState(**copy.deepcopy(field_state))
                for name, field_state in claim.get("fields", {}).items()
            },
        )
    harness.bindings = {
        claim_id: {
            field_name: FieldBinding(**copy.deepcopy(binding))
            for field_name, binding in fields.items()
        }
        for claim_id, fields in state.get("field_bindings", {}).items()
    }
    for claim_id in harness.claims:
        harness.bindings.setdefault(claim_id, {})
    harness.dropped_evidence_ids = list(state.get("dropped_evidence_ids", []))
    harness.terminal = bool(state.get("terminal", False))
    harness.stop_decision = copy.deepcopy(state.get("stop_decision"))
    return harness


def _bindable_fields(harness: VideoForensicHarness, claim_id: str) -> set[str]:
    claim = harness.claims.get(claim_id)
    if claim is None:
        return set()
    result: set[str] = set()
    for field_name, field_state in claim.fields.items():
        if field_state.evidence_status == "supported":
            continue
        acceptable = set(harness.claim_schema["fields"][field_name]["acceptable_evidence_kinds"])
        for atom in harness.evidence_store.values():
            if (
                atom.active
                and field_name in atom.supports_fields
                and atom.kind in acceptable
                and harness._atom_supports_claim_value(field_name, field_state.value, atom)
            ):
                result.add(field_name)
                break
    return result


def _claim_grounded(harness: VideoForensicHarness, arguments: dict[str, Any]) -> bool:
    values = arguments.get("field_values")
    if not isinstance(values, dict):
        return False
    for field_name, value in values.items():
        acceptable = set(harness.claim_schema["fields"][field_name]["acceptable_evidence_kinds"])
        if not any(
            atom.active
            and field_name in atom.supports_fields
            and atom.kind in acceptable
            and harness._atom_supports_claim_value(field_name, value, atom)
            for atom in harness.evidence_store.values()
        ):
            return False
    return True


def audit(report_path: Path, sft_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_by_id = {str(row["metadata"]["row_id"]): row for row in _jsonl(sft_path)}
    details: list[dict[str, Any]] = []
    generated_rows = [item for item in report["rows"] if item.get("generated", True)]
    for generated in generated_rows:
        row = rows_by_id[str(generated["row_id"])]
        reference_name = str(row["messages"][-1]["tool_calls"][0]["function"]["name"])
        name, arguments = _parse_generated_call(str(generated.get("generated_text", "")))
        payload = _payload(row)
        eligible = set(payload["current_harness_state"].get("eligible_tools", []))
        schema_valid = _arguments_validate_visible_schema(row, name, arguments)
        is_state_row = reference_name in STATE_TOOLS
        generated_is_state_tool = name in STATE_TOOLS if name else False
        item: dict[str, Any] = {
            "row_id": generated["row_id"],
            "dataset": row["metadata"]["dataset"],
            "source_kind": row["metadata"]["source_kind"],
            "reference_tool": reference_name,
            "generated_tool": name,
            "parseable": name is not None and arguments is not None,
            "schema_valid": schema_valid,
            "prestate_eligible": bool(name and name in eligible),
            "evaluation_mode": (
                "stateful_harness_shadow_execution" if is_state_row else "external_contract_preflight"
            ),
        }
        if not (
            is_state_row
            and generated_is_state_tool
            and schema_valid
            and name in eligible
            and arguments is not None
        ):
            item.update(
                {
                    "shadow_execution_attempted": False,
                    "shadow_status": None,
                    "state_changed": None,
                    "grounded_or_maximal": None,
                    "stateful_quality_pass": None,
                }
            )
            details.append(item)
            continue
        harness = _restore_harness(row)
        before = harness.render_state()
        bindable_before = (
            _bindable_fields(harness, str(arguments.get("claim_id"))) if name == "bind_evidence" else set()
        )
        grounded_before = _claim_grounded(harness, arguments) if name == "create_candidate_claim" else None
        try:
            result = harness.execute_tool(
                name,
                _normalize_harness_params(name, arguments),
                "shadow-generated-call",
            )
            error = None
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            result = {"status": "exception"}
            error = f"{type(exc).__name__}:{exc}"
        after = harness.render_state()
        status = str(result.get("status"))
        state_changed = before != after
        if name == "bind_evidence":
            requested = {
                str(entry.get("field")) for entry in arguments.get("binding_items", []) if isinstance(entry, dict)
            }
            grounded_or_maximal = bindable_before.issubset(requested) and status == "accepted"
        elif name == "create_candidate_claim":
            grounded_or_maximal = bool(grounded_before)
        else:
            grounded_or_maximal = True
        accepted = status in {"accepted", "executed"}
        meaningful = accepted and state_changed
        quality_pass = accepted and meaningful and grounded_or_maximal
        item.update(
            {
                "shadow_execution_attempted": True,
                "shadow_status": status,
                "shadow_error": error,
                "state_changed": state_changed,
                "grounded_or_maximal": grounded_or_maximal,
                "bindable_fields_before": sorted(bindable_before),
                "stateful_quality_pass": quality_pass,
                "open_obligations_before": before.get("open_high_priority_obligations"),
                "open_obligations_after": after.get("open_high_priority_obligations"),
            }
        )
        details.append(item)

    stateful = [item for item in details if item["evaluation_mode"] == "stateful_harness_shadow_execution"]
    external = [item for item in details if item["evaluation_mode"] == "external_contract_preflight"]
    return {
        "audit_type": "public_prestate_generated_action_shadow_audit",
        "generation_report": str(report_path),
        "rows": len(details),
        "stateful_actions": {
            "rows": len(stateful),
            "shadow_execution_attempted": sum(item["shadow_execution_attempted"] for item in stateful),
            "accepted_and_meaningful_and_grounded_or_maximal": sum(
                item.get("stateful_quality_pass") is True for item in stateful
            ),
            "quality_rate": round(
                sum(item.get("stateful_quality_pass") is True for item in stateful) / max(len(stateful), 1), 4
            ),
        },
        "external_actions": {
            "rows": len(external),
            "parse_schema_eligible": sum(
                item["parseable"] and item["schema_valid"] and item["prestate_eligible"] for item in external
            ),
            "note": "External media/feature tools were not re-executed in this state-only audit.",
        },
        "per_dataset_stateful": {
            dataset: {
                "rows": sum(item["dataset"] == dataset for item in stateful),
                "quality_pass": sum(
                    item["dataset"] == dataset and item.get("stateful_quality_pass") is True
                    for item in stateful
                ),
            }
            for dataset in sorted({str(item["dataset"]) for item in stateful})
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.report, args.sft)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("rows", "stateful_actions", "external_actions", "per_dataset_stateful")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
