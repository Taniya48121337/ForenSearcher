"""Mix protected-case-disjoint replacement-route examples with queued evidence assessment SFT."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from run_student_closed_loop_canary import (
    PROGRESSIVE_EVIDENCE_SYSTEM,
    _adjudication_reasoning_frontier,
    _claim_value_conflict_fields,
    _evidence_assessment_reasoning_frontier,
    _post_meta_policy_view,
    _preexecution_reasoning_problems,
)


ARM = "decision_sufficient_state_v6_executable_affordances"
DEFAULT_QUEUED_ROOT = (
    ROOT / "outputs/student_external_contract_v1/progressive_evidence_assessment_warmup_v2_queued"
)
DEFAULT_ROUTE_ROOT = ROOT / "outputs/student_external_contract_v1/repair_policy_increment_v1"
DEFAULT_ADJUDICATION_ROOT = (
    ROOT / "outputs/student_external_contract_v1/evidence_adjudication_warmup_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/replacement_lineage_plus_progressive_curriculum_v1"
)
PROTECTED_CASES = {"viddiff-surgery_16", "viddiff-surgery_26"}
ROUTE_ROW_IDS = {
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:3",
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:4",
    "VidDiffBench:viddiff-surgery_117:student-closed-loop:repair:3",
    "VidDiffBench:viddiff-surgery_117:student-closed-loop:repair:4",
}
MULTIATOM_ADJUDICATION_ROW_IDS = {
    "VidDiffBench:viddiff-surgery_145:adjudication_warmup:repair:1",
}

MEASUREMENT_PURPOSES = {
    "compare_action_refined_symmetric": (
        "it tests whether predicate-relevant local behavior remains stable in both A/B orders"
    ),
    "compare_uniform_timeline": (
        "it checks what direction coarse whole-timeline coverage independently supports"
    ),
    "compare_global_motion_symmetric": (
        "it tests whether whole-video motion supports the same direction under both A/B role orders"
    ),
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _case_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("case_id") or "")


def _row_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("row_id") or "")


def _strengthen_route_reasoning(
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = row.get("messages") or []
    user = next((item for item in reversed(messages) if item.get("role") == "user"), None)
    assistant = next(
        (item for item in reversed(messages) if item.get("role") == "assistant"), None
    )
    if user is None or assistant is None:
        raise ValueError(f"replacement route lacks user/assistant messages: {_row_id(row)}")
    payload = json.loads(str(user.get("content") or "{}"))
    state = payload.get("current_harness_state") or {}
    latest = payload.get("latest_environment_observation") or {}
    created_ids = [str(item) for item in latest.get("created_evidence_ids") or []]
    dropped_ids = [str(item) for item in state.get("dropped_evidence_ids") or []]
    calls = assistant.get("tool_calls") or []
    if len(created_ids) != 1 or not dropped_ids or len(calls) != 1:
        raise ValueError(
            f"replacement route lacks one-new-versus-dropped identity contrast: {_row_id(row)}"
        )
    function = calls[0].get("function") or {}
    tool_name = str(function.get("name") or "")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if tool_name not in MEASUREMENT_PURPOSES:
        raise ValueError(f"unsupported replacement measurement tool: {tool_name}")
    new_id = created_ids[0]
    old_id = dropped_ids[0]
    new_atom = (state.get("evidence_store") or {}).get(new_id) or {}
    if new_atom.get("active") is not True or new_id in dropped_ids:
        raise ValueError(f"latest replacement atom is not unambiguously active: {new_id}")
    observation = new_atom.get("observation") or {}
    method = observation.get("measurement_method")
    preferred = observation.get("preferred_video")
    order = observation.get("order_consistent")
    order_text = "unobserved" if order is None else str(order).lower()
    active_directions = {
        (atom.get("observation") or {}).get("preferred_video")
        for atom in (state.get("evidence_store") or {}).values()
        if atom.get("active") is True
        and (atom.get("observation") or {}).get("measurement_method")
    }
    gap = (
        "active measurement candidates still disagree on direction"
        if len(active_directions) > 1
        else "the executable multi-method evidence route remains incomplete"
    )
    eligible = [str(item) for item in state.get("eligible_tools") or []]
    choice = "best next eligible action" if len(eligible) > 1 else "required next eligible action"
    reasoning = (
        f"The latest observation created active EvidenceAtom `{new_id}`, which reports "
        f"`preferred_video={preferred}`, `order_consistent={order_text}`, and "
        f"`measurement_method={method}`. `{new_id}` is ID-distinct from dropped tombstone "
        f"`{old_id}`; dropped status belongs only to `{old_id}` and does not transfer. "
        f"The blocking gap is that {gap} while the comparison fields remain open, so a "
        "complementary observation is needed before adjudication or binding. "
        f"`{tool_name}` is the {choice} because {MEASUREMENT_PURPOSES[tool_name]}, without "
        "reusing the tombstone or changing supported identity fields."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        tool_name,
        arguments,
        meta_visible=False,
        visible_state={**state, "latest_environment_observation": latest},
    )
    if problems:
        raise ValueError(f"replacement route reasoning failed current rubric: {problems}")
    assistant["content"] = reasoning
    metadata = row.setdefault("metadata", {})
    metadata["identity_contrast_reasoning"] = True
    metadata["reasoning_recompiled_from_student_visible_state"] = True
    return row, {
        "row_id": _row_id(row),
        "new_active_evidence_id": new_id,
        "dropped_tombstone_id": old_id,
        "target_tool": tool_name,
        "eligible_tool_count": len(eligible),
        "reasoning_contract_problems": problems,
    }


def _progressive_action(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    messages = row.get("messages") or []
    user = next((item for item in reversed(messages) if item.get("role") == "user"), None)
    assistant = next(
        (item for item in reversed(messages) if item.get("role") == "assistant"), None
    )
    if user is None or assistant is None:
        raise ValueError(f"progressive row lacks user/assistant messages: {_row_id(row)}")
    payload = json.loads(str(user.get("content") or "{}"))
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"progressive row needs exactly one tool call: {_row_id(row)}")
    function = calls[0].get("function") or {}
    name = str(function.get("name") or "")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return payload, assistant, name, arguments


def _defect_phrases(
    state: dict[str, Any], arguments: dict[str, Any], evidence_id: str, codes: list[str]
) -> list[str]:
    observation = ((state.get("evidence_store") or {}).get(evidence_id) or {}).get(
        "observation"
    ) or {}
    phrases: list[str] = []
    if "claim_value_conflict" in codes:
        fields = _claim_value_conflict_fields(
            state,
            str(arguments.get("claim_id") or ""),
            evidence_id,
            [str(field) for field in arguments.get("target_fields") or []],
        )
        if not fields:
            raise ValueError(f"claim_value_conflict lacks a mismatched field: {evidence_id}")
        phrases.append("claim mismatch in " + ", ".join(f"`{field}`" for field in fields))
    if "order_inconsistent" in codes:
        phrases.append("`order_consistent=false`")
    if "lower_confidence" in codes:
        phrases.append(f"lower confidence={observation.get('confidence')}")
    if "abstention_conflicts_with_claim" in codes:
        phrases.append("an abstaining `preferred_video=c` that conflicts with the claim")
    if "redundant_same_conclusion" in codes:
        phrases.append("redundant same-conclusion support")
    return phrases


def _strengthen_progressive_reasoning(
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, assistant, name, arguments = _progressive_action(row)
    state = payload.get("current_harness_state") or {}
    if name == "adjudicate_evidence" and "evidence_assessments" not in state:
        claim_id = str(arguments.get("claim_id") or "")
        assessments: dict[str, Any] = {}
        for evidence_id in arguments.get("selected_evidence_ids") or []:
            assessments[str(evidence_id)] = {
                "assessment_id": f"ASSESS-{len(assessments) + 1:03d}",
                "disposition": "retain",
                "defect_codes": [],
            }
        for item in arguments.get("rejected_evidence") or []:
            evidence_id = str(item.get("evidence_id") or "")
            assessments[evidence_id] = {
                "assessment_id": f"ASSESS-{len(assessments) + 1:03d}",
                "disposition": "reject",
                "defect_codes": list(item.get("defect_codes") or []),
            }
        state["evidence_assessments"] = {claim_id: assessments}
        state["evidence_assessment"] = {
            "required": True,
            "pending_evidence_ids": [],
            "current_candidate_id": None,
            "deferred_candidate_ids": [],
        }
    compact_state, compact_history = _post_meta_policy_view(
        state, list(payload.get("past_execution_history") or [])
    )
    state = compact_state
    payload["current_harness_state"] = compact_state
    payload["past_execution_history"] = compact_history
    action = {"tool_name": name, "arguments": arguments}
    store = state.get("evidence_store") or {}
    if name == "assess_evidence_candidate":
        evidence_id = str(arguments.get("evidence_id") or "")
        observation = (store.get(evidence_id) or {}).get("observation") or {}
        preferred = observation.get("preferred_video")
        order = observation.get("order_consistent")
        order_text = "unobserved" if order is None else str(order).lower()
        disposition = str(arguments.get("disposition") or "")
        first = (
            f"I select `assess_evidence_candidate` because the remaining gap is the pending "
            f"ledger entry for `{evidence_id}`, and this atomic action changes no claim value."
        )
        second = (
            f"The atom reports `preferred_video={preferred}`, `order_consistent` is "
            f"{order_text}, and `measurement_method={observation.get('measurement_method')}`."
        )
        if disposition == "retain":
            tail = (
                "I retain this candidate as usable support because its visible target fields "
                "match the claim; any unobserved order status remains uncertainty, not a defect."
            )
        else:
            codes = [str(code) for code in arguments.get("defect_codes") or []]
            phrases = _defect_phrases(state, arguments, evidence_id, codes)
            tail = (
                "I reject this candidate because "
                + "; ".join(phrases)
                + ", which records its visible defect before final adjudication."
            )
        reasoning = " ".join((first, second, tail))
        payload["evidence_assessment_frontier"] = _evidence_assessment_reasoning_frontier(
            state, [action]
        )
    elif name == "adjudicate_evidence":
        selected_ids = [str(item) for item in arguments.get("selected_evidence_ids") or []]
        rejected = list(arguments.get("rejected_evidence") or [])
        first = (
            "I select `adjudicate_evidence` because the remaining gap is an unresolved "
            f"{len(selected_ids) + len(rejected)}-atom partition for the reopened fields."
        )
        selected_clauses = []
        handles = {
            evidence_id: f"@E{index}"
            for index, evidence_id in enumerate(
                [
                    *selected_ids,
                    *(str(item.get("evidence_id") or "") for item in rejected),
                ],
                start=1,
            )
        }
        for evidence_id in selected_ids:
            observation = (store.get(evidence_id) or {}).get("observation") or {}
            selected_clauses.append(
                f"`{handles[evidence_id]}` (`preferred_video={observation.get('preferred_video')}`, "
                f"`measurement_method={observation.get('measurement_method')}`)"
            )
        second = "I retain " + ", ".join(selected_clauses) + " because its visible fields fit the claim."
        third = (
            f"The completed per-candidate assessment ledger already records the visible defects "
            f"for all {len(rejected)} rejected alternatives; the native call preserves their exact "
            "IDs and defect codes without repeating those explanations in prose."
        )
        reasoning = " ".join((first, second, third))
        payload["evidence_adjudication_frontier"] = _adjudication_reasoning_frontier(
            state, [action]
        )
    else:
        raise ValueError(f"unsupported progressive target tool: {name}")

    problems = _preexecution_reasoning_problems(
        reasoning,
        name,
        arguments,
        meta_visible=False,
        visible_state=state,
    )
    if problems:
        raise ValueError(
            f"progressive reasoning failed current rubric for {_row_id(row)}: {problems}"
        )
    messages = row.get("messages") or []
    system = next((item for item in messages if item.get("role") == "system"), None)
    user = next((item for item in reversed(messages) if item.get("role") == "user"), None)
    if system is None or user is None:
        raise ValueError(f"progressive row lacks system/user messages: {_row_id(row)}")
    system["content"] = PROGRESSIVE_EVIDENCE_SYSTEM
    user["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assistant["content"] = reasoning
    metadata = row.setdefault("metadata", {})
    metadata["field_grounded_reasoning"] = True
    metadata["reasoning_recompiled_from_student_visible_state"] = True
    return row, {
        "row_id": _row_id(row),
        "target_tool": name,
        "competitor_count": len(
            arguments.get("selected_evidence_ids") or []
        )
        + len(arguments.get("rejected_evidence") or []),
        "reasoning_contract_problems": problems,
    }
def compile_curriculum(
    queued_root: Path,
    route_root: Path,
    output: Path,
    *,
    adjudication_root: Path | None = None,
    strengthen_progressive: bool = False,
) -> dict[str, Any]:
    queued_path = queued_root / ARM / "train" / "sft_all.jsonl"
    route_path = route_root / ARM / "train" / "sft_all.jsonl"
    queued = _jsonl(queued_path)
    if not queued:
        raise ValueError("progressive queue is empty")
    queued_report_path = queued_root / "compile_report.json"
    if queued_report_path.exists():
        queued_report = json.loads(queued_report_path.read_text(encoding="utf-8"))
        if not bool(
            (queued_report.get("quality_report") or {}).get(
                "all_applicable_checks_passed"
            )
        ):
            raise ValueError("progressive queue quality checks did not pass")
    route_candidates = _jsonl(route_path)
    route = [row for row in route_candidates if _row_id(row) in ROUTE_ROW_IDS]
    found = {_row_id(row) for row in route}
    if found != ROUTE_ROW_IDS:
        raise ValueError(f"missing replacement-route rows: {sorted(ROUTE_ROW_IDS - found)}")
    multiatom: list[dict[str, Any]] = []
    if adjudication_root is not None:
        candidates = _jsonl(adjudication_root / ARM / "train" / "sft_all.jsonl")
        multiatom = [
            row for row in candidates if _row_id(row) in MULTIATOM_ADJUDICATION_ROW_IDS
        ]
        multiatom_found = {_row_id(row) for row in multiatom}
        if multiatom_found != MULTIATOM_ADJUDICATION_ROW_IDS:
            raise ValueError(
                "missing multi-atom adjudication rows: "
                f"{sorted(MULTIATOM_ADJUDICATION_ROW_IDS - multiatom_found)}"
            )

    combined: list[dict[str, Any]] = []
    route_reasoning_audits: list[dict[str, Any]] = []
    progressive_reasoning_audits: list[dict[str, Any]] = []
    sources = [("progressive_queue", queued), ("replacement_route", route)]
    if multiatom:
        sources.append(("multiatom_adjudication", multiatom))
    for source_name, rows in sources:
        for source in rows:
            row = copy.deepcopy(source)
            case_id = _case_id(row)
            serialized = json.dumps(row, ensure_ascii=False)
            if case_id in PROTECTED_CASES or any(case in serialized for case in PROTECTED_CASES):
                raise ValueError(f"protected case leaked into curriculum: {case_id or _row_id(row)}")
            metadata = row.setdefault("metadata", {})
            metadata["curriculum_source"] = source_name
            metadata["protected_case_disjoint"] = True
            if source_name == "replacement_route":
                row, route_audit = _strengthen_route_reasoning(row)
                route_reasoning_audits.append(route_audit)
            elif strengthen_progressive:
                row, progressive_audit = _strengthen_progressive_reasoning(row)
                progressive_reasoning_audits.append(progressive_audit)
            combined.append(row)

    row_ids = [_row_id(row) for row in combined]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate row_id in combined curriculum")

    train_dir = output / ARM / "train"
    write_jsonl(train_dir / "sft_all.jsonl", combined)
    report = {
        "artifact": output.name,
        "rows": len(combined),
        "progressive_queue_rows": len(queued),
        "replacement_route_rows": len(route),
        "multiatom_adjudication_rows": len(multiatom),
        "replacement_route_row_ids": sorted(found),
        "protected_cases_excluded": sorted(PROTECTED_CASES),
        "teacher_actions_used": False,
        "oracle_or_verifier_used": False,
        "external_transfer_performed": False,
        "route_identity_contrast_rows": len(route_reasoning_audits),
        "route_reasoning_all_clean": all(
            not item["reasoning_contract_problems"] for item in route_reasoning_audits
        ),
        "route_reasoning_audits": route_reasoning_audits,
        "field_grounded_progressive_rows": len(progressive_reasoning_audits),
        "progressive_reasoning_all_clean": all(
            not item["reasoning_contract_problems"]
            for item in progressive_reasoning_audits
        ),
        "progressive_reasoning_audits": progressive_reasoning_audits,
        "target_tool_counts": {},
    }
    for row in combined:
        tool = str((row.get("metadata") or {}).get("target_tool") or "")
        report["target_tool_counts"][tool] = report["target_tool_counts"].get(tool, 0) + 1
    write_json(output / "compile_report.json", report)
    write_json(
        output / ARM / "manifest.json",
        {
            "arm": ARM,
            "splits": {"train": {"rows": len(combined)}},
            "source": "accepted_student_visible_rows_only",
            "protected_case_disjoint": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queued-root", type=Path, default=DEFAULT_QUEUED_ROOT)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    parser.add_argument(
        "--adjudication-root", type=Path, default=DEFAULT_ADJUDICATION_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            compile_curriculum(
                args.queued_root,
                args.route_root,
                args.output,
                adjudication_root=args.adjudication_root,
                strengthen_progressive=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
