"""Programmatic policy-view arms for the control-plane falsification study."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


ARMS = ("full", "structure_only", "evidence_only", "shuffled_evidence")

_SEMANTIC_KEYS = {
    "value",
    "observation",
    "result",
    "visible_rationale",
    "rationale",
    "reason",
    "description",
    "definition",
    "comparison_statement",
    "action_description",
    "preferred_video",
    "direction",
    "confidence",
    "score",
    "measurement",
    "measurements",
    "interval",
    "interval_s",
    "video_a_interval_s",
    "video_b_interval_s",
    "affected_source_interval_s",
    "affected_suspect_interval_s",
}

_REDUCED_CONTROL_KEYS = {
    "state_action_affordances",
    "action_constraints",
    "execution_frontier",
    "evidence_adjudication_frontier",
    "evidence_assessment_frontier",
    "repair_restoration_candidates",
    "executable_binding_actions",
    "executable_decision_actions",
    "executable_evidence_assessment_actions",
    "executable_adjudication_actions",
    "binding_candidates",
    "meaningful_decision_transitions",
    "recommended_tool",
    "required_tool",
    "native_action_templates",
    "native_action_templates_reference",
}


def _structure(value: Any, *, key: str = "") -> Any:
    if key in _SEMANTIC_KEYS:
        if isinstance(value, list):
            return {"redacted": True, "type": "array", "length": len(value)}
        if isinstance(value, dict):
            return {"redacted": True, "type": "object", "keys": sorted(value)}
        return {"redacted": True, "type": type(value).__name__}
    if isinstance(value, dict):
        return {str(item_key): _structure(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_structure(item) for item in value]
    return copy.deepcopy(value)


def _reduced_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _reduced_control(item)
            for key, item in value.items()
            if str(key) not in _REDUCED_CONTROL_KEYS
        }
    if isinstance(value, list):
        return [_reduced_control(item) for item in value]
    return copy.deepcopy(value)


def _tool_without_description(tool: dict[str, Any]) -> dict[str, Any]:
    revised = copy.deepcopy(tool)
    function = revised.get("function") or {}
    function.pop("description", None)
    return revised


def evidence_signature(atom: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(atom.get("kind") or ""),
        tuple(sorted(str(item) for item in atom.get("supports_fields") or [])),
    )


def build_evidence_donor_bank(rows: list[dict[str, Any]]) -> dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Index student-visible atoms only; oracle/verifier fields are never read."""

    bank: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        dataset = str(metadata.get("dataset") or "")
        case_id = str(metadata.get("case_id") or metadata.get("contract_id") or metadata.get("row_id") or "")
        messages = row.get("messages") or []
        if len(messages) < 2:
            continue
        try:
            import json

            payload = json.loads(str(messages[1].get("content") or "{}"))
        except (TypeError, ValueError):
            continue
        state = payload.get("current_harness_state") or {}
        for atom in (state.get("evidence_store") or {}).values():
            if not isinstance(atom, dict) or not isinstance(atom.get("observation"), dict):
                continue
            bank[(dataset, *evidence_signature(atom))].append(
                {
                    "case_id": case_id,
                    "observation": copy.deepcopy(atom["observation"]),
                    "provenance": copy.deepcopy(atom.get("provenance") or {}),
                }
            )
    for candidates in bank.values():
        candidates.sort(key=lambda item: (item["case_id"], repr(item["observation"])))
    return dict(bank)


def _donor_for_atom(
    atom: dict[str, Any],
    *,
    dataset: str,
    case_id: str,
    donor_bank: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    key = (dataset, *evidence_signature(atom))
    return next(
        (candidate for candidate in donor_bank.get(key, []) if candidate["case_id"] != case_id),
        None,
    )


def _swap_evidence_atoms(
    value: Any,
    *,
    dataset: str,
    case_id: str,
    donor_bank: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]],
) -> tuple[Any, int]:
    swaps = 0
    if isinstance(value, dict):
        revised = copy.deepcopy(value)
        if (
            "evidence_id" in revised
            and isinstance(revised.get("observation"), dict)
            and "kind" in revised
        ):
            donor = _donor_for_atom(
                revised, dataset=dataset, case_id=case_id, donor_bank=donor_bank
            )
            if donor is not None:
                revised["observation"] = copy.deepcopy(donor["observation"])
                if "provenance" in revised:
                    revised["provenance"] = {
                        "shuffled_from_different_case": True,
                        "original_interface_keys": sorted((revised.get("provenance") or {}).keys()),
                    }
                revised["semantic_content_shuffled"] = True
                swaps += 1
        for key, item in list(revised.items()):
            if key in {"observation", "provenance"} and revised.get("semantic_content_shuffled"):
                continue
            revised[key], child_swaps = _swap_evidence_atoms(
                item, dataset=dataset, case_id=case_id, donor_bank=donor_bank
            )
            swaps += child_swaps
        return revised, swaps
    if isinstance(value, list):
        revised_list = []
        for item in value:
            revised_item, child_swaps = _swap_evidence_atoms(
                item, dataset=dataset, case_id=case_id, donor_bank=donor_bank
            )
            revised_list.append(revised_item)
            swaps += child_swaps
        return revised_list, swaps
    return copy.deepcopy(value), 0


def apply_policy_view_arm(
    arm: str,
    *,
    dataset: str,
    case_id: str,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    task: dict[str, Any],
    latest: dict[str, Any],
    tools: list[dict[str, Any]],
    donor_bank: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown falsification arm: {arm}")
    if arm == "full":
        return {
            "state": copy.deepcopy(state),
            "history": copy.deepcopy(history),
            "task": copy.deepcopy(task),
            "latest": copy.deepcopy(latest),
            "tools": copy.deepcopy(tools),
            "evidence_swaps": 0,
        }
    if arm == "structure_only":
        return {
            "state": _structure(state),
            "history": _structure(history),
            "task": _structure(task),
            "latest": _structure(latest),
            "tools": [_tool_without_description(tool) for tool in tools],
            "evidence_swaps": 0,
        }
    if arm == "evidence_only":
        return {
            "state": _reduced_control(state),
            "history": _reduced_control(history),
            "task": _reduced_control(task),
            "latest": _reduced_control(latest),
            "tools": copy.deepcopy(tools),
            "evidence_swaps": 0,
        }
    if donor_bank is None:
        raise ValueError("shuffled_evidence requires a donor bank")
    state_view, state_swaps = _swap_evidence_atoms(
        state, dataset=dataset, case_id=case_id, donor_bank=donor_bank
    )
    history_view, history_swaps = _swap_evidence_atoms(
        history, dataset=dataset, case_id=case_id, donor_bank=donor_bank
    )
    latest_view, latest_swaps = _swap_evidence_atoms(
        latest, dataset=dataset, case_id=case_id, donor_bank=donor_bank
    )
    return {
        "state": state_view,
        "history": history_view,
        "task": copy.deepcopy(task),
        "latest": latest_view,
        "tools": copy.deepcopy(tools),
        "evidence_swaps": state_swaps + history_swaps + latest_swaps,
    }

