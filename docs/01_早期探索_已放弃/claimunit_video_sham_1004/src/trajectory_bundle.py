"""Build ECSO-shaped clean and evidence-dropout repair trajectories."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "source_id",
    "claim_family",
    "manipulation_type",
    "direction",
    "comparison_scope_s",
    "retained_source_interval_s",
    "omitted_source_intervals_s",
    "estimated_removed_time_s",
]

TARGET_FIELDS = [
    "claim_family",
    "manipulation_type",
    "direction",
    "omitted_source_intervals_s",
    "estimated_removed_time_s",
]

INVARIANT_FIELDS = [
    "source_id",
    "comparison_scope_s",
    "retained_source_interval_s",
]


def _binding(field: str, value: Any, evidence_ids: list[str], status: str = "verified") -> dict[str, Any]:
    return {"field": field, "value": value, "evidence_ids": evidence_ids, "status": status}


def _bindings(claim: dict[str, Any], omission_evidence_id: str) -> list[dict[str, Any]]:
    return [
        _binding("source_id", claim["source_id"], ["EA-ALIGN-001"]),
        _binding("claim_family", claim["claim_family"], ["EA-ALIGN-001", omission_evidence_id]),
        _binding("manipulation_type", claim["manipulation_type"], [omission_evidence_id]),
        _binding("direction", claim["direction"], ["EA-ALIGN-001", omission_evidence_id]),
        _binding("comparison_scope_s", claim["comparison_scope_s"], ["EA-SCOPE-001"]),
        _binding("retained_source_interval_s", claim["retained_source_interval_s"], ["EA-ALIGN-001"]),
        _binding("omitted_source_intervals_s", claim["omitted_source_intervals_s"], ["EA-SCOPE-001", omission_evidence_id]),
        _binding("estimated_removed_time_s", claim["estimated_removed_time_s"], ["EA-META-001", omission_evidence_id]),
    ]


def _event(
    step: int,
    action: str,
    kind: str,
    *,
    consumes: list[str] | None = None,
    produces: list[str] | None = None,
    state_delta: dict[str, Any] | None = None,
    progress: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "action": action,
        "kind": kind,
        "consumes": consumes or [],
        "produces": produces or [],
        "state_delta": state_delta or {},
        "obligation_progress": round(progress, 3),
    }


def _provenance_graph(
    clean_events: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    repair_events: list[dict[str, Any]],
    repaired_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    for event in clean_events:
        action_id = f"ACTION-{event['step']:02d}"
        nodes.append({"node_id": action_id, "node_type": "action", "label": event["action"]})
        for evidence_id in event["produces"]:
            edges.append({"from": action_id, "to": evidence_id, "relation": "produces"})
    for event in repair_events:
        action_id = f"REPAIR-ACTION-{event['step']:02d}"
        nodes.append({"node_id": action_id, "node_type": "repair_action", "label": event["action"]})
        for evidence_id in event["produces"]:
            edges.append({"from": action_id, "to": evidence_id, "relation": "produces"})
    for evidence_id in ["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001", "EA-OMIT-001"]:
        nodes.append({"node_id": evidence_id, "node_type": "evidence", "label": evidence_id})
    nodes.append({"node_id": "EA-REPAIR-001", "node_type": "repair_evidence", "label": "EA-REPAIR-001"})
    nodes.extend(
        [
            {"node_id": "OBL-1004-001", "node_type": "obligation", "label": "selective omission obligation"},
            {"node_id": "CL-1004-001", "node_type": "claim", "label": "verified omission claim"},
            {"node_id": "STOP-1004-001", "node_type": "stop", "label": "verified stop"},
        ]
    )
    for binding in bindings:
        field_id = f"FIELD-{binding['field']}"
        nodes.append({"node_id": field_id, "node_type": "claim_field", "label": binding["field"]})
        for evidence_id in binding["evidence_ids"]:
            edges.append({"from": evidence_id, "to": field_id, "relation": "supports"})
        edges.append({"from": field_id, "to": "CL-1004-001", "relation": "composes"})
        edges.append({"from": field_id, "to": "OBL-1004-001", "relation": "satisfies"})
    edges.append({"from": "OBL-1004-001", "to": "STOP-1004-001", "relation": "enables"})
    for binding in repaired_bindings:
        if "EA-REPAIR-001" in binding["evidence_ids"]:
            edges.append({"from": "EA-REPAIR-001", "to": f"FIELD-{binding['field']}", "relation": "repairs_support"})
    return {"graph_type": "FieldProvenanceGraph", "nodes": nodes, "edges": edges}


def build_trajectory_bundle(
    *,
    root: Path,
    manifest: dict[str, Any],
    evidence: list[dict[str, Any]],
    claim: dict[str, Any],
    repair_atom: dict[str, Any],
) -> dict[str, Any]:
    clean_claim = copy.deepcopy(claim)
    clean_claim["evidence_ids"] = ["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001", "EA-OMIT-001"]
    clean_bindings = _bindings(clean_claim, "EA-OMIT-001")
    obligation_initial = {
        "obligation_id": "OBL-1004-001",
        "hypothesis_type": "selective_temporal_omission",
        "priority": "high",
        "required_fields": REQUIRED_FIELDS,
        "status": "open",
        "verified_fields": [],
        "missing_evidence": ["comparison_scope", "source_alignment", "omission_measurement"],
        "claim_id": None,
    }
    clean_events = [
        _event(1, "inspect_media", "tool", produces=["EA-META-001"], state_delta={"available_evidence_added": ["EA-META-001"]}, progress=0.0),
        _event(2, "bind_comparison_scope", "state_edit", consumes=["case_manifest"], produces=["EA-SCOPE-001"], state_delta={"verified_fields_added": ["comparison_scope_s"], "missing_evidence_removed": ["comparison_scope"]}, progress=1 / 8),
        _event(3, "align_visual_timeline", "tool", consumes=["source_media", "edited_media"], produces=["EA-ALIGN-001"], state_delta={"verified_fields_added": ["source_id", "retained_source_interval_s"], "missing_evidence_removed": ["source_alignment"]}, progress=3 / 8),
        _event(4, "measure_source_time_contraction", "tool", consumes=["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001"], produces=["EA-OMIT-001"], state_delta={"verified_fields_added": TARGET_FIELDS, "missing_evidence_removed": ["omission_measurement"], "obligation_status": "evidence_ready"}, progress=1.0),
        _event(5, "propose_claim", "state_edit", consumes=[item["evidence_id"] for item in evidence], produces=["CL-1004-001"], state_delta={"obligation_status": "proposed", "claim_id": "CL-1004-001"}, progress=1.0),
        _event(6, "verify_claim", "verification", consumes=["CL-1004-001", *clean_claim["evidence_ids"]], state_delta={"obligation_status": "verified", "field_bindings_verified": REQUIRED_FIELDS}, progress=1.0),
        _event(7, "stop", "terminal", consumes=["OBL-1004-001", "CL-1004-001"], produces=["STOP-1004-001"], state_delta={"stop_allowed": True, "reason_code": "all_high_priority_obligations_verified"}, progress=1.0),
    ]
    clean = {
        "trajectory_id": "TRJ-1004-CLEAN-001",
        "case_id": manifest["case_id"],
        "trajectory_type": "clean_acquisition",
        "initial_state": {"obligations": [obligation_initial], "claim_ledger": [], "available_evidence_ids": []},
        "events": clean_events,
        "evidence": evidence,
        "final_state": {
            "obligations": [
                {
                    **obligation_initial,
                    "status": "verified",
                    "verified_fields": REQUIRED_FIELDS,
                    "missing_evidence": [],
                    "claim_id": "CL-1004-001",
                }
            ],
            "claim_ledger": [
                {"claim_id": "CL-1004-001", "status": "verified", "claim": clean_claim, "field_bindings": clean_bindings}
            ],
            "available_evidence_ids": [item["evidence_id"] for item in evidence],
            "stop_decision": {"allowed": True, "reason_code": "all_high_priority_obligations_verified", "open_high_priority_obligations": []},
        },
    }

    mutated_evidence = [item for item in evidence if item["evidence_id"] != "EA-OMIT-001"]
    mutated_bindings: list[dict[str, Any]] = []
    for binding in clean_bindings:
        copied = copy.deepcopy(binding)
        if binding["field"] in TARGET_FIELDS:
            copied["evidence_ids"] = []
            copied["status"] = "unsupported"
        mutated_bindings.append(copied)
    repaired_claim = copy.deepcopy(clean_claim)
    repaired_claim["evidence_ids"] = ["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001", "EA-REPAIR-001"]
    repaired_bindings = _bindings(repaired_claim, "EA-REPAIR-001")
    repair_events = [
        _event(1, "inject_evidence_dropout", "environment_mutation", consumes=["EA-OMIT-001"], state_delta={"evidence_removed": ["EA-OMIT-001"], "media_changed": False, "oracle_changed": False, "claim_text_changed": False}, progress=3 / 8),
        _event(2, "audit_field_support", "verification", consumes=["claim_ledger", "available_evidence"], state_delta={"invalidated_fields": TARGET_FIELDS, "preserved_fields": INVARIANT_FIELDS}, progress=3 / 8),
        _event(3, "reopen_affected_obligation", "state_edit", consumes=["OBL-1004-001"], state_delta={"obligation_status": "open", "reopened_fields": TARGET_FIELDS, "preserved_fields": INVARIANT_FIELDS, "missing_evidence_added": ["omission_measurement"]}, progress=3 / 8),
        _event(4, "targeted_alignment_remeasurement", "tool", consumes=["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001"], produces=["EA-REPAIR-001"], state_delta={"available_evidence_added": ["EA-REPAIR-001"]}, progress=3 / 8),
        _event(5, "rebind_affected_fields", "state_edit", consumes=["EA-REPAIR-001"], state_delta={"rebound_fields": TARGET_FIELDS, "preserved_fields": INVARIANT_FIELDS, "obligation_status": "proposed"}, progress=1.0),
        _event(6, "verify_repaired_claim", "verification", consumes=["CL-1004-001", "EA-REPAIR-001"], state_delta={"obligation_status": "verified", "field_bindings_verified": REQUIRED_FIELDS}, progress=1.0),
        _event(7, "stop", "terminal", consumes=["OBL-1004-001", "CL-1004-001"], produces=["STOP-1004-REPAIR-001"], state_delta={"stop_allowed": True, "reason_code": "all_reopened_fields_reverified"}, progress=1.0),
    ]
    repair = {
        "trajectory_id": "TRJ-1004-REPAIR-001",
        "case_id": manifest["case_id"],
        "trajectory_type": "evidence_dropout_local_repair",
        "clean_prefix_id": clean["trajectory_id"],
        "mutation": {
            "operator": "evidence_dropout",
            "surface": "evidence_store",
            "removed_evidence_ids": ["EA-OMIT-001"],
            "target_fields": TARGET_FIELDS,
            "invariant_fields": INVARIANT_FIELDS,
            "media_changed": False,
            "oracle_changed": False,
            "claim_text_changed_at_injection": False,
        },
        "mutated_state": {
            "available_evidence_ids": [item["evidence_id"] for item in mutated_evidence],
            "field_bindings": mutated_bindings,
            "obligation_status": "open",
            "reopened_fields": TARGET_FIELDS,
            "preserved_fields": INVARIANT_FIELDS,
        },
        "events": repair_events,
        "repair_evidence": repair_atom,
        "final_state": {
            "available_evidence_ids": [item["evidence_id"] for item in mutated_evidence] + ["EA-REPAIR-001"],
            "obligations": [{**clean["final_state"]["obligations"][0], "status": "verified"}],
            "claim_ledger": [
                {"claim_id": "CL-1004-001", "status": "verified", "claim": repaired_claim, "field_bindings": repaired_bindings}
            ],
            "stop_decision": {"allowed": True, "reason_code": "all_reopened_fields_reverified", "open_high_priority_obligations": []},
        },
    }

    graph = _provenance_graph(clean_events, clean_bindings, repair_events, repaired_bindings)
    clean_sft = {
        "messages": [
            {"role": "system", "content": "Maintain typed evidence-to-claim state. Bind every verified field to visible evidence, reopen only invalidated fields, and stop only after all high-priority obligations are verified."},
            {"role": "user", "content": json.dumps({"task": "Determine whether and how the edited clip omits source content.", "case_manifest": manifest}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"events": clean_events, "final_state": clean["final_state"]}, ensure_ascii=False)},
        ],
        "metadata": {"case_id": manifest["case_id"], "trajectory_id": clean["trajectory_id"], "trajectory_type": clean["trajectory_type"], "generator": "deterministic_ecso_v2"},
    }
    repair_sft = {
        "messages": [
            {"role": "system", "content": "Maintain typed evidence-to-claim state. Bind every verified field to visible evidence, reopen only invalidated fields, and stop only after all high-priority obligations are verified."},
            {"role": "user", "content": json.dumps({"task": "Repair the state after an evidence-store mutation without changing independently supported fields.", "case_manifest": manifest, "available_evidence_ids": repair["mutated_state"]["available_evidence_ids"]}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"action": "audit_field_support", "invalidated_fields": TARGET_FIELDS, "preserved_fields": INVARIANT_FIELDS, "next_action": "reopen_affected_obligation"}, ensure_ascii=False)},
            {"role": "tool", "content": json.dumps(repair_atom, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"events": repair_events[2:], "final_state": repair["final_state"]}, ensure_ascii=False)},
        ],
        "metadata": {"case_id": manifest["case_id"], "trajectory_id": repair["trajectory_id"], "trajectory_type": repair["trajectory_type"], "generator": "deterministic_ecso_v2"},
    }
    return {"clean": clean, "repair": repair, "provenance_graph": graph, "clean_sft": clean_sft, "repair_sft": repair_sft}
