"""Deterministic evaluator for the ECSO trajectory quality rubric."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "rubric" / "trajectory_quality_rubric.json"

ALLOWED_KIND_GROUPS: dict[str, list[set[str]]] = {
    "source_id": [{"monotonic_visual_alignment"}],
    "claim_family": [{"monotonic_visual_alignment"}, {"source_time_contraction", "targeted_alignment_remeasurement"}],
    "manipulation_type": [{"source_time_contraction", "targeted_alignment_remeasurement"}],
    "direction": [{"monotonic_visual_alignment"}, {"source_time_contraction", "targeted_alignment_remeasurement"}],
    "comparison_scope_s": [{"case_scope_locator"}],
    "retained_source_interval_s": [{"monotonic_visual_alignment"}],
    "omitted_source_intervals_s": [{"case_scope_locator"}, {"source_time_contraction", "targeted_alignment_remeasurement"}],
    "estimated_removed_time_s": [{"media_metadata"}, {"source_time_contraction", "targeted_alignment_remeasurement"}],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def binding_map(trajectory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ledger = trajectory["final_state"]["claim_ledger"][0]
    return {item["field"]: item for item in ledger["field_bindings"]}


def claim_of(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory["final_state"]["claim_ledger"][0]["claim"]


def fraction(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 1.0


def criterion(spec: dict[str, Any], normalized: float, evidence: list[str], failures: list[str] | None = None) -> dict[str, Any]:
    normalized = max(0.0, min(1.0, normalized))
    earned = round(spec["weight"] * normalized, 3)
    return {
        "criterion_id": spec["id"],
        "name": spec["name"],
        "earned": earned,
        "max_points": spec["weight"],
        "normalized_score": round(normalized, 4),
        "passed_minimum": normalized >= 0.8,
        "evidence": evidence,
        "failure_codes": failures or [],
    }


def has_path(edges: list[dict[str, str]], starts: set[str], target: str) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["from"]].append(edge["to"])
    queue = deque(starts)
    visited = set(starts)
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def binding_is_compatible(field: str, binding: dict[str, Any], evidence_map: dict[str, dict[str, Any]]) -> bool:
    if not binding["evidence_ids"] or not set(binding["evidence_ids"]).issubset(evidence_map):
        return False
    kinds = {evidence_map[evidence_id]["kind"] for evidence_id in binding["evidence_ids"]}
    groups = ALLOWED_KIND_GROUPS[field]
    allowed_union = set().union(*groups)
    return all(bool(kinds.intersection(group)) for group in groups) and kinds.issubset(allowed_union)


def evaluate(
    overrides: dict[str, Any] | None = None,
    *,
    write_report: bool = True,
    raise_on_reject: bool = True,
) -> dict[str, Any]:
    overrides = overrides or {}
    rubric = overrides.get("rubric", load_json(RUBRIC_PATH))
    oracle = overrides.get("oracle", load_json(ROOT / "oracle" / "claim_oracle.json"))
    manifest = overrides.get("manifest", load_json(ROOT / "student_view" / "case_manifest.json"))
    clean = overrides.get("clean", load_json(ROOT / "outputs" / "clean_trajectory.json"))
    repair = overrides.get("repair", load_json(ROOT / "outputs" / "repair_trajectory.json"))
    graph = overrides.get("graph", load_json(ROOT / "outputs" / "provenance_graph.json"))
    clean_sft = overrides.get("clean_sft", (ROOT / "outputs" / "sft_clean.jsonl").read_text(encoding="utf-8"))
    repair_sft = overrides.get("repair_sft", (ROOT / "outputs" / "sft_repair.jsonl").read_text(encoding="utf-8"))
    student_text = clean_sft + repair_sft

    clean_claim = claim_of(clean)
    repair_claim = claim_of(repair)
    clean_bindings = binding_map(clean)
    repair_bindings = binding_map(repair)
    clean_evidence = {item["evidence_id"]: item for item in clean["evidence"]}
    repair_evidence = {**clean_evidence, repair["repair_evidence"]["evidence_id"]: repair["repair_evidence"]}
    clean_available = set(clean["final_state"]["available_evidence_ids"])
    repair_available = set(repair["final_state"]["available_evidence_ids"])
    required_fields = clean["initial_state"]["obligations"][0]["required_fields"]
    removed = set(repair["mutation"]["removed_evidence_ids"])

    clean_refs_valid = all(
        set(binding["evidence_ids"]).issubset(clean_available) and set(binding["evidence_ids"]).issubset(clean_evidence)
        for binding in clean_bindings.values()
    )
    repair_refs_valid = all(
        set(binding["evidence_ids"]).issubset(repair_available) and set(binding["evidence_ids"]).issubset(repair_evidence)
        for binding in repair_bindings.values()
    )
    no_false_closure = all(binding["status"] == "verified" and binding["evidence_ids"] for binding in clean_bindings.values())
    no_false_closure &= all(binding["status"] == "verified" and binding["evidence_ids"] for binding in repair_bindings.values())
    no_false_closure &= not clean["final_state"]["stop_decision"]["open_high_priority_obligations"]
    no_false_closure &= not repair["final_state"]["stop_decision"]["open_high_priority_obligations"]
    oracle_valid = (
        clean_claim["manipulation_type"] == oracle["manipulation_type"]
        and clean_claim["comparison_scope_s"] == oracle["edit_window_s"]
        and abs(manifest["edited_media"]["duration_s"] - oracle["target_output_duration_s"]) <= 0.05
        and clean_claim["claim_family"] == "selective_temporal_omission"
    )
    mutation_integrity = all(
        repair["mutation"][key] is False
        for key in ("media_changed", "oracle_changed", "claim_text_changed_at_injection")
    ) and repair["mutation"]["surface"] == "evidence_store"
    repair_non_leakage = not any(removed.intersection(binding["evidence_ids"]) for binding in repair_bindings.values())
    semantic_binding_integrity = all(
        binding_is_compatible(field, bindings[field], evidence_map)
        for bindings, evidence_map in ((clean_bindings, clean_evidence), (repair_bindings, repair_evidence))
        for field in required_fields
    )
    hard_gate_values = {
        "G1": oracle["annotation_text"] not in student_text and "claim_oracle.json" not in student_text,
        "G2": clean_refs_valid and repair_refs_valid and semantic_binding_integrity,
        "G3": no_false_closure,
        "G4": oracle_valid,
        "G5": mutation_integrity,
        "G6": repair_non_leakage,
    }
    hard_gates = [
        {
            "gate_id": item["id"],
            "name": item["name"],
            "passed": hard_gate_values[item["id"]],
            "failure": None if hard_gate_values[item["id"]] else item["failure"],
        }
        for item in rubric["hard_gates"]
    ]

    specs = {item["id"]: item for item in rubric["criteria"]}
    results: list[dict[str, Any]] = []

    outcome_flags = [
        clean_claim["manipulation_type"] == oracle["manipulation_type"],
        clean_claim["comparison_scope_s"] == oracle["edit_window_s"],
        abs(manifest["edited_media"]["duration_s"] - oracle["target_output_duration_s"]) <= 0.05,
        clean_claim["estimated_removed_time_s"] > 0,
        repair_claim == {**clean_claim, "evidence_ids": repair_claim["evidence_ids"]},
    ]
    results.append(criterion(specs["O1"], fraction(outcome_flags), ["oracle/claim_oracle.json", "clean.final_state", "repair.final_state"]))

    completeness_flags = [field in clean_claim and clean_claim[field] not in (None, "", []) for field in required_fields]
    results.append(criterion(specs["O2"], fraction(completeness_flags), [f"required_fields={required_fields}"]))

    correctness_flags: list[bool] = []
    for bindings, evidence_map in ((clean_bindings, clean_evidence), (repair_bindings, repair_evidence)):
        for field in required_fields:
            correctness_flags.append(binding_is_compatible(field, bindings[field], evidence_map))
    # Precision is conjunctive here: one wrongly attributed field can teach the
    # student a spurious evidence shortcut, even when every other binding is sound.
    results.append(
        criterion(
            specs["E1"],
            1.0 if correctness_flags and all(correctness_flags) else 0.0,
            ["field-to-evidence-kind compatibility matrix"],
        )
    )

    completeness = [
        bool(bindings[field]["evidence_ids"]) and bindings[field]["status"] == "verified"
        for bindings in (clean_bindings, repair_bindings)
        for field in required_fields
    ]
    results.append(criterion(specs["E2"], fraction(completeness), ["clean and repair field_bindings"]))

    action_nodes = {node["node_id"] for node in graph["nodes"] if node["node_type"] == "action"}
    repair_action_nodes = {node["node_id"] for node in graph["nodes"] if node["node_type"] == "repair_action"}
    provenance_flags = []
    for field in required_fields:
        field_id = f"FIELD-{field}"
        evidence_starts = set(clean_bindings[field]["evidence_ids"])
        provenance_flags.extend(
            [
                all(any(has_path(graph["edges"], {action}, evidence_id) for action in action_nodes) for evidence_id in evidence_starts),
                all(has_path(graph["edges"], {evidence_id}, field_id) for evidence_id in evidence_starts),
                has_path(graph["edges"], {field_id}, "CL-1004-001"),
                has_path(graph["edges"], {field_id}, "OBL-1004-001"),
            ]
        )
    for field in repair["mutation"]["target_fields"]:
        field_id = f"FIELD-{field}"
        provenance_flags.extend(
            [
                any(has_path(graph["edges"], {action}, "EA-REPAIR-001") for action in repair_action_nodes),
                has_path(graph["edges"], {"EA-REPAIR-001"}, field_id),
            ]
        )
    # Provenance closure is also conjunctive: an isolated field cannot be audited
    # or repaired locally, so averaging it away would hide the exact failure that
    # the ECSO design is meant to expose.
    results.append(
        criterion(
            specs["E3"],
            1.0 if provenance_flags and all(provenance_flags) else 0.0,
            ["outputs/provenance_graph.json"],
        )
    )

    required_order = [
        "inspect_media",
        "bind_comparison_scope",
        "align_visual_timeline",
        "measure_source_time_contraction",
        "propose_claim",
        "verify_claim",
        "stop",
    ]
    observed = [event["action"] for event in clean["events"]]
    order_flags = [action in observed for action in required_order]
    order_flags.append(
        all(action in observed for action in required_order)
        and [observed.index(action) for action in required_order]
        == sorted(observed.index(action) for action in required_order)
    )
    repair_order = ["audit_field_support", "reopen_affected_obligation", "targeted_alignment_remeasurement", "rebind_affected_fields", "verify_repaired_claim", "stop"]
    repair_observed = [event["action"] for event in repair["events"]]
    order_flags.extend(action in repair_observed for action in repair_order)
    order_flags.append(
        all(action in repair_observed for action in repair_order)
        and [repair_observed.index(action) for action in repair_order]
        == sorted(repair_observed.index(action) for action in repair_order)
    )
    results.append(criterion(specs["P1"], fraction(order_flags), [f"clean_order={observed}", f"repair_order={repair_observed}"]))

    clean_statuses = [clean["initial_state"]["obligations"][0]["status"]] + [
        event["state_delta"]["obligation_status"] for event in clean["events"] if "obligation_status" in event["state_delta"]
    ]
    lifecycle_flags = [clean_statuses == ["open", "evidence_ready", "proposed", "verified"]]
    repair_statuses = [
        event["state_delta"]["obligation_status"] for event in repair["events"] if "obligation_status" in event["state_delta"]
    ]
    lifecycle_flags.append(repair_statuses == ["open", "proposed", "verified"])
    lifecycle_flags.append("verify_claim" in observed and observed.index("verify_claim") < observed.index("stop"))
    lifecycle_flags.append("verify_repaired_claim" in repair_observed and repair_observed.index("verify_repaired_claim") < repair_observed.index("stop"))
    results.append(criterion(specs["P2"], fraction(lifecycle_flags), [f"clean_statuses={clean_statuses}", f"repair_statuses={repair_statuses}"]))

    clean_progress = [event["obligation_progress"] for event in clean["events"]]
    repair_progress = [event["obligation_progress"] for event in repair["events"]]
    progress_flags = [
        all(left <= right for left, right in zip(clean_progress, clean_progress[1:])),
        clean_progress[-1] == 1.0,
        repair_progress[0] < 1.0,
        all(left <= right for left, right in zip(repair_progress, repair_progress[1:])),
        repair_progress[-1] == 1.0,
    ]
    results.append(criterion(specs["P3"], fraction(progress_flags), [f"clean={clean_progress}", f"repair={repair_progress}"]))

    mutated_bindings = {item["field"]: item for item in repair["mutated_state"]["field_bindings"]}
    unsupported = {field for field, item in mutated_bindings.items() if item["status"] == "unsupported"}
    reopened = set(repair["mutated_state"]["reopened_fields"])
    target = set(repair["mutation"]["target_fields"])
    sensitivity_flags = [unsupported == target, reopened == target]
    results.append(criterion(specs["M1"], fraction(sensitivity_flags), [f"target={sorted(target)}", f"unsupported={sorted(unsupported)}", f"reopened={sorted(reopened)}"]))

    invariant = set(repair["mutation"]["invariant_fields"])
    invariance_flags = []
    for field in invariant:
        invariance_flags.extend(
            [
                mutated_bindings[field]["value"] == clean_bindings[field]["value"],
                mutated_bindings[field]["evidence_ids"] == clean_bindings[field]["evidence_ids"],
                mutated_bindings[field]["status"] == "verified",
                repair_bindings[field]["value"] == clean_bindings[field]["value"],
                repair_bindings[field]["evidence_ids"] == clean_bindings[field]["evidence_ids"],
            ]
        )
    results.append(criterion(specs["M2"], fraction(invariance_flags), [f"invariant_fields={sorted(invariant)}"]))

    recovery_flags = [
        repair["repair_evidence"]["evidence_id"] == "EA-REPAIR-001",
        "EA-REPAIR-001" in repair_available,
        "EA-OMIT-001" not in repair_available,
        all(repair_bindings[field]["status"] == "verified" for field in target),
        all("EA-REPAIR-001" in repair_bindings[field]["evidence_ids"] for field in target),
        all(repair_claim[field] == clean_claim[field] for field in required_fields),
        set(repair["mutated_state"]["preserved_fields"]) == invariant,
    ]
    results.append(criterion(specs["M3"], fraction(recovery_flags), ["repair evidence, target bindings, and recovered claim"]))

    alignment_p05 = next(item for item in clean["evidence"] if item["evidence_id"] == "EA-ALIGN-001")["observation"]["p05_similarity"]
    stop_flags = [
        clean["final_state"]["stop_decision"]["allowed"],
        repair["final_state"]["stop_decision"]["allowed"],
        observed[-2:] == ["verify_claim", "stop"],
        repair_observed[-2:] == ["verify_repaired_claim", "stop"],
        len(observed) == len(set(observed)),
        clean_claim["confidence"] != "high" or alignment_p05 >= 0.9,
    ]
    results.append(criterion(specs["S1"], fraction(stop_flags), [f"alignment_p05={alignment_p05}", "explicit verification before stop"]))

    total_score = round(sum(item["earned"] for item in results), 3)
    gates_pass = all(item["passed"] for item in hard_gates)
    criteria_pass = all(item["passed_minimum"] for item in results)
    accepted = gates_pass and total_score >= 95 and criteria_pass
    grade = "reject"
    if gates_pass:
        for band in rubric["aggregation"]["grades"]:
            if total_score >= band["min"]:
                grade = band["label"]
                break
    failed_criteria = [item["criterion_id"] for item in results if not item["passed_minimum"]]
    report = {
        "rubric_id": rubric["rubric_id"],
        "trajectory_bundle": [clean["trajectory_id"], repair["trajectory_id"]],
        "hard_gates_passed": gates_pass,
        "hard_gates": hard_gates,
        "total_score": total_score,
        "max_score": 100,
        "grade": grade,
        "accepted_for_sft": accepted,
        "criteria": results,
        "diagnosis": {
            "failed_gates": [item["gate_id"] for item in hard_gates if not item["passed"]],
            "failed_criteria": failed_criteria,
            "earliest_root_cause": failed_criteria[0] if failed_criteria else None,
        },
        "judge_mode": "deterministic_structured_state",
        "llm_judge_used": False,
    }
    if write_report:
        write_json(ROOT / "outputs" / "trajectory_quality_report.json", report)
    if not accepted and raise_on_reject:
        raise RuntimeError(f"trajectory bundle rejected by rubric: score={total_score}, diagnosis={report['diagnosis']}")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
