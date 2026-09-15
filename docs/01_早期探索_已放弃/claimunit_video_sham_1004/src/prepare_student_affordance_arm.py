"""Add visible state-action affordances to the external-contract student arm."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from interactive_harness import derive_state_action_affordances
from pipeline import ROOT, write_json


DATA_ROOT = ROOT / "outputs/student_external_contract_v1"
SOURCE_ARM = "decision_sufficient_state_v4_external_tool_contract"
OUTPUT_ARM = "decision_sufficient_state_v5_action_affordances"
STATE_TOOLS = {
    "create_candidate_claim",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
    "stop",
}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = row["messages"][-1]["tool_calls"][0]["function"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return str(function["name"]), arguments


def _alignment_errors(
    tool_name: str, arguments: dict[str, Any], affordances: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if tool_name == "bind_evidence":
        claim_id = str(arguments.get("claim_id"))
        candidates = affordances["binding_candidates"].get(claim_id, {})
        for item in arguments.get("binding_items") or []:
            field = str(item.get("field"))
            allowed = set(candidates.get(field, []))
            requested = set(str(value) for value in item.get("support_ids") or [])
            if not requested or not requested.issubset(allowed):
                errors.append(f"bind:{claim_id}:{field}:{sorted(requested - allowed)}")
    elif tool_name == "set_field_decision":
        claim_id = str(arguments.get("claim_id"))
        status = str(arguments.get("decision_status"))
        requested = set(str(value) for value in arguments.get("fields") or [])
        matching = [
            item
            for item in affordances["meaningful_decision_transitions"]
            if item["claim_id"] == claim_id and item["decision_status"] == status
        ]
        allowed = set(matching[0]["fields"]) if matching else set()
        if not requested or not requested.issubset(allowed):
            errors.append(f"decision:{claim_id}:{status}:{sorted(requested - allowed)}")
    return errors


def build(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    target_counts: Counter[str] = Counter()
    alignment_failures: list[dict[str, Any]] = []
    affordance_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        rows = _read(data_root / SOURCE_ARM / split / "sft_all.jsonl")
        split_counts[split] = len(rows)
        for row in rows:
            payload = json.loads(row["messages"][1]["content"])
            state = payload["current_harness_state"]
            schema = payload["task"]["claim_schema"]
            affordances = derive_state_action_affordances(state, schema)
            state["state_action_affordances"] = affordances
            payload["compression_contract"] += (
                " state_action_affordances externalizes schema-valid argument choices derived "
                "from the same visible state; it is authoritative. repair_restoration_candidates "
                "identify an eligible prior measurement_method after evidence dropout, but its "
                "new observation must still be executed and validated. executable_binding_actions "
                "are complete native calls that cover every currently bindable field with a "
                "minimum-cardinality set of compatible active EvidenceAtoms."
            )
            row["messages"][1]["content"] = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            metadata = row["metadata"]
            metadata["context_arm"] = OUTPUT_ARM
            metadata["context_view"] = OUTPUT_ARM
            metadata["state_action_affordance_contract"] = "visible_derived_v1"
            tool_name, arguments = _target(row)
            target_counts[tool_name] += 1
            errors = _alignment_errors(tool_name, arguments, affordances)
            if errors:
                alignment_failures.append({"row_id": metadata["row_id"], "errors": errors})
            affordance_counts["rows_with_binding_candidates"] += bool(
                affordances["binding_candidates"]
            )
            affordance_counts["rows_with_decision_transitions"] += bool(
                affordances["meaningful_decision_transitions"]
            )
            affordance_counts["rows_with_repair_restoration_candidates"] += bool(
                affordances["repair_restoration_candidates"]
            )
            affordance_counts["rows_with_executable_binding_actions"] += bool(
                affordances["executable_binding_actions"]
            )
        _write(data_root / OUTPUT_ARM / split / "sft_all.jsonl", rows)
    report = {
        "experiment": "student_visible_state_action_affordances_v1",
        "source_arm": SOURCE_ARM,
        "output_arm": OUTPUT_ARM,
        "split_counts": split_counts,
        "rows": sum(split_counts.values()),
        "target_tool_counts": dict(target_counts),
        "affordance_counts": dict(affordance_counts),
        "target_affordance_alignment_failures": len(alignment_failures),
        "alignment_failure_examples": alignment_failures[:20],
        "oracle_inputs_used": False,
    }
    if alignment_failures:
        raise ValueError(f"teacher targets disagree with visible affordances: {alignment_failures[:3]}")
    write_json(data_root / OUTPUT_ARM / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
