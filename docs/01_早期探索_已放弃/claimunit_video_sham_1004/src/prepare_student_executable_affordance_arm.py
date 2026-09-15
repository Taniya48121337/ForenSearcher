"""Compile meaningful visible decision transitions into native executable calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from student_tool_contract import executable_argument_schema


DATA_ROOT = ROOT / "outputs/student_external_contract_v1"
SOURCE_ARM = "decision_sufficient_state_v5_action_affordances"
OUTPUT_ARM = "decision_sufficient_state_v6_executable_affordances"


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


def build(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    split_counts: dict[str, int] = {}
    compiled_rows = 0
    binding_candidate_rows_compacted = 0
    dynamically_constrained_tool_rows = 0
    exact_target_matches = 0
    mismatches: list[str] = []
    for split in ("train", "validation", "test"):
        rows = _read(data_root / SOURCE_ARM / split / "sft_all.jsonl")
        split_counts[split] = len(rows)
        for row in rows:
            payload = json.loads(row["messages"][1]["content"])
            state = payload["current_harness_state"]
            affordances = state["state_action_affordances"]
            claim_field_order = {
                str(claim["claim_id"]): list((claim.get("fields") or {}).keys())
                for claim in state.get("candidate_claims") or []
            }
            compiled = [
                {
                    "tool_name": "set_field_decision",
                    "arguments": {
                        "claim_id": transition["claim_id"],
                        "fields": [
                            field
                            for field in claim_field_order.get(
                                str(transition["claim_id"]), transition["fields"]
                            )
                            if field in set(transition["fields"])
                        ],
                        "decision_status": transition["decision_status"],
                    },
                }
                for transition in affordances.get("meaningful_decision_transitions", [])
            ]
            affordances["executable_decision_actions"] = compiled
            affordances["derivation"] = "Compiled from student-visible state."
            complete_actions = list(affordances.get("executable_binding_actions") or [])
            complete_actions.extend(compiled)
            actions_by_tool: dict[str, list[dict[str, Any]]] = {}
            for action in complete_actions:
                actions_by_tool.setdefault(str(action["tool_name"]), []).append(
                    action["arguments"]
                )
            constrained = False
            for tool in row.get("tools") or []:
                function = tool.get("function") or {}
                name = str(function.get("name"))
                if name in actions_by_tool:
                    function["parameters"] = executable_argument_schema(
                        actions_by_tool[name]
                    )
                    constrained = True
            dynamically_constrained_tool_rows += int(constrained)
            if affordances.get("executable_binding_actions") and affordances.get(
                "binding_candidates"
            ):
                affordances["binding_candidates"] = {}
                binding_candidate_rows_compacted += 1
            if compiled:
                compiled_rows += 1
            tool_name, arguments = _target(row)
            if tool_name == "set_field_decision":
                if any(
                    item["tool_name"] == tool_name and item["arguments"] == arguments
                    for item in compiled
                ):
                    exact_target_matches += 1
                else:
                    mismatches.append(str(row["metadata"]["row_id"]))
            payload["compression_contract"] += (
                " executable_decision_actions are complete native calls compiled from visible "
                "meaningful transitions; copy their argument keys exactly when selecting one. "
                "When executable_binding_actions are present, they replace redundant lower-level "
                "binding_candidates in this policy view."
            )
            row["messages"][1]["content"] = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            row["metadata"]["context_arm"] = OUTPUT_ARM
            row["metadata"]["context_view"] = OUTPUT_ARM
            row["metadata"]["executable_affordance_contract"] = "visible_compiled_native_call_v1"
        _write(data_root / OUTPUT_ARM / split / "sft_all.jsonl", rows)
    report = {
        "experiment": "student_visible_executable_affordances_v1",
        "source_arm": SOURCE_ARM,
        "output_arm": OUTPUT_ARM,
        "split_counts": split_counts,
        "rows": sum(split_counts.values()),
        "rows_with_compiled_decision_action": compiled_rows,
        "rows_with_raw_binding_candidates_compacted": binding_candidate_rows_compacted,
        "rows_with_dynamic_native_argument_constraints": dynamically_constrained_tool_rows,
        "set_decision_targets_exactly_present": exact_target_matches,
        "set_decision_target_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "oracle_inputs_used": False,
    }
    if mismatches:
        raise ValueError(f"compiled visible actions disagree with teacher targets: {mismatches[:3]}")
    write_json(data_root / OUTPUT_ARM / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
