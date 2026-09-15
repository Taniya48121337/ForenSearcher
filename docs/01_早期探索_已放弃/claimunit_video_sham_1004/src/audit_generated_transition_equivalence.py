"""Match generated actions to recorded executed transitions without claiming live replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import _arguments_semantically_equal, _parse_generated_call
from pipeline import ROOT, write_json
from student_tool_contract import canonical_student_arguments


DEFAULT_REPORT = (
    ROOT / "outputs/student_external_contract_v1/eval_adapter120_v4_4k_gen9_512.json"
)
DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v4_external_tool_contract"
    / "validation/sft_all.jsonl"
)
DEFAULT_TRANSITIONS = ROOT / "outputs/production_sft_v7_external_contract/validation/transition_sidecar.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/student_external_contract_v1/generated_transition_equivalence_gen9.json"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(report_path: Path, sft_path: Path, transition_path: Path) -> dict[str, Any]:
    generation_report = json.loads(report_path.read_text(encoding="utf-8"))
    sft_by_id = {str(row["metadata"]["row_id"]): row for row in _jsonl(sft_path)}
    transition_by_key = {
        (
            str(row["dataset"]),
            str(row["case_id"]),
            str(row["trajectory_id"]),
            str(row["episode_mode"]),
            int(row["turn_index"]),
        ): row
        for row in _jsonl(transition_path)
    }
    rows: list[dict[str, Any]] = []
    for generated in generation_report["rows"]:
        if not generated.get("generated"):
            continue
        row_id = str(generated["row_id"])
        sft = sft_by_id[row_id]
        metadata = sft["metadata"]
        key = (
            str(metadata["dataset"]),
            str(metadata["case_id"]),
            str(metadata["trajectory_id"]),
            "repair" if metadata.get("trajectory_type") == "repair" else "clean",
            int(metadata["turn_index"]),
        )
        transition = transition_by_key.get(key)
        generated_name, generated_arguments = _parse_generated_call(str(generated.get("generated_text", "")))
        recorded_name = None
        recorded_external_arguments: dict[str, Any] | None = None
        if transition is not None:
            recorded = transition["chosen_action"]
            recorded_name = str(recorded["tool_name"])
            recorded_external_arguments = canonical_student_arguments(
                recorded_name, recorded.get("arguments") or {}
            )
        semantic_match = bool(
            transition is not None
            and generated_name == recorded_name
            and _arguments_semantically_equal(
                str(recorded_name), generated_arguments, recorded_external_arguments or {}
            )
        )
        labels = (transition or {}).get("step_labels", {})
        status = str(labels.get("execution_status", ""))
        recorded_executable = status in {"accepted", "executed"}
        rows.append(
            {
                "row_id": row_id,
                "dataset": metadata["dataset"],
                "generated_tool": generated_name,
                "recorded_tool": recorded_name,
                "semantic_action_match": semantic_match,
                "recorded_execution_status": status,
                "recorded_action_accepted_or_executed": recorded_executable,
                "recorded_downstream_consumed": labels.get("downstream_consumed") is True,
                "recorded_causally_necessary_or_branch_enabling": (
                    labels.get("causally_necessary") is True
                    or labels.get("branch_enabling_measurement") is True
                ),
                "recorded_transition": (transition or {}).get("transition"),
                "equivalent_to_verified_transition": bool(
                    semantic_match
                    and recorded_executable
                    and labels.get("downstream_consumed") is True
                    and (
                        labels.get("causally_necessary") is True
                        or labels.get("branch_enabling_measurement") is True
                    )
                ),
            }
        )
    count = len(rows)
    equivalent = sum(row["equivalent_to_verified_transition"] for row in rows)
    return {
        "audit_type": "generated_action_to_recorded_verified_transition_equivalence",
        "live_harness_reexecution_performed": False,
        "claim_boundary": (
            "A semantic match proves equivalence to the action previously accepted/executed at this recorded state; "
            "it is not an independent live replay of the generated action."
        ),
        "generation_report": str(report_path),
        "rows": count,
        "equivalent_to_verified_transition": equivalent,
        "equivalence_rate": round(equivalent / max(count, 1), 4),
        "per_dataset": {
            dataset: {
                "rows": sum(row["dataset"] == dataset for row in rows),
                "equivalent": sum(
                    row["dataset"] == dataset and row["equivalent_to_verified_transition"] for row in rows
                ),
            }
            for dataset in sorted({str(row["dataset"]) for row in rows})
        },
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.report, args.sft, args.transitions)
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("rows", "equivalent_to_verified_transition", "equivalence_rate", "per_dataset")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
