"""Summarize the staged native-tool SFT canaries without rerunning generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import (
    _arguments_semantically_equal,
    _arguments_validate_visible_schema,
    _parse_generated_call,
    _reference_call,
)
from pipeline import ROOT, write_json


OUTPUT = ROOT / "outputs" / "student_external_contract_v1" / "native_tool_canary_summary.json"


STAGES = [
    (
        "v1_history_envelope_collision_12_steps_160",
        ROOT / "outputs/student_context_ablation_v2/eval_seeded_balanced_compact_4k_gen9.json",
        ROOT / "outputs/student_context_ablation_v2/decision_sufficient_state_v1/validation/sft_all.jsonl",
    ),
    (
        "v2_role_separated_no_schemas_12_steps_160",
        ROOT / "outputs/student_role_separation_ablation_v1/eval_adapter_v2_4k_gen9.json",
        ROOT / "outputs/student_role_separation_ablation_v1/decision_sufficient_state_v2_role_separated/validation/sft_all.jsonl",
    ),
    (
        "v3_native_schemas_internal_target_12_steps_320",
        ROOT / "outputs/student_tool_contract_ablation_v1/eval_adapter_v3_4k_gen9.json",
        ROOT / "outputs/student_tool_contract_ablation_v1/decision_sufficient_state_v3_native_tool_schemas/validation/sft_all.jsonl",
    ),
    (
        "v4_external_contract_12_steps_320",
        ROOT / "outputs/student_external_contract_v1/eval_adapter_v4_4k_gen9.json",
        ROOT / "outputs/student_external_contract_v1/decision_sufficient_state_v4_external_tool_contract/validation/sft_all.jsonl",
    ),
    (
        "v4_external_contract_120_steps_320",
        ROOT / "outputs/student_external_contract_v1/eval_adapter120_v4_4k_gen9.json",
        ROOT / "outputs/student_external_contract_v1/decision_sufficient_state_v4_external_tool_contract/validation/sft_all.jsonl",
    ),
    (
        "v4_external_contract_120_steps_512",
        ROOT / "outputs/student_external_contract_v1/eval_adapter120_v4_4k_gen9_512.json",
        ROOT / "outputs/student_external_contract_v1/decision_sufficient_state_v4_external_tool_contract/validation/sft_all.jsonl",
    ),
]


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["metadata"]["row_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def build() -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for name, report_path, rows_path in STAGES:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows_by_id = _read_rows(rows_path)
        generated = [item for item in report["rows"] if item.get("generated")]
        semantic_flags: list[bool] = []
        schema_flags: list[bool] = []
        for item in generated:
            row = rows_by_id[str(item["row_id"])]
            generated_name, generated_arguments = _parse_generated_call(str(item.get("generated_text", "")))
            reference_name, reference_arguments = _reference_call(row)
            semantic_flags.append(
                generated_name == reference_name
                and _arguments_semantically_equal(reference_name, generated_arguments, reference_arguments)
            )
            if row.get("tools"):
                schema_flags.append(
                    _arguments_validate_visible_schema(row, generated_name, generated_arguments)
                )
        denominator = max(len(generated), 1)
        summary = report["summary"]
        stages.append(
            {
                "stage": name,
                "generation_rows": len(generated),
                "target_nll": summary.get("target_nll"),
                "parse_rate": summary.get("tool_call_parse_rate"),
                "tool_name_exact_rate": summary.get("tool_name_exact_rate"),
                "schema_valid_rate": (
                    round(sum(schema_flags) / denominator, 4) if schema_flags else None
                ),
                "strict_argument_exact_rate": summary.get("arguments_exact_rate"),
                "semantic_argument_equivalence_rate": round(sum(semantic_flags) / denominator, 4),
                "prestate_eligible_rate": summary.get("prestate_eligible_rate"),
                "token_limit_hit_rate": summary.get("generation_token_limit_hit_rate"),
                "report": str(report_path),
            }
        )
    result = {
        "experiment": "native_tool_contract_canary",
        "status": "canary_not_mass_production_proof",
        "fixed_validation_rows": 18,
        "fixed_generation_rows": 9,
        "final_observation": (
            "At 120 balanced exposures and a fixed 512-token action budget, all 9 generated actions were parseable, "
            "schema-valid, pre-state eligible, tool-name exact, and semantically argument-equivalent."
        ),
        "stages": stages,
        "limitations": [
            "nine generation rows are too small for a release claim",
            "closed-loop generated-action execution remains to be measured",
            "the 320- and 512-token results are separate latency/completion operating points",
        ],
    }
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
