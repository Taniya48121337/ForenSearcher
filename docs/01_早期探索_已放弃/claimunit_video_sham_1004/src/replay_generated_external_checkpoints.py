"""Replay representative generated external actions on local video/features.

The replay is intentionally student-visible: it reconstructs inputs from the
generated evaluation row and its public pre-action state.  It does not load a
dataset label, verifier score, or saved teacher observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import _parse_generated_call
from pipeline import ROOT, decode, write_json
from shadow_execute_generated_actions import _jsonl, _payload
from vcsl_batch import VCSLFeatureEnvironment
from vcsl_probe import DEFAULT_FEATURE_ROOT
from videosham_batch import EDITED_ROOT, SOURCE_ROOT, MultiFamilyVideoEnvironment


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
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "cross_dataset_generated_external_checkpoint_replay.json"
)

VIDEO_INSPECT_ROW = (
    "VideoSham:videosham-4189:LLM-TRJ-4189-CLEAN-001:clean:1"
)
VIDEO_MEASURE_ROW = (
    "VideoSham:videosham-4189:LLM-TRJ-4189-CLEAN-001:clean:2"
)
VIDEO_REPAIR_ROW = (
    "VideoSham:videosham-4189:LLM-TRJ-4189-REPAIR-001:repair:2"
)
VCSL_TN_INSPECT_ROW = (
    "VCSL:vcsl-06feccaaee94443a8ddfcd59d7938bad-"
    "808d18d803c649e0badaf17caff18f5c:LLM-TRJ-VCSL-"
    "06FECCAAEE94443A8DDFCD59D7938BAD-808D18D803C649E0BADAF17CAFF18F5C-"
    "CLEAN-001:clean:1"
)
VCSL_TN_MEASURE_ROW = VCSL_TN_INSPECT_ROW.replace(":clean:1", ":clean:2")
VCSL_DP_REPAIR_ROW = (
    "VCSL:vcsl-a6264ea57d364eaca85e8eef6bf788d7-"
    "c8c2d0a9f1174aa6a63971328c8c586f:LLM-TRJ-VCSL-"
    "A6264EA57D364EACA85E8EEF6BF788D7-C8C2D0A9F1174AA6A63971328C8C586F-"
    "REPAIR-001:repair:2"
)


def _generated_call(rows: dict[str, dict[str, Any]], row_id: str) -> tuple[str, dict[str, Any]]:
    name, arguments = _parse_generated_call(rows[row_id]["generated_text"])
    if name is None or arguments is None:
        raise ValueError(f"generated call is not parseable: {row_id}")
    return name, arguments


def _atom_summary(observation: dict[str, Any]) -> dict[str, Any]:
    atom = observation["evidence_atoms"][0]
    return {
        "evidence_id": atom.evidence_id,
        "kind": atom.kind,
        "supports_fields": list(atom.supports_fields),
        "observation": atom.observation,
        "provenance": atom.provenance,
    }


def _state_source_id_and_scope(sft_rows: dict[str, dict[str, Any]]) -> tuple[str, list[float]]:
    state = _payload(sft_rows[VIDEO_MEASURE_ROW])["current_harness_state"]
    source_id = None
    scope = None
    for atom in state["evidence_store"].values():
        visible = atom.get("observation") or {}
        if "source_id" in visible:
            source_id = visible["source_id"]
        if "comparison_scope_s" in visible:
            scope = visible["comparison_scope_s"]
    if source_id is None or scope is None:
        raise ValueError("public VideoSham prestate lacks source identity or comparison scope")
    return str(source_id), list(scope)


def _video_replay(
    generated_rows: dict[str, dict[str, Any]],
    sft_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    inspect_name, inspect_args = _generated_call(generated_rows, VIDEO_INSPECT_ROW)
    measure_name, measure_args = _generated_call(generated_rows, VIDEO_MEASURE_ROW)
    repair_name, repair_args = _generated_call(generated_rows, VIDEO_REPAIR_ROW)
    source_id, scope = _state_source_id_and_scope(sft_rows)
    case_number = "4189"
    source_path = SOURCE_ROOT / f"{case_number}_unedited.mp4"
    suspect_path = EDITED_ROOT / f"{case_number}_processed.mp4"
    source_frames = decode(source_path)
    suspect_frames = decode(suspect_path)
    environment = MultiFamilyVideoEnvironment(
        manifest={
            "case_id": f"videosham-{case_number}",
            "source_id": source_id,
            "comparison_scope_s": scope,
        },
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        source_path=source_path,
        suspect_path=suspect_path,
    )
    eligible_before_inspect = environment.eligible_tool_names()
    inspect = environment.execute(inspect_name, inspect_args)
    eligible_before_measure = environment.eligible_tool_names()
    clean = environment.execute(measure_name, measure_args)
    eligible_before_repair = environment.eligible_tool_names()
    repair = environment.execute(repair_name, repair_args)
    clean_atom = _atom_summary(clean)
    repair_atom = _atom_summary(repair)
    semantic_repeat = (
        clean["result"] == repair["result"]
        and clean_atom["kind"] == repair_atom["kind"]
        and clean_atom["supports_fields"] == repair_atom["supports_fields"]
        and clean_atom["observation"] == repair_atom["observation"]
    )
    checks = {
        "inspect_was_eligible": inspect_name in eligible_before_inspect,
        "measurement_was_eligible": measure_name in eligible_before_measure,
        "repair_measurement_was_eligible": repair_name in eligible_before_repair,
        "generated_clean_and_repair_calls_match": (
            measure_name == repair_name and measure_args == repair_args
        ),
        "clean_and_repair_semantics_identical": semantic_repeat,
        "material_difference_measured": bool(
            clean_atom["observation"].get("material_difference_detected")
        ),
    }
    return {
        "dataset": "VideoSham",
        "case_id": "videosham-4189",
        "audit_type": "raw_local_video_reexecution",
        "raw_video_reprocessed": True,
        "dataset_label_or_editor_task_loaded": False,
        "verifier_or_teacher_observation_loaded": False,
        "generated_actions": {
            "inspect": {"name": inspect_name, "arguments": inspect_args},
            "clean_measurement": {"name": measure_name, "arguments": measure_args},
            "repair_measurement": {"name": repair_name, "arguments": repair_args},
        },
        "inspect_result": {
            "tool_name": inspect["tool_name"],
            "created_evidence_ids": [atom.evidence_id for atom in inspect["evidence_atoms"]],
            "modality_triage": inspect["result"]["modality_triage"],
        },
        "clean_measurement": clean_atom,
        "repair_measurement": repair_atom,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _vcsl_case(query_id: str, reference_id: str) -> dict[str, str]:
    return {
        "pair_id": f"vcsl-{query_id}-{reference_id}",
        "query_id": query_id,
        "reference_id": reference_id,
    }


def _vcsl_tn_replay(generated_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inspect_name, inspect_args = _generated_call(generated_rows, VCSL_TN_INSPECT_ROW)
    measure_name, measure_args = _generated_call(generated_rows, VCSL_TN_MEASURE_ROW)
    case = _vcsl_case(
        "06feccaaee94443a8ddfcd59d7938bad",
        "808d18d803c649e0badaf17caff18f5c",
    )
    environment = VCSLFeatureEnvironment(
        case=case,
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="tn_stability",
    )
    eligible_before_inspect = environment.eligible_tool_names()
    inspect = environment.execute(inspect_name, inspect_args)
    eligible_before_measure = environment.eligible_tool_names()
    measurement = environment.execute(measure_name, measure_args)
    atom = _atom_summary(measurement)
    checks = {
        "inspect_was_eligible": inspect_name in eligible_before_inspect,
        "measurement_was_eligible": measure_name in eligible_before_measure,
        "nonempty_localization": bool(measurement["result"]["copied_segments_frames"]),
        "method_matches_generated_call": (
            measurement["result"]["localization_method"] == measure_args["method"]
        ),
        "label_access_false_in_provenance": atom["provenance"].get("label_access") is False,
    }
    return {
        "dataset": "VCSL",
        "case_id": case["pair_id"],
        "audit_type": "raw_official_feature_reexecution",
        "raw_official_features_reprocessed": True,
        "dataset_label_loaded": False,
        "verifier_or_teacher_observation_loaded": False,
        "generated_actions": {
            "inspect": {"name": inspect_name, "arguments": inspect_args},
            "measurement": {"name": measure_name, "arguments": measure_args},
        },
        "inspect_result": {
            "tool_name": inspect["tool_name"],
            "feature_shapes": {
                "query": inspect["result"]["query_shape"],
                "reference": inspect["result"]["reference_shape"],
            },
            "recommended_localizer_tool": inspect["result"]["recommended_localizer_tool"],
        },
        "measurement": atom,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _vcsl_dp_repair_replay(generated_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    repair_name, repair_args = _generated_call(generated_rows, VCSL_DP_REPAIR_ROW)
    case = _vcsl_case(
        "a6264ea57d364eaca85e8eef6bf788d7",
        "c8c2d0a9f1174aa6a63971328c8c586f",
    )
    environment = VCSLFeatureEnvironment(
        case=case,
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="dp",
    )
    environment.execute(
        "inspect_pair_features",
        {"query_ref": "query", "reference_ref": "reference"},
    )
    clean = environment.execute(repair_name, repair_args)
    eligible_after_clean = environment.eligible_tool_names()
    environment.on_evidence_dropout(
        "EA-VCSL-LOCALIZATION-001",
        ["relationship", "copied_segments_frames", "localization_method"],
    )
    eligible_after_dropout = environment.eligible_tool_names()
    repair = environment.execute(repair_name, repair_args)
    clean_atom = _atom_summary(clean)
    repair_atom = _atom_summary(repair)
    semantic_repeat = (
        clean["result"] == repair["result"]
        and clean_atom["kind"] == repair_atom["kind"]
        and clean_atom["supports_fields"] == repair_atom["supports_fields"]
        and clean_atom["observation"] == repair_atom["observation"]
    )
    checks = {
        "localizer_closed_after_clean_measurement": repair_name not in eligible_after_clean,
        "localizer_reopened_after_dropout": repair_name in eligible_after_dropout,
        "repair_recreated_identical_semantic_measurement": semantic_repeat,
        "nonempty_repair_localization": bool(repair["result"]["copied_segments_frames"]),
        "label_access_false_in_provenance": repair_atom["provenance"].get("label_access") is False,
    }
    return {
        "dataset": "VCSL",
        "case_id": case["pair_id"],
        "audit_type": "raw_official_feature_dropout_repair_reexecution",
        "raw_official_features_reprocessed": True,
        "dataset_label_loaded": False,
        "verifier_or_teacher_observation_loaded": False,
        "generated_action": {"name": repair_name, "arguments": repair_args},
        "clean_measurement": clean_atom,
        "repair_measurement": repair_atom,
        "checks": checks,
        "passed": all(checks.values()),
    }


def replay(report_path: Path, sft_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated_rows = {str(item["row_id"]): item for item in report["rows"]}
    sft_rows = {
        str(item["metadata"]["row_id"]): item
        for item in _jsonl(sft_path)
    }
    audits = [
        _video_replay(generated_rows, sft_rows),
        _vcsl_tn_replay(generated_rows),
        _vcsl_dp_repair_replay(generated_rows),
    ]
    return {
        "audit": "cross_dataset_generated_external_checkpoint_replay",
        "input_policy": {
            "generated_action_source": "student model evaluation output",
            "reconstruction_source": "student-visible pre-action state and logical case IDs",
            "dataset_labels_loaded": False,
            "teacher_observations_loaded": False,
            "local_filesystem_paths_exported": False,
        },
        "summary": {
            "audit_count": len(audits),
            "passed": sum(bool(item["passed"]) for item in audits),
            "failed": sum(not bool(item["passed"]) for item in audits),
            "all_passed": all(bool(item["passed"]) for item in audits),
        },
        "audits": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = replay(args.report, args.sft)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
