"""Execute one label-blind evidence-choice branch set in each real local adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidence_choice_search import commit_selected_branch, select_evidence_branch
from interactive_harness import TaskContract, VideoForensicHarness
from llm_teacher_rollout import TeacherAction
from pipeline import ROOT, decode, write_json
from vcsl_batch import (
    LOCALIZER_SPECS,
    VCSL_SCHEMA,
    VCSLFeatureEnvironment,
    _manifest as vcsl_manifest,
    vcsl_tool_registry_for_variant,
)
from vcsl_probe import DEFAULT_FEATURE_ROOT, select_cases
from viddiff_batch import (
    DEFAULT_ADAPTIVE,
    VIDDIFF_CAUSAL_TOOL_REGISTRY,
    VIDDIFF_SCHEMA,
    VidDiffAdaptiveMeasurementEnvironment,
    _manifest as viddiff_manifest,
)
from videosham_batch import (
    EDITED_ROOT,
    MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
    MULTIFAMILY_SCHEMA,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
)


VIDEOSHAM_MANIFEST = (
    ROOT
    / "outputs/batch/videosham_cases_4055-4094-4005-4144-1045-0069-4080_n7"
    / "cases/videosham-4055/student_view/case_manifest.json"
)


def _contract(manifest: dict[str, Any], schema: dict[str, Any], assets: dict[str, Any]) -> TaskContract:
    return TaskContract(
        contract_id=f"TASK-{manifest['case_id'].upper()}",
        episode_mode="paired_compare",
        task=str(manifest["task"]),
        visible_assets=assets,
        claim_schema_id=str(schema["schema_id"]),
    )


def _action(name: str, params: dict[str, Any], index: int) -> TeacherAction:
    reasoning = (
        f"The comparison still lacks evidence from the {name} measurement family. "
        "This branch tests a distinct visible explanation and can return either positive or negative evidence "
        "without presuming the result."
    )
    return TeacherAction(
        response_id=f"LOCAL-BRANCH-RESPONSE-{index:02d}",
        call_id=f"LOCAL-BRANCH-CALL-{index:02d}",
        tool_name=name,
        reasoning=reasoning,
        params=params,
        raw_arguments=json.dumps({"reasoning": reasoning, **params}),
        usage={},
        replay_items=[],
    )


def _compact_selection(
    harness: VideoForensicHarness,
    actions: list[TeacherAction],
) -> dict[str, Any]:
    visible = [action.tool_name for action in actions]
    counts_before = dict(harness.environment.call_counts)
    selection = select_evidence_branch(
        harness=harness,
        actions=actions,
        visible_semantic_tools=visible,
    )
    counts_after_preview = dict(harness.environment.call_counts)
    selected = selection["selected"]
    if selected is None:
        raise RuntimeError("real environment produced no branch")
    committed = commit_selected_branch(harness, selected)
    branch_rows = []
    for branch in selection["audit"]["branches"]:
        result = branch["result"]
        branch_rows.append(
            {
                "tool": branch["action"]["tool_name"],
                "utility": branch["utility"],
                "status": result.get("status", "executed"),
                "goal_satisfied": result.get("goal_satisfied"),
                "evidence_kinds": [
                    getattr(atom, "kind", None)
                    if not isinstance(atom, dict)
                    else atom.get("kind")
                    for atom in result.get("evidence_atoms") or []
                ],
            }
        )
    return {
        "visible_semantic_tools": visible,
        "proposal_coverage_passed": selection["audit"]["proposal_coverage_passed"],
        "selection_ambiguous_without_teacher_reasoning_scores": selection["audit"][
            "selection_ambiguous"
        ],
        "selection_frozen_before_branch_execution": selection["audit"][
            "selection_frozen_before_branch_execution"
        ],
        "hindsight_outcome_used_for_selection": selection["audit"][
            "hindsight_outcome_used_for_selection"
        ],
        "selected_branch_execution_admissible": selection["audit"][
            "selected_branch_execution_admissible"
        ],
        "higher_utility_branch_was_not_allowed_to_reroute_selection": selection[
            "audit"
        ]["higher_utility_branch_was_not_allowed_to_reroute_selection"],
        "parent_call_counts_unchanged_by_preview": counts_before == counts_after_preview,
        "selected_tool": selected.action.tool_name,
        "deterministic_replay_match": committed["deterministic_replay_match"],
        "committed_call_count_delta": {
            name: harness.environment.call_counts.get(name, 0) - counts_before.get(name, 0)
            for name in harness.environment.call_counts
            if harness.environment.call_counts.get(name, 0) != counts_before.get(name, 0)
        },
        "branches": branch_rows,
    }


def _videosham() -> dict[str, Any]:
    manifest = json.loads(VIDEOSHAM_MANIFEST.read_text(encoding="utf-8"))
    case_number = manifest["case_id"].removeprefix("videosham-")
    source_path = SOURCE_ROOT / f"{case_number}_unedited.mp4"
    suspect_path = EDITED_ROOT / f"{case_number}_processed.mp4"
    source_frames = decode(source_path)
    suspect_frames = decode(suspect_path)
    environment = MultiFamilyVideoEnvironment(
        manifest=manifest,
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        source_path=source_path,
        suspect_path=suspect_path,
    )
    harness = VideoForensicHarness(
        _contract(
            manifest,
            MULTIFAMILY_SCHEMA,
            {
                "source": "source",
                "suspect": "suspect",
                "comparison_scope_s": manifest["comparison_scope_s"],
            },
        ),
        environment,
        claim_schema=MULTIFAMILY_SCHEMA,
        tool_registry=MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
        target_fields=MULTIFAMILY_SCHEMA["target_fields"],
        invariant_fields=MULTIFAMILY_SCHEMA["invariant_fields"],
    )
    harness.execute_tool(
        "inspect_media", {"media_refs": ["source", "suspect"]}, "LOCAL-INSPECT"
    )
    params = {
        "source_ref": "source",
        "suspect_ref": "suspect",
        "comparison_scope_s": manifest["comparison_scope_s"],
    }
    names = [
        "detect_temporal_edit",
        "compare_audio_tracks",
        "compare_spatial_regions",
    ]
    return _compact_selection(
        harness, [_action(name, params, index) for index, name in enumerate(names, 1)]
    )


def _vcsl() -> dict[str, Any]:
    case = select_cases(20260818, 1)[0]
    manifest = vcsl_manifest(case, DEFAULT_FEATURE_ROOT)
    environment = VCSLFeatureEnvironment(
        case=case, feature_root=DEFAULT_FEATURE_ROOT, routing_variant="portfolio"
    )
    harness = VideoForensicHarness(
        _contract(
            manifest,
            VCSL_SCHEMA,
            {"query": "query", "reference": "reference", "comparison_scope_s": manifest["comparison_scope_s"]},
        ),
        environment,
        claim_schema=VCSL_SCHEMA,
        tool_registry=vcsl_tool_registry_for_variant("portfolio"),
        target_fields=VCSL_SCHEMA["target_fields"],
        invariant_fields=VCSL_SCHEMA["invariant_fields"],
    )
    harness.execute_tool(
        "inspect_pair_features",
        {"query_ref": "query", "reference_ref": "reference"},
        "LOCAL-INSPECT",
    )
    actions = [
        _action(
            spec["tool_name"],
            {
                "query_ref": "query",
                "reference_ref": "reference",
                "method": spec["method"],
            },
            index,
        )
        for index, spec in enumerate(LOCALIZER_SPECS.values(), 1)
    ]
    return _compact_selection(harness, actions)


def _viddiff() -> dict[str, Any]:
    item = next(
        json.loads(line)
        for line in DEFAULT_ADAPTIVE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = viddiff_manifest(item)
    environment = VidDiffAdaptiveMeasurementEnvironment(item, routing_variant="balanced")
    harness = VideoForensicHarness(
        _contract(
            manifest,
            VIDDIFF_SCHEMA,
            {"video_a": "video_a", "video_b": "video_b", "comparison_scope_s": manifest["comparison_scope_s"]},
        ),
        environment,
        claim_schema=VIDDIFF_SCHEMA,
        tool_registry=VIDDIFF_CAUSAL_TOOL_REGISTRY,
        target_fields=VIDDIFF_SCHEMA["target_fields"],
        invariant_fields=VIDDIFF_SCHEMA["invariant_fields"],
    )
    harness.execute_tool(
        "inspect_comparison_task", {"sample_ref": "current_sample"}, "LOCAL-INSPECT"
    )
    by_name = {entry["name"]: entry for entry in item["candidate_tools"]}
    actions = [
        _action(name, by_name[name]["arguments"], index)
        for index, name in enumerate(environment.eligible_tool_names(), 1)
    ]
    return _compact_selection(harness, actions)


def audit() -> dict[str, Any]:
    datasets = {
        "VideoSham": _videosham(),
        "VCSL": _vcsl(),
        "VidDiffBench": _viddiff(),
    }
    return {
        "artifact": "real_environment_preaction_frozen_branch_execution_v2",
        "scope": "one_existing_local_case_per_dataset_no_teacher_no_oracle",
        "datasets": datasets,
        "all_passed": all(
            row["proposal_coverage_passed"]
            and row["selection_frozen_before_branch_execution"]
            and not row["hindsight_outcome_used_for_selection"]
            and row["selected_branch_execution_admissible"]
            and row["parent_call_counts_unchanged_by_preview"]
            and row["deterministic_replay_match"]
            and sum(row["committed_call_count_delta"].values()) == 1
            for row in datasets.values()
        ),
    }


def main() -> None:
    report = audit()
    output = (
        ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1"
        / "real_environment_branch_execution.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
