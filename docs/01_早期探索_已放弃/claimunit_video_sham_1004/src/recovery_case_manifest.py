"""Manifest and environment adapters for case-disjoint causal-recovery batches."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from pipeline import ROOT, decode, write_json
from run_student_closed_loop_canary import _viddiff_item
from shadow_execute_generated_actions import _jsonl, _restore_harness
from unannounced_repair_interventions import INTERVENTIONS
from vcsl_batch import VCSL_TOOL_REGISTRY, VCSLFeatureEnvironment
from vcsl_probe import DEFAULT_FEATURE_ROOT
from viddiff_batch import (
    VIDDIFF_TOOL_REGISTRY,
    VidDiffAdaptiveMeasurementEnvironment,
)
from videosham_batch import (
    EDITED_ROOT,
    MULTIFAMILY_TOOL_REGISTRY,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
)


DEFAULT_SOURCES = {
    "VideoSham": ROOT
    / "outputs/batch/videosham_seed_20260818_n10/sft_clean_accepted.jsonl",
    "VCSL": ROOT
    / "outputs/batch/vcsl_seed_20260818_n10/sft_clean_accepted.jsonl",
    "VidDiffBench": ROOT
    / "outputs/batch/viddiff_verified_seed_20260818_accepted_8/sft_clean_accepted.jsonl",
}

DEFAULT_MANIFEST = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_manifest_v1/manifest.json"
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _terminal_checkpoints(path: Path) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    candidates = [
        row
        for row in rows
        if row.get("metadata", {}).get("target_tool") == "stop"
        and row.get("metadata", {}).get("target_execution_status") == "accepted"
    ]
    by_case: dict[str, dict[str, Any]] = {}
    for row in candidates:
        case_id = str(row["metadata"]["case_id"])
        if case_id in by_case:
            raise ValueError(f"multiple accepted terminal checkpoints for {case_id}")
        by_case[case_id] = row
    return [by_case[case_id] for case_id in sorted(by_case)]


def _case_entities(dataset: str, case_id: str) -> set[str]:
    if dataset == "VCSL":
        match = re.fullmatch(r"vcsl-([0-9a-f]{32})-([0-9a-f]{32})", case_id)
        if match is None:
            raise ValueError(f"invalid VCSL case ID: {case_id}")
        return {match.group(1), match.group(2)}
    return {case_id}


def _maximum_entity_disjoint_rows(
    dataset: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select the largest deterministic subset with no shared media entity."""

    ordered = sorted(rows, key=lambda row: str(row["metadata"]["case_id"]))
    best: list[dict[str, Any]] = []

    def visit(index: int, used: set[str], selected: list[dict[str, Any]]) -> None:
        nonlocal best
        if len(selected) + len(ordered) - index <= len(best):
            return
        if index == len(ordered):
            if len(selected) > len(best):
                best = selected.copy()
            return
        row = ordered[index]
        entities = _case_entities(dataset, str(row["metadata"]["case_id"]))
        if entities.isdisjoint(used):
            selected.append(row)
            visit(index + 1, used | entities, selected)
            selected.pop()
        visit(index + 1, used, selected)

    visit(0, set(), [])
    return best


def build_manifest(
    sources: dict[str, Path | list[Path]] | None = None,
    *,
    exclude_case_ids: set[str] | None = None,
    enforce_entity_disjoint: bool = False,
) -> dict[str, Any]:
    sources = sources or DEFAULT_SOURCES
    excluded = {item.lower() for item in (exclude_case_ids or set())}
    interventions = {
        dataset: [item for item in INTERVENTIONS if item["dataset"] == dataset]
        for dataset in sources
    }
    jobs: list[dict[str, Any]] = []
    for dataset, source_value in sources.items():
        paths = source_value if isinstance(source_value, list) else [source_value]
        row_source: dict[str, tuple[dict[str, Any], Path]] = {}
        for path in paths:
            for row in _terminal_checkpoints(path):
                case_id = str(row["metadata"]["case_id"])
                if case_id.lower() in excluded:
                    continue
                # Source order is a declared checkpoint-quality priority.  It
                # never inspects oracle labels or rubric scores.
                row_source.setdefault(case_id, (row, path))
        rows = [row_source[case_id][0] for case_id in sorted(row_source)]
        if enforce_entity_disjoint:
            rows = _maximum_entity_disjoint_rows(dataset, rows)
        family = interventions[dataset]
        for index, row in enumerate(rows):
            metadata = row["metadata"]
            path = row_source[str(metadata["case_id"])][1]
            intervention = family[index % len(family)]
            case_id = str(metadata["case_id"])
            jobs.append(
                {
                    # Keep the public trajectory identifier opaque.  The private
                    # manifest already carries case and intervention provenance;
                    # repeating the operator name in an SFT row would reveal why
                    # the teacher challenged the evidence.
                    "episode_id": f"batch-recovery-{len(jobs) + 1:03d}",
                    "dataset": dataset,
                    "case_id": case_id,
                    "checkpoint": {
                        "file": _relative(path),
                        "trajectory_id": metadata.get("trajectory_id"),
                        "turn_index": metadata.get("turn_index"),
                        "target_tool": "stop",
                    },
                    "intervention": copy.deepcopy(intervention),
                }
            )
    return {
        "artifact": "causal_recovery_case_manifest_v1",
        "status": "development_manifest_not_sealed",
        "selection": (
            "All locally available accepted clean terminal checkpoints from the named "
            "three batch artifacts after declared exclusions; one intervention family "
            "is assigned round-robin within each dataset without reading oracle labels."
        ),
        "oracle_fields_included": False,
        "private_generation_manifest": True,
        "excluded_case_ids": sorted(excluded),
        "entity_disjoint_selection": enforce_entity_disjoint,
        "datasets": {
            dataset: sum(job["dataset"] == dataset for job in jobs)
            for dataset in sources
        },
        "jobs": jobs,
    }


def write_default_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = build_manifest()
    write_json(path, manifest)
    return manifest


def load_checkpoint(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = job["checkpoint"]
    path = ROOT / str(checkpoint["file"])
    matches = [
        row
        for row in _jsonl(path)
        if str(row.get("metadata", {}).get("case_id")) == str(job["case_id"])
        and row.get("metadata", {}).get("trajectory_id")
        == checkpoint.get("trajectory_id")
        and row.get("metadata", {}).get("turn_index")
        == checkpoint.get("turn_index")
        and row.get("metadata", {}).get("target_tool") == "stop"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint selector for {job['case_id']} matched {len(matches)} rows"
        )
    return normalize_checkpoint_row(matches[0])


def normalize_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project full-trajectory SFT into the checkpoint shape used by restoration."""

    payload = json.loads(row["messages"][1]["content"])
    if "current_harness_state" in payload:
        return copy.deepcopy(row)
    state = None
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        result = json.loads(content) if isinstance(content, str) else content
        if isinstance(result, dict) and isinstance(result.get("harness_state"), dict):
            state = copy.deepcopy(result["harness_state"])
            break
    if state is None:
        state = copy.deepcopy(payload.get("harness_state"))
    if not isinstance(state, dict):
        raise ValueError("full-trajectory checkpoint contains no Harness state")
    task_contract = payload.get("task_contract") or state.get("task_contract") or {}
    normalized_payload = {
        "task": {
            "instruction": str(task_contract.get("task") or ""),
            "claim_schema": copy.deepcopy(payload["claim_schema"]),
        },
        "current_harness_state": state,
    }
    normalized = copy.deepcopy(row)
    normalized["messages"] = [
        copy.deepcopy(row["messages"][0]),
        {
            "role": "user",
            "content": json.dumps(
                normalized_payload, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]
    return normalized


def _claim_value(state: dict[str, Any], field_name: str) -> Any:
    claims = state.get("candidate_claims") or []
    if len(claims) != 1:
        raise ValueError("terminal checkpoint must contain exactly one claim")
    return copy.deepcopy(claims[0]["fields"][field_name]["value"])


def _videosham_paths(case_id: str) -> tuple[Path, Path]:
    match = re.fullmatch(r"videosham-(\d+)", case_id)
    if match is None:
        raise ValueError(f"invalid VideoSham case ID: {case_id}")
    number = match.group(1)
    return (
        SOURCE_ROOT / f"{number}_unedited.mp4",
        EDITED_ROOT / f"{number}_processed.mp4",
    )


def _vcsl_ids(case_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"vcsl-([0-9a-f]{32})-([0-9a-f]{32})", case_id)
    if match is None:
        raise ValueError(f"invalid VCSL case ID: {case_id}")
    return match.group(1), match.group(2)


def _viddiff_variant(state: dict[str, Any]) -> str:
    for atom in (state.get("evidence_store") or {}).values():
        routing = (atom.get("observation") or {}).get("routing_state") or {}
        variant = (routing.get("routing_contract") or {}).get("variant")
        if variant in VidDiffAdaptiveMeasurementEnvironment.ROUTING_VARIANTS:
            return str(variant)
    return "balanced"


def preflight_job(job: dict[str, Any]) -> dict[str, Any]:
    row = load_checkpoint(job)
    harness = _restore_harness(row)
    state = harness.render_state()
    dataset = str(job["dataset"])
    reasons: list[str] = []
    if state.get("open_high_priority_obligations"):
        reasons.append("checkpoint_has_open_obligations")
    if dataset == "VideoSham":
        source, suspect = _videosham_paths(str(job["case_id"]))
        if not source.exists():
            reasons.append("source_video_missing")
        if not suspect.exists():
            reasons.append("suspect_video_missing")
    elif dataset == "VCSL":
        query, reference = _vcsl_ids(str(job["case_id"]))
        if not (DEFAULT_FEATURE_ROOT / f"{query}.npy").exists():
            reasons.append("query_feature_missing")
        if not (DEFAULT_FEATURE_ROOT / f"{reference}.npy").exists():
            reasons.append("reference_feature_missing")
    elif dataset == "VidDiffBench":
        sample_key = str(job["case_id"]).removeprefix("viddiff-")
        try:
            _viddiff_item(sample_key)
        except ValueError:
            reasons.append("student_visible_candidate_measurements_missing")
    else:
        reasons.append("unsupported_dataset")
    return {
        "episode_id": job["episode_id"],
        "dataset": dataset,
        "case_id": job["case_id"],
        "passed": not reasons,
        "reasons": reasons,
    }


def build_job(job: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    row = load_checkpoint(job)
    harness = _restore_harness(row)
    state = harness.render_state()
    dataset = str(job["dataset"])
    case_id = str(job["case_id"])
    if dataset == "VideoSham":
        source_path, suspect_path = _videosham_paths(case_id)
        scope = _claim_value(state, "comparison_scope_s")
        environment = MultiFamilyVideoEnvironment(
            manifest={
                "case_id": case_id,
                "source_id": _claim_value(state, "source_id"),
                "comparison_scope_s": scope,
            },
            source_frames=decode(source_path),
            suspect_frames=decode(suspect_path),
            source_path=source_path,
            suspect_path=suspect_path,
        )
        registry = MULTIFAMILY_TOOL_REGISTRY
    elif dataset == "VCSL":
        query_id, reference_id = _vcsl_ids(case_id)
        environment = VCSLFeatureEnvironment(
            case={
                "pair_id": case_id,
                "query_id": query_id,
                "reference_id": reference_id,
            },
            feature_root=DEFAULT_FEATURE_ROOT,
            routing_variant="portfolio",
        )
        registry = VCSL_TOOL_REGISTRY
    elif dataset == "VidDiffBench":
        sample_key = case_id.removeprefix("viddiff-")
        environment = VidDiffAdaptiveMeasurementEnvironment(
            _viddiff_item(sample_key), routing_variant=_viddiff_variant(state)
        )
        registry = VIDDIFF_TOOL_REGISTRY
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return environment, registry, row


def compile_spec(job: dict[str, Any]) -> dict[str, Any]:
    return {
        **copy.deepcopy(job["intervention"]),
        "episode_id": str(job["episode_id"]),
        "case_id": str(job["case_id"]),
        "checkpoint": copy.deepcopy(job["checkpoint"]),
    }


def main() -> None:
    manifest = write_default_manifest()
    preflight = [preflight_job(job) for job in manifest["jobs"]]
    report = {
        "artifact": "causal_recovery_case_manifest_preflight_v1",
        "jobs": len(preflight),
        "passed": sum(item["passed"] for item in preflight),
        "failed": sum(not item["passed"] for item in preflight),
        "results": preflight,
    }
    write_json(DEFAULT_MANIFEST.parent / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
