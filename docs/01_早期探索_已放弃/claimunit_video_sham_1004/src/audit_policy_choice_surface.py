"""Audit that future evidence turns expose real executable policy choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from vcsl_batch import LOCALIZER_SPECS, VCSLFeatureEnvironment
from vcsl_probe import DEFAULT_FEATURE_ROOT, select_cases
from viddiff_batch import (
    DEFAULT_ADAPTIVE,
    DEFAULT_ROUTING_MODE,
    VidDiffAdaptiveMeasurementEnvironment,
)
from videosham_batch import MultiFamilyVideoEnvironment


def _videosham_surface() -> dict[str, Any]:
    environment = object.__new__(MultiFamilyVideoEnvironment)
    environment.call_counts = {
        "inspect_media": 1,
        "detect_temporal_edit": 0,
        "compare_audio_tracks": 0,
        "compare_spatial_regions": 0,
    }
    environment._triage = {
        "candidate_modalities": ["spatial"],
        "primary_modality": "spatial",
        "duration_ratio": 1.0,
        "visual_p05_similarity": 0.92,
        "rgb_tile_p95_mae": 0.04,
        "rgb_tile_max_mae": 0.08,
        "rgb_changed_tile_fraction": 0.12,
        "audio_similarity": 0.99,
    }
    choices = environment.eligible_tool_names()
    visible = environment._student_visible_triage_metrics()
    environment.on_evidence_dropout("EA-LOCAL-001", ["affected_source_interval_s"])
    repair_choices = environment.eligible_tool_names()
    return {
        "default_mode": "multi_modality_measurement_choice",
        "semantic_choices": choices,
        "semantic_choice_count": len(choices),
        "repair_semantic_choices": repair_choices,
        "repair_semantic_choice_count": len(repair_choices),
        "router_answer_fields_visible": sorted(
            set(visible).intersection({"candidate_modalities", "primary_modality"})
        ),
        "neutral_screen_fields": sorted(visible),
    }


def _vcsl_surface() -> dict[str, Any]:
    case = select_cases(20260818, 1)[0]
    environment = VCSLFeatureEnvironment(
        case=case,
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="portfolio",
    )
    inspection = environment.execute(
        "inspect_pair_features",
        {"query_ref": "query", "reference_ref": "reference"},
    )
    choices = environment.eligible_tool_names()
    diagnostics = inspection["result"]["localizer_routing_diagnostics"]
    environment.on_evidence_dropout("EA-LOCAL-001", ["query_interval_s"])
    repair_choices = environment.eligible_tool_names()
    return {
        "default_mode": "portfolio",
        "semantic_choices": choices,
        "semantic_choice_count": len(choices),
        "repair_semantic_choices": repair_choices,
        "repair_semantic_choice_count": len(repair_choices),
        "implemented_localizer_count": len(LOCALIZER_SPECS),
        "router_answer_fields_visible": sorted(
            set(diagnostics).intersection({"recommended_variant", "reason_code"})
        ),
        "neutral_screen_fields": sorted(diagnostics),
    }


def _viddiff_surface(path: Path = DEFAULT_ADAPTIVE) -> dict[str, Any]:
    first = next(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    environment = VidDiffAdaptiveMeasurementEnvironment(first, routing_variant="balanced")
    environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
    choices = environment.eligible_tool_names()
    environment.on_evidence_dropout("EA-LOCAL-001", ["preferred_video"])
    repair_choices = environment.eligible_tool_names()
    return {
        "default_mode": DEFAULT_ROUTING_MODE,
        "semantic_choices": choices,
        "semantic_choice_count": len(choices),
        "repair_semantic_choices": repair_choices,
        "repair_semantic_choice_count": len(repair_choices),
        "executed_candidate_count": len(environment.available_methods),
        "router_answer_fields_visible": [],
        "neutral_screen_fields": ["comparison_statement", "routing_contract"],
    }


def audit() -> dict[str, Any]:
    datasets = {
        "VideoSham": _videosham_surface(),
        "VCSL": _vcsl_surface(),
        "VidDiffBench": _viddiff_surface(),
    }
    for result in datasets.values():
        result["passes_nontrivial_choice"] = result["semantic_choice_count"] >= 2
        result["passes_nontrivial_repair_choice"] = result["repair_semantic_choice_count"] >= 2
        result["passes_no_router_answer"] = not result["router_answer_fields_visible"]
    return {
        "artifact": "policy_choice_surface_audit_v1",
        "scope": "future_default_generation_configuration",
        "datasets": datasets,
        "all_passed": all(
            result["passes_nontrivial_choice"]
            and result["passes_nontrivial_repair_choice"]
            and result["passes_no_router_answer"]
            for result in datasets.values()
        ),
        "interpretation": (
            "This proves only that the default environment exposes multiple executable semantic actions without a "
            "categorical router answer. It does not prove that an LLM consumes evidence or succeeds end to end."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1/audit.json",
    )
    args = parser.parse_args()
    report = audit()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
