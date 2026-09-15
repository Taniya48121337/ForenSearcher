"""Audit whether every evidence tool exposes a distinguishable public mechanism."""

from __future__ import annotations

import json
from typing import Any

from causal_trajectory_generation import _normalized_concept_phrases, _normalized_terms
from llm_teacher_rollout import teacher_tool_schemas
from pipeline import ROOT, write_json
from vcsl_batch import VCSL_CAUSAL_TOOL_REGISTRY, VCSL_SCHEMA
from viddiff_batch import VIDDIFF_CAUSAL_TOOL_REGISTRY, VIDDIFF_SCHEMA
from videosham_batch import (
    MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
    MULTIFAMILY_SCHEMA,
)


def _names(registry: list[dict[str, Any]], prefix: str) -> list[str]:
    return [str(item["name"]) for item in registry if str(item["name"]).startswith(prefix)]


def _dataset_audit(
    schema: dict[str, Any],
    registry: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any]:
    visible = {
        str(item["name"]): item
        for item in teacher_tool_schemas(schema, registry)
        if str(item["name"]) in names
    }
    terms = {
        name: _normalized_terms(str(item.get("description") or ""))
        for name, item in visible.items()
    }
    phrases = {
        name: _normalized_concept_phrases(str(item.get("description") or ""))
        for name, item in visible.items()
    }
    rows = []
    for name in names:
        other_terms: set[str] = set()
        for other_name, other in terms.items():
            if other_name != name:
                other_terms.update(other)
        distinctive = sorted(terms.get(name, set()) - other_terms)
        other_phrases: set[str] = set()
        for other_name, other in phrases.items():
            if other_name != name:
                other_phrases.update(other)
        distinctive_phrases = sorted(phrases.get(name, set()) - other_phrases)
        rows.append(
            {
                "tool": name,
                "description_present": bool(
                    str((visible.get(name) or {}).get("description") or "").strip()
                ),
                "distinctive_concept_count": len(distinctive),
                "distinctive_concepts": distinctive,
                "distinctive_mechanism_phrase_count": len(distinctive_phrases),
                "distinctive_mechanism_phrases": distinctive_phrases,
            }
        )
    return {
        "tool_count": len(rows),
        "all_tools_have_public_descriptions": all(
            row["description_present"] for row in rows
        ),
        "all_tools_have_distinctive_mechanism_concepts": all(
            row["distinctive_concept_count"] >= 2 for row in rows
        ) and all(
            row["distinctive_mechanism_phrase_count"] >= 1 for row in rows
        ),
        "tools": rows,
    }


def audit() -> dict[str, Any]:
    datasets = {
        "VideoSham": _dataset_audit(
            MULTIFAMILY_SCHEMA,
            MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
            [
                "detect_temporal_edit",
                "compare_audio_tracks",
                "compare_spatial_regions",
            ],
        ),
        "VCSL": _dataset_audit(
            VCSL_SCHEMA,
            VCSL_CAUSAL_TOOL_REGISTRY,
            _names(VCSL_CAUSAL_TOOL_REGISTRY, "localize_copy_segment"),
        ),
        "VidDiffBench": _dataset_audit(
            VIDDIFF_SCHEMA,
            VIDDIFF_CAUSAL_TOOL_REGISTRY,
            _names(VIDDIFF_CAUSAL_TOOL_REGISTRY, "compare_"),
        ),
    }
    return {
        "artifact": "student_visible_method_interface_distinctiveness_v1",
        "scope": "all_current_evidence_tools_no_teacher_no_execution_no_oracle",
        "datasets": datasets,
        "all_passed": all(
            row["all_tools_have_public_descriptions"]
            and row["all_tools_have_distinctive_mechanism_concepts"]
            for row in datasets.values()
        ),
    }


def main() -> None:
    report = audit()
    output = (
        ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1"
        / "method_interface_distinctiveness.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
