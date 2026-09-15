"""Attack real tool interfaces with tool-swappable, keyword-stuffed rationales."""

from __future__ import annotations

import json
from typing import Any

from audit_method_interface_distinctiveness import _names
from causal_trajectory_generation import (
    _normalized_concept_phrases,
    _normalized_terms,
    prospective_method_fit_audit,
)
from llm_teacher_rollout import TeacherAction, teacher_tool_schemas
from pipeline import ROOT, write_json
from vcsl_batch import VCSL_CAUSAL_TOOL_REGISTRY, VCSL_SCHEMA
from viddiff_batch import VIDDIFF_CAUSAL_TOOL_REGISTRY, VIDDIFF_SCHEMA
from videosham_batch import MULTIFAMILY_CAUSAL_TOOL_REGISTRY, MULTIFAMILY_SCHEMA


def _action(tool_name: str, reasoning: str, suffix: str) -> TeacherAction:
    return TeacherAction(
        response_id=f"response-{suffix}",
        call_id=f"call-{suffix}",
        tool_name=tool_name,
        reasoning=reasoning,
        params={"student_visible_ref": "current"},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )


def _dataset_attack(
    schema: dict[str, Any],
    registry: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any]:
    tools = [
        tool
        for tool in teacher_tool_schemas(schema, registry)
        if str(tool.get("name")) in names
    ]
    selected, alternative = tools[:2]
    all_terms = {
        str(tool["name"]): _normalized_terms(str(tool.get("description") or ""))
        for tool in tools
    }
    all_phrases = {
        str(tool["name"]): _normalized_concept_phrases(
            str(tool.get("description") or "")
        )
        for tool in tools
    }

    def distinctive(name: str) -> list[str]:
        others = set().union(
            *(terms for other, terms in all_terms.items() if other != name),
            set(),
        )
        return sorted(all_terms[name] - others)

    def distinctive_phrases(name: str) -> list[str]:
        others = set().union(
            *(phrases for other, phrases in all_phrases.items() if other != name),
            set(),
        )
        unique_terms = set(distinctive(name))
        return sorted(
            phrase
            for phrase in all_phrases[name] - others
            if len(_normalized_terms(phrase).intersection(unique_terms)) >= 2
        )

    selected_terms = distinctive(str(selected["name"]))[:4]
    alternative_terms = distinctive(str(alternative["name"]))[:4]
    selected_phrase = distinctive_phrases(str(selected["name"]))[0]
    alternative_phrase = distinctive_phrases(str(alternative["name"]))[0]
    input_items = [
        {
            "role": "user",
            "content": json.dumps(
                {"student_visible_diagnostics": {"diagnostic_signal": 1.0}}
            ),
        }
    ]
    stuffed_reasoning = (
        "Given diagnostic signal 1.0, because "
        + selected_phrase
        + " "
        + " ".join(selected_terms)
        + " can test the visible gap, the alternative "
        + str(alternative["name"])
        + " with "
        + alternative_phrase
        + " "
        + " ".join(alternative_terms)
        + " could also test it; these mechanisms are all potentially relevant."
    )
    directional_reasoning = (
        "Given diagnostic signal 1.0, because "
        + selected_phrase
        + " can test the visible gap, I select this method, unlike "
        + str(alternative["name"])
        + ", whose "
        + alternative_phrase
        + " target a different failure mode."
    )
    stuffed = prospective_method_fit_audit(
        _action(str(selected["name"]), stuffed_reasoning, "stuffed"),
        input_items=input_items,
        tools=tools,
    )
    directional = prospective_method_fit_audit(
        _action(str(selected["name"]), directional_reasoning, "directional"),
        input_items=input_items,
        tools=tools,
    )
    old_components = {
        name: passed
        for name, passed in stuffed["components"].items()
        if name != "counterfactual_tool_swap_specificity"
    }
    return {
        "tool_count": len(tools),
        "selected_tool": selected["name"],
        "alternative_tool": alternative["name"],
        "keyword_stuffing_attack": {
            "old_five_components_all_pass": all(old_components.values()),
            "counterfactual_tool_swap_passed": stuffed[
                "counterfactual_tool_swap_passed"
            ],
            "new_contract_rejects": stuffed["score"] < 1.0,
            "failures": stuffed["failures"],
        },
        "directional_control": {
            "counterfactual_tool_swap_passed": directional[
                "counterfactual_tool_swap_passed"
            ],
            "new_contract_passes": directional["score"] == 1.0,
            "directional_contrast": directional["directional_contrast"],
        },
    }


def audit() -> dict[str, Any]:
    datasets = {
        "VideoSham": _dataset_attack(
            MULTIFAMILY_SCHEMA,
            MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
            [
                "detect_temporal_edit",
                "compare_audio_tracks",
                "compare_spatial_regions",
            ],
        ),
        "VCSL": _dataset_attack(
            VCSL_SCHEMA,
            VCSL_CAUSAL_TOOL_REGISTRY,
            _names(VCSL_CAUSAL_TOOL_REGISTRY, "localize_copy_segment"),
        ),
        "VidDiffBench": _dataset_attack(
            VIDDIFF_SCHEMA,
            VIDDIFF_CAUSAL_TOOL_REGISTRY,
            _names(VIDDIFF_CAUSAL_TOOL_REGISTRY, "compare_"),
        ),
    }
    all_passed = all(
        row["keyword_stuffing_attack"]["old_five_components_all_pass"]
        and row["keyword_stuffing_attack"]["new_contract_rejects"]
        and not row["keyword_stuffing_attack"]["counterfactual_tool_swap_passed"]
        and row["directional_control"]["new_contract_passes"]
        and row["directional_control"]["counterfactual_tool_swap_passed"]
        for row in datasets.values()
    )
    return {
        "audit_id": "COUNTERFACTUAL-METHOD-SPECIFICITY-V1",
        "scope": "real student-visible tool descriptions; synthetic neutral preaction diagnostic; no execution, teacher, oracle, or verifier",
        "datasets": datasets,
        "all_passed": all_passed,
    }


def main() -> None:
    report = audit()
    output = (
        ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1"
        / "counterfactual_method_specificity.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
