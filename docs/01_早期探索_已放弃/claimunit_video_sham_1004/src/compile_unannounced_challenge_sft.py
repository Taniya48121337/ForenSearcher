"""Compile executed, reduced-control SFT targets for unannounced evidence challenges.

The private intervention operator is used only to construct and verify a local
counterfactual.  It is never placed in the student-visible row.  Every positive
target names active EvidenceAtoms, selects a generic defect class, and is
executed by the real Harness before export.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from compile_evidence_first_sft import sanitize_payload
from causal_trajectory_generation import CAUSAL_SYSTEM_PROMPT
from pipeline import ROOT, write_json, write_jsonl
from run_student_closed_loop_canary import (
    _vcsl_environment,
    _video_environment,
    _viddiff_item,
)
from run_unannounced_repair_falsification import (
    DEFAULT_SFT,
    STOP_ROWS,
    _active_registry,
    _builder,
)
from shadow_execute_generated_actions import _jsonl, _restore_harness
from student_tool_contract import build_student_tools
from unannounced_repair_interventions import INTERVENTIONS, apply_intervention


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/causal_challenge_curriculum_v1"
)

OPERATOR_ACTION = {
    "active_wrong_interval": ("scope_mismatch", "remeasure_evidence"),
    "role_swap": ("role_mismatch", "remeasure_evidence"),
    "confidence_rationale_contradiction": (
        "value_rationale_conflict",
        "remeasure_evidence",
    ),
    "incompatible_binding": ("contradictory_observations", "remeasure_evidence"),
    "conflicting_active": ("contradictory_observations", "remeasure_evidence"),
    "stale_active": ("stale_or_superseded", "remeasure_evidence"),
    "new_result_conflict": ("contradictory_observations", "remeasure_evidence"),
    "partial_interval": ("incomplete_observation", "remeasure_evidence"),
    "empty_result": ("empty_observation", "remeasure_evidence"),
    "insufficient_evidence": ("incomplete_observation", "remeasure_evidence"),
}


def _changed_evidence_ids(
    before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    left = before.get("evidence_store") or {}
    right = after.get("evidence_store") or {}
    return sorted(
        evidence_id
        for evidence_id, atom in right.items()
        if evidence_id not in left or atom != left[evidence_id]
    )


def _binding_evidence_ids(
    before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    if before.get("field_bindings") == after.get("field_bindings"):
        return []
    result: set[str] = set()
    for fields in (after.get("field_bindings") or {}).values():
        for binding in fields.values():
            result.update(str(item) for item in binding.get("support_ids") or [])
            result.update(str(item) for item in binding.get("attack_ids") or [])
    return sorted(result)


def _fallback_measurement_ids(state: dict[str, Any]) -> list[str]:
    invariant_kinds = {
        "paired_media_identity",
        "case_scope_locator",
        "feature_pair_identity",
        "comparison_task_identity",
    }
    return sorted(
        evidence_id
        for evidence_id, atom in (state.get("evidence_store") or {}).items()
        if atom.get("active") is True
        and atom.get("kind") not in invariant_kinds
        and (atom.get("supports_fields") or [])
    )


def challenged_evidence_ids(
    before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    candidates = (
        _changed_evidence_ids(before, after)
        or _binding_evidence_ids(before, after)
        or _fallback_measurement_ids(after)
    )
    active = after.get("evidence_store") or {}
    return [
        evidence_id
        for evidence_id in candidates
        if evidence_id in active and active[evidence_id].get("active") is True
    ]


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _claim_values(state: dict[str, Any]) -> dict[str, Any]:
    claims = state.get("candidate_claims") or []
    if not claims:
        return {}
    return {
        str(name): copy.deepcopy(field.get("value"))
        for name, field in (claims[0].get("fields") or {}).items()
    }


def _visible_atom_summary(atom: dict[str, Any]) -> str:
    observation = atom.get("observation") or {}
    provenance = atom.get("provenance") or {}
    preferred_keys = (
        "source_role",
        "suspect_role",
        "affected_source_interval_s",
        "affected_suspect_interval_s",
        "suspect_comparison_scope_s",
        "preferred_video",
        "video_a_interval_s",
        "video_b_interval_s",
        "relationship",
        "copied_segments_frames",
        "localization_method",
        "measurement_method",
        "segments",
        "confidence",
        "order_consistent",
        "visible_rationale",
    )
    def display(key: str, value: Any) -> str:
        if key == "visible_rationale" and isinstance(value, str):
            compact = " ".join(value.split())
            # The challenge needs the semantic cue, not a verbatim replay of a
            # paragraph already visible in state.  Keep the first concrete
            # proposition so long rationales do not drown out the prospective
            # decision or create artificial sentence-count failures.
            compact = re.split(r"[.!?]\s+", compact, maxsplit=1)[0]
            if len(compact) > 180:
                compact = compact[:177].rstrip() + "..."
            value = compact
        if key == "copied_segments_frames" and isinstance(value, list) and len(value) > 3:
            value = {
                "segment_count": len(value),
                "first_box": value[0],
                "last_box": value[-1],
            }
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if key == "preferred_video" and value == "c":
            return f'{rendered} [abstention: neither/equal/insufficient evidence]'
        return rendered

    observation_facts = [
        f"{key}={display(key, observation[key])}"
        for key in preferred_keys
        if key in observation
    ]
    if not observation_facts:
        observation_facts = [
            f"{key}={display(key, value)}"
            for key, value in list(observation.items())[:4]
        ]
    facts = [
        f"kind={json.dumps(atom.get('kind'), ensure_ascii=False)}",
        "supports_fields="
        f"{json.dumps(atom.get('supports_fields') or [], ensure_ascii=False, separators=(',', ':'))}",
        *observation_facts,
    ]
    for key in ("execution_sequence", "supersedes_execution_sequence", "method"):
        if key in provenance:
            facts.append(
                f"provenance.{key}="
                f"{json.dumps(provenance[key], ensure_ascii=False, separators=(',', ':'))}"
            )
    return ", ".join(facts)


def challenge_reasoning(
    *,
    state: dict[str, Any],
    evidence_ids: list[str],
    defect_code: str,
    target_fields: list[str] | None = None,
    style: int = 0,
) -> str:
    atoms = state.get("evidence_store") or {}
    atom_facts = "; ".join(
        f"{evidence_id} reports {_visible_atom_summary(atoms[evidence_id])}"
        for evidence_id in evidence_ids
    )
    target_fields = target_fields or []
    incompatible_edges = [
        f"{evidence_id} advertises only "
        f"{json.dumps(atoms[evidence_id].get('supports_fields') or [], ensure_ascii=False, separators=(',', ':'))}"
        for evidence_id in evidence_ids
        if target_fields
        and not set(target_fields).issubset(
            set(atoms[evidence_id].get("supports_fields") or [])
        )
    ]
    explanations = {
        "scope_mismatch": (
            "The reported interval placement does not form a trustworthy match to the "
            "current visible comparison scope."
        ),
        "role_mismatch": (
            "The source/suspect roles encoded by the evidence do not agree with the "
            "roles used by the current task."
        ),
        "value_rationale_conflict": (
            "The numeric conclusion and its visible rationale point in incompatible "
            "directions, so confidence cannot validate the value."
        ),
        "contradictory_observations": (
            "The active evidence content is incompatible with the current claim or "
            "another active observation."
        ),
        "incompatible_binding": (
            f"The support edge is invalid because {'; '.join(incompatible_edges)}; "
            f"that evidence may remain valid for its own fields, but it cannot justify "
            f"{', '.join(target_fields)}."
        ),
        "stale_or_superseded": (
            "The active observations encode competing versions, so recency alone is "
            "not a valid reason to retain either one."
        ),
        "incomplete_observation": (
            "The observation omits values needed for a complete paired conclusion."
        ),
        "empty_observation": (
            "The observation contains no usable comparison segment and cannot support "
            "a committed claim."
        ),
    }
    explanation_key = "incompatible_binding" if incompatible_edges else defect_code
    explanation = explanations[explanation_key]
    action = (
        "I will challenge only the cited evidence-field links and request a fresh "
        "measurement before any new binding or stop."
    )
    variants = (
        f"The current evidence requires verification: {atom_facts}. {explanation} {action}",
        f"{explanation} The relevant visible records are {atom_facts}. {action}",
        f"Before preserving the current support, compare {atom_facts}. {explanation} {action}",
        f"A local integrity check is needed: {atom_facts}. {explanation} {action}",
    )
    return variants[style % len(variants)]


def _schema_for(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [
        (tool.get("function") or {}).get("parameters") or {}
        for tool in tools
        if (tool.get("function") or {}).get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"tool schema is not unique: {name}")
    return matches[0]


def compile_candidates(
    source_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sft_rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    for index, spec in enumerate(INTERVENTIONS, start=1):
        environment, registry = _builder(spec["dataset"], source_rows)
        harness = _restore_harness(source_rows[STOP_ROWS[spec["dataset"]]])
        harness.environment = environment
        harness.tool_registry = _active_registry(registry)
        before = harness.render_state()
        private_metadata = apply_intervention(harness, spec)
        preaction_state = harness.render_state()
        evidence_ids = challenged_evidence_ids(before, preaction_state)
        if not evidence_ids:
            raise ValueError(f"no visible evidence can be challenged for {spec['id']}")
        defect_code, requested_check = OPERATOR_ACTION[spec["operator"]]
        target_fields = [
            str(field_name)
            for field_name in private_metadata.get(
                "affected_fields_verifier_only", []
            )
        ]
        if not target_fields:
            raise ValueError(f"no local target fields for {spec['id']}")
        reasoning = challenge_reasoning(
            state=preaction_state,
            evidence_ids=evidence_ids,
            defect_code=defect_code,
            target_fields=target_fields,
            style=index - 1,
        )
        arguments = {
            "evidence_ids": evidence_ids,
            "defect_codes": [defect_code],
            "target_fields": target_fields,
            "requested_check": requested_check,
        }
        eligible = [str(item) for item in preaction_state.get("eligible_tools") or []]
        if "challenge_evidence" not in eligible:
            raise ValueError(f"challenge_evidence is not eligible for {spec['id']}")
        tools = build_student_tools(
            harness.claim_schema,
            harness.tool_registry,
            eligible,
            executable_actions=[],
        )
        errors = sorted(
            Draft202012Validator(_schema_for(tools, "challenge_evidence")).iter_errors(
                arguments
            ),
            key=lambda item: list(item.path),
        )
        if errors:
            raise ValueError(f"challenge target is invalid for {spec['id']}: {errors[0].message}")

        execution = harness.execute_tool(
            "challenge_evidence",
            {"reasoning": reasoning, **copy.deepcopy(arguments)},
            f"CALL-CHALLENGE-CURRICULUM-{index:03d}",
        )
        if execution.get("status") != "accepted":
            raise ValueError(f"challenge execution failed for {spec['id']}")

        payload = sanitize_payload(
            {
                "task": {
                    "instruction": harness.contract.task,
                    "claim_schema": copy.deepcopy(harness.claim_schema),
                },
                "current_harness_state": preaction_state,
                "trajectory_contract": (
                    "Detect evidence integrity failures from visible content. No error "
                    "location, repair recipe, or correct next tool is supplied."
                ),
            }
        )
        row_id = f"causal-challenge-{index:03d}"
        visible_blob = json.dumps(payload, ensure_ascii=False)
        forbidden_values = [
            str(spec["id"]),
            str(spec["operator"]),
            "affected_fields_verifier_only",
        ]
        leakage_hits = [value for value in forbidden_values if value in visible_blob]
        row = {
            "messages": [
                {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
                {
                    "role": "assistant",
                    "content": reasoning,
                    "tool_calls": [
                        {
                            "id": f"CALL-CHALLENGE-CURRICULUM-{index:03d}",
                            "type": "function",
                            "function": {
                                "name": "challenge_evidence",
                                "arguments": copy.deepcopy(arguments),
                            },
                        }
                    ],
                },
            ],
            "tools": tools,
            "metadata": {
                "row_id": row_id,
                "dataset": spec["dataset"],
                "case_id": str(harness.contract.contract_id),
                "trajectory_mode": "unannounced_integrity_challenge",
                "target_tool": "challenge_evidence",
                "policy_view": "evidence_first_reduced_control_v1",
                "loss_scope": "last_assistant_action_only",
                "supervision_class": "uncertainty_or_repair_detection",
                "supervision_subtype": "belief_repair",
                "learning_value_weight": 1.2,
                "target_was_executed": True,
                "private_intervention_visible": False,
                "compiled_actions_visible": False,
                "repair_recipe_visible": False,
                "structural_candidate_only": True,
            },
        }
        sft_rows.append(row)
        sidecars.append(
            {
                "row_id": row_id,
                "intervention_id": spec["id"],
                "operator": spec["operator"],
                "dataset": spec["dataset"],
                "private_generation_metadata": private_metadata,
                "challenged_evidence_ids": evidence_ids,
                "defect_code": defect_code,
                "target_fields": target_fields,
                "requested_check": requested_check,
                "preaction_eligible_tools": eligible,
                "target_schema_valid": not errors,
                "target_executed": execution.get("status") == "accepted",
                "postaction_open_challenges": (
                    execution.get("harness_state") or {}
                ).get("open_evidence_challenges"),
                "student_visible_leakage_hits": leakage_hits,
            }
        )

    summary = {
        "artifact": "unannounced_evidence_challenge_sft_candidate_v1",
        "artifact_status": "structural_candidate_only_not_training_qualified",
        "rows": len(sft_rows),
        "datasets": {
            dataset: sum(row["metadata"]["dataset"] == dataset for row in sft_rows)
            for dataset in ("VideoSham", "VCSL", "VidDiffBench")
        },
        "target_schema_valid": sum(item["target_schema_valid"] for item in sidecars),
        "target_executed": sum(item["target_executed"] for item in sidecars),
        "student_visible_leakage_rows": sum(
            bool(item["student_visible_leakage_hits"]) for item in sidecars
        ),
        "qualification_blockers": [
            "deterministic_reasoning_requires_local_teacher_naturalization",
            "fresh_measurement_and_local_repair_suffix_not_yet_compiled",
            "closed_loop_student_gain_not_evaluated",
            "sealed_cross_dataset_evaluation_not_available",
        ],
    }
    return sft_rows, sidecars, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_rows = {
        str(row["metadata"]["row_id"]): row for row in _jsonl(args.sft)
    }
    rows, sidecars, summary = compile_candidates(source_rows)
    write_jsonl(args.output_root / "challenge_sft_candidate.jsonl", rows)
    write_jsonl(args.output_root / "private_intervention_sidecar.jsonl", sidecars)
    write_json(args.output_root / "compile_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
