"""Compile oracle-verified VidDiff measurements into executable CLI Harness trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cli_teacher_rollout import CodexCLITeacher
from causal_probe_suite import qualify_and_write_causal_packet
from evidence_choice_search import build_evidence_choice_teacher
from generation_admission import generation_admission_summary, generation_failure_metadata
from interactive_harness import EVIDENCE_INTEGRITY_TOOL_NAMES, EvidenceAtom, with_evidence_integrity_tools
from llm_teacher_rollout import _write_bundle, build_llm_teacher_bundle
from pipeline import ROOT, write_json, write_jsonl
from sft_v2_sidecar import compile_transition_sidecar
from step_utility_auditor import apply_step_utility_filter, audit_step_utility
from unified_trajectory_rubric import (
    evaluate_bundle as evaluate_unified_quality,
    load_saved_bundle,
)


SELECTION_ROOT = ROOT / "outputs" / "selection" / "viddiff_best_of_tree_seed_20260818_n10"
DEFAULT_ACCEPTED = SELECTION_ROOT / "accepted_measurements.student.jsonl"
DEFAULT_ADAPTIVE = SELECTION_ROOT / "candidate_measurements.student.jsonl"
DEFAULT_AUDIT = SELECTION_ROOT / "selection_audit.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "batch"
DEFAULT_ROUTING_MODE = "adaptive"


REASONING = {
    "type": "string",
    "description": "Explain the visible evidence gap and why this exact next action closes it.",
}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _external_schema(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": _object_schema({"reasoning": REASONING, **properties}),
        "strict": True,
    }


VIDDIFF_SCHEMA = {
    "schema_id": "viddiff_comparative_measurement_claim_v1",
    "required_fields": [
        "sample_key",
        "comparison_statement",
        "preferred_video",
        "video_a_interval_s",
        "video_b_interval_s",
        "measurement_method",
    ],
    "invariant_fields": ["sample_key", "comparison_statement"],
    "target_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method"],
    "fields": {
        "sample_key": {
            "types": ["str"],
            "definition": "Visible VidDiff sample key.",
            "acceptable_evidence_kinds": ["comparison_task_identity"],
        },
        "comparison_statement": {
            "types": ["str"],
            "definition": "Exact visible statement being compared.",
            "acceptable_evidence_kinds": ["comparison_task_identity"],
        },
        "preferred_video": {
            "types": ["str"],
            "enum": ["a", "b", "c"],
            "definition": "a, b, or c for neither/equal/insufficient evidence.",
            "acceptable_evidence_kinds": ["action_comparison_measurement"],
        },
        "video_a_interval_s": {
            "types": ["list"],
            "definition": "Visible supporting local interval in video A.",
            "acceptable_evidence_kinds": ["action_comparison_measurement"],
        },
        "video_b_interval_s": {
            "types": ["list"],
            "definition": "Visible supporting local interval in video B.",
            "acceptable_evidence_kinds": ["action_comparison_measurement"],
        },
        "measurement_method": {
            "types": ["str"],
            "enum": [
                "compare_uniform_timeline",
                "compare_global_motion_symmetric",
                "compare_action_refined_symmetric",
                "compare_predicate_independent",
                "compare_symmetry_residual_abstention",
            ],
            "definition": "Executed label-blind evidence method; a method may measure prepared views or derive a role-invariant abstention from prior measurements.",
            "acceptable_evidence_kinds": ["action_comparison_measurement"],
        },
    },
}


PAIR_PROPERTIES = {
    "video_a_ref": {"type": "string", "enum": ["video_a"]},
    "video_b_ref": {"type": "string", "enum": ["video_b"]},
    "comparison_statement": {"type": "string"},
}


VIDDIFF_METHOD_DESCRIPTIONS = {
    "compare_uniform_timeline": (
        "Compare coarse, uniformly sampled chronological mosaics over the whole A/B "
        "scope. Use it for broad temporal coverage and an initial direction; treat "
        "brief or locally occluded action details as a known limitation."
    ),
    "compare_global_motion_symmetric": (
        "Compare whole-video motion strips in both A/B presentation orders. Use it to "
        "test whether a direction survives role swapping and global-motion emphasis; "
        "report order inconsistency instead of forcing a winner."
    ),
    "compare_action_refined_symmetric": (
        "Compare dense action-focused local strips in both A/B presentation orders. "
        "Use it to inspect brief, localized predicate-relevant behavior that coarse "
        "timeline or global-motion views can miss; report order inconsistency explicitly."
    ),
    "compare_predicate_independent": (
        "Record a typed predicate-specific observation for each video independently, then "
        "apply a deterministic label-blind comparator. Use exact event definitions, geometric "
        "variables, visible force/tension proxies, or universal-claim counterexamples as "
        "specified by the observation contract. It may abstain even when another route guesses "
        "a direction; use it to expose insufficient or semantically mismatched evidence."
    ),
}


TYPED_OBSERVATION_FIELDS = (
    "predicate_family",
    "metric_a",
    "metric_b",
    "usable_a",
    "usable_b",
    "reasons_a",
    "reasons_b",
    "independent_typed_records",
)


def _normalized_measurement_observation(
    raw: dict[str, Any],
    *,
    method_name: str,
    routing_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the student-visible facts needed to judge method fit.

    The explicit allowlist prevents verifier/oracle metadata from crossing into the
    executable environment while retaining predicate-specific proof obligations.
    """

    normalized = {
        "preferred_video": raw.get("preferred_video"),
        "video_a_interval_s": raw.get("video_a_interval_s"),
        "video_b_interval_s": raw.get("video_b_interval_s"),
        "measurement_method": method_name,
        "confidence": raw.get("confidence"),
        "measurement_id": raw.get("measurement_id"),
        "order_consistent": raw.get("order_consistent"),
        "decision_rule": raw.get("decision_rule", "direct_visual_comparison"),
        "visible_rationale": raw.get(
            "comparative_rationale",
            raw.get("visual_rationale", raw.get("forward", {}).get("comparative_rationale")),
        ),
    }
    if method_name == "compare_predicate_independent":
        for field in TYPED_OBSERVATION_FIELDS:
            if field in raw:
                normalized[field] = copy.deepcopy(raw[field])
    if routing_state is not None:
        normalized["routing_state"] = copy.deepcopy(routing_state)
    return normalized


VIDDIFF_TOOL_REGISTRY = [
    {
        "name": "inspect_comparison_task",
        "domain": "external_media",
        "satisfies_goal": "G-CONTEXT",
        "teacher_schema": _external_schema(
            "inspect_comparison_task",
            "Inspect the visible sample identity, comparison statement, and prepared visual assets.",
            {"sample_ref": {"type": "string", "enum": ["current_sample"]}},
        ),
    },
    *[
        {
            "name": name,
            "domain": "external_media",
            "satisfies_goal": "G-DIFFERENCE",
            "teacher_schema": _external_schema(
                name,
                (
                    "Aggregate the already executed global-motion and action-refined "
                    "symmetric observations. Return only role-invariant c abstention "
                    "when both routes independently fail A/B-swap consistency; this "
                    "tool does not inspect new pixels and cannot produce a or b. In the "
                    "reasoning, name compare_global_motion_symmetric and "
                    "compare_action_refined_symmetric, or unambiguously refer to both fixed "
                    "symmetric source routes after their exact atoms were grounded earlier; "
                    "also say that no new pixels are "
                    "inspected. The resulting meta observation is role-invariant with "
                    "order_consistent=true; do not transfer either source route's "
                    "order inconsistency onto the meta result."
                    if name == "compare_symmetry_residual_abstention"
                    else VIDDIFF_METHOD_DESCRIPTIONS[name]
                ),
                PAIR_PROPERTIES,
            ),
        }
        for name in (
            "compare_uniform_timeline",
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
            "compare_predicate_independent",
            "compare_symmetry_residual_abstention",
        )
    ],
    {"name": "create_candidate_claim", "domain": "harness_state"},
    {"name": "bind_evidence", "domain": "harness_state"},
    {"name": "set_field_decision", "domain": "harness_state"},
    {"name": "stop", "domain": "terminal"},
]
VIDDIFF_CAUSAL_TOOL_REGISTRY = with_evidence_integrity_tools(VIDDIFF_TOOL_REGISTRY)


class VidDiffRecordedMeasurementEnvironment:
    """Replays one real accepted measurement with strict input/provenance checks."""

    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item
        self.selected_tool = item["selected_tool"]["name"]
        self.call_counts = {"inspect_comparison_task": 0, **{name: 0 for name in (
            "compare_uniform_timeline",
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
            "compare_predicate_independent",
            "compare_symmetry_residual_abstention",
        )}}
        self._repair_due = False
        self._context_recheck_due = False

    def eligible_tool_names(self) -> list[str]:
        if self.call_counts["inspect_comparison_task"] == 0 or getattr(
            self, "_context_recheck_due", False
        ):
            return ["inspect_comparison_task"]
        if self.call_counts[self.selected_tool] == 0 or self._repair_due:
            return [self.selected_tool]
        return []

    def on_evidence_dropout(self, _evidence_id: str, affected_fields: list[str]) -> None:
        if set(affected_fields).intersection(VIDDIFF_SCHEMA["target_fields"]):
            self._repair_due = True

    def on_evidence_challenge(
        self,
        requested_check: str,
        _evidence_ids: list[str],
        _defect_codes: list[str],
    ) -> None:
        if requested_check == "reinspect_context":
            self._context_recheck_due = True
        elif requested_check == "remeasure_evidence":
            self._repair_due = True

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        task = self.item["task"]
        if name == "inspect_comparison_task":
            if arguments != {"sample_ref": "current_sample"}:
                raise ValueError("inspect_comparison_task requires sample_ref=current_sample")
            self.call_counts[name] += 1
            self._context_recheck_due = False
            index = self.call_counts[name]
            execution_id = f"EXEC-VIDDIFF-INSPECT-{index:03d}"
            return {
                "observation_id": f"OBS-VIDDIFF-INSPECT-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(task),
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-VIDDIFF-TASK-{index:03d}",
                        kind="comparison_task_identity",
                        supports_fields=list(VIDDIFF_SCHEMA["invariant_fields"]),
                        observation={
                            "sample_key": self.item["sample_key"],
                            "comparison_statement": task["comparison_statement"],
                        },
                        provenance={
                            "generated_by": execution_id,
                            "method": "inspect_student_visible_comparison_task",
                        },
                    )
                ],
            }

        if name != self.selected_tool:
            raise ValueError(f"measurement method is not selected for this accepted trajectory: {name}")
        expected = self.item["selected_tool"]["arguments"]
        if arguments != expected:
            raise ValueError(f"measurement tool requires exactly {expected}")
        self.call_counts[name] += 1
        index = self.call_counts[name]
        self._repair_due = False
        execution_id = f"EXEC-VIDDIFF-MEASURE-{index:03d}"
        raw = self.item["tool_observation"]
        normalized = _normalized_measurement_observation(raw, method_name=name)
        return {
            "observation_id": f"OBS-VIDDIFF-MEASURE-{index:03d}",
            "execution_id": execution_id,
            "tool_name": name,
            "arguments": copy.deepcopy(arguments),
            "result": copy.deepcopy(normalized),
            "evidence_atoms": [
                EvidenceAtom(
                    evidence_id=f"EA-VIDDIFF-MEASUREMENT-{index:03d}",
                    kind="action_comparison_measurement",
                    supports_fields=list(VIDDIFF_SCHEMA["target_fields"]),
                    observation=copy.deepcopy(normalized),
                    provenance={
                        "generated_by": execution_id,
                        "method": name,
                        "source_measurement": copy.deepcopy(self.item["provenance"]),
                        "label_visible_to_tool": False,
                    },
                )
            ],
        }


class VidDiffAdaptiveMeasurementEnvironment:
    """Expose all executed label-blind branches and route only from observations."""

    METHOD_NAMES = (
        "compare_uniform_timeline",
        "compare_global_motion_symmetric",
        "compare_action_refined_symmetric",
    )
    OPTIONAL_METHOD_NAMES = (
        "compare_predicate_independent",
        "compare_symmetry_residual_abstention",
    )
    SYMMETRY_ABSTENTION_SOURCES = {
        "compare_global_motion_symmetric",
        "compare_action_refined_symmetric",
    }

    ROUTING_VARIANTS = {
        "balanced": None,
        "uniform_first": (
            "compare_uniform_timeline",
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
            "compare_symmetry_residual_abstention",
        ),
        "global_first": (
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
            "compare_uniform_timeline",
            "compare_symmetry_residual_abstention",
        ),
        "action_first": (
            "compare_action_refined_symmetric",
            "compare_global_motion_symmetric",
            "compare_uniform_timeline",
            "compare_symmetry_residual_abstention",
        ),
        "predicate_first": (
            "compare_predicate_independent",
            "compare_action_refined_symmetric",
            "compare_global_motion_symmetric",
            "compare_uniform_timeline",
            "compare_symmetry_residual_abstention",
        ),
    }

    def __init__(
        self,
        item: dict[str, Any],
        *,
        confidence_threshold: float = 0.65,
        routing_variant: str = "balanced",
    ) -> None:
        self.item = item
        self.confidence_threshold = confidence_threshold
        if routing_variant not in self.ROUTING_VARIANTS:
            raise ValueError(f"unknown adaptive routing variant: {routing_variant}")
        self.routing_variant = routing_variant
        self.candidates = {entry["name"]: entry for entry in item["candidate_tools"]}
        self._context_recheck_due = False
        missing = set(self.METHOD_NAMES) - set(self.candidates)
        if missing:
            raise ValueError(f"adaptive VidDiff item is missing candidate tools: {sorted(missing)}")
        self.available_methods = tuple(
            name
            for name in self.METHOD_NAMES + self.OPTIONAL_METHOD_NAMES
            if name in self.candidates
        )
        configured_order = self.ROUTING_VARIANTS[routing_variant]
        self.method_order = tuple(
            name
            for name in (configured_order or self.available_methods)
            if name in self.available_methods
        )
        self.call_counts = {
            "inspect_comparison_task": 0,
            **{name: 0 for name in self.available_methods},
        }
        self._round_called: list[str] = []
        self._measurement_complete = False
        self._repair_due = False
        self._evidence_tool_by_id: dict[str, str] = {}

    def eligible_tool_names(self) -> list[str]:
        if self.call_counts["inspect_comparison_task"] == 0 or getattr(
            self, "_context_recheck_due", False
        ):
            return ["inspect_comparison_task"]
        if self._measurement_complete and not self._repair_due:
            return []
        remaining = [name for name in self.method_order if name not in self._round_called]
        if (
            "compare_symmetry_residual_abstention" in remaining
            and not self.SYMMETRY_ABSTENTION_SOURCES.issubset(self._round_called)
        ):
            remaining.remove("compare_symmetry_residual_abstention")
        if self.routing_variant == "balanced":
            return remaining
        return remaining[:1]

    def _routing_contract(self) -> dict[str, Any]:
        return {
            "variant": self.routing_variant,
            "method_order": list(self.method_order),
            "confidence_threshold": self.confidence_threshold,
            "predicate_independent_threshold": 0.55,
            "label_visible_to_router": False,
            "selection_basis": "visible_observations_and_state_only",
        }

    def on_evidence_dropout(self, evidence_id: str, affected_fields: list[str]) -> None:
        if set(affected_fields).intersection(VIDDIFF_SCHEMA["target_fields"]):
            self._repair_due = True
            self._measurement_complete = False
            # Reopen a fresh evidence-choice round.  The invalidated producing
            # branch remains available, but it is not disclosed as the unique
            # repair answer; the policy must choose among real measurements.
            self._round_called = []

    def on_evidence_challenge(
        self,
        requested_check: str,
        _evidence_ids: list[str],
        _defect_codes: list[str],
    ) -> None:
        if requested_check == "reinspect_context":
            self._context_recheck_due = True
        elif requested_check == "remeasure_evidence":
            self._repair_due = True
            self._measurement_complete = False
            self._round_called = []

    def on_measurement_incompatible_with_claim(
        self, _tool_name: str, _incompatible_claim_fields: list[str]
    ) -> None:
        """Keep the label-blind branch open when visible evidence cannot repair the current claim."""
        self._measurement_complete = False

    def _routing_state(self) -> dict[str, Any]:
        called_observations = [
            self.candidates[name]["tool_observation"] for name in self._round_called
        ]
        directions = [
            observation.get("preferred_video")
            for observation in called_observations
            if observation.get("preferred_video") in {"a", "b", "c"}
        ]
        stable_directions = [
            observation.get("preferred_video")
            for observation in called_observations
            if observation.get("preferred_video") in {"a", "b", "c"}
            and observation.get("order_consistent") is not False
        ]
        unique = sorted(set(directions))
        disagreement = len(unique) > 1
        stable_unique = sorted(set(stable_directions))
        agreement = len(stable_directions) >= 2 and len(stable_unique) == 1
        latest = self.candidates[self._round_called[-1]]["tool_observation"]
        confidence = latest.get("confidence")
        latest_method = self._round_called[-1]
        effective_threshold = (
            0.55 if latest_method == "compare_predicate_independent" else self.confidence_threshold
        )
        high_confidence = (
            isinstance(confidence, (int, float))
            and float(confidence) >= effective_threshold
            and latest.get("order_consistent") is not False
            and latest.get("preferred_video") in {"a", "b", "c"}
        )
        exhausted = len(self._round_called) == len(self.available_methods)
        goal_satisfied = exhausted or agreement or (high_confidence and not disagreement)
        if disagreement and not exhausted:
            reason = "cross_method_disagreement"
        elif not goal_satisfied and not high_confidence:
            reason = "low_confidence_or_order_inconsistency"
        elif agreement:
            reason = "two_methods_agree"
        elif exhausted:
            reason = "all_candidates_exhausted"
        else:
            reason = "high_confidence_order_consistent_result"
        return {
            "routing_contract": self._routing_contract(),
            "called_methods": list(self._round_called),
            "remaining_methods": [name for name in self.method_order if name not in self._round_called],
            "observed_directions": directions,
            "stable_observed_directions": stable_directions,
            "cross_method_disagreement": disagreement,
            "goal_satisfied": goal_satisfied,
            "reason_code": reason,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        task = self.item["task"]
        if name == "inspect_comparison_task":
            if arguments != {"sample_ref": "current_sample"}:
                raise ValueError("inspect_comparison_task requires sample_ref=current_sample")
            self.call_counts[name] += 1
            self._context_recheck_due = False
            index = self.call_counts[name]
            execution_id = f"EXEC-VIDDIFF-ADAPTIVE-INSPECT-{index:03d}"
            return {
                "observation_id": f"OBS-VIDDIFF-ADAPTIVE-INSPECT-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": copy.deepcopy(arguments),
                "result": {
                    **copy.deepcopy(task),
                    "routing_contract": self._routing_contract(),
                },
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-VIDDIFF-ADAPTIVE-TASK-{index:03d}",
                        kind="comparison_task_identity",
                        supports_fields=list(VIDDIFF_SCHEMA["invariant_fields"]),
                        observation={
                            "sample_key": self.item["sample_key"],
                            "comparison_statement": task["comparison_statement"],
                        },
                        provenance={
                            "generated_by": execution_id,
                            "method": "inspect_student_visible_comparison_task",
                        },
                    )
                ],
            }

        candidate = self.candidates.get(name)
        if candidate is None:
            raise ValueError(f"unknown adaptive measurement tool: {name}")
        if name in self._round_called:
            raise ValueError(f"measurement tool already executed in this routing round: {name}")
        if arguments != candidate["arguments"]:
            raise ValueError(f"measurement tool requires exactly {candidate['arguments']}")
        self.call_counts[name] += 1
        index = self.call_counts[name]
        self._round_called.append(name)
        routing_state = self._routing_state()
        self._measurement_complete = routing_state["goal_satisfied"]
        self._repair_due = False
        execution_id = f"EXEC-VIDDIFF-ADAPTIVE-{name.upper()}-{index:03d}"
        raw = candidate["tool_observation"]
        normalized = _normalized_measurement_observation(
            raw,
            method_name=name,
            routing_state=routing_state,
        )
        evidence_id = f"EA-VIDDIFF-{name.upper()}-{index:03d}"
        self._evidence_tool_by_id[evidence_id] = name
        return {
            "observation_id": f"OBS-VIDDIFF-ADAPTIVE-MEASURE-{name}-{index:03d}",
            "execution_id": execution_id,
            "tool_name": name,
            "arguments": copy.deepcopy(arguments),
            "goal_satisfied": routing_state["goal_satisfied"],
            "result": copy.deepcopy(normalized),
            "evidence_atoms": [
                EvidenceAtom(
                    evidence_id=evidence_id,
                    kind="action_comparison_measurement",
                    supports_fields=list(VIDDIFF_SCHEMA["target_fields"]),
                    observation=copy.deepcopy(normalized),
                    provenance={
                        "generated_by": execution_id,
                        "method": name,
                        "source_measurement": copy.deepcopy(candidate["provenance"]),
                        "label_visible_to_tool": False,
                    },
                )
            ],
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest(item: dict[str, Any]) -> dict[str, Any]:
    durations = []
    assets = item["task"].get("visible_assets")
    for payload in assets if isinstance(assets, list) else []:
        if isinstance(payload, dict):
            duration = payload.get("local_duration_s")
            if isinstance(duration, (int, float)):
                durations.append(float(duration))
            if isinstance(payload.get("global"), dict):
                duration = payload["global"].get("local_duration_s")
                if isinstance(duration, (int, float)):
                    durations.append(float(duration))
    return {
        "case_id": f"viddiff-{item['sample_key']}",
        "task": "Produce one evidence-bound comparative claim for the visible video pair.",
        "source_id": item["sample_key"],
        "source_media": {"path": f"viddiff://{item['sample_key']}/video_a"},
        "edited_media": {"path": f"viddiff://{item['sample_key']}/video_b"},
        "comparison_scope_s": [0.0, max(durations, default=1.0)],
    }


def _validate(bundle: dict[str, Any], oracle: str, item: dict[str, Any]) -> dict[str, Any]:
    clean = bundle["clean_episode"]
    repair = bundle["repair_episode"]
    final = repair.get("final_claim") or {}
    summary = bundle["execution_summary"]
    target_states = repair["final_state"]["candidate_claims"][0]["fields"]
    adaptive = "candidate_tools" in item
    candidate_names = (
        {entry["name"] for entry in item["candidate_tools"]}
        if adaptive
        else {item["selected_tool"]["name"]}
    )
    selected_method = final.get("measurement_method")
    required_calls = 2 if summary.get("repair_operator") == "evidence_dropout" else 1
    student_payload = json.dumps(
        {"clean": bundle["clean_turn_sft"], "repair": bundle["repair_turn_sft"]}, ensure_ascii=False
    )
    checks = {
        "clean_terminated": clean["terminated"],
        "repair_terminated": repair["terminated"],
        "measurement_reexecuted": summary["external_tool_call_counts"].get(selected_method, 0) >= required_calls,
        "oracle_direction_match": final.get("preferred_video") == oracle,
        "selected_method_preserved": selected_method in candidate_names,
        "sample_identity_preserved": final.get("sample_key") == item["sample_key"],
        "statement_preserved": final.get("comparison_statement") == item["task"]["comparison_statement"],
        "all_fields_supported_and_committed": all(
            state["evidence_status"] == "supported" and state["decision_status"] == "committed"
            for state in target_states.values()
        ),
        "off_target_invariance": repair["invariant_fields_before"] == repair["invariant_fields_after"],
        "dropped_evidence_not_reused": summary.get("repair_operator") != "evidence_dropout"
        or summary["dropped_evidence_id"] not in repair["final_state"]["available_evidence_ids"],
        "step_utility_counterfactual_audit": bundle.get("step_utility_report", {}).get("passed") is True,
        "no_oracle_leak": not any(
            token in student_payload
            for token in ("oracle_direction", "oracle_match", "differences_gt", "retrieval_frames")
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "execution_summary": summary}


def _candidate_set_reachability(item: dict[str, Any], reference_direction: str) -> dict[str, Any]:
    """Verifier-only feasibility check; never used to choose a rollout route."""
    candidates = item.get("candidate_tools")
    if not isinstance(candidates, list):
        return {
            "reachable": True,
            "candidate_directions": [],
            "candidate_tool_count": 0,
            "reason_code": "not_an_adaptive_candidate_set",
        }
    directions = [
        entry.get("tool_observation", {}).get("preferred_video")
        for entry in candidates
        if isinstance(entry, dict)
    ]
    reachable = reference_direction in directions
    return {
        "reachable": reachable,
        "candidate_directions": directions,
        "candidate_tool_count": len(candidates),
        "reason_code": (
            "reference_supported_by_candidate_atom"
            if reachable
            else "no_candidate_atom_supports_reference"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    routing_mode = getattr(args, "routing_mode", DEFAULT_ROUTING_MODE)
    input_path = (
        getattr(args, "adaptive_path", DEFAULT_ADAPTIVE)
        if routing_mode == "adaptive"
        else args.accepted_path
    )
    accepted = _load_jsonl(input_path)
    audit = json.loads(args.audit_path.read_text(encoding="utf-8"))
    oracle = {item["sample_key"]: item["oracle_direction"] for item in audit["cases"]}
    sample_keys = getattr(args, "sample_keys", None)
    if sample_keys:
        requested = set(sample_keys)
        accepted = [item for item in accepted if item["sample_key"] in requested]
        missing = requested - {item["sample_key"] for item in accepted}
        if missing:
            raise ValueError(f"requested VidDiff sample keys are unavailable: {sorted(missing)}")
    if args.limit:
        accepted = accepted[: args.limit]
    rollout_variants = list(getattr(args, "rollout_variants", ["balanced"]))
    if routing_mode != "adaptive":
        rollout_variants = ["legacy"]
    invalid_variants = set(rollout_variants) - set(VidDiffAdaptiveMeasurementEnvironment.ROUTING_VARIANTS)
    if routing_mode == "adaptive" and invalid_variants:
        raise ValueError(f"unknown rollout variants: {sorted(invalid_variants)}")
    run_id = (
        f"viddiff_adaptive_seed_{audit['seed']}_n{len(accepted)}_bon{len(rollout_variants)}"
        if routing_mode == "adaptive"
        else f"viddiff_verified_seed_{audit['seed']}_accepted_{len(accepted)}"
    )
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    base_teacher = CodexCLITeacher(model=args.model) if args.run_cli else None
    teacher = (
        None
        if base_teacher is None
        else base_teacher
        if getattr(args, "legacy_full_control", False)
        else build_evidence_choice_teacher(
            base_teacher,
            candidates_per_turn=getattr(args, "candidates_per_turn", 3),
            proposal_rounds_per_turn=getattr(args, "proposal_rounds_per_turn", 2),
        )
    )
    results = []
    clean_sft = []
    repair_sft = []
    transition_sidecar = []
    selected_calls: Counter[str] = Counter()
    indispensable_calls: Counter[str] = Counter()
    repair_operators = getattr(args, "repair_operators", ["evidence_dropout"])
    for item_index, item in enumerate(accepted):
        result: dict[str, Any] = {"case_id": item["case_id"], "status": "ready"}
        reachability = _candidate_set_reachability(item, oracle[item["sample_key"]])
        if getattr(args, "skip_oracle_unreachable", False) and not reachability["reachable"]:
            result.update(
                {
                    "status": "oracle_unreachable_candidate_set",
                    "verifier_only_reachability": reachability,
                    "attempts": [],
                }
            )
            results.append(result)
            write_json(
                output_root / "batch_report.json",
                {
                    "run_id": run_id,
                    "dataset": "VidDiffBench",
                    "input_policy": (
                        "visible_observation_adaptive_branching_v2"
                        if routing_mode == "adaptive"
                        else audit["policy"]
                    ),
                    "routing_mode": routing_mode,
                    "rollout_variants": rollout_variants,
                    "accepted_input_count": len(accepted),
                    "cases": results,
                },
            )
            continue
        if args.run_cli:
            attempts: list[dict[str, Any]] = []
            chosen: tuple[
                dict[str, Any], dict[str, Any], dict[str, Any], Counter[str], Counter[str], int, str
            ] | None = None
            # The full sample key and routing variant are recorded in the
            # report.  Compact physical directories prevent long audit-export
            # filenames from crossing the Windows path limit.
            sample_slug = str(item["sample_key"]).replace("_", "")[:14]
            case_root = output_root / "c" / f"{item_index:03d}_{sample_slug}"
            repair_operator = repair_operators[item_index % len(repair_operators)]
            for attempt_index, variant in enumerate(rollout_variants, start=1):
                variant_slug = variant.replace("_", "")[:6]
                attempt_root = case_root / "a" / f"{attempt_index:02d}_{variant_slug}"
                try:
                    environment = (
                        VidDiffAdaptiveMeasurementEnvironment(item, routing_variant=variant)
                        if routing_mode == "adaptive"
                        else VidDiffRecordedMeasurementEnvironment(item)
                    )
                    manifest = _manifest(item)
                    bundle = build_llm_teacher_bundle(
                        manifest=manifest,
                        source_frames=None,
                        suspect_frames=None,
                        source_path=Path(manifest["source_media"]["path"]),
                        suspect_path=Path(manifest["edited_media"]["path"]),
                        teacher=teacher,
                        claim_schema=VIDDIFF_SCHEMA,
                        tool_registry=VIDDIFF_CAUSAL_TOOL_REGISTRY,
                        external_environment=environment,
                        target_fields=VIDDIFF_SCHEMA["target_fields"],
                        invariant_fields=VIDDIFF_SCHEMA["invariant_fields"],
                        max_clean_turns=args.max_clean_turns,
                        max_repair_turns=args.max_repair_turns,
                        repair_operator=repair_operator,
                    )
                    bundle["execution_summary"].update(
                        {
                            "routing_variant": variant,
                            "best_of_n_attempt_index": attempt_index,
                            "selection_policy": "full_trajectory_post_rollout_verification",
                            "oracle_visible_to_teacher": False,
                        }
                    )
                    for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]:
                        datum["metadata"]["routing_variant"] = variant
                        datum["metadata"]["best_of_n_attempt_index"] = attempt_index
                    step_report = audit_step_utility(bundle)
                    apply_step_utility_filter(bundle, step_report)
                    validation = _validate(bundle, oracle[item["sample_key"]], item)
                    quality = evaluate_unified_quality(bundle, validation, dataset="VidDiffBench")
                    _write_bundle(bundle, validation, promote=False, output_dir=attempt_root)
                    write_json(attempt_root / "unified_trajectory_quality_report.json", quality)
                    causal_quality = None
                    if (
                        getattr(args, "run_causal_packets", False)
                        and validation["passed"]
                        and quality["accepted_for_sft"]
                    ):
                        def probe_environment_factory():
                            return (
                                VidDiffAdaptiveMeasurementEnvironment(item, routing_variant=variant)
                                if routing_mode == "adaptive"
                                else VidDiffRecordedMeasurementEnvironment(item)
                            )

                        _packet, causal_quality = qualify_and_write_causal_packet(
                            bundle=bundle,
                            teacher=teacher,
                            dataset="VidDiffBench",
                            validation=validation,
                            output_dir=attempt_root,
                            environment_factory=probe_environment_factory,
                            run_blind_judges=getattr(args, "blind_judges", False),
                            judge_model=args.model,
                            max_probe_turns=getattr(args, "max_probe_turns", 6),
                        )
                    selected = Counter(
                        turn["tool_name"]
                        for episode_name in ("clean_episode", "repair_episode")
                        for turn in bundle[episode_name]["turn_history"]
                    )
                    indispensable = Counter(
                        entry["tool_name"]
                        for entries in bundle["step_utility_report"]["episodes"].values()
                        for entry in entries
                        if entry["verdict"] == "indispensable"
                    )
                    attempt_row = {
                        "attempt_index": attempt_index,
                        "routing_variant": variant,
                        "status": (
                            "passed" if (
                                validation["passed"]
                                and quality["accepted_for_sft"]
                                and (causal_quality is None or causal_quality["accepted_for_sft"])
                            )
                            else "utqr_v7_rejected" if causal_quality is not None
                            else "rubric_rejected" if validation["passed"]
                            else "validation_failed"
                        ),
                        "failed_checks": [name for name, value in validation["checks"].items() if not value],
                        "rubric_score": quality["total_score"],
                        "rubric_grade": quality["grade"],
                        "rubric_failed_gates": quality["diagnosis"]["failed_gates"],
                        "rubric_failed_criteria": quality["diagnosis"]["failed_criteria"],
                        "utqr_v7_score": None if causal_quality is None else causal_quality["total_score"],
                        "utqr_v7_accepted": None if causal_quality is None else causal_quality["accepted_for_sft"],
                        "utqr_v7_failed_gates": None if causal_quality is None else causal_quality["failed_gates"],
                        "selected_measurement_tool": (
                            bundle["repair_episode"].get("final_claim") or {}
                        ).get("measurement_method"),
                        "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                        "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                        "usage": bundle["execution_summary"]["api_usage"],
                        "artifact_dir": str(attempt_root),
                    }
                    attempts.append(attempt_row)
                    if (
                        validation["passed"]
                        and quality["accepted_for_sft"]
                        and (causal_quality is None or causal_quality["accepted_for_sft"])
                    ):
                        chosen = (bundle, validation, quality, selected, indispensable, attempt_index, variant)
                        _write_bundle(bundle, validation, promote=False, output_dir=case_root)
                        write_json(case_root / "unified_trajectory_quality_report.json", quality)
                        break
                except Exception as error:
                    failure_artifact = getattr(error, "failure_artifact", None)
                    failure_path = attempt_root / "incomplete_teacher_episode.json"
                    if failure_artifact is not None:
                        write_json(failure_path, failure_artifact)
                    attempts.append(
                        {
                            "attempt_index": attempt_index,
                            "routing_variant": variant,
                            "status": "execution_failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "artifact_dir": str(attempt_root),
                            "failure_artifact": str(failure_path) if failure_artifact is not None else None,
                            **generation_failure_metadata(error),
                        }
                    )
            if chosen is not None:
                bundle, validation, quality, selected, indispensable, attempt_index, variant = chosen
                result.update(
                    {
                        "status": "passed",
                        "failed_checks": [],
                        "selected_attempt": attempt_index,
                        "selected_routing_variant": variant,
                        "selected_measurement_tool": (
                            bundle["repair_episode"].get("final_claim") or {}
                        ).get("measurement_method"),
                        "repair_operator": bundle["execution_summary"].get("repair_operator"),
                        "rubric_score": quality["total_score"],
                        "rubric_grade": quality["grade"],
                        "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                        "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                        "usage": bundle["execution_summary"]["api_usage"],
                        "selected_calls": dict(selected),
                        "indispensable_calls": dict(indispensable),
                        "attempts": attempts,
                    }
                )
                clean_sft.extend(bundle["clean_turn_sft"])
                repair_sft.extend(bundle["repair_turn_sft"])
                transition_sidecar.extend(compile_transition_sidecar(bundle))
                selected_calls.update(selected)
                indispensable_calls.update(indispensable)
            else:
                last = attempts[-1] if attempts else {"status": "execution_failed"}
                result.update(
                    {
                        "status": last["status"],
                        "failed_checks": last.get("failed_checks", []),
                        "repair_operator": repair_operator,
                        "attempts": attempts,
                    }
                )
                if "error" in last:
                    result.update({"error_type": last.get("error_type"), "error": last["error"]})
        results.append(result)
        write_json(
            output_root / "batch_report.json",
            {
                "run_id": run_id,
                "dataset": "VidDiffBench",
                "input_policy": (
                    "visible_observation_adaptive_branching_v2"
                    if routing_mode == "adaptive"
                    else audit["policy"]
                ),
                "routing_mode": routing_mode,
                "rollout_variants": rollout_variants,
                "accepted_input_count": len(accepted),
                "cases": results,
            },
        )
    passed = sum(item["status"] == "passed" for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VidDiffBench",
        "input_policy": (
            "visible_observation_adaptive_branching_v2" if routing_mode == "adaptive" else audit["policy"]
        ),
        "routing_mode": routing_mode,
        "rollout_variants": rollout_variants,
        "accepted_input_count": len(accepted),
        "summary": {
            "passed": passed,
            "attempted_rollouts": sum(len(item.get("attempts", [])) for item in results),
            "oracle_unreachable_candidate_sets": sum(
                item["status"] == "oracle_unreachable_candidate_set" for item in results
            ),
            "failed_after_cli_start": sum(
                item["status"] in {"execution_failed", "validation_failed", "rubric_rejected"}
                for item in results
            ),
            "safe_to_compile_sft": passed == len(accepted) and bool(accepted),
            "safe_to_compile_accepted_sft": passed > 0,
        },
        "tool_telemetry": {
            "selected_calls": dict(selected_calls),
            "indispensable_calls": dict(indispensable_calls),
            "selected_but_not_indispensable": {
                name: count - indispensable_calls.get(name, 0)
                for name, count in selected_calls.items()
                if count > indispensable_calls.get(name, 0)
            },
        },
        "generation_admission": generation_admission_summary(results),
        "cases": results,
    }
    write_json(output_root / "batch_report.json", report)
    write_jsonl(output_root / "sft_clean_accepted.jsonl", clean_sft)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", repair_sft)
    write_jsonl(output_root / "transition_sidecar.jsonl", transition_sidecar)
    return report


def _saved_attempt_dir(run_root: Path, attempt: dict[str, Any], sample_key: str) -> Path:
    recorded = Path(str(attempt.get("artifact_dir", "")))
    candidates = [
        recorded,
        ROOT / recorded,
        run_root / "cases" / sample_key / "attempts"
        / f"{int(attempt.get('attempt_index', 1)):02d}_{attempt.get('routing_variant', 'balanced')}",
    ]
    for candidate in candidates:
        if (candidate / "llm_teacher_clean_episode.json").exists():
            return candidate
    raise FileNotFoundError(f"saved VidDiff attempt artifacts are unavailable: {candidates}")


def _restore_viddiff_harness_config(bundle: dict[str, Any]) -> None:
    """Restore the exact model-visible runtime contract used by the saved rollout."""

    payload: dict[str, Any] = {}
    for message in bundle.get("clean_episode", {}).get("messages", []):
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            candidate = json.loads(message["content"])
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, dict) and candidate.get("tool_registry"):
            payload = candidate
            break
    claim_schema = copy.deepcopy(payload.get("claim_schema") or VIDDIFF_SCHEMA)
    tool_registry = copy.deepcopy(
        payload.get("tool_registry") or VIDDIFF_CAUSAL_TOOL_REGISTRY
    )
    bundle["harness_config"] = {
        "claim_schema": claim_schema,
        "tool_registry": tool_registry,
        "target_fields": list(claim_schema.get("target_fields") or VIDDIFF_SCHEMA["target_fields"]),
        "invariant_fields": list(
            claim_schema.get("invariant_fields") or VIDDIFF_SCHEMA["invariant_fields"]
        ),
    }


def reconcile_saved_run(
    run_root: Path,
    *,
    accepted_path: Path = DEFAULT_ACCEPTED,
    adaptive_path: Path = DEFAULT_ADAPTIVE,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    """Re-audit saved executions with current validators/rubric without new LLM calls."""
    run_root = Path(run_root)
    report_path = run_root / "batch_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    routing_mode = report.get("routing_mode", "adaptive")
    input_path = adaptive_path if routing_mode == "adaptive" else accepted_path
    items = {item["sample_key"]: item for item in _load_jsonl(input_path)}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    oracle = {item["sample_key"]: item["oracle_direction"] for item in audit["cases"]}

    clean_sft: list[dict[str, Any]] = []
    repair_sft: list[dict[str, Any]] = []
    transition_sidecar: list[dict[str, Any]] = []
    selected_calls: Counter[str] = Counter()
    indispensable_calls: Counter[str] = Counter()

    for case in report.get("cases", []):
        sample_key = str(case.get("case_id", "")).removeprefix("viddiff-")
        item = items.get(sample_key)
        if item is None or sample_key not in oracle or not case.get("attempts"):
            continue
        chosen: tuple[dict[str, Any], dict[str, Any], dict[str, Any], Counter[str], Counter[str], dict[str, Any]] | None = None
        for attempt in case["attempts"]:
            try:
                attempt_root = _saved_attempt_dir(run_root, attempt, sample_key)
                bundle, _saved_validation = load_saved_bundle(attempt_root)
                _restore_viddiff_harness_config(bundle)
                step_report = audit_step_utility(bundle)
                apply_step_utility_filter(bundle, step_report)
                validation = _validate(bundle, oracle[sample_key], item)
                quality = evaluate_unified_quality(bundle, validation, dataset="VidDiffBench")
                _write_bundle(bundle, validation, promote=False, output_dir=attempt_root)
                write_json(attempt_root / "unified_trajectory_quality_report.json", quality)

                selected = Counter(
                    turn["tool_name"]
                    for episode_name in ("clean_episode", "repair_episode")
                    for turn in bundle[episode_name]["turn_history"]
                )
                indispensable = Counter(
                    entry["tool_name"]
                    for entries in step_report["episodes"].values()
                    for entry in entries
                    if entry["verdict"] == "indispensable"
                )
                attempt.update(
                    {
                        "status": (
                            "passed"
                            if validation["passed"] and quality["accepted_for_sft"]
                            else "rubric_rejected" if validation["passed"] else "validation_failed"
                        ),
                        "failed_checks": [
                            name for name, value in validation["checks"].items() if not value
                        ],
                        "rubric_score": quality["total_score"],
                        "rubric_grade": quality["grade"],
                        "rubric_failed_gates": quality["diagnosis"]["failed_gates"],
                        "rubric_failed_criteria": quality["diagnosis"]["failed_criteria"],
                        "artifact_dir": str(attempt_root),
                        "reconciled_from_saved_execution": True,
                    }
                )
                if validation["passed"] and quality["accepted_for_sft"]:
                    chosen = (bundle, validation, quality, selected, indispensable, attempt)
                    break
            except Exception as error:
                attempt.update(
                    {
                        "status": "reconciliation_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

        if chosen is None:
            last = case["attempts"][-1]
            case.update(
                {
                    "status": last["status"],
                    "failed_checks": last.get("failed_checks", []),
                }
            )
            continue

        bundle, validation, quality, selected, indispensable, attempt = chosen
        case_root = run_root / "cases" / sample_key
        _write_bundle(bundle, validation, promote=False, output_dir=case_root)
        write_json(case_root / "unified_trajectory_quality_report.json", quality)
        case.update(
            {
                "status": "passed",
                "failed_checks": [],
                "selected_attempt": attempt["attempt_index"],
                "selected_routing_variant": attempt["routing_variant"],
                "selected_measurement_tool": (
                    bundle["repair_episode"].get("final_claim") or {}
                ).get("measurement_method"),
                "repair_operator": bundle["execution_summary"].get("repair_operator"),
                "rubric_score": quality["total_score"],
                "rubric_grade": quality["grade"],
                "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                "usage": bundle["execution_summary"].get("api_usage", {}),
                "selected_calls": dict(selected),
                "indispensable_calls": dict(indispensable),
                "reconciled_from_saved_execution": True,
            }
        )
        case.pop("error", None)
        case.pop("error_type", None)
        clean_sft.extend(bundle["clean_turn_sft"])
        repair_sft.extend(bundle["repair_turn_sft"])
        transition_sidecar.extend(compile_transition_sidecar(bundle))
        selected_calls.update(selected)
        indispensable_calls.update(indispensable)

    cases = report.get("cases", [])
    passed = sum(case.get("status") == "passed" for case in cases)
    accepted_count = int(report.get("accepted_input_count", len(cases)))
    report["summary"] = {
        "passed": passed,
        "attempted_rollouts": sum(len(case.get("attempts", [])) for case in cases),
        "oracle_unreachable_candidate_sets": sum(
            case.get("status") == "oracle_unreachable_candidate_set" for case in cases
        ),
        "failed_after_cli_start": sum(
            case.get("status")
            in {"execution_failed", "validation_failed", "rubric_rejected", "reconciliation_failed"}
            for case in cases
        ),
        "safe_to_compile_sft": passed == accepted_count and bool(accepted_count),
        "safe_to_compile_accepted_sft": passed > 0,
    }
    report["tool_telemetry"] = {
        "selected_calls": dict(selected_calls),
        "indispensable_calls": dict(indispensable_calls),
        "selected_but_not_indispensable": {
            name: count - indispensable_calls.get(name, 0)
            for name, count in selected_calls.items()
            if count > indispensable_calls.get(name, 0)
        },
    }
    report["reconciliation"] = {
        "used_saved_executions": True,
        "new_llm_calls": 0,
        "counterfactual_step_audit_rerun": True,
        "validator_rerun": True,
        "current_rubric_rerun": True,
    }
    write_json(report_path, report)
    write_jsonl(run_root / "sft_clean_accepted.jsonl", clean_sft)
    write_jsonl(run_root / "sft_repair_accepted.jsonl", repair_sft)
    write_jsonl(run_root / "transition_sidecar.jsonl", transition_sidecar)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-path", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reconcile-run",
        type=Path,
        help="Re-audit a saved VidDiff run with current code; makes no new teacher calls.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-keys", nargs="+")
    parser.add_argument("--run-cli", action="store_true")
    parser.add_argument(
        "--skip-oracle-unreachable",
        action="store_true",
        help="Verifier-only pre-screen: skip CLI when no recorded candidate atom can support the reference.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=["legacy", "adaptive"],
        default=DEFAULT_ROUTING_MODE,
        help=(
            "Adaptive is the evidence-first default and exposes multiple executed label-blind methods. "
            "Legacy replays one preselected method for diagnostic compatibility only."
        ),
    )
    parser.add_argument(
        "--rollout-variants",
        nargs="+",
        choices=sorted(VidDiffAdaptiveMeasurementEnvironment.ROUTING_VARIANTS),
        default=["balanced"],
        help="Label-blind route portfolio; attempts stop after the first fully verified trajectory.",
    )
    parser.add_argument(
        "--repair-operators",
        nargs="+",
        choices=["evidence_dropout", "binding_invalidation", "decision_reopen"],
        default=["evidence_dropout"],
    )
    parser.add_argument("--max-clean-turns", type=int, default=10)
    parser.add_argument("--max-repair-turns", type=int, default=8)
    parser.add_argument("--candidates-per-turn", type=int, default=3)
    parser.add_argument("--proposal-rounds-per-turn", type=int, default=2)
    parser.add_argument(
        "--run-causal-packets",
        action="store_true",
        help="Execute all eight UTQR-7 causal probes; a failing packet is excluded from SFT.",
    )
    parser.add_argument(
        "--blind-judges",
        action="store_true",
        help="Add two isolated blind process judges. Required for final UTQR-7 acceptance.",
    )
    parser.add_argument("--max-probe-turns", type=int, default=8)
    parser.add_argument(
        "--legacy-full-control",
        action="store_true",
        help="Diagnostic compatibility mode; causal reduced-control teacher view is the default.",
    )
    args = parser.parse_args()
    report = (
        reconcile_saved_run(
            args.reconcile_run,
            accepted_path=args.accepted_path,
            adaptive_path=args.adaptive_path,
            audit_path=args.audit_path,
        )
        if args.reconcile_run
        else run(args)
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
