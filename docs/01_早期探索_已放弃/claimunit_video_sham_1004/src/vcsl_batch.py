"""Run executable Codex-CLI teacher trajectories on ten VCSL feature pairs."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cli_teacher_rollout import CodexCLITeacher
from causal_probe_suite import qualify_and_write_causal_packet
from evidence_choice_search import build_evidence_choice_teacher
from generation_admission import generation_admission_summary, generation_failure_metadata
from interactive_harness import (
    EVIDENCE_INTEGRITY_TOOL_NAMES,
    EvidenceAtom,
    VideoForensicHarness,
    with_evidence_integrity_tools,
)
from llm_teacher_rollout import _write_bundle, build_llm_teacher_bundle
from pipeline import ROOT, write_json, write_jsonl
from sft_v2_sidecar import compile_transition_sidecar
from step_utility_auditor import apply_step_utility_filter, audit_step_utility
from unified_trajectory_rubric import (
    evaluate_bundle as evaluate_unified_quality,
    load_saved_bundle,
)
from vcsl_probe import (
    DEFAULT_FEATURE_ROOT,
    axis_saliency_localize,
    auto_route_localize,
    dp_localize,
    hv_localize,
    hv_segment_localize,
    local_alignment_localize,
    segment_precision_recall,
    select_cases,
    temporal_network_localize,
    temporal_network_stability_localize,
)
from vcsl_official_spd import (
    DEFAULT_CONFIDENCE as SPD_DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD as SPD_DEFAULT_IOU_THRESHOLD,
    official_spd_localize,
)


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "batch"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_selected_localization(
    case: dict[str, Any],
    item: dict[str, Any],
    feature_root: Path,
) -> dict[str, Any]:
    """Re-execute a selected localizer before crossing the teacher-service boundary."""
    serialized = json.dumps(item, ensure_ascii=False)
    forbidden = [token for token in ("oracle_boxes", '"metrics"', "strict_utqr") if token in serialized]
    if forbidden:
        raise ValueError(f"student selection contains verifier-only fields: {forbidden}")
    for field in ("pair_id", "query_id", "reference_id"):
        if item.get(field) != case.get(field):
            raise ValueError(f"selection identity mismatch for {field}: {item.get(field)!r} != {case.get(field)!r}")
    variant = str(item.get("selected_localizer", ""))
    if variant not in LOCALIZER_SPECS:
        raise ValueError(f"selection names unknown localizer: {variant}")
    spec = LOCALIZER_SPECS[variant]
    expected_tool = {
        "name": spec["tool_name"],
        "arguments": {
            "query_ref": "query",
            "reference_ref": "reference",
            "method": spec["method"],
        },
    }
    if item.get("selected_tool") != expected_tool:
        raise ValueError(f"selection tool contract mismatch for {case['pair_id']}")

    environment = VCSLFeatureEnvironment(
        case=case,
        feature_root=feature_root,
        routing_variant=variant,
    )
    environment.execute(
        "inspect_pair_features",
        {"query_ref": "query", "reference_ref": "reference"},
    )
    observation = environment.execute(expected_tool["name"], expected_tool["arguments"])
    actual_boxes = observation["result"]["copied_segments_frames"]
    expected_boxes = item.get("tool_observation", {}).get("copied_segments_frames")
    if actual_boxes != expected_boxes:
        raise ValueError(f"selected localization is not exactly replayable for {case['pair_id']}")
    return {
        "pair_id": case["pair_id"],
        "selected_localizer": variant,
        "box_count": len(actual_boxes),
        "exact_replay": True,
    }


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


REASONING = {
    "type": "string",
    "description": (
        "Explain what the latest visible observation established, what evidence or decision gap remains, "
        "and why this action closes that gap without using hidden labels."
    ),
}


def _external_schema(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    all_properties = {"reasoning": REASONING, **properties}
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": _object_schema(all_properties, list(all_properties)),
        "strict": True,
    }


VCSL_SCHEMA = {
    "schema_id": "vcsl_copy_localization_claim_v1",
    "required_fields": [
        "query_id",
        "reference_id",
        "relationship",
        "copied_segments_frames",
        "localization_method",
    ],
    "invariant_fields": ["query_id", "reference_id"],
    "target_fields": ["relationship", "copied_segments_frames", "localization_method"],
    "fields": {
        "query_id": {
            "types": ["str"],
            "definition": "Visible identifier of the query feature sequence.",
            "acceptable_evidence_kinds": ["feature_pair_identity"],
        },
        "reference_id": {
            "types": ["str"],
            "definition": "Visible identifier of the reference feature sequence.",
            "acceptable_evidence_kinds": ["feature_pair_identity"],
        },
        "relationship": {
            "types": ["str"],
            "enum": ["partial_copy"],
            "definition": "Measured relationship between the two videos.",
            "acceptable_evidence_kinds": ["copy_localization_measurement"],
        },
        "copied_segments_frames": {
            "types": ["list"],
            "definition": "One or more [query_start, reference_start, query_end, reference_end] boxes.",
            "acceptable_evidence_kinds": ["copy_localization_measurement"],
        },
        "localization_method": {
            "types": ["str"],
            "enum": [
                "HV",
                "HV-SEGMENT",
                "LOCAL-ALIGNMENT",
                "DP",
                "TEMPORAL-NETWORK",
                "TN-STABILITY",
                "AXIS-SALIENCY",
                "SPD-OFFICIAL-ISC",
            ],
            "definition": "Public temporal alignment method used to produce the boxes.",
            "acceptable_evidence_kinds": ["copy_localization_measurement"],
        },
    },
}


LOCALIZER_SPECS = {
    "hv": {
        "tool_name": "localize_copy_segment",
        "method": "HV",
        "description": (
            "Rank temporal offsets by global Hough-vote mass and emit an unsplit full-diagonal "
            "box for each retained offset."
        ),
    },
    "hv_segment": {
        "tool_name": "localize_copy_segment_hv_segment",
        "method": "HV-SEGMENT",
        "description": "Run segment-refined Hough voting that splits sparse diagonal votes into locally supported runs.",
    },
    "local_alignment": {
        "tool_name": "localize_copy_segment_local_alignment",
        "method": "LOCAL-ALIGNMENT",
        "description": "Run label-blind local sequence alignment with speed-change tolerance.",
    },
    "dp": {
        "tool_name": "localize_copy_segment_dp",
        "method": "DP",
        "description": "Run the non-learned dynamic-programming temporal localizer on the real similarity map.",
    },
    "temporal_network": {
        "tool_name": "localize_copy_segment_temporal_network",
        "method": "TEMPORAL-NETWORK",
        "description": (
            "Extract raw top-k monotonic temporal-network paths with rectangular suppression "
            "and IoU-based non-overlap filtering, without stability refinements."
        ),
    },
    "tn_stability": {
        "tool_name": "localize_copy_segment_temporal_stability",
        "method": "TN-STABILITY",
        "description": "Run temporal-network proposals with label-blind weak-bridge splitting, slope regularization, dominant-path suppression, and endpoint-uncertainty margins for two-fragment paths.",
    },
    "axis_saliency": {
        "tool_name": "localize_copy_segment_axis_saliency",
        "method": "AXIS-SALIENCY",
        "description": "Bound broad speed-changed copies using robust row/column upper-tail saliency anchored by the strongest monotonic path.",
    },
    "spd_official": {
        "tool_name": "localize_copy_segment_spd_official",
        "method": "SPD-OFFICIAL-ISC",
        "description": "Run VCSL's released ISC SPD detector on the real frame-similarity map.",
    },
}


def _localizer_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": spec["tool_name"],
            "domain": "external_media",
            "satisfies_goal": "G-DIFFERENCE",
            "teacher_schema": _external_schema(
                spec["tool_name"],
                spec["description"] + " It never reads benchmark labels.",
                {
                    "query_ref": {"type": "string", "enum": ["query"]},
                    "reference_ref": {"type": "string", "enum": ["reference"]},
                    "method": {"type": "string", "enum": [spec["method"]]},
                },
            ),
        }
        for spec in LOCALIZER_SPECS.values()
    ]


VCSL_TOOL_REGISTRY = [
    {
        "name": "inspect_pair_features",
        "domain": "external_media",
        "satisfies_goal": "G-CONTEXT",
        "teacher_schema": _external_schema(
            "inspect_pair_features",
            "Inspect the identities, frame counts, dimensions, and normalization of the visible feature pair.",
            {
                "query_ref": {"type": "string", "enum": ["query"]},
                "reference_ref": {"type": "string", "enum": ["reference"]},
            },
        ),
    },
    *_localizer_tools(),
    {"name": "create_candidate_claim", "domain": "harness_state"},
    {"name": "bind_evidence", "domain": "harness_state"},
    {"name": "set_field_decision", "domain": "harness_state"},
    {"name": "stop", "domain": "terminal"},
]
VCSL_CAUSAL_TOOL_REGISTRY = with_evidence_integrity_tools(VCSL_TOOL_REGISTRY)


def vcsl_tool_registry_for_variant(variant: str) -> list[dict[str, Any]]:
    """Expose only external localizers that this environment can execute.

    Internal claim/evidence tools remain shared across routes.  Keeping an
    unexecutable localizer in the teacher function list is actively harmful:
    a sensible counterfactual policy may select it and receive a harness
    rejection even though the visible tool contract advertised it as usable.
    """
    if variant == "portfolio":
        return copy.deepcopy(VCSL_CAUSAL_TOOL_REGISTRY)
    if variant not in LOCALIZER_SPECS:
        raise ValueError(f"unknown fixed VCSL routing variant: {variant}")
    selected_tool = LOCALIZER_SPECS[variant]["tool_name"]
    localizer_names = {spec["tool_name"] for spec in LOCALIZER_SPECS.values()}
    route_tools = [
        copy.deepcopy(item)
        for item in VCSL_TOOL_REGISTRY
        if item["name"] not in localizer_names or item["name"] == selected_tool
    ]
    return with_evidence_integrity_tools(route_tools)


class VCSLFeatureEnvironment:
    """Label-blind VCSL adapter backed by the authors' released ISC features."""

    def __init__(self, *, case: dict[str, Any], feature_root: Path, routing_variant: str = "portfolio") -> None:
        if routing_variant not in {"portfolio", "auto", *LOCALIZER_SPECS}:
            raise ValueError(f"unknown VCSL routing variant: {routing_variant}")
        self.case = case
        self.requested_routing_variant = routing_variant
        self.query_path = feature_root / f"{case['query_id']}.npy"
        self.reference_path = feature_root / f"{case['reference_id']}.npy"
        self.query = np.load(self.query_path)
        self.reference = np.load(self.reference_path)
        self.similarity = self.query @ self.reference.T
        self.selected_routing_variant: str | None = None
        self._context_recheck_due = False
        if routing_variant == "portfolio":
            _recommended, _preview_boxes, diagnostics = auto_route_localize(self.similarity)
            self.routing_diagnostics = {
                key: value
                for key, value in diagnostics.items()
                if key not in {"recommended_variant", "reason_code"}
            }
            self.routing_diagnostics.update(
                {
                    "selection_basis": "student_visible_feature_structure",
                    "policy_choice_required": True,
                }
            )
        elif routing_variant == "auto":
            routing_variant, _preview_boxes, self.routing_diagnostics = auto_route_localize(self.similarity)
        else:
            self.routing_diagnostics = {
                "label_visible_to_router": False,
                "selection_basis": "fixed_ablation_variant",
                "recommended_variant": routing_variant,
                "reason_code": "fixed_ablation_variant",
            }
        self.routing_variant = routing_variant
        self.localizer_spec = (
            None if routing_variant == "portfolio" else LOCALIZER_SPECS[routing_variant]
        )
        self.call_counts = {
            "inspect_pair_features": 0,
            **{spec["tool_name"]: 0 for spec in LOCALIZER_SPECS.values()},
        }
        self._repair_localization_due = False

    def eligible_tool_names(self) -> list[str]:
        if self.call_counts["inspect_pair_features"] == 0 or getattr(
            self, "_context_recheck_due", False
        ):
            return ["inspect_pair_features"]
        if self.routing_variant == "portfolio":
            if self._repair_localization_due:
                return [spec["tool_name"] for spec in LOCALIZER_SPECS.values()]
            return [
                spec["tool_name"]
                for spec in LOCALIZER_SPECS.values()
                if self.call_counts[spec["tool_name"]] == 0
            ]
        assert self.localizer_spec is not None
        tool_name = self.localizer_spec["tool_name"]
        if self.call_counts[tool_name] == 0 or self._repair_localization_due:
            return [tool_name]
        return []

    def on_evidence_dropout(self, _evidence_id: str, affected_fields: list[str]) -> None:
        if set(affected_fields).intersection(VCSL_SCHEMA["target_fields"]):
            self._repair_localization_due = True

    def on_evidence_challenge(
        self,
        requested_check: str,
        _evidence_ids: list[str],
        _defect_codes: list[str],
    ) -> None:
        if requested_check == "reinspect_context":
            self._context_recheck_due = True
        elif requested_check == "remeasure_evidence":
            self._repair_localization_due = True

    def _validate_refs(
        self,
        arguments: dict[str, Any],
        *,
        include_method: bool,
        localizer_spec: dict[str, Any] | None = None,
    ) -> None:
        expected = {"query_ref": "query", "reference_ref": "reference"}
        if include_method:
            spec = localizer_spec or self.localizer_spec
            if spec is None:
                raise ValueError("localizer method must be selected before validation")
            expected["method"] = spec["method"]
        if arguments != expected:
            raise ValueError(f"feature tool requires exactly {expected}")

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_pair_features":
            self._validate_refs(arguments, include_method=False)
            self.call_counts[name] += 1
            self._context_recheck_due = False
            index = self.call_counts[name]
            execution_id = f"EXEC-VCSL-INSPECT-{index:03d}"
            observation = {
                "query_id": self.case["query_id"],
                "reference_id": self.case["reference_id"],
                "query_shape": list(self.query.shape),
                "reference_shape": list(self.reference.shape),
                "feature_dimension": int(self.query.shape[1]),
                "query_mean_l2_norm": round(float(np.linalg.norm(self.query, axis=1).mean()), 6),
                "reference_mean_l2_norm": round(float(np.linalg.norm(self.reference, axis=1).mean()), 6),
                "localizer_routing_diagnostics": copy.deepcopy(self.routing_diagnostics),
            }
            if self.routing_variant != "portfolio":
                assert self.localizer_spec is not None
                observation["recommended_localizer_tool"] = self.localizer_spec["tool_name"]
            return {
                "observation_id": f"OBS-VCSL-INSPECT-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": copy.deepcopy(arguments),
                "result": observation,
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-VCSL-PAIR-{index:03d}",
                        kind="feature_pair_identity",
                        supports_fields=list(VCSL_SCHEMA["invariant_fields"]),
                        observation={
                            "query_id": self.case["query_id"],
                            "reference_id": self.case["reference_id"],
                        },
                        provenance={
                            "generated_by": execution_id,
                            "method": "inspect_official_isc_feature_headers",
                            "inputs": [
                                f"VCSL_ISC256_QUERY:{self.case['query_id']}",
                                f"VCSL_ISC256_REFERENCE:{self.case['reference_id']}",
                            ],
                        },
                    )
                ],
            }

        variant_by_tool = {
            spec["tool_name"]: variant for variant, spec in LOCALIZER_SPECS.items()
        }
        if self.routing_variant == "portfolio":
            actual_variant = variant_by_tool.get(name)
            if actual_variant is None:
                raise ValueError(f"unknown external VCSL tool: {name}")
            localizer_spec = LOCALIZER_SPECS[actual_variant]
        else:
            assert self.localizer_spec is not None
            if name != self.localizer_spec["tool_name"]:
                raise ValueError(f"unknown external VCSL tool: {name}")
            actual_variant = self.routing_variant
            localizer_spec = self.localizer_spec
        self._validate_refs(
            arguments,
            include_method=True,
            localizer_spec=localizer_spec,
        )
        similarity = self.similarity
        if actual_variant == "hv":
            boxes = hv_localize(similarity, min_similarity=0.7, max_iou=0.9)
            parameters = {"min_similarity": 0.7, "max_iou": 0.9}
        elif actual_variant == "hv_segment":
            boxes = hv_segment_localize(
                similarity,
                min_similarity=0.7,
                max_iou=0.9,
                max_gap=2,
                min_matches=4,
            )
            parameters = {
                "min_similarity": 0.7,
                "max_iou": 0.9,
                "max_gap": 2,
                "min_matches": 4,
            }
        elif actual_variant == "local_alignment":
            boxes = local_alignment_localize(
                similarity,
                min_similarity=0.7,
                max_iou=0.9,
                gap_penalty=0.08,
                min_length=4,
            )
            parameters = {
                "min_similarity": 0.7,
                "max_iou": 0.9,
                "gap_penalty": 0.08,
                "min_length": 4,
            }
        elif actual_variant == "dp":
            boxes = dp_localize(similarity)
            parameters = {
                "discontinue": 3,
                "point_similarity": 0.0,
                "average_similarity": 0.3,
                "min_length": 5,
                "diagonal_threshold": 30,
            }
        elif actual_variant == "temporal_network":
            boxes = temporal_network_localize(
                similarity,
                top_k=5,
                max_step=10,
                min_similarity=0.7,
                min_length=4,
                max_iou=0.9,
            )
            parameters = {
                "top_k": 5,
                "max_step": 10,
                "min_similarity": 0.7,
                "min_length": 4,
                "max_iou": 0.9,
            }
        elif actual_variant == "tn_stability":
            boxes = temporal_network_stability_localize(similarity)
            parameters = {
                "top_k": 5,
                "max_step": 10,
                "min_similarity": 0.4,
                "min_average_similarity": 0.5,
                "weak_bridge_split": True,
                "slope_regularization": True,
            }
        elif actual_variant == "axis_saliency":
            boxes = axis_saliency_localize(similarity)
            parameters = {"axis_percentile": 80.0, "selection_percentile": 60.0}
        else:
            boxes = official_spd_localize(similarity)
            parameters = {
                "confidence": SPD_DEFAULT_CONFIDENCE,
                "iou_threshold": SPD_DEFAULT_IOU_THRESHOLD,
                "device": "cpu",
                "released_model": "VCSL_ISC_SPD",
            }
        self.call_counts[name] += 1
        index = self.call_counts[name]
        if not boxes:
            method_slug = actual_variant.upper().replace("_", "-")
            execution_id = f"EXEC-VCSL-{method_slug}-NEGATIVE-{index:03d}"
            negative = {
                "localization_method": localizer_spec["method"],
                "copied_segment_detected": False,
                "similarity_shape": list(similarity.shape),
                "similarity_maximum": round(float(similarity.max()), 6),
                "parameters": parameters,
                "routing_variant": actual_variant,
            }
            return {
                "observation_id": f"OBS-VCSL-{method_slug}-NEGATIVE-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": copy.deepcopy(arguments),
                "goal_satisfied": False,
                "result": copy.deepcopy(negative),
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-VCSL-{method_slug}-NEGATIVE-{index:03d}",
                        kind="negative_copy_localization_measurement",
                        supports_fields=[],
                        observation=copy.deepcopy(negative),
                        provenance={
                            "generated_by": execution_id,
                            "method": f"VCSL_{localizer_spec['method']}_on_official_ISC_256d_features",
                            "inputs": [
                                f"VCSL_ISC256_QUERY:{self.case['query_id']}",
                                f"VCSL_ISC256_REFERENCE:{self.case['reference_id']}",
                            ],
                            "label_access": False,
                        },
                    )
                ],
            }
        self._repair_localization_due = False
        self.localizer_spec = localizer_spec
        self.selected_routing_variant = actual_variant
        method_slug = actual_variant.upper().replace("_", "-")
        execution_id = f"EXEC-VCSL-{method_slug}-{index:03d}"
        measurement = {
            "relationship": "partial_copy",
            "copied_segments_frames": boxes,
            "localization_method": localizer_spec["method"],
            "similarity_shape": list(similarity.shape),
            "similarity_maximum": round(float(similarity.max()), 6),
            "parameters": parameters,
            "routing_variant": actual_variant,
        }
        return {
            "observation_id": f"OBS-VCSL-LOCALIZE-{index:03d}",
            "execution_id": execution_id,
            "tool_name": name,
            "arguments": copy.deepcopy(arguments),
            "result": measurement,
            "evidence_atoms": [
                EvidenceAtom(
                    evidence_id=f"EA-VCSL-{method_slug}-{index:03d}",
                    kind="copy_localization_measurement",
                    supports_fields=list(VCSL_SCHEMA["target_fields"]),
                    observation=copy.deepcopy(measurement),
                        provenance={
                            "generated_by": execution_id,
                            "method": f"VCSL_{localizer_spec['method']}_on_official_ISC_256d_features",
                            "inputs": [
                                f"VCSL_ISC256_QUERY:{self.case['query_id']}",
                                f"VCSL_ISC256_REFERENCE:{self.case['reference_id']}",
                            ],
                        "label_access": False,
                    },
                )
            ],
        }


def _manifest(case: dict[str, Any], feature_root: Path) -> dict[str, Any]:
    query_path = feature_root / f"{case['query_id']}.npy"
    reference_path = feature_root / f"{case['reference_id']}.npy"
    query_frames = int(np.load(query_path, mmap_mode="r").shape[0])
    reference_frames = int(np.load(reference_path, mmap_mode="r").shape[0])
    return {
        "case_id": f"vcsl-{case['pair_id']}",
        "task": (
            "Using only the visible released feature sequences, determine whether the query and reference share "
            "copied content and report the measured temporal boxes."
        ),
        "source_id": case["query_id"],
        "source_media": {
            "path": f"VCSL_ISC256_QUERY:{case['query_id']}",
            "role": "query_feature_sequence",
        },
        "edited_media": {
            "path": f"VCSL_ISC256_REFERENCE:{case['reference_id']}",
            "role": "reference_feature_sequence",
        },
        "comparison_scope_s": [0.0, float(max(query_frames, reference_frames))],
        "visible_feature_metadata": {
            "feature": "official_ISC_256d",
            "query_frames": query_frames,
            "reference_frames": reference_frames,
        },
    }


def _validate_bundle(bundle: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    clean = bundle["clean_episode"]
    repair = bundle["repair_episode"]
    summary = bundle["execution_summary"]
    final_claim = repair.get("final_claim") or {}
    predicted = final_claim.get("copied_segments_frames", [])
    metrics = segment_precision_recall(predicted, case["oracle_boxes"]) if isinstance(predicted, list) else {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    final_fields = repair["final_state"]["candidate_claims"][0]["fields"]
    selected_method = final_claim.get("localization_method")
    selected_tools = [
        spec["tool_name"] for spec in LOCALIZER_SPECS.values() if spec["method"] == selected_method
    ]
    required_calls = 2 if summary.get("repair_operator") == "evidence_dropout" else 1
    student_payload = json.dumps(
        {"clean": bundle["clean_turn_sft"], "repair": bundle["repair_turn_sft"]},
        ensure_ascii=False,
    )
    checks = {
        "clean_terminated": clean["terminated"],
        "repair_terminated": repair["terminated"],
        "real_localizer_executed_or_reexecuted": bool(selected_tools)
        and summary["external_tool_call_counts"].get(selected_tools[0], 0) >= required_calls,
        "query_identity_preserved": final_claim.get("query_id") == case["query_id"],
        "reference_identity_preserved": final_claim.get("reference_id") == case["reference_id"],
        "relationship_supported": final_claim.get("relationship") == "partial_copy",
        "method_supported": selected_method in {spec["method"] for spec in LOCALIZER_SPECS.values()},
        "localization_precision_at_least_half": metrics["precision"] >= 0.5,
        "localization_recall_at_least_half": metrics["recall"] >= 0.5,
        "localization_f1_at_least_point_six": metrics["f1"] >= 0.6,
        "all_fields_supported_and_committed": all(
            item["evidence_status"] == "supported" and item["decision_status"] == "committed"
            for item in final_fields.values()
        ),
        "off_target_invariance": repair["invariant_fields_before"] == repair["invariant_fields_after"],
        "dropped_evidence_not_reused": summary.get("repair_operator") != "evidence_dropout"
        or summary["dropped_evidence_id"] not in repair["final_state"]["available_evidence_ids"],
        "step_utility_counterfactual_audit": bundle.get("step_utility_report", {}).get("passed") is True,
        "no_oracle_box_leak": "oracle_boxes" not in student_payload
        and "label_file_uuid_total.json" not in student_payload,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "localization_metrics": {key: round(value, 6) for key, value in metrics.items()},
        "execution_summary": summary,
    }


def _telemetry(bundle: dict[str, Any]) -> dict[str, Counter[str]]:
    selected = Counter(
        turn["tool_name"]
        for episode_name in ("clean_episode", "repair_episode")
        for turn in bundle[episode_name]["turn_history"]
    )
    indispensable = Counter(
        item["tool_name"]
        for episode_items in bundle["step_utility_report"]["episodes"].values()
        for item in episode_items
        if item["verdict"] == "indispensable"
    )
    return {"selected": selected, "indispensable": indispensable}


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_cases(args.seed, args.sample_size)
    sampled_pair_ids = {case["pair_id"] for case in selected}
    requested_case_indices = getattr(args, "case_indices", None)
    if requested_case_indices:
        invalid_indices = [index for index in requested_case_indices if index < 0 or index >= len(selected)]
        if invalid_indices:
            raise ValueError(f"case indices outside sampled range: {invalid_indices}")
        selected = [selected[index] for index in requested_case_indices]
    selection_path = getattr(args, "selection_path", None)
    selection_items: dict[str, dict[str, Any]] = {}
    if selection_path is not None:
        loaded = _load_jsonl(selection_path)
        selection_items = {item["pair_id"]: item for item in loaded}
        duplicate_count = len(loaded) - len(selection_items)
        if duplicate_count:
            raise ValueError(f"selection file contains {duplicate_count} duplicate pair_id values")
        unknown = sorted(set(selection_items) - sampled_pair_ids)
        if unknown:
            raise ValueError(f"selection file contains pairs outside the seeded sample: {unknown}")
        selected = [case for case in selected if case["pair_id"] in selection_items]
        if not selected:
            raise ValueError("selection file contains no accepted pair from the requested seeded sample")
    selection_replay = (
        [
            _verify_selected_localization(case, selection_items[case["pair_id"]], args.feature_root)
            for case in selected
        ]
        if selection_items
        else []
    )
    rollout_variants = list(getattr(args, "rollout_variants", ["portfolio"]))
    invalid_variants = set(rollout_variants) - {"portfolio", "auto", *LOCALIZER_SPECS}
    if invalid_variants:
        raise ValueError(f"unknown VCSL rollout variants: {sorted(invalid_variants)}")
    # The exact indices remain in batch_report.json. Keep the directory label
    # compact because a verbose 12-case label pushes nested Windows artifacts
    # past MAX_PATH before the first episode can be saved.
    suffix = (
        ""
        if not requested_case_indices
        else f"_idx{len(requested_case_indices)}_{min(requested_case_indices)}to{max(requested_case_indices)}"
    )
    selection_suffix = "_selected" if selection_items else f"_bon{len(rollout_variants)}"
    run_id = f"vcsl_seed_{args.seed}_n{len(selected)}{suffix}{selection_suffix}"
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
    aggregate_clean: list[dict[str, Any]] = []
    aggregate_repair: list[dict[str, Any]] = []
    aggregate_transitions: list[dict[str, Any]] = []
    selected_calls: Counter[str] = Counter()
    indispensable_calls: Counter[str] = Counter()

    repair_operators = getattr(args, "repair_operators", ["evidence_dropout"])
    for case_index, case in enumerate(selected):
        feature_paths = [
            args.feature_root / f"{case['query_id']}.npy",
            args.feature_root / f"{case['reference_id']}.npy",
        ]
        case_result: dict[str, Any] = {
            "case_id": f"vcsl-{case['pair_id']}",
            "query_id": case["query_id"],
            "reference_id": case["reference_id"],
            "status": "ready" if all(path.exists() for path in feature_paths) else "preflight_blocked",
            "preflight_reasons": [f"feature_missing:{path}" for path in feature_paths if not path.exists()],
        }
        if args.run_cli and case_result["status"] == "ready":
            attempts: list[dict[str, Any]] = []
            chosen: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Counter[str]], int, str] | None = None
            # Keep the on-disk routing keys compact.  The complete pair and
            # localizer identities remain in the manifest/report, while the
            # shorter directories leave room for audit filenames such as
            # ``llm_teacher_sft_repair_full_trajectory.jsonl`` on Windows.
            case_directory = f"{case_index:03d}_{case['query_id'][:6]}_{case['reference_id'][:6]}"
            case_root = output_root / "c" / case_directory
            repair_operator = repair_operators[case_index % len(repair_operators)]
            case_variants = (
                [selection_items[case["pair_id"]]["selected_localizer"]]
                if selection_items
                else rollout_variants
            )
            for attempt_index, variant in enumerate(case_variants, start=1):
                variant_slug = variant.replace("_", "")[:6]
                attempt_root = case_root / "a" / f"{attempt_index:02d}_{variant_slug}"
                try:
                    manifest = _manifest(case, args.feature_root)
                    environment = VCSLFeatureEnvironment(
                        case=case,
                        feature_root=args.feature_root,
                        routing_variant=variant,
                    )
                    case_tool_registry = vcsl_tool_registry_for_variant(
                        environment.routing_variant
                    )
                    bundle = build_llm_teacher_bundle(
                        manifest=manifest,
                        source_frames=None,
                        suspect_frames=None,
                        source_path=feature_paths[0],
                        suspect_path=feature_paths[1],
                        teacher=teacher,
                        claim_schema=VCSL_SCHEMA,
                        tool_registry=case_tool_registry,
                        external_environment=environment,
                        target_fields=VCSL_SCHEMA["target_fields"],
                        invariant_fields=VCSL_SCHEMA["invariant_fields"],
                        max_clean_turns=args.max_clean_turns,
                        max_repair_turns=args.max_repair_turns,
                        repair_operator=repair_operator,
                    )
                    bundle["execution_summary"].update(
                        {
                            "routing_variant": variant,
                            "resolved_routing_variant": (
                                environment.selected_routing_variant
                                or environment.routing_variant
                            ),
                            "routing_diagnostics": copy.deepcopy(environment.routing_diagnostics),
                            "best_of_n_attempt_index": attempt_index,
                            "selection_policy": (
                                "post_execution_oracle_bounded_rejection_sampling_v1"
                                if selection_items
                                else "full_trajectory_post_rollout_verification"
                            ),
                            "oracle_visible_to_teacher": False,
                        }
                    )
                    for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]:
                        datum["metadata"]["routing_variant"] = variant
                        datum["metadata"]["resolved_routing_variant"] = (
                            environment.selected_routing_variant
                            or environment.routing_variant
                        )
                        datum["metadata"]["best_of_n_attempt_index"] = attempt_index
                    step_report = audit_step_utility(bundle)
                    apply_step_utility_filter(bundle, step_report)
                    validation = _validate_bundle(bundle, case)
                    quality = evaluate_unified_quality(bundle, validation, dataset="VCSL")
                    _write_bundle(bundle, validation, promote=False, output_dir=attempt_root)
                    write_json(attempt_root / "unified_trajectory_quality_report.json", quality)
                    causal_quality = None
                    if (
                        getattr(args, "run_causal_packets", False)
                        and validation["passed"]
                        and quality["accepted_for_sft"]
                    ):
                        _packet, causal_quality = qualify_and_write_causal_packet(
                            bundle=bundle,
                            teacher=teacher,
                            dataset="VCSL",
                            validation=validation,
                            output_dir=attempt_root,
                            environment_factory=lambda: VCSLFeatureEnvironment(
                                case=case,
                                feature_root=args.feature_root,
                                routing_variant=variant,
                            ),
                            run_blind_judges=getattr(args, "blind_judges", False),
                            judge_model=args.model,
                            max_probe_turns=getattr(args, "max_probe_turns", 6),
                        )
                    telemetry = _telemetry(bundle)
                    accepted = (
                        validation["passed"]
                        and quality["accepted_for_sft"]
                        and (causal_quality is None or causal_quality["accepted_for_sft"])
                    )
                    attempts.append(
                        {
                            "attempt_index": attempt_index,
                            "routing_variant": variant,
                            "resolved_routing_variant": (
                                environment.selected_routing_variant
                                or environment.routing_variant
                            ),
                            "localization_method": environment.localizer_spec["method"],
                            "status": (
                                "passed" if accepted
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
                            "localization_metrics": validation["localization_metrics"],
                            "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                            "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                            "usage": bundle["execution_summary"]["api_usage"],
                            "artifact_dir": str(attempt_root),
                        }
                    )
                    if accepted:
                        chosen = (bundle, validation, quality, telemetry, attempt_index, variant)
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
                bundle, validation, quality, telemetry, attempt_index, variant = chosen
                case_result.update(
                    {
                        "status": "passed",
                        "failed_checks": [],
                        "selected_attempt": attempt_index,
                        "selected_routing_variant": variant,
                        "selected_localization_method": (bundle["repair_episode"].get("final_claim") or {}).get("localization_method"),
                        "rubric_score": quality["total_score"],
                        "rubric_grade": quality["grade"],
                        "localization_metrics": validation["localization_metrics"],
                        "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                        "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                        "repair_operator": bundle["execution_summary"].get("repair_operator"),
                        "usage": bundle["execution_summary"]["api_usage"],
                        "tool_telemetry": {
                            "selected_calls": dict(telemetry["selected"]),
                            "indispensable_calls": dict(telemetry["indispensable"]),
                        },
                        "attempts": attempts,
                    }
                )
                aggregate_clean.extend(bundle["clean_turn_sft"])
                aggregate_repair.extend(bundle["repair_turn_sft"])
                aggregate_transitions.extend(compile_transition_sidecar(bundle))
                selected_calls.update(telemetry["selected"])
                indispensable_calls.update(telemetry["indispensable"])
            else:
                last = attempts[-1] if attempts else {"status": "execution_failed"}
                case_result.update(
                    {
                        "status": last["status"],
                        "failed_checks": last.get("failed_checks", []),
                        "repair_operator": repair_operator,
                        "attempts": attempts,
                    }
                )
                if "error" in last:
                    case_result.update({"error_type": last.get("error_type"), "error": last["error"]})
        results.append(case_result)
        write_json(
            output_root / "batch_report.json",
            {
                "run_id": run_id,
                "dataset": "VCSL",
                "seed": args.seed,
                "sample_size": len(selected),
                "source_sample_size": args.sample_size,
                "case_indices": requested_case_indices,
                "rollout_variants": rollout_variants,
                "selection_path": str(selection_path) if selection_path is not None else None,
                "selection_policy": (
                    "post_execution_oracle_bounded_rejection_sampling_v1"
                    if selection_items
                    else "full_trajectory_post_rollout_verification"
                ),
                "selection_replay_verified": len(selection_replay),
                "cli_requested": args.run_cli,
                "cases": results,
            },
        )

    implemented = {item["name"] for item in VCSL_CAUSAL_TOOL_REGISTRY}
    passed = sum(item["status"] == "passed" for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VCSL",
        "seed": args.seed,
        "sample_size": len(selected),
        "source_sample_size": args.sample_size,
        "case_indices": requested_case_indices,
        "rollout_variants": rollout_variants,
        "selection_path": str(selection_path) if selection_path is not None else None,
        "selection_policy": (
            "post_execution_oracle_bounded_rejection_sampling_v1"
            if selection_items
            else "full_trajectory_post_rollout_verification"
        ),
        "selection_replay_verified": len(selection_replay),
        "cli_requested": args.run_cli,
        "summary": {
            "passed": passed,
            "attempted_rollouts": sum(len(item.get("attempts", [])) for item in results),
            "preflight_blocked": sum(item["status"] == "preflight_blocked" for item in results),
            "failed_after_cli_start": sum(
                item["status"] in {"execution_failed", "validation_failed", "rubric_rejected"}
                for item in results
            ),
            "safe_to_compile_sft": passed == len(selected) and bool(selected),
        },
        "tool_telemetry": {
            "selected_calls": dict(selected_calls),
            "indispensable_calls": dict(indispensable_calls),
            "unused_implemented_tools": sorted(implemented - EVIDENCE_INTEGRITY_TOOL_NAMES - set(selected_calls)),
            "conditional_integrity_tools": sorted(EVIDENCE_INTEGRITY_TOOL_NAMES),
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
    write_jsonl(output_root / "sft_clean_accepted.jsonl", aggregate_clean)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", aggregate_repair)
    write_jsonl(output_root / "transition_sidecar.jsonl", aggregate_transitions)
    return report


def _restore_vcsl_harness_config(bundle: dict[str, Any]) -> None:
    variant = str(
        bundle.get("execution_summary", {}).get("resolved_routing_variant")
        or bundle.get("execution_summary", {}).get("routing_variant")
        or "hv"
    )
    if variant == "auto":
        variant = "hv"
    bundle["harness_config"] = {
        "claim_schema": copy.deepcopy(VCSL_SCHEMA),
        "tool_registry": vcsl_tool_registry_for_variant(variant),
        "target_fields": list(VCSL_SCHEMA["target_fields"]),
        "invariant_fields": list(VCSL_SCHEMA["invariant_fields"]),
    }


def _resolve_saved_attempt(attempt: dict[str, Any]) -> Path:
    recorded = Path(str(attempt.get("artifact_dir", "")))
    for candidate in (recorded, ROOT / recorded):
        if (candidate / "llm_teacher_clean_episode.json").exists():
            return candidate
    raise FileNotFoundError(f"saved VCSL attempt artifacts are unavailable: {recorded}")


def reconcile_saved_run(
    run_root: Path,
    *,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
) -> dict[str, Any]:
    """Re-audit saved VCSL executions and rebuild accepted exports without teacher calls."""
    run_root = Path(run_root)
    report_path = run_root / "batch_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampled = select_cases(int(report["seed"]), int(report.get("source_sample_size", report["sample_size"])))
    case_indices = report.get("case_indices")
    if case_indices:
        sampled = [sampled[index] for index in case_indices]
    cases_by_pair = {case["pair_id"]: case for case in sampled}

    aggregate_clean: list[dict[str, Any]] = []
    aggregate_repair: list[dict[str, Any]] = []
    aggregate_transitions: list[dict[str, Any]] = []
    selected_calls: Counter[str] = Counter()
    indispensable_calls: Counter[str] = Counter()

    for case_row in report.get("cases", []):
        pair_id = str(case_row.get("case_id", "")).removeprefix("vcsl-")
        case = cases_by_pair.get(pair_id)
        if case is None or not case_row.get("attempts"):
            continue
        chosen: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Counter[str]], dict[str, Any], Path] | None = None
        last_result: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Counter[str]], dict[str, Any], Path] | None = None
        for attempt in case_row["attempts"]:
            try:
                attempt_root = _resolve_saved_attempt(attempt)
                bundle, _saved_validation = load_saved_bundle(attempt_root)
                _restore_vcsl_harness_config(bundle)
                step_report = audit_step_utility(bundle)
                apply_step_utility_filter(bundle, step_report)
                validation = _validate_bundle(bundle, case)
                quality = evaluate_unified_quality(bundle, validation, dataset="VCSL")
                _write_bundle(bundle, validation, promote=False, output_dir=attempt_root)
                write_json(attempt_root / "unified_trajectory_quality_report.json", quality)
                telemetry = _telemetry(bundle)
                accepted = validation["passed"] and quality["accepted_for_sft"]
                attempt.update(
                    {
                        "status": "passed" if accepted else "rubric_rejected" if validation["passed"] else "validation_failed",
                        "failed_checks": [name for name, value in validation["checks"].items() if not value],
                        "rubric_score": quality["total_score"],
                        "rubric_grade": quality["grade"],
                        "rubric_failed_gates": quality["diagnosis"]["failed_gates"],
                        "rubric_failed_criteria": quality["diagnosis"]["failed_criteria"],
                        "localization_metrics": validation["localization_metrics"],
                        "artifact_dir": str(attempt_root),
                        "reconciled_from_saved_execution": True,
                    }
                )
                last_result = (bundle, validation, quality, telemetry, attempt, attempt_root)
                if accepted:
                    chosen = last_result
                    break
            except Exception as error:
                attempt.update(
                    {
                        "status": "reconciliation_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

        result = chosen or last_result
        if result is None:
            case_row["status"] = "reconciliation_failed"
            continue
        bundle, validation, quality, telemetry, attempt, attempt_root = result
        accepted = chosen is not None
        case_root = attempt_root.parent.parent
        _write_bundle(bundle, validation, promote=False, output_dir=case_root)
        write_json(case_root / "unified_trajectory_quality_report.json", quality)
        case_row.update(
            {
                "status": "passed" if accepted else "rubric_rejected" if validation["passed"] else "validation_failed",
                "failed_checks": [name for name, value in validation["checks"].items() if not value],
                "selected_attempt": attempt["attempt_index"],
                "selected_routing_variant": attempt["routing_variant"],
                "selected_localization_method": (bundle["repair_episode"].get("final_claim") or {}).get("localization_method"),
                "rubric_score": quality["total_score"],
                "rubric_grade": quality["grade"],
                "rubric_failed_gates": quality["diagnosis"]["failed_gates"],
                "rubric_failed_criteria": quality["diagnosis"]["failed_criteria"],
                "localization_metrics": validation["localization_metrics"],
                "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                "repair_operator": bundle["execution_summary"].get("repair_operator"),
                "usage": bundle["execution_summary"].get("api_usage", {}),
                "tool_telemetry": {
                    "selected_calls": dict(telemetry["selected"]),
                    "indispensable_calls": dict(telemetry["indispensable"]),
                },
                "reconciled_from_saved_execution": True,
            }
        )
        case_row.pop("error", None)
        case_row.pop("error_type", None)
        if accepted:
            aggregate_clean.extend(bundle["clean_turn_sft"])
            aggregate_repair.extend(bundle["repair_turn_sft"])
            aggregate_transitions.extend(compile_transition_sidecar(bundle))
            selected_calls.update(telemetry["selected"])
            indispensable_calls.update(telemetry["indispensable"])

    cases = report.get("cases", [])
    passed = sum(case.get("status") == "passed" for case in cases)
    implemented = {item["name"] for item in VCSL_CAUSAL_TOOL_REGISTRY}
    report["summary"] = {
        "passed": passed,
        "attempted_rollouts": sum(len(case.get("attempts", [])) for case in cases),
        "preflight_blocked": sum(case.get("status") == "preflight_blocked" for case in cases),
        "failed_after_cli_start": sum(
            case.get("status") in {"execution_failed", "validation_failed", "rubric_rejected", "reconciliation_failed"}
            for case in cases
        ),
        "safe_to_compile_sft": passed == len(cases) and bool(cases),
        "safe_to_compile_accepted_sft": passed > 0,
    }
    report["tool_telemetry"] = {
        "selected_calls": dict(selected_calls),
        "indispensable_calls": dict(indispensable_calls),
        "unused_implemented_tools": sorted(implemented - EVIDENCE_INTEGRITY_TOOL_NAMES - set(selected_calls)),
        "conditional_integrity_tools": sorted(EVIDENCE_INTEGRITY_TOOL_NAMES),
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
    write_jsonl(run_root / "sft_clean_accepted.jsonl", aggregate_clean)
    write_jsonl(run_root / "sft_repair_accepted.jsonl", aggregate_repair)
    write_jsonl(run_root / "transition_sidecar.jsonl", aggregate_transitions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument(
        "--case-indices",
        nargs="+",
        type=int,
        help="Optional zero-based subset of the seeded sample, useful for targeted canaries or resume runs.",
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--selection-path",
        type=Path,
        help="Student-visible accepted-localization JSONL; selects one preverified label-blind localizer per pair.",
    )
    parser.add_argument(
        "--reconcile-run",
        type=Path,
        help="Re-audit a saved VCSL run with current code; makes no new teacher calls.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--run-cli", action="store_true")
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
    parser.add_argument(
        "--rollout-variants",
        nargs="+",
        choices=["portfolio", "auto", *sorted(LOCALIZER_SPECS)],
        default=["portfolio"],
        help=(
            "Portfolio is the evidence-first default and exposes every executable localizer. "
            "Auto/fixed variants retain the old preselected-route diagnostic behavior."
        ),
    )
    parser.add_argument(
        "--repair-operators",
        nargs="+",
        choices=["evidence_dropout", "binding_invalidation", "decision_reopen"],
        default=["evidence_dropout"],
    )
    args = parser.parse_args()
    report = (
        reconcile_saved_run(args.reconcile_run, feature_root=args.feature_root)
        if args.reconcile_run
        else run(args)
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
