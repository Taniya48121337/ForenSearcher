"""Executable multi-turn tool environment for ClaimUnitTest case 1004.

The policy side only proposes tool calls and state edits.  External media tools
create observations and EvidenceAtoms; the harness validates bindings and
recomputes evidence coverage.  Hidden dataset annotations are intentionally not
accepted by this module.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


REQUIRED_FIELDS = [
    "source_id",
    "claim_family",
    "manipulation_type",
    "direction",
    "comparison_scope_s",
    "retained_source_interval_s",
    "omitted_source_intervals_s",
    "estimated_removed_time_s",
]

TARGET_FIELDS = [
    "claim_family",
    "manipulation_type",
    "direction",
    "omitted_source_intervals_s",
    "estimated_removed_time_s",
]

INVARIANT_FIELDS = [
    "source_id",
    "comparison_scope_s",
    "retained_source_interval_s",
]


CLAIM_SCHEMA = {
    "schema_id": "selective_temporal_omission_v1",
    "required_fields": REQUIRED_FIELDS,
    "fields": {
        "source_id": {
            "types": ["str"],
            "acceptable_evidence_kinds": ["monotonic_visual_alignment"],
            "definition": "Neutral identifier of the source video matched by alignment.",
        },
        "claim_family": {
            "types": ["str"],
            "enum": [
                "selective_temporal_omission",
                "temporal_reordering",
                "temporal_rate_change",
                "no_material_difference",
            ],
            "acceptable_evidence_kinds": ["source_time_contraction"],
            "definition": "High-level temporal relation inferred from measured source coverage.",
        },
        "manipulation_type": {
            "types": ["str"],
            "enum": ["frame_dropping", "frame_insertion", "temporal_reordering", "speed_change", "undetermined"],
            "acceptable_evidence_kinds": ["source_time_contraction"],
            "definition": (
                "frame_dropping means source frames inside the comparison scope are absent from the suspect, "
                "including omission at scope boundaries."
            ),
        },
        "direction": {
            "types": ["str"],
            "enum": ["source_to_edited", "edited_to_source", "undetermined"],
            "acceptable_evidence_kinds": ["source_time_contraction"],
            "definition": "Direction of the measured transformation between the two declared media roles.",
        },
        "comparison_scope_s": {"types": ["list"], "acceptable_evidence_kinds": ["case_scope_locator"]},
        "retained_source_interval_s": {"types": ["list"], "acceptable_evidence_kinds": ["monotonic_visual_alignment"]},
        "omitted_source_intervals_s": {"types": ["list"], "acceptable_evidence_kinds": ["source_time_contraction"]},
        "estimated_removed_time_s": {"types": ["float", "int"], "acceptable_evidence_kinds": ["source_time_contraction"]},
    },
}


TOOL_REGISTRY = [
    {
        "name": "inspect_media",
        "domain": "external_media",
        "description": "Decode media metadata and read the neutral comparison scope.",
        "produces": ["media_metadata", "case_scope_locator"],
        "preconditions": ["source and suspect assets are visible"],
    },
    {
        "name": "align_videos_dense",
        "domain": "external_media",
        "description": "Execute dense monotonic visual alignment over the real video pair.",
        "produces": ["monotonic_visual_alignment", "source_time_contraction"],
        "preconditions": ["source and suspect assets are visible", "comparison scope is supplied"],
    },
    {
        "name": "create_candidate_claim",
        "domain": "harness_state",
        "description": "Create policy-proposed claim values and expand public field obligations.",
        "produces": ["CandidateClaim", "FieldObligations"],
    },
    {
        "name": "bind_evidence",
        "domain": "harness_state",
        "description": "Propose evidence-to-field links for deterministic contract validation.",
        "produces": ["FieldBinding"],
    },
    {
        "name": "set_field_decision",
        "domain": "harness_state",
        "description": "Set selected claim fields to open, committed, or abstained.",
        "produces": ["decision_status transition"],
    },
    {
        "name": "stop",
        "domain": "harness_terminal",
        "description": "Request termination after every high-priority field obligation is closed.",
        "produces": ["terminal decision"],
    },
]


ADJUDICATION_DEFECT_CODES = [
    "claim_value_conflict",
    "cross_method_direction_conflict",
    "order_inconsistent",
    "lower_confidence",
    "abstention_conflicts_with_claim",
    "role_assignment_mismatch",
    "redundant_same_conclusion",
]

EVIDENCE_CHALLENGE_DEFECT_CODES = [
    "identity_mismatch",
    "scope_mismatch",
    "role_mismatch",
    "contradictory_observations",
    "stale_or_superseded",
    "incomplete_observation",
    "value_rationale_conflict",
    "empty_observation",
]

EVIDENCE_CHALLENGE_TOOL = {
    "name": "challenge_evidence",
    "domain": "harness_state",
    "description": (
        "Record a policy-discovered evidence integrity concern and request a fresh "
        "context inspection or measurement. The policy must identify the suspect "
        "EvidenceAtom IDs and defect type; the Harness does not announce either."
    ),
    "produces": ["EvidenceChallenge", "VerificationObligation"],
    "teacher_schema": {
        "type": "function",
        "name": "challenge_evidence",
        "description": (
            "Use only when visible EvidenceAtom content is inconsistent, incomplete, "
            "stale, or incompatible with the current task. Name the suspect evidence, "
            "name only the claim fields whose support is put in doubt, and choose whether "
            "fresh context or a fresh measurement is needed. This action does not reveal "
            "a repair; target_fields are a policy decision, not a Harness hint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Grounded 2-4 sentence diagnosis citing every challenged "
                        "EvidenceAtom and the concrete visible inconsistency."
                    ),
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "defect_codes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": EVIDENCE_CHALLENGE_DEFECT_CODES,
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "target_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": (
                        "Claim fields whose current support is specifically put in doubt "
                        "by the cited evidence. Select the minimal visible subset."
                    ),
                },
                "requested_check": {
                    "type": "string",
                    "enum": ["reinspect_context", "remeasure_evidence"],
                },
            },
            "required": [
                "reasoning",
                "evidence_ids",
                "defect_codes",
                "target_fields",
                "requested_check",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

CLAIM_REVISION_TOOL = {
    "name": "revise_claim_fields",
    "domain": "harness_state",
    "description": (
        "Revise only reopened claim fields when fresh active evidence directly "
        "supports a different value. The Harness validates the proposed values; "
        "it does not choose or reveal them."
    ),
    "produces": ["CandidateClaim field-value revision"],
}

EVIDENCE_ADJUDICATION_TOOL = {
    "name": "adjudicate_evidence",
    "domain": "harness_state",
    "description": (
        "Compare every active EvidenceAtom competing for the same open claim fields, "
        "select the support set to use, and explicitly reject all alternatives using "
        "student-visible defects before binding."
    ),
    "produces": ["EvidenceAdjudication"],
    "teacher_schema": {
        "type": "function",
        "name": "adjudicate_evidence",
        "description": (
            "Resolve a visible multi-EvidenceAtom conflict before bind_evidence. Partition "
            "every listed competitor into selected_evidence_ids or rejected_evidence; each "
            "rejection must copy defect_codes exactly, including order, from that evidence "
            "ID's accepted visible candidate assessment. Do not add a newly inferred defect "
            "during adjudication even when it would be defensible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Grounded 2-4 sentence comparison that cites every competing "
                        "EvidenceAtom ID, explains the selected support, and names each "
                        "rejected atom's visible defect."
                    ),
                },
                "claim_id": {"type": "string"},
                "target_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Copy this list exactly from the visible "
                        "evidence_adjudication.requirement.target_fields; do not broaden it."
                    ),
                },
                "selected_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "rejected_evidence": {
                    "type": "array",
                    "description": (
                        "For each rejected evidence ID, copy defect_codes exactly and in the "
                        "same order from evidence_assessments in the visible harness state; "
                        "adjudication consumes the ledger and does not revise it."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "evidence_id": {"type": "string"},
                            "defect_codes": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ADJUDICATION_DEFECT_CODES,
                                },
                                "minItems": 1,
                            },
                        },
                        "required": ["evidence_id", "defect_codes"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
            },
            "required": [
                "reasoning",
                "claim_id",
                "target_fields",
                "selected_evidence_ids",
                "rejected_evidence",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVIDENCE_ASSESSMENT_TOOL = {
    "name": "assess_evidence_candidate",
    "domain": "harness_state",
    "description": (
        "Record one student-visible candidate assessment before final evidence "
        "adjudication. Each assessment has a real state effect and covers exactly one "
        "competing EvidenceAtom."
    ),
    "produces": ["EvidenceCandidateAssessment"],
    "teacher_schema": {
        "type": "function",
        "name": "assess_evidence_candidate",
        "description": (
            "Assess exactly one listed EvidenceAtom as retainable or rejectable using "
            "only its visible structured facts. Every competitor must be assessed before "
            "adjudicate_evidence becomes eligible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Grounded 2-4 sentence assessment naming this EvidenceAtom or its "
                        "unique method, its observed direction and order-consistency status, "
                        "and why the recorded disposition follows."
                    ),
                },
                "claim_id": {"type": "string"},
                "target_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Copy this list exactly from the visible "
                        "evidence_adjudication.requirement.target_fields; do not broaden it."
                    ),
                },
                "evidence_id": {"type": "string"},
                "disposition": {
                    "type": "string",
                    "enum": ["retain", "reject"],
                    "description": (
                        "Use retain only for claim-compatible evidence with no hard defect; "
                        "a retain action must set defect_codes to an empty array. Use reject "
                        "only when at least one listed defect is visibly supported."
                    ),
                },
                "defect_codes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ADJUDICATION_DEFECT_CODES,
                    },
                    "description": (
                        "MUST be [] when disposition=retain. When disposition=reject, "
                        "provide only defects literally supported by visible facts: "
                        "claim_value_conflict means the candidate cannot support the proposed field value; "
                        "order_inconsistent requires observation.order_consistent=false or a visible interval/box "
                        "whose start is after its end; "
                        "role_assignment_mismatch requires provenance to mark the current role assignment invalid; "
                        "abstention_conflicts_with_claim requires an explicit abstention against a non-abstaining claim; "
                        "redundant_same_conclusion and lower_confidence require comparison with a named selected atom."
                    ),
                },
            },
            "required": [
                "reasoning",
                "claim_id",
                "target_fields",
                "evidence_id",
                "disposition",
                "defect_codes",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


EVIDENCE_INTEGRITY_TOOL_NAMES = frozenset(
    {
        "challenge_evidence",
        "revise_claim_fields",
        "assess_evidence_candidate",
        "adjudicate_evidence",
    }
)


def _visible_temporal_order_inconsistent(
    observation: dict[str, Any], target_fields: list[str]
) -> bool:
    """Recognize temporal inversions directly encoded in visible values."""

    def numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    for field_name in target_fields:
        value = observation.get(field_name)
        if field_name == "copied_segments_frames" and isinstance(value, list):
            for box in value:
                if (
                    isinstance(box, (list, tuple))
                    and len(box) == 4
                    and all(numeric(item) for item in box)
                    and (float(box[0]) > float(box[2]) or float(box[1]) > float(box[3]))
                ):
                    return True
        elif (
            (field_name.endswith("_interval_s") or field_name.endswith("_range_s"))
            and isinstance(value, (list, tuple))
            and len(value) == 2
            and all(numeric(item) for item in value)
            and float(value[0]) > float(value[1])
        ):
            return True
    return False


def with_evidence_integrity_tools(registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add universal visible-evidence assessment tools exactly once."""

    result = copy.deepcopy(registry)
    names = {item.get("name") for item in result}
    for tool in (
        EVIDENCE_CHALLENGE_TOOL,
        CLAIM_REVISION_TOOL,
        EVIDENCE_ASSESSMENT_TOOL,
        EVIDENCE_ADJUDICATION_TOOL,
    ):
        if tool["name"] not in names:
            result.append(copy.deepcopy(tool))
    return result


@dataclass
class TaskContract:
    contract_id: str
    episode_mode: str
    task: str
    visible_assets: dict[str, Any]
    claim_schema_id: str


@dataclass
class CaseGoal:
    goal_id: str
    goal_type: str
    description: str
    priority: str = "high"
    status: str = "open"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceAtom:
    evidence_id: str
    kind: str
    supports_fields: list[str]
    observation: dict[str, Any]
    provenance: dict[str, Any]
    active: bool = True


def evidence_interval_scope_issues(
    evidence_atoms: Any, comparison_scope_s: Any
) -> list[dict[str, Any]]:
    """Return student-visible interval/scope contradictions in evidence atoms."""

    if not isinstance(comparison_scope_s, list) or len(comparison_scope_s) != 2:
        return []
    try:
        low, high = map(float, comparison_scope_s)
    except (TypeError, ValueError):
        return []
    values = evidence_atoms.values() if isinstance(evidence_atoms, dict) else evidence_atoms
    issues: list[dict[str, Any]] = []
    for raw_atom in values or []:
        atom = raw_atom if isinstance(raw_atom, EvidenceAtom) else EvidenceAtom(**raw_atom)
        if not atom.active:
            continue
        for field_name in ("affected_source_interval_s", "affected_suspect_interval_s"):
            interval = atom.observation.get(field_name)
            if not isinstance(interval, list) or len(interval) != 2:
                continue
            # The contract locator is on the source timeline. A temporal edit
            # may lengthen or shorten the suspect, so a measurement can publish
            # its separate, student-visible suspect coordinate frame. Older or
            # generic evidence keeps the conservative source-scope fallback.
            field_low, field_high = low, high
            if field_name == "affected_suspect_interval_s":
                suspect_scope = atom.observation.get("suspect_comparison_scope_s")
                if isinstance(suspect_scope, list) and len(suspect_scope) == 2:
                    try:
                        candidate_low, candidate_high = map(float, suspect_scope)
                    except (TypeError, ValueError):
                        pass
                    else:
                        if candidate_low <= candidate_high:
                            field_low, field_high = candidate_low, candidate_high
            try:
                start, end = map(float, interval)
            except (TypeError, ValueError):
                issues.append({"evidence_id": atom.evidence_id, "field": field_name, "reason": "non_numeric"})
                continue
            if start < field_low - 1e-6 or end > field_high + 1e-6 or start > end:
                issues.append(
                    {
                        "evidence_id": atom.evidence_id,
                        "field": field_name,
                        "interval_s": [start, end],
                        "comparison_scope_s": [field_low, field_high],
                        "coordinate_frame": (
                            "suspect"
                            if field_name == "affected_suspect_interval_s"
                            else "source"
                        ),
                        "reason": "outside_visible_scope",
                    }
                )
    return issues


@dataclass
class EvidenceAdjudication:
    adjudication_id: str
    claim_id: str
    target_fields: list[str]
    selected_evidence_ids: list[str]
    rejected_evidence: list[dict[str, Any]]
    considered_evidence_ids: list[str]


@dataclass
class EvidenceCandidateAssessment:
    assessment_id: str
    claim_id: str
    target_fields: list[str]
    evidence_id: str
    disposition: str
    defect_codes: list[str]
    considered_evidence_ids: list[str]


@dataclass
class EvidenceChallenge:
    challenge_id: str
    evidence_ids: list[str]
    defect_codes: list[str]
    target_fields: list[str]
    requested_check: str
    status: str = "open"
    resolved_by_evidence_ids: list[str] = field(default_factory=list)


@dataclass
class FieldBinding:
    field: str
    support_ids: list[str]
    attack_ids: list[str] = field(default_factory=list)
    proposed_by: str = "policy"
    validated_by: str = "contract_checker"
    validation_status: str = "accepted"


@dataclass
class ClaimFieldState:
    value: Any
    evidence_status: str = "unsupported"
    decision_status: str = "open"
    support_ids: list[str] = field(default_factory=list)
    attack_ids: list[str] = field(default_factory=list)


@dataclass
class CandidateClaim:
    claim_id: str
    schema_id: str
    fields: dict[str, ClaimFieldState]


class RealVideoToolEnvironment:
    """Adapter that turns real video functions into replayable observations."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        source_frames: Any,
        suspect_frames: Any,
        source_path: Path,
        suspect_path: Path,
        metadata_fn: Callable[[Any, Path], dict[str, Any]],
        align_fn: Callable[[Any, Any], dict[str, Any]],
        omission_fn: Callable[[dict[str, Any], float], list[dict[str, Any]]],
    ) -> None:
        self.manifest = manifest
        self.source_frames = source_frames
        self.suspect_frames = suspect_frames
        self.source_path = source_path
        self.suspect_path = suspect_path
        self.metadata_fn = metadata_fn
        self.align_fn = align_fn
        self.omission_fn = omission_fn
        self.call_counts = {"inspect_media": 0, "align_videos_dense": 0}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_media":
            return self._inspect(arguments)
        if name == "align_videos_dense":
            return self._align(arguments)
        raise ValueError(f"unknown external tool: {name}")

    def _inspect(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expected = ["source", "suspect"]
        if arguments.get("media_refs") != expected:
            raise ValueError(f"inspect_media requires media_refs={expected}")
        self.call_counts["inspect_media"] += 1
        call_index = self.call_counts["inspect_media"]
        source_meta = self.metadata_fn(self.source_frames, self.source_path)
        suspect_meta = self.metadata_fn(self.suspect_frames, self.suspect_path)
        execution_id = f"EXEC-INSPECT-{call_index:03d}"
        return {
            "observation_id": f"OBS-INSPECT-{call_index:03d}",
            "execution_id": execution_id,
            "tool_name": "inspect_media",
            "arguments": copy.deepcopy(arguments),
            "result": {
                "source_media": source_meta,
                "suspect_media": suspect_meta,
                "comparison_scope_s": self.manifest["comparison_scope_s"],
            },
            "evidence_atoms": [
                EvidenceAtom(
                    evidence_id=f"EA-META-{call_index:03d}",
                    kind="media_metadata",
                    supports_fields=[],
                    observation={
                        "source_duration_s": source_meta["duration_s"],
                        "suspect_duration_s": suspect_meta["duration_s"],
                        "source_fps": source_meta["fps"],
                        "suspect_fps": suspect_meta["fps"],
                    },
                    provenance={
                        "generated_by": execution_id,
                        "method": "container_decode",
                        "inputs": [source_meta["path"], suspect_meta["path"]],
                    },
                ),
                EvidenceAtom(
                    evidence_id=f"EA-SCOPE-{call_index:03d}",
                    kind="case_scope_locator",
                    supports_fields=["comparison_scope_s"],
                    observation={"comparison_scope_s": self.manifest["comparison_scope_s"]},
                    provenance={
                        "generated_by": execution_id,
                        "method": "read_visible_case_manifest",
                        "input": "student_view/case_manifest.json",
                    },
                ),
            ],
        }

    def _align(self, arguments: dict[str, Any]) -> dict[str, Any]:
        required = {
            "source_ref": "source",
            "suspect_ref": "suspect",
            "comparison_scope_s": self.manifest["comparison_scope_s"],
        }
        if arguments != required:
            raise ValueError(f"align_videos_dense requires exactly {required}")

        self.call_counts["align_videos_dense"] += 1
        call_index = self.call_counts["align_videos_dense"]
        alignment = self.align_fn(self.source_frames, self.suspect_frames)
        localized = self.omission_fn(alignment, self.source_frames.fps)
        scope = self.manifest["comparison_scope_s"]
        aligned_start = float(alignment["source_times_s"][0])
        aligned_end = float(alignment["source_times_s"][-1])
        retained_end = aligned_end + 1.0 / self.source_frames.fps
        aligned_span = retained_end - aligned_start
        contraction = max(0.0, (scope[1] - scope[0]) - aligned_span)
        boundary: list[dict[str, Any]] = []
        if aligned_start > scope[0] + 1.0 / self.source_frames.fps:
            boundary.append(
                {
                    "source_interval_s": [scope[0], round(aligned_start, 3)],
                    "estimated_missing_s": round(aligned_start - scope[0], 3),
                    "position": "leading_boundary",
                }
            )
        if retained_end < scope[1] - 1.0 / self.source_frames.fps:
            boundary.append(
                {
                    "source_interval_s": [round(retained_end, 3), scope[1]],
                    "estimated_missing_s": round(scope[1] - retained_end, 3),
                    "position": "trailing_boundary",
                }
            )

        execution_id = f"EXEC-ALIGN-{call_index:03d}"
        alignment_id = f"EA-ALIGN-{call_index:03d}"
        gap_id = f"EA-GAP-{call_index:03d}"
        alignment_summary = {
            "source_id": self.manifest["source_id"],
            "retained_source_interval_s": [round(aligned_start, 3), round(retained_end, 3)],
            "matched_frames": self.suspect_frames.frame_count,
            "strictly_monotonic": True,
            "mean_similarity": round(float(alignment["mean_similarity"]), 4),
            "p05_similarity": round(float(alignment["p05_similarity"]), 4),
        }
        gap_summary = {
            "source_role": "source",
            "suspect_role": "edited",
            "source_frames_absent": bool(boundary or localized) and contraction > 1.0 / self.source_frames.fps,
            "temporal_order_preserved": True,
            "comparison_scope_s": scope,
            "omitted_source_intervals_s": [item["source_interval_s"] for item in boundary],
            "estimated_removed_time_s": round(contraction, 3),
            "boundary_omission_segments": boundary,
            "localized_jump_segments": localized,
        }
        atoms = [
            EvidenceAtom(
                evidence_id=alignment_id,
                kind="monotonic_visual_alignment",
                supports_fields=["source_id", "retained_source_interval_s"],
                observation=alignment_summary,
                provenance={
                    "generated_by": execution_id,
                    "method": "visual_viterbi_v1",
                    "inputs": [self.manifest["source_media"]["path"], self.manifest["edited_media"]["path"]],
                    "feature": "center_crop_16:9_gray+gradients_64x36",
                },
            ),
            EvidenceAtom(
                evidence_id=gap_id,
                kind="source_time_contraction",
                supports_fields=TARGET_FIELDS,
                observation=gap_summary,
                provenance={
                    "generated_by": execution_id,
                    "method": "coverage_gap_from_fresh_alignment",
                    "depends_on": [alignment_id],
                    "inputs": [self.manifest["source_media"]["path"], self.manifest["edited_media"]["path"]],
                },
            ),
        ]
        return {
            "observation_id": f"OBS-ALIGN-{call_index:03d}",
            "execution_id": execution_id,
            "tool_name": "align_videos_dense",
            "arguments": copy.deepcopy(arguments),
            "result": {"alignment": alignment_summary, "source_time_contraction": gap_summary},
            "evidence_atoms": atoms,
        }


class VideoForensicHarness:
    """Deterministic transition kernel; it never supplies claim values."""

    def __init__(
        self,
        contract: TaskContract,
        environment: RealVideoToolEnvironment,
        *,
        claim_schema: dict[str, Any] | None = None,
        tool_registry: list[dict[str, Any]] | None = None,
        target_fields: list[str] | None = None,
        invariant_fields: list[str] | None = None,
    ) -> None:
        self.contract = contract
        self.environment = environment
        self.claim_schema = copy.deepcopy(claim_schema or CLAIM_SCHEMA)
        self.required_fields = list(self.claim_schema["required_fields"])
        self.tool_registry = copy.deepcopy(tool_registry or TOOL_REGISTRY)
        self.target_fields = list(target_fields or self.claim_schema.get("target_fields", TARGET_FIELDS))
        self.invariant_fields = list(
            invariant_fields or self.claim_schema.get("invariant_fields", INVARIANT_FIELDS)
        )
        self.goals = {
            "G-CONTEXT": CaseGoal(
                "G-CONTEXT", "ESTABLISH_COMPARISON_CONTEXT", "Inspect the visible media pair and comparison scope."
            ),
            "G-DIFFERENCE": CaseGoal(
                "G-DIFFERENCE", "DISCOVER_MATERIAL_DIFFERENCES", "Measure material source-to-suspect differences."
            ),
            "G-REPORT": CaseGoal(
                "G-REPORT", "PRODUCE_SUPPORTED_CLAIMS", "Produce claims whose required fields have visible support."
            ),
        }
        self.evidence_store: dict[str, EvidenceAtom] = {}
        self.evidence_assessments: dict[
            str, dict[str, EvidenceCandidateAssessment]
        ] = {}
        self.evidence_adjudications: dict[str, EvidenceAdjudication] = {}
        self.evidence_challenges: list[EvidenceChallenge] = []
        self.claims: dict[str, CandidateClaim] = {}
        self.bindings: dict[str, dict[str, FieldBinding]] = {}
        self.events: list[dict[str, Any]] = []
        self.dropped_evidence_ids: list[str] = []
        self.terminal = False
        self.stop_decision: dict[str, Any] | None = None

    def clone_state(
        self, *, environment: RealVideoToolEnvironment | None = None
    ) -> "VideoForensicHarness":
        clone = VideoForensicHarness(
            self.contract,
            self.environment if environment is None else environment,
            claim_schema=self.claim_schema,
            tool_registry=self.tool_registry,
            target_fields=self.target_fields,
            invariant_fields=self.invariant_fields,
        )
        clone.goals = copy.deepcopy(self.goals)
        clone.evidence_store = copy.deepcopy(self.evidence_store)
        clone.evidence_assessments = copy.deepcopy(self.evidence_assessments)
        clone.evidence_adjudications = copy.deepcopy(self.evidence_adjudications)
        clone.evidence_challenges = copy.deepcopy(self.evidence_challenges)
        clone.claims = copy.deepcopy(self.claims)
        clone.bindings = copy.deepcopy(self.bindings)
        clone.events = copy.deepcopy(self.events)
        clone.dropped_evidence_ids = copy.deepcopy(self.dropped_evidence_ids)
        clone.terminal = False
        clone.stop_decision = None
        return clone

    def render_state(self) -> dict[str, Any]:
        active_atoms = {
            evidence_id: asdict(atom)
            for evidence_id, atom in self.evidence_store.items()
            if atom.active
        }
        must_reopen_before_rebinding = []
        for claim_id, claim in self.claims.items():
            blocked_fields = [
                field_name
                for field_name, field in claim.fields.items()
                if field.decision_status == "committed" and field.evidence_status in {"unsupported", "conflicted"}
            ]
            if blocked_fields:
                must_reopen_before_rebinding.append({"claim_id": claim_id, "fields": blocked_fields})
        state = {
            "task_contract": asdict(self.contract),
            "case_goals": [asdict(goal) for goal in self.goals.values()],
            "candidate_claims": [self._claim_dict(claim) for claim in self.claims.values()],
            "field_bindings": {
                claim_id: {name: asdict(binding) for name, binding in fields.items()}
                for claim_id, fields in self.bindings.items()
            },
            "evidence_store": active_atoms,
            "evidence_assessments": {
                claim_id: {
                    evidence_id: asdict(assessment)
                    for evidence_id, assessment in assessments.items()
                }
                for claim_id, assessments in self.evidence_assessments.items()
            },
            "evidence_adjudications": {
                claim_id: asdict(adjudication)
                for claim_id, adjudication in self.evidence_adjudications.items()
            },
            "evidence_challenges": [
                asdict(challenge) for challenge in self.evidence_challenges
            ],
            "open_evidence_challenges": [
                asdict(challenge)
                for challenge in self.evidence_challenges
                if challenge.status == "open"
            ],
            "available_evidence_ids": list(active_atoms),
            "dropped_evidence_ids": list(self.dropped_evidence_ids),
            "open_high_priority_obligations": self._open_obligations(),
            "eligible_tools": [item["name"] for item in self.eligible_tool_registry()],
            "action_constraints": {
                "must_reopen_before_rebinding": must_reopen_before_rebinding,
                "rule": (
                    "For every listed claim field, call set_field_decision with decision_status=open before "
                    "attempting bind_evidence. A bind attempted while that field remains committed is rejected."
                ),
            },
            "terminal": self.terminal,
            "stop_decision": copy.deepcopy(self.stop_decision),
        }
        if any(item.get("name") == "adjudicate_evidence" for item in self.tool_registry):
            requirement = self._evidence_adjudication_requirement()
            state["evidence_adjudication"] = {
                "required": requirement is not None,
                "requirement": copy.deepcopy(requirement),
                "rule": (
                    "When required=true, adjudicate_evidence must partition every competing "
                    "active EvidenceAtom before bind_evidence. The subsequent bind must use "
                    "only selected_evidence_ids for the adjudicated fields. For every rejected "
                    "ID, defect_codes must exactly copy the accepted evidence_assessments ledger "
                    "in its recorded order; do not add, remove, or reorder codes."
                ),
            }
            if any(
                item.get("name") == "assess_evidence_candidate"
                for item in self.tool_registry
            ):
                pending = (
                    self._pending_evidence_assessment_ids(requirement)
                    if requirement is not None
                    else []
                )
                state["evidence_assessment"] = {
                    "required": bool(pending),
                    "pending_evidence_ids": pending,
                    "current_candidate_id": pending[0] if pending else None,
                    "deferred_candidate_ids": pending[1:] if pending else [],
                    "rule": (
                        "Assess only current_candidate_id this turn. Each accepted action "
                        "advances the queue by one; adjudication unlocks only after every "
                        "competitor is assessed. An assessment records visible comparison "
                        "state; it does not create media evidence or change claim values."
                    ),
                }
        state["state_action_affordances"] = derive_state_action_affordances(
            state, self.claim_schema
        )
        return state

    def _claim_revision_candidates(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Find unique alternative values stated by active visible evidence."""

        # Once competing evidence has been adjudicated, the selected visible atoms
        # are authoritative for a *proposed* local revision.  This deliberately lets
        # the claim follow the evidence instead of making claim compatibility a
        # prerequisite for selecting evidence.
        adjudicated: dict[str, dict[str, dict[str, Any]]] = {}
        for claim_id, decision in self.evidence_adjudications.items():
            claim = self.claims.get(claim_id)
            if claim is None:
                continue
            for field_name in decision.target_fields:
                field_state = claim.fields.get(field_name)
                values: list[tuple[Any, str]] = []
                for evidence_id in decision.selected_evidence_ids:
                    atom = self.evidence_store.get(evidence_id)
                    if (
                        atom is not None
                        and atom.active
                        and field_name in atom.supports_fields
                        and field_name in (atom.observation or {})
                    ):
                        values.append((copy.deepcopy(atom.observation[field_name]), evidence_id))
                if not values or any(
                    not self._same_measurement(values[0][0], value)
                    for value, _ in values[1:]
                ):
                    continue
                if field_state is not None and not self._same_measurement(
                    field_state.value, values[0][0]
                ):
                    adjudicated.setdefault(claim_id, {})[field_name] = {
                        "value": values[0][0],
                        "evidence_ids": [evidence_id for _, evidence_id in values],
                    }
        # Do not let rejected alternatives re-enter through the ordinary
        # single-alternative revision path after an adjudication, even when the
        # selected values already match the claim and no revision remains.
        if self.evidence_adjudications:
            return adjudicated

        # Ordinary unsupported draft claims must still go through binding, and
        # multi-atom disagreement must still go through adjudication.  Revision
        # is only a post-challenge recovery action.
        resolved_challenges = [
            challenge
            for challenge in self.evidence_challenges
            if challenge.status == "resolved" and challenge.resolved_by_evidence_ids
        ]
        if not resolved_challenges or self._evidence_adjudication_requirement() is not None:
            return {}
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for claim_id, claim in self.claims.items():
            for field_name, field_state in claim.fields.items():
                if field_state.evidence_status not in {"unsupported", "conflicted"}:
                    continue
                resolving_ids = {
                    evidence_id
                    for challenge in resolved_challenges
                    if field_name in challenge.target_fields
                    for evidence_id in challenge.resolved_by_evidence_ids
                }
                if not resolving_ids:
                    continue
                acceptable = set(
                    self.claim_schema["fields"][field_name].get(
                        "acceptable_evidence_kinds", []
                    )
                )
                alternatives: list[dict[str, Any]] = []
                for evidence_id, atom in self.evidence_store.items():
                    if (
                        evidence_id not in resolving_ids
                        or
                        not atom.active
                        or field_name not in atom.supports_fields
                        or atom.kind not in acceptable
                        or self._challenge_blocks_field_support(evidence_id, field_name)
                        or field_name not in atom.observation
                    ):
                        continue
                    value = copy.deepcopy(atom.observation[field_name])
                    if self._same_measurement(value, field_state.value):
                        continue
                    if not self._atom_supports_claim_value(field_name, value, atom):
                        continue
                    match = next(
                        (
                            item
                            for item in alternatives
                            if self._same_measurement(item["value"], value)
                        ),
                        None,
                    )
                    if match is None:
                        alternatives.append(
                            {"value": value, "evidence_ids": [evidence_id]}
                        )
                    else:
                        match["evidence_ids"].append(evidence_id)
                # Ambiguous competing values require evidence adjudication, not
                # an automatic-looking value revision surface.
                current_supported = any(
                    evidence_id in resolving_ids
                    and atom.active
                    and field_name in atom.supports_fields
                    and atom.kind in acceptable
                    and not self._challenge_blocks_field_support(evidence_id, field_name)
                    and self._atom_supports_claim_value(field_name, field_state.value, atom)
                    for evidence_id, atom in self.evidence_store.items()
                )
                if len(alternatives) == 1 and not current_supported:
                    result.setdefault(claim_id, {})[field_name] = alternatives[0]
        return result

    def eligible_tool_registry(self) -> list[dict[str, Any]]:
        """Return the state-dependent tool surface exposed to the policy."""
        if self.terminal:
            return []
        open_obligations = self._open_obligations()
        challenge_registered = any(
            item.get("name") == "challenge_evidence" for item in self.tool_registry
        )
        challenge_available = bool(
            challenge_registered
            and self.evidence_store
            and not any(
                challenge.status == "open" for challenge in self.evidence_challenges
            )
        )
        if not open_obligations:
            names = {"stop"}
            if challenge_available:
                names.add("challenge_evidence")
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["name"] in names
            ]
        must_reopen = any(
            field.decision_status == "committed"
            and field.evidence_status in {"unsupported", "conflicted"}
            for claim in self.claims.values()
            for field in claim.fields.values()
        )
        challenge_measurement_pending = bool(
            self.goals["G-DIFFERENCE"].status != "satisfied"
            and any(
                challenge.status == "resolved"
                and challenge.requested_check == "remeasure_evidence"
                for challenge in self.evidence_challenges
            )
        )
        revision_candidates = self._claim_revision_candidates()
        if revision_candidates:
            next_name = "set_field_decision" if must_reopen else "revise_claim_fields"
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["name"] == next_name
            ]
        if must_reopen and not challenge_measurement_pending:
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["name"] == "set_field_decision"
            ]
        selector = getattr(self.environment, "eligible_tool_names", None)
        enabled_external = set(selector()) if callable(selector) else None
        open_goal_ids = {
            goal.goal_id
            for goal in self.goals.values()
            if goal.priority == "high" and goal.status != "satisfied"
        }
        active_external_goal = (
            "G-CONTEXT"
            if "G-CONTEXT" in open_goal_ids
            else "G-DIFFERENCE"
            if "G-DIFFERENCE" in open_goal_ids
            else None
        )
        if active_external_goal is not None:
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["domain"] == "external_media"
                and item.get(
                    "satisfies_goal",
                    "G-CONTEXT" if item["name"] == "inspect_media" else "G-DIFFERENCE",
                )
                == active_external_goal
                and (enabled_external is None or item["name"] in enabled_external)
            ]
        adjudication_required = self._evidence_adjudication_requirement()
        assessment_registered = any(
            item.get("name") == "assess_evidence_candidate"
            for item in self.tool_registry
        )
        adjudication_registered = any(
            item.get("name") == "adjudicate_evidence" for item in self.tool_registry
        )
        if (
            adjudication_required is not None
            and assessment_registered
            and self._pending_evidence_assessment_ids(adjudication_required)
        ):
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["name"] == "assess_evidence_candidate"
            ]
        if adjudication_required is not None and adjudication_registered:
            return [
                copy.deepcopy(item)
                for item in self.tool_registry
                if item["name"] == "adjudicate_evidence"
            ]
        if not self.claims:
            meaningful_internal = {"create_candidate_claim"}
        else:
            needs_binding = any(
                field.decision_status == "open"
                and field.evidence_status in {"unsupported", "conflicted"}
                for claim in self.claims.values()
                for field in claim.fields.values()
            )
            needs_commit = any(
                field.decision_status == "open" and field.evidence_status == "supported"
                for claim in self.claims.values()
                for field in claim.fields.values()
            )
            if needs_binding:
                meaningful_internal = {"bind_evidence"}
            elif needs_commit:
                meaningful_internal = {"set_field_decision"}
            else:
                # Recovery surface for unusual nonterminal states. Normal dependency
                # order resolves through create -> bind -> commit -> stop above.
                meaningful_internal = {"set_field_decision"}
        if challenge_available:
            meaningful_internal.add("challenge_evidence")
        return [
            copy.deepcopy(item)
            for item in self.tool_registry
            if item["name"] in meaningful_internal
        ]

    @staticmethod
    def _claim_dict(claim: CandidateClaim) -> dict[str, Any]:
        return {
            "claim_id": claim.claim_id,
            "schema_id": claim.schema_id,
            "fields": {name: asdict(value) for name, value in claim.fields.items()},
        }

    def execute_tool(self, name: str, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        registry_entry = next((item for item in self.eligible_tool_registry() if item["name"] == name), None)
        if registry_entry is None:
            raise ValueError(f"tool is not eligible in the current state: {name}")
        if registry_entry["domain"] == "external_media":
            result = self._execute_external(name, arguments, tool_call_id)
        elif name == "create_candidate_claim":
            result = self._create_claim(arguments, tool_call_id)
        elif name == "bind_evidence":
            result = self._bind(arguments, tool_call_id)
        elif name == "assess_evidence_candidate":
            result = self._assess_evidence_candidate(arguments, tool_call_id)
        elif name == "adjudicate_evidence":
            result = self._adjudicate_evidence(arguments, tool_call_id)
        elif name == "challenge_evidence":
            result = self._challenge_evidence(arguments, tool_call_id)
        elif name == "revise_claim_fields":
            result = self._revise_claim_fields(arguments, tool_call_id)
        elif name == "set_field_decision":
            result = self._set_decision(arguments, tool_call_id)
        elif name == "stop":
            result = self._stop(tool_call_id)
        else:
            raise ValueError(f"tool is not registered: {name}")
        return result

    def _challenge_evidence(
        self, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        """Turn a policy diagnosis into a real, non-prescriptive verification step."""

        evidence_ids = [str(item) for item in arguments.get("evidence_ids") or []]
        defect_codes = [str(item) for item in arguments.get("defect_codes") or []]
        target_fields = [str(item) for item in arguments.get("target_fields") or []]
        requested_check = str(arguments.get("requested_check") or "")
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("challenge_evidence requires unique active evidence_ids")
        missing = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in self.evidence_store
            or not self.evidence_store[evidence_id].active
        ]
        if missing:
            raise ValueError(f"challenged evidence is not active: {missing}")
        if (
            not defect_codes
            or len(defect_codes) != len(set(defect_codes))
            or not set(defect_codes).issubset(EVIDENCE_CHALLENGE_DEFECT_CODES)
        ):
            raise ValueError("challenge_evidence defect_codes are invalid")
        if (
            not target_fields
            or len(target_fields) != len(set(target_fields))
            or not set(target_fields).issubset(self.required_fields)
        ):
            raise ValueError("challenge_evidence target_fields are invalid")
        connected_fields = {
            field_name
            for evidence_id in evidence_ids
            for field_name in self.evidence_store[evidence_id].supports_fields
        }
        connected_fields.update(
            field_name
            for claim_bindings in self.bindings.values()
            for field_name, binding in claim_bindings.items()
            if set(binding.support_ids).intersection(evidence_ids)
        )
        if not set(target_fields).issubset(connected_fields):
            raise ValueError(
                "challenge_evidence target_fields must be visibly connected to the "
                "challenged evidence"
            )
        if requested_check not in {"reinspect_context", "remeasure_evidence"}:
            raise ValueError("challenge_evidence requested_check is invalid")
        if any(challenge.status == "open" for challenge in self.evidence_challenges):
            raise ValueError("an evidence challenge is already awaiting verification")

        challenge = EvidenceChallenge(
            challenge_id=f"CHALLENGE-{len(self.evidence_challenges) + 1:03d}",
            evidence_ids=evidence_ids,
            defect_codes=defect_codes,
            target_fields=target_fields,
            requested_check=requested_check,
        )
        self.evidence_challenges.append(challenge)
        challenge_hook = getattr(self.environment, "on_evidence_challenge", None)
        if callable(challenge_hook):
            challenge_hook(
                requested_check,
                list(evidence_ids),
                list(defect_codes),
            )
        goal_id = (
            "G-CONTEXT"
            if requested_check == "reinspect_context"
            else "G-DIFFERENCE"
        )
        self.goals[goal_id].status = "open"
        self.goals[goal_id].evidence_ids = []
        result = {
            "status": "accepted",
            "authority": "policy_evidence_challenge_checker",
            "challenge": asdict(challenge),
            "created_evidence_ids": [],
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="evidence_integrity_challenge_validation",
            action="challenge_evidence",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _pending_evidence_assessment_ids(
        self, requirement: dict[str, Any] | None
    ) -> list[str]:
        if not requirement:
            return []
        claim_id = str(requirement["claim_id"])
        target_fields = [str(item) for item in requirement["target_fields"]]
        considered = [
            str(item) for item in requirement["competing_evidence_ids"]
        ]
        existing = self.evidence_assessments.get(claim_id, {})
        return [
            evidence_id
            for evidence_id in considered
            if evidence_id not in existing
            or existing[evidence_id].target_fields != target_fields
            or existing[evidence_id].considered_evidence_ids != considered
        ]

    def _assess_evidence_candidate(
        self, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        requirement = self._evidence_adjudication_requirement()
        if requirement is None:
            raise ValueError("no visible evidence conflict currently requires assessment")
        claim_id = str(arguments.get("claim_id"))
        if claim_id != requirement["claim_id"]:
            raise ValueError("assessment claim_id does not match the visible conflict")
        target_fields = [str(item) for item in arguments.get("target_fields") or []]
        if len(target_fields) != len(set(target_fields)) or set(target_fields) != set(
            requirement["target_fields"]
        ):
            raise ValueError("assessment target_fields must cover the visible conflict exactly")
        evidence_id = str(arguments.get("evidence_id"))
        pending = self._pending_evidence_assessment_ids(requirement)
        if not pending or evidence_id != pending[0]:
            raise ValueError("assessment must address the current visible candidate only")
        disposition = str(arguments.get("disposition"))
        defect_codes = [str(item) for item in arguments.get("defect_codes") or []]
        supported_defects = self._visible_evidence_defects(requirement, evidence_id)
        atom = self.evidence_store[evidence_id]
        supports_all_targets = all(
            field_name in atom.supports_fields
            and field_name in (atom.observation or {})
            for field_name in requirement["target_fields"]
        )
        hard_defect_names = {"role_assignment_mismatch"}
        if (atom.observation or {}).get("preferred_video") != "c":
            hard_defect_names.add("order_inconsistent")
        hard_defects = hard_defect_names.intersection(supported_defects)
        if disposition == "retain":
            if defect_codes or hard_defects or not supports_all_targets:
                raise ValueError(
                    "retain requires complete visible evidence with no role or directional-order defect"
                )
        elif disposition == "reject":
            if (
                not defect_codes
                or len(defect_codes) != len(set(defect_codes))
                or not set(defect_codes).issubset(supported_defects)
            ):
                raise ValueError(
                    "reject defect_codes must be nonempty and supported by visible evidence"
                )
        else:
            raise ValueError("assessment disposition must be retain or reject")
        assessment = EvidenceCandidateAssessment(
            assessment_id=(
                f"ASSESS-{sum(len(items) for items in self.evidence_assessments.values()) + 1:03d}"
            ),
            claim_id=claim_id,
            target_fields=list(requirement["target_fields"]),
            evidence_id=evidence_id,
            disposition=disposition,
            defect_codes=defect_codes,
            considered_evidence_ids=list(requirement["competing_evidence_ids"]),
        )
        self.evidence_assessments.setdefault(claim_id, {})[
            assessment.evidence_id
        ] = assessment
        result = {
            "status": "accepted",
            "authority": "visible_candidate_assessment_checker",
            "assessment": asdict(assessment),
            "created_evidence_ids": [],
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="evidence_candidate_assessment_validation",
            action="assess_evidence_candidate",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def record_policy_call(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        reasoning: str,
    ) -> None:
        registry_entry = next(item for item in self.tool_registry if item["name"] == name)
        action_type = (
            "external_tool_proposal"
            if registry_entry["domain"] == "external_media"
            else "state_edit_proposal"
            if registry_entry["domain"] == "harness_state"
            else "terminal_proposal"
        )
        self._record(
            actor="policy",
            action_type=action_type,
            action=name,
            tool_call_id=tool_call_id,
            result={
                "reasoning": reasoning,
                "proposed_arguments": copy.deepcopy(arguments),
                "state_changed": False,
            },
        )

    def record_rejected_call(
        self,
        name: str,
        tool_call_id: str,
        result: dict[str, Any],
    ) -> None:
        """Record a failed proposal without pretending that it mutated harness state."""
        self._record(
            actor="harness",
            action_type="execution_rejection",
            action=name,
            tool_call_id=tool_call_id,
            result={**copy.deepcopy(result), "state_changed": False},
        )

    def _record(
        self,
        *,
        actor: str,
        action_type: str,
        action: str,
        tool_call_id: str | None,
        result: dict[str, Any],
    ) -> None:
        self.events.append(
            {
                "event_id": f"EVENT-{len(self.events) + 1:03d}",
                "actor": actor,
                "action_type": action_type,
                "action": action,
                "tool_call_id": tool_call_id,
                "result": copy.deepcopy(result),
                "state_after": self.render_state(),
            }
        )

    def _execute_external(self, name: str, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        observation = self.environment.execute(name, arguments)
        atoms = observation.pop("evidence_atoms")
        for atom in atoms:
            if atom.evidence_id in self.evidence_store:
                raise ValueError(f"duplicate evidence id: {atom.evidence_id}")
            self.evidence_store[atom.evidence_id] = atom
        if atoms:
            self.evidence_adjudications.clear()

        produced_ids = [atom.evidence_id for atom in atoms]
        registry_entry = next(item for item in self.tool_registry if item["name"] == name)
        goal_effect = registry_entry.get(
            "satisfies_goal",
            "G-CONTEXT" if name == "inspect_media" else "G-DIFFERENCE",
        )
        for challenge in self.evidence_challenges:
            expected_goal = (
                "G-CONTEXT"
                if challenge.requested_check == "reinspect_context"
                else "G-DIFFERENCE"
            )
            if challenge.status == "open" and goal_effect == expected_goal:
                challenge.status = "resolved"
                challenge.resolved_by_evidence_ids = list(produced_ids)
                self._reconcile_resolved_challenge(challenge)
        if goal_effect == "G-CONTEXT":
            self.goals["G-CONTEXT"].status = "satisfied"
            self.goals["G-CONTEXT"].evidence_ids = produced_ids
        if goal_effect == "G-DIFFERENCE":
            # Adaptive environments may return a real but explicitly
            # inconclusive measurement.  Such an observation is useful for
            # routing the next tool, but must not close the evidence goal.
            goal_satisfied = observation.get("goal_satisfied", True) is True
            comparison_scope = self.contract.visible_assets.get("comparison_scope_s")
            scope_issues = evidence_interval_scope_issues(atoms, comparison_scope)
            if scope_issues:
                goal_satisfied = False
            observation["scope_integrity"] = {
                "checked": isinstance(comparison_scope, list),
                "passed": not scope_issues,
                "issues": scope_issues,
            }
            open_claim_gaps: list[tuple[str, str, Any]] = [
                (claim_id, field_name, field_state.value)
                for claim_id, claim in self.claims.items()
                for field_name, field_state in claim.fields.items()
                if field_state.evidence_status in {"unsupported", "conflicted"}
            ]
            incompatible_gaps: list[str] = []
            for claim_id, field_name, value in open_claim_gaps:
                field_contract = self.claim_schema["fields"].get(field_name, {})
                acceptable = set(field_contract.get("acceptable_evidence_kinds", []))
                compatible = any(
                    atom.active
                    and field_name in atom.supports_fields
                    and atom.kind in acceptable
                    and self._atom_supports_claim_value(field_name, value, atom)
                    for atom in self.evidence_store.values()
                )
                if not compatible:
                    incompatible_gaps.append(f"{claim_id}:{field_name}")
            if goal_satisfied and incompatible_gaps:
                goal_satisfied = False
                incompatibility_hook = getattr(
                    self.environment, "on_measurement_incompatible_with_claim", None
                )
                if callable(incompatibility_hook):
                    incompatibility_hook(name, list(incompatible_gaps))
            observation["claim_gap_compatibility"] = {
                "checked": bool(open_claim_gaps),
                "all_open_claim_values_supported": not incompatible_gaps,
                "incompatible_claim_fields": incompatible_gaps,
                "selection_basis": "active student-visible evidence and current claim values only",
            }
            self.goals["G-DIFFERENCE"].status = "satisfied" if goal_satisfied else "open"
            self.goals["G-DIFFERENCE"].evidence_ids = produced_ids

        result = {
            "status": "executed",
            "authority": "external_tool_environment",
            "observation": observation,
            "evidence_atoms": [asdict(atom) for atom in atoms],
            "created_evidence_ids": produced_ids,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="environment",
            action_type="external_tool_execution",
            action=name,
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _challenge_blocks_field_support(
        self, evidence_id: str, field_name: str
    ) -> bool:
        """A fresh check supersedes challenged support only for policy-named fields."""

        return any(
            challenge.status == "resolved"
            and evidence_id in challenge.evidence_ids
            and field_name in challenge.target_fields
            for challenge in self.evidence_challenges
        )

    def _reconcile_resolved_challenge(self, challenge: EvidenceChallenge) -> None:
        """Open only the policy-selected field dependencies after fresh evidence.

        The Harness does not select fields or claim values.  It only prevents the
        exact challenged evidence/field pair from silently remaining authoritative
        after the requested fresh check has actually executed.
        """

        changed = False
        challenged = set(challenge.evidence_ids)
        for claim_id, claim in self.claims.items():
            claim_bindings = self.bindings.get(claim_id, {})
            for field_name in challenge.target_fields:
                if field_name not in claim.fields:
                    continue
                field_state = claim.fields[field_name]
                binding = claim_bindings.get(field_name)
                removed = []
                if binding is not None:
                    removed = [
                        evidence_id
                        for evidence_id in binding.support_ids
                        if evidence_id in challenged
                    ]
                visibly_connected = any(
                    field_name in self.evidence_store[evidence_id].supports_fields
                    for evidence_id in challenge.evidence_ids
                    if evidence_id in self.evidence_store
                )
                if not removed and not visibly_connected:
                    continue
                if binding is not None and removed:
                    binding.support_ids = [
                        evidence_id
                        for evidence_id in binding.support_ids
                        if evidence_id not in challenged
                    ]
                    binding.validation_status = (
                        "superseded_by_policy_challenge_and_fresh_check"
                    )
                    field_state.support_ids = list(binding.support_ids)
                    field_state.evidence_status = (
                        "supported" if binding.support_ids else "unsupported"
                    )
                else:
                    # An unbound but active competitor can still invalidate confidence
                    # in the current field.  The policy chose this minimal field subset;
                    # the Harness merely makes that uncertainty executable.
                    field_state.evidence_status = "conflicted"
                    field_state.attack_ids = sorted(
                        set(field_state.attack_ids).union(challenged)
                    )
                changed = True
        if changed:
            self.evidence_assessments.clear()
            self.evidence_adjudications.clear()
            self.goals["G-REPORT"].status = "open"
            self.goals["G-REPORT"].evidence_ids = []

    def _create_claim(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        claim_id = arguments.get("claim_id")
        values = arguments.get("field_values")
        if not isinstance(claim_id, str) or not isinstance(values, dict):
            raise ValueError("create_candidate_claim requires claim_id and field_values")
        if claim_id in self.claims:
            raise ValueError(f"claim already exists: {claim_id}")
        missing = [name for name in self.required_fields if name not in values]
        extra = [name for name in values if name not in self.required_fields]
        if missing or extra:
            raise ValueError(f"claim schema mismatch; missing={missing}, extra={extra}")
        value_errors = self._validate_claim_values(values)
        if value_errors:
            raise ValueError(f"claim value validation failed: {value_errors}")
        fields = {name: ClaimFieldState(value=copy.deepcopy(values[name])) for name in self.required_fields}
        self.claims[claim_id] = CandidateClaim(claim_id, self.claim_schema["schema_id"], fields)
        self.bindings[claim_id] = {}
        result = {
            "status": "accepted",
            "authority": "schema_checker",
            "created_claim_id": claim_id,
            "field_obligations": [
                {
                    "obligation_id": f"OBL-{claim_id}-{name}",
                    "field": name,
                    "acceptable_evidence_kinds": self.claim_schema["fields"][name]["acceptable_evidence_kinds"],
                    "evidence_status": "unsupported",
                    "decision_status": "open",
                }
                for name in self.required_fields
            ],
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="policy_state_edit_validation",
            action="create_candidate_claim",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _bind(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        claim_id = arguments.get("claim_id")
        proposals = arguments.get("bindings")
        if claim_id not in self.claims or not isinstance(proposals, dict):
            raise ValueError("bind_evidence requires an existing claim and binding map")

        accepted: list[str] = []
        rejected: dict[str, str] = {}
        for field_name, support_ids in proposals.items():
            if field_name not in self.required_fields:
                rejected[field_name] = "unknown_field"
                continue
            if not isinstance(support_ids, list) or not support_ids:
                rejected[field_name] = "support_ids_must_be_nonempty"
                continue
            if any(
                self._challenge_blocks_field_support(evidence_id, field_name)
                for evidence_id in support_ids
            ):
                rejected[field_name] = (
                    "evidence_superseded_by_resolved_challenge_for_field"
                )
                continue
            atoms = [self.evidence_store.get(evidence_id) for evidence_id in support_ids]
            if any(atom is None or not atom.active for atom in atoms):
                rejected[field_name] = "missing_or_inactive_evidence"
                continue
            if any(field_name not in atom.supports_fields for atom in atoms if atom is not None):
                rejected[field_name] = "evidence_does_not_support_field"
                continue
            acceptable = set(self.claim_schema["fields"][field_name]["acceptable_evidence_kinds"])
            if any(atom.kind not in acceptable for atom in atoms if atom is not None):
                rejected[field_name] = "incompatible_evidence_kind"
                continue
            field_state = self.claims[claim_id].fields[field_name]
            adjudication = self.evidence_adjudications.get(claim_id)
            if (
                adjudication is not None
                and field_name in adjudication.target_fields
                and not set(support_ids).issubset(adjudication.selected_evidence_ids)
            ):
                rejected[field_name] = "evidence_not_selected_by_latest_adjudication"
                continue
            if field_state.decision_status == "committed" and field_state.evidence_status != "supported":
                rejected[field_name] = "must_reopen_invalidated_field_before_rebinding"
                continue
            existing_binding = self.bindings[claim_id].get(field_name)
            if (
                existing_binding is not None
                and field_state.evidence_status == "supported"
                and existing_binding.support_ids == support_ids
            ):
                rejected[field_name] = "binding_already_active"
                continue
            claim_value = self.claims[claim_id].fields[field_name].value
            if any(
                not self._atom_supports_claim_value(field_name, claim_value, atom)
                for atom in atoms
                if atom is not None
            ):
                rejected[field_name] = "claim_value_not_supported_by_evidence"
                continue
            binding = FieldBinding(field=field_name, support_ids=list(support_ids))
            self.bindings[claim_id][field_name] = binding
            field_state.support_ids = list(support_ids)
            field_state.attack_ids = []
            field_state.evidence_status = "supported"
            accepted.append(field_name)

        result = {
            "status": "accepted" if accepted and not rejected else "partially_accepted" if accepted else "rejected",
            "authority": "contract_and_provenance_checker",
            "accepted_fields": accepted,
            "rejected_fields": rejected,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="binding_validation",
            action="bind_evidence",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _evidence_adjudication_requirement(self) -> dict[str, Any] | None:
        """Find one visible conflict that must be resolved before binding."""

        for claim_id, claim in self.claims.items():
            open_fields = [
                field_name
                for field_name, field_state in claim.fields.items()
                if field_state.decision_status == "open"
            ]
            if not open_fields:
                continue
            compatible_by_field: dict[str, list[str]] = {}
            candidates_by_field: dict[str, list[str]] = {}
            target_fields: list[str] = []
            for field_name in open_fields:
                acceptable = set(
                    (self.claim_schema.get("fields") or {})
                    .get(field_name, {})
                    .get("acceptable_evidence_kinds", [])
                )
                candidates = sorted(
                    evidence_id
                    for evidence_id, atom in self.evidence_store.items()
                    if atom.active
                    and field_name in atom.supports_fields
                    and atom.kind in acceptable
                    and not self._challenge_blocks_field_support(
                        evidence_id, field_name
                    )
                )
                compatible = []
                for evidence_id in candidates:
                    atom = self.evidence_store[evidence_id]
                    role_invalid = (
                        (atom.provenance or {}).get(
                            "role_assignment_valid_for_current_contract"
                        )
                        is False
                        and field_name
                        in {
                            "direction",
                            "affected_source_interval_s",
                            "affected_suspect_interval_s",
                        }
                    )
                    if not role_invalid and self._atom_supports_claim_value(
                        field_name,
                        claim.fields[field_name].value,
                        atom,
                    ):
                        compatible.append(evidence_id)
                if candidates:
                    candidates_by_field[field_name] = candidates
                if compatible:
                    compatible_by_field[field_name] = compatible
                visible_values = [
                    (self.evidence_store[evidence_id].observation or {}).get(field_name)
                    for evidence_id in candidates
                    if field_name in (self.evidence_store[evidence_id].observation or {})
                ]
                has_visible_conflict = len(candidates) >= 2 and any(
                    not self._same_measurement(visible_values[0], value)
                    for value in visible_values[1:]
                ) if visible_values else False
                if has_visible_conflict:
                    target_fields.append(field_name)
            if not target_fields:
                continue
            compatible_by_field = {
                field_name: compatible_by_field.get(field_name, [])
                for field_name in target_fields
            }
            candidates_by_field = {
                field_name: candidates_by_field.get(field_name, [])
                for field_name in target_fields
            }
            considered = sorted(
                {
                    evidence_id
                    for candidates in candidates_by_field.values()
                    for evidence_id in candidates
                }
            )
            if len(considered) < 2:
                continue
            previous = self.evidence_adjudications.get(claim_id)
            if (
                previous is not None
                and previous.target_fields == target_fields
                and previous.considered_evidence_ids == considered
            ):
                continue
            return {
                "claim_id": claim_id,
                "target_fields": target_fields,
                "competing_evidence_ids": considered,
                "compatible_evidence_ids_by_field": compatible_by_field,
                "selection_basis": (
                    "active student-visible EvidenceAtoms, one vote per measurement method, "
                    "cross-method direction agreement, completeness, confidence, and order "
                    "consistency; current claim values do not select the winner"
                ),
            }
        return None

    def _evidence_first_selection(self, requirement: dict[str, Any]) -> list[str]:
        """Select a visible candidate without consulting the current claim value."""

        candidates = [
            evidence_id
            for evidence_id in requirement["competing_evidence_ids"]
            if evidence_id in self.evidence_store and self.evidence_store[evidence_id].active
        ]
        target_fields = list(requirement["target_fields"])
        complete = [
            evidence_id
            for evidence_id in candidates
            if all(
                field_name in self.evidence_store[evidence_id].supports_fields
                and field_name in (self.evidence_store[evidence_id].observation or {})
                for field_name in target_fields
            )
            and (self.evidence_store[evidence_id].provenance or {}).get(
                "role_assignment_valid_for_current_contract", True
            ) is not False
        ]
        if not complete:
            raise ValueError("no complete visible evidence candidate can be adjudicated")

        def confidence(evidence_id: str) -> float:
            value = (self.evidence_store[evidence_id].observation or {}).get("confidence")
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else -1.0

        if all(
            (self.evidence_store[evidence_id].observation or {}).get("preferred_video")
            in {"a", "b", "c"}
            for evidence_id in complete
        ):
            # Repeated executions of one measurement method are one independent vote.
            method_representatives: dict[str, str] = {}
            for evidence_id in complete:
                observation = self.evidence_store[evidence_id].observation or {}
                method = str(observation.get("measurement_method") or (self.evidence_store[evidence_id].provenance or {}).get("method") or evidence_id)
                incumbent = method_representatives.get(method)
                if incumbent is None or (confidence(evidence_id), evidence_id) > (
                    confidence(incumbent), incumbent
                ):
                    method_representatives[method] = evidence_id
            votes = Counter(
                str((self.evidence_store[evidence_id].observation or {}).get("preferred_video"))
                for evidence_id in method_representatives.values()
            )
            directional = [value for value in ("a", "b") if votes[value] >= 2]
            if len(directional) == 1 and votes["b" if directional[0] == "a" else "a"] == 0:
                selected_value = directional[0]
            elif votes["c"]:
                selected_value = "c"
            elif votes["a"] != votes["b"]:
                selected_value = "a" if votes["a"] > votes["b"] else "b"
            else:
                raise ValueError("cross-method evidence is tied without an abstention candidate")
            matching = [
                evidence_id
                for evidence_id in complete
                if (self.evidence_store[evidence_id].observation or {}).get("preferred_video")
                == selected_value
            ]
            stable = [
                evidence_id
                for evidence_id in matching
                if selected_value == "c"
                or (self.evidence_store[evidence_id].observation or {}).get("order_consistent")
                is not False
            ]
            pool = stable or matching
            return [max(pool, key=lambda evidence_id: (confidence(evidence_id), evidence_id))]

        return [max(complete, key=lambda evidence_id: (confidence(evidence_id), evidence_id))]

    def _visible_evidence_defects(
        self,
        requirement: dict[str, Any],
        evidence_id: str,
        *,
        selected_evidence_ids: list[str] | None = None,
    ) -> set[str]:
        """Derive defensible defect labels only from student-visible structured facts."""

        if evidence_id not in requirement["competing_evidence_ids"]:
            raise ValueError("evidence_id is not a visible competitor")
        atom = self.evidence_store[evidence_id]
        target_fields = list(requirement["target_fields"])
        compatible_by_field = requirement["compatible_evidence_ids_by_field"]
        supported_target_fields = [
            field_name for field_name in target_fields if field_name in atom.supports_fields
        ]
        defects: set[str] = set()
        if any(
            evidence_id not in compatible_by_field.get(field_name, [])
            for field_name in supported_target_fields
        ) or not supported_target_fields:
            defects.add("claim_value_conflict")
        observation = atom.observation or {}
        if (atom.provenance or {}).get(
            "role_assignment_valid_for_current_contract"
        ) is False:
            defects.add("role_assignment_mismatch")
        if observation.get("order_consistent") is False or _visible_temporal_order_inconsistent(
            observation, target_fields
        ):
            defects.add("order_inconsistent")
        claim = self.claims[str(requirement["claim_id"])]
        if "preferred_video" in claim.fields:
            preferred_claim = claim.fields["preferred_video"].value
            if observation.get("preferred_video") == "c" and preferred_claim != "c":
                defects.add("abstention_conflicts_with_claim")

        comparison_ids = selected_evidence_ids or [
            candidate_id
            for candidate_id in requirement["competing_evidence_ids"]
            if candidate_id != evidence_id
        ]
        preferred_value = observation.get("preferred_video")
        comparison_values = {
            (self.evidence_store[candidate_id].observation or {}).get("preferred_video")
            for candidate_id in comparison_ids
            if candidate_id in self.evidence_store
        }
        if preferred_value is not None and any(
            value is not None and value != preferred_value for value in comparison_values
        ):
            defects.add("cross_method_direction_conflict")
        if preferred_value is not None and preferred_value in comparison_values:
            defects.add("redundant_same_conclusion")
        confidence = observation.get("confidence")
        other_confidences = [
            (self.evidence_store[candidate_id].observation or {}).get("confidence")
            for candidate_id in comparison_ids
            if candidate_id in self.evidence_store
        ]
        numeric_others = [
            float(value)
            for value in other_confidences
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and numeric_others
            and float(confidence) < max(numeric_others)
        ):
            defects.add("lower_confidence")

        if selected_evidence_ids:
            candidate_compatible_fields = {
                field_name
                for field_name in target_fields
                if evidence_id in compatible_by_field.get(field_name, [])
            }
            selected_compatible_fields = {
                field_name
                for field_name in target_fields
                if set(selected_evidence_ids).intersection(
                    compatible_by_field.get(field_name, [])
                )
            }
            if candidate_compatible_fields and candidate_compatible_fields.issubset(
                selected_compatible_fields
            ):
                defects.add("redundant_same_conclusion")
        else:
            candidate_compatible_fields = {
                field_name
                for field_name in target_fields
                if evidence_id in compatible_by_field.get(field_name, [])
            }
            alternative_compatible_fields = {
                field_name
                for field_name in target_fields
                if set(comparison_ids).intersection(
                    compatible_by_field.get(field_name, [])
                )
            }
            if (
                candidate_compatible_fields
                and candidate_compatible_fields.issubset(
                    alternative_compatible_fields
                )
            ):
                defects.add("redundant_same_conclusion")
        return defects

    def _adjudicate_evidence(
        self, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        requirement = self._evidence_adjudication_requirement()
        if requirement is None:
            raise ValueError("no visible evidence conflict currently requires adjudication")
        claim_id = str(arguments.get("claim_id"))
        if claim_id != requirement["claim_id"]:
            raise ValueError("adjudication claim_id does not match the visible conflict")
        target_fields = [str(item) for item in arguments.get("target_fields") or []]
        if len(target_fields) != len(set(target_fields)) or set(target_fields) != set(
            requirement["target_fields"]
        ):
            raise ValueError("adjudication target_fields must cover the visible conflict exactly")
        selected = [str(item) for item in arguments.get("selected_evidence_ids") or []]
        rejected_items = copy.deepcopy(arguments.get("rejected_evidence") or [])
        rejected_ids = [str(item.get("evidence_id")) for item in rejected_items]
        competing = list(requirement["competing_evidence_ids"])
        if (
            not selected
            or not rejected_items
            or len(selected) != len(set(selected))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(selected).intersection(rejected_ids)
            or set(selected).union(rejected_ids) != set(competing)
        ):
            raise ValueError(
                "adjudication partition must exactly match every visible competitor once"
            )
        if selected != self._evidence_first_selection(requirement):
            raise ValueError("selected evidence does not match the label-blind cross-method rule")
        if any(
            not any(
                field_name in self.evidence_store[evidence_id].supports_fields
                and field_name in (self.evidence_store[evidence_id].observation or {})
                for evidence_id in selected
            )
            for field_name in requirement["target_fields"]
        ):
            raise ValueError("selected evidence does not observe every target field")
        for item in rejected_items:
            defect_codes = [str(code) for code in item.get("defect_codes") or []]
            supported = self._visible_evidence_defects(
                requirement,
                str(item.get("evidence_id")),
                selected_evidence_ids=selected,
            )
            if (
                not defect_codes
                or len(defect_codes) != len(set(defect_codes))
                or not set(defect_codes).issubset(supported)
            ):
                raise ValueError(
                    "each rejected evidence item needs a visible, defensible defect code"
                )
        if any(
            item.get("name") == "assess_evidence_candidate"
            for item in self.tool_registry
        ):
            assessments = self.evidence_assessments.get(claim_id, {})
            expected_dispositions = {evidence_id: ("retain", []) for evidence_id in selected}
            expected_dispositions.update(
                {
                    item["evidence_id"]: (
                        "reject",
                        list(item["defect_codes"]),
                    )
                    for item in rejected_items
                }
            )
            for evidence_id in requirement["competing_evidence_ids"]:
                assessment = assessments.get(evidence_id)
                expected = expected_dispositions.get(evidence_id)
                if (
                    assessment is None
                    or expected is None
                    or assessment.disposition != expected[0]
                    or assessment.defect_codes != expected[1]
                    or assessment.target_fields != requirement["target_fields"]
                    or assessment.considered_evidence_ids
                    != requirement["competing_evidence_ids"]
                ):
                    raise ValueError(
                        "adjudicate_evidence must consume a complete matching visible "
                        "candidate-assessment ledger"
                    )
        adjudication = EvidenceAdjudication(
            adjudication_id=f"ADJ-{len(self.evidence_adjudications) + 1:03d}",
            claim_id=claim_id,
            target_fields=list(requirement["target_fields"]),
            selected_evidence_ids=selected,
            rejected_evidence=rejected_items,
            considered_evidence_ids=list(requirement["competing_evidence_ids"]),
        )
        self.evidence_adjudications[claim_id] = adjudication
        result = {
            "status": "accepted",
            "authority": "visible_evidence_adjudication_checker",
            "adjudication": asdict(adjudication),
            "created_evidence_ids": [],
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="evidence_adjudication_validation",
            action="adjudicate_evidence",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _validate_claim_values(self, values: dict[str, Any]) -> dict[str, str]:
        errors: dict[str, str] = {}
        for field_name, spec in self.claim_schema["fields"].items():
            value = values[field_name]
            allowed_types = spec["types"]
            type_ok = (
                ("str" in allowed_types and isinstance(value, str))
                or ("list" in allowed_types and isinstance(value, list))
                or (
                    any(name in allowed_types for name in ("float", "int"))
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                )
            )
            if not type_ok:
                errors[field_name] = "wrong_type"
                continue
            if "enum" in spec and value not in spec["enum"]:
                errors[field_name] = "outside_public_ontology"

        for field_name in [
            name for name in self.required_fields if name.endswith("_interval_s") and name != "omitted_source_intervals_s"
        ]:
            value = values[field_name]
            if not self._is_interval(value):
                errors[field_name] = "invalid_interval"
        if "omitted_source_intervals_s" in values:
            omitted = values["omitted_source_intervals_s"]
            if not isinstance(omitted, list) or not omitted or not all(self._is_interval(item) for item in omitted):
                errors["omitted_source_intervals_s"] = "requires_nonempty_valid_intervals"
            elif any(omitted[index][0] < omitted[index - 1][1] for index in range(1, len(omitted))):
                errors["omitted_source_intervals_s"] = "intervals_overlap_or_are_unsorted"

        scope = values.get("comparison_scope_s")
        visible_scope = self.contract.visible_assets.get("comparison_scope_s")
        if self._is_interval(scope) and not self._same_measurement(scope, visible_scope):
            errors["comparison_scope_s"] = "does_not_match_visible_task_scope"
        if isinstance(values.get("source_id"), str) and not values["source_id"].strip():
            errors["source_id"] = "empty_identifier"
        removed = values.get("estimated_removed_time_s")
        if isinstance(removed, (int, float)) and not isinstance(removed, bool) and removed < 0:
            errors["estimated_removed_time_s"] = "must_be_nonnegative"
        return errors

    @staticmethod
    def _is_interval(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
                for item in value
            )
            and value[0] < value[1]
        )

    @classmethod
    def _same_measurement(cls, left: Any, right: Any, tolerance: float = 0.002) -> bool:
        if isinstance(left, (int, float)) and not isinstance(left, bool):
            return (
                isinstance(right, (int, float))
                and not isinstance(right, bool)
                and math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
            )
        if isinstance(left, list):
            return isinstance(right, list) and len(left) == len(right) and all(
                cls._same_measurement(a, b, tolerance) for a, b in zip(left, right)
            )
        return left == right

    @classmethod
    def _atom_supports_claim_value(cls, field_name: str, value: Any, atom: EvidenceAtom) -> bool:
        observation = atom.observation
        direct_key = {
            "source_id": "source_id",
            "comparison_scope_s": "comparison_scope_s",
            "retained_source_interval_s": "retained_source_interval_s",
            "omitted_source_intervals_s": "omitted_source_intervals_s",
            "estimated_removed_time_s": "estimated_removed_time_s",
        }.get(field_name)
        if direct_key is not None:
            return direct_key in observation and cls._same_measurement(value, observation[direct_key])
        if field_name == "affected_source_interval_s":
            return cls._same_measurement(value, observation.get("affected_source_interval_s"))
        if field_name == "affected_suspect_interval_s":
            return cls._same_measurement(value, observation.get("affected_suspect_interval_s"))
        if field_name == "claim_family":
            if value == "paired_media_difference":
                return observation.get("material_difference_detected") is True
            return value == "selective_temporal_omission" and observation.get("source_frames_absent") is True
        if field_name == "manipulation_type":
            generic_signal = {
                "temporal_edit": "temporal_edit_detected",
                "audio_edit": "audio_edit_detected",
                "spatial_edit": "spatial_edit_detected",
            }.get(value)
            if generic_signal:
                return observation.get(generic_signal) is True
            return (
                value == "frame_dropping"
                and observation.get("source_frames_absent") is True
                and observation.get("temporal_order_preserved") is True
            )
        if field_name == "direction":
            return (
                value == "source_to_edited"
                and observation.get("source_role") == "source"
                and observation.get("suspect_role") == "edited"
            )
        if field_name in observation:
            return cls._same_measurement(value, observation[field_name])
        return False

    def _revise_claim_fields(
        self, arguments: dict[str, Any], tool_call_id: str
    ) -> dict[str, Any]:
        """Validate a policy-proposed local value revision against visible atoms."""

        claim_id = str(arguments.get("claim_id") or "")
        updates = arguments.get("field_updates")
        if claim_id not in self.claims or not isinstance(updates, list) or not updates:
            raise ValueError(
                "revise_claim_fields requires an existing claim and field_updates"
            )
        fields = [
            str(item.get("field") or "") if isinstance(item, dict) else ""
            for item in updates
        ]
        if any(not item for item in fields) or len(fields) != len(set(fields)):
            raise ValueError("revision fields must be nonempty and unique")
        unknown = [field_name for field_name in fields if field_name not in self.required_fields]
        if unknown:
            raise ValueError(f"unknown revision fields: {unknown}")
        not_open = [
            field_name
            for field_name in fields
            if self.claims[claim_id].fields[field_name].decision_status != "open"
        ]
        if not_open:
            raise ValueError(f"revision fields must be reopened first: {not_open}")

        candidates = self._claim_revision_candidates().get(claim_id, {})
        proposed = {
            field_name: copy.deepcopy(updates[index].get("value"))
            for index, field_name in enumerate(fields)
        }
        unsupported = [
            field_name
            for field_name, value in proposed.items()
            if field_name not in candidates
            or not self._same_measurement(value, candidates[field_name]["value"])
        ]
        if unsupported:
            raise ValueError(
                "revision values require one unambiguous active visible support: "
                f"{unsupported}"
            )
        merged_values = {
            field_name: copy.deepcopy(field_state.value)
            for field_name, field_state in self.claims[claim_id].fields.items()
        }
        merged_values.update(copy.deepcopy(proposed))
        value_errors = self._validate_claim_values(merged_values)
        if value_errors:
            raise ValueError(f"revised claim value validation failed: {value_errors}")

        revisions: list[dict[str, Any]] = []
        for field_name, value in proposed.items():
            field_state = self.claims[claim_id].fields[field_name]
            old_value = copy.deepcopy(field_state.value)
            if self._same_measurement(old_value, value):
                raise ValueError(f"revision does not change {field_name}")
            field_state.value = copy.deepcopy(value)
            field_state.support_ids = []
            field_state.attack_ids = []
            field_state.evidence_status = "unsupported"
            binding = self.bindings.get(claim_id, {}).get(field_name)
            if binding is not None:
                binding.support_ids = []
                binding.attack_ids = []
                binding.validation_status = "invalidated_by_claim_value_revision"
            revisions.append(
                {
                    "field": field_name,
                    "old_value": old_value,
                    "new_value": copy.deepcopy(value),
                    "candidate_evidence_ids": copy.deepcopy(
                        candidates[field_name]["evidence_ids"]
                    ),
                }
            )

        preserve_adjudication = (
            claim_id in self.evidence_adjudications
            and set(fields).issubset(
                set(self.evidence_adjudications[claim_id].target_fields)
            )
        )
        if not preserve_adjudication:
            self.evidence_assessments.clear()
            self.evidence_adjudications.clear()
        self.goals["G-REPORT"].status = "open"
        self.goals["G-REPORT"].evidence_ids = []

        gap_support_ids: list[str] = []
        incompatible: list[str] = []
        for current_claim_id, claim in self.claims.items():
            for field_name, field_state in claim.fields.items():
                if field_state.evidence_status not in {"unsupported", "conflicted"}:
                    continue
                acceptable = set(
                    self.claim_schema["fields"][field_name].get(
                        "acceptable_evidence_kinds", []
                    )
                )
                supporting = [
                    evidence_id
                    for evidence_id, atom in self.evidence_store.items()
                    if atom.active
                    and field_name in atom.supports_fields
                    and atom.kind in acceptable
                    and not self._challenge_blocks_field_support(
                        evidence_id, field_name
                    )
                    and self._atom_supports_claim_value(
                        field_name, field_state.value, atom
                    )
                ]
                if supporting:
                    gap_support_ids.extend(supporting)
                else:
                    incompatible.append(f"{current_claim_id}:{field_name}")
        self.goals["G-DIFFERENCE"].status = (
            "satisfied" if not incompatible else "open"
        )
        self.goals["G-DIFFERENCE"].evidence_ids = sorted(set(gap_support_ids))
        result = {
            "status": "accepted",
            "authority": "schema_and_visible_evidence_checker",
            "revisions": revisions,
            "remaining_incompatible_claim_fields": incompatible,
            "created_evidence_ids": [],
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="claim_value_revision_validation",
            action="revise_claim_fields",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def _set_decision(self, arguments: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
        claim_id = arguments.get("claim_id")
        fields = arguments.get("fields")
        status = arguments.get("decision_status")
        if claim_id not in self.claims or not isinstance(fields, list):
            raise ValueError("set_field_decision requires an existing claim and fields")
        if status not in {"open", "committed", "abstained"}:
            raise ValueError("invalid decision_status")
        unknown = [name for name in fields if name not in self.required_fields]
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")
        if status == "committed":
            unsupported = [
                name for name in fields if self.claims[claim_id].fields[name].evidence_status != "supported"
            ]
            if unsupported:
                result = {
                    "status": "rejected",
                    "authority": "schema_checker",
                    "reason": "cannot_commit_without_visible_support",
                    "unsupported_fields": unsupported,
                    "harness_state": self.render_state(),
                }
                self._record(
                    actor="harness",
                    action_type="policy_state_edit_validation",
                    action="set_field_decision",
                    tool_call_id=tool_call_id,
                    result=result,
                )
                return result
        for name in fields:
            self.claims[claim_id].fields[name].decision_status = status
        if all(
            item.evidence_status == "supported" and item.decision_status == "committed"
            for item in self.claims[claim_id].fields.values()
        ):
            self.goals["G-REPORT"].status = "satisfied"
            self.goals["G-REPORT"].evidence_ids = sorted(
                {evidence_id for item in self.claims[claim_id].fields.values() for evidence_id in item.support_ids}
            )
        else:
            self.goals["G-REPORT"].status = "open"
        result = {
            "status": "accepted",
            "authority": "schema_checker",
            "updated_fields": fields,
            "decision_status": status,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="policy_state_edit_validation",
            action="set_field_decision",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result

    def inject_evidence_dropout(self, evidence_id: str) -> dict[str, Any]:
        atom = self.evidence_store.get(evidence_id)
        if atom is None or not atom.active:
            raise ValueError(f"cannot drop unavailable evidence: {evidence_id}")
        atom.active = False
        self.dropped_evidence_ids.append(evidence_id)
        # Candidate assessments and the final partition are scoped to one exact
        # competing-evidence set. Dropping any member invalidates both ledgers;
        # retaining stale entries would make a later replacement set impossible
        # to partition exactly once.
        self.evidence_assessments.clear()
        self.evidence_adjudications.clear()
        affected_fields: list[str] = []
        for claim_id, claim_bindings in self.bindings.items():
            for field_name, binding in claim_bindings.items():
                if evidence_id not in binding.support_ids:
                    continue
                binding.support_ids = [item for item in binding.support_ids if item != evidence_id]
                field_state = self.claims[claim_id].fields[field_name]
                field_state.support_ids = list(binding.support_ids)
                field_state.evidence_status = "supported" if binding.support_ids else "unsupported"
                if not binding.support_ids:
                    binding.validation_status = "invalidated_by_environment_mutation"
                affected_fields.append(field_name)
        self.goals["G-DIFFERENCE"].status = "open"
        self.goals["G-REPORT"].status = "open"
        self.terminal = False
        self.stop_decision = None
        dropout_hook = getattr(self.environment, "on_evidence_dropout", None)
        if callable(dropout_hook):
            dropout_hook(evidence_id, list(affected_fields))
        result = {
            "status": "executed",
            "authority": "mutation_compiler",
            "operator": "evidence_dropout",
            "removed_evidence_ids": [evidence_id],
            "affected_fields": affected_fields,
            "decision_status_edited_by_compiler": False,
            "media_changed": False,
            "claim_values_changed": False,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="mutation_compiler",
            action_type="environment_mutation",
            action="inject_evidence_dropout",
            tool_call_id=None,
            result=result,
        )
        return result

    def inject_binding_invalidation(self, evidence_id: str) -> dict[str, Any]:
        """Invalidate claim links while leaving the underlying evidence active.

        This models a recoverable provenance-index or state-serialization
        failure.  It is intentionally different from evidence dropout: the
        policy should reuse the still-visible atom after reopening the damaged
        fields instead of paying for another external measurement.
        """
        atom = self.evidence_store.get(evidence_id)
        if atom is None or not atom.active:
            raise ValueError(f"cannot invalidate bindings for unavailable evidence: {evidence_id}")
        self.evidence_adjudications.clear()
        affected_fields: list[str] = []
        for claim_id, claim_bindings in self.bindings.items():
            for field_name, binding in claim_bindings.items():
                if evidence_id not in binding.support_ids:
                    continue
                binding.support_ids = [item for item in binding.support_ids if item != evidence_id]
                binding.validation_status = "invalidated_by_binding_mutation"
                field_state = self.claims[claim_id].fields[field_name]
                field_state.support_ids = list(binding.support_ids)
                field_state.evidence_status = "supported" if binding.support_ids else "unsupported"
                affected_fields.append(field_name)
        if not affected_fields:
            raise ValueError(f"evidence is not bound to any claim field: {evidence_id}")
        self.goals["G-REPORT"].status = "open"
        self.terminal = False
        self.stop_decision = None
        result = {
            "status": "executed",
            "authority": "mutation_compiler",
            "operator": "binding_invalidation",
            "invalidated_binding_evidence_ids": [evidence_id],
            "affected_fields": affected_fields,
            "decision_status_edited_by_compiler": False,
            "media_changed": False,
            "claim_values_changed": False,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="mutation_compiler",
            action_type="environment_mutation",
            action="inject_binding_invalidation",
            tool_call_id=None,
            result=result,
        )
        return result

    def inject_decision_reopen(self, fields: list[str] | None = None) -> dict[str, Any]:
        """Reopen an already supported decision slice without changing evidence."""
        if len(self.claims) != 1:
            raise ValueError("decision_reopen requires exactly one existing claim")
        self.evidence_adjudications.clear()
        claim_id = next(iter(self.claims))
        selected = list(fields or self.target_fields)
        unknown = [name for name in selected if name not in self.required_fields]
        if unknown:
            raise ValueError(f"unknown fields for decision reopen: {unknown}")
        invalid = [
            name
            for name in selected
            if self.claims[claim_id].fields[name].evidence_status != "supported"
            or self.claims[claim_id].fields[name].decision_status != "committed"
        ]
        if invalid:
            raise ValueError(f"decision_reopen requires supported committed fields: {invalid}")
        for name in selected:
            self.claims[claim_id].fields[name].decision_status = "open"
        self.goals["G-REPORT"].status = "open"
        self.terminal = False
        self.stop_decision = None
        result = {
            "status": "executed",
            "authority": "mutation_compiler",
            "operator": "decision_reopen",
            "reopened_decision_count": len(selected),
            "affected_fields": selected,
            "decision_status_edited_by_compiler": True,
            "media_changed": False,
            "claim_values_changed": False,
            "harness_state": self.render_state(),
        }
        self._record(
            actor="mutation_compiler",
            action_type="environment_mutation",
            action="inject_decision_reopen",
            tool_call_id=None,
            result=result,
        )
        return result

    def _open_obligations(self) -> list[str]:
        open_items = [goal.goal_id for goal in self.goals.values() if goal.priority == "high" and goal.status != "satisfied"]
        for claim in self.claims.values():
            for name, item in claim.fields.items():
                if item.evidence_status != "supported" or item.decision_status not in {"committed", "abstained"}:
                    open_items.append(f"OBL-{claim.claim_id}-{name}")
        return open_items

    def _stop(self, tool_call_id: str) -> dict[str, Any]:
        open_items = self._open_obligations()
        allowed = not open_items and bool(self.claims)
        self.terminal = allowed
        self.stop_decision = {
            "allowed": allowed,
            "reason_code": "all_high_priority_obligations_closed" if allowed else "open_high_priority_obligations",
            "open_high_priority_obligations": open_items,
        }
        result = {
            "status": "accepted" if allowed else "rejected",
            "authority": "terminal_checker",
            "stop_decision": copy.deepcopy(self.stop_decision),
            "harness_state": self.render_state(),
        }
        self._record(
            actor="harness",
            action_type="terminal_validation",
            action="stop",
            tool_call_id=tool_call_id,
            result=result,
        )
        return result


def derive_state_action_affordances(
    state: dict[str, Any], claim_schema: dict[str, Any]
) -> dict[str, Any]:
    """Externalize executable state-action arguments derived from visible state only."""
    atoms = {
        str(evidence_id): EvidenceAtom(**copy.deepcopy(atom))
        for evidence_id, atom in (state.get("evidence_store") or {}).items()
        if atom.get("active", True)
    }
    schema_fields = claim_schema.get("fields") or {}
    challenge_barred_pairs = {
        (str(evidence_id), str(field_name))
        for challenge in state.get("evidence_challenges") or []
        if challenge.get("status") == "resolved"
        for evidence_id in challenge.get("evidence_ids") or []
        for field_name in challenge.get("target_fields") or []
    }
    binding_candidates: dict[str, dict[str, list[str]]] = {}
    executable_binding_actions: list[dict[str, Any]] = []
    repair_restoration_candidates: list[dict[str, Any]] = []
    commit_fields: dict[str, list[str]] = {}
    reopen_fields: dict[str, list[str]] = {}
    eligible_tools = {str(name) for name in state.get("eligible_tools") or []}
    evidence_dropped = bool(state.get("dropped_evidence_ids"))
    adjudications = state.get("evidence_adjudications") or {}
    for raw_claim in state.get("candidate_claims") or []:
        claim_id = str(raw_claim["claim_id"])
        raw_fields = raw_claim.get("fields") or {}
        adjudication = adjudications.get(claim_id) or {}
        adjudicated_fields = set(adjudication.get("target_fields") or [])
        adjudicated_ids = set(adjudication.get("selected_evidence_ids") or [])
        claim_candidates: dict[str, list[str]] = {}
        for field_name, field_state in raw_fields.items():
            evidence_status = str(field_state.get("evidence_status", "unsupported"))
            decision_status = str(field_state.get("decision_status", "open"))
            if evidence_status == "supported" and decision_status == "open":
                commit_fields.setdefault(claim_id, []).append(str(field_name))
            if evidence_status in {"unsupported", "conflicted"} and decision_status == "committed":
                reopen_fields.setdefault(claim_id, []).append(str(field_name))
            if evidence_status == "supported" or decision_status == "committed":
                continue
            field_contract = schema_fields.get(field_name) or {}
            acceptable = set(field_contract.get("acceptable_evidence_kinds") or [])
            compatible = sorted(
                evidence_id
                for evidence_id, atom in atoms.items()
                if field_name in atom.supports_fields
                and atom.kind in acceptable
                and (evidence_id, str(field_name)) not in challenge_barred_pairs
                and (
                    str(field_name) not in adjudicated_fields
                    or evidence_id in adjudicated_ids
                )
                and VideoForensicHarness._atom_supports_claim_value(
                    str(field_name), field_state.get("value"), atom
                )
            )
            if compatible:
                claim_candidates[str(field_name)] = compatible
        if claim_candidates and "bind_evidence" in eligible_tools:
            binding_candidates[claim_id] = claim_candidates
            ordered_fields = [
                str(field_name)
                for field_name in raw_fields
                if str(field_name) in claim_candidates
            ]
            evidence_ids = sorted(
                {
                    evidence_id
                    for candidates in claim_candidates.values()
                    for evidence_id in candidates
                }
            )
            coverage = {
                evidence_id: sum(
                    evidence_id in claim_candidates[field_name]
                    for field_name in ordered_fields
                )
                for evidence_id in evidence_ids
            }
            minimal_covers: list[tuple[str, ...]] = []
            for size in range(1, len(evidence_ids) + 1):
                for subset in itertools.combinations(evidence_ids, size):
                    if all(
                        set(claim_candidates[field_name]).intersection(subset)
                        for field_name in ordered_fields
                    ):
                        minimal_covers.append(subset)
                if minimal_covers:
                    break
            for subset in minimal_covers[:4]:
                executable_binding_actions.append(
                    {
                        "tool_name": "bind_evidence",
                        "arguments": {
                            "claim_id": claim_id,
                            "binding_items": [
                                {
                                    "field": field_name,
                                    "support_ids": [
                                        min(
                                            set(claim_candidates[field_name]).intersection(subset),
                                            key=lambda evidence_id: (
                                                -coverage[evidence_id],
                                                evidence_id,
                                            ),
                                        )
                                    ],
                                }
                                for field_name in ordered_fields
                            ],
                        },
                        "coverage": {
                            "fields": ordered_fields,
                            "field_count": len(ordered_fields),
                            "distinct_evidence_count": len(subset),
                        },
                    }
                )
        method_state = raw_fields.get("measurement_method") or {}
        prior_method = method_state.get("value")
        if (
            evidence_dropped
            and isinstance(prior_method, str)
            and prior_method in eligible_tools
            and method_state.get("evidence_status") in {"unsupported", "conflicted"}
            and method_state.get("decision_status") == "open"
        ):
            repair_restoration_candidates.append(
                {
                    "claim_id": claim_id,
                    "tool_name": prior_method,
                    "evidence_gap_fields": [
                        str(field_name)
                        for field_name, field_state in raw_fields.items()
                        if field_state.get("evidence_status") in {"unsupported", "conflicted"}
                        and field_state.get("decision_status") == "open"
                    ],
                    "basis": (
                        "The current open measurement_method value names this eligible tool "
                        "after an evidence dropout. Re-measure it first, then validate the "
                        "new observation before rebinding."
                    ),
                }
            )
    transitions = [
        {"claim_id": claim_id, "decision_status": "open", "fields": sorted(fields)}
        for claim_id, fields in sorted(reopen_fields.items())
    ] + [
        {"claim_id": claim_id, "decision_status": "committed", "fields": sorted(fields)}
        for claim_id, fields in sorted(commit_fields.items())
    ]
    executable_adjudication_actions: list[dict[str, Any]] = []
    executable_evidence_assessment_actions: list[dict[str, Any]] = []
    adjudication_state = state.get("evidence_adjudication") or {}
    requirement = adjudication_state.get("requirement") or {}
    progressive_visible = "assess_evidence_candidate" in eligible_tools
    if requirement and (
        "adjudicate_evidence" in eligible_tools or progressive_visible
    ):
        claim_id = str(requirement["claim_id"])
        target_fields = [str(item) for item in requirement["target_fields"]]
        competing_ids = [
            str(item) for item in requirement["competing_evidence_ids"]
        ]
        complete_ids = [
            evidence_id
            for evidence_id in competing_ids
            if evidence_id in atoms
            and all(
                field_name in atoms[evidence_id].supports_fields
                and field_name in (atoms[evidence_id].observation or {})
                for field_name in target_fields
            )
            and (atoms[evidence_id].provenance or {}).get(
                "role_assignment_valid_for_current_contract", True
            ) is not False
        ]

        def visible_confidence(evidence_id: str) -> float:
            value = (atoms[evidence_id].observation or {}).get("confidence")
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else -1.0

        selected_covers: list[tuple[str, ...]] = []
        if complete_ids and all(
            (atoms[evidence_id].observation or {}).get("preferred_video")
            in {"a", "b", "c"}
            for evidence_id in complete_ids
        ):
            method_representatives: dict[str, str] = {}
            for evidence_id in complete_ids:
                observation = atoms[evidence_id].observation or {}
                method = str(observation.get("measurement_method") or (atoms[evidence_id].provenance or {}).get("method") or evidence_id)
                incumbent = method_representatives.get(method)
                if incumbent is None or (visible_confidence(evidence_id), evidence_id) > (
                    visible_confidence(incumbent), incumbent
                ):
                    method_representatives[method] = evidence_id
            votes = Counter(
                str((atoms[evidence_id].observation or {}).get("preferred_video"))
                for evidence_id in method_representatives.values()
            )
            directional = [value for value in ("a", "b") if votes[value] >= 2]
            if len(directional) == 1 and votes["b" if directional[0] == "a" else "a"] == 0:
                selected_value = directional[0]
            elif votes["c"]:
                selected_value = "c"
            elif votes["a"] != votes["b"]:
                selected_value = "a" if votes["a"] > votes["b"] else "b"
            else:
                selected_value = None
            matching = [
                evidence_id
                for evidence_id in complete_ids
                if (atoms[evidence_id].observation or {}).get("preferred_video") == selected_value
            ]
            stable = [
                evidence_id
                for evidence_id in matching
                if selected_value == "c"
                or (atoms[evidence_id].observation or {}).get("order_consistent") is not False
            ]
            pool = stable or matching
            if pool:
                selected_covers = [
                    (max(pool, key=lambda evidence_id: (visible_confidence(evidence_id), evidence_id)),)
                ]
        elif complete_ids:
            selected_covers = [
                (max(complete_ids, key=lambda evidence_id: (visible_confidence(evidence_id), evidence_id)),)
            ]
        for subset in selected_covers[:4]:
            selected_confidences = [
                (atoms[evidence_id].observation or {}).get("confidence")
                for evidence_id in subset
                if evidence_id in atoms
            ]
            selected_numeric_confidences = [
                float(value)
                for value in selected_confidences
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            rejected_items: list[dict[str, Any]] = []
            for evidence_id in competing_ids:
                if evidence_id in subset:
                    continue
                atom = atoms[evidence_id]
                observation = atom.observation or {}
                selected_observations = [
                    atoms[selected_id].observation or {}
                    for selected_id in subset
                    if selected_id in atoms
                ]
                selected_preferred = {
                    item.get("preferred_video")
                    for item in selected_observations
                    if item.get("preferred_video") is not None
                }
                if (atom.provenance or {}).get(
                    "role_assignment_valid_for_current_contract"
                ) is False:
                    defects = ["role_assignment_mismatch"]
                elif observation.get("preferred_video") is not None and any(
                    value != observation.get("preferred_video")
                    for value in selected_preferred
                ):
                    defects = ["cross_method_direction_conflict"]
                elif observation.get("preferred_video") in selected_preferred:
                    defects = ["redundant_same_conclusion"]
                elif (
                    observation.get("preferred_video") != "c"
                    and observation.get("order_consistent") is False
                ):
                    defects = ["order_inconsistent"]
                else:
                    defects = []
                confidence = observation.get("confidence")
                if (
                    not defects
                    and
                    isinstance(confidence, (int, float))
                    and not isinstance(confidence, bool)
                    and selected_numeric_confidences
                    and float(confidence) < max(selected_numeric_confidences)
                ):
                    defects = ["lower_confidence"]
                if not defects:
                    defects.append("redundant_same_conclusion")
                rejected_items.append(
                    {
                        "evidence_id": evidence_id,
                        "defect_codes": defects,
                    }
                )
            if rejected_items:
                adjudication_action = {
                    "tool_name": "adjudicate_evidence",
                    "arguments": {
                        "claim_id": claim_id,
                        "target_fields": target_fields,
                        "selected_evidence_ids": list(subset),
                        "rejected_evidence": rejected_items,
                    },
                    "coverage": {
                        "competing_evidence_count": len(competing_ids),
                        "selected_evidence_count": len(subset),
                        "rejected_evidence_count": len(rejected_items),
                    },
                }
                executable_adjudication_actions.append(adjudication_action)
                if progressive_visible and len(selected_covers) == 1:
                    existing = (
                        (state.get("evidence_assessments") or {}).get(claim_id)
                        or {}
                    )
                    rejected_by_id = {
                        str(item["evidence_id"]): list(item["defect_codes"])
                        for item in rejected_items
                    }
                    current_candidate_id = str(
                        (state.get("evidence_assessment") or {}).get(
                            "current_candidate_id"
                        )
                        or ""
                    )
                    for evidence_id in competing_ids:
                        if current_candidate_id and evidence_id != current_candidate_id:
                            continue
                        prior = existing.get(evidence_id) or {}
                        if (
                            prior.get("target_fields") == target_fields
                            and prior.get("considered_evidence_ids") == competing_ids
                        ):
                            continue
                        is_selected = evidence_id in subset
                        executable_evidence_assessment_actions.append(
                            {
                                "tool_name": "assess_evidence_candidate",
                                "arguments": {
                                    "claim_id": claim_id,
                                    "target_fields": target_fields,
                                    "evidence_id": evidence_id,
                                    "disposition": (
                                        "retain" if is_selected else "reject"
                                    ),
                                    "defect_codes": (
                                        []
                                        if is_selected
                                        else rejected_by_id[evidence_id]
                                    ),
                                },
                                "coverage": {
                                    "competing_evidence_count": len(competing_ids),
                                    "assesses_exactly_one_candidate": True,
                                },
                            }
                        )
    return {
        "derivation": (
            "Computed only from active student-visible EvidenceAtoms, one vote per "
            "measurement method, cross-method agreement, claim schema compatibility, "
            "and current field statuses; claim values do not choose adjudication winners."
        ),
        "binding_candidates": binding_candidates,
        "executable_binding_actions": executable_binding_actions,
        "executable_evidence_assessment_actions": (
            executable_evidence_assessment_actions
        ),
        "executable_adjudication_actions": executable_adjudication_actions,
        "repair_restoration_candidates": repair_restoration_candidates,
        "meaningful_decision_transitions": transitions,
    }


def _assistant_tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    reasoning: str,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        # OpenAI-compatible export of the Harmony analysis message used by
        # Harness-1.  The raw turn_history below also stores this separately.
        "content": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))},
            }
        ],
    }


def _tool_message(call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
    }


def _run_call(
    harness: VideoForensicHarness,
    messages: list[dict[str, Any]],
    turn_history: list[dict[str, Any]],
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    reasoning: str,
) -> dict[str, Any]:
    messages.append(_assistant_tool_call(call_id, name, arguments, reasoning))
    harness.record_policy_call(name, arguments, call_id, reasoning)
    result = harness.execute_tool(name, arguments, call_id)
    messages.append(_tool_message(call_id, name, result))
    turn_history.append(
        {
            "turn_idx": len(turn_history) + 1,
            "reasoning": reasoning,
            "tool_name": name,
            "params": copy.deepcopy(arguments),
            "observation": copy.deepcopy(result),
        }
    )
    return result


def _expand_turn_datums(
    messages: list[dict[str, Any]],
    *,
    case_id: str,
    trajectory_id: str,
    trajectory_type: str,
) -> list[dict[str, Any]]:
    """Mirror Harness-1: one supervised datum per assistant action turn."""

    datums: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        tool_name = message["tool_calls"][0]["function"]["name"]
        datums.append(
            {
                "messages": copy.deepcopy(messages[: message_index + 1]),
                "metadata": {
                    "case_id": case_id,
                    "trajectory_id": trajectory_id,
                    "trajectory_type": trajectory_type,
                    "turn_index": len(datums) + 1,
                    "target_tool": tool_name,
                    "target_message_index": message_index,
                    "target_includes_observation": False,
                    "generator": "claimunittest_harness1_style_v2",
                    "loss_roles": ["assistant"],
                    "loss_scope": "last_assistant_action_only",
                },
            }
        )
    return datums


def _claim_values_from_observation(manifest: dict[str, Any], align_result: dict[str, Any]) -> dict[str, Any]:
    observation = align_result["observation"]["result"]
    alignment = observation["alignment"]
    gap = observation["source_time_contraction"]
    return {
        "source_id": alignment["source_id"],
        "claim_family": "selective_temporal_omission",
        "manipulation_type": "frame_dropping",
        "direction": "source_to_edited",
        "comparison_scope_s": manifest["comparison_scope_s"],
        "retained_source_interval_s": alignment["retained_source_interval_s"],
        "omitted_source_intervals_s": gap["omitted_source_intervals_s"],
        "estimated_removed_time_s": gap["estimated_removed_time_s"],
    }


def _final_claim(harness: VideoForensicHarness, claim_id: str) -> dict[str, Any]:
    claim = harness.claims[claim_id]
    return {
        "claim_id": claim_id,
        **{name: copy.deepcopy(item.value) for name, item in claim.fields.items()},
        "field_support": {name: list(item.support_ids) for name, item in claim.fields.items()},
    }


def build_interactive_bundle(
    *,
    manifest: dict[str, Any],
    source_frames: Any,
    suspect_frames: Any,
    source_path: Path,
    suspect_path: Path,
    metadata_fn: Callable[[Any, Path], dict[str, Any]],
    align_fn: Callable[[Any, Any], dict[str, Any]],
    omission_fn: Callable[[dict[str, Any], float], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Execute one clean acquisition and one evidence-dropout repair episode."""

    contract = TaskContract(
        contract_id="TASK-VIDEOSHAM-1004",
        episode_mode="paired_compare",
        task="Determine whether and how the suspect clip omits source content.",
        visible_assets={
            "source": manifest["source_media"]["path"],
            "suspect": manifest["edited_media"]["path"],
            "comparison_scope_s": manifest["comparison_scope_s"],
        },
        claim_schema_id=CLAIM_SCHEMA["schema_id"],
    )
    environment = RealVideoToolEnvironment(
        manifest=manifest,
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        source_path=source_path,
        suspect_path=suspect_path,
        metadata_fn=metadata_fn,
        align_fn=align_fn,
        omission_fn=omission_fn,
    )
    harness = VideoForensicHarness(contract, environment)
    system = {
        "role": "system",
        "content": (
            "You are a policy inside a state-externalizing video-forensic harness. Choose executable tools; "
            "never invent observations or EvidenceAtoms. Propose claim values and evidence bindings explicitly. "
            "Evidence coverage and decision status are separate. Stop only when all visible high-priority "
            "obligations are closed."
        ),
    }
    clean_messages: list[dict[str, Any]] = [
        system,
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_contract": asdict(contract),
                    "claim_schema": CLAIM_SCHEMA,
                    "tool_registry": TOOL_REGISTRY,
                    "harness_state": harness.render_state(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    clean_turn_history: list[dict[str, Any]] = []

    inspect_result = _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_inspect_001",
        "inspect_media",
        {"media_refs": ["source", "suspect"]},
        (
            "I have a visible source-suspect pair and a neutral comparison window, but no decoded media "
            "facts yet. Before proposing any manipulation claim, I need to establish the media context and "
            "scope from executable inputs. I will inspect both files first."
        ),
    )
    align_args = {
        "source_ref": "source",
        "suspect_ref": "suspect",
        "comparison_scope_s": manifest["comparison_scope_s"],
    }
    align_result = _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_align_001",
        "align_videos_dense",
        align_args,
        (
            f"The media inspection found a {inspect_result['observation']['result']['source_media']['duration_s']:.3f}s "
            f"source and an {inspect_result['observation']['result']['suspect_media']['duration_s']:.3f}s suspect, "
            f"with comparison scope {manifest['comparison_scope_s']}. Duration mismatch suggests temporal contraction, "
            "but it does not identify which source frames were retained or omitted. I need dense monotonic alignment "
            "to localize the difference and obtain replayable evidence."
        ),
    )
    values = _claim_values_from_observation(manifest, align_result)
    _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_create_001",
        "create_candidate_claim",
        {"claim_id": "CL-1004-001", "field_values": values},
        (
            f"The alignment matched {align_result['observation']['result']['alignment']['matched_frames']} suspect "
            f"frames in strictly monotonic order with p05 similarity "
            f"{align_result['observation']['result']['alignment']['p05_similarity']:.4f}. It retains source interval "
            f"{values['retained_source_interval_s']} and measures boundary omissions {values['omitted_source_intervals_s']} "
            f"totaling {values['estimated_removed_time_s']:.3f}s. The evidence supports a selective temporal omission "
            "candidate, so I will now propose the atomic claim values; the harness should still leave every field "
            "open until evidence bindings are checked."
        ),
    )
    clean_align_ids = align_result["created_evidence_ids"]
    scope_id = next(item for item in inspect_result["created_evidence_ids"] if item.startswith("EA-SCOPE-"))
    alignment_id = next(item for item in clean_align_ids if item.startswith("EA-ALIGN-"))
    gap_id = next(item for item in clean_align_ids if item.startswith("EA-GAP-"))
    clean_bindings = {
        "source_id": [alignment_id],
        "claim_family": [gap_id],
        "manipulation_type": [gap_id],
        "direction": [gap_id],
        "comparison_scope_s": [scope_id],
        "retained_source_interval_s": [alignment_id],
        "omitted_source_intervals_s": [gap_id],
        "estimated_removed_time_s": [gap_id],
    }
    _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_bind_001",
        "bind_evidence",
        {"claim_id": "CL-1004-001", "bindings": clean_bindings},
        (
            f"The candidate claim exists, but its field obligations are still unsupported. I will bind source identity "
            f"and retained interval to {alignment_id}, the neutral scope to {scope_id}, and only the omission-dependent "
            f"fields to {gap_id}. This keeps the provenance granular enough that losing the gap evidence will not "
            "invalidate independently supported source and scope fields."
        ),
    )
    _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_commit_001",
        "set_field_decision",
        {"claim_id": "CL-1004-001", "fields": REQUIRED_FIELDS, "decision_status": "committed"},
        (
            "The contract checker accepted all eight bindings: every required field is now supported and none is "
            "conflicted. The remaining gap is a policy decision rather than more evidence acquisition, so I will "
            "commit the supported fields."
        ),
    )
    _run_call(
        harness,
        clean_messages,
        clean_turn_history,
        "call_clean_stop_001",
        "stop",
        {},
        (
            "All case goals are satisfied, every required field is supported and committed, and no high-priority "
            "obligation remains open. Additional tool calls would not close any evidence gap, so I will request "
            "termination."
        ),
    )
    clean_messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {"action": "final_answer", "claim": _final_claim(harness, "CL-1004-001")},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    clean_final_state = harness.render_state()
    clean_events = copy.deepcopy(harness.events)

    repair_harness = harness.clone_state()
    mutation = repair_harness.inject_evidence_dropout(gap_id)
    mutated_state = copy.deepcopy(mutation["harness_state"])
    invariant_before = {
        name: copy.deepcopy(clean_final_state["candidate_claims"][0]["fields"][name]) for name in INVARIANT_FIELDS
    }
    target_after_dropout = {
        name: copy.deepcopy(mutated_state["candidate_claims"][0]["fields"][name]) for name in TARGET_FIELDS
    }
    visible_mutation = {
        "operator": mutation["operator"],
        "removed_evidence_ids": mutation["removed_evidence_ids"],
        "media_changed": mutation["media_changed"],
        "claim_values_changed": mutation["claim_values_changed"],
    }
    repair_messages: list[dict[str, Any]] = [
        system,
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Repair the restored state after an evidence-store mutation. Preserve independently supported fields.",
                    "shared_clean_snapshot_id": "SNAPSHOT-1004-CLEAN-FINAL",
                    "mutation": visible_mutation,
                    "claim_schema": CLAIM_SCHEMA,
                    "tool_registry": TOOL_REGISTRY,
                    "harness_state": mutated_state,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    repair_turn_history: list[dict[str, Any]] = []
    _run_call(
        repair_harness,
        repair_messages,
        repair_turn_history,
        "call_repair_reopen_001",
        "set_field_decision",
        {"claim_id": "CL-1004-001", "fields": TARGET_FIELDS, "decision_status": "open"},
        (
            f"The restored state reports that {gap_id} is no longer available. The five fields that depended on it "
            "are now unsupported, while source_id, comparison_scope_s, and retained_source_interval_s still retain "
            "their independent support. Their old committed decisions are therefore stale. I will reopen exactly "
            "the unsupported fields and preserve the three unaffected fields."
        ),
    )
    post_reopen_state = repair_harness.render_state()
    repair_align_result = _run_call(
        repair_harness,
        repair_messages,
        repair_turn_history,
        "call_repair_align_001",
        "align_videos_dense",
        align_args,
        (
            f"The affected fields are open, but the evidence gap remains because {gap_id} cannot be reused. The raw "
            "media and comparison scope are unchanged, so the appropriate recovery is to execute dense alignment "
            "again from the media rather than copy the dropped observation or reset the whole claim."
        ),
    )
    repair_gap_id = next(
        item for item in repair_align_result["created_evidence_ids"] if item.startswith("EA-GAP-")
    )
    _run_call(
        repair_harness,
        repair_messages,
        repair_turn_history,
        "call_repair_bind_001",
        "bind_evidence",
        {"claim_id": "CL-1004-001", "bindings": {name: [repair_gap_id] for name in TARGET_FIELDS}},
        (
            f"Fresh execution {repair_align_result['observation']['execution_id']} reproduced the two boundary "
            f"omissions and created {repair_gap_id}. I only need to bind this new gap atom to the five reopened "
            "omission-dependent fields; rebinding source, scope, or retained interval would erase valid independent "
            "provenance and violate local repair."
        ),
    )
    _run_call(
        repair_harness,
        repair_messages,
        repair_turn_history,
        "call_repair_commit_001",
        "set_field_decision",
        {"claim_id": "CL-1004-001", "fields": TARGET_FIELDS, "decision_status": "committed"},
        (
            f"The contract checker accepted {repair_gap_id} for all five reopened fields, so their evidence status "
            "is supported again. The invariant fields never changed. I will commit only the repaired fields and "
            "leave the independently supported decisions untouched."
        ),
    )
    _run_call(
        repair_harness,
        repair_messages,
        repair_turn_history,
        "call_repair_stop_001",
        "stop",
        {},
        (
            f"The dropped evidence {gap_id} is still absent, the repaired fields now cite fresh evidence "
            f"{repair_gap_id}, and all off-target field values, decisions, and support IDs were preserved. No "
            "high-priority obligation remains open, so the locally repaired episode can terminate."
        ),
    )
    repair_messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {"action": "final_answer", "claim": _final_claim(repair_harness, "CL-1004-001")},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    )
    repair_final_state = repair_harness.render_state()
    invariant_after = {
        name: copy.deepcopy(repair_final_state["candidate_claims"][0]["fields"][name]) for name in INVARIANT_FIELDS
    }

    clean_sft = {
        "messages": clean_messages,
        "metadata": {
            "case_id": manifest["case_id"],
            "trajectory_id": "ITRJ-1004-CLEAN-001",
            "trajectory_type": "real_tool_clean_acquisition",
            "generator": "claimunittest_harness1_style_v2",
            "loss_roles": ["assistant"],
        },
    }
    repair_sft = {
        "messages": repair_messages,
        "metadata": {
            "case_id": manifest["case_id"],
            "trajectory_id": "ITRJ-1004-REPAIR-001",
            "trajectory_type": "evidence_dropout_suffix_replay",
            "shared_clean_snapshot_id": "SNAPSHOT-1004-CLEAN-FINAL",
            "generator": "claimunittest_harness1_style_v2",
            "loss_roles": ["assistant"],
        },
    }
    clean_turn_sft = _expand_turn_datums(
        clean_messages,
        case_id=manifest["case_id"],
        trajectory_id="ITRJ-1004-CLEAN-001",
        trajectory_type="real_tool_clean_acquisition",
    )
    repair_turn_sft = _expand_turn_datums(
        repair_messages,
        case_id=manifest["case_id"],
        trajectory_id="ITRJ-1004-REPAIR-001",
        trajectory_type="evidence_dropout_suffix_replay",
    )
    return {
        "clean_episode": {
            "trajectory_id": "ITRJ-1004-CLEAN-001",
            "task_contract": asdict(contract),
            "claim_schema": CLAIM_SCHEMA,
            "tool_registry": TOOL_REGISTRY,
            "turn_history": clean_turn_history,
            "events": clean_events,
            "messages": clean_messages,
            "final_state": clean_final_state,
            "final_claim": _final_claim(harness, "CL-1004-001"),
        },
        "repair_episode": {
            "trajectory_id": "ITRJ-1004-REPAIR-001",
            "shared_clean_snapshot_id": "SNAPSHOT-1004-CLEAN-FINAL",
            "mutation": {key: value for key, value in mutation.items() if key != "harness_state"},
            "student_visible_mutation": visible_mutation,
            "mutated_state": mutated_state,
            "post_reopen_state": post_reopen_state,
            "target_fields_after_dropout": target_after_dropout,
            "invariant_fields_before": invariant_before,
            "turn_history": repair_turn_history,
            "events": repair_harness.events,
            "messages": repair_messages,
            "final_state": repair_final_state,
            "invariant_fields_after": invariant_after,
            "final_claim": _final_claim(repair_harness, "CL-1004-001"),
        },
        "clean_sft": clean_sft,
        "repair_sft": repair_sft,
        "clean_turn_sft": clean_turn_sft,
        "repair_turn_sft": repair_turn_sft,
        "execution_summary": {
            "external_tool_call_counts": copy.deepcopy(environment.call_counts),
            "clean_alignment_execution_id": align_result["observation"]["execution_id"],
            "repair_alignment_execution_id": repair_align_result["observation"]["execution_id"],
            "dropped_evidence_id": gap_id,
            "repair_evidence_id": repair_gap_id,
        },
    }


def validate_interactive_bundle(bundle: dict[str, Any], forbidden_text: str) -> dict[str, Any]:
    clean = bundle["clean_episode"]
    repair = bundle["repair_episode"]
    summary = bundle["execution_summary"]

    def fields(state: dict[str, Any]) -> dict[str, Any]:
        return state["candidate_claims"][0]["fields"]

    clean_fields = fields(clean["final_state"])
    mutated_fields = fields(repair["mutated_state"])
    reopened_fields = fields(repair["post_reopen_state"])
    final_fields = fields(repair["final_state"])
    student_payload = json.dumps(
        {"clean_sft": bundle["clean_sft"], "repair_sft": bundle["repair_sft"]}, ensure_ascii=False
    )
    repair_user_payload = json.loads(bundle["repair_sft"]["messages"][1]["content"])
    clean_tool_calls = [message for message in clean["messages"] if message["role"] == "assistant" and message.get("tool_calls")]
    clean_tool_results = [message for message in clean["messages"] if message["role"] == "tool"]
    repair_tool_calls = [message for message in repair["messages"] if message["role"] == "assistant" and message.get("tool_calls")]
    reasoning_messages = clean_tool_calls + repair_tool_calls
    turn_datums = bundle["clean_turn_sft"] + bundle["repair_turn_sft"]
    checks = {
        "real_alignment_executed_twice": summary["external_tool_call_counts"]["align_videos_dense"] == 2,
        "alignment_reexecution_is_distinct": summary["clean_alignment_execution_id"] != summary["repair_alignment_execution_id"],
        "assistant_tool_calls_have_tool_results": len(clean_tool_calls) == len(clean_tool_results) == 6,
        "natural_language_reasoning_on_every_action": all(
            isinstance(message.get("content"), str)
            and len(message["content"].split()) >= 18
            and not message["content"].lstrip().startswith("{")
            for message in reasoning_messages
        ),
        "harness1_turn_history_shape": len(clean["turn_history"]) == 6
        and len(repair["turn_history"]) == 5
        and all(
            set(["turn_idx", "reasoning", "tool_name", "params", "observation"]).issubset(turn)
            for turn in clean["turn_history"] + repair["turn_history"]
        ),
        "one_supervised_datum_per_action_turn": len(turn_datums) == 11
        and all(
            datum["messages"][-1]["role"] == "assistant"
            and bool(datum["messages"][-1].get("tool_calls"))
            and datum["metadata"]["target_includes_observation"] is False
            and datum["metadata"]["loss_scope"] == "last_assistant_action_only"
            for datum in turn_datums
        ),
        "evidence_created_by_environment": all(
            event["actor"] == "environment"
            for event in clean["events"]
            if event["action_type"] == "external_tool_execution"
        ),
        "clean_fields_supported_and_committed": all(
            item["evidence_status"] == "supported" and item["decision_status"] == "committed"
            for item in clean_fields.values()
        ),
        "dropout_targets_only_lose_support": all(
            mutated_fields[name]["evidence_status"] == "unsupported" for name in TARGET_FIELDS
        )
        and all(mutated_fields[name] == clean_fields[name] for name in INVARIANT_FIELDS),
        "policy_reopens_only_targets": all(reopened_fields[name]["decision_status"] == "open" for name in TARGET_FIELDS)
        and all(reopened_fields[name] == clean_fields[name] for name in INVARIANT_FIELDS),
        "repair_restores_targets": all(
            final_fields[name]["evidence_status"] == "supported"
            and final_fields[name]["decision_status"] == "committed"
            and summary["repair_evidence_id"] in final_fields[name]["support_ids"]
            for name in TARGET_FIELDS
        ),
        "off_target_invariance": repair["invariant_fields_before"] == repair["invariant_fields_after"],
        "removed_evidence_not_reused": summary["dropped_evidence_id"] not in repair["final_state"]["available_evidence_ids"],
        "terminal_checks_pass": clean["final_state"]["stop_decision"]["allowed"]
        and repair["final_state"]["stop_decision"]["allowed"],
        "no_oracle_leak": forbidden_text not in student_payload
        and "claim_oracle.json" not in student_payload
        and "oracle_only" not in student_payload,
        "no_mutation_target_hint_leak": "affected_fields" not in repair_user_payload["mutation"]
        and "target_fields" not in repair_user_payload["mutation"]
        and "invariant_fields" not in repair_user_payload["mutation"],
    }
    return {"passed": all(checks.values()), "checks": checks, "execution_summary": summary}
