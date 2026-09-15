"""Run the local student policy in a real multi-turn harness loop.

Unlike per-turn teacher-state evaluation, every next state here is produced by
the student's own previous action and the executable environment.  Teacher
actions and verifier labels are not read during generation or execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from control_plane_falsification_arms import ARMS, apply_policy_view_arm, build_evidence_donor_bank
from evaluate_student_context_arm import _load_model, _parse_generated_call
from interactive_harness import EVIDENCE_ADJUDICATION_TOOL, EVIDENCE_ASSESSMENT_TOOL
from llm_teacher_rollout import _normalize_harness_params
from pipeline import ROOT, decode, write_json
from shadow_execute_generated_actions import _jsonl, _payload, _restore_harness
from student_tool_contract import build_student_tools, constrain_tools_with_visible_invariants
from train_student_context_arm import DEFAULT_MODEL
from vcsl_batch import VCSLFeatureEnvironment, VCSL_TOOL_REGISTRY
from vcsl_probe import DEFAULT_FEATURE_ROOT
from viddiff_batch import (
    DEFAULT_ADAPTIVE,
    VIDDIFF_TOOL_REGISTRY,
    VidDiffAdaptiveMeasurementEnvironment,
)
from videosham_batch import (
    EDITED_ROOT,
    MULTIFAMILY_TOOL_REGISTRY,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
)


DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "decision_sufficient_state_v6_executable_affordances"
    / "validation/sft_all.jsonl"
)
DEFAULT_ADAPTER = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "v6_executable_decision_repair16_lr2e5"
    / "adapter"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_canary_three_datasets.json"
)
DEFAULT_EVIDENCE_DONOR_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_current40"
    / "current40_sft_v5_diverse.jsonl"
)


class MaterializableToolCallStoppingCriteria(StoppingCriteria):
    """Stop after a chosen tool name when visible state uniquely supplies its arguments."""

    def __init__(
        self,
        tokenizer: Any,
        tool_names: list[str],
        *,
        prompt_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        alternatives = "|".join(
            sorted((re.escape(name) for name in tool_names), key=len, reverse=True)
        )
        self.pattern = re.compile(
            rf"<tool_call>\s*\{{?\s*[\"']name[\"']\s*:\s*[\"']"
            rf"(?P<name>{alternatives})[\"']",
            re.I,
        )
        self.matched_tool_name: str | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if not self.pattern.pattern or input_ids.shape[0] != 1:
            return False
        generated = self.tokenizer.decode(
            input_ids[0, self.prompt_length :], skip_special_tokens=False
        )
        match = self.pattern.search(generated)
        if match is None:
            return False
        self.matched_tool_name = match.group("name")
        return True
# Explicitly enumerate only the seed-20260818 development catalog.  Do not use
# a recursive glob here: similarly named external-holdout shards must remain
# outside training/recovery generation unless a caller explicitly supplies one.
VIDDIFF_DEVELOPMENT_CANDIDATE_PATHS = (
    DEFAULT_ADAPTIVE,
    ROOT
    / "outputs/scale30_viddiff_selection"
    / "viddiff_best_of_tree_seed_20260818_n5"
    / "candidate_measurements.student.jsonl",
    ROOT
    / "outputs/scale30_viddiff_selection_15_19"
    / "viddiff_best_of_tree_seed_20260818_n5"
    / "candidate_measurements.student.jsonl",
    ROOT
    / "outputs/scale30_viddiff_selection_20_24"
    / "viddiff_best_of_tree_seed_20260818_n5"
    / "candidate_measurements.student.jsonl",
    ROOT
    / "outputs/scale30_viddiff_selection_25_29"
    / "viddiff_best_of_tree_seed_20260818_n5"
    / "candidate_measurements.student.jsonl",
)
# Backward-compatible name for code that imports the original single shard.
VIDDIFF_CANDIDATES = VIDDIFF_DEVELOPMENT_CANDIDATE_PATHS[2]

VIDEO_FIRST = "VideoSham:videosham-4189:LLM-TRJ-4189-CLEAN-001:clean:1"
VIDEO_SECOND = "VideoSham:videosham-4189:LLM-TRJ-4189-CLEAN-001:clean:2"
VIDEO_REPAIR_FIRST = "VideoSham:videosham-4189:LLM-TRJ-4189-REPAIR-001:repair:1"
VCSL_FIRST = (
    "VCSL:vcsl-06feccaaee94443a8ddfcd59d7938bad-"
    "808d18d803c649e0badaf17caff18f5c:LLM-TRJ-VCSL-"
    "06FECCAAEE94443A8DDFCD59D7938BAD-808D18D803C649E0BADAF17CAFF18F5C-"
    "CLEAN-001:clean:1"
)
VCSL_REPAIR_FIRST = (
    "VCSL:vcsl-a6264ea57d364eaca85e8eef6bf788d7-"
    "c8c2d0a9f1174aa6a63971328c8c586f:LLM-TRJ-VCSL-"
    "A6264EA57D364EACA85E8EEF6BF788D7-C8C2D0A9F1174AA6A63971328C8C586F-"
    "REPAIR-001:repair:1"
)
VIDDIFF_FIRST = (
    "VidDiffBench:viddiff-surgery_70:LLM-TRJ-VIDDIFF-SURGERY_70-"
    "CLEAN-001:clean:1"
)
VIDDIFF_REPAIR_FIRST = (
    "VidDiffBench:viddiff-surgery_155:LLM-TRJ-VIDDIFF-SURGERY_155-"
    "REPAIR-001:repair:1"
)

COMPRESSION_CONTRACT = (
    "The full executable trajectory is retained in the audit source. "
    "This training view removes duplicate historical states and, after a derived meta-evidence "
    "atom is created, replaces the two incorporated source rationales with decision-equivalent "
    "summaries while retaining their IDs, directions, intervals, confidence, order consistency, "
    "and provenance; "
    "current_harness_state is the authoritative pre-action state. "
    "past_execution_history is environment history, not an output template; "
    "the assistant must emit natural-language reasoning plus exactly one native tool call. "
    "Every EvidenceAtom created by a measurement must be consumed later: cite its exact ID "
    "in reasoning or bind it to a claim field. If the atom is inconclusive, cite it when "
    "explaining why a different measurement is necessary. "
    "state_action_affordances externalizes schema-valid argument choices derived from the "
    "same visible state; it is authoritative. executable_decision_actions are complete "
    "native calls compiled from visible meaningful transitions; copy their argument keys "
    "exactly when selecting one. repair_restoration_candidates identify an eligible prior "
    "measurement_method after evidence dropout, but the new observation must still be "
    "executed and validated. executable_binding_actions are complete native calls that cover "
    "all currently bindable fields with a minimum-cardinality compatible evidence set; when "
    "present, they replace the redundant lower-level binding_candidates in the policy view."
)

PROGRESSIVE_COMPRESSION_CONTRACT = (
    "current_harness_state is the authoritative pre-action state; history records past "
    "outcomes, not an output template. Emit concise natural-language reasoning and exactly "
    "one eligible native tool call. Consume every measured EvidenceAtom by exact citation, "
    "candidate assessment, or binding. For assess/adjudicate states, the eligible tool "
    "schema const options and matching evidence frontier are authoritative; preserve null "
    "as unobserved, complete the per-candidate ledger, and make final adjudication agree with it. "
    "When eligible_tools contains multiple actions, selecting one best next action does not make "
    "it the only eligible action."
)

NATURAL_REASONING_CONTRACT = (
    " Write like an investigator deciding from evidence, not like a checker narrating an "
    "interface. State the concrete missing fact or state transition directly. Across turns, "
    "vary sentence structure and do not repeatedly use stock frames such as `the latest visible "
    "state`, `the only remaining gap`, `the blocking gap`, or `the only eligible tool`. Never "
    "expose internal state-key or constraint-key names in the reasoning."
)

PROGRESSIVE_MEASUREMENT_SYSTEM = (
    "You are the semantic measurement policy inside a state-externalizing video-evidence "
    "harness. Use only the current student-visible state. Emit 2-4 grounded sentences "
    "(40-120 words) plus exactly one eligible native tool call. Name the selected tool and the "
    "unresolved observation gap; a measurement gap is never merely that evidence is unbound, "
    "uncommitted, or not stopped. Describe a new measurement prospectively—what it will inspect "
    "or test—without promising its result. For VidDiff routes: action_refined_symmetric checks "
    "predicate-relevant local behavior under both A/B orders; global_motion_symmetric checks "
    "whole-video motion under both orders; uniform_timeline checks coarse uniformly sampled "
    "chronological visual coverage and observes neither global-motion statistics nor A/B order "
    "consistency. When repair_restoration_candidates names an eligible dropped method, execute "
    "that method first rather than rerunning an already-active substitute route. Cite the latest "
    "visible EvidenceAtom and preserve its structured facts. An active atom's supports_fields do "
    "not disappear merely because claim bindings are absent; use visible uncertainty or cross-method "
    "disagreement as the reason for complementary sensing. If a named atom has preferred_video=c, "
    "order_consistent=false, and an order-disagreement abstention rule, explain that its abstention "
    "is caused by disagreement across A/B presentation orders; visual cue strength and routing "
    "state are context, not the direct cause. Under evidence_dropout, a removed ID "
    "stays invalid and a new observation must have a distinct ID. "
    "Never infer oracle labels, verifier results, or unobserved values."
)

PROGRESSIVE_VCSL_LOCALIZATION_SYSTEM = (
    "You are the semantic localization policy inside a state-externalizing video-copy "
    "evidence harness. Use only the current student-visible state. Emit 2-4 grounded "
    "sentences (40-120 words) plus exactly one eligible native tool call. Name the exact "
    "selected localize_copy_segment tool and the unresolved localization gap. For a DP route, "
    "write `dynamic programming (DP)` explicitly in the prose; for another route, spell out "
    "its literal method name. The initial "
    "pair EvidenceAtom establishes only the query/reference identities and any fields "
    "explicitly present in its observation; never add preferred_video, order_consistent, "
    "confidence, or a direction judgment. A localizer operates on the query/reference "
    "feature-similarity map to estimate their temporal relationship and copied segment "
    "boundaries. Dynamic programming or other local alignment does not test A/B presentation "
    "orders, role swapping, or VidDiff direction symmetry. Describe the localization "
    "prospectively—what structure it will inspect and which missing temporal evidence it can "
    "produce—without promising the result. Never infer oracle labels, verifier results, or "
    "unobserved values."
)

PROGRESSIVE_EVIDENCE_SYSTEM = (
    "You are the semantic policy inside a state-externalizing evidence harness. "
    "The current_harness_state is authoritative. Emit 2-3 concise, grounded sentences plus "
    "exactly one eligible native tool call. Identify the selected action by its literal name, "
    "but explain the decision in natural case-specific prose rather than using a fixed opening. "
    "The harness may materialize the sole "
    "schema-valid arguments from the visible action frontier, so spend language on the decision "
    "rather than restating JSON. Use only student-visible "
    "facts; never infer hidden labels, oracle values, or unobserved order consistency. "
    "For assess_evidence_candidate, judge exactly the named EvidenceAtom and cite its exact ID "
    "or unique method. For measurement/comparison atoms, state preferred_video and the exact "
    "order_consistent value (explicitly unobserved/null when absent). For identity/context atoms, "
    "instead cite their exact supported identity facts and do not invent absent comparison fields. "
    "Record retain or reject, and name a visible defect for a rejection. When "
    "claim_value_conflict is validated, name one of the frontier's actual "
    "mismatched fields and never call a matching preferred_video a conflict. Each assessment "
    "with preferred_video=c, order_consistent=false, and an order-disagreement abstention rule "
    "must explain in natural language that disagreement across A/B presentation orders causes "
    "the abstention; do not merely repeat a decision_rule token or routing reason. Each assessment "
    "must close one pending-candidate gap without changing claim "
    "values or creating visual evidence. Only current_candidate_id is actionable on this turn; "
    "do not summarize or decide deferred_candidate_ids. An assessment call records only that one "
    "candidate; it never carries the full rejected-ID/defect partition. After the ledger is complete, adjudicate_evidence "
    "must identify every selected atom by exact ID or its visible turn-local reasoning_handle, "
    "then justify it with a concrete visible field/value that matches the claim. It must agree "
    "with all recorded dispositions. Refer explicitly to the completed "
    "per-candidate ledger instead of repeating every already-validated rejection explanation; the "
    "native call still carries every exact rejected ID and defect code. Do not use "
    "newest, majority, or confidence alone as the decision rule; preserve null versus false. "
    "An assessment records retain/reject only; it never binds evidence or closes field support."
)

PROGRESSIVE_ASSESSMENT_SYSTEM = (
    "You are the candidate-assessment policy in a state-externalizing evidence harness. "
    "Use only the student-visible current state. Write 2-4 concise sentences and identify "
    "`assess_evidence_candidate` as the selected action before emitting that one native tool "
    "call. Express the supplied decision-gap facts in natural case-specific prose; do not copy "
    "a stock clause merely because it is present in the state. Cite "
    "the exact current EvidenceAtom ID. For an identity/context atom, state every exact "
    "supported_observation_fact and do not invent comparison fields. For a measurement atom, "
    "state preferred_video and order_consistent exactly, including null/unobserved. State the "
    "visible retain/reject disposition and a validated defect for rejection. When c/false uses "
    "an order-disagreement abstention rule, explain in your own words why A/B presentation-order "
    "disagreement causes the abstention. Preserve the directed causal edge: disagreement across "
    "the two A/B presentation orders is the cause and preferred_video=c abstention is the effect; "
    "never reverse that relation. This "
    "action records only one disposition: it never binds or supports claim "
    "fields, and it cannot complete the ledger while deferred candidates remain. Never infer "
    "oracle labels, verifier results, or hidden values."
)

PROGRESSIVE_ADJUDICATION_SYSTEM = (
    "You are the evidence-adjudication policy in a state-externalizing harness. Use only the "
    "student-visible completed assessment ledger. Write 2-4 concise sentences, identify "
    "`adjudicate_evidence` as the selected action, then emit that one native tool call. Name the missing final "
    "selected/rejected partition as the decision gap. Before any generic summary, naturally "
    "express every selected_support_facts item in the reasoning: each selected EvidenceAtom "
    "must have its turn-local reasoning_handle (or exact ID) and a literal field=value that "
    "matches the claim. Do not mention prompt fields, requirements, or clauses. Say the completed ledger "
    "supplies the already-validated rejection reasons. Adjudication records the partition only; "
    "it never binds fields or creates visual evidence. Never infer oracle labels or hidden facts."
)

POST_ADJUDICATION_SYSTEM = (
    "You are the semantic policy inside a state-externalizing evidence harness. The current "
    "harness state is authoritative. Emit 2-4 complete, grounded sentences (45-120 words, or "
    "25-120 for stop) plus exactly one eligible native tool call. Use only student-visible facts "
    "and preserve null as unobserved. The evidence ledger is already adjudicated: bind only the "
    "selected EvidenceAtom IDs to the exact supported open fields, never a rejected atom. Binding "
    "does not commit a field; after successful binding, commit only the still-open supported fields. "
    "For evidence_dropout, the removed EvidenceAtom ID is invalid and only a newly generated active "
    "ID may be rebound; never describe dropout as the same atom remaining valid, which is instead "
    "the binding_invalidation case. "
    "Call stop only when all required fields are supported and committed and no high-priority "
    "obligation remains. Cite the concrete selected EvidenceAtom or state fact, name the single "
    "remaining gap, and explain why the chosen action closes it without changing independent fields."
)


META_ABSTENTION_METHOD = "compare_symmetry_residual_abstention"
META_SOURCE_METHODS = {
    "compare_global_motion_symmetric",
    "compare_action_refined_symmetric",
}
DERIVED_META_POLICY_CONTRACT = (
    " For any derived/meta action, ground every required source with its exact EvidenceAtom ID "
    "in the current or an earlier turn; the meta rationale must name both source methods or "
    "unambiguously refer to the two fixed symmetric source routes, and "
    "explicitly state that the action aggregates prior measurements without inspecting new "
    "pixels. The derived meta observation is role-invariant and has order_consistent=true; "
    "never describe that result as order-inconsistent merely because its source routes were. "
    "Keep reasoning and native arguments aligned: if the rationale says "
    "the derived meta atom is strongest, create and bind the claim from that atom's method and "
    "intervals. Choosing an older source instead requires naming a concrete visible defect in "
    "the meta atom. The derived meta atom supports only preferred_video, both interval fields, "
    "and measurement_method; the task-identity atom separately supports sample_key and "
    "comparison_statement. If later reasoning summarizes all six fields as supported, it must "
    "name both evidence atoms rather than attributing all six to the meta atom."
    " More generally, preserve the latest visible atom's structured facts: if it is active and "
    "contains A/B intervals, do not say that measurement evidence or interval evidence is absent. "
    "When more evidence is needed, name the real visible defect such as low confidence, order "
    "inconsistency, or cross-method disagreement."
    " For an absolute never/always statement, counterexamples and sampling coverage outrank a "
    "generic overall impression. If an active typed predicate atom reports usable independent "
    "A/B records, zero violations for both, and sparse_sampling_cannot_prove_never, preserve its "
    "c abstention; do not replace it with an a/b result from a coarse route merely because that "
    "route has higher confidence or joins a majority."
    " Whenever a visible atom has preferred_video=c, order_consistent=false, and an "
    "order-disagreement abstention rule, say that disagreement across the two A/B presentation "
    "orders directly causes the abstention; cross-method disagreement or visual strength is "
    "context only."
    " create_candidate_claim records field values and obligations only; it never binds evidence, "
    "marks fields supported, or commits them. Reserve those state changes for bind_evidence and "
    "set_field_decision."
    " For create_candidate_claim and bind_evidence calls with long native arguments, keep the "
    "reasoning compact at 45-65 words so the complete JSON call remains inside the generation "
    "budget."
)


class EligibleToolNameLogitsProcessor(LogitsProcessor):
    """Constrain only the JSON tool-name string to the visible eligible set.

    Natural-language reasoning and argument JSON remain unconstrained. The
    processor activates only after an open `"name": "` field inside
    `<tool_call>` and releases control after the closing quote.
    """

    def __init__(
        self,
        tokenizer: Any,
        eligible_tools: list[str],
        prompt_length: int,
    ) -> None:
        if not eligible_tools:
            raise ValueError("eligible tool-name constraint requires at least one tool")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.sequences = [
            # Include the JSON comma so a tokenizer token such as `",` is
            # forced intact rather than being replaced by a bare quote.
            list(tokenizer.encode(f'{name}",', add_special_tokens=False))
            for name in eligible_tools
        ]
        if any(not sequence for sequence in self.sequences):
            raise ValueError("eligible tool name tokenized to an empty sequence")
        self.name_start: int | None = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise ValueError("eligible tool-name constraint currently supports batch size one")
        generated = input_ids[0, self.prompt_length :].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        tool_call = text.rsplit("<tool_call>", 1)[-1] if "<tool_call>" in text else ""
        open_name = re.search(r'\{\s*"name"\s*:\s*"([^"]*)$', tool_call)
        if self.name_start is None:
            if open_name is None or open_name.group(1) != "":
                return scores
            self.name_start = len(generated)
        elif open_name is None:
            return scores

        prefix = generated[self.name_start :]
        matching = [
            sequence
            for sequence in self.sequences
            if len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix
        ]
        if not matching:
            return scores
        allowed = sorted({sequence[len(prefix)] for sequence in matching})
        constrained = torch.full_like(scores, float("-inf"))
        constrained[:, allowed] = scores[:, allowed]
        return constrained


def _post_meta_policy_view(
    state: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep decision facts while compacting redundant multi-measurement text."""

    evidence_store = state.get("evidence_store") or {}
    measurement_items = [
        (evidence_id, atom)
        for evidence_id, atom in evidence_store.items()
        if atom.get("active") is True
        and str((atom.get("observation") or {}).get("measurement_method", "")).startswith(
            "compare_"
        )
    ]
    meta_items = [
        (evidence_id, atom)
        for evidence_id, atom in measurement_items
        if (atom.get("observation") or {}).get("measurement_method") == META_ABSTENTION_METHOD
    ]
    if not meta_items and len(measurement_items) < 2:
        return state, history

    meta_id = str(meta_items[-1][0]) if meta_items else None
    latest_measurement_id = str(measurement_items[-1][0])
    compact_state = copy.deepcopy(state)
    progressive_assessment = "evidence_assessment" in compact_state
    compacted_ids: list[str] = []
    for evidence_id, atom in (compact_state.get("evidence_store") or {}).items():
        observation = atom.get("observation") or {}
        method = observation.get("measurement_method")
        routing_state = observation.get("routing_state") or {}
        if str(method).startswith("compare_") and routing_state:
            compact_routing = {
                key: routing_state.get(key)
                for key in (
                    "cross_method_disagreement",
                    "goal_satisfied",
                    "reason_code",
                    "stable_observed_directions",
                )
                if key in routing_state
            }
            if compact_routing:
                observation["routing_state"] = compact_routing
            else:
                observation.pop("routing_state", None)
        if meta_id is not None and method in META_SOURCE_METHODS:
            observation["visible_rationale"] = (
                f"Full source rationale was incorporated into {meta_id}. Retained decision facts: "
                f"preferred_video={observation.get('preferred_video')}, "
                f"confidence={observation.get('confidence')}, "
                f"order_consistent={observation.get('order_consistent')}."
            )
            compacted_ids.append(str(evidence_id))
        if progressive_assessment and str(method).startswith("compare_"):
            rationale = str(observation.get("visible_rationale") or "").strip()
            excerpt = rationale[:140].rstrip()
            if len(rationale) > len(excerpt):
                excerpt += "…"
            observation["visible_rationale"] = (
                f"Visible cue excerpt: {excerpt} Retained structured facts are "
                f"authoritative for assessment."
            )
            provenance = atom.get("provenance") or {}
            atom["provenance"] = {
                key: provenance.get(key)
                for key in ("generated_by", "method", "label_visible_to_tool")
                if key in provenance
            }
            if str(evidence_id) not in compacted_ids:
                compacted_ids.append(str(evidence_id))
        elif (
            meta_id is None
            and str(evidence_id) != latest_measurement_id
            and str(method).startswith("compare_")
        ):
            rationale = str(observation.get("visible_rationale") or "").strip()
            excerpt = rationale[:220].rstrip()
            if len(rationale) > len(excerpt):
                excerpt += "…"
            observation["visible_rationale"] = (
                f"Earlier measurement rationale excerpt: {excerpt} Retained decision facts: "
                f"preferred_video={observation.get('preferred_video')}, "
                f"confidence={observation.get('confidence')}, "
                f"order_consistent={observation.get('order_consistent')}."
            )
            compacted_ids.append(str(evidence_id))
    compact_state["policy_view_compaction"] = {
        "kind": (
            "post_meta_source_rationale_summary_v1"
            if meta_id is not None
            else "multi_measurement_rationale_summary_v1"
        ),
        "latest_full_measurement_evidence_id": latest_measurement_id,
        "compacted_source_evidence_ids": compacted_ids,
        "decision_fields_retained": [
            "evidence_id",
            "preferred_video",
            "video_a_interval_s",
            "video_b_interval_s",
            "measurement_method",
            "confidence",
            "order_consistent",
            "decision_rule",
            "provenance",
            "active",
        ],
        "full_state_reconstructable_from_audit_turns": True,
    }
    if meta_id is not None:
        compact_state["policy_view_compaction"]["derived_evidence_id"] = meta_id

    if progressive_assessment:
        assessment_state = compact_state.get("evidence_assessments") or {}
        compact_state["evidence_assessments"] = {
            claim_id: {
                evidence_id: {
                    "assessment_id": assessment.get("assessment_id"),
                    "disposition": assessment.get("disposition"),
                    "defect_codes": list(assessment.get("defect_codes") or []),
                }
                for evidence_id, assessment in assessments.items()
            }
            for claim_id, assessments in assessment_state.items()
        }
        compact_state["policy_view_compaction"][
            "progressive_assessment_ledger_compacted"
        ] = True
        compact_state["policy_view_compaction"][
            "assessment_scope_reference"
        ] = "evidence_adjudication.requirement"
        affordances = compact_state.get("state_action_affordances") or {}
        if "assess_evidence_candidate" in set(compact_state.get("eligible_tools") or []):
            # Final adjudication arguments are not executable until the ledger is
            # complete and duplicate the same partition shown by the assessment
            # options. Keep the current executable assessment actions only.
            affordances["executable_adjudication_actions"] = []
            affordances["deferred_adjudication"] = (
                "Unlocked only after pending_evidence_ids is empty."
            )
            assessment_actions = list(
                affordances.get("executable_evidence_assessment_actions") or []
            )
            affordances["evidence_assessment_action_summary"] = [
                {
                    key: (action.get("arguments") or {}).get(key)
                    for key in ("evidence_id", "disposition", "defect_codes")
                }
                for action in assessment_actions
            ]
            affordances["executable_evidence_assessment_actions"] = []
            affordances["current_native_actions_reference"] = (
                "eligible tool schema const options and evidence_assessment_frontier"
            )
            compact_state["policy_view_compaction"][
                "deferred_noneligible_adjudication_removed"
            ] = True
            compact_state["policy_view_compaction"][
                "duplicate_assessment_native_actions_summarized"
            ] = True
        elif compact_state.get("eligible_tools") == ["adjudicate_evidence"]:
            # The exact partition is already represented by the eligible tool
            # schema and the adjudication frontier. A third full copy in the
            # state affordances consumes context without adding a choice.
            affordances["executable_adjudication_actions"] = []
            affordances["current_native_actions_reference"] = (
                "eligible tool schema const options and evidence_adjudication_frontier"
            )
            compact_state["policy_view_compaction"][
                "duplicate_adjudication_native_action_removed"
            ] = True

    candidate_claims = compact_state.get("candidate_claims") or []
    selected_methods = {
        str((claim.get("fields") or {}).get("measurement_method", {}).get("value", ""))
        for claim in candidate_claims
    }
    if meta_id is not None and META_ABSTENTION_METHOD in selected_methods:
        nonselected_ids: list[str] = []
        for evidence_id, atom in (compact_state.get("evidence_store") or {}).items():
            observation = atom.get("observation") or {}
            method = str(observation.get("measurement_method", ""))
            if not method.startswith("compare_") or method == META_ABSTENTION_METHOD:
                continue
            atom["observation"] = {
                "preferred_video": observation.get("preferred_video"),
                "measurement_method": method,
                "confidence": observation.get("confidence"),
                "order_consistent": observation.get("order_consistent"),
                "decision_rule": observation.get("decision_rule"),
                "summary": (
                    "Nonselected measurement already consumed before claim creation; full "
                    "observation remains reconstructable from prior audit turns."
                ),
            }
            nonselected_ids.append(str(evidence_id))
        compact_state["policy_view_compaction"]["post_claim_nonselected_evidence_ids"] = (
            nonselected_ids
        )

    field_bindings = compact_state.get("field_bindings") or {}
    if field_bindings:
        compact_state["field_bindings_summary"] = {
            claim_id: {
                field: {
                    "support_ids": list(binding.get("support_ids") or []),
                    "validation_status": binding.get("validation_status"),
                }
                for field, binding in claim_bindings.items()
            }
            for claim_id, claim_bindings in field_bindings.items()
        }
        compact_state["field_bindings"] = {}
        compact_state["policy_view_compaction"]["field_bindings_compacted"] = True

    compact_history = copy.deepcopy(history)
    for item in compact_history:
        past_action = item.get("past_action") or {}
        tool = str(past_action.get("executed_tool", ""))
        past_input = past_action.get("past_input") or {}
        if tool.startswith("compare_"):
            past_input.pop("comparison_statement", None)
        elif tool == "create_candidate_claim":
            past_action["past_input"] = {"claim_id": past_input.get("claim_id")}
        elif tool == "bind_evidence":
            past_action["past_input"] = {
                "claim_id": past_input.get("claim_id"),
                "bound_fields": [
                    binding.get("field")
                    for binding in past_input.get("binding_items") or []
                ],
            }
        elif tool == "assess_evidence_candidate":
            past_action["past_input"] = {
                "evidence_id": past_input.get("evidence_id"),
                "disposition": past_input.get("disposition"),
                "defect_codes": list(past_input.get("defect_codes") or []),
            }
    if progressive_assessment:
        compact_state, compact_history = _compact_progressive_decision_view(
            compact_state, compact_history
        )
    compact_state, compact_history = _compact_viddiff_state_transition_view(
        compact_state, compact_history
    )
    return compact_state, compact_history


def _compact_progressive_decision_view(
    state: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove facts duplicated by the evidence store, frontier, and sole native schema."""

    eligible = list(state.get("eligible_tools") or [])
    if eligible not in (["assess_evidence_candidate"], ["adjudicate_evidence"]):
        return state, history
    compact = copy.deepcopy(state)
    compact["candidate_claims"] = [
        {
            "claim_id": claim.get("claim_id"),
            "fields": {
                field_name: {
                    key: field.get(key)
                    for key in ("value", "evidence_status", "decision_status")
                    if key in field
                }
                for field_name, field in (claim.get("fields") or {}).items()
            },
        }
        for claim in compact.get("candidate_claims") or []
    ]
    adjudication = compact.get("evidence_adjudication") or {}
    requirement = adjudication.get("requirement") or {}
    compact["evidence_adjudication"] = {
        "required": adjudication.get("required"),
        "requirement": {
            key: copy.deepcopy(requirement.get(key))
            for key in ("claim_id", "target_fields", "competing_evidence_ids")
            if key in requirement
        },
    }
    assessment = compact.get("evidence_assessment") or {}
    if eligible == ["assess_evidence_candidate"]:
        compact["evidence_assessment"] = {
            key: copy.deepcopy(assessment.get(key))
            for key in (
                "required",
                "pending_evidence_ids",
                "current_candidate_id",
                "deferred_candidate_ids",
            )
            if key in assessment
        }
    else:
        compact.pop("evidence_assessment", None)
    compact["policy_view_compaction"] = {
        "kind": "progressive_decision_state_v2",
        "authoritative_candidate_facts": "evidence_store",
        "native_arguments_reference": (
            "eligible tool schema const options and matching evidence frontier"
        ),
        "full_state_reconstructable_from_audit_turns": True,
    }
    for key in (
        "case_goals",
        "task_contract",
        "available_evidence_ids",
        "dropped_evidence_ids",
        "field_bindings",
        "field_bindings_summary",
        "evidence_adjudications",
        "stop_decision",
    ):
        compact.pop(key, None)
    constraints = compact.get("action_constraints") or {}
    if not constraints.get("must_reopen_before_rebinding"):
        compact.pop("action_constraints", None)

    compact_history = copy.deepcopy(history)
    for item in compact_history:
        outcome = item.get("environment_outcome") or {}
        item["environment_outcome"] = {
            key: copy.deepcopy(outcome.get(key))
            for key in (
                "status",
                "created_evidence_ids",
                "assessment_evidence_id",
                "assessment_disposition",
            )
            if key in outcome
        }
    return compact, compact_history


def _compact_viddiff_state_transition_view(
    state: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep decision-sufficient facts for sole bind/commit/stop frontiers."""

    eligible = list(state.get("eligible_tools") or [])
    if eligible not in (["bind_evidence"], ["set_field_decision"], ["stop"]):
        return state, history
    schema_id = str((state.get("task_contract") or {}).get("claim_schema_id") or "")
    if schema_id != "viddiff_comparative_measurement_claim_v1":
        return state, history
    compact = copy.deepcopy(state)
    compact["candidate_claims"] = [
        {
            "claim_id": claim.get("claim_id"),
            "fields": {
                field_name: {
                    key: copy.deepcopy(field.get(key))
                    for key in (
                        "value",
                        "evidence_status",
                        "decision_status",
                        "support_ids",
                    )
                    if key in field
                }
                for field_name, field in (claim.get("fields") or {}).items()
            },
        }
        for claim in compact.get("candidate_claims") or []
    ]
    observation_keys = (
        "sample_key",
        "comparison_statement",
        "preferred_video",
        "video_a_interval_s",
        "video_b_interval_s",
        "measurement_method",
        "confidence",
        "order_consistent",
        "decision_rule",
    )
    compact["evidence_store"] = {
        evidence_id: {
            "evidence_id": atom.get("evidence_id", evidence_id),
            "kind": atom.get("kind"),
            "supports_fields": list(atom.get("supports_fields") or []),
            "observation": {
                key: copy.deepcopy((atom.get("observation") or {}).get(key))
                for key in observation_keys
                if key in (atom.get("observation") or {})
            },
            "provenance": {
                key: copy.deepcopy((atom.get("provenance") or {}).get(key))
                for key in ("generated_by", "method", "label_visible_to_tool")
                if key in (atom.get("provenance") or {})
            },
            "active": atom.get("active"),
        }
        for evidence_id, atom in (compact.get("evidence_store") or {}).items()
    }
    affordances = compact.get("state_action_affordances") or {}
    compact["state_action_affordances"] = {
        key: copy.deepcopy(affordances.get(key))
        for key in (
            "derivation",
            "executable_binding_actions",
            "repair_restoration_candidates",
            "meaningful_decision_transitions",
            "executable_decision_actions",
        )
        if affordances.get(key)
    }
    compact["policy_view_compaction"] = {
        "kind": "viddiff_state_transition_v1",
        "authoritative_evidence_facts": "evidence_store",
        "native_arguments_reference": "execution_frontier.native_action_templates",
        "removed_duplicate_visual_rationales": True,
        "full_state_reconstructable_from_audit_turns": True,
    }
    for key in (
        "case_goals",
        "task_contract",
        "available_evidence_ids",
        "field_bindings",
        "evidence_assessment",
        "evidence_assessments",
        "evidence_adjudication",
        "evidence_adjudications",
        "stop_decision",
    ):
        compact.pop(key, None)
    return compact, history


def _post_meta_task_view(
    task: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("policy_view_compaction"):
        return task
    claim_schema = task.get("claim_schema") or {}
    compact = {
        "claim_schema_id": claim_schema.get("schema_id"),
        "required_fields": list(claim_schema.get("required_fields") or []),
        "claim_field_schemas_reference": "currently eligible native tool schema",
    }
    if (state.get("policy_view_compaction") or {}).get("kind") not in {
        "progressive_decision_state_v2",
        "viddiff_state_transition_v1",
    }:
        compact["task_contract_reference"] = "current_harness_state.task_contract"
    if task.get("task") is not None:
        compact["task"] = task.get("task")
    if task.get("mutation") is not None:
        compact["mutation"] = copy.deepcopy(task.get("mutation"))
    return compact


def _public_source_and_scope(row: dict[str, Any]) -> tuple[str, list[float]]:
    state = _payload(row)["current_harness_state"]
    source_id = None
    scope = None
    for atom in (state.get("evidence_store") or {}).values():
        observation = atom.get("observation") or {}
        source_id = observation.get("source_id", source_id)
        scope = observation.get("comparison_scope_s", scope)
    if source_id is None or scope is None:
        raise ValueError("public VideoSham state lacks source identity or scope")
    return str(source_id), list(scope)


def _video_environment(rows: dict[str, dict[str, Any]]) -> MultiFamilyVideoEnvironment:
    source_id, scope = _public_source_and_scope(rows[VIDEO_SECOND])
    source_path = SOURCE_ROOT / "4189_unedited.mp4"
    suspect_path = EDITED_ROOT / "4189_processed.mp4"
    return MultiFamilyVideoEnvironment(
        manifest={
            "case_id": "videosham-4189",
            "source_id": source_id,
            "comparison_scope_s": scope,
        },
        source_frames=decode(source_path),
        suspect_frames=decode(suspect_path),
        source_path=source_path,
        suspect_path=suspect_path,
    )


def _vcsl_environment() -> VCSLFeatureEnvironment:
    return VCSLFeatureEnvironment(
        case={
            "pair_id": (
                "vcsl-06feccaaee94443a8ddfcd59d7938bad-"
                "808d18d803c649e0badaf17caff18f5c"
            ),
            "query_id": "06feccaaee94443a8ddfcd59d7938bad",
            "reference_id": "808d18d803c649e0badaf17caff18f5c",
        },
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="tn_stability",
    )


def _video_repair_environment(
    rows: dict[str, dict[str, Any]],
) -> MultiFamilyVideoEnvironment:
    environment = _video_environment(rows)
    environment.execute("inspect_media", {"media_refs": ["source", "suspect"]})
    environment.execute(
        "detect_temporal_edit",
        {
            "source_ref": "source",
            "suspect_ref": "suspect",
            "comparison_scope_s": [0.0, 10.043],
        },
    )
    return environment


def _vcsl_repair_environment() -> VCSLFeatureEnvironment:
    environment = VCSLFeatureEnvironment(
        case={
            "pair_id": (
                "vcsl-a6264ea57d364eaca85e8eef6bf788d7-"
                "c8c2d0a9f1174aa6a63971328c8c586f"
            ),
            "query_id": "a6264ea57d364eaca85e8eef6bf788d7",
            "reference_id": "c8c2d0a9f1174aa6a63971328c8c586f",
        },
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant="dp",
    )
    environment.execute(
        "inspect_pair_features",
        {"query_ref": "query", "reference_ref": "reference"},
    )
    environment.execute(
        "localize_copy_segment_dp",
        {"query_ref": "query", "reference_ref": "reference", "method": "DP"},
    )
    environment.on_evidence_dropout(
        "EA-VCSL-LOCALIZATION-001",
        ["relationship", "copied_segments_frames", "localization_method"],
    )
    return environment


def _viddiff_item(sample_key: str) -> dict[str, Any]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in VIDDIFF_DEVELOPMENT_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        for item in _jsonl(path):
            if item["sample_key"] == sample_key:
                matches.append((path, item))
    if len(matches) == 1:
        return copy.deepcopy(matches[0][1])
    if len(matches) > 1:
        locations = [str(path.relative_to(ROOT)) for path, _ in matches]
        raise ValueError(
            f"ambiguous development measurements for {sample_key}: {locations}"
        )
    raise ValueError(f"student-visible VidDiff measurements unavailable: {sample_key}")


def _viddiff_repair_environment() -> VidDiffAdaptiveMeasurementEnvironment:
    item = _viddiff_item("surgery_155")
    environment = VidDiffAdaptiveMeasurementEnvironment(item)
    environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
    by_name = {candidate["name"]: candidate for candidate in item["candidate_tools"]}
    for name in (
        "compare_action_refined_symmetric",
        "compare_uniform_timeline",
        "compare_global_motion_symmetric",
    ):
        environment.execute(name, by_name[name]["arguments"])
    environment.on_evidence_dropout(
        "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001",
        ["preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method"],
    )
    return environment


def _compile_visible_decision_actions(state: dict[str, Any]) -> None:
    affordances = state.get("state_action_affordances") or {}
    order = {
        str(claim["claim_id"]): list((claim.get("fields") or {}).keys())
        for claim in state.get("candidate_claims") or []
    }
    affordances["executable_decision_actions"] = [
        {
            "tool_name": "set_field_decision",
            "arguments": {
                "claim_id": transition["claim_id"],
                "fields": [
                    field
                    for field in order.get(
                        str(transition["claim_id"]), transition["fields"]
                    )
                    if field in set(transition["fields"])
                ],
                "decision_status": transition["decision_status"],
            },
        }
        for transition in affordances.get("meaningful_decision_transitions", [])
    ]
    affordances["derivation"] = "Compiled from student-visible state."
    state["state_action_affordances"] = affordances


def _prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[int], int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        tools=tools,
    )["input_ids"]
    source = len(encoded)
    return encoded[-max_length:], source


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": result.get("status"),
        "authority": result.get("authority"),
        "created_evidence_ids": list(result.get("created_evidence_ids") or []),
    }
    if result.get("error"):
        summary["error"] = str(result["error"])
    if result.get("stop_decision"):
        summary["stop_decision"] = copy.deepcopy(result["stop_decision"])
    return summary


def _schema_valid(
    tools: list[dict[str, Any]], name: str | None, arguments: dict[str, Any] | None
) -> bool:
    if name is None or arguments is None:
        return False
    function = next(
        (item["function"] for item in tools if item["function"]["name"] == name),
        None,
    )
    return bool(
        function is not None
        and not list(Draft202012Validator(function["parameters"]).iter_errors(arguments))
    )


def _schema_retry_feedback(
    tools: list[dict[str, Any]],
    name: str | None,
    arguments: dict[str, Any] | None,
) -> str:
    """Return concise, native-schema-grounded repair guidance for one draft."""
    if name is None or arguments is None:
        return ""
    function = next(
        (item["function"] for item in tools if item["function"]["name"] == name),
        None,
    )
    if function is None:
        return ""
    schema = function["parameters"]
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if not errors:
        return ""
    details: list[str] = []
    for error in errors[:4]:
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        details.append(f"at `{location}`: {error.message}")
    required = list(schema.get("required") or [])
    properties = list((schema.get("properties") or {}).keys())
    shape = ""
    if required or properties:
        shape = (
            f" The allowed top-level keys are {properties}; required keys are {required}."
        )
    migration = ""
    if "binding_items" in required and "field_bindings" in arguments:
        migration = (
            " Rename the top-level key `field_bindings` to exactly `binding_items`; "
            "do not keep both keys. The valid shape is "
            '`{"claim_id":"...","binding_items":'
            '[{"field":"...","support_ids":["..."]}]}`.'
        )
    return (
        f"The native `{name}` argument schema rejected the draft "
        + "; ".join(details)
        + shape
        + migration
        + " Preserve arrays as JSON arrays and emit no legacy or extra keys."
    )


def _eligibility_retry_feedback(name: str | None, eligible: list[str]) -> str:
    if name is None or name in eligible:
        return ""
    return (
        f"`{name}` is not executable from the unchanged visible state. "
        f"Choose exactly one currently eligible tool: {eligible}. Do not skip the "
        "remaining measurement or dependency step."
    )


def _action_lock_safe(problems: list[str]) -> bool:
    """Lock a chosen branch only for surface alignment/format defects.

    Evidence-fact, support, binding, or state-status errors require a fresh
    plan and must never be reinforced by an action lock.
    """

    if not problems:
        return False
    safe_exact = {
        "tool_arguments_not_schema_valid",
        "reasoning_does_not_name_decision_gap",
        "reasoning_missing_terminal_punctuation",
        "generation_hit_limit",
        "adjudication_selection_not_justified",
        "binding_described_as_assessment_transition",
    }
    safe_prefixes = (
        "chosen_measurement_tool_not_justified:",
        "reasoning_sentence_count:",
        "reasoning_word_count:",
        "eligible_tool_count_mismatch:",
        "assessment_retained_uncertainty_called_defect:",
        "assessment_candidate_not_cited:",
        "assessment_identity_fact_not_stated:",
        "typed_universal_abstention_overridden:",
        "assessment_preferred_video_not_stated:",
        "assessment_unobserved_order_not_stated:",
        "assessment_order_inconsistency_not_stated:",
        "assessment_reject_disposition_not_explained:",
        "assessment_rejection_defect_not_explained:",
        "assessment_abstention_decision_rule_not_explained:",
        "adjudication_selected_id_not_cited:",
        "adjudication_gap_misframed_as_binding",
    )
    return all(
        problem in safe_exact or problem.startswith(safe_prefixes)
        for problem in problems
    )


def _next_action_lock_tool(
    pinned_tool_name: str | None,
    semantic_name: str | None,
    parsed_name: str | None,
    *,
    parseable: bool,
    eligible_call: bool,
    problems: list[str],
) -> str | None:
    """Retain a branch only when every rejection is a surface-form defect."""
    if not _action_lock_safe(problems):
        return None
    if pinned_tool_name:
        return pinned_tool_name if semantic_name in {None, pinned_tool_name} else None
    if semantic_name:
        return semantic_name
    if parseable and eligible_call:
        return parsed_name
    return None


def _recommended_retry_tool(problems: list[str], eligible: list[str]) -> str | None:
    prefixes = (
        "student_visible_measurement_recommendation_bypassed:",
        "repair_restoration_candidate_bypassed:",
    )
    recommended = {
        problem.split(":", 1)[1]
        for problem in problems
        if problem.startswith(prefixes) and ":" in problem
    }
    if len(recommended) != 1:
        return None
    tool_name = next(iter(recommended))
    return tool_name if tool_name in eligible else None


def _initial_recommended_tool(
    payload: dict[str, Any], eligible: list[str]
) -> str | None:
    """Return a unique, student-visible measurement recommendation, if any."""

    contract = (
        (payload.get("execution_frontier") or {}).get("measurement_gap_contract")
        or {}
    )
    recommendation = contract.get("recommended_next_measurement") or {}
    tool_name = str(recommendation.get("tool_name") or "")
    return tool_name if tool_name in eligible else None


def _compact_recommended_measurement_retry_payload(
    payload: dict[str, Any], pinned_tool_name: str
) -> dict[str, Any]:
    """Keep the full structured decision surface while removing retry-only redundancy."""

    state = payload.get("current_harness_state") or {}
    compact_store: dict[str, Any] = {}
    for evidence_id, atom in (state.get("evidence_store") or {}).items():
        observation = atom.get("observation") or {}
        if atom.get("active") is not True or not (
            observation.get("measurement_method")
            or observation.get("query_id")
            or observation.get("reference_id")
        ):
            continue
        compact_observation = {
            key: copy.deepcopy(value)
            for key, value in observation.items()
            if key
            in {
                "preferred_video",
                "video_a_interval_s",
                "video_b_interval_s",
                "measurement_method",
                "confidence",
                "order_consistent",
                "decision_rule",
                "routing_state",
                "query_id",
                "reference_id",
                "relationship",
                "copied_segments_frames",
                "localization_method",
            }
        }
        compact_store[evidence_id] = {
            "evidence_id": evidence_id,
            "active": True,
            "supports_fields": copy.deepcopy(atom.get("supports_fields") or []),
            "observation": compact_observation,
        }
    compact_claims = []
    for claim in state.get("candidate_claims") or []:
        compact_claims.append(
            {
                "claim_id": claim.get("claim_id"),
                "fields": {
                    field: {
                        key: copy.deepcopy(value)
                        for key, value in (field_state or {}).items()
                        if key in {"value", "evidence_status", "decision_status", "support_ids"}
                    }
                    for field, field_state in (claim.get("fields") or {}).items()
                },
            }
        )
    frontier = copy.deepcopy(payload.get("execution_frontier") or {})
    purposes = frontier.get("tool_purposes") or {}
    logical_eligible = list(
        frontier.get("eligible_tools") or state.get("eligible_tools") or []
    )
    latest_ids = list(
        (payload.get("latest_environment_observation") or {}).get(
            "created_evidence_ids"
        )
        or []
    )
    latest_id = next((item for item in latest_ids if item in compact_store), None)
    latest_atom = compact_store.get(latest_id or "") or {}
    route_atoms = [
        atom
        for atom in compact_store.values()
        if (atom.get("observation") or {}).get("measurement_method")
        == pinned_tool_name
    ]
    prior_route_atom = max(
        route_atoms,
        key=lambda atom: float((atom.get("observation") or {}).get("confidence") or 0.0),
        default=None,
    )
    latest_observation = latest_atom.get("observation") or {}
    prior_observation = (prior_route_atom or {}).get("observation") or {}
    pair_relation = "not_comparable"
    if latest_observation.get("preferred_video") is not None and prior_observation.get(
        "preferred_video"
    ) is not None:
        pair_relation = (
            f"same:{latest_observation.get('preferred_video')}"
            if latest_observation.get("preferred_video")
            == prior_observation.get("preferred_video")
            else "different"
        )
    route_scope = {
        "compare_uniform_timeline": (
            "coarse uniformly sampled chronological visual coverage; global-motion "
            "statistics and A/B order consistency remain unobserved"
        ),
        "compare_global_motion_symmetric": (
            "whole-video motion under both A/B role orders"
        ),
        "compare_action_refined_symmetric": (
            "predicate-relevant local behavior under both A/B role orders"
        ),
    }.get(pinned_tool_name, purposes.get(pinned_tool_name, ""))
    all_visible_uncertainties = (
        (frontier.get("measurement_gap_contract") or {}).get(
            "valid_observation_uncertainties"
        )
        or []
    )
    latest_visible_uncertainties = [
        item
        for item in all_visible_uncertainties
        if isinstance(item, dict)
        and str(item.get("evidence_id") or "") == str(latest_id or "")
    ]
    grounded_reasoning_plan = {
        "kind": "student_visible_grounded_measurement_reasoning_plan_v1",
        "latest_atom_exact_facts": (
            {
                "evidence_id": latest_id,
                "supports_fields": copy.deepcopy(latest_atom.get("supports_fields") or []),
                "preferred_video": latest_observation.get("preferred_video"),
                "confidence": latest_observation.get("confidence"),
                "order_consistent": latest_observation.get("order_consistent"),
                "measurement_method": latest_observation.get("measurement_method"),
                "decision_rule": latest_observation.get("decision_rule"),
            }
            if latest_id
            else None
        ),
        "recommended_route_prior_exact_facts": (
            {
                "evidence_id": (prior_route_atom or {}).get("evidence_id"),
                "supports_fields": copy.deepcopy(
                    (prior_route_atom or {}).get("supports_fields") or []
                ),
                "preferred_video": prior_observation.get("preferred_video"),
                "confidence": prior_observation.get("confidence"),
                "order_consistent": prior_observation.get("order_consistent"),
                "measurement_method": prior_observation.get("measurement_method"),
                "decision_rule": prior_observation.get("decision_rule"),
            }
            if prior_route_atom
            else None
        ),
        "preferred_video_relation": pair_relation,
        "preferred_video_relation_instruction": (
            "Explicitly say the two named atoms agree on preferred_video; do not call "
            "their relation cross-method disagreement."
            if pair_relation.startswith("same:")
            else "Describe only the exact relation above."
        ),
        "valid_primary_gap": latest_visible_uncertainties
        or all_visible_uncertainties,
        "required_tool": pinned_tool_name,
        "required_route_scope": route_scope,
        "forbidden_inferences": [
            "active supports_fields disappeared",
            "claim fields are unsupported merely because bindings are absent",
            "same preferred_video values conflict",
            "an unobserved order_consistent value is true or false",
            "a tool's required or recommended status is an observation gap",
        ],
        "composition_instruction": (
            "Write the rationale from these exact facts: first state the latest atom and "
            "its real observation uncertainty, then state the prior route relation, then "
            "name the required tool and its exact observable scope. Do not add claim-status "
            "or pairwise-conflict assertions not licensed here."
        ),
    }
    if (
        latest_observation.get("preferred_video") == "c"
        and latest_observation.get("order_consistent") is False
        and latest_observation.get("decision_rule")
        in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
    ):
        grounded_reasoning_plan["required_abstention_causal_explanation"] = (
            "State explicitly that this atom abstains because its direction disagrees across "
            "the two A/B presentation orders. Do not attribute the abstention directly to "
            "weak visual cues, routing state, or a tool recommendation."
        )
    if pinned_tool_name.startswith("localize_copy_segment_"):
        grounded_reasoning_plan.update(
            {
                "latest_atom_exact_facts": (
                    {
                        "evidence_id": latest_id,
                        "supports_fields": copy.deepcopy(
                            latest_atom.get("supports_fields") or []
                        ),
                        "observed_fields": copy.deepcopy(latest_observation),
                    }
                    if latest_id
                    else None
                ),
                "recommended_route_prior_exact_facts": None,
                "preferred_video_relation": "not_applicable_to_vcsl_localization",
                "preferred_video_relation_instruction": (
                    "Do not discuss preferred-video direction or A/B order consistency; "
                    "this task localizes a query/reference relationship."
                ),
                "valid_primary_gap": (
                    latest_visible_uncertainties
                    or all_visible_uncertainties
                    or [
                        {
                            "gap": "missing_query_reference_temporal_localization",
                            "needed_observation": (
                                "relationship and copied_segments_frames from the visible "
                                "query/reference similarity map"
                            ),
                        }
                    ]
                ),
                "required_route_scope": (
                    "local alignment over the query/reference feature-similarity map to "
                    "estimate relationship and copied temporal segment boundaries"
                ),
                "forbidden_inferences": [
                    "preferred_video",
                    "A/B presentation-order consistency",
                    "role-swapping symmetry",
                    "confidence when absent from the visible atom",
                    "a localization result before the localizer executes",
                ],
                "composition_instruction": (
                    "State only the identity fields actually present in the latest pair atom; "
                    "name the missing query/reference temporal localization as the decision "
                    "gap; then repeat the literal required localizer name in the prose (for "
                    "a DP route also say dynamic programming) and say prospectively that it "
                    "examines the similarity map to estimate relationship and segment "
                    "boundaries. Do not borrow VidDiff A/B comparison language."
                ),
            }
        )
    frontier.update(
        {
            "eligible_tools": logical_eligible,
            "required_tool_for_revision": pinned_tool_name,
            "tool_purposes": {
                pinned_tool_name: purposes.get(pinned_tool_name, "")
            },
            "native_action_templates": [
                item
                for item in frontier.get("native_action_templates") or []
                if item.get("tool_name") == pinned_tool_name
            ],
            "planning_instruction": (
                "Use the required recommended measurement, while acknowledging that the other "
                "listed routes remain logically eligible. Ground the rationale in one visible "
                "uncertainty from measurement_gap_contract, describe the measurement "
                "prospectively, and emit its native call. When naming an EvidenceAtom, copy its "
                "preferred_video, confidence, and order_consistent values exactly from the "
                "structured observation; never substitute null or missing. Do not use claim "
                "binding, support, open, or decision status as the measurement gap. Follow "
                "grounded_reasoning_plan exactly."
            ),
        }
    )
    return {
        "task": copy.deepcopy(payload.get("task") or {}),
        "current_harness_state": {
            "candidate_claims": compact_claims,
            "evidence_store": compact_store,
            "dropped_evidence_ids": copy.deepcopy(state.get("dropped_evidence_ids") or []),
            "eligible_tools": logical_eligible,
            "terminal": bool(state.get("terminal")),
            "policy_view_compaction": {
                "kind": "recommended_measurement_retry_view_v1",
                "omitted": "visual rationales, duplicated history, and unrelated tools",
                "authoritative_structured_facts_preserved": True,
            },
        },
        "latest_environment_observation": copy.deepcopy(
            payload.get("latest_environment_observation") or {}
        ),
        "execution_frontier": frontier,
        "grounded_reasoning_plan": grounded_reasoning_plan,
        "compression_contract": (
            "This retry view preserves the authoritative active EvidenceAtom facts, claim "
            "statuses, unique recommended tool, and native action surface needed for this turn."
        ),
    }


def _compact_progressive_decision_retry_payload(
    payload: dict[str, Any], pinned_tool_name: str
) -> dict[str, Any]:
    """Keep only the exact candidate facts needed to rewrite assess/adjudicate prose."""

    state = payload.get("current_harness_state") or {}
    assessment_frontier = copy.deepcopy(
        payload.get("evidence_assessment_frontier") or {}
    )
    adjudication_frontier = copy.deepcopy(
        payload.get("evidence_adjudication_frontier") or {}
    )
    relevant_ids: list[str] = []
    if pinned_tool_name == "assess_evidence_candidate":
        relevant_ids.extend(
            str(item.get("evidence_id") or "")
            for item in assessment_frontier.get("pending_candidate_assessments") or []
        )
    else:
        relevant_ids.extend(
            str(item.get("evidence_id") or "")
            for item in adjudication_frontier.get("selected_support") or []
        )
        relevant_ids.extend(
            str(item.get("evidence_id") or "")
            for item in adjudication_frontier.get("rejected_alternatives") or []
        )
    relevant_ids = list(dict.fromkeys(item for item in relevant_ids if item))

    compact_store: dict[str, Any] = {}
    for evidence_id in relevant_ids:
        atom = (state.get("evidence_store") or {}).get(evidence_id) or {}
        observation = atom.get("observation") or {}
        observation_keys = {
            "preferred_video",
            "video_a_interval_s",
            "video_b_interval_s",
            "measurement_method",
            "confidence",
            "order_consistent",
            "decision_rule",
            "routing_state",
            *(str(field) for field in atom.get("supports_fields") or []),
        }
        compact_store[evidence_id] = {
            "evidence_id": evidence_id,
            "active": atom.get("active"),
            "supports_fields": copy.deepcopy(atom.get("supports_fields") or []),
            "observation": {
                key: copy.deepcopy(observation.get(key))
                for key in observation_keys
                if key in observation
            },
        }

    compact_claims = []
    for claim in state.get("candidate_claims") or []:
        compact_claims.append(
            {
                "claim_id": claim.get("claim_id"),
                "fields": {
                    field: {
                        key: copy.deepcopy(field_state.get(key))
                        for key in ("value", "evidence_status", "decision_status")
                        if key in field_state
                    }
                    for field, field_state in (claim.get("fields") or {}).items()
                },
            }
        )

    if pinned_tool_name == "adjudicate_evidence":
        selected = copy.deepcopy(adjudication_frontier.get("selected_support") or [])
        required_grounding_clauses = [
            str(item.get("required_minimal_grounding_clause"))
            for item in selected
            if item.get("required_minimal_grounding_clause")
        ]
        required_compact_grounding_clauses = [
            str(item.get("required_compact_grounding_clause"))
            for item in selected
            if item.get("required_compact_grounding_clause")
        ]
        reasoning_plan: dict[str, Any] = {
            "kind": "student_visible_adjudication_reasoning_plan_v1",
            "selected_support": selected,
            "full_selected_support_facts": required_grounding_clauses,
            "selected_support_facts": required_compact_grounding_clauses,
            "rejected_partition": copy.deepcopy(
                adjudication_frontier.get("rejected_alternatives") or []
            ),
            "selected_support_instruction": (
                "Before any generic summary, express every fact in selected_support_facts "
                "directly as natural evidence reasoning. Exact copying is neither required nor "
                "preferred. Do not refer to fields, requirements, or clauses. "
                "Each reasoning_handle maps to one exact selected evidence_id in selected_support. "
                "A list of field names without a concrete value is insufficient."
            ),
            "ledger_instruction": (
                "Refer to the completed per-candidate assessment ledger for rejection reasons; "
                "the native call still carries the exact rejected IDs and defect codes."
            ),
            "forbidden_inferences": [
                "Never change order_consistent=null into true or false.",
                "Never omit a selected evidence_id from the natural-language reasoning.",
                "Do not describe adjudication as binding or new visual measurement.",
            ],
        }
    else:
        pending = copy.deepcopy(
            assessment_frontier.get("pending_candidate_assessments") or []
        )
        required_candidate = pending[0] if pending else None
        measurement_semantics_required = bool(
            (required_candidate or {}).get("measurement_semantics_required")
        )
        reasoning_plan = {
            "kind": "student_visible_candidate_assessment_reasoning_plan_v1",
            "required_candidate": required_candidate,
            "required_decision_gap_clause": (
                (required_candidate or {}).get("required_decision_gap_clause")
            ),
            "required_abstention_causal_clause": (
                (required_candidate or {}).get("required_abstention_causal_clause")
            ),
            "assessment_instruction": (
                (
                    "Cite the exact evidence_id, state preferred_video and order_consistent "
                    "exactly, then state the visible retain/reject disposition and a validated "
                    "defect when rejecting. If preferred_video=c and order_consistent=false "
                    "under an order-disagreement abstention rule, explain the direct A/B-order "
                    "cause in natural language before explaining the rejection."
                )
                if measurement_semantics_required
                else (
                    "Cite the exact evidence_id and its supported_observation_facts, then state "
                    "the visible retain/reject disposition. This is an identity/context atom: "
                    "do not invent or discuss absent preferred_video or order_consistent fields."
                )
            ),
            "forbidden_inferences": [
                "Do not merely repeat a decision_rule token as an explanation.",
                "Do not assess any deferred candidate.",
                "Do not claim this call carries the full adjudication partition.",
            ],
        }

    frontier = copy.deepcopy(payload.get("execution_frontier") or {})
    frontier["eligible_tools"] = [pinned_tool_name]
    return {
        "task": copy.deepcopy(payload.get("task") or {}),
        "current_harness_state": {
            "candidate_claims": compact_claims,
            "evidence_store": compact_store,
            "evidence_assessments": copy.deepcopy(
                state.get("evidence_assessments") or {}
            ),
            "eligible_tools": [pinned_tool_name],
            "evidence_assessment": copy.deepcopy(state.get("evidence_assessment") or {}),
            "evidence_adjudication": copy.deepcopy(
                state.get("evidence_adjudication") or {}
            ),
            "policy_view_compaction": {
                "kind": "progressive_decision_retry_view_v1",
                "omitted": "visual rationales, duplicated history, unrelated atoms and tools",
                "authoritative_structured_facts_preserved": True,
            },
        },
        "execution_frontier": frontier,
        "evidence_assessment_frontier": assessment_frontier,
        "evidence_adjudication_frontier": adjudication_frontier,
        "grounded_reasoning_plan": reasoning_plan,
        "compression_contract": (
            "This retry-only view preserves the exact claim values, candidate facts, native "
            "decision frontier, and ledger information required for the same unchanged turn."
        ),
    }


def _initial_progressive_decision_message(
    user_message: dict[str, Any], tool_name: str
) -> dict[str, Any]:
    """Use the decision-sufficient view on the first unique assess/adjudicate attempt."""

    revised = copy.deepcopy(user_message)
    payload = json.loads(str(revised.get("content", "")))
    compact = _compact_progressive_decision_retry_payload(payload, tool_name)
    reasoning_plan = compact.get("grounded_reasoning_plan") or {}
    initial_view: dict[str, Any] = {
        "state_unchanged": True,
        "unique_eligible_tool": tool_name,
        "instruction": (
            "Generate grounded natural-language reasoning from grounded_reasoning_plan, then "
            "emit the unique eligible native tool call."
        ),
    }
    if tool_name == "adjudicate_evidence":
        frontier = compact.get("evidence_adjudication_frontier") or {}
        initial_view.update(
            {
                "decision_gap": frontier.get("decision_gap"),
                "selected_support_facts_to_state": copy.deepcopy(
                    reasoning_plan.get("selected_support_facts")
                    or reasoning_plan.get("full_selected_support_facts")
                    or []
                ),
                "ledger_rule": (
                    "Say the completed assessment ledger supplies validated rejection reasons."
                ),
            }
        )
        compact["evidence_adjudication_frontier"] = {
            "decision_gap": frontier.get("decision_gap"),
            "rejected_alternatives_externalized_in_complete_ledger": frontier.get(
                "rejected_alternatives_externalized_in_complete_ledger"
            ),
        }
    else:
        initial_view["required_candidate"] = copy.deepcopy(
            reasoning_plan.get("required_candidate")
        )
        required_facts = [
            clause
            for clause in (
                reasoning_plan.get("required_decision_gap_clause"),
                reasoning_plan.get("required_abstention_causal_clause"),
            )
            if clause
        ]
        if required_facts:
            initial_view["facts_to_express_naturally_before_tool_call"] = required_facts
    compact["initial_decision_view"] = {
        **initial_view,
    }
    revised["content"] = json.dumps(
        compact, ensure_ascii=False, separators=(",", ":")
    )
    return revised


def _generation_hit_limit_problem(
    generated_tokens: int,
    token_limit: int,
    *,
    raw_call_parseable: bool,
    arguments_materialized: bool,
) -> str | None:
    """A complete native call is not truncated merely because it ends at the cap."""
    if (
        generated_tokens >= token_limit
        and not raw_call_parseable
        and not arguments_materialized
    ):
        return "generation_hit_limit"
    return None


def _attempt_user_message(
    user_message: dict[str, Any],
    *,
    pinned_tool_name: str | None,
    retry_feedback: str,
) -> dict[str, Any]:
    """Append pre-execution revision guidance where tail truncation preserves it.

    The accepted SFT datum still uses the untouched authoritative user message. This
    extra field exists only while regenerating a rejected draft from the unchanged
    visible state.
    """

    revised = copy.deepcopy(user_message)
    if not retry_feedback:
        return revised
    try:
        payload = json.loads(str(revised.get("content", "")))
    except json.JSONDecodeError:
        payload = {"authoritative_user_content": revised.get("content", "")}
    if (
        pinned_tool_name
        and pinned_tool_name.startswith(("compare_", "detect_", "localize_"))
    ):
        payload = _compact_recommended_measurement_retry_payload(
            payload, pinned_tool_name
        )
    elif pinned_tool_name in {
        "assess_evidence_candidate",
        "adjudicate_evidence",
    }:
        payload = _compact_progressive_decision_retry_payload(
            payload, pinned_tool_name
        )
    revision: dict[str, Any] = {
        "kind": "preexecution_revision_v1",
        "state_unchanged": True,
        "requirements": retry_feedback,
        "instruction": (
            "Regenerate this same turn from the authoritative visible state. Emit "
            "2-4 grounded sentences and one complete native tool call. When a tool is "
            "locked, use 45-65 words total and emit the tool call immediately afterward. "
            "Do not mention "
            "drafts, retries, revision feedback, or generation audits in the reasoning."
        ),
    }
    if pinned_tool_name:
        revision.update(
            {
                "required_tool": pinned_tool_name,
                "tool_alignment_instruction": (
                    f"Keep exactly `{pinned_tool_name}`. Literally name it in the "
                    "reasoning and explain its method-specific fit to the visible gap; "
                    "do not present another tool as the next action."
                ),
            }
        )
    payload["preexecution_revision"] = revision
    revised["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return revised


def _execution_frontier(
    eligible: list[str],
    tools: list[dict[str, Any]],
    executable_actions: list[dict[str, Any]],
    visible_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact, tail-visible state-to-action interface for planning."""

    descriptions = {
        str((tool.get("function") or {}).get("name")): str(
            (tool.get("function") or {}).get("description", "")
        )[:320]
        for tool in tools
        if str((tool.get("function") or {}).get("name")) in eligible
    }
    const_schema_frontier = eligible in (
        ["assess_evidence_candidate"],
        ["adjudicate_evidence"],
    )
    frontier = {
        "kind": "eligible_execution_frontier_v1",
        "eligible_tools": list(eligible),
        "tool_purposes": descriptions,
        "native_action_templates": (
            [] if const_schema_frontier else copy.deepcopy(executable_actions)
        ),
        "planning_instruction": (
            "Plan only from this current executable frontier. Select exactly one eligible "
            "tool, literally name it in the reasoning before the tool call, and explain why its specific method "
            "or state transition closes the visible blocking gap. Keep the natural-language "
            "plan and emitted tool call aligned. If multiple tools are listed, describe the "
            "selection as the best next choice, never as the only eligible action. When choosing "
            "a measurement tool, name the unresolved observation gap that requires new sensing; "
            "never claim that binding, committing, or stopping is the only remaining step. "
            "Describe a not-yet-executed measurement prospectively: say what it will test or "
            "whether a result is stable, and never promise an order-consistent outcome."
        ),
    }
    if const_schema_frontier:
        frontier["native_action_templates_reference"] = (
            "The eligible tool schema contains the exact const argument options; "
            "the matching evidence frontier contains the visible facts needed to justify them."
        )
    measurement_gap_contract = _measurement_gap_contract(visible_state or {}, eligible)
    if measurement_gap_contract is not None:
        frontier["measurement_gap_contract"] = measurement_gap_contract
    state_transition_contract = _state_transition_language_contract(eligible)
    if state_transition_contract is not None:
        frontier["state_transition_language_contract"] = state_transition_contract
    return frontier


def _state_transition_language_contract(
    eligible: list[str],
) -> dict[str, Any] | None:
    """Externalize precise language for state-only post-adjudication gaps."""

    if eligible == ["bind_evidence"]:
        return {
            "transition": "supported_unbound_to_bound",
            "required_gap_clause": (
                "The selected evidence supports these claim fields, but the fields are "
                "not yet bound to those EvidenceAtom IDs."
            ),
            "prohibited_descriptions": [
                "the bindable fields are unsupported",
                "binding completes the candidate-assessment ledger",
                "binding commits the claim fields",
            ],
        }
    if eligible == ["set_field_decision"]:
        return {
            "transition": "supported_bound_open_to_committed",
            "required_gap_clause": (
                "The evidence is already bound and supports these fields, but their "
                "field decisions are not yet committed."
            ),
            "prohibited_descriptions": [
                "the fields are unsupported or unbound",
                "committing creates new evidence",
            ],
        }
    if eligible == ["stop"]:
        return {
            "transition": "complete_committed_state_to_terminal",
            "required_gap_clause": (
                "All required fields are supported and committed and no high-priority "
                "obligation remains; only the terminal stop decision is unrecorded."
            ),
            "prohibited_descriptions": [
                "additional evidence, binding, or field commitment is still required",
            ],
        }
    return None


def _measurement_gap_contract(
    visible_state: dict[str, Any], eligible: list[str]
) -> dict[str, Any] | None:
    if not eligible or not all(
        name.startswith(("compare_", "detect_", "localize_")) for name in eligible
    ):
        return None
    uncertainties: list[dict[str, Any]] = []
    for evidence_id, atom in (visible_state.get("evidence_store") or {}).items():
        if atom.get("active") is not True:
            continue
        observation = atom.get("observation") or {}
        method = observation.get("measurement_method")
        if not method:
            continue
        reasons: list[str] = []
        confidence = observation.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < 0.65:
            reasons.append(f"low_confidence={confidence}")
        if observation.get("order_consistent") is False:
            reasons.append("order_inconsistent")
        if observation.get("preferred_video") == "c":
            reasons.append("abstaining_direction")
        routing = observation.get("routing_state") or {}
        if routing.get("cross_method_disagreement") is True:
            reasons.append("cross_method_disagreement")
        if reasons:
            uncertainties.append(
                {
                    "evidence_id": evidence_id,
                    "measurement_method": method,
                    "visible_uncertainties": reasons,
                }
            )
    if not uncertainties:
        return None
    contract = {
        "kind": "student_visible_measurement_gap_contract_v1",
        "valid_observation_uncertainties": uncertainties,
        "required_reasoning_rule": (
            "Justify new sensing with at least one listed visible uncertainty or the need for "
            "independent cross-method corroboration."
        ),
        "prohibited_gap_frames": [
            "claim fields are unbound",
            "claim fields are uncommitted",
            "claim fields are open or undecided when active measurement uncertainty is visible",
            "claim fields are unsupported despite an active atom declaring supports_fields",
            "no active or replacement evidence exists when a new active ID is visible",
        ],
    }
    latest_ids = [
        str(item)
        for item in (visible_state.get("latest_environment_observation") or {}).get(
            "created_evidence_ids"
        )
        or []
    ]
    latest_atom = next(
        (
            (visible_state.get("evidence_store") or {}).get(evidence_id)
            for evidence_id in latest_ids
            if (visible_state.get("evidence_store") or {}).get(evidence_id)
        ),
        None,
    )
    latest_observation = (latest_atom or {}).get("observation") or {}
    latest_direction = latest_observation.get("preferred_video")
    latest_method = latest_observation.get("measurement_method")
    if (
        latest_direction == "c"
        and latest_observation.get("order_consistent") is False
        and latest_observation.get("decision_rule")
        in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
    ):
        contract["required_latest_atom_causal_explanation"] = {
            "preferred_video": "c",
            "order_consistent": False,
            "decision_rule": latest_observation.get("decision_rule"),
            "required_natural_language": (
                "State explicitly that the latest atom abstains because the comparison "
                "direction disagrees across the two A/B presentation orders. Cross-method "
                "disagreement, visual strength, and routing state are not the direct cause."
            ),
        }
    if latest_direction in {"a", "b"} and latest_method:
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for tool_name in eligible:
            if tool_name == latest_method:
                continue
            route_atoms = [
                atom
                for atom in (visible_state.get("evidence_store") or {}).values()
                if atom.get("active") is True
                and (atom.get("observation") or {}).get("measurement_method") == tool_name
            ]
            if not route_atoms:
                continue
            best_atom = max(
                route_atoms,
                key=lambda atom: float((atom.get("observation") or {}).get("confidence") or 0.0),
            )
            observation = best_atom.get("observation") or {}
            score = float(observation.get("confidence") or 0.0)
            if observation.get("preferred_video") == latest_direction:
                score += 3.0
            if observation.get("preferred_video") in {"a", "b"}:
                score += 1.0
            if observation.get("order_consistent") is not False:
                score += 1.0
            ranked.append(
                (
                    score,
                    tool_name,
                    {
                        "prior_visible_direction": observation.get("preferred_video"),
                        "prior_visible_confidence": observation.get("confidence"),
                        "prior_visible_order_consistent": observation.get(
                            "order_consistent"
                        ),
                    },
                )
            )
        if ranked:
            ranked.sort(key=lambda item: (-item[0], item[1]))
            best_score, best_tool, best_facts = ranked[0]
            if len(ranked) == 1 or best_score > ranked[1][0]:
                contract["recommended_next_measurement"] = {
                    "tool_name": best_tool,
                    "basis": (
                        "Among eligible complementary routes, its prior active student-visible "
                        "observation best matches the latest non-abstaining direction without an "
                        "observed order-inconsistency defect. Execute it for fresh independent "
                        "corroboration; this recommendation uses no hidden label."
                    ),
                    **best_facts,
                }
    return contract


def _turn_generation_budget(eligible: list[str], global_cap: int) -> int:
    """Allocate output budget by the largest eligible native action envelope."""

    names = set(eligible)
    if "adjudicate_evidence" in names:
        return min(global_cap, 448)
    if "assess_evidence_candidate" in names:
        return min(global_cap, 320)
    if names.intersection({"create_candidate_claim", "bind_evidence"}):
        return min(global_cap, 320)
    if names and names.issubset({"set_field_decision", "stop"}):
        return min(global_cap, 192)
    return min(global_cap, 320)


def _progressive_turn_generation_budget(
    eligible: list[str],
    global_cap: int,
    executable_actions: list[dict[str, Any]] | None = None,
) -> int:
    """Keep state actions compact while allowing a complete measured-observation rationale."""

    names = set(eligible)
    if "adjudicate_evidence" in names:
        adjudication = next(
            (
                item.get("arguments") or {}
                for item in (executable_actions or [])
                if item.get("tool_name") == "adjudicate_evidence"
            ),
            {},
        )
        competitor_count = len(adjudication.get("selected_evidence_ids") or []) + len(
            adjudication.get("rejected_evidence") or []
        )
        # Three-way adjudication fits the compact 160-token envelope. A fourth
        # competitor needs enough room for two selected field=value clauses plus
        # an ID-scoped rejection clause; native arguments remain materializable.
        return min(global_cap, 160 + 80 * max(0, competitor_count - 3))
    if names == {"assess_evidence_candidate"}:
        return min(global_cap, 176)
    if names == {"create_candidate_claim"}:
        return min(global_cap, 320)
    if names == {"stop"}:
        return min(global_cap, 128)
    compact_native_actions = {
        "assess_evidence_candidate",
        "create_candidate_claim",
        "bind_evidence",
        "set_field_decision",
        "stop",
    }
    if names and names.issubset(compact_native_actions):
        return min(global_cap, 144)
    return min(global_cap, 192)


def _use_post_adjudication_system(
    eligible: list[str],
    visible_state: dict[str, Any],
    *,
    require_progressive_evidence_assessment: bool,
) -> bool:
    if not require_progressive_evidence_assessment:
        return False
    if not (visible_state.get("evidence_adjudications") or {}):
        return False
    return bool(eligible) and set(eligible).issubset(
        {"bind_evidence", "set_field_decision", "stop"}
    )


def _use_progressive_measurement_system(
    eligible: list[str], *, require_progressive_evidence_assessment: bool
) -> bool:
    if not require_progressive_evidence_assessment or not eligible:
        return False
    return all(
        name.startswith(("compare_", "detect_", "localize_")) for name in eligible
    )


def _use_progressive_vcsl_localization_system(
    eligible: list[str], *, require_progressive_evidence_assessment: bool
) -> bool:
    if not require_progressive_evidence_assessment or not eligible:
        return False
    return all(name.startswith("localize_copy_segment_") for name in eligible)


def _generated_tool_name(text: str) -> str | None:
    match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _semantic_tool_name(text: str, eligible: list[str]) -> str | None:
    mentioned = [
        name
        for name in eligible
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    ]
    return mentioned[0] if len(mentioned) == 1 else None


def _visible_field_value(state: dict[str, Any], field_name: str) -> Any:
    values = []
    for claim in state.get("candidate_claims") or []:
        field = (claim.get("fields") or {}).get(field_name) or {}
        value = field.get("value")
        if value is not None and value not in values:
            values.append(value)
    return values[0] if len(values) == 1 else None


def _materialize_visible_action_arguments(
    name: str,
    *,
    state: dict[str, Any],
    tools: list[dict[str, Any]],
    executable_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve repetitive native arguments only when visible state makes them unique."""

    matching_actions = [
        action
        for action in executable_actions
        if action.get("tool_name") == name and isinstance(action.get("arguments"), dict)
    ]
    if len(matching_actions) == 1:
        return copy.deepcopy(matching_actions[0]["arguments"])

    matching_tools = [
        tool for tool in tools if (tool.get("function") or {}).get("name") == name
    ]
    if len(matching_tools) != 1:
        return None
    parameters = (matching_tools[0].get("function") or {}).get("parameters") or {}
    properties = parameters.get("properties") or {}
    arguments: dict[str, Any] = {}
    for field_name in parameters.get("required") or []:
        property_schema = properties.get(field_name) or {}
        if "const" in property_schema:
            arguments[field_name] = copy.deepcopy(property_schema["const"])
            continue
        enum = list(property_schema.get("enum") or [])
        if len(enum) == 1:
            arguments[field_name] = copy.deepcopy(enum[0])
            continue
        visible_value = _visible_field_value(state, str(field_name))
        if visible_value is None:
            return None
        arguments[field_name] = copy.deepcopy(visible_value)
    return arguments if _schema_valid(matching_tools, name, arguments) else None


def _evidence_consumption_frontier(
    state: dict[str, Any], latest_observation: dict[str, Any]
) -> dict[str, Any] | None:
    """Externalize newly created measurement debt using student-visible facts only."""

    evidence_store = state.get("evidence_store") or {}
    summaries: list[dict[str, Any]] = []
    for evidence_id in latest_observation.get("created_evidence_ids") or []:
        atom = evidence_store.get(str(evidence_id)) or {}
        observation = atom.get("observation") or {}
        method = observation.get("measurement_method") or observation.get(
            "localization_method"
        )
        if not method:
            continue
        summary = {
            "evidence_id": str(evidence_id),
            "active": atom.get("active"),
            "method": method,
        }
        for key in (
            "preferred_video",
            "confidence",
            "order_consistent",
            "decision_rule",
            "video_a_interval_s",
            "video_b_interval_s",
        ):
            if key in observation:
                summary[key] = copy.deepcopy(observation[key])
        visible_rationale = str(observation.get("visible_rationale") or "").strip()
        if visible_rationale:
            summary["visible_rationale"] = visible_rationale[:520]
        summaries.append(summary)
    if not summaries:
        return None
    frontier = {
        "kind": "new_measurement_consumption_frontier_v1",
        "new_measurement_atoms": summaries,
        "current_turn_requirement": (
            "Consume every listed new atom in the next decision: either include its exact "
            "EvidenceAtom ID in the native support_ids, or cite that exact ID in reasoning "
            "and name a concrete visible defect that justifies measuring again or selecting "
            "different support. Do not silently omit a newly created atom."
        ),
    }
    active_created_ids = [
        str(item["evidence_id"])
        for item in summaries
        if item.get("active") is True
    ]
    dropped_ids = [str(item) for item in state.get("dropped_evidence_ids") or []]
    if active_created_ids and dropped_ids:
        frontier["evidence_identity_guard"] = {
            "newly_created_active_ids": active_created_ids,
            "distinct_dropped_tombstone_ids": dropped_ids,
            "rule": (
                "Evidence status is ID-scoped and never transfers between similarly named IDs. "
                "Every newly_created_active_id is active and is not any dropped tombstone; do not "
                "describe it as dropped, inactive, invalidated, unavailable, or a replacement that "
                "still needs to be recreated."
            ),
        }
    return frontier


def _eligible_tool_claim_problems(
    reasoning: str, visible_state: dict[str, Any]
) -> list[str]:
    eligible = [str(item) for item in visible_state.get("eligible_tools") or []]
    if len(eligible) <= 1:
        return []
    if re.search(
        r"\b(?:the\s+)?only\s+(?:remaining\s+)?eligible\s+"
        r"(?:(?:native|measurement)\s+)?(?:tool|action|method|route)\b",
        reasoning,
        re.I,
    ):
        return [f"eligible_tool_count_mismatch:said_only_one!={len(eligible)}"]
    return []


def _minimum_reasoning_words(name: str) -> int:
    """Keep deterministic state transitions concise while preserving strict checks."""
    if name == "stop":
        return 25
    if name == "set_field_decision":
        return 35
    if name == "assess_evidence_candidate":
        return 35
    if name.startswith(("compare_", "detect_", "localize_")):
        return 40
    return 45


def _repair_typed_order_abstention_causal_edge(
    reasoning: str,
    name: str | None,
    arguments: dict[str, Any] | None,
    visible_state: dict[str, Any],
) -> tuple[str, list[str]]:
    """Repair one known reversed causal phrase using only visible typed facts.

    This does not invent evidence or tool arguments. It is a narrow surface realization
    repair for an atom whose visible decision rule already fixes the directed edge:
    disagreement across A/B presentation orders -> preferred_video=c abstention.
    """

    if name != "assess_evidence_candidate" or not arguments:
        return reasoning, []
    evidence_id = str(arguments.get("evidence_id") or "")
    atom = (visible_state.get("evidence_store") or {}).get(evidence_id) or {}
    observation = atom.get("observation") or {}
    if not (
        observation.get("preferred_video") == "c"
        and observation.get("order_consistent") is False
        and observation.get("decision_rule")
        in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
    ):
        return reasoning, []

    repaired, count = re.subn(
        r"(?:the\s+)?order[-_ ]disagreement\s+abstention\s+rule\s+"
        r"(?:directly\s+)?(?:makes|causes)\s+(?:its|the)\s+"
        r"(?:A\s*/\s*B\s+)?direction\s+(?:to\s+be\s+)?unstable",
        (
            "its direction disagrees across the two A/B presentation orders, so the atom "
            "abstains with preferred_video=c"
        ),
        reasoning,
        flags=re.I,
    )
    if not count:
        return reasoning, []
    return repaired, ["typed_order_abstention_causal_edge_repaired"]


def _preexecution_reasoning_problems(
    reasoning: str,
    name: str | None,
    arguments: dict[str, Any] | None,
    *,
    meta_visible: bool,
    visible_state: dict[str, Any] | None = None,
) -> list[str]:
    if not name:
        return []
    problems: list[str] = []
    words = re.findall(r"\b\w+[\w'-]*\b", reasoning)
    minimum_words = _minimum_reasoning_words(name)
    if not minimum_words <= len(words) <= 120:
        problems.append(f"reasoning_word_count:{len(words)}")
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", reasoning))
    if not 2 <= sentence_count <= 4:
        problems.append(f"reasoning_sentence_count:{sentence_count}")
    if not re.search(r"[.!?][\"'`)\]]?\s*$", reasoning):
        problems.append("reasoning_missing_terminal_punctuation")
    measurement_uncertainty_named = name.startswith(
        ("compare_", "detect_", "localize_")
    ) and re.search(
        r"low[-_ ]confidence|confidence\s*(?::|=|is|of|\()?\s*0?\.[0-6]|"
        r"uncertain|disagree|inconsisten|abstain|independent\s+(?:check|evidence|"
        r"measurement|corroboration)|cross[-_ ]method|occlud|sparse|limitation",
        reasoning,
        re.I,
    )
    names_decision_gap = bool(
        re.search(
            r"gap|block|remaining|missing|not\s+yet|still\s+(?:open|unbound|"
            r"unsupported|uncommitted)|no observation|no available",
            reasoning,
            re.I,
        )
        or (
            name == "stop"
            and re.search(r"terminal\s+(?:validation|finalization)", reasoning, re.I)
        )
    )
    assessment_decision_named = bool(
        name == "assess_evidence_candidate"
        and arguments
        and str(arguments.get("evidence_id") or "") in reasoning
        and re.search(r"\b(?:retain|retained|reject|rejected)\b", reasoning, re.I)
    )
    if (
        not names_decision_gap
        and not measurement_uncertainty_named
        and not assessment_decision_named
    ):
        problems.append("reasoning_does_not_name_decision_gap")
    if re.search(r"\boracle\b|\bverifier\b|teacher target", reasoning, re.I):
        problems.append("reasoning_mentions_hidden_evaluation_information")
    chosen_tool_problem = _chosen_measurement_tool_problem(name, reasoning)
    if chosen_tool_problem:
        problems.append(chosen_tool_problem)
    route_semantic_problem = _measurement_route_semantic_problem(name, reasoning)
    if route_semantic_problem:
        problems.append(route_semantic_problem)
    restoration_choice_problem = _repair_restoration_choice_problem(
        name, visible_state or {}
    )
    if restoration_choice_problem:
        problems.append(restoration_choice_problem)
    observation_gap_problem = _measurement_observation_gap_problem(
        name, reasoning, visible_state or {}
    )
    if observation_gap_problem:
        problems.append(observation_gap_problem)
    recommendation_problem = _measurement_recommendation_problem(
        name, visible_state or {}
    )
    if recommendation_problem:
        problems.append(recommendation_problem)
    measurement_need_problem = _measurement_need_problem(name, reasoning)
    if measurement_need_problem:
        problems.append(measurement_need_problem)
    outcome_presupposition_problem = _measurement_outcome_presupposition_problem(
        name, reasoning
    )
    if outcome_presupposition_problem:
        problems.append(outcome_presupposition_problem)
    bind_scope_problem = _bind_evidence_reasoning_problem(name, arguments, reasoning)
    if bind_scope_problem:
        problems.append(bind_scope_problem)
    latest_measurement_problem = _bind_latest_measurement_problem(
        name, arguments, reasoning, visible_state or {}
    )
    if latest_measurement_problem:
        problems.append(latest_measurement_problem)
    creation_source_problem = _claim_creation_source_problem(
        name, arguments, reasoning, visible_state or {}
    )
    if creation_source_problem:
        problems.append(creation_source_problem)
    typed_quantifier_problem = _typed_universal_claim_choice_problem(
        name, arguments, visible_state or {}
    )
    if typed_quantifier_problem:
        problems.append(typed_quantifier_problem)
    problems.extend(
        _adjudication_reasoning_problems(
            name, arguments, reasoning, visible_state=visible_state or {}
        )
    )
    problems.extend(
        _evidence_assessment_reasoning_problems(
            name, arguments, reasoning, visible_state=visible_state or {}
        )
    )
    problems.extend(
        _reasoning_process_artifact_problems(name, reasoning, visible_state or {})
    )
    # Candidate assessment has its own ID-scoped fact checker above. Applying the generic
    # latest-created-atom checker here confuses the candidate under assessment with a newer
    # derived atom still visible in the execution history.
    if name != "assess_evidence_candidate":
        problems.extend(_visible_observation_fact_problems(reasoning, visible_state or {}))
    problems.extend(_eligible_tool_claim_problems(reasoning, visible_state or {}))

    if name == META_ABSTENTION_METHOD:
        if not re.search(r"aggregat|combin|already executed|prior (?:route|measurement)", reasoning, re.I):
            problems.append("meta_not_described_as_aggregation")
        if not re.search(
            r"(?:without|does not|doesn't|cannot|no new).{0,45}(?:inspect|pixel|frame|image|visual|sens)",
            reasoning,
            re.I,
        ):
            problems.append("meta_omits_no_new_sensing_contract")
        source_pair_named = (
            re.search(r"global[-_ ]motion", reasoning, re.I)
            and re.search(r"action[-_ ]refined", reasoning, re.I)
        ) or re.search(
            r"both (?:fixed )?(?:symmetric )?(?:source )?routes|"
            r"(?:two|both) (?:already executed )?symmetric (?:observations|measurements|routes)|"
            r"already executed symmetric (?:observations|measurements|routes)",
            reasoning,
            re.I,
        )
        if not source_pair_named:
            problems.append("meta_source_pair_not_named")

    if meta_visible:
        meta_reference = (
            r"(?:EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-\d{3}"
            r"|meta[-_ ](?:atom|evidence|abstention|result))"
        )
        false_order = (
            r"(?:order[-_ ]inconsisten\w*|order[-_ ]consisten(?:t|cy)"
            r"\s*[`*]*\s*(?:(?::|=|is|was|of)\s*)?[`*]*\s*false)"
        )
        if re.search(rf"{meta_reference}.{{0,220}}{false_order}", reasoning, re.I):
            problems.append("role_invariant_meta_described_as_order_inconsistent")
        all_fields = (
            r"(?:all\s+(?:six|6)\s+(?:required\s+)?(?:claim\s+)?fields"
            r"|the\s+(?:six|6)\s+required\s+(?:claim\s+)?fields"
            r"|all\s+required\s+(?:claim\s+)?fields|every\s+(?:claim\s+)?field)"
        )
        task_identity = (
            r"(?:EA-VIDDIFF-(?:ADAPTIVE-)?TASK-\d{3}"
            r"|(?:the\s+)?task[-_ ](?:identity[-_ ])?(?:atom|evidence))"
        )
        direct_meta_scope = re.search(
            rf"(?:{all_fields}.{{0,35}}\b(?:via|from|by|through)\b\s*(?:the\s+)?{meta_reference}"
            rf"|{meta_reference}.{{0,80}}\b(?:supports?|covers?|handles?)\b.{{0,50}}{all_fields}"
            rf"|{meta_reference}.{{0,100}}\b(?:support|evidence)\s+for\s+{all_fields})",
            reasoning,
            re.I,
        )
        if (
            re.search(meta_reference, reasoning, re.I)
            and re.search(all_fields, reasoning, re.I)
            and (direct_meta_scope or not re.search(task_identity, reasoning, re.I))
        ):
            problems.append("meta_evidence_overclaims_identity_field_support")
        if name == "create_candidate_claim":
            selected_method = str(
                ((arguments or {}).get("field_values") or {}).get(
                    "measurement_method", ""
                )
            )
            explicit_rejection = re.search(
                r"(?:reject|weaker|defect|not suitable|not reliable|cannot use).{0,80}"
                r"(?:meta|symmetry[-_ ]residual)",
                reasoning,
                re.I,
            ) or re.search(
                r"(?:meta|symmetry[-_ ]residual).{0,80}"
                r"(?:reject|weaker|defect|not suitable|not reliable|cannot use)",
                reasoning,
                re.I,
            )
            if selected_method != META_ABSTENTION_METHOD and not explicit_rejection:
                problems.append("meta_reasoning_and_claim_arguments_disagree")
    return problems


def _chosen_measurement_tool_problem(name: str, reasoning: str) -> str | None:
    if not name.startswith(("compare_", "detect_", "localize_")):
        return None
    cues = {
        "compare_action_refined_symmetric": (
            r"(?:action[-_ ]refined|refined[-_ ]action|fine[-_ ]grained action|"
            r"action[-_ ]focused (?:local )?strips?|dense (?:action[-_ ]focused )?local strips?)"
        ),
        "compare_uniform_timeline": (
            r"(?:uniform[-_ ]timeline|uniform(?:ly)? sampled (?:chronological )?"
            r"(?:timeline|mosaics?|frames?)|coarse (?:whole[-_ ]video )?chronological)"
        ),
        "compare_global_motion_symmetric": (
            r"(?:global[-_ ]motion|whole[-_ ]video motion strips?|"
            r"motion strips?.{0,35}(?:both|two) A/B (?:orders?|roles?))"
        ),
        "compare_symmetry_residual_abstention": r"(?:symmetry[-_ ]residual|residual abstention|meta[-_ ]aggregation)",
        "compare_predicate_independent": r"predicate[-_ ]independent",
        "compare_spatial_regions": r"spatial[-_ ]regions?",
        "compare_audio_tracks": r"audio[-_ ]tracks?",
        "detect_temporal_edit": r"temporal[-_ ]edit",
        "localize_copy_segment_spd_official": r"(?:official\s+SPD|SPD[-_ ]OFFICIAL)",
        "localize_copy_segment_dp": r"(?:\bDP\b|dynamic[-_ ]programming)",
        "localize_copy_segment_local_alignment": r"local[-_ ]alignment",
        "localize_copy_segment_temporal_stability": r"(?:temporal[-_ ]stability|TN[-_ ]STABILITY)",
        "localize_copy_segment_axis_saliency": r"axis[-_ ]saliency",
        "localize_copy_segment_hv_segment": r"HV[-_ ]SEGMENT",
        "localize_copy_segment_temporal_network": r"temporal[-_ ]network",
        "localize_copy_segment": r"(?:copy[-_ ]segment|localization)",
    }
    pattern = rf"(?:{re.escape(name)}|{cues.get(name, re.escape(name))})"
    if re.search(pattern, reasoning, re.I):
        return None
    return f"chosen_measurement_tool_not_justified:{name}"


def _measurement_route_semantic_problem(name: str, reasoning: str) -> str | None:
    """Reject method-purpose claims that exceed a route's observable output."""

    if name.startswith("localize_copy_segment_") and re.search(
        r"(?:under|across|in)\s+both\s+A/B\s+(?:orders|roles)|"
        r"\brole[- ]swapping\b|\bA/B\s+presentation[- ]order",
        reasoning,
        re.I,
    ):
        return "vcsl_localizer_overclaims_ab_order_symmetry"
    if name != "compare_uniform_timeline":
        return None
    claims_order_test = re.search(
        r"(?:can|will|would|to|which).{0,55}"
        r"(?:resolve|test|check|determine|establish|verify).{0,45}"
        r"order[-_ ]consisten",
        reasoning,
        re.I,
    ) or re.search(
        r"\b(?:test|check|resolve|determine)\b.{0,85}"
        r"(?:under|in)\s+both\s+A/B\s+(?:orders|roles)|"
        r"\b(?:test|check|resolve|determine)\b.{0,70}role\s+swapping",
        reasoning,
        re.I,
    )
    claims_global_motion_test = re.search(
        r"\b(?:test|measure|inspect|analy[sz]e|compare)\b.{0,70}"
        r"(?:global[-_ ]motion|whole[-_ ]video motion)|"
        r"same global[-_ ]motion comparison|"
        r"(?:faster[-_ ]movement|motion)[^.!?]{0,55}"
        r"(?:across\s+(?:the\s+)?(?:whole|entire)\s+(?:scope|video)|stop[-_ ]and[-_ ]go)",
        reasoning,
        re.I,
    )
    if claims_order_test or claims_global_motion_test:
        return "uniform_timeline_overclaims_motion_or_order_observation"
    return None


def _repair_restoration_choice_problem(
    name: str, visible_state: dict[str, Any]
) -> str | None:
    candidates = (
        (visible_state.get("state_action_affordances") or {}).get(
            "repair_restoration_candidates"
        )
        or []
    )
    if isinstance(candidates, dict):
        candidates = [candidates]
    required_tools = {
        str(item.get("tool_name") or "")
        for item in candidates
        if isinstance(item, dict) and item.get("tool_name")
    }
    if len(required_tools) == 1 and name not in required_tools:
        return f"repair_restoration_candidate_bypassed:{next(iter(required_tools))}"
    return None


def _measurement_observation_gap_problem(
    name: str, reasoning: str, visible_state: dict[str, Any]
) -> str | None:
    contract = _measurement_gap_contract(visible_state, [name])
    if contract is None:
        return None
    if re.search(
        r"(?:does\s+not|doesn't|fails?\s+to|cannot)\s+resolve[^.!?]{0,80}"
        r"because[^.!?]{0,90}(?:not\s+(?:the\s+)?(?:required|recommended)\s+tool|"
        r"(?:required|recommended)\s+tool\s+is)",
        reasoning,
        re.I,
    ):
        return f"measurement_tool_policy_substituted_for_observation_gap:{name}"
    if re.search(
        r"(?:open|undecided)\s+(?:claim\s+)?fields?[^.!?]{0,45}"
        r"(?:measurement\s+)?gap|(?:measurement\s+)?gap[^.!?]{0,45}"
        r"(?:open|undecided)\s+(?:claim\s+)?fields?",
        reasoning,
        re.I,
    ):
        return f"measurement_action_denies_need_for_measurement:{name}"
    if re.search(
        r"low\s+confidence|confidence\s*(?::|=|is|of)?\s*0?\.[0-6]|"
        r"uncertain|disagree|inconsisten|corroborat|independent\s+(?:check|evidence|"
        r"measurement|route|view)|cross[-_ ]method|abstain|limitation|occlud|sparse|"
        r"whether.{0,45}(?:stable|survives?|agrees?)",
        reasoning,
        re.I,
    ):
        return None
    return f"measurement_reasoning_omits_visible_observation_uncertainty:{name}"


def _measurement_recommendation_problem(
    name: str, visible_state: dict[str, Any]
) -> str | None:
    eligible = [str(item) for item in visible_state.get("eligible_tools") or []]
    contract = _measurement_gap_contract(visible_state, eligible or [name])
    recommendation = (contract or {}).get("recommended_next_measurement") or {}
    required_tool = str(recommendation.get("tool_name") or "")
    if required_tool and name != required_tool:
        return f"student_visible_measurement_recommendation_bypassed:{required_tool}"
    return None


def _measurement_need_problem(name: str, reasoning: str) -> str | None:
    if not name.startswith(("compare_", "detect_", "localize_")):
        return None
    gap_reasoning = re.sub(
        r"\b(?:rather\s+than|instead\s+of|not(?:\s+because\s+of)?)\b"
        r"[^.!?]{0,45}\b(?:rebind|bind|bound|unbound|binding|commit|stop)\b",
        "",
        reasoning,
        flags=re.I,
    )
    if re.search(
        r"(?:only\s+remaining\s+(?:step|gap)|(?:blocking|remaining)\s+gap\s+is)"
        r".{0,100}\b(?:rebind|bind|bound|unbound|binding|commit|stop)\b|"
        r"(?:blocking|remaining)\s+gap\s+is(?:\s+that)?\s+(?:the\s+)?"
        r"(?:claim\s+)?fields?[^.!?]{0,160}\bunsupported\b|"
        r"(?:blocking|remaining)\s+gap\s+is\s+that[^.!?]{0,180}"
        r"\bleave(?:s)?\b[^.!?]{0,100}\bunsupported\b|"
        r"(?:fields?|claims?).{0,45}\b(?:still\s+)?lack(?:ing)?\s+(?:of\s+)?"
        r"(?:evidence|support)\b|\black\s+of\s+(?:field\s+)?support\b",
        gap_reasoning,
        re.I,
    ):
        return f"measurement_action_denies_need_for_measurement:{name}"
    return None


def _measurement_outcome_presupposition_problem(
    name: str, reasoning: str
) -> str | None:
    if not name.startswith(("compare_", "detect_", "localize_")):
        return None
    for match in re.finditer(
        r"\b(?:to|will|would)\b.{0,80}\border[-_ ]consistent\b",
        reasoning,
        re.I,
    ):
        segment = match.group(0)
        if re.search(
            r"\b(?:whether|if|test|check|determine|assess|evaluate)\b",
            segment,
            re.I,
        ):
            continue
        if re.search(
            r"\b(?:establish|create|produce|yield|obtain|generate|regenerate|return|show)\b",
            segment,
            re.I,
        ):
            return f"measurement_result_presupposed_before_execution:{name}"
    return None


def _bind_evidence_reasoning_problem(
    name: str, arguments: dict[str, Any] | None, reasoning: str
) -> str | None:
    if name != "bind_evidence":
        return None
    support_ids = {
        str(support_id)
        for item in (arguments or {}).get("binding_items") or []
        for support_id in item.get("support_ids") or []
    }
    task_and_meta_bound = any("TASK" in item for item in support_ids) and any(
        "SYMMETRY_RESIDUAL_ABSTENTION" in item for item in support_ids
    )
    exclusive_meta_wording = re.search(
        r"bind(?:ing)?\s+(?:only\s+)?(?:that|the)\s+meta\s+"
        r"(?:atom|evidence)\s+to\s+(?:every|all)\b",
        reasoning,
        re.I,
    )
    if task_and_meta_bound and exclusive_meta_wording:
        return "bind_reasoning_excludes_required_task_atom"
    return None


def _bind_latest_measurement_problem(
    name: str,
    arguments: dict[str, Any] | None,
    reasoning: str,
    visible_state: dict[str, Any],
) -> str | None:
    """A just-executed measurement must be bound or explicitly dispositioned."""
    if name != "bind_evidence":
        return None
    latest_ids = list(
        ((visible_state.get("latest_environment_observation") or {}).get(
            "created_evidence_ids"
        ) or [])
    )
    bound_ids = {
        str(support_id)
        for item in (arguments or {}).get("binding_items") or []
        for support_id in item.get("support_ids") or []
    }
    evidence_store = visible_state.get("evidence_store") or {}
    for evidence_id in latest_ids:
        atom = evidence_store.get(str(evidence_id)) or {}
        observation = atom.get("observation") or {}
        is_measurement = bool(
            observation.get("measurement_method")
            or observation.get("localization_method")
            or str(atom.get("kind") or "").endswith("measurement")
        )
        if (
            is_measurement
            and str(evidence_id) not in bound_ids
            and not re.search(re.escape(str(evidence_id)), reasoning, re.I)
        ):
            return f"bind_omits_latest_measurement_rationale:{evidence_id}"
    return None


def _claim_creation_source_problem(
    name: str,
    arguments: dict[str, Any] | None,
    reasoning: str,
    visible_state: dict[str, Any],
) -> str | None:
    """Require prose to identify the measurement source actually copied into a claim."""

    if name != "create_candidate_claim":
        return None
    method = str(
        ((arguments or {}).get("field_values") or {}).get("measurement_method") or ""
    )
    if not method:
        return None
    matching_ids = [
        str(evidence_id)
        for evidence_id, atom in (visible_state.get("evidence_store") or {}).items()
        if str((atom.get("observation") or {}).get("measurement_method") or "")
        == method
    ]
    if re.search(re.escape(method), reasoning, re.I) or any(
        re.search(re.escape(evidence_id), reasoning, re.I)
        for evidence_id in matching_ids
    ):
        return None
    return f"claim_creation_source_not_identified:{method}"


def _typed_universal_claim_choice_problem(
    name: str,
    arguments: dict[str, Any] | None,
    visible_state: dict[str, Any],
) -> str | None:
    """Protect universal-quantifier claims from generic-route overclaiming.

    This uses only student-visible structured observations. It is intentionally
    narrow: both unary records must be usable, both must report zero sampled
    violations, and both must explicitly carry the sparse-sampling limitation.
    """

    if name != "create_candidate_claim":
        return None
    field_values = (arguments or {}).get("field_values") or {}
    for evidence_id, atom in (visible_state.get("evidence_store") or {}).items():
        if atom.get("active") is False:
            continue
        observation = atom.get("observation") or {}
        records = observation.get("independent_typed_records") or {}
        record_a = records.get("a") or {}
        record_b = records.get("b") or {}
        reasons_a = set(observation.get("reasons_a") or [])
        reasons_b = set(observation.get("reasons_b") or [])
        is_symmetric_sparse_absence = (
            observation.get("measurement_method") == "compare_predicate_independent"
            and observation.get("predicate_family") == "visibility_coverage"
            and observation.get("decision_rule")
            == "universal_claim_no_observed_counterexample_abstain"
            and observation.get("preferred_video") == "c"
            and observation.get("usable_a") is True
            and observation.get("usable_b") is True
            and record_a.get("violation_count") == 0
            and record_b.get("violation_count") == 0
            and "sparse_sampling_cannot_prove_never" in reasons_a
            and "sparse_sampling_cannot_prove_never" in reasons_b
        )
        if not is_symmetric_sparse_absence:
            continue
        if (
            field_values.get("preferred_video") != "c"
            or field_values.get("measurement_method")
            != "compare_predicate_independent"
        ):
            return f"typed_universal_abstention_overridden:{evidence_id}"
    return None


def _claim_value_conflict_fields(
    visible_state: dict[str, Any],
    claim_id: str,
    evidence_id: str,
    target_fields: list[str],
) -> list[str]:
    claim = next(
        (
            item
            for item in visible_state.get("candidate_claims") or []
            if str(item.get("claim_id")) == claim_id
        ),
        {},
    )
    claim_fields = claim.get("fields") or {}
    observation = (
        ((visible_state.get("evidence_store") or {}).get(evidence_id) or {}).get(
            "observation"
        )
        or {}
    )
    return [
        field
        for field in target_fields
        if field in observation
        and field in claim_fields
        and observation.get(field) != (claim_fields.get(field) or {}).get("value")
    ]


def _reasoning_names_conflict_field(
    reasoning: str,
    evidence_id: str,
    conflict_fields: list[str],
    visible_state: dict[str, Any],
) -> bool:
    explanation = re.sub(re.escape(evidence_id), "", reasoning, flags=re.I)
    observation = (
        ((visible_state.get("evidence_store") or {}).get(evidence_id) or {}).get(
            "observation"
        )
        or {}
    )
    field_aliases = {
        "preferred_video": (r"preferred_video", r"preferred video", r"direction"),
        "video_a_interval_s": (
            r"video_a_interval_s",
            r"video a interval",
            r"a interval",
            r"intervals?",
        ),
        "video_b_interval_s": (
            r"video_b_interval_s",
            r"video b interval",
            r"b interval",
            r"intervals?",
        ),
        "measurement_method": (
            r"measurement_method",
            r"measurement method",
            r"method",
        ),
    }
    for field in conflict_fields:
        aliases = field_aliases.get(field, (re.escape(field),))
        if any(re.search(alias, explanation, re.I) for alias in aliases):
            return True
        value = observation.get(field)
        if isinstance(value, str) and re.search(re.escape(value), explanation, re.I):
            return True
    return False


ADJUDICATION_DEFECT_ALIASES = {
    "claim_value_conflict": (
        r"claim[-_ ]value conflict|conflict(?:s|ing)? with (?:the )?claim|"
        r"does not match|mismatch|contradict|incompatible",
    ),
    "order_inconsistent": (
        r"order[-_ ]inconsisten|swap (?:disagree|inconsisten)|"
        r"A/B (?:swap|order).{0,30}(?:disagree|fail)",
    ),
    "lower_confidence": (r"lower confidence|less confident|weaker confidence",),
    "abstention_conflicts_with_claim": (
        r"abstain|insufficient evidence|neither|equal result|preferred_video.?=.?c",
    ),
    "redundant_same_conclusion": (
        r"redundant|same conclusion|duplicate support|adds no new support",
    ),
}


def _complete_assessment_ledger_matches_adjudication(
    visible_state: dict[str, Any], arguments: dict[str, Any]
) -> bool:
    """Return whether a visible atomic ledger already proves the full partition.

    The final adjudication may summarize rejected alternatives only when every
    competitor has already received a disposition and every rejected defect set
    exactly matches the proposed native call. Exact EvidenceAtom identities remain
    mandatory in the tool arguments and are still checked by the harness.
    """

    claim_id = str(arguments.get("claim_id") or "")
    ledger = ((visible_state.get("evidence_assessments") or {}).get(claim_id) or {})
    if not ledger:
        return False
    selected_ids = [str(item) for item in arguments.get("selected_evidence_ids") or []]
    rejected_items = list(arguments.get("rejected_evidence") or [])
    competitor_ids = [
        *selected_ids,
        *(str(item.get("evidence_id") or "") for item in rejected_items),
    ]
    if not competitor_ids or any(not evidence_id for evidence_id in competitor_ids):
        return False
    if set(ledger) != set(competitor_ids):
        return False
    for evidence_id in selected_ids:
        entry = ledger.get(evidence_id) or {}
        if entry.get("disposition") != "retain" or list(entry.get("defect_codes") or []):
            return False
    for item in rejected_items:
        evidence_id = str(item.get("evidence_id") or "")
        entry = ledger.get(evidence_id) or {}
        if entry.get("disposition") != "reject":
            return False
        if set(str(code) for code in entry.get("defect_codes") or []) != set(
            str(code) for code in item.get("defect_codes") or []
        ):
            return False
    return True


def _adjudication_reasoning_problems(
    name: str,
    arguments: dict[str, Any] | None,
    reasoning: str,
    *,
    visible_state: dict[str, Any] | None = None,
) -> list[str]:
    if name != "adjudicate_evidence":
        return []
    arguments = arguments or {}
    problems: list[str] = []
    if not re.search(r"adjudicat|weigh|compare|resolve", reasoning, re.I):
        problems.append("adjudication_action_not_explained")
    selected_ids = [str(item) for item in arguments.get("selected_evidence_ids") or []]
    rejected_ids = [
        str(item.get("evidence_id") or "")
        for item in arguments.get("rejected_evidence") or []
    ]
    competitor_ids = [*selected_ids, *rejected_ids]
    competitor_handles = {
        evidence_id: f"@E{index}"
        for index, evidence_id in enumerate(competitor_ids, start=1)
    }
    store = (visible_state or {}).get("evidence_store") or {}
    competitor_methods = {
        evidence_id: str(
            ((store.get(evidence_id) or {}).get("observation") or {}).get(
                "measurement_method", ""
            )
        )
        for evidence_id in competitor_ids
    }
    method_counts = {
        method: sum(value == method for value in competitor_methods.values())
        for method in set(competitor_methods.values())
        if method
    }
    ledger_complete = _complete_assessment_ledger_matches_adjudication(
        visible_state or {}, arguments
    )

    def cited(evidence_id: str) -> bool:
        if re.search(re.escape(evidence_id), reasoning, re.I):
            return True
        handle = competitor_handles.get(evidence_id, "")
        if handle and re.search(rf"(?<!\w){re.escape(handle)}\b", reasoning, re.I):
            return True
        method = competitor_methods.get(evidence_id, "")
        return bool(
            method
            and method_counts.get(method) == 1
            and re.search(re.escape(method), reasoning, re.I)
        )

    def selection_grounded(evidence_id: str) -> bool:
        references = [evidence_id, competitor_handles.get(evidence_id, "")]
        method = competitor_methods.get(evidence_id, "")
        if method and method_counts.get(method) == 1:
            references.append(method)
        observation = ((store.get(evidence_id) or {}).get("observation") or {})
        fact_patterns: list[str] = []
        for field, value in observation.items():
            if isinstance(value, bool):
                rendered = str(value).lower()
            elif value is None:
                rendered = r"(?:null|none|unobserved)"
            elif isinstance(value, (str, int, float)):
                rendered = re.escape(str(value))
            else:
                continue
            fact_patterns.append(
                rf"{re.escape(str(field))}\s*[`*]*\s*(?::|=|is|was)\s*"
                rf"[`*\"']*{rendered}\b"
            )
        for reference in (item for item in references if item):
            for match in re.finditer(re.escape(reference), reasoning, re.I):
                sentence_start = max(
                    reasoning.rfind(".", 0, match.start()),
                    reasoning.rfind("!", 0, match.start()),
                    reasoning.rfind("?", 0, match.start()),
                ) + 1
                sentence_end_match = re.search(r"[.!?]", reasoning[match.end():])
                sentence_end = (
                    match.end() + sentence_end_match.end()
                    if sentence_end_match
                    else len(reasoning)
                )
                sentence = reasoning[sentence_start:sentence_end]
                reference_offset = match.start() - sentence_start
                # A sentence may introduce a coordinated list once ("the selected
                # atoms are A ... and B ..."). Keep the shared selection predicate
                # visible for later IDs while still requiring an exact fact in the
                # same sentence.
                around_reference = sentence[max(0, reference_offset - 220):]
                has_selection_reason = bool(
                    re.search(
                        r"(?:retain|select(?:ing|ed)?|support\s+to\s+retain|use\s+as\s+support|"
                        r"select(?:ing|ed)?[^.!?]{0,70}as\s+support)"
                        r"[^.!?]{0,130}(?:because|since|matches?|supports?|fits?|agrees?)|"
                        r"(?:matches?|supports?|fits?|agrees?)[^.!?]{0,90}"
                        r"(?:retain|select(?:ing|ed)?|use\s+as\s+support|select(?:ing|ed)?[^.!?]{0,70}"
                        r"as\s+support)",
                        around_reference,
                        re.I,
                    )
                )
                has_exact_fact = not fact_patterns or any(
                    re.search(pattern, sentence, re.I) for pattern in fact_patterns
                )
                if has_selection_reason and has_exact_fact:
                    return True
        return False

    if re.search(
        r"(?:blocking|remaining|visible)\s+gap\s+(?:is|was|:).{0,120}\b(?:unbound|not bound)\b",
        reasoning,
        re.I,
    ):
        problems.append("adjudication_gap_misframed_as_binding")
    for evidence_id in selected_ids:
        if not cited(evidence_id):
            problems.append(f"adjudication_selected_id_not_cited:{evidence_id}")
    if selected_ids and not all(selection_grounded(item) for item in selected_ids):
        problems.append("adjudication_selection_not_justified")
    if rejected_ids and re.search(
        r"(?:rejected|excluded)\s+(?:atoms?|alternatives?|evidence)[^.!?]{0,25}"
        r"(?:is|are|were|remain)\s+retain|retain\w*[^.!?]{0,25}"
        r"(?:rejected|excluded)\s+(?:atoms?|alternatives?|evidence)",
        reasoning,
        re.I,
    ):
        problems.append("adjudication_rejected_alternatives_called_retained")
    if ledger_complete and not re.search(
        r"(?:assessment|candidate|evidence)[-_ ]ledger|ledger[-_ ](?:backed|complete|recorded)|"
        r"recorded dispositions|prior assessments",
        reasoning,
        re.I,
    ):
        problems.append("adjudication_complete_ledger_not_referenced")
    for item in arguments.get("rejected_evidence") or []:
        evidence_id = str(item.get("evidence_id") or "")
        if ledger_complete:
            continue
        if not evidence_id or not cited(evidence_id):
            problems.append(f"adjudication_rejected_id_not_cited:{evidence_id}")
            continue
        defect_codes = [str(code) for code in item.get("defect_codes") or []]
        aliases = [
            alias
            for code in defect_codes
            for alias in ADJUDICATION_DEFECT_ALIASES.get(code, ())
        ]
        if aliases and not any(re.search(alias, reasoning, re.I) for alias in aliases):
            problems.append(f"adjudication_rejected_defect_not_explained:{evidence_id}")
        if "claim_value_conflict" in defect_codes:
            conflict_fields = _claim_value_conflict_fields(
                visible_state or {},
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
            if conflict_fields and not _reasoning_names_conflict_field(
                reasoning, evidence_id, conflict_fields, visible_state or {}
            ):
                problems.append(
                    f"adjudication_claim_conflict_field_not_named:{evidence_id}"
                )
    return problems


def _evidence_assessment_reasoning_problems(
    name: str,
    arguments: dict[str, Any] | None,
    reasoning: str,
    *,
    visible_state: dict[str, Any] | None = None,
) -> list[str]:
    if name != "assess_evidence_candidate":
        return []
    arguments = arguments or {}
    evidence_id = str(arguments.get("evidence_id") or "")
    atom = (((visible_state or {}).get("evidence_store") or {}).get(evidence_id) or {})
    observation = atom.get("observation") or {}
    method = str(observation.get("measurement_method") or "")
    requires_measurement_semantics = bool(
        method
        or "preferred_video" in observation
        or "order_consistent" in observation
        or str(atom.get("kind") or "").endswith("measurement")
    )
    problems: list[str] = []
    cites_id = bool(
        evidence_id and re.search(re.escape(evidence_id), reasoning, re.I)
    )
    cites_method = bool(method and re.search(re.escape(method), reasoning, re.I))
    if not (cites_id or cites_method):
        problems.append(f"assessment_candidate_not_cited:{evidence_id}")

    if not requires_measurement_semantics:
        for field in atom.get("supports_fields") or []:
            if field not in observation:
                continue
            value = observation.get(field)
            if not isinstance(value, (str, int, float, bool)) or value is None:
                continue
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            if not re.search(
                rf"{re.escape(str(field))}\s*[`*]*\s*(?::|=|is|was)\s*"
                rf"[`*\"']*{re.escape(rendered)}(?!\w)",
                reasoning,
                re.I,
            ):
                problems.append(
                    f"assessment_identity_fact_not_stated:{evidence_id}:{field}"
                )

    pending_ids = (
        ((visible_state or {}).get("evidence_assessment") or {}).get(
            "pending_evidence_ids"
        )
        or []
    )
    if pending_ids and re.search(
        r"(?:assessment|candidate|evidence)[-_ ]ledger[^.!?]{0,25}"
        r"(?:is|was|:)?\s*complete|ledger[-_ ]complete",
        reasoning,
        re.I,
    ) and not re.search(r"(?:not\s+complete|incomplete)", reasoning, re.I):
        problems.append("assessment_falsely_claims_complete_ledger")
    claim_id = str(arguments.get("claim_id") or "")
    recorded_ledger = (
        ((visible_state or {}).get("evidence_assessments") or {}).get(claim_id)
        or {}
    )
    if evidence_id not in recorded_ledger and re.search(
        r"\b(?:already|previously)\s+(?:rejected|retained|recorded|assessed)\b|"
        r"\b(?:rejected|retained)\s+(?:already|previously)\s+in\s+(?:the\s+)?ledger\b",
        reasoning,
        re.I,
    ):
        problems.append(
            f"pending_assessment_described_as_already_recorded:{evidence_id}"
        )
    if len(pending_ids) > 1 and re.search(
        r"(?:clos(?:e|es|ing)|complet(?:e|es|ing)|finish(?:es|ing)?)"
        r"[^.!?]{0,30}(?:assessment|candidate|evidence)[-_ ]ledger",
        reasoning,
        re.I,
    ):
        issue = "assessment_falsely_claims_complete_ledger"
        if issue not in problems:
            problems.append(issue)

    claim = next(
        (
            item
            for item in (visible_state or {}).get("candidate_claims") or []
            if str(item.get("claim_id") or "")
            == str(arguments.get("claim_id") or "")
        ),
        {},
    )
    claim_fields = claim.get("fields") or {}
    has_unsupported_target = any(
        (claim_fields.get(str(field)) or {}).get("evidence_status") != "supported"
        for field in arguments.get("target_fields") or []
        if str(field) in claim_fields
    )
    if has_unsupported_target and re.search(
        r"claim\s+(?:field|value)s?[^.!?]{0,25}"
        r"(?:is|are|remain)\s+(?:already\s+)?supported",
        reasoning,
        re.I,
    ):
        problems.append("assessment_claims_premature_field_support")
    if re.search(
        r"(?:this|the)\s+(?:evidence\s+)?atom\s+is\s+the\s+only\s+eligible\s+action",
        reasoning,
        re.I,
    ):
        problems.append("assessment_atom_called_action")

    preferred = observation.get("preferred_video")
    if preferred is not None and not re.search(
        rf"preferred_video\s*[`*]*\s*(?::|=|is|was|as)\s*"
        rf"[`*\"']*{re.escape(str(preferred))}\b[`*\"']*",
        reasoning,
        re.I,
    ):
        problems.append(f"assessment_preferred_video_not_stated:{evidence_id}")

    order_consistent = observation.get("order_consistent")
    if requires_measurement_semantics and order_consistent is None:
        if not re.search(
            r"order_consistent.{0,30}(?:unobserved|unknown|not (?:observed|reported)|null|none)",
            reasoning,
            re.I,
        ):
            problems.append(f"assessment_unobserved_order_not_stated:{evidence_id}")
    elif requires_measurement_semantics and order_consistent is False and not re.search(
        r"order_consistent\s*[`*]*\s*(?::|=|is|was)\s*[`*\"']*false\b[`*\"']*|"
        r"order[-_ ]inconsisten",
        reasoning,
        re.I,
    ):
        problems.append(f"assessment_order_inconsistency_not_stated:{evidence_id}")
    elif requires_measurement_semantics and order_consistent is True and not re.search(
        r"order_consistent\s*[`*]*\s*(?::|=|is|was)\s*[`*\"']*true\b[`*\"']*|"
        r"order[-_ ]consisten",
        reasoning,
        re.I,
    ):
        problems.append(f"assessment_order_consistency_not_stated:{evidence_id}")

    disposition = str(arguments.get("disposition") or "")
    if disposition == "retain":
        if not re.search(r"retain|select|keep|usable support", reasoning, re.I):
            problems.append(f"assessment_retain_disposition_not_explained:{evidence_id}")
        if re.search(
            r"(?:visible\s+)?defect\s+(?:is|was|:).{0,45}"
            r"(?:missing|unobserved|unknown|absent).{0,30}order",
            reasoning,
            re.I,
        ):
            problems.append(f"assessment_retained_uncertainty_called_defect:{evidence_id}")
        if re.search(
            r"(?:only|sole)\s+remaining\s+gap[^.!?]{0,70}"
            r"(?:missing|unobserved|null|absent)[^.!?]{0,35}order|"
            r"(?:only|sole)\s+remaining\s+gap[^.!?]{0,70}order[-_ ]"
            r"consisten[^.!?]{0,25}(?:signal|value|observation)",
            reasoning,
            re.I,
        ):
            problems.append(
                f"assessment_retained_uncertainty_misframed_as_action_gap:{evidence_id}"
            )
    if disposition == "reject":
        if not re.search(r"reject|exclude|not retain|weaker", reasoning, re.I):
            problems.append(f"assessment_reject_disposition_not_explained:{evidence_id}")
        aliases = [
            alias
            for code in arguments.get("defect_codes") or []
            for alias in ADJUDICATION_DEFECT_ALIASES.get(str(code), ())
        ]
        if aliases and not any(re.search(alias, reasoning, re.I) for alias in aliases):
            problems.append(f"assessment_rejection_defect_not_explained:{evidence_id}")
        defect_codes = [str(code) for code in arguments.get("defect_codes") or []]
        if "claim_value_conflict" in defect_codes:
            conflict_fields = _claim_value_conflict_fields(
                visible_state or {},
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
            if conflict_fields and not _reasoning_names_conflict_field(
                reasoning, evidence_id, conflict_fields, visible_state or {}
            ):
                problems.append(
                    f"assessment_claim_conflict_field_not_named:{evidence_id}"
                )
            if re.search(
                r"(?:visible\s+)?mismatch[^.!?]{0,70}"
                r"(?:claim\s+)?(?:values?|fields?)[^.!?]{0,35}unsupported|"
                r"unsupported[^.!?]{0,55}(?:visible\s+)?mismatch",
                reasoning,
                re.I,
            ):
                problems.append(
                    f"assessment_claim_conflict_misframed_as_support_status:{evidence_id}"
                )
        routing_state = observation.get("routing_state") or {}
        decision_rule = str(observation.get("decision_rule") or "")
        abstention_cause = re.search(
            r"abstain\w*\s+because(?P<cause>[^.!?]{0,100})",
            reasoning,
            re.I,
        )
        if (
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            and routing_state.get("reason_code") == "all_candidates_exhausted"
            and re.search(
                r"abstain\w*\s+because[^.!?]{0,45}"
                r"all\s+candidates?\s+(?:are\s+)?exhausted",
                reasoning,
                re.I,
            )
        ):
            problems.append(
                f"assessment_abstention_cause_confuses_routing_termination:{evidence_id}"
            )
        elif (
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            and decision_rule
            in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
            and abstention_cause
            and not re.search(
                r"\border[-_ ]disagreement\b|"
                r"(?:direction|result|comparison)?[^.!?]{0,45}"
                r"(?:disagree|inconsisten|differ|change|flip|fail(?:s|ed)?\s+to\s+survive)"
                r"[^.!?]{0,55}(?:A\s*/\s*B\s+(?:presentation\s+)?orders?|"
                r"presentation\s+orders?|role[- ]?swapp)|"
                r"(?:A\s*/\s*B\s+(?:presentation\s+)?orders?|presentation\s+orders?|"
                r"role[- ]?swapp)[^.!?]{0,55}"
                r"(?:disagree|inconsisten|differ|change|flip|fail(?:s|ed)?\s+to\s+survive)",
                abstention_cause.group("cause"),
                re.I,
            )
        ):
            problems.append(
                f"assessment_abstention_cause_omits_decision_rule:{evidence_id}"
            )
        if (
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            and decision_rule
            in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
            and not re.search(
                r"(?:abstain\w*[^.!?]{0,90}(?:because|due\s+to|from)"
                r"[^.!?]{0,70}(?:order|role[- ]?swapp|a\s*/\s*b\s+presentation)|"
                r"(?:abstention|abstain\w*)[^.!?]{0,45}"
                r"(?:(?:is|was)\s+caus\w*\s+(?:by|from)|"
                r"caus\w*[^.!?]{0,20}(?:is|was|comes?\s+from|due\s+to))"
                r"[^.!?]{0,45}"
                r"(?:disagree|inconsisten|differ|change|flip)[^.!?]{0,45}"
                r"(?:orders?|a\s*/\s*b\s+presentation)|"
                r"a\s*/\s*b\s+(?:presentation[- ]?)?orders?[^.!?]{0,30}"
                r"(?:disagree|inconsisten|differ)[^.!?]{0,40}"
                r"(?:caus\w*|lead\w*|therefore|so)[^.!?]{0,30}"
                r"(?:abstain\w*|abstention)|"
                r"(?:order[-_ ]inconsisten|directions?[^.!?]{0,30}disagree[^.!?]{0,30}"
                r"(?:orders?|a\s*/\s*b))[^.!?]{0,70}"
                r"(?:caus\w*|lead\w*|therefore|so)[^.!?]{0,30}abstain|"
                r"preferred_video\s*[`*]*\s*(?::|=|is)\s*[`*\"']*c\b"
                r"[^.!?]{0,90}order_consistent\s*[`*]*\s*(?::|=|is)\s*"
                r"[`*\"']*false\b[^.!?]{0,90}order[-_ ]disagreement\s+"
                r"abstention\s+rule|"
                r"(?:direct\s+)?abstention\s+cause\s+(?:is|was)\s+(?:that\s+)?"
                r"[^.!?]{0,65}(?:a\s*/\s*b\s+(?:presentation[- ]?)?orders?|"
                r"presentation\s+orders?|role[- ]?swapp)[^.!?]{0,45}"
                r"(?:disagree|inconsisten|unstable)|"
                r"(?:a\s*/\s*b\s+direction[^.!?]{0,25}unstable[^.!?]{0,20})?"
                r"due\s+to\s+(?:the\s+)?presentation[- ]order\s+disagreement"
                r"[^.!?]{0,25}caus(?:e[sd]?|ing)\s+(?:the\s+)?abstention)",
                reasoning,
                re.I,
            )
        ):
            issue = f"assessment_abstention_decision_rule_not_explained:{evidence_id}"
            if issue not in problems:
                problems.append(issue)
    if re.search(
        r"(?:this|the)\s+(?:native\s+)?call[^.!?]{0,70}"
        r"(?:every|all)\s+(?:exact\s+)?rejected\s+(?:evidence\s+)?IDs?"
        r"[^.!?]{0,45}defect",
        reasoning,
        re.I,
    ):
        problems.append("assessment_claims_full_adjudication_partition_payload")
    return problems


def _evidence_assessment_reasoning_frontier(
    state: dict[str, Any], executable_actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    actions = [
        item
        for item in executable_actions
        if item.get("tool_name") == "assess_evidence_candidate"
    ]
    if not actions:
        return None
    store = state.get("evidence_store") or {}
    claims = {
        str(claim.get("claim_id") or ""): claim
        for claim in state.get("candidate_claims") or []
    }
    candidates = []
    for action in actions:
        arguments = action.get("arguments") or {}
        evidence_id = str(arguments.get("evidence_id") or "")
        atom = store.get(evidence_id) or {}
        observation = atom.get("observation") or {}
        measurement_semantics_required = bool(
            observation.get("measurement_method")
            or "preferred_video" in observation
            or "order_consistent" in observation
            or str(atom.get("kind") or "").endswith("measurement")
        )
        defect_codes = [str(code) for code in arguments.get("defect_codes") or []]
        candidate = {
            "evidence_id": evidence_id,
            "evidence_kind": atom.get("kind"),
            "supports_fields": copy.deepcopy(atom.get("supports_fields") or []),
            "supported_observation_facts": {
                field: copy.deepcopy(observation.get(field))
                for field in atom.get("supports_fields") or []
                if field in observation
            },
            "measurement_semantics_required": measurement_semantics_required,
            "measurement_method": observation.get("measurement_method"),
            "preferred_video": observation.get("preferred_video"),
            "confidence": observation.get("confidence"),
            "disposition": arguments.get("disposition"),
            "validated_defect_codes": defect_codes,
            "required_decision_gap_clause": (
                f"The visible decision gap is the unrecorded assessment for {evidence_id}."
            ),
        }
        if measurement_semantics_required:
            candidate["order_consistent"] = observation.get("order_consistent")
            order_value = observation.get("order_consistent")
            rendered_order = (
                "unobserved"
                if order_value is None
                else str(bool(order_value)).lower()
            )
            candidate["required_measurement_semantics_clause"] = (
                f"{evidence_id} reports preferred_video="
                f"{json.dumps(observation.get('preferred_video'), ensure_ascii=False, separators=(',', ':'))} "
                f"and order_consistent={rendered_order}."
            )
        if "claim_value_conflict" in defect_codes:
            candidate["validated_claim_conflict_fields"] = _claim_value_conflict_fields(
                state,
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
        rejection_clauses: list[str] = []
        conflict_fields = candidate.get("validated_claim_conflict_fields") or []
        claim = claims.get(str(arguments.get("claim_id") or "")) or {}
        claim_fields = claim.get("fields") or {}
        if conflict_fields:
            field = str(conflict_fields[0])
            observed_value = observation.get(field)
            claim_value = (claim_fields.get(field) or {}).get("value")
            rejection_clauses.append(
                f"Reject {evidence_id} because {field}="
                f"{json.dumps(observed_value, ensure_ascii=False, separators=(',', ':'))} "
                f"conflicts with the claim value {field}="
                f"{json.dumps(claim_value, ensure_ascii=False, separators=(',', ':'))}."
            )
        if "order_inconsistent" in defect_codes:
            rejection_clauses.append(
                f"Reject {evidence_id} because order_consistent=false makes its "
                "A/B direction unstable."
            )
        if "abstention_conflicts_with_claim" in defect_codes:
            claim_preferred = (claim_fields.get("preferred_video") or {}).get("value")
            rejection_clauses.append(
                f"Reject {evidence_id} because preferred_video=\"c\" abstains while "
                f"the claim value is preferred_video="
                f"{json.dumps(claim_preferred, ensure_ascii=False, separators=(',', ':'))}."
            )
        if rejection_clauses:
            candidate["required_rejection_grounding_clauses"] = rejection_clauses
        if (
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            and observation.get("decision_rule")
            in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
        ):
            candidate["required_abstention_causal_clause"] = (
                "This atom abstains because its direction disagrees across the two A/B "
                "presentation orders."
            )
            candidate["causal_edges"] = [
                {
                    "cause": "direction_disagreement_across_two_ab_presentation_orders",
                    "effect": "preferred_video_c_abstention",
                    "forbid_reverse_relation": True,
                }
            ]
        candidates.append(candidate)
    return {
        "kind": "progressive_evidence_candidate_assessment_frontier_v1",
        "pending_candidate_assessments": candidates,
        "natural_language_requirement": (
            "Choose exactly one candidate assessment. Identify it by exact EvidenceAtom "
            "ID or unique measurement_method. When measurement_semantics_required=true, "
            "copy required_measurement_semantics_clause or explicitly state preferred_video "
            "and order_consistent as true, false, or unobserved exactly as shown. Otherwise cite "
            "exact supported_observation_facts and do not invent "
            "comparison fields. Then explain the visible retain/reject disposition and at least "
            "one validated defect for a rejection. Copy every supplied "
            "required_rejection_grounding_clauses sentence or state an exact semantic "
            "equivalent. For claim_value_conflict, name an actually mismatched claim field "
            "or its visible mismatched value; a matching preferred_video is not itself a "
            "conflict. If required_abstention_causal_clause is supplied, copy it or explain "
            "the same direct A/B-order cause in natural language. Do not summarize multiple "
            "candidates in this atomic turn."
        ),
    }


def _adjudication_reasoning_frontier(
    state: dict[str, Any], executable_actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    action = next(
        (
            item
            for item in executable_actions
            if item.get("tool_name") == "adjudicate_evidence"
        ),
        None,
    )
    if action is None:
        return None
    arguments = action.get("arguments") or {}
    store = state.get("evidence_store") or {}

    competitor_ids = [
        *(str(value) for value in arguments.get("selected_evidence_ids") or []),
        *(
            str(item["evidence_id"])
            for item in arguments.get("rejected_evidence") or []
        ),
    ]
    handles = {
        evidence_id: f"@E{index}"
        for index, evidence_id in enumerate(competitor_ids, start=1)
    }

    def summary(evidence_id: str) -> dict[str, Any]:
        atom = store.get(evidence_id) or {}
        observation = atom.get("observation") or {}
        claim = next(
            (
                item
                for item in state.get("candidate_claims") or []
                if str(item.get("claim_id") or "")
                == str(arguments.get("claim_id") or "")
            ),
            {},
        )
        claim_fields = claim.get("fields") or {}
        matching_facts = {
            field: copy.deepcopy(observation.get(field))
            for field in arguments.get("target_fields") or []
            if field in observation
            and field in claim_fields
            and observation.get(field) == (claim_fields.get(field) or {}).get("value")
        }
        result = {
            "evidence_id": evidence_id,
            "reasoning_handle": handles[evidence_id],
            "supports_fields": copy.deepcopy(atom.get("supports_fields") or []),
            "claim_matching_facts": matching_facts,
            "measurement_method": observation.get("measurement_method"),
            "preferred_video": observation.get("preferred_video"),
            "confidence": observation.get("confidence"),
            "order_consistent": observation.get("order_consistent"),
        }
        return {key: value for key, value in result.items() if value is not None}

    selected = [summary(str(value)) for value in arguments.get("selected_evidence_ids") or []]
    for item in selected:
        matching_facts = item.get("claim_matching_facts") or {}
        if matching_facts:
            field, value = next(iter(matching_facts.items()))
            item["required_minimal_grounding_clause"] = (
                f"Retain {item['evidence_id']} because {field}="
                f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))} "
                "matches the claim."
            )
            item["required_compact_grounding_clause"] = (
                f"Retain {item['reasoning_handle']} because {field}="
                f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))} "
                "matches the claim."
            )
    rejected = []
    for item in arguments.get("rejected_evidence") or []:
        evidence_id = str(item["evidence_id"])
        entry = summary(evidence_id)
        defect_codes = [str(code) for code in item.get("defect_codes") or []]
        entry["validated_defect_codes"] = defect_codes
        if "claim_value_conflict" in defect_codes:
            entry["validated_claim_conflict_fields"] = _claim_value_conflict_fields(
                state,
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
        rejected.append(entry)
    ledger_complete = _complete_assessment_ledger_matches_adjudication(state, arguments)
    return {
        "kind": "evidence_adjudication_reasoning_frontier_v1",
        "selected_support": selected,
        "rejected_alternatives": rejected,
        "required_atom_mentions": [
            {
                "reasoning_handle": item["reasoning_handle"],
                "evidence_id": item["evidence_id"],
            }
            for item in (selected if ledger_complete else [*selected, *rejected])
        ],
        "rejected_alternatives_externalized_in_complete_ledger": ledger_complete,
        "decision_gap": (
            "The candidate-assessment ledger is complete, but the final selected/rejected "
            "evidence partition has not yet been recorded for the claim."
        ),
        "natural_language_requirement": (
            "Identify every selected alternative by its turn-local reasoning_handle, exact "
            "EvidenceAtom ID, or unique measurement_method, and justify each with at least one "
            "exact field/value from its claim_matching_facts. Copy each selected atom's "
            "required_minimal_grounding_clause or write an equivalent literal field=value "
            "clause; a field-name list is insufficient. Explicitly name the decision_gap. When "
            "rejected_alternatives_externalized_in_complete_ledger is true, explicitly say the "
            "completed per-candidate assessment ledger supplies the already-validated rejection "
            "reasons; do not redundantly repeat every rejected ID in prose. The native tool call "
            "must still retain every exact rejected ID and defect code and agree with the ledger. "
            "When that flag is false, identify every rejection and verbalize a validated defect; "
            "for claim_value_conflict, name an actually mismatched field."
        ),
    }


def _reasoning_process_artifact_problems(
    name: str, reasoning: str, visible_state: dict[str, Any]
) -> list[str]:
    """Reject retry/audit leakage and false claims that a first-use tool is rerun."""
    problems: list[str] = []
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    claims = visible_state.get("candidate_claims") or []
    stated_field_count = re.search(
        r"\ball\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:required\s+)?(?:claim\s+)?fields\b",
        reasoning,
        re.I,
    )
    if len(claims) == 1 and stated_field_count:
        raw_count = stated_field_count.group(1).lower()
        stated_count = number_words.get(raw_count, int(raw_count) if raw_count.isdigit() else 0)
        actual_count = len((claims[0].get("fields") or {}))
        if stated_count != actual_count:
            problems.append(
                f"claim_field_count_mismatch:said_{stated_count}!={actual_count}"
            )
    if re.search(
        r"active measurement count mismatch|generation (?:retry|attempt)|retry feedback",
        reasoning,
        re.I,
    ):
        problems.append("reasoning_mentions_generation_audit_artifact")
    if re.search(
        r"\brequired\s+(?:rejection\s+grounding|measurement\s+semantics|"
        r"abstention\s+causal|decision\s+gap|compact\s+grounding|"
        r"minimal\s+grounding)(?:\s+clauses?)?\b|"
        r"\bnatural_language_requirement\b|\brevision requirements?\b",
        reasoning,
        re.I,
    ):
        problems.append("reasoning_mentions_prompt_scaffolding")
    if re.search(
        r"\b(?:current_harness_state|state_action_affordances|eligible_tools|"
        r"execution_frontier|compression_contract|grounded_reasoning_plan|"
        r"action_constraints|must_reopen_before_rebinding|"
        r"executable_(?:binding|decision|evidence_assessment|adjudication)_actions)\b|"
        r"\bstudent[- ]visible\s+(?:state|field|fact|atom|evidence|observation)s?\b",
        reasoning,
        re.I,
    ):
        problems.append("reasoning_mentions_internal_policy_scaffolding")
    claims_all_supported = re.search(
        r"(?:all\s+(?:six|6|required)\s+(?:claim\s+)?fields.{0,45}support"
        r"|support.{0,45}all\s+(?:six|6|required)\s+(?:claim\s+)?fields)",
        reasoning,
        re.I,
    )
    claims_all_unsupported = re.search(
        r"(?:every|all)\s+(?:claim\s+)?fields?.{0,25}\bunsupported\b",
        reasoning,
        re.I,
    )
    if claims_all_supported and claims_all_unsupported:
        problems.append("reasoning_self_contradicts_field_support")
    binding_ready = bool(
        ((visible_state.get("state_action_affordances") or {}).get(
            "executable_binding_actions"
        ))
    )
    denies_bindable_support = re.search(
        r"(?:(?:every|all|those|the)\s+(?:(?:one|two|three|four|five|six|"
        r"seven|eight|nine|ten|\d+)\s+)?"
        r"(?:required\s+)?(?:claim\s+|measurement\s+)?fields?|"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:required\s+)?(?:claim\s+|measurement\s+)?fields?)"
        r".{0,30}"
        r"\bunsupported\b|none\s+of\s+(?:the\s+)?(?:required\s+)?fields?.{0,25}"
        r"\bsupported\b",
        reasoning,
        re.I,
    )
    if name == "bind_evidence" and binding_ready and denies_bindable_support:
        problems.append("bindable_fields_described_as_unsupported")
    if name == "bind_evidence" and re.search(
        r"\b(?:bind(?:ing)?|bind_evidence)\b[^.!?]{0,120}"
        r"\b(?:closes?|completes?|finishes?)\b[^.!?]{0,60}"
        r"\b(?:candidate[- ]assessment|assessment(?:\s+ledger|\s+step)?|ledger)\b|"
        r"\b(?:closes?|completes?|finishes?)\b[^.!?]{0,60}"
        r"\b(?:candidate[- ]assessment|assessment(?:\s+ledger|\s+step)?|ledger)\b"
        r"[^.!?]{0,120}\b(?:bind(?:ing)?|bind_evidence)\b",
        reasoning,
        re.I,
    ):
        problems.append("binding_described_as_assessment_transition")
    if name == "assess_evidence_candidate" and re.search(
        r"\b(?:retain(?:ing|ed)?|assessment)\b[^.!?;]{0,100}"
        r"\b(?:closes?|creates?|establishes?|adds?|provides?)\b[^.!?;]{0,70}"
        r"\b(?:field\s+)?support\b|"
        r"\b(?:closes?|creates?|establishes?)\b[^.!?;]{0,70}"
        r"\b(?:field\s+)?support\b[^.!?;]{0,70}\bretain(?:ing|ed)?\b",
        reasoning,
        re.I,
    ):
        problems.append("assessment_described_as_binding_field_support")
    if name == "create_candidate_claim" and re.search(
        r"\b(?:creat(?:e|es|ing)|turn(?:s|ing)?)\b[^.!?]{0,120}"
        r"\b(?:into\s+)?(?:a\s+)?(?:supported|bound|committed)\s+"
        r"(?:candidate\s+)?claim\b|"
        r"\b(?:candidate\s+)?claim\b[^.!?]{0,90}"
        r"\b(?:becomes?|is\s+now)\s+(?:supported|bound|committed)\b",
        reasoning,
        re.I,
    ):
        problems.append("claim_creation_described_as_binding_or_commitment")
    reopen_before_rebinding = list(
        ((visible_state.get("action_constraints") or {}).get(
            "must_reopen_before_rebinding"
        ) or [])
    )
    if (
        name == "set_field_decision"
        and not reopen_before_rebinding
        and re.search(
            r"(?:allowed|valid|correct|appropriate|can proceed|closes? that gap)"
            r"[^.!?]{0,80}(?:must_reopen_before_rebinding|reopen[- ]before[- ]rebinding)|"
            r"(?:must_reopen_before_rebinding|reopen[- ]before[- ]rebinding)"
            r"[^.!?]{0,80}(?:allowed|valid|correct|appropriate|can proceed|closes? that gap)",
            reasoning,
            re.I,
        )
    ):
        problems.append("irrelevant_rebinding_constraint_justifies_decision_transition")
    if re.search(r"\brebind_evidence\b", reasoning, re.I):
        problems.append("nonexistent_tool_name_in_reasoning:rebind_evidence")
    if re.search(
        r"\bintervals?\s+(?:are|is|remain(?:s)?|be|become(?:s)?)\s+"
        r"order[-_ ]consistent\b|\border[-_ ]consistent\s+intervals?\b",
        reasoning,
        re.I,
    ):
        problems.append("order_consistency_misapplied_to_intervals")
    rerun_matches = list(
        re.finditer(
            r"\b(?:re-?run(?:ning)?|run(?:ning)?\s+(?:it\s+)?again|repeat(?:ing)?)\b",
            reasoning,
            re.I,
        )
    )
    affirmative_rerun_claim = any(
        not re.search(
            r"(?:rather\s+than|instead\s+of|without|not|avoid(?:ing)?|no\s+need\s+to)\s*$",
            reasoning[max(0, match.start() - 45):match.start()],
            re.I,
        )
        and not re.match(
            r"\s+(?:a\s+)?(?:different|another|new)\s+(?:eligible\s+)?"
            r"(?:measurement\s+)?(?:method|route|tool)\b",
            reasoning[match.end():match.end() + 70],
            re.I,
        )
        for match in rerun_matches
    )
    if name.startswith("compare_") and affirmative_rerun_claim:
        executed_tools = {
            str((item.get("past_action") or {}).get("executed_tool") or "")
            for item in visible_state.get("past_execution_history") or []
        }
        dropped_ids = [
            str(item) for item in visible_state.get("dropped_evidence_ids") or []
        ]
        prior_snapshot_same_method = any(
            name.upper() in evidence_id.upper() for evidence_id in dropped_ids
        )
        if not prior_snapshot_same_method:
            prior_snapshot_same_method = any(
                str((atom.get("observation") or {}).get("measurement_method")) == name
                for atom in (visible_state.get("evidence_store") or {}).values()
            )
        if not prior_snapshot_same_method and dropped_ids:
            prior_snapshot_same_method = any(
                str(((claim.get("fields") or {}).get("measurement_method") or {}).get("value"))
                == name
                and ((claim.get("fields") or {}).get("measurement_method") or {}).get(
                    "evidence_status"
                )
                in {"unsupported", "conflicted"}
                for claim in visible_state.get("candidate_claims") or []
            )
        if name not in executed_tools and not prior_snapshot_same_method:
            problems.append(f"first_use_measurement_described_as_rerun:{name}")
    method_is_bound = any(
        isinstance((bindings or {}).get("measurement_method"), dict)
        and bool((bindings.get("measurement_method") or {}).get("support_ids"))
        and (bindings.get("measurement_method") or {}).get("validation_status")
        == "accepted"
        for bindings in (visible_state.get("field_bindings") or {}).values()
        if isinstance(bindings, dict)
    )
    if not method_is_bound and re.search(
        r"\b(?:the\s+)?(?:current\s+)?claim\s+(?:still\s+)?binds?\s+"
        r"[`*]*compare_[a-z0-9_]+[`*]*\s+as\s+(?:the\s+)?method\b",
        reasoning,
        re.I,
    ):
        problems.append("unbound_claim_method_described_as_bound")
    dropped_ids = [
        str(item) for item in visible_state.get("dropped_evidence_ids") or []
    ]
    if dropped_ids and re.search(
        r"(?:evidence[-_ ]dropout|dropped|removed)[^.!?]{0,140}"
        r"rebind(?:ing)?\s+(?:the\s+)?same\s+(?:atom|evidence)|"
        r"rebind(?:ing)?\s+(?:the\s+)?same\s+(?:atom|evidence)"
        r"[^.!?]{0,140}(?:evidence[-_ ]dropout|dropped|removed)",
        reasoning,
        re.I,
    ):
        problems.append("dropped_atom_described_as_rebindable")
    if dropped_ids and re.search(
        r"(?:dropout|mutation|removed|dropped).{0,120}"
        r"(?:same|original|removed|dropped) (?:active )?(?:atom|evidence)"
        r".{0,60}(?:remain(?:s|ed)? valid|still valid|intended to be rebound|rebind)",
        reasoning,
        re.I,
    ):
        problems.append("evidence_dropout_misdescribed_as_binding_invalidation")
    return problems


def _repetitive_reasoning_style_problems(
    reasoning: str, prior_reasonings: list[str]
) -> list[str]:
    """Reject recurring discourse scaffolds while allowing one natural use."""
    patterns = {
        "latest_visible_state": r"\b(?:the\s+)?latest\s+visible\s+state\b",
        "only_remaining_gap": r"\bthe\s+only\s+(?:remaining|blocking)\s+gap\b",
        "blocking_gap_label": r"\bthe\s+blocking\s+gap\b",
    }
    problems: list[str] = []
    if re.search(r"\bthe\s+only\s+eligible\s+(?:tool|action)\b", reasoning, re.I):
        problems.append("stock_reasoning_scaffold:only_eligible_action")
    for label, pattern in patterns.items():
        if not re.search(pattern, reasoning, re.I):
            continue
        if any(re.search(pattern, prior, re.I) for prior in prior_reasonings):
            problems.append(f"repeated_reasoning_scaffold:{label}")
    return problems


def _visible_observation_fact_problems(
    reasoning: str, visible_state: dict[str, Any]
) -> list[str]:
    """Check explicit claims about any named visible atom against structured facts."""
    evidence_store = visible_state.get("evidence_store") or {}
    problems: list[str] = []
    denies_nonempty_store = re.search(
        r"\bevidence\s+store\s+is\s+empty\b|"
        r"\bno\s+(?:visible\s+|active\s+)?(?:observation|atom|evidence(?:atom)?)s?\s+"
        r"(?:has|have)\s+been\s+established\b|"
        r"\bthere\s+(?:are|is)\s+no\s+(?:available\s+|active\s+)?"
        r"(?:evidenceatom\s+ids?|atoms?|evidence)\b",
        reasoning,
        re.I,
    )
    if evidence_store and denies_nonempty_store:
        problems.append(
            f"visible_evidence_store_denied_nonempty:actual_{len(evidence_store)}"
        )
    active_measurement_count = sum(
        atom.get("active") is True
        and bool((atom.get("observation") or {}).get("measurement_method"))
        for atom in evidence_store.values()
    )
    latest_created_ids = [
        str(item)
        for item in (
            (visible_state.get("latest_environment_observation") or {}).get(
                "created_evidence_ids"
            )
            or []
        )
    ]
    latest_reference = re.search(
        r"\b(?:the\s+)?latest\s+(?:active\s+)?(?:observation|atom|evidence)\b",
        reasoning,
        re.I,
    )
    if len(latest_created_ids) == 1 and latest_reference:
        latest_id = latest_created_ids[0]
        latest_observation = (
            (evidence_store.get(latest_id) or {}).get("observation") or {}
        )
        latest_sentence = reasoning[latest_reference.start():]
        sentence_end = re.search(r"[.!?](?:\s|$)", latest_sentence)
        if sentence_end:
            latest_sentence = latest_sentence[: sentence_end.end()]
        asserted_patterns = {
            "preferred_video": r"preferred_video\s*[`*]*\s*(?::|=|is)\s*[`*]*[abc]\b",
            "order_consistent": r"order_consistent\s*[`*]*\s*(?::|=|is)\s*[`*]*(?:true|false|null|none)\b",
            "confidence": r"confidence\s*[`*]*\s*(?::|=|is)\s*[`*]*\d+(?:\.\d+)?\b",
        }
        for field, pattern in asserted_patterns.items():
            if field not in latest_observation and re.search(
                pattern, latest_sentence, re.I
            ):
                problems.append(
                    f"latest_created_atom_asserts_absent_field:{latest_id}:{field}"
                )
    says_only_one_measurement_exists = re.search(
        r"\bonly\s+one\s+active\s+measurement\b|"
        r"\b(?:the\s+)?only\s+active\s+measurement\b"
        r"(?!\s+(?:(?:atom|evidence)\s+)?(?:that|which|to|with|reaching|supporting|yielding|returning|"
        r"showing|providing|producing))",
        reasoning,
        re.I,
    )
    if active_measurement_count > 1 and says_only_one_measurement_exists:
        problems.append(
            f"active_measurement_count_mismatch:said_only_one!={active_measurement_count}"
        )
    denies_any_active_field_evidence = re.search(
        r"\b(?:no|without)\s+(?:currently\s+)?active\s+(?:measurement\s+)?evidence\b"
        r".{0,45}(?:for\s+)?(?:the\s+)?(?:open\s+|comparison\s+|measurement\s+)?fields?\b|"
        r"\b(?:new|another|complementary)\s+measurement\b.{0,55}"
        r"(?:reestablish|re-establish|restore|recreate)\s+(?:field\s+)?support\b|"
        r"\bmissing\s+(?:active\s+)?measurement\b|"
        r"\bno\s+new\s+(?:measurement\s+)?evidence\b.{0,35}"
        r"(?:replace|replaced|replacement)|"
        r"\b(?:left|leaves?)\s+(?:the\s+)?(?:four|six|\d+)?\s*"
        r"(?:claim\s+|measurement\s+)?fields?\s+unsupported\b",
        reasoning,
        re.I,
    )
    if active_measurement_count and denies_any_active_field_evidence:
        problems.append("active_measurement_evidence_denied_for_fields")
    method_atoms = {
        str((atom.get("observation") or {}).get("measurement_method") or ""): (
            evidence_id,
            atom,
        )
        for evidence_id, atom in evidence_store.items()
        if (atom.get("observation") or {}).get("measurement_method")
    }
    coordinated_preference = re.compile(
        r"(?P<first>compare_[a-z0-9_]+)\s*[`*]*\s+(?:and|,)\s*[`*]*"
        r"(?P<second>compare_[a-z0-9_]+).{0,35}?both\s+supports?\s+[`*\"']*"
        r"(?P<value>[abc])",
        re.I,
    )
    for match in coordinated_preference.finditer(reasoning):
        stated = match.group("value").lower()
        for group in ("first", "second"):
            method = match.group(group).lower()
            evidence = method_atoms.get(method)
            if not evidence:
                continue
            evidence_id, atom = evidence
            expected = (atom.get("observation") or {}).get("preferred_video")
            if expected is not None and stated != str(expected).lower():
                problems.append(
                    f"latest_atom_preferred_video_mismatch:{evidence_id}:"
                    f"{stated}!={expected}"
                )
    pair_conflict = re.compile(
        r"(?P<first>compare_[a-z0-9_]+)\s*[`*]*\s+(?:and|versus|vs\.?|with)\s+"
        r"[`*]*(?P<second>compare_[a-z0-9_]+).{0,40}"
        r"(?:conflict|disagree).{0,30}preferred_video",
        re.I,
    )
    for match in pair_conflict.finditer(reasoning):
        first = match.group("first").lower()
        second = match.group("second").lower()
        first_evidence = method_atoms.get(first)
        second_evidence = method_atoms.get(second)
        if not first_evidence or not second_evidence:
            continue
        first_value = (first_evidence[1].get("observation") or {}).get(
            "preferred_video"
        )
        second_value = (second_evidence[1].get("observation") or {}).get(
            "preferred_video"
        )
        if first_value is not None and first_value == second_value:
            problems.append(
                f"measurement_pair_conflict_mismatch:{first}:{second}:"
                f"{first_value}=={second_value}"
            )
    id_pair_conflict = re.compile(
        r"(?P<first>EA-[A-Z0-9_-]+-\d{3})\s*[`*]*\s+(?:and|versus|vs\.?|with)\s+"
        r"[`*]*(?P<second>EA-[A-Z0-9_-]+-\d{3}).{0,45}"
        r"(?:conflict|disagree).{0,35}preferred_video",
        re.I,
    )
    for match in id_pair_conflict.finditer(reasoning):
        first_id = next(
            (
                evidence_id
                for evidence_id in evidence_store
                if evidence_id.lower() == match.group("first").lower()
            ),
            None,
        )
        second_id = next(
            (
                evidence_id
                for evidence_id in evidence_store
                if evidence_id.lower() == match.group("second").lower()
            ),
            None,
        )
        if first_id is None or second_id is None:
            continue
        first_value = (evidence_store[first_id].get("observation") or {}).get(
            "preferred_video"
        )
        second_value = (evidence_store[second_id].get("observation") or {}).get(
            "preferred_video"
        )
        if first_value is not None and first_value == second_value:
            problems.append(
                f"measurement_pair_conflict_mismatch:{first_id}:{second_id}:"
                f"{first_value}=={second_value}"
            )
    for evidence_id, atom in evidence_store.items():
        atom = evidence_store.get(evidence_id) or {}
        observation = atom.get("observation") or {}
        id_match = re.search(re.escape(str(evidence_id)), reasoning, re.I)
        method = str(observation.get("measurement_method") or "")
        method_match = re.search(re.escape(method), reasoning, re.I) if method else None
        reference_matches = [item for item in (id_match, method_match) if item]
        if not observation or not reference_matches:
            continue
        reference_match = min(reference_matches, key=lambda item: item.start())
        fact_text = reasoning[reference_match.start():]
        next_id = re.search(r"EA-[A-Z0-9_-]+-\d{3}", fact_text[len(str(evidence_id)):], re.I)
        cut_points: list[int] = []
        if next_id:
            cut_points.append(len(str(evidence_id)) + next_id.start())
        for other_method in method_atoms:
            if other_method == method:
                continue
            other_match = re.search(re.escape(other_method), fact_text[reference_match.end() - reference_match.start():], re.I)
            if other_match:
                cut_points.append(
                    reference_match.end() - reference_match.start() + other_match.start()
                )
        contrast_match = re.search(
            r"[,;]\s*(?:while|whereas|but|although)\b|\b(?:while|whereas)\b",
            fact_text[reference_match.end() - reference_match.start():],
            re.I,
        )
        if contrast_match:
            contrast_tail = fact_text[
                reference_match.end() - reference_match.start()
                + contrast_match.end():
            ]
            if re.match(
                r"\s*(?:the\s+)?(?:earlier|existing|source|other)\b",
                contrast_tail,
                re.I,
            ):
                cut_points.append(
                    reference_match.end() - reference_match.start()
                    + contrast_match.start()
                )
        if cut_points:
            fact_text = fact_text[: min(cut_points)]
        active = atom.get("active")
        if active is True and re.search(
            r"\b(?:this\s+atom|it|the\s+(?:atom|evidence))\s+"
            r"(?:was|is|has\s+been)\s+(?:dropped|inactive|invalidated|unavailable)\b|"
            r"\b(?:this\s+atom|it|the\s+(?:atom|evidence))\s+is\s+no\s+longer\s+"
            r"(?:active|available)\b",
            fact_text,
            re.I,
        ):
            problems.append(f"visible_atom_active_status_mismatch:{evidence_id}:true")
        elif active is False and re.search(
            r"\b(?:this\s+atom|it|the\s+(?:atom|evidence))\s+is\s+active\b",
            fact_text,
            re.I,
        ):
            problems.append(f"visible_atom_active_status_mismatch:{evidence_id}:false")
        sentence_end = re.search(r"[.!?](?:\s|$)", fact_text)
        atom_fact_text = fact_text[: sentence_end.end()] if sentence_end else fact_text
        decision_rule = str(observation.get("decision_rule") or "")
        abstention_cause = re.search(
            r"abstain\w*[^.!?]{0,80}\bbecause(?P<cause>[^.!?]{0,120})",
            atom_fact_text,
            re.I,
        )
        if (
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            and decision_rule
            in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
            and abstention_cause
            and not re.search(
                r"\border[-_ ]disagreement\b|"
                r"(?:direction|result|comparison)?[^.!?]{0,45}"
                r"(?:disagree|inconsisten|differ|change|flip|fail(?:s|ed)?\s+to\s+survive)"
                r"[^.!?]{0,55}(?:A\s*/\s*B\s+(?:presentation\s+)?orders?|"
                r"presentation\s+orders?|role[- ]?swapp)|"
                r"(?:A\s*/\s*B\s+(?:presentation\s+)?orders?|presentation\s+orders?|"
                r"role[- ]?swapp)[^.!?]{0,55}"
                r"(?:disagree|inconsisten|differ|change|flip|fail(?:s|ed)?\s+to\s+survive)",
                abstention_cause.group("cause"),
                re.I,
            )
        ):
            problems.append(
                f"latest_atom_abstention_cause_omits_decision_rule:{evidence_id}"
            )
        if observation.get("order_consistent") is False and re.search(
            r"\b(?:without|no)\s+(?:observed\s+)?order[-_ ]inconsisten|"
            r"\bnot\s+order[-_ ]inconsisten|\border[-_ ]consistent\s+(?:result|atom|evidence)",
            fact_text,
            re.I,
        ):
            problems.append(
                f"visible_atom_order_consistency_denied:{evidence_id}:false"
            )
        stable_directions = list(
            ((observation.get("routing_state") or {}).get("stable_observed_directions"))
            or []
        )
        if stable_directions and re.search(
            r"\b(?:without|no|lacks?)\s+(?:a\s+)?stable\s+"
            r"(?:observed\s+)?direction\b",
            atom_fact_text,
            re.I,
        ):
            problems.append(
                f"latest_atom_stable_direction_denied:{evidence_id}:"
                f"{','.join(str(value) for value in stable_directions)}"
            )
        declared_support = set(atom.get("supports_fields") or [])
        denies_declared_support = re.search(
            r"\b(?:this\s+atom|it|the\s+(?:atom|evidence))\b.{0,25}"
            r"(?:does\s+not|doesn't|fails?\s+to|cannot)\b.{0,30}"
            r"supports?\b.{0,55}(?:claim\s+)?fields?\b",
            fact_text,
            re.I,
        )
        if declared_support and denies_declared_support:
            problems.append(f"visible_atom_declared_support_denied:{evidence_id}")
        for field, expected_value in observation.items():
            if field in {"preferred_video", "order_consistent"}:
                continue
            if not isinstance(expected_value, (str, int, float, bool)) and expected_value is not None:
                continue
            if isinstance(expected_value, bool):
                value_pattern = r"true|false"
                value_boundary = r"(?![A-Za-z0-9_-])"
            elif isinstance(expected_value, (int, float)):
                value_pattern = r"[-+]?\d+(?:\.\d+)?"
                value_boundary = r"(?![A-Za-z0-9_.-])"
            elif expected_value is None:
                value_pattern = r"null|none"
                value_boundary = r"(?![A-Za-z0-9_-])"
            else:
                value_pattern = r"[A-Za-z][A-Za-z0-9_-]*"
                value_boundary = r"(?![A-Za-z0-9_-])"
            if isinstance(expected_value, str) and re.search(
                rf"[`*]*{re.escape(str(field))}[`*]*\s*(?::|=|\bis\b)\s*"
                rf"[`*]*[\"']{re.escape(expected_value)}[\"'][`*]*",
                atom_fact_text,
                re.I,
            ):
                continue
            explicit_value = re.search(
                rf"[`*]*{re.escape(str(field))}[`*]*\s*(?::|=|\bis\b)\s*"
                rf"[`*\"']*({value_pattern}){value_boundary}",
                atom_fact_text,
                re.I,
            )
            if not explicit_value:
                continue
            raw_value = explicit_value.group(1)
            if raw_value.lower() == "true":
                stated_value: Any = True
            elif raw_value.lower() == "false":
                stated_value = False
            elif raw_value.lower() == "null":
                stated_value = None
            elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw_value):
                stated_value = float(raw_value)
            else:
                stated_value = raw_value
            scalar_matches = (
                abs(float(stated_value) - float(expected_value)) < 1e-6
                if isinstance(stated_value, (int, float))
                and not isinstance(stated_value, bool)
                and isinstance(expected_value, (int, float))
                and not isinstance(expected_value, bool)
                else str(stated_value).lower() == str(expected_value).lower()
            )
            if not scalar_matches:
                problems.append(
                    f"visible_atom_scalar_mismatch:{evidence_id}:{field}:"
                    f"{stated_value}!={expected_value}"
                )
        expected_boxes = observation.get("copied_segments_frames") or []
        box_match = re.search(
            r"\bbox\s*`?\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
            atom_fact_text,
            re.I,
        )
        if box_match and expected_boxes:
            stated_box = [int(item) for item in box_match.groups()]
            if stated_box not in expected_boxes:
                problems.append(
                    f"visible_atom_localization_box_mismatch:{evidence_id}:"
                    f"{stated_box}!={expected_boxes}"
                )
        claims_identity_support = re.search(
            r"supports?\s+(?:the\s+)?(?:task[-_ ]identity|identity)\s+fields"
            r"|supports?.{0,35}sample_key.{0,35}comparison_statement",
            atom_fact_text,
            re.I,
        )
        identity_fields = {"sample_key", "comparison_statement"}
        if claims_identity_support and not identity_fields.issubset(declared_support):
            problems.append(
                f"latest_atom_support_scope_mismatch:{evidence_id}:claimed_identity_fields"
            )
        preferred_match = re.search(
            r"(?:preferred_video\s*[`*]*\s*[:=]\s*[`*\"']*"
            r"|(?:visible\s+)?preference\s+for\s+(?:video\s+)?"
            r"|prefers?\s+(?:video\s+)?|supports?\s+(?:an?\s+)?[`*\"']*)([abc])\b",
            atom_fact_text,
            re.I,
        )
        preferred = observation.get("preferred_video")
        if preferred_match and preferred is not None and preferred_match.group(1).lower() != str(preferred).lower():
            problems.append(
                f"latest_atom_preferred_video_mismatch:{evidence_id}:"
                f"{preferred_match.group(1).lower()}!={preferred}"
            )
        order_match = re.search(
            r"order[-_ ]consisten(?:t|cy)\s*[`*]*\s*"
            r"(?:(?::|=|is|was|of)\s*)?[`*]*\s*(true|false)",
            atom_fact_text,
            re.I,
        )
        expected_order = observation.get("order_consistent")
        asserted_inconsistent = re.search(
            r"order[-_ ]inconsisten\w*", atom_fact_text, re.I
        )
        negated_inconsistency = re.search(
            r"\b(?:without|no)\b[^.!?]{0,35}order[-_ ]inconsisten\w*",
            atom_fact_text,
            re.I,
        )
        asserted_consistent = re.search(
            r"(?:(?:is|was|remains)\s+order[-_ ]consistent"
            r"|order[-_ ]consistent\s+(?:result|intervals?|measurement|comparison))",
            atom_fact_text,
            re.I,
        )
        asserted_unobserved = re.search(
            r"(?:no|without|missing)\s+[`*]*order[-_ ]consistent(?:cy|t)?\s+"
            r"(?:field|value|observation)|"
            r"order[-_ ]consistent(?:cy|t)?\s+(?:is|was|remains)\s+"
            r"(?:unobserved|missing|null|none)",
            atom_fact_text,
            re.I,
        )
        if expected_order is None and (
            order_match
            or (asserted_inconsistent and not negated_inconsistency)
            or asserted_consistent
        ):
            problems.append(
                f"latest_atom_order_consistency_unobserved_but_asserted:{evidence_id}"
            )
        elif asserted_unobserved and expected_order is not None:
            problems.append(
                f"latest_atom_order_consistency_mismatch:{evidence_id}:"
                f"unobserved!={str(bool(expected_order)).lower()}"
            )
        elif order_match and expected_order is not None:
            stated_order = order_match.group(1).lower() == "true"
            if stated_order is not bool(expected_order):
                problems.append(
                    f"latest_atom_order_consistency_mismatch:{evidence_id}:"
                    f"{str(stated_order).lower()}!={str(bool(expected_order)).lower()}"
                )
        elif asserted_inconsistent and not negated_inconsistency and expected_order is True:
            problems.append(
                f"latest_atom_order_consistency_mismatch:{evidence_id}:false!=true"
            )
        elif asserted_consistent and expected_order is False and not (
            order_match and order_match.group(1).lower() == "false"
        ):
            problems.append(
                f"latest_atom_order_consistency_mismatch:{evidence_id}:true!=false"
            )
        has_intervals = bool(observation.get("video_a_interval_s")) and bool(
            observation.get("video_b_interval_s")
        )
        denies_intervals = re.search(
            r"(?:lack(?:s|ed|ing)?|without|has\s+no|there\s+(?:is|are)\s+no)"
            r".{0,45}(?:interval\s+evidence|supporting\s+intervals?|visible\s+intervals?)",
            fact_text,
            re.I,
        )
        if has_intervals and denies_intervals:
            problems.append(f"latest_atom_interval_evidence_denied:{evidence_id}")
        is_active_measurement = atom.get("active") is True and bool(
            observation.get("measurement_method")
        )
        denies_active_measurement = re.search(
            r"(?:"
            r"(?:no|without)\s+(?:currently\s+)?(?:active|visible)\s+measurement\s+evidence"
            r"|missing\s+measurement\s+evidence"
            r")",
            fact_text,
            re.I,
        )
        if is_active_measurement and denies_active_measurement:
            problems.append(f"latest_atom_active_measurement_denied:{evidence_id}")
    if "active_measurement_evidence_denied_for_fields" in problems and any(
        item.startswith("latest_atom_active_measurement_denied:") for item in problems
    ):
        problems.remove("active_measurement_evidence_denied_for_fields")
    return problems


def _generation_retry_feedback(
    problems: list[str],
    *,
    visible_state: dict[str, Any] | None = None,
    arguments: dict[str, Any] | None = None,
) -> str:
    visible_state = visible_state or {}
    arguments = arguments or {}
    guidance: list[str] = []
    for problem in problems:
        if problem == "meta_evidence_overclaims_identity_field_support":
            guidance.append(
                "Do not attribute all six fields to the meta atom: explicitly name the task "
                "atom for sample_key/comparison_statement and the meta atom only for "
                "preferred_video, both intervals, and measurement_method."
            )
        elif problem == "role_invariant_meta_described_as_order_inconsistent":
            guidance.append(
                "State that the derived meta result is order_consistent=true; only its two "
                "source routes are order-inconsistent."
            )
        elif problem == "generation_hit_limit":
            guidance.append(
                "Shorten the reasoning to 45-65 words while keeping the complete native JSON "
                "tool call."
            )
        elif problem.startswith("latest_atom_order_consistency_unobserved_but_asserted:"):
            guidance.append(
                "The named visible EvidenceAtom has no observed order_consistent field. Do not "
                "infer or assert true/false; justify the next method using only fields that are "
                "actually present."
            )
        elif problem == "uniform_timeline_overclaims_motion_or_order_observation":
            guidance.append(
                "`compare_uniform_timeline` inspects coarse uniformly sampled chronological "
                "visual coverage. It does not measure global motion and does not observe A/B "
                "order consistency; describe the route only as a complementary whole-timeline "
                "visual check and leave order_consistent unobserved."
            )
        elif problem == "vcsl_localizer_overclaims_ab_order_symmetry":
            guidance.append(
                "A VCSL copy-segment localizer operates on the query/reference similarity map "
                "and returns temporal localization boxes. It does not run a VidDiff-style A/B "
                "presentation-order symmetry test; describe DP or local alignment in its own "
                "query/reference localization terms."
            )
        elif problem.startswith("latest_created_atom_asserts_absent_field:"):
            _, evidence_id, field = problem.split(":", 2)
            guidance.append(
                f"The latest created atom `{evidence_id}` has no student-visible `{field}` "
                "field. Do not transfer VidDiff comparison fields onto a VCSL or VideoSham "
                "identity atom; cite only properties present in its structured observation."
            )
        elif problem.startswith("visible_evidence_store_denied_nonempty:"):
            guidance.append(
                "The student-visible evidence_store is not empty. Acknowledge the existing "
                "context EvidenceAtoms and distinguish them from the still-missing measurement; "
                "do not say that no atom, observation, evidence, or available EvidenceAtom ID "
                "exists."
            )
        elif problem == "evidence_dropout_misdescribed_as_binding_invalidation":
            guidance.append(
                "This is evidence_dropout: the removed EvidenceAtom ID is invalid. Distinguish "
                "the newly generated active atom from the dropped tombstone and say that only "
                "the new ID is rebound. Do not say the same atom remained valid; that wording "
                "belongs to binding_invalidation."
            )
        elif problem == "dropped_atom_described_as_rebindable":
            guidance.append(
                "Under evidence_dropout, the removed EvidenceAtom ID cannot be rebound or "
                "reactivated. Say that a newly executed measurement must create a distinct "
                "replacement atom; only binding_invalidation permits rebinding the same atom."
            )
        elif problem == "order_consistency_misapplied_to_intervals":
            guidance.append(
                "order_consistent describes whether the comparison direction survives A/B "
                "presentation-order swapping. Do not call intervals order-consistent; intervals "
                "localize supporting time spans and should be described separately."
            )
        elif problem.startswith("nonexistent_tool_name_in_reasoning:"):
            invalid_name = problem.split(":", 1)[1]
            guidance.append(
                f"`{invalid_name}` is not a native tool. Use the exact visible tool name "
                "`bind_evidence` when describing the completed or proposed binding action."
            )
        elif problem.startswith("visible_atom_declared_support_denied:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"`{evidence_id}` visibly declares supports_fields. Do not say the atom itself "
                "fails to support those fields merely because claim bindings are absent. For a "
                "complementary measurement, name the real visible uncertainty—such as low "
                "confidence or cross-method disagreement—while keeping atom support distinct "
                "from claim-to-evidence binding."
            )
        elif problem == "active_measurement_evidence_denied_for_fields":
            guidance.append(
                "The visible state already contains active measurement evidence. Do not say "
                "there is no active/new replacement evidence, a measurement is missing, or that "
                "sensing must recreate field support. "
                "Keep missing claim bindings separate, and justify a complementary measurement "
                "with a real observation uncertainty such as low confidence, order inconsistency, "
                "or cross-method disagreement."
            )
        elif problem.startswith("repair_restoration_candidate_bypassed:"):
            required_tool = problem.split(":", 1)[1]
            guidance.append(
                f"The student-visible repair_restoration_candidates selects `{required_tool}`. "
                "Execute that dropped claim method first to create a distinct active EvidenceAtom; "
                "do not substitute a route whose prior atom is already active."
            )
        elif problem.startswith(
            "measurement_reasoning_omits_visible_observation_uncertainty:"
        ):
            guidance.append(
                "Use the tail-visible measurement_gap_contract. Name at least one listed "
                "observation uncertainty—such as low confidence, order inconsistency, abstention, "
                "or cross-method disagreement—or say the route supplies independent corroboration. "
                "Do not frame sensing as fixing unbound, uncommitted, or unsupported claim fields."
            )
        elif problem.startswith("measurement_tool_policy_substituted_for_observation_gap:"):
            guidance.append(
                "A tool's required or recommended status is a policy choice, not an observation "
                "gap. Use grounded_reasoning_plan.valid_primary_gap and the exact pair relation; "
                "explain what remains uncertain in the visible measurement itself."
            )
        elif problem.startswith("student_visible_measurement_recommendation_bypassed:"):
            required_tool = problem.split(":", 1)[1]
            guidance.append(
                f"Follow measurement_gap_contract.recommended_next_measurement and select "
                f"`{required_tool}`. Its prior student-visible route evidence provides the best "
                "non-abstaining, non-order-inconsistent match for independent corroboration; "
                "do not rerun a visibly weaker substitute route."
            )
        elif problem.startswith("latest_atom_preferred_video_mismatch:"):
            parts = problem.split(":", 2)
            evidence_id = parts[1] if len(parts) > 1 else "the named EvidenceAtom"
            mismatch = parts[2] if len(parts) > 2 else ""
            stated, separator, expected = mismatch.partition("!=")
            if separator:
                guidance.append(
                    f"For `{evidence_id}`, the visible preferred_video is `{expected}`, not "
                    f"`{stated}`. State that exact fact. If the tool call binds a different "
                    "older atom, also cite this latest atom and name its concrete visible "
                    "defect that makes the older atom stronger."
                )
            else:
                guidance.append(
                    f"Correct the stated preferred_video for `{evidence_id}` to match its "
                    "student-visible observation exactly."
                )
        elif problem.startswith("measurement_pair_conflict_mismatch:"):
            parts = problem.split(":")
            first = parts[1] if len(parts) > 1 else "the first method"
            second = parts[2] if len(parts) > 2 else "the second method"
            guidance.append(
                f"Do not say `{first}` and `{second}` conflict on preferred_video: their visible "
                "directions match. Frame the real gap using the listed low confidence, an "
                "abstaining/order-inconsistent alternative, or the need for fresh independent "
                "corroboration."
            )
        elif problem.startswith("latest_atom_stable_direction_denied:"):
            parts = problem.split(":", 2)
            evidence_id = parts[1] if len(parts) > 1 else "the named EvidenceAtom"
            directions = parts[2] if len(parts) > 2 else "the listed value"
            guidance.append(
                f"`{evidence_id}` visibly records stable_observed_directions={directions}. "
                "Do not deny that stable direction. If confidence is low or order_consistent "
                "is unobserved, state that limitation separately despite the visible direction."
            )
        elif problem.startswith("latest_atom_abstention_cause_omits_decision_rule:"):
            guidance.append(
                "The named atom's preferred_video=c follows an order-disagreement abstention "
                "rule. If you say why it abstains, attribute that result to disagreement across "
                "A/B presentation orders; visual cue strength or cross-method routing state is "
                "context, not the direct decision cause."
            )
        elif problem.startswith("latest_atom_"):
            guidance.append(
                "Correct the statement to match the latest visible EvidenceAtom's exact "
                "supports_fields, preferred_video, order_consistent value, and A/B intervals."
            )
        elif problem.startswith("active_measurement_count_mismatch:"):
            guidance.append(
                "Use the visible evidence store's actual number of active measurement atoms; "
                "describe disagreement or insufficiency instead of saying only one exists."
            )
        elif problem.startswith("eligible_tool_count_mismatch:"):
            guidance.append(
                "The visible eligible_tools list contains multiple actions. Describe why the selected "
                "method is the best next choice for this gap, not as the only eligible tool, action, "
                "method, or route."
            )
        elif problem.startswith("chosen_measurement_tool_not_justified:"):
            selected_tool = problem.split(":", 1)[1]
            guidance.append(
                f"Use the literal selected method name `{selected_tool}` in the reasoning and "
                "explain why that method, rather than an unspecified available method, closes "
                "the gap."
            )
        elif problem.startswith("typed_universal_abstention_overridden:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"Use the typed predicate atom `{evidence_id}` for claim creation. Its usable "
                "independent A/B records both report zero sampled violations, while both carry "
                "sparse_sampling_cannot_prove_never; therefore the absolute never claim remains "
                "preferred_video=c under compare_predicate_independent. Do not override that "
                "proof-aware abstention with a coarse route's confidence or vote count."
            )
        elif problem.startswith("measurement_action_denies_need_for_measurement:"):
            guidance.append(
                "A measurement action cannot be justified by unbound, uncommitted, unsupported, "
                "or evidence-lacking claim fields, nor by saying binding, committing, or stopping "
                "is the only remaining step. Replan from the unchanged visible state and make a "
                "visible observation uncertainty or independent corroboration need the primary "
                "unresolved observation gap."
            )
        elif problem.startswith("measurement_result_presupposed_before_execution:"):
            guidance.append(
                "Do not promise an order-consistent or positive result before executing the "
                "measurement. State what the tool will test, using whether/if language, and let "
                "the environment observation determine the outcome."
            )
        elif problem == "reasoning_mentions_generation_audit_artifact":
            guidance.append(
                "Describe only the visible evidence conflict or insufficiency; do not mention "
                "generation attempts, retry feedback, or an audit/count mismatch."
            )
        elif problem == "reasoning_mentions_prompt_scaffolding":
            guidance.append(
                "State the observed evidence facts and decision consequence directly. Do not "
                "mention a required clause, natural-language requirement, revision instruction, "
                "prompt, rubric, or checker in the trajectory reasoning."
            )
        elif problem == "reasoning_mentions_internal_policy_scaffolding":
            guidance.append(
                "Describe the case evidence and the unresolved decision directly. Do not expose "
                "internal policy-view keys such as current_harness_state, eligible_tools, an "
                "execution frontier, executable actions, or student-visible state."
            )
        elif problem.startswith("repeated_reasoning_scaffold:"):
            guidance.append(
                "Keep the same evidence and action, but rewrite the rationale with a different "
                "sentence structure. State the concrete missing fact or transition directly; "
                "do not reuse `the latest visible state`, `the only remaining gap`, or `the "
                "blocking gap` from an earlier turn."
            )
        elif problem.startswith("stock_reasoning_scaffold:"):
            guidance.append(
                "Keep the same eligible action, but justify it from the missing evidence or "
                "state transition instead of saying it is `the only eligible tool`."
            )
        elif problem.startswith("claim_field_count_mismatch:"):
            mismatch = problem.split(":", 1)[1]
            stated, _, actual = mismatch.partition("!=")
            stated = stated.removeprefix("said_")
            guidance.append(
                f"The visible candidate claim has exactly {actual} fields, not {stated}. "
                "Use the schema-derived field count or avoid stating a numeric total; do not "
                "transfer a field count from another dataset."
            )
        elif problem == "reasoning_self_contradicts_field_support":
            guidance.append(
                "Keep evidence support distinct from binding: if all required fields already "
                "have active support, say they are supported but still unbound, never unsupported."
            )
        elif problem == "unbound_claim_method_described_as_bound":
            guidance.append(
                "The claim's measurement_method value may remain visible after dropout, but its "
                "evidence binding is absent. Say the value remains; do not say the claim still "
                "binds the method."
            )
        elif problem == "bindable_fields_described_as_unsupported":
            guidance.append(
                "An executable binding action means compatible active evidence already supports "
                "its fields. Describe them as supported but unbound, not unsupported."
            )
        elif problem == "binding_described_as_assessment_transition":
            guidance.append(
                "The candidate-assessment ledger and evidence partition are already complete. "
                "Describe `bind_evidence` only as creating claim-field-to-EvidenceAtom support "
                "links; it does not close or complete an assessment step."
            )
        elif problem == "claim_creation_described_as_binding_or_commitment":
            guidance.append(
                "`create_candidate_claim` creates field values and obligations only. It does "
                "not bind evidence, mark fields supported, or commit them; describe support as "
                "a later `bind_evidence` transition."
            )
        elif problem.startswith("claim_creation_source_not_identified:"):
            method = problem.split(":", 1)[1]
            guidance.append(
                f"The claim JSON takes measurement values from `{method}`. Name that exact "
                "method or its matching EvidenceAtom in the reasoning and explain why its "
                "visible fields are used; do not leave the source implicit while discussing a "
                "different atom."
            )
        elif problem == "assessment_described_as_binding_field_support":
            guidance.append(
                "`assess_evidence_candidate` records only the current candidate's retain/reject "
                "disposition. Say it closes the pending assessment gap; it does not bind evidence "
                "or create/close field support."
            )
        elif problem.startswith("pending_assessment_described_as_already_recorded:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"`{evidence_id}` is still pending and absent from the recorded assessment "
                "ledger. State the visible retain/reject disposition prospectively; do not say "
                "it was already assessed, retained, rejected, or recorded."
            )
        elif problem.startswith("visible_atom_order_consistency_denied:"):
            evidence_id = problem.split(":", 2)[1]
            guidance.append(
                f"`{evidence_id}` visibly has order_consistent=false. Do not call it free of "
                "order inconsistency or order-consistent; preserve that defect exactly."
            )
        elif problem == "irrelevant_rebinding_constraint_justifies_decision_transition":
            guidance.append(
                "Do not justify a field-decision transition with "
                "`must_reopen_before_rebinding` when no field is awaiting rebinding. For a "
                "commit action, cite the supported fields and their open decision status; the "
                "reopen-before-rebinding constraint is relevant only to an actual repair bind."
            )
        elif problem.startswith("first_use_measurement_described_as_rerun:"):
            selected_tool = problem.split(":", 1)[1]
            guidance.append(
                f"`{selected_tool}` has not been executed in the visible history. Describe it "
                "as a new complementary measurement, not a rerun or repeated action."
            )
        elif problem.startswith("visible_atom_active_status_mismatch:"):
            evidence_id = problem.split(":", 2)[1]
            guidance.append(
                f"EvidenceAtom `{evidence_id}` is the active new atom. Match that exact ID's visible "
                "active status; do not transfer dropped status from a different, similarly named "
                "tombstone. In particular, do not describe an active new atom as dropped, inactive, "
                "invalidated, or unavailable."
            )
        elif problem.startswith("visible_atom_"):
            guidance.append(
                "Correct every explicit field value and localization box to match the named "
                "student-visible EvidenceAtom exactly."
            )
        elif problem == "bind_reasoning_excludes_required_task_atom":
            guidance.append(
                "Align the binding rationale with the native binding_items: say that the task "
                "atom binds identity fields and the meta atom binds comparative fields, or "
                "refer to both together as the two-atom evidence set."
            )
        elif problem.startswith("bind_omits_latest_measurement_rationale:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"The just-executed measurement `{evidence_id}` must be consumed. Either bind "
                "it when compatible, or cite its exact ID and a visible defect explaining why "
                "another atom is selected."
            )
        elif problem == "adjudication_action_not_explained":
            guidance.append(
                "Describe adjudicate_evidence as comparing and resolving the visible "
                "competing atoms before binding, not as a new visual measurement."
            )
        elif problem == "adjudication_gap_misframed_as_binding":
            guidance.append(
                "The adjudication-stage gap is the unresolved evidence partition or incomplete "
                "ledger, not that atoms are unbound. Adjudication selects/rejects; binding happens "
                "only in the following state."
            )
        elif problem.startswith("adjudication_selected_id_not_cited:"):
            evidence_id = problem.split(":", 1)[1]
            frontier = _adjudication_reasoning_frontier(
                visible_state,
                [{"tool_name": "adjudicate_evidence", "arguments": arguments}],
            )
            selected = next(
                (
                    item
                    for item in (frontier or {}).get("selected_support") or []
                    if item.get("evidence_id") == evidence_id
                ),
                {},
            )
            exact_clause = selected.get("required_minimal_grounding_clause")
            if exact_clause:
                guidance.append(
                    "Ground the selection with the following student-visible facts, but express "
                    f"them naturally rather than copying a stock sentence: {exact_clause}"
                )
            else:
                guidance.append(
                    f"Cite selected EvidenceAtom `{evidence_id}` or its visible turn-local "
                    "reasoning_handle and state why it is the support to retain."
                )
        elif problem == "adjudication_selection_not_justified":
            frontier = _adjudication_reasoning_frontier(
                visible_state,
                [{"tool_name": "adjudicate_evidence", "arguments": arguments}],
            )
            exact_clauses = [
                str(item["required_minimal_grounding_clause"])
                for item in (frontier or {}).get("selected_support") or []
                if item.get("required_minimal_grounding_clause")
            ]
            if exact_clauses:
                guidance.append(
                    "Express every selected-support fact naturally before any generic summary; "
                    "do not copy stock sentences: " + " ".join(exact_clauses)
                )
            else:
                guidance.append(
                    "Explicitly say which atom is selected/retained and why its visible facts "
                    "fit the current claim fields."
                )
        elif problem == "adjudication_rejected_alternatives_called_retained":
            guidance.append(
                "Never call rejected alternatives retained. Retain only the selected support; "
                "say the completed ledger supplies the validated rejection reasons for the rest."
            )
        elif problem == "adjudication_complete_ledger_not_referenced":
            guidance.append(
                "Say explicitly that the completed per-candidate assessment ledger already "
                "records each rejected atom's validated defect. Keep the selected support "
                "justification in prose; exact rejected IDs and defect codes remain in the "
                "native adjudication call and need not be repeated."
            )
        elif problem.startswith("adjudication_rejected_id_not_cited:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"Cite rejected EvidenceAtom `{evidence_id}` or its visible turn-local "
                "reasoning_handle and explain its visible defect rather than omitting it."
            )
        elif problem.startswith("adjudication_rejected_defect_not_explained:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"For `{evidence_id}`, verbalize at least one structured rejection reason: "
                "claim-value conflict, A/B order inconsistency, lower confidence, "
                "conflicting abstention, or redundant same-conclusion support."
            )
        elif problem.startswith("adjudication_claim_conflict_field_not_named:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"For `{evidence_id}`, name a claim field that actually differs, such as "
                "measurement_method or an interval, and state its visible mismatch. Do not "
                "say preferred_video conflicts when the evidence and claim values match."
            )
        elif problem.startswith("assessment_retained_uncertainty_called_defect:"):
            guidance.append(
                "This candidate is retained. Describe unobserved order consistency as a remaining "
                "uncertainty or limitation, not as a defect; explain why the visible matching facts "
                "still make the atom usable support."
            )
        elif problem.startswith(
            "assessment_retained_uncertainty_misframed_as_action_gap:"
        ):
            guidance.append(
                "This turn's actionable gap is the pending candidate assessment. An unobserved "
                "order_consistent value may be noted as a limitation, but do not call it the only "
                "remaining gap after retaining the candidate; the next state advances the ledger."
            )
        elif problem == "assessment_claims_full_adjudication_partition_payload":
            guidance.append(
                "This assess_evidence_candidate call records only the current candidate's "
                "disposition and defect codes. Do not say it carries every rejected ID or the "
                "full partition; that belongs only to the later adjudicate_evidence call."
            )
        elif problem.startswith("assessment_claim_conflict_field_not_named:"):
            evidence_id = problem.split(":", 1)[1]
            conflict_fields = _claim_value_conflict_fields(
                visible_state,
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
            claim = next(
                (
                    item
                    for item in visible_state.get("candidate_claims") or []
                    if str(item.get("claim_id") or "")
                    == str(arguments.get("claim_id") or "")
                ),
                {},
            )
            observation = (
                ((visible_state.get("evidence_store") or {}).get(evidence_id) or {}).get(
                    "observation"
                )
                or {}
            )
            if conflict_fields:
                field = conflict_fields[0]
                observed_value = json.dumps(
                    observation.get(field), ensure_ascii=False, separators=(",", ":")
                )
                claim_value = json.dumps(
                    ((claim.get("fields") or {}).get(field) or {}).get("value"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                guidance.append(
                    f"State this concrete mismatch in ordinary prose: `{evidence_id}` has "
                    f"`{field}={observed_value}`, while the claim has "
                    f"`{field}={claim_value}`. Therefore that field conflicts. Do not "
                    "generalize one scalar value to all target fields."
                )
            else:
                guidance.append(
                    f"For `{evidence_id}`, name a claim field that actually differs, such as "
                    "measurement_method or an interval, and state its visible mismatch. Do not "
                    "say preferred_video conflicts when the evidence and claim values match."
                )
        elif problem.startswith(
            "assessment_claim_conflict_misframed_as_support_status:"
        ):
            guidance.append(
                "Unsupported or unbound claim status is not a claim_value_conflict defect. "
                "Explain the conflict only with the named field's differing visible value, "
                "interval, or measurement_method."
            )
        elif problem.startswith(
            "assessment_abstention_cause_confuses_routing_termination:"
        ):
            guidance.append(
                "The visible order_consistent=false and decision_rule cause this candidate's "
                "abstention. all_candidates_exhausted only says the router stopped searching; "
                "do not present routing termination as the causal reason for preferred_video=c."
            )
        elif problem.startswith(
            "assessment_abstention_cause_omits_decision_rule:"
        ):
            guidance.append(
                "When preferred_video=c, order_consistent=false, and the visible decision_rule "
                "is an order-disagreement abstention, attribute the abstention to disagreement "
                "across A/B presentation orders. Pixel strength and cross-method disagreement "
                "may be context, but they are not the direct decision-rule cause."
            )
        elif problem.startswith(
            "assessment_abstention_decision_rule_not_explained:"
        ):
            guidance.append(
                "Preserve the directed cause and effect; do not reverse them. Say naturally: "
                "its direction disagrees across the two A/B presentation orders, so the atom "
                "abstains with preferred_video=c. Then state why that order-inconsistent atom "
                "is rejected."
            )
        elif problem.startswith("assessment_identity_fact_not_stated:"):
            _, evidence_id, field = problem.split(":", 2)
            guidance.append(
                f"For identity/context atom `{evidence_id}`, state the exact visible "
                f"`{field}=value` fact. Do not replace its identity facts with comparison fields."
            )
        elif problem == "assessment_falsely_claims_complete_ledger":
            guidance.append(
                "Pending candidate assessments remain. Name only the current candidate's "
                "unrecorded disposition as the gap; do not call the assessment ledger complete "
                "until pending_evidence_ids is empty."
            )
        elif problem == "assessment_claims_premature_field_support":
            guidance.append(
                "Assessment records retain/reject only. Say the candidate values match the claim, "
                "but do not say claim fields or values are already supported before bind_evidence."
            )
        elif problem == "assessment_atom_called_action":
            guidance.append(
                "The EvidenceAtom is the object being assessed, not an action. Name "
                "assess_evidence_candidate as the eligible action and the atom as its candidate."
            )
        elif problem.startswith("assessment_order_consistency_not_stated:"):
            evidence_id = problem.split(":", 1)[1]
            observation = (
                ((visible_state.get("evidence_store") or {}).get(evidence_id) or {}).get(
                    "observation"
                )
                or {}
            )
            claim = next(
                (
                    item
                    for item in visible_state.get("candidate_claims") or []
                    if str(item.get("claim_id") or "")
                    == str(arguments.get("claim_id") or "")
                ),
                {},
            )
            preferred = json.dumps(
                observation.get("preferred_video"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            claim_preferred = json.dumps(
                (((claim.get("fields") or {}).get("preferred_video") or {}).get("value")),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            guidance.append(
                "Use this complete evidence sentence before explaining the gap: "
                f"`{evidence_id} reports preferred_video={preferred} and "
                f"order_consistent={str(observation.get('order_consistent')).lower()}; I reject "
                f"it because preferred_video={preferred} conflicts with the claim's "
                f"preferred_video={claim_preferred}.` Do not omit either structured fact or "
                "refer to a prompt clause."
            )
        elif problem.startswith("assessment_order_inconsistency_not_stated:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"State directly that `{evidence_id}` reports `order_consistent=false`, then "
                "explain the resulting A/B-order instability when relevant."
            )
        elif problem.startswith("assessment_unobserved_order_not_stated:"):
            evidence_id = problem.split(":", 1)[1]
            guidance.append(
                f"State directly that `{evidence_id}` does not report order consistency "
                "(`order_consistent` is unobserved); do not infer true or false."
            )
        elif problem.startswith("assessment_rejection_defect_not_explained:"):
            evidence_id = problem.split(":", 1)[1]
            conflict_fields = _claim_value_conflict_fields(
                visible_state,
                str(arguments.get("claim_id") or ""),
                evidence_id,
                [str(field) for field in arguments.get("target_fields") or []],
            )
            claim = next(
                (
                    item
                    for item in visible_state.get("candidate_claims") or []
                    if str(item.get("claim_id") or "")
                    == str(arguments.get("claim_id") or "")
                ),
                {},
            )
            observation = (
                ((visible_state.get("evidence_store") or {}).get(evidence_id) or {}).get(
                    "observation"
                )
                or {}
            )
            if "claim_value_conflict" in {
                str(code) for code in arguments.get("defect_codes") or []
            } and conflict_fields:
                field = conflict_fields[0]
                observed_value = json.dumps(
                    observation.get(field), ensure_ascii=False, separators=(",", ":")
                )
                claim_value = json.dumps(
                    ((claim.get("fields") or {}).get(field) or {}).get("value"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                preferred = json.dumps(
                    observation.get("preferred_video"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                order_value = observation.get("order_consistent")
                rendered_order = (
                    "unobserved"
                    if order_value is None
                    else str(order_value).lower()
                )
                guidance.append(
                    "Use one complete evidence sentence before explaining the gap: "
                    f"`{evidence_id} reports preferred_video={preferred} and "
                    f"order_consistent={rendered_order}; I reject it because "
                    f"{field}={observed_value} conflicts with the claim's "
                    f"{field}={claim_value}.` Do not omit either structured fact, generalize "
                    "one value to all fields, or refer to a prompt clause."
                )
            else:
                guidance.append(
                    f"For `{evidence_id}`, verbalize at least one native rejection defect "
                    "using the candidate's concrete visible facts."
                )
        elif problem.startswith("assessment_"):
            evidence_id = problem.rsplit(":", 1)[-1]
            guidance.append(
                f"Assess only `{evidence_id}` in this turn. Cite its exact ID or unique "
                "method, explicitly state preferred_video and whether order_consistent "
                "is true, false, or unobserved, then explain the native retain/reject "
                "disposition using a visible defect when rejecting."
            )
        elif problem.startswith("reasoning_sentence_count:"):
            guidance.append("Use 2-4 complete sentences.")
        elif problem == "reasoning_missing_terminal_punctuation":
            guidance.append(
                "Finish the final reasoning sentence with terminal punctuation before the tool call; "
                "do not leave a trailing connector or incomplete clause."
            )
        elif problem.startswith("reasoning_word_count:"):
            guidance.append(
                "Use 45-120 words; measurement actions may use 40-120 words, "
                "set_field_decision may use 35-120 words, and stop may use 25-120 words."
            )
    return (
        "Revision requirements (satisfy all of them simultaneously): "
        + " ".join(dict.fromkeys(guidance))
    )


def _run_episode(
    *,
    tokenizer: Any,
    model: Any,
    seed_row: dict[str, Any],
    environment: Any,
    tool_registry: list[dict[str, Any]],
    max_turns: int,
    max_length: int,
    max_new_tokens: int,
    include_sft_data: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_generation_attempts: int = 1,
    constrain_eligible_tool_names: bool = False,
    require_evidence_adjudication: bool = False,
    require_progressive_evidence_assessment: bool = False,
    prefer_natural_reasoning: bool = False,
    policy_view_arm: str = "full",
    evidence_donor_bank: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] | None = None,
    turn_checkpoint_callback: Any | None = None,
    harness_setup_callback: Any | None = None,
    task_integrity_instruction: str | None = None,
) -> dict[str, Any]:
    harness = _restore_harness(seed_row)
    harness.environment = environment
    active_tool_registry = copy.deepcopy(tool_registry)
    if (
        require_evidence_adjudication or require_progressive_evidence_assessment
    ) and not any(
        item.get("name") == "adjudicate_evidence" for item in active_tool_registry
    ):
        active_tool_registry.append(copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL))
    if require_progressive_evidence_assessment and not any(
        item.get("name") == "assess_evidence_candidate"
        for item in active_tool_registry
    ):
        active_tool_registry.append(copy.deepcopy(EVIDENCE_ASSESSMENT_TOOL))
    harness.tool_registry = active_tool_registry
    intervention_metadata = (
        harness_setup_callback(harness) if harness_setup_callback is not None else None
    )
    seed_payload = _payload(seed_row)
    system = copy.deepcopy(seed_row["messages"][0])
    if DERIVED_META_POLICY_CONTRACT not in str(system.get("content", "")):
        system["content"] = str(system.get("content", "")) + DERIVED_META_POLICY_CONTRACT
    task = copy.deepcopy(seed_payload["task"])
    if task_integrity_instruction:
        task["integrity_audit_instruction"] = task_integrity_instruction
    history: list[dict[str, Any]] = []
    latest: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    terminal_success = False

    def emit_turn_checkpoint() -> None:
        if turn_checkpoint_callback is None:
            return
        turn_checkpoint_callback(
            {
                "artifact": "student_closed_loop_partial_turn_checkpoint_v1",
                "case_id": harness.contract.contract_id,
                "dataset": seed_row["metadata"]["dataset"],
                "policy_view_arm": policy_view_arm,
                "terminal_success": terminal_success,
                "turn_count": len(turns),
                "turns": copy.deepcopy(turns),
                "current_harness_state": harness.render_state(),
            }
        )

    for turn_index in range(1, max_turns + 1):
        state = harness.render_state()
        _compile_visible_decision_actions(state)
        affordances = state.get("state_action_affordances") or {}
        if affordances.get("executable_binding_actions"):
            affordances["binding_candidates"] = {}
        policy_state, policy_history = _post_meta_policy_view(state, history)
        policy_task = _post_meta_task_view(task, policy_state)
        eligible = list(state.get("eligible_tools") or [])
        if not eligible:
            turns.append({
                "turn": turn_index,
                "status": "no_eligible_action_before_terminal",
            })
            emit_turn_checkpoint()
            break
        executable_actions = list(affordances.get("executable_binding_actions") or [])
        executable_actions.extend(affordances.get("executable_decision_actions") or [])
        executable_actions.extend(
            affordances.get("executable_evidence_assessment_actions") or []
        )
        executable_actions.extend(
            affordances.get("executable_adjudication_actions") or []
        )
        tool_executable_actions = (
            [] if policy_view_arm == "evidence_only" else executable_actions
        )
        tools = build_student_tools(
            harness.claim_schema,
            harness.tool_registry,
            eligible,
            executable_actions=tool_executable_actions,
        )
        if policy_view_arm != "evidence_only":
            tools = constrain_tools_with_visible_invariants(tools, state)
        arm_view = apply_policy_view_arm(
            policy_view_arm,
            dataset=str(seed_row["metadata"]["dataset"]),
            case_id=str(harness.contract.contract_id),
            state=policy_state,
            history=policy_history,
            task=policy_task,
            latest=latest,
            tools=tools,
            donor_bank=evidence_donor_bank,
        )
        policy_state = arm_view["state"]
        policy_history = arm_view["history"]
        policy_task = arm_view["task"]
        policy_latest = arm_view["latest"]
        tools = arm_view["tools"]
        visible_executable_actions = (
            [] if policy_view_arm == "evidence_only" else executable_actions
        )
        turn_max_new_tokens = _turn_generation_budget(eligible, max_new_tokens)
        if require_progressive_evidence_assessment:
            turn_max_new_tokens = _progressive_turn_generation_budget(
                eligible,
                turn_max_new_tokens,
                executable_actions,
            )
        payload = {
            "task": policy_task,
            "current_harness_state": policy_state,
            "past_execution_history": policy_history,
            "latest_environment_observation": policy_latest,
            "compression_contract": (
                PROGRESSIVE_COMPRESSION_CONTRACT
                if require_progressive_evidence_assessment
                else COMPRESSION_CONTRACT
            ),
        }
        if constrain_eligible_tool_names:
            payload["execution_frontier"] = _execution_frontier(
                eligible,
                tools,
                visible_executable_actions,
                {
                    **policy_state,
                    "latest_environment_observation": policy_latest,
                },
            )
            adjudication_frontier = _adjudication_reasoning_frontier(
                policy_state, visible_executable_actions
            )
            if adjudication_frontier is not None:
                payload["evidence_adjudication_frontier"] = adjudication_frontier
            assessment_frontier = _evidence_assessment_reasoning_frontier(
                policy_state, visible_executable_actions
            )
            if assessment_frontier is not None:
                payload["evidence_assessment_frontier"] = assessment_frontier
            evidence_frontier = _evidence_consumption_frontier(policy_state, policy_latest)
            if evidence_frontier is not None:
                payload["evidence_consumption_frontier"] = evidence_frontier
        if require_progressive_evidence_assessment and eligible == [
            "assess_evidence_candidate"
        ]:
            turn_system = {
                "role": "system",
                "content": PROGRESSIVE_ASSESSMENT_SYSTEM,
            }
        elif require_progressive_evidence_assessment and eligible == [
            "adjudicate_evidence"
        ]:
            turn_system = {
                "role": "system",
                "content": PROGRESSIVE_ADJUDICATION_SYSTEM,
            }
        elif _use_progressive_vcsl_localization_system(
            eligible,
            require_progressive_evidence_assessment=require_progressive_evidence_assessment,
        ):
            turn_system = {
                "role": "system",
                "content": PROGRESSIVE_VCSL_LOCALIZATION_SYSTEM,
            }
        elif _use_progressive_measurement_system(
            eligible,
            require_progressive_evidence_assessment=require_progressive_evidence_assessment,
        ):
            turn_system = {"role": "system", "content": PROGRESSIVE_MEASUREMENT_SYSTEM}
        elif _use_post_adjudication_system(
            eligible,
            policy_state,
            require_progressive_evidence_assessment=require_progressive_evidence_assessment,
        ):
            turn_system = {"role": "system", "content": POST_ADJUDICATION_SYSTEM}
        else:
            turn_system = system
        if NATURAL_REASONING_CONTRACT not in str(turn_system.get("content", "")):
            turn_system["content"] = (
                str(turn_system.get("content", "")) + NATURAL_REASONING_CONTRACT
            )
        messages = [
            turn_system,
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        sft_user_content = messages[1]["content"]
        attempt_audit: list[dict[str, Any]] = []
        seen_contract_problems: list[str] = []
        total_generation_seconds = 0.0
        retry_feedback = ""
        pinned_tool_name = _initial_recommended_tool(payload, eligible)
        recommendation_lock_tool = pinned_tool_name
        candidate_accepted = False
        input_ids: list[int] = []
        source_tokens = 0
        generated_text = ""
        generated_tokens = 0
        name: str | None = None
        arguments: dict[str, Any] | None = None
        parseable = False
        schema_valid = False
        eligible_call = False
        reasoning = ""
        arguments_materialized = False
        generation_stopped_after_materializable_tool_name = False
        previous_rejected_signature: tuple[str, tuple[str, ...]] | None = None
        soft_style_fallback: dict[str, Any] | None = None
        for attempt_index in range(1, max(1, max_generation_attempts) + 1):
            arguments_materialized = False
            generation_stopped_after_materializable_tool_name = False
            attempt_system = copy.deepcopy(turn_system)
            attempt_tools = (
                [
                    tool
                    for tool in tools
                    if str((tool.get("function") or {}).get("name"))
                    == pinned_tool_name
                ]
                if pinned_tool_name and retry_feedback
                else tools
            )
            attempt_user = _attempt_user_message(
                messages[1],
                pinned_tool_name=pinned_tool_name,
                retry_feedback=retry_feedback,
            )
            if (
                not retry_feedback
                and len(eligible) == 1
                and eligible[0]
                in {"assess_evidence_candidate", "adjudicate_evidence"}
            ):
                attempt_user = _initial_progressive_decision_message(
                    messages[1], eligible[0]
                )
            attempt_messages = [attempt_system, attempt_user]
            input_ids, source_tokens = _prompt_ids(
                tokenizer, attempt_messages, attempt_tools, max_length
            )
            tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                generation_started = time.perf_counter()
                generation_args: dict[str, Any] = {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                    "max_new_tokens": turn_max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if constrain_eligible_tool_names:
                    generation_args["logits_processor"] = LogitsProcessorList(
                        [
                            EligibleToolNameLogitsProcessor(
                                tokenizer,
                                [pinned_tool_name] if pinned_tool_name else eligible,
                                prompt_length=tensor.shape[1],
                            )
                        ]
                    )
                materializable_stopper: MaterializableToolCallStoppingCriteria | None = None
                if require_progressive_evidence_assessment:
                    materializable_names = [
                        tool_name
                        for tool_name in (
                            [pinned_tool_name]
                            if pinned_tool_name
                            else [
                                str((tool.get("function") or {}).get("name") or "")
                                for tool in attempt_tools
                            ]
                        )
                        if tool_name
                        and _materialize_visible_action_arguments(
                            tool_name,
                            state=policy_state,
                            tools=attempt_tools,
                            executable_actions=visible_executable_actions,
                        )
                        is not None
                    ]
                    if materializable_names:
                        materializable_stopper = MaterializableToolCallStoppingCriteria(
                            tokenizer,
                            list(dict.fromkeys(materializable_names)),
                            prompt_length=tensor.shape[1],
                        )
                        generation_args["stopping_criteria"] = StoppingCriteriaList(
                            [materializable_stopper]
                        )
                if do_sample:
                    generation_args.update(
                        {
                            "temperature": temperature,
                            "top_p": top_p,
                        }
                    )
                generated = model.generate(**generation_args)
                attempt_seconds = time.perf_counter() - generation_started
            total_generation_seconds += attempt_seconds
            generated_tokens = int(generated.shape[1] - len(input_ids))
            generated_text = tokenizer.decode(
                generated[0, len(input_ids):], skip_special_tokens=False
            )
            name, arguments = _parse_generated_call(generated_text)
            raw_call_parseable = name is not None and arguments is not None
            stopped_tool_name = (
                materializable_stopper.matched_tool_name
                if materializable_stopper is not None
                else None
            )
            generation_stopped_after_materializable_tool_name = bool(stopped_tool_name)
            semantic_name = stopped_tool_name or _semantic_tool_name(generated_text, eligible)
            if require_progressive_evidence_assessment:
                candidate_name = name or _generated_tool_name(generated_text) or semantic_name
                if candidate_name and candidate_name in eligible:
                    materialized_arguments = _materialize_visible_action_arguments(
                        candidate_name,
                        state=policy_state,
                        tools=attempt_tools,
                        executable_actions=visible_executable_actions,
                    )
                    if materialized_arguments is not None and not _schema_valid(
                        attempt_tools, candidate_name, arguments
                    ):
                        name = candidate_name
                        arguments = materialized_arguments
                        arguments_materialized = True
            parseable = name is not None and arguments is not None
            schema_valid = _schema_valid(attempt_tools, name, arguments)
            eligible_call = bool(name and name in eligible)
            reasoning = generated_text.split("<tool_call>", 1)[0].strip()
            meta_visible = any(
                (atom.get("observation") or {}).get("measurement_method")
                == META_ABSTENTION_METHOD
                for atom in (policy_state.get("evidence_store") or {}).values()
            )
            reasoning_problems = (
                _preexecution_reasoning_problems(
                    reasoning,
                    name,
                    arguments,
                    meta_visible=meta_visible,
                    visible_state={
                        **policy_state,
                        "latest_environment_observation": policy_latest,
                    },
                )
                if parseable and schema_valid and eligible_call
                else []
            )
            semantic_repairs: list[str] = []
            if any(
                problem.startswith("assessment_abstention_decision_rule_not_explained:")
                for problem in reasoning_problems
            ):
                repaired_reasoning, semantic_repairs = (
                    _repair_typed_order_abstention_causal_edge(
                        reasoning,
                        name,
                        arguments,
                        {
                            **policy_state,
                            "latest_environment_observation": policy_latest,
                        },
                    )
                )
                if semantic_repairs:
                    reasoning = repaired_reasoning
                    if "<tool_call>" in generated_text:
                        generated_text = (
                            reasoning
                            + "\n<tool_call>"
                            + generated_text.split("<tool_call>", 1)[1]
                        )
                    reasoning_problems = _preexecution_reasoning_problems(
                        reasoning,
                        name,
                        arguments,
                        meta_visible=meta_visible,
                        visible_state={
                            **policy_state,
                            "latest_environment_observation": policy_latest,
                        },
                    )
            contract_problems = []
            if not parseable:
                contract_problems.append("tool_call_not_parseable")
            if parseable and not eligible_call:
                contract_problems.append("tool_not_currently_eligible")
            elif parseable and not schema_valid:
                contract_problems.append("tool_arguments_not_schema_valid")
            contract_problems.extend(reasoning_problems)
            style_problems = _repetitive_reasoning_style_problems(
                reasoning,
                [
                    str(turn.get("reasoning") or "")
                    for turn in turns
                    if turn.get("reasoning")
                ],
            )
            limit_problem = _generation_hit_limit_problem(
                generated_tokens,
                turn_max_new_tokens,
                raw_call_parseable=raw_call_parseable,
                arguments_materialized=arguments_materialized,
            )
            if limit_problem:
                contract_problems.append(limit_problem)
            attempt_audit.append(
                {
                    "attempt": attempt_index,
                    "max_new_tokens": turn_max_new_tokens,
                    "problems": contract_problems,
                    "style_problems": style_problems,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": round(attempt_seconds, 3),
                    "action_lock_tool": pinned_tool_name,
                    "raw_call_parseable": raw_call_parseable,
                    "arguments_materialized_from_visible_state": arguments_materialized,
                    "generation_stopped_after_materializable_tool_name": (
                        generation_stopped_after_materializable_tool_name
                    ),
                    "semantic_tool_name": semantic_name,
                    "semantic_repairs": semantic_repairs,
                }
            )
            if not contract_problems:
                if (
                    prefer_natural_reasoning
                    and style_problems
                    and soft_style_fallback is None
                    and attempt_index < max(1, max_generation_attempts)
                ):
                    soft_style_fallback = {
                        "attempt_audit_index": len(attempt_audit) - 1,
                        "input_ids": input_ids,
                        "source_tokens": source_tokens,
                        "generated_text": generated_text,
                        "generated_tokens": generated_tokens,
                        "name": name,
                        "arguments": copy.deepcopy(arguments),
                        "parseable": parseable,
                        "schema_valid": schema_valid,
                        "eligible_call": eligible_call,
                        "reasoning": reasoning,
                        "arguments_materialized": arguments_materialized,
                        "generation_stopped_after_materializable_tool_name": (
                            generation_stopped_after_materializable_tool_name
                        ),
                        "pinned_tool_name": pinned_tool_name,
                    }
                    retry_feedback = _generation_retry_feedback(style_problems)
                    if constrain_eligible_tool_names:
                        pinned_tool_name = name
                    continue
                candidate_accepted = True
                break
            rejected_signature = (generated_text, tuple(contract_problems))
            if rejected_signature == previous_rejected_signature:
                attempt_audit[-1]["deterministic_duplicate_rejection"] = True
                break
            previous_rejected_signature = rejected_signature
            for problem in contract_problems:
                if problem not in seen_contract_problems:
                    seen_contract_problems.append(problem)
            if constrain_eligible_tool_names:
                recommended_retry_tool = _recommended_retry_tool(
                    contract_problems, eligible
                )
                if recommended_retry_tool is not None:
                    recommendation_lock_tool = recommended_retry_tool
                    pinned_tool_name = recommended_retry_tool
                elif (
                    recommendation_lock_tool is not None
                    and name == recommendation_lock_tool
                ):
                    pinned_tool_name = recommendation_lock_tool
                else:
                    pinned_tool_name = _next_action_lock_tool(
                        pinned_tool_name,
                        semantic_name,
                        name,
                        parseable=parseable,
                        eligible_call=eligible_call,
                        problems=contract_problems,
                    )
            retry_feedback = _generation_retry_feedback(
                seen_contract_problems,
                visible_state={
                    **policy_state,
                    "latest_environment_observation": policy_latest,
                },
                arguments=arguments,
            )
            if parseable and not eligible_call:
                eligibility_feedback = _eligibility_retry_feedback(name, eligible)
                if eligibility_feedback:
                    retry_feedback = " ".join(
                        part for part in (retry_feedback, eligibility_feedback) if part
                    )
            elif parseable and not schema_valid:
                schema_feedback = _schema_retry_feedback(attempt_tools, name, arguments)
                if schema_feedback:
                    retry_feedback = " ".join(
                        part for part in (retry_feedback, schema_feedback) if part
                    )

        if not candidate_accepted and soft_style_fallback is not None:
            input_ids = soft_style_fallback["input_ids"]
            source_tokens = soft_style_fallback["source_tokens"]
            generated_text = soft_style_fallback["generated_text"]
            generated_tokens = soft_style_fallback["generated_tokens"]
            name = soft_style_fallback["name"]
            arguments = soft_style_fallback["arguments"]
            parseable = soft_style_fallback["parseable"]
            schema_valid = soft_style_fallback["schema_valid"]
            eligible_call = soft_style_fallback["eligible_call"]
            reasoning = soft_style_fallback["reasoning"]
            arguments_materialized = soft_style_fallback["arguments_materialized"]
            generation_stopped_after_materializable_tool_name = soft_style_fallback[
                "generation_stopped_after_materializable_tool_name"
            ]
            pinned_tool_name = soft_style_fallback["pinned_tool_name"]
            attempt_audit[soft_style_fallback["attempt_audit_index"]][
                "accepted_as_style_fallback_after_rewrite_failures"
            ] = True
            candidate_accepted = True

        if not candidate_accepted:
            turns.append({
                "turn": turn_index,
                "status": "generation_contract_failure",
                "visible_eligible_tools": eligible,
                "generated_tool": name,
                "generated_arguments": arguments,
                "parseable": parseable,
                "schema_valid": schema_valid,
                "eligible": eligible_call,
                "prompt_tokens_source": source_tokens,
                "prompt_tokens_kept": len(input_ids),
                "generated_text": generated_text,
                "generation_attempts": len(attempt_audit),
                "generation_attempt_audit": attempt_audit,
                "generation_seconds": round(total_generation_seconds, 3),
                "eligible_tool_name_constrained": constrain_eligible_tool_names,
                "evidence_adjudication_enabled": (
                    require_evidence_adjudication
                    or require_progressive_evidence_assessment
                ),
                "progressive_evidence_assessment_enabled": (
                    require_progressive_evidence_assessment
                ),
                "action_lock_tool": pinned_tool_name,
                "policy_view_arm": policy_view_arm,
                "evidence_semantic_swaps_visible": arm_view["evidence_swaps"],
            })
            emit_turn_checkpoint()
            break
        call_id = f"student-free-run-{turn_index:03d}"
        internal_arguments = _normalize_harness_params(name, arguments)
        harness.record_policy_call(name, internal_arguments, call_id, generated_text.split("<tool_call>", 1)[0].strip())
        try:
            result = harness.execute_tool(name, internal_arguments, call_id)
        except (ValueError, RuntimeError) as error:
            result = {
                "status": "rejected",
                "authority": "tool_or_harness_precondition",
                "error": str(error),
            }
            harness.record_rejected_call(name, call_id, result)
        latest = _result_summary(result)
        history.append({
            "step_index": turn_index - 1,
            "past_action": {
                "executed_tool": name,
                "past_input": copy.deepcopy(arguments),
            },
            "environment_outcome": copy.deepcopy(latest),
        })
        turn_record = {
            "turn": turn_index,
            "status": result.get("status"),
            "visible_eligible_tools": eligible,
            "generated_tool": name,
            "generated_arguments": arguments,
            "prompt_tokens_source": source_tokens,
            "prompt_tokens_kept": len(input_ids),
            "generated_tokens": generated_tokens,
            "turn_max_new_tokens": turn_max_new_tokens,
            "arguments_materialized_from_visible_state": arguments_materialized,
            "generation_stopped_after_materializable_tool_name": (
                generation_stopped_after_materializable_tool_name
            ),
            "generation_attempts": len(attempt_audit),
            "generation_attempt_audit": attempt_audit,
            "generation_seconds": round(total_generation_seconds, 3),
            "eligible_tool_name_constrained": constrain_eligible_tool_names,
            "evidence_adjudication_enabled": (
                require_evidence_adjudication
                or require_progressive_evidence_assessment
            ),
            "progressive_evidence_assessment_enabled": (
                require_progressive_evidence_assessment
            ),
            "action_lock_tool": pinned_tool_name,
            "policy_view_arm": policy_view_arm,
            "evidence_semantic_swaps_visible": arm_view["evidence_swaps"],
            "reasoning": reasoning,
            "environment_outcome": latest,
        }
        if include_sft_data:
            turn_record["sft_datum"] = {
                "messages": [
                    copy.deepcopy(turn_system),
                    {
                        "role": "user",
                        "content": sft_user_content,
                    },
                    {
                        "role": "assistant",
                        "content": reasoning,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": copy.deepcopy(arguments),
                                },
                            }
                        ],
                    },
                ],
                "tools": copy.deepcopy(tools),
                "metadata": {
                    "dataset": seed_row["metadata"]["dataset"],
                    "source_seed_row_id": seed_row["metadata"]["row_id"],
                    "turn_index": turn_index,
                    "target_tool": name,
                    "teacher_action_used": False,
                    "oracle_or_verifier_used": False,
                    "generation_source": "accepted_student_closed_loop_rollout",
                },
            }
        turns.append(turn_record)
        if name == "stop" and bool((result.get("stop_decision") or {}).get("allowed")):
            terminal_success = True
            emit_turn_checkpoint()
            break
        emit_turn_checkpoint()
    final_state = harness.render_state()
    rejection_count = sum(item.get("status") == "rejected" for item in turns)
    return {
        "case_id": harness.contract.contract_id,
        "dataset": seed_row["metadata"]["dataset"],
        "policy_view_arm": policy_view_arm,
        "intervention_metadata": copy.deepcopy(intervention_metadata),
        "teacher_actions_used_during_generation": False,
        "verifier_or_dataset_label_used_during_generation": False,
        "turn_count": len(turns),
        "terminal_success": terminal_success,
        "rejection_count": rejection_count,
        "claim_count": len(final_state.get("candidate_claims") or []),
        "final_terminal": bool(final_state.get("terminal")),
        "final_stop_decision": final_state.get("stop_decision"),
        "final_state": final_state,
        "tool_sequence": [item.get("generated_tool") for item in turns if item.get("generated_tool")],
        "turns": turns,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    sft_rows = {
        str(item["metadata"]["row_id"]): item for item in _jsonl(args.sft)
    }
    clean_builders = {
        "VideoSham": (
            VIDEO_FIRST,
            lambda: _video_environment(sft_rows),
            MULTIFAMILY_TOOL_REGISTRY,
        ),
        "VCSL": (VCSL_FIRST, _vcsl_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (
            VIDDIFF_FIRST,
            lambda: VidDiffAdaptiveMeasurementEnvironment(_viddiff_item("surgery_70")),
            VIDDIFF_TOOL_REGISTRY,
        ),
    }
    repair_builders = {
        "VideoSham": (
            VIDEO_REPAIR_FIRST,
            lambda: _video_repair_environment(sft_rows),
            MULTIFAMILY_TOOL_REGISTRY,
        ),
        "VCSL": (VCSL_REPAIR_FIRST, _vcsl_repair_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (
            VIDDIFF_REPAIR_FIRST,
            _viddiff_repair_environment,
            VIDDIFF_TOOL_REGISTRY,
        ),
    }
    builders = clean_builders if args.mode == "clean" else repair_builders
    selected = args.dataset or list(builders)
    policy_view_arm = str(getattr(args, "policy_view_arm", "full"))
    evidence_donor_bank = (
        build_evidence_donor_bank(
            list(_jsonl(getattr(args, "evidence_donor_sft", DEFAULT_EVIDENCE_DONOR_SFT)))
        )
        if policy_view_arm == "shuffled_evidence"
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(
        args.model,
        args.adapter,
        load_in_4bit=not bool(getattr(args, "bf16_inference", False)),
    )
    episodes = []
    for dataset in selected:
        row_id, environment_builder, registry = builders[dataset]
        episode = _run_episode(
                tokenizer=tokenizer,
                model=model,
                seed_row=sft_rows[row_id],
                environment=environment_builder(),
                tool_registry=registry,
                max_turns=args.max_turns,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                policy_view_arm=policy_view_arm,
                evidence_donor_bank=evidence_donor_bank,
            )
        episodes.append(episode)
        write_json(
            args.output.parent
            / f"student_closed_loop_checkpoint_{args.mode}_{dataset.lower()}_{policy_view_arm}.json",
            episode,
        )
    return {
        "audit": f"student_closed_loop_three_dataset_{args.mode}_canary",
        "episode_mode": args.mode,
        "adapter": str(args.adapter),
        "generation_mode": "deterministic_greedy_local_model",
        "policy_view_arm": policy_view_arm,
        "teacher_actions_used_during_generation": False,
        "summary": {
            "episodes": len(episodes),
            "terminal_successes": sum(item["terminal_success"] for item in episodes),
            "episodes_with_rejection": sum(item["rejection_count"] > 0 for item in episodes),
            "all_terminal_success": all(item["terminal_success"] for item in episodes),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("clean", "repair"), default="clean")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("VideoSham", "VCSL", "VidDiffBench"),
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--policy-view-arm", choices=ARMS, default="full")
    parser.add_argument(
        "--evidence-donor-sft", type=Path, default=DEFAULT_EVIDENCE_DONOR_SFT
    )
    parser.add_argument(
        "--bf16-inference",
        action="store_true",
        help="Load the local model in BF16 instead of 4-bit NF4 for faster generation when VRAM permits.",
    )
    args = parser.parse_args()
    result = run(args)
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for episode in result["episodes"]:
        print(json.dumps({
            "dataset": episode["dataset"],
            "terminal_success": episode["terminal_success"],
            "rejection_count": episode["rejection_count"],
            "tool_sequence": episode["tool_sequence"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
