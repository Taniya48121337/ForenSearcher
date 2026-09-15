"""Control-reduced, evidence-sensitive teacher generation for UTQR-7.

This module does not alter the environment or verifier.  It wraps an existing
teacher transport, removes answer-like control affordances from the policy view,
samples several label-blind action proposals, and retains an audit explaining
why the selected proposal is the best grounded decision note.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict
from typing import Any

from llm_teacher_rollout import TeacherAction
from sft_learning_value import supervision_profile


COMPILED_CONTROL_KEYS = {
    "state_action_affordances",
    "execution_frontier",
    "evidence_adjudication_frontier",
    "evidence_assessment_frontier",
    "repair_restoration_candidates",
    "executable_binding_actions",
    "executable_decision_actions",
    "executable_evidence_assessment_actions",
    "executable_adjudication_actions",
    "compatible_evidence_ids_by_field",
    "binding_candidates",
    "meaningful_decision_transitions",
    "recommended_tool",
    "required_tool",
    "native_action_templates",
    "native_action_templates_reference",
}

REPAIR_RECIPE_KEYS = {
    "must_reopen_before_rebinding",
    "dropped_evidence_ids",
    "unsupported_fields",
    "affected_fields",
    "repair_operator",
    "restoration_method",
}

# These operations maintain or inspect harness state.  Every other executable
# environment function is treated as a semantic evidence measurement.  Keeping
# this as a negative list makes the audit cover newly added dataset tools by
# default instead of silently treating them as non-semantic.
NON_MEASUREMENT_TOOL_NAMES = {
    "create_candidate_claim",
    "challenge_evidence",
    "bind_evidence",
    "set_field_decision",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "stop",
}

# A categorical routing answer tells the policy which evidence branch to take,
# even when no oracle label is present.  Neutral diagnostics (durations,
# confidence, coverage) are intentionally not included.
CATEGORICAL_ROUTER_KEYS = {
    "candidate_modalities",
    "primary_modality",
    "recommended_variant",
    "recommended_localizer_tool",
    "recommended_tool",
    "required_tool",
    "selected_tool",
}

PROCESS_ARTIFACTS = (
    "the prompt requires",
    "the rubric requires",
    "the checker requires",
    "only eligible tool",
    "only eligible action",
    "copy these arguments exactly",
    "teacher target",
    "hidden label",
    "oracle",
    "verifier",
)

EVIDENCE_ID = re.compile(r"EA-[A-Za-z0-9_-]+")
WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
STOCK_REASONING_PHRASES = (
    "the latest observation",
    "the latest visible",
    "the visible state",
    "the visible change",
    "the remaining gap",
    "the blocking gap",
    "the remaining blocker",
    "what remains",
    "what still blocks",
    "the accepted assessment",
    "the accepted assessments",
    "the adjudication",
    "the bind step",
    "the direct next step",
    "that matters because",
    "is warranted now",
    "warranted next step",
    "the warranted next action",
)

STYLE_FRAME_MARKERS = {
    "gap_frame": (" gap", "what remains", "what still blocks"),
    "warranted_frame": ("warranted",),
    "that_matters_frame": ("that matters because",),
    "next_action_frame": ("next step", "next action"),
    "rather_than_frame": ("rather than",),
    "no_conflict_inventory": (
        "no competing",
        "no conflicting",
        "no visible role",
        "no visible task",
        "no open obligation",
    ),
}

# Method-fit scoring is deliberately based on the student-visible tool
# descriptions and pre-action diagnostics.  These are lexical guardrails, not
# a hidden router: no tool is assigned a preferred dataset-specific score.
METHOD_CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "already",
    "between",
    "by",
    "compare",
    "comparison",
    "current",
    "detector",
    "execute",
    "for",
    "from",
    "inside",
    "it",
    "label",
    "localizer",
    "measure",
    "measurement",
    "media",
    "method",
    "never",
    "observation",
    "of",
    "on",
    "pair",
    "real",
    "report",
    "result",
    "run",
    "scope",
    "source",
    "suspect",
    "the",
    "this",
    "to",
    "tool",
    "use",
    "video",
    "visible",
    "when",
    "with",
    "also",
    "another",
    "are",
    "based",
    "can",
    "do",
    "does",
    "each",
    "even",
    "into",
    "is",
    "may",
    "name",
    "new",
    "no",
    "not",
    "only",
    "over",
    "reasoning",
    "resulting",
    "style",
    "that",
    "their",
    "then",
    "true",
    "using",
    "were",
    "whether",
    "without",
    "after",
    "action",
    "all",
    "any",
    "changes",
    "benchmark",
    "blind",
    "brief",
    "details",
    "different",
    "emit",
    "executed",
    "explicitly",
    "forcing",
    "full",
    "given",
    "initial",
    "known",
    "limitation",
    "potentially",
    "produce",
    "reads",
    "released",
    "relevant",
    "retained",
    "say",
    "route",
    "routes",
    "supported",
    "test",
    "two",
    "treat",
    "under",
    "whole",
    "winner",
}

DIAGNOSTIC_KEY_STOPWORDS = {
    "arguments",
    "call_id",
    "content",
    "evidence_atoms",
    "evidence_id",
    "execution_id",
    "harness_state",
    "name",
    "observation_id",
    "output",
    "parameters",
    "provenance",
    "reasoning",
    "result",
    "role",
    "status",
    "tool_name",
    "type",
}

CONTRAST_MARKERS = (
    "rather than",
    "instead of",
    "unlike",
    "whereas",
    "compared with",
    "compared to",
    "alternative",
    "over ",
    "而不是",
    "相比",
    "区别于",
)
DIRECTIONAL_CONTRAST_MARKERS = (
    "rather than",
    "instead of",
    "unlike",
    "whereas",
    "compared with",
    "compared to",
    "而不是",
    "相比",
    "区别于",
)

METHOD_LINK_MARKERS = (
    "because",
    "given",
    "therefore",
    "so ",
    "suited",
    "appropriate",
    "matches",
    "fit",
    "preserve",
    "resolve",
    "test",
    "distinguish",
    "由于",
    "因为",
    "适合",
    "匹配",
)


def _style_frame_signature(note: str) -> set[str]:
    lowered = str(note or "").lower()
    signature = {
        name
        for name, markers in STYLE_FRAME_MARKERS.items()
        if any(marker in lowered for marker in markers)
    }
    if re.search(r"\bbecause\b", lowered):
        signature.add("because_connector")
    if re.search(r"\bso\b", lowered):
        signature.add("so_connector")
    if re.search(r"\bsince\b", lowered):
        signature.add("since_connector")
    return signature


CAUSAL_SYSTEM_PROMPT = (
    "You are a careful evidence analyst acting inside an executable video-tool environment. "
    "Choose exactly one available harness function per turn. In the first turn after a tool creates EvidenceAtoms, "
    "name at least one newly created EvidenceAtom ID and state one concrete semantic result before choosing the next action. "
    "In 2-4 sentences, state the visible fact that changed "
    "or preserved your belief and explain why the selected operation follows from it. For a measurement or localization "
    "operation, name the unresolved evidence question, mention a plausible alternative when the visible evidence "
    "supports one, and explain how the measurement can discriminate without presuming its result. For claim creation, "
    "evidence binding, field decisions, adjudication, or stop, explain the exact evidence match or conflict, the state "
    "consequence, and which task or field invariants are preserved; do not invent a hypothetical uncertainty merely to "
    "fit a template. Before creating or binding a claim, verify "
    "that every EvidenceAtom belongs to the current task identity, entity pair, role assignment, statement, and scope. "
    "Treat an identity mismatch, stale active atom, contradictory rationale/value, partial result, or empty result as "
    "an ordinary evidence-integrity problem: use challenge_evidence to identify the suspect atom, select only the "
    "minimally affected claim fields, and request a fresh context check or measurement, or abstain, without waiting "
    "for a repair hint. The availability of challenge_evidence "
    "is not itself evidence that any atom is wrong; choose it only from visible semantic content. "
    "Immediately before stop, compare each committed claim value with the observation in every bound active "
    "EvidenceAtom, and compare its provenance, entity pair, role assignment, and interval against the current task "
    "contract. Active or supported status is not proof of semantic consistency; any visible mismatch blocks stop. "
    "A task, entity, role, or scope mismatch also blocks claim creation, binding, and commitment until compatible "
    "evidence is measured or the policy abstains. When only a subset of bound values conflicts, reopen and repair "
    "exactly that minimal field subset; preserve independently supported fields. A rejected tool call supplies no "
    "negative evidence and must not be used to justify a later claim or stop. "
    "For assess_evidence_candidate, a retained candidate must always use defect_codes=[]; defect codes belong only "
    "to rejected candidates and must be visibly supported. For assess_evidence_candidate and adjudicate_evidence, "
    "copy target_fields exactly from the visible evidence_adjudication.requirement.target_fields; do not broaden the "
    "conflict merely because an atom supports additional fields. "
    "Write like a compact analyst notebook, not a form: vary sentence openings and organization with the situation; "
    "do not repeatedly use stock frames such as 'the latest observation', 'the remaining gap', or 'the next step'. "
    "Do not force every note into a fact-then-'remaining gap'-then-tool template. Depending on the turn, lead with a "
    "contrast between atoms, a consequence of the last result, a concrete uncertainty, a rejected alternative, or "
    "the evidential reason the action is now warranted. Avoid starting consecutive notes with 'The ...'. "
    "Describe harness mechanics only when recovering from an actual returned tool error; otherwise state the "
    "evidence question in ordinary analytical language. "
    "Do not infer semantic answers from the number of available tools, schema singleton values, compiled actions, or "
    "restoration recipes. Do not mention prompts, rubrics, hidden labels, oracle data, or verifier results. The "
    "environment alone creates observations and EvidenceAtoms; never invent either."
)


def _json_object(value: Any) -> Any:
    if not isinstance(value, str):
        return copy.deepcopy(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _sanitize_value(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        revised: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name in COMPILED_CONTROL_KEYS or name in REPAIR_RECIPE_KEYS:
                continue
            if name in {"const", "default", "examples"}:
                continue
            revised[name] = _sanitize_value(child, path=(*path, name))
        return revised
    if isinstance(value, list):
        return [_sanitize_value(item, path=path) for item in value]
    return copy.deepcopy(value)


def sanitize_teacher_transcript(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove compiled control answers from JSON-bearing transcript items."""

    revised: list[dict[str, Any]] = []
    for item in input_items:
        row = copy.deepcopy(item)
        if row.get("role") == "system":
            row["content"] = CAUSAL_SYSTEM_PROMPT
        for key in ("content", "output", "arguments"):
            if key not in row:
                continue
            parsed = _json_object(row[key])
            if isinstance(parsed, (dict, list)):
                row[key] = json.dumps(
                    _sanitize_value(parsed), ensure_ascii=False, separators=(",", ":")
                )
        revised.append(row)
    return revised


def _recent_reasoning(input_items: list[dict[str, Any]], limit: int = 4) -> list[str]:
    notes: list[str] = []
    for item in input_items:
        if item.get("role") == "assistant" and str(item.get("content") or "").strip():
            notes.append(str(item.get("content")))
            continue
        if item.get("type") == "function_call":
            arguments = _json_object(item.get("arguments"))
            if isinstance(arguments, dict) and str(arguments.get("reasoning") or "").strip():
                notes.append(str(arguments["reasoning"]))
    return notes[-limit:]


def _add_style_memory(
    input_items: list[dict[str, Any]], recent_notes: list[str]
) -> list[dict[str, Any]]:
    if not recent_notes:
        return input_items
    repeated = sorted(
        phrase
        for phrase in STOCK_REASONING_PHRASES
        if any(phrase in note.lower() for note in recent_notes)
    )
    revised = copy.deepcopy(input_items)
    openings = []
    for note in recent_notes:
        words = WORD.findall(note.lower())
        if words:
            openings.append(" ".join(words[:4]))
    reminder = " Style memory for this turn:"
    if repeated:
        reminder += " recent notes already used " + ", ".join(
            repr(item) for item in repeated
        ) + "; do not reuse any of those gap/state frames."
    if openings:
        reminder += " Recent sentence openings include " + ", ".join(
            repr(item) for item in openings[-3:]
        ) + "; choose a different opening and sentence organization."
    recent_frames = [_style_frame_signature(note) for note in recent_notes]
    used_frames = sorted(set().union(*recent_frames)) if recent_frames else []
    if used_frames:
        reminder += (
            " Recent notes already used these discourse frames: "
            + ", ".join(used_frames)
            + ". Do not reuse any high-salience frame from that list. In particular: next_action_frame bans "
            "'next step/action/move'; rather_than_frame bans 'rather than' and 'instead'; no_conflict_inventory "
            "bans another list beginning 'no competing/conflicting...'; gap_frame bans gap/blocker language; and "
            "warranted_frame bans 'warranted'. Vary connectors as well."
        )
    reminder += (
        " Begin from the specific evidence contrast, consequence, uncertainty, or rejected alternative that matters "
        "now. Preserve the analytical content without narrating a form."
    )
    if revised and revised[0].get("role") == "system":
        revised[0]["content"] = str(revised[0].get("content") or "") + reminder
    return revised


def _sanitize_claim_field_values(tool: dict[str, Any]) -> None:
    if tool.get("name") != "create_candidate_claim":
        return
    properties = (
        tool.get("parameters", {})
        .get("properties", {})
        .get("field_values", {})
        .get("properties", {})
    )
    for field_schema in properties.values():
        enum = field_schema.get("enum")
        if isinstance(enum, list) and len(enum) == 1:
            field_schema.pop("enum", None)
            field_schema.setdefault("type", "string")
        for name in ("const", "default", "examples"):
            field_schema.pop(name, None)


def sanitize_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep executable interfaces while removing singleton semantic answers."""

    revised = [_sanitize_value(tool) for tool in tools]
    for tool in revised:
        _sanitize_claim_field_values(tool)
        description = str(tool.get("description") or "")
        if any(artifact in description.lower() for artifact in PROCESS_ARTIFACTS):
            tool["description"] = "Execute this harness operation when its ordinary interface preconditions hold."
    return revised


def _walk(value: Any, *, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    rows = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_walk(child, path=(*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_walk(child, path=(*path, str(index))))
    return rows


def is_semantic_measurement_tool(name: str) -> bool:
    """Return whether an executable tool gathers task-semantic evidence."""

    normalized = str(name or "").strip()
    return bool(normalized) and normalized not in NON_MEASUREMENT_TOOL_NAMES and not normalized.startswith(
        "inspect_"
    )


def control_plane_exposure_audit(
    input_items: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    parsed_items = []
    for item in input_items:
        parsed = copy.deepcopy(item)
        for key in ("content", "output", "arguments"):
            if key in parsed:
                parsed[key] = _json_object(parsed[key])
        parsed_items.append(parsed)
    compiled_hits: list[str] = []
    repair_hits: list[str] = []
    router_hits: list[str] = []
    for path, value in _walk(parsed_items):
        if path and path[-1] in COMPILED_CONTROL_KEYS:
            compiled_hits.append(".".join(path))
        if path and path[-1] in REPAIR_RECIPE_KEYS:
            repair_hits.append(".".join(path))
        if path and path[-1] in CATEGORICAL_ROUTER_KEYS and value not in (None, "", [], {}):
            router_hits.append(".".join(path))
    singleton_fields: list[str] = []
    fixed_values: list[str] = []
    for tool in tools:
        if tool.get("name") != "create_candidate_claim":
            continue
        fields = (
            tool.get("parameters", {})
            .get("properties", {})
            .get("field_values", {})
            .get("properties", {})
        )
        for field, schema in fields.items():
            if isinstance(schema.get("enum"), list) and len(schema["enum"]) == 1:
                singleton_fields.append(str(field))
            if "const" in schema or "default" in schema:
                fixed_values.append(str(field))
    semantic_tools = sorted(
        {
            str(tool.get("name"))
            for tool in tools
            if is_semantic_measurement_tool(str(tool.get("name") or ""))
        }
    )
    return {
        "compiled_semantic_actions_visible": bool(compiled_hits),
        "compiled_control_paths": sorted(set(compiled_hits)),
        "repair_recipe_visible": bool(repair_hits),
        "repair_recipe_paths": sorted(set(repair_hits)),
        "singleton_target_values_visible": bool(singleton_fields),
        "singleton_target_fields": sorted(set(singleton_fields)),
        "fixed_target_fields": sorted(set(fixed_values)),
        "full_target_call_materializable": bool(singleton_fields or fixed_values),
        "semantic_measurement_tool_names": semantic_tools,
        "semantic_measurement_choice_count": len(semantic_tools),
        "categorical_router_answer_visible": bool(router_hits),
        "categorical_router_answer_paths": sorted(set(router_hits)),
    }


def _visible_evidence_ids(input_items: list[dict[str, Any]]) -> set[str]:
    return set(EVIDENCE_ID.findall(json.dumps(input_items, ensure_ascii=False)))


def _latest_created_evidence_ids(input_items: list[dict[str, Any]]) -> set[str]:
    """Return IDs from only the immediately preceding tool observation."""

    for item in reversed(input_items):
        if item.get("type") != "function_call_output" and item.get("role") != "tool":
            continue
        parsed = _json_object(item.get("output", item.get("content")))
        if not isinstance(parsed, dict):
            return set()
        direct = {str(value) for value in parsed.get("created_evidence_ids") or []}
        atoms = {
            str(atom.get("evidence_id"))
            for atom in parsed.get("evidence_atoms") or []
            if isinstance(atom, dict) and atom.get("evidence_id")
        }
        return direct.union(atoms)
    return set()


def _latest_harness_state(input_items: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for item in input_items:
        for key in ("content", "output"):
            parsed = _json_object(item.get(key))
            if not isinstance(parsed, dict):
                continue
            candidate = parsed.get("harness_state")
            if isinstance(candidate, dict):
                latest = candidate
            elif "task_contract" in parsed and "evidence_store" in parsed:
                latest = parsed
    return latest


def visible_integrity_issues(input_items: list[dict[str, Any]]) -> list[str]:
    """Derive semantic conflicts from the same state visible to the policy."""

    state = _latest_harness_state(input_items)
    if not state:
        return []
    issues: set[str] = set()
    contract = state.get("task_contract") or {}
    contract_id = str(contract.get("contract_id") or "")
    assets = contract.get("visible_assets") or {}
    scope = assets.get("comparison_scope_s")
    evidence = state.get("evidence_store") or {}
    bindings = state.get("field_bindings") or {}
    bound_fields_by_evidence: dict[str, set[str]] = {}
    for claim_bindings in bindings.values():
        for field_name, binding in (claim_bindings or {}).items():
            for evidence_id in (binding or {}).get("support_ids") or []:
                bound_fields_by_evidence.setdefault(str(evidence_id), set()).add(
                    str(field_name)
                )
    rejected_fields_by_evidence: dict[str, set[str]] = {}
    for adjudication in (state.get("evidence_adjudications") or {}).values():
        target_fields = {str(item) for item in adjudication.get("target_fields") or []}
        for rejected in adjudication.get("rejected_evidence") or []:
            rejected_fields_by_evidence.setdefault(
                str(rejected.get("evidence_id")), set()
            ).update(target_fields)
    for evidence_id, atom in evidence.items():
        provenance = atom.get("provenance") or {}
        for key in ("task_contract_id", "contract_id"):
            value = provenance.get(key)
            if value and contract_id and str(value) != contract_id:
                issues.add(f"{evidence_id}:task_identity_mismatch")
        if any("MISMATCH" in str(provenance.get(key) or "") for key in ("entity_pair_id", "scope_identity")):
            issues.add(f"{evidence_id}:entity_or_scope_mismatch")
        if "swapped" in str(provenance.get("role_assignment") or "").lower():
            issues.add(f"{evidence_id}:role_assignment_mismatch")
        observation = atom.get("observation") or {}
        if isinstance(scope, list) and len(scope) == 2:
            low, high = map(float, scope)
            for key, value in observation.items():
                if "interval" not in str(key) or not isinstance(value, list) or len(value) != 2:
                    continue
                # Interval defects are field-local. Once an atom is rejected for that
                # interval and the field is rebound elsewhere, its unrelated semantic
                # support must not trigger broad reopening of the whole claim.
                if (
                    bindings
                    and str(key) not in bound_fields_by_evidence.get(str(evidence_id), set())
                ):
                    continue
                if str(key) in rejected_fields_by_evidence.get(str(evidence_id), set()):
                    continue
                try:
                    start, end = map(float, value)
                except (TypeError, ValueError):
                    continue
                if start < low or end > high or start > end:
                    issues.add(f"{evidence_id}:{key}_outside_scope")

    claims = state.get("candidate_claims") or []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        fields = claim.get("fields") or {}
        claim_bindings = bindings.get(claim_id) or {}
        for field_name, binding in claim_bindings.items():
            claim_field = fields.get(field_name) or {}
            claim_value = claim_field.get("value")
            for evidence_id in binding.get("support_ids") or []:
                observation = (evidence.get(evidence_id) or {}).get("observation") or {}
                if field_name in observation and observation[field_name] is not None and observation[field_name] != claim_value:
                    issues.add(f"{evidence_id}:{field_name}_claim_evidence_conflict")
    return sorted(issues)


def _normalized_terms(value: str) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    normalized = normalized.replace("_", " ").replace("-", " ").lower()
    return {
        token
        for token in WORD.findall(normalized)
        if len(token) >= 2 and token not in METHOD_CONCEPT_STOPWORDS
    }


def _normalized_concept_phrases(value: str) -> set[str]:
    """Extract contiguous technical bigrams/trigrams from a public description."""

    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    normalized = normalized.replace("_", " ").replace("-", " ").lower()
    tokens = [
        token
        for token in WORD.findall(normalized)
        if len(token) >= 3 and token not in METHOD_CONCEPT_STOPWORDS
    ]
    return {
        " ".join(tokens[index : index + width])
        for width in (2, 3)
        for index in range(0, len(tokens) - width + 1)
    }


def _concept_phrase_visible(phrase: str, reasoning: str) -> bool:
    def filtered(value: str) -> str:
        normalized = str(value or "").replace("_", " ").replace("-", " ").lower()
        return " ".join(
            token
            for token in WORD.findall(normalized)
            if len(token) >= 3 and token not in METHOD_CONCEPT_STOPWORDS
        )

    phrase_text = filtered(phrase)
    reasoning_text = filtered(reasoning)
    return bool(phrase_text) and phrase_text in reasoning_text


def _phrase_visible(phrase: str, reasoning: str) -> bool:
    phrase_text = " ".join(WORD.findall(str(phrase).replace("_", " ").replace("-", " ").lower()))
    reasoning_text = " ".join(WORD.findall(str(reasoning).replace("_", " ").replace("-", " ").lower()))
    return bool(phrase_text) and phrase_text in reasoning_text


def _visible_diagnostic_keys(input_items: list[dict[str, Any]]) -> set[str]:
    """Return neutral fact names present before the proposed action.

    System instructions and assistant prose are excluded so that repeating a
    method description cannot masquerade as grounding in a measured fact.
    """

    keys: set[str] = set()

    def collect(value: Any) -> None:
        parsed = _json_object(value)
        if isinstance(parsed, dict):
            for key, child in parsed.items():
                normalized = str(key).strip().lower()
                if (
                    normalized
                    and normalized not in DIAGNOSTIC_KEY_STOPWORDS
                    and not normalized.endswith("_id")
                    and not normalized.endswith("_ref")
                ):
                    keys.add(normalized)
                collect(child)
        elif isinstance(parsed, list):
            for child in parsed:
                collect(child)

    for item in input_items:
        if item.get("role") in {"system", "assistant"}:
            continue
        for field in ("content", "output"):
            if field in item:
                collect(item[field])
    return keys


def prospective_method_fit_audit(
    action: TeacherAction,
    *,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit whether a pre-action note connects a method to visible diagnostics.

    The audit derives method concepts from the currently visible schemas and
    diagnostics from the current transcript.  It never contains a preferred
    tool table, benchmark label, verifier result, or post-execution outcome.
    """

    semantic_tools = [
        tool
        for tool in tools
        if is_semantic_measurement_tool(str(tool.get("name") or ""))
    ]
    applicable = (
        len(semantic_tools) >= 2
        and is_semantic_measurement_tool(action.tool_name)
    )
    if not applicable:
        return {
            "applicable": False,
            "score": 0.0,
            "reason": "requires_multiple_visible_semantic_tools_and_a_measurement_action",
            "oracle_or_verifier_used": False,
        }

    selected = next(
        (tool for tool in semantic_tools if str(tool.get("name")) == action.tool_name),
        None,
    )
    if selected is None:
        return {
            "applicable": True,
            "score": 0.0,
            "failures": ["selected_tool_schema_missing"],
            "oracle_or_verifier_used": False,
        }

    description_terms_by_tool = {
        str(tool.get("name")): _normalized_terms(str(tool.get("description") or ""))
        for tool in semantic_tools
    }
    description_phrases_by_tool = {
        str(tool.get("name")): _normalized_concept_phrases(
            str(tool.get("description") or "")
        )
        for tool in semantic_tools
    }
    selected_terms = description_terms_by_tool[action.tool_name]
    other_term_counts: dict[str, int] = {}
    for name, terms in description_terms_by_tool.items():
        if name == action.tool_name:
            continue
        for term in terms:
            other_term_counts[term] = other_term_counts.get(term, 0) + 1
    distinctive_terms = {
        term for term in selected_terms if other_term_counts.get(term, 0) == 0
    }
    reasoning_terms = _normalized_terms(action.reasoning)
    matched_method_terms = sorted(distinctive_terms.intersection(reasoning_terms))
    selected_other_phrases = set().union(
        *(
            phrases
            for name, phrases in description_phrases_by_tool.items()
            if name != action.tool_name
        ),
        set(),
    )
    distinctive_phrases = (
        description_phrases_by_tool[action.tool_name] - selected_other_phrases
    )
    matched_method_phrases = sorted(
        phrase
        for phrase in distinctive_phrases
        if _concept_phrase_visible(phrase, action.reasoning)
    )

    diagnostic_keys = _visible_diagnostic_keys(input_items)
    matched_diagnostics = sorted(
        key for key in diagnostic_keys if _phrase_visible(key, action.reasoning)
    )

    alternative_tools = {
        str(tool.get("name")): tool
        for tool in semantic_tools
        if str(tool.get("name")) != action.tool_name
    }
    named_alternatives = sorted(
        name
        for name in alternative_tools
        if _phrase_visible(name, action.reasoning)
    )
    if not named_alternatives:
        for name in alternative_tools:
            alt_distinctive = description_terms_by_tool[name] - selected_terms
            if len(alt_distinctive.intersection(reasoning_terms)) >= 2:
                named_alternatives.append(name)
    lowered = str(action.reasoning or "").lower()
    contrastive = bool(named_alternatives) and any(
        marker in lowered for marker in CONTRAST_MARKERS
    )
    directional_contrast_candidates: list[dict[str, Any]] = []
    for marker in DIRECTIONAL_CONTRAST_MARKERS:
        marker_index = lowered.find(marker)
        if marker_index < 0:
            continue
        selected_clause_terms = _normalized_terms(lowered[:marker_index])
        rejected_clause_terms = _normalized_terms(
            lowered[marker_index + len(marker) :]
        )
        selected_clause_matches = sorted(
            distinctive_terms.intersection(selected_clause_terms)
        )
        selected_clause_phrase_matches = sorted(
            phrase
            for phrase in distinctive_phrases
            if _concept_phrase_visible(phrase, lowered[:marker_index])
        )
        for alternative_name in alternative_tools:
            alternative_other_terms = set().union(
                *(
                    terms
                    for name, terms in description_terms_by_tool.items()
                    if name != alternative_name
                ),
                set(),
            )
            alternative_distinctive = (
                description_terms_by_tool[alternative_name]
                - alternative_other_terms
            )
            alternative_other_phrases = set().union(
                *(
                    phrases
                    for name, phrases in description_phrases_by_tool.items()
                    if name != alternative_name
                ),
                set(),
            )
            alternative_distinctive_phrases = (
                description_phrases_by_tool[alternative_name]
                - alternative_other_phrases
            )
            rejected_clause_matches = sorted(
                alternative_distinctive.intersection(rejected_clause_terms)
            )
            rejected_clause_phrase_matches = sorted(
                phrase
                for phrase in alternative_distinctive_phrases
                if _concept_phrase_visible(
                    phrase, lowered[marker_index + len(marker) :]
                )
            )
            directional_contrast_candidates.append(
                {
                    "marker": marker,
                    "alternative_tool": alternative_name,
                    "selected_clause_matches": selected_clause_matches,
                    "selected_clause_phrase_matches": selected_clause_phrase_matches,
                    "rejected_clause_matches": rejected_clause_matches,
                    "rejected_clause_phrase_matches": rejected_clause_phrase_matches,
                    "passed": (
                        len(selected_clause_matches) >= 2
                        and bool(selected_clause_phrase_matches)
                        and len(rejected_clause_matches) >= 2
                        and bool(rejected_clause_phrase_matches)
                    ),
                }
            )
    directional_contrast = next(
        (
            item
            for item in directional_contrast_candidates
            if item["passed"]
            and item["alternative_tool"] in set(named_alternatives)
        ),
        None,
    )
    prospective = any(
        marker in lowered
        for marker in (
            "would",
            "could",
            "can ",
            "if ",
            "depending",
            "将",
            "可以",
            "若",
        )
    )
    method_diagnostic_link = (
        len(matched_method_terms) >= 2
        and bool(matched_method_phrases)
        and bool(matched_diagnostics)
        and any(marker in lowered for marker in METHOD_LINK_MARKERS)
    )
    components = {
        "chosen_method_mechanism_grounded": (
            len(matched_method_terms) >= 2 and bool(matched_method_phrases)
        ),
        "visible_diagnostic_grounded": bool(matched_diagnostics),
        "plausible_alternative_contrasted": contrastive,
        "counterfactual_tool_swap_specificity": directional_contrast is not None,
        "prospective_not_outcome_assuming": prospective,
        "method_diagnostic_link_explicit": method_diagnostic_link,
    }
    failures = [name for name, passed in components.items() if not passed]
    component_score = sum(components.values()) / len(components)
    # The pass score remains an interpretable component contract.  A separate,
    # capped specificity score ranks two fully valid pre-action notes without
    # rewarding unlimited keyword enumeration.
    mechanism_specificity = (
        min(len(matched_method_terms), 4) / 8.0
        + min(len(matched_method_phrases), 2) / 4.0
    )
    diagnostic_specificity = min(len(matched_diagnostics), 2) / 2.0
    ranking_score = (
        0.8 * component_score
        + 0.12 * mechanism_specificity
        + 0.08 * diagnostic_specificity
    )
    return {
        "applicable": True,
        "score": round(component_score, 4),
        "ranking_score": round(ranking_score, 4),
        "mechanism_specificity": round(mechanism_specificity, 4),
        "diagnostic_specificity": round(diagnostic_specificity, 4),
        "components": components,
        "matched_method_terms": matched_method_terms,
        "matched_method_phrases": matched_method_phrases,
        "distinctive_method_phrases": sorted(distinctive_phrases),
        "matched_diagnostic_keys": matched_diagnostics,
        "named_alternative_tools": sorted(set(named_alternatives)),
        "counterfactual_tool_swap_passed": directional_contrast is not None,
        "directional_contrast": directional_contrast,
        "directional_contrast_candidates": directional_contrast_candidates,
        "selected_tool": action.tool_name,
        "failures": failures,
        "diagnostic_source": "student_visible_preaction_transcript_only",
        "method_source": "student_visible_tool_descriptions_only",
        "oracle_or_verifier_used": False,
    }


def _reasoning_score(
    action: TeacherAction,
    *,
    visible_evidence_ids: set[str],
    eligible_count: int,
    integrity_issues: list[str] | None = None,
    recent_reasoning: list[str] | None = None,
    latest_observation_evidence_ids: set[str] | None = None,
) -> tuple[float, list[str]]:
    reasoning = str(action.reasoning or "")
    lowered = reasoning.lower()
    words = WORD.findall(reasoning)
    flags: list[bool] = []
    failures: list[str] = []
    flags.append(35 <= len(words) <= 140)
    if not flags[-1]:
        failures.append("decision_note_length_outside_35_140")
    if visible_evidence_ids:
        grounded = bool(visible_evidence_ids.intersection(EVIDENCE_ID.findall(reasoning)))
        flags.append(grounded)
        if not grounded:
            failures.append("does_not_name_visible_evidence")
    latest_observation_evidence_ids = set(latest_observation_evidence_ids or [])
    if latest_observation_evidence_ids:
        latest_named = bool(
            latest_observation_evidence_ids.intersection(EVIDENCE_ID.findall(reasoning))
        )
        flags.append(latest_named)
        if not latest_named:
            failures.append("does_not_name_newly_created_evidence")
        uptake_markers = (
            "show",
            "report",
            "indicat",
            "support",
            "observ",
            "measur",
            "detect",
            "localiz",
            "conflict",
            "mismatch",
            "negative",
            "inconclusive",
            "no material",
            "未",
            "显示",
            "表明",
            "支持",
            "冲突",
        )
        interpreted = any(marker in lowered for marker in uptake_markers)
        flags.append(interpreted)
        if not interpreted:
            failures.append("new_observation_semantic_update_missing")
    tool_parts = [part for part in action.tool_name.lower().split("_") if len(part) >= 4]
    tool_grounded = action.tool_name.lower() in lowered or any(part in lowered for part in tool_parts)
    flags.append(tool_grounded)
    if not tool_grounded:
        failures.append("chosen_tool_not_justified")
    exploratory = action.tool_name.startswith(
        ("inspect_", "compare_", "detect_", "localize_", "align_")
    )
    gap_markers = ("remain", "unresolved", "need", "missing", "question", "uncertain", "尚", "缺", "问题")
    state_relevance_markers = (
        "support",
        "bind",
        "claim",
        "commit",
        "conflict",
        "field",
        "terminal",
        "abstain",
        "provenance",
        "scope",
        "identity",
    )
    decision_relevance = any(
        marker in lowered
        for marker in (gap_markers if exploratory else (*gap_markers, *state_relevance_markers))
    )
    flags.append(decision_relevance)
    if not decision_relevance:
        failures.append(
            "unresolved_question_missing" if exploratory else "state_consequence_missing"
        )
    prospective_markers = (
        "would",
        "could",
        "can test",
        "can distinguish",
        "can establish",
        "can reveal",
        "can determine",
        "can measure",
        "can localize",
        "can verify",
        "distinguish",
        "if ",
        "depending",
        "将",
        "可以检验",
        "若",
    )
    prospective = any(marker in lowered for marker in prospective_markers)
    flags.append(prospective or not exploratory)
    if exploratory and not prospective:
        failures.append("prospective_discrimination_missing")
    artifacts = [item for item in PROCESS_ARTIFACTS if item in lowered]
    flags.append(not artifacts)
    if artifacts:
        failures.extend(f"process_artifact:{item}" for item in artifacts)
    cardinality_honest = not (
        eligible_count > 1 and ("only eligible" in lowered or "only available" in lowered)
    )
    flags.append(cardinality_honest)
    if not cardinality_honest:
        failures.append("false_singleton_eligibility")
    flags.append(bool(action.params))
    if not action.params:
        failures.append("empty_arguments")
    integrity_issues = list(integrity_issues or [])
    unsafe_semantic_transition = bool(integrity_issues) and (
        action.tool_name in {"create_candidate_claim", "bind_evidence", "stop"}
        or (
            action.tool_name == "set_field_decision"
            and action.params.get("decision_status") == "committed"
        )
    )
    flags.append(not unsafe_semantic_transition)
    if unsafe_semantic_transition:
        failures.append("semantic_commit_with_visible_integrity_issue")
    recent_text = " ".join(recent_reasoning or []).lower()
    repeated_stock = [
        phrase
        for phrase in STOCK_REASONING_PHRASES
        if phrase in lowered and phrase in recent_text
    ]
    flags.append(not repeated_stock)
    if repeated_stock:
        failures.extend(f"repeated_stock_phrase:{phrase}" for phrase in repeated_stock)
    current_stock = [phrase for phrase in STOCK_REASONING_PHRASES if phrase in lowered]
    recent_stock = [phrase for phrase in STOCK_REASONING_PHRASES if phrase in recent_text]
    repeated_gap_template = bool(current_stock and recent_stock)
    flags.append(not repeated_gap_template)
    if repeated_gap_template:
        failures.append("repeated_gap_state_template")
    current_opening = " ".join(WORD.findall(lowered)[:4])
    prior_openings = {
        " ".join(WORD.findall(note.lower())[:4])
        for note in recent_reasoning or []
        if WORD.findall(note.lower())
    }
    repeated_opening = bool(current_opening and current_opening in prior_openings)
    flags.append(not repeated_opening)
    if repeated_opening:
        failures.append(f"repeated_sentence_opening:{current_opening}")
    current_signature = _style_frame_signature(reasoning)
    recent_signatures = [_style_frame_signature(note) for note in recent_reasoning or []]
    high_salience_frames = {
        "gap_frame",
        "warranted_frame",
        "that_matters_frame",
        "next_action_frame",
        "rather_than_frame",
        "no_conflict_inventory",
    }
    repeated_discourse_frame = any(
        bool(current_signature.intersection(previous).intersection(high_salience_frames))
        or (
            len(current_signature.intersection(previous)) >= 2
            and len(current_signature.intersection(previous))
            / max(1, len(current_signature.union(previous)))
            >= 0.5
        )
        for previous in recent_signatures
    )
    flags.append(not repeated_discourse_frame)
    if repeated_discourse_frame:
        failures.append(
            "repeated_discourse_frame:" + "+".join(sorted(current_signature))
        )
    return sum(flags) / len(flags), failures


def classify_supervision(
    *,
    eligible_tool_count: int,
    action: TeacherAction,
    exposure: dict[str, Any],
    evidence_ids_in_reasoning: int,
) -> dict[str, Any]:
    materialized = bool(exposure.get("full_target_call_materializable"))
    semantic_names = set(exposure.get("semantic_measurement_tool_names") or [])
    profile = supervision_profile(
        action.tool_name,
        params=action.params,
        external_measurement=action.tool_name in semantic_names,
        evidence_ids_in_reasoning=evidence_ids_in_reasoning,
        full_target_call_materializable=materialized,
    )
    return {
        **profile,
        "eligible_tool_count": eligible_tool_count,
        "policy_choice_bits": round(math.log2(max(1, eligible_tool_count)), 4),
    }


class CausalPolicyViewTeacher:
    """Best-of-N wrapper that never sends compiled control answers to its delegate."""

    def __init__(
        self,
        delegate: Any,
        *,
        candidates_per_turn: int = 3,
        early_accept_score: float = 0.875,
    ) -> None:
        if candidates_per_turn < 1:
            raise ValueError("candidates_per_turn must be positive")
        if not 0.0 <= early_accept_score <= 1.0:
            raise ValueError("early_accept_score must be between zero and one")
        self.delegate = delegate
        self.model = getattr(delegate, "model", "unknown_teacher")
        self.source = getattr(delegate, "source", "unknown_teacher_transport")
        self.policy_view = "causal_reduced_control_v1"
        self.candidates_per_turn = candidates_per_turn
        self.early_accept_score = early_accept_score
        self.audit_log: list[dict[str, Any]] = []

    def next_action(
        self, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> TeacherAction:
        recent_notes = _recent_reasoning(input_items)
        visible_input = _add_style_memory(
            sanitize_teacher_transcript(input_items), recent_notes
        )
        visible_tools = sanitize_tool_schemas(tools)
        exposure = control_plane_exposure_audit(visible_input, visible_tools)
        evidence_ids = _visible_evidence_ids(visible_input)
        latest_observation_evidence_ids = _latest_created_evidence_ids(visible_input)
        integrity_issues = visible_integrity_issues(visible_input)
        candidates: list[tuple[TeacherAction, float, list[str], dict[str, Any]]] = []
        errors: list[str] = []
        early_stop = False
        for _index in range(self.candidates_per_turn):
            candidate_input = copy.deepcopy(visible_input)
            if candidates:
                previous_failures = candidates[-1][2]
                style_failures = [
                    item
                    for item in previous_failures
                    if item.startswith("repeated_")
                ]
                content_failures = [
                    item
                    for item in previous_failures
                    if item
                    in {
                        "does_not_name_visible_evidence",
                        "chosen_tool_not_justified",
                        "unresolved_question_missing",
                        "prospective_discrimination_missing",
                        "chosen_method_mechanism_grounded",
                        "visible_diagnostic_grounded",
                        "plausible_alternative_contrasted",
                        "method_diagnostic_link_explicit",
                    }
                ]
                retry_note = (
                    " A prior independent draft for this turn was rejected. Re-decide from the visible state; "
                    "do not assume its tool choice was correct."
                )
                if style_failures:
                    retry_note += (
                        " Its prose repeated a recent sentence/discourse frame ("
                        + ", ".join(style_failures)
                        + "). Use a genuinely different organization and connective pattern."
                    )
                if content_failures:
                    retry_note += (
                        " It also missed these content properties: "
                        + ", ".join(content_failures)
                        + ". Ground the new note in a named visible fact, a concrete uncertainty, and how the chosen "
                        "measurement or state edit would resolve it."
                    )
                if candidate_input and candidate_input[0].get("role") == "system":
                    candidate_input[0]["content"] = (
                        str(candidate_input[0].get("content") or "") + retry_note
                    )
            try:
                action = self.delegate.next_action(
                    candidate_input, copy.deepcopy(visible_tools)
                )
            except Exception as error:  # transport errors are recorded, not hidden
                errors.append(f"{type(error).__name__}:{error}")
                continue
            score, failures = _reasoning_score(
                action,
                visible_evidence_ids=evidence_ids,
                eligible_count=len(visible_tools),
                integrity_issues=integrity_issues,
                recent_reasoning=recent_notes,
                latest_observation_evidence_ids=latest_observation_evidence_ids,
            )
            method_fit = prospective_method_fit_audit(
                action,
                input_items=visible_input,
                tools=visible_tools,
            )
            if method_fit.get("applicable"):
                failures = [*failures, *list(method_fit.get("failures") or [])]
            candidates.append((action, score, failures, method_fit))
            critical_failures = {
                failure
                for failure in failures
                if failure != "decision_note_length_outside_35_140"
            }
            method_fit_ready = (
                not method_fit.get("applicable")
                or float(method_fit.get("score", 0.0)) >= 0.8
            )
            if score >= self.early_accept_score and method_fit_ready and not critical_failures:
                early_stop = True
                break
        if not candidates:
            raise RuntimeError(f"all causal teacher candidates failed: {errors}")
        selected_index, selected = max(
            enumerate(candidates),
            key=lambda item: (
                float(item[1][3].get("ranking_score", item[1][3].get("score", 0.0)))
                if item[1][3].get("applicable")
                else 0.0,
                item[1][1],
                -item[0],
            ),
        )
        action, score, failures, method_fit = selected
        supervision = classify_supervision(
            eligible_tool_count=len(visible_tools),
            action=action,
            exposure=exposure,
            evidence_ids_in_reasoning=len(evidence_ids.intersection(EVIDENCE_ID.findall(action.reasoning))),
        )
        self.audit_log.append(
            {
                "selected_candidate_index": selected_index,
                "selected_score": round(score, 4),
                "selected_method_fit_score": float(method_fit.get("score", 0.0)),
                "selected_method_fit_ranking_score": float(
                    method_fit.get("ranking_score", method_fit.get("score", 0.0))
                ),
                "selected_method_fit": method_fit,
                "selected_failures": failures,
                "candidate_scores": [round(item[1], 4) for item in candidates],
                "candidate_failures": [item[2] for item in candidates],
                "candidate_method_fit_scores": [
                    float(item[3].get("score", 0.0)) for item in candidates
                ],
                "candidate_method_fit_ranking_scores": [
                    float(item[3].get("ranking_score", item[3].get("score", 0.0)))
                    for item in candidates
                ],
                "transport_errors": errors,
                "candidates_attempted": len(candidates) + len(errors),
                "candidate_budget": self.candidates_per_turn,
                "early_accept_score": self.early_accept_score,
                "early_stopped": early_stop,
                "control_plane_exposure": exposure,
                "visible_integrity_issues": integrity_issues,
                "latest_observation_evidence_ids": sorted(
                    latest_observation_evidence_ids
                ),
                "latest_observation_uptake_expected": bool(
                    latest_observation_evidence_ids
                ),
                "latest_observation_uptake_passed": not any(
                    failure
                    in {
                        "does_not_name_newly_created_evidence",
                        "new_observation_semantic_update_missing",
                    }
                    for failure in failures
                ),
                "supervision": supervision,
                "selected_action": asdict(action),
            }
        )
        return action
