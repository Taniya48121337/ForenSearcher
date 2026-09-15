"""Learning-value annotations for indispensable executable SFT actions.

Step utility answers whether an action may enter the corpus.  This module
answers a different question: how much supervised loss should an admitted
action contribute?  Keeping these concerns separate prevents necessary
bookkeeping from receiving the same training mass as evidence acquisition,
interpretation, and uncertainty repair.
"""

from __future__ import annotations

from typing import Any


UNCERTAINTY_TOOLS = {
    "compare_symmetry_residual_abstention",
    "challenge_evidence",
    "reopen_claim_fields",
    "invalidate_evidence",
}
EVIDENCE_INTERPRETATION_TOOLS = {
    "create_candidate_claim",
    "assess_evidence_candidate",
    "adjudicate_evidence",
}


def supervision_profile(
    tool_name: str,
    *,
    params: dict[str, Any] | None = None,
    external_measurement: bool = False,
    evidence_ids_in_reasoning: int = 0,
    full_target_call_materializable: bool = False,
) -> dict[str, Any]:
    """Return a tool-semantic base weight without using execution outcomes."""

    params = params or {}
    if tool_name in UNCERTAINTY_TOOLS or (
        tool_name == "set_field_decision"
        and params.get("decision_status") == "open"
    ):
        label, subtype, weight = (
            "uncertainty_or_repair_detection",
            "belief_repair",
            1.2,
        )
    elif external_measurement:
        label, subtype, weight = (
            "semantic_tool_decision",
            "evidence_measurement",
            1.0,
        )
    elif tool_name in EVIDENCE_INTERPRETATION_TOOLS:
        weight = 1.1 if tool_name in {
            "assess_evidence_candidate",
            "adjudicate_evidence",
        } else 1.0
        label, subtype = "evidence_interpretation", "evidence_synthesis"
    elif tool_name == "bind_evidence":
        label, subtype, weight = (
            "evidence_interpretation",
            "evidence_attribution",
            0.7,
        )
    elif tool_name == "set_field_decision":
        label, subtype, weight = (
            "semantic_decision",
            "belief_commitment",
            0.6,
        )
    elif tool_name.startswith("inspect_"):
        label, subtype, weight = (
            "context_acquisition",
            "context_inspection",
            0.4,
        )
    elif tool_name == "stop":
        label, subtype, weight = "control_only", "terminal_verification", 0.2
    elif evidence_ids_in_reasoning > 0:
        label, subtype, weight = (
            "evidence_interpretation",
            "evidence_grounded_state_transition",
            0.8,
        )
    else:
        label, subtype, weight = "control_only", "ordinary_control", 0.1

    return {
        "class": label,
        "subtype": subtype,
        "training_weight": weight,
        "full_target_call_materializable": bool(
            full_target_call_materializable
        ),
    }


def learning_value_profile(
    tool_name: str,
    *,
    params: dict[str, Any] | None = None,
    external_measurement: bool = False,
    evidence_ids_in_reasoning: int = 0,
    full_target_call_materializable: bool = False,
    branch_enabling_measurement: bool = False,
) -> dict[str, Any]:
    """Add a bounded value-of-information bonus and assign a loss tier."""

    supervision = supervision_profile(
        tool_name,
        params=params,
        external_measurement=external_measurement,
        evidence_ids_in_reasoning=evidence_ids_in_reasoning,
        full_target_call_materializable=full_target_call_materializable,
    )
    bonus = 0.2 if branch_enabling_measurement and external_measurement else 0.0
    weight = round(min(1.3, float(supervision["training_weight"]) + bonus), 3)
    tier = "core_reasoning" if weight >= 0.9 else "supporting" if weight >= 0.5 else "minimal_control"
    return {
        "supervision_class": supervision["class"],
        "supervision_subtype": supervision["subtype"],
        "base_training_weight": supervision["training_weight"],
        "learning_value_weight": weight,
        "learning_value_tier": tier,
        "branch_information_gain_bonus": bonus,
    }
