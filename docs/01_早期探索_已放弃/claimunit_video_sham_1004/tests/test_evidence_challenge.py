from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_harness import (  # noqa: E402
    EvidenceAtom,
    TaskContract,
    VideoForensicHarness,
    with_evidence_integrity_tools,
)
from viddiff_batch import VIDDIFF_SCHEMA  # noqa: E402


class _FreshContextEnvironment:
    def __init__(self) -> None:
        self.calls = 0

    def eligible_tool_names(self) -> list[str]:
        return ["inspect_comparison_task", "compare_uniform_timeline"]

    def execute(self, name: str, _arguments: dict) -> dict:
        self.calls += 1
        if name != "inspect_comparison_task":
            raise AssertionError(name)
        atom = EvidenceAtom(
            evidence_id=f"EA-FRESH-CONTEXT-{self.calls:03d}",
            kind="comparison_task_identity",
            supports_fields=list(VIDDIFF_SCHEMA["invariant_fields"]),
            observation={
                "sample_key": "surgery_current",
                "comparison_statement": "current visible task",
            },
            provenance={"generated_by": f"EXEC-FRESH-{self.calls:03d}"},
        )
        return {
            "goal_satisfied": True,
            "evidence_atoms": [atom],
            "result": copy.deepcopy(atom.observation),
        }


def _harness() -> VideoForensicHarness:
    registry = with_evidence_integrity_tools(
        [
            {
                "name": "inspect_comparison_task",
                "domain": "external_media",
                "satisfies_goal": "G-CONTEXT",
            },
            {
                "name": "compare_uniform_timeline",
                "domain": "external_media",
                "satisfies_goal": "G-DIFFERENCE",
            },
            {"name": "create_candidate_claim", "domain": "harness_state"},
            {"name": "bind_evidence", "domain": "harness_state"},
            {"name": "set_field_decision", "domain": "harness_state"},
            {"name": "stop", "domain": "harness_terminal"},
        ]
    )
    harness = VideoForensicHarness(
        TaskContract(
            contract_id="TASK-CHALLENGE-001",
            episode_mode="paired_compare",
            task="Compare the visible pair.",
            visible_assets={"source": "video_a", "suspect": "video_b"},
            claim_schema_id=VIDDIFF_SCHEMA["schema_id"],
        ),
        _FreshContextEnvironment(),
        claim_schema=VIDDIFF_SCHEMA,
        tool_registry=registry,
        target_fields=VIDDIFF_SCHEMA["target_fields"],
        invariant_fields=VIDDIFF_SCHEMA["invariant_fields"],
    )
    harness.goals["G-CONTEXT"].status = "satisfied"
    harness.goals["G-DIFFERENCE"].status = "satisfied"
    harness.evidence_store["EA-DONOR-001"] = EvidenceAtom(
        evidence_id="EA-DONOR-001",
        kind="comparison_task_identity",
        supports_fields=list(VIDDIFF_SCHEMA["invariant_fields"]),
        observation={
            "sample_key": "surgery_donor",
            "comparison_statement": "different task",
        },
        provenance={"generated_by": "EXEC-DONOR"},
    )
    return harness


def test_preclaim_state_offers_claim_or_nonprescriptive_integrity_challenge() -> None:
    harness = _harness()
    names = [item["name"] for item in harness.eligible_tool_registry()]
    assert names == ["create_candidate_claim", "challenge_evidence"]
    state = harness.render_state()
    assert state["open_evidence_challenges"] == []
    assert "repair_restoration_candidates" not in state


def test_challenge_reopens_generic_check_and_is_resolved_only_by_execution() -> None:
    harness = _harness()
    result = harness.execute_tool(
        "challenge_evidence",
        {
            "reasoning": (
                "EA-DONOR-001 names surgery_donor while the current task is different; "
                "fresh context is required before creating a claim."
            ),
            "evidence_ids": ["EA-DONOR-001"],
            "defect_codes": ["identity_mismatch"],
            "target_fields": ["sample_key"],
            "requested_check": "reinspect_context",
        },
        "CALL-CHALLENGE-001",
    )
    assert result["challenge"]["status"] == "open"
    assert [item["name"] for item in harness.eligible_tool_registry()] == [
        "inspect_comparison_task"
    ]
    harness.execute_tool(
        "inspect_comparison_task",
        {"video_a_ref": "video_a", "video_b_ref": "video_b"},
        "CALL-INSPECT-001",
    )
    challenge = harness.render_state()["evidence_challenges"][0]
    assert challenge["status"] == "resolved"
    assert challenge["resolved_by_evidence_ids"] == ["EA-FRESH-CONTEXT-001"]
    assert [item["name"] for item in harness.eligible_tool_registry()] == [
        "create_candidate_claim",
        "challenge_evidence",
    ]


def test_challenge_rejects_nonactive_ids_and_unknown_defects() -> None:
    harness = _harness()
    with pytest.raises(ValueError, match="not active"):
        harness.execute_tool(
            "challenge_evidence",
            {
                "reasoning": "Visible evidence is suspect and needs a fresh check.",
                "evidence_ids": ["EA-MISSING"],
                "defect_codes": ["identity_mismatch"],
                "target_fields": ["sample_key"],
                "requested_check": "reinspect_context",
            },
            "CALL-BAD-ID",
        )
    with pytest.raises(ValueError, match="defect_codes"):
        harness.execute_tool(
            "challenge_evidence",
            {
                "reasoning": "Visible evidence is suspect and needs a fresh check.",
                "evidence_ids": ["EA-DONOR-001"],
                "defect_codes": ["oracle_disagrees"],
                "target_fields": ["sample_key"],
                "requested_check": "reinspect_context",
            },
            "CALL-BAD-DEFECT",
        )
