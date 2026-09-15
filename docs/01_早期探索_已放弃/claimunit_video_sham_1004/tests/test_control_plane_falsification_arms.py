from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control_plane_falsification_arms import apply_policy_view_arm  # noqa: E402


def _atom(value: str) -> dict:
    return {
        "evidence_id": "EA-1",
        "kind": "measurement",
        "supports_fields": ["direction"],
        "observation": {"direction": value, "confidence": 0.9},
        "active": True,
    }


def test_structure_only_redacts_semantics_but_keeps_control() -> None:
    view = apply_policy_view_arm(
        "structure_only",
        dataset="D",
        case_id="C",
        state={
            "eligible_tools": ["bind_evidence"],
            "evidence_store": {"EA-1": _atom("a_to_b")},
            "state_action_affordances": {"executable_binding_actions": [{"tool_name": "bind_evidence"}]},
        },
        history=[],
        task={"description": "which direction"},
        latest={},
        tools=[],
    )
    assert view["state"]["eligible_tools"] == ["bind_evidence"]
    assert "executable_binding_actions" in view["state"]["state_action_affordances"]
    assert view["state"]["evidence_store"]["EA-1"]["observation"]["redacted"] is True
    assert view["task"]["description"]["redacted"] is True


def test_evidence_only_removes_compiled_control_but_keeps_observation() -> None:
    view = apply_policy_view_arm(
        "evidence_only",
        dataset="D",
        case_id="C",
        state={
            "eligible_tools": ["bind_evidence"],
            "evidence_store": {"EA-1": _atom("a_to_b")},
            "state_action_affordances": {"executable_binding_actions": [{"tool_name": "bind_evidence"}]},
        },
        history=[],
        task={},
        latest={},
        tools=[],
    )
    assert view["state"]["evidence_store"]["EA-1"]["observation"]["direction"] == "a_to_b"
    assert "state_action_affordances" not in view["state"]
    assert view["state"]["eligible_tools"] == ["bind_evidence"]


def test_shuffle_uses_schema_compatible_different_case_donor() -> None:
    bank = {
        ("D", "measurement", ("direction",)): [
            {"case_id": "OTHER", "observation": {"direction": "b_to_a", "confidence": 0.2}, "provenance": {}}
        ]
    }
    view = apply_policy_view_arm(
        "shuffled_evidence",
        dataset="D",
        case_id="C",
        state={"evidence_store": {"EA-1": _atom("a_to_b")}},
        history=[],
        task={},
        latest={},
        tools=[],
        donor_bank=bank,
    )
    atom = view["state"]["evidence_store"]["EA-1"]
    assert atom["observation"]["direction"] == "b_to_a"
    assert atom["semantic_content_shuffled"] is True
    assert view["evidence_swaps"] == 1
