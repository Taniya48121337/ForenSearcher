from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_student_closed_loop_canary import (
    EligibleToolNameLogitsProcessor,
    MaterializableToolCallStoppingCriteria,
    _action_lock_safe,
    _attempt_user_message,
    _evidence_consumption_frontier,
    _execution_frontier,
)


class _FakeTokenizer:
    pieces = {
        10: '<tool_call>\n{"name": "',
        20: "compare",
        21: "_uniform_timeline",
        22: "_action_refined_symmetric",
        23: "_global_motion_symmetric",
        30: '",',
        40: "reasoning",
    }
    names = {
        'compare_uniform_timeline",': [20, 21, 30],
        'compare_action_refined_symmetric",': [20, 22, 30],
        'compare_global_motion_symmetric",': [20, 23, 30],
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.names[text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.pieces[item] for item in ids)


def _scores() -> torch.Tensor:
    return torch.arange(50, dtype=torch.float32).reshape(1, 50)


def test_processor_masks_noneligible_tool_branch_only_inside_name() -> None:
    processor = EligibleToolNameLogitsProcessor(
        _FakeTokenizer(),
        ["compare_uniform_timeline", "compare_action_refined_symmetric"],
        prompt_length=1,
    )
    first = processor(torch.tensor([[99, 10]]), _scores())
    assert torch.isfinite(first[0, 20])
    assert torch.isneginf(first[0, 23])

    second = processor(torch.tensor([[99, 10, 20]]), _scores())
    assert torch.isfinite(second[0, 21])
    assert torch.isfinite(second[0, 22])
    assert torch.isneginf(second[0, 23])


def test_processor_does_not_mask_reasoning_outside_tool_name() -> None:
    processor = EligibleToolNameLogitsProcessor(
        _FakeTokenizer(), ["compare_uniform_timeline"], prompt_length=1
    )
    original = _scores()
    result = processor(torch.tensor([[99, 40]]), original.clone())
    assert torch.equal(result, original)


def test_materializable_stopper_waits_for_native_tool_call_name() -> None:
    stopper = MaterializableToolCallStoppingCriteria(
        _FakeTokenizer(), ["compare_uniform_timeline"], prompt_length=1
    )
    assert stopper(torch.tensor([[99, 40]]), _scores()) is False
    assert stopper(torch.tensor([[99, 10, 20, 21, 30]]), _scores()) is True
    assert stopper.matched_tool_name == "compare_uniform_timeline"


def test_action_lock_accepts_surface_alignment_problem() -> None:
    assert _action_lock_safe(
        ["chosen_measurement_tool_not_justified:compare_action_refined_symmetric"]
    )
    assert _action_lock_safe(["tool_arguments_not_schema_valid"])


def test_action_lock_rejects_visible_evidence_fact_problem() -> None:
    assert not _action_lock_safe(
        [
            "chosen_measurement_tool_not_justified:compare_uniform_timeline",
            "visible_atom_active_status_mismatch:EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001:true",
        ]
    )


def test_retry_revision_is_appended_to_user_json_without_mutating_source() -> None:
    original = {"role": "user", "content": '{"current_harness_state":{"terminal":false}}'}
    revised = _attempt_user_message(
        original,
        pinned_tool_name="compare_action_refined_symmetric",
        retry_feedback="Use the exact visible preference.",
    )
    assert "preexecution_revision" not in original["content"]
    assert revised["content"].endswith("}}")
    assert '"required_tool":"compare_action_refined_symmetric"' in revised["content"]
    assert '"state_unchanged":true' in revised["content"]
    assert "Do not mention drafts" in revised["content"]


def test_execution_frontier_exposes_only_current_eligible_tools() -> None:
    tools = [
        {"function": {"name": "compare_uniform_timeline", "description": "Uniform scan."}},
        {"function": {"name": "bind_evidence", "description": "Bind visible support."}},
    ]
    frontier = _execution_frontier(
        ["compare_uniform_timeline"], tools, executable_actions=[]
    )
    assert frontier["eligible_tools"] == ["compare_uniform_timeline"]
    assert frontier["tool_purposes"] == {
        "compare_uniform_timeline": "Uniform scan."
    }
    assert "bind_evidence" not in frontier["planning_instruction"]
    assert "literally name it" in frontier["planning_instruction"]
    assert "unresolved observation gap" in frontier["planning_instruction"]
    assert "never promise an order-consistent outcome" in frontier["planning_instruction"]


def test_evidence_consumption_frontier_summarizes_latest_visible_measurement() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-GLOBAL-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "confidence": 0.7,
                    "order_consistent": False,
                    "visible_rationale": "The order swap disagrees, so the route abstains.",
                },
            }
        }
    }
    frontier = _evidence_consumption_frontier(
        state, {"created_evidence_ids": [evidence_id]}
    )
    assert frontier is not None
    assert frontier["new_measurement_atoms"][0]["preferred_video"] == "c"
    assert frontier["new_measurement_atoms"][0]["order_consistent"] is False
    assert "exact EvidenceAtom ID" in frontier["current_turn_requirement"]


def test_evidence_consumption_frontier_separates_active_replacement_from_tombstone() -> None:
    active_id = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-002"
    dropped_id = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
    state = {
        "dropped_evidence_ids": [dropped_id],
        "evidence_store": {
            active_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "a",
                },
            }
        },
    }
    frontier = _evidence_consumption_frontier(
        state, {"created_evidence_ids": [active_id]}
    )
    assert frontier is not None
    guard = frontier["evidence_identity_guard"]
    assert guard["newly_created_active_ids"] == [active_id]
    assert guard["distinct_dropped_tombstone_ids"] == [dropped_id]
    assert "never transfers" in guard["rule"]
