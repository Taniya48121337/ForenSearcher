from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_counterfactual_repair_seeds import mutate_clean_stop_row


def _clean_stop_row() -> dict:
    measurement_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    target_fields = [
        "preferred_video",
        "video_a_interval_s",
        "video_b_interval_s",
        "measurement_method",
    ]
    fields = {
        "sample_key": {
            "value": "surgery_999",
            "evidence_status": "supported",
            "decision_status": "committed",
            "support_ids": ["EA-TASK-001"],
            "attack_ids": [],
        },
        "comparison_statement": {
            "value": "The action is faster",
            "evidence_status": "supported",
            "decision_status": "committed",
            "support_ids": ["EA-TASK-001"],
            "attack_ids": [],
        },
    }
    for field in target_fields:
        fields[field] = {
            "value": "a" if field == "preferred_video" else (
                "compare_uniform_timeline" if field == "measurement_method" else [0.0, 1.0]
            ),
            "evidence_status": "supported",
            "decision_status": "committed",
            "support_ids": [measurement_id],
            "attack_ids": [],
        }
    schema = {
        "schema_id": "viddiff_comparative_measurement_claim_v1",
        "required_fields": ["sample_key", "comparison_statement", *target_fields],
        "invariant_fields": ["sample_key", "comparison_statement"],
        "target_fields": target_fields,
        "fields": {field: {"types": ["str"], "acceptable_evidence_kinds": []} for field in fields},
    }
    bindings = {
        field: {
            "field": field,
            "support_ids": copy.deepcopy(state["support_ids"]),
            "attack_ids": [],
            "proposed_by": "policy",
            "validated_by": "contract_checker",
            "validation_status": "accepted",
        }
        for field, state in fields.items()
    }
    payload = {
        "task": {"task_contract": {}, "claim_schema": schema},
        "current_harness_state": {
            "task_contract": {
                "contract_id": "TASK-VIDDIFF-SURGERY_999",
                "episode_mode": "paired_compare",
                "task": "compare",
                "visible_assets": {},
                "claim_schema_id": schema["schema_id"],
            },
            "case_goals": [
                {"goal_id": "G-CONTEXT", "status": "satisfied"},
                {"goal_id": "G-DIFFERENCE", "status": "satisfied"},
                {"goal_id": "G-REPORT", "status": "satisfied"},
            ],
            "candidate_claims": [
                {"claim_id": "claim-1", "schema_id": schema["schema_id"], "fields": fields}
            ],
            "field_bindings": {"claim-1": bindings},
            "evidence_store": {
                "EA-TASK-001": {
                    "evidence_id": "EA-TASK-001",
                    "kind": "comparison_task_identity",
                    "supports_fields": ["sample_key", "comparison_statement"],
                    "observation": {},
                    "provenance": {},
                    "active": True,
                },
                measurement_id: {
                    "evidence_id": measurement_id,
                    "kind": "action_comparison_measurement",
                    "supports_fields": target_fields,
                    "observation": {},
                    "provenance": {},
                    "active": True,
                },
            },
            "available_evidence_ids": ["EA-TASK-001", measurement_id],
            "dropped_evidence_ids": [],
            "open_high_priority_obligations": [],
            "eligible_tools": ["stop"],
            "terminal": False,
            "stop_decision": None,
        },
        "past_execution_history": [],
        "latest_environment_observation": {},
    }
    return {
        "messages": [
            {"role": "system", "content": "clean"},
            {"role": "user", "content": json.dumps(payload)},
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [
                    {"type": "function", "function": {"name": "stop", "arguments": {}}}
                ],
            },
        ],
        "metadata": {
            "dataset": "VidDiffBench",
            "case_id": "viddiff-surgery_999",
            "row_id": "VidDiffBench:viddiff-surgery_999:clean:8",
            "source_split": "train",
        },
        "tools": [],
    }


def test_mutation_changes_only_measurement_support_slice() -> None:
    seed = mutate_clean_stop_row(_clean_stop_row())
    payload = json.loads(seed["messages"][1]["content"])
    state = payload["current_harness_state"]
    claim = state["candidate_claims"][0]
    assert seed["metadata"]["mutation_operator"] == "evidence_dropout"
    assert ":repair:1" in seed["metadata"]["row_id"]
    assert seed["messages"][0]["content"].startswith("You are the semantic policy")
    assert "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001" not in state["evidence_store"]
    assert state["dropped_evidence_ids"] == ["EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"]
    assert state["eligible_tools"] == ["set_field_decision"]
    assert claim["fields"]["sample_key"]["support_ids"] == ["EA-TASK-001"]
    assert claim["fields"]["sample_key"]["evidence_status"] == "supported"
    for field in payload["task"]["claim_schema"]["target_fields"]:
        assert claim["fields"][field]["support_ids"] == []
        assert claim["fields"][field]["evidence_status"] == "unsupported"
        assert claim["fields"][field]["decision_status"] == "committed"
        binding = state["field_bindings"]["claim-1"][field]
        assert binding["validation_status"] == "invalidated_by_environment_mutation"


def test_non_stop_source_is_rejected() -> None:
    row = _clean_stop_row()
    row["messages"][-1]["tool_calls"][0]["function"]["name"] = "bind_evidence"
    with pytest.raises(ValueError, match="clean stop"):
        mutate_clean_stop_row(row)
