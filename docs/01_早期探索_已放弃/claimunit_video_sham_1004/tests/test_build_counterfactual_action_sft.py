from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_counterfactual_action_sft import build  # noqa: E402


def _row(tool: str) -> dict:
    arguments = (
        {"claim_id": "claim_001", "field_values": {"sample_key": "x"}}
        if tool == "create_candidate_claim"
        else {
            "claim_id": "claim_001",
            "binding_items": [{"field": "sample_key", "support_ids": ["EA-1"]}],
        }
    )
    return {
        "messages": [
            {"role": "system", "content": "copy exact IDs"},
            {"role": "user", "content": "visible claim_id=claim_001"},
            {
                "role": "assistant",
                "content": "A deliberately verbose old explanation.",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": tool, "arguments": arguments}}
                ],
            },
        ],
        "tools": [{"function": {"name": tool, "parameters": {"properties": {"claim_id": {"const": "claim_001"}}}}}],
        "metadata": {"row_id": f"D:c:{tool}", "dataset": "D", "target_tool": tool},
    }


def test_counterfactual_rename_is_consistent_and_target_is_compact() -> None:
    rows, report = build([_row("create_candidate_claim")], variants=2)
    assert len(rows) == 3
    assert report["counterfactual_rows"] == 2
    assert "state-provided ID `claim_001`" in rows[0]["messages"][-1]["content"]
    for variant in rows[1:]:
        target_id = variant["messages"][-1]["tool_calls"][0]["function"]["arguments"]["claim_id"]
        assert target_id.startswith("CF-CLAIM-D-")
        assert target_id in variant["messages"][1]["content"]
        assert variant["tools"][0]["function"]["parameters"]["properties"]["claim_id"]["const"] == target_id


def test_bind_compaction_cites_support_and_preserves_call() -> None:
    source = _row("bind_evidence")
    rows, _ = build([source], variants=0)
    assert "active atoms cover all 1 fields" in rows[0]["messages"][-1]["content"]
    assert rows[0]["messages"][-1]["tool_calls"] == source["messages"][-1]["tool_calls"]


def test_unobservable_create_id_is_not_counterfactually_relabelled() -> None:
    source = _row("create_candidate_claim")
    source["messages"][1]["content"] = "no proposed claim handle is visible"
    source["tools"][0]["function"]["parameters"]["properties"]["claim_id"] = {
        "type": "string"
    }
    rows, report = build([source], variants=2)
    assert len(rows) == 1
    assert report["counterfactual_rows"] == 0
    assert report["counterfactual_source_rows_skipped_unobservable_id"] == 1
    assert "fresh opaque claim ID" in rows[0]["messages"][-1]["content"]
