"""Negative controls showing that the UTQR scorecard is not pass-by-construction."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from pipeline import ROOT, write_json
from score_utqr_recovery_episodes import score_batch
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio"
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = DEFAULT_BATCH / "utqr_falsification_report.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for(rows: list[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(row)
        for row in rows
        if str(row["metadata"]["episode_id"]) == episode_id
    ]


def _set_final_nonterminal(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    payload = json.loads(episode["messages"][-1]["content"])
    payload["harness_state"]["terminal"] = False
    payload["harness_state"]["stop_decision"] = None
    payload["harness_state"]["open_high_priority_obligations"] = ["G-REPORT"]
    episode["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False)


def _break_counterfactual(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    episode["clean_control_suppresses_challenge"] = False


def _break_locality(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    episode["independent_fields_preserved"] = False


def _break_fresh_uptake(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    episode["fresh_evidence_consumed"] = False


def _add_private_leak(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    episode["student_visible_leakage_hits"] = ["private_intervention"]


def _empty_reasoning(_: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    rows[0]["messages"][-1]["content"] = ""


def _remove_state_delta(episode: dict[str, Any], _: list[dict[str, Any]]) -> None:
    initial = json.loads(episode["messages"][1]["content"])["current_harness_state"]
    result = json.loads(episode["messages"][3]["content"])
    result["harness_state"] = initial
    episode["messages"][3]["content"] = json.dumps(result, ensure_ascii=False)


def _add_unused_executed_measurement(
    episode: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Insert a legal, state-changing external call whose evidence is never used."""

    measurement_row = next(
        row
        for row in rows
        if row["metadata"]["causal_recovery_stage"] == "fresh_measurement"
    )
    original_tool = str(measurement_row["metadata"]["target_tool"])
    messages = episode["messages"]
    measurement_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and (message.get("tool_calls") or [])[0]["function"]["name"] == original_tool
    )
    prior_result = json.loads(messages[measurement_index - 1]["content"])
    prior_state = prior_result["harness_state"]
    eligible = [
        str(name)
        for name in prior_state.get("eligible_tools") or []
        if str(name) != original_tool
    ]
    if not eligible:
        raise ValueError("unused-measurement mutation requires an eligible alternative")
    extra_tool = eligible[0]
    original_call = copy.deepcopy(messages[measurement_index]["tool_calls"][0])
    original_call["id"] = "CALL-UNUSED-EXTRA-001"
    original_call["function"]["name"] = extra_tool
    extra_assistant = {
        "role": "assistant",
        "content": (
            "This additional eligible measurement records another observation, but it "
            "does not answer a distinct unresolved question and will not affect the decision."
        ),
        "tool_calls": [original_call],
    }
    evidence_id = "EA-UNUSED-EXTRA-001"
    atom = {
        "evidence_id": evidence_id,
        "kind": "unused_auxiliary_measurement",
        "supports_fields": list(episode.get("target_fields") or []),
        "observation": {"measurement_method": extra_tool, "confidence": 0.51},
        "provenance": {"generated_by": "CALL-UNUSED-EXTRA-001", "method": extra_tool},
        "active": True,
    }
    extra_state = copy.deepcopy(prior_state)
    extra_state.setdefault("evidence_store", {})[evidence_id] = copy.deepcopy(atom)
    available = extra_state.setdefault("available_evidence_ids", [])
    if evidence_id not in available:
        available.append(evidence_id)
    extra_result = {
        "status": "executed",
        "created_evidence_ids": [evidence_id],
        "evidence_atoms": [copy.deepcopy(atom)],
        "observation": {"measurement_method": extra_tool, "goal_satisfied": True},
        "harness_state": extra_state,
    }
    extra_tool_message = {
        "role": "tool",
        "tool_call_id": "CALL-UNUSED-EXTRA-001",
        "name": extra_tool,
        "content": json.dumps(extra_result, ensure_ascii=False),
    }
    messages[measurement_index:measurement_index] = [extra_assistant, extra_tool_message]
    for message in messages[measurement_index + 2 :]:
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        state = payload.get("harness_state") or {}
        state.setdefault("evidence_store", {})[evidence_id] = copy.deepcopy(atom)
        visible = state.setdefault("available_evidence_ids", [])
        if evidence_id not in visible:
            visible.append(evidence_id)
        message["content"] = json.dumps(payload, ensure_ascii=False)

    extra_row = copy.deepcopy(measurement_row)
    extra_row["metadata"]["row_id"] = (
        f"{episode['episode_id']}:unused-executed-measurement"
    )
    extra_row["metadata"]["target_tool"] = extra_tool
    original_reasoning = str(extra_row["messages"][-1].get("content") or "")
    extra_row["messages"][-1]["content"] = (
        "As an unnecessary control, "
        + original_reasoning.replace(original_tool, extra_tool).lstrip()
    )
    insert_at = rows.index(measurement_row)
    rows.insert(insert_at, extra_row)
    episode["turns"] = int(episode["turns"]) + 1


def _uses_tool(episode: dict[str, Any], tool_name: str) -> bool:
    return any(
        message.get("role") == "assistant"
        and any(
            call.get("function", {}).get("name") == tool_name
            for call in message.get("tool_calls") or []
        )
        for message in episode["messages"]
    )


def _reenter_rejected_evidence(
    episode: dict[str, Any], _: list[dict[str, Any]]
) -> None:
    """Make the final binding reuse a rejected candidate while keeping it terminal."""

    messages = episode["messages"]
    adjudication_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and (message.get("tool_calls") or [])[0]["function"]["name"]
        == "adjudicate_evidence"
    )
    adjudication_call = messages[adjudication_index]["tool_calls"][0]["function"]
    adjudication_args = adjudication_call["arguments"]
    if isinstance(adjudication_args, str):
        adjudication_args = json.loads(adjudication_args)
    rejected_id = adjudication_args["rejected_evidence"][0]["evidence_id"]
    claim_id = str(adjudication_args["claim_id"])
    target_fields = list(adjudication_args["target_fields"])
    bind_index = next(
        index
        for index in range(adjudication_index + 1, len(messages))
        if messages[index].get("role") == "assistant"
        and (messages[index].get("tool_calls") or [])[0]["function"]["name"]
        == "bind_evidence"
    )
    bind_call = messages[bind_index]["tool_calls"][0]["function"]
    bind_args = bind_call["arguments"]
    if isinstance(bind_args, str):
        bind_args = json.loads(bind_args)
    for item in bind_args.get("binding_items") or []:
        if item.get("field") in target_fields:
            item["support_ids"] = [rejected_id]
    bind_call["arguments"] = bind_args

    for index in range(bind_index + 1, len(messages)):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        state = payload.get("harness_state") or {}
        evidence = state.get("evidence_store") or {}
        rejected_observation = (evidence.get(rejected_id) or {}).get("observation") or {}
        bindings = (state.get("field_bindings") or {}).get(claim_id) or {}
        claims = {
            str(item.get("claim_id")): item for item in state.get("candidate_claims") or []
        }
        fields = (claims.get(claim_id) or {}).get("fields") or {}
        for field in target_fields:
            if field in bindings:
                bindings[field]["support_ids"] = [rejected_id]
            if field in fields:
                fields[field]["support_ids"] = [rejected_id]
                if field in rejected_observation:
                    fields[field]["value"] = rejected_observation[field]
        message["content"] = json.dumps(payload, ensure_ascii=False)


MUTATIONS: list[tuple[str, str, Callable[[dict[str, Any], list[dict[str, Any]]], None]]] = [
    ("premature_nonterminal_stop", "C1", _set_final_nonterminal),
    ("counterfactual_insensitivity", "C3", _break_counterfactual),
    ("off_target_field_mutation", "C4", _break_locality),
    ("fresh_evidence_not_consumed", "C7", _break_fresh_uptake),
    ("student_visible_private_leak", "C2", _add_private_leak),
    ("empty_reasoning_target", "C10", _empty_reasoning),
    ("executed_action_without_material_delta", "C5", _remove_state_delta),
]


def run_falsification(
    episodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    compile_report: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    representatives: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        representatives.setdefault(str(episode["dataset"]), episode)
    ordered = [representatives[key] for key in sorted(representatives)]
    mutations = list(MUTATIONS)
    if any(item.get("id") == "C11" for item in rubric.get("criteria") or []):
        mutations.append(
            (
                "rejected_evidence_reenters_final_binding",
                "C11",
                _reenter_rejected_evidence,
            )
        )
    if str(rubric.get("rubric_id")) == "UTQR-12":
        mutations.append(
            (
                "executed_but_unused_external_measurement",
                "C5",
                _add_unused_executed_measurement,
            )
        )
    results: list[dict[str, Any]] = []
    for index, (name, expected, mutate) in enumerate(mutations):
        if expected == "C11":
            source = next(
                episode for episode in episodes if _uses_tool(episode, "adjudicate_evidence")
            )
        elif name == "executed_but_unused_external_measurement":
            source = next(
                episode
                for episode in episodes
                if episode.get("dataset") == "VideoSham"
            )
        else:
            source = ordered[index % len(ordered)]
        episode = copy.deepcopy(source)
        episode_rows = _rows_for(rows, str(source["episode_id"]))
        baseline = score_batch(
            [copy.deepcopy(source)],
            _rows_for(rows, str(source["episode_id"])),
            compile_report,
            rubric,
        )["scorecards"][0]
        mutate(episode, episode_rows)
        corrupted = score_batch(
            [episode], episode_rows, compile_report, rubric
        )["scorecards"][0]
        results.append(
            {
                "mutation": name,
                "dataset": source["dataset"],
                "expected_criterion": expected,
                "baseline_score": baseline["automated_core_score"],
                "mutated_score": corrupted["automated_core_score"],
                "score_drop": round(
                    baseline["automated_core_score"]
                    - corrupted["automated_core_score"],
                    3,
                ),
                "expected_criterion_before": baseline["criteria"][expected],
                "expected_criterion_after": corrupted["criteria"][expected],
                "automated_hard_gates_after": corrupted["hard_gates"][
                    "H1_to_H7_automated"
                ],
            }
        )
    return {
        "artifact": "utqr_scorecard_falsification_v1",
        "purpose": (
            "General defect mutations must lower the evidence-linked score and fail "
            "the automated hard-gate summary; no case-specific rule is used."
        ),
        "mutations": len(results),
        "all_detected": all(
            item["score_drop"] > 0
            and item["expected_criterion_after"]
            < item["expected_criterion_before"]
            and item["automated_hard_gates_after"] is False
            for item in results
        ),
        "minimum_score_drop": min(item["score_drop"] for item in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_falsification(
        _jsonl(args.batch_root / "recovery_episodes.jsonl"),
        _jsonl(args.batch_root / "recovery_sft_candidate.jsonl"),
        _load(args.batch_root / "compile_report.json"),
        _load(args.rubric),
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
