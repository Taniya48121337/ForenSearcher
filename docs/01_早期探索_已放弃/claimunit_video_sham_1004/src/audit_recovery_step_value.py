"""Audit whether every executed recovery step has a visible downstream role.

Execution and a material state delta are necessary but not sufficient for useful
SFT supervision.  In particular, an eligible external call can create a fresh
EvidenceAtom that is never assessed, adjudicated, bound, or used to resolve the
challenge while the episode still reaches the same valid terminal state.  This
audit follows identifiers and field scopes through the recorded student-visible
transcript so such busywork does not receive full method-fit credit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v31_evidence_first55"
DEFAULT_OUTPUT = DEFAULT_BATCH / "step_value_audit.json"
ACCEPTED = {"accepted", "executed"}


def _content(message: dict[str, Any]) -> dict[str, Any]:
    value = message.get("content")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _state(message: dict[str, Any], *, initial: bool = False) -> dict[str, Any]:
    payload = _content(message)
    key = "current_harness_state" if initial else "harness_state"
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _events(episode: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = episode["messages"]
    before = _state(messages[1], initial=True)
    events: list[dict[str, Any]] = []
    row_index = 0
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or len(calls) != 1:
            continue
        if index + 1 >= len(messages) or row_index >= len(rows):
            continue
        call = calls[0].get("function") or {}
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        result = _content(messages[index + 1])
        after = _state(messages[index + 1])
        events.append(
            {
                "turn": row_index + 1,
                "stage": str(rows[row_index]["metadata"]["causal_recovery_stage"]),
                "tool": str(call.get("name")),
                "arguments": arguments if isinstance(arguments, dict) else {},
                "result": result,
                "before": before,
                "after": after,
            }
        )
        before = after
        row_index += 1
    return events


def _binding_fields(arguments: dict[str, Any]) -> set[str]:
    return {
        str(item.get("field"))
        for item in arguments.get("binding_items") or []
        if isinstance(item, dict) and item.get("field")
    }


def _binding_ids(arguments: dict[str, Any]) -> set[str]:
    return {
        str(evidence_id)
        for item in arguments.get("binding_items") or []
        if isinstance(item, dict)
        for evidence_id in item.get("support_ids") or []
    }


def _adjudication_ids(arguments: dict[str, Any]) -> set[str]:
    selected = {str(item) for item in arguments.get("selected_evidence_ids") or []}
    rejected = {
        str(item.get("evidence_id"))
        for item in arguments.get("rejected_evidence") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    return selected | rejected


def _later_evidence_references(events: list[dict[str, Any]], start: int) -> set[str]:
    result: set[str] = set()
    for event in events[start:]:
        arguments = event["arguments"]
        result.update(_binding_ids(arguments))
        evidence_id = arguments.get("evidence_id")
        if evidence_id:
            result.add(str(evidence_id))
        result.update(_adjudication_ids(arguments))
        result.update(str(item) for item in arguments.get("evidence_ids") or [])
    return result


def _final_support_ids(state: dict[str, Any]) -> set[str]:
    return {
        str(evidence_id)
        for fields in (state.get("field_bindings") or {}).values()
        for binding in (fields or {}).values()
        for evidence_id in (binding or {}).get("support_ids") or []
    }


def _resolved_ids(state: dict[str, Any]) -> set[str]:
    return {
        str(evidence_id)
        for challenge in state.get("evidence_challenges") or []
        for evidence_id in challenge.get("resolved_by_evidence_ids") or []
    }


def _field_status(state: dict[str, Any], field: str) -> tuple[Any, Any]:
    claims = state.get("candidate_claims") or []
    item = ((claims[0].get("fields") or {}).get(field) or {}) if claims else {}
    return item.get("evidence_status"), item.get("decision_status")


def audit_episode_step_value(
    episode: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    events = _events(episode, rows)
    target_fields = {str(item) for item in episode.get("target_fields") or []}
    fresh_ids = {str(item) for item in episode.get("fresh_evidence_ids") or []}
    final_state = events[-1]["after"] if events else {}
    final_support = _final_support_ids(final_state)
    final_resolved = _resolved_ids(final_state)
    seen_external_tools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for index, event in enumerate(events):
        stage = event["stage"]
        tool = event["tool"]
        arguments = event["arguments"]
        result = event["result"]
        later = events[index + 1 :]
        later_tools = [item["tool"] for item in later]
        later_refs = _later_evidence_references(events, index + 1)
        checks: dict[str, bool] = {
            "executed_or_accepted": result.get("status") in ACCEPTED,
        }

        if stage == "challenge":
            challenged_fields = {str(item) for item in arguments.get("target_fields") or []}
            checks.update(
                {
                    "minimal_target_scope": bool(target_fields)
                    and challenged_fields == target_fields,
                    "requests_real_followup": bool(arguments.get("requested_check"))
                    and any(item["stage"] == "fresh_measurement" for item in later),
                }
            )
        elif stage == "fresh_measurement":
            created = {str(item) for item in result.get("created_evidence_ids") or []}
            consumed = created.intersection(later_refs | final_support | final_resolved)
            atoms = event["after"].get("evidence_store") or {}
            supported_targets = {
                field
                for evidence_id in created
                for field in (atoms.get(evidence_id) or {}).get("supports_fields") or []
                if field in target_fields
            }
            previous_same = seen_external_tools[tool]
            repeated_call_justified = not previous_same or any(
                prior["arguments"] != arguments
                and (
                    ((prior["result"].get("observation") or {}).get("goal_satisfied") is False)
                    or (
                        (((prior["result"].get("observation") or {}).get("routing_state") or {}).get("goal_satisfied")
                        is False)
                    )
                )
                for prior in previous_same
            )
            checks.update(
                {
                    "creates_fresh_evidence": bool(created)
                    and created.issubset(set(atoms)),
                    "all_created_evidence_consumed": bool(created)
                    and consumed == created,
                    "targets_open_question": bool(supported_targets),
                    "repeated_method_has_adaptive_reason": repeated_call_justified,
                }
            )
            seen_external_tools[tool].append(event)
        elif stage == "local_reopen":
            fields = {str(item) for item in arguments.get("fields") or []}
            checks.update(
                {
                    "opens_exact_target_fields": arguments.get("decision_status") == "open"
                    and fields == target_fields,
                    "fields_were_committed": bool(fields)
                    and all(_field_status(event["before"], field)[1] == "committed" for field in fields),
                    "open_state_is_consumed": "bind_evidence" in later_tools
                    and "set_field_decision" in later_tools,
                }
            )
        elif stage == "evidence_assessment":
            evidence_id = str(arguments.get("evidence_id") or "")
            adjudicated = any(
                evidence_id in _adjudication_ids(item["arguments"])
                for item in later
                if item["tool"] == "adjudicate_evidence"
            )
            checks.update(
                {
                    "assesses_exact_target_scope": set(arguments.get("target_fields") or [])
                    == target_fields,
                    "assessment_is_consumed": bool(evidence_id) and adjudicated,
                }
            )
        elif stage == "evidence_adjudication":
            selected = {str(item) for item in arguments.get("selected_evidence_ids") or []}
            later_bound = set().union(
                *[
                    _binding_ids(item["arguments"])
                    for item in later
                    if item["tool"] == "bind_evidence"
                ],
                set(),
            )
            checks.update(
                {
                    "adjudicates_exact_target_scope": set(arguments.get("target_fields") or [])
                    == target_fields,
                    "selected_partition_is_consumed": bool(selected)
                    and bool(selected.intersection(later_bound)),
                }
            )
        elif stage in {"claim_revision", "final_visible_claim_revision"}:
            fields = {
                str(item.get("field"))
                for item in arguments.get("field_updates") or []
                if isinstance(item, dict) and item.get("field")
            }
            checks.update(
                {
                    "revision_is_local": bool(fields) and fields.issubset(target_fields),
                    "revision_is_consumed": "bind_evidence" in later_tools
                    and "set_field_decision" in later_tools,
                }
            )
        elif stage == "fresh_binding":
            fields = _binding_fields(arguments)
            support_ids = _binding_ids(arguments)
            checks.update(
                {
                    "binds_exact_target_fields": fields == target_fields,
                    "binds_only_fresh_selected_evidence": bool(support_ids)
                    and support_ids.issubset(fresh_ids),
                    "binding_is_consumed": "set_field_decision" in later_tools,
                }
            )
        elif stage == "recommit":
            fields = {str(item) for item in arguments.get("fields") or []}
            checks.update(
                {
                    "commits_exact_target_fields": arguments.get("decision_status")
                    == "committed"
                    and fields == target_fields,
                    "commit_is_consumed": "stop" in later_tools,
                }
            )
        elif stage == "terminal_verification":
            checks.update(
                {
                    "stop_closes_episode": event["before"].get("terminal") is not True
                    and event["after"].get("terminal") is True,
                    "no_open_high_priority_obligation": not event["before"].get(
                        "open_high_priority_obligations"
                    ),
                }
            )
        else:
            checks["known_causal_stage"] = False

        details.append(
            {
                "turn": event["turn"],
                "stage": stage,
                "tool": tool,
                "checks": checks,
                "failures": [name for name, passed in checks.items() if not passed],
                "passed": all(checks.values()),
            }
        )

    checks = {
        "event_row_alignment": len(events) == len(rows) == int(episode.get("turns", 0)),
        "every_step_has_downstream_decision_role": bool(details)
        and all(item["passed"] for item in details),
    }
    return {
        "episode_id": str(episode.get("episode_id")),
        "dataset": str(episode.get("dataset")),
        "turns": len(events),
        "checks": checks,
        "passed": all(checks.values()),
        "failed_turns": sum(not item["passed"] for item in details),
        "details": details,
    }


def audit_batch(
    episodes: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_episode[str(row["metadata"]["episode_id"])].append(row)
    reports = [
        audit_episode_step_value(
            episode, rows_by_episode[str(episode["episode_id"])]
        )
        for episode in episodes
    ]
    by_dataset = {
        dataset: {
            "episodes": len(items),
            "passed": sum(item["passed"] for item in items),
            "failed_turns": sum(item["failed_turns"] for item in items),
        }
        for dataset in sorted({item["dataset"] for item in reports})
        for items in [[item for item in reports if item["dataset"] == dataset]]
    }
    failure_counts = Counter(
        failure
        for report in reports
        for detail in report["details"]
        for failure in detail["failures"]
    )
    return {
        "artifact": "recovery_step_value_audit_v1",
        "episodes": len(reports),
        "turns": sum(item["turns"] for item in reports),
        "episodes_passed": sum(item["passed"] for item in reports),
        "all_steps_have_downstream_decision_role": all(
            item["passed"] for item in reports
        ),
        "by_dataset": by_dataset,
        "failure_counts": dict(sorted(failure_counts.items())),
        "reports": reports,
        "scope_limit": (
            "This transcript-level dataflow audit rejects executed busywork and scope "
            "drift. It does not replace independent human judgment or measured student gain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=Path, default=DEFAULT_BATCH / "recovery_episodes.jsonl"
    )
    parser.add_argument(
        "--rows", type=Path, default=DEFAULT_BATCH / "recovery_sft_candidate.jsonl"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_batch(_jsonl(args.episodes), _jsonl(args.rows))
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "episodes": report["episodes"],
                "turns": report["turns"],
                "episodes_passed": report["episodes_passed"],
                "failure_counts": report["failure_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
