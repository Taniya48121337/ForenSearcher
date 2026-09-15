"""Compile verifier-only transition supervision beside standard agent SFT JSONL.

The sidecar never changes the inference conversation format.  It records the
visible action surface, state progress, action family, and causal audit labels
needed for balanced SFT, preference construction, or later credit assignment.
Hidden dataset labels and oracle decisions are deliberately absent.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def _initial_state(episode: dict[str, Any]) -> dict[str, Any]:
    return json.loads(episode["messages"][1]["content"])["harness_state"]


def _action_family(tool_name: str, registry: list[dict[str, Any]]) -> str:
    entry = next((item for item in registry if item.get("name") == tool_name), {})
    domain = entry.get("domain")
    if domain == "external_media":
        return "external_semantic_action"
    if domain == "harness_state":
        return "harness_state_action"
    return "terminal_action"


def compile_transition_sidecar(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    registry = bundle.get("harness_config", {}).get("tool_registry", [])
    audit_episodes = bundle.get("step_utility_report", {}).get("episodes", {})
    rows: list[dict[str, Any]] = []
    for short_name, episode_key in (("clean", "clean_episode"), ("repair", "repair_episode")):
        episode = bundle[episode_key]
        turns = episode["turn_history"]
        if not turns:
            continue
        audit_by_turn = {
            item["turn_idx"]: item for item in audit_episodes.get(short_name, [])
        }
        before = _initial_state(episode)
        trajectory_id = episode["trajectory_id"]
        for turn in turns:
            after = turn["observation"].get("harness_state", before)
            chosen = turn["tool_name"]
            audit = audit_by_turn.get(turn["turn_idx"], {})
            eligible = list(before.get("eligible_tools", []))
            evidence_before = set(before.get("available_evidence_ids", []))
            evidence_after = set(after.get("available_evidence_ids", []))
            rows.append(
                {
                    "state_id": f"{trajectory_id}:turn-{turn['turn_idx']:03d}",
                    "case_id": bundle["case_id"],
                    "trajectory_id": trajectory_id,
                    "episode_mode": short_name,
                    "turn_index": turn["turn_idx"],
                    "eligible_actions": eligible,
                    "chosen_action": {
                        "tool_name": chosen,
                        "arguments": copy.deepcopy(turn["params"]),
                    },
                    "candidate_actions_unjudged": [name for name in eligible if name != chosen],
                    "transition": {
                        "open_obligations_before": list(before.get("open_high_priority_obligations", [])),
                        "open_obligations_after": list(after.get("open_high_priority_obligations", [])),
                        "evidence_added": sorted(evidence_after - evidence_before),
                        "terminal_before": bool(before.get("terminal")),
                        "terminal_after": bool(after.get("terminal")),
                    },
                    "step_labels": {
                        "execution_status": turn["observation"].get("status"),
                        "reasoning_grounded": audit.get("reasoning_grounded"),
                        "downstream_consumed": audit.get("downstream_consumed"),
                        "causally_necessary": audit.get("causally_necessary"),
                        "observation_sensitive": audit.get("observation_sensitive"),
                        "branch_enabling_measurement": audit.get("branch_enabling_measurement", False),
                        "decision_role": audit.get("decision_role"),
                        "utility_verdict": audit.get("verdict"),
                    },
                    "training_weights": {
                        "episode_normalized": 1.0 / len(turns),
                        "action_family": _action_family(chosen, registry),
                    },
                    "visibility": "verifier_sidecar_not_rendered_to_student",
                }
            )
            before = after
    return rows

