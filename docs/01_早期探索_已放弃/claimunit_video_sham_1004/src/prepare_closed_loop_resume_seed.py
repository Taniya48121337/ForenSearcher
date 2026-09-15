"""Create a resumable seed after an accepted harness-state turn.

This utility is intentionally limited to state-only turns. External media or feature
tools cannot be replayed from a JSON trace because their observations belong to the
real environment, not the harness snapshot.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from interactive_harness import EVIDENCE_ADJUDICATION_TOOL, EVIDENCE_ASSESSMENT_TOOL
from llm_teacher_rollout import _normalize_harness_params
from shadow_execute_generated_actions import _restore_harness
from vcsl_batch import VCSL_TOOL_REGISTRY
from viddiff_batch import VIDDIFF_TOOL_REGISTRY
from videosham_batch import MULTIFAMILY_TOOL_REGISTRY


STATE_ONLY_TOOLS = {
    "create_candidate_claim",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
    "stop",
}


def _dataset_tool_registry(dataset: str) -> list[dict[str, Any]]:
    registries = {
        "VCSL": VCSL_TOOL_REGISTRY,
        "VidDiffBench": VIDDIFF_TOOL_REGISTRY,
        "VideoSham": MULTIFAMILY_TOOL_REGISTRY,
    }
    if dataset not in registries:
        raise ValueError(f"unsupported resume dataset: {dataset}")
    return copy.deepcopy(registries[dataset])


def _load_base_seed(
    seed_path: Path,
    *,
    dataset: str,
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if (row.get("metadata") or {}).get("dataset") == dataset
        and (row.get("metadata") or {}).get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "base seed must contain exactly one row matching the resumed "
            f"dataset/case; found {len(matches)} for {dataset}/{case_id}"
        )
    datum = matches[0]
    payload = json.loads(datum["messages"][1]["content"])
    return copy.deepcopy(datum), payload


def build_resume_seed(
    episode_path: Path,
    through_turn: int,
    base_seed_path: Path | None = None,
) -> dict[str, Any]:
    artifact = json.loads(episode_path.read_text(encoding="utf-8"))
    episode = artifact["episode"]
    turn = next(
        (item for item in episode.get("turns") or [] if int(item.get("turn") or 0) == through_turn),
        None,
    )
    if turn is None:
        raise ValueError(f"turn {through_turn} is absent from {episode_path}")
    if turn.get("status") not in {"accepted", "executed"}:
        raise ValueError(f"turn {through_turn} is not accepted/executed")
    name = str(turn.get("generated_tool") or "")
    if name not in STATE_ONLY_TOOLS:
        raise ValueError(f"turn {through_turn} uses non-resumable external tool: {name}")
    datum = copy.deepcopy(turn.get("sft_datum") or {})
    if not datum:
        raise ValueError(f"turn {through_turn} has no saved SFT pre-action snapshot")

    prefix_payload = json.loads(datum["messages"][1]["content"])
    prefix_state = prefix_payload.get("current_harness_state") or {}
    if "task_contract" not in prefix_state or not (
        (prefix_payload.get("task") or {}).get("claim_schema")
    ):
        base_datum = None
        base_payload = None
        for earlier in reversed(episode.get("turns") or []):
            if int(earlier.get("turn") or 0) > through_turn:
                continue
            candidate = earlier.get("sft_datum") or {}
            if not candidate:
                continue
            candidate_payload = json.loads(candidate["messages"][1]["content"])
            candidate_state = candidate_payload.get("current_harness_state") or {}
            if "task_contract" in candidate_state and (
                (candidate_payload.get("task") or {}).get("claim_schema")
            ):
                base_datum = copy.deepcopy(candidate)
                base_payload = candidate_payload
                break
        if base_datum is None or base_payload is None:
            if base_seed_path is None:
                raise ValueError(
                    "no earlier full harness snapshot is available for restoration; "
                    "provide --base-seed for a nested resume"
                )
            base_datum, base_payload = _load_base_seed(
                base_seed_path,
                dataset=str(artifact["dataset"]),
                case_id=str(artifact["case_id"]),
            )
        merged_state = copy.deepcopy(base_payload["current_harness_state"])
        for key in (
            "candidate_claims",
            "field_bindings",
            "evidence_store",
            "evidence_assessments",
            "evidence_adjudications",
            "evidence_adjudication",
            "evidence_assessment",
            "dropped_evidence_ids",
            "terminal",
            "stop_decision",
        ):
            if key in prefix_state:
                merged_state[key] = copy.deepcopy(prefix_state[key])
        open_goals = set(prefix_state.get("open_high_priority_obligations") or [])
        for goal in merged_state.get("case_goals") or []:
            if goal.get("priority") == "high":
                goal["status"] = (
                    "open" if str(goal.get("goal_id")) in open_goals else "satisfied"
                )
        prefix_payload = copy.deepcopy(base_payload)
        prefix_payload["current_harness_state"] = merged_state
        prefix_payload["task"] = copy.deepcopy(base_payload["task"])
        datum["messages"][1]["content"] = json.dumps(
            prefix_payload, ensure_ascii=False, separators=(",", ":")
        )

    harness = _restore_harness(datum)
    harness.tool_registry = _dataset_tool_registry(str(artifact["dataset"]))
    if not any(
        item.get("name") == "adjudicate_evidence" for item in harness.tool_registry
    ):
        harness.tool_registry.append(copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL))
    if not any(
        item.get("name") == "assess_evidence_candidate"
        for item in harness.tool_registry
    ):
        harness.tool_registry.append(copy.deepcopy(EVIDENCE_ASSESSMENT_TOOL))
    arguments = _normalize_harness_params(name, copy.deepcopy(turn.get("generated_arguments") or {}))
    try:
        result = harness.execute_tool(name, arguments, f"resume-prefix-{through_turn:03d}")
    except ValueError as error:
        state = harness.render_state()
        raise ValueError(
            f"prefix action replay failed: {error}; input={arguments}; "
            f"assessment={state.get('evidence_assessment')}; "
            f"actions={(state.get('state_action_affordances') or {}).get('executable_evidence_assessment_actions')}"
        ) from error
    if result.get("status") not in {"accepted", "executed"}:
        raise ValueError(f"prefix action could not be replayed: {result}")

    payload = json.loads(datum["messages"][1]["content"])
    payload["current_harness_state"] = harness.render_state()
    for stale_key in (
        "past_execution_history",
        "latest_environment_observation",
        "execution_frontier",
        "evidence_assessment_frontier",
        "evidence_adjudication_frontier",
        "evidence_consumption_frontier",
        "grounded_reasoning_plan",
        "preexecution_revision",
    ):
        payload.pop(stale_key, None)
    datum["messages"] = [
        copy.deepcopy(datum["messages"][0]),
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    episode_mode = str(artifact.get("mode") or "clean")
    datum["metadata"] = {
        **copy.deepcopy(datum.get("metadata") or {}),
        "dataset": artifact["dataset"],
        "case_id": artifact["case_id"],
        "row_id": (
            f"{artifact['dataset']}:{artifact['case_id']}:"
            f"resume-through-{through_turn}:{episode_mode}:1"
        ),
        "trajectory_type": f"accepted_prefix_resume_{episode_mode}",
        "source_kind": episode_mode,
        "trajectory_mode": episode_mode,
        "repair_subtype": episode.get("repair_subtype"),
        "resume_through_turn": through_turn,
        "resume_source_episode": str(episode_path.name),
    }
    return datum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--through-turn", type=int, required=True)
    parser.add_argument(
        "--base-seed",
        type=Path,
        help="Earlier one-row resume seed used when the episode is itself a suffix.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = build_resume_seed(args.episode, args.through_turn, args.base_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_id": row["metadata"]["case_id"],
                "resume_through_turn": args.through_turn,
                "eligible_tools": json.loads(row["messages"][1]["content"])[
                    "current_harness_state"
                ]["eligible_tools"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
