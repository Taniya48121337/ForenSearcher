"""Replay accepted VidDiff clean episodes into full evidence-dropout repair seeds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from interactive_harness import (
    CLAIM_REVISION_TOOL,
    EVIDENCE_ADJUDICATION_TOOL,
    EVIDENCE_ASSESSMENT_TOOL,
)
from llm_teacher_rollout import _normalize_harness_params
from pipeline import write_json, write_jsonl
from prepare_counterfactual_repair_seeds import mutate_clean_stop_row
from shadow_execute_generated_actions import _jsonl, _payload, _restore_harness
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, VidDiffAdaptiveMeasurementEnvironment
from viddiff_rejection_sampler import _student_visible


ACCEPTED_STATUSES = {"accepted", "executed"}


def _migrate_integrity_arguments(
    harness: Any,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    """Project legacy all-field adjudication onto the current conflict fields."""

    if name == "bind_evidence":
        state = harness.render_state()
        actions = (state.get("state_action_affordances") or {}).get(
            "executable_binding_actions"
        ) or []
        if not actions:
            return arguments
        current = copy.deepcopy(actions[0]["arguments"])
        return {
            "claim_id": current["claim_id"],
            "bindings": {
                item["field"]: list(item["support_ids"])
                for item in current.get("binding_items") or []
            },
        }
    if name not in {"assess_evidence_candidate", "adjudicate_evidence"}:
        return arguments
    state = harness.render_state()
    requirement = (state.get("evidence_adjudication") or {}).get("requirement") or {}
    competing = {
        str(evidence_id)
        for evidence_id in requirement.get("competing_evidence_ids") or []
    }
    if not requirement or not competing:
        return arguments
    migrated = copy.deepcopy(arguments)
    migrated["target_fields"] = list(requirement["target_fields"])
    if name == "assess_evidence_candidate":
        if str(migrated.get("evidence_id")) not in competing:
            return None
        actions = (state.get("state_action_affordances") or {}).get(
            "executable_evidence_assessment_actions"
        ) or []
        exact = next(
            (
                item.get("arguments")
                for item in actions
                if str((item.get("arguments") or {}).get("evidence_id"))
                == str(migrated.get("evidence_id"))
            ),
            None,
        )
        return copy.deepcopy(exact) if exact is not None else migrated
    # Legacy clean episodes may contain a claim-first partition.  Replay is a
    # migration into the current executable environment, so consume the current
    # student-visible, label-blind partition rather than preserving stale targets.
    actions = (state.get("state_action_affordances") or {}).get(
        "executable_adjudication_actions"
    ) or []
    return copy.deepcopy(actions[0]["arguments"]) if actions else migrated


def _candidate_items(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item["sample_key"]): _student_visible(item) for item in _jsonl(path)
    }


def replay_pre_stop_row(
    episode_path: Path,
    candidate_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact = json.loads(episode_path.read_text(encoding="utf-8"))
    if artifact.get("dataset") != "VidDiffBench" or artifact.get("mode") != "clean":
        raise ValueError("replayed repair seed requires a clean VidDiffBench episode")
    episode = artifact["episode"]
    turns = list(episode.get("turns") or [])
    if not turns or turns[-1].get("generated_tool") != "stop":
        raise ValueError("accepted clean episode must end with stop")
    if not episode.get("terminal_success"):
        raise ValueError("source clean episode did not terminate successfully")
    if any(turn.get("status") not in ACCEPTED_STATUSES for turn in turns):
        raise ValueError("source clean episode contains a non-accepted action")

    first_datum = copy.deepcopy(turns[0].get("sft_datum") or {})
    stop_datum = copy.deepcopy(turns[-1].get("sft_datum") or {})
    if not first_datum or not stop_datum:
        raise ValueError("source episode lacks replayable SFT snapshots")
    sample_key = str(artifact["case_id"]).removeprefix("viddiff-")
    item = candidate_items.get(sample_key)
    if item is None:
        raise ValueError(f"candidate set lacks source case: {sample_key}")

    routing_variant = str(
        (first_datum.get("metadata") or {}).get("resolved_routing_variant")
        or (first_datum.get("metadata") or {}).get("routing_variant")
        or "balanced"
    )
    if routing_variant not in VidDiffAdaptiveMeasurementEnvironment.ROUTING_VARIANTS:
        routing_variant = "balanced"
    environment = VidDiffAdaptiveMeasurementEnvironment(
        item, routing_variant=routing_variant
    )
    harness = _restore_harness(first_datum)
    harness.environment = environment
    harness.tool_registry = copy.deepcopy(VIDDIFF_TOOL_REGISTRY)
    harness.tool_registry.extend(
        [
            copy.deepcopy(CLAIM_REVISION_TOOL),
            copy.deepcopy(EVIDENCE_ASSESSMENT_TOOL),
            copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL),
        ]
    )

    replayed: list[dict[str, Any]] = []
    for turn in turns[:-1]:
        name = str(turn.get("generated_tool") or "")
        if name == "bind_evidence" and harness.render_state().get("eligible_tools") == [
            "revise_claim_fields"
        ]:
            claim_id = next(iter(harness._claim_revision_candidates()))
            candidates = harness._claim_revision_candidates()[claim_id]
            revision = harness.execute_tool(
                "revise_claim_fields",
                {
                    "claim_id": claim_id,
                    "field_updates": [
                        {"field": field_name, "value": item["value"]}
                        for field_name, item in candidates.items()
                    ],
                },
                f"repair-replay-{int(turn['turn']):03d}-revision",
            )
            replayed.append(
                {
                    "turn": int(turn["turn"]),
                    "tool": "revise_claim_fields",
                    "status": revision.get("status"),
                }
            )
        arguments = _normalize_harness_params(
            name, copy.deepcopy(turn.get("generated_arguments") or {})
        )
        arguments = _migrate_integrity_arguments(harness, name, arguments)
        if arguments is None:
            continue
        if name not in (harness.render_state().get("eligible_tools") or []):
            raise ValueError(
                f"legacy replay action {name} is not eligible after migration; "
                f"eligible={harness.render_state().get('eligible_tools')}, "
                f"revision_candidates={harness._claim_revision_candidates()}"
            )
        result = harness.execute_tool(
            name, arguments, f"repair-replay-{int(turn['turn']):03d}"
        )
        if result.get("status") not in ACCEPTED_STATUSES:
            raise ValueError(
                f"clean replay diverged at turn {turn.get('turn')} {name}: {result}"
            )
        replayed.append(
            {
                "turn": int(turn["turn"]),
                "tool": name,
                "status": result.get("status"),
            }
        )

    state = harness.render_state()
    if state.get("terminal") is True or state.get("eligible_tools") != ["stop"]:
        raise ValueError(
            f"replay did not reach the pre-stop state: {state.get('eligible_tools')}"
        )
    initial_payload = copy.deepcopy(_payload(first_datum))
    initial_payload["current_harness_state"] = state
    initial_payload.pop("past_execution_history", None)
    initial_payload.pop("latest_environment_observation", None)
    initial_payload.pop("execution_frontier", None)
    stop_datum["messages"][1]["content"] = json.dumps(
        initial_payload, ensure_ascii=False, separators=(",", ":")
    )
    metadata = stop_datum.setdefault("metadata", {})
    metadata.update(
        {
            "dataset": "VidDiffBench",
            "case_id": str(artifact["case_id"]),
            "source_split": str(episode.get("source_split") or "external_holdout"),
            "data_split": str(episode.get("source_split") or "external_holdout"),
            "routing_variant": routing_variant,
            "resolved_routing_variant": routing_variant,
            "replayed_clean_actions": replayed,
            "teacher_action_used": False,
            "oracle_or_verifier_used": False,
        }
    )
    return stop_datum


def build(
    episode_paths: list[Path],
    candidates_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = _candidate_items(candidates_path)
    source_rows = [replay_pre_stop_row(path, items) for path in episode_paths]
    seeds = [mutate_clean_stop_row(row) for row in source_rows]
    return seeds, {
        "artifact": "viddiff_replayed_counterfactual_repair_seeds_v1",
        "source_episodes": [str(path) for path in episode_paths],
        "student_visible_candidates": str(candidates_path),
        "rows": len(seeds),
        "case_ids": [str((seed.get("metadata") or {}).get("case_id")) for seed in seeds],
        "mutation_operator": "evidence_dropout",
        "clean_actions_replayed": sum(
            len((row.get("metadata") or {}).get("replayed_clean_actions") or [])
            for row in source_rows
        ),
        "teacher_actions_used": False,
        "oracle_or_verifier_used": False,
        "external_teacher_called": False,
        "media_changed": False,
        "claim_values_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--viddiff-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds, manifest = build(args.episode, args.viddiff_candidates)
    write_jsonl(args.output / "seeds.jsonl", seeds)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
