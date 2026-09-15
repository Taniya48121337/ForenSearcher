"""Audit that recovery adjudication is evidence-first rather than claim-first."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import write_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assistant_calls(episode: dict[str, Any], name: str) -> list[dict[str, Any]]:
    result = []
    for message in episode.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == name:
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result.append(arguments)
    return result


def _final_state(episode: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(episode.get("messages") or []):
        if message.get("role") != "tool" or message.get("name") != "stop":
            continue
        payload = message.get("content") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload.get("harness_state") or {}
    return {}


def _confidence(atom: dict[str, Any]) -> float:
    value = (atom.get("observation") or {}).get("confidence")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else -1.0


def _expected_selection(state: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    store = state.get("evidence_store") or {}
    target_fields = list(decision.get("target_fields") or [])
    candidates = [
        evidence_id
        for evidence_id in decision.get("considered_evidence_ids") or []
        if evidence_id in store
        and store[evidence_id].get("active", True)
        and all(
            field_name in (store[evidence_id].get("supports_fields") or [])
            and field_name in (store[evidence_id].get("observation") or {})
            for field_name in target_fields
        )
        and (store[evidence_id].get("provenance") or {}).get(
            "role_assignment_valid_for_current_contract", True
        ) is not False
    ]
    if not candidates:
        return []
    if all(
        (store[evidence_id].get("observation") or {}).get("preferred_video")
        in {"a", "b", "c"}
        for evidence_id in candidates
    ):
        representatives: dict[str, str] = {}
        for evidence_id in candidates:
            atom = store[evidence_id]
            observation = atom.get("observation") or {}
            method = str(
                observation.get("measurement_method")
                or (atom.get("provenance") or {}).get("method")
                or evidence_id
            )
            incumbent = representatives.get(method)
            if incumbent is None or (_confidence(atom), evidence_id) > (
                _confidence(store[incumbent]), incumbent
            ):
                representatives[method] = evidence_id
        votes = Counter(
            (store[evidence_id].get("observation") or {}).get("preferred_video")
            for evidence_id in representatives.values()
        )
        directional = [value for value in ("a", "b") if votes[value] >= 2]
        if len(directional) == 1 and votes["b" if directional[0] == "a" else "a"] == 0:
            selected_value = directional[0]
        elif votes["c"]:
            selected_value = "c"
        elif votes["a"] != votes["b"]:
            selected_value = "a" if votes["a"] > votes["b"] else "b"
        else:
            return []
        matching = [
            evidence_id
            for evidence_id in candidates
            if (store[evidence_id].get("observation") or {}).get("preferred_video")
            == selected_value
        ]
        stable = [
            evidence_id
            for evidence_id in matching
            if selected_value == "c"
            or (store[evidence_id].get("observation") or {}).get("order_consistent")
            is not False
        ]
        pool = stable or matching
    else:
        pool = candidates
    return [max(pool, key=lambda evidence_id: (_confidence(store[evidence_id]), evidence_id))]


def audit(episodes: list[dict[str, Any]], baseline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    baseline_selected = {
        str(episode["episode_id"]): _assistant_calls(episode, "adjudicate_evidence")[0].get("selected_evidence_ids")
        for episode in baseline or []
        if _assistant_calls(episode, "adjudicate_evidence")
    }
    records = []
    for episode in episodes:
        calls = _assistant_calls(episode, "adjudicate_evidence")
        if not calls:
            continue
        state = _final_state(episode)
        claim_id = str(calls[0]["claim_id"])
        decision = (state.get("evidence_adjudications") or {}).get(claim_id) or {}
        expected = _expected_selection(state, decision)
        selected = list(calls[0].get("selected_evidence_ids") or [])
        claim = next(
            item for item in state.get("candidate_claims") or []
            if str(item.get("claim_id")) == claim_id
        )
        bindings = (state.get("field_bindings") or {}).get(claim_id) or {}
        selected_store = state.get("evidence_store") or {}
        target_fields = list(decision.get("target_fields") or [])
        bound_only_selected = all(
            set((bindings.get(field_name) or {}).get("support_ids") or []).issubset(selected)
            and bool((bindings.get(field_name) or {}).get("support_ids"))
            for field_name in target_fields
        )
        selected_matches_final_claim = all(
            any(
                (selected_store.get(evidence_id) or {}).get("observation", {}).get(field_name)
                == (claim.get("fields") or {}).get(field_name, {}).get("value")
                for evidence_id in selected
            )
            for field_name in target_fields
        )
        records.append(
            {
                "episode_id": episode["episode_id"],
                "case_id": episode["case_id"],
                "selected_evidence_ids": selected,
                "expected_label_blind_selection": expected,
                "selection_matches_rule": selected == expected,
                "bound_only_selected": bound_only_selected,
                "selected_matches_final_claim": selected_matches_final_claim,
                "changed_from_baseline": baseline_selected.get(str(episode["episode_id"])) != selected,
                "baseline_selected_evidence_ids": baseline_selected.get(str(episode["episode_id"])),
            }
        )
    checks = {
        "all_adjudications_match_label_blind_rule": all(item["selection_matches_rule"] for item in records),
        "all_target_bindings_use_selected_only": all(item["bound_only_selected"] for item in records),
        "all_selected_values_match_final_claim": all(item["selected_matches_final_claim"] for item in records),
    }
    return {
        "artifact": "recovery_evidence_first_adjudication_audit_v1",
        "adjudicated_episodes": len(records),
        "changed_from_baseline": sum(item["changed_from_baseline"] for item in records),
        "checks": checks,
        "passed": bool(records) and all(checks.values()),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        _jsonl(args.episodes),
        _jsonl(args.baseline) if args.baseline else None,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
