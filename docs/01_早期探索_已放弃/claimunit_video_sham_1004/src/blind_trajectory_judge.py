"""Two-context blind process judging for UTQR-7 causal packets."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cli_teacher_rollout import DEFAULT_CODEX_JS, DEFAULT_NODE


DIMENSIONS = ("prospective_reasoning", "belief_update", "method_fit", "naturalness")

FORBIDDEN_KEYS = {
    "dataset",
    "case_id",
    "packet_id",
    "hidden_outcome_correct",
    "oracle",
    "oracle_direction",
    "verifier",
    "verifier_result",
    "validation",
    "validation_report",
    "blind_judgments",
    "source_path",
    "suspect_path",
    "video_path",
    "local_path",
    "url",
}


def _redact_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return copy.deepcopy(value)
    lowered = value.lower()
    if ":\\" in value or value.startswith(("/", "file://", "http://", "https://")):
        return "<redacted_locator>"
    if any(token in lowered for token in ("oracle", "verifier_result", "differences_gt")):
        return "<redacted_private_field>"
    return value


def blind_view(value: Any) -> Any:
    """Remove identity, answers, private locators, and previous judge outputs."""

    if isinstance(value, dict):
        return {
            str(key): blind_view(child)
            for key, child in value.items()
            if str(key).lower() not in FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [blind_view(item) for item in value]
    return _redact_scalar(value)


def _compact_state(
    state: dict[str, Any], *, include_evidence: bool = True
) -> dict[str, Any]:
    evidence = {}
    for evidence_id, atom in (state.get("evidence_store") or {}).items():
        evidence[str(evidence_id)] = {
            "kind": atom.get("kind"),
            "supports_fields": copy.deepcopy(atom.get("supports_fields") or []),
            "observation": copy.deepcopy(atom.get("observation") or {}),
            "provenance": copy.deepcopy(atom.get("provenance") or {}),
            "active": atom.get("active"),
        }
    compact = {
        "candidate_claims": copy.deepcopy(state.get("candidate_claims") or []),
        "field_bindings": copy.deepcopy(state.get("field_bindings") or {}),
        "evidence_assessments": copy.deepcopy(state.get("evidence_assessments") or {}),
        "evidence_adjudications": copy.deepcopy(state.get("evidence_adjudications") or {}),
        "open_high_priority_obligations": copy.deepcopy(
            state.get("open_high_priority_obligations") or []
        ),
        "eligible_tools": copy.deepcopy(state.get("eligible_tools") or []),
        "terminal": bool(state.get("terminal")),
    }
    if include_evidence:
        compact["evidence_store"] = evidence
    return compact


def _initial_state(episode: dict[str, Any]) -> dict[str, Any]:
    messages = episode.get("messages") or []
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (TypeError, ValueError):
            continue
        state = payload.get("harness_state")
        if isinstance(state, dict):
            return _compact_state(state, include_evidence=True)
    return {}


def _compact_turn(turn: dict[str, Any]) -> dict[str, Any]:
    result = turn.get("observation") or {}
    compact_result = {
        key: copy.deepcopy(result.get(key))
        for key in (
            "status",
            "authority",
            "reason",
            "accepted_fields",
            "rejected_fields",
            "updated_fields",
            "decision_status",
            "created_evidence_ids",
            "assessment",
            "adjudication",
            "observation",
            "evidence_atoms",
        )
        if key in result
    }
    if isinstance(result.get("harness_state"), dict):
        # Evidence observations are already present once in the episode's
        # initial state or in the tool return that created them. Repeating the
        # full store after every turn can exceed the judge transport limit and
        # adds no new process information.
        compact_result["post_action_state"] = _compact_state(
            result["harness_state"], include_evidence=False
        )
    return {
        "turn_idx": turn.get("turn_idx"),
        "reasoning": turn.get("reasoning"),
        "selected_tool": turn.get("tool_name"),
        "arguments": copy.deepcopy(turn.get("params") or {}),
        "returned_observation": compact_result,
    }


def blind_process_view(packet: dict[str, Any]) -> dict[str, Any]:
    """Project a causal packet to process evidence without repeated chat snapshots."""

    probes = {
        str(name): {
            str(key): copy.deepcopy(value)
            for key, value in (probe or {}).items()
            if key not in {"episode", "reasoning"}
        }
        for name, probe in (packet.get("probes") or {}).items()
    }
    episodes = {}
    episode_sources = {
        **{
            f"base_{name}": episode
            for name, episode in (packet.get("raw_base_episodes") or {}).items()
        },
        **{
            f"probe_{name}": episode
            for name, episode in (packet.get("raw_probe_episodes") or {}).items()
        },
    }
    for name, episode in episode_sources.items():
        episodes[str(name)] = {
            "probe_name": episode.get("probe_name") or name,
            "terminated": bool(episode.get("terminated")),
            "initial_state": _initial_state(episode),
            "turns": [_compact_turn(turn) for turn in episode.get("turn_history") or []],
            "final_claim": copy.deepcopy(episode.get("final_claim")),
        }
    projected = {
        "base_process_checks": copy.deepcopy(packet.get("base") or {}),
        "control_plane": copy.deepcopy(packet.get("control_plane") or {}),
        "step_intervention": copy.deepcopy(packet.get("step_intervention") or {}),
        "probe_outcomes": probes,
        "probe_process_episodes": episodes,
        "sft_target_checks": copy.deepcopy(packet.get("sft") or {}),
    }
    return blind_view(projected)


def judge_schema() -> dict[str, Any]:
    score_properties = {
        name: {"type": "integer", "minimum": 1, "maximum": 5}
        for name in DIMENSIONS
    }
    note_properties = {name: {"type": "string"} for name in DIMENSIONS}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dimension_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": score_properties,
                "required": list(DIMENSIONS),
            },
            "dimension_evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": note_properties,
                "required": list(DIMENSIONS),
            },
        },
        "required": ["dimension_scores", "dimension_evidence"],
    }


def _prompt(packet_view: dict[str, Any], *, perspective: str) -> str:
    return (
        "You are one blind trajectory-process judge. You cannot see the dataset name, case identity, oracle, "
        "verifier outcome, or another judge's scores. Judge only the visible pre-action state, natural-language "
        "reasoning, selected executable tool, returned observation, and the fixed counterfactual probes. "
        "Do not reward the final answer merely for being confident or terminal.\n\n"
        "Use integer scores 1-5 with demanding anchors:\n"
        "5 prospective_reasoning: nearly every semantic step states the latest material fact, a live alternative "
        "or uncertainty, the unresolved question, and why the next tool can discriminate before seeing its result.\n"
        "5 belief_update: observations visibly change or preserve beliefs and downstream actions; conflicts, empty "
        "results, and identity/scope issues cause an appropriate update rather than a scripted continuation.\n"
        "5 method_fit: each tool is well matched to the information gap, scopes and identities align, repeated calls "
        "add information, and a cheaper sufficient action was not obviously available.\n"
        "5 naturalness: concise, varied analyst notes sound like genuine decision-making, calibrate uncertainty, and "
        "never mention prompts, rubrics, checkers, oracle labels, or copy a fixed template.\n"
        "A 4 is strong with one minor lapse; 3 has a material but recoverable weakness; 2 is mostly post-hoc, "
        "template-driven, or poorly matched; 1 is ungrounded or misleading. Cite a concrete visible turn or probe in "
        "every evidence note. Do not assume missing information passed.\n"
        f"Independent perspective: {perspective}.\n\n"
        "BLIND_CAUSAL_PACKET_JSON:\n"
        + json.dumps(packet_view, ensure_ascii=False, separators=(",", ":"))
    )


def run_codex_blind_judge(
    packet: dict[str, Any],
    *,
    judge_id: str,
    perspective: str,
    model: str = "gpt-5.4",
    node_path: Path = DEFAULT_NODE,
    codex_js_path: Path = DEFAULT_CODEX_JS,
    timeout_s: int = 300,
) -> dict[str, Any]:
    visible = blind_process_view(packet)
    with tempfile.TemporaryDirectory(prefix="utqr7_blind_judge_") as directory:
        directory_path = Path(directory)
        schema_path = directory_path / "judge.schema.json"
        output_path = directory_path / "judge.json"
        schema_path.write_text(json.dumps(judge_schema(), ensure_ascii=False), encoding="utf-8")
        command = [
            str(node_path),
            str(codex_js_path),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--model",
            model,
            "-",
        ]
        completed = subprocess.run(
            command,
            input=_prompt(visible, perspective=perspective),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_s,
            cwd=directory,
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"blind judge failed with exit code {completed.returncode}: {diagnostic}")
        if not output_path.exists():
            raise RuntimeError("blind judge did not write a structured result")
        result = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        "judge_id": judge_id,
        "dimension_scores": result["dimension_scores"],
        "dimension_evidence": result["dimension_evidence"],
        "blind_to_forbidden_fields": True,
        "independent_context": True,
        "perspective": perspective,
        "model": model,
    }


def add_two_blind_judgments(
    packet: dict[str, Any],
    *,
    model: str = "gpt-5.4",
    timeout_s: int = 300,
) -> dict[str, Any]:
    judged = copy.deepcopy(packet)
    judged["blind_judgments"] = [
        run_codex_blind_judge(
            judged,
            judge_id="blind_process_judge_a",
            perspective="Audit evidence-to-belief updates and whether each action was chosen prospectively.",
            model=model,
            timeout_s=timeout_s,
        ),
        run_codex_blind_judge(
            judged,
            judge_id="blind_process_judge_b",
            perspective="Audit tool-method fit, economy, uncertainty calibration, and naturalness.",
            model=model,
            timeout_s=timeout_s,
        ),
    ]
    return judged
