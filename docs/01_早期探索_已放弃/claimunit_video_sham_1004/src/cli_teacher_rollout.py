"""Codex CLI transport for the same executable teacher rollout used by the API backend."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from llm_teacher_rollout import (
    TeacherAction,
    _substantive_reasoning,
    _write_bundle,
    build_llm_teacher_bundle,
    teacher_tool_schemas,
    validate_llm_teacher_bundle,
)
from evidence_choice_search import build_evidence_choice_teacher
from pipeline import EDITED, ROOT, SOURCE, decode
from step_utility_auditor import apply_step_utility_filter, audit_step_utility


DEFAULT_NODE = Path(
    r"C:\Users\Zya\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
DEFAULT_CODEX_JS = ROOT / "tools" / "codex_cli" / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
DEFAULT_ACTION_SCHEMA = ROOT / "tools" / "codex_cli" / "teacher_action.schema.json"


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _observation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "status",
        "authority",
        "reason",
        "error_type",
        "error",
        "accepted_fields",
        "rejected_fields",
        "updated_fields",
        "decision_status",
        "created_evidence_ids",
        "claim_id",
        "stop_decision",
    )
    return {key: copy.deepcopy(payload[key]) for key in summary_keys if key in payload}


def compact_teacher_context(input_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a decision-sufficient view without duplicating historical harness states."""
    system_instruction = ""
    initial_payload: dict[str, Any] = {}
    current_state: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    pending_action: dict[str, Any] | None = None
    latest_observation: dict[str, Any] = {}

    for item in input_items:
        if item.get("role") == "system" and not system_instruction:
            system_instruction = str(item.get("content", ""))
            continue
        if item.get("role") == "user" and not initial_payload:
            initial_payload = _json_object(item.get("content")) or {"task": item.get("content", "")}
            state = initial_payload.get("harness_state")
            if isinstance(state, dict):
                current_state = copy.deepcopy(state)
            continue
        if item.get("type") == "function_call":
            arguments = _json_object(item.get("arguments")) or {}
            prior_reasoning = arguments.pop("reasoning", None)
            pending_action = {
                "tool_name": item.get("name"),
                "arguments": arguments,
                "call_id": item.get("call_id"),
            }
            if isinstance(prior_reasoning, str) and prior_reasoning.strip():
                pending_action["reasoning"] = prior_reasoning
            continue
        if item.get("type") == "function_call_output":
            observation = _json_object(item.get("output")) or {}
            state = observation.get("harness_state")
            if isinstance(state, dict):
                current_state = copy.deepcopy(state)
            latest_observation = _observation_summary(observation)
            history.append({
                **(pending_action or {"call_id": item.get("call_id")}),
                "observation": copy.deepcopy(latest_observation),
            })
            pending_action = None

    task_summary = {
        key: copy.deepcopy(value)
        for key, value in initial_payload.items()
        if key not in {"harness_state", "tool_registry"}
    }
    return {
        "system_instruction": system_instruction,
        "task": task_summary,
        "current_harness_state": current_state,
        "action_history": history,
        "latest_observation_summary": latest_observation,
        "compression_contract": (
            "The complete trajectory is stored by the harness. This view removes only duplicate historical states; "
            "current_harness_state is authoritative for the next decision."
        ),
    }


def _usage_from_events(stdout: str) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in totals):
                for key in totals:
                    token_count = value.get(key)
                    if isinstance(token_count, int):
                        totals[key] = max(totals[key], token_count)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in stdout.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    if totals["total_tokens"] == 0 and (totals["input_tokens"] or totals["output_tokens"]):
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def action_schema_for_tools(template_path: Path, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Constrain the CLI's structured output to the tools enabled for this episode."""
    schema = json.loads(template_path.read_text(encoding="utf-8"))
    schema["properties"]["tool_name"]["enum"] = [tool["name"] for tool in tools]
    return schema


class CodexCLITeacher:
    """Policy-only teacher backed by an authenticated local Codex CLI session."""

    source = "codex_cli"

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        node_path: Path = DEFAULT_NODE,
        codex_js_path: Path = DEFAULT_CODEX_JS,
        action_schema_path: Path = DEFAULT_ACTION_SCHEMA,
        timeout_s: int = 300,
    ) -> None:
        self.model = model
        self.node_path = Path(node_path)
        self.codex_js_path = Path(codex_js_path)
        self.action_schema_path = Path(action_schema_path)
        self.timeout_s = timeout_s
        for path in (self.node_path, self.codex_js_path, self.action_schema_path):
            if not path.exists():
                raise FileNotFoundError(f"required CLI runtime file is missing: {path}")

    @staticmethod
    def _prompt(input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        compact_context = compact_teacher_context(input_items)
        return (
            "You are a policy-only teacher for an executable video-forensic harness. "
            "Do not call shell, file, web, or any Codex-side tool. The JSON below is a decision-sufficient compact "
            "view of the visible transcript plus the available harness functions. Choose exactly one next harness "
            "function. Return only the object "
            "required by the output schema. Put a grounded explanation of what the latest observation established, "
            "why that fact changes the decision, and how the chosen action follows in reasoning. Put the chosen function parameters, "
            "excluding reasoning, as a valid JSON object encoded in arguments_json. "
            "Write reasoning as 2-3 natural sentences, normally 35-80 English words. Organize it to fit the state: "
            "you may lead with an evidence contrast, a consequence, an uncertainty, a rejected alternative, or the "
            "reason this tool is now warranted. Never use the phrases 'blocking gap', 'remaining gap', 'what remains', "
            "or 'what still blocks'. Do not use 'warranted', 'next step', 'next action', 'next move', or repeated "
            "'rather than' contrasts as action-justification formulas. Do not recite "
            "every field, goal, or constraint, and do not reuse a fixed template, causal-connective sequence, or sentence "
            "opening across turns. "
            "Never invent observations, EvidenceAtom IDs, hidden labels, or oracle data. Before claim creation, "
            "binding, commitment, and especially stop, independently compare the current task identity, logical "
            "source/suspect or A/B role assignment, scope, bound EvidenceAtom provenance, observation values, and "
            "claim values. An active or supported flag is not semantic proof. A mismatched task/entity/scope, a "
            "swapped role marker, an interval outside the visible scope, a partial or empty result, or a value that "
            "contradicts its bound claim blocks claim creation, binding, commitment, and stop until it is resolved. "
            "When only some bound values conflict, reopen and repair exactly that minimal field subset while "
            "preserving independently supported fields. A rejected tool call produces no evidence and cannot justify "
            "a later decision. Use an ordinary measurement, assessment, localized reopen, or "
            "abstention appropriate to the visible problem; do not wait for or infer a hidden repair recipe. "
            "Obey ordinary function preconditions, but do not infer the semantic answer from action cardinality. "
            "If active measurements conflict, do not default to the latest atom, a vote count, or confidence alone. "
            "Compare exact-query fit, order-consistency, direct counterexamples, abstention semantics, and visible "
            "rationales; the reasoning for claim creation must identify the chosen atom and why the alternatives are "
            "weaker.\n\n"
            "COMPACT_VISIBLE_CONTEXT_JSON:\n"
            + json.dumps(compact_context, ensure_ascii=False, separators=(",", ":"))
            + "\n\nAVAILABLE_FUNCTIONS_JSON:\n"
            + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        )

    def next_action(self, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> TeacherAction:
        prompt = self._prompt(input_items, tools)
        with tempfile.TemporaryDirectory(prefix="claimunit_cli_teacher_") as directory:
            output_path = Path(directory) / "last_message.json"
            dynamic_schema_path = Path(directory) / "teacher_action.schema.json"
            dynamic_schema_path.write_text(
                json.dumps(action_schema_for_tools(self.action_schema_path, tools), ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                str(self.node_path),
                str(self.codex_js_path),
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
                str(dynamic_schema_path),
                "--output-last-message",
                str(output_path),
                "--model",
                self.model,
                "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_s,
                cwd=directory,
                check=False,
            )
            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout)[-4000:]
                raise RuntimeError(f"Codex CLI teacher failed with exit code {completed.returncode}: {diagnostic}")
            if not output_path.exists():
                raise RuntimeError("Codex CLI did not write its structured final response")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("Codex CLI returned invalid structured JSON") from error

        reasoning = payload.get("reasoning")
        tool_name = payload.get("tool_name")
        if not _substantive_reasoning(reasoning):
            raise RuntimeError("Codex CLI teacher reasoning is missing or too short")
        allowed_names = {tool["name"] for tool in tools}
        if tool_name not in allowed_names:
            raise RuntimeError(f"Codex CLI selected an unavailable function: {tool_name}")
        try:
            params = json.loads(payload.get("arguments_json", ""))
        except json.JSONDecodeError as error:
            raise RuntimeError("Codex CLI arguments_json is not valid JSON") from error
        if not isinstance(params, dict):
            raise RuntimeError("Codex CLI arguments_json must encode an object")

        response_id = f"cli_resp_{uuid.uuid4().hex}"
        call_id = f"cli_call_{uuid.uuid4().hex}"
        raw_arguments = json.dumps(
            {"reasoning": reasoning, **copy.deepcopy(params)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return TeacherAction(
            response_id=response_id,
            call_id=call_id,
            tool_name=tool_name,
            reasoning=reasoning,
            params=params,
            raw_arguments=raw_arguments,
            usage=_usage_from_events(completed.stdout),
            replay_items=[
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": raw_arguments,
                }
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VideoSham teacher pilot through an authenticated Codex CLI")
    parser.add_argument("--model", default=os.environ.get("CLAIMUNIT_CLI_MODEL", "gpt-5.4"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "student_view" / "case_manifest.json")
    parser.add_argument("--oracle", type=Path, default=ROOT / "oracle" / "claim_oracle.json")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--suspect", type=Path, default=EDITED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-clean-turns", type=int, default=12)
    parser.add_argument("--max-repair-turns", type=int, default=10)
    parser.add_argument("--candidates-per-turn", type=int, default=3)
    parser.add_argument("--proposal-rounds-per-turn", type=int, default=2)
    parser.add_argument(
        "--legacy-full-control",
        action="store_true",
        help="Diagnostic compatibility mode. The default causal view removes answer-like control affordances.",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    base_teacher = CodexCLITeacher(model=args.model)
    teacher = (
        base_teacher
        if args.legacy_full_control
        else build_evidence_choice_teacher(
            base_teacher,
            candidates_per_turn=args.candidates_per_turn,
            proposal_rounds_per_turn=args.proposal_rounds_per_turn,
        )
    )
    bundle = build_llm_teacher_bundle(
        manifest=manifest,
        source_frames=decode(args.source),
        suspect_frames=decode(args.suspect),
        teacher=teacher,
        source_path=args.source,
        suspect_path=args.suspect,
        max_clean_turns=args.max_clean_turns,
        max_repair_turns=args.max_repair_turns,
    )
    step_report = audit_step_utility(bundle)
    apply_step_utility_filter(bundle, step_report)
    report = validate_llm_teacher_bundle(bundle, oracle=oracle)
    _write_bundle(bundle, report, promote=args.promote, output_dir=args.output_dir)
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"CLI teacher validation failed: {failed}")
    print(
        json.dumps(
            {
                "passed": True,
                "teacher_source": "codex_cli",
                "policy_view": getattr(teacher, "policy_view", "legacy_full_control"),
                "model": args.model,
                "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                "sft_turns_after_step_audit": len(bundle["clean_turn_sft"]) + len(bundle["repair_turn_sft"]),
                "step_utility": step_report["summary"],
                "promoted": args.promote,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
