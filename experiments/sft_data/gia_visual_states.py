"""Generate task-conditioned GIA decisions from an actual visual HI report."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

from gia_states import ROOT, load
from cli_generator import CliGenerator
from foren_searcher import GlobalStore
from foren_searcher.local_models import TransformersGiaPolicy


def make_states(spec_file):
    spec = load(spec_file)
    source = ROOT / "runs" / spec["source_run"]
    config = load(source / "run_config.json")
    events = [json.loads(line) for line in (source / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    last = [event["payload"] for event in events
            if event["actor_role"] == "GIA" and event["event_type"] == "model_input"][-1]
    template = copy.deepcopy(last["model_view"])
    if not template["latest_reports"]:
        raise ValueError("The source must contain a completed HI report")
    template["original_materials"] = [item for item in template["original_materials"]
                                      if item.get("record_type") == "evidence"]
    for report in template["latest_reports"]:
        report["materials"] = [item for item in report["materials"] if item.startswith("E")]
    template["runtime_constraints"] = {"round_index": 2, "can_create_round": True,
                                       "turn_index": 1, "max_turns": 8, "remaining_turns": 7}
    states = []
    evidence_ids = list(dict.fromkeys(item["evidence_id"] for item in
        template["original_materials"] + template["current_materials"] if "evidence_id" in item))
    for index, item in enumerate(spec["tasks"], 1):
        view = copy.deepcopy(template)
        view["task"] = item["question"]
        states.append({"id": f"state_{index:02d}", "view": view, "actions": last["actions"],
                       "reference_expectation": item["expectation"],
                       "provenance": {"dataset": config["dataset"], "case": str(config["case"]),
                           "source_run": spec["source_run"], "usage_split": config["usage"],
                           "source_evidence_ids": evidence_ids,
                           "input_source": "offline_state_from_actual_visual_investigation",
                           "report_source": "actual_hi_report_retained_without_rewriting",
                           "state_changes": "new_global_question_and_available_round; previous_global_claim_card_removed; report_findings_and_evidence_retained",
                           "is_executed_tool_trajectory": False}})
    return source, states


async def main(args):
    source, states = make_states(args.spec)
    if args.preview:
        print(json.dumps({"source": source.name, "states": [
            {"id": state["id"], "task": state["view"]["task"],
             "reports": len(state["view"]["latest_reports"]),
             "evidence_ids": state["provenance"]["source_evidence_ids"]} for state in states]},
             ensure_ascii=False))
        return
    destination = ROOT / "gia_decisions" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    store = GlobalStore(source / "store.sqlite3", source / "artifacts")
    instruction = (ROOT / "actor_gia_r2.txt").read_text(encoding="utf-8")
    for state in states:
        folder = destination / state["id"]
        folder.mkdir()
        (folder / "reference.json").write_text(json.dumps({k: v for k, v in state.items()
            if k not in {"view", "actions"}}, ensure_ascii=False, indent=2), encoding="utf-8")
        generator = CliGenerator(store=store, output_dir=folder / "requests", instruction=instruction)
        decision = await TransformersGiaPolicy(generator).decide(model_view=state["view"], history=(),
                                                                 actions=tuple(state["actions"]))
        (folder / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"state": state["id"], "decision": decision, "executed": False},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--preview", action="store_true")
    asyncio.run(main(parser.parse_args()))
