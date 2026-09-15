"""Execute a recorded GIA read on a copied store and generate its next decision."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sqlite3
import sys

from gia_states import ROOT, load
from cli_generator import CliGenerator
from foren_searcher import GlobalStore
from foren_searcher.local_models import TransformersGiaPolicy, extract_json_object
from foren_searcher.memory import MemoryPolicy, RecordAccess
from foren_searcher.orchestration import GiaRunner


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def main(args):
    source = ROOT / "gia_decisions" / args.source
    destination = ROOT / "gia_decisions" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    instruction = (ROOT / "actor_gia_r2.txt").read_text(encoding="utf-8")
    for index, previous in enumerate(args.states, 1):
        old = source / previous
        request_dir = old / "requests/request_0001"
        request = load(request_dir / "request.json")
        action = load(old / "decision.json")
        raw = extract_json_object((request_dir / "response.json").read_text(encoding="utf-8"))
        transport = load(request_dir / "transport.json")
        reference = load(old / "reference.json")
        provenance = reference["provenance"]
        if transport["returncode"] or transport.get("external_tool_items") or raw != action:
            raise ValueError("Source decision is not the clean original actor output")
        if action.get("action") != "READ_RECORD":
            raise ValueError("Only recorded evidence reads are supported")
        if action["arguments"]["record_id"] not in provenance["source_evidence_ids"]:
            raise ValueError("Read target is outside the constructed state's evidence")
        view = copy.deepcopy(request["payload"]["global_view"])
        constraints = view["runtime_constraints"]
        if constraints["remaining_turns"] < 1:
            raise ValueError("The recorded state has no next decision available")
        folder = destination / f"state_{index:02d}"
        folder.mkdir()
        run = ROOT / "runs" / provenance["source_run"]
        # RecordAccess audits reads, so its writes belong to this continuation.
        with sqlite3.connect((run / "store.sqlite3").as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(folder / "store.sqlite3") as copied:
                original.backup(copied)
        store = GlobalStore(folder / "store.sqlite3", run / "artifacts")
        access = RecordAccess(store=store, role="GIA", actor_id="GIA",
                              round_index=constraints["round_index"], policy=MemoryPolicy())
        observation = GiaRunner._execute_memory_action(action, access)
        history = copy.deepcopy(request["payload"].get("memory_history", []))
        history.append({"turn_index": constraints["turn_index"], "output": action,
                        "observation": observation})
        constraints["turn_index"] += 1
        constraints["remaining_turns"] -= 1
        reference["id"] = folder.name
        reference["provenance"] = {**provenance,
            "input_source": "offline_state_after_executed_memory_read",
            "previous_construction": args.source, "previous_state": previous,
            "executed_memory_actions": len(history), "external_tool_actions": 0,
            "state_changes": "same_question_and_reports_plus_actual_memory_observation",
            "is_executed_tool_trajectory": False}
        save(folder / "reference.json", reference)
        save(folder / "executed_history.json", history)
        generator = CliGenerator(store=store, output_dir=folder / "requests", instruction=instruction)
        decision = await TransformersGiaPolicy(generator).decide(model_view=view, history=tuple(history),
            actions=tuple(request["payload"]["available_actions"]))
        save(folder / "decision.json", decision)
        print(json.dumps({"state": folder.name, "previous_state": previous,
                          "executed_read": action, "next_decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--states", nargs="+", required=True)
    asyncio.run(main(parser.parse_args()))
