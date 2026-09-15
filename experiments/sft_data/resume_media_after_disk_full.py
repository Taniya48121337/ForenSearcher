"""Resume a saved media investigation after its recorded disk-full failure."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from run_case import ROOT
from cli_generator import CliGenerator
from foren_searcher import GlobalStore
from foren_searcher.harness1 import harness1_workspace_type
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import TransformersGiaPolicy, TransformersHiPolicy
from foren_searcher.memory import MemoryPolicy
from foren_searcher.orchestration import InvestigationRuntime


async def main(args):
    run = (ROOT / "runs" / args.run).resolve()
    run.relative_to((ROOT / "runs").resolve())
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    if config["source_condition"] != "recorded_plan_execution_with_provided_evidence" or config["dataset"] not in {"videosham", "qvhighlights", "vcsl"}:
        raise ValueError("This recovery is limited to recorded-plan media investigations")
    recovery = run / "recovery_disk_full"
    if not (recovery / "store_before_recovery.sqlite3").is_file() or (recovery / "resume.json").exists():
        raise ValueError("Inspect the saved pre-recovery database and any previous resume first")
    with sqlite3.connect((run / "store.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("The saved database did not pass its integrity check")
        completed = connection.execute("SELECT count(*) FROM trajectory_events WHERE actor_id=? AND event_type='raw_model_output'", (args.workspace,)).fetchone()[0]
        outcome = connection.execute("SELECT payload_json FROM trajectory_events WHERE actor_id=? AND event_type='workspace_outcome' ORDER BY id DESC LIMIT 1", (args.workspace,)).fetchone()
    store = GlobalStore(run / "store.sqlite3", run / "artifacts")
    prior = store.workspace_task_record(args.workspace)
    if prior["run_status"] != "runtime_error" or prior["error"] != {"type": "OSError", "message": "[Errno 28] No space left on device"}:
        raise ValueError("Selected workspace is not the recorded disk-full failure")
    rounds = store.round_records()
    if rounds[-1]["status"] != "running" or prior["round_index"] != rounds[-1]["round_index"]:
        raise ValueError("Recovery requires the still-open original round")
    if any(x["run_status"] != "reported" for x in store.workspace_task_records() if x["task_id"] != args.workspace):
        raise ValueError("Other workspaces must already be reported")
    first_request = next(json.loads((p / "request.json").read_text(encoding="utf-8"))
                         for p in sorted((run / "requests").glob("request_*"))
                         if json.loads((p / "request.json").read_text(encoding="utf-8"))["payload"].get("workspace_id") == args.workspace)
    original_turn_limit = first_request["payload"]["model_view"]["runtime_constraints"]["max_turns"]
    remaining_turns = original_turn_limit - completed
    original_seconds = first_request["payload"]["model_view"]["runtime_constraints"]["remaining_seconds"]
    elapsed_seconds = json.loads(outcome[0])["elapsed_ms"] / 1000
    remaining_seconds = original_seconds - elapsed_seconds
    if remaining_turns <= 0 or remaining_seconds <= 0:
        raise ValueError("No original execution budget remains")
    request_numbers = [int(re.match(r"request_(\d+)", p.name)[1]) for p in (run / "requests").glob("request_*")]
    request_numbers += [int(re.match(r"request_(\d+)", p.name)[1]) for p in recovery.glob("request_*_unwritten")]
    generator = CliGenerator(store=store, output_dir=run / "requests", instruction=config["instruction"])
    generator.sequence = max(request_numbers)
    if generator.model_config != config["configured_model"]:
        raise ValueError("Configured teacher model differs from the interrupted run")
    def record(value):
        workspace = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace else "GIA", actor_id=workspace or "GIA",
                           event_type="raw_model_output", payload=value)
    generator.generation_observer = record
    details = {"workspace": args.workspace, "prior_status": prior, "completed_outputs": completed,
               "original_turn_limit": original_turn_limit, "remaining_output_limit": remaining_turns,
               "remaining_execution_seconds": remaining_seconds, "next_request_number": generator.sequence + 1,
               "scope": "Explicit recovery after verified process termination and restored disk space. Preserve prior errors and executed history; restore only the failed workspace. Local turn indices restart within the remaining output budget."}
    (recovery / "resume.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    store.record_event(actor_role="HARNESS", actor_id=args.workspace, event_type="experiment_recovery", payload=details)
    store.set_workspace_run_status(args.workspace, "pending")
    inspect = create_inspect_service(store=store, enabled_operations=frozenset(config["inspect_operations"]))
    runtime = InvestigationRuntime(store=store, gia_policy=TransformersGiaPolicy(generator),
        hi_policy=TransformersHiPolicy(generator), inspect_service=inspect, search_service=None,
        memory_policy=MemoryPolicy(max_read_chars=config["memory_policy"]["max_read_chars"]),
        max_rounds=3, max_gia_turns=8, gia_turn_timeout_s=660, max_workspace_turns=remaining_turns,
        workspace_timeout_s=remaining_seconds, max_concurrency=2, max_workspaces_per_round=3,
        report_review_delay_s=None, workspace_harness_type=harness1_workspace_type(Path(config["harness_root"])))
    try:
        async with runtime:
            result = await runtime.run(task=config["task"], original_material_ids=tuple(config["plan_origin"]["evidence_id_mapping"].values()))
            (run / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"run": args.run, "resumed_runtime_returned": True}), flush=True)
    finally:
        store.export_trajectory(run / "trajectory.jsonl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--workspace", required=True)
    asyncio.run(main(parser.parse_args()))
