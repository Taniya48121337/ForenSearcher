"""Review a terminal HI workspace while the rest of its investigation continues."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from assessment_packet import ROOT, build_packet
from audit_run import link_events
from judge_run import assess_packet
from foren_searcher.local_models import extract_json_object


def completed_packet(run, actor):
    with sqlite3.connect((run / "store.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        events = []
        for row in connection.execute("SELECT * FROM trajectory_events ORDER BY id"):
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)
    outcomes = [e for e in events if e["actor_id"] == actor and e["event_type"] == "workspace_outcome"]
    reported = [e for e in outcomes if e["payload"]["run_status"] == "reported"]
    if actor == "GIA" or len(reported) != 1 or outcomes[-1] != reported[0]:
        raise ValueError("The selected HI must have one committed reported outcome")
    terminal = reported[0]
    if any(e["actor_id"] == actor and e["id"] > terminal["id"] for e in events):
        raise ValueError("Selected actor has activity after its terminal outcome")
    links = [link for link in link_events(events) if link["actor"] == actor]
    requests = []
    for directory in sorted((run / "requests").glob("request_*")):
        request_path = directory / "request.json"
        if not request_path.exists():
            continue
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request["payload"].get("workspace_id") != actor:
            continue
        transport = json.loads((directory / "transport.json").read_text(encoding="utf-8"))
        if transport.get("returncode") != 0 or transport.get("external_tool_items"):
            raise ValueError("This snapshot helper requires clean completed actor requests")
        requests.append(directory)
    if len(requests) != len(links):
        raise ValueError("Actor request count differs from recorded raw decisions")
    for directory, link in zip(requests, links):
        output = extract_json_object((directory / "response.json").read_text(encoding="utf-8"))
        if output != link["raw_output"]:
            raise ValueError("Actor requests and events have different order or output")
        link.update(request_id=directory.name, request_file=str(directory / "request.json"))
    packet, images = build_packet(run, max_images=None, actor=actor,
                                  event_snapshot=events, execution_links=links)
    packet["snapshot"] = {"scope": "complete_terminal_actor_only", "last_event_id": events[-1]["id"],
                          "actor_terminal_event_id": terminal["id"],
                          "other_workspaces_may_still_be_running": True}
    packet["all_workspaces_reported"] = None
    packet["execution_summary"] = {"raw_proposals": len(links), "cli_requests": len(requests),
                                    "incomplete_requests": [], "scope": "selected_terminal_actor_only"}
    return packet, images


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("actor")
    parser.add_argument("--assessment", default="assessment_r5_by_actor")
    parser.add_argument("--rubric", default="rubric_r5.md")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    packet, images = completed_packet(run, args.actor)
    assess_packet(packet, images, run / args.assessment / args.actor, args.rubric)
