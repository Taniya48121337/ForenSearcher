"""Link actual CLI proposals to accepted decisions and executed observations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import LocalModelOutputError, extract_json_object


def link_events(events):
    records, pending, accepted = {}, {}, {}
    for event in events:
        actor = event["actor_id"]
        kind = event["event_type"]
        if kind == "raw_model_output":
            record = {"raw_event_id": event["id"], "actor": actor, "role": event["actor_role"],
                      "raw_output": extract_json_object(event["payload"]["raw_output"]),
                      "accepted_event_id": None, "observation_event_id": None,
                      "terminal_event_id": None, "status": "proposal_not_accepted"}
            records[event["id"]] = record
            pending[actor] = event["id"]
        elif kind == "model_output" and actor in pending:
            record = records[pending.pop(actor)]
            record["accepted_event_id"] = event["id"]
            record["accepted_output"] = event["payload"]["output"]
            record["raw_equals_accepted"] = record["raw_output"] == record["accepted_output"]
            record["status"] = "accepted_without_observation"
            accepted[actor] = record["raw_event_id"]
        elif kind == "observation" and actor in accepted:
            record = records[accepted.pop(actor)]
            record["observation_event_id"] = event["id"]
            record["status"] = "executed"
        elif kind == "workspace_report" and actor in accepted:
            record = records[accepted.pop(actor)]
            record["terminal_event_id"] = event["id"]
            record["status"] = "report_committed"
    for record in records.values():
        if record["role"] == "GIA" and record["accepted_event_id"]:
            record["status"] = "gia_decision_accepted"
    return list(records.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    directory = ROOT / "runs" / args.run
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    links = link_events(events)
    requests = sorted((directory / "requests").glob("request_*"))
    successful = []
    failed_requests = []
    for request in requests:
        transport_file = request / "transport.json"
        response_file = request / "response.json"
        transport = json.loads(transport_file.read_text(encoding="utf-8")) if transport_file.exists() else {}
        if transport.get("returncode") != 0 or transport.get("external_tool_items") or not response_file.exists():
            failed_requests.append({"request_id": request.name, "reason": "no_valid_actor_completion", "transport": transport})
        else:
            try:
                extract_json_object(response_file.read_text(encoding="utf-8"))
            except LocalModelOutputError:
                failed_requests.append({"request_id": request.name, "reason": "unparsed_actor_output", "transport": transport})
            else:
                successful.append(request)
    if len(successful) != len(links):
        raise RuntimeError("Successful CLI output count differs from recorded proposals; inspect unparsed outputs")
    for request, link in zip(successful, links):
        output = extract_json_object((request / "response.json").read_text(encoding="utf-8"))
        if output != link["raw_output"]:
            raise RuntimeError("CLI output and store event ordering do not match")
        link["request_id"] = request.name
        link["request_file"] = str(request / "request.json")
    (directory / "execution_links.json").write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"raw_proposals": len(links), "statuses": dict(Counter(r["status"] for r in links)),
               "cli_requests": len(requests),
               "normalized_outputs": [r["request_id"] for r in links if r.get("raw_equals_accepted") is False],
               "incomplete_requests": failed_requests,
               "scope": "execution linkage only; does not establish factual or decision quality"}
    (directory / "execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
