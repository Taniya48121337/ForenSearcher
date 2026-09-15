"""Measure simultaneous, equal-time frame attachments in a supplied video pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measure(run, actor, parents):
    requests = []
    seen_times = set()
    for folder in sorted((run / "requests").glob("request_*")):
        request = read(folder / "request.json")
        if request["payload"].get("workspace_id") != actor:
            continue
        transport = read(folder / "transport.json")
        if transport["returncode"] != 0 or transport.get("external_tool_items"):
            raise ValueError(f"The actor request did not finish cleanly: {folder}")
        output = read(folder / "response.json")
        records = {}
        for note in request["image_notes"]:
            try:
                record = json.loads(note)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "evidence_id" in record:
                records[record["evidence_id"]] = record
        times = {}
        unpaired = []
        for eid in request["image_evidence_ids"]:
            record = records.get(eid)
            if record is None:
                raise ValueError(f"No actual attachment metadata for {eid}: {folder}")
            ancestors = record.get("derived_from", [])
            locator = record.get("locator")
            if isinstance(locator, str):
                try:
                    locator = json.loads(locator)
                except json.JSONDecodeError:
                    locator = None
            if (len(ancestors) != 1 or ancestors[0] not in parents
                    or not isinstance(locator, dict) or "time_s" not in locator):
                unpaired.append(eid)
                continue
            time = round(float(locator["time_s"]), 6)
            times.setdefault(time, {}).setdefault(ancestors[0], []).append(eid)
        pairs = [{"time_s": time, **frames} for time, frames in sorted(times.items())
                 if all(parent in frames for parent in parents)]
        seen_times.update(pair["time_s"] for pair in pairs)
        requests.append({"request": folder.name, "image_ids": request["image_evidence_ids"],
                         "co_visible_pairs": pairs, "not_direct_frames_of_selected_parents": unpaired,
                         "reported_status": output.get("status") if "findings" in output else None})
    if not requests or requests[-1]["reported_status"] is None:
        raise ValueError("Select a terminal HI actor with its actual report response")
    return {"run": run.name, "actor": actor, "parent_evidence_ids": list(parents),
            "requests": requests, "unique_co_visible_times_s": sorted(seen_times),
            "final_co_visible_pairs": requests[-1]["co_visible_pairs"],
            "scope": "Actual completed request attachments only. Equal timestamps do not establish alignment, semantic verification, memory, task completion, or an optimal evidence selection. Use only where equal-time pairing is appropriate; assess task-critical evidence separately."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--actor", default="W1-1")
    parser.add_argument("--parents", nargs=2, default=["E1", "E2"])
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    result = measure(run, args.actor, args.parents)
    folder = run / "reference_review"
    folder.mkdir(exist_ok=True)
    path = folder / (args.actor + "_visible_frame_pairs.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run": args.run, "actor": args.actor,
                      "unique_times": len(result["unique_co_visible_times_s"]),
                      "final_pairs": len(result["final_co_visible_pairs"])}))
