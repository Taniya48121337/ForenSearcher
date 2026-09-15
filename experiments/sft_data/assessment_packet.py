"""Provide exact actor-visible state to assessors without repeating shared content."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pack_repeated(value):
    counts = Counter()

    def key(item):
        if not isinstance(item, (dict, list, str)):
            return None
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return encoded if len(encoded) >= 160 else None

    def count(item):
        encoded = key(item)
        if encoded:
            counts[encoded] += 1
        children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else []
        for child in children:
            count(child)

    count(value)
    identifiers, objects = {}, {}

    def encode(item, allow_reference=True):
        encoded = key(item)
        if allow_reference and encoded and counts[encoded] > 1:
            if encoded not in identifiers:
                identifier = "o" + str(len(identifiers) + 1)
                identifiers[encoded] = identifier
                objects[identifier] = encode(item, False)
            return {"$object": identifiers[encoded]}
        if isinstance(item, dict):
            return {k: encode(v) for k, v in item.items()}
        if isinstance(item, list):
            return [encode(v) for v in item]
        return item

    root = encode(value)
    return {"root": root, "objects": objects}


def unpack_repeated(packet):
    def decode(item):
        if isinstance(item, dict):
            if set(item) == {"$object"}:
                return decode(packet["objects"][item["$object"]])
            return {k: decode(v) for k, v in item.items()}
        if isinstance(item, list):
            return [decode(v) for v in item]
        return item
    return decode(packet["root"])


def workspace_outcome_summary(events):
    history = [{"actor": event["actor_id"], **event["payload"]} for event in events
               if event["event_type"] == "workspace_outcome"]
    latest = {item["actor"]: item for item in history}
    scheduled = {event["payload"]["round_index"]: event["payload"]["workspace_count"]
                 for event in events if event["event_type"] == "round_started"}
    complete = (bool(scheduled) and len(latest) == sum(scheduled.values())
                and all(item.get("run_status") == "reported" for item in latest.values()))
    return history, list(latest.values()), complete


def build_packet(run, *, dataset=None, case=None, max_images=24, actor=None,
                 event_snapshot=None, execution_links=None):
    from foren_searcher.local_models import extract_json_object

    run = Path(run)
    config = read_json(run / "run_config.json") if (run / "run_config.json").exists() else {}
    dataset = dataset or config.get("dataset", "videosham")
    case = config.get("case", case)
    if dataset == "videosham":
        reference_path = ROOT / "data/cases" / f"videosham_{case}" / "reference.json"
    else:
        reference_path = ROOT / "data" / dataset / str(case) / "reference.json"
    links = execution_links if execution_links is not None else read_json(run / "execution_links.json")
    links = {link["request_id"]: link for link in links}
    steps, images, report_images = [], {}, []
    for directory in sorted((run / "requests").glob("request_*")):
        if directory.name not in links or not (directory / "response.json").is_file():
            continue
        request = read_json(directory / "request.json")
        if actor is not None and request["payload"].get("workspace_id", "GIA") != actor:
            continue
        output = extract_json_object((directory / "response.json").read_text(encoding="utf-8"))
        transport = read_json(directory / "transport.json")
        steps.append({"request": directory.name, "actor": request["payload"].get("workspace_id", "GIA"),
                      "actor_input": request, "output": output,
                      "execution_link": links.get(directory.name), "transport": transport})
        for i, eid in enumerate(request["image_evidence_ids"]):
            images.setdefault(eid, directory / "actor" / f"image_{i + 1:02d}.png")
        if "findings" in output or output.get("decision") == "FINISH":
            report_images.extend(request["image_evidence_ids"])
            for finding in output.get("findings", []):
                report_images.extend(finding.get("evidence_ids", []))
    priority = list(dict.fromkeys(report_images + list(images)))
    selected = [(eid, images[eid]) for eid in priority if eid in images]
    if max_images is not None:
        selected = selected[:max_images]
    selected_ids = [eid for eid, _ in selected]
    result_path = run / "result.json"
    trajectory_path = run / "trajectory.jsonl"
    events = event_snapshot if event_snapshot is not None else (
        [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()] if trajectory_path.exists() else [])
    outcomes, latest_outcomes, all_reported = workspace_outcome_summary(events)
    packet = {"dataset": dataset, "case": case, "run_config": config,
              "reference": read_json(reference_path), "steps": steps,
              "result": read_json(result_path) if result_path.exists() else None,
              "runtime_returned_result": result_path.exists(), "workspace_outcomes": outcomes,
              "latest_workspace_outcomes": latest_outcomes,
              "all_workspaces_reported": all_reported,
              "experiment_recoveries": [event["payload"] for event in events
                                         if event["event_type"] == "experiment_recovery"],
              "attached_images_in_order": selected_ids,
              "unattached_image_ids": [eid for eid in images if eid not in selected_ids]}
    summary_path = run / "execution_summary.json"
    if summary_path.exists():
        packet["execution_summary"] = read_json(summary_path)
        if actor is not None:
            incomplete = []
            for item in packet["execution_summary"].get("incomplete_requests", []):
                path = run / "requests" / item["request_id"] / "request.json"
                if path.exists() and read_json(path)["payload"].get("workspace_id", "GIA") == actor:
                    incomplete.append(item)
            packet["execution_summary"]["incomplete_requests"] = incomplete
    if actor is not None:
        packet["evaluation_actor"] = actor
        packet["evaluation_scope"] = "only_this_actor_outputs; complete_actual_role_inputs_and_available_actor_images"
    return packet, selected
