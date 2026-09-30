"""run 结束后把教师的每次原始回答对到"被接受的决定 / 执行结果"，写 execution_links.json（runlog 读它）。

逻辑与 experiments/sft_data/audit_run.py 相同（那边是试点时期的单 run 脚本，保留不动）；这里做成函数供批量驱动调用，
并认 EndpointTeacher 的 transport.json（returncode=0 为成功，与 Codex CLI 相同）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..local_models import LocalModelOutputError, extract_json_object, strip_rationale


class LinkError(RuntimeError):
    """原始回答数与轨迹里记录的提议数对不上，或顺序不一致：这个 run 不能进 runlog。"""


def link_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[Any, dict[str, Any]] = {}
    pending: dict[str, Any] = {}
    accepted: dict[str, Any] = {}
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
            # rationale 是可选的一句话理由，运行时校验前剥掉；比较时也剥掉，这个标记只反映 harness 有没有改动作本身
            record["raw_equals_accepted"] = strip_rationale(record["raw_output"]) == record["accepted_output"]
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


def write_execution_links(run_dir: str | Path) -> dict[str, Any]:
    """读 trajectory.jsonl 与 requests/，写 execution_links.json 与 execution_summary.json，返回 summary。"""

    directory = Path(run_dir)
    events = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    links = link_events(events)
    requests = sorted((directory / "requests").glob("request_*"))
    successful: list[Path] = []
    failed: list[dict[str, Any]] = []
    for request in requests:
        transport_file, response_file = request / "transport.json", request / "response.json"
        transport = json.loads(transport_file.read_text(encoding="utf-8")) if transport_file.exists() else {}
        if transport.get("returncode") != 0 or transport.get("external_tool_items") or not response_file.exists():
            failed.append({"request_id": request.name, "reason": "no_valid_actor_completion", "transport": transport})
            continue
        try:
            extract_json_object(response_file.read_text(encoding="utf-8"))
        except LocalModelOutputError:
            failed.append({"request_id": request.name, "reason": "unparsed_actor_output", "transport": transport})
        else:
            successful.append(request)
    if len(successful) != len(links):
        raise LinkError(f"{directory.name}: 成功的教师回答 {len(successful)} 个，轨迹里的提议 {len(links)} 个，对不上")
    for request, link in zip(successful, links):
        if extract_json_object((request / "response.json").read_text(encoding="utf-8")) != link["raw_output"]:
            raise LinkError(f"{directory.name}: {request.name} 的回答与轨迹事件顺序不一致")
        link["request_id"] = request.name
        link["request_file"] = str(request / "request.json")
    (directory / "execution_links.json").write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"raw_proposals": len(links), "statuses": dict(Counter(r["status"] for r in links)),
               "cli_requests": len(requests),
               "normalized_outputs": [r["request_id"] for r in links if r.get("raw_equals_accepted") is False],
               "incomplete_requests": failed,
               "scope": "execution linkage only; does not establish factual or decision quality"}
    (directory / "execution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["LinkError", "link_events", "write_execution_links"]
