"""网站对已保存调查的只读投影，不参与 GIA 或 HI 决策。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .inspect.contracts import InspectValidationError, parse_inspect_call


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    staged.replace(path)


def _inspection_executions(events: list[dict[str, Any]], executions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """旧轨迹按工作区、实际请求及返回结果关联执行记录；新轨迹使用执行主键。"""
    records = {row["id"]: {**row, "request": json.loads(row["request_json"]),
                         "details": json.loads(row["details_json"] or "{}")} for row in executions}
    pending: dict[str, Any] = {}
    used: set[int] = set()
    linked: dict[int, dict[str, Any]] = {}
    for event in events:
        payload = json.loads(event["payload_json"])
        actor = event["actor_id"]
        if event["event_type"] == "model_output":
            output = payload.get("output") or {}
            pending[actor] = output.get("arguments") if output.get("action") == "INSPECT" else None
        elif event["event_type"] == "observation":
            observation = payload.get("observation") or {}
            if observation.get("action") != "INSPECT":
                continue
            details = observation.get("details") or {}
            execution = details.get("execution") or {}
            record = records.get(execution.get("id"))
            if record is None and pending.get(actor) is not None:
                try:
                    request = parse_inspect_call(pending[actor]).to_dict()
                except InspectValidationError:
                    continue
                record = next((r for r in records.values() if r["id"] not in used
                    and r["workspace_id"] == actor and r["request"] == request
                    and r["status"] == details.get("status")
                    and (f'E{r["result_evidence_id"]}' if r["result_evidence_id"] else None)
                        == details.get("result_evidence_id")), None)
            pending.pop(actor, None)
            if record is not None and record["workspace_id"] == actor:
                used.add(record["id"])
                info = record["details"]
                linked[event["id"]] = {"id": record["id"], "reused": bool(info.get("reused")),
                    "created_evidence_ids": info.get("created_evidence_ids", []),
                    "reused_evidence_ids": info.get("reused_evidence_ids", [])}
    return linked


def _result_evidence_ids(details: dict[str, Any]) -> list[str]:
    identifiers: list[str] = []
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if (key == "evidence_id" or key.endswith("_evidence_id")) and isinstance(child, str):
                    if child not in identifiers:
                        identifiers.append(child)
                elif isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    if details.get("result_evidence_id"):
        identifiers.append(details["result_evidence_id"])
    collect(details.get("result"))
    return identifiers


class InvestigationReader:
    """从同一 SQLite 快照读取轨迹、工作区及已提交证据。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.database = directory / "investigation.sqlite3"

    def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if not self.database.is_file():
            return []
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, params)]

    def event(self, event_id: int) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM trajectory_events WHERE id=?", (event_id,))
        if not rows:
            return None
        row = rows[0]
        row["payload"] = json.loads(row.pop("payload_json"))
        return row

    def investigation_task(self) -> str | None:
        rows = self._rows("SELECT task FROM investigation_meta WHERE id=1")
        return rows[0]["task"] if rows else None

    def evidence(self, evidence_id: str) -> dict[str, Any] | None:
        rows = self._rows("""SELECT e.*, a.media_type, a.byte_size, c.source_url, c.final_url
            FROM evidence e LEFT JOIN artifacts a ON e.artifact_id=a.id
            LEFT JOIN candidates c ON e.source_ref='R'||c.id WHERE e.id=?""", (int(evidence_id[1:]),))
        if not rows:
            return None
        row = rows[0]
        return {"evidence_id": evidence_id, "label": row["label"], "content": row["content"],
                "modalities": json.loads(row["modalities_json"]), "source_ref": row["source_ref"],
                "derived_from": json.loads(row["derived_from_json"]), "locator": row["locator"],
                "acquired_by": row["acquired_by"], "discovered_by": row["discovered_by"],
                "production_method": row["production_method"], "media_type": row["media_type"],
                "has_artifact": row["artifact_id"] is not None,
                "source_url": row["final_url"] or row["source_url"]}

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        if not self.database.is_file():
            return {"events": [], "cursor": 0, "workspaces": [], "evidence": [], "answer": "", "usage": None}
        # 一次只读事务避免展示已提交 E 时读到上一时刻的工作区状态。
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            events = [dict(row) for row in db.execute("SELECT * FROM trajectory_events WHERE id>? ORDER BY id", (after,))]
            cursor = db.execute("SELECT COALESCE(MAX(id),0) FROM trajectory_events").fetchone()[0]
            workspaces = [dict(row) for row in db.execute("SELECT task_id,round_index,task,materials_json,run_status,error_json FROM workspace_tasks ORDER BY round_index,task_id")]
            evidence = [dict(row) for row in db.execute("""SELECT e.id,e.label,e.modalities_json,e.source_ref,e.derived_from_json,
                e.discovered_by,e.acquired_by,e.artifact_id,a.media_type FROM evidence e
                LEFT JOIN artifacts a ON e.artifact_id=a.id ORDER BY e.id""")]
            inspection_events = [dict(row) for row in db.execute("SELECT id,actor_id,event_type,payload_json FROM trajectory_events WHERE event_type IN ('model_output','observation') ORDER BY id")]
            executions = [dict(row) for row in db.execute("SELECT * FROM inspect_executions ORDER BY id")]
        inspection_by_event = _inspection_executions(inspection_events, executions)
        compact = []
        for event in events:
            payload = json.loads(event.pop("payload_json"))
            typ = event["event_type"]
            value: dict[str, Any] = {**event, "round_index": payload.get("round_index"), "turn_index": payload.get("turn_index")}
            if typ == "raw_model_output":
                value.update(reasoning_preview=str(payload.get("reasoning") or "")[:180],
                             has_reasoning=bool(payload.get("reasoning")), model=payload.get("model"),
                             image_evidence_ids=payload.get("image_evidence_ids", []),
                             output_preview=str(payload.get("raw_output") or "")[:180])
            elif typ == "model_input":
                value["task"] = payload.get("model_view", {}).get("task", "")
            elif typ == "model_output":
                value["output"] = payload.get("output", {})
            elif typ == "observation":
                obs = payload.get("observation", {})
                details = obs.get("details", {})
                value["observation"] = {"action": obs.get("action"), "items": obs.get("items", []),
                    "status": details.get("status"), "error": details.get("error"),
                    "evidence": details.get("evidence", []), "operation": details.get("operation"),
                    "result_evidence_ids": _result_evidence_ids(details),
                    "match_count": len(details["result"]["matches"]) if isinstance(details.get("result"), dict) and isinstance(details["result"].get("matches"), list) else None,
                    "execution": inspection_by_event.get(event["id"]), "feedback": details.get("feedback", [])}
            elif typ in {"model_retry", "workspace_report", "workspace_outcome", "round_started", "round_finished", "investigation_finished", "investigation_failed"}:
                value["output"] = payload
            compact.append(value)
        for w in workspaces:
            w["materials"] = json.loads(w.pop("materials_json"))
            w["error"] = json.loads(w.pop("error_json") or "null")
        for e in evidence:
            e["evidence_id"] = f"E{e.pop('id')}"
            e["modalities"] = json.loads(e.pop("modalities_json"))
            e["derived_from"] = json.loads(e.pop("derived_from_json"))
            e["has_artifact"] = e.pop("artifact_id") is not None
        result = read_json(self.directory / "result.json", {})
        usage = read_json(self.directory / "billing-verification.json") or result.get("model_usage")
        return {"events": compact, "cursor": cursor, "workspaces": workspaces, "evidence": evidence,
                "answer": (result.get("decision") or {}).get("answer", ""),
                "reports": result.get("workspace_reports", []), "error": result.get("error"),
                "usage": ({k: usage.get(k) for k in ("request_count", "actual_cost_usd", "unsettled_reserved_usd")} if usage else None)}
