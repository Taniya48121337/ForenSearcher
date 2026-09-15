"""GIA 每轮看到的 Global View：五块紧凑输入的构造与预算裁剪。

从原 memory.py 拆出（2026-09-15 拆包）。卡片渲染等共享辅助函数在 store/records.py。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..contracts import WorkspaceReport
from ..store.global_store import GlobalStore
from ..store.records import (
    MemoryPolicy,
    _clip_text,
    _evidence_card,
    _evidence_page,
    _finding_card,
    _report_card,
    _report_detail,
    _task_context,
    _workspace_action_observations,
    _workspace_task_card,
    compact_reference,
    trim_visual_observation_summaries,
    workspace_visual_observations,
)

@dataclass(frozen=True)
class GlobalView:
    """每次调用 GIA 时提供的压缩全局视图。

    original 是用户最初材料，current 是最新一轮实际读取的 Evidence，latest_reports
    是最新轮结果，history_index 只提供更早历史的索引摘要。
    """

    task: str
    original_materials: tuple[dict[str, Any], ...]
    current_materials: tuple[dict[str, Any], ...]
    latest_reports: tuple[dict[str, Any], ...]
    history_index: tuple[dict[str, Any], ...]
    context_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """复制为 JSON 兼容结构，避免模型适配器修改内部 tuple。"""

        return {
            "task": self.task,
            "original_materials": list(self.original_materials),
            "current_materials": list(self.current_materials),
            "latest_reports": list(self.latest_reports),
            "history_index": list(self.history_index),
            **({"context_truncated": True} if self.context_truncated else {}),
        }


def bound_global_view(value: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """预算不足时将报告正文退回可检索索引，保留任务和每个工作区终态。"""

    value = json.loads(json.dumps(value, ensure_ascii=False))
    def oversized() -> bool:
        return len(json.dumps(value, ensure_ascii=False)) > max_chars
    for key in ("history_index", "current_materials"):
        while value.get(key) and oversized():
            value[key].pop(0)
            value["context_truncated"] = True
    reports = value.get("latest_reports", [])
    while value.get("image_inputs") and oversized():
        value["image_inputs"].pop()
        value["image_inputs_truncated"] = True
        value["context_truncated"] = True
    for item in reports:
        support = item.get("supporting_records", [])
        while support and oversized():
            support.pop()
            item["supporting_records_truncated"] = True
            value["context_truncated"] = True
        observations = item.get("action_observations", [])
        while observations and oversized():
            observations.pop(0)
            item["action_observations_truncated"] = True
            value["context_truncated"] = True
        images = item.get("image_inputs", [])
        while images and oversized():
            images.pop()
            item["image_inputs_truncated"] = True
            value["context_truncated"] = True
    trim_visual_observation_summaries(value, max_chars)
    for item in sorted(reports, key=lambda row: len(json.dumps(row)), reverse=True):
        if not oversized():
            break
        report = item.get("report")
        if isinstance(report, dict) and "findings" in report:
            report.pop("findings")
            report["details_truncated"] = True
            value["context_truncated"] = True
    for item in reports:
        if not oversized():
            break
        report = item.get("report")
        if isinstance(report, dict):
            item["report"] = {key: report[key] for key in ("task_id", "status") if key in report}
            item["report"]["details_truncated"] = True
            value["context_truncated"] = True
    for item in reports:
        if not oversized():
            break
        # 原任务可通过同一 W 编号完整读取，包括尚未提交报告的工作区。
        if "task" in item:
            item.pop("task")
            item.pop("materials", None)
            item["task_context_truncated"] = True
            value["context_truncated"] = True
    if oversized():
        raise ValueError("GIA task, original materials and Workspace status index exceed max_model_view_chars")
    return value


class GlobalViewBuilder:
    """从完整 Store 构造每一轮 GIA 所见的四层全局摘要。"""

    def __init__(self, store: GlobalStore, policy: MemoryPolicy) -> None:
        """保存 Store 与全局历史条数上限。"""

        self.store = store
        self.policy = policy

    def _unfinished_progress(self, task_id: str) -> dict[str, Any]:
        visual = self._visual_context(task_id)
        if not visual["image_inputs"]:
            visual = {}
        observations = _workspace_action_observations(self.store, task_id)
        if not observations:
            return visual
        retained = observations[-self.policy.max_history_entries:]
        value: dict[str, Any] = {"action_observations": retained}
        def oversized() -> bool:
            return len(json.dumps(value, ensure_ascii=False)) > self.policy.max_observation_chars
        while retained and oversized():
            retained.pop(0)
        if len(retained) < len(observations):
            value["action_observations_truncated"] = True
        ids = list(dict.fromkeys(eid for row in reversed(observations)
                                 for eid in row.get("evidence_ids", [])))
        evidence = [self.store.evidence(eid) for eid in ids]
        evidence.sort(key=lambda item: (item.production_method != "structured_analysis",
                                       "text" not in item.modalities))
        records: list[dict[str, Any]] = []
        value["supporting_records"] = records
        for item in evidence[:self.policy.max_evidence_cards]:
            records.append(_evidence_page(item, self.policy.max_read_chars))
            if oversized():
                records.pop()
                break
        if len(records) < len(ids):
            value["supporting_records_truncated"] = True
        return {**value, **visual}

    def _visual_context(self, task_id: str) -> dict[str, Any]:
        images, summary = workspace_visual_observations(self.store, task_id)
        return {"image_inputs": images[-self.policy.max_evidence_cards:],
                **({"image_inputs_truncated": True} if len(images) > self.policy.max_evidence_cards else {}),
                "visual_observation_summary": summary}

    def _report_support(self, report: WorkspaceReport) -> dict[str, Any]:
        identifiers = list(dict.fromkeys(
            [eid for finding in report.findings for eid in finding.evidence_ids]
            + list(report.selected_evidence_ids)))
        evidence = [self.store.evidence(eid) for eid in identifiers]
        # 工具原始结果优先于原始媒体的文件描述，保留可分页读取的正文。
        evidence.sort(key=lambda item: (item.production_method != "structured_analysis",
                                       "text" not in item.modalities))
        records = []
        for item in evidence[:self.policy.max_evidence_cards]:
            page = _evidence_page(item, self.policy.max_read_chars)
            if item.source_ref and item.source_ref.startswith("R"):
                source = self.store.resolve(item.source_ref)
                page["source"] = {key: source.get(key) for key in
                                  ("candidate_id", "source_url", "final_url")}
            records.append(page)
        return {
            "supporting_records": records,
            **({"supporting_records_truncated": True} if len(evidence) > len(records) else {}),
            **self._visual_context(report.task_id),
        }

    def build(
        self,
        *,
        task: str,
        original_material_ids: Iterable[str],
        latest_round_index: int | None,
    ) -> GlobalView:
        """汇总原始材料、最新读取证据、最新报告和更早历史索引。"""

        # tuple 固化调用方给出的顺序，set 用于后续快速排除重复编号。
        original_values = tuple(original_material_ids)
        original_ids = set(original_values)
        originals = tuple(compact_reference(self.store, value) for value in original_values)
        current: list[dict[str, Any]] = []
        current_ids: set[str] = set()
        latest: list[dict[str, Any]] = []
        latest_task_ids: set[str] = set()
        if latest_round_index is not None:
            # 收集最新轮中 HI 真正读取过、且不是原始输入的 E。
            seen: set[str] = set()
            for read in self.store.record_reads(
                actor_role="HI", round_index=latest_round_index
            ):
                reference_id = read["reference_id"]
                if (
                    reference_id.startswith("E")
                    and reference_id not in original_ids
                    and reference_id not in seen
                ):
                    seen.add(reference_id)
                    current_ids.add(reference_id)
                    current.append(compact_reference(self.store, reference_id))
            for record in self.store.workspace_task_records(latest_round_index):
                # 原任务与局部完成状态并列，避免把局部完成误当作总任务已解决。
                task_id = record["task_id"]
                latest_task_ids.add(task_id)
                if record["run_status"] == "reported":
                    latest.append(
                        {
                            "task": record["task"],
                            "materials": record["materials"],
                            "run_status": "reported",
                            "report": _report_detail(
                                self.store.workspace_report(task_id)
                            ),
                            **self._report_support(self.store.workspace_report(task_id)),
                        }
                    )
                else:
                    latest.append(
                        {
                            "task_id": task_id,
                            "task": record["task"],
                            "materials": record["materials"],
                            "run_status": record["run_status"],
                            **({"error": record["error"]} if record["error"] else {}),
                            **(self._unfinished_progress(task_id) if record["run_status"] in
                               {"runtime_error", "timed_out", "cancelled"} else {}),
                        }
                    )
        # 原件属性独立于 HI 是否成功提交报告；跨轮仍可按索引读取时长、页数等测量值。
        for reference_id in original_values:
            if not reference_id.startswith("E"):
                continue
            for profile in self.store.find_derived_evidence(
                derived_from=(reference_id,), acquired_by="INSPECT/media_profile"
            ):
                if profile.evidence_id not in current_ids:
                    current_ids.add(profile.evidence_id)
                    current.append(_evidence_card(profile))
        history: list[dict[str, Any]] = []
        # 历史 Evidence 排除原始输入、当前轮已读内容和当前轮刚生成的内容。
        for evidence in self.store.evidence_index_records():
            if evidence["evidence_id"] in current_ids or evidence["evidence_id"] in original_ids:
                continue
            if latest_round_index is not None and evidence["discovered_by"].startswith(
                f"W{latest_round_index}-"
            ):
                continue
            history.append(
                {
                    "record_type": "evidence",
                    "evidence_id": evidence["evidence_id"],
                    "label": _clip_text(evidence["label"], 1000),
                    "source_workspace": _clip_text(evidence["discovered_by"], 500),
                }
            )
        for record in self.store.workspace_task_records():
            if record["task_id"] not in latest_task_ids and record["run_status"] != "reported":
                history.append(_workspace_task_card(record))
        for report in self.store.workspace_report_records():
            # 最新轮报告已在 latest_reports，旧报告和 Finding 才进入历史索引。
            if report.task_id in latest_task_ids:
                continue
            history.append({**_report_card(report), **_task_context(self.store, report.task_id, compact=True)})
            history.extend(_finding_card(finding) for finding in report.findings)
        history_truncated = len(history) > self.policy.max_history_entries
        if history_truncated:
            # 只保留最近的固定条数，防止调查越久 GIA 上下文无限增长。
            history = history[-self.policy.max_history_entries :]
        view = GlobalView(
            task=task,
            original_materials=originals,
            current_materials=tuple(current),
            latest_reports=tuple(latest),
            history_index=tuple(history),
            context_truncated=history_truncated,
        )
        bounded = bound_global_view(view.to_dict(), self.policy.max_model_view_chars)
        return GlobalView(task=bounded["task"], original_materials=tuple(bounded["original_materials"]),
            current_materials=tuple(bounded["current_materials"]),
            latest_reports=tuple(bounded["latest_reports"]), history_index=tuple(bounded["history_index"]),
            context_truncated=bounded.get("context_truncated", False))

