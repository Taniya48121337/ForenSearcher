"""按角色控制历史记录访问，并为模型构造有大小上限的调查视图。

GlobalStore 保存完整记录，但不能把整个数据库都塞进模型上下文。本模块负责三件事：
1. 把完整记录压缩成 Evidence、Candidate、Finding、Report 和工作区任务卡片；
2. 让 GIA 能看全局历史，而 HI 只能看公共 Evidence 和自己的局部记录；
3. 按任务相关性、显式材料、关注集合和来源 Workspace 给卡片排序并截断。

这里的“Memory”不是另一个会自主决策的模型，而是 Harness 控制的只读视图。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Literal

from .contracts import Evidence, Finding, WorkspaceReport, WorkspaceTask
from .store import GlobalStore

# 两个 Literal 把字符串取值限制在框架真正支持的角色和记录类型中。
ActorRole = Literal["GIA", "HI"]
RecordType = Literal["candidate", "evidence", "finding", "report", "workspace_task"]


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """所有模型记忆和报告大小上限的集中配置。

    这些上限既控制 token/内存成本，也阻止模型一次返回过多材料造成拒绝服务。
    """

    max_evidence_cards: int = 12
    max_candidate_cards: int = 8
    max_history_entries: int = 32
    query_page_size: int = 10
    max_read_chars: int = 6000
    max_observation_chars: int = 16000
    max_model_view_chars: int = 180000
    max_focus_evidence_ids: int = 128
    max_findings_per_report: int = 64
    max_report_chars: int = 64000
    max_query_chars: int = 4000

    def __post_init__(self) -> None:
        """构造后逐项确认所有上限都是正整数。"""

        # 用字段名循环可以保证新增上限时复用相同的严格校验。
        for name in (
            "max_evidence_cards",
            "max_candidate_cards",
            "max_history_entries",
            "query_page_size",
            "max_read_chars",
            "max_observation_chars",
            "max_model_view_chars",
            "max_focus_evidence_ids",
            "max_findings_per_report",
            "max_report_chars",
            "max_query_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RecordPage:
    """QUERY_MEMORY 的一页结果；next_cursor 为 None 表示没有下一页。"""

    results: tuple[dict[str, Any], ...]
    next_cursor: int | None

    def to_dict(self) -> dict[str, Any]:
        """转换成模型可接收的普通 JSON 对象。"""

        return {"results": list(self.results), "next_cursor": self.next_cursor}


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


def trim_visual_observation_summaries(
    value: dict[str, Any], max_chars: int, *, details_only: bool = False,
) -> None:
    """统计超出上下文预算时先省略覆盖明细，保留计数并标明缺失范围。"""

    def oversized() -> bool:
        return len(json.dumps(value, ensure_ascii=False)) > max_chars

    containers = [value, *value.get("latest_reports", [])]
    for container in containers:
        summary = container.get("visual_observation_summary")
        if not isinstance(summary, dict) or not oversized():
            continue
        for video in summary.get("videos", []):
            if not oversized():
                break
            fields = ("observed_ranges_s", "unconfirmed_ranges_s", "unknown_duration_frame_times_s")
            if any(key in video for key in fields):
                value["context_truncated"] = True
                for key in fields:
                    video.pop(key, None)
                video["coverage_truncated"] = True
    if details_only:
        return
    for container in containers:
        if "visual_observation_summary" in container and oversized():
            value["context_truncated"] = True
            container.pop("visual_observation_summary")
            container["visual_observation_summary_truncated"] = True


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


_UNSEGMENTED_CHARACTERS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0002fa1f"
_UNSEGMENTED_TEXT = re.compile(f"[{_UNSEGMENTED_CHARACTERS}]+")
_WORD_CHARACTER = rf"[^\W{_UNSEGMENTED_CHARACTERS}]"


def _tokens(value: str) -> set[str]:
    """生成用于轻量相关性排序的词集合和连续双字符集合。"""

    # casefold 比 lower 更适合多语言大小写归一化。
    lowered = value.casefold()
    # 分隔无空格文字，保留混合文本中的英文词与数字。
    words = set(re.findall(r"\w+", _UNSEGMENTED_TEXT.sub(" ", lowered)))
    for run in _UNSEGMENTED_TEXT.findall(lowered):
        words.add(run)
        words.update(run[index:index + 2] for index in range(len(run) - 1))
    return {item for item in words if item}


def _relevance(query: str, text: str, *other_texts: str) -> float:
    """优先连续短语，其余按查询词覆盖率排序；分数不是事实置信度。"""

    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    phrase = " ".join(query.casefold().split()).strip('"\'“”‘’')
    # 英文词不能命中较长单词的内部；汉字短语可处于连续中文句子中。
    left = rf"(?<!{_WORD_CHARACTER})" if phrase and re.match(_WORD_CHARACTER, phrase[0]) else ""
    right = rf"(?!{_WORD_CHARACTER})" if phrase and re.match(_WORD_CHARACTER, phrase[-1]) else ""
    text_tokens: set[str] = set()
    for section in (text, *other_texts):
        normalized = " ".join(section.casefold().split())
        if phrase and re.search(left + re.escape(phrase) + right, normalized):
            return 1.0
        text_tokens.update(_tokens(section))
    return 0.5 * len(query_tokens & text_tokens) / len(query_tokens)


def _evidence_index(evidence_id: str) -> int:
    """取出 E 后面的数字用于稳定排序；异常格式按 0 处理。"""

    return int(evidence_id[1:]) if evidence_id[1:].isdigit() else 0


def _clip_text(value: Any, limit: int) -> Any:
    """只截断超长字符串；非字符串原样返回。"""

    if isinstance(value, str) and len(value) > limit:
        return value[:limit - 1] + "…"
    return value


def _task_context(store: GlobalStore, task_id: str, *, compact: bool = False) -> dict[str, Any]:
    """报告的完成状态仅针对其分配任务；将该任务与报告一起提供给读取者。"""

    try:
        record = store.workspace_task_record(task_id)
    except KeyError:
        return {}
    task, materials = record["task"], record["materials"]
    if not compact:
        return {"task": task, "materials": materials}
    return {"task": _clip_text(task, 2000), "materials": materials[:128],
            **({"task_context_truncated": True} if len(task) > 2000 or len(materials) > 128 else {})}


def _workspace_task_card(record: dict[str, Any]) -> dict[str, Any]:
    """未提交报告的任务索引；完整任务和运行错误通过 W 编号读取。"""

    task, materials = record["task"], record["materials"]
    card = {"record_type": "workspace_task", "task_id": record["task_id"],
            "round_index": record["round_index"], "run_status": record["run_status"],
            "task": _clip_text(task, 2000), "materials": materials[:128]}
    if len(task) > 2000 or len(materials) > 128:
        card["task_context_truncated"] = True
    if record["error"]:
        error = json.dumps(record["error"], ensure_ascii=False)
        card["error_excerpt"] = _clip_text(error, 2000)
        if len(error) > 2000:
            card["error_truncated"] = True
    return card


def _evidence_card(value: Evidence) -> dict[str, Any]:
    """从完整 Evidence 构造有字段上限的模型卡片。"""

    card = value.card().to_dict()
    # 派生链最多展示 128 项，防止很长的处理链撑爆上下文。
    derived_from = list(card["derived_from"][:128])
    card["label"] = _clip_text(card["label"], 512)
    card["source_ref"] = _clip_text(card["source_ref"], 512)
    card["acquired_by"] = _clip_text(card["acquired_by"], 256)
    card["derived_from"] = derived_from
    if len(value.derived_from) > len(derived_from):
        card["derived_from_truncated"] = True
    return {"record_type": "evidence", **card}


def _workspace_action_observations(store: GlobalStore, task_id: str) -> list[dict[str, Any]]:
    """仅整理收到响应的动作，不包含推理、未执行请求或其他工作区记录。"""

    events = store.trajectory_events(actor_role="HI", actor_id=task_id, event_type="model_output")
    events.extend(store.trajectory_events(actor_role="HI", actor_id=task_id, event_type="observation"))
    outputs: dict[Any, Any] = {}
    rows = []
    for event in sorted(events, key=lambda row: row["id"]):
        payload = event["payload"]
        if event["event_type"] == "model_output":
            outputs[payload.get("turn_index")] = payload.get("output", {})
            continue
        output = outputs.pop(payload.get("turn_index"), {})
        observation = payload.get("observation", {})
        action = observation.get("action")
        if not isinstance(action, str):
            continue
        row: dict[str, Any] = {"turn_index": payload.get("turn_index"), "action": action}
        details = observation.get("details", {})
        if action == "HARNESS_REJECTION":
            row["error"] = observation.get("error") or details.get("error", "")
        else:
            if isinstance(output, dict) and output.get("action") == action:
                row["arguments"] = output.get("arguments", {})
            if isinstance(details, dict):
                if action == "INSPECT":
                    row.update(operation=details.get("operation"), status=details.get("status"),
                               feedback=details.get("feedback", []))
                    # 取帧等操作可只返回媒体 E，没有独立的结果正文 E。
                    identifiers = ([details["result_evidence_id"]] if details.get("result_evidence_id") else [])
                    identifiers.extend(card["evidence_id"] for card in details.get("evidence", [])
                                       if isinstance(card, dict) and isinstance(card.get("evidence_id"), str))
                    if identifiers:
                        row["evidence_ids"] = list(dict.fromkeys(identifiers))
                elif action == "READ_RECORD":
                    row.update({key: details[key] for key in
                                ("record_id", "evidence_id", "content_offset", "next_offset") if key in details})
                    if details.get("evidence_id"):
                        row["evidence_ids"] = [details["evidence_id"]]
                elif action == "SEARCH":
                    row.update(stop_reason=details.get("stop_reason"), error=details.get("error"))
                    sources = []
                    ids = []
                    for source in details.get("results", []):
                        materials = []
                        for material in source.get("materials", []):
                            evidence_ids = [card["evidence_id"] for card in material.get("evidence", [])]
                            ids.extend(evidence_ids)
                            materials.append({key: material[key] for key in ("kind", "status", "error") if key in material})
                        sources.append({"candidate_id": source.get("candidate_id"),
                                        "source_url": source.get("source_url"), "materials": materials})
                    row.update(results=sources, evidence_ids=list(dict.fromkeys(ids)))
                elif action == "QUERY_MEMORY":
                    row.update(result_count=len(details.get("results", [])), next_cursor=details.get("next_cursor"))
        rows.append(row)
    return rows


def _evidence_index_card(value: dict[str, Any]) -> dict[str, Any]:
    """把 Store 返回的 Evidence 索引行转换成与模型约定一致的卡片。"""

    derived_from = list(value["derived_from"][:128])
    return {
        "record_type": "evidence",
        "evidence_id": value["evidence_id"],
        "modalities": list(value["modalities"]),
        "label": _clip_text(value["label"], 512),
        "source_ref": _clip_text(value["source_ref"], 512),
        "derived_from": derived_from,
        **(
            {"derived_from_truncated": True}
            if len(value["derived_from"]) > len(derived_from)
            else {}
        ),
        "acquired_by": _clip_text(value["acquired_by"], 256),
    }


def _candidate_card(value: dict[str, Any]) -> dict[str, Any]:
    """压缩一个候选来源 R 记录，只保留检索和选择所需字段。"""

    return {
        "record_type": "candidate",
        "candidate_id": value["candidate_id"],
        "source_url": _clip_text(value["source_url"], 4000),
        "title": _clip_text(value.get("search_title"), 1000),
        "snippet": _clip_text(value.get("search_snippet"), 2000),
        "candidate_type": _clip_text(value.get("candidate_type"), 200),
    }


def _finding_card(value: Finding) -> dict[str, Any]:
    """压缩 Finding，同时显式标注引用或 limitations 是否被截断。"""

    evidence_ids = list(value.evidence_ids[:128])
    return {
        "record_type": "finding",
        "finding_id": value.finding_id,
        "workspace_id": value.workspace_id,
        "statement": _clip_text(value.statement, 4000),
        "evidence_ids": evidence_ids,
        **(
            {"evidence_ids_truncated": True}
            if len(value.evidence_ids) > len(evidence_ids)
            else {}
        ),
        "limitations": [_clip_text(item, 1000) for item in value.limitations[:16]],
        **({"limitations_truncated": True}
           if len(value.limitations) > 16 or any(len(item) > 1000 for item in value.limitations[:16]) else {}),
    }


def _report_card(value: WorkspaceReport) -> dict[str, Any]:
    """构造报告索引卡：保留 F/E 编号，不在索引中展开全部 Finding。"""

    finding_ids = [item.finding_id for item in value.findings[:64]]
    selected_evidence_ids = list(value.selected_evidence_ids[:128])
    return {
        "record_type": "report",
        "task_id": value.task_id,
        "status": value.status,
        "finding_ids": finding_ids,
        "selected_evidence_ids": selected_evidence_ids,
        **({"findings_truncated": True} if len(value.findings) > len(finding_ids) else {}),
        **(
            {"selected_evidence_ids_truncated": True}
            if len(value.selected_evidence_ids) > len(selected_evidence_ids)
            else {}
        ),
        **(
            {"status_reason": _clip_text(value.status_reason, 2000)}
            if value.status_reason
            else {}
        ),
    }


def _report_detail(value: WorkspaceReport) -> dict[str, Any]:
    """在报告索引卡基础上展开最多 64 条 Finding，供 READ_RECORD 使用。"""

    result = _report_card(value)
    result["findings"] = [_finding_card(item) for item in value.findings[:64]]
    return result


def _evidence_page(value: Evidence, max_chars: int, offset: int = 0) -> dict[str, Any]:
    """正文页与媒体描述使用相同的分页和内容范围标记。"""

    end = min(len(value.content), offset + max_chars)
    payload = {"record_type": "evidence", **value.to_dict(),
               "content": value.content[offset:end], "content_offset": offset,
               "next_offset": end if end < len(value.content) else None}
    if value.artifact_ref is not None and (
        value.acquired_by == "HARNESS/ingest"
        or ("text" not in value.modalities and set(value.modalities) & {"image", "audio", "video"})
    ):
        payload["content_notice"] = (
            "本记录正文是图片元数据，不是像素。direct_visual_input=true 时，"
            "READ_RECORD 本 E 可请求这张图片供后续决策查看，实际附图以 image_inputs 为准。"
            "不支持直接视觉输入时，使用可用的视觉 INSPECT 操作核查图片。"
        ) if "image" in value.modalities else (
            "这里只包含媒体记录的文字或元数据，不包含文件正文、像素或声音内容。"
            "重复读取记录不会取得媒体内容。图片须实际提供给多模态输入或适用的 INSPECT 操作；"
            "PDF 先获取页面图片，视频先提取帧；文字和语音可分别使用 OCR 与 ASR。"
        )
    return payload


def compact_reference(store: GlobalStore, reference_id: str) -> dict[str, Any]:
    """读取任意已知编号，并按它的真实记录类型选择压缩方法。"""

    value = store.resolve(reference_id)
    # isinstance 分支保证不同记录不会因字段名相似而被错误解释。
    if isinstance(value, Evidence):
        return _evidence_card(value)
    if isinstance(value, Finding):
        return _finding_card(value)
    if isinstance(value, WorkspaceReport):
        return {**_report_card(value), **_task_context(store, value.task_id, compact=True)}
    if isinstance(value, WorkspaceTask):
        return _workspace_task_card(store.workspace_task_record(value.task_id))
    if isinstance(value, dict) and "candidate_id" in value:
        return _candidate_card(value)
    # C 等普通材料没有专用 dataclass，序列化后只展示前 512 字符。
    rendered = json.dumps(value, ensure_ascii=False, default=str)
    return {
        "record_type": "material",
        "reference_id": reference_id,
        "content": rendered[:512],
        "truncated": len(rendered) > 512,
    }


class RecordAccess:
    """Harness 拥有的 QUERY_MEMORY 与 READ_RECORD 实现。

    模型只提交查询参数；实际权限判断、分页、截断和读取审计都在这里执行。
    """

    def __init__(
        self,
        *,
        store: GlobalStore,
        role: ActorRole,
        actor_id: str,
        round_index: int | None,
        policy: MemoryPolicy,
        local_reference_ids: set[str] | None = None,
    ) -> None:
        """绑定 Store、调用角色、调用者编号、轮次和内存策略。"""

        self.store = store
        self.role = role
        self.actor_id = actor_id
        self.round_index = round_index
        self.policy = policy
        self.local_reference_ids = local_reference_ids

    def _allowed(self, record_id: str, value: Any) -> bool:
        """判断当前角色是否允许读取已经解析出的记录。"""

        # GIA 是全局协调者，可以读取所有记录。
        if self.role == "GIA":
            return True
        # Evidence 是 Workspace 间唯一共享的完整调查产物。
        if isinstance(value, Evidence):
            return True
        # 候选来源和普通材料必须已进入当前 Workspace 的局部引用集合。
        if isinstance(value, dict):
            return (
                self.local_reference_ids is not None
                and record_id in self.local_reference_ids
            )
        # HI 只能看到自己产生的 Finding 和自己的正式报告。
        if isinstance(value, Finding):
            return value.workspace_id == self.actor_id
        if isinstance(value, (WorkspaceReport, WorkspaceTask)):
            return value.task_id == self.actor_id
        return True

    def _structured_page(
        self, payload: dict[str, Any], identity: dict[str, Any], offset: int,
    ) -> dict[str, Any]:
        rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if offset == 0 and len(rendered) <= self.policy.max_read_chars:
            return {**payload, **identity, "content_offset": 0, "next_offset": None}
        end = min(len(rendered), offset + self.policy.max_read_chars)
        return {**identity, "content": rendered[offset:end], "content_offset": offset,
                "next_offset": end if end < len(rendered) else None}

    def read(self, record_id: str, *, offset: int = 0) -> dict[str, Any]:
        """按编号读取一条记录，并对长正文执行基于字符位置的分页。"""

        # 先校验编号文本和分页偏移，避免负数切片或空编号产生含糊行为。
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be non-empty text")
        record_id = record_id.strip()
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        value = self.store.resolve(record_id)
        if not self._allowed(record_id, value):
            raise PermissionError(f"{self.role} cannot read {record_id}")
        # 通过权限检查后才写读取审计，记录谁在第几轮看过哪条数据。
        self.store.record_read(
            actor_role=self.role,
            actor_id=self.actor_id,
            reference_id=record_id,
            round_index=self.round_index,
        )
        if isinstance(value, Evidence):
            # Evidence 正文可能很长：只返回本页，并给出下一页 offset。
            payload = _evidence_page(value, self.policy.max_read_chars, offset)
            if value.source_ref:
                try:
                    source = self.store.resolve(value.source_ref)
                except KeyError:
                    source = None
                if (isinstance(source, dict) and "candidate_id" in source
                        and self._allowed(value.source_ref, source)):
                    payload["source"] = {
                        "candidate_id": source["candidate_id"],
                        "source_url": _clip_text(source["source_url"], 4000),
                        "final_url": _clip_text(source.get("final_url"), 4000),
                    }
                    self.store.record_read(actor_role=self.role, actor_id=self.actor_id,
                        reference_id=value.source_ref, round_index=self.round_index)
            return payload
        if isinstance(value, WorkspaceTask):
            record = self.store.workspace_task_record(value.task_id)
            payload = {**value.to_dict(), "round_index": record["round_index"],
                       "run_status": record["run_status"], "error": record["error"]}
            observations = _workspace_action_observations(self.store, value.task_id)
            if observations:
                payload["action_observations"] = observations
            return self._structured_page(payload, {"record_type": "workspace_task", "task_id": value.task_id}, offset)
        if isinstance(value, (Finding, WorkspaceReport)):
            payload = value.to_dict()
            if isinstance(value, Finding):
                identity = {"record_type": "finding", "finding_id": value.finding_id,
                            "workspace_id": value.workspace_id}
            else:
                identity = {"record_type": "report", "task_id": value.task_id,
                            "status": value.status}
                payload["finding_ids"] = [finding.finding_id for finding in value.findings]
                payload.update(_task_context(self.store, value.task_id))
            # 长判断按完整 JSON 分页，保留卡片无法容纳的正文、限定与引用。
            return self._structured_page(payload, identity, offset)
        if isinstance(value, dict) and "candidate_id" in value:
            payload = dict(value)
            payload["title"] = payload.pop("search_title", None)
            payload["snippet"] = payload.pop("search_snippet", None)
            # 最终跳转地址可能随后续获取更新，作为每页的当前元数据返回。
            identity = {"record_type": "candidate", "candidate_id": value["candidate_id"],
                        "final_url": payload.pop("final_url", None)}
            return self._structured_page(payload, identity, offset)
        # 普通 C 材料先转成字符串，再应用同样的分页规则。
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        end = min(len(rendered), offset + self.policy.max_read_chars)
        return {
            "record_type": "material",
            "reference_id": record_id,
            "content": rendered[offset:end],
            "content_offset": offset,
            "next_offset": end if end < len(rendered) else None,
        }

    def query(
        self,
        query: str,
        *,
        record_types: Iterable[RecordType] | None = None,
        cursor: int = 0,
    ) -> RecordPage:
        """在允许的记录类型中做轻量全文查询并返回一页相关结果。"""

        # 查询文本、最大长度和 cursor 都由 Harness 校验，不信任模型参数。
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text")
        if len(query) > self.policy.max_query_chars:
            raise ValueError(
                f"query exceeds the configured {self.policy.max_query_chars}-character limit"
            )
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        # HI 默认且最多查询 R/E；GIA 额外可查询判断、报告和未提交报告的任务。
        requested = tuple(
            record_types
            or (
                ("candidate", "evidence")
                if self.role == "HI"
                else ("candidate", "evidence", "finding", "report", "workspace_task")
            )
        )
        allowed_types = (
            {"candidate", "evidence"}
            if self.role == "HI"
            else {"candidate", "evidence", "finding", "report", "workspace_task"}
        )
        if not requested or any(value not in allowed_types for value in requested):
            raise PermissionError(f"{self.role} cannot query the requested record types")
        # 排序项依次保存：相关分、插入顺序、展示卡片、真实引用编号。
        ranked: list[tuple[float, int, dict[str, Any], str]] = []
        order = 0
        if "candidate" in requested:
            # HI 查询候选时仍受局部引用集合限制，不能偷看其他 Workspace 的 R。
            for candidate in self.store.candidate_index_records():
                if self.role == "HI" and (
                    self.local_reference_ids is None
                    or candidate["candidate_id"] not in self.local_reference_ids
                ):
                    continue
                card = _candidate_card(candidate)
                texts = [str(value or "") for value in candidate.values()]
                ranked.append((_relevance(query, *texts), order, card, candidate["candidate_id"]))
                order += 1
        if "evidence" in requested:
            # 排序检查完整正文；返回值仍只包含六字段卡片。
            for evidence in self.store.evidence_index_records(content_chars=None):
                card = _evidence_index_card(evidence)
                ranked.append(
                    (
                        _relevance(
                            query,
                            evidence["label"], evidence["content_excerpt"],
                        ),
                        order,
                        card,
                        evidence["evidence_id"],
                    )
                )
                order += 1
        if "finding" in requested:
            # 这两个分支只可能由 GIA 请求，因为上面已经限制 allowed_types。
            for finding in self.store.finding_records():
                card = _finding_card(finding)
                ranked.append(
                    (
                        _relevance(query, json.dumps(finding.to_dict(), ensure_ascii=False)),
                        order,
                        card,
                        finding.finding_id,
                    )
                )
                order += 1
        if "report" in requested:
            for report in self.store.workspace_report_records():
                context = _task_context(self.store, report.task_id)
                card = {**_report_card(report), **_task_context(self.store, report.task_id, compact=True)}
                ranked.append(
                    (
                        _relevance(query, json.dumps({**report.to_dict(), **context}, ensure_ascii=False)),
                        order,
                        card,
                        report.task_id,
                    )
                )
                order += 1
        if "workspace_task" in requested:
            for record in self.store.workspace_task_records():
                if record["run_status"] == "reported":
                    continue
                ranked.append((_relevance(query, json.dumps(record, ensure_ascii=False)),
                               order, _workspace_task_card(record), record["task_id"]))
                order += 1
        # 先按相关分降序；同分时较新的插入项（order 较大）在前。
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        positive = [item for item in ranked if item[0] > 0]
        fallback = not positive
        matches = ranked if fallback else positive
        page = matches[cursor : cursor + self.policy.query_page_size]
        for _, _, _, reference_id in page:
            self.store.record_read(
                actor_role=self.role,
                actor_id=self.actor_id,
                reference_id=reference_id,
                round_index=self.round_index,
            )
        next_cursor = cursor + len(page) if cursor + len(page) < len(matches) else None
        cards = []
        for score, _, card, _ in page:
            cards.append(
                {
                    **card,
                    "retrieval": {
                        "match": "recent_fallback" if fallback else "lexical",
                        "score": round(score, 6),
                    },
                }
            )
        return RecordPage(tuple(cards), next_cursor)


class EvidenceViewBuilder:
    """为一个 HI Workspace 选择本轮最值得放进上下文的 Evidence 卡片。"""

    def __init__(self, store: GlobalStore, policy: MemoryPolicy) -> None:
        """保存 Store 与容量策略。"""

        self.store = store
        self.policy = policy

    def build(
        self,
        *,
        task: str,
        workspace_id: str,
        explicit_material_ids: Iterable[str],
        focus_evidence_ids: Iterable[str],
        records: Iterable[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """按显式材料、focus、任务相关性、本 Workspace 发现和新旧程度排序。"""

        # set 使后续成员检查为常数时间，并自然忽略重复输入。
        explicit = set(explicit_material_ids)
        focus = set(focus_evidence_ids)
        ranked: list[tuple[int, int, float, int, int, dict[str, Any]]] = []
        # 测试或调用方可以传入同一时点快照；否则直接读取 Store 当前索引。
        evidence_records = (
            tuple(records) if records is not None else self.store.evidence_index_records()
        )
        for evidence in evidence_records:
            ranked.append(
                (
                    int(evidence["evidence_id"] in explicit),
                    int(evidence["evidence_id"] in focus),
                    _relevance(
                        task, evidence["label"], evidence["content_excerpt"]
                    ),
                    int(evidence["discovered_by"] == workspace_id),
                    _evidence_index(evidence["evidence_id"]),
                    evidence,
                )
            )
        # 前五项就是上述五级排序键；最后一项保留原记录用于构造卡片。
        ranked.sort(key=lambda item: item[:5], reverse=True)
        return tuple(
            _evidence_index_card(item[5])
            for item in ranked[: self.policy.max_evidence_cards]
        )


def _frame_coverage(frames: dict[float, float | None], duration_s: float | None = None) -> dict[str, Any]:
    """按容器帧时长合并覆盖；已知媒体时长时包含尚未查看的尾段。"""

    ranges: list[list[float]] = []
    for start, duration in sorted(frames.items()):
        if duration is None:
            continue
        end = float(Decimal(str(start)) + Decimal(str(duration)))
        # 帧起点和时长分别转为 float，相邻端点可能只差一次浮点舍入。
        if ranges and start <= math.nextafter(ranges[-1][1], math.inf):
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    span_end = max([duration_s or 0.0, *frames, *(end for _, end in ranges)])
    gaps: list[list[float]] = []
    cursor = 0.0
    for start, end in ranges:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if cursor < span_end:
        gaps.append([cursor, span_end])
    return {**({"media_duration_s": duration_s} if duration_s is not None else {}),
            "coverage_span_s": [0.0, span_end], "observed_ranges_s": ranges,
            "unconfirmed_ranges_s": gaps,
            "unknown_duration_frame_times_s": sorted(time for time, duration in frames.items() if duration is None)}


def workspace_visual_observations(
    store: GlobalStore, task_id: str, current_image_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """汇总本 HI 成功请求的图片与当前附图；裁剪图不计为查看了完整视频帧。"""

    identifiers = dict.fromkeys(eid for event in store.trajectory_events(
        actor_role="HI", actor_id=task_id, event_type="raw_model_output")
        for eid in event["payload"].get("image_evidence_ids", []))
    identifiers.update(dict.fromkeys(current_image_ids))
    return _visual_observations(store, identifiers)


def investigation_visual_observations(store: GlobalStore) -> dict[str, Any]:
    """按实际附图记录合并所有 HI 的观察统计，不传递局部判断。"""

    identifiers = (eid for event in store.trajectory_events(actor_role="HI", event_type="raw_model_output")
                   for eid in event["payload"].get("image_evidence_ids", []))
    return _visual_observations(store, identifiers)[1]


def _visual_observations(
    store: GlobalStore, identifiers: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = []
    videos: dict[str, dict[float, float | None]] = {}
    sources: dict[str, Evidence] = {}
    for eid in dict.fromkeys(identifiers):
        image = store.evidence(eid)
        images.append({"evidence_id": eid, "derived_from": list(image.derived_from), "locator": image.locator})
        try:
            locator = json.loads(image.locator)
        except (TypeError, ValueError):
            continue
        time_s = locator.get("time_s") if isinstance(locator, dict) else None
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
            continue
        duration = locator.get("frame_duration_s")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0 or not math.isfinite(time_s + duration)):
            duration = None
        for parent_id in image.derived_from:
            if parent_id not in sources:
                sources[parent_id] = store.evidence(parent_id)
            if "video" in sources[parent_id].modalities:
                frames = videos.setdefault(parent_id, {})
                previous = frames.get(time_s)
                frames[time_s] = (min(previous, duration) if previous is not None and duration is not None
                                  else previous if previous is not None else duration)
    coverage = []
    for eid, frames in videos.items():
        duration_s = None
        profiles = store.find_derived_evidence(derived_from=(eid,), acquired_by="INSPECT/media_profile")
        for profile in profiles:
            try:
                value = json.loads(profile.content)
            except (TypeError, ValueError):
                continue
            length = value.get("duration_s") if isinstance(value, dict) and value.get("media_kind") == "video" else None
            if isinstance(length, (int, float)) and not isinstance(length, bool) and math.isfinite(length) and length > 0:
                duration_s = float(length)
                break
        coverage.append({"evidence_id": eid, "observed_frame_count": len(frames),
                         **_frame_coverage(frames, duration_s)})
    return images, {"image_evidence_count": len(images), "videos": coverage}


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
