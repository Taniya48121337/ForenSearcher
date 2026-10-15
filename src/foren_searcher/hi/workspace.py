"""单个 Workspace 的局部记忆、动作路由和 HI 多轮循环。

可以把 Workspace 理解成一名只负责一个局部任务的调查员：``WorkspaceMemory`` 保存
它看过什么，``WorkspaceHarness`` 决定它能做什么并实际调用工具，``WorkspaceRunner``
重复执行“给模型视图 → 接收决定 → 校验/执行 → 返回 Observation”，直到报告完成。

重要边界：模型只提出动作，不能直接写 Store、伪造 Observation 或分配 Evidence/Finding
编号；这些副作用全部由 Harness 完成。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from ..contracts import (
    HiActionDecision,
    WorkspaceReport,
    WorkspaceReportDraft,
    WorkspaceTask,
    parse_hi_decision,
)
from ..inspect.contracts import DEFAULT_OPERATIONS as DEFAULT_INSPECT_OPERATIONS, InspectCall, InspectResponse, OcrResult, inspect_action_schema, is_merged_catalog, ocr_summary_line
from ..inspect.service import InspectService
from .memory import EvidenceViewBuilder, MemoryPolicy, RecordAccess, compact_reference, trim_visual_observation_summaries
from ..store.records import _evidence_index_card
from ..i18n import tr
from ..policies import HiPolicy, OutputRepairBudget, OutputRepairLimit, PolicyOutputError, active_guards, guard_state, output_repair_scope
from ..search.contracts import SearchRequest, SearchResponse
from ..search.service import SearchService
from ..store.global_store import GlobalStore


_LOGGER = logging.getLogger(__name__)


def _observation_evidence_cards(value: Any) -> list[dict[str, Any]]:
    """只取工具响应和记忆响应中的证据元数据，不遍历来源正文。"""

    if not isinstance(value, dict) or not isinstance(value.get("details"), dict):
        return []
    details = value["details"]
    action = value.get("action")
    cards: Any = []
    if action == "INSPECT":
        cards = details.get("evidence", [])
    elif action == "QUERY_MEMORY":
        cards = details.get("results", [])
    elif action == "READ_RECORD" and details.get("record_type") == "evidence":
        cards = [details]
    elif action == "SEARCH":
        for result in details.get("results", []):
            if isinstance(result, dict):
                for material in result.get("materials", []):
                    if isinstance(material, dict) and isinstance(material.get("evidence"), list):
                        cards.extend(material["evidence"])
    return [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []


def _bounded_model_value(value: Any, max_chars: int) -> Any:
    """深拷贝较小值；超限值保留大小、定长预览和有界证据编号。"""

    # 紧凑 JSON 的字符数就是这里统一采用的大小口径。
    rendered = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(rendered) <= max_chars:
        # 深拷贝避免模型适配器修改 Harness 保存的原始历史对象。
        return copy.deepcopy(value)
    # 明确标记 truncated，模型不会把预览误认为完整结果。
    bounded = {
        "truncated": True,
        "original_size_chars": len(rendered),
        "preview": rendered[:max_chars],
    }
    # 证据卡片来自 Store。保留可继续访问的编号，不从预览正文解析编号，
    # 也不把这些编号登记为已观察内容。
    cards = _observation_evidence_cards(value)
    ids = list(dict.fromkeys(card["evidence_id"] for card in cards
        if isinstance(card.get("evidence_id"), str)
        and re.fullmatch(r"E[1-9][0-9]*", card["evidence_id"])))
    all_ids = list(ids)
    references: dict[str, Any] = {"evidence_ids": ids}
    while ids and len(json.dumps(references)) > max_chars // 2:
        ids.pop()
        references["evidence_ids_truncated"] = True
    if (ids and len(ids) < len(all_ids) and value.get("action") == "INSPECT"
            and value["details"].get("operation") in {"get_frames", "get_pages"}):
        # 截断引用列表仍须保留批次后段，不能只留下视频或文档开头。
        count = len(ids)
        image_ids = list(dict.fromkeys(card["evidence_id"] for card in cards
            if card.get("evidence_id") in all_ids and "image" in card.get("modalities", [])))
        metadata_ids = [identifier for identifier in all_ids if identifier not in image_ids]
        while count:
            metadata = metadata_ids[:max(0, count - 2)] if image_ids else []
            sequence = image_ids or all_ids
            slots = min(count - len(metadata), len(sequence))
            ids[:] = metadata + ([sequence[len(sequence) // 2]] if slots == 1 else
                [sequence[round(index * (len(sequence) - 1) / (slots - 1))] for index in range(slots)])
            if len(json.dumps(references)) <= max_chars // 2:
                break
            count -= 1
        if not count:
            ids.clear()
    if ids:
        bounded.update(references)
        bounded["preview"] = rendered[:max(0, max_chars - len(json.dumps(references)))]
        if value.get("action") == "INSPECT":
            bounded["action"] = "INSPECT"
            bounded["details"] = {"operation": value["details"].get("operation")}
    return bounded


def _focused_evidence_read(action: str, request: Any, observation: Any, focus: set[str]) -> bool:
    """仅已成功读取的关注证据正文享有保留优先级。"""

    if action != "READ_RECORD" or not isinstance(request, dict) or not isinstance(observation, dict):
        return False
    details = observation.get("details")
    return (observation.get("action") == "READ_RECORD" and isinstance(details, dict)
            and details.get("record_type") == "evidence" and isinstance(details.get("content"), str)
            and details.get("evidence_id") == request.get("record_id")
            and request.get("record_id") in focus)


@dataclass(frozen=True)
class ToolObservation:
    """Harness 执行动作后交给 HI 的统一观察结果。

    items 是便于模型快速阅读的摘要句，details 保留该工具的完整结构化响应。
    """

    action: str
    items: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容结构，并复制 details 防止外部原地修改。"""

        return {
            "action": self.action,
            "items": list(self.items),
            "details": copy.deepcopy(self.details),
        }


@dataclass(frozen=True)
class WorkspaceHistoryEntry:
    """Workspace 内一次已执行步骤的原始请求、响应和模型观察。"""

    action: str
    request: Any
    response: Any
    observation: dict[str, Any] | None = None
    focus_evidence_ids: tuple[str, ...] | None = None


@dataclass
class WorkspaceMemory:
    """由 Harness 维护、彼此隔离的单个 Workspace Working Memory。

    candidate_source_ids 保存该 Workspace 发现的 R；evidence_pool_ids 是 Store 中可见的
    E 索引；focus_evidence_ids 由 HI 每轮主动更新；history 保存已经执行过的动作。
    """

    task_id: str
    task: str
    materials: dict[str, Any]
    candidate_source_ids: list[str] = field(default_factory=list)
    evidence_pool_ids: list[str] = field(default_factory=list)
    focus_evidence_ids: list[str] = field(default_factory=list)
    content_observed_evidence_ids: list[str] = field(default_factory=list)
    history: list[WorkspaceHistoryEntry] = field(default_factory=list)
    latest_observation: dict[str, Any] | None = None
    evidence_view: tuple[dict[str, Any], ...] = ()

    @property
    def selected_evidence_ids(self) -> list[str]:
        """兼容旧调用方的名称；实际含义仍是当前局部 focus 集合。"""

        return self.focus_evidence_ids

    def to_model_view(
        self, *, history_limit: int = 8, observation_char_limit: int = 16000
    ) -> dict[str, Any]:
        """构造一次 HI 调用所需的有限、可序列化模型视图。"""

        # 已完成的相同处理和同一 E 的相同读取只展示最近一次。
        # 原始动作不变；动态查询、可重试失败、不同观察和不同焦点均保留。
        history: list[WorkspaceHistoryEntry] = []
        occurrences: dict[int, int] = {}
        for entry in reversed(self.history):
            observation = entry.observation or {}
            details = observation.get("details")
            match = None
            feedback = details.get("feedback") if isinstance(details, dict) else None
            completed = isinstance(details, dict) and (details.get("status") == "ready" or (
                details.get("status") == "partial" and details.get("result") is not None
                and isinstance(feedback, list) and bool(feedback)
                and all(isinstance(item, dict) and item.get("retryable") is False for item in feedback)))
            if entry.action == "INSPECT" and completed and isinstance(details, dict):
                match = next((previous for previous in history
                            if previous.action == entry.action and previous.request == entry.request
                            and previous.focus_evidence_ids == entry.focus_evidence_ids
                            and {k: v for k, v in ((previous.observation or {}).get("details") or {}).items() if k != "execution"}
                                == {k: v for k, v in details.items() if k != "execution"}
                            and (previous.observation or {}).get("items", [])[1:] == observation.get("items", [])[1:]
                            ), None)
            if (entry.action == "READ_RECORD" and observation.get("action") == "READ_RECORD"
                    and isinstance(entry.request, dict) and isinstance(details, dict)
                    and details.get("record_type") == "evidence"
                    and re.fullmatch(r"E[1-9][0-9]*", str(entry.request.get("record_id", "")))
                    and details.get("evidence_id") == entry.request["record_id"]
                    ):
                match = next((previous for previous in history if previous.action == entry.action
                            and previous.request == entry.request
                            and previous.observation == entry.observation
                            and previous.focus_evidence_ids == entry.focus_evidence_ids
                            ), None)
            if match is not None:
                occurrences[id(match)] += 1
                continue
            history.append(entry)
            occurrences[id(entry)] = 1
        history.reverse()
        focus = set(self.focus_evidence_ids)
        while len(history) > max(1, history_limit):
            # 最新动作及反馈优先保留；其余先淘汰非关注证据的最旧记录。
            index = min(range(len(history) - 1), key=lambda i: _focused_evidence_read(
                history[i].action, history[i].request, history[i].observation, focus))
            history.pop(index)
        return copy.deepcopy(
            {
                "task_id": self.task_id,
                "task": self.task,
                "materials": self.materials,
                "candidate_sources": list(self.candidate_source_ids),
                "evidence_view": list(self.evidence_view),
                "focus_evidence_ids": list(self.focus_evidence_ids),
                "content_observed_evidence_ids": list(
                    self.content_observed_evidence_ids
                ),
                "history": [
                    {
                        "action": entry.action,
                        **({"completed_occurrences": occurrences[id(entry)]} if occurrences[id(entry)] > 1 else {}),
                        "request": _bounded_model_value(
                            entry.request, observation_char_limit
                        ),
                        **({"focus_evidence_ids": list(entry.focus_evidence_ids)}
                           if entry.focus_evidence_ids is not None else {}),
                        "observation": _bounded_model_value(
                            # 最后一条 Observation 已在 latest_observation 单独出现，
                            # 历史处放 None，避免同一大响应重复占用上下文。
                            None
                            if index == len(history) - 1
                            and entry.observation == self.latest_observation
                            else entry.observation,
                            observation_char_limit,
                        ),
                    }
                    for index, entry in enumerate(history)
                ],
                "latest_observation": _bounded_model_value(
                    self.latest_observation, observation_char_limit
                ),
                **({"context_truncated": True} if len(self.history) > len(history) else {}),
            }
        )


# 10-13：SEARCH 日期查询的说明（只在 search_service.date_lookup 打开时附在 SEARCH 的 description 后面）
DATE_LOOKUP_NOTE = (" To check when a source appeared, SEARCH {\"source\": an R from earlier results or a URL, "
                    "\"output_type\": \"text\", \"include\": [\"dates\"]} returns the publication date the page declares, the "
                    "platform upload date for videos, and the earliest Internet Archive capture of the page and of its image "
                    "file. A page's date is when that page was published, not when its picture or video first appeared; a "
                    "capture date shows only that it was online by then; no date found does not mean it is new. To find the "
                    "earliest copy, check the dates of several sources and compare them.")

# 自定义动作处理器接收模型参数和 Workspace ID，异步返回 Harness Observation。
CustomActionHandler = Callable[[dict[str, Any], str], Awaitable[ToolObservation]]

# 下面三份字典是展示给模型的 JSON Schema，不负责真正执行动作。
_E_ID_SCHEMA = {"type": "string", "pattern": "^E[1-9][0-9]*$"}
_SEARCH_TEXT_SCHEMA = {
    "type": "string", "minLength": 1, "maxLength": 500,
    "pattern": r"^[\s\S]*\S[\s\S]*$",
}
_SEARCH_SCHEMA: dict[str, Any] = {
    "oneOf": [
        *({
            "type": "object",
            "required": ["input_type", "query", "output_type"],
            "additionalProperties": False,
            "properties": {
                "input_type": {"type": "string", "const": input_type},
                "query": query_schema,
                "output_type": {"enum": ["text", "image", "video"]},
            },
        } for input_type, query_schema in (
            ("text", _SEARCH_TEXT_SCHEMA),
            ("image", _E_ID_SCHEMA),
            ("text_image", {
                "type": "object",
                "required": ["evidence_id", "text"],
                "additionalProperties": False,
                "properties": {"evidence_id": _E_ID_SCHEMA, "text": _SEARCH_TEXT_SCHEMA},
            }),
        )),
        {
            "type": "object",
            "required": ["source", "output_type"],
            "additionalProperties": False,
            "properties": {
                "source": {"type": "string"},
                "output_type": {"enum": ["text", "image", "video"]},
                "include": {"type": "array", "items": {"const": "subtitles"}, "maxItems": 0},
                "subtitle_language": {"type": "null"},
            },
        },
        {
            "type": "object",
            "required": ["source", "output_type", "include"],
            "additionalProperties": False,
            "properties": {
                "source": {"type": "string"},
                "output_type": {"const": "video"},
                "include": {
                    "type": "array", "items": {"const": "subtitles"},
                    "minItems": 1, "maxItems": 1,
                },
                "subtitle_language": {
                    "anyOf": [
                        {"type": "null"},
                        {"type": "string", "pattern": r"^[\s\S]*\S[\s\S]*$"},
                    ],
                },
            },
        },
    ]
}
_INSPECT_SCHEMA = inspect_action_schema()
_INSPECT_SCHEMA_BY_LANGUAGE: dict[tuple[bool, bool, bool], dict[str, Any]] = {}

# 10-13（规格 harness_params.compact_tool_params，hi@0.10.0 起）：打开视频时自动带原语言字幕，SEARCH 不再列
# include=subtitles / subtitle_language（search_service.auto_subtitles 打开时）
AUTO_SUBTITLES_NOTE = (" Opening a video source (output_type=video) also fetches its subtitles in the video's original "
                       "language when the platform has them (subtitle=ready with an E you can READ_RECORD).")


def _inspect_schema(merged: bool = False, compact: bool = False) -> dict[str, Any]:
    """工具 schema 的说明文字随语言开关（10-05 晚，规格 language=en）；按语言缓存，中文仍是模块导入时那份。
    10-12：merged=True 是合并目录（hi@0.8.0 起，compare_media 开着），get_frames / asr 的说明与参数不同。
    10-13：compact=True（hi@0.10.0 起）compare_media 不列 mode。"""

    english = active_guards().english
    if not english and not merged and not compact:
        return _INSPECT_SCHEMA
    key = (bool(english), merged, compact)
    if key not in _INSPECT_SCHEMA_BY_LANGUAGE:
        _INSPECT_SCHEMA_BY_LANGUAGE[key] = inspect_action_schema(merged=merged, compact=compact)
    return _INSPECT_SCHEMA_BY_LANGUAGE[key]


class WorkspaceHarness:
    """验证并执行 HI 动作；工具响应和 Observation 始终由环境产生。"""

    def __init__(
        self,
        *,
        task: WorkspaceTask,
        store: GlobalStore,
        search_service: SearchService | None = None,
        inspect_service: InspectService | None = None,
        memory_policy: MemoryPolicy | None = None,
        round_index: int | None = None,
    ) -> None:
        """验证 WorkspaceTask，建立局部记忆、权限集合和 Evidence 视图。"""

        # task_id/task 是每次模型调用的身份和目标，空值会让审计失去意义。
        if (
            not isinstance(task.task_id, str)
            or not task.task_id.strip()
            or not isinstance(task.task, str)
            or not task.task.strip()
        ):
            raise ValueError("Workspace Task requires non-empty task_id and task text")
        if len(task.task) > 16000:
            # 任务文本先设字符上限，避免仅初始化 Workspace 就制造超大上下文。
            raise ValueError("Workspace task exceeds the 16000-character safety limit")
        if (
            len(task.materials) > 128
            or any(not isinstance(item, str) or not item for item in task.materials)
            or len(task.materials) != len(set(task.materials))
        ):
            # 材料必须有限、非空且不重复；是否真实存在由 compact_reference 继续检查。
            raise ValueError(
                "Workspace Task materials must be unique and contain at most 128 references"
            )
        # 保存工具、策略和轮次配置；None 表示该工具不会出现在 eligible_actions。
        self.store = store
        self.search_service = search_service
        self.inspect_service = inspect_service
        self.memory_policy = memory_policy or MemoryPolicy()
        self.round_index = round_index
        # explicit 是 GIA 明确交给本 Workspace 的初始引用；local 集合会随 SEARCH 扩展。
        self.explicit_material_ids = task.materials
        self._local_reference_ids = set(task.materials)
        # 初始化时把每个编号压缩为模型卡片，而不是把完整 Artifact 放进内存。
        self.memory = WorkspaceMemory(
            task_id=task.task_id,
            task=task.task,
            materials={
                reference_id: compact_reference(store, reference_id)
                for reference_id in task.materials
            },
        )
        # RecordAccess 固定 role=HI，因此会执行 Workspace 级读取隔离规则。
        self.record_access = RecordAccess(
            store=store,
            role="HI",
            actor_id=task.task_id,
            round_index=round_index,
            policy=self.memory_policy,
            local_reference_ids=self._local_reference_ids,
        )
        self.view_builder = EvidenceViewBuilder(store, self.memory_policy)
        self._custom_actions: dict[
            str, tuple[dict[str, Any], CustomActionHandler]
        ] = {}
        # 初始材料视为已交付给 HI，逐条写入读取审计。
        for reference_id in task.materials:
            self.store.record_read(
                actor_role="HI",
                actor_id=task.task_id,
                reference_id=reference_id,
                round_index=round_index,
            )
        self._seed_candidate_materials()
        self.refresh_evidence_view()

    def _seed_candidate_materials(self) -> None:
        """P1.5①（10-05）：GIA 把上一轮找到的候选 R 交给本工作区时，R 进 candidate_sources，
        它已打开的页面 / 封面 / 视频 E 进 materials（模型可直接对页面 E 取帧触发下载）。

        只在 RuntimeGuards.cross_round_handoff 打开时 GIA 才能派 R（gia/runtime.py 校验），旧规格下 materials 里不会有 R。
        """

        for reference_id in self.explicit_material_ids:
            if not (reference_id.startswith("R") and reference_id[1:].isdigit()):
                continue
            if reference_id not in self.memory.candidate_source_ids:
                self.memory.candidate_source_ids.append(reference_id)
            self._local_reference_ids.add(reference_id)
            try:
                linked = self.store.candidate_evidence(reference_id)
            except KeyError:
                continue
            for kind, status, evidence in linked:
                if status not in {"ready", "partial"} or evidence.evidence_id in self.memory.materials:
                    continue
                self.memory.materials[evidence.evidence_id] = compact_reference(self.store, evidence.evidence_id)
                self._local_reference_ids.add(evidence.evidence_id)
                self.store.record_read(actor_role="HI", actor_id=self.memory.task_id,
                                       reference_id=evidence.evidence_id, round_index=self.round_index)

    def restore_working_memory(self) -> None:
        """从本工作区已提交的观察恢复记忆，不执行轨迹中的动作。"""

        self.memory.history.clear()
        self.memory.latest_observation = None
        self.memory.candidate_source_ids.clear()
        self.memory.content_observed_evidence_ids.clear()
        self.memory.focus_evidence_ids.clear()
        self._local_reference_ids.clear()
        self._local_reference_ids.update(self.explicit_material_ids)
        pending: HiActionDecision | None = None
        for event in self.store.trajectory_events(actor_role="HI", actor_id=self.memory.task_id):
            payload = event["payload"]
            kind = event["event_type"]
            if kind == "raw_model_output":
                for identifier in payload.get("image_evidence_ids", []):
                    self._mark_content_observed(identifier)
            if kind == "model_input":
                # 旧轨迹的输入同样由 Harness 生成，保留当时已经展示的内容状态。
                view = payload["model_view"]
                self.memory.focus_evidence_ids = list(view.get("focus_evidence_ids", []))
                if self.memory.history and view.get("history"):
                    focus = view["history"][-1].get("focus_evidence_ids")
                    if focus is not None:
                        self.memory.history[-1] = replace(self.memory.history[-1], focus_evidence_ids=tuple(focus))
                for identifier in view.get("content_observed_evidence_ids", []):
                    self._mark_content_observed(identifier)
                pending = None
            elif kind == "model_output":
                try:
                    parsed = parse_hi_decision(payload["output"])
                except (KeyError, ValueError, TypeError):
                    pending = None
                else:
                    pending = parsed if isinstance(parsed, HiActionDecision) else None
            elif kind == "observation":
                observation = copy.deepcopy(payload["observation"])
                rejected = observation["action"] == "HARNESS_REJECTION"
                if not rejected and pending is None:
                    continue
                if rejected:
                    action, request = "HARNESS_REJECTION", {}
                else:
                    assert pending is not None
                    action, request = pending.action, copy.deepcopy(pending.arguments)
                    self.memory.focus_evidence_ids = list(pending.focus_evidence_ids)
                details = observation["details"]
                if action in {"READ_RECORD", "QUERY_MEMORY"}:
                    observation = self._memory_observation(action, request, details).to_dict()
                if action == "SEARCH":
                    for item in details.get("results", []):
                        identifier = item["candidate_id"]
                        if identifier not in self.memory.candidate_source_ids:
                            self.memory.candidate_source_ids.append(identifier)
                elif (action == "READ_RECORD" and details.get("record_type") == "evidence"
                      and details.get("content") and not details.get("content_notice")):
                    self._mark_content_observed(request["record_id"])
                elif action == "INSPECT":
                    for card in details.get("evidence", []):
                        if card.get("acquired_by") == "INSPECT/media_profile":
                            self._mark_content_observed(card["evidence_id"])
                    result_id = details.get("result_evidence_id")
                    if result_id and details.get("status") in {"ready", "partial"}:
                        self._mark_content_observed(result_id)
                        if details.get("operation") in {"visual_qa", "ocr", "asr", "audio_qa", "locate_regions"}:
                            for identifier in request["targets"]:
                                self._mark_content_observed(identifier)
                state = payload.get("working_memory")
                if state is not None:
                    # 包含拒绝动作前可能已更新的 focus；不会从模型的失败输出推测状态。
                    self.memory.focus_evidence_ids = list(state["focus_evidence_ids"])
                    self.memory.content_observed_evidence_ids = list(state["content_observed_evidence_ids"])
                    self.memory.candidate_source_ids = list(state["candidate_source_ids"])
                self.memory.latest_observation = observation
                self.memory.history.append(WorkspaceHistoryEntry(action, request, copy.deepcopy(details),
                    observation, tuple(self.memory.focus_evidence_ids)))
                pending = None
        self._local_reference_ids.update(self.memory.candidate_source_ids)
        self._seed_candidate_materials()
        self.refresh_evidence_view()

    def register_action(
        self,
        name: str,
        *,
        schema: dict[str, Any],
        handler: CustomActionHandler,
    ) -> None:
        """注册一个业务自定义动作；内建动作名不能被覆盖。"""

        if not isinstance(name, str):
            raise ValueError("action name must be text")
        normalized = name.strip().upper()
        if not normalized or normalized in {"SEARCH", "INSPECT", "QUERY_MEMORY", "READ_RECORD"}:
            raise ValueError(f"action name is reserved or invalid: {name}")
        # 深拷贝 schema，调用方之后修改原字典不会悄悄改变已注册接口。
        self._custom_actions[normalized] = (copy.deepcopy(schema), handler)

    def eligible_actions(self) -> tuple[dict[str, Any], ...]:
        """根据已配置服务生成本 Workspace 当前可见的动作清单。"""

        actions: list[dict[str, Any]] = []
        if self.search_service is not None:
            # 未配置 SearchService 时完全不暴露 SEARCH，模型不会调用一个占位动作。
            search_schema = copy.deepcopy(_SEARCH_SCHEMA)
            description = "Search external sources or open one known URL/R source."
            if getattr(self.search_service, "local_image_search", False) is True:
                # 10-06 找原片实验 imageclip（search/providers/thumbnail_clip.py）：以图搜视频在本机做，只能输出视频。
                for branch in search_schema["oneOf"]:
                    if branch["properties"].get("input_type", {}).get("const") in {"image", "text_image"}:
                        branch["properties"]["output_type"] = {"const": "video"}
                description += (" Image-input search (output_type=video only) runs locally: text_image searches"
                                " YouTube with the text and ranks the results by how closely their thumbnail frames"
                                " match the image; image ranks the video candidates already found. The image is not"
                                " sent to any search engine.")
            elif isinstance(getattr(self.search_service, "image_search_note", None), str):
                # 10-12 gcvimage：真正的以图搜图（Google Cloud Vision Web Detection），说明它能做什么。
                description += " " + self.search_service.image_search_note
            elif not getattr(self.search_service, "allows_image_uploads", True):
                # 10-05：部署不允许上传调查图片时，image / text_image 输入一定被拒（ACCESS_RESTRICTED）。
                # 仍展示它们会让 HI 反复以图搜图（evidb3 video_151 七次全被拒、一次文字搜索都没做），所以直接不列。
                search_schema["oneOf"] = [
                    branch for branch in search_schema["oneOf"]
                    if branch["properties"].get("input_type", {}).get("const") not in {"image", "text_image"}
                ]
                description += (" Image-input search is not available in this deployment;"
                                " describe what the image shows in a text query instead.")
            if getattr(self.search_service, "auto_subtitles", False) is True:
                # 10-13 compact_tool_params：打开来源只剩 {source, output_type}，视频自动带原语言字幕
                kept = []
                for branch in search_schema["oneOf"]:
                    props = branch["properties"]
                    if "source" not in props:
                        kept.append(branch)
                    elif "minItems" not in props.get("include", {}):
                        props.pop("include", None)
                        props.pop("subtitle_language", None)
                        kept.append(branch)
                search_schema["oneOf"] = kept
                description += AUTO_SUBTITLES_NOTE
            if getattr(self.search_service, "date_lookup", False):
                # 10-13（规格 harness_params.search_dates，hi@0.8.1 起）：日期查询，见 search/openers/dates.py
                search_schema["oneOf"].append({
                    "type": "object",
                    "required": ["source", "output_type", "include"],
                    "additionalProperties": False,
                    "properties": {
                        "source": {"type": "string"},
                        "output_type": {"const": "text"},
                        "include": {"type": "array", "items": {"const": "dates"}, "minItems": 1, "maxItems": 1},
                    },
                })
                description += DATE_LOOKUP_NOTE
            actions.append(
                {
                    "name": "SEARCH",
                    "kind": "tool",
                    "description": description,
                    "schema": search_schema,
                }
            )
        if self.inspect_service is not None:
            # INSPECT 采用同样原则：只有真实服务存在才加入动作列表。
            enabled = getattr(self.inspect_service, "enabled_operations", None)
            if enabled is None:
                # 未声明时只展示默认操作；diff_media 等需显式开启的操作不进入旧运行的工具目录。
                enabled = DEFAULT_INSPECT_OPERATIONS
            inspect_schema = copy.deepcopy(_inspect_schema(
                merged=is_merged_catalog(enabled), compact=getattr(self.inspect_service, "compact_params", False) is True))
            inspect_schema["oneOf"] = [branch for branch in inspect_schema["oneOf"]
                if branch["properties"]["request"]["properties"]["operation"]["const"] in enabled]
            for branch in inspect_schema["oneOf"]:
                if branch["properties"]["request"]["properties"]["operation"]["const"] == "visual_qa":
                    branch["properties"]["targets"]["maxItems"] = self.inspect_service.policy.maximum_visual_qa_images
            actions.append(
                {
                    "name": "INSPECT",
                    "kind": "tool",
                    "description": "Run one approved INSPECT operation on one or more E targets.",
                    "schema": inspect_schema,
                }
            )
        # 两个只读记忆动作总是由 Harness 提供，不需要外部服务。
        actions.extend(
            (
                {
                    "name": "QUERY_MEMORY",
                    "kind": "harness",
                    "description": "Search accessible Candidate Source and Evidence cards without creating Evidence.",
                    "schema": {
                        "type": "object",
                        "required": ["query"],
                        "additionalProperties": False,
                        "properties": {
                            "query": {"type": "string"},
                            "record_types": {
                                "type": "array",
                                "items": {"enum": ["candidate", "evidence"]},
                            },
                            "cursor": {"type": "integer", "minimum": 0},
                        },
                    },
                },
                {
                    "name": "READ_RECORD",
                    "kind": "harness",
                    "description": "Read a saved record by ID. Long text or structured records return content pages; pass next_offset as offset to continue. Does not decode PDF, images, audio or video.",
                    "schema": {
                        "type": "object",
                        "required": ["record_id"],
                        "additionalProperties": False,
                        "properties": {
                            "record_id": {"type": "string"},
                            "offset": {"type": "integer", "minimum": 0},
                        },
                    },
                },
            )
        )
        for name, (schema, _) in self._custom_actions.items():
            # 自定义动作排在内建动作后，并再次复制 schema 保护注册状态。
            actions.append({"name": name, "kind": "tool", "schema": copy.deepcopy(schema)})
        return tuple(actions)

    def refresh_evidence_view(self) -> None:
        """读取 Store 最新 Evidence 索引，并重建有限的 HI Evidence View。"""

        records = self.store.evidence_index_records()
        # Evidence 可以跨 Workspace 共享；这里只保存轻量编号池。
        self.memory.evidence_pool_ids = [record["evidence_id"] for record in records]
        for record in records:
            source_ref = record["source_ref"]
            if source_ref and source_ref.startswith("R"):
                # Evidence 指回的 R 也进入局部可读集合，使后续可按 R 重新打开来源。
                self._local_reference_ids.add(source_ref)
        cards = self.view_builder.build(
            task=self.memory.task,
            workspace_id=self.memory.task_id,
            explicit_material_ids=self.explicit_material_ids,
            focus_evidence_ids=self.memory.focus_evidence_ids,
            records=records,
        )
        slots = active_guards().workspace_original_slots
        if slots:
            cards = self._reserve_original_frames(cards, records, slots)
        self.memory.evidence_view = cards

    def _reserve_original_frames(self, cards: tuple[dict[str, Any], ...], records: Iterable[dict[str, Any]],
                                 slots: int) -> tuple[dict[str, Any], ...]:
        """P1.5②（10-05）：evidence_view 固定给原件派生帧留 slots 个名额（按时间均匀）。

        审计：第二轮工作区的 12 张卡全被任务材料（别的视频的页面和封面）占满，第一轮取的 30 帧一张不见。
        顺序：显式材料与焦点卡 → 预留的原件帧 → 其余按原排序；总数仍是 max_evidence_cards（预留帧在前，裁剪时最后才丢）。
        """

        from ..view_annotations import reserved_original_frames

        reserved = reserved_original_frames(self.store, slots)
        if not reserved:
            return cards
        by_id = {record["evidence_id"]: record for record in records}
        pinned = set(self.explicit_material_ids) | set(self.memory.focus_evidence_ids)
        head = [card for card in cards if card.get("evidence_id") in pinned]
        frames = [next((card for card in cards if card.get("evidence_id") == eid), None)
                  or (_evidence_index_card(by_id[eid]) if eid in by_id else None) for eid in reserved]
        frames = [card for card in frames if card is not None and card not in head]
        taken = {card.get("evidence_id") for card in head + frames}
        rest = [card for card in cards if card.get("evidence_id") not in taken]
        return tuple((head + frames + rest)[: max(self.memory_policy.max_evidence_cards, len(head) + len(frames))])

    def model_view(self, *, runtime_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        """刷新动态索引，再把整个 HI 输入裁剪到 max_model_view_chars 内。"""

        # 每轮先刷新，保证其他并行 Workspace 新提交的 E 可以被看到。
        self.refresh_evidence_view()
        value = self.memory.to_model_view(
            history_limit=self.memory_policy.max_history_entries,
            observation_char_limit=self.memory_policy.max_observation_chars,
        )
        if runtime_constraints is not None:
            value["runtime_constraints"] = copy.deepcopy(runtime_constraints)
        # 候选 R 只保留最近若干张卡；完整候选仍在 Store。
        value["candidate_sources"] = [
            compact_reference(self.store, candidate_id)
            for candidate_id in self.memory.candidate_source_ids[
                -self.memory_policy.max_candidate_cards :
            ]
        ]
        self.bound_model_view(value)
        return value

    def bound_model_view(self, value: dict[str, Any]) -> None:
        """在文本视图及附图信息准备后统一裁剪可恢复的历史和索引。"""

        trim_visual_observation_summaries(value, self.memory_policy.max_model_view_chars, details_only=True)
        history_truncated = False
        # 总大小仍超限时沿用正文保留优先级，随后再裁剪候选来源。
        focus = set(self.memory.focus_evidence_ids)
        while value["history"] and len(
            json.dumps(value, ensure_ascii=False, default=str)
        ) > self.memory_policy.max_model_view_chars:
            index = min(range(max(1, len(value["history"]) - 1)), key=lambda i: _focused_evidence_read(
                value["history"][i]["action"], value["history"][i]["request"],
                value["history"][i]["observation"], focus))
            value["history"].pop(index)
            history_truncated = True
        while value["candidate_sources"] and len(
            json.dumps(value, ensure_ascii=False, default=str)
        ) > self.memory_policy.max_model_view_chars:
            value["candidate_sources"].pop(0)
            history_truncated = True
        if history_truncated:
            # 显式告诉模型上下文曾被截断，避免它误判“历史从未发生”。
            value["context_truncated"] = True
        for key in ("evidence_view",):
            while value[key] and len(json.dumps(value, ensure_ascii=False)) > self.memory_policy.max_model_view_chars:
                value[key].pop()
                value["context_truncated"] = True
        trim_visual_observation_summaries(value, self.memory_policy.max_model_view_chars)
        if len(json.dumps(value, ensure_ascii=False)) > self.memory_policy.max_model_view_chars:
            raise ValueError(
                "Workspace task, materials and latest observation exceed max_model_view_chars; "
                "use a larger context budget or a smaller task/material selection"
            )

    def set_focus(self, evidence_ids: tuple[str, ...]) -> None:
        """用 HI 本轮给出的完整列表替换 focus，并验证每个 E 确实存在。"""

        if len(evidence_ids) > self.memory_policy.max_focus_evidence_ids:
            raise ValueError(
                "focus_evidence_ids exceeds the configured limit of "
                f"{self.memory_policy.max_focus_evidence_ids}"
            )
        # 按首次出现顺序去重；每个有效 E 同时记为本 Workspace 已读取。
        seen: set[str] = set()
        values: list[str] = []
        for evidence_id in evidence_ids:
            self.store.evidence(evidence_id)
            if evidence_id not in seen:
                seen.add(evidence_id)
                values.append(evidence_id)
                self.store.record_read(
                    actor_role="HI",
                    actor_id=self.memory.task_id,
                    reference_id=evidence_id,
                    round_index=self.round_index,
                )
        self.memory.focus_evidence_ids = values

    def _record_evidence_references(self, value: Any) -> None:
        """递归扫描自定义参数/响应中的 E 编号并写读取审计。"""

        if isinstance(value, str):
            if value.startswith("E") and value[1:].isdigit():
                try:
                    self._record_evidence_access([value])
                except KeyError:
                    # 不存在的疑似 E 留给动作校验报错，不把它伪装成已读 Evidence。
                    pass
            return
        if isinstance(value, dict):
            for item in value.values():
                self._record_evidence_references(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._record_evidence_references(item)

    def _record_action_access(self, decision: HiActionDecision) -> None:
        """在执行工具前登记动作参数实际引用的 R/E，并实施局部 R 权限。"""

        arguments = decision.arguments
        if decision.action == "SEARCH":
            # 直接打开 R 时，它必须来自本 Workspace 已知候选或显式材料。
            source = arguments.get("source")
            if isinstance(source, str) and source.startswith("R") and source[1:].isdigit():
                if source not in self._local_reference_ids:
                    raise PermissionError(
                        f"HI cannot access Candidate Source {source} in this workspace"
                    )
                self.store.record_read(
                    actor_role="HI",
                    actor_id=self.memory.task_id,
                    reference_id=source,
                    round_index=self.round_index,
                )
            input_type = arguments.get("input_type")
            query = arguments.get("query")
            # 图片和图文搜索会把查询图 E 记作一次实际读取。
            if input_type == "image" and isinstance(query, str):
                self._record_evidence_references(query)
            elif input_type == "text_image" and isinstance(query, dict):
                self._record_evidence_references(query.get("evidence_id"))
        elif decision.action == "INSPECT":
            # INSPECT 的 targets 和 params 都可能引用 Evidence。
            self._record_evidence_references(arguments.get("targets"))
            request = arguments.get("request")
            if isinstance(request, dict):
                self._record_evidence_references(request.get("params"))
        elif decision.action not in {"QUERY_MEMORY", "READ_RECORD"}:
            # 自定义动作没有固定 schema 语义，因此递归扫描全部参数。
            self._record_evidence_references(arguments)

    def _record_evidence_access(self, evidence_ids: list[str]) -> None:
        """把新 E 加入内存编号池，并为每次访问写 Store 审计。"""

        for evidence_id in evidence_ids:
            if evidence_id not in self.memory.evidence_pool_ids:
                self.memory.evidence_pool_ids.append(evidence_id)
            self.store.record_read(
                actor_role="HI",
                actor_id=self.memory.task_id,
                reference_id=evidence_id,
                round_index=self.round_index,
            )

    def _mark_content_observed(self, evidence_id: str) -> None:
        """记录已展示的正文、分析结果或实际探测属性；普通材料标题不算正文。"""

        if evidence_id not in self.memory.content_observed_evidence_ids:
            self.memory.content_observed_evidence_ids.append(evidence_id)

    async def execute_inspect(self, action: InspectCall | dict[str, Any]) -> InspectResponse:
        """调用 INSPECT，登记新 E，并生成模型可见的统一 Observation。"""

        if self.inspect_service is None:
            raise RuntimeError("INSPECT is not configured for this workspace")
        result = await self.inspect_service.execute(
            action, workspace_id=self.memory.task_id
        )
        # 工具响应里的 Evidence 已由 INSPECT 提交完成，现在登记到 Workspace。
        evidence_ids = [card.evidence_id for card in result.evidence]
        self._record_evidence_access(evidence_ids)
        response_value = result.to_dict()
        result_ids = list(dict.fromkeys(
            ([result.result_evidence_id] if result.result_evidence_id else [])
            + list(result.result.referenced_evidence_ids() if result.result else ())))
        execution = None
        if result.execution_id is not None:
            audit = self.store.inspect_execution(result.execution_id)
            info = audit["details"] or {}
            execution = {"id": result.execution_id, "reused": bool(info.get("reused")),
                "created_evidence_ids": info.get("created_evidence_ids", []),
                "reused_evidence_ids": info.get("reused_evidence_ids", [])}
            response_value["execution"] = execution
        for card in result.evidence:
            if card.acquired_by == "INSPECT/media_profile":
                # Profile 标签由实际 probe 属性生成，属性值已经呈现在本次响应中。
                self._mark_content_observed(card.evidence_id)
        observations = [f"INSPECT {result.operation or 'invalid'} returned {result.status}; "
                        f"Result Evidence: {', '.join(result_ids) or 'none'}; "
                        "evidence cards also include source materials and metadata."]
        if execution and execution["reused"]:
            observations[0] += " Same completed request: reused its saved result; no new Evidence. Use focus_evidence_ids with READ_RECORD to review existing images; repeating extraction is unnecessary."
        if result.result is not None and result.operation in {"compare_media", "video_structure"}:
            # 10-12：比较 / 结构分析的结论逐行放进观察，模型不必先读结果 E 才看到差异
            # 10-14：按 DETECT v2.4 的观察格式（图按 E 编号列出、结尾一句 Next），见 _report_observation_lines
            observations.extend(self._report_observation_lines(result))
        if isinstance(result.result, OcrResult):
            # 10-15（A7）：一次 OCR = 一个结果 E。观察只给汇总（出现最多的若干条文字 + 总数），逐框结果在结果 E 里，
            # READ_RECORD 读全；以前 details.result 带全部逐帧结果（fmnv_a_1Kk 31 万字），被截成 preview
            summary = result.result.text_summary()
            targets = action.targets if isinstance(action, InspectCall) else tuple((action or {}).get("targets") or ())
            observations.append(ocr_summary_line(summary, result.result_evidence_id, ", ".join(str(t) for t in targets) or None))
            compact = {"summary": summary}
            if result.result.coverage is not None:
                compact["coverage"] = result.result.coverage.to_dict()
            response_value["result"] = compact
        if result.result_evidence_id is not None and result.status in {"ready", "partial"}:
            # 识别来源和算法适用范围与 result 一起可见，不能只标记结果已经读过。
            saved_result = json.loads(self.store.evidence(result.result_evidence_id).content)
            # 10-14：compare_media / video_structure 的 limitations 不进观察（"字幕、台标也会造成差异"这类句子让 27B
            # 把换物解释成字幕 / 台标；DETECT 的观察里也没有），仍保存在结果 E 里
            skip = {"limitations"} if result.operation in {"compare_media", "video_structure"} else set()
            context = {key: saved_result[key] for key in
                       ("backend", "model", "provider", "sampling_interval_s", "boundary_uncertainty_s", "speed_ratio_range", "vad_filter", "speech_duration_s", "limitations")
                       if key in saved_result and key not in (response_value.get("result") or {}) and key not in skip}
            if context:
                observations.append(json.dumps(context, ensure_ascii=False))
            self._mark_content_observed(result.result_evidence_id)
        observation = ToolObservation(
            "INSPECT",
            tuple(observations),
            response_value,
        )
        request_value = (
            action.to_dict()
            if isinstance(action, InspectCall)
            else copy.deepcopy(action)
        )
        if result.result_evidence_id is not None and result.status in {"ready", "partial"} and result.operation in {
            "visual_qa", "ocr", "asr", "audio_qa", "locate_regions"
        }:
            for evidence_id in request_value["targets"]:
                self._mark_content_observed(evidence_id)
        # latest 供下一轮快速查看，history 保留请求—响应—观察三件套。
        self.memory.latest_observation = observation.to_dict()
        self.memory.history.append(
            WorkspaceHistoryEntry(
                "INSPECT", request_value, response_value, observation.to_dict(),
                tuple(self.memory.focus_evidence_ids),
            )
        )
        return result

    async def execute_search(self, action: SearchRequest | dict[str, Any]) -> SearchResponse:
        """调用 SEARCH，把候选 R、取得的 E 和材料状态写入局部记忆。"""

        if self.search_service is None:
            raise RuntimeError("SEARCH is not configured for this workspace")
        result = await self.search_service.execute(action, workspace_id=self.memory.task_id)
        observations: list[str] = []
        evidence_ids: list[str] = []
        matched: list[tuple[float, str, str, str]] = []      # 10-14 --auto-tools：(fraction, R, 材料 E, 页面 E)
        for item in result.results:
            # 每个 SearchResult 对应一个候选来源 R；保持首次发现顺序。
            if item.candidate_id not in self.memory.candidate_source_ids:
                self.memory.candidate_source_ids.append(item.candidate_id)
            self._local_reference_ids.add(item.candidate_id)
            material_states: list[str] = []
            for material in item.materials:
                # 一个来源可产生页面、图片、字幕等多类材料，每类分别汇总状态和 E。
                card_ids = [card.evidence_id for card in material.evidence]
                evidence_ids.extend(card_ids)
                material_states.append(
                    f"{material.kind}={material.status}"
                    + (f"({','.join(card_ids)})" if card_ids else "")
                )
            observations.append(
                f"[{item.candidate_id}] {item.source_url}; " + ", ".join(material_states)
            )
            for material in item.materials:
                if material.kind == "dates" and material.status in {"ready", "partial"} and material.evidence:
                    # 10-13 日期查询：把查到的日期写成一行（全文在该 E 里，可 READ_RECORD）
                    from ..search.openers.dates import date_summary

                    try:
                        text = self.store.evidence(material.evidence[0].evidence_id).content or ""
                    except Exception:  # noqa: BLE001
                        text = ""
                    observations.append(f"dates[{item.candidate_id}]: {date_summary(text)} "
                                        f"(details and what each date means: {material.evidence[0].evidence_id})")
                    # 10-14：这一行就是日期 E 的内容（各日期都在），算已看过、可以引用（wk8 E 族报告引用不了日期）
                    self._mark_content_observed(material.evidence[0].evidence_id)
            match = getattr(item, "content_match", None)
            if isinstance(match, dict):
                # 10-12：SEARCH 的画面比对结果（原 DETECT find_source）；对上的给出完整比对的调用
                from ..search.source_match import content_match_line

                observations.append(content_match_line(item.candidate_id, match))
                page = next((m.evidence[0].evidence_id for m in item.materials
                             if m.kind == "page" and m.status in {"ready", "partial"} and m.evidence), None)
                if match.get("status") in {"source", "partial"} and page and match.get("material"):
                    if match.get("status") == "source" and active_guards().auto_compare_on_match:
                        matched.append((float(match.get("fraction") or 0.0), item.candidate_id, match["material"], page))
                    call = {"action": "INSPECT", "arguments": {"targets": [match["material"], page], "request": {
                        "operation": "compare_media", "params": {}}}, "focus_evidence_ids": []}
                    observations.append(f"full_comparison[{item.candidate_id}]: " + json.dumps(call, ensure_ascii=False))
            if active_guards().open_as_video_hint:
                observations.extend(self._comparison_hints(item))
        auto_compare = await self._auto_compare(matched) if matched else None
        if auto_compare is not None:
            observations.extend(auto_compare.pop("lines"))
        if result.error is not None:
            # 部分成功时仍可能带顶层错误，把它保留在摘要中供 HI 决定是否继续。
            observations.append(f"SEARCH {result.error.code}: {result.error.message}")
        observations.extend(self._search_excerpts(action, result))
        self._record_evidence_access(evidence_ids)
        response_value = result.to_dict()
        if auto_compare is not None:
            response_value["auto_compare"] = auto_compare
        observation = ToolObservation(
            "SEARCH",
            tuple(observations) or (f"SEARCH stopped with {result.stop_reason}; no results",),
            response_value,
        )
        request_value = (
            action.to_dict()
            if hasattr(action, "to_dict")
            else copy.deepcopy(action)
        )
        self.memory.latest_observation = observation.to_dict()
        self.memory.history.append(
            WorkspaceHistoryEntry("SEARCH", request_value, response_value, observation.to_dict(),
                                  tuple(self.memory.focus_evidence_ids))
        )
        return result

    async def _auto_compare(self, matched: list[tuple[float, str, str, str]]) -> dict[str, Any] | None:
        """10-14 评测协议 --auto-tools on：content_match 判为原片时，运行时替 HI 跑 compare_media [材料, 页面 E]。

        恢复 DETECT 的自动比对（evidM27a–d：合并目录四轮平均 12.75/44，DETECT v2.4 两轮 19.5；找到原片却没比、
        比了一半的 run 都有）。每次 SEARCH 只比匹配比例最高的一个；同一对在本调查里已比过时 INSPECT 直接复用结果。
        结论逐行并进这次 SEARCH 的观察，结果 E 记为已看过，HI 可以直接引用。失败只记一行，不影响 SEARCH。"""

        if self.inspect_service is None or not is_merged_catalog(getattr(self.inspect_service, "enabled_operations", None)):
            return None
        fraction, candidate_id, material, page = max(matched)
        request = {"targets": [material, page], "request": {"operation": "compare_media", "params": {}}}
        head = (f"auto_compare[{candidate_id}]: content_match found {material}'s footage in {candidate_id} "
                f"({100 * fraction:.0f}% of its samples), so INSPECT compare_media on [{material}, {page}] was run automatically")
        try:
            response = await self.inspect_service.execute(request, workspace_id=self.memory.task_id)
        except Exception as exc:  # noqa: BLE001 —— 自动比对失败不能拖垮 SEARCH
            return {"candidate_id": candidate_id, "request": request, "status": "error",
                    "lines": [f"{head}, but it failed ({type(exc).__name__}); you can run it yourself."]}
        self._record_evidence_access([card.evidence_id for card in response.evidence])
        for card in response.evidence:
            if card.acquired_by == "INSPECT/media_profile":
                self._mark_content_observed(card.evidence_id)
        lines = [f"{head}: returned {response.status}; Result Evidence: {response.result_evidence_id or 'none'}."]
        if response.result is not None:
            lines.extend(self._report_observation_lines(response))
        else:
            lines.extend(f"{item.code}: {item.message}"[:300] for item in response.feedback[:2])
        if response.result_evidence_id is not None and response.status in {"ready", "partial"}:
            self._mark_content_observed(response.result_evidence_id)
        return {"candidate_id": candidate_id, "request": request, "status": response.status,
                "result_evidence_id": response.result_evidence_id, "lines": lines}

    COMPARE_NEXT = ("Next: for each listed span, region and person, say what A shows and what B shows there (see images); "
                    "report each as a separate forgery point with its time.")
    STRUCTURE_NEXT = "Next: view the listed shots and judge whether they fit the rest of the footage; cues are not proof."

    def _report_observation_lines(self, response: Any) -> list[str]:
        """compare_media / video_structure 的观察行（10-14，恢复 DETECT v2.4 的观察格式）。

        evidM27a–f 与 evidC27e 逐条对照：DETECT 的观察把每张对照图按 E 编号列出（"Image E44: …"，描述行写成
        "Visual check of E44"），结尾一句 Next 让 HI 把每处差异写成一个伪造点；合并后只写"comparison image 2"，HI 在报告里
        自己猜图的编号、引用没看过的 E，校验退回 4 次后整份报告作废（M27e/f 的 156、194、208）。对照图的内容已经由
        描述模型写进这些行，记为已看过，可以引用。"""

        result = response.result
        images = list(getattr(result, "images", ()) or ())
        ids = [getattr(img, "evidence_id", None) for img in images]

        def name(match: re.Match[str]) -> str:
            k = int(match.group(1))
            return ids[k - 1] if 0 < k <= len(ids) and ids[k - 1] else match.group(0)

        lines = [re.sub(r"comparison image (\d+)", name, str(x)) for x in (getattr(result, "summary_lines", ()) or ())[:20]]
        for img in images[:8]:
            if getattr(img, "evidence_id", None):
                lines.append(f"Image {img.evidence_id}: {getattr(img, 'caption', '')}")
                self._mark_content_observed(img.evidence_id)
        report = getattr(result, "report", None) or {}
        operation = getattr(response, "operation", None)
        if operation == "compare_media" and report.get("source_match"):
            lines.append(self.COMPARE_NEXT)
        elif operation == "video_structure" and report.get("generator_length_shots"):
            lines.append(self.STRUCTURE_NEXT)
        return lines

    def _comparison_hints(self, item: Any) -> list[str]:
        """档 A（eval_plus / 规格 candidate_affordance，10-05）：把"怎么调"写进观察，不写"该做什么"。

        视频候选的下载是 INSPECT 内部的事（inspect/service.py:_resolve_or_acquire）：对候选**页面 E** 做
        get_frames / ocr / asr / align_media 才会下载源视频；SEARCH 的视频打开器 skip_download，再搜一次不会取得视频。
        （10-05 早先的版本给的是 SEARCH source=R 的调用——那条路不下载，已改。）
        - 还没有视频 E → 给出对页面 E 取帧的调用；
        - 本工作区有材料视频 → 给出材料视频与候选（页面 E 或已下载的视频 E）align_media(visual) 的调用。
        evidb4 里 136 个 run 一次都没下载过候选视频，14 次 get_frames 打在封面图 E 上。
        """

        hints: list[str] = []
        video = next((m for m in item.materials if m.kind == "video"), None)
        video_ids = [card.evidence_id for card in video.evidence] if video is not None and video.status in {"ready", "partial"} else []
        page = next((m for m in item.materials if m.kind == "page" and m.status in {"ready", "partial"} and m.evidence), None)
        try:
            from ..platform_video import is_video_candidate

            video_candidate = is_video_candidate(self.store.candidate(item.candidate_id))
        except KeyError:
            video_candidate = False
        if not video_ids and (page is None or not video_candidate):
            return hints
        target = video_ids[0] if video_ids else page.evidence[0].evidence_id
        if not video_ids:
            call = {"action": "INSPECT", "arguments": {"targets": [target], "request": {
                "operation": "get_frames", "params": {"times_s": [0, 15, 30, 60, 120, 240]}}}, "focus_evidence_ids": []}
            ops = ("get_frames/ocr/asr/compare_media" if is_merged_catalog(getattr(self.inspect_service, "enabled_operations", None))
                   else "get_frames/ocr/asr/align_media")
            hints.append(f"open_as_video[{item.candidate_id}]: INSPECT {ops} on {target} "
                         "downloads the source video, e.g. " + json.dumps(call, ensure_ascii=False))
        material = self._material_video_id()
        matched = getattr(item, "content_match", None)
        if isinstance(matched, dict) and matched.get("status") in {"source", "partial"}:
            material = None                  # 10-12：上面的 full_comparison 已给出比对调用
        if material and material != target:
            merged = is_merged_catalog(getattr(self.inspect_service, "enabled_operations", None))
            request = ({"operation": "compare_media", "params": {}} if merged
                       else {"operation": "align_media", "params": {"mode": "visual"}})
            call = {"action": "INSPECT", "arguments": {"targets": [material, target], "request": request}, "focus_evidence_ids": []}
            hints.append(f"compare_with_material[{item.candidate_id}]: " + json.dumps(call, ensure_ascii=False))
        return hints

    def _material_video_id(self) -> str | None:
        from ..view_annotations import original_ids

        # 显式材料优先；GIA 没把原件交给本工作区时退回调查的原始材料（P1.5 跨轮交接的工作区常只拿到 R）
        for reference_id in (*self.explicit_material_ids, *original_ids(self.store)):
            if not reference_id.startswith("E"):
                continue
            try:
                evidence = self.store.evidence(reference_id)
            except Exception:  # noqa: BLE001 —— 不存在的编号
                continue
            if "video" in evidence.modalities and not (evidence.source_ref or "").startswith("R"):
                return reference_id
        return None

    def _search_excerpts(self, action: Any, result: SearchResponse) -> list[str]:
        """harness 参数 search_excerpt_chars > 0 时（09-30 对照项）：SEARCH 自动打开的文本页各附一段与查询最相关的摘录。

        只是节选，不标记为已读正文（content_observed 不变）；完整内容仍需 READ_RECORD。默认 0，行为与以前完全相同。
        """

        limit = self.memory_policy.search_excerpt_chars
        if limit <= 0:
            return []
        query = action.get("query") if isinstance(action, dict) else getattr(action, "query", None)
        if isinstance(query, dict):
            query = query.get("text")
        lines: list[str] = []
        for item in result.results:
            for material in item.materials:
                if material.status not in {"ready", "partial"}:
                    continue
                for card in material.evidence:
                    if "text" not in tuple(getattr(card, "modalities", ()) or ()):
                        continue
                    try:
                        text = self.store.evidence(card.evidence_id).content
                    except Exception:  # noqa: BLE001 —— 取不到正文就不附，不影响 SEARCH 本身
                        continue
                    excerpt = best_excerpt(text if isinstance(text, str) else str(text), str(query or ""), limit)
                    if excerpt:
                        lines.append(tr(f"{card.evidence_id} 摘录（与查询最相关的一段；完整正文用 READ_RECORD）：{excerpt}",
                                        f"{card.evidence_id} excerpt (the passage most relevant to the query; READ_RECORD for the full text): {excerpt}"))
        return lines

    @staticmethod
    def _memory_observation(
        action: str, request: dict[str, Any], details: dict[str, Any]
    ) -> ToolObservation:
        """摘要仅标明读取位置；完整记录在 details 中保留一份。"""
        if action == "READ_RECORD":
            summary = (
                f"READ_RECORD {request['record_id']}; offset={request.get('offset', 0)}; "
                f"next_offset={details.get('next_offset')}"
            )
        elif action == "QUERY_MEMORY":
            summary = (
                f"QUERY_MEMORY returned {len(details['results'])} records; "
                f"next_cursor={details['next_cursor']}"
            )
        else:
            summary = json.dumps(details, ensure_ascii=False, default=str)
        return ToolObservation(action, (summary,), details)

    def _append_harness_observation(
        self, action: str, request: dict[str, Any], details: dict[str, Any]
    ) -> ToolObservation:
        """把 Harness 内建只读动作的 details 统一加入 latest 与 history。"""

        observation = self._memory_observation(action, request, details)
        self.memory.latest_observation = observation.to_dict()
        self.memory.history.append(
            WorkspaceHistoryEntry(
                action, copy.deepcopy(request), copy.deepcopy(details), observation.to_dict(),
                tuple(self.memory.focus_evidence_ids),
            )
        )
        return observation

    async def execute_action(self, decision: HiActionDecision) -> ToolObservation:
        """路由一条已解析 HI 决策，并只返回 Harness 实际产生的 Observation。"""

        # focus 与参数引用在产生副作用前校验；失败会由 Runner 转成 rejection。
        self.set_focus(decision.focus_evidence_ids)
        self._record_action_access(decision)
        if decision.action == "SEARCH":
            await self.execute_search(decision.arguments)
        elif decision.action == "INSPECT":
            await self.execute_inspect(decision.arguments)
        elif decision.action == "QUERY_MEMORY":
            # 精确字段检查避免模型夹带未定义参数。
            allowed = {"query", "record_types", "cursor"}
            if not set(decision.arguments) <= allowed or "query" not in decision.arguments:
                raise ValueError("QUERY_MEMORY accepts query, record_types, and cursor")
            page = self.record_access.query(
                decision.arguments["query"],
                record_types=decision.arguments.get("record_types"),
                cursor=decision.arguments.get("cursor", 0),
            )
            return self._append_harness_observation(
                "QUERY_MEMORY", decision.arguments, page.to_dict()
            )
        elif decision.action == "READ_RECORD":
            allowed = {"record_id", "offset"}
            if set(decision.arguments) - allowed or "record_id" not in decision.arguments:
                raise ValueError("READ_RECORD accepts record_id and offset")
            value = self.record_access.read(
                decision.arguments["record_id"],
                offset=decision.arguments.get("offset", 0),
            )
            if (value.get("record_type") == "evidence"
                    and value.get("content") and not value.get("content_notice")):
                self._mark_content_observed(decision.arguments["record_id"])
            return self._append_harness_observation("READ_RECORD", decision.arguments, value)
        else:
            # 内建分支都不匹配时，只能查找显式注册的自定义动作。
            custom = self._custom_actions.get(decision.action)
            if custom is None:
                raise ValueError(f"action is not available: {decision.action}")
            observation = await custom[1](decision.arguments, self.memory.task_id)
            if not isinstance(observation, ToolObservation):
                # 自定义处理器也必须遵守 Harness Observation 边界。
                raise TypeError("custom action handlers must return ToolObservation")
            self._record_evidence_references(observation.details)
            self.memory.latest_observation = observation.to_dict()
            self.memory.history.append(
                WorkspaceHistoryEntry(
                    decision.action,
                    copy.deepcopy(decision.arguments),
                    copy.deepcopy(observation.details),
                    observation.to_dict(),
                    tuple(self.memory.focus_evidence_ids),
                )
            )
            return observation
        # SEARCH/INSPECT 已在专用方法中写入 latest，这里返回它的只读形态。
        assert self.memory.latest_observation is not None
        return ToolObservation(
            decision.action,
            tuple(self.memory.latest_observation["items"]),
            self.memory.latest_observation["details"],
        )

    def note_repeated_action(self, decision: HiActionDecision, previous: ToolObservation,
                             first_turn: int) -> ToolObservation:
        """评测兜底 3c：完全相同的调用不再执行，把上次的观察原样交回并加一句提示。"""

        self.set_focus(decision.focus_evidence_ids)
        note = (f"REPEATED_ACTION: this exact {decision.action} call already ran at turn {first_turn}; it was not "
                "executed again and its earlier result is repeated below. Choose a different action or submit the "
                "workspace report.")
        observation = ToolObservation(previous.action, (note, *previous.items), copy.deepcopy(previous.details))
        self.memory.latest_observation = observation.to_dict()
        self.memory.history.append(WorkspaceHistoryEntry(
            "REPEATED_ACTION", decision.to_dict(), {"first_turn": first_turn}, observation.to_dict(),
            tuple(self.memory.focus_evidence_ids)))
        return observation

    def reject(self, message: str) -> ToolObservation:
        """把可修复的模型/参数错误变成 HARNESS_REJECTION 交回下一轮 HI。"""

        return self._append_harness_observation("HARNESS_REJECTION", {}, {"error": message})

    def start_episode(self, max_turns: int) -> None:
        """为可选运行时适配器提供工作区开始通知。"""

    async def submit_report(self, draft: WorkspaceReportDraft) -> WorkspaceReport:
        """接收模型直接提交的终止报告。"""

        return self.finish(draft)

    def validate_report(self, draft: WorkspaceReportDraft) -> None:
        """检查报告归属、大小和引用，不提交记录。"""

        # 报告必须属于当前 Workspace，不能替其他 Workspace 提交结果。
        if draft.task_id != self.memory.task_id:
            raise ValueError("Workspace Report task_id does not match this workspace")
        if len(draft.findings) > self.memory_policy.max_findings_per_report:
            # Finding 条数和报告总字符数分别限制结构膨胀与文本膨胀。
            raise ValueError(
                "Workspace Report findings exceed the configured limit of "
                f"{self.memory_policy.max_findings_per_report}"
            )
        rendered = json.dumps(
            draft.to_dict(), ensure_ascii=False, separators=(",", ":")
        )
        if len(rendered) > self.memory_policy.max_report_chars:
            raise ValueError(
                "Workspace Report exceeds the configured "
                f"{self.memory_policy.max_report_chars}-character limit"
            )
        # selected 集合及每条 finding 引用都必须解析成 Store 中真实存在的 Evidence。
        self.set_focus(draft.selected_evidence_ids)
        for finding in draft.findings:
            for evidence_id in finding.evidence_ids:
                self.store.evidence(evidence_id)

    def finish(self, draft: WorkspaceReportDraft) -> WorkspaceReport:
        """执行最终跨字段校验，原子提交报告，再尽力记录报告轨迹事件。"""

        self.validate_report(draft)
        # Store 在事务中分配 Finding 编号并保存正式 WorkspaceReport。
        report = self.store.commit_workspace_report(draft)
        try:
            self.store.record_event(
                actor_role="HI",
                actor_id=self.memory.task_id,
                event_type="workspace_report",
                payload=report.to_dict(),
            )
        except Exception:
            # 报告已经提交，轨迹附加失败不能回滚或伪装成报告失败。
            _LOGGER.exception(
                "workspace report committed but its trajectory event could not be recorded",
                extra={"workspace_id": self.memory.task_id},
            )
        return report


class WorkspaceTurnLimit(RuntimeError):
    """HI 连续多轮仍未提交有效报告时抛出的有限终止错误。"""

    pass


AUTO_REPORT_PREFIX = "auto_report:"


def _action_fingerprint(decision: HiActionDecision) -> str:
    """动作 + 参数的规范化指纹（focus 不算在内：只换关注集合、请求完全相同，仍是重复调用）。"""

    return json.dumps({"action": decision.action, "arguments": decision.arguments}, sort_keys=True,
                      ensure_ascii=False, default=str)


class WorkspaceRunner:
    """驱动一个 HI Workspace 的“模型—工具—观察”循环。"""

    def __init__(
        self, *, harness: WorkspaceHarness, policy: HiPolicy, max_turns: int = 32
    ) -> None:
        """绑定 Harness、HI 策略和最大模型轮数。"""

        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self.harness = harness
        self.policy = policy
        self.max_turns = max_turns

    def _record_observation(self, turn_index: int, observation: ToolObservation) -> None:
        memory = self.harness.memory
        self.harness.store.record_event(
            actor_role="HI", actor_id=memory.task_id, event_type="observation",
            payload={"turn_index": turn_index, "observation": observation.to_dict(),
                "working_memory": {
                    "focus_evidence_ids": list(memory.focus_evidence_ids),
                    "content_observed_evidence_ids": list(memory.content_observed_evidence_ids),
                    "candidate_source_ids": list(memory.candidate_source_ids)}})

    def _auto_incomplete(self, reason: str) -> WorkspaceReport | None:
        """评测兜底 1c：工作区撞上限时自动提交一份 INCOMPLETE 报告，GIA 至少知道它看过哪些 E。

        findings 为空、selected = 已观察内容的 E（取最近的、不超过 focus 上限），status_reason 以 auto_report: 开头，
        SFT 导出据此识别（这不是模型写的报告，不能当教师样本）。没有打开评测兜底时返回 None，保持原来抛 WorkspaceTurnLimit。
        """

        if not active_guards().auto_incomplete_report:
            return None
        memory = self.harness.memory
        observed = list(dict.fromkeys(memory.content_observed_evidence_ids))
        limit = max(0, self.harness.memory_policy.max_focus_evidence_ids)
        selected = tuple(observed[-limit:]) if limit else ()
        draft = WorkspaceReportDraft(task_id=memory.task_id, status="INCOMPLETE", findings=(),
                                     selected_evidence_ids=selected,
                                     status_reason=f"{AUTO_REPORT_PREFIX} {reason}; the workspace produced no report of its own")
        try:
            report = self.harness.finish(draft)
        except (KeyError, PermissionError, ValueError):
            _LOGGER.exception("auto INCOMPLETE report was rejected", extra={"workspace_id": memory.task_id})
            return None
        state = guard_state()
        if state is not None:
            state.count("auto_incomplete_report")
        try:
            self.harness.store.record_event(actor_role="HI", actor_id=memory.task_id, event_type="auto_report",
                                            payload={"reason": reason, "selected_evidence_ids": list(selected)})
        except Exception:
            _LOGGER.exception("could not record auto report event", extra={"workspace_id": memory.task_id})
        return report

    async def run(self, *, deadline: float | None = None) -> WorkspaceReport:
        """运行到报告或轮数上限；deadline 提供外部 loop.time 预算，取消由调度器负责。"""

        with output_repair_scope() as repairs:
            return await self._run(repairs, deadline=deadline)

    async def _run(self, repairs: OutputRepairBudget, *, deadline: float | None) -> WorkspaceReport:

        self.harness.start_episode(self.max_turns)
        loop = asyncio.get_running_loop()
        last_model_call_seconds: float | None = None
        guards = active_guards()
        # 3c（评测兜底）：已执行动作的指纹 → (首次轮次, 观察)；完全相同的调用不再执行、不消耗轮次，最多 repeat_budget 次。
        executed: dict[str, tuple[int, ToolObservation]] = {}
        repeats_left = guards.repeat_budget or 0
        # 10-05（evidb7：27 个工作区里 26 个因重复用完而自动 INCOMPLETE，HI 一份自己的报告都没写，GIA 收不到任何判断）：
        # 约束可见（state_dependent_actions）时，重复用完先给一个"只可交报告"的回合（不耗轮次，最多 2 次），交不出再自动 INCOMPLETE。
        report_only: str | None = None
        report_only_attempts = 0
        turn_index = 0
        while turn_index < self.max_turns:
            turn_index += 1
            if report_only is not None and report_only_attempts >= 2:
                auto = self._auto_incomplete(f"{report_only}, and no valid report followed")
                if auto is not None:
                    return auto
                break
            # 每轮重新生成视图和动作，确保新 Evidence 与服务状态及时生效。
            constraints: dict[str, Any] = {
                "turn_index": turn_index,
                "max_turns": self.max_turns,
                "remaining_turns": self.max_turns - turn_index,
                "max_model_view_chars": self.harness.memory_policy.max_model_view_chars,
                "state_renderer": self.harness.memory_policy.state_renderer,
            }
            if deadline is not None:
                constraints["remaining_seconds"] = max(0.0, deadline - loop.time())
                if last_model_call_seconds is not None:
                    constraints["last_model_call_seconds"] = last_model_call_seconds
            if guards.state_dependent_actions and turn_index == self.max_turns:
                # 3b（eval_plus）：最后一轮动作目录清空，写明只能交报告；8B 不看 remaining_turns == 0（evidb4 139 次）
                constraints["next_output"] = "WORKSPACE REPORT only: this is the last turn; tool actions will be rejected"
            if report_only is not None:
                constraints["next_output"] = (f"WORKSPACE REPORT only: {report_only}; tool actions will be rejected. "
                                              "Report what the observations so far show and what is still missing.")
            model_view = self.harness.model_view(runtime_constraints=constraints)
            actions = () if (guards.state_dependent_actions and turn_index == self.max_turns) or report_only is not None \
                else self.harness.eligible_actions()
            prepare = getattr(self.policy, "prepare_model_view", None)
            if callable(prepare):
                prepare(model_view)
                self.harness.bound_model_view(model_view)
            if len(json.dumps(model_view, ensure_ascii=False)) > self.harness.memory_policy.max_model_view_chars:
                raise ValueError("Workspace multimodal view exceeds max_model_view_chars")
            # 模型输入先进入轨迹，便于完整复现实验和排查决策。
            self.harness.store.record_event(
                actor_role="HI",
                actor_id=self.harness.memory.task_id,
                event_type="model_input",
                payload={"turn_index": turn_index, "model_view": model_view, "actions": list(actions)},
            )
            if report_only is not None:
                report_only_attempts += 1
                turn_index -= 1                  # 只可交报告的回合不耗轮次
            failures_before = repairs.failures
            try:
                # Policy 产生决策；模型适配器只读取可见材料，动作仍由 Harness 执行。
                model_started = loop.time()
                try:
                    raw = await self.policy.decide(
                        workspace_id=self.harness.memory.task_id,
                        model_view=model_view,
                        actions=actions,
                    )
                finally:
                    last_model_call_seconds = loop.time() - model_started
                    supplied_images = {item["evidence_id"] for item in model_view.get("image_inputs", [])}
                    for identifier in model_view.get("model_observed_image_ids", []):
                        if identifier in supplied_images:
                            self.harness._mark_content_observed(identifier)
            except OutputRepairLimit:
                raise
            except PolicyOutputError as exc:
                if repairs.failures == failures_before:
                    repairs.reject(exc)
                observation = self.harness.reject(str(exc)[:2000])
                self._record_observation(turn_index, observation)
                continue
            except BaseException as exc:
                # 包括取消在内的异常都先尽力写一条有限长度的 runtime_error。
                try:
                    message = str(exc)
                    self.harness.store.record_event(
                        actor_role="HI",
                        actor_id=self.harness.memory.task_id,
                        event_type="runtime_error",
                        payload={
                            "turn_index": turn_index,
                            "error": {
                                "type": type(exc).__name__,
                                "message": message[:2000],
                                **(
                                    {"message_truncated": True}
                                    if len(message) > 2000
                                    else {}
                                ),
                            },
                        },
                    )
                except Exception:
                    # 记录异常失败不能覆盖真正的策略异常。
                    _LOGGER.exception(
                        "could not persist a Workspace policy failure event",
                        extra={"workspace_id": self.harness.memory.task_id},
                    )
                raise
            # 原始模型输出单独保存，之后才进入严格解析和执行。
            self.harness.store.record_event(
                actor_role="HI",
                actor_id=self.harness.memory.task_id,
                event_type="model_output",
                payload={"turn_index": turn_index, "output": copy.deepcopy(raw)},
            )
            try:
                decision = parse_hi_decision(raw)
                if isinstance(decision, WorkspaceReportDraft):
                    # Report 是终止输出；finish 成功后立即结束循环。
                    return await self.harness.submit_report(decision)
                if report_only is not None:
                    auto = self._auto_incomplete(f"{report_only}, and the report-only turn produced a tool action")
                    if auto is not None:
                        return auto
                fingerprint = _action_fingerprint(decision) if guards.repeat_budget else None
                if fingerprint is not None and fingerprint in executed:
                    if repeats_left <= 0:
                        if guards.state_dependent_actions and report_only is None:
                            report_only = "identical actions were repeated until the repeat budget ran out"
                            first_turn, previous = executed[fingerprint]
                            observation = self.harness.note_repeated_action(decision, previous, first_turn)
                            repairs.progress()
                            self._record_observation(turn_index, observation)
                            state = guard_state()
                            if state is not None:
                                state.count("report_only_after_repeats")
                            turn_index -= 1      # 不执行、不消耗轮次
                            continue
                        auto = self._auto_incomplete("repeated identical actions exhausted the repeat budget")
                        if auto is not None:
                            return auto
                    else:
                        repeats_left -= 1
                        first_turn, previous = executed[fingerprint]
                        observation = self.harness.note_repeated_action(decision, previous, first_turn)
                        repairs.progress()
                        self._record_observation(turn_index, observation)
                        state = guard_state()
                        if state is not None:
                            state.count("repeated_action_short_circuit")
                        turn_index -= 1          # 不执行、不消耗轮次
                        continue
                observation = await self.harness.execute_action(decision)
                if fingerprint is not None:
                    executed.setdefault(fingerprint, (turn_index, observation))
            except (KeyError, PermissionError, ValueError) as exc:
                repairs.reject(exc)
                # 可修复错误不终止 Workspace，而是作为 Observation 反馈给 HI。
                observation = self.harness.reject(str(exc))
            else:
                repairs.progress()
            self._record_observation(turn_index, observation)
        # 循环自然结束表示模型始终未提交有效报告。
        auto = self._auto_incomplete("turn limit reached")
        if auto is not None:
            return auto
        message = (
            f"Workspace {self.harness.memory.task_id} exceeded its configured turn limit"
        )
        try:
            self.harness.store.record_event(
                actor_role="HI",
                actor_id=self.harness.memory.task_id,
                event_type="runtime_error",
                payload={"error": {"type": "WorkspaceTurnLimit", "message": message}},
            )
        except Exception:
            _LOGGER.exception(
                "could not persist a Workspace turn-limit event",
                extra={"workspace_id": self.harness.memory.task_id},
            )
        raise WorkspaceTurnLimit(message)


_EXCERPT_WORD = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9'’-]{2,}")
_EXCERPT_CJK = __import__("re").compile(r"[\u4e00-\u9fff]{2,}")
_EXCERPT_SPLIT = __import__("re").compile(r"(?<=[。！？.!?])\s+|\n+")
_EXCERPT_STOP = frozenset("the and for with that this from was were are has have had not but its his her their what when where which who how "
                          "site http https www com org".split())


def best_excerpt(text: str, query: str, limit: int) -> str:
    """在正文里找与查询词重合最多的一段连续句子（不超过 limit 字符）；查询为空或都不命中时取开头。纯程序，不调模型。"""

    text = " ".join(text.split()) if "\n" not in text else text
    if not text.strip() or limit <= 0:
        return ""
    terms = {w.lower() for w in _EXCERPT_WORD.findall(query) if w.lower() not in _EXCERPT_STOP}
    for run in _EXCERPT_CJK.findall(query):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    sentences = [x.strip() for x in _EXCERPT_SPLIT.split(text) if x and x.strip()]
    if not sentences:
        return ""

    def score(chunk: str) -> int:
        low = chunk.lower()
        return sum(1 for t in terms if t in low)

    best, best_score = "", (-1, False)
    for start in range(len(sentences)):
        window = ""
        for sentence in sentences[start:]:
            candidate = (window + " " + sentence).strip()
            if len(candidate) > limit and window:
                break
            window = candidate
        value = (score(window), score(sentences[start]) > 0)      # 同样命中数时，优先从命中句开头的一段
        if value > best_score:
            best, best_score = window, value
        if not terms:
            break
    best = best[:limit]
    return best + ("…" if len(best) >= limit else "")
