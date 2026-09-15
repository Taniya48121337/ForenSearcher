"""GIA 会话、动态调查轮次和顶层 InvestigationRuntime。

主流程从下往上分三层：``GiaRunner`` 驱动一次全局模型决策；``RoundCoordinator``
并发运行一轮独立 Workspace；``InvestigationRuntime`` 在轮次之间串行推进、负责恢复、
资源限制和关闭。每轮内部可以并行，但下一轮一定等待上一轮全部进入终态。

本文件只编排组件，不实现 SEARCH/INSPECT 的内部算法，也不让 Workspace 创建子树。
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    Evidence,
    GiaDecision,
    GiaFinishDecision,
    GiaInvestigateDecision,
    WorkspaceReport,
    WorkspaceTask,
    parse_gia_decision,
)
from .ids import assign_workspace_tasks, is_evidence_id
from .inspect.service import InspectService
from .memory import GlobalView, GlobalViewBuilder, MemoryPolicy, RecordAccess, bound_global_view
from .policies import GiaPolicy, HiPolicy, OutputRepairBudget, OutputRepairLimit, PolicyOutputError, PolicyReviewDeferred, output_repair_scope
from .search.service import SearchService
from .store import GlobalStore
from .workspace import CustomActionHandler, WorkspaceHarness, WorkspaceRunner

# 三个正则分别识别规范 E、看起来像 E 但可能不规范的编号、以及 C 原始说法编号。
_EVIDENCE_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])E[1-9][0-9]*(?![A-Za-z0-9_])")
_EVIDENCE_LIKE_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])E[0-9]+(?![A-Za-z0-9_])")
_CLAIM_REFERENCE = re.compile(r"C(?:0|[1-9][0-9]*)\Z")
_LOGGER = logging.getLogger(__name__)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """把任意异常压缩成可持久化、最多 2000 字符的错误对象。"""

    message = str(exc)
    return {
        "type": type(exc).__name__,
        "message": message[:2000],
        **({"message_truncated": True} if len(message) > 2000 else {}),
    }


def _record_runtime_event(
    store: GlobalStore,
    *,
    actor_role: str,
    actor_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """尽力写运行轨迹；轨迹失败不能覆盖真正的调查结果或异常。"""

    try:
        store.record_event(
            actor_role=actor_role,
            actor_id=actor_id,
            event_type=event_type,
            payload=payload,
        )
    except Exception:
        # 数据主流程与审计附加记录分离，日志保留事件身份供运维修复。
        _LOGGER.exception(
            "could not persist a runtime trajectory event",
            extra={"actor_role": actor_role, "actor_id": actor_id, "event_type": event_type},
        )


@dataclass(frozen=True)
class WorkspaceOutcome:
    """RoundCoordinator 收集的一项 Workspace 终态：报告或结构化错误。"""

    task_id: str
    run_status: str
    report: WorkspaceReport | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """只导出实际存在的 report/error，避免用 null 混淆状态。"""

        value: dict[str, Any] = {
            "task_id": self.task_id,
            "run_status": self.run_status,
        }
        if self.report is not None:
            value["report"] = self.report.to_dict()
        if self.error is not None:
            value["error"] = copy.deepcopy(self.error)
        return value


@dataclass(frozen=True)
class InvestigationResult:
    """整个调查的最终 GIA 决定，以及答案中实际引用的完整 Evidence。"""

    decision: GiaFinishDecision
    cited_evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, Any]:
        """导出最终答案与可独立检查的证据记录。"""

        return {
            "decision": self.decision.to_dict(),
            "cited_evidence": [item.to_dict() for item in self.cited_evidence],
        }


@dataclass(frozen=True)
class InitialMedia:
    """调用 ``run_new`` 时尚未进入 Store 的原始媒体。

    调用方必须且只能提供 ``data`` 或 ``path`` 之一；Runtime 会复制并提交 Artifact，
    随后用真正的 E 编号替换这个临时输入。
    """

    source_ref: str
    media_type: str
    modalities: tuple[str, ...]
    label: str
    original_filename: str
    data: bytes | None = None
    path: str | Path | None = None

    def __post_init__(self) -> None:
        """验证来源、媒体描述、模态和二进制输入的基本不变量。"""

        # 等价比较为真表示两者同时为空或同时有值，这两种都不允许。
        if (self.data is None) == (self.path is None):
            raise ValueError("InitialMedia requires exactly one of data or path")
        # 这些文字会进入审计和 Artifact 元数据，所以都必须明确非空。
        for name in ("source_ref", "media_type", "label", "original_filename"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"InitialMedia {name} must be non-empty text")
        if (
            not isinstance(self.modalities, tuple)
            or not self.modalities
            or any(not isinstance(item, str) or not item.strip() for item in self.modalities)
            or len(self.modalities) != len(set(self.modalities))
        ):
            # modalities 必须是非空、无重复 tuple，保持不可变公共契约。
            raise ValueError("InitialMedia modalities must be unique non-empty strings")
        if self.data is not None and (
            not isinstance(self.data, bytes) or not self.data
        ):
            raise ValueError("InitialMedia data must be non-empty bytes")
        if self.path is not None and not isinstance(self.path, (str, Path)):
            raise ValueError("InitialMedia path must be a filesystem path")
        if isinstance(self.path, str) and not self.path.strip():
            raise ValueError("InitialMedia path must not be empty")


class GiaTurnLimit(RuntimeError):
    """GIA 连续使用记忆动作却未给出决定，达到轮次上限。"""

    pass


class GiaPolicyTimeout(TimeoutError):
    """单次 GIA 模型调用超过配置截止时间。"""

    pass


class InvestigationLimit(RuntimeError):
    """调查试图开启超过 ``max_rounds`` 的新轮次。"""

    pass


class RuntimeClosedError(RuntimeError):
    """调用方在 Runtime 已关闭或正在关闭时提交新工作。"""

    pass


class GiaRunner:
    """执行 GIA 的有限多轮循环，并提供两个只读 Memory 动作。"""

    def __init__(
        self,
        *,
        store: GlobalStore,
        policy: GiaPolicy,
        memory_policy: MemoryPolicy,
        max_turns: int = 16,
        turn_timeout_s: float | None = 120.0,
    ) -> None:
        """绑定 Store、GIA Policy、MemoryPolicy、轮数和单轮超时。"""

        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if turn_timeout_s is not None and (
            isinstance(turn_timeout_s, bool)
            or not isinstance(turn_timeout_s, (int, float))
            or not math.isfinite(turn_timeout_s)
            or turn_timeout_s <= 0
        ):
            raise ValueError("turn_timeout_s must be positive or null")
        self.store = store
        self.policy = policy
        self.memory_policy = memory_policy
        self.max_turns = max_turns
        self.turn_timeout_s = float(turn_timeout_s) if turn_timeout_s is not None else None

    @staticmethod
    def actions() -> tuple[dict[str, Any], ...]:
        """返回 GIA 唯一可用的 QUERY_MEMORY 与 READ_RECORD schema。"""

        return (
            {
                "name": "QUERY_MEMORY",
                "kind": "harness",
                "description": "Search Evidence, Finding, Candidate Source, Report, and unreported Workspace Task cards. Workspace tasks include saved run status and errors; reported tasks are searchable as reports.",
                "schema": {
                    "type": "object",
                    "required": ["query"],
                    "additionalProperties": False,
                    "properties": {
                        "query": {"type": "string"},
                        "record_types": {
                            "type": "array",
                            "items": {
                                "enum": ["candidate", "evidence", "finding", "report", "workspace_task"]
                            },
                        },
                        "cursor": {"type": "integer", "minimum": 0},
                    },
                },
            },
            {
                "name": "READ_RECORD",
                "kind": "harness",
                "description": "Read a saved record by ID. Long text or structured records return content pages; pass next_offset as offset to continue. Image Evidence can supply pixels for review. New frame extraction, cropping or transcription belongs in a Workspace.",
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

    async def run(
        self,
        view: GlobalView,
        *,
        round_index: int,
        can_create_round: bool = True,
        validator: Callable[[GiaDecision], None] | None = None,
        round_in_progress: bool = False,
    ) -> GiaDecision | None:
        """让 GIA 查询记忆若干次，直到返回可验证的 INVESTIGATE/FINISH。"""

        with output_repair_scope() as repairs:
            return await self._run(view, round_index=round_index, can_create_round=can_create_round,
                                   validator=validator, repairs=repairs, round_in_progress=round_in_progress)

    async def _run(self, view: GlobalView, *, round_index: int, can_create_round: bool,
                   validator: Callable[[GiaDecision], None] | None, repairs: OutputRepairBudget,
                   round_in_progress: bool) -> GiaDecision | None:

        # 私有观察随本次会话的记忆动作保留，不写入共享证据或后续会话。
        history: list[dict[str, Any]] = []
        # role=GIA 让 RecordAccess 使用全局读取权限。
        access = RecordAccess(
            store=self.store,
            role="GIA",
            actor_id="GIA",
            round_index=round_index,
            policy=self.memory_policy,
        )
        for turn_index in range(1, self.max_turns + 1):
            # GlobalView 本轮保持不变；变化仅来自 GIA 的只读历史。
            model_view = view.to_dict()
            model_view["runtime_constraints"] = {
                "round_index": round_index,
                "can_create_round": can_create_round,
                "turn_index": turn_index,
                "max_turns": self.max_turns,
                "remaining_turns": self.max_turns - turn_index,
            }
            if round_in_progress:
                model_view["runtime_constraints"]["round_in_progress"] = True
            # 相同记录页保留最近一次读取；完整行动轨迹仍在 Store 中。
            seen_pages: set[tuple[str, int]] = set()
            retained = []
            for step in reversed(history):
                output = step.get("output", {})
                observation = step.get("observation", {})
                if output.get("action") == "READ_RECORD" and isinstance(observation.get("record"), dict):
                    arguments = output.get("arguments", {})
                    page = (arguments["record_id"], arguments.get("offset", 0))
                    if page in seen_pages:
                        continue
                    seen_pages.add(page)
                retained.append(step)
            history = list(reversed(retained))[-self.memory_policy.max_history_entries:]
            model_history = copy.deepcopy(history)
            prepare = getattr(self.policy, "prepare_model_view", None)
            if callable(prepare):
                prepare(model_view, history=tuple(model_history))
            # 原始证据与报告优先；私有分析按整块省略，不截成可能改变含义的句段。
            for step in model_history:
                if len(json.dumps(model_view, ensure_ascii=False)) + len(json.dumps(model_history, ensure_ascii=False)) <= self.memory_policy.max_model_view_chars:
                    break
                if step.pop("reasoning", None):
                    model_view["reasoning_context_omitted"] = True
            # 已请求的正文页与全局视图共用总预算，避免仍有空余却只留下两页。
            while True:
                history_chars = len(json.dumps(model_history, ensure_ascii=False))
                try:
                    model_view = bound_global_view(model_view, self.memory_policy.max_model_view_chars - history_chars)
                    break
                except ValueError:
                    if len(model_history) <= 1:
                        raise
                    model_history.pop(0)
            actions = () if round_in_progress else self.actions()
            # 调用前保存完整模型输入，使一次决策可以从轨迹复现。
            self.store.record_event(
                actor_role="GIA",
                actor_id="GIA",
                event_type="model_input",
                payload={
                    "turn_index": turn_index,
                    "round_index": round_index,
                    "model_view": model_view,
                    "history": copy.deepcopy(model_history),
                    "actions": list(actions),
                },
            )
            failures_before = repairs.failures
            try:
                if self.turn_timeout_s is None:
                    # None 只适合受控环境；生产通常使用下面的有限超时分支。
                    raw = await self.policy.decide(
                        model_view=model_view,
                        history=tuple(copy.deepcopy(model_history)),
                        actions=actions,
                    )
                else:
                    # asyncio.timeout 会取消超时中的模型协程。
                    async with asyncio.timeout(self.turn_timeout_s):
                        raw = await self.policy.decide(
                            model_view=model_view,
                            history=tuple(copy.deepcopy(model_history)),
                            actions=actions,
                        )
            except OutputRepairLimit:
                raise
            except PolicyOutputError as exc:
                if repairs.failures == failures_before:
                    repairs.reject(exc)
                observation = {"action":"HARNESS_REJECTION", "error":str(exc)[:2000]}
                history.append({"turn_index":turn_index, "observation":observation})
                _record_runtime_event(self.store, actor_role="GIA", actor_id="GIA",
                    event_type="observation", payload={"round_index":round_index,
                    "turn_index":turn_index, "observation":observation})
                continue
            except TimeoutError as exc:
                # 对外统一抛 GiaPolicyTimeout，同时保存发生轮次和 GIA turn。
                error = {
                    "type": "GiaPolicyTimeout",
                    "message": "GIA policy turn timed out",
                }
                _record_runtime_event(
                    self.store,
                    actor_role="GIA",
                    actor_id="GIA",
                    event_type="runtime_error",
                    payload={
                        "round_index": round_index,
                        "turn_index": turn_index,
                        "error": error,
                    },
                )
                raise GiaPolicyTimeout(error["message"]) from exc
            except BaseException as exc:
                # 取消和模型异常也要先尽力进入轨迹，然后保持原异常语义。
                if round_in_progress and isinstance(exc, PolicyReviewDeferred):
                    raise
                _record_runtime_event(
                    self.store,
                    actor_role="GIA",
                    actor_id="GIA",
                    event_type="runtime_error",
                    payload={
                        "round_index": round_index,
                        "turn_index": turn_index,
                        "error": _error_payload(exc),
                    },
                )
                raise
            # 模型输出先原样保存，再交给严格契约解析。
            self.store.record_event(
                actor_role="GIA",
                actor_id="GIA",
                event_type="model_output",
                payload={
                    "round_index": round_index,
                    "turn_index": turn_index,
                    "output": copy.deepcopy(raw),
                },
            )
            try:
                if not isinstance(raw, dict):
                    raise ValueError("GIA output must be an object")
                if round_in_progress:
                    if raw == {"decision": "WAIT"}:
                        return None
                    if raw.get("decision") != "FINISH":
                        raise ValueError("An active-round review must WAIT or FINISH")
                if "decision" in raw:
                    # 有 decision 就按终止 GIA 输出解析；validator 检查 Store 相关约束。
                    decision = parse_gia_decision(raw)
                    if (
                        isinstance(decision, GiaInvestigateDecision)
                        and not can_create_round
                    ):
                        raise ValueError(
                            "investigation round budget is exhausted; GIA must FINISH"
                        )
                    if validator is not None:
                        validator(decision)
                    return decision
                # 没有 decision 时，只可能是一个受控的只读记忆动作。
                observation = self._execute_memory_action(raw, access)
            except (KeyError, PermissionError, ValueError) as exc:
                repairs.reject(exc)
                # 可修复格式/权限错误作为 HARNESS_REJECTION 交给下一轮 GIA。
                observation = {"action": "HARNESS_REJECTION", "error": str(exc)}
            else:
                repairs.progress()
            step = {"turn_index": turn_index, "output":copy.deepcopy(raw), "observation": observation}
            reasoning = model_view.pop("model_reasoning", None)
            if observation.get("action") != "HARNESS_REJECTION" and isinstance(reasoning, str) and reasoning:
                step["reasoning"] = reasoning
            history.append(step)
            self.store.record_event(
                actor_role="GIA",
                actor_id="GIA",
                event_type="observation",
                payload={
                    "round_index": round_index,
                    "turn_index": turn_index,
                    "observation": observation,
                },
            )
        # 循环耗尽说明 GIA 始终没有给出有效决定。
        error = {
            "type": "GiaTurnLimit",
            "message": "GIA exceeded its configured turn limit",
        }
        _record_runtime_event(
            self.store,
            actor_role="GIA",
            actor_id="GIA",
            event_type="runtime_error",
            payload={"round_index": round_index, "error": error},
        )
        raise GiaTurnLimit(error["message"])

    @staticmethod
    def _execute_memory_action(raw: dict[str, Any], access: RecordAccess) -> dict[str, Any]:
        """校验并执行一条 GIA 只读动作，返回模型可见 Observation。"""

        # 精确顶层字段检查阻止模型夹带未定义控制信息。
        if set(raw) != {"action", "arguments"}:
            raise ValueError("GIA memory action requires action and arguments")
        action = raw["action"]
        arguments = raw["arguments"]
        if not isinstance(action, str) or not isinstance(arguments, dict):
            raise ValueError("invalid GIA memory action")
        action = action.upper()
        if action == "QUERY_MEMORY":
            # 每个内建动作再次检查自己的参数白名单和必填字段。
            allowed = {"query", "record_types", "cursor"}
            if not set(arguments) <= allowed or "query" not in arguments:
                raise ValueError("QUERY_MEMORY accepts query, record_types, and cursor")
            return {
                "action": action,
                **access.query(
                    arguments["query"],
                    record_types=arguments.get("record_types"),
                    cursor=arguments.get("cursor", 0),
                ).to_dict(),
            }
        if action == "READ_RECORD":
            allowed = {"record_id", "offset"}
            if not set(arguments) <= allowed or "record_id" not in arguments:
                raise ValueError("READ_RECORD accepts record_id and offset")
            return {
                "action": action,
                "record": access.read(
                    arguments["record_id"], offset=arguments.get("offset", 0)
                ),
            }
        raise ValueError(f"GIA action is not available: {action}")


class RoundCoordinator:
    """并发执行同一调查轮中的多个独立 Workspace，并汇总它们的终态。"""

    def __init__(
        self,
        *,
        store: GlobalStore,
        hi_policy: HiPolicy,
        search_service: SearchService | None,
        inspect_service: InspectService | None,
        memory_policy: MemoryPolicy,
        max_workspace_turns: int = 32,
        workspace_timeout_s: float | None = 900.0,
        max_concurrency: int | None = None,
        custom_actions: dict[str, tuple[dict[str, Any], CustomActionHandler]] | None = None,
        workspace_harness_type: type[WorkspaceHarness] = WorkspaceHarness,
    ) -> None:
        """绑定 HI 与工具服务，并验证 Workspace 轮数、超时和并发配置。"""

        if (
            isinstance(max_workspace_turns, bool)
            or not isinstance(max_workspace_turns, int)
            or max_workspace_turns < 1
        ):
            raise ValueError("max_workspace_turns must be a positive integer")
        if (
            max_concurrency is not None
            and (
                isinstance(max_concurrency, bool)
                or not isinstance(max_concurrency, int)
                or max_concurrency < 1
            )
        ):
            raise ValueError("max_concurrency must be a positive integer or null")
        if (
            workspace_timeout_s is not None
            and (
                isinstance(workspace_timeout_s, bool)
                or not isinstance(workspace_timeout_s, (int, float))
                or not math.isfinite(workspace_timeout_s)
                or workspace_timeout_s <= 0
            )
        ):
            raise ValueError("workspace_timeout_s must be positive or null")
        # 这些对象由同一个顶层 Runtime 组装，所有 Workspace 共享 Store 和工具服务。
        self.store = store
        self.hi_policy = hi_policy
        self.search_service = search_service
        self.inspect_service = inspect_service
        self.memory_policy = memory_policy
        self.max_workspace_turns = max_workspace_turns
        self.workspace_timeout_s = workspace_timeout_s
        self.max_concurrency = max_concurrency
        self.custom_actions = custom_actions or {}
        self.workspace_harness_type = workspace_harness_type

    async def run(
        self, *, round_index: int, tasks: tuple[WorkspaceTask, ...], resume: bool = False,
        report_observer: Callable[[tuple[WorkspaceOutcome, ...]], Awaitable[bool]] | None = None,
        review_delay_s: float = 30.0,
    ) -> tuple[WorkspaceOutcome, ...]:
        """运行一整轮；单个 Workspace 错误会成为 outcome，不取消其他 Workspace。"""

        # 轮次、任务非空性和 task_id 唯一性先在产生异步任务前完成检查。
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("round_index must be a positive integer")
        if not tasks:
            raise ValueError("a round requires at least one Workspace Task")
        if any(
            not isinstance(task.task_id, str)
            or not task.task_id
            or not isinstance(task.task, str)
            or not task.task.strip()
            for task in tasks
        ):
            raise ValueError("Workspace Tasks require non-empty task_id and task text")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("Workspace Task IDs must be unique within a round")
        for task in tasks:
            # 输入任务必须逐字段等于 start_round 已原子保存的记录，防止恢复时漂移。
            stored = self.store.workspace_task_record(task.task_id)
            if (
                int(stored["round_index"]) != round_index
                or stored["task"] != task.task
                or tuple(stored["materials"]) != task.materials
            ):
                raise ValueError(
                    f"Workspace Task {task.task_id} does not match its stored round record"
                )
        # 标准日志只记录数量；完整任务内容留在受控 trajectory_events。
        _LOGGER.info(
            "investigation round started",
            extra={"round_index": round_index, "workspace_count": len(tasks)},
        )
        _record_runtime_event(
            self.store,
            actor_role="HARNESS",
            actor_id="HARNESS",
            event_type="round_started",
            payload={"round_index": round_index, "workspace_count": len(tasks)},
        )
        # semaphore 限制同时运行数量；未配置时允许本轮所有 Workspace 并行。
        semaphore = asyncio.Semaphore(
            self.max_concurrency or max(1, len(tasks))
        )

        async def run_one(task: WorkspaceTask) -> WorkspaceOutcome:
            """执行一个 Workspace，并把所有终止情况归一化为 WorkspaceOutcome。"""

            async with semaphore:
                # 恢复运行时，已经终结的 Workspace 直接复用，不重复调用模型或工具。
                existing = self.store.workspace_task_record(task.task_id)
                if existing["run_status"] == "reported":
                    return WorkspaceOutcome(
                        task.task_id,
                        "reported",
                        report=self.store.workspace_report(task.task_id),
                    )
                if existing["run_status"] in {
                    "runtime_error", "cancelled", "timed_out"
                }:
                    return WorkspaceOutcome(
                        task.task_id,
                        existing["run_status"],
                        error=existing["error"],
                    )
                # 只有 pending 才进入真正执行阶段，并先持久化 running 状态。
                self.store.set_workspace_run_status(task.task_id, "running")
                started = time.monotonic()
                harness = self.workspace_harness_type(
                    task=task,
                    store=self.store,
                    search_service=self.search_service,
                    inspect_service=self.inspect_service,
                    memory_policy=self.memory_policy,
                    round_index=round_index,
                )
                for name, (schema, handler) in self.custom_actions.items():
                    # 每个 Harness 独立注册同一组自定义动作，局部内存不会共享。
                    harness.register_action(name, schema=schema, handler=handler)
                if resume:
                    harness.restore_working_memory()
                runner = WorkspaceRunner(
                    harness=harness,
                    policy=self.hi_policy,
                    max_turns=self.max_workspace_turns,
                )
                try:
                    if self.workspace_timeout_s is None:
                        # 无超时分支保留给离线受控运行。
                        report = await runner.run()
                    else:
                        # wait_for 到时会取消 WorkspaceRunner，包括在途模型/工具协程。
                        report = await asyncio.wait_for(
                            runner.run(deadline=asyncio.get_running_loop().time() + self.workspace_timeout_s),
                            timeout=self.workspace_timeout_s
                        )
                    outcome = WorkspaceOutcome(
                        task.task_id, "reported", report=report
                    )
                    _record_runtime_event(
                        self.store,
                        actor_role="HARNESS",
                        actor_id=task.task_id,
                        event_type="workspace_outcome",
                        payload={
                            "round_index": round_index,
                            "run_status": outcome.run_status,
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                        },
                    )
                    return outcome
                except asyncio.TimeoutError:
                    # 超时是可持久化终态，不再把该 Workspace 当作 pending 自动重跑。
                    error = {"type": "timeout", "message": "Workspace runtime timed out"}
                    self.store.set_workspace_run_status(
                        task.task_id, "timed_out", error=error
                    )
                    _record_runtime_event(
                        self.store,
                        actor_role="HARNESS",
                        actor_id=task.task_id,
                        event_type="workspace_outcome",
                        payload={
                            "round_index": round_index,
                            "run_status": "timed_out",
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                            "error": error,
                        },
                    )
                    return WorkspaceOutcome(task.task_id, "timed_out", error=error)
                except asyncio.CancelledError as exc:
                    # 外层取消需要写状态后继续 raise，保持 asyncio 的取消传播语义。
                    error = {"type": "cancelled", "message": str(exc) or "Workspace runtime was cancelled"}
                    self.store.set_workspace_run_status(
                        task.task_id, "cancelled", error=error
                    )
                    _record_runtime_event(
                        self.store,
                        actor_role="HARNESS",
                        actor_id=task.task_id,
                        event_type="workspace_outcome",
                        payload={
                            "round_index": round_index,
                            "run_status": "cancelled",
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                            "error": error,
                        },
                    )
                    raise
                except Exception as exc:
                    # 普通运行错误只终止当前 Workspace，其他并行任务继续完成。
                    error = _error_payload(exc)
                    self.store.set_workspace_run_status(
                        task.task_id, "runtime_error", error=error
                    )
                    _record_runtime_event(
                        self.store,
                        actor_role="HARNESS",
                        actor_id=task.task_id,
                        event_type="workspace_outcome",
                        payload={
                            "round_index": round_index,
                            "run_status": "runtime_error",
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                            "error": error,
                        },
                    )
                    return WorkspaceOutcome(task.task_id, "runtime_error", error=error)

        running = {asyncio.create_task(run_one(task)): task for task in tasks}
        pending = set(running)
        collected: dict[str, WorkspaceOutcome] = {}
        finished_from_reports = False
        review_at: float | None = None
        try:
            while pending:
                timeout = max(0.0, review_at - time.monotonic()) if review_at is not None else None
                done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    outcome = future.result()
                    collected[outcome.task_id] = outcome
                    if report_observer is not None and outcome.run_status == "reported" and review_at is None:
                        review_at = time.monotonic() + review_delay_s
                if not pending or done or review_at is None:
                    continue
                review_at = None
                assert report_observer is not None
                review_task = asyncio.ensure_future(report_observer(tuple(collected.values())))
                try:
                    changed, _ = await asyncio.wait({review_task, *pending}, return_when=asyncio.FIRST_COMPLETED)
                    if changed.intersection(pending):
                        # 新报告已使本次视图过期，不再等待旧汇总阻挡下一次决策。
                        should_finish = False
                        _record_runtime_event(self.store, actor_role="GIA", actor_id="GIA",
                            event_type="round_review_superseded", payload={"round_index": round_index})
                    else:
                        should_finish = review_task.result()
                except PolicyReviewDeferred as exc:
                    _record_runtime_event(self.store, actor_role="GIA", actor_id="GIA",
                        event_type="round_review_deferred", payload={"round_index": round_index, "reason": str(exc)})
                    should_finish = False
                except Exception as exc:
                    _record_runtime_event(self.store, actor_role="GIA", actor_id="GIA",
                        event_type="round_review_failed", payload={"round_index": round_index, "error": _error_payload(exc)})
                    should_finish = False
                finally:
                    if not review_task.done():
                        review_task.cancel()
                    await asyncio.gather(review_task, return_exceptions=True)
                if should_finish:
                    finished_from_reports = True
                    message = "GIA concluded the investigation from completed reports"
                    for future in pending:
                        if not future.done():
                            future.cancel(message)
                    await asyncio.gather(*pending, return_exceptions=True)
                    for future in pending:
                        task = running[future]
                        if future.cancelled():
                            error = {"type": "cancelled", "message": message}
                            if self.store.workspace_task_record(task.task_id)["run_status"] == "pending":
                                self.store.set_workspace_run_status(task.task_id, "cancelled", error=error)
                            collected[task.task_id] = WorkspaceOutcome(task.task_id, "cancelled", error=error)
                        else:
                            collected[task.task_id] = future.result()
                    pending.clear()
        except BaseException:
            for future in running:
                if not future.done():
                    future.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise
        outcomes = tuple(collected[task.task_id] for task in tasks)
        status = (
            "completed"
            if all(item.run_status == "reported" or (
                finished_from_reports and item.run_status == "cancelled"
                and item.error == {"type": "cancelled", "message": "GIA concluded the investigation from completed reports"}
            ) for item in outcomes)
            else "completed_with_errors"
        )
        # 所有 Workspace 都到终态后，才提交轮次终态。
        self.store.finish_round(round_index, status=status)
        _record_runtime_event(
            self.store,
            actor_role="HARNESS",
            actor_id="HARNESS",
            event_type="round_finished",
            payload={
                "round_index": round_index,
                "status": status,
                "outcomes": [item.to_dict() for item in outcomes],
            },
        )
        _LOGGER.info(
            "investigation round finished",
            extra={"round_index": round_index, "round_status": status},
        )
        return outcomes


class InvestigationRuntime:
    """在共享 Store 上运行由 GIA 指导的动态调查轮次。

    一个 Runtime 同时只推进一个调查；``_run_lock`` 也让关闭操作等待当前调查到达
    安全边界。若要横向扩展，应按调查拆分 Store，而不是多个 Runtime 抢同一数据库。
    """

    def __init__(
        self,
        *,
        store: GlobalStore,
        gia_policy: GiaPolicy,
        hi_policy: HiPolicy,
        search_service: SearchService | None = None,
        inspect_service: InspectService | None = None,
        memory_policy: MemoryPolicy | None = None,
        max_rounds: int = 8,
        max_gia_turns: int = 16,
        gia_turn_timeout_s: float | None = 120.0,
        max_workspace_turns: int = 32,
        workspace_timeout_s: float | None = 900.0,
        max_concurrency: int | None = None,
        max_workspaces_per_round: int = 32,
        max_materials_per_workspace: int = 64,
        max_original_materials: int = 64,
        max_task_chars: int = 16000,
        custom_actions: dict[str, tuple[dict[str, Any], CustomActionHandler]] | None = None,
        close_services: bool = True,
        workspace_harness_type: type[WorkspaceHarness] = WorkspaceHarness,
        report_review_delay_s: float | None = 30.0,
    ) -> None:
        """验证顶层资源限制，并组装 GIA、RoundCoordinator 与 GlobalViewBuilder。"""

        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        if report_review_delay_s is not None and (isinstance(report_review_delay_s, bool)
                or not isinstance(report_review_delay_s, (int, float)) or not math.isfinite(report_review_delay_s)
                or report_review_delay_s < 0):
            raise ValueError("report_review_delay_s must be nonnegative or null")
        self.report_review_delay_s = report_review_delay_s
        for name, value in (
            ("max_workspaces_per_round", max_workspaces_per_round),
            ("max_materials_per_workspace", max_materials_per_workspace),
            ("max_original_materials", max_original_materials),
            ("max_task_chars", max_task_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Runtime 可选择是否拥有两个服务的关闭责任，默认拥有。
        self.store = store
        self.search_service = search_service
        self.inspect_service = inspect_service
        self.close_services = close_services
        self._closed = False
        self._closing = False
        # 同一实例的 run/run_new/aclose 通过一把异步锁串行化。
        self._run_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self.memory_policy = memory_policy or MemoryPolicy()
        self.max_rounds = max_rounds
        self.max_workspaces_per_round = max_workspaces_per_round
        self.max_materials_per_workspace = max_materials_per_workspace
        self.max_original_materials = max_original_materials
        self.max_task_chars = max_task_chars
        # 把顶层配置下发到两个更小的执行器，Runtime 本身只管理轮次循环。
        self.gia = GiaRunner(
            store=store,
            policy=gia_policy,
            memory_policy=self.memory_policy,
            max_turns=max_gia_turns,
            turn_timeout_s=gia_turn_timeout_s,
        )
        self.rounds = RoundCoordinator(
            store=store,
            hi_policy=hi_policy,
            search_service=search_service,
            inspect_service=inspect_service,
            memory_policy=self.memory_policy,
            max_workspace_turns=max_workspace_turns,
            workspace_timeout_s=workspace_timeout_s,
            max_concurrency=max_concurrency,
            custom_actions=custom_actions,
            workspace_harness_type=workspace_harness_type,
        )
        self.view_builder = GlobalViewBuilder(store, self.memory_policy)

    async def __aenter__(self) -> "InvestigationRuntime":
        """进入 async with 时确认实例仍接受新调查。"""

        self._ensure_accepting()
        return self

    async def __aexit__(self, *_args: object) -> None:
        """离开 async with 时执行可等待、可重复调用的优雅关闭。"""

        await self.aclose()

    async def aclose(self) -> None:
        """只创建一次关闭任务；调用者取消等待也不会取消真正的清理任务。"""

        if self._close_task is None:
            # 先置 closing，使新 run 在关闭任务取得锁之前也会被拒绝。
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        """等待在途调查释放运行锁，再依次关闭自己拥有的所有服务。"""

        errors: list[BaseException] = []
        try:
            async with self._run_lock:
                if self.close_services:
                    # 一个服务失败仍继续关闭另一个，最后统一抛出收集到的异常。
                    for service in (self.inspect_service, self.search_service):
                        if service is None:
                            continue
                        try:
                            await service.aclose()
                        except BaseException as exc:
                            errors.append(exc)
        finally:
            # 即使服务关闭失败，Runtime 自身也必须保持 closed，不能继续接单。
            self._closed = True
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("multiple services failed to close", errors)

    def _ensure_accepting(self) -> None:
        """集中检查 Runtime 生命周期；所有公开运行入口都先调用它。"""

        if self._closed or self._closing:
            raise RuntimeClosedError("InvestigationRuntime is closing or closed")

    @staticmethod
    def _path_matches(path: Path, expected: Path) -> bool:
        """分块逐字节比较两个文件，验证幂等重试提供的是同一份媒体。"""

        # strict=True 同时取得规范绝对路径并要求文件真实存在。
        path = path.resolve(strict=True)
        expected = expected.resolve(strict=True)
        if not path.is_file() or not expected.is_file():
            return False
        if path.stat().st_size != expected.stat().st_size:
            return False
        # 每次只读 1 MiB，不会把大视频整体载入内存。
        with path.open("rb") as left, expected.open("rb") as right:
            while True:
                left_chunk = left.read(1024 * 1024)
                right_chunk = right.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True

    @staticmethod
    def _bytes_match(path: Path, expected: bytes) -> bool:
        """把已保存文件与调用方直接提供的 bytes 分块比较。"""

        if path.stat().st_size != len(expected):
            return False
        offset = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    return offset == len(expected)
                if chunk != expected[offset : offset + len(chunk)]:
                    return False
                offset += len(chunk)

    def _validate_existing_media(self, item: InitialMedia, evidence: Evidence) -> None:
        """确认相同 source_ref 的重试与 Store 中媒体元数据、字节完全一致。"""

        # 先比较无需读文件的 Evidence 字段。
        if (
            evidence.source_ref != item.source_ref
            or evidence.modalities != item.modalities
            or evidence.label != item.label
            or evidence.artifact_ref is None
        ):
            raise ValueError(
                f"initial media {item.source_ref} does not match its stored Evidence"
            )
        try:
            # 初始媒体 Evidence.content 保存 JSON 元数据，必须仍可解析。
            description = json.loads(evidence.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"stored initial media {item.source_ref} has invalid metadata"
            ) from exc
        artifact = self.store.artifact(evidence.artifact_ref)
        # 文件名、MIME 和 Artifact 自身声明必须共同一致。
        if (
            not isinstance(description, dict)
            or description.get("original_filename") != item.original_filename
            or description.get("media_type") != item.media_type
            or artifact.media_type != item.media_type
        ):
            raise ValueError(
                f"initial media {item.source_ref} does not match its stored metadata"
            )
        # 根据本次输入形式选择 bytes 或 path 比较，不依赖文件名猜测。
        matches = (
            self._bytes_match(artifact.path, item.data)
            if item.data is not None
            else self._path_matches(artifact.path, Path(item.path))  # type: ignore[arg-type]
        )
        if not matches:
            raise ValueError(
                f"initial media {item.source_ref} does not match its stored bytes"
            )

    def prepare_original_materials(
        self,
        *,
        claim_materials: dict[str, Any],
        media: Iterable[InitialMedia] = (),
    ) -> tuple[str, ...]:
        """验证并提交新调查的 C 材料与初始媒体，返回正式 C/E 引用顺序。"""

        # 公开辅助入口也遵守生命周期，关闭后不能继续写 Store。
        self._ensure_accepting()
        if not isinstance(claim_materials, dict):
            raise ValueError("claim_materials must be an object")
        try:
            # 先物化 iterable，后续校验和提交使用完全相同的一批媒体。
            media_items = tuple(media)
        except TypeError as exc:
            raise ValueError("media must be an iterable of InitialMedia values") from exc
        if len(claim_materials) + len(media_items) > self.max_original_materials:
            # C 与媒体合计受同一个原始材料上限保护。
            raise ValueError(
                "original materials exceed the configured limit of "
                f"{self.max_original_materials}"
            )
        if any(not isinstance(item, InitialMedia) for item in media_items):
            raise ValueError("media must contain only InitialMedia values")
        # source_ref 是初始媒体的幂等键，同一请求中不能重复。
        source_refs = [item.source_ref for item in media_items]
        if len(source_refs) != len(set(source_refs)):
            raise ValueError("initial media source_ref values must be unique")

        # 第一遍只验证，不写入；这样任意一项错误都不会留下部分 C 材料。
        claim_exists: dict[str, bool] = {}
        for reference_id, value in claim_materials.items():
            if not isinstance(reference_id, str) or not _CLAIM_REFERENCE.fullmatch(
                reference_id
            ):
                raise ValueError("claim material IDs must use C followed by a number")
            try:
                # allow_nan=False 保证数据是标准 JSON，不接受 NaN/Infinity。
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"claim material {reference_id} must be valid JSON data"
                ) from exc
            exists = self.store.has_reference(reference_id)
            claim_exists[reference_id] = exists
            if exists:
                # 幂等重试只能复用逐值相等的 C，同编号不同内容必须拒绝。
                if self.store.resolve(reference_id) != value:
                    raise ValueError(
                        f"claim material {reference_id} does not match the stored value"
                    )

        # 同样先检查所有文件与已存在 Evidence，再进入真正的提交阶段。
        existing_media: dict[str, Evidence | None] = {}
        for item in media_items:
            if item.path is not None:
                # path 输入必须解析到一个当前存在的非空普通文件。
                try:
                    path = Path(item.path).resolve(strict=True)
                except OSError as exc:
                    raise ValueError(
                        f"initial media {item.source_ref} path is unavailable"
                    ) from exc
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(
                        f"initial media {item.source_ref} path must be a non-empty file"
                    )
            # 已有媒体会做完整字节比较；没有则在第二阶段提交。
            evidence = self.store.find_ingested_evidence(item.source_ref)
            if evidence is not None:
                self._validate_existing_media(item, evidence)
            existing_media[item.source_ref] = evidence

        # 所有前置检查通过后才开始写入，并保持调用方提供的 C、媒体顺序。
        references: list[str] = []
        for reference_id, value in claim_materials.items():
            if not claim_exists[reference_id]:
                self.store.add_material(reference_id, value)
            references.append(reference_id)
        for item in media_items:
            evidence = existing_media[item.source_ref]
            if evidence is None:
                # Store 同时写 Artifact 与 Evidence；返回的正式 E 进入原始引用列表。
                evidence = self.store.commit_ingested_evidence(
                    source_ref=item.source_ref,
                    media_type=item.media_type,
                    modalities=item.modalities,
                    label=item.label,
                    original_filename=item.original_filename,
                    data=item.data,
                    path=item.path,
                )
            else:
                self._validate_existing_media(item, evidence)
            references.append(evidence.evidence_id)
        return tuple(references)

    async def run_new(
        self,
        *,
        task: str,
        claim_materials: dict[str, Any],
        media: Iterable[InitialMedia] = (),
    ) -> InvestigationResult:
        """从未登记的 C/媒体创建或幂等恢复调查，然后运行到最终答案。"""

        self._ensure_accepting()
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be non-empty text")
        if len(task) > self.max_task_chars:
            raise ValueError(
                f"task exceeds the configured {self.max_task_chars}-character limit"
            )
        if not isinstance(claim_materials, dict):
            raise ValueError("claim_materials must be an object")
        try:
            media_items = tuple(media)
        except TypeError as exc:
            raise ValueError("media must be an iterable of InitialMedia values") from exc
        # 锁内再次检查生命周期，覆盖等待锁期间恰好开始关闭的竞态。
        async with self._run_lock:
            self._ensure_accepting()
            originals = self.prepare_original_materials(
                claim_materials=claim_materials, media=media_items
            )
            return await self._run(
                task=task.strip(), original_material_ids=originals
            )

    def _validate_originals(self, values: Iterable[str]) -> tuple[str, ...]:
        """验证 ``run`` 接收的已有原始引用只能是 C 材料或真实 Evidence。"""

        result = tuple(values)
        if len(result) > self.max_original_materials:
            raise ValueError(
                "original_material_ids exceed the configured limit of "
                f"{self.max_original_materials}"
            )
        if any(not isinstance(reference_id, str) for reference_id in result):
            raise ValueError("original_material_ids must contain only reference strings")
        if len(result) != len(set(result)):
            raise ValueError("original_material_ids must not contain duplicates")
        for reference_id in result:
            # resolve 同时验证记录存在；Evidence 可直接接受，其他类型只能是 C。
            value = self.store.resolve(reference_id)
            if isinstance(value, Evidence):
                continue
            if not _CLAIM_REFERENCE.fullmatch(reference_id) or not self.store.has_material(
                reference_id
            ):
                raise ValueError(
                    "original_material_ids may contain only C materials and Evidence IDs"
                )
        return result

    def _validate_tasks(self, decision: GiaInvestigateDecision) -> None:
        """在开启轮次前验证 GIA 的任务数量、长度和每个材料引用。"""

        if len(decision.tasks) > self.max_workspaces_per_round:
            raise ValueError(
                "INVESTIGATE tasks exceed the configured limit of "
                f"{self.max_workspaces_per_round}"
            )
        for task in decision.tasks:
            # 任务文本长度与材料数量分别限制模型输出的两种膨胀方式。
            if len(task.task) > self.max_task_chars:
                raise ValueError(
                    "Workspace task exceeds the configured "
                    f"{self.max_task_chars}-character limit"
                )
            if len(task.materials) > self.max_materials_per_workspace:
                raise ValueError(
                    "task materials exceed the configured limit of "
                    f"{self.max_materials_per_workspace}"
                )
            for reference_id in task.materials:
                # Workspace 材料只接受已有 E 或已登记 C，R/F/W 等编号不能直接进入。
                value = self.store.resolve(reference_id)
                if not isinstance(value, Evidence) and not (
                    isinstance(reference_id, str)
                    and _CLAIM_REFERENCE.fullmatch(reference_id)
                    and self.store.has_material(reference_id)
                ):
                    raise ValueError(
                        "GIA task materials may contain only original C references and Evidence IDs"
                    )

    def _finish(self, decision: GiaFinishDecision) -> InvestigationResult:
        """解析答案引用、保存唯一最终结果，并返回完整 Evidence 对象。"""

        # 先扫描所有“像 E 的编号”，专门拒绝 E0/E01 等非规范形式。
        for reference_id in _EVIDENCE_LIKE_REFERENCE.findall(decision.answer):
            if not is_evidence_id(reference_id):
                raise ValueError(
                    f"FINISH answer contains a non-canonical Evidence ID: {reference_id}"
                )
        # dict.fromkeys 在去重同时保留答案中的首次出现顺序。
        evidence_ids = tuple(
            dict.fromkeys(_EVIDENCE_REFERENCE.findall(decision.answer))
        )
        # 每个编号都必须能从 Store 取出，防止最终答案引用不存在的证据。
        evidence = tuple(self.store.evidence(evidence_id) for evidence_id in evidence_ids)
        if not self.store.has_final_result():
            # 恢复运行读到已保存结果时不重复写最终事件。
            self.store.save_final_result(decision.to_dict())
            _record_runtime_event(
                self.store,
                actor_role="HARNESS",
                actor_id="HARNESS",
                event_type="investigation_finished",
                payload={"evidence_ids": list(evidence_ids)},
            )
            _LOGGER.info(
                "investigation finished",
                extra={"cited_evidence_count": len(evidence_ids)},
            )
        return InvestigationResult(decision=decision, cited_evidence=evidence)

    def _validate_gia_decision(self, decision: GiaDecision) -> None:
        """给 GiaRunner 的 Store 级 validator：任务引用或最终 E 必须真实存在。"""

        if isinstance(decision, GiaInvestigateDecision):
            self._validate_tasks(decision)
            return
        for evidence_id in _EVIDENCE_LIKE_REFERENCE.findall(decision.answer):
            if not is_evidence_id(evidence_id):
                raise ValueError(
                    f"FINISH answer contains a non-canonical Evidence ID: {evidence_id}"
                )
            self.store.evidence(evidence_id)

    def _completed_result(self) -> InvestigationResult | None:
        """若 Store 已有最终答案则解析复用，否则返回 None 继续调查。"""

        if not self.store.has_final_result():
            return None
        decision = parse_gia_decision(self.store.final_result())
        if not isinstance(decision, GiaFinishDecision):
            raise RuntimeError("stored final result is not a FINISH decision")
        return self._finish(decision)

    async def _execute_round(self, *, round_index: int, tasks: tuple[WorkspaceTask, ...],
                             task: str, originals: tuple[str, ...], resume: bool = False) -> GiaFinishDecision | None:
        early_finish: GiaFinishDecision | None = None

        async def review(outcomes: tuple[WorkspaceOutcome, ...]) -> bool:
            nonlocal early_finish
            statuses = {item["task_id"]: item["run_status"] for item in self.store.workspace_task_records(round_index)}
            view = self.view_builder.build(task=task, original_material_ids=originals, latest_round_index=round_index)
            decision = await self.gia.run(view, round_index=round_index + 1, can_create_round=False,
                validator=self._validate_gia_decision, round_in_progress=True)
            current = {item["task_id"]: item["run_status"] for item in self.store.workspace_task_records(round_index)}
            # 评估期间到达的报告可能包含反例，不能使用旧视图提前提交答案。
            if current != statuses:
                _record_runtime_event(self.store, actor_role="GIA", actor_id="GIA", event_type="round_review_superseded",
                    payload={"round_index": round_index})
                return False
            if isinstance(decision, GiaFinishDecision):
                early_finish = decision
                return True
            return False

        outcomes = await self.rounds.run(round_index=round_index, tasks=tasks, resume=resume,
            report_observer=review if self.report_review_delay_s is not None and len(tasks) > 1 else None,
            review_delay_s=self.report_review_delay_s or 0.0)
        accepted = any(item.run_status == "cancelled" and item.error == {
            "type": "cancelled", "message": "GIA concluded the investigation from completed reports"}
            for item in outcomes)
        return early_finish if accepted else None

    async def _resume_open_round(self, round_record: dict[str, Any], *, task: str,
                                 originals: tuple[str, ...]) -> GiaFinishDecision | None:
        """从 Store 恢复一轮尚未完成的 INVESTIGATE，并复用已保存 Workspace。"""

        round_index = int(round_record["round_index"])
        decision = parse_gia_decision(round_record["gia_decision"])
        if not isinstance(decision, GiaInvestigateDecision):
            raise RuntimeError("running round does not contain an INVESTIGATE decision")
        self._validate_tasks(decision)
        # 重新按确定规则计算 W 编号，再与数据库逐项核对防止状态漂移。
        expected_tasks = assign_workspace_tasks(round_index, decision.tasks)
        records = self.store.workspace_task_records(round_index)
        if not records:
            # 兼容“轮次已写、Workspace 尚未写”这一崩溃边界，原子补齐任务。
            self.store.add_workspace_tasks(round_index, expected_tasks)
            records = self.store.workspace_task_records(round_index)
        tasks = tuple(
            WorkspaceTask(
                task_id=record["task_id"],
                task=record["task"],
                materials=tuple(record["materials"]),
            )
            for record in records
        )
        if tasks != expected_tasks:
            raise RuntimeError(
                "stored Workspace Tasks do not match the running round decision"
            )
        for record in records:
            if record["run_status"] in {"running", "cancelled"}:
                # 进程重启后没有真实在途协程，running/cancelled 可恢复为 pending。
                self.store.set_workspace_run_status(record["task_id"], "pending")
        return await self._execute_round(round_index=round_index, tasks=tasks, task=task, originals=originals, resume=True)

    async def run(
        self, *, task: str, original_material_ids: Iterable[str]
    ) -> InvestigationResult:
        """使用已经存在于 Store 的 C/E 引用运行或恢复一项调查。"""

        self._ensure_accepting()
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be non-empty text")
        if len(task) > self.max_task_chars:
            raise ValueError(
                f"task exceeds the configured {self.max_task_chars}-character limit"
            )
        # 在取得锁前只物化 iterable；依赖 Store 的验证在 _run 内完成。
        originals = tuple(original_material_ids)
        async with self._run_lock:
            self._ensure_accepting()
            return await self._run(
                task=task.strip(), original_material_ids=originals
            )

    async def _run(
        self, *, task: str, original_material_ids: Iterable[str]
    ) -> InvestigationResult:
        """顶层串行轮次循环；调用方必须已经持有 ``_run_lock``。"""

        # ensure_investigation 建立调查身份；已有身份必须与 task/originals 完全相同。
        originals = self._validate_originals(original_material_ids)
        self.store.ensure_investigation(
            task=task, original_material_ids=originals
        )
        _record_runtime_event(
            self.store,
            actor_role="HARNESS",
            actor_id="HARNESS",
            event_type="investigation_run",
            payload={
                "resume": bool(self.store.round_records()),
                "original_material_ids": list(originals),
            },
        )
        # 已有最终结果是最强恢复边界，直接返回，不再调用模型。
        completed = self._completed_result()
        if completed is not None:
            return completed
        existing_rounds = self.store.round_records()
        # 轮次必须从 1 连续增长；空洞表示数据库状态不可安全推断。
        expected_indices = list(range(1, len(existing_rounds) + 1))
        if [record["round_index"] for record in existing_rounds] != expected_indices:
            raise RuntimeError("stored rounds are not contiguous")
        latest_round: int | None = None
        if existing_rounds:
            # 用第一轮持久化输入再次核对任务和原始材料，拒绝错任务恢复。
            first_input = existing_rounds[0]["gia_input"]
            expected_originals = self.view_builder.build(
                task=task,
                original_material_ids=originals,
                latest_round_index=None,
            ).to_dict()
            if first_input["task"] != expected_originals["task"] or first_input[
                "original_materials"
            ] != expected_originals["original_materials"]:
                raise ValueError("resume task and original materials do not match the stored investigation")
            latest_record = existing_rounds[-1]
            if latest_record["status"] == "running":
                # 开放轮次先恢复至完整终态，再让 GIA 看见它的报告/错误。
                early = await self._resume_open_round(latest_record, task=task, originals=originals)
                if early is not None:
                    return self._finish(early)
            latest_round = int(latest_record["round_index"])
        start_round = (latest_round or 0) + 1
        round_index = start_round
        while True:
            # 每轮都从当前 Store 重建 GlobalView，不复用可能过期的模型上下文。
            view = self.view_builder.build(
                task=task,
                original_material_ids=originals,
                latest_round_index=latest_round,
            )
            decision = await self.gia.run(
                view,
                round_index=round_index,
                can_create_round=round_index <= self.max_rounds,
                validator=self._validate_gia_decision,
            )
            assert decision is not None
            if isinstance(decision, GiaFinishDecision):
                # FINISH 不创建新轮次，直接进入最终引用校验与保存。
                return self._finish(decision)
            if round_index > self.max_rounds:
                # 只有 GIA 试图继续调查时才检查上限；已有 FINISH 仍可正常返回。
                _record_runtime_event(
                    self.store,
                    actor_role="HARNESS",
                    actor_id="HARNESS",
                    event_type="runtime_error",
                    payload={
                        "type": "InvestigationLimit",
                        "message": "investigation exceeded its configured round limit",
                    },
                )
                raise InvestigationLimit("investigation exceeded its configured round limit")
            # Harness 分配 W 编号，并把 GIA 输入、决定、任务作为一个轮次边界提交。
            tasks = assign_workspace_tasks(round_index, decision.tasks)
            self.store.start_round(
                round_index,
                gia_input=view.to_dict(),
                gia_decision=decision.to_dict(),
                tasks=tasks,
            )
            early = await self._execute_round(round_index=round_index, tasks=tasks, task=task, originals=originals)
            if early is not None:
                return self._finish(early)
            latest_round = round_index
            round_index += 1
