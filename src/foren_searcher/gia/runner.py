"""GiaRunner：驱动一次 GIA 决策（INVESTIGATE / FINISH / 只读记忆动作）。

从原 orchestration.py 拆出（2026-09-15 拆包）。
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import Callable
from typing import Any

from ..contracts import GiaDecision, GiaInvestigateDecision, parse_gia_decision
from ..policies import GiaPolicy, OutputRepairBudget, OutputRepairLimit, PolicyOutputError, PolicyReviewDeferred, output_repair_scope
from ..store.global_store import GlobalStore
from ..store.records import MemoryPolicy, RecordAccess
from ._common import _error_payload, _record_runtime_event
from .global_view import GlobalView, bound_global_view

class GiaTurnLimit(RuntimeError):
    """GIA 连续使用记忆动作却未给出决定，达到轮次上限。"""

    pass


class GiaPolicyTimeout(TimeoutError):
    """单次 GIA 模型调用超过配置截止时间。"""

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

