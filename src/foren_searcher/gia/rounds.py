"""RoundCoordinator：并发运行一轮独立 Workspace，收集报告，支持轮中复核。

从原 orchestration.py 拆出（2026-09-15 拆包）。
"""

from __future__ import annotations

import asyncio
import copy
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..contracts import WorkspaceReport, WorkspaceTask
from ..hi.workspace import CustomActionHandler, WorkspaceHarness, WorkspaceRunner
from ..inspect.service import InspectService
from ..policies import HiPolicy, PolicyReviewDeferred
from ..search.service import SearchService
from ..store.global_store import GlobalStore
from ..store.records import MemoryPolicy
from ._common import _LOGGER, _error_payload, _record_runtime_event

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

