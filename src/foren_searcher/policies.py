"""GIA 与 HI 的可替换策略接口。

这里使用 ``Protocol``（结构化接口）而不是固定基类：只要外部对象提供同名的异步
``decide`` 方法，就能接入 Runtime。这样测试脚本、本地模型和远程模型客户端都可
使用同一主框架，而不让框架依赖某一家模型 SDK。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Protocol


class PolicyOutputError(RuntimeError):
    """模型输出未通过解析或角色校验，可在原有调查循环中纠正。"""


class PolicyReviewDeferred(RuntimeError):
    """可选的轮内检查让出共享推理资源，调查工作继续执行。"""


class OutputRepairLimit(PolicyOutputError):
    """没有完成有效动作，连续输出修复已耗尽。"""


@dataclass
class OutputRepairBudget:
    maximum_failures: int = 4
    failures: int = 0

    def reject(self, error: BaseException) -> None:
        self.failures += 1
        if self.failures >= self.maximum_failures:
            raise OutputRepairLimit(
                f"Model output repair limit reached after {self.failures} invalid outputs without a completed action: {error}"
            ) from error

    def progress(self) -> None:
        self.failures = 0


_OUTPUT_REPAIRS: ContextVar[OutputRepairBudget | None] = ContextVar("output_repairs", default=None)


@contextmanager
def output_repair_scope() -> Iterator[OutputRepairBudget]:
    """解析、角色校验及外层执行共享次数；各并发角色保持独立。"""
    budget = OutputRepairBudget()
    token = _OUTPUT_REPAIRS.set(budget)
    try:
        yield budget
    finally:
        _OUTPUT_REPAIRS.reset(token)


def record_output_rejection(error: BaseException) -> None:
    budget = _OUTPUT_REPAIRS.get()
    if budget is not None:
        budget.reject(error)


class HiPolicy(Protocol):
    """Workspace 内 HI 每一轮决策必须实现的异步接口。"""

    async def decide(
        self,
        *,
        workspace_id: str,
        model_view: dict[str, Any],
        actions: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        """根据局部视图和可用动作，返回一次工具动作或最终报告 JSON。"""

        ...


class GiaPolicy(Protocol):
    """全局 GIA 每一轮规划或汇总必须实现的异步接口。"""

    async def decide(
        self,
        *,
        model_view: dict[str, Any],
        history: tuple[dict[str, Any], ...],
        actions: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        """根据全局视图、只读动作历史和接口清单返回一个 GIA JSON 决定。"""

        ...
