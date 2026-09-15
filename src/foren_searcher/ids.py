"""编号规则：由 Harness 统一生成和检查 Evidence、Workspace 编号。

新手阅读提示：编号不是普通展示文字，而是不同模块之间引用同一条记录的“主键”。
Evidence 使用 ``E1``、``E2``；Workspace 使用 ``W轮次-序号``，例如 ``W2-3``。
模型只能使用 Harness 已经给出的编号，不能自行创造编号。
"""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import InvestigationTask, WorkspaceTask


def is_evidence_id(value: object) -> bool:
    """判断输入是否为规范 Evidence 编号，例如 ``E1``，返回布尔值。"""

    # 先排除非字符串，以及没有 E 前缀的普通字符串。
    if not isinstance(value, str) or not value.startswith("E"):
        return False
    # E 后面必须是 ASCII 正整数；E0、E01、中文数字都不是合法编号。
    digits = value[1:]
    return bool(digits) and digits.isascii() and digits.isdigit() and digits[0] != "0"


def _require_positive_integer(value: int, field_name: str) -> None:
    """复用的参数检查：布尔值虽然是 int 子类，也不能当作编号。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def format_workspace_id(round_index: int, workspace_index: int) -> str:
    """把轮次和本轮序号组合成 ``W<round>-<index>`` 格式。"""

    # 两个数字都先经过相同校验，防止生成 W0-1 一类无效编号。
    _require_positive_integer(round_index, "round_index")
    _require_positive_integer(workspace_index, "workspace_index")
    return f"W{round_index}-{workspace_index}"


def assign_workspace_tasks(
    round_index: int, tasks: Iterable[InvestigationTask]
) -> tuple[WorkspaceTask, ...]:
    """按输入顺序为 GIA 给出的局部任务分配 Workspace 编号。

    ``enumerate(..., start=1)`` 让每轮第一个 Workspace 从 1 开始编号；返回
    tuple 表示这批任务一旦分配完成，就不应在调用方手中被原地修改。
    """

    _require_positive_integer(round_index, "round_index")
    return tuple(
        WorkspaceTask(
            task_id=format_workspace_id(round_index, workspace_index),
            task=task.task,
            materials=task.materials,
        )
        for workspace_index, task in enumerate(tasks, start=1)
    )
