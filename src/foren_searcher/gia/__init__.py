"""GIA（Global Investigation Agent）一侧：全局视图、单次决策、轮次调度、整体运行时。

- global_view.py：GIA 每轮看到的 Global View 及其预算裁剪
- runner.py：GiaRunner，驱动一次 GIA 决策
- rounds.py：RoundCoordinator，并发跑一轮 Workspace 并收报告（轮内并行、轮间串行）
- runtime.py：InvestigationRuntime，整个调查的入口：初始化、逐轮推进、恢复、关闭
GIA 只决定"下一轮查什么"和"证据够不够"，不碰工具。
"""

from .global_view import GlobalView, GlobalViewBuilder, bound_global_view
from .rounds import RoundCoordinator, WorkspaceOutcome
from .runner import GiaPolicyTimeout, GiaRunner, GiaTurnLimit
from .runtime import InitialMedia, InvestigationLimit, InvestigationResult, InvestigationRuntime, RuntimeClosedError

__all__ = [
    "GiaPolicyTimeout", "GiaRunner", "GiaTurnLimit", "GlobalView", "GlobalViewBuilder", "InitialMedia",
    "InvestigationLimit", "InvestigationResult", "InvestigationRuntime", "RoundCoordinator", "RuntimeClosedError",
    "WorkspaceOutcome", "bound_global_view",
]
