"""【已弃用，2026-09-15】把私有分叉的 Harness-1 episode 壳接到调查工作区上。

这层只做透传：真正的工具执行、证据登记、报告校验全在 ``hi/workspace.py`` 的
``WorkspaceHarness`` 里。它导入的 ``EpisodeEnv`` / ``ToolCatalog`` 等类来自本机私有分叉，
公开的 Harness-1 仓库里并不存在。设计决定（DESIGN.md 第 7 节）是从运行时剥离；
本模块仅为兼容旧命令行参数保留，调用时会发出 DeprecationWarning，之后归档删除。
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

from .contracts import HiActionDecision, WorkspaceReport, WorkspaceReportDraft
from .hi.workspace import ToolObservation, WorkspaceHarness


def load_harness1_runtime(root: str | Path) -> ModuleType:
    """从用户指定的 Harness-1 源码目录加载纯运行时包。"""

    directory = Path(root).resolve() / "harness" / "runtime"
    entry = directory / "__init__.py"
    if not entry.is_file():
        raise ValueError(f"Harness-1 runtime package is missing: {entry}")
    name = "foren_searcher._harness1_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(str(existing.__file__)).resolve() != entry:
            raise ValueError("a different Harness-1 runtime is already loaded in this process")
        return existing
    spec = importlib.util.spec_from_file_location(name, entry, submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Harness-1 from {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise
    return module


def harness1_workspace_type(root: str | Path) -> type[WorkspaceHarness]:
    """每个工作区使用独立 EpisodeEnv、工具执行器和任务状态。"""

    warnings.warn("harness1_workspace_type is deprecated; WorkspaceHarness runs without Harness-1 (DESIGN.md §7)",
                  DeprecationWarning, stacklevel=2)
    runtime: Any = load_harness1_runtime(root)

    class Harness1Workspace(WorkspaceHarness):
        episode: Any = None
        _decision: HiActionDecision | None = None
        _report_draft: WorkspaceReportDraft | None = None

        def start_episode(self, max_turns: int) -> None:
            catalog = runtime.ToolCatalog()
            for action in self.eligible_actions():
                schema = action.get("schema", {})

                async def handle(arguments: Any, context: Any) -> ToolObservation:
                    assert self._decision is not None
                    return await super(Harness1Workspace, self).execute_action(
                        HiActionDecision(self._decision.action, dict(arguments), self._decision.focus_evidence_ids))

                catalog.register(runtime.RuntimeToolBinding(
                    schema=runtime.RuntimeToolSchema(name=action["name"],
                        description=action.get("description") or action["name"],
                        parameters=schema.get("properties", {}), required=tuple(schema.get("required", ())),
                        schema_keywords={key: value for key, value in schema.items() if key not in {"type", "properties", "required"}}),
                    handler=handle))
            # 终止映射仅供适配器调用；HI 仍直接输出报告，不新增模型动作。
            catalog.register(runtime.RuntimeToolBinding(
                schema=runtime.RuntimeToolSchema(name="workspace_report", description="Workspace terminal output", parameters={}),
                termination=True, advertised=False))

            def terminal_output(state: Any, partial: bool) -> WorkspaceReport | None:
                if self._report_draft is None:
                    return None
                return self.finish(self._report_draft)

            adapter = runtime.CallbackTaskRuntimeAdapter(
                initialize=lambda task: self.memory,
                prompt=lambda state: state.task,
                terminal_output=terminal_output,
                decide_termination=lambda state, request: (
                    runtime.TerminationDecision.force_partial("Workspace reported incomplete")
                    if self._report_draft is not None and self._report_draft.status == "INCOMPLETE"
                    else runtime.TerminationDecision.accept()))
            self.episode = runtime.EpisodeEnv(
                runtime_spec=runtime.RuntimeSpec(name="ForenSearcher", catalog=catalog),
                task_adapter=adapter,
                budget=runtime.EpisodeBudget(max_turns=max_turns, max_tool_calls=max_turns))
            self.episode.start(self.memory.task, episode_id=self.memory.task_id)

        async def _episode_step(self, calls: tuple[Any, ...]) -> Any:
            try:
                return await self.episode.step(calls)
            except runtime.ToolBatchCancelled as exc:
                # Python 3.11 的 asyncio.timeout 只转换原生 CancelledError；批次详情保留在原因链中。
                raise asyncio.CancelledError() from exc

        async def execute_action(self, decision: HiActionDecision) -> ToolObservation:
            if decision.action not in {item["name"] for item in self.eligible_actions()}:
                raise ValueError(f"HI action is not currently available: {decision.action}")
            if self.episode is None:
                self.start_episode(32)
            assert self.episode is not None
            self._decision = decision
            step = await self._episode_step((runtime.RuntimeToolCall(
                name=decision.action, arguments=decision.arguments),))
            if not step.results:
                raise ValueError(f"Harness-1 episode ended: {step.termination_reason}")
            result = step.results[0]
            if not result.ok:
                raise ValueError(f"Harness-1 {result.error.code}: {result.error.message}")
            if not isinstance(result.output, ToolObservation):
                raise TypeError("Harness-1 handler returned an invalid observation")
            return result.output

        async def submit_report(self, draft: WorkspaceReportDraft) -> WorkspaceReport:
            self.validate_report(draft)
            if self.episode is None:
                self.start_episode(32)
            assert self.episode is not None
            self._report_draft = draft
            step = await self._episode_step((runtime.RuntimeToolCall(name="workspace_report", arguments={}),))
            report = self.episode.state.output
            if not isinstance(report, WorkspaceReport):
                self._report_draft = None
                raise ValueError(f"Harness-1 could not finish Workspace: {step.termination_reason}")
            return report

    return Harness1Workspace
