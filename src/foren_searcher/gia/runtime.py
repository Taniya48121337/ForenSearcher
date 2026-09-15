"""InvestigationRuntime：在轮次之间串行推进整个调查，负责初始化、恢复、限制和关闭。

从原 orchestration.py 拆出（2026-09-15 拆包）。
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..contracts import Evidence, GiaDecision, GiaFinishDecision, GiaInvestigateDecision, WorkspaceTask, parse_gia_decision
from ..hi.workspace import CustomActionHandler, WorkspaceHarness
from ..ids import assign_workspace_tasks, is_evidence_id
from ..inspect.service import InspectService
from ..policies import GiaPolicy, HiPolicy
from ..search.service import SearchService
from ..store.global_store import GlobalStore
from ..store.records import MemoryPolicy
from ._common import _CLAIM_REFERENCE, _EVIDENCE_LIKE_REFERENCE, _EVIDENCE_REFERENCE, _LOGGER, _record_runtime_event
from .global_view import GlobalViewBuilder
from .rounds import RoundCoordinator, WorkspaceOutcome
from .runner import GiaRunner

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


class InvestigationLimit(RuntimeError):
    """调查试图开启超过 ``max_rounds`` 的新轮次。"""

    pass


class RuntimeClosedError(RuntimeError):
    """调用方在 Runtime 已关闭或正在关闭时提交新工作。"""

    pass


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

