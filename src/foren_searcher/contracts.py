"""GIA、Workspace、工具与 Harness 之间交换的核心数据契约。

“契约”可以理解为各模块共同遵守的数据格式。本文件只负责定义结构、做严格校验和
转换成普通 dict，不访问网络、数据库或模型。所有模型输出都先经过这里的解析函数，
因此字段多一个、少一个、类型不对或列表重复都会被拒绝，而不是悄悄猜测模型意思。

主要数据流：GIA 输出 InvestigationTask → Harness 分配 WorkspaceTask → HI 反复输出
HiActionDecision → HI 最后输出 WorkspaceReportDraft → Harness 保存为 WorkspaceReport。
Evidence 是原始/派生证据，Finding 是 Workspace 基于 Evidence 得出的判断，两者不能混用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias


def _require_nonempty_text(value: Any, field_name: str) -> str:
    """要求字段是去掉首尾空白后仍非空的字符串，并返回清理后的值。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, field_name: str) -> list[str]:
    """要求字段是字符串列表；逐项拒绝空字符串，并统一去掉首尾空白。"""

    # any(...) 只要发现一个非法元素就立即为真，避免后续构造半合法对象。
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _require_unique_string_list(value: Any, field_name: str) -> list[str]:
    """在字符串列表校验基础上，再保证没有重复项。"""

    result = _require_string_list(value, field_name)
    # set 会去重；长度变化就说明原列表至少有一个重复字符串。
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicate values")
    return result


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    """要求 JSON 对象字段与 expected 完全相等，不允许悄悄忽略多余字段。"""

    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields must be {sorted(expected)}; got {sorted(actual)}"
        )


@dataclass(frozen=True)
class EvidenceCard:
    """已提交 Evidence 的精简卡片，供模型浏览索引而不加载完整正文。

    ``label`` 用于快速判断相关性；``source_ref`` 指回来源；``derived_from`` 说明
    这条证据由哪些 E 派生；``acquired_by`` 记录由哪个工具和路径取得。
    来源引用定位入库记录；不同引用不代表真实出处彼此独立。
    """

    evidence_id: str
    modalities: tuple[str, ...]
    label: str
    source_ref: str | None
    derived_from: tuple[str, ...]
    acquired_by: str

    def to_dict(self) -> dict[str, Any]:
        """转成可写入 JSON 的普通字典；tuple 在这里转成 list。"""

        return {
            "evidence_id": self.evidence_id,
            "modalities": list(self.modalities),
            "label": self.label,
            "source_ref": self.source_ref,
            "derived_from": list(self.derived_from),
            "acquired_by": self.acquired_by,
        }


@dataclass(frozen=True, init=False)
class Evidence:
    """完整、可追溯的证据记录；“登记为 Evidence”不等于内容已经被判真。

    ``content`` 是可读文本或结构化描述；大文件由 ``artifact_ref`` 指向 Artifact。
    ``locator`` 说明证据位于原材料何处，两个 ``*_by`` 字段记录取得者和发现者。
    类使用 ``frozen=True``，创建后不能被原地改写，避免审计记录随意漂移。
    """

    evidence_id: str
    modalities: tuple[str, ...]
    label: str
    content: str
    source_ref: str | None
    derived_from: tuple[str, ...]
    artifact_ref: str | None
    locator: str
    acquired_by: str
    discovered_by: str
    production_method: str

    def __init__(
        self,
        *,
        evidence_id: str,
        modalities: tuple[str, ...] = ("text",),
        label: str | None = None,
        content: str,
        source_ref: str | None = None,
        derived_from: tuple[str, ...] = (),
        artifact_ref: str | None = None,
        locator: str,
        acquired_by: str,
        discovered_by: str,
        production_method: str = "source_excerpt",
    ) -> None:
        """验证必要字段，并用 object.__setattr__ 初始化冻结 dataclass。"""

        # Evidence 至少要有可读内容和取得方法，否则无法解释它是什么、从何而来。
        if not content:
            raise ValueError("content must be non-empty")
        if not acquired_by:
            raise ValueError("acquired_by must be non-empty")
        # frozen dataclass 禁止普通赋值；构造函数中必须使用 object.__setattr__。
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "modalities", tuple(modalities))
        # 调用方没给短标签时，用正文前 120 字符生成一个仅供浏览的标签。
        object.__setattr__(self, "label", label or content[:120])
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "derived_from", tuple(derived_from))
        object.__setattr__(self, "artifact_ref", artifact_ref)
        object.__setattr__(self, "locator", locator)
        object.__setattr__(self, "acquired_by", acquired_by)
        object.__setattr__(self, "discovered_by", discovered_by)
        object.__setattr__(self, "production_method", production_method)

    def card(self) -> EvidenceCard:
        """生成不含完整 content、artifact_ref 和 locator 的模型索引卡。"""

        return EvidenceCard(
            evidence_id=self.evidence_id,
            modalities=self.modalities,
            label=self.label,
            source_ref=self.source_ref,
            derived_from=self.derived_from,
            acquired_by=self.acquired_by,
        )

    def to_dict(self) -> dict[str, Any]:
        """导出包含全部审计字段的 JSON 兼容字典。"""

        return {
            "evidence_id": self.evidence_id,
            "modalities": list(self.modalities),
            "label": self.label,
            "content": self.content,
            "source_ref": self.source_ref,
            "derived_from": list(self.derived_from),
            "artifact_ref": self.artifact_ref,
            "locator": self.locator,
            "acquired_by": self.acquired_by,
            "discovered_by": self.discovered_by,
            "production_method": self.production_method,
        }


@dataclass(frozen=True)
class Finding:
    """Workspace 基于 Evidence 得出的判断；它本身不是原始证据。

    ``evidence_ids`` 是判断依据，``limitations`` 写明尚不能证明的范围。Finding 编号
    由 Harness 在保存报告时分配，``workspace_id`` 指明是谁得出的判断。
    """

    finding_id: str
    workspace_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """把 dataclass 导出，并把两个 tuple 显式转为 JSON 数组。"""

        # asdict 先递归复制字段，后两行统一公共 JSON 形态。
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        value["limitations"] = list(self.limitations)
        return value


@dataclass(frozen=True)
class FindingDraft:
    """HI 写出的 Finding 草稿；此时 Harness 还没有为它分配 F 编号。"""

    statement: str
    evidence_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换成报告 JSON 中单条 finding 的结构。"""

        return {
            "statement": self.statement,
            "evidence_ids": list(self.evidence_ids),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class InvestigationTask:
    """GIA 在 INVESTIGATE 决定中提出的一项局部调查任务。

    ``materials`` 只能引用当前全局视图里已经存在的 C/E 编号。
    """

    task: str
    materials: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """导出模型/Harness 共同使用的 task 与 materials 两字段对象。"""

        return {"task": self.task, "materials": list(self.materials)}


@dataclass(frozen=True)
class WorkspaceTask:
    """Harness 为 InvestigationTask 分配 ``W轮次-序号`` 后的可执行任务。"""

    task_id: str
    task: str
    materials: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """导出含 task_id 的 Workspace 输入。"""

        return {
            "task_id": self.task_id,
            "task": self.task,
            "materials": list(self.materials),
        }


# Workspace 只能明确完成或明确证据不足，不使用含糊的第三种状态。
WorkspaceStatus: TypeAlias = Literal["COMPLETED", "INCOMPLETE"]


@dataclass(frozen=True)
class WorkspaceReport:
    """已由 Harness 验证、分配 F 编号并保存的正式 Workspace 报告。"""

    task_id: str
    status: WorkspaceStatus
    findings: tuple[Finding, ...]
    selected_evidence_ids: tuple[str, ...]
    status_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """导出报告；只有确实存在 status_reason 时才写入该可选字段。"""

        value = {
            "task_id": self.task_id,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "selected_evidence_ids": list(self.selected_evidence_ids),
        }
        if self.status_reason is not None:
            value["status_reason"] = self.status_reason
        return value


@dataclass(frozen=True)
class WorkspaceReportDraft:
    """HI 的终止输出；Finding 仍是草稿，等待 Harness 校验和编号。"""

    task_id: str
    status: WorkspaceStatus
    findings: tuple[FindingDraft, ...]
    selected_evidence_ids: tuple[str, ...]
    status_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """导出模型直接返回的报告结构。"""

        value: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "selected_evidence_ids": list(self.selected_evidence_ids),
        }
        if self.status_reason is not None:
            value["status_reason"] = self.status_reason
        return value


@dataclass(frozen=True)
class HiActionDecision:
    """HI 的一次非终止决策：调用一个动作，并更新完整的局部关注证据集合。"""

    action: str
    arguments: dict[str, Any]
    focus_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """导出 action、arguments、focus_evidence_ids 三个严格字段。"""

        return {
            "action": self.action,
            "arguments": self.arguments,
            "focus_evidence_ids": list(self.focus_evidence_ids),
        }


# HI 每轮要么继续行动，要么提交报告结束，不存在第三种顶层输出。
HiDecision: TypeAlias = HiActionDecision | WorkspaceReportDraft


def _parse_finding_draft(value: Any, index: int) -> FindingDraft:
    """解析报告中的第 index 条 finding，并在错误中带上位置方便定位。"""

    if not isinstance(value, dict):
        raise ValueError(f"finding {index} must be an object")
    _require_exact_keys(
        value,
        {"statement", "evidence_ids", "limitations"},
        f"finding {index}",
    )
    # 三个字段分别经过专门校验，再组装成不可变草稿对象。
    return FindingDraft(
        statement=_require_nonempty_text(value["statement"], f"finding {index}.statement"),
        evidence_ids=tuple(
            _require_unique_string_list(
                value["evidence_ids"], f"finding {index}.evidence_ids"
            )
        ),
        limitations=tuple(
            _require_unique_string_list(
                value["limitations"], f"finding {index}.limitations"
            )
        ),
    )


def parse_workspace_report(value: Any) -> WorkspaceReportDraft:
    """严格解析 HI 直接提交的 Workspace Report JSON。"""

    # 顶层必须是对象，status 决定允许出现哪些字段。
    if not isinstance(value, dict):
        raise ValueError("Workspace Report must be an object")
    status = value.get("status")
    if status not in {"COMPLETED", "INCOMPLETE"}:
        raise ValueError("Workspace Report status must be COMPLETED or INCOMPLETE")
    expected = {"task_id", "status", "findings", "selected_evidence_ids"}
    # 只有 INCOMPLETE 可以解释“为什么没完成”；COMPLETED 出现该字段会被严格拒绝。
    if status == "INCOMPLETE" and "status_reason" in value:
        expected.add("status_reason")
    _require_exact_keys(value, expected, "Workspace Report")
    raw_findings = value["findings"]
    if not isinstance(raw_findings, list):
        raise ValueError("Workspace Report findings must be a list")
    # enumerate 从 1 开始，使报错位置和人阅读的“第 1 条”一致。
    findings = tuple(
        _parse_finding_draft(item, index)
        for index, item in enumerate(raw_findings, start=1)
    )
    for index, finding in enumerate(findings, start=1):
        # 判断必须有证据依据；空 evidence_ids 不能进入正式报告。
        if not finding.evidence_ids:
            raise ValueError(f"finding {index}.evidence_ids must not be empty")
    status_reason = value.get("status_reason")
    if status_reason is not None:
        status_reason = _require_nonempty_text(status_reason, "status_reason")
    selected_evidence_ids = tuple(
        _require_unique_string_list(
            value["selected_evidence_ids"], "selected_evidence_ids"
        )
    )
    # 报告引用的每条 Evidence 也必须进入 selected 集合，便于后续统一审计和汇总。
    selected = set(selected_evidence_ids)
    missing = sorted(
        {
            evidence_id
            for finding in findings
            for evidence_id in finding.evidence_ids
            if evidence_id not in selected
        }
    )
    if missing:
        raise ValueError(
            "Finding Evidence IDs must be included in selected_evidence_ids: "
            + ", ".join(missing)
        )
    # 所有交叉字段检查通过后，才构造草稿对象返回给 Harness。
    return WorkspaceReportDraft(
        task_id=_require_nonempty_text(value["task_id"], "task_id"),
        status=status,
        findings=findings,
        selected_evidence_ids=selected_evidence_ids,
        status_reason=status_reason,
    )


def parse_hi_decision(value: Any) -> HiDecision:
    """区分并解析 HI 的“继续行动”或“提交报告”两种输出。"""

    if not isinstance(value, dict):
        raise ValueError("HI output must be an object")
    if "action" not in value:
        # 没有 action 字段时，按终止报告走完整报告校验。
        return parse_workspace_report(value)
    _require_exact_keys(
        value,
        {"action", "arguments", "focus_evidence_ids"},
        "HI action",
    )
    # 动作名统一大写，使模型输出 search 与 SEARCH 得到相同行为。
    action = _require_nonempty_text(value["action"], "action").upper()
    arguments = value["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("HI action arguments must be an object")
    return HiActionDecision(
        action=action,
        arguments=arguments,
        focus_evidence_ids=tuple(
            _require_unique_string_list(
                value["focus_evidence_ids"], "focus_evidence_ids"
            )
        ),
    )


@dataclass(frozen=True)
class GiaInvestigateDecision:
    """GIA 决定开启新一轮调查时的结构化结果。"""

    decision: Literal["INVESTIGATE"]
    tasks: tuple[InvestigationTask, ...]

    def to_dict(self) -> dict[str, Any]:
        """把所有 InvestigationTask 逐个导出为 JSON 数组。"""

        return {
            "decision": self.decision,
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class GiaFinishDecision:
    """GIA 依据已完成报告给出最终答案时的结构化结果。"""

    decision: Literal["FINISH"]
    answer: str

    def to_dict(self) -> dict[str, Any]:
        """导出最小的 decision + answer 对象。"""

        return {"decision": self.decision, "answer": self.answer}


# 顶层 GIA 只能继续调查或完成答案。
GiaDecision: TypeAlias = GiaInvestigateDecision | GiaFinishDecision


def parse_gia_decision(value: Any) -> GiaDecision:
    """严格解析 GIA 的 INVESTIGATE 或 FINISH 输出。"""

    if not isinstance(value, dict):
        raise ValueError("GIA decision must be an object")

    decision = value.get("decision")
    if decision == "INVESTIGATE":
        # INVESTIGATE 只能包含 decision 和非空 tasks，禁止夹带额外控制字段。
        _require_exact_keys(value, {"decision", "tasks"}, "INVESTIGATE")
        raw_tasks = value["tasks"]
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("INVESTIGATE tasks must be a non-empty list")

        tasks: list[InvestigationTask] = []
        for index, raw_task in enumerate(raw_tasks, start=1):
            # 每个局部任务也使用精确字段检查，随后清理 task 文本和 materials。
            if not isinstance(raw_task, dict):
                raise ValueError(f"task {index} must be an object")
            _require_exact_keys(raw_task, {"task", "materials"}, f"task {index}")
            tasks.append(
                InvestigationTask(
                    task=_require_nonempty_text(raw_task["task"], f"task {index}.task"),
                    materials=tuple(
                        _require_unique_string_list(
                            raw_task["materials"], f"task {index}.materials"
                        )
                    ),
                )
            )
        return GiaInvestigateDecision(decision="INVESTIGATE", tasks=tuple(tasks))

    if decision == "FINISH":
        # FINISH 保持最小结构，最终答案必须是非空文本。
        _require_exact_keys(value, {"decision", "answer"}, "FINISH")
        return GiaFinishDecision(
            decision="FINISH",
            answer=_require_nonempty_text(value["answer"], "answer"),
        )

    raise ValueError("decision must be INVESTIGATE or FINISH")
