"""错误类型清单：自我发展循环的通用语言（DESIGN.md 第 8 节）。

核验器给每条 Finding / 每个动作 / 每个状态判定时只能用这里的类型；诊断按类型计数；
对照轨迹生成器按类型出错题；rubric 条目按类型绑定。**这是唯一的定义处**，其它模块只
import，不自己写字符串。

大白话：这是"错题分类表"。审计员批案卷时不能写"这里不太对"，必须从表里挑一个名字，
这样才能统计"哪类错最多"，也才能对症下药。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    HI = "HI"
    GIA = "GIA"
    ENV = "ENV"


class Level(str, Enum):
    """错误发生在哪一层。"""

    FINDING = "finding"      # 报告里的某条 Finding，或 GIA 最终答案里的某句话
    STATUS = "status"        # 报告的 COMPLETED / INCOMPLETE 状态
    ACTION = "action"        # HI 的某一步工具动作
    PLANNING = "planning"    # GIA 的派任务 / 结束决定
    ENVIRONMENT = "environment"  # 不是演员的错，但要记下来


class Severity(str, Enum):
    CRITICAL = "critical"    # 一票否决：结论不可信
    MAJOR = "major"          # 结论可能仍对，但过程有实质缺陷
    MINOR = "minor"          # 效率或表述问题


class ErrorType(str, Enum):
    """26 种错误（23 + 2026-09-21 从 true:4124903 示例补的 incomplete_citation_set、overlapping_sibling_tasks，+ 接核验模型判 plan 单元时补的 premise_smuggling；早期文档写 22 是漏数了 empty_report）。值是写进 verdicts.jsonl 的字符串，改名等于改数据格式，谨慎。"""

    # ---- Finding 级：说了什么、引了什么 ----
    UNSUPPORTED_ASSERTION = "unsupported_assertion"      # 断言没有任何所引 E 支持
    WRONG_CITATION = "wrong_citation"                    # 引用的 E 存在，但内容不是这回事
    FABRICATED_EVIDENCE_ID = "fabricated_evidence_id"    # 引用的 E 不存在，或当时对它不可见
    SOURCE_RESTRICTION_MISS = "source_restriction_miss"  # 用了被排除的来源 / 违反日期或来源条件
    ATTRIBUTION_DRIFT = "attribution_drift"              # "来源声称 X" 写成 "X 成立"；申请/批准/执行阶段互换
    SCOPE_OVERREACH = "scope_overreach"                  # 几帧推全片、采样推连续区间、局部推全局
    MISREAD_FINE_TEXT = "misread_fine_text"              # 图/页面里的文字、数字、日期读错
    SLOPPY_NUMERIC = "sloppy_numeric"                    # 口径/单位/分母/年份不一致却直接比较
    OMITTED_REQUIRED_FACT = "omitted_required_fact"      # 任务要求、材料里有，却没写进报告
    INCOMPLETE_CITATION_SET = "incomplete_citation_set"  # 断言的支持存在于本工作区可见且已被引用的其它 E 里，只是这条 Finding 没引它
    # ---- 状态级 ----
    COMPLETED_WITHOUT_FINISHING = "completed_without_finishing"  # 没查完却报 COMPLETED
    INCOMPLETE_THOUGH_ANSWERABLE = "incomplete_though_answerable"  # 材料够却报 INCOMPLETE
    EMPTY_REPORT = "empty_report"                        # 一条 Finding 都没有（且不属于合理的 INCOMPLETE）
    # ---- 动作级 ----
    ILLEGAL_ACTION = "illegal_action"                    # 接口校验失败 / 参数不合法 / 目标不可见
    REDUNDANT_ACTION = "redundant_action"                # 与历史重复且没带来新信息
    EVIDENCE_DROPPED = "evidence_dropped"                # focus 换掉了后续必需的对照材料
    WRONG_TOOL = "wrong_tool"                            # 用 SEARCH 读已有 R、用取帧看已有帧等
    IGNORED_FEEDBACK = "ignored_feedback"                # 工具报错后原样重复
    # ---- GIA 级 ----
    TASK_NOT_GAP_DRIVEN = "task_not_gap_driven"          # 派的任务不对应未解决的缺口
    PREMATURE_FINISH = "premature_finish"                # 证据不够就结束
    UNNECESSARY_ROUND = "unnecessary_round"              # 证据已够还开新一轮
    DEPENDENT_SIBLING_TASKS = "dependent_sibling_tasks"  # 同轮任务互相依赖，违反 DIR 规则
    OVERLAPPING_SIBLING_TASKS = "overlapping_sibling_tasks"  # 同轮任务目标重叠，导致重复取证（不是 HI 的 redundant_action）
    PREMISE_SMUGGLING = "premise_smuggling"              # 派任务的文字把尚未核实的说法当成既定事实写给 HI（前提走私）
    # ---- 环境级 ----
    ENV_CONTAMINATION = "env_contamination"              # 搜索返回了本应排除的来源
    ENV_TOOL_FAILURE = "env_tool_failure"                # 工具本身失败（超时、解码错误）


@dataclass(frozen=True)
class ErrorSpec:
    """每种错误的元信息：层级、适用角色、默认严重度、是否向后传播、能否程序核验。"""

    level: Level
    roles: tuple[Role, ...]
    default_severity: Severity
    propagates: bool          # 为真时，之后所有因果依赖此步的步骤都视为受污染（AgentProcessBench 的传播规则）
    exec_checkable: bool      # 为真时第一层程序核验就能判，不需要模型
    description: str


SPECS: dict[ErrorType, ErrorSpec] = {
    ErrorType.UNSUPPORTED_ASSERTION: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.CRITICAL, True, False,
        "Finding 的陈述在所引 E 的原文/像素里找不到支持"),
    ErrorType.WRONG_CITATION: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.CRITICAL, True, False,
        "所引 E 存在但讲的是别的事，或该内容在另一条 E 里"),
    ErrorType.FABRICATED_EVIDENCE_ID: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.CRITICAL, True, True,
        "引用了不存在的 E，或该 E 在该步对该角色不可见"),
    ErrorType.SOURCE_RESTRICTION_MISS: ErrorSpec(Level.FINDING, (Role.HI,), Severity.CRITICAL, True, True,
        "引用来源命中排除表，或早于/晚于允许的日期范围"),
    ErrorType.ATTRIBUTION_DRIFT: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.MAJOR, True, False,
        "把来源的说法当作事实，或把申请/批准/执行/发生等状态互换"),
    ErrorType.SCOPE_OVERREACH: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.MAJOR, True, False,
        "结论覆盖的范围大于实际检查的范围"),
    ErrorType.MISREAD_FINE_TEXT: ErrorSpec(Level.FINDING, (Role.HI,), Severity.MAJOR, True, False,
        "对图片/页面中的细小文字、数字、日期识别错误（可半程序核验：OCR 放大比对）"),
    ErrorType.SLOPPY_NUMERIC: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.MAJOR, True, True,
        "比较的数值口径、单位、分母或时间范围不一致"),
    ErrorType.OMITTED_REQUIRED_FACT: ErrorSpec(Level.FINDING, (Role.HI,), Severity.MAJOR, False, False,
        "任务明确要求且材料中可得的事实没有出现在报告里"),
    ErrorType.INCOMPLETE_CITATION_SET: ErrorSpec(Level.FINDING, (Role.HI, Role.GIA), Severity.MAJOR, False, False,
        "比较句/关系句只引了一边：支持它的 E 在本工作区可见且被同报告其它 Finding 引用，本条未引（校准规则 v0.2.0 从 unsupported_assertion 降级而来）"),
    ErrorType.COMPLETED_WITHOUT_FINISHING: ErrorSpec(Level.STATUS, (Role.HI,), Severity.CRITICAL, False, True,
        "任务要求项未全部覆盖却报 COMPLETED"),
    ErrorType.INCOMPLETE_THOUGH_ANSWERABLE: ErrorSpec(Level.STATUS, (Role.HI,), Severity.MINOR, False, False,
        "可见材料已足以回答却报 INCOMPLETE"),
    ErrorType.EMPTY_REPORT: ErrorSpec(Level.STATUS, (Role.HI,), Severity.MAJOR, False, True,
        "报告没有任何 Finding，且 status_reason 未说明为何无法取得材料"),
    ErrorType.ILLEGAL_ACTION: ErrorSpec(Level.ACTION, (Role.HI, Role.GIA), Severity.MINOR, False, True,
        "被 Harness 拒绝的动作"),
    ErrorType.REDUNDANT_ACTION: ErrorSpec(Level.ACTION, (Role.HI,), Severity.MINOR, False, True,
        "与历史中某步参数相同且未产生新 E"),
    ErrorType.EVIDENCE_DROPPED: ErrorSpec(Level.ACTION, (Role.HI,), Severity.MAJOR, True, True,
        "focus_evidence_ids 丢掉了后续步骤仍需对照的 E"),
    ErrorType.WRONG_TOOL: ErrorSpec(Level.ACTION, (Role.HI,), Severity.MINOR, False, False,
        "工具选择与目标不匹配"),
    ErrorType.IGNORED_FEEDBACK: ErrorSpec(Level.ACTION, (Role.HI,), Severity.MINOR, False, True,
        "上一步工具返回 failed/feedback 后，原样重复同一请求"),
    ErrorType.TASK_NOT_GAP_DRIVEN: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.MAJOR, False, False,
        "派出的任务不对应尚缺的必需证据"),
    ErrorType.PREMATURE_FINISH: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.CRITICAL, False, False,
        "必需证据尚缺却 FINISH"),
    ErrorType.UNNECESSARY_ROUND: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.MINOR, False, True,
        "必需证据已齐却继续 INVESTIGATE"),
    ErrorType.DEPENDENT_SIBLING_TASKS: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.MAJOR, False, False,
        "同一轮的任务之间存在依赖，违反轮内并行规则"),
    ErrorType.OVERLAPPING_SIBLING_TASKS: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.MINOR, False, True,
        "同轮两个任务对准同一要求，各自的工作区用同一批必需证据写了同类 Finding（派工效率问题，按 hindsight 程序判）"),
    ErrorType.PREMISE_SMUGGLING: ErrorSpec(Level.PLANNING, (Role.GIA,), Severity.MAJOR, True, False,
        "任务文字把案卷没给、此前也没核实的说法当成既定事实（会把 HI 的取证方向带偏，按传播处理；需核验模型读任务文字判）"),
    ErrorType.ENV_CONTAMINATION: ErrorSpec(Level.ENVIRONMENT, (Role.ENV,), Severity.MAJOR, True, True,
        "搜索结果含应被排除的来源（不算演员错，但会污染其后的步骤）"),
    ErrorType.ENV_TOOL_FAILURE: ErrorSpec(Level.ENVIRONMENT, (Role.ENV,), Severity.MINOR, False, True,
        "工具执行失败"),
}

assert set(SPECS) == set(ErrorType), "每个 ErrorType 都必须有 ErrorSpec"


@dataclass(frozen=True)
class ErrorRecord:
    """一次具体的错误判定。写进 verdicts.jsonl 的 errors[] 里。"""

    type: ErrorType
    severity: Severity
    detail: str
    span: str | None = None                  # 出错的那句话 / 那个字段（原文摘录）
    evidence_ids: tuple[str, ...] = ()       # 涉及的 E
    method: str = "model"                    # exec | model | pixels | human | injected
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, ErrorType):
            raise TypeError("type must be an ErrorType")
        if not isinstance(self.severity, Severity):
            raise TypeError("severity must be a Severity")
        if not self.detail.strip():
            raise ValueError("detail must be non-empty")
        if self.method not in {"exec", "model", "pixels", "human", "injected"}:
            raise ValueError(f"unknown method: {self.method}")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @property
    def spec(self) -> ErrorSpec:
        return SPECS[self.type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value, "level": self.spec.level.value, "severity": self.severity.value,
            "detail": self.detail, "span": self.span, "evidence_ids": list(self.evidence_ids),
            "method": self.method, "propagates": self.spec.propagates, **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ErrorRecord":
        return cls(
            type=ErrorType(value["type"]), severity=Severity(value["severity"]), detail=value["detail"],
            span=value.get("span"), evidence_ids=tuple(value.get("evidence_ids", ())),
            method=value.get("method", "model"), extra=dict(value.get("extra", {})),
        )


def error(type_: ErrorType, detail: str, **kwargs: Any) -> ErrorRecord:
    """用默认严重度快速构造一条错误记录。"""

    severity = kwargs.pop("severity", SPECS[type_].default_severity)
    return ErrorRecord(type=type_, severity=severity, detail=detail, **kwargs)


def types_for(role: Role, *, level: Level | None = None, exec_checkable: bool | None = None) -> tuple[ErrorType, ...]:
    """按角色/层级/是否可程序核验筛选错误类型。"""

    result = []
    for kind, spec in SPECS.items():
        if role not in spec.roles:
            continue
        if level is not None and spec.level != level:
            continue
        if exec_checkable is not None and spec.exec_checkable != exec_checkable:
            continue
        result.append(kind)
    return tuple(result)


def tally(errors: list[ErrorRecord] | tuple[ErrorRecord, ...]) -> dict[str, int]:
    """按类型计数，键固定为全部类型（没出现的为 0），方便跨迭代画曲线。"""

    counts = {kind.value: 0 for kind in ErrorType}
    for item in errors:
        counts[item.type.value] += 1
    return counts


__all__ = ["ErrorRecord", "ErrorSpec", "ErrorType", "Level", "Role", "SPECS", "Severity", "error", "tally", "types_for"]
