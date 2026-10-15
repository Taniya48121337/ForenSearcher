"""规格 harness_params → 运行时配置的显式映射（让内环改的参数真的生效）。

为什么要有：specs/<hi|gia>@x.y.z/spec.json 的 harness_params 以前只被记进 run_config，run_cases 实际用的是
configs/*.toml 的 [runtime] 和各 Policy 的内置默认值。内环按 rubric 改了 harness_params、发了新版本规格，
跑出来的数据却还是旧参数——版本号和行为对不上，这比没有版本号更糟。

这里做三件事：
1. ``HARNESS_PARAM_MAP``：每个键对应运行时哪个对象的哪个字段、类型、取值范围、归属角色、是否已接线。
2. ``resolve_harness_params``：从 HI / GIA 两份规格取值。未知键直接报错；类型、范围不对报错；两份规格对
   "共享"键给了不同的值也报错——运行时只有一份 MemoryPolicy / InspectPolicy，不能静默挑一个。
   只属于一个角色的键（max_workspace_turns 属于 HI）取该角色规格的值，另一份规格里的值记为 not_applicable。
3. ``build_harness_settings``：在 load_config() 的结果上覆盖，产出 RuntimeConfig / MemoryPolicy / InspectPolicy，
   并给出 run_config 要记的 ``harness_params_applied``（键 → 值 → 来源规格）与"相对默认配置的差异"。

默认配置指：configs/*.toml 的 [runtime]（没有配置文件时是内置默认）+ MemoryPolicy() + InspectPolicy()——也就是
接线之前 run_cases 实际使用的那一套。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .specs import DEFAULT_HARNESS_PARAMS, Spec

Owner = Literal["shared", "hi", "gia"]
HarnessMode = Literal["apply", "record"]
HARNESS_MODES: tuple[str, ...] = ("apply", "record")


class HarnessParamError(ValueError):
    """harness_params 有未知键、类型或范围不对、两份规格冲突。"""


@dataclass(frozen=True)
class ParamBinding:
    key: str
    target: str                         # 运行时位置，形如 memory_policy.max_read_chars
    kind: type
    owner: Owner
    wired: bool
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    note: str = ""

    def validate(self, value: Any, source: str) -> Any:
        if self.kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise HarnessParamError(f"{source}.harness_params.{self.key} 应为整数，得到 {value!r}")
        elif self.kind is str:
            if not isinstance(value, str):
                raise HarnessParamError(f"{source}.harness_params.{self.key} 应为字符串，得到 {value!r}")
        elif self.kind is bool:
            if not isinstance(value, bool):
                raise HarnessParamError(f"{source}.harness_params.{self.key} 应为 true / false，得到 {value!r}")
            return value
        if self.choices and value not in self.choices:
            raise HarnessParamError(f"{source}.harness_params.{self.key} 只能是 {' / '.join(self.choices)}，得到 {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise HarnessParamError(f"{source}.harness_params.{self.key} 必须是有限数，得到 {value!r}")
            if self.minimum is not None and value < self.minimum:
                raise HarnessParamError(f"{source}.harness_params.{self.key}={value} 小于下限 {self.minimum:g}")
            if self.maximum is not None and value > self.maximum:
                raise HarnessParamError(f"{source}.harness_params.{self.key}={value} 大于上限 {self.maximum:g}")
        return value


def _renderer_tiers() -> tuple[str, ...]:
    from ..state_render import TIERS

    return tuple(TIERS)


# 范围取"工程上合理"的区间：下限防止把视图压到装不下任务本身，上限防止内环把上下文调到模型装不下。
HARNESS_PARAM_MAP: dict[str, ParamBinding] = {
    "image_input_limit": ParamBinding(
        "image_input_limit", "inspect_policy.maximum_visual_qa_images", int, "shared", True, 1, 64,
        note="同一个 InspectPolicy 同时给 INSPECT 服务（visual_qa 输入上限、HI 动作 schema 的 targets.maxItems）"
             "和教师的附图上限（ModelImageInputs.image_policy，两个角色共用一个教师实例）"),
    "max_workspace_turns": ParamBinding(
        "max_workspace_turns", "runtime.max_workspace_turns", int, "hi", True, 1, 200,
        note="InvestigationRuntime(max_workspace_turns=…)；只约束 HI 的 Workspace，GIA 规格里的同名值不适用"),
    "read_page_chars": ParamBinding(
        "read_page_chars", "memory_policy.max_read_chars", int, "shared", True, 200, 1_000_000,
        note="READ_RECORD / 全局视图里 Evidence 分页的每页字符数"),
    "max_model_view_chars": ParamBinding(
        "max_model_view_chars", "memory_policy.max_model_view_chars", int, "shared", True, 4_000, 4_000_000,
        note="GIA 全局视图与 HI 输入的总字符上限"),
    "max_evidence_cards": ParamBinding(
        "max_evidence_cards", "memory_policy.max_evidence_cards", int, "shared", True, 1, 1_000),
    "max_history_entries": ParamBinding(
        "max_history_entries", "memory_policy.max_history_entries", int, "shared", True, 1, 1_000,
        note="10-13：HI 工作区 / GIA 视图里保留的历史条数（默认 MemoryPolicy 的 32）；只用于环境变体（walker V9："
             "max_evidence_cards=6 + max_history_entries=4，早先的结果掉出视图，要 QUERY_MEMORY 查回来），规格里不写"),
    "max_candidate_cards": ParamBinding(
        "max_candidate_cards", "memory_policy.max_candidate_cards", int, "shared", True, 1, 1_000),
    "search_excerpt_chars": ParamBinding(
        "search_excerpt_chars", "memory_policy.search_excerpt_chars", int, "hi", True, 0, 4000,
        note="09-30：SEARCH 自动打开网页后，观察里附每页与查询最相关的一段（字符数）；0 = 不附。只作用于 HI 的 SEARCH 观察"),
    "state_renderer": ParamBinding(
        "state_renderer", "memory_policy.state_renderer", str, "shared", True, choices=(),
        note="部分生效：写入 MemoryPolicy → runtime_constraints.state_renderer，本地学生（local_models）与导出据此渲染；"
             "EndpointTeacher / Codex 教师不按档位渲染输入（教师看的始终是 legacy 全量视图）"),
    "inspect_diff_media": ParamBinding(
        "inspect_diff_media", "inspect_operations.diff_media", bool, "hi", True,
        note="10-04：true 时 run_cases 在 --inspect-ops 解析出的操作集上加开 INSPECT diff_media（逐帧像素差异），"
             "HI 的工具目录随之出现该操作；未写或 false 时不开放（hi@0.4.1 及以前的行为）"),
    "inspect_merged_catalog": ParamBinding(
        "inspect_merged_catalog", "inspect_operations.merged_catalog", bool, "shared", True,
        note="10-12 DETECT 并入 INSPECT：true 时 HI 的 INSPECT 目录换成合并后的六项 get_frames / ocr / asr / locate / "
             "video_structure / compare_media（get_pages、crop、align_media、diff_media、locate_frames、locate_regions 由服务"
             "内部翻译，不再单列；asr 带声音时间线；比对图由固定描述模型逐张描述）。asr / locate 仍要权重在。"
             "GIA 提示随之加比对说明（RuntimeGuards.merged_catalog），所以两份规格都写（hi@0.8.0 / gia@0.9.0 起）；"
             "未写或 false = 旧目录"),
    "search_dates": ParamBinding(
        "search_dates", "input_features.search_dates", bool, "hi", True,
        note="10-13：true 时 HI 的 SEARCH 多一种请求 {source: R/URL, output_type: text, include: [\"dates\"]}：页面声明的发布 / 修改"
             "日期、视频平台上传日期、Wayback Machine 最早存档（页面与原图地址），全部免费（search/openers/dates.py）；"
             "观察里多一行 dates[R]。只作用于 HI 的 SEARCH；未写或 false = 没有这种请求（hi@0.8.0 及以前）"),
    "compact_tool_params": ParamBinding(
        "compact_tool_params", "input_features.compact_tool_params", bool, "hi", True,
        note="10-13：true 时 HI 工具目录少两组参数，功能并入默认：INSPECT compare_media 不列 mode——画面对不上且两边都有"
             "声音时自动只比声音（原 mode=audio），InspectService._translate 写入内部的 auto_fallback；SEARCH 打开来源只剩"
             " {source, output_type}——打开视频自动带原语言字幕（原 include=[\"subtitles\"] / subtitle_language）。"
             "未写或 false = 旧参数（hi@0.9.0 及以前）"),
    # ---- 10-05 修正优化计划 P1（输入补全）：hi@0.6.0 / gia@0.7.0 起打开；旧规格不写 = 全关。
    #      运行时经 policies.RuntimeGuards（run_cases 合并进 runtime_guard_scope）生效，见 policies.FEATURE_GUARD_FIELDS
    "ingest_keyframes": ParamBinding(
        "ingest_keyframes", "input_features.ingest_keyframes", int, "shared", True, 0, 16,
        note="P1.1：登记原件视频后立刻产出媒体属性 E 与 N 张均匀关键帧（INSPECT get_frames，discovered_by=HARNESS），"
             "GIA 首轮与 HI 首轮就看得到画面与时长；0 = 不做"),
    "gia_original_frames_first": ParamBinding(
        "gia_original_frames_first", "input_features.gia_original_frames_first", bool, "gia", True,
        note="P1.2：GIA 附图里原件派生帧绝对优先（按时间均匀抽，至少占一半名额），报告引用只填余量"),
    "evidence_origin": ParamBinding(
        "evidence_origin", "input_features.evidence_origin", bool, "shared", True,
        note="P1.3：证据卡加 origin（original / derived_from_original / candidate_source / tool_result），原件卡带 duration_s；"
             "GIA 视图把候选来源的素材单列为 candidate_materials"),
    "candidate_affordance": ParamBinding(
        "candidate_affordance", "input_features.candidate_affordance", bool, "shared", True,
        note="P1.4：候选页面 E / R 卡写明对页面 E 做 get_frames/ocr/asr/align_media 会下载源视频；SEARCH 观察给出调用示例"),
    "cross_round_handoff": ParamBinding(
        "cross_round_handoff", "input_features.cross_round_handoff", bool, "gia", True,
        note="P1.5①：GIA 的 INVESTIGATE materials 可带 R；运行时把 R 及其页面/封面 E 种进新工作区"),
    "workspace_original_slots": ParamBinding(
        "workspace_original_slots", "input_features.workspace_original_slots", int, "hi", True, 0, 8,
        note="P1.5②：工作区 evidence_view 给原件派生帧预留的名额（按时间均匀）；首轮附图也带这几帧；0 = 不预留"),
    "constraints_visible": ParamBinding(
        "constraints_visible", "input_features.constraints_visible", bool, "shared", True,
        note="P1.6：每轮任务数 / 每任务材料数上限进 GIA runtime_constraints，预算耗尽只留 FINISH，HI 最后一轮只能交报告"),
    "language": ParamBinding(
        "language", "input_features.language", str, "shared", True, choices=("zh", "en"),
        note="10-05 晚：模型读到的框架文字的语言（角色提示、卡片标签、工具说明、观察与修复附注）；en 自 hi@0.7.0 / gia@0.8.0 起。"
             "外部取得的内容（网页、字幕、转写、搜索标题）不翻译"),
}


@dataclass(frozen=True)
class InspectOperationSwitches:
    """规格控制的 INSPECT 可选操作开关（默认全关，与旧规格一致）。"""

    diff_media: bool = False
    merged_catalog: bool = False        # 10-12：合并目录（见 inspect.contracts.MERGED_OPERATIONS）

    def operations(self) -> tuple[str, ...]:
        return ("diff_media",) if self.diff_media and not self.merged_catalog else ()


@dataclass(frozen=True)
class InputFeatures:
    """规格控制的模型输入补全特性（10-05 修正优化计划 P1；默认全关，与旧规格一致）。"""

    ingest_keyframes: int = 0
    gia_original_frames_first: bool = False
    evidence_origin: bool = False
    candidate_affordance: bool = False
    cross_round_handoff: bool = False
    workspace_original_slots: int = 0
    constraints_visible: bool = False
    language: str = "zh"
    search_dates: bool = False          # 10-13：SEARCH 日期查询（只在 run_cases 里打开 search_service.date_lookup）
    compact_tool_params: bool = False   # 10-13：compare_media 无 mode、打开视频自动带字幕（run_cases 打开两个服务的开关）

    def enabled(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value and not (key == "language" and value == "zh")}


def binding(key: str) -> ParamBinding:
    try:
        found = HARNESS_PARAM_MAP[key]
    except KeyError as exc:
        raise HarnessParamError(f"harness_params 键 {key!r} 没有运行时映射；已知键：{', '.join(sorted(HARNESS_PARAM_MAP))}") from exc
    if key == "state_renderer" and not found.choices:
        found = replace(found, choices=_renderer_tiers())
    return found


def mapping_table() -> list[dict[str, Any]]:
    """给文档和 run_config 用的映射表。"""

    return [{"key": b.key, "target": b.target, "owner": b.owner, "wired": b.wired, "type": b.kind.__name__,
             "range": [b.minimum, b.maximum] if b.minimum is not None or b.maximum is not None else None,
             "note": b.note} for b in (binding(k) for k in HARNESS_PARAM_MAP)]


def unwired_keys() -> list[str]:
    return [key for key, b in HARNESS_PARAM_MAP.items() if not b.wired]


def resolve_harness_params(hi: Spec, gia: Spec) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """→ (applied, not_applicable)。applied[key] = {"value", "source", "target"}。"""

    if set(DEFAULT_HARNESS_PARAMS) - set(HARNESS_PARAM_MAP):
        # specs.py 加了新键但这里没写映射：宁可跑不起来，也不要"记了但不生效"。
        missing = sorted(set(DEFAULT_HARNESS_PARAMS) - set(HARNESS_PARAM_MAP))
        raise HarnessParamError(f"specs.DEFAULT_HARNESS_PARAMS 里的键没有运行时映射：{', '.join(missing)}")
    values: dict[str, dict[str, Any]] = {}
    for spec in (hi, gia):
        for key, value in spec.harness_params.items():
            values.setdefault(key, {})[spec.spec_id] = binding(key).validate(value, spec.name)
    applied: dict[str, dict[str, Any]] = {}
    not_applicable: dict[str, dict[str, Any]] = {}
    names = {"hi": hi.name, "gia": gia.name}
    for key, by_role in values.items():
        item = binding(key)
        if item.owner in {"hi", "gia"}:
            if item.owner not in by_role:
                continue
            applied[key] = {"value": by_role[item.owner], "source": names[item.owner], "target": item.target}
            for role, value in by_role.items():
                if role != item.owner:
                    not_applicable[key] = {"value": value, "source": names[role],
                                           "reason": f"{key} 只作用于 {item.owner.upper()}，{names[role]} 里的值不适用"}
            continue
        distinct = {repr(v) for v in by_role.values()}
        if len(distinct) > 1:
            detail = "，".join(f"{names[r]}={v!r}" for r, v in by_role.items())
            raise HarnessParamError(f"harness_params.{key} 在两份规格里不同（{detail}）；运行时只有一份 {item.target.split('.')[0]}，"
                                    "请让两份规格取同一值")
        role = "hi" if "hi" in by_role else "gia"
        applied[key] = {"value": by_role[role], "source": " + ".join(names[r] for r in by_role), "target": item.target}
    return applied, not_applicable


@dataclass
class HarnessSettings:
    """run_one 构造运行时要用的三样配置，以及要写进 run_config 的记录。"""

    mode: HarnessMode
    runtime: Any                    # config.RuntimeConfig
    memory_policy: Any              # store.records.MemoryPolicy
    inspect_policy: Any             # inspect.policy.InspectPolicy
    inspect_operations: InspectOperationSwitches = field(default_factory=InspectOperationSwitches)
    input_features: InputFeatures = field(default_factory=InputFeatures)
    applied: dict[str, dict[str, Any]] = field(default_factory=dict)
    not_applicable: dict[str, dict[str, Any]] = field(default_factory=dict)
    differences: dict[str, dict[str, Any]] = field(default_factory=dict)
    specified: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        value: dict[str, Any] = {"mode": self.mode, "applied": self.applied}
        if self.mode != "apply":
            value["specified_not_applied"] = self.specified      # record 模式：规格写了什么，但运行时没用
        return {**value, "not_applicable": self.not_applicable, "differences_from_default": self.differences,
                "unwired": unwired_keys()}


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """["image_input_limit=16", "search_excerpt_chars=600"] → {键: 按映射表类型转换并校验过的值}。"""

    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep:
            raise HarnessParamError(f"--harness-override 应写成 键=值，得到 {item!r}")
        b = binding(key)
        value: Any = raw.strip()
        if b.kind is bool:
            lowered = value.lower()
            if lowered not in {"true", "false", "1", "0"}:
                raise HarnessParamError(f"--harness-override {key} 应为 true / false，得到 {raw!r}")
            value = lowered in {"true", "1"}
        elif b.kind is int:
            try:
                value = int(value)
            except ValueError as exc:
                raise HarnessParamError(f"--harness-override {key} 应为整数，得到 {raw!r}") from exc
        out[key] = b.validate(value, "override")
    return out


def build_harness_settings(config: Any, hi: Spec, gia: Spec, *, mode: HarnessMode = "apply",
                           overrides: dict[str, Any] | None = None) -> HarnessSettings:
    """在 load_config() 结果上用规格覆盖。mode=record 时只校验并记录，运行时仍用默认配置（旧行为，对照用）。

    overrides（09-30）：对照实验用的临时覆盖（run_cases --harness-override），优先于规格；来源记为 "override"，
    run_config 里单独记一份，不改规格文件、不发新版本——试出来有效再发规格。
    """

    from ..inspect.policy import InspectPolicy
    from ..store.records import MemoryPolicy

    if mode not in HARNESS_MODES:
        raise HarnessParamError(f"harness params 模式应为 apply / record，得到 {mode!r}")
    specified, not_applicable = resolve_harness_params(hi, gia)
    for key, value in (overrides or {}).items():
        b = binding(key)
        specified[key] = {"value": b.validate(value, "override"), "source": "override", "target": b.target}
    defaults = {"runtime": config.runtime, "memory_policy": MemoryPolicy(), "inspect_policy": InspectPolicy(),
                "inspect_operations": InspectOperationSwitches(), "input_features": InputFeatures()}
    targets: dict[str, dict[str, Any]] = {name: {} for name in defaults}
    for key, item in specified.items():
        section, attribute = item["target"].split(".", 1)
        targets[section][attribute] = item["value"]
    built = dict(defaults)
    differences: dict[str, dict[str, Any]] = {}
    for section, overrides in targets.items():
        for attribute, value in overrides.items():
            default_value = getattr(defaults[section], attribute)
            if default_value != value:
                key = next(k for k, v in specified.items() if v["target"] == f"{section}.{attribute}")
                differences[key] = {"target": f"{section}.{attribute}", "default": default_value, "spec": value,
                                    "source": specified[key]["source"],
                                    "effective": value if mode == "apply" else default_value}
        if mode == "apply" and overrides:
            try:
                built[section] = replace(defaults[section], **overrides)
            except (TypeError, ValueError) as exc:     # Policy 自己的 __post_init__ 校验兜底
                raise HarnessParamError(f"harness_params 覆盖 {section} 失败：{exc}") from exc
    return HarnessSettings(mode=mode, runtime=built["runtime"], memory_policy=built["memory_policy"],
                           inspect_policy=built["inspect_policy"], inspect_operations=built["inspect_operations"],
                           input_features=built["input_features"],
                           applied=dict(specified) if mode == "apply" else {},
                           not_applicable=not_applicable, differences=differences, specified=specified)


def settings_snapshot(settings: HarnessSettings) -> dict[str, Any]:
    """运行时实际用的三样配置的字典形态（写 run_config 用，便于事后核对"到底跑的是什么参数"）。"""

    return {"runtime": asdict(settings.runtime), "memory_policy": asdict(settings.memory_policy),
            "inspect_policy": settings.inspect_policy.to_dict(),
            "inspect_operations": asdict(settings.inspect_operations),
            "input_features": asdict(settings.input_features)}


__all__ = ["HARNESS_MODES", "HARNESS_PARAM_MAP", "HarnessParamError", "HarnessSettings", "InputFeatures", "InspectOperationSwitches",
           "ParamBinding", "binding",
           "build_harness_settings", "mapping_table", "resolve_harness_params", "settings_snapshot", "unwired_keys"]
