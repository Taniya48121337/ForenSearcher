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
    "max_candidate_cards": ParamBinding(
        "max_candidate_cards", "memory_policy.max_candidate_cards", int, "shared", True, 1, 1_000),
    "search_excerpt_chars": ParamBinding(
        "search_excerpt_chars", "memory_policy.search_excerpt_chars", int, "hi", True, 0, 4000,
        note="09-30：SEARCH 自动打开网页后，观察里附每页与查询最相关的一段（字符数）；0 = 不附。只作用于 HI 的 SEARCH 观察"),
    "state_renderer": ParamBinding(
        "state_renderer", "memory_policy.state_renderer", str, "shared", True, choices=(),
        note="部分生效：写入 MemoryPolicy → runtime_constraints.state_renderer，本地学生（local_models）与导出据此渲染；"
             "EndpointTeacher / Codex 教师不按档位渲染输入（教师看的始终是 legacy 全量视图）"),
}


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
        if b.kind is int:
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
    defaults = {"runtime": config.runtime, "memory_policy": MemoryPolicy(), "inspect_policy": InspectPolicy()}
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
                           inspect_policy=built["inspect_policy"],
                           applied=dict(specified) if mode == "apply" else {},
                           not_applicable=not_applicable, differences=differences, specified=specified)


def settings_snapshot(settings: HarnessSettings) -> dict[str, Any]:
    """运行时实际用的三样配置的字典形态（写 run_config 用，便于事后核对"到底跑的是什么参数"）。"""

    return {"runtime": asdict(settings.runtime), "memory_policy": asdict(settings.memory_policy),
            "inspect_policy": settings.inspect_policy.to_dict()}


__all__ = ["HARNESS_MODES", "HARNESS_PARAM_MAP", "HarnessParamError", "HarnessSettings", "ParamBinding", "binding",
           "build_harness_settings", "mapping_table", "resolve_harness_params", "settings_snapshot", "unwired_keys"]
