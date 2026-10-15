"""GIA 与 HI 的可替换策略接口。

这里使用 ``Protocol``（结构化接口）而不是固定基类：只要外部对象提供同名的异步
``decide`` 方法，就能接入 Runtime。这样测试脚本、本地模型和远程模型客户端都可
使用同一主框架，而不让框架依赖某一家模型 SDK。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------- 评测运行时兜底（10-05，EVID 基线）
#
# 依据 claude/EVID基线架构修改方案_20261005.md 第 1 步：只改运行时与解码层、不改模型输入格式。
# 默认全关（SFT 数据生成、r0e 等不受影响）；run_cases --runtime-guards eval 打开。
# 用 ContextVar 传递：run_cases 在 runtime.run_new 外设置一次，GIA / HI / 教师（含 to_thread 里的调用）都读得到，
# 不必改各层构造函数。GuardState 是同一 run 内共享的可变计数，结束后写进 result.json / run_config.json。


@dataclass(frozen=True)
class RuntimeGuards:
    accept_uncited_finish_after: int | None = None   # 1a：同一 run 里第 N 次"FINISH 不引 E"时接受并打标
    auto_incomplete_report: bool = False             # 1c：工作区撞轮次上限 / 重复预算用尽 → 自动 INCOMPLETE 报告
    repeat_budget: int | None = None                 # 3c：同一工作区完全相同的动作不再执行、不耗轮次，最多 N 次
    forced_finish: bool = False                      # 1b-3：GIA 修复次数 / 轮数耗尽时用最后一份 FINISH 草稿或 HI 报告收尾
    uniform_gia_frames: bool = False                 # 2：GIA 的材料帧按时间均匀抽，而不是取最前面几张
    json_mode: bool = False                          # 0：教师请求带 response_format=json_object（服务不支持时自动关掉）
    dedupe_finish_points: bool = False               # 10-05：FINISH 里有重复的点 → 第一次退回重写，之后自动去重并打标
    numbered_finish: bool = False                    # 10-13：任务要编号列表而 FINISH 不是 → 第一次退回按 Findings 重写，之后照收并打标
    salvage_report_citations: bool = False           # 10-14：HI 报告的引用错误（正文提到没列的 E、列了没看过 / 不在视图里的 E）
                                                     #        退回一次仍错时由运行时修补，报告保留并打标，不再整份作废
    # ---- 方案第 2 步：会改模型输入（runtime_constraints 多键、动作目录随状态变、观察里多提示），
    #      只在 --runtime-guards eval_plus 打开；正式采用要起 hi/gia 新版本并同步教师数据
    state_dependent_actions: bool = False            # 1b-1 / 3b：GIA 不能再开轮时标"只可 FINISH"；HI 最后一轮动作目录清空、只可交报告
    gia_limits_visible: bool = False                 # 1b-2：每轮任务数上限、每任务材料数上限写进 GIA 的 runtime_constraints
    open_as_video_hint: bool = False                 # 档 A：SEARCH 结果卡给出"打开为视频""与材料视频对齐"的可执行调用
    # ---- 修正优化计划 P0（10-05，claude/EVID修正优化计划_20261005.md）：不改模型输入，eval 档起用 ----
    json_schema: bool = False                        # P0.2：教师请求带 response_format=json_schema（HI 动作/报告、GIA 决定各自的 schema），
                                                     #       服务拒收时退回 json_object，再退回不带
    # ---- P1 输入补全：改模型输入，由规格 hi@0.6.0 / gia@0.7.0 的 harness_params 打开（run_cases 合并进来），
    #      不跟 --runtime-guards 档位走；旧规格全为默认值，r0e 等不受影响
    ingest_keyframes: int = 0                        # P1.1：登记原件视频后立即产出媒体属性 E 与 N 张均匀关键帧（0 = 不做）
    gia_original_first: bool = False                 # P1.2：GIA 附图里原件派生帧绝对优先（按时间均匀抽），报告引用只填余量
    evidence_origin: bool = False                    # P1.3：证据卡加 origin；GIA 视图把候选来源的素材单列为 candidate_materials
    candidate_affordance: bool = False               # P1.4：候选页面 E / R 卡写明"对页面 E 做 get_frames/ocr/asr/align_media 会下载视频"
    cross_round_handoff: bool = False                # P1.5①：GIA 的 INVESTIGATE materials 可带 R，运行时把 R 及其页面/封面 E 种进新工作区
    workspace_original_slots: int = 0                # P1.5②：工作区 evidence_view 给原件派生帧预留的名额（按时间均匀；0 = 不预留）
    # 10-05 晚：所有由框架写给模型的文字（角色提示、卡片标签、工具说明、观察与修复附注）用英文；外部取得的内容原样。
    # 由规格 harness_params.language="en" 打开（hi@0.7.0 / gia@0.8.0 起），旧规格仍是中文、可复现
    english: bool = False
    # 10-12：INSPECT 合并目录（规格 harness_params.inspect_merged_catalog，hi@0.8.0 起；run_cases 按 HI 规格打开）：
    # HI 提示换成合并目录的写法，GIA 提示加比对说明（prompts_en.merged_catalog_hi / GIA_COMPARE_NOTE）
    merged_catalog: bool = False
    # 10-14 评测协议 --auto-tools on（claude/合并目录修复2_收尾检查与text_image放宽_20261013.md 选项 A）：在合并目录上恢复
    # DETECT 的自动流程——SEARCH 的 content_match 判为 source 时运行时自动跑 compare_media、结果并进 SEARCH 观察；
    # 登记材料后自动跑 video_structure，结果摘要放进 GIA 视图。改模型输入，只作评测协议，不进规格
    auto_compare_on_match: bool = False
    auto_video_structure: bool = False

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


EVAL_GUARDS = RuntimeGuards(accept_uncited_finish_after=3, auto_incomplete_report=True, repeat_budget=3,
                            forced_finish=True, uniform_gia_frames=True, json_mode=True, dedupe_finish_points=True,
                            json_schema=True, numbered_finish=True, salvage_report_citations=True)     # json_schema：修正优化计划 P0.2（evidb6 起）
EVAL_PLUS_GUARDS = RuntimeGuards(**{**EVAL_GUARDS.to_dict(), "state_dependent_actions": True,
                                    "gia_limits_visible": True, "open_as_video_hint": True})


# 规格 harness_params（datagen/harness_params.py 的 input_features 段）→ RuntimeGuards 字段。
# constraints_visible 一键打开 eval_plus 的两项"约束可见"（P1.6 转正）；candidate_affordance 同时打开档 A 的观察提示。
FEATURE_GUARD_FIELDS: dict[str, tuple[str, ...]] = {
    "ingest_keyframes": ("ingest_keyframes",),
    "gia_original_frames_first": ("gia_original_first",),
    "evidence_origin": ("evidence_origin",),
    "candidate_affordance": ("candidate_affordance", "open_as_video_hint"),
    "cross_round_handoff": ("cross_round_handoff",),
    "workspace_original_slots": ("workspace_original_slots",),
    "constraints_visible": ("state_dependent_actions", "gia_limits_visible"),
    "language": ("english",),
}


def merge_input_features(guards: RuntimeGuards | None, features: dict[str, Any]) -> RuntimeGuards | None:
    """把规格打开的输入特性并进档位兜底；什么都没开时原样返回（None 表示不开 scope，与旧行为相同）。"""

    from dataclasses import replace

    changes: dict[str, Any] = {}
    for key, value in features.items():
        if key == "language":
            value = value == "en"
        if not value:
            continue
        for name in FEATURE_GUARD_FIELDS.get(key, ()):
            changes[name] = value if isinstance(value, int) and not isinstance(value, bool) and name in (
                "ingest_keyframes", "workspace_original_slots") else True
    if not changes:
        return guards
    return replace(guards or RuntimeGuards(), **changes)


@dataclass
class GuardState:
    guards: RuntimeGuards
    uncited_finish_rejections: int = 0
    duplicate_point_rejections: int = 0
    unnumbered_finish_rejections: int = 0
    contract_flags: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    last_finish_draft: str | None = None
    last_gia_output: dict[str, Any] | None = None
    last_gia_error: str | None = None
    # GIA 的任务数 / 材料数上限（GiaRunner 每次决策前写入）：P0.2 的 schema 解码据此给 tasks / materials 设 maxItems，
    # 不写进模型输入（那是 P1.6 constraints_visible 的事）
    task_limits: dict[str, int] = field(default_factory=dict)

    def count(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def flag(self, name: str) -> None:
        if name not in self.contract_flags:
            self.contract_flags.append(name)
        self.count(name)

    def summary(self) -> dict[str, Any]:
        return {"guards": self.guards.to_dict(), "contract_flags": list(self.contract_flags), "counters": dict(self.counters)}


_GUARD_STATE: ContextVar[GuardState | None] = ContextVar("runtime_guard_state", default=None)


def guard_state() -> GuardState | None:
    return _GUARD_STATE.get()


def active_guards() -> RuntimeGuards:
    state = _GUARD_STATE.get()
    return state.guards if state is not None else RuntimeGuards()


@contextmanager
def runtime_guard_scope(guards: RuntimeGuards | None) -> Iterator[GuardState | None]:
    """在一个 run 内打开评测兜底；guards 为 None 时什么都不做。"""

    if guards is None:
        yield None
        return
    state = GuardState(guards)
    token = _GUARD_STATE.set(state)
    try:
        yield state
    finally:
        _GUARD_STATE.reset(token)


def note_gia_output(output: Any, error: BaseException | str | None = None) -> None:
    """记下 GIA 最近一次输出（含被拒的）；FINISH 的非空答案留作 forced_finish 的草稿。"""

    state = _GUARD_STATE.get()
    if state is None or not isinstance(output, dict):
        return
    state.last_gia_output = output
    state.last_gia_error = str(error) if error is not None else None
    if output.get("decision") == "FINISH":
        answer = output.get("answer")
        if isinstance(answer, list):
            answer = "\n".join(str(item) for item in answer if str(item).strip())
        if isinstance(answer, str) and answer.strip():
            state.last_finish_draft = answer.strip()


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
