"""生成第一版规格 hi@0.1.0 / gia@0.1.0：把现在散落各处的东西收进一个带版本的目录。

来源：
- student_prompt：运行时正在用的角色提示（local_models.py 里的 _HI_MULTIMODAL_SYSTEM_PROMPT、_GIA_SYSTEM_PROMPT、_GIA_DISPATCH_PROMPT）。
- teacher_instruction：试点里给教师的指令文件（actor_generation_r8.txt、actor_evidence_retention.txt、actor_report_r6.txt、
  actor_gia_r2.txt）——以前被拼进 system prompt，现在单独放。
- rubric：按 errors.py 的 22 种类型逐条生成"不得……"条目（absence），再加少数"应该……"条目（presence）。

用法：python -m foren_searcher.datagen.bootstrap [--root specs] [--sft-data experiments/sft_data]
运行时提示以后应改为从这里读取；在那之前，tests/test_datagen_specs.py 会检查两边没有悄悄漂移。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import SPECS, ErrorType, Role
from .specs import DEFAULT_HARNESS_PARAMS, RubricItem, Spec

GIA_PROMPT_SEPARATOR = "\n\n<!-- ===== 首轮派工变体（运行时在没有历史报告时使用） ===== -->\n\n"

_ABSENCE_TEXTS: dict[ErrorType, str] = {
    ErrorType.UNSUPPORTED_ASSERTION: "不得写出任何在所引 E 的原文或像素中找不到支持的陈述。",
    ErrorType.WRONG_CITATION: "不得把讲别的事的 E 当作某句话的依据；引用必须指向真正支持该句的 E。",
    ErrorType.FABRICATED_EVIDENCE_ID: "不得引用不存在或本步不可见的 E 编号。",
    ErrorType.SOURCE_RESTRICTION_MISS: "不得引用任务排除的来源，或超出允许日期范围的来源。",
    ErrorType.ATTRIBUTION_DRIFT: "不得把“来源声称 X”写成“X 成立”，不得把申请/批准/执行/发生互换。",
    ErrorType.SCOPE_OVERREACH: "不得用少数帧、采样点或局部片段推断整段视频、连续区间或全局。",
    ErrorType.MISREAD_FINE_TEXT: "不得把图片/页面中的细小文字、数字、日期读成别的内容；看不清时保留未知。",
    ErrorType.SLOPPY_NUMERIC: "不得比较口径、单位、分母或时间范围不一致的数值。",
    ErrorType.OMITTED_REQUIRED_FACT: "不得遗漏任务明确要求且材料中可得的事实。",
    ErrorType.COMPLETED_WITHOUT_FINISHING: "任务要求项未全部覆盖时不得报 COMPLETED。",
    ErrorType.INCOMPLETE_THOUGH_ANSWERABLE: "可见材料已足以回答时不应报 INCOMPLETE。",
    ErrorType.EMPTY_REPORT: "不得提交没有 Finding 且未说明无法取得材料原因的报告。",
    ErrorType.ILLEGAL_ACTION: "不得输出接口校验通不过的动作或参数。",
    ErrorType.REDUNDANT_ACTION: "不得重复与历史参数相同且不会带来新信息的动作。",
    ErrorType.EVIDENCE_DROPPED: "不得在 focus_evidence_ids 中丢掉后续比较仍需要的 E。",
    ErrorType.WRONG_TOOL: "不得用 SEARCH 读已有 R，或用取帧查看已有帧等与目标不匹配的工具。",
    ErrorType.IGNORED_FEEDBACK: "工具返回 failed/feedback 后不得原样重复同一请求。",
    ErrorType.TASK_NOT_GAP_DRIVEN: "不得派出与尚缺必需证据无关的任务。",
    ErrorType.PREMATURE_FINISH: "必需证据尚缺时不得 FINISH。",
    ErrorType.UNNECESSARY_ROUND: "必需证据已齐时不得再开新一轮。",
    ErrorType.DEPENDENT_SIBLING_TASKS: "同一轮的任务之间不得存在依赖。",
}

_PRESENCE_ITEMS: dict[str, tuple[Role, ErrorType, str]] = {
    "hi.finding_cites_evidence": (Role.HI, ErrorType.UNSUPPORTED_ASSERTION, "每条 Finding 至少引用一条真正支持它的 E。"),
    "hi.completed_covers_task": (Role.HI, ErrorType.COMPLETED_WITHOUT_FINISHING, "COMPLETED 报告覆盖任务列出的全部要求项。"),
    "hi.incomplete_states_reason": (Role.HI, ErrorType.EMPTY_REPORT, "INCOMPLETE 报告在 status_reason 说明缺什么、为何取不到。"),
    "gia.answer_cites_per_fact": (Role.GIA, ErrorType.UNSUPPORTED_ASSERTION, "FINISH 的回答里每个可核验事实就近引用支持它的 E。"),
    "gia.tasks_target_gaps": (Role.GIA, ErrorType.TASK_NOT_GAP_DRIVEN, "INVESTIGATE 的每个任务对应一个尚缺的必需证据或未回答的要点。"),
}


def initial_rubric(role: Role) -> tuple[RubricItem, ...]:
    items = [
        RubricItem(id=f"{role.value.lower()}.no_{kind.value}", error_type=kind, axis="absence", text=text, role=role)
        for kind, text in _ABSENCE_TEXTS.items() if role in SPECS[kind].roles
    ]
    items += [RubricItem(id=item_id, error_type=kind, axis="presence", text=text, role=r)
              for item_id, (r, kind, text) in _PRESENCE_ITEMS.items() if r is role]
    return tuple(items)


def _read_optional(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else fallback


def build_initial_specs(sft_data: str | Path) -> tuple[Spec, Spec]:
    """从运行时常量和试点指令文件构造 hi@0.1.0 与 gia@0.1.0（不落盘）。"""

    from .. import local_models  # 导入较慢（会加载 torch），只在需要时做

    sft = Path(sft_data)
    hi_teacher = (
        _read_optional(sft / "actor_generation_r8.txt", "（教师行动指令：待补，试点文件 actor_generation_r8.txt 未找到）")
        + "\n\n---- 证据保留（focus_evidence_ids）----\n\n"
        + _read_optional(sft / "actor_evidence_retention.txt", "（证据保留指令：待补，试点文件 actor_evidence_retention.txt 未找到）")
        + "\n\n---- 报告阶段 ----\n\n"
        + _read_optional(sft / "actor_report_r6.txt", "（教师报告指令：待补，试点文件 actor_report_r6.txt 未找到）")
    )
    gia_teacher = _read_optional(sft / "actor_gia_r2.txt", "（GIA 教师指令：待补，试点文件 actor_gia_r2.txt 未找到）")
    changelog = ({"targets": [kind.value for kind in ErrorType if kind in _ABSENCE_TEXTS],
                  "rationale": "初始版本：收编运行时提示与试点指令，rubric 按 errors.py 逐类生成"},)
    hi = Spec(
        spec_id="hi", version="0.1.0", parent=None, student_prompt=local_models._HI_MULTIMODAL_SYSTEM_PROMPT,
        teacher_instruction=hi_teacher, rubric=initial_rubric(Role.HI), harness_params=dict(DEFAULT_HARNESS_PARAMS),
        changelog=changelog, output_schema="contracts.parse_hi_decision", verifier="verifier@0.0.0",
        notes="student_prompt 来自 local_models._HI_MULTIMODAL_SYSTEM_PROMPT；运行时的时间预算附注不在此内。",
    )
    gia = Spec(
        spec_id="gia", version="0.1.0", parent=None,
        student_prompt=local_models._GIA_SYSTEM_PROMPT + GIA_PROMPT_SEPARATOR + local_models._GIA_DISPATCH_PROMPT,
        teacher_instruction=gia_teacher, rubric=initial_rubric(Role.GIA), harness_params=dict(DEFAULT_HARNESS_PARAMS),
        changelog=changelog, output_schema="contracts.parse_gia_decision", verifier="verifier@0.0.0",
        notes="student_prompt 含主提示与首轮派工变体，以分隔注释区分；运行时按是否有历史报告选择。",
    )
    return hi, gia


def bootstrap(root: str | Path, sft_data: str | Path, *, overwrite: bool = False) -> list[Path]:
    root = Path(root)
    written = []
    for spec in build_initial_specs(sft_data):
        target = root / spec.name
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} 已存在；已发布的版本不能覆盖，除非 --overwrite")
        written.append(spec.save(root))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成第一版 hi@0.1.0 / gia@0.1.0 规格目录")
    parser.add_argument("--root", default="specs")
    parser.add_argument("--sft-data", default="experiments/sft_data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for path in bootstrap(args.root, args.sft_data, overwrite=args.overwrite):
        print(f"写入 {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
