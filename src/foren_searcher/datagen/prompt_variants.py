"""运行时 system 提示的全部固定变体：规格要逐一声明，导出按运行时实际用的那一段取 system。

GIA 的 system 由 local_models._gia_turn_prompt 按状态拼出，共 8 种固定文本（不含任何当次数据）：
    findings                 有可汇总的 Finding、还能开新轮（gia@0.2.0 的"主提示"）
    findings_budget_out      有 Finding、轮次预算用完
    budget_out               没有 Finding、轮次预算用完
    wait                     本轮还有 HI 在跑
    dispatch                 首轮派工（没有任何报告与历史）
    dispatch_empty_material  首轮派工且没有任何可用 C/E 引用
    default                  有报告但没有可汇总的 Finding、还能开新轮
    default_empty_material   同上且没有任何可用 C/E 引用
另外，模型输出没通过接口校验时，策略会在 system 末尾拼一段"修复"附注（local_models._repair_prompt），HI、GIA 都有。
这段属于 harness 的重试机制，不属于角色提示；导出时识别出来，记 repair_turn=true，system 按运行时原样保留。

这里的枚举直接调用运行时函数生成，所以测试可以断言"规格里的每个变体 == 运行时现在会生成的文本"，两边漂移立刻报错。
"""

from __future__ import annotations

import re
from typing import Any

VARIANT_MARKER = re.compile(r"\n*<!-- ===== 变体：([a-z_]+) ===== -->\n*")
LEGACY_MARKER = "<!-- ===== 首轮派工变体（运行时在没有历史报告时使用） ===== -->"    # gia@0.1.0 / 0.2.0 的写法 = dispatch
MAIN_VARIANT = "findings"
GIA_VARIANTS = ("findings", "findings_budget_out", "budget_out", "wait", "dispatch", "dispatch_empty_material",
                "default", "default_empty_material")


def _view(*, reports: list[dict[str, Any]] | None = None, can_create: bool = True, in_progress: bool = False,
          materials: bool = True) -> dict[str, Any]:
    view: dict[str, Any] = {"task": "t", "original_materials": [], "latest_reports": reports or [], "history_index": [],
                            "runtime_constraints": {"can_create_round": can_create}}
    if materials:
        view["original_materials"] = [{"record_type": "evidence", "evidence_id": "E1"}]
    if in_progress:
        view["runtime_constraints"]["round_in_progress"] = True
    return view


_COMPLETED = [{"run_status": "reported", "materials": ["E1"],
               "report": {"status": "COMPLETED", "findings": [{"statement": "s", "evidence_ids": ["E1"]}]}}]
_FAILED = [{"run_status": "failed", "materials": ["E1"]}]


def runtime_gia_variants() -> dict[str, str]:
    """调用运行时 _gia_turn_prompt，生成 8 个变体的确切文本。"""

    from ..local_models import _gia_turn_prompt

    views = {
        "findings": _view(reports=_COMPLETED),
        "findings_budget_out": _view(reports=_COMPLETED, can_create=False),
        "budget_out": _view(reports=_FAILED, can_create=False),
        "wait": _view(in_progress=True),
        "dispatch": _view(),
        "dispatch_empty_material": _view(materials=False),
        "default": _view(reports=_FAILED),
        "default_empty_material": _view(reports=[{"run_status": "failed"}], materials=False),
    }
    return {name: _gia_turn_prompt(view, history=()) for name, view in views.items()}


def repair_suffix(role: str) -> str:
    """策略重试时拼在 system 末尾的修复附注（运行时 _repair_prompt 的增量部分）。"""

    from ..local_models import _repair_prompt

    return _repair_prompt("", role)


def split_variants(student_prompt: str) -> dict[str, str]:
    """规格 student_prompt → {变体名: 文本}。第一段是主变体（GIA 为 findings）；兼容 0.1.0/0.2.0 的旧分隔符。"""

    text = student_prompt
    if LEGACY_MARKER in text:
        main, dispatch = text.split(LEGACY_MARKER, 1)
        return {MAIN_VARIANT: main.strip(), "dispatch": dispatch.strip()}
    parts = VARIANT_MARKER.split(text)
    variants = {MAIN_VARIANT: parts[0].strip()}
    for name, body in zip(parts[1::2], parts[2::2]):
        variants[name] = body.strip()
    return variants


def join_variants(variants: dict[str, str]) -> str:
    """{变体名: 文本} → 规格 student_prompt（主变体在前，其余按 GIA_VARIANTS 顺序，带命名分隔符）。"""

    ordered = [MAIN_VARIANT] + [n for n in GIA_VARIANTS if n != MAIN_VARIANT and n in variants]
    ordered += [n for n in variants if n not in ordered]
    chunks = [variants[MAIN_VARIANT].strip()]
    for name in ordered[1:]:
        chunks.append(f"<!-- ===== 变体：{name} ===== -->\n\n{variants[name].strip()}")
    return "\n\n".join(chunks) + "\n"


def match_runtime(student_prompt: str, runtime_system: str, role: str) -> tuple[str, bool, str | None, bool]:
    """运行时 system 对规格：返回 (训练用 system, 是否逐字对上, 对上的变体名, 是否修复轮)。

    先剥掉修复附注（修复轮的 system = 变体 + 附注，按运行时原样保留），再和规格的每个变体逐字比较。
    对不上时 system 取规格主变体（与 0.2.0 行为一致），变体名 None。
    """

    runtime = runtime_system.strip()
    suffix = repair_suffix(role).strip()
    repair = bool(suffix) and runtime.endswith(suffix)
    core = runtime[: -len(suffix)].strip() if repair else runtime
    variants = split_variants(student_prompt)
    for name, text in variants.items():
        if core == text.strip():
            if role == "HI" and name == MAIN_VARIANT:
                name = "main"                             # HI 规格只有一段
            return runtime, True, name, repair          # 对上了：训练 system 就是运行时原文（修复轮含附注）
    return variants[MAIN_VARIANT].strip(), False, None, repair


__all__ = ["GIA_VARIANTS", "LEGACY_MARKER", "MAIN_VARIANT", "VARIANT_MARKER", "join_variants", "match_runtime",
           "repair_suffix", "runtime_gia_variants", "split_variants"]
