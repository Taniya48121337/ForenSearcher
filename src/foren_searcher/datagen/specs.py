"""版本化规格（spec）：给实习生看的"岗位说明书"和给老师傅看的"办案注意事项"，分开、带版本、带指纹。

大白话：以前改一次提示词就是改个文件名后缀（r7→r8），谁也说不清第 k 轮的数据是按哪版规格生成的，
而且给老师傅的注意事项被直接拼进了训练输入。现在每个规格版本是一个目录：

    specs/hi@0.1.0/
        spec.json               元数据：角色、版本、父版本、harness 参数、changelog、hash
        student_prompt.md       进入训练 system 的固定文本（改动必须升 minor 或 major）
        teacher_instruction.md  只给教师，不进训练输入（改动升 patch 即可）
        rubric.json             评分条目，每条绑定一种 ErrorType，标 presence/absence

hash 由四个文件的内容算出，写进每一行训练数据的 metadata，这样任何一行都能追溯到它是按
哪版规格产生的；同一版本的文件被改动会被 verify() 抓出来。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import SPECS as ERROR_SPECS
from .errors import ErrorType, Role, Severity

_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SPEC_ID = re.compile(r"^(hi|gia)$")
Axis = Literal["presence", "absence"]


class SpecError(ValueError):
    """规格文件不合法或版本规则被违反。"""


def parse_version(value: str) -> tuple[int, int, int]:
    match = _VERSION.match(value)
    if not match:
        raise SpecError(f"版本号应为 x.y.z，得到 {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


@dataclass(frozen=True)
class RubricItem:
    """一条评分条目。绑定错误类型；axis 说明它是"应该有"还是"不得有"。"""

    id: str
    error_type: ErrorType
    axis: Axis
    text: str
    role: Role
    severity: Severity | None = None          # 缺省用错误类型的默认严重度
    applies_when: str = "always"              # 何时适用（自然语言或简单条件）
    evidence_refs: tuple[str, ...] = ()       # 案例专属条目可引用 E 编号（可重放）
    replayable: bool = True                   # 同一条目对同一 episode 在新旧版本下判定应一致

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z0-9_.-]+$", self.id):
            raise SpecError(f"rubric id 只能用小写字母数字 . _ -，得到 {self.id!r}")
        if self.axis not in ("presence", "absence"):
            raise SpecError(f"axis 只能是 presence/absence，得到 {self.axis!r}")
        if not self.text.strip():
            raise SpecError(f"rubric[{self.id}].text 不能为空")
        spec = ERROR_SPECS[self.error_type]
        if self.role not in spec.roles and self.role is not Role.ENV:
            raise SpecError(f"rubric[{self.id}]：{self.error_type.value} 不适用于角色 {self.role.value}")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    @property
    def effective_severity(self) -> Severity:
        return self.severity or ERROR_SPECS[self.error_type].default_severity

    @property
    def exec_checkable(self) -> bool:
        return ERROR_SPECS[self.error_type].exec_checkable

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "error_type": self.error_type.value, "axis": self.axis, "text": self.text,
            "role": self.role.value, "severity": self.severity.value if self.severity else None,
            "applies_when": self.applies_when, "evidence_refs": list(self.evidence_refs), "replayable": self.replayable,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RubricItem":
        return cls(
            id=value["id"], error_type=ErrorType(value["error_type"]), axis=value["axis"], text=value["text"],
            role=Role(value["role"]), severity=Severity(value["severity"]) if value.get("severity") else None,
            applies_when=value.get("applies_when", "always"), evidence_refs=tuple(value.get("evidence_refs", ())),
            replayable=bool(value.get("replayable", True)),
        )


DEFAULT_HARNESS_PARAMS: dict[str, Any] = {
    "image_input_limit": 8,
    "max_workspace_turns": 12,
    "read_page_chars": 6000,
    "max_model_view_chars": 180000,
    "max_evidence_cards": 12,
    "max_candidate_cards": 8,
    "state_renderer": "legacy",       # legacy | compact_8k | compact_16k（渲染器待实现）
    "search_excerpt_chars": 0,        # 09-30：SEARCH 打开网页后附相关摘录的字符数；0 = 不附
    "inspect_diff_media": False,      # 10-04：开放 INSPECT diff_media（逐帧像素差异）；旧规格未写此键即不开放
    "inspect_merged_catalog": False,  # 10-12：INSPECT 合并目录（DETECT 并入；compare_media / video_structure / locate）；hi@0.8.0 起
    # 10-05 修正优化计划 P1（输入补全），hi@0.6.0 / gia@0.7.0 起打开；旧规格未写 = 关（datagen/harness_params.py）
    "ingest_keyframes": 0,
    "gia_original_frames_first": False,
    "evidence_origin": False,
    "candidate_affordance": False,
    "cross_round_handoff": False,
    "workspace_original_slots": 0,
    "constraints_visible": False,
    "language": "zh",                 # 10-05 晚：en = 模型读到的框架文字全用英文（hi@0.7.0 / gia@0.8.0 起）
    "search_dates": False,            # 10-13：SEARCH 日期查询（include=["dates"]，页面元数据 + Wayback CDX）；hi@0.9.0 起
    "compact_tool_params": False,     # 10-13：compare_media 去掉 mode（声音单比并入默认）、打开视频自动带字幕；hi@0.10.0 起
    "max_history_entries": 32,        # 10-13：视图里保留的历史条数（= MemoryPolicy 默认）；规格不写，只给环境变体（walker V9）覆盖
}

# 改变模型输入分布的 harness 参数（卡片字段、附图、视图结构、渲染档位）：只改它们时也可以升 minor——
# student_prompt 没变，但训练输入的分布变了，升 minor 才能和"只改教师指令 / 上限"的 patch 区分开。
INPUT_SHAPING_PARAMS = frozenset({
    "state_renderer", "ingest_keyframes", "gia_original_frames_first", "evidence_origin", "candidate_affordance",
    "cross_round_handoff", "workspace_original_slots", "constraints_visible", "language", "inspect_merged_catalog",
    "search_dates", "compact_tool_params",
})


@dataclass(frozen=True)
class Spec:
    spec_id: str                       # hi | gia
    version: str                       # x.y.z
    parent: str | None
    student_prompt: str
    teacher_instruction: str
    rubric: tuple[RubricItem, ...]
    harness_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_HARNESS_PARAMS))
    changelog: tuple[dict[str, Any], ...] = ()   # [{"targets": [error_type...], "rationale": str}]
    output_schema: str | None = None
    verifier: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _SPEC_ID.match(self.spec_id):
            raise SpecError(f"spec_id 只能是 hi 或 gia，得到 {self.spec_id!r}")
        parse_version(self.version)
        if self.parent is not None:
            parse_version(self.parent)
            if parse_version(self.parent) >= parse_version(self.version):
                raise SpecError(f"parent {self.parent} 必须早于 version {self.version}")
        if not self.student_prompt.strip():
            raise SpecError("student_prompt 不能为空")
        unknown = set(self.harness_params) - set(DEFAULT_HARNESS_PARAMS)
        if unknown:
            raise SpecError(f"harness_params 含未知参数: {', '.join(sorted(unknown))}")
        ids = [item.id for item in self.rubric]
        if len(ids) != len(set(ids)):
            raise SpecError("rubric id 重复")
        role = Role.HI if self.spec_id == "hi" else Role.GIA
        for item in self.rubric:
            if item.role not in (role, Role.ENV):
                raise SpecError(f"rubric[{item.id}] 的角色 {item.role.value} 与规格 {self.spec_id} 不符")
        for entry in self.changelog:
            for target in entry.get("targets", ()):
                ErrorType(target)   # 必须是已知错误类型
        object.__setattr__(self, "rubric", tuple(self.rubric))
        object.__setattr__(self, "changelog", tuple(self.changelog))

    # ---- 指纹 ----
    @property
    def name(self) -> str:
        return f"{self.spec_id}@{self.version}"

    def hash(self) -> str:
        """对进入训练与运行的内容算 SHA-256：student_prompt、teacher_instruction、harness_params、rubric。"""

        canonical = json.dumps(
            {"spec_id": self.spec_id, "version": self.version, "student_prompt": self.student_prompt,
             "teacher_instruction": self.teacher_instruction, "harness_params": self.harness_params,
             "rubric": [item.to_dict() for item in self.rubric]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def role(self) -> Role:
        return Role.HI if self.spec_id == "hi" else Role.GIA

    def rubric_by_axis(self) -> dict[str, int]:
        counts = {"presence": 0, "absence": 0}
        for item in self.rubric:
            counts[item.axis] += 1
        return counts

    # ---- 落盘 ----
    def save(self, root: str | Path) -> Path:
        directory = Path(root) / self.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "student_prompt.md").write_text(self.student_prompt, encoding="utf-8")
        (directory / "teacher_instruction.md").write_text(self.teacher_instruction, encoding="utf-8")
        (directory / "rubric.json").write_text(
            json.dumps([item.to_dict() for item in self.rubric], ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "spec_id": self.spec_id, "version": self.version, "parent": self.parent,
            "harness_params": self.harness_params, "changelog": list(self.changelog),
            "output_schema": self.output_schema, "verifier": self.verifier, "notes": self.notes,
            "hash": self.hash(),
        }
        (directory / "spec.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, verify_hash: bool = True) -> "Spec":
        directory = Path(directory)
        meta = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
        spec = cls(
            spec_id=meta["spec_id"], version=meta["version"], parent=meta.get("parent"),
            student_prompt=(directory / "student_prompt.md").read_text(encoding="utf-8"),
            teacher_instruction=(directory / "teacher_instruction.md").read_text(encoding="utf-8"),
            rubric=tuple(RubricItem.from_dict(v) for v in json.loads((directory / "rubric.json").read_text(encoding="utf-8"))),
            harness_params=dict(meta.get("harness_params", DEFAULT_HARNESS_PARAMS)),
            changelog=tuple(meta.get("changelog", ())), output_schema=meta.get("output_schema"),
            verifier=meta.get("verifier"), notes=meta.get("notes", ""),
        )
        if verify_hash and meta.get("hash") and meta["hash"] != spec.hash():
            raise SpecError(f"{spec.name} 的文件内容与 spec.json 记录的 hash 不一致：已发布的版本不能原地修改，请新建版本")
        return spec


# --------------------------------------------------------------------------- 版本规则

def validate_bump(parent: Spec, child: Spec) -> list[str]:
    """检查子版本相对父版本的升号是否符合规则；返回违规说明（空即通过）。

    规则：student_prompt 变了 → minor 或 major 必须升；只改 teacher_instruction / harness_params / rubric → patch 即可；
    什么都没变 → 不允许发新版本。changelog 至少一条且 targets 非空。
    """

    problems: list[str] = []
    if child.spec_id != parent.spec_id:
        return [f"spec_id 不同：{parent.spec_id} vs {child.spec_id}"]
    if child.parent != parent.version:
        problems.append(f"child.parent 应为 {parent.version}，得到 {child.parent}")
    p, c = parse_version(parent.version), parse_version(child.version)
    student_changed = parent.student_prompt != child.student_prompt
    other_changed = (parent.teacher_instruction != child.teacher_instruction
                     or parent.harness_params != child.harness_params
                     or [i.to_dict() for i in parent.rubric] != [i.to_dict() for i in child.rubric])
    if not student_changed and not other_changed:
        problems.append("内容与父版本完全相同，不应发新版本")
    if student_changed and (c[0], c[1]) == (p[0], p[1]):
        problems.append("student_prompt 改动进入训练输入分布，必须升 minor 或 major")
    input_shaping_changed = any(parent.harness_params.get(key) != child.harness_params.get(key)
                                for key in INPUT_SHAPING_PARAMS)
    if not student_changed and other_changed and (c[0], c[1]) != (p[0], p[1]) and not input_shaping_changed:
        problems.append("只改教师指令/参数/rubric 时应升 patch，不要升 minor（避免把训练分布变化和规格变化混淆）")
    if not child.changelog or not any(entry.get("targets") for entry in child.changelog):
        problems.append("新版本必须在 changelog 里写明针对哪些错误类型（targets）")
    return problems


def diff(parent: Spec, child: Spec) -> dict[str, Any]:
    """给人看的差异摘要。"""

    import difflib

    def text_diff(a: str, b: str, name: str) -> list[str]:
        return list(difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=f"{parent.name}/{name}",
                                         tofile=f"{child.name}/{name}", lineterm="", n=1))

    parent_ids = {i.id: i for i in parent.rubric}
    child_ids = {i.id: i for i in child.rubric}
    return {
        "student_prompt": text_diff(parent.student_prompt, child.student_prompt, "student_prompt.md"),
        "teacher_instruction": text_diff(parent.teacher_instruction, child.teacher_instruction, "teacher_instruction.md"),
        "harness_params": {k: (parent.harness_params.get(k), child.harness_params.get(k))
                           for k in set(parent.harness_params) | set(child.harness_params)
                           if parent.harness_params.get(k) != child.harness_params.get(k)},
        "rubric_added": sorted(set(child_ids) - set(parent_ids)),
        "rubric_removed": sorted(set(parent_ids) - set(child_ids)),
        "rubric_changed": sorted(i for i in set(parent_ids) & set(child_ids) if parent_ids[i].to_dict() != child_ids[i].to_dict()),
    }


def list_versions(root: str | Path, spec_id: str) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    versions = [p.name.split("@", 1)[1] for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{spec_id}@")]
    return sorted(versions, key=parse_version)


def latest(root: str | Path, spec_id: str) -> Spec | None:
    versions = list_versions(root, spec_id)
    return Spec.load(Path(root) / f"{spec_id}@{versions[-1]}") if versions else None


__all__ = ["DEFAULT_HARNESS_PARAMS", "INPUT_SHAPING_PARAMS", "RubricItem", "Spec", "SpecError", "diff", "latest", "list_versions",
           "parse_version", "validate_bump"]
