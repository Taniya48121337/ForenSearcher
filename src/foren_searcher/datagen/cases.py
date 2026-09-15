"""案例文件（case.json）：一个案例在进入循环之前必须说清楚的东西。

大白话：每个案子一张"案卷封面"，写明题目、材料、来源规则、**办这个案子必须拿到哪几份证据**
（required_evidence），以及**程序能自动核对的断言**（checks）。有了这两样，核验器的第一层
和"反向线索评分"（ABSeeker）就都能不靠模型跑起来。

划分（split）规则：
- dev：可以用来看结果、改规格。
- heldout：只用来检验，一旦它的核验结果被用来改规格，就自动降级为 dev（记录在 used_for_revision）。
- contrast_only：只用于生成对照轨迹，不做评测。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Split = Literal["dev", "heldout", "contrast_only"]
TaskKind = Literal["claim_check", "local_question", "video_diff", "temporal_locate", "copy_detect", "provenance", "geolocate"]

_CASE_ID = re.compile(r"^[a-z0-9_]+:[A-Za-z0-9_\-\.]+$")
_MEDIA_REF = re.compile(r"^[VIAD]\d+$")
_EVIDENCE_KEY = re.compile(r"^[a-z0-9_]+$")


class CaseError(ValueError):
    """case.json 不合法。"""


@dataclass(frozen=True)
class Media:
    ref: str                 # V0 / I0 / A0 / D0，与运行时登记的来源编号一致
    path: str                # 相对案例目录
    media_type: str          # image/jpeg, video/mp4, ...

    def __post_init__(self) -> None:
        if not _MEDIA_REF.match(self.ref):
            raise CaseError(f"media.ref 必须形如 V0/I0/A0/D0，得到 {self.ref!r}")
        if not self.path or Path(self.path).is_absolute():
            raise CaseError(f"media.path 必须是相对案例目录的路径，得到 {self.path!r}")
        if "/" not in self.media_type:
            raise CaseError(f"media.media_type 应为 MIME 类型，得到 {self.media_type!r}")


@dataclass(frozen=True)
class SourcePolicy:
    mode: Literal["unrestricted", "exclude_original", "source_bound", "provided_media"] = "unrestricted"
    excluded_keys: tuple[str, ...] = ()    # 被排除来源的规范化键（域名+路径）
    date_cutoff: str | None = None         # ISO 日期；晚于此日期的来源不算

    def __post_init__(self) -> None:
        if self.date_cutoff is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.date_cutoff):
            raise CaseError(f"date_cutoff 应为 YYYY-MM-DD，得到 {self.date_cutoff!r}")
        object.__setattr__(self, "excluded_keys", tuple(self.excluded_keys))


@dataclass(frozen=True)
class RequiredEvidence:
    """办案必需的一份证据（"线索"）。核验器据此判断某步是否新增了必需证据。"""

    key: str                  # 短名，如 upload_date / original_video / official_notice
    description: str          # 人能读懂的一句话：这份证据要说明什么
    source_hint: str | None = None    # 期望的来源（URL、平台、文件），可为空
    locator_hint: str | None = None   # 期望在来源里的位置（段落、时间、页码），可为空
    must_cite: bool = True            # 最终结论是否必须引用它

    def __post_init__(self) -> None:
        if not _EVIDENCE_KEY.match(self.key):
            raise CaseError(f"required_evidence.key 只能用小写字母数字下划线，得到 {self.key!r}")
        if not self.description.strip():
            raise CaseError(f"required_evidence[{self.key}].description 不能为空")


CheckKind = Literal[
    "numeric_in_evidence",     # params: {value, tolerance?}  结论里的数字必须能在某条 E 原文定位
    "entity_in_evidence",      # params: {text}               结论里的实体/短语必须能在某条 E 原文定位
    "status_expected",         # params: {status}             报告状态应为 COMPLETED / INCOMPLETE
    "evidence_key_required",   # params: {key}                某份必需证据必须被最终引用
    "date_in_range",           # params: {min?, max?}         结论中的日期应落在范围内
    "interval_iou",            # params: {start, end, min_iou} 片段定位类任务的区间重叠
    "label_expected",          # params: {label}              分类类任务的标签
]


@dataclass(frozen=True)
class Check:
    kind: CheckKind
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        required = {
            "numeric_in_evidence": ("value",), "entity_in_evidence": ("text",), "status_expected": ("status",),
            "evidence_key_required": ("key",), "date_in_range": (), "interval_iou": ("start", "end", "min_iou"),
            "label_expected": ("label",),
        }
        if self.kind not in required:
            raise CaseError(f"未知的 check.kind: {self.kind!r}")
        missing = [name for name in required[self.kind] if name not in self.params]
        if missing:
            raise CaseError(f"check[{self.kind}] 缺少参数: {', '.join(missing)}")
        if self.kind == "status_expected" and self.params["status"] not in {"COMPLETED", "INCOMPLETE"}:
            raise CaseError("status_expected.status 必须是 COMPLETED 或 INCOMPLETE")


@dataclass(frozen=True)
class Case:
    case_id: str                       # "<dataset>:<source_id>"，全小写数据集名
    dataset: str
    source_id: str
    split: Split
    task_kind: TaskKind
    task_text: str
    family_key: str                    # 用于划分去重：同一来源视频/事件的案例共用一个键
    media: tuple[Media, ...] = ()
    claims: tuple[str, ...] = ()       # C0, C1, ... 的文字
    source_policy: SourcePolicy = field(default_factory=SourcePolicy)
    required_evidence: tuple[RequiredEvidence, ...] = ()
    checks: tuple[Check, ...] = ()
    reference_path: str | None = None  # 官方标注文件，演员不可见
    used_for_revision: tuple[str, ...] = ()   # 被用来改规格的记录（spec 版本号）
    notes: str = ""

    def __post_init__(self) -> None:
        if not _CASE_ID.match(self.case_id):
            raise CaseError(f"case_id 必须形如 dataset:source_id（数据集名小写），得到 {self.case_id!r}")
        if self.case_id != f"{self.dataset}:{self.source_id}":
            raise CaseError("case_id 必须等于 dataset + ':' + source_id")
        if self.dataset != self.dataset.lower():
            raise CaseError("dataset 必须全小写（试点语料里 true/True/TRUE 混用过）")
        if self.split not in {"dev", "heldout", "contrast_only"}:
            raise CaseError(f"split 只能是 dev/heldout/contrast_only，得到 {self.split!r}")
        if not self.task_text.strip():
            raise CaseError("task_text 不能为空")
        if not self.family_key.strip():
            raise CaseError("family_key 不能为空")
        refs = [m.ref for m in self.media]
        if len(refs) != len(set(refs)):
            raise CaseError("media.ref 重复")
        keys = [r.key for r in self.required_evidence]
        if len(keys) != len(set(keys)):
            raise CaseError("required_evidence.key 重复")
        for check in self.checks:
            if check.kind == "evidence_key_required" and check.params["key"] not in keys:
                raise CaseError(f"check 引用了不存在的 required_evidence.key: {check.params['key']}")
        for name in ("media", "claims", "required_evidence", "checks", "used_for_revision"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    # ---- 划分规则 ----
    @property
    def effective_split(self) -> Split:
        """heldout 一旦被用来改过规格，就不再是留出集。"""

        if self.split == "heldout" and self.used_for_revision:
            return "dev"
        return self.split

    def mark_used_for_revision(self, spec_version: str) -> "Case":
        if spec_version in self.used_for_revision:
            return self
        return Case(**{**asdict_shallow(self), "used_for_revision": (*self.used_for_revision, spec_version)})

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checks"] = [{"kind": c.kind, "params": c.params, "note": c.note} for c in self.checks]
        value["effective_split"] = self.effective_split
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        data = dict(value)
        data.pop("effective_split", None)
        data["media"] = tuple(Media(**m) for m in data.get("media", ()))
        data["source_policy"] = SourcePolicy(**data.get("source_policy", {}))
        data["required_evidence"] = tuple(RequiredEvidence(**r) for r in data.get("required_evidence", ()))
        data["checks"] = tuple(Check(kind=c["kind"], params=dict(c.get("params", {})), note=c.get("note", ""))
                               for c in data.get("checks", ()))
        for name in ("claims", "used_for_revision"):
            data[name] = tuple(data.get(name, ()))
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise CaseError(f"case.json 含未知字段: {', '.join(sorted(unknown))}")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Case":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def asdict_shallow(case: Case) -> dict[str, Any]:
    """保留嵌套 dataclass 对象的浅拷贝（asdict 会把它们变成 dict）。"""

    return {name: getattr(case, name) for name in case.__dataclass_fields__}


def check_split_leakage(cases: list[Case] | tuple[Case, ...]) -> list[str]:
    """同一 family_key 不能同时出现在 dev 和 heldout；返回冲突说明列表（空即通过）。"""

    families: dict[str, set[str]] = {}
    for case in cases:
        families.setdefault(case.family_key, set()).add(case.effective_split)
    return [f"family_key={key!r} 同时出现在 {sorted(splits)}" for key, splits in families.items()
            if "dev" in splits and "heldout" in splits]


__all__ = ["Case", "CaseError", "Check", "Media", "RequiredEvidence", "SourcePolicy", "check_split_leakage"]
