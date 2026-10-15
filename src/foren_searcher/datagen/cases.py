"""案例文件（case.json）：一个案例在进入循环之前必须说清楚的东西。

大白话：每个案子一张"案卷封面"，写明题目、材料、来源规则、**办这个案子必须拿到哪几份证据**
（required_evidence），以及**程序能自动核对的断言**（checks）。有了这两样，核验器的第一层
和"反向线索评分"（ABSeeker）就都能不靠模型跑起来。

划分（split）规则（2026-09-24 起五种 + contrast_only；按 family_key = 事件/源视频分组，一个 family 只属于一个划分）：
- dev_train：生成训练数据、看结果、改规格，随便用。
- dev_select：门 A、B 用（候选规格之间挑选），参与决策。
- dev_confirm：确认门用，参与决策。dev_select / dev_confirm 是轮换的门控集，用过不降级。
- dev_eval：只画曲线，从不参与决策；一旦它的核验结果被用来改规格（used_for_revision 非空），就自动降级为 dev_train。
- test：EVID-Bench + TRUE test，锁定；降级规则同 dev_eval。
- contrast_only：只用于生成对照轨迹，不做评测。
旧值兼容：读到 dev 当 dev_train，读到 heldout 当 dev_eval（构造时静默映射，写文件只写新值）。
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

_CJK_TEXT = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")   # 汉字 / 假名 / 谚文

Split = Literal["dev_train", "dev_select", "dev_confirm", "dev_eval", "test", "contrast_only"]
SPLITS: tuple[Split, ...] = ("dev_train", "dev_select", "dev_confirm", "dev_eval", "test", "contrast_only")
LEGACY_SPLITS: dict[str, Split] = {"dev": "dev_train", "heldout": "dev_eval"}   # 旧三分法 → 新五分法
DECISION_SPLITS: frozenset[str] = frozenset({"dev_train", "dev_select", "dev_confirm"})  # 训练 / 参与决策
EVAL_SPLITS: frozenset[str] = frozenset({"dev_eval", "test"})                          # 只画曲线；用来改过规格就降级
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
    supports: tuple[str, ...] = ()    # 它支持哪些 requirements[].id（hindsight 用它算 round_yield）
    url_pattern: str | None = None    # 来源 URL 的 glob（如 "twitter.com/prefpolice/status/7537*"），程序匹配 E
    acquired_by: str | None = None    # 派生证据按获取方式匹配（如 "INSPECT/get_frames"）

    def __post_init__(self) -> None:
        if not _EVIDENCE_KEY.match(self.key):
            raise CaseError(f"required_evidence.key 只能用小写字母数字下划线，得到 {self.key!r}")
        if not self.description.strip():
            raise CaseError(f"required_evidence[{self.key}].description 不能为空")
        object.__setattr__(self, "supports", tuple(self.supports))

    def matches(self, *, source_url: str | None, acquired_by: str | None, label: str = "") -> bool:
        """程序判断某条 E 是否就是这份必需证据：url_pattern（glob，不区分协议/www）> acquired_by 前缀 > source_hint 子串。"""

        if self.url_pattern:
            key = _normalize_url(source_url or "")
            if key and fnmatch.fnmatch(key, self.url_pattern.lower()):
                return True
        if self.acquired_by and acquired_by and acquired_by.startswith(self.acquired_by):
            return True
        if self.source_hint and not self.url_pattern and not self.acquired_by:
            hint = self.source_hint.lower()
            if hint in ((source_url or "") + " " + label).lower():
                return True
        return False


def _normalize_url(url: str) -> str:
    value = re.sub(r"^https?://", "", url.strip(), flags=re.I)
    value = re.sub(r"^(www|mobile|m)\.", "", value, flags=re.I)
    return value.split("#", 1)[0].rstrip("/").lower()      # 保留 query（YouTube 的视频号在 ?v= 里）；glob 里的问号写成 [?]


@dataclass(frozen=True)
class Requirement:
    """总任务/局部任务的一项要求（缺口评判和充分性层的单位）。"""

    id: str                   # REQ1 / REQ2 …
    text: str                 # 人能读懂的一句话
    must_cite: bool = True    # 为真时：FINISH/COMPLETED 前必须有 supported Finding 覆盖它
    text_en: str | None = None   # 英文同义表述（10-07）：英文规格下工作区任务是英文，scope 兜底按它比对

    def __post_init__(self) -> None:
        if not re.match(r"^REQ\d+$", self.id):
            raise CaseError(f"requirements.id 必须形如 REQ1，得到 {self.id!r}")
        if not self.text.strip():
            raise CaseError(f"requirements[{self.id}].text 不能为空")


CheckKind = Literal[
    "numeric_in_evidence",     # params: {value, tolerance?}  结论里的数字必须能在某条 E 原文定位
    "entity_in_evidence",      # params: {text, aliases?}     结论里的实体/短语必须能在某条 E 原文定位（aliases：E 原文里的等价写法）
    "status_expected",         # params: {status}             报告状态应为 COMPLETED / INCOMPLETE
    "evidence_key_required",   # params: {key}                某份必需证据必须被最终引用
    "date_in_range",           # params: {min?, max?}         结论中的日期应落在范围内
    "interval_iou",            # params: {start, end, min_iou} 片段定位类任务的区间重叠
    "label_expected",          # params: {label, aliases?}    分类类任务的标签（aliases：答案里可接受的等价措辞）
]


@dataclass(frozen=True)
class Check:
    kind: CheckKind
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    requirement: str | None = None     # 只对对准这条 REQ 的工作区报告生效；None = 整案（各工作区都查）。answer 单元不分 requirement 全查

    def applies_to(self, requirement_ids: set[str] | None) -> bool:
        """工作区报告是否该跑这条检查：requirement_ids 为 None（不收窄）或本检查不限 REQ 时总是跑。"""

        return self.requirement is None or requirement_ids is None or self.requirement in requirement_ids

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "params": self.params, "note": self.note}
        if self.requirement is not None:
            value["requirement"] = self.requirement
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Check":
        return cls(kind=value["kind"], params=dict(value.get("params", {})), note=value.get("note", ""),
                   requirement=value.get("requirement"))

    def __post_init__(self) -> None:
        if self.requirement is not None and not re.match(r"^REQ\d+$", str(self.requirement)):
            raise CaseError(f"check.requirement 必须形如 REQ1，得到 {self.requirement!r}")
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
    requirements: tuple[Requirement, ...] = ()        # 缺口评判 / 充分性层 / round_yield 的单位
    required_evidence_rule: str = "all"               # "all" | "any_one_of" | 分组文本 "any_one_of:a,b; all:c"（evidence_groups() 解析）
    checks: tuple[Check, ...] = ()
    reference_path: str | None = None  # 官方标注文件，演员不可见
    used_for_revision: tuple[str, ...] = ()   # 被用来改规格的记录（spec 版本号）
    notes: str = ""
    # 英文任务文字（10-07）：规格 harness_params.language=en（hi@0.7.0 / gia@0.8.0 起）时交给模型的是它，
    # 模型输入里不混中文框架文字；声明原文本来就是英文。核验仍用 task_text（核验提示是中文）。
    task_text_en: str | None = None

    def task_for(self, language: str) -> str:
        """交给模型的任务文字：language=en 用 task_text_en（缺了就报错，不悄悄混用中文）。

        10-07：task_text 本身就是纯英文（没有一个汉字，例如 EVID-Bench 的 222 个案卷）时直接用它——这不是混用中文。
        """

        if language == "en":
            if (self.task_text_en or "").strip():
                return str(self.task_text_en)
            if not _CJK_TEXT.search(self.task_text):
                return self.task_text
            raise CaseError(f"{self.case_id} 没有 task_text_en，不能按英文规格跑")
        return self.task_text

    def __post_init__(self) -> None:
        if not _CASE_ID.match(self.case_id):
            raise CaseError(f"case_id 必须形如 dataset:source_id（数据集名小写），得到 {self.case_id!r}")
        if self.case_id != f"{self.dataset}:{self.source_id}":
            raise CaseError("case_id 必须等于 dataset + ':' + source_id")
        if self.dataset != self.dataset.lower():
            raise CaseError("dataset 必须全小写（试点语料里 true/True/TRUE 混用过）")
        if self.split in LEGACY_SPLITS:
            object.__setattr__(self, "split", LEGACY_SPLITS[self.split])
        if self.split not in SPLITS:
            raise CaseError(f"split 只能是 {'/'.join(SPLITS)}（旧值 dev/heldout 兼容），得到 {self.split!r}")
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
        req_ids = [r.id for r in self.requirements]
        if len(req_ids) != len(set(req_ids)):
            raise CaseError("requirements.id 重复")
        for item in self.required_evidence:
            for rid in item.supports:
                if rid not in req_ids:
                    raise CaseError(f"required_evidence[{item.key}].supports 引用了不存在的 requirement {rid}")
        for check in self.checks:
            if check.kind == "evidence_key_required" and check.params["key"] not in keys:
                raise CaseError(f"check 引用了不存在的 required_evidence.key: {check.params['key']}")
            if check.requirement is not None and check.requirement not in req_ids:
                raise CaseError(f"check[{check.kind}].requirement 引用了不存在的 requirement {check.requirement}")
        for name in ("media", "claims", "required_evidence", "requirements", "checks", "used_for_revision"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def must_cite_requirements(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.requirements if r.must_cite)

    def orphan_requirements(self) -> tuple[str, ...]:
        """must_cite=True 却没有任何 required_evidence.supports 指向的要求（2026-10-04）。

        充分性/premature_finish 只按"干净 Finding 引用了支撑该要求的必需证据"算覆盖，这样的要求永远覆盖不了，
        每个 run 的答案都会被判 premature_finish（r0cn 的 averimatec_423 REQ2 就是这样）。案卷检查应为空。
        """

        supported = {rid for item in self.required_evidence for rid in item.supports}
        return tuple(rid for rid in self.must_cite_requirements if rid not in supported)

    def evidence_groups(self) -> list[tuple[str, tuple[str, ...]]]:
        """required_evidence_rule 解析成 [(mode, keys)]，mode ∈ {all, any_one_of}。

        "all" → 每个 key 单独一组（各自必须）；"any_one_of" → 全部 key 一组任取其一；
        "any_one_of:a,b; all:c,d" → 逐段解析；解析不了的段按 all 处理并把该段原文当 key 忽略。
        """

        keys = [r.key for r in self.required_evidence]
        rule = (self.required_evidence_rule or "all").strip()
        if rule == "all":
            return [("all", (k,)) for k in keys]
        if rule == "any_one_of":
            return [("any_one_of", tuple(keys))] if keys else []
        groups: list[tuple[str, tuple[str, ...]]] = []
        seen: set[str] = set()
        for segment in rule.split(";"):
            segment = segment.strip()
            if not segment or ":" not in segment:
                continue
            mode, _, rest = segment.partition(":")
            mode = mode.strip()
            names = tuple(k for k in (x.strip() for x in rest.split(",")) if k in keys)
            if mode not in ("all", "any_one_of") or not names:
                continue
            seen.update(names)
            if mode == "all":
                groups.extend(("all", (k,)) for k in names)
            else:
                groups.append(("any_one_of", names))
        groups.extend(("all", (k,)) for k in keys if k not in seen)      # 规则没提到的 key 视为各自必须
        return groups

    def scoped_required_evidence(self, requirement_ids: set[str] | None) -> tuple["RequiredEvidence", ...]:
        """只要求对准这些 requirements 的必需证据；supports 为空的项视为不分工作区、始终在范围内。None → 全部。"""

        if requirement_ids is None:
            return tuple(self.required_evidence)
        return tuple(r for r in self.required_evidence if not r.supports or set(r.supports) & set(requirement_ids))

    # ---- 划分规则 ----
    @property
    def effective_split(self) -> Split:
        """dev_eval / test 一旦被用来改过规格，就不再是评测集，降级为 dev_train；其它划分不降级。"""

        if self.split in EVAL_SPLITS and self.used_for_revision:
            return "dev_train"
        return self.split

    def mark_used_for_revision(self, spec_version: str) -> "Case":
        if spec_version in self.used_for_revision:
            return self
        return Case(**{**asdict_shallow(self), "used_for_revision": (*self.used_for_revision, spec_version)})

    # ---- 序列化 ----
    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checks"] = [c.to_dict() for c in self.checks]
        value["effective_split"] = self.effective_split
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        data = dict(value)
        data.pop("effective_split", None)
        data["media"] = tuple(Media(**m) for m in data.get("media", ()))
        data["source_policy"] = SourcePolicy(**data.get("source_policy", {}))
        data["required_evidence"] = tuple(RequiredEvidence(**r) for r in data.get("required_evidence", ()))
        data["requirements"] = tuple(Requirement(**r) for r in data.get("requirements", ()))
        data["checks"] = tuple(Check.from_dict(c) for c in data.get("checks", ()))
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


def load_cases(directory: str | Path) -> dict[str, Case]:
    """读取目录下全部 *.json 案卷，按 case_id 索引。"""

    root = Path(directory)
    cases: dict[str, Case] = {}
    if not root.is_dir():
        return cases
    for path in sorted(root.glob("*.json")):
        case = Case.load(path)
        if case.case_id in cases:
            raise CaseError(f"重复的 case_id {case.case_id}（{path.name}）")
        cases[case.case_id] = case
    return cases


def check_split_leakage(cases: list[Case] | tuple[Case, ...]) -> list[str]:
    """同一 family_key 只能属于一个划分（按 effective_split）；返回冲突说明列表（空即通过）。

    最低要求是 family 不跨 {dev_train, dev_select, dev_confirm}（训练/决策）与 {dev_eval, test}（评测）两组，
    但这里取更严的规则：组内也不许重复。理由：dev_select / dev_confirm 是门控集，若同一事件的案例也在
    dev_train 里，门 A/B 与确认门量到的是"见过的事件"上的表现，挑选会偏向记住该事件的规格；dev_select 与
    dev_confirm 共享 family 则确认不再独立于挑选；dev_eval 与 test 共享 family 则 dev 曲线会预支 test。
    contrast_only 同样算一个划分。冲突说明里标出是否跨组（跨组 = 评测泄漏，更严重）。
    """

    families: dict[str, set[str]] = {}
    for case in cases:
        families.setdefault(case.family_key, set()).add(case.effective_split)
    problems: list[str] = []
    for key, splits in families.items():
        if len(splits) <= 1:
            continue
        cross = bool(splits & DECISION_SPLITS) and bool(splits & EVAL_SPLITS)
        problems.append(f"family_key={key!r} 同时出现在 {sorted(splits)}" + ("（跨训练/评测组）" if cross else "（组内重复）"))
    return problems


def split_summary(cases: list[Case] | tuple[Case, ...]) -> dict[str, dict[str, int]]:
    """按 effective_split 统计每个划分的案例数与 family 数（全部划分都列出，没有的记 0），供 manifest 用。"""

    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        members = [c for c in cases if c.effective_split == split]
        summary[split] = {"cases": len(members), "families": len({c.family_key for c in members})}
    return summary


__all__ = ["Case", "CaseError", "Check", "Media", "RequiredEvidence", "Requirement", "SPLITS", "SourcePolicy", "Split", "check_split_leakage",
           "load_cases", "split_summary"]
