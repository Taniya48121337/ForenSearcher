"""核验器读取证据的统一入口：从 GlobalStore 读（真实运行），或从字典读（测试、校准）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

_EVIDENCE_ID = re.compile(r"^E\d+$")
_MATERIAL_LABEL = re.compile(r"^\s*材料\s*([A-Za-z])\s*$")


@dataclass(frozen=True)
class EvidenceText:
    """核验器看到的一条证据：全文（不是演员看到的分页片段）、模态、来源地址、图片路径。"""

    evidence_id: str
    content: str
    modalities: tuple[str, ...] = ("text",)
    label: str = ""
    source_ref: str | None = None
    source_url: str | None = None
    derived_from: tuple[str, ...] = ()
    image_path: Path | None = None      # 图片类证据的像素文件（第二层看图用）
    acquired_by: str | None = None      # 取得方式（SEARCH/open、INSPECT/get_frames …），案卷 required_evidence.acquired_by 用它匹配
    provenance: str = ""                # 人能读的来源归属，如"来源材料：B（E2 → 视频E2在0.000秒的帧）"；见 describe_provenance

    @property
    def is_text(self) -> bool:
        return "text" in self.modalities and bool(self.content.strip())

    @property
    def is_image(self) -> bool:
        return "image" in self.modalities


def _host(url: str) -> str:
    host = urlparse(url if "://" in url else "//" + url).netloc.lower()
    return re.sub(r"^www\.", "", host) or url


def _root_name(text: EvidenceText) -> tuple[str, str]:
    """根证据的称呼：("material", "A") / ("web", "外部网页 x.com") / ("other", label)。"""

    match = _MATERIAL_LABEL.match(text.label or "")
    if match:
        return "material", match.group(1).upper()
    if text.source_url:
        return "web", f"外部网页 {_host(text.source_url)}"
    return "other", (text.label or text.evidence_id).strip()


def describe_provenance(evidence_id: str, resolve: Callable[[str], EvidenceText | None]) -> str:
    """沿 derived_from 回溯到根，生成一句来源归属。

    - 根是输入材料（label 形如"材料 A"）：自身 → "来源材料：A（E1，原始材料）"；
      派生 → "来源材料：B（E2 → 视频E2在0.000秒的帧）"（链上每一环写 label；多个父证据时只列根 ID）。
    - 根是外部网页（有 source_url）：自身 → "来源：外部网页 snopes.com"；派生同上加链。
    - 其他（没有 derived_from、也不是材料/网页）：空串。
    """

    def base(eid: str) -> EvidenceText | None:
        try:
            return resolve(eid)
        except KeyError:
            return None

    this = base(evidence_id)
    if this is None:
        return ""

    roots: list[str] = []

    def trail(text: EvidenceText, seen: frozenset[str]) -> str:
        parents = [p for p in text.derived_from if p not in seen]
        resolved = [(p, base(p)) for p in parents]
        resolved_ok = [(p, t) for p, t in resolved if t is not None]
        if not text.derived_from or not resolved_ok:
            if text.evidence_id not in roots:
                roots.append(text.evidence_id)
            return text.evidence_id
        here = (text.label or text.evidence_id).strip()
        if len(here) > 40:
            here = here[:40] + "…"
        if len(resolved_ok) == 1:
            p, parent = resolved_ok[0]
            return f"{trail(parent, seen | {p})} → {here}"
        for p, parent in resolved_ok:
            trail(parent, seen | {p})
        return f"{'、'.join(roots)} → {here}"

    chain = trail(this, frozenset({evidence_id}))
    names = [_root_name(t) for t in (base(r) for r in roots) if t is not None]
    if not names:
        return ""
    is_root = chain == evidence_id
    if all(kind == "material" for kind, _ in names):
        who = "、".join(name for _, name in names)
        return f"来源材料：{who}（{evidence_id}，原始材料）" if is_root else f"来源材料：{who}（{chain}）"
    if is_root:
        kind, name = names[0]
        return f"来源：{name}" if kind == "web" else ""
    who = "、".join(f"材料 {name}" if kind == "material" else name for kind, name in names)
    return f"来源：{who}（{chain}）"


class EvidenceLookup(Protocol):
    def has(self, evidence_id: str) -> bool: ...
    def get(self, evidence_id: str) -> EvidenceText: ...


@dataclass
class DictLookup:
    """测试与校准用：{E: EvidenceText | str}。provenance 可直接给；没给时按记录里的 derived_from / label 现算。"""

    records: dict[str, EvidenceText | str] = field(default_factory=dict)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.records

    def _base(self, evidence_id: str) -> EvidenceText:
        value = self.records[evidence_id]
        if isinstance(value, str):
            return EvidenceText(evidence_id=evidence_id, content=value)
        return value

    def get(self, evidence_id: str) -> EvidenceText:
        text = self._base(evidence_id)
        if text.provenance:
            return text
        provenance = describe_provenance(evidence_id, self._base)
        return replace(text, provenance=provenance) if provenance else text


class StoreLookup:
    """从 GlobalStore 读全文；来源地址取自 E 指回的 R（candidate.final_url / source_url）。"""

    def __init__(self, store: Any) -> None:
        self.store = store
        self._cache: dict[str, EvidenceText] = {}
        self._base_cache: dict[str, EvidenceText] = {}

    def has(self, evidence_id: str) -> bool:
        if not _EVIDENCE_ID.match(evidence_id):
            return False
        try:
            self.get(evidence_id)
        except KeyError:
            return False
        return True

    def get(self, evidence_id: str) -> EvidenceText:
        if evidence_id in self._cache:
            return self._cache[evidence_id]
        text = self._base(evidence_id)
        text = replace(text, provenance=describe_provenance(evidence_id, self._base))
        self._cache[evidence_id] = text
        return text

    def _base(self, evidence_id: str) -> EvidenceText:
        """不含 provenance 的记录（provenance 回溯链上的父证据都走这里，避免递归算 provenance）。"""

        if evidence_id in self._base_cache:
            return self._base_cache[evidence_id]
        record = self.store.evidence(evidence_id)
        url = None
        if isinstance(record.source_ref, str) and record.source_ref.startswith("R"):
            try:
                candidate = self.store.candidate(record.source_ref)
                url = candidate.get("final_url") or candidate.get("source_url")
            except KeyError:
                url = None
        image_path = None
        if "image" in record.modalities and record.artifact_ref:
            try:
                image_path = Path(self.store.artifact_path(record.artifact_ref))
            except (KeyError, ValueError, OSError):
                image_path = None
        text = EvidenceText(
            evidence_id=evidence_id, content=record.content or "", modalities=tuple(record.modalities),
            label=record.label or "", source_ref=record.source_ref, source_url=url,
            derived_from=tuple(record.derived_from), image_path=image_path,
            acquired_by=getattr(record, "acquired_by", None),
        )
        self._base_cache[evidence_id] = text
        return text


def normalize_source_key(url: str) -> str:
    """来源排除名单用的规范化键：去协议、去 www、去末尾斜杠、小写域名。与 cases.SourcePolicy.excluded_keys 对齐。"""

    value = url.strip()
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = re.sub(r"^www\.", "", value, flags=re.I)
    value = value.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    host, _, path = value.partition("/")
    return host.lower() + ("/" + path if path else "")


__all__ = ["DictLookup", "EvidenceLookup", "EvidenceText", "StoreLookup", "describe_provenance", "normalize_source_key"]
