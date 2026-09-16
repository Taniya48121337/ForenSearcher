"""核验器读取证据的统一入口：从 GlobalStore 读（真实运行），或从字典读（测试、校准）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_EVIDENCE_ID = re.compile(r"^E\d+$")


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

    @property
    def is_text(self) -> bool:
        return "text" in self.modalities and bool(self.content.strip())

    @property
    def is_image(self) -> bool:
        return "image" in self.modalities


class EvidenceLookup(Protocol):
    def has(self, evidence_id: str) -> bool: ...
    def get(self, evidence_id: str) -> EvidenceText: ...


@dataclass
class DictLookup:
    """测试与校准用：{E: EvidenceText | str}。"""

    records: dict[str, EvidenceText | str] = field(default_factory=dict)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self.records

    def get(self, evidence_id: str) -> EvidenceText:
        value = self.records[evidence_id]
        if isinstance(value, str):
            return EvidenceText(evidence_id=evidence_id, content=value)
        return value


class StoreLookup:
    """从 GlobalStore 读全文；来源地址取自 E 指回的 R（candidate.final_url / source_url）。"""

    def __init__(self, store: Any) -> None:
        self.store = store
        self._cache: dict[str, EvidenceText] = {}

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
        )
        self._cache[evidence_id] = text
        return text


def normalize_source_key(url: str) -> str:
    """来源排除名单用的规范化键：去协议、去 www、去末尾斜杠、小写域名。与 cases.SourcePolicy.excluded_keys 对齐。"""

    value = url.strip()
    value = re.sub(r"^https?://", "", value, flags=re.I)
    value = re.sub(r"^www\.", "", value, flags=re.I)
    value = value.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    host, _, path = value.partition("/")
    return host.lower() + ("/" + path if path else "")


__all__ = ["DictLookup", "EvidenceLookup", "EvidenceText", "StoreLookup", "normalize_source_key"]
