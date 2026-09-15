"""INSPECT Service 与具体后端之间交换的规范化内部值。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


class BackendFailure(RuntimeError):
    """后端可预期失败，包含稳定错误码、阶段和可选定位信息。"""

    def __init__(self, code: str, message: str, *, stage: str = "process", retryable: bool = False, locator: dict[str, Any] | None = None, severity: str = "error") -> None:
        """保存结构化错误，同时以 message 初始化普通异常文本。"""

        super().__init__(message)
        self.code, self.stage, self.retryable, self.locator, self.severity = code, stage, retryable, locator, severity


@dataclass(frozen=True, slots=True)
class MediaProfileData:
    """后端探测出的跨媒体统一属性。"""

    media_kind: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    page_count: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    audio_tracks: int = 0
    format: str | None = None
    backend: str = "local"
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        """去掉 None 字段，生成紧凑的可序列化画像。"""

        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    """已解析到本地文件且完成 probe 的 Evidence。"""

    evidence_id: str
    path: Path
    media_kind: str
    profile: MediaProfileData
    source_context: str = ""


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """后端在内存中产生、等待 Service 原子提交的派生材料。"""

    handle: str
    data: bytes
    media_type: str
    suffix: str
    role: str
    parent_evidence_id: str
    label: str
    locator: Mapping[str, Any]
    acquired_by: str
    production_method: str = "derived_media"


@dataclass(frozen=True, slots=True)
class BackendOutcome:
    """一次后端操作的结果、待提交文件、问题和诊断细节。"""

    result: Mapping[str, Any]
    artifacts: tuple[PendingArtifact, ...] = ()
    issues: tuple[BackendFailure, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
