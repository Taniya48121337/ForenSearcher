"""把 Evidence 编号安全地解析成磁盘 Artifact。

Evidence 是数据库中的可追溯记录，Artifact 是它指向的实际文件。本模块把
``E 编号 → Evidence → Artifact 元数据 → 受控路径`` 串起来，并依据文件头判断
真实媒体类型。SEARCH 和 INSPECT 共用这里的规则，避免各写一套路径检查。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .contracts import Evidence
from .ids import is_evidence_id
from .store import ArtifactMetadata, GlobalStore

MediaKind = Literal["image", "video", "audio", "document", "unknown"]


class EvidenceResolutionError(RuntimeError):
    """可机器识别的解析错误；``code`` 供上层转换成稳定接口错误。"""

    def __init__(self, code: str, message: str) -> None:
        """同时保存稳定错误码和普通异常消息。"""

        # RuntimeError 保存给人看的 message，本类再额外保存稳定错误码。
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResolvedEvidence:
    """解析成功后的只读结果：数据库记录、文件元数据、路径和媒体大类。"""

    evidence: Evidence
    artifact: ArtifactMetadata
    path: Path
    media_kind: MediaKind


def sniff_media_kind(path: Path, media_type: str) -> MediaKind:
    """优先检查文件魔数，无法识别时再参考声明的 MIME 类型。

    文件扩展名和 MIME 可能标错，所以可靠性顺序是：文件头特征 > MIME > unknown。
    """

    try:
        # 只读前 32 字节足够覆盖这里使用的常见文件签名，不必加载整个大文件。
        with path.open("rb") as handle:
            header = handle.read(32)
    except OSError:
        # 文件不可读时不在这里抛异常；调用方会把 unknown 作为不支持的材料处理。
        return "unknown"
    # PDF、常见位图格式都有稳定的起始字节。
    if header.startswith(b"%PDF-"):
        return "document"
    if header.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"II*\x00", b"MM\x00*", b"BM")):
        return "image"
    if header.startswith(b"RIFF"):
        # RIFF 是容器总标记，第 9—12 字节才区分 WAV、WEBP、AVI。
        riff_kind = header[8:12]
        if riff_kind == b"WAVE":
            return "audio"
        if riff_kind == b"WEBP":
            return "image"
        if riff_kind == b"AVI ":
            return "video"
    if header.startswith((b"ID3", b"fLaC")):
        return "audio"
    if (len(header) >= 12 and header[4:8] == b"ftyp") or header.startswith((b"\x1aE\xdf\xa3", b"OggS")):
        # MP4、Matroska、Ogg 都能仅包含音频；封面图不算视频轨。
        import av
        try:
            with av.open(str(path)) as container:
                if any(not (stream.disposition & type(stream.disposition).attached_pic)
                       for stream in container.streams.video):
                    return "video"
                if container.streams.audio:
                    return "audio"
        except (OSError, ValueError, av.FFmpegError):
            pass
        return "audio" if media_type.startswith("audio/") or header.startswith(b"OggS") else "video"
    # 没有识别出魔数时，才使用 Artifact 保存的 MIME 声明兜底。
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("video/"):
        return "video"
    if media_type.startswith("audio/"):
        return "audio"
    if media_type == "application/pdf":
        return "document"
    return "unknown"


class EvidenceResolver:
    """持有 GlobalStore，并为工具提供统一、受目录约束的文件解析入口。"""

    def __init__(self, store: GlobalStore) -> None:
        """保存 Store 引用；构造阶段不读取文件。"""

        self.store = store

    def resolve(self, evidence_id: str) -> ResolvedEvidence:
        """解析一个 E 编号；任何缺失或越界路径都转成稳定业务错误。"""

        # 第一步只接受规范 E 编号，避免把任意路径或其他记录编号传入 Store。
        if not is_evidence_id(evidence_id):
            raise EvidenceResolutionError("INVALID_INPUT", "invalid Evidence ID")
        try:
            # 第二步从 Store 取 Evidence；不存在时保留原异常作为 cause 便于调试。
            evidence = self.store.evidence(evidence_id)
        except KeyError as exc:
            raise EvidenceResolutionError("NOT_FOUND", f"Evidence {evidence_id} does not exist") from exc
        if evidence.artifact_ref is None:
            # 纯文本 Evidence 可以没有文件，但媒体工具不能处理它。
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", f"Evidence {evidence_id} has no media Artifact")
        try:
            # strict=True 要求目录和文件此刻真实存在。
            artifact = self.store.artifact(evidence.artifact_ref)
            root = self.store.artifact_root.resolve(strict=True)
            path = artifact.path.resolve(strict=True)
            # relative_to 是关键安全检查：文件必须位于 Store 的 Artifact 根目录下。
            path.relative_to(root)
        except (KeyError, OSError, ValueError) as exc:
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", f"Evidence {evidence_id} Artifact is unavailable") from exc
        # 最后把四项信息一起返回，后续工具无需再次查库或猜媒体类型。
        return ResolvedEvidence(
            evidence,
            artifact,
            path,
            sniff_media_kind(path, artifact.media_type),
        )

    def resolve_many(self, evidence_ids: tuple[str, ...]) -> tuple[ResolvedEvidence, ...]:
        """保持输入顺序批量解析；其中任意一个失败就立即向上抛错。"""

        return tuple(self.resolve(evidence_id) for evidence_id in evidence_ids)
