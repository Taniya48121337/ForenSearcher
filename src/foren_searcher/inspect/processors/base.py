"""不访问 Store 的 INSPECT Processor 层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, cast

from ..backends.base import BackendOutcome, PreparedMedia
from ..contracts import InspectOperation, LocateFramesParams
from ..policy import InspectPolicy


class InspectProcessor(Protocol):
    """Dispatcher 所需的单操作处理器协议。"""

    @property
    def operation(self) -> InspectOperation:
        """该 Processor 唯一负责的公共 operation。"""

        ...

    async def process(
        self,
        media: tuple[PreparedMedia, ...],
        params: Any,
        policy: InspectPolicy,
        query_media: PreparedMedia | None = None,
    ) -> BackendOutcome:
        """执行一次已经完成材料准备和真实范围校验的操作。"""

        ...


@dataclass(frozen=True, slots=True)
class BackendOperationProcessor:
    """把一个稳定 operation 绑定到 Backend 的同名窄方法。"""

    operation: InspectOperation
    backend: Any

    async def process(
        self,
        media: tuple[PreparedMedia, ...],
        params: Any,
        policy: InspectPolicy,
        query_media: PreparedMedia | None = None,
    ) -> BackendOutcome:
        """调用绑定能力；只有 locate_frames 接收辅助查询图片。"""

        method = getattr(self.backend, self.operation, None)
        if not callable(method):
            raise ValueError(f"backend is missing INSPECT operation: {self.operation}")
        if isinstance(params, LocateFramesParams):
            return cast(
                BackendOutcome,
                await method(media, params, policy, query_media),
            )
        return cast(BackendOutcome, await method(media, params, policy))


_OPERATIONS: tuple[InspectOperation, ...] = (
    "get_frames",
    "get_pages",
    "crop",
    "locate_regions",
    "ocr",
    "asr",
    "audio_qa",
    "visual_qa",
    "locate_frames",
    "align_media",
    "diff_media",
    "compare_media",
    "video_structure",
)


def create_processors(backend: Any) -> Mapping[InspectOperation, InspectProcessor]:
    """为默认或注入 Backend 构造完整且显式的 operation 映射。"""

    missing = [name for name in _OPERATIONS if not callable(getattr(backend, name, None))]
    if missing:
        raise ValueError(f"backend is missing INSPECT operations: {missing}")
    return {name: BackendOperationProcessor(name, backend) for name in _OPERATIONS}
