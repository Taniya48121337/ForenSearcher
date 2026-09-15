"""离线测试用的确定性 Provider。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..contracts import CandidateBatch, CandidateSource, SearchQueryRequest


class FixtureSearchProvider:
    """直接返回预置候选，并记录调用参数供测试断言。"""

    def __init__(self, candidates: tuple[CandidateSource, ...] | Callable[[SearchQueryRequest], tuple[CandidateSource, ...]]) -> None:
        """接受固定候选元组，或按请求动态生成候选的函数。"""

        self._candidates = candidates
        self.calls: list[dict[str, object]] = []

    async def search(self, request: SearchQueryRequest, *, offset: int, limit: int, image_path: Path | None = None) -> CandidateBatch:
        """模拟分页并返回与真实 Provider 相同形状的 CandidateBatch。"""

        # calls 让测试能确认 Service 是否正确传入 offset、limit 和图片路径。
        self.calls.append({"request": request, "offset": offset, "limit": limit, "image_path": image_path})
        values = self._candidates(request) if callable(self._candidates) else self._candidates
        return CandidateBatch("fixture", "fixture", tuple(values[offset:offset + limit]), {"fixture": True, "count": len(values)})
