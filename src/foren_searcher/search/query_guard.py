"""SEARCH 查询守卫（2026-10-05）：拒收含本次调查内部编号或材料代称的文字查询，并告诉 HI 怎么改写。

为什么要有：EVID 冒烟里底座 HI 的查询是 "original source footage of Material A video E1"、"original source of video E1"。
E1、"Material A" 是本次调查内部的名字，搜索引擎看不到，这样的查询只会搜回"如何反向搜视频"之类的教程页，
还白白消耗一次 SEARCH。守卫在调用搜索后端之前拦下，返回 INVALID_INPUT（不重试），反馈里写明改写方法。

规则（通用，不针对任何数据集）：
- 查询文字里出现本次调查**实际存在**的 E / R / F / W / C 编号（由 known 回调判定，避免误伤 "F1 赛车"、"E3 展会"）；
- 或出现材料代称："Material A"、"材料A/甲"、"给定视频"、"the given video" 等。
只检查文字部分；纯图片查询没有文字，放行。拒收记在 log_path（jsonl）和 summary() 里。

10-06（找原片实验 guardlen，search/arms.py）：video_rules=True 时，输出视频的查询再加两条——超过 12 个词、
或含日期 / 时刻戳（屏幕上的录制时间）就拒收并提示"3–8 个词"。evidb12bf 的 50 次视频搜索有 12 次 YouTube 零结果，
全是这两类。默认关，与以前完全相同。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contracts import SearchError, SearchFailure

_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])([ERFWC]\d{1,5})(?![A-Za-z0-9_])")
_MATERIAL_LABEL = re.compile(
    r"(?<![A-Za-z])material\s+[A-H](?![A-Za-z])|材料\s*[A-HＡ-Ｈ甲乙丙丁]|给定的?(?:视频|图片|材料|素材)|"
    r"the\s+given\s+(?:video|image|material|clip)",
    re.IGNORECASE,
)

MESSAGE = ("Query contains internal investigation references ({found}). Search engines cannot see Evidence IDs or "
           "material labels. Rewrite the query using what is actually visible or stated in the material: on-screen "
           "text or captions quoted verbatim, channel logos or watermarks, account names, people, places, events, "
           "dates, distinctive objects or actions.")


def query_text(request: Any) -> str:
    """SearchQueryRequest 的文字部分：query 可能是字符串，也可能是带 text 字段的对象。"""

    if getattr(request, "input_type", None) == "image":
        return ""            # 纯图片查询的 query 是 E 编号本身，不是文字（10-06 修：以前会把它当内部编号拒收）
    query = getattr(request, "query", None)
    if isinstance(query, str):
        return query
    if isinstance(query, dict):
        return str(query.get("text") or "")
    text = getattr(query, "text", None)
    return text if isinstance(text, str) else ""


def internal_references(text: str, known: Callable[[str], bool] | None = None) -> list[str]:
    """查询文字里的内部引用：实际存在的编号 + 材料代称（按出现顺序去重）。known=None 时编号一律算。"""

    found: list[str] = []
    for match in _REFERENCE.finditer(text):
        token = match.group(1).upper()
        if known is None or known(token):
            found.append(token)
    found += [" ".join(m.group(0).split()) for m in _MATERIAL_LABEL.finditer(text)]
    return list(dict.fromkeys(found))


class QueryGuard:
    def __init__(self, known: Callable[[str], bool] | None = None, *, log_path: str | Path | None = None,
                 video_rules: bool = False) -> None:
        self.known = known
        self.log_path = Path(log_path) if log_path else None
        self.video_rules = video_rules
        self.checked = 0
        self.rejected = 0
        self.rejected_video_rules = 0

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def check(self, request: Any) -> None:
        text = query_text(request)
        if not text:
            return
        self.checked += 1
        found = internal_references(text, self.known)
        if found:
            self.rejected += 1
            self._log({"query": text, "found": found})
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", MESSAGE.format(found=", ".join(found))))
        if self.video_rules and getattr(request, "output_type", None) == "video":
            from .arms import video_query_problem

            problem = video_query_problem(text)
            if problem:
                self.rejected += 1
                self.rejected_video_rules += 1
                self._log({"query": text, "rule": "video_query", "message": problem})
                raise SearchFailure(SearchError("validate", "INVALID_INPUT", problem))

    def summary(self) -> dict[str, Any]:
        out = {"enabled": True, "checked": self.checked, "rejected": self.rejected}
        if self.video_rules:
            out.update(video_rules=True, rejected_video_rules=self.rejected_video_rules)
        return out


class QueryGuardedProvider:
    """套在 provider 最外层：先查查询文字，通过了才调用真实后端（拒收的查询不联网、不进录制快照）。"""

    def __init__(self, provider: Any, guard: QueryGuard) -> None:
        self.provider, self.guard = provider, guard

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> Any:
        self.guard.check(request)
        return await self.provider.search(request, offset=offset, limit=limit, image_path=image_path)

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if callable(close):
            await close()


def apply_query_guard(service: Any, guard: QueryGuard | None) -> Any:
    if guard is not None:
        service.provider = QueryGuardedProvider(service.provider, guard)
    return service


__all__ = ["MESSAGE", "QueryGuard", "QueryGuardedProvider", "apply_query_guard", "internal_references", "query_text"]
