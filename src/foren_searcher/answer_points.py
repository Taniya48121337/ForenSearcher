"""最终答案的编号点：切分与去重（10-05，评测兜底"重复点检查"）。

evidb5 冒烟里 8B 为了写满"3 到 5 点"把同一点复制多遍（151 的 5 点一字不差，001 有 3 点相同）。
这里只做文本层面的判断：去掉证据编号、空白与标点后完全相同，或相似度 ≥ 0.9，算同一点。
"""

from __future__ import annotations

import difflib
import re

_LINE_NUMBER = re.compile(r"^[ \t]*(?:\*\*)?(\d{1,2})\s*[.)、:：](?:\*\*)?\s+", re.M)
_INLINE_NUMBER = re.compile(r"(?:(?<=[\s；;。，,！!？?])|^)(\d{1,2})\s*[.)、:：]\s*(?=\D)", re.M)
_CITATION = re.compile(r"[\[(（]\s*[EFR]\d+(?:\s*[,，;；/、]\s*(?:and\s+)?[EFR]\d+)*\s*[\])）]|\b[EFR]\d+\b")
SIMILAR = 0.9


def split_points(answer: str) -> tuple[str, list[str]]:
    """返回 (编号前的引言, 各点正文)；找不到连续递增的编号（≥2 个）时各点为空列表。"""

    text = answer.strip()
    for pattern in (_LINE_NUMBER, _INLINE_NUMBER):
        starts, expected = [], 1
        for match in pattern.finditer(text):
            if int(match.group(1)) == expected:
                starts.append((match.start(), match.end()))
                expected += 1
        if len(starts) >= 2:
            points = [text[body:(starts[i + 1][0] if i + 1 < len(starts) else len(text))].strip()
                      for i, (_, body) in enumerate(starts)]
            return text[: starts[0][0]].strip(), points
    return text, []


def _key(point: str) -> str:
    return re.sub(r"[\W_]+", "", _CITATION.sub("", point).lower())


def duplicate_indices(points: list[str]) -> list[int]:
    """与前面某一点相同或几乎相同的点的下标（保留第一次出现的）。"""

    kept: list[str] = []
    duplicates: list[int] = []
    for index, point in enumerate(points):
        key = _key(point)
        if not key or any(key == other or difflib.SequenceMatcher(None, key, other).ratio() >= SIMILAR for other in kept):
            duplicates.append(index)
        else:
            kept.append(key)
    return duplicates


def dedupe_answer(answer: str) -> tuple[str, int]:
    """去掉重复的点并重新编号；返回 (新答案, 去掉的点数)。没有编号或没有重复时原样返回。"""

    intro, points = split_points(answer)
    duplicates = set(duplicate_indices(points))
    if not duplicates:
        return answer, 0
    kept = [point for index, point in enumerate(points) if index not in duplicates]
    body = "\n".join(f"{number}. {point}" for number, point in enumerate(kept, 1))
    return (f"{intro}\n{body}" if intro else body), len(duplicates)
