"""模型视图的输入补全（10-05 修正优化计划 P1，claude/EVID修正优化计划_20261005.md）。

审计（claude/EVID_GIA-HI逐步输入输出审计_20261005.md）发现的"看不到"问题都在展示层：
- 证据卡只靠 label 前缀区分"材料 A 的帧"和"别的视频的封面"，GIA 把封面当成被查对象派给 HI；
- 原件卡片没有时长，HI 只能猜 range_s [0, 60]；
- 候选页面 E 的卡片是 text，看不出"对它做 get_frames 会下载视频"。

这里只做两件纯展示的事，开关都在 policies.RuntimeGuards（由规格 harness_params 打开，旧规格全关）：
1. ``annotate_view``：给视图里的每张证据卡加 origin（original / derived_from_original / candidate_source / tool_result），
   原件视频卡加 duration_s，候选页面 E 与 R 卡加 video_access 说明；
2. ``original_frames`` / ``spread``：按时间排好的原件派生帧，供 GIA 附图优先级（P1.2）与工作区预留名额（P1.5②）共用。

不改 Store、不改工具行为；同一 Store 状态 + 同一开关 = 同一输出。
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Iterable

from .policies import active_guards

ORIGINS = ("original", "derived_from_original", "candidate_source", "tool_result")
VIDEO_ACCESS = ("INSPECT get_frames / ocr / asr / align_media on this page E downloads the source video "
                "(the result is a video E of this candidate)")


def original_ids(store: Any) -> tuple[str, ...]:
    """本次调查的原始材料编号（C/E）；调查尚未登记时为空。"""

    try:
        return tuple(store.investigation_meta()["original_material_ids"])
    except (KeyError, RuntimeError):
        return ()


class OriginResolver:
    """沿 derived_from 链给 E 定性；一次视图构造内缓存。"""

    def __init__(self, store: Any, originals: Iterable[str] | None = None) -> None:
        self.store = store
        self.originals = set(original_ids(store) if originals is None else originals)
        self._cache: dict[str, str] = {}

    def __call__(self, evidence_id: str, _depth: int = 0) -> str | None:
        if evidence_id in self._cache:
            return self._cache[evidence_id]
        if _depth > 64:                       # 异常深的派生链：不定性
            return None
        try:
            evidence = self.store.evidence(evidence_id)
        except KeyError:
            return None
        for parent in evidence.derived_from:
            self(parent, _depth + 1)
        self._cache[evidence_id] = self._classify(evidence_id, evidence)
        return self._cache[evidence_id]

    def _classify(self, evidence_id: str, evidence: Any) -> str:
        if evidence_id in self.originals or evidence.acquired_by == "HARNESS/ingest":
            return "original"
        if isinstance(evidence.source_ref, str) and evidence.source_ref.startswith("R"):
            return "candidate_source"
        kinds = {self._cache.get(parent) for parent in evidence.derived_from}
        if not kinds or None in kinds:
            return "tool_result"
        if kinds <= {"original", "derived_from_original"}:
            return "derived_from_original"
        if kinds == {"candidate_source"}:
            return "candidate_source"
        return "tool_result"


def media_duration(store: Any, evidence_id: str) -> float | None:
    """原件媒体的时长（秒）：取 INSPECT/media_profile 记录；还没探测过时为 None。"""

    try:
        profiles = store.find_derived_evidence(derived_from=(evidence_id,), acquired_by="INSPECT/media_profile")
    except Exception:  # noqa: BLE001 —— 只是展示字段，取不到就不写
        return None
    for profile in profiles:
        try:
            value = json.loads(profile.content).get("duration_s")
        except (ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return round(float(value), 3)
    return None


def _locator_position(locator: Any) -> float | None:
    try:
        value = json.loads(locator) if isinstance(locator, str) else locator
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    for key in ("time_s", "page_number"):
        position = value.get(key)
        if isinstance(position, (int, float)) and not isinstance(position, bool) and math.isfinite(position):
            return float(position)
    return None


def original_frames(store: Any, originals: Iterable[str] | None = None) -> list[tuple[str, str, float]]:
    """原件直接派生的帧 / 页面图片 → [(E, 原件 E, 时刻或页码)]，按原件、时刻排序；同一时刻只留最早的 E。"""

    roots = [value for value in (original_ids(store) if originals is None else originals) if value.startswith("E")]
    if not roots:
        return []
    root_set = set(roots)
    found: dict[tuple[str, float], str] = {}
    for record in store.evidence_index_records(content_chars=1):
        parents = record["derived_from"]
        if len(parents) != 1 or parents[0] not in root_set or "image" not in record["modalities"]:
            continue
        try:
            position = _locator_position(store.evidence(record["evidence_id"]).locator)
        except KeyError:
            continue
        if position is None:
            continue
        found.setdefault((parents[0], position), record["evidence_id"])
    order = {root: index for index, root in enumerate(roots)}
    return [(eid, root, position) for (root, position), eid in sorted(found.items(), key=lambda kv: (order[kv[0][0]], kv[0][1]))]


def spread(items: list[Any], limit: int) -> list[Any]:
    """按顺序均匀取 limit 个（含首尾）；不多于 limit 时原样返回。"""

    if limit < 1:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (limit - 1)
    picked: list[Any] = []
    for index in range(limit):
        item = items[round(index * step)]
        if item not in picked:
            picked.append(item)
    return picked


def reserved_original_frames(store: Any, slots: int) -> list[str]:
    """工作区预留名额（P1.5②）：原件派生帧按时间均匀取 slots 张。"""

    if slots < 1:
        return []
    return [eid for eid, _, _ in spread(original_frames(store), slots)]


def _candidate_links(store: Any, candidate_id: str) -> dict[str, Any]:
    try:
        candidate = store.candidate(candidate_id)
        linked = store.candidate_evidence(candidate_id)
    except KeyError:
        return {}
    out: dict[str, Any] = {"candidate_type": candidate.get("candidate_type")}
    for kind, status, evidence in linked:
        if status not in {"ready", "partial"}:
            continue
        out.setdefault(f"{kind}_evidence_id", evidence.evidence_id)
    return out


def annotate_view(view: dict[str, Any], store: Any) -> None:
    """按开关给视图里的卡片补字段（原地改 view；被改动的顶层值先深拷贝，不改调用方共享的卡片对象）。"""

    guards = active_guards()
    origin_on = guards.evidence_origin
    duration_on = guards.evidence_origin or guards.ingest_keyframes > 0
    access_on = guards.candidate_affordance
    if not (origin_on or duration_on or access_on) or store is None:
        return
    resolver = OriginResolver(store)
    durations: dict[str, float | None] = {}
    candidates: dict[str, dict[str, Any]] = {}

    def candidate(candidate_id: str) -> dict[str, Any]:
        if candidate_id not in candidates:
            candidates[candidate_id] = _candidate_links(store, candidate_id)
        return candidates[candidate_id]

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        eid = value.get("evidence_id")
        is_card = isinstance(eid, str) and eid.startswith("E") and (
            "acquired_by" in value or "derived_from" in value or "modalities" in value)
        if is_card:
            origin = resolver(eid)
            if origin_on and origin is not None:
                value["origin"] = origin
            if duration_on and origin == "original" and "video" in (value.get("modalities") or ["video"]):
                if eid not in durations:
                    durations[eid] = media_duration(store, eid)
                if durations[eid] is not None:
                    value["duration_s"] = durations[eid]
            source = value.get("source_ref")
            if access_on and isinstance(source, str) and source.startswith("R"):
                links = candidate(source)
                if links.get("candidate_type") == "video" and links.get("page_evidence_id") == eid:
                    value["video_access"] = (f"video already downloaded as {links['video_evidence_id']}"
                                             if links.get("video_evidence_id") else VIDEO_ACCESS)
        cid = value.get("candidate_id")
        if access_on and not isinstance(eid, str) and isinstance(cid, str) and cid.startswith("R") and (
                "source_url" in value or value.get("record_type") == "candidate"):
            links = candidate(cid)
            for key in ("page_evidence_id", "cover_evidence_id", "video_evidence_id"):
                if links.get(key):
                    value[key] = links[key]
            if links.get("candidate_type") == "video" and links.get("page_evidence_id"):
                value["video_access"] = (f"video already downloaded as {links['video_evidence_id']}"
                                         if links.get("video_evidence_id") else
                                         f"INSPECT get_frames / ocr / asr / align_media on {links['page_evidence_id']} "
                                         "downloads the source video")
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                visit(item)

    for key in list(view):
        item = view[key]
        if isinstance(item, (dict, list)):
            view[key] = copy.deepcopy(item)
            visit(view[key])


__all__ = ["ORIGINS", "OriginResolver", "VIDEO_ACCESS", "annotate_view", "media_duration", "original_frames",
           "original_ids", "reserved_original_frames", "spread"]
