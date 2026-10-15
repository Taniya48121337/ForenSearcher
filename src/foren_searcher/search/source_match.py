"""SEARCH video results checked against the investigation's own video (10-12; was DETECT find_source).

After a SEARCH with output_type=video, the new video results are ranked as possible originals of the questioned video
(the investigation's original material that is a video) and the best few are downloaded at low resolution, picture
only, and aligned with it. Each result then carries ``content_match``:

    {"material": "E1", "status": "source" | "partial" | "none" | "download_failed" | "not_checked",
     "fraction": 0.92, "b_duration_s": 357.0, "runs": [{"a_range_s": [...], "b_range_s": [...], ...}], "note": "..."}

The result says how much of the material's picture was found in that video, nothing more; the full comparison
(added / removed / re-ordered footage, changed regions, faces, sound) is INSPECT compare_media on [material, page E].

Layering: SEARCH does not import INSPECT. run_cases injects ``fetch`` (low-resolution download through INSPECT's
controlled acquirer, so recording / replay applies) and ``align`` (inspect.source_compare.SourceComparer.match).
State (what was tried, failed downloads, the per-material limit) lives in the Store (candidate_matches), so it
survives a resumed run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

SOURCE_FRACTION = 0.3      # DETECT find_source's threshold for "Source found"
PARTIAL_FRACTION = 0.08    # some footage of the material is in this video (one source of a compilation)

Fetch = Callable[[str, str, str], Awaitable[Path]]          # (candidate R, source_url, page E) -> local video path
Align = Callable[[Path, Path], dict]                        # SourceComparer.match (sync; run in a worker thread)


def duration_tier(duration: Any, dur_a: float | None) -> tuple[int, float]:
    """10-11 ranking: an excerpt is shorter than its source and inserted clips make A a little longer than B.

    Tier 0: B/A in [0.8, 2.0]; 1: unknown duration; 2: [0.4, 0.8) or (2.0, 2.5]; 3: anything else (tried last; a
    compilation's sources can be much longer than the compilation)."""

    if isinstance(duration, (int, float)) and duration > 0 and dur_a:
        q = duration / dur_a
        closeness = abs(math.log(q))
        if 0.8 <= q <= 2.0:
            return 0, closeness
        if 0.4 <= q <= 2.5:
            return 2, closeness
        return 3, closeness
    return 1, 0.0


class SourceMatcher:
    PER_SEARCH = 5          # candidates downloaded and aligned per SEARCH call
    MAX_ATTEMPTS = 20       # per material over the whole investigation (downloads are the slow part)
    TIME_BUDGET_S = 600.0   # per SEARCH call

    def __init__(self, store: Any, *, fetch: Fetch, align: Align, material_path: Callable[[str], Path | None],
                 material: str = "auto", per_search: int | None = None, max_attempts: int | None = None) -> None:
        self.store = store
        self.fetch, self.align, self.material_path = fetch, align, material_path
        self.material = material
        self.per_search = per_search or self.PER_SEARCH
        self.max_attempts = max_attempts or self.MAX_ATTEMPTS
        self._lock = asyncio.Lock()
        self.calls: list[dict[str, Any]] = []
        self.errors: list[str] = []

    # ------------------------------------------------------------------ material

    def questioned_video(self) -> str | None:
        """The investigation's original material that is a video (first one)."""

        if self.material != "auto":
            return self.material if self.material.startswith("E") else None
        try:
            originals = self.store.investigation_meta().get("original_material_ids") or []
        except Exception:  # noqa: BLE001
            return None
        for evidence_id in originals:
            try:
                if self.material_path(evidence_id) is not None:
                    return evidence_id
            except Exception:  # noqa: BLE001
                continue
        return None

    # ------------------------------------------------------------------ main entry

    async def annotate(self, response: Any, workspace_id: str, *, per_search: int | None = None) -> Any:
        """Return the SearchResponse with content_match on each video result (never raises).

        per_search overrides PER_SEARCH for this call (10-13: a text_image search merges two result lists)."""

        if getattr(response, "output_type", None) != "video" or not getattr(response, "results", None):
            return response
        try:
            material = self.questioned_video()
            path_a = self.material_path(material) if material else None
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"material: {type(exc).__name__}: {exc}"[:200])
            return response
        if material is None or path_a is None:
            return response
        async with self._lock:
            try:
                matches = await self._match(response, material, path_a, per_search or self.per_search)
            except Exception as exc:  # noqa: BLE001 - matching must never break a SEARCH
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
                return response
        results = tuple(dataclasses.replace(item, content_match=matches[item.candidate_id])
                        if item.candidate_id in matches and item.video_material is not None else item
                        for item in response.results)
        return dataclasses.replace(response, results=results)

    async def _match(self, response: Any, material: str, path_a: Path, per_search: int | None = None) -> dict[str, dict[str, Any]]:
        started = time.monotonic()
        known = self.store.candidate_matches(material)
        attempts = sum(int(row.get("attempts", 0)) for row in known.values())
        stats = self._discovery_stats()
        dur_a = self._duration(path_a)
        out: dict[str, dict[str, Any]] = {}
        queue = []
        for position, item in enumerate(response.results):
            rid = item.candidate_id
            page = self._page_evidence(item)
            row = known.get(rid)
            if row is not None and not (row["status"] == "download_failed" and row["attempts"] < 2):
                out[rid] = self._public(material, row)          # already compared: reuse, never re-download
                continue
            if page is None:
                continue
            try:
                record = self.store.candidate(rid)
            except KeyError:
                continue
            duration = (record.get("extra") or {}).get("duration")
            tier, closeness = duration_tier(duration, dur_a)
            best, found = stats.get(rid, (record.get("provider_rank"), 1))
            rank = int(best or record.get("provider_rank") or position + 1)
            queue.append(((tier, rank, -min(found, 3), closeness, position), rid, page, record.get("source_url") or item.source_url))
        queue.sort(key=lambda x: x[0])
        room = max(0, min(per_search or self.per_search, self.max_attempts - attempts))
        for k, (_key, rid, page, url) in enumerate(queue):
            if k >= room or time.monotonic() - started > self.TIME_BUDGET_S:
                why = ("the per-search comparison limit was reached" if k >= room and attempts + k < self.max_attempts
                       else f"{self.max_attempts} videos have already been compared with this material" if k >= room
                       else "the time limit for comparisons in one SEARCH was reached")
                out[rid] = {"material": material, "status": "not_checked", "note": why}
                continue
            out[rid] = await self._try(material, path_a, rid, page, url)
        return out

    async def _try(self, material: str, path_a: Path, rid: str, page: str, url: str) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            path_b = await self.fetch(rid, url, page)
        except Exception as exc:  # noqa: BLE001
            row = self.store.record_candidate_match(material, rid, status="download_failed",
                                                    details={"error": f"{type(exc).__name__}: {exc}"[:300], "page": page})
            self.calls.append({"candidate": rid, "status": "download_failed", "s": round(time.monotonic() - t0, 1)})
            return self._public(material, row)
        try:
            found = await asyncio.to_thread(self.align, path_a, Path(path_b))
        except Exception as exc:  # noqa: BLE001
            row = self.store.record_candidate_match(material, rid, status="none",
                                                    details={"error": f"{type(exc).__name__}: {exc}"[:300], "page": page})
            self.calls.append({"candidate": rid, "status": "error", "s": round(time.monotonic() - t0, 1)})
            return self._public(material, row)
        fraction = float(found.get("fraction") or 0.0)
        status = "source" if fraction >= SOURCE_FRACTION else "partial" if fraction >= PARTIAL_FRACTION else "none"
        details = {"page": page, "engine": found.get("engine"), "b_duration_s": found.get("b_duration_s"),
                   "runs": found.get("runs", [])[:8]}
        row = self.store.record_candidate_match(material, rid, status=status, fraction=round(fraction, 3), details=details)
        self.calls.append({"candidate": rid, "status": status, "fraction": round(fraction, 3),
                           "s": round(time.monotonic() - t0, 1)})
        return self._public(material, row)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _public(material: str, row: dict[str, Any]) -> dict[str, Any]:
        details = row.get("details") or {}
        value: dict[str, Any] = {"material": material, "status": row["status"]}
        if row.get("fraction") is not None:
            value["fraction"] = row["fraction"]
        if row["status"] in {"source", "partial"}:
            if details.get("b_duration_s") is not None:
                value["b_duration_s"] = details["b_duration_s"]
            if details.get("runs"):
                value["runs"] = details["runs"]
            if details.get("engine") == "registered":
                value["note"] = "matched after warping this video onto the material's framing (crop, zoom or border)"
        elif row["status"] == "download_failed":
            value["note"] = "the low-resolution picture download failed" + (
                "; it is tried once more by a later SEARCH that returns it" if row.get("attempts", 0) < 2 else "")
        return value

    @staticmethod
    def _page_evidence(item: Any) -> str | None:
        for material in item.materials:
            if material.kind == "page" and material.status in {"ready", "partial"} and material.evidence:
                return material.evidence[0].evidence_id
        return None

    def _discovery_stats(self) -> dict[str, tuple[int | None, int]]:
        getter = getattr(self.store, "candidate_discovery_stats", None)
        try:
            return getter() if callable(getter) else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _duration(path: Path) -> float | None:
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 0.0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            return float(n / fps) if fps > 0 and n > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for call in self.calls:
            by_status[call["status"]] = by_status.get(call["status"], 0) + 1
        return {"enabled": True, "material": self.material, "per_search": self.per_search,
                "max_attempts": self.max_attempts, "compared": len(self.calls), "by_status": by_status,
                "calls": self.calls[:60], "errors": self.errors[:10]}


def content_match_line(candidate_id: str, match: dict[str, Any]) -> str:
    """One observation line for the HI (facts only)."""

    status, material = match.get("status"), match.get("material")
    pct = f"{100 * float(match['fraction']):.0f}%" if match.get("fraction") is not None else None
    if status == "source":
        head = f"content_match[{candidate_id}]: {pct} of {material}'s picture samples are found in this video"
    elif status == "partial":
        head = f"content_match[{candidate_id}]: part of {material} ({pct} of its picture samples) is found in this video"
    elif status == "none":
        head = f"content_match[{candidate_id}]: {material}'s picture was not found in this video ({pct or '0%'} of samples)"
    elif status == "download_failed":
        head = f"content_match[{candidate_id}]: not compared with {material}"
    else:
        head = f"content_match[{candidate_id}]: not compared with {material}"
    # 10-14：只列 A 上 ≥3 s 的对应段；更短的段多是剪辑点附近的对齐误差，过去带着 "(reversed)" 进了答案（evidM27h video_154）
    runs = [r for r in match.get("runs") or []
            if isinstance(r, dict) and r.get("a_range_s") and float(r["a_range_s"][1]) - float(r["a_range_s"][0]) >= 3.0]
    if status in {"source", "partial"} and runs:
        spans = "; ".join(f"{material} {r['a_range_s'][0]}-{r['a_range_s'][1]} s = this video {r['b_range_s'][0]}-{r['b_range_s'][1]} s"
                          + (f" ({r['kind']})" if r.get("kind") not in {None, "normal"} else "")
                          + (" (mirrored)" if r.get("mirrored") else "") for r in runs[:4])
        head += f" ({spans})"
    if match.get("note"):
        head += f"; {match['note']}"
    return head + "."


__all__ = ["PARTIAL_FRACTION", "SOURCE_FRACTION", "SourceMatcher", "content_match_line", "duration_tier"]
