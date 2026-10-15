"""Backend methods of the merged INSPECT catalog (10-12): compare_media and video_structure, plus the non-speech sound
timeline attached to asr. Mixed into LocalInspectBackend; see inspect/source_compare.py and inspect/video_structure.py
for the measurements themselves."""

from __future__ import annotations

import re

import os
import time
from threading import Event
from typing import Any

import numpy as np

from ...i18n import tr
from ..contracts import AlignMediaParams, CompareMediaParams, DiffMapping, DiffMediaParams, VideoStructureParams
from ..policy import InspectPolicy
from .. import media_diff
from ..source_compare import (CompareConfig, CompareUnavailable, FaceIdentity, Run, SourceComparer, audio_compare,
                              audio_offset, grab, video_info)
from .base import BackendFailure, BackendOutcome, PendingArtifact, PreparedMedia

DESCRIBE_QUESTIONS = {
    "region": ("Top row: the same moment of video A (left) and video B (right); bottom row: the red-boxed area enlarged, "
               "A detail left and B detail right. What object is in the boxed area in B, and what is there instead in A? "
               "Ignore people's movement. Answer in one short line exactly as: B: <object>; A: <object>."),
    "person": ("Left: a whole frame of video A with one face boxed in red; then pairs of that face in A (left) and the face at "
               "the same position in video B (right). Describe the boxed person in one short line: gender, approximate age, "
               "hair, clothing, and where they are in the scene (for example front passenger seat, back seat, left of the group)."),
    "segment": ("These frames come from a part of video A that is not in video B (the last tile, if labelled B, is B just before "
                "it). In one short line, describe what the A frames show: people, objects, action and setting."),
}
DESCRIBE_QUESTIONS["shot"] = ("These three frames are the beginning, middle and end of one shot of a video. In one short line, "
                              "describe what the shot shows: people, objects, action and setting.")
DESCRIBE_MAX = 6


def _unavailable(exc: CompareUnavailable) -> BackendFailure:
    code = {"TIME_LIMIT": "TIMEOUT", "CANCELLED": "TIMEOUT", "UNSUPPORTED": "UNSUPPORTED_MEDIA",
            "INSUFFICIENT_FRAMES": "DECODE_FAILED"}.get(exc.code, "PROCESS_FAILED")
    return BackendFailure(code, str(exc), stage="decode" if code == "DECODE_FAILED" else "process",
                          retryable=code == "TIMEOUT")


class MergedOperationsMixin:
    """compare_media / video_structure / asr sound timeline for LocalInspectBackend."""

    describer: Any = None          # fixed VLM endpoint (inspect/describe.py); set by the run driver
    _faces: FaceIdentity | None = None
    _faces_checked = False

    # ------------------------------------------------------------------ helpers

    def _face_identity(self) -> FaceIdentity | None:
        if not self._faces_checked:
            self._faces_checked = True
            if os.environ.get("FOREN_COMPARE_FACES", "auto") != "off" and FaceIdentity.available():
                device = os.environ.get("FOREN_FACE_DEVICE", "auto")
                if device == "auto":
                    try:
                        import torch

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except Exception:  # noqa: BLE001
                        device = "cpu"
                self._faces = FaceIdentity(device)
        return self._faces

    def _report_images(self, media: PreparedMedia, other: PreparedMedia, images: list[dict], prefix: str,
                       acquired_by: str) -> tuple[list[PendingArtifact], list[dict]]:
        artifacts, rows = [], []
        for k, img in enumerate(images):
            handle = f"{prefix}:{k}"
            locator = {"image": k, "kind": img.get("kind"), "index": img.get("index")}
            artifacts.append(PendingArtifact(handle, img["jpeg"], "image/jpeg", ".jpg", "comparison_image", media.evidence_id,
                                             str(img.get("caption", "comparison image"))[:300], locator, acquired_by,
                                             production_method="structured_analysis",
                                             extra_parent_evidence_ids=(other.evidence_id,) if other is not media else ()))
            rows.append({"handle": handle, "kind": img.get("kind"), "index": img.get("index"), "caption": img.get("caption", "")})
        return artifacts, rows

    def _describe(self, images: list[dict]) -> list[str]:
        """One short line per comparison image from the fixed describer VLM (10-11 v2.4: names what the boxes show)."""

        describer = getattr(self, "describer", None)
        if describer is None:
            return []
        lines = []
        jobs = [(k, img) for k, img in enumerate(images) if img.get("kind") in DESCRIBE_QUESTIONS]
        jobs.sort(key=lambda x: {"region": 0, "person": 1, "segment": 2, "shot": 3}[x[1]["kind"]])
        for k, img in jobs[:DESCRIBE_MAX]:
            try:
                answer = describer.describe(img["jpeg"], DESCRIBE_QUESTIONS[img["kind"]])
            except Exception as exc:  # noqa: BLE001 - a failed description only drops that line
                self.__dict__.setdefault("describe_errors", []).append(f"{type(exc).__name__}: {exc}"[:200])
                continue
            if isinstance(answer, str) and answer.strip():
                label = {"region": "changed region", "person": "person", "segment": "added footage", "shot": "shot"}[img["kind"]]
                img["description"] = " ".join(answer.split())[:300]
                lines.append(f"Visual check of comparison image {k + 1} ({label} {int(img.get('index') or 0) + 1}): {img['description']}")
        return lines

    def _audio_8k(self, item: PreparedMedia) -> np.ndarray:
        samples, rate, _ = self._audio_samples(item)  # type: ignore[attr-defined]
        return self._resample_for_correlation(samples.astype(np.float32), rate, 8000)  # type: ignore[attr-defined]

    @staticmethod
    def _mapping_from_runs(runs: list[Run], step: float = 0.5):
        moving = [r for r in runs if r.kind != "frozen"]

        def mapping(t: float) -> float | None:
            for r in moving:
                if r.a_start - step / 2 <= t <= r.a_end + step / 2 - 1.0:
                    return r.b_at(t)
            return None
        return mapping

    async def _robust_runs(self, media: tuple[PreparedMedia, ...], policy: InspectPolicy) -> tuple[list[Run], Any]:
        """SIFT/RANSAC alignment (align_media) for crops, borders and overlays; returns runs and a B->A registration."""

        try:
            outcome = await self.align_media(media, AlignMediaParams("visual", None, None, 8), policy)  # type: ignore[attr-defined]
        except BackendFailure:
            return [], None
        runs: list[Run] = []
        for match in outcome.result.get("matches", ()):
            anchors = match.get("anchors") or []
            pts = [(float(x["time_a_s"]), float(x["time_b_s"])) for x in anchors]
            (a0, a1), (b0, b1) = match["range_a_s"], match["range_b_s"]
            if len(pts) < 2:
                pts = [(a0, b0), (a1, b1)]
            slope, offset = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1) if len(pts) >= 2 else (1.0, b0 - a0)
            ta = list(np.arange(a0 + 0.25, a1, 0.5)) or [a0, a1]
            runs.append(Run([float(t) for t in ta], [float(slope * t + offset) for t in ta], engine="sift", score=match.get("score")))
        # align_media returns overlapping / split candidates; keep the best-scoring one per A span and merge neighbours
        # that follow the same line (same offset and speed), so a continuous excerpt is not reported as several cuts.
        runs.sort(key=lambda r: -(r.score or 0))
        kept: list[Run] = []
        for run in runs:
            if all(min(run.a_end, k.a_end) - max(run.a_start, k.a_start) < 0.5 for k in kept):
                kept.append(run)
        kept.sort(key=lambda r: r.a_start)
        merged: list[Run] = []
        for run in kept:
            if merged:
                prev = merged[-1]
                (s1, o1), (s2, o2) = prev.fit(), run.fit()
                if abs(s1 - s2) <= 0.1 and abs(prev.b_at(run.a_start) - run.b_at(run.a_start)) <= 0.6 \
                        and run.a_start - prev.a_end <= 3.0:
                    merged[-1] = Run(prev.ta + run.ta, prev.tb + run.tb, engine="sift", score=max(prev.score or 0, run.score or 0))
                    continue
            merged.append(run)
        runs = merged
        # one B->A registration per run: different parts of an edit can be cropped or framed differently
        registrations: dict[int, Any] = {}
        try:
            fa, _, wa, ha = video_info(media[0].path)
            fb, _, wb, hb = video_info(media[1].path)
            for k, run in enumerate(runs):
                pairs = []
                for q in (0.2, 0.5, 0.8):
                    t = run.a_start + q * (run.a_end - run.a_start)
                    ia, ib = int(round(t * fa)), int(round(run.b_at(t) * fb))
                    ga, gb = grab(media[0].path, [ia]), grab(media[1].path, [ib])
                    if ia in ga and ib in gb:
                        pairs.append((ga[ia], gb[ib]))
                if pairs:
                    registration = media_diff.estimate_registration(pairs, (wa, ha), (wb, hb), True)
                    if registration.info.get("applied"):
                        registrations[k] = registration
        except Exception:  # noqa: BLE001 - without a registration the plain ECC pass is used
            registrations = {}
        registration = {id(runs[k]): reg for k, reg in registrations.items()} or None
        return runs, registration

    # ------------------------------------------------------------------ compare_media

    async def compare_media(self, media: tuple[PreparedMedia, ...], params: CompareMediaParams, policy: InspectPolicy) -> BackendOutcome:
        a, b = media
        started = time.monotonic()
        if a.media_kind == "image" and b.media_kind == "image":
            outcome = await self.diff_media(media, DiffMediaParams(None, None, params.region, params.min_area, True), policy)  # type: ignore[attr-defined]
            raw = dict(outcome.result)
            raw.pop("timeline", None)
            return BackendOutcome({"media_kind": "image", "summary_lines": [raw.get("summary", "")], "frame_diff": raw, "images": []},
                                  outcome.artifacts, outcome.issues, {"backend": "compare_media image pair (frame differencing)"})
        mode = params.mode or "auto"
        if mode == "audio":
            return self._compare_audio_only(a, b, started)
        cfg = CompareConfig(time_budget_s=max(60.0, min(900.0, float(getattr(policy, "execution_timeout_s", 900.0)) - 60.0)))
        comparer = SourceComparer(self._face_identity(), cfg)
        try:
            report, images = comparer.compare(a.path, b.path, Event())
            engine = getattr(comparer, "last_engine", "near_duplicate")
            if engine == "registered":
                report["registration"] = getattr(comparer, "last_registration", None)
                report["summary_lines"].insert(1, "Matched after warping B onto A's framing (A shows B cropped, scaled, "
                                                  "letterboxed or framed differently).")
        except CompareUnavailable as exc:
            raise _unavailable(exc) from exc
        lines = list(report.pop("summary_lines"))
        runs_rows = report.get("correspondences") or []
        issues: list[BackendFailure] = []
        # sound at the matched moments
        if mode in {"auto", "auto_fallback"} and report.get("source_match") and a.profile.audio_tracks and b.profile.audio_tracks:
            try:
                runs = [Run([r["a_range_s"][0] + 0.25, r["a_range_s"][1] - 0.25],
                            [r["mapping"]["scale"] * (r["a_range_s"][0] + 0.25) + r["mapping"]["offset_s"],
                             r["mapping"]["scale"] * (r["a_range_s"][1] - 0.25) + r["mapping"]["offset_s"]])
                        for r in runs_rows if r["kind"] != "frozen"]
                sa, sb = self._audio_8k(a), self._audio_8k(b)
                audio = audio_compare(sa, sb, 8000, self._mapping_from_runs(runs), len(sa) / 8000)
                report["audio"] = audio
                for span in audio["spans_differ"][:6]:
                    what = {"differs": "the sound differs from B's sound at the same moment",
                            "sound_only_in_a": "A has sound where B is silent at the same moment",
                            "silent_in_a": "A is silent where B has sound at the same moment"}[span["state"]]
                    lines.append(f"Sound: A {span['a_range_s'][0]}-{span['a_range_s'][1]} s: {what} (B {span['b_range_s'][0]}-"
                                 f"{span['b_range_s'][1]} s; spectral similarity {span['median_similarity']}).")
                if audio["compared_windows"] and not audio["spans_differ"]:
                    lines.append(f"Sound: A's sound matches B's at the matched moments ({audio['same_windows']} of "
                                 f"{audio['compared_windows']} one-second windows the same).")
            except BackendFailure as exc:
                issues.append(BackendFailure(exc.code, f"sound comparison skipped: {exc}", stage=exc.stage, severity="error"))
            except Exception as exc:  # noqa: BLE001
                report["audio"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        if mode == "auto_fallback" and not report.get("source_match") and a.profile.audio_tracks and b.profile.audio_tracks:
            # 10-13 compact_tool_params：画面对不上（换了画面、配了别的视频）时只比声音，原 mode=audio 的结果附在后面
            try:
                sound = self._compare_audio_only(a, b, started).result
                report["audio_only"] = sound["audio"]
                lines.append("The pictures do not match, so the sound was compared on its own:")
                lines += list(sound["summary_lines"])
                report.setdefault("limitations", []).extend(sound.get("limitations") or [])
            except BackendFailure as exc:
                issues.append(BackendFailure(exc.code, f"sound-only comparison skipped: {exc}", stage=exc.stage, severity="error"))
            except Exception as exc:  # noqa: BLE001
                report["audio_only"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        # frame-exact differences inside one span
        diff_outcome = None
        target = params.range_a_s
        if target is None and report.get("source_match") and len(runs_rows) == 1 and runs_rows[0]["kind"] == "normal" \
                and (a.profile.duration_s or 999) <= 60 and abs((a.profile.duration_s or 0) - (b.profile.duration_s or 0)) <= 1.0:
            target = (0.0, float(a.profile.duration_s or 0.0))      # short equal pair (VideoSham style): whole A
        if target is not None and report.get("source_match"):
            mid = (target[0] + target[1]) / 2
            run = next((r for r in runs_rows if r["a_range_s"][0] - 0.5 <= mid <= r["a_range_s"][1] + 0.5), None)
            if run is not None and not run["mirrored"]:
                scale, offset = abs(float(run["mapping"]["scale"])), float(run["mapping"]["offset_s"])
                same_length = abs((a.profile.duration_s or 0) - (b.profile.duration_s or -9)) <= 0.1
                if same_length and abs(scale - 1.0) <= 0.12 and abs(offset) <= 0.5:
                    # equal-length pair whose fit is close to identity (VideoSham 4126: a person added in B pulls the
                    # thumbnail fit to scale 0.91 / offset 0.26): difference frame by frame on the same timeline
                    scale, offset = 1.0, 0.0
                mapping = DiffMapping(offset, max(1e-3, scale))
                try:
                    diff_outcome = await self.diff_media(media, DiffMediaParams(mapping, (max(0.0, target[0]), min(target[1], a.profile.duration_s or target[1])),  # type: ignore[attr-defined]
                                                                                  params.region, params.min_area, True), policy)
                    diff = dict(diff_outcome.result)
                    diff.pop("timeline", None)
                    report["frame_diff"] = diff
                    if diff.get("differs"):
                        lines = [x for x in lines if not x.startswith("No added, missing")]
                    lines.append(f"Frame-exact differences in A {target[0]:.2f}-{target[1]:.2f} s: {diff.get('summary', '')}")
                except BackendFailure as exc:
                    issues.append(BackendFailure(exc.code, f"frame-exact differencing skipped: {exc}", stage=exc.stage))
            elif params.range_a_s is not None:
                lines.append(f"range_a_s {target[0]}-{target[1]} s does not lie in one span of A that matches B unmirrored; "
                             "no frame-exact differencing was done.")
        lines = prioritise_compare_lines(lines + self._describe(images))
        artifacts, rows = self._report_images(a, b, images, "cmp", "INSPECT/compare_media")
        if diff_outcome is not None:
            artifacts += list(diff_outcome.artifacts)
            issues += [x for x in diff_outcome.issues if x.severity == "warning"]
        report["engine"] = engine
        report["timings_s"] = {**report.get("timings_s", {}), "total_s": round(time.monotonic() - started, 1)}
        result = {"media_kind": "video", "summary_lines": lines, **report, "images": rows}
        describer = getattr(self, "describer", None)
        return BackendOutcome(result, tuple(artifacts), tuple(issues),
                              {"backend": "compare_media", "model": comparer.identity(),
                               "describer": describer.identity() if describer is not None else None,
                               "limitations": report.get("limitations", [])})

    def _compare_audio_only(self, a: PreparedMedia, b: PreparedMedia, started: float) -> BackendOutcome:
        sa, sb = self._audio_8k(a), self._audio_8k(b)
        offset, score = audio_offset(sa, sb, 8000)
        audio = audio_compare(sa, sb, 8000, lambda t: t + offset, len(sa) / 8000)
        lines = [f"Sound of A lines up with B at B = A {offset:+.2f} s (envelope correlation {score:.2f})."]
        if audio["compared_windows"]:
            share = audio["same_windows"] / audio["compared_windows"]
            lines.append(f"{audio['same_windows']} of {audio['compared_windows']} one-second windows sound the same ({100 * share:.0f}%).")
        for span in audio["spans_differ"][:8]:
            lines.append(f"Sound: A {span['a_range_s'][0]}-{span['a_range_s'][1]} s differs from B {span['b_range_s'][0]}-"
                         f"{span['b_range_s'][1]} s ({span['state']}, similarity {span['median_similarity']}).")
        return BackendOutcome({"media_kind": "audio", "summary_lines": lines, "audio": {**audio, "offset_s": round(offset, 2),
                               "offset_score": round(score, 3)}, "images": [],
                               "limitations": ["Sound is compared as spectral envelopes in one-second windows at a single global "
                                               "offset; a re-timed or re-ordered track is only partly matched."],
                               "timings_s": {"total_s": round(time.monotonic() - started, 1)}},
                              details={"backend": "compare_media audio (log band energies, windowed)"})

    # ------------------------------------------------------------------ video_structure

    async def video_structure(self, media: tuple[PreparedMedia, ...], params: VideoStructureParams, policy: InspectPolicy) -> BackendOutcome:
        from ..video_structure import analyse

        budget = max(60.0, min(600.0, float(getattr(policy, "execution_timeout_s", 600.0)) - 30.0))
        try:
            report, images = analyse(media[0].path, range_s=params.range_s, cancel=Event(), time_budget_s=budget)
        except CompareUnavailable as exc:
            raise _unavailable(exc) from exc
        lines = list(report.pop("summary_lines"))
        # 10-14：约 10 s 镜头的三帧图交给描述模型，描述接在该镜头那一行后面
        for k, img in enumerate(images):
            if img.get("kind") != "shot":
                continue
            for line in self._describe([img]):
                text = line.split(": ", 1)[1] if ": " in line else line
                prefix = f"Shot {int(img.get('index') or 0) + 1} ("
                for j, existing in enumerate(lines):
                    if existing.startswith(prefix):
                        lines[j] = f"{existing} Visual check of comparison image {k + 1}: {text}"
                        break
        artifacts, rows = self._report_images(media[0], media[0], images, "shots", "INSPECT/video_structure")
        return BackendOutcome({"summary_lines": lines, **report, "images": rows}, tuple(artifacts),
                              details={"backend": "video_structure (HSV histogram cuts, frame differences)",
                                       "limitations": report.get("limitations", [])})

    # ------------------------------------------------------------------ asr: sound timeline

    def sound_timeline(self, samples: np.ndarray, rate: int, start: float, speech: list[dict]) -> tuple[list[dict], str]:
        """speech / other_sound / silence spans from frame energy and the transcript's speech segments.

        With FOREN_AUDIO_TAGGER_MODEL (a local transformers audio-classification model such as an AudioSet AST) each
        other_sound span gets the tagger's top label; without it the span is only 'other sound'."""

        hop = int(0.05 * rate)
        if len(samples) < hop:
            return [], "energy"
        frames = samples[: len(samples) // hop * hop].reshape(-1, hop)
        rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
        quiet = max(1e-4, float(np.percentile(rms, 95)) * 0.03)
        t = start + np.arange(len(rms)) * 0.05
        is_speech = np.zeros(len(rms), dtype=bool)
        for seg in speech:
            is_speech |= (t >= seg["start_s"]) & (t < seg["end_s"])
        kind = np.where(is_speech, 2, np.where(rms > quiet, 1, 0))
        names = {0: "silence", 1: "other_sound", 2: "speech"}
        spans: list[dict] = []
        i = 0
        while i < len(kind):
            j = i
            while j + 1 < len(kind) and kind[j + 1] == kind[i]:
                j += 1
            a, b = float(t[i]), float(t[j] + 0.05)
            if spans and (b - a < 0.5) and spans[-1]["kind"] != "speech":
                spans[-1]["end_s"] = round(b, 2)       # short blips merge into the previous span
            else:
                spans.append({"start_s": round(a, 2), "end_s": round(b, 2), "kind": names[int(kind[i])], "label": None})
            i = j + 1
        merged: list[dict] = []
        for s in spans:
            if merged and merged[-1]["kind"] == s["kind"]:
                merged[-1]["end_s"] = s["end_s"]
            else:
                merged.append(s)
        method = "energy + transcript speech segments"
        tagger_path = os.environ.get("FOREN_AUDIO_TAGGER_MODEL")
        if tagger_path:
            try:
                from transformers import pipeline

                with self._model_lock:  # type: ignore[attr-defined]
                    tagger = getattr(self, "_tagger", None)
                    if tagger is None:
                        tagger = pipeline("audio-classification", model=tagger_path, device=-1)
                        self._tagger = tagger
                for s in [x for x in merged if x["kind"] == "other_sound" and x["end_s"] - x["start_s"] >= 1.0][:12]:
                    lo, hi = int((s["start_s"] - start) * rate), int((min(s["end_s"], s["start_s"] + 10) - start) * rate)
                    out = tagger({"raw": samples[lo:hi].astype(np.float32), "sampling_rate": rate}, top_k=1)
                    if out:
                        s["label"] = str(out[0]["label"])
                method += f" + audio tagger ({os.path.basename(tagger_path.rstrip('/'))})"
            except Exception as exc:  # noqa: BLE001 - labels are optional
                method += f" (audio tagger failed: {type(exc).__name__})"
        return merged[:400], method


# ---------------------------------------------------------------------------------------------------------------------
# 10-13 compare_media summary order. evidM27a: the HI observation shows the first 20 lines and the history keeps 8, and
# the comparer wrote them in a fixed order — up to 8 short re-timed spans first, then added footage, changed regions,
# faces, and the describer's "Visual check" lines last. In video_208 the HI saw eight 1.5 s "plays B backwards / slower"
# spans (3 half-second samples each), wrote them up as forgery points, and never saw "B: Wrench; A: Hammer". Now the
# content comes first, each visual check is attached to the line it describes, and short re-timed spans share one line.

_TIMING = re.compile(r"^A (\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) s plays B (backwards|slower \(([-\d.]+)x|faster \(([-\d.]+)x)")
_CHECK = re.compile(r"^Visual check of comparison image (\d+) \((added footage|changed region|person) (\d+)\): (.*)$")
SHORT_RETIMED_S = 2.0


def _group(line: str) -> int:
    if line.startswith(("B matches A", "B is probably not", "Matched after warping", "A uses B")):
        return 0
    if line.startswith(("Not in B:", "Changed region:", "Face identity", "Frame-exact", "Visual check")) or (
            line.startswith("Sound: A ") and not line.startswith("Sound: A's sound matches")):
        return 1
    if line.startswith(("Missing from A:", "Order differs:", "Repeated:")) or " lines up in time with B " in line \
            or "held still" in line:
        return 2
    if _TIMING.match(line):
        return 3
    return 4


def prioritise_compare_lines(lines: list[str]) -> list[str]:
    """Content first (added footage, changed regions, faces, sound), then structure, then timing; facts unchanged."""

    lines = [str(x) for x in lines]
    checks: list[tuple[str, int, str, str]] = []
    rest: list[str] = []
    for line in lines:
        m = _CHECK.match(line)
        if m:
            checks.append((m.group(2), int(m.group(3)), m.group(1), m.group(4)))
        else:
            rest.append(line)
    prefixes = {"added footage": "Not in B:", "changed region": "Changed region:"}
    unattached: list[str] = []
    for label, index, image, text in checks:
        if label == "person":
            targets = [k for k, x in enumerate(rest) if x.startswith("Face identity") and f"person {index} (" in x]
        else:
            numbered = [k for k, x in enumerate(rest) if x.startswith(prefixes[label])]
            targets = [numbered[index - 1]] if 0 < index <= len(numbered) else []
        if targets:
            rest[targets[0]] = f"{rest[targets[0]].rstrip()} Visual check of comparison image {image}: {text}"
        else:
            unattached.append(f"Visual check of comparison image {image} ({label} {index}): {text}")
    short: list[str] = []
    kept: list[str] = []
    for line in rest:
        m = _TIMING.match(line)
        if m and float(m.group(2)) - float(m.group(1)) <= SHORT_RETIMED_S:
            how = "backwards" if m.group(3) == "backwards" else f"{m.group(4) or m.group(5)}x"
            short.append(f"A {m.group(1)}-{m.group(2)} s {how}")
        else:
            kept.append(line)
    ordered = sorted(kept + unattached, key=_group)            # stable: original order inside each group
    if short:
        ordered.append(f"Short spans of {SHORT_RETIMED_S:.0f} s or less that play B at another speed or backwards ("
                       + "; ".join(short) + "). Each rests on 3-4 half-second samples, so it can also come from "
                       "imprecise alignment around a cut.")
    return ordered
