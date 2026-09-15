"""Reproducible VideoSham pilot sampling, capability preflight, and CLI execution."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cli_teacher_rollout import CodexCLITeacher
from causal_probe_suite import qualify_and_write_causal_packet
from evidence_choice_search import build_evidence_choice_teacher
from generation_admission import generation_admission_summary, generation_failure_metadata
from interactive_harness import EVIDENCE_INTEGRITY_TOOL_NAMES, EvidenceAtom, with_evidence_integrity_tools
from llm_teacher_rollout import _write_bundle, build_llm_teacher_bundle, validate_llm_teacher_bundle
from pipeline import ROOT, decode, media_metadata, write_json, write_jsonl
from sft_v2_sidecar import compile_transition_sidecar
from step_utility_auditor import apply_step_utility_filter, audit_step_utility
from unified_trajectory_rubric import evaluate_bundle as evaluate_unified_quality, load_saved_bundle

import av
import numpy as np
from PIL import Image


DATASET_ROOT = ROOT / "datasets" / "videosham"
METADATA_PATH = DATASET_ROOT / "VideoSourceandEdits.csv"
SOURCE_ROOT = DATASET_ROOT / "media" / "source"
EDITED_ROOT = DATASET_ROOT / "media" / "edited"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "batch"

FAMILY_TO_REQUIRED_CAPABILITIES = {
    "temporal_omission": ["inspect_media", "detect_temporal_edit"],
    "temporal_warp": ["inspect_media", "detect_temporal_edit"],
    "audio_replacement": ["inspect_media", "compare_audio_tracks"],
    "spatial_removal": ["inspect_media", "compare_spatial_regions"],
    "spatial_insertion": ["inspect_media", "compare_spatial_regions"],
    "appearance_change": ["inspect_media", "compare_spatial_regions"],
    "text_edit": ["inspect_media", "compare_spatial_regions"],
}
COMMON_STATE_TOOLS = ["create_candidate_claim", "bind_evidence", "set_field_decision", "stop"]
MULTIFAMILY_SCHEMA = {
    "schema_id": "paired_media_difference_v1",
    "required_fields": [
        "source_id",
        "claim_family",
        "manipulation_type",
        "direction",
        "comparison_scope_s",
        "affected_source_interval_s",
        "affected_suspect_interval_s",
    ],
    "target_fields": [
        "claim_family",
        "manipulation_type",
        "direction",
        "affected_source_interval_s",
        "affected_suspect_interval_s",
    ],
    "invariant_fields": ["source_id", "comparison_scope_s"],
    "fields": {
        "source_id": {
            "types": ["str"],
            "definition": "Neutral identifier of the paired source media.",
            "acceptable_evidence_kinds": ["paired_media_identity"],
        },
        "claim_family": {
            "types": ["str"],
            "enum": ["paired_media_difference"],
            "definition": "A material source-to-suspect media difference.",
            "acceptable_evidence_kinds": [
                "temporal_edit_measurement",
                "audio_track_comparison",
                "spatial_region_comparison",
            ],
        },
        "manipulation_type": {
            "types": ["str"],
            "enum": ["temporal_edit", "audio_edit", "spatial_edit"],
            "definition": "The modality-level edit type supported by a real comparison tool.",
            "acceptable_evidence_kinds": [
                "temporal_edit_measurement",
                "audio_track_comparison",
                "spatial_region_comparison",
            ],
        },
        "direction": {
            "types": ["str"],
            "enum": ["source_to_edited"],
            "definition": "The suspect is the edited derivative of the paired source.",
            "acceptable_evidence_kinds": [
                "temporal_edit_measurement",
                "audio_track_comparison",
                "spatial_region_comparison",
            ],
        },
        "comparison_scope_s": {
            "types": ["list"],
            "definition": "Dataset-provided neutral source locator.",
            "acceptable_evidence_kinds": ["case_scope_locator"],
        },
        "affected_source_interval_s": {
            "types": ["list"],
            "definition": "Measured source interval used by the comparison.",
            "acceptable_evidence_kinds": [
                "temporal_edit_measurement",
                "audio_track_comparison",
                "spatial_region_comparison",
            ],
        },
        "affected_suspect_interval_s": {
            "types": ["list"],
            "definition": "Measured suspect interval covered by the comparison.",
            "acceptable_evidence_kinds": [
                "temporal_edit_measurement",
                "audio_track_comparison",
                "spatial_region_comparison",
            ],
        },
    },
}

MULTIFAMILY_TOOL_REGISTRY = [
    {"name": "inspect_media", "domain": "external_media"},
    {"name": "detect_temporal_edit", "domain": "external_media"},
    {"name": "compare_audio_tracks", "domain": "external_media"},
    {"name": "compare_spatial_regions", "domain": "external_media"},
    {"name": "create_candidate_claim", "domain": "harness_state"},
    {"name": "bind_evidence", "domain": "harness_state"},
    {"name": "set_field_decision", "domain": "harness_state"},
    {"name": "stop", "domain": "terminal"},
]
MULTIFAMILY_CAUSAL_TOOL_REGISTRY = with_evidence_integrity_tools(
    MULTIFAMILY_TOOL_REGISTRY
)

VIDEOSHAM_ADAPTER_ID = "paired_media_difference_scope_v2"
IMPLEMENTED_ADAPTERS = {
    "temporal_omission": VIDEOSHAM_ADAPTER_ID,
    "temporal_warp": VIDEOSHAM_ADAPTER_ID,
    "audio_replacement": VIDEOSHAM_ADAPTER_ID,
    "spatial_removal": VIDEOSHAM_ADAPTER_ID,
    "spatial_insertion": VIDEOSHAM_ADAPTER_ID,
    "appearance_change": VIDEOSHAM_ADAPTER_ID,
    "text_edit": VIDEOSHAM_ADAPTER_ID,
}
IMPLEMENTED_TOOLS = {item["name"] for item in MULTIFAMILY_CAUSAL_TOOL_REGISTRY}


def _suspect_interval_covered_by_comparison(
    comparison_scope_s: list[float], suspect_duration_s: float
) -> list[float]:
    """Report the suspect span actually compared, not the entire media duration."""

    start, end = map(float, comparison_scope_s)
    duration = max(0.0, float(suspect_duration_s))
    # When the suspect still contains the requested timeline, both streams are
    # compared inside the same visible window. A genuinely shortened suspect
    # is compared over its complete available span instead.
    if duration >= end - 0.25:
        return [round(max(0.0, start), 3), round(min(end, duration), 3)]
    return [0.0, round(duration, 3)]


class MultiFamilyVideoEnvironment:
    """Measurement-only VideoSham adapter; it never reads the editor task or manipulation label."""

    def __init__(self, *, manifest: dict[str, Any], source_frames: Any, suspect_frames: Any, source_path: Path, suspect_path: Path) -> None:
        self.manifest = manifest
        self.source_frames = source_frames
        self.suspect_frames = suspect_frames
        self.source_path = source_path
        self.suspect_path = suspect_path
        self.call_counts = {name: 0 for name in [
            "inspect_media", "detect_temporal_edit", "compare_audio_tracks", "compare_spatial_regions"
        ]}
        self._triage: dict[str, Any] | None = None
        self._visual_cache: dict[str, Any] | None = None
        self._audio_cache: dict[str, Any] | None = None
        self._repair_due = False
        self._context_recheck_due = False
        self._last_successful_tool: str | None = None

    def eligible_tool_names(self) -> list[str]:
        if self.call_counts["inspect_media"] == 0 or getattr(
            self, "_context_recheck_due", False
        ):
            return ["inspect_media"]
        # All three measurements are implemented and executable.  Exposing
        # only the router-selected modality made the Harness choose the
        # semantic action for the policy.
        choices = [
            "detect_temporal_edit",
            "compare_audio_tracks",
            "compare_spatial_regions",
        ]
        if getattr(self, "_repair_due", False):
            return choices
        return [name for name in choices if self.call_counts[name] == 0]

    def on_evidence_dropout(self, _evidence_id: str, affected_fields: list[str]) -> None:
        if set(affected_fields).intersection(MULTIFAMILY_SCHEMA["target_fields"]):
            self._repair_due = True

    def on_evidence_challenge(
        self,
        requested_check: str,
        _evidence_ids: list[str],
        _defect_codes: list[str],
    ) -> None:
        if requested_check == "reinspect_context":
            self._context_recheck_due = True
        elif requested_check == "remeasure_evidence":
            self._repair_due = True

    def _validate_pair_args(self, arguments: dict[str, Any]) -> None:
        expected = {
            "source_ref": "source",
            "suspect_ref": "suspect",
            "comparison_scope_s": self.manifest["comparison_scope_s"],
        }
        if arguments != expected:
            raise ValueError(f"pair comparison requires exactly {expected}")

    def _scope_features(self) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.manifest["comparison_scope_s"]
        source_mask = (self.source_frames.times >= start) & (self.source_frames.times <= end)
        source = self.source_frames.features[source_mask]
        if len(source) < 2:
            raise ValueError("visible comparison scope has too few source frames")
        # Most release pairs retain the original timeline, so compare the same
        # visible interval on both sides. A shortened suspect cannot cover the
        # source interval; in that case its full clip is the corresponding span.
        if self.suspect_frames.duration_s >= end - 0.25:
            suspect_mask = (self.suspect_frames.times >= start) & (self.suspect_frames.times <= end)
            suspect = self.suspect_frames.features[suspect_mask]
        else:
            suspect = self.suspect_frames.features
        if len(suspect) < 2:
            raise ValueError("visible comparison scope has too few suspect frames")
        return source, suspect

    def _visual_metrics(self) -> dict[str, Any]:
        if self._visual_cache is not None:
            return dict(self._visual_cache)
        source, suspect = self._scope_features()
        similarity = suspect @ source.T
        nearest = np.argmax(similarity, axis=1)
        nearest_similarity = similarity[np.arange(len(suspect)), nearest]
        steps = np.diff(nearest.astype(np.float64))
        positive = steps[steps > 0]
        step_cv = float(np.std(positive) / max(np.mean(positive), 1e-6)) if len(positive) else 10.0
        duration_ratio = self.source_frames.duration_s / max(self.suspect_frames.duration_s, 1e-6)
        negative_fraction = float(np.mean(steps < 0)) if len(steps) else 0.0
        reused_fraction = float(np.mean(steps <= 0)) if len(steps) else 0.0
        p05_similarity = float(np.quantile(nearest_similarity, 0.05))
        temporal_detected = bool(
            abs(duration_ratio - 1.0) > 0.08
            or (
                p05_similarity > 0.9995
                and (negative_fraction > 0.02 or (reused_fraction > 0.08 and step_cv > 0.8))
            )
        )
        rgb = self._rgb_tile_metrics()
        spatial_detected = bool(
            not temporal_detected
            and (p05_similarity < 0.9995 or rgb["localized_rgb_edit_detected"])
        )
        self._visual_cache = {
            "mean_nearest_similarity": round(float(np.mean(nearest_similarity)), 4),
            "p05_nearest_similarity": round(p05_similarity, 4),
            "negative_step_fraction": round(negative_fraction, 4),
            "reused_or_reversed_step_fraction": round(reused_fraction, 4),
            "positive_step_cv": round(step_cv, 4),
            "source_to_suspect_duration_ratio": round(duration_ratio, 4),
            "temporal_edit_detected": temporal_detected,
            "spatial_edit_detected": spatial_detected,
            **rgb,
        }
        return dict(self._visual_cache)

    @staticmethod
    def _sample_rgb(path: Path, scope: list[float], max_frames: int = 24) -> np.ndarray:
        frames: list[np.ndarray] = []
        start, end = scope
        targets = np.linspace(start, end, max_frames)
        target_index = 0
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                time_s = float(frame.time) if frame.time is not None else float(index)
                if time_s < start or time_s > end:
                    continue
                if target_index >= len(targets) or time_s + 1e-6 < targets[target_index]:
                    continue
                image = Image.fromarray(frame.to_ndarray(format="rgb24")).resize(
                    (192, 108), Image.Resampling.BILINEAR
                )
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                while target_index < len(targets) and time_s + 1e-6 >= targets[target_index]:
                    frames.append(pixels)
                    target_index += 1
        if not frames:
            return np.empty((0, 108, 192, 3), dtype=np.float32)
        return np.stack(frames)

    def _rgb_tile_metrics(self) -> dict[str, Any]:
        start, end = self.manifest["comparison_scope_s"]
        suspect_scope = (
            [start, end]
            if self.suspect_frames.duration_s >= end - 0.25
            else [0.0, float(self.suspect_frames.duration_s)]
        )
        source = self._sample_rgb(self.source_path, [start, end])
        suspect = self._sample_rgb(self.suspect_path, suspect_scope)
        if not len(source) or not len(suspect):
            return {
                "rgb_frames_compared": 0,
                "rgb_tile_p95_mae": 0.0,
                "rgb_tile_max_mae": 0.0,
                "rgb_changed_tile_fraction": 0.0,
                "localized_rgb_edit_detected": False,
            }
        paired = min(len(source), len(suspect))
        source = source[np.linspace(0, len(source) - 1, paired).round().astype(int)]
        suspect = suspect[np.linspace(0, len(suspect) - 1, paired).round().astype(int)]
        residual = np.abs(source - suspect)
        # Remove the per-frame/channel median codec drift, while retaining local
        # color and text edits. A 12x16 grid retains small inscriptions that are
        # almost invisible to the global 64x36 grayscale alignment feature.
        drift = np.median(source - suspect, axis=(1, 2), keepdims=True)
        residual = np.abs((source - suspect) - drift)
        tile_scores = residual.reshape(paired, 12, 9, 16, 12, 3).mean(axis=(2, 4, 5))
        p95 = float(np.quantile(tile_scores, 0.95))
        maximum = float(np.max(tile_scores))
        changed_fraction = float(np.mean(tile_scores > 0.03))
        detected = bool(p95 > 0.025 or maximum > 0.025 or changed_fraction > 0.04)
        return {
            "rgb_frames_compared": paired,
            "rgb_tile_p95_mae": round(p95, 4),
            "rgb_tile_max_mae": round(maximum, 4),
            "rgb_changed_tile_fraction": round(changed_fraction, 4),
            "localized_rgb_edit_detected": detected,
        }

    @staticmethod
    def _audio_series(path: Path, scope: list[float] | None) -> tuple[np.ndarray, np.ndarray]:
        envelope = []
        roughness = []
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return np.asarray([]), np.asarray([])
            stream = container.streams.audio[0]
            for index, frame in enumerate(container.decode(audio=0)):
                time_s = float(frame.time) if frame.time is not None else index * frame.samples / float(frame.sample_rate)
                if scope and (time_s < scope[0] or time_s > scope[1]):
                    continue
                samples = frame.to_ndarray().astype(np.float32)
                if samples.ndim > 1:
                    samples = samples.mean(axis=0)
                samples = samples.reshape(-1)
                if not len(samples):
                    continue
                peak = max(float(np.max(np.abs(samples))), 1.0)
                samples = samples / peak
                envelope.append(float(np.sqrt(np.mean(samples * samples))))
                roughness.append(float(np.mean(np.abs(np.diff(samples)))))
        return np.asarray(envelope), np.asarray(roughness)

    @staticmethod
    def _series_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 3 or len(right) < 3:
            return 0.0
        grid = np.linspace(0.0, 1.0, 128)
        a = np.interp(grid, np.linspace(0.0, 1.0, len(left)), left)
        b = np.interp(grid, np.linspace(0.0, 1.0, len(right)), right)
        if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
            return float(np.isclose(np.mean(a), np.mean(b), atol=1e-4))
        return float(np.clip((np.corrcoef(a, b)[0, 1] + 1.0) / 2.0, 0.0, 1.0))

    def _audio_metrics(self) -> dict[str, Any]:
        if self._audio_cache is not None:
            return dict(self._audio_cache)
        source_env, source_rough = self._audio_series(self.source_path, self.manifest["comparison_scope_s"])
        suspect_env, suspect_rough = self._audio_series(self.suspect_path, None)
        available = bool(len(source_env) and len(suspect_env))
        envelope_similarity = self._series_similarity(source_env, suspect_env) if available else 0.0
        roughness_similarity = self._series_similarity(source_rough, suspect_rough) if available else 0.0
        combined = (envelope_similarity + roughness_similarity) / 2.0
        self._audio_cache = {
            "source_audio_available": bool(len(source_env)),
            "suspect_audio_available": bool(len(suspect_env)),
            "envelope_similarity": round(envelope_similarity, 4),
            "roughness_similarity": round(roughness_similarity, 4),
            "combined_audio_similarity": round(combined, 4),
            "audio_edit_detected": bool(available and combined < 0.72),
        }
        return dict(self._audio_cache)

    def _triage_metrics(self) -> dict[str, Any]:
        if self._triage is None:
            visual = self._visual_metrics()
            audio = self._audio_metrics()
            if visual["temporal_edit_detected"]:
                primary = "temporal"
            elif visual["spatial_edit_detected"]:
                primary = "spatial"
            elif audio["audio_edit_detected"]:
                primary = "audio"
            else:
                primary = None
            self._triage = {
                "candidate_modalities": [
                    modality
                    for modality, detected in (
                        ("temporal", visual["temporal_edit_detected"]),
                        ("spatial", visual["spatial_edit_detected"]),
                        ("audio", audio["audio_edit_detected"]),
                    )
                    if detected
                ],
                "primary_modality": primary,
                "duration_ratio": visual["source_to_suspect_duration_ratio"],
                "visual_p05_similarity": visual["p05_nearest_similarity"],
                "rgb_tile_p95_mae": visual["rgb_tile_p95_mae"],
                "rgb_tile_max_mae": visual["rgb_tile_max_mae"],
                "rgb_changed_tile_fraction": visual["rgb_changed_tile_fraction"],
                "audio_similarity": audio["combined_audio_similarity"],
            }
        return dict(self._triage)

    def _student_visible_triage_metrics(self) -> dict[str, Any]:
        """Expose neutral measurements without the router's categorical answer."""

        return {
            key: value
            for key, value in self._triage_metrics().items()
            if key not in {"candidate_modalities", "primary_modality"}
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_media":
            if arguments.get("media_refs") != ["source", "suspect"]:
                raise ValueError("inspect_media requires media_refs=['source','suspect']")
            self.call_counts[name] += 1
            self._context_recheck_due = False
            index = self.call_counts[name]
            source_meta = media_metadata(self.source_frames, self.source_path)
            suspect_meta = media_metadata(self.suspect_frames, self.suspect_path)
            source_meta = {"media_ref": "source", **{key: value for key, value in source_meta.items() if key != "path"}}
            suspect_meta = {"media_ref": "suspect", **{key: value for key, value in suspect_meta.items() if key != "path"}}
            execution_id = f"EXEC-INSPECT-{index:03d}"
            return {
                "observation_id": f"OBS-INSPECT-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": dict(arguments),
                "result": {
                    "source_media": source_meta,
                    "suspect_media": suspect_meta,
                    "comparison_scope_s": self.manifest["comparison_scope_s"],
                    "modality_screen": self._student_visible_triage_metrics(),
                },
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-PAIR-{index:03d}",
                        kind="paired_media_identity",
                        supports_fields=["source_id"],
                        observation={"source_id": self.manifest["source_id"]},
                        provenance={"generated_by": execution_id, "method": "visible_pair_identity"},
                    ),
                    EvidenceAtom(
                        evidence_id=f"EA-SCOPE-{index:03d}",
                        kind="case_scope_locator",
                        supports_fields=["comparison_scope_s"],
                        observation={"comparison_scope_s": self.manifest["comparison_scope_s"]},
                        provenance={"generated_by": execution_id, "method": "visible_dataset_locator"},
                    ),
                ],
            }

        self._validate_pair_args(arguments)
        if name not in {"detect_temporal_edit", "compare_audio_tracks", "compare_spatial_regions"}:
            raise ValueError(f"unknown external tool: {name}")
        metrics = self._audio_metrics() if name == "compare_audio_tracks" else self._visual_metrics()
        expected_modality = {
            "detect_temporal_edit": "temporal",
            "compare_audio_tracks": "audio",
            "compare_spatial_regions": "spatial",
        }[name]
        signal_key = {
            "detect_temporal_edit": "temporal_edit_detected",
            "compare_audio_tracks": "audio_edit_detected",
            "compare_spatial_regions": "spatial_edit_detected",
        }[name]
        kind = {
            "detect_temporal_edit": "temporal_edit_measurement",
            "compare_audio_tracks": "audio_track_comparison",
            "compare_spatial_regions": "spatial_region_comparison",
        }[name]
        if self._triage_metrics()["primary_modality"] != expected_modality or not metrics.get(signal_key):
            self.call_counts[name] += 1
            index = self.call_counts[name]
            execution_id = f"EXEC-{name.upper().replace('_', '-')}-NEGATIVE-{index:03d}"
            negative = {
                "tested_modality": expected_modality,
                "material_difference_detected": False,
                "measurements": copy.deepcopy(metrics),
            }
            return {
                "observation_id": f"OBS-{name.upper().replace('_', '-')}-NEGATIVE-{index:03d}",
                "execution_id": execution_id,
                "tool_name": name,
                "arguments": dict(arguments),
                "goal_satisfied": False,
                "result": copy.deepcopy(negative),
                "evidence_atoms": [
                    EvidenceAtom(
                        evidence_id=f"EA-NEGATIVE-{name.upper().replace('_', '-')}-{index:03d}",
                        kind=f"negative_{kind}",
                        supports_fields=[],
                        observation=copy.deepcopy(negative),
                        provenance={
                            "generated_by": execution_id,
                            "method": name,
                            "inputs": ["source", "suspect"],
                        },
                    )
                ],
            }
        self.call_counts[name] += 1
        index = self.call_counts[name]
        self._repair_due = False
        self._last_successful_tool = name
        execution_id = f"EXEC-{name.upper().replace('_', '-')}-{index:03d}"
        suspect_interval = _suspect_interval_covered_by_comparison(
            self.manifest["comparison_scope_s"], self.suspect_frames.duration_s
        )
        observation = {
            **metrics,
            "material_difference_detected": True,
            "source_role": "source",
            "suspect_role": "edited",
            "source_comparison_scope_s": list(self.manifest["comparison_scope_s"]),
            "suspect_comparison_scope_s": suspect_interval,
            "affected_source_interval_s": self.manifest["comparison_scope_s"],
            "affected_suspect_interval_s": suspect_interval,
        }
        return {
            "observation_id": f"OBS-{name.upper().replace('_', '-')}-{index:03d}",
            "execution_id": execution_id,
            "tool_name": name,
            "arguments": dict(arguments),
            "result": {"measurements": metrics},
            "evidence_atoms": [
                EvidenceAtom(
                    evidence_id=f"EA-DIFF-{name.upper().replace('_', '-')}-{index:03d}",
                    kind=kind,
                    supports_fields=list(MULTIFAMILY_SCHEMA["target_fields"]),
                    observation=observation,
                    provenance={
                        "generated_by": execution_id,
                        "method": name,
                        # Keep teacher-visible provenance stable and machine-neutral. The
                        # harness resolves these logical refs locally; filesystem paths
                        # are neither needed for the policy decision nor safe to export.
                        "inputs": ["source", "suspect"],
                    },
                )
            ],
        }


def _rows() -> list[dict[str, str]]:
    with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select_rows(
    rows: list[dict[str, str]],
    *,
    seed: int,
    sample_size: int,
    case_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """Select a reproducible random pilot or an explicit capability-coverage slice."""
    if not case_ids:
        return random.Random(seed).sample(rows, sample_size)
    normalized = [str(case_id).removeprefix("videosham-") for case_id in case_ids]
    by_number = {_case_number(row): row for row in rows}
    missing = [case_id for case_id in normalized if case_id not in by_number]
    if missing:
        raise ValueError(f"requested VideoSham case IDs are unavailable: {missing}")
    return [by_number[case_id] for case_id in normalized]


def _case_number(row: dict[str, str]) -> str:
    return Path(row["Video-Name"]).stem


def _time_window(task: str) -> list[float] | None:
    pattern = re.compile(r"(\d{1,2})\s*:\s*(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*:\s*(\d{1,2})", re.I)
    match = pattern.search(task)
    if match:
        start = int(match.group(1)) * 60 + int(match.group(2))
        end = int(match.group(3)) * 60 + int(match.group(4))
        return [float(start), float(end)] if end > start else None
    first_seconds = re.search(r"first\s+(\d+(?:\.\d+)?)\s*seconds?", task, re.I)
    if first_seconds:
        return [0.0, float(first_seconds.group(1))]
    return None


def _source_id(row: dict[str, str]) -> str:
    match = re.search(r"vimeo\.com/(\d+)", row.get("Name/Id in source dataset", ""), re.I)
    return f"vimeo-{match.group(1)}" if match else f"videosham-source-{_case_number(row)}"


def localize_comparison_scope(annotation_scope: list[float], source_duration_s: float) -> list[float]:
    """Map an original-video locator onto VideoSham's sometimes pre-clipped media timeline."""
    start, end = annotation_scope
    annotated_span = end - start
    if start >= source_duration_s or (start > 0 and abs(annotated_span - source_duration_s) <= 0.75):
        return [0.0, round(float(source_duration_s), 3)]
    return [float(start), round(min(float(end), float(source_duration_s)), 3)]


def capability_family(row: dict[str, str]) -> str:
    label = row["Manipulation"].lower()
    task = row["Task"].lower()
    if "frame duplication" in label:
        if re.search(r"drop\w*\s+(?:some\s+)?frames|frame\s+dropping", task):
            return "temporal_omission"
        return "temporal_warp"
    if "audio" in label:
        return "audio_replacement"
    if "removing an entity" in label:
        return "spatial_removal"
    if "adding an entity" in label:
        return "spatial_insertion"
    if "background/color" in label:
        return "appearance_change"
    if "text replaced" in label:
        return "text_edit"
    return "unknown"


def preflight_case(row: dict[str, str]) -> dict[str, Any]:
    number = _case_number(row)
    family = capability_family(row)
    window = _time_window(row["Task"])
    source_path = SOURCE_ROOT / f"{number}_unedited.mp4"
    suspect_path = EDITED_ROOT / f"{number}_processed.mp4"
    required = FAMILY_TO_REQUIRED_CAPABILITIES.get(family, []) + COMMON_STATE_TOOLS
    missing = [name for name in required if name not in IMPLEMENTED_TOOLS]
    reasons = []
    if family not in IMPLEMENTED_ADAPTERS:
        reasons.append(f"adapter_missing:{family}")
    if window is None:
        reasons.append("dataset_edit_window_unparseable")
    if not source_path.exists():
        reasons.append("source_media_missing")
    if not suspect_path.exists():
        reasons.append("suspect_media_missing")
    return {
        "case_id": f"videosham-{number}",
        "video_name": row["Video-Name"],
        "capability_family": family,
        "required_capabilities": required,
        "missing_capabilities": missing,
        "adapter_id": IMPLEMENTED_ADAPTERS.get(family),
        "comparison_scope_s": window,
        "source_path": str(source_path),
        "suspect_path": str(suspect_path),
        "cli_eligible": not reasons,
        "preflight_reasons": reasons,
    }


def compile_case(row: dict[str, str], preflight: dict[str, Any], case_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    number = _case_number(row)
    source_path = Path(preflight["source_path"])
    suspect_path = Path(preflight["suspect_path"])
    window = preflight["comparison_scope_s"]
    manifest = {
        "case_id": f"videosham-{number}",
        "dataset": "VideoSham",
        "task": (
            "Determine which media modality contains a material source-to-suspect difference, "
            "then localize the compared source and suspect intervals."
        ),
        "source_id": _source_id(row),
        "comparison_scope_s": window,
        "scope_provenance": "dataset-provided edit window used only as a neutral locator",
        "source_media": {"ref": "source"},
        "edited_media": {"ref": "suspect"},
        "allowed_inputs": ["the two media files", "their neutral identifiers", "the comparison scope"],
        "forbidden_inputs": ["editor task text", "manipulation label", "oracle fields"],
        "dataset_adapter": preflight["adapter_id"],
    }
    manipulation_type = {
        "temporal_omission": "temporal_edit",
        "temporal_warp": "temporal_edit",
        "audio_replacement": "audio_edit",
        "spatial_removal": "spatial_edit",
        "spatial_insertion": "spatial_edit",
        "appearance_change": "spatial_edit",
        "text_edit": "spatial_edit",
    }[preflight["capability_family"]]
    oracle = {
        "case_id": manifest["case_id"],
        "annotation_source": "VideoSourceandEdits.csv",
        "original_vimeo_id": manifest["source_id"].replace("vimeo-", ""),
        "edit_window_s": window,
        "original_annotation_window_s": preflight.get("original_annotation_window_s", window),
        "manipulation_type": manipulation_type,
        "annotation_text": row["Task"],
        "visibility": "oracle_only",
    }
    write_json(case_root / "student_view" / "case_manifest.json", manifest)
    write_json(case_root / "oracle" / "claim_oracle.json", oracle)
    return manifest, oracle


def _tool_telemetry(bundle: dict[str, Any]) -> dict[str, Any]:
    selected = Counter(
        turn["tool_name"]
        for episode_name in ("clean_episode", "repair_episode")
        for turn in bundle[episode_name]["turn_history"]
    )
    indispensable = Counter(
        step["tool_name"]
        for episode in bundle["step_utility_report"]["episodes"].values()
        for step in episode
        if step["accepted_for_sft"]
    )
    return {
        "selected_calls": dict(selected),
        "indispensable_calls": dict(indispensable),
        "selected_but_not_indispensable": {
            name: count - indispensable.get(name, 0)
            for name, count in selected.items()
            if count > indispensable.get(name, 0)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _rows()
    case_ids = getattr(args, "case_ids", None)
    selected_rows = select_rows(
        rows,
        seed=args.seed,
        sample_size=args.sample_size,
        case_ids=case_ids,
    )
    sample_size = len(selected_rows)
    if case_ids:
        selected_numbers = "-".join(_case_number(row) for row in selected_rows)
        run_id = f"videosham_cases_{selected_numbers}_n{sample_size}"
    else:
        run_id = f"videosham_seed_{args.seed}_n{sample_size}"
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    base_teacher = CodexCLITeacher(model=args.model) if args.run_cli else None
    teacher = (
        None
        if base_teacher is None
        else base_teacher
        if getattr(args, "legacy_full_control", False)
        else build_evidence_choice_teacher(
            base_teacher,
            candidates_per_turn=getattr(args, "candidates_per_turn", 3),
            proposal_rounds_per_turn=getattr(args, "proposal_rounds_per_turn", 2),
        )
    )
    results = []
    aggregate_clean = []
    aggregate_repair = []
    aggregate_transitions = []
    aggregate_selected: Counter[str] = Counter()
    aggregate_indispensable: Counter[str] = Counter()

    repair_operators = getattr(args, "repair_operators", ["evidence_dropout"])
    for row_index, row in enumerate(selected_rows):
        preflight = preflight_case(row)
        case_root = output_root / "cases" / preflight["case_id"]
        result = {
            **preflight,
            "dataset_manipulation_family": row["Manipulation"],
            "status": "preflight_blocked" if not preflight["cli_eligible"] else "ready",
        }
        if args.run_cli and preflight["cli_eligible"]:
            try:
                source_path = Path(preflight["source_path"])
                suspect_path = Path(preflight["suspect_path"])
                source_frames = decode(source_path)
                suspect_frames = decode(suspect_path)
                preflight["original_annotation_window_s"] = list(preflight["comparison_scope_s"])
                preflight["comparison_scope_s"] = localize_comparison_scope(
                    preflight["comparison_scope_s"], source_frames.duration_s
                )
                result["original_annotation_window_s"] = preflight["original_annotation_window_s"]
                result["comparison_scope_s"] = preflight["comparison_scope_s"]
                manifest, oracle = compile_case(row, preflight, case_root)
                environment = MultiFamilyVideoEnvironment(
                    manifest=manifest,
                    source_frames=source_frames,
                    suspect_frames=suspect_frames,
                    source_path=source_path,
                    suspect_path=suspect_path,
                )
                artifact_dir = case_root / "outputs"
                resume_base = (
                    getattr(args, "resume_base_artifacts", False)
                    and (artifact_dir / "llm_teacher_clean_episode.json").exists()
                )
                if resume_base:
                    bundle, report = load_saved_bundle(artifact_dir)
                    bundle["harness_config"] = {
                        "claim_schema": copy.deepcopy(MULTIFAMILY_SCHEMA),
                        "tool_registry": copy.deepcopy(MULTIFAMILY_CAUSAL_TOOL_REGISTRY),
                        "target_fields": list(MULTIFAMILY_SCHEMA["target_fields"]),
                        "invariant_fields": list(MULTIFAMILY_SCHEMA["invariant_fields"]),
                    }
                else:
                    bundle = build_llm_teacher_bundle(
                        manifest=manifest,
                        source_frames=source_frames,
                        suspect_frames=suspect_frames,
                        source_path=source_path,
                        suspect_path=suspect_path,
                        teacher=teacher,
                        max_clean_turns=args.max_clean_turns,
                        max_repair_turns=args.max_repair_turns,
                        claim_schema=MULTIFAMILY_SCHEMA,
                        tool_registry=MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
                        external_environment=environment,
                        target_fields=MULTIFAMILY_SCHEMA["target_fields"],
                        invariant_fields=MULTIFAMILY_SCHEMA["invariant_fields"],
                        repair_operator=repair_operators[row_index % len(repair_operators)],
                    )
                    step_report = audit_step_utility(bundle)
                    apply_step_utility_filter(bundle, step_report)
                    report = validate_llm_teacher_bundle(bundle, oracle=oracle)
                    _write_bundle(bundle, report, promote=False, output_dir=artifact_dir)
                quality = evaluate_unified_quality(bundle, report, dataset="VideoSham")
                write_json(case_root / "outputs" / "unified_trajectory_quality_report.json", quality)
                causal_quality = None
                if getattr(args, "run_causal_packets", False):
                    _packet, causal_quality = qualify_and_write_causal_packet(
                        bundle=bundle,
                        teacher=teacher,
                        dataset="VideoSham",
                        validation=report,
                        output_dir=case_root / "outputs",
                        environment_factory=lambda: MultiFamilyVideoEnvironment(
                            manifest=manifest,
                            source_frames=source_frames,
                            suspect_frames=suspect_frames,
                            source_path=source_path,
                            suspect_path=suspect_path,
                        ),
                        run_blind_judges=getattr(args, "blind_judges", False),
                        judge_model=args.model,
                        max_probe_turns=getattr(args, "max_probe_turns", 6),
                    )
                telemetry = _tool_telemetry(bundle)
                accepted = (
                    report["passed"]
                    and quality["accepted_for_sft"]
                    and (causal_quality is None or causal_quality["accepted_for_sft"])
                )
                result.update({
                    "status": (
                        "passed" if accepted
                        else "utqr_v7_rejected" if causal_quality is not None
                        else "rubric_rejected" if report["passed"]
                        else "validation_failed"
                    ),
                    "validation_passed": report["passed"],
                    "failed_checks": [name for name, passed in report["checks"].items() if not passed],
                    "rubric_score": quality["total_score"],
                    "rubric_grade": quality["grade"],
                    "rubric_failed_gates": quality["diagnosis"]["failed_gates"],
                    "rubric_failed_criteria": quality["diagnosis"]["failed_criteria"],
                    "utqr_v7_score": None if causal_quality is None else causal_quality["total_score"],
                    "utqr_v7_accepted": None if causal_quality is None else causal_quality["accepted_for_sft"],
                    "utqr_v7_failed_gates": None if causal_quality is None else causal_quality["failed_gates"],
                    "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                    "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                    "repair_operator": bundle["execution_summary"].get("repair_operator"),
                    "usage": bundle["execution_summary"]["api_usage"],
                    "tool_telemetry": telemetry,
                    "base_artifacts_resumed": resume_base,
                })
                if accepted:
                    aggregate_clean.extend(bundle["clean_turn_sft"])
                    aggregate_repair.extend(bundle["repair_turn_sft"])
                    aggregate_transitions.extend(compile_transition_sidecar(bundle))
                    aggregate_selected.update(telemetry["selected_calls"])
                    aggregate_indispensable.update(telemetry["indispensable_calls"])
            except Exception as error:  # keep the batch report even when one real case exposes a failure
                failure_artifact = getattr(error, "failure_artifact", None)
                if failure_artifact is not None:
                    failure_path = case_root / "outputs" / "incomplete_teacher_episode.json"
                    write_json(failure_path, failure_artifact)
                    result["failure_artifact"] = str(failure_path)
                result.update(
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        **generation_failure_metadata(error),
                    }
                )
        results.append(result)
        write_json(
            output_root / "batch_report.json",
            {
                "run_id": run_id,
                "dataset": "VideoSham",
                "seed": args.seed,
                "sample_size": sample_size,
                "selection_mode": "explicit_case_ids" if case_ids else "seeded_random_sample",
                "cli_requested": args.run_cli,
                "cases": results,
            },
        )

    passed = sum(item["status"] == "passed" for item in results)
    blocked = sum(item["status"] == "preflight_blocked" for item in results)
    failed_after_cli_start = sum(
        item["status"] in {"execution_failed", "validation_failed", "rubric_rejected", "utqr_v7_rejected"}
        for item in results
    )
    ready_not_run = sum(item["status"] == "ready" for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VideoSham",
        "seed": args.seed,
        "sample_size": sample_size,
        "selection_mode": "explicit_case_ids" if case_ids else "seeded_random_sample",
        "cli_requested": args.run_cli,
        "summary": {
            "passed": passed,
            "preflight_blocked": blocked,
            "ready_not_run": ready_not_run,
            "failed_after_cli_start": failed_after_cli_start,
            "safe_to_expand_to_other_datasets": passed == sample_size,
        },
        "sample_capability_distribution": dict(Counter(item["capability_family"] for item in results)),
        "tool_telemetry": {
            "selected_calls": dict(aggregate_selected),
            "indispensable_calls": dict(aggregate_indispensable),
            "unused_implemented_tools": sorted(
                IMPLEMENTED_TOOLS - EVIDENCE_INTEGRITY_TOOL_NAMES - set(aggregate_selected)
            ),
            "conditional_integrity_tools": sorted(EVIDENCE_INTEGRITY_TOOL_NAMES),
        },
        "generation_admission": generation_admission_summary(results),
        "cases": results,
    }
    write_json(output_root / "batch_report.json", report)
    write_jsonl(output_root / "sft_clean_accepted.jsonl", aggregate_clean)
    write_jsonl(output_root / "sft_repair_accepted.jsonl", aggregate_repair)
    write_jsonl(output_root / "transition_sidecar.jsonl", aggregate_transitions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="Explicit VideoSham IDs for targeted capability coverage; overrides random sampling.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-cli", action="store_true")
    parser.add_argument("--max-clean-turns", type=int, default=12)
    parser.add_argument("--max-repair-turns", type=int, default=10)
    parser.add_argument("--candidates-per-turn", type=int, default=3)
    parser.add_argument("--proposal-rounds-per-turn", type=int, default=2)
    parser.add_argument(
        "--run-causal-packets",
        action="store_true",
        help="Execute all eight UTQR-7 causal probes; a failing packet is excluded from SFT.",
    )
    parser.add_argument(
        "--blind-judges",
        action="store_true",
        help="Add two isolated blind process judges. Required for final UTQR-7 acceptance.",
    )
    parser.add_argument("--max-probe-turns", type=int, default=8)
    parser.add_argument(
        "--resume-base-artifacts",
        action="store_true",
        help="Reuse already validated clean/repair artifacts and run only missing causal qualification.",
    )
    parser.add_argument(
        "--legacy-full-control",
        action="store_true",
        help="Diagnostic compatibility mode; causal reduced-control teacher view is the default.",
    )
    parser.add_argument(
        "--repair-operators",
        nargs="+",
        choices=["evidence_dropout", "binding_invalidation", "decision_reopen"],
        default=["evidence_dropout"],
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
