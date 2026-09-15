"""Predicate-decomposed independent VidDiff measurement to reduce comparison anchoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from cli_teacher_rollout import DEFAULT_CODEX_JS, DEFAULT_NODE, _usage_from_events
from pipeline import ROOT, write_json
from viddiff_batch import DEFAULT_ADAPTIVE, DEFAULT_AUDIT, _load_jsonl


DEFAULT_V2_ROOT = ROOT / "outputs" / "probes" / "viddiff_v2_surgery_seed_20260818_n10"
DEFAULT_V3_ROOT = ROOT / "outputs" / "probes" / "viddiff_v3_surgery_seed_20260818_n10"
DEFAULT_OUTPUT = ROOT / "outputs" / "probes" / "viddiff_v4_independent_predicate"
DEFAULT_SAMPLE_KEYS = ["surgery_116", "surgery_159", "surgery_60"]


def predicate_family(statement: str) -> str:
    text = statement.lower()
    if "both graspers" in text and "efficient" in text:
        return "dual_grasper_efficiency"
    if "grasped closer to the tip" in text or ("grasping" in text and "needle tip" in text):
        return "needle_tip_grasp_position"
    if "dot" in text and ("accurately hit" in text or "accuracy" in text):
        return "target_dot_accuracy"
    if "never out of view" in text or "occluded" in text or "out of frame" in text:
        return "universal_visibility"
    if "stabalize" in text or "stabilize" in text:
        return "supporting_grasper_stability"
    if "force" in text:
        return "force_deformation_proxy"
    if "tangle" in text or "entangle" in text:
        return "suture_tangle_topology"
    if "radial" in text and "hoop" in text:
        return "hoop_radial_trajectory"
    if "perpendicular" in text or " angle " in f" {text} ":
        return "needle_insertion_angle"
    if "tension" in text or "taut" in text or "slack" in text:
        return "suture_tension_proxy"
    return "generic_action_evidence"


def measurement_schema() -> dict[str, Any]:
    interval = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    string_list = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "support_score": {"type": "number", "minimum": 0, "maximum": 100},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "best_interval_s": interval,
            "positive_cues": string_list,
            "counterevidence": string_list,
            "uncertainty_sources": string_list,
            "measurement_summary": {"type": "string", "minLength": 100},
        },
        "required": [
            "support_score",
            "confidence",
            "best_interval_s",
            "positive_cues",
            "counterevidence",
            "uncertainty_sources",
            "measurement_summary",
        ],
        "additionalProperties": False,
    }


FAMILY_INSTRUCTIONS = {
    "dual_grasper_efficiency": (
        "Operationalize efficient dual-grasper use as role-complementary work, not mere visibility: identify intervals "
        "where one grasper actively manipulates the suture while the other maintains useful contact, stabilizes the "
        "tube, transfers the suture, or prevents slack/backsliding. Penalize a parked, hovering, redundant, colliding, "
        "or repeatedly repositioned second grasper. Score sustained useful coordination relative to visible "
        "opportunities, and cite both productive and wasted intervals."
    ),
    "needle_tip_grasp_position": (
        "Measure geometric grasp position on the needle. In frames where the needle arc and jaw contact are both "
        "visible, estimate the two needle endpoints, identify the sharper tip, trace the arc length, and report whether "
        "the jaw contact lies closer to that tip or toward the middle/tail. Do not substitute grasper approach speed, "
        "shaft angle, later handoffs, or subject skill. If the tip/contact is occluded, lower confidence rather than "
        "guessing."
    ),
    "target_dot_accuracy": (
        "Measure the actual needle penetration/contact point relative to the intended red target dot during suturing. "
        "Use adjacent frames to distinguish the needle tip from grasper jaws and to identify first tissue contact; "
        "normalize miss distance by neighboring dot spacing. Do not use grasper-tip distance, the needle pose before "
        "contact, or overall trajectory smoothness as a substitute. If penetration is not visible, abstain with low "
        "confidence."
    ),
    "universal_visibility": (
        "Treat this as a universal claim and actively search for counterexamples: any clear tip loss behind another "
        "instrument or beyond the image boundary sharply lowers support. Do not reward a shorter video or denser "
        "sampling. Sparse frames cannot prove 'never'; put that limitation in uncertainty_sources."
    ),
    "supporting_grasper_stability": (
        "Operationalize stabilization as sustained contact with the target plus lower frame-to-frame displacement of "
        "the supporting grasper while the working grasper moves. Distinguish this from both tools jointly manipulating "
        "the needle or merely passing near the hoop cluster."
    ),
    "force_deformation_proxy": (
        "Infer force only from physical proxies: local tissue indentation/compression, needle curvature, taut suture, "
        "and opposing tool contact across adjacent frames. Do not use speed, duration, general busyness, or tool pose alone."
    ),
    "needle_insertion_angle": (
        "Estimate the needle tangent at the tissue entry point relative to the visible incision line during actual "
        "penetration. Perpendicular means close to 90 degrees. Do not substitute grasper shaft angle, camera angle, "
        "needle pose before contact, or a single occluded frame; use adjacent frames around entry and name ambiguity."
    ),
    "suture_tension_proxy": (
        "Infer thread tension only from visible taut-versus-slack geometry across adjacent frames: straightening, "
        "reduced catenary curvature, transmitted tissue motion, and persistent opposing traction. Do not use hand speed, "
        "clip duration, general activity, or needle pose as tension evidence."
    ),
    "suture_tangle_topology": (
        "Measure tangling only from persistent thread topology: visible self-crossings, closed or tightening loops, "
        "snagging around an instrument, and failure of a crossing to disappear across adjacent frames. Do not call "
        "ordinary slack curvature, projection overlap in one frame, or a loose arc a tangle."
    ),
    "hoop_radial_trajectory": (
        "Estimate the hoop center and compare the needle-tip displacement across adjacent frames with the local radius "
        "from that center. Radial motion aligns inward or outward with the radius; tangential motion follows the hoop. "
        "Do not infer radiality from needle orientation, grasper pose, or a single frame."
    ),
    "generic_action_evidence": (
        "Use only action-specific changes across adjacent frames and name both supporting and contradicting cues."
    ),
}


class IndependentPredicateMeasurer:
    def __init__(self, model: str = "gpt-5.4", timeout_s: int = 300) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def measure(
        self,
        *,
        global_image: Path,
        targeted_image: Path,
        role: str,
        statement: str,
        action_description: str,
        family: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are a blinded single-video measurement backend. Both attached images show the SAME video: image 1 "
            "is a global chronological mosaic and image 2 contains dense chronological strips proposed without labels. "
            "You cannot see the other comparison video and must not guess which is better. Measure only how strongly "
            f"this video's pixels support the exact statement {statement!r}. Shared action: {action_description!r}. "
            f"Predicate family: {family}. {FAMILY_INSTRUCTIONS[family]} Return a calibrated 0-100 support_score, "
            "confidence in the visible measurement, the best local interval, concrete positive cues, concrete "
            "counterevidence, and uncertainty sources. A high score requires positive evidence and little counterevidence; "
            "absence of a visible counterexample is not proof. Never use subject identity, duration, filename, labels, or "
            "assumed skill.\n"
            f"ROLE={role}\nMETADATA={json.dumps(metadata, ensure_ascii=False)}"
        )
        with tempfile.TemporaryDirectory(prefix="viddiff_v4_cli_") as directory:
            temp_root = Path(directory)
            schema_path = temp_root / "schema.json"
            output_path = temp_root / "output.json"
            schema_path.write_text(json.dumps(measurement_schema()), encoding="utf-8")
            command = [
                str(DEFAULT_NODE),
                str(DEFAULT_CODEX_JS),
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--json",
                "--image",
                str(global_image.resolve()),
                "--image",
                str(targeted_image.resolve()),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--model",
                self.model,
                "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_s,
                cwd=directory,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "measurement_id": f"VIDDIFF-V4-{uuid.uuid4().hex}",
            "role": role,
            "predicate_family": family,
            "usage": _usage_from_events(completed.stdout),
            **payload,
        }


def compare_independent(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    score_a = float(left["support_score"])
    score_b = float(right["support_score"])
    confidence = min(float(left["confidence"]), float(right["confidence"]))
    delta = score_a - score_b
    calibrated_margin = abs(delta) * confidence
    if confidence < 0.45 or calibrated_margin < 8.0:
        preferred = "c"
        rule = "independent_scores_insufficient_margin_abstain"
    else:
        preferred = "a" if delta > 0 else "b"
        rule = "independent_predicate_score_margin"
    return {
        "preferred_video": preferred,
        "decision_rule": rule,
        "order_consistent": True,
        "support_score_a": score_a,
        "support_score_b": score_b,
        "score_delta_a_minus_b": round(delta, 3),
        "calibrated_margin": round(calibrated_margin, 3),
        "confidence": round(confidence, 3),
        "video_a_interval_s": left["best_interval_s"],
        "video_b_interval_s": right["best_interval_s"],
        "independent_a": left,
        "independent_b": right,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = {item["sample_key"]: item for item in _load_jsonl(args.adaptive_path)}
    audit = json.loads(args.audit_path.read_text(encoding="utf-8"))
    oracle = {item["sample_key"]: item["oracle_direction"] for item in audit["cases"]}
    measurer = IndependentPredicateMeasurer(args.model) if args.run_cli else None
    results = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample_key in args.sample_keys:
        item = items[sample_key]
        family = predicate_family(item["task"]["comparison_statement"])
        case_root = args.output_root / sample_key
        result: dict[str, Any] = {
            "case_id": item["case_id"],
            "sample_key": sample_key,
            "predicate_family": family,
            "status": "ready",
        }
        if measurer is not None:
            try:
                measurements = []
                for role in ("a", "b"):
                    targeted_meta = next(
                        asset for asset in json.loads((args.v3_root / "probe_report.json").read_text(encoding="utf-8"))["cases"]
                        if asset["sample_key"] == sample_key
                    )["targeted_assets"][0 if role == "a" else 1]
                    measurement = measurer.measure(
                        global_image=args.v2_root / sample_key / f"video_{role}_global.jpg",
                        targeted_image=args.v3_root / sample_key / f"video_{role}_targeted.jpg",
                        role=role.upper(),
                        statement=item["task"]["comparison_statement"],
                        action_description=item["task"]["action_description"],
                        family=family,
                        metadata=targeted_meta,
                    )
                    measurements.append(measurement)
                    write_json(case_root / f"independent_{role}.json", measurement)
                comparison = compare_independent(measurements[0], measurements[1])
                result.update(
                    {
                        "status": "passed" if comparison["preferred_video"] == oracle[sample_key] else "failed",
                        "measurement": comparison,
                        "oracle_direction": oracle[sample_key],
                        "direction_correct": comparison["preferred_video"] == oracle[sample_key],
                    }
                )
            except Exception as error:
                result.update({"status": "execution_failed", "error_type": type(error).__name__, "error": str(error)})
        results.append(result)
        write_json(args.output_root / "probe_report.json", {"cases": results})
    report = {
        "run_id": "viddiff_v4_independent_predicate",
        "dataset": "VidDiffBench",
        "method": "predicate_decomposed_independent_single_video_measurement_then_deterministic_compare",
        "label_access_policy": "oracle used only after both independent measurements and deterministic comparison",
        "summary": {
            "cases": len(results),
            "passed": sum(item["status"] == "passed" for item in results),
            "execution_failed": sum(item["status"] == "execution_failed" for item in results),
        },
        "cases": results,
    }
    write_json(args.output_root / "probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-keys", nargs="+", default=DEFAULT_SAMPLE_KEYS)
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--run-cli", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args)["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
