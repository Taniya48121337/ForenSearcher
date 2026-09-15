"""Audit locally available exact-case holdouts before any sealed model run.

The audit treats any exact case ID mentioned by prior source, tests, reports, or
experiment outputs as development-exposed. Raw dataset metadata is not scanned.
Selection is deterministic from a public seed and never uses oracle direction or
localization quality.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from pipeline import ROOT, write_json
from viddiff_probe import available_surgery_rows


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1/manifests"
)
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".ps1", ".py", ".txt", ".yaml", ".yml"}
SCAN_ROOTS = ("src", "tests", "outputs")
EXCLUDED_OUTPUT_PARTS = {
    "control_plane_falsification_v1/manifests",
}


def _history_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in SCAN_ROOTS:
        scan_root = root / relative
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            normalized = path.relative_to(root).as_posix()
            if any(part in normalized for part in EXCLUDED_OUTPUT_PARTS):
                continue
            files.append(path)
    files.extend(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )
    return sorted(set(files))


def _read_history(files: Iterable[Path]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        records.append((path.as_posix(), text.lower()))
    return records


def _references(records: list[tuple[str, str]], tokens: Iterable[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for token in tokens:
        needle = token.lower()
        hits = [path for path, text in records if needle in text.lower()]
        result[token] = hits
    return result


_CASE_TOKEN_PATTERN = re.compile(
    r"videosham-\d{4}|(?:vcsl-)?[0-9a-f]{32}-[0-9a-f]{32}|(?:viddiff-)?surgery_\d+",
    re.I,
)


def _references_in_files(files: Iterable[Path], tokens: Iterable[str]) -> dict[str, list[str]]:
    """Build the same exact-token ledger in one streaming pass over history."""
    canonical = {token.lower(): token for token in tokens}
    result: dict[str, list[str]] = {token: [] for token in tokens}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found: set[str] = set()
        for match in _CASE_TOKEN_PATTERN.finditer(text):
            value = match.group(0).lower()
            if value.startswith("surgery_"):
                value = f"viddiff-{value}"
            elif len(value) == 65 and "-" in value:
                value = f"vcsl-{value}"
            if value in canonical:
                found.add(value)
        normalized_path = path.as_posix()
        for value in found:
            result[canonical[value]].append(normalized_path)
    return result


def _videosham_candidates(root: Path) -> list[str]:
    source_root = root / "datasets/videosham/media/source"
    edited_root = root / "datasets/videosham/media/edited"
    source = {
        path.name.removesuffix("_unedited.mp4")
        for path in source_root.glob("*_unedited.mp4")
        if path.stat().st_size >= 1024 and not path.name.startswith("._")
    }
    edited = {
        path.name.removesuffix("_processed.mp4")
        for path in edited_root.glob("*_processed.mp4")
        if path.stat().st_size >= 1024 and not path.name.startswith("._")
    }
    return [f"videosham-{number}" for number in sorted(source & edited)]


def _vcsl_candidates(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vcsl_root = root.parent / "external/VCSL"
    feature_root = vcsl_root / "features_isc_30case/eff256d"
    available = {path.stem for path in feature_root.glob("*.npy")}
    labels = json.loads(
        (vcsl_root / "data/label_file_uuid_total.json").read_text(encoding="utf-8")
    )
    candidates: list[dict[str, Any]] = []
    verifier: dict[str, Any] = {}
    for pair_id, boxes in labels.items():
        query_id, separator, reference_id = pair_id.partition("-")
        if not separator or query_id not in available or reference_id not in available:
            continue
        case_id = f"vcsl-{pair_id}"
        candidates.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "query_id": query_id,
                "reference_id": reference_id,
                "feature_entities_previously_available": True,
            }
        )
        verifier[case_id] = {"oracle_boxes": boxes}
    return sorted(candidates, key=lambda item: item["case_id"]), verifier


def _viddiff_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    verifier: dict[str, Any] = {}
    for row in available_surgery_rows():
        difference_keys = sorted(
            str(key)
            for key in row["differences_annotated"]
            if row["differences_gt"].get(str(key)) in {"a", "b", "c"}
        )
        if not difference_keys:
            continue
        difference_key = difference_keys[0]
        descriptor = row["differences_annotated"][difference_key]
        case_id = f"viddiff-{row['sample_key']}"
        candidates.append(
            {
                "case_id": case_id,
                "sample_key": row["sample_key"],
                "difference_key": difference_key,
                "comparison_statement": descriptor.get("description"),
                "action_description": row.get("action_description"),
                "source_dataset": row.get("source_dataset"),
            }
        )
        verifier[case_id] = {
            "difference_key": difference_key,
            "oracle_direction": row["differences_gt"][difference_key],
        }
    return sorted(candidates, key=lambda item: item["case_id"]), verifier


def _is_exposed(
    dataset: str,
    candidate: dict[str, Any],
    references: dict[str, list[str]],
) -> tuple[bool, list[str]]:
    case_id = str(candidate["case_id"])
    paths = list(references.get(case_id, []))
    if dataset == "VCSL":
        reverse_id = f"vcsl-{candidate['reference_id']}-{candidate['query_id']}"
        paths.extend(references.get(reverse_id, []))
    return bool(paths), sorted(set(paths))


def build(seed: int, requested_per_dataset: int) -> tuple[dict[str, Any], dict[str, Any]]:
    files = _history_files(ROOT)
    videosham = [{"case_id": case_id} for case_id in _videosham_candidates(ROOT)]
    vcsl, vcsl_verifier = _vcsl_candidates(ROOT)
    viddiff, viddiff_verifier = _viddiff_candidates()
    pools = {"VideoSham": videosham, "VCSL": vcsl, "VidDiffBench": viddiff}
    tokens = []
    for dataset, candidates in pools.items():
        for candidate in candidates:
            tokens.append(str(candidate["case_id"]))
            if dataset == "VCSL":
                tokens.append(f"vcsl-{candidate['reference_id']}-{candidate['query_id']}")
    references = _references_in_files(files, tokens)

    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    availability: dict[str, Any] = {}
    verifier_selected: dict[str, Any] = {}
    verifier_pool = {"VCSL": vcsl_verifier, "VidDiffBench": viddiff_verifier}
    for dataset, candidates in pools.items():
        unexposed: list[dict[str, Any]] = []
        exposed_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            exposed, paths = _is_exposed(dataset, candidate, references)
            if exposed:
                exposed_rows.append(
                    {
                        "case_id": candidate["case_id"],
                        "reference_count": len(paths),
                        "example_references": paths[:5],
                    }
                )
            else:
                unexposed.append(candidate)
        chosen = rng.sample(unexposed, min(requested_per_dataset, len(unexposed)))
        chosen = sorted(chosen, key=lambda item: item["case_id"])
        selected[dataset] = chosen
        availability[dataset] = {
            "local_executable_candidates": len(candidates),
            "development_exposed_exact_cases": len(exposed_rows),
            "unexposed_exact_cases": len(unexposed),
            "requested": requested_per_dataset,
            "selected": len(chosen),
            "shortage": max(0, requested_per_dataset - len(chosen)),
            "selected_case_ids": [item["case_id"] for item in chosen],
            "exposed_cases": exposed_rows,
        }
        if dataset in verifier_pool:
            verifier_selected[dataset] = {
                item["case_id"]: verifier_pool[dataset][item["case_id"]]
                for item in chosen
            }

    complete = all(
        availability[dataset]["selected"] == requested_per_dataset for dataset in pools
    )
    manifest = {
        "artifact": "control_plane_falsification_sealed_case_manifest_v1",
        "seed": seed,
        "selection_time_policy": "fixed_before_any_sealed_model_result",
        "selection_rule": (
            "uniform seeded sample from locally executable exact cases absent from all prior "
            "source/test/report/output text; no oracle direction or quality used"
        ),
        "history_scan": {
            "roots": list(SCAN_ROOTS),
            "files_scanned": len(files),
            "raw_dataset_metadata_scanned_as_exposure": False,
        },
        "requested_per_dataset": requested_per_dataset,
        "cross_dataset_complete": complete,
        "availability": availability,
        "selected_cases": selected,
        "fixed_arms": ["full", "structure_only", "evidence_only", "shuffled_evidence"],
        "fixed_decode": {
            "adapter": "student_external_contract_v1 v4 bind-compact",
            "local_quantization": "4bit",
            "greedy": True,
            "max_length": 6144,
            "max_new_tokens": 384,
            "max_generation_attempts": 3,
            "max_turns": 16,
            "utqr": "original unchanged",
        },
        "blocked_reason": (
            None
            if complete
            else "At least one dataset has fewer unexposed local exact cases than requested."
        ),
    }
    verifier = {
        "artifact": "control_plane_falsification_sealed_verifier_sidecar_v1",
        "student_visible": False,
        "selection_uses_oracle_value": False,
        "cases": verifier_selected,
    }
    return manifest, verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--requested-per-dataset", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest, verifier = build(args.seed, args.requested_per_dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "sealed_case_availability_audit.json", manifest)
    write_json(args.output / "sealed_cases.json", manifest)
    write_json(args.output / "sealed_cases_verifier.json", verifier)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
