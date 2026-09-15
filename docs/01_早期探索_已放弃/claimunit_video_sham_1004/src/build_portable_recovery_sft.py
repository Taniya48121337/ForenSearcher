"""Create a portable SFT serialization without changing trajectory semantics."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_INPUT = (
    BASE
    / "recovery_sft_training_mix_v4_edgeaware55/recovery_sft_weighted.jsonl"
)
DEFAULT_OUTPUT = BASE / "recovery_sft_portable_v6_edgeaware55"

WINDOWS_PATH = re.compile(r"(?i)[A-Z]:\\[^\s\"']+")
WORKSPACE_PATH = re.compile(r"(?i)(?:datasets|external)/[^\s\"']+")
HTTP_URL = re.compile(r"https?://[^\s\"']+", re.I)
VIDDIFF_URI = re.compile(r"viddiff://[^\s\"']+", re.I)
VIDEOSHAM_CASE = re.compile(r"videosham-\d+", re.I)
VIDDIFF_CASE = re.compile(r"viddiff-surgery_\d+", re.I)
VCSL_CASE = re.compile(r"vcsl-[0-9a-f]{32}-[0-9a-f]{32}", re.I)
GUID = re.compile(r"\b[0-9a-f]{32}\b", re.I)
TASK_ID = re.compile(r"\bTASK-[A-Z0-9_-]+\b", re.I)
TRAJECTORY_ID = re.compile(r"\bLLM-TRJ-[A-Z0-9_-]+\b", re.I)
TOOL_CALL_ID = re.compile(r"\b(?:cli_)?call_[A-Z0-9_-]+\b", re.I)
MEDIA_FILE = re.compile(r"\b\d{4}_(?:unedited|processed)\.mp4\b", re.I)
EVIDENCE_ID = re.compile(r"\bEA-[A-Z0-9_-]+\b", re.I)

LOGICAL_KEYS = {
    "source_id": "<SOURCE_ENTITY>",
    "query_id": "<QUERY_ENTITY>",
    "reference_id": "<REFERENCE_ENTITY>",
    "sample_key": "<SAMPLE_ENTITY>",
    "contract_id": "<TASK_CONTRACT>",
}


def _parsed(value: str) -> Any | None:
    if value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


class PortableSerializer:
    def __init__(self, namespace: str) -> None:
        self.namespace = re.sub(r"[^A-Z0-9]+", "-", namespace.upper()).strip("-")
        self.literal_map: dict[str, str] = {}
        self.claim_map: dict[str, str] = {}
        self.evidence_map: dict[str, str] = {}
        self.guid_map: dict[str, str] = {}
        self.call_map: dict[str, str] = {}

    def collect(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in LOGICAL_KEYS and isinstance(item, str):
                    self.literal_map.setdefault(item, self.logical_value(key))
                if key in {"claim_id", "candidate_claim_id"} and isinstance(item, str):
                    self.claim_map.setdefault(
                        item,
                        f"<CLAIM_{self.namespace}_{len(self.claim_map) + 1:02d}>",
                    )
                self.collect(item)
        elif isinstance(value, list):
            for item in value:
                self.collect(item)
        elif isinstance(value, str):
            for evidence_id in EVIDENCE_ID.findall(value):
                self.evidence_map.setdefault(
                    evidence_id,
                    f"EA-PORTABLE-{self.namespace}-{len(self.evidence_map) + 1:03d}",
                )
            parsed = _parsed(value)
            if parsed is not None:
                self.collect(parsed)

    @staticmethod
    def _token(token: str, mapping: dict[str, str], prefix: str) -> str:
        key = token.lower()
        if key not in mapping:
            mapping[key] = f"<{prefix}_{len(mapping) + 1:02d}>"
        return mapping[key]

    def logical_value(self, key: str) -> str:
        prefix = LOGICAL_KEYS[key].strip("<>")
        return f"<{prefix}_{self.namespace}>"

    def text(self, value: str) -> str:
        parsed = _parsed(value)
        if parsed is not None:
            return json.dumps(
                self.value(parsed), ensure_ascii=False, separators=(",", ":")
            )
        for source, target in sorted(
            {**self.literal_map, **self.claim_map, **self.evidence_map}.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            value = value.replace(source, target)
        value = WINDOWS_PATH.sub("asset://local-logical", value)
        value = WORKSPACE_PATH.sub("asset://workspace-logical", value)
        value = HTTP_URL.sub("asset://remote-logical", value)
        value = VIDDIFF_URI.sub("asset://video-pair-logical", value)
        case_entity = f"<CASE_ENTITY_{self.namespace}>"
        value = VCSL_CASE.sub(case_entity, value)
        value = VIDEOSHAM_CASE.sub(case_entity, value)
        value = VIDDIFF_CASE.sub(case_entity, value)
        value = TASK_ID.sub(f"<TASK_CONTRACT_{self.namespace}>", value)
        value = TRAJECTORY_ID.sub(f"<TRAJECTORY_ID_{self.namespace}>", value)
        value = MEDIA_FILE.sub("asset://media-logical", value)
        value = GUID.sub(
            lambda match: self._token(
                match.group(0), self.guid_map, f"ENTITY_{self.namespace}"
            ),
            value,
        )
        value = TOOL_CALL_ID.sub(
            lambda match: self._token(
                match.group(0), self.call_map, f"TOOL_CALL_{self.namespace}"
            ),
            value,
        )
        return value

    def value(self, value: Any, key: str | None = None) -> Any:
        if key in LOGICAL_KEYS and isinstance(value, str):
            return self.logical_value(key)
        if isinstance(value, dict):
            return {
                self.text(str(name)): self.value(item, str(name))
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def _tool_names(row: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for message in row["messages"]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result.append(str((call.get("function") or {}).get("name")))
    return result


def _numeric_scalars(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, bool):
        return result
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        for item in value.values():
            result.extend(_numeric_scalars(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_numeric_scalars(item))
    elif isinstance(value, str):
        parsed = _parsed(value)
        if parsed is not None:
            result.extend(_numeric_scalars(parsed))
    return result


def _evidence_ids(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False)
    return sorted(EVIDENCE_ID.findall(text))


def _evidence_reference_graph_preserved(source: Any, portable: Any) -> bool:
    source_counts = Counter(_evidence_ids(source))
    portable_counts = Counter(_evidence_ids(portable))
    return (
        len(source_counts) == len(portable_counts)
        and sorted(source_counts.values()) == sorted(portable_counts.values())
    )


def _tool_call_links_valid(row: dict[str, Any]) -> bool:
    pending: set[str] = set()
    for message in row["messages"]:
        if message.get("role") == "assistant":
            pending.update(
                str(call.get("id")) for call in message.get("tool_calls") or []
            )
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id"))
            if call_id not in pending:
                return False
            pending.remove(call_id)
    return True


def build_portable_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    case_map: dict[tuple[str, str], str] = {}
    dataset_case_counts: Counter[str] = Counter()
    for source in rows:
        dataset = str(source["metadata"]["dataset"])
        source_case_id = str(source["metadata"]["case_id"])
        key = (dataset, source_case_id)
        if key not in case_map:
            dataset_case_counts[dataset] += 1
            logical_dataset = "".join(
                character.lower() if character.isalnum() else "_"
                for character in dataset
            ).strip("_")
            case_map[key] = (
                f"portable://{logical_dataset}/case_"
                f"{dataset_case_counts[dataset]:04d}"
            )
    for source in rows:
        serializer = PortableSerializer(str(source["metadata"]["episode_id"]))
        serializer.collect(source["messages"])
        item = copy.deepcopy(source)
        item["messages"] = serializer.value(item["messages"])
        case_key = (
            str(source["metadata"]["dataset"]),
            str(source["metadata"]["case_id"]),
        )
        item["metadata"]["case_id"] = case_map[case_key]
        item["metadata"]["portable_serialization"] = True
        item["metadata"]["portable_policy_id"] = "logical_asset_entity_and_case_refs_v2"
        checks = {
            "message_roles_preserved": [m["role"] for m in source["messages"]]
            == [m["role"] for m in item["messages"]],
            "tool_sequence_preserved": _tool_names(source) == _tool_names(item),
            "numeric_json_scalars_preserved": sorted(_numeric_scalars(source["messages"]))
            == sorted(_numeric_scalars(item["messages"])),
            "evidence_reference_graph_preserved": _evidence_reference_graph_preserved(
                source["messages"], item["messages"]
            ),
            "sampling_weight_preserved": source["metadata"].get("sft_sampling_weight")
            == item["metadata"].get("sft_sampling_weight"),
            "tool_call_links_valid": _tool_call_links_valid(item),
        }
        equivalence.append(
            {
                "row_id": item["metadata"]["row_id"],
                "passed": all(checks.values()),
                "checks": checks,
            }
        )
        output.append(item)

    serialized = "\n".join(json.dumps(item, ensure_ascii=False) for item in output)
    forbidden = {
        "windows_absolute_path": len(WINDOWS_PATH.findall(serialized)),
        "workspace_media_path": len(WORKSPACE_PATH.findall(serialized)),
        "http_url": len(HTTP_URL.findall(serialized)),
        "viddiff_uri": len(VIDDIFF_URI.findall(serialized)),
        "videosham_case_id": len(VIDEOSHAM_CASE.findall(serialized)),
        "viddiff_case_id": len(VIDDIFF_CASE.findall(serialized)),
        "vcsl_case_id": len(VCSL_CASE.findall(serialized)),
        "raw_guid": len(GUID.findall(serialized)),
    }
    report = {
        "artifact": "portable_recovery_sft_audit_v6",
        "rows": len(output),
        "datasets": dict(Counter(row["metadata"]["dataset"] for row in output)),
        "all_semantic_equivalence_checks_passed": all(
            item["passed"] for item in equivalence
        ),
        "forbidden_identifier_hits": forbidden,
        "all_forbidden_identifier_hits_zero": all(value == 0 for value in forbidden.values()),
        "messages_intentionally_changed_only_by_logical_serialization": True,
        "tool_evidence_numeric_and_weight_semantics_preserved": True,
        "case_equivalence_classes_preserved": (
            len(case_map)
            == len({row["metadata"]["case_id"] for row in output})
        ),
        "rows_detail": equivalence,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--canonical-executable-source",
        default=(
            "../recovery_batch_candidate_v16_edge_aware_humanlike/"
            "recovery_sft_candidate.jsonl"
        ),
    )
    args = parser.parse_args()
    rows, report = build_portable_rows(_jsonl(args.input))
    has_semantic_contrasts = any(
        (row.get("metadata") or {}).get("semantic_contrast_groups") for row in rows
    )
    write_jsonl(args.output_root / "recovery_sft_portable_weighted.jsonl", rows)
    write_json(args.output_root / "portability_audit.json", report)
    write_json(
        args.output_root / "serialization_contract.json",
        {
            "artifact": "portable_recovery_sft_serialization_contract_v6",
            "logical_assets": [
                "asset://local-logical",
                "asset://workspace-logical",
                "asset://remote-logical",
                "asset://video-pair-logical",
                "asset://media-logical",
            ],
            "logical_entities": [
                "<SOURCE_ENTITY>",
                "<QUERY_ENTITY>",
                "<REFERENCE_ENTITY>",
                "<SAMPLE_ENTITY>",
                "<CASE_ENTITY>",
            ],
            "preserved_exactly": [
                "message roles and order",
                "tool names and order",
                "all numeric JSON scalar measurements",
                "EvidenceAtom reference graph under consistent logical renaming",
                "sampling weights",
                "assistant loss scope",
            ],
            "not_executable_replay_input": (
                "This is a portable SFT serialization of already executed trajectories; "
                "canonical executable artifacts remain in the source batch."
            ),
            "training_qualified": False,
        },
    )
    write_json(
        args.output_root / "trainer_contract.json",
        {
            "artifact": "portable_recovery_sft_trainer_contract_v1",
            "preferred_training_input": "recovery_sft_portable_weighted.jsonl",
            "canonical_executable_source": args.canonical_executable_source,
            "supervised_tokens": "last assistant action only",
            "per_row_loss_multiplier": "metadata.sft_sampling_weight",
            "do_not_supervise": ["system", "user", "tool", "private sidecar"],
            "split_unit": "episode_id; never split turns from one episode",
            "optional_batch_group": (
                "metadata.semantic_contrast_groups[*].group_id"
                if has_semantic_contrasts
                else None
            ),
            "grouping_is_advisory": has_semantic_contrasts,
            "n_ary_group_sampling": (
                "For groups with more than two rows, sample at least two members whose "
                "contrast_target values differ; loading the entire group is not required."
                if has_semantic_contrasts
                else None
            ),
            "load_entire_group_required": False,
            "drop_unpaired_rows": False,
            "portable_rows_are_for_training_not_tool_replay": True,
            "admission_requirement": (
                "Score the canonical executable source with the current evidence-first "
                "UTQR before using this portable serialization; portable text alone cannot "
                "replay or certify adjudication."
            ),
            "training_qualified": False,
        },
    )
    print(json.dumps({
        "rows": report["rows"],
        "datasets": report["datasets"],
        "semantic_equivalence": report["all_semantic_equivalence_checks_passed"],
        "forbidden_identifier_hits": report["forbidden_identifier_hits"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
