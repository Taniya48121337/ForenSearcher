"""Build a policy-target view that teaches exact opaque-ID copying.

The source trajectories remain unchanged.  For claim creation and evidence
binding turns, this view (1) shortens only the natural-language preamble and
(2) creates semantics-preserving variants by consistently renaming claim_id in
the student-visible prompt, tool schema, and target call.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FOCUS_TOOLS = {"create_candidate_claim", "bind_evidence"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target_call(row: dict[str, Any]) -> dict[str, Any]:
    assistant = row["messages"][-1]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"expected one target call in {row['metadata'].get('row_id')}")
    return calls[0]["function"]


def _deep_replace(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_deep_replace(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _deep_replace(item, old, new) for key, item in value.items()}
    return value


def _compact_reasoning(row: dict[str, Any], *, id_is_visible: bool) -> None:
    function = _target_call(row)
    name = function["name"]
    arguments = function["arguments"]
    claim_id = str(arguments["claim_id"])
    if name == "create_candidate_claim":
        fields = list((arguments.get("field_values") or {}).keys())
        id_phrase = (
            f"the state-provided ID `{claim_id}`"
            if id_is_visible
            else "a fresh opaque claim ID"
        )
        row["messages"][-1]["content"] = (
            f"Active evidence supplies {len(fields)} visible field values, but no claim exists. "
            f"I will call `create_candidate_claim` with {id_phrase}, copying those values "
            "exactly before binding."
        )
    elif name == "bind_evidence":
        items = arguments.get("binding_items") or []
        row["messages"][-1]["content"] = (
            f"`bind_evidence` is now required because active atoms cover all {len(items)} "
            f"fields of `{claim_id}`. I will attach the exact visible mapping before any decision."
        )


def _safe_dataset(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")


def build(rows: list[dict[str, Any]], *, variants: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    augmented = 0
    compressed = 0
    skipped_unobservable_ids = 0
    for ordinal, source in enumerate(rows, start=1):
        base = copy.deepcopy(source)
        target_tool = str(base.get("metadata", {}).get("target_tool"))
        if target_tool in FOCUS_TOOLS:
            original_id = str(_target_call(base)["arguments"]["claim_id"])
            visible_source = json.dumps(
                {"messages": base["messages"][:-1], "tools": base.get("tools")},
                ensure_ascii=False,
            )
            id_is_visible = original_id in visible_source
            _compact_reasoning(base, id_is_visible=id_is_visible)
            base["metadata"]["policy_target_view"] = "concise_exact_action_v1"
            base["metadata"]["target_claim_id_student_visible"] = id_is_visible
            compressed += 1
        output.append(base)
        if target_tool not in FOCUS_TOOLS:
            continue
        if not id_is_visible:
            skipped_unobservable_ids += 1
            continue
        dataset = _safe_dataset(str(base["metadata"].get("dataset") or "DATA"))
        for variant in range(1, variants + 1):
            renamed = f"CF-CLAIM-{dataset}-{ordinal:04d}-V{variant}"
            item = _deep_replace(copy.deepcopy(base), original_id, renamed)
            item["metadata"]["row_id"] = f"{base['metadata']['row_id']}:cfid:{variant}"
            item["metadata"]["counterfactual_id_augmentation"] = {
                "kind": "opaque_claim_id_consistent_rename_v1",
                "variant": variant,
                "semantics_changed": False,
            }
            if original_id in json.dumps(
                {"messages": item["messages"], "tools": item.get("tools")},
                ensure_ascii=False,
            ):
                raise AssertionError("original claim_id survived counterfactual rename")
            if _target_call(item)["arguments"]["claim_id"] != renamed:
                raise AssertionError("target claim_id was not renamed")
            output.append(item)
            augmented += 1
    report = {
        "artifact": "counterfactual_exact_action_sft_view_v1",
        "source_rows": len(rows),
        "output_rows": len(output),
        "compressed_focus_rows": compressed,
        "counterfactual_rows": augmented,
        "counterfactual_source_rows_skipped_unobservable_id": skipped_unobservable_ids,
        "counterfactual_variants_per_focus_row": variants,
        "datasets": dict(Counter(str(row["metadata"].get("dataset")) for row in output)),
        "tools": dict(Counter(str(row["metadata"].get("target_tool")) for row in output)),
        "source_trajectories_modified": False,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    rows, report = build(_read_jsonl(args.input), variants=args.variants)
    _write_jsonl(args.output, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
