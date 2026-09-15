"""Verify that contrast batching is structure-matched and invisible to the policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_control_plane_exposure import _target_call
from build_recovery_semantic_contrast_view import (
    _abstract_target,
    _canonical,
    _control_signature,
    _jsonl,
)
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_ROOT = BASE / "recovery_sft_semantic_contrast_v4_state_transition55"
DEFAULT_OUTPUT = DEFAULT_ROOT / "contrast_integrity_audit.json"


def audit(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row["metadata"]["row_id"]): row for row in rows}
    details: list[dict[str, Any]] = []
    forbidden_hits: list[dict[str, str]] = []
    for group in groups:
        group_id = str(group["group_id"])
        family = str(group["family"])
        row_ids = [str(item) for item in group.get("row_ids") or []]
        members = [by_id.get(row_id) for row_id in row_ids]
        rows_exist = bool(row_ids) and all(member is not None for member in members)
        present_members = [member for member in members if member is not None]
        signatures = [_control_signature(member) for member in present_members]
        control_equal = bool(signatures) and all(
            signature == signatures[0] for signature in signatures[1:]
        )
        metadata_links_exact = rows_exist and all(
            sum(
                item.get("group_id") == group_id
                for item in (member["metadata"].get("semantic_contrast_groups") or [])
            )
            == 1
            for member in present_members
        )
        target_metadata_hidden = True
        for member in present_members:
            student_payload = json.dumps(
                {"messages": member["messages"], "tools": member.get("tools")},
                ensure_ascii=False,
                sort_keys=True,
            )
            if group_id in student_payload:
                target_metadata_hidden = False

        family_check = False
        abstract_targets: list[dict[str, Any] | None] = []
        if family == "same_control_surface_different_semantic_target":
            abstract_targets = [_abstract_target(member) for member in present_members]
            canonical_targets = [
                _canonical(item) for item in abstract_targets if item is not None
            ]
            target_counts = Counter(canonical_targets)
            count_values = list(target_counts.values())
            implied_pairs = sum(
                count_values[left] * count_values[right]
                for left in range(len(count_values))
                for right in range(left + 1, len(count_values))
            )
            neighborhood_ledger_valid = True
            if "distinct_abstract_targets" in group:
                neighborhood_ledger_valid = neighborhood_ledger_valid and (
                    group["distinct_abstract_targets"] == len(target_counts)
                )
            if "target_distribution" in group:
                neighborhood_ledger_valid = neighborhood_ledger_valid and (
                    group["target_distribution"]
                    == {target: target_counts[target] for target in sorted(target_counts)}
                )
            if "implied_cross_target_pairs" in group:
                neighborhood_ledger_valid = neighborhood_ledger_valid and (
                    group["implied_cross_target_pairs"] == implied_pairs
                )
            family_check = (
                len(present_members) >= 2
                and all(item is not None for item in abstract_targets)
                and len(set(canonical_targets)) >= 2
                and abstract_targets == group.get("abstract_targets")
                and neighborhood_ledger_valid
            )
        elif family == "same_menu_different_tool":
            target_tools = [_target_call(member)[0] for member in present_members]
            family_check = len(present_members) == 2 and len(set(target_tools)) == 2
        elif family == "same_menu_different_arguments":
            dispositions = [
                str(_target_call(member)[1].get("disposition"))
                for member in present_members
            ]
            family_check = {"retain", "reject"}.issubset(set(dispositions))
        elif family == "same_case_cross_variant_different_tool":
            target_tools = {_target_call(member)[0] for member in present_members}
            case_ids = {
                str(member["metadata"]["case_id"]) for member in present_members
            }
            family_check = (
                len(present_members) >= 2
                and target_tools == {"challenge_evidence", "stop"}
                and len(case_ids) == 1
                and group.get("same_case") is True
                and group.get("distinct_targets") == 2
                and int(group.get("distinct_episodes") or 0) >= 2
            )

        checks = {
            "rows_exist": rows_exist,
            "control_signature_equal": control_equal,
            "family_semantic_targets_differ": family_check,
            "metadata_links_exact": metadata_links_exact,
            "target_grouping_hidden_from_student": target_metadata_hidden,
        }
        details.append(
            {
                "group_id": group_id,
                "family": family,
                "row_ids": row_ids,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        serialized = json.dumps(group, ensure_ascii=False).lower()
        for term in ("oracle", "verifier", "private_intervention"):
            if term in serialized:
                forbidden_hits.append({"group_id": group_id, "term": term})

    grouped_row_ids = {
        row_id for group in groups for row_id in group.get("row_ids") or []
    }
    by_stage = Counter(
        str(by_id[row_id]["metadata"]["causal_recovery_stage"])
        for row_id in grouped_row_ids
        if row_id in by_id
    )
    by_dataset = Counter(
        str(by_id[row_id]["metadata"]["dataset"])
        for row_id in grouped_row_ids
        if row_id in by_id
    )
    all_row_metadata_groups = {
        str(item["group_id"])
        for row in rows
        for item in row["metadata"].get("semantic_contrast_groups") or []
    }
    declared_group_ids = {str(group["group_id"]) for group in groups}
    checks = {
        "all_groups_pass": all(item["passed"] for item in details),
        "no_duplicate_group_ids": len(declared_group_ids) == len(groups),
        "group_ledger_matches_row_metadata": (
            all_row_metadata_groups == declared_group_ids
        ),
        "forbidden_control_terms_absent": not forbidden_hits,
        "messages_tools_untouched_by_group_audit": True,
    }
    return {
        "artifact": "recovery_semantic_contrast_integrity_audit_v1",
        "rows": len(rows),
        "groups": len(groups),
        "covered_rows": len(grouped_row_ids),
        "covered_row_fraction": round(len(grouped_row_ids) / max(len(rows), 1), 6),
        "covered_rows_by_stage": dict(sorted(by_stage.items())),
        "covered_rows_by_dataset": dict(sorted(by_dataset.items())),
        "checks": checks,
        "forbidden_hits": forbidden_hits,
        "passed": all(checks.values()),
        "details": details,
        "scope_limit": (
            "This proves neighborhood construction and policy invisibility, not that a trained "
            "student uses the contrasts; student free-run ablation remains required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=Path, default=DEFAULT_ROOT / "recovery_sft_contrastive.jsonl"
    )
    parser.add_argument(
        "--groups", type=Path, default=DEFAULT_ROOT / "contrast_groups.jsonl"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(_jsonl(args.rows), _jsonl(args.groups))
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "groups": report["groups"],
                "covered_row_fraction": report["covered_row_fraction"],
                "passed": report["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
