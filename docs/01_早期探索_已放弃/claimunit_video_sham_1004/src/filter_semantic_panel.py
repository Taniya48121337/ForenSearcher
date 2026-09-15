"""Split a panel manifest into current-contract accepted and quarantine manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import write_json


def split(manifest_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    report_by_identity = {
        (str(row["dataset"]), str(row["case_id"]), str(row["mode"])): row
        for row in audit.get("episode_reports") or []
    }
    accepted = []
    quarantined = []
    for item in manifest.get("items") or []:
        identity = (str(item["dataset"]), str(item["case_id"]), str(item["mode"]))
        report = report_by_identity.get(identity)
        if report is not None and report.get("semantic_passed") is True:
            accepted.append(item)
            continue
        failure_codes = sorted(
            {
                str(code)
                for turn in (report or {}).get("turn_reports") or []
                for code in [*(turn.get("reasoning_problems") or []), *(turn.get("privacy_problems") or [])]
            }
        )
        quarantined.append({**item, "failure_codes": failure_codes})
    accepted_manifest = {
        "artifact": "current_contract_accepted_panel_manifest_v1",
        "episodes": len(accepted),
        "items": accepted,
    }
    quarantine_manifest = {
        "artifact": "current_contract_quarantine_panel_manifest_v1",
        "episodes": len(quarantined),
        "items": quarantined,
    }
    return accepted_manifest, quarantine_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True)
    args = parser.parse_args()
    accepted, quarantine = split(args.manifest, args.semantic_audit)
    write_json(args.accepted_output, accepted)
    write_json(args.quarantine_output, quarantine)
    print(
        json.dumps(
            {"accepted": accepted["episodes"], "quarantined": quarantine["episodes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
