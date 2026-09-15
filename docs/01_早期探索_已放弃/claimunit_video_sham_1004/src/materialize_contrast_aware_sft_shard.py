"""Materialize one compact contrast-aware schedule shard into trainer JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_contrast_aware_sft_schedule import (
    DEFAULT_OUTPUT,
    DEFAULT_ROWS,
    _read_jsonl,
    _write_jsonl,
    materialize_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument(
        "--schedule-shard",
        type=Path,
        default=DEFAULT_OUTPUT / "schedules/epoch_001/shard_001.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "materialized_epoch001_shard001.jsonl",
    )
    args = parser.parse_args()
    source_rows = _read_jsonl(args.rows)
    batches = _read_jsonl(args.schedule_shard)
    materialized = materialize_schedule(source_rows, batches)
    _write_jsonl(args.output, materialized)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "batches": len(batches),
                "rows": len(materialized),
                "messages_and_tools_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
