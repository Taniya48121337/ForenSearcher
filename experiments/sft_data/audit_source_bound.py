"""Check saved source-block associations and replay known-article exclusions."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from gia_states import ROOT, load
from source_policy import ArticleExclusion, FilteredProvider
from foren_searcher.search.contracts import CandidateBatch, CandidateSource, SearchQueryRequest


async def replay_exclusion():
    run = ROOT / "runs/averimatec_660_q0_source_validation"
    with sqlite3.connect((run / "store.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM candidates WHERE discovered_execution_id=2 ORDER BY provider_rank")]
    actual = tuple(CandidateSource(row["source_url"], row["search_title"], row["search_snippet"]) for row in rows)
    class ReplayProvider:
        async def search(self, request, *, offset, limit, image_path=None):
            return CandidateBatch("actual_capture_replay", "recorded", actual, {})
    log = ROOT / "source_filter_replay.jsonl"
    # A new audit replaces this derived replay log, never the original trajectory.
    log.write_text("", encoding="utf-8")
    article = load(ROOT / "data/averimatec/660/reference.json")["article"]
    wrapper = FilteredProvider(ReplayProvider(), ArticleExclusion(article, log))
    result = await wrapper.search(SearchQueryRequest("text", "replay", "text"), offset=0, limit=len(actual))
    return {"scope": "replay_of_actual_candidates; no_new_search_or_actor_execution",
        "source_run": run.name, "execution_id": 2,
        "before_urls": [x.source_url for x in actual], "after_urls": [x.source_url for x in result.candidates],
        "excluded": [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]}


def main():
    results, errors = [], []
    for run in sorted((ROOT / "runs").iterdir()):
        captures = sorted((run / "search_captures").glob("*.json"))
        if not captures:
            continue
        checked, empty = 0, 0
        for path in captures:
            record = load(path)
            rows = record["results"]
            if not rows:
                empty += 1
                continue
            blocks = [x for x in record["captures"][-1]["blocks"] if x["outcome"] == "accepted"]
            checked += len(rows)
            if [x["source_url"] for x in blocks] != [x["href"] for x in rows] or any(x["href_count"] != 1 for x in blocks):
                errors.append({"run": run.name, "capture": path.name})
        results.append({"run": run.name, "calls": len(captures), "parsed_source_rows": checked,
                        "calls_without_usable_records": empty})
    summary = {"scope": "saved_parser_block_links; no_claim_of_source_truth_or_relevance",
               "runs": results, "association_errors": errors,
               "known_article_replay": asyncio.run(replay_exclusion())}
    (ROOT / "source_bound_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
