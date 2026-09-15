"""Locate original fact-check articles and known same-case URL variants."""
from __future__ import annotations

import json
import sys

from gia_states import ROOT, load
from foren_searcher import GlobalStore
from source_policy import ArticleExclusion, canonical_url


def urls_in_payload(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_url", "final_url"} and isinstance(item, str):
                yield item
            else:
                yield from urls_in_payload(item)
    elif isinstance(value, list):
        for item in value:
            yield from urls_in_payload(item)


def main():
    results = []
    for run in sorted((ROOT / "runs").iterdir()):
        if not (run / "run_config.json").is_file() or not (run / "store.sqlite3").is_file():
            continue
        config = load(run / "run_config.json")
        dataset = config["dataset"].lower()
        if dataset not in {"averitec", "averimatec"}:
            continue
        reference = load(ROOT / "data" / dataset / str(config["case"]) / "reference.json")
        article = reference.get("fact_checking_article", reference.get("article", ""))
        exclusion = ArticleExclusion(article, run / "source_exclusions.jsonl") if article else None
        store = GlobalStore(run / "store.sqlite3", run / "artifacts")
        candidates = {r["candidate_id"]: r for r in store.candidate_records()}
        matched = []
        for evidence in store.evidence_records():
            candidate = candidates.get(evidence.source_ref, {})
            urls = [candidate.get("source_url", ""), candidate.get("final_url", "")]
            if exclusion and any(exclusion.matches(url) for url in urls):
                matched.append({"evidence_id": evidence.evidence_id, "source_ref": evidence.source_ref,
                                "source_url": candidate.get("source_url"), "final_url": candidate.get("final_url")})
        visible = []
        for request in sorted((run / "requests").glob("request_*")):
            payload = load(request / "request.json") if (request / "request.json").is_file() else {}
            urls = sorted({url for url in urls_in_payload(payload.get("payload", {}))
                           if exclusion and exclusion.matches(url)})
            if urls:
                visible.append({"request": request.name, "matching_source_urls": urls})
        results.append({"run": run.name, "dataset": dataset, "case": str(config["case"]),
                        "original_fact_check_article": article, "matching_acquired_evidence": matched,
                        "matching_source_fields_in_model_inputs": visible,
                        "evidence_source_policy": config.get("evidence_source_policy", "unrestricted_retrospective_web"),
                        "text_search_parser": config.get("text_search_parser", "installed_ddgs"),
                        "known_article_keys": sorted(exclusion.keys) if exclusion else []})
    summary = {"scope": "known_URL_and_explicit_alias_comparison; model_input_source_fields; acquisition_not_proof_of_citation",
               "time_policy": "unrestricted_retrospective_web; claim_date_filter_not_enforced",
               "limitations": "no_match_does_not_rule_out_mirrors_or_post_claim_sources; dates_not_verified; text_only_URL_mentions_not_exhaustively_matched; known_alias_equivalence_basis_in_source_article_aliases.json",
               "runs_checked": len(results), "runs_with_matches": sum(bool(x["matching_acquired_evidence"]) for x in results),
               "results": results}
    (ROOT / "evidence_source_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
