"""Measure actual search responses and local parsing on recorded task queries."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ddgs.engines.bing import Bing
from ddgs.engines.brave import Brave
from ddgs.engines.duckduckgo import Duckduckgo
from ddgs.engines.google import Google
from ddgs.engines.yahoo import Yahoo
from lxml import html

from source_bound_search import parse_bound_results

ROOT = Path(__file__).resolve().parent
ENGINES = {e.name: e for e in (Brave, Bing, Google, Yahoo, Duckduckgo)}


def probe(job):
    index, query, name, directory = job
    engine = ENGINES[name](timeout=10)
    started = time.monotonic()
    record = {"query": query, "engine": name, "method": engine.search_method}
    try:
        payload = engine.build_payload(query, "us-en", "moderate", None, page=1)
        record.update(request_url=engine.search_url, request_payload=payload)
        kwargs = {"params" if engine.search_method == "GET" else "data": payload}
        response = engine.http_client.request(engine.search_method, engine.search_url, **kwargs)
        record.update(status=response.status_code, html=response.text)
        tree = html.fromstring(response.text)
        record["page_title"] = " ".join(tree.xpath("//title/text()"))
        if response.status_code == 200:
            try:
                record["installed_results"] = [r.__dict__ for r in engine.post_extract_results(engine.extract_results(response.text))]
            except Exception as error:
                record["installed_error"] = {"type": type(error).__name__, "message": str(error)}
            try:
                record["bound_results"], record["blocks"] = parse_bound_results(engine, response.text)
            except Exception as error:
                record["bound_error"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        record["request_error"] = {"type": type(error).__name__, "message": str(error)}
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    filename = f"query_{index:02d}_{name}.json"
    (directory / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"file": filename, **{k: v for k, v in record.items() if k not in {"html", "installed_results", "bound_results", "blocks"}},
               "installed_count": len(record.get("installed_results", [])),
               "bound_results": record.get("bound_results", [])}
    print(json.dumps({"file": filename, "status": record.get("status"),
                      "installed_count": summary["installed_count"], "bound_count": len(summary["bound_results"]),
                      "page_title": record.get("page_title"), "error": record.get("request_error")}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--engines", default="brave,bing,google,yahoo,duckduckgo")
    args = parser.parse_args()
    directory = ROOT / args.output
    directory.mkdir(exist_ok=False)
    queries = []
    for run in ("averimatec_63_q0_source_bound_r1", "averitec_2872_q0_binding_validation", "averitec_2694_q8_source_validation"):
        database = ROOT / "runs" / run / "store.sqlite3"
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            request = json.loads(connection.execute("SELECT request_json FROM search_executions WHERE request_json LIKE '%query%' ORDER BY id LIMIT 1").fetchone()[0])
        queries.append({"run": run, "query": request["query"]})
    (directory / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = [(i, q["query"], engine, directory) for i, q in enumerate(queries, 1) for engine in args.engines.split(",")]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(probe, jobs))
    (directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
