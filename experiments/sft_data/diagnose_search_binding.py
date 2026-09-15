"""Capture public search HTML to inspect candidate/source field binding."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ddgs.ddgs import DDGS


class CapturingDdgs(DDGS):
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.engine_selections = []

    def _get_engines(self, category, backend):
        if backend == "bing_direct":
            from ddgs.engines.bing import Bing
            engines = [Bing(timeout=self._timeout, verify=self._verify)]
        else:
            engines = super()._get_engines(category, backend)
        self.engine_selections.append({"requested_backend": backend,
                                       "actual_engines": [e.name for e in engines]})
        for engine in engines:
            if getattr(engine, "_pilot_captured", False):
                continue
            engine._pilot_captured = True
            original = engine.request

            def capture(*args, _engine=engine, _original=original, **kwargs):
                value = _original(*args, **kwargs)
                if value:
                    path = self.directory / (_engine.name + ".html")
                    path.write_text(value, encoding="utf-8")
                    tree = _engine.extract_tree(_engine.pre_process_html(value))
                    items = tree.xpath(_engine.items_xpath)
                    parsed = _engine.post_extract_results(_engine.extract_results(value))
                    (path.with_suffix(".json")).write_text(json.dumps({
                        "engine": _engine.name, "provider": _engine.provider,
                        "items_xpath": _engine.items_xpath,
                        "items": [{"path": tree.getroottree().getpath(item),
                                   "attributes": dict(item.attrib),
                                   "href_matches": item.xpath(_engine.elements_xpath["href"]),
                                   "child_tags": [child.tag for child in item]}
                                  for item in items],
                        "results": [r.__dict__ for r in parsed]}, ensure_ascii=False, indent=2), encoding="utf-8")
                return value

            engine.request = capture
        return engines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="brave")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / args.output
    output.mkdir(exist_ok=False)
    database = root / "runs/averimatec_63_q0_source_policy_r1/store.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        request = json.loads(connection.execute("SELECT request_json FROM search_executions WHERE id=6").fetchone()[0])
    client = CapturingDdgs(output, timeout=8)
    result = {"query": request["query"], "requested_backend": args.backend}
    try:
        result["results"] = client.text(request["query"], region="us-en", safesearch="moderate",
                                       backend=args.backend, max_results=10)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result["engine_selections"] = client.engine_selections
    (output / "capture.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "engine_selections": client.engine_selections,
                      "results": len(result.get("results", [])), "error": result.get("error")}))


if __name__ == "__main__":
    main()
