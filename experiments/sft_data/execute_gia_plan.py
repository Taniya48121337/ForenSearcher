"""Execute a recorded GIA plan with its provided evidence in a fresh investigation."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from run_case import ROOT, PROJECT
from cli_generator import CliGenerator
from import_saved_media import import_media_evidence
from source_policy import article_key, exclude_original_article
from source_bound_search import PARSER_VERSION, SourceBoundTextProvider
from foren_searcher import GlobalStore
from foren_searcher.harness1 import harness1_workspace_type
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import TransformersGiaPolicy, TransformersHiPolicy
from foren_searcher.memory import MemoryPolicy
from foren_searcher.orchestration import InvestigationRuntime
from foren_searcher.search.factory import create_free_search_service


def import_source_evidence(source: Path, store: GlobalStore, evidence_ids: list[str]):
    """Copy saved source text and bytes, retaining acquisition provenance separately."""
    mapping, origins, candidate_mapping = {}, [], {}
    with sqlite3.connect((source / "store.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for old_id in evidence_ids:
            row = dict(connection.execute("SELECT * FROM evidence WHERE id=?", (int(old_id[1:]),)).fetchone())
            if json.loads(row["derived_from_json"]) or row["production_method"] != "source_excerpt":
                raise ValueError("This importer supports source excerpts, not derived evidence")
            old_candidate = row["source_ref"]
            candidate = dict(connection.execute("SELECT * FROM candidates WHERE id=?", (int(old_candidate[1:]),)).fetchone())
            artifact = dict(connection.execute("SELECT * FROM artifacts WHERE id=?", (row["artifact_id"],)).fetchone())
            relations = connection.execute("SELECT * FROM candidate_evidence WHERE evidence_id=?", (row["id"],)).fetchall()
            if len(relations) != 1 or relations[0]["candidate_id"] != candidate["id"]:
                raise ValueError("Source material must have one matching candidate relationship")
            if old_candidate not in candidate_mapping:
                new_candidate, _ = store.get_or_create_candidate(
                    identity_key=candidate["identity_key"], source_url=candidate["source_url"],
                    final_url=candidate["final_url"], candidate_type=candidate["candidate_type"],
                    search_title=candidate["search_title"], provider="provided_saved_source",
                    extra={"source_run": source.name, "source_candidate_id": old_candidate},
                )
                candidate_mapping[old_candidate] = new_candidate
                if candidate["final_url"]:
                    store.update_candidate_final_url(new_candidate, candidate["final_url"], "url:" + candidate["final_url"])
            artifact_path = (source / "artifacts" / artifact["relative_path"]).resolve()
            if not artifact_path.is_relative_to((source / "artifacts").resolve()):
                raise ValueError("Source artifact is outside its store")
            raw = artifact_path.read_bytes()
            if len(raw) != artifact["byte_size"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                raise ValueError("Source artifact failed its existing integrity check")
            new = store.commit_material_evidence(
                candidate_id=candidate_mapping[old_candidate],
                material_kind=relations[0]["material_kind"], material_status=relations[0]["material_status"],
                artifact_bytes=raw, media_type=artifact["media_type"], suffix=artifact_path.suffix,
                modalities=tuple(json.loads(row["modalities_json"])), label=row["label"],
                content=row["content"], locator=row["locator"], acquired_by="provided_saved_source",
                discovered_by="provided_input", production_method=row["production_method"], role=artifact["role"],
            )
            mapping[old_id] = new.evidence_id
            origins.append({"source_run": source.name, "source_evidence_id": old_id,
                            "source_candidate_id": old_candidate, "source_artifact": artifact,
                            "source_acquired_by": row["acquired_by"], "source_discovered_by": row["discovered_by"],
                            "source_created_at": row["created_at"], "evidence_id": new.evidence_id,
                            "candidate_id": new.source_ref, "artifact_ref": new.artifact_ref,
                            "source_url": candidate["source_url"], "final_url": candidate["final_url"]})
    return mapping, origins


class RecordedFirstPlan:
    def __init__(self, decision: dict, policy):
        self.decision, self.policy, self.used = decision, policy, False

    async def decide(self, *, model_view, history, actions):
        if not self.used:
            self.used = True
            return copy.deepcopy(self.decision)
        return await self.policy.decide(model_view=model_view, history=history, actions=actions)


async def run(args):
    state = ROOT / "gia_decisions" / args.construction / args.state
    reference = json.loads((state / "reference.json").read_text(encoding="utf-8"))
    provenance = reference["provenance"]
    source = ROOT / "runs" / provenance["source_run"]
    source_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    request = json.loads((state / "requests/request_0001/request.json").read_text(encoding="utf-8"))
    task = request["payload"]["global_view"]["task"]
    decision = json.loads((state / "decision.json").read_text(encoding="utf-8"))
    if decision.get("decision") != "INVESTIGATE":
        raise ValueError("Select a state with a recorded INVESTIGATE decision")
    output = ROOT / "runs" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    store = GlobalStore(output / "store.sqlite3", output / "artifacts")
    media_case = source_config["dataset"] in {"videosham", "qvhighlights", "vcsl"}
    importer = import_media_evidence if media_case else import_source_evidence
    mapping, origins = importer(source, store, provenance["source_evidence_ids"])
    replay = copy.deepcopy(decision)
    for item in replay["tasks"]:
        item["materials"] = [mapping[eid] for eid in item["materials"]]
    plan_origin = {"source_state": str(state.relative_to(ROOT)), "source_request": "requests/request_0001/request.json",
                   "recorded_decision": decision, "replayed_decision": replay, "evidence_id_mapping": mapping,
                   "provided_sources": origins, "condition": "recorded_plan_execution_with_provided_evidence",
                   "first_plan_is_new_model_target": False,
                   "initial_report_history": "not_imported; fresh investigation begins with recorded tasks and saved source evidence"}
    (output / "plan_origin.json").write_text(json.dumps(plan_origin, ensure_ascii=False, indent=2), encoding="utf-8")
    instruction_path = Path(args.instruction_file) if args.instruction_file else ROOT / "actor_generation_r5.txt"
    instruction = instruction_path.read_text(encoding="utf-8")
    generator = CliGenerator(store=store, output_dir=output / "requests", instruction=instruction)

    def record_generation(value):
        workspace = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace else "GIA", actor_id=workspace or "GIA",
                           event_type="raw_model_output", payload=value)

    generator.generation_observer = record_generation
    search = None
    exclusion_keys = []
    if not media_case:
        search = create_free_search_service(store=store, reverse_engine="none")
        search.provider = SourceBoundTextProvider(search.provider, capture_dir=output / "search_captures")
        data_reference = json.loads((ROOT / "data" / source_config["dataset"] / str(source_config["case"]) / "reference.json").read_text(encoding="utf-8"))
        article = data_reference.get("fact_checking_article", data_reference.get("article"))
        exclude_original_article(search, article, output / "source_exclusions.jsonl")
        exclusion_keys = sorted(search.provider.exclusion.keys)
    aliases = [group for group in json.loads((ROOT / "source_article_aliases.json").read_text(encoding="utf-8"))
               if article_key(group["reference_url"]) in exclusion_keys]
    inspect = create_inspect_service(store=store, enabled_operations=frozenset({"get_frames", "get_pages", "crop", "ocr", "align_media"}))
    harness_root = Path(source_config["harness_root"])
    config = {**source_config, "mode": "bridge", "question_index": None, "task": task,
              "gia_output_source": "recorded_first_plan_then_codex_cli",
              "source_condition": "recorded_plan_execution_with_provided_evidence", "provided_source_url": None,
              "text_search_parser": PARSER_VERSION if search else None, "instruction": instruction,
              "instruction_file": str(instruction_path.resolve()),
              "source_exclusion_keys": exclusion_keys, "source_exclusion_aliases": aliases,
              "configured_model": generator.model_config, "plan_origin": plan_origin,
              "memory_policy": {"max_read_chars": args.read_chars},
              "reference_scope": "task constructed from saved evidence; original dataset annotation is not automatically its ground-truth answer"}
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = InvestigationRuntime(
        store=store, gia_policy=RecordedFirstPlan(replay, TransformersGiaPolicy(generator)),
        hi_policy=TransformersHiPolicy(generator), search_service=search, inspect_service=inspect,
        memory_policy=MemoryPolicy(max_read_chars=args.read_chars),
        max_rounds=3, max_gia_turns=8, gia_turn_timeout_s=660, max_workspace_turns=12,
        workspace_timeout_s=3600, max_concurrency=2, max_workspaces_per_round=3,
        report_review_delay_s=None, workspace_harness_type=harness1_workspace_type(harness_root),
    )
    try:
        async with runtime:
            result = await runtime.run(task=task, original_material_ids=tuple(mapping.values()))
            (output / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"run": args.run_name, "runtime_returned": True}, ensure_ascii=False), flush=True)
    finally:
        store.export_trajectory(output / "trajectory.jsonl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--read-chars", type=int, default=6000)
    parser.add_argument("--instruction-file")
    asyncio.run(run(parser.parse_args()))
