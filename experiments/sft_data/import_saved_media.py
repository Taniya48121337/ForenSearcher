"""Import a complete, saved local-media evidence set without its old investigation."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def import_media_evidence(source: Path, store, evidence_ids: list[str]):
    mapping, origins = {}, []
    with sqlite3.connect((source / "store.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM evidence ORDER BY id")]
        expected = [f"E{i}" for i in range(1, len(rows) + 1)]
        # Saved analysis text can contain E/A references; preserve these IDs exactly.
        if [f"E{row['id']}" for row in rows] != expected or set(evidence_ids) != set(expected):
            raise ValueError("Media replay requires the complete contiguous saved evidence set")
        for row in rows:
            old_id = f"E{row['id']}"
            parents = tuple(json.loads(row["derived_from_json"]))
            artifact = None
            raw = None
            path = None
            if row["artifact_id"] is not None:
                artifact = dict(connection.execute("SELECT * FROM artifacts WHERE id=?", (row["artifact_id"],)).fetchone())
                path = (source / "artifacts" / artifact["relative_path"]).resolve()
                if not path.is_relative_to((source / "artifacts").resolve()):
                    raise ValueError("Source artifact is outside its store")
                raw = path.read_bytes()
                if len(raw) != artifact["byte_size"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                    raise ValueError("Saved media failed its existing artifact integrity check")
            if not parents:
                if row["acquired_by"] != "HARNESS/ingest" or artifact is None:
                    raise ValueError("Only original ingested media and its derived evidence are supported")
                content = json.loads(row["content"])
                new = store.commit_ingested_evidence(
                    source_ref=row["source_ref"], media_type=artifact["media_type"],
                    modalities=tuple(json.loads(row["modalities_json"])), label=row["label"],
                    original_filename=content["original_filename"], data=raw,
                    discovered_by="provided_saved_media",
                )
            else:
                if row["source_ref"] is not None or any(parent not in mapping for parent in parents):
                    raise ValueError("Derived evidence must have already imported local-media parents")
                new = store.commit_derived_evidence(
                    derived_from=parents, modalities=tuple(json.loads(row["modalities_json"])),
                    label=row["label"], content=row["content"], locator=row["locator"],
                    acquired_by=row["acquired_by"], discovered_by="provided_saved_media",
                    production_method=row["production_method"], artifact_bytes=raw,
                    media_type=artifact["media_type"] if artifact else None,
                    suffix=path.suffix if path else ".bin",
                    role=artifact["role"] if artifact else "derived_material",
                )
            expected_artifact = f"A{row['artifact_id']}" if artifact else None
            if new.evidence_id != old_id or new.artifact_ref != expected_artifact:
                raise ValueError("Saved evidence references changed during import")
            mapping[old_id] = new.evidence_id
            origins.append({"source_run": source.name, "source_evidence_id": old_id,
                            "evidence_id": new.evidence_id, "artifact_ref": new.artifact_ref,
                            "source_acquired_by": row["acquired_by"],
                            "source_discovered_by": row["discovered_by"],
                            "source_created_at": row["created_at"],
                            "source_artifact": artifact, "derived_from": list(parents),
                            "new_tool_execution": False})
    return mapping, origins
