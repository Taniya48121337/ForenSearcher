from __future__ import annotations

from collections import Counter

from recovery_case_manifest import build_manifest, preflight_job


def test_manifest_covers_all_local_accepted_terminal_checkpoints() -> None:
    manifest = build_manifest()
    jobs = manifest["jobs"]
    assert manifest["oracle_fields_included"] is False
    assert manifest["private_generation_manifest"] is True
    assert Counter(job["dataset"] for job in jobs) == {
        "VideoSham": 10,
        "VCSL": 10,
        "VidDiffBench": 8,
    }
    assert [job["episode_id"] for job in jobs] == [
        f"batch-recovery-{index:03d}" for index in range(1, 29)
    ]
    assert all(job["intervention"]["id"] not in job["episode_id"] for job in jobs)


def test_manifest_preflight_resolves_every_local_resource() -> None:
    manifest = build_manifest()
    results = [preflight_job(job) for job in manifest["jobs"]]
    assert len(results) == 28
    assert all(item["passed"] for item in results)
