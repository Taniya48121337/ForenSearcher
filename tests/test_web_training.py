from pathlib import Path

import httpx
import pytest

from foren_searcher.web_demo import create_app
from foren_searcher.web_demo_data import write_json
from foren_searcher.web_training import TrainingReader


@pytest.fixture
def training_project(tmp_path):
    root = tmp_path / "experiments/sft_data"
    run = root / "runs/videosham_4126_split_intervals_r7"
    for i in (1, 2):
        request_id = f"request_{i:04d}"
        directory = run / "requests" / request_id
        write_json(directory / "request.json", {"system_prompt": "Use only visible evidence.",
            "payload": {"workspace_id": "W1-1", "model_view": {"task": "Compare A and B", "history": []}},
            "image_evidence_ids": ["E1"], "image_notes": []})
        (directory / "actor").mkdir()
        (directory / "actor/image_01.png").write_bytes(b"test image")
    write_json(run / "run_config.json", {"dataset": "videosham", "case": "4126", "mode": "bridge"})
    write_json(run / "selected_assessment.json", {"assessment": "assessment_r6", "rubric_file": "rubric_r6.md"})
    write_json(run / "assessment_r6/judgment.json", {"roles": [{"actor": "W1-1", "dimensions": [
        {"id": "task", "rating": 2, "status": "scored", "reason": "Incomplete overall task"}]}],
        "target_findings": [
            {"request_id": "request_0001", "disposition": "accept_candidate", "target_rating": 4,
             "critical_unknown": False, "reason": "Useful comparison"},
            {"request_id": "request_0002", "disposition": "accept_candidate", "target_rating": 3,
             "critical_unknown": False, "reason": "Automated initial acceptance"}]})
    write_json(run / "execution_links.json", [
        {"request_id": f"request_{i:04d}", "actor": "W1-1", "role": "HI", "status": "executed",
         "raw_output": {"action": "READ_RECORD", "arguments": {"record_id": "E1"}},
         "observation_event_id": i*10} for i in (1, 2)])
    import json
    (run / "trajectory.jsonl").write_text("\n".join(json.dumps({"id": i*10, "event_type": "observation",
        "payload": {"observation": {"items": [f"Actual observation {i}"], "details": {"content": "Source text"}}}})
        for i in (1, 2)), encoding="utf-8")
    corpus = root / "corpus"
    write_json(corpus / "manifest.json", {"files": [{"file": "hi_development.jsonl", "role": "HI",
        "usage_split": "development", "rows": 1, "case_families": 1}], "images_used": 1})
    (corpus / "hi_development.jsonl").write_text(json.dumps({"metadata": {
        "run": run.name, "request": "request_0001", "dataset": "VideoSham", "role": "HI",
        "case": "4126", "usage_split": "development", "target_rating": 4}})+"\n", encoding="utf-8")
    (root / "rubric_r6.md").write_text("# Quality criteria", encoding="utf-8")
    return tmp_path


def test_actual_corpus_membership_controls_inclusion_not_aggregate_score(training_project):
    reader = TrainingReader(training_project)
    sample = reader.sample("video-intervals")
    assert (sample["step_count"], sample["included_count"], sample["excluded_count"]) == (2, 1, 1)
    assert sample["steps"][1]["disposition"] == "accept_candidate"
    assert sample["steps"][1]["included"] is False
    first = reader.step("video-intervals", "request_0001")
    assert first["rating"] == 4
    assert first["dimensions"][0]["rating"] == 2
    assert first["observation"]["items"] == ["Actual observation 1"]
    assert first["payload"]["model_view"]["history"] == []
    assert first["images"][0]["evidence_id"] == "E1"
    assert all(not k.startswith("_") for k in sample["steps"][0])


@pytest.mark.asyncio
async def test_training_routes_only_serve_selected_materials(training_project):
    app = create_app(project=training_project)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        overview = await client.get("/api/training")
        assert overview.status_code == 200
        assert overview.json()["stats"]["total"] == 1
        assert len(overview.json()["samples"]) == 1
        image = await client.get("/api/training/samples/video-intervals/steps/request_0001/images/0")
        assert image.content == b"test image"
        assert image.headers["content-type"] == "image/png"
        for url in ["/api/training/samples/not-selected", "/api/training/documents/config",
                    "/api/training/samples/video-intervals/steps/request_9999",
                    "/api/training/samples/video-intervals/steps/request_0001/images/-1",
                    "/api/training/samples/video-intervals/steps/request_0001/images/1"]:
            assert (await client.get(url)).status_code == 404
        assert (await client.get("/api/training", headers={"Origin": "https://external.example"})).status_code == 403
        assert "frame-ancestors 'none'" in overview.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_trial_training_requires_existing_authentication(training_project):
    app = create_app(project=training_project, trial=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        for url in ["/api/training", "/api/training/samples/video-intervals",
                    "/api/training/samples/video-intervals/steps/request_0001/images/0",
                    "/api/training/documents/rubric", "/assets/training.js"]:
            assert (await client.get(url)).status_code == 401


def test_corpus_changes_refresh_sample_counts(training_project):
    reader = TrainingReader(training_project)
    assert reader.sample("video-intervals")["included_count"] == 1
    path = training_project / "experiments/sft_data/corpus/hi_development.jsonl"
    path.write_text("", encoding="utf-8")
    assert reader.sample("video-intervals")["included_count"] == 0


def test_verification_summary_feeds_overview(training_project):
    root = training_project / "experiments/sft_data"
    reader = TrainingReader(training_project)
    assert reader.verification() == {}
    write_json(root / "verdicts_r0_judged.jsonl.summary.json", {"reports": 88})
    assert reader.verification()["reports"] == 88
    write_json(root / "verdicts_r0_judged.summary49.json", {"runs": 49, "reports": 80, "pairs": 641, "findings": 203,
        "by_dataset": {"true": {"n": 15}, "videosham": {"n": 21}, "unknown": {"n": 2}}, "verifier": "verifier@0.1.0"})
    stats = reader.overview()["verification"]
    assert (stats["runs"], stats["reports"], stats["pairs"], stats["datasets"]) == (49, 80, 641, 2)
    assert stats["source"] == "verdicts_r0_judged.summary49.json"

