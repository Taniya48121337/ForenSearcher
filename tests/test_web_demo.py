import asyncio
import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")
from foren_searcher import web_demo
from foren_searcher.store import GlobalStore
from foren_searcher.web_demo_data import InvestigationReader, read_json, write_json


@pytest.fixture
def website(tmp_path, monkeypatch):
    harness = tmp_path / "harness/runtime"
    harness.mkdir(parents=True)
    (harness / "tool_executor.py").write_text("")
    catalog = tmp_path / "catalog.json"
    write_json(catalog, [{"id":"sample", "title":"Sample", "task":"Inspect the supplied source", "dataset":"Fixture",
                         "media":[], "tags":[], "record":"saved", "source_url":"https://example.com"}])
    write_json(tmp_path / ".local/openrouter.json", {"model":"qwen/qwen3.8-27b", "budget_usd":24,
               "encrypted_key":"not-a-real-test-secret"})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return web_demo.create_app(project=tmp_path, harness_root=tmp_path, catalog=catalog)


def client_for(app, **kwargs):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://testserver",
                             headers={"X-Foren-Request":"1"}, **kwargs)


@pytest.mark.asyncio
async def test_website_calls_original_runtime_with_only_task_and_materials(website, monkeypatch):
    calls=[]
    async def run(**kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True)
        write_json(kwargs["output_dir"] / "result.json", {"decision":{"answer":"Real runtime output"}})
    monkeypatch.setattr(web_demo,"run_demo",run)
    async with client_for(website) as client:
        response=await client.post("/api/runs",json={"case_id":"sample"})
        assert response.status_code==201
        run_id=response.json()["id"]
        await website.state.website.tasks[run_id]
        assert len(calls)==1
        assert calls[0]["task"]=="Inspect the supplied source"
        assert calls[0]["model_backend"]=="openrouter"
        assert calls[0]["media_paths"]==()
        assert "expected" not in calls[0] and "reference" not in calls[0]
        meta=(await client.get("/api/runs")).json()[0]
        assert meta["status"]=="completed"
        assert meta["id"]==run_id
        assert "not-a-real-test-secret" not in json.dumps(meta)


@pytest.mark.asyncio
async def test_duplicate_run_is_rejected_and_stop_cancels_runtime(website, monkeypatch):
    entered=asyncio.Event();cancelled=asyncio.Event()
    async def run(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(web_demo,"run_demo",run)
    async with client_for(website) as client:
        first=(await client.post("/api/runs",json={"task":"Investigate"})).json()
        await entered.wait()
        assert (await client.post("/api/runs",json={"task":"Duplicate"})).status_code==409
        assert (await client.post(f"/api/runs/{first['id']}/stop")).json()["status"]=="stopping"
        await website.state.website.tasks[first["id"]]
        assert cancelled.is_set()
        assert website.state.website.jobs[first["id"]]["status"]=="stopped"
        assert (await client.post(f"/api/runs/{first['id']}/stop")).status_code==409


@pytest.mark.asyncio
@pytest.mark.parametrize("origin",["https://evil.example","http://127.0.0.1:9999","null"])
async def test_cross_origin_cannot_spend_or_read_evidence(website,origin):
    async with client_for(website) as client:
        assert (await client.post("/api/runs",json={"task":"x"},headers={"Origin":origin})).status_code==403
        assert (await client.get("/api/cases",headers={"Origin":origin})).status_code==403
    assert not website.state.website.tasks


@pytest.mark.asyncio
async def test_dns_rebinding_and_missing_browser_header_are_rejected(website):
    async with client_for(website) as client:
        assert (await client.get("/api/cases",headers={"Host":"evil.example"})).status_code==400
        client.headers.pop("X-Foren-Request")
        assert (await client.post("/api/runs",json={"task":"x"})).status_code==403


@pytest.mark.asyncio
async def test_public_status_and_assets_never_include_saved_key(website):
    async with client_for(website) as client:
        status=await client.get("/api/status")
        assert status.status_code==200 and status.json()["key_configured"]
        assert "encrypted_key" not in status.text and "not-a-real-test-secret" not in status.text
        assert (await client.get("/.local/openrouter.json")).status_code==404
        assert (await client.get("/assets/../openrouter_settings.py")).status_code==404
        index=await client.get("/")
        assert "unsafe-eval" not in index.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in index.headers["content-security-policy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name",["../key.png","..\\key.png","a.html","a.svg","C:secret.png"])
async def test_upload_rejects_paths_and_active_documents(website,name):
    async with client_for(website) as client:
        response=await client.post("/api/uploads",params={"name":name},content=b"payload")
        assert response.status_code==400


@pytest.mark.asyncio
async def test_upload_size_limit_removes_partial_file(website,monkeypatch):
    monkeypatch.setattr(web_demo,"MAX_UPLOAD_BYTES",8)
    async with client_for(website) as client:
        assert (await client.post("/api/uploads?name=large.png",content=b"123456789")).status_code==413
    uploads=website.state.website.root/"uploads"
    assert not list(uploads.rglob("*.png")) and not list(uploads.rglob("metadata.json"))


@pytest.mark.asyncio
async def test_uploaded_media_is_preserved_for_real_run_and_rerun(website,monkeypatch):
    calls=[]
    async def run(**kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(web_demo,"run_demo",run)
    async with client_for(website) as client:
        upload=(await client.post("/api/uploads?name=材料.png",content=b"sample-image-bytes")).json()
        response=await client.post("/api/runs",json={"task":"Read image", "upload_ids":[upload["id"]]})
        job=response.json()
        await website.state.website.tasks[job["id"]]
        assert job["upload_ids"]==[upload["id"]]
        assert Path(calls[0]["media_paths"][0]).read_bytes()==b"sample-image-bytes"


@pytest.mark.asyncio
async def test_invalid_and_chunked_oversize_request_do_not_start_work(website):
    async def chunks():
        yield b'{"task":"'+b"x"*100001
        yield b'"}'
    async with client_for(website) as client:
        assert (await client.post("/api/runs",content=chunks())).status_code==413
        assert (await client.post("/api/runs",content=b"not json")).status_code==400
        assert (await client.post("/api/runs",json=[])).status_code==400
        assert (await client.post("/api/runs",json={"case_id":"does-not-exist"})).status_code==400
        assert (await client.post("/api/runs",json={"task":"x","upload_ids":["../private"]})).status_code==400
    assert not website.state.website.tasks


def test_restart_preserves_finished_and_marks_unfinished_without_reexecuting(website):
    site=website.state.website
    write_json(site.jobs_dir/"interrupted/metadata.json",{"id":"interrupted","status":"running","started_at":1})
    write_json(site.jobs_dir/"finished/metadata.json",{"id":"finished","status":"running","started_at":1})
    write_json(site.jobs_dir/"finished/investigation/result.json",{"decision":{"answer":"Saved answer"}})
    restored=web_demo.DemoWebsite(site.project,site.harness_root,site.config,site.catalog_path)
    assert restored.jobs["interrupted"]["status"]=="interrupted"
    assert restored.jobs["finished"]["status"]=="completed"
    assert not restored.tasks


@pytest.mark.asyncio
async def test_trace_projection_and_html_download_use_actual_saved_records(website):
    directory=website.state.website.project/"saved"
    store=GlobalStore(directory/"investigation.sqlite3",directory/"artifacts")
    e=store.commit_derived_evidence(artifact_bytes=b'<html><script>alert(1)</script></html>',media_type="text/html",suffix=".html",
        role="source_material",modalities=("text",),label="Saved source",content="Source text",
        derived_from=(),locator="paragraph 1",acquired_by="SEARCH",discovered_by="W1-1",production_method="source_excerpt")
    first=store.record_event(actor_role="HI",actor_id="W1-1",event_type="raw_model_output",payload={
        "reasoning":"Actual returned analysis", "raw_output":"Actual returned JSON", "model":"qwen/qwen3.8-27b", "image_evidence_ids":[]})
    second=store.record_event(actor_role="HI",actor_id="W1-2",event_type="raw_model_output",payload={"reasoning":"Separate workspace analysis"})
    write_json(directory/"result.json",{"decision":{"answer":"Source text (E1)"},
        "model_usage":{"request_count":1,"actual_cost_usd":0,"unsettled_reserved_usd":0.08}})
    write_json(directory/"billing-verification.json",{"request_count":1,"actual_cost_usd":0.01,"unsettled_reserved_usd":0})
    write_json(directory/"content-verification.json",{"flow_passed":True,"content_verdict":"fail","remaining_issues":["Incorrect conclusion"]})
    before=store.trajectory_events()
    async with client_for(website) as client:
        snapshot=(await client.get(f"/api/runs/example-sample?after={first}")).json()
        assert [x["id"] for x in snapshot["events"]]==[second]
        assert snapshot["events"][0]["actor_id"]=="W1-2"
        assert snapshot["events"][0]["reasoning_preview"]=="Separate workspace analysis"
        assert snapshot["usage"]["actual_cost_usd"]==0.01 and snapshot["usage"]["unsettled_reserved_usd"]==0
        assert snapshot["verification"]["flow_passed"] is True and snapshot["verification"]["content_verdict"]=="fail"
        detail=(await client.get(f"/api/runs/example-sample/events/{first}")).json()
        assert detail["payload"]["reasoning"]=="Actual returned analysis"
        material=(await client.get(f"/api/runs/example-sample/evidence/{e.evidence_id}")).json()
        assert material["content"]=="Source text"
        file=await client.get(f"/api/runs/example-sample/evidence/{e.evidence_id}/file")
        assert file.headers["content-disposition"].startswith("attachment")
        assert file.headers["content-type"]=="application/octet-stream"
        assert file.content==b'<html><script>alert(1)</script></html>'
        assert (await client.get('/api/runs/example-sample/evidence/not-an-evidence/file')).status_code==404
    assert store.trajectory_events()==before
    assert read_json(directory/"result.json")["model_usage"]["unsettled_reserved_usd"]==0.08


def test_empty_investigation_has_no_fabricated_steps(tmp_path):
    snapshot=InvestigationReader(tmp_path).snapshot()
    assert snapshot["events"]==[] and snapshot["evidence"]==[] and snapshot["answer"]==""


def test_inspect_projection_links_legacy_reuse_and_keeps_inputs_out_of_results(tmp_path):
    from foren_searcher.inspect.contracts import parse_inspect_call
    store=GlobalStore(tmp_path/'investigation.sqlite3',tmp_path/'artifacts')
    arguments={'targets':['E1'],'request':{'operation':'get_frames','params':{'times_s':[0,1]}}}
    observations=[]
    for actor,reused in [('W1-1',False),('W1-2',True),('W1-1',True)]:
        output_id=store.record_event(actor_role='HI',actor_id=actor,event_type='model_output',
            payload={'output':{'action':'INSPECT','arguments':arguments,'focus_evidence_ids':[]}})
        execution_id=store.create_inspect_execution(parse_inspect_call(arguments).to_dict(),actor)
        store.finish_inspect_execution(execution_id,status='ready',result_evidence_id=None,feedback=[],
            details={'reused':reused,'created_evidence_ids':[] if reused else ['E3','E4'],
                     'reused_evidence_ids':['E1','E2','E3','E4'] if reused else []})
        observations.append(store.record_event(actor_role='HI',actor_id=actor,event_type='observation',
            payload={'observation':{'action':'INSPECT','items':['Saved output'],'details':{
                'operation':'get_frames','status':'ready','result_evidence_id':None,
                'result':{'frames':[{'evidence_id':'E3','time_s':0},{'evidence_id':'E4','time_s':1}]},
                'evidence':[{'evidence_id':e} for e in ['E1','E2','E3','E4']]}}}))
    reader=InvestigationReader(tmp_path)
    before=store.trajectory_events()
    events=[e for e in reader.snapshot()['events'] if e['event_type']=='observation']
    assert [e['observation']['execution']['id'] for e in events]==[1,2,3]
    assert [e['observation']['execution']['reused'] for e in events]==[False,True,True]
    assert all(e['observation']['result_evidence_ids']==['E3','E4'] for e in events)
    incremental=reader.snapshot(after=output_id)['events']
    assert incremental[0]['id']==observations[-1]
    assert incremental[0]['observation']['execution']['id']==3
    assert store.trajectory_events()==before


def test_model_retry_events_preserve_actor_and_wait_for_website(tmp_path):
    store=GlobalStore(tmp_path/"investigation.sqlite3",tmp_path/"artifacts")
    waiting={"state":"waiting","wait_seconds":30,"resume_at":1788883000,"retry_number":1,"maximum_retries":2}
    store.record_event(actor_role="HI",actor_id="W1-2",event_type="model_retry",payload=waiting)
    event=InvestigationReader(tmp_path).snapshot()["events"][0]
    assert event["actor_id"]=="W1-2" and event["output"]==waiting


@pytest.mark.asyncio
async def test_stop_before_worker_starts_still_persists_terminal_status(website,monkeypatch):
    called=[]
    async def run(**kwargs):
        called.append(kwargs)
    monkeypatch.setattr(web_demo,"run_demo",run)
    site=website.state.website
    job=await site.start({"task":"Cancel immediately"})
    task=site.tasks[job["id"]]
    task.cancel()
    await asyncio.gather(task,return_exceptions=True)
    assert not called
    assert read_json(site.jobs_dir/job["id"]/"metadata.json")["status"]=="stopped"
