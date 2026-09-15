import asyncio
import json
import time

import httpx
import pytest

pytest.importorskip("fastapi")
from foren_searcher import web_demo
from foren_searcher.web_access import COOKIE_NAME, TrialAccess
from foren_searcher.web_demo_data import write_json


@pytest.fixture
def app(tmp_path, monkeypatch):
    runtime = tmp_path / "harness/runtime"
    runtime.mkdir(parents=True)
    (runtime / "tool_executor.py").touch()
    catalog = tmp_path / "cases.json"
    write_json(catalog, [{"id": "sample", "title": "Sample", "task": "Read this chart", "media": []}])
    write_json(tmp_path / ".local/openrouter.json", {"model": "qwen/qwen3.8-27b", "budget_usd": 24, "encrypted_key": "fixture"})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return web_demo.create_app(project=tmp_path, harness_root=tmp_path, catalog=catalog, trial=True, daily_runs=2)


def client(app, base_url="http://testserver"):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url, headers={"X-Foren-Request": "1"})


async def sign_in(http, app, name):
    invite = app.state.access.invite(name)
    result = await http.post("/api/session", json={"code": invite["invite_code"]})
    assert result.status_code == 200
    assert "httponly" in result.headers["set-cookie"].lower()
    assert "samesite=strict" in result.headers["set-cookie"].lower()
    return invite


@pytest.mark.asyncio
async def test_anonymous_cannot_read_data_or_spend_and_logout_invalidates_session(app):
    async with client(app) as http:
        assert "邀请试用" in (await http.get("/")).text
        for path in ("/api/status", "/api/cases", "/api/runs", "/api/runs/known/evidence/E1/file", "/assets/cases.json"):
            assert (await http.get(path)).status_code == 401
        assert (await http.post("/api/runs", json={"task": "x"})).status_code == 401
        assert (await http.post("/api/uploads?name=x.png", content=b"x")).status_code == 401
        await sign_in(http, app, "Alice")
        old = http.cookies.get(COOKIE_NAME)
        assert (await http.get("/api/runs")).status_code == 200
        assert "调查脑图" in (await http.get("/")).text
        assert (await http.delete("/api/session")).status_code == 200
        http.cookies.set(COOKIE_NAME, old)
        assert (await http.get("/api/runs")).status_code == 401


@pytest.mark.asyncio
async def test_two_users_cannot_read_stop_or_reuse_each_others_materials(app, monkeypatch):
    entered = asyncio.Event()
    async def run(**kwargs):
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(web_demo, "run_demo", run)
    async with client(app) as alice, client(app) as bob:
        await sign_in(alice, app, "Alice")
        await sign_in(bob, app, "Bob")
        upload = (await alice.post("/api/uploads?name=private.png", content=b"fixture")).json()
        response = await bob.post("/api/runs", json={"task": "Read", "upload_ids": [upload["id"]]})
        assert response.status_code == 400
        job = (await alice.post("/api/runs", json={"case_id": "sample", "upload_ids": [upload["id"]]})).json()
        await entered.wait()
        assert len((await alice.get("/api/runs")).json()) == 1
        assert (await bob.get("/api/runs")).json() == []
        case = (await bob.get("/api/cases")).json()[0]
        assert case["record_id"] is None and case["latest_run_id"] is None
        assert (await bob.get("/api/status")).json()["active_run_id"] is None
        for suffix in ("", "/events/1", "/evidence/E1", "/evidence/E1/file"):
            assert (await bob.get(f"/api/runs/{job['id']}{suffix}")).status_code == 404
        assert (await bob.post(f"/api/runs/{job['id']}/stop")).status_code == 404
        assert (await bob.post(f"/api/runs/{job['id']}/resume")).status_code == 404
        assert not app.state.website.tasks[job["id"]].done()
        await alice.post(f"/api/runs/{job['id']}/stop")
        await app.state.website.tasks[job["id"]]


@pytest.mark.asyncio
async def test_resume_uses_owned_saved_progress_and_prevents_simultaneous_attempts(app, monkeypatch):
    from foren_searcher.store import GlobalStore
    calls = []
    continuing, finish = asyncio.Event(), asyncio.Event()
    async def run(**kwargs):
        calls.append(kwargs)
        directory = kwargs["output_dir"]
        if not kwargs["resume"]:
            store = GlobalStore(directory / "investigation.sqlite3", directory / "artifacts")
            store.ensure_investigation(task=kwargs["task"], original_material_ids=())
            write_json(directory / "result.json", {"task": kwargs["task"], "decision": None,
                       "error": {"message": "GIA timeout"}})
            raise web_demo.DemoRunError(directory / "result.json")
        continuing.set()
        await finish.wait()
        write_json(directory / "result.json", {"task": kwargs["task"], "decision": {"answer": "18 (E1)"}})
    monkeypatch.setattr(web_demo, "run_demo", run)
    async with client(app) as alice, client(app) as bob:
        await sign_in(alice, app, "Alice")
        await sign_in(bob, app, "Bob")
        upload = (await alice.post("/api/uploads?name=private.png", content=b"fixture")).json()
        job = (await alice.post("/api/runs", json={"task": "Count", "upload_ids": [upload["id"]]})).json()
        website = app.state.website
        await website.tasks[job["id"]]
        route = f"/api/runs/{job['id']}"
        assert (await alice.get(route)).json()["can_resume"] is True
        assert (await bob.post(route + "/resume")).status_code == 404
        response = await alice.post(route + "/resume")
        assert response.status_code == 202 and response.json()["id"] == job["id"]
        await continuing.wait()
        snapshot = (await alice.get(route)).json()
        assert snapshot["status"] == "running" and snapshot["error"] is None and not snapshot["can_resume"]
        assert (await alice.post(route + "/resume")).status_code == 409
        assert (await bob.post("/api/runs", json={"task": "Concurrent"})).status_code == 409
        assert calls[1]["output_dir"] == calls[0]["output_dir"]
        assert calls[0]["media_paths"] and calls[1]["media_paths"] == ()
        assert calls[1]["task"] == calls[0]["task"]
        finish.set()
        await website.tasks[job["id"]]
        assert (await alice.get(route)).json()["status"] == "completed"
        assert (await alice.post(route + "/resume")).status_code == 409
        website.cases["sample"]["record"] = calls[0]["output_dir"].relative_to(website.project).as_posix()
        assert (await bob.post("/api/runs/example-sample/resume")).status_code == 409
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_failed_resume_counts_toward_quota_after_restart_including_legacy_jobs(app, monkeypatch):
    from foren_searcher.store import GlobalStore
    async def fail(**kwargs):
        directory = kwargs["output_dir"]
        store = GlobalStore(directory / "investigation.sqlite3", directory / "artifacts")
        store.ensure_investigation(task=kwargs["task"], original_material_ids=())
        write_json(directory / "result.json", {"task": kwargs["task"], "decision": None})
        raise web_demo.DemoRunError(directory / "result.json")
    monkeypatch.setattr(web_demo, "run_demo", fail)
    async with client(app) as alice:
        invitation = await sign_in(alice, app, "Alice")
        job = (await alice.post("/api/runs", json={"task": "Count"})).json()
        website = app.state.website
        await website.tasks[job["id"]]
        website.jobs[job["id"]].pop("attempt_starts")
        route = f"/api/runs/{job['id']}/resume"
        assert (await alice.post(route)).status_code == 202
        await website.tasks[job["id"]]
        assert len(website.jobs[job["id"]]["attempt_starts"]) == 2
        assert (await alice.post(route)).status_code == 429
        restarted = web_demo.create_app(project=website.project, harness_root=website.harness_root,
            catalog=website.catalog_path, trial=True, daily_runs=2)
        async with client(restarted) as again:
            await again.post("/api/session", json={"code": invitation["invite_code"]})
            assert (await again.post(route)).status_code == 429
            assert (await again.post("/api/runs", json={"task": "New"})).status_code == 429


@pytest.mark.asyncio
async def test_failure_before_investigation_initialization_cannot_resume(app, monkeypatch):
    from foren_searcher.store import GlobalStore
    async def fail(**kwargs):
        directory = kwargs["output_dir"]
        GlobalStore(directory / "investigation.sqlite3", directory / "artifacts")
        write_json(directory / "result.json", {"task": kwargs["task"], "decision": None})
        raise web_demo.DemoRunError(directory / "result.json")
    monkeypatch.setattr(web_demo, "run_demo", fail)
    async with client(app) as http:
        await sign_in(http, app, "Alice")
        job = (await http.post("/api/runs", json={"task": "Count"})).json()
        await app.state.website.tasks[job["id"]]
        route = f"/api/runs/{job['id']}"
        assert not (await http.get(route)).json()["can_resume"]
        assert (await http.post(route + "/resume")).status_code == 409


@pytest.mark.asyncio
async def test_quota_survives_restart_and_is_separate_for_each_user(app, monkeypatch):
    async def run(**kwargs):
        pass
    monkeypatch.setattr(web_demo, "run_demo", run)
    async with client(app) as alice, client(app) as bob:
        invitation = await sign_in(alice, app, "Alice")
        await sign_in(bob, app, "Bob")
        for _ in range(2):
            response = await alice.post("/api/runs", json={"task": "x"})
            assert response.status_code == 201
            await app.state.website.tasks[response.json()["id"]]
        assert (await alice.post("/api/runs", json={"task": "x"})).status_code == 429
        website = app.state.website
        restarted = web_demo.create_app(project=website.project, harness_root=website.harness_root, catalog=website.catalog_path, trial=True, daily_runs=2)
        async with client(restarted) as again:
            assert (await again.post("/api/session", json={"code": invitation["invite_code"]})).status_code == 200
            assert (await again.post("/api/runs", json={"task": "x"})).status_code == 429
        response = await bob.post("/api/runs", json={"task": "x"})
        assert response.status_code == 201
        await app.state.website.tasks[response.json()["id"]]


@pytest.mark.asyncio
async def test_trial_upload_space_is_per_user_and_partial_files_are_removed(app, monkeypatch):
    monkeypatch.setattr(web_demo, "MAX_UPLOAD_BYTES", 4)
    async with client(app) as alice, client(app) as bob:
        await sign_in(alice, app, "Alice")
        await sign_in(bob, app, "Bob")
        for _ in range(8):
            assert (await alice.post("/api/uploads?name=x.png", content=b"1234")).status_code == 201
        assert (await alice.post("/api/uploads?name=x.png", content=b"1")).status_code == 413
        assert (await bob.post("/api/uploads?name=x.png", content=b"1")).status_code == 201
        assert len(list((app.state.website.root / "uploads").glob("*/x.png"))) == 9


@pytest.mark.asyncio
async def test_public_mode_requires_trial_https_exact_host_and_secure_cookie(app):
    website = app.state.website
    for options in ({"public_origin": "https://trial.example"}, {"trial": True, "public_origin": "http://trial.example"},
                    {"trial": True, "public_origin": "https://*.example"}):
        with pytest.raises(ValueError):
            web_demo.create_app(project=website.project, **options)
    public = web_demo.create_app(project=website.project, trial=True, public_origin="https://trial.example")
    async with client(public, "https://trial.example") as http:
        invite = public.state.access.invite("Alice")
        wrong = await http.post("/api/session", json={"code": invite["invite_code"]}, headers={"Origin": "https://evil.example"})
        assert wrong.status_code == 403
        response = await http.post("/api/session", json={"code": invite["invite_code"]}, headers={"Origin": "https://trial.example"})
        assert response.status_code == 200
        assert "; secure" in response.headers["set-cookie"].lower()
        assert "max-age" in response.headers["strict-transport-security"]
        assert (await http.get("/api/runs", headers={"Host": "evil.example"})).status_code == 400


def test_expiration_revocation_and_credentials_not_stored_in_plaintext(tmp_path, monkeypatch):
    access = TrialAccess(tmp_path / "access.sqlite3")
    invite = access.invite("Alice")
    token, user = access.login(invite["invite_code"], "test")
    assert access.identity(token) == user
    raw = access.path.read_bytes()
    assert token.encode() not in raw and invite["invite_code"].encode() not in raw
    assert access.revoke(invite["id"])
    assert access.identity(token) is None
    assert access.login(invite["invite_code"], "test") is None
    other = access.invite("Bob", days=1)
    other_token, _ = access.login(other["invite_code"], "test")
    monkeypatch.setattr(time, "time", lambda: other["expires_at"] + 1)
    assert access.identity(other_token) is None
    assert access.login(other["invite_code"], "test") is None


def test_failed_login_is_rate_limited_without_creating_users_or_sessions(tmp_path):
    access = TrialAccess(tmp_path / "access.sqlite3")
    for _ in range(10):
        assert access.login("invalid", "client") is None
    with pytest.raises(PermissionError):
        access.login("invalid", "client")
    with access.connect() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 0
