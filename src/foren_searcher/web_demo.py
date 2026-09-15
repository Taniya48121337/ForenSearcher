"""本机调查网站：调用原 Demo 运行时，读取真实事件并提供证据预览。"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .demo import DemoRunError, capability_report, run_demo
from .openrouter import UsageLedger
from .openrouter_settings import DEFAULT_CONFIG, DEFAULT_MODEL, key_configured, read_settings
from .store import GlobalStore
from .web_demo_data import InvestigationReader, read_json, write_json
from .web_access import COOKIE_NAME, SESSION_SECONDS, TrialAccess
from .web_training import TrainingReader


ASSETS = Path(__file__).with_name("web")
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".pdf", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class DemoWebsite:
    def __init__(self, project: Path, harness_root: Path | None, config: Path, catalog: Path | None = None,
                 *, trial: bool = False, daily_runs: int = 3) -> None:
        self.project, self.harness_root, self.config = project.resolve(), harness_root, config.resolve()
        self.root = self.project / ("outputs/web-trial" if trial else "outputs/web-demo")
        self.trial, self.daily_runs = trial, daily_runs
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = catalog or ASSETS / "cases.json"
        self.cases = {c["id"]: c for c in read_json(self.catalog_path, [])}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.start_lock = asyncio.Lock()
        self.upload_lock = asyncio.Lock()
        for p in self.jobs_dir.glob("*/metadata.json"):
            value = read_json(p)
            if not value or not IDENTIFIER.fullmatch(p.parent.name) or value.get("id") != p.parent.name:
                continue
            if value["status"] in {"running", "stopping", "queued"}:
                answer = read_json(p.parent / "investigation/result.json", {})
                value["status"] = "completed" if (answer.get("decision") or {}).get("answer") else "interrupted"
                value["ended_at"] = value.get("ended_at") or time.time()
                write_json(p, value)
            self.jobs[value["id"]] = value

    def local_path(self, relative: str) -> Path:
        path = (self.project / relative).resolve()
        if not path.is_relative_to(self.project):
            raise HTTPException(400, "材料地址超出项目范围")
        return path

    def owns(self, job: dict[str, Any], owner_id: str) -> bool:
        return job.get("owner_id", "local") == owner_id

    def resolve_run(self, run_id: str, owner_id: str = "local") -> tuple[dict[str, Any], Path]:
        if run_id in self.jobs and self.owns(self.jobs[run_id], owner_id):
            return self.jobs[run_id].copy(), self.jobs_dir / run_id / "investigation"
        for c in self.cases.values():
            if run_id == f"example-{c['id']}" and c.get("record"):
                directory = self.local_path(c["record"])
                if (directory / "result.json").is_file():
                    return {"id": run_id, "case_id": c["id"], "title": c["title"], "task": c["task"],
                            "status": "completed", "mode": "record", "started_at": None, "ended_at": None}, directory
        raise HTTPException(404, "调查记录不存在")

    def case_list(self, owner_id: str = "local") -> list[dict[str, Any]]:
        values = []
        for c in self.cases.values():
            ready = all(self.local_path(p).is_file() for p in c.get("media", []))
            jobs = sorted((j for j in self.jobs.values() if j.get("case_id") == c["id"] and self.owns(j, owner_id)), key=lambda j: j["started_at"], reverse=True)
            completed = next((j for j in jobs if j["status"] == "completed"), None)
            record = completed["id"] if completed else (f"example-{c['id']}" if c.get("record") and (self.local_path(c["record"]) / "result.json").is_file() else None)
            values.append({k: c.get(k) for k in ("id", "title", "description", "dataset", "sample", "tags", "task", "source_url", "license", "kind")}
                          | {"available": ready, "record_id": record, "latest_run_id": jobs[0]["id"] if jobs else None,
                             "media_count": len(c.get("media", [])), "thumbnail": f"/api/cases/{c['id']}/thumbnail" if c.get("thumbnail") else None})
        return values

    def can_resume(self, job: dict[str, Any], directory: Path) -> bool:
        if job.get("mode") != "live" or job["status"] not in {"failed", "stopped", "interrupted"}:
            return False
        saved = read_json(directory / "result.json", {})
        return (isinstance(saved, dict) and saved.get("task") == job["task"] and not saved.get("decision")
                and InvestigationReader(directory).investigation_task() == job["task"])

    def check_start(self, owner_id: str) -> None:
        if any(not t.done() for t in self.tasks.values()):
            raise HTTPException(409, "服务正在处理调查，请稍后再试")
        recent = sum(start > time.time() - 86400 for j in self.jobs.values() if self.owns(j, owner_id)
                     for start in j.get("attempt_starts", [j["started_at"]]))
        if self.trial and recent >= self.daily_runs:
            raise HTTPException(429, f"每人24小时内最多开始{self.daily_runs}次调查，继续调查也计入次数，请稍后再试")
        if not key_configured(self.config):
            raise HTTPException(409, "请先在 OpenRouter 设置窗口填写密钥")

    def launch(self, job: dict[str, Any], media: tuple[str, ...], *, resume: bool = False) -> None:
        identifier = job["id"]
        write_json(self.jobs_dir / identifier / "metadata.json", job)
        self.jobs[identifier] = job
        self.tasks[identifier] = asyncio.create_task(self.execute(job, media, resume=resume))
        def persist_early_cancellation(task: asyncio.Task[Any]) -> None:
            if task.cancelled() and job["status"] in {"running", "stopping"}:
                job.update(status="stopped", ended_at=time.time())
                write_json(self.jobs_dir / identifier / "metadata.json", job)
        self.tasks[identifier].add_done_callback(persist_early_cancellation)

    async def execute(self, job: dict[str, Any], media: tuple[str, ...], *, resume: bool = False) -> None:
        path = self.jobs_dir / job["id"]
        try:
            await run_demo(task=job["task"], media_paths=media, model_root=self.project / ".models",
                output_dir=path / "investigation", harness_root=self.harness_root,
                openrouter_config=self.config, model_backend="openrouter", search="ddgs", resume=resume)
            job["status"] = "completed"
        except asyncio.CancelledError:
            job["status"] = "stopped"
        except DemoRunError:
            job["status"] = "failed"
        except Exception as exc:
            job.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)[:1000]})
        finally:
            job["ended_at"] = time.time()
            write_json(path / "metadata.json", job)

    async def start(self, body: dict[str, Any], owner_id: str = "local") -> dict[str, Any]:
        async with self.start_lock:
            self.check_start(owner_id)
            case_id = body.get("case_id")
            case = self.cases.get(case_id) if isinstance(case_id, str) else None
            if case_id is not None and case is None:
                raise HTTPException(400, "案例不存在")
            task = body.get("task", case["task"] if case else "")
            if not isinstance(task, str) or not task.strip() or len(task) > 16000:
                raise HTTPException(400, "请输入1至16000字的调查问题")
            inputs = body.get("upload_ids", [])
            if not isinstance(inputs, list) or len(inputs) > 8:
                raise HTTPException(400, "每次最多添加8份材料")
            media = [str(self.local_path(p)) for p in case.get("media", [])] if case else []
            for identifier in inputs:
                if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
                    raise HTTPException(400, "上传材料编号无效")
                info = read_json(self.root / "uploads" / identifier / "metadata.json")
                if not info or info.get("owner_id", "local") != owner_id:
                    raise HTTPException(400, "上传材料不存在")
                media.append(str(self.root / "uploads" / identifier / info["filename"]))
            if not all(Path(p).is_file() for p in media):
                raise HTTPException(400, "案例材料尚未准备好")
            identifier = uuid.uuid4().hex
            started_at = time.time()
            job = {"id": identifier, "owner_id": owner_id, "case_id": case_id, "title": case["title"] if case else task.strip()[:48],
                   "task": task.strip(), "upload_ids": inputs, "status": "running", "mode": "live",
                   "started_at": started_at, "attempt_starts": [started_at], "ended_at": None}
            self.launch(job, tuple(media))
            return job.copy()

    async def resume(self, run_id: str, owner_id: str = "local") -> dict[str, Any]:
        async with self.start_lock:
            job, directory = self.resolve_run(run_id, owner_id)
            if not self.can_resume(job, directory):
                raise HTTPException(409, "这项调查没有可继续的中断记录")
            self.check_start(owner_id)
            job["attempt_starts"] = [*job.get("attempt_starts", [job["started_at"]]), time.time()]
            job.update(status="running", ended_at=None)
            job.pop("error", None)
            self.launch(job, (), resume=True)
            return job.copy()


def create_app(*, project: str | Path = ".", harness_root: str | Path | None = None,
               config: str | Path = DEFAULT_CONFIG, catalog: Path | None = None,
               trial: bool = False, public_origin: str | None = None, daily_runs: int = 3) -> FastAPI:
    if daily_runs < 1 or daily_runs > 100:
        raise ValueError("daily_runs must be between 1 and 100")
    if public_origin:
        origin_parts = urlsplit(public_origin)
        if (not trial or origin_parts.scheme != "https" or not origin_parts.hostname
                or origin_parts.username or origin_parts.password or origin_parts.path not in {"", "/"}
                or origin_parts.query or origin_parts.fragment or "*" in origin_parts.netloc):
            raise ValueError("外网访问必须启用邀请试用并提供 HTTPS 网站地址")
        public_origin = public_origin.rstrip("/")
    project_path = Path(project).resolve()
    harness = harness_root or os.environ.get("FOREN_HARNESS_ROOT")
    website = DemoWebsite(project_path, Path(harness).resolve() if harness else None,
                          (project_path / config).resolve(), catalog, trial=trial, daily_runs=daily_runs)
    access = TrialAccess(project_path / ".local/trial-access.sqlite3") if trial else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        pending = [t for t in website.tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(title="ForenSearcher", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.website = website
    app.state.access = access
    training = TrainingReader(project_path)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(public_origin).hostname] if public_origin else ["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_browser_boundary(request: Request, call_next: Any) -> Any:
        origin = request.headers.get("origin")
        if origin and origin != (public_origin or str(request.base_url).rstrip("/")):
            return JSONResponse({"detail": "仅允许从本网站访问调查数据"}, status_code=403)
        if request.method not in {"GET", "HEAD"} and request.headers.get("x-foren-request") != "1":
            return JSONResponse({"detail": "请从网站发起操作"}, status_code=403)
        request.state.owner_id = "local"
        request.state.user = None
        if access:
            user = await asyncio.to_thread(access.identity, request.cookies.get(COOKIE_NAME, ""))
            request.state.user = user
            request.state.owner_id = user["id"] if user else None
            public_path = request.url.path in {"/", "/api/session", "/assets/login.js", "/assets/style.css"}
            if not user and not public_path:
                response = JSONResponse({"detail": "请先使用邀请码登录"}, status_code=401)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers["Cache-Control"] = "no-store"
        if public_origin:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/api/session")
    async def session(request: Request) -> Any:
        return {"trial": trial, "user": request.state.user}

    @app.post("/api/session")
    async def login(request: Request) -> Any:
        if not access:
            raise HTTPException(404)
        import json
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > 1024:
                raise HTTPException(413, "登录请求过大")
            data.extend(chunk)
        try:
            body = json.loads(data)
        except ValueError as exc:
            raise HTTPException(400, "登录请求无效") from exc
        code = body.get("code") if isinstance(body, dict) else None
        if not isinstance(code, str):
            raise HTTPException(400, "请输入邀请码")
        try:
            result = await asyncio.to_thread(access.login, code.strip(), request.client.host if request.client else "unknown")
        except PermissionError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "60"}) from exc
        if not result:
            raise HTTPException(401, "邀请码无效或已过期")
        token, user = result
        response = JSONResponse({"user": user, "trial": True})
        response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True, secure=bool(public_origin), samesite="strict")
        return response

    @app.delete("/api/session")
    async def logout(request: Request) -> Any:
        if access:
            await asyncio.to_thread(access.logout, request.cookies.get(COOKIE_NAME, ""))
        response = JSONResponse({"user": None})
        response.delete_cookie(COOKIE_NAME, secure=bool(public_origin), httponly=True, samesite="strict")
        return response

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        settings = read_settings(website.config)
        ledger = UsageLedger(website.config.parent / "openrouter-usage.sqlite3", settings["budget_usd"])
        usage = await asyncio.to_thread(ledger.summary)
        report = capability_report(model_root=project_path / ".models", harness_root=website.harness_root,
                                   model_backend="openrouter", openrouter_config=website.config, search="ddgs")
        return {"model": settings.get("model", DEFAULT_MODEL), "configured": report["configured_for_text_demo"],
                "key_configured": key_configured(website.config), "harness_configured": report["harness1"]["configured"],
                "budget": {k: usage[k] for k in ("budget_usd", "remaining_usd", "cumulative_accounted_usd")},
                "trial": trial, "daily_runs": daily_runs if trial else None,
                "active_run_id": next((i for i, t in website.tasks.items() if not t.done() and website.owns(website.jobs[i], request.state.owner_id)), None)}

    @app.get("/api/cases")
    async def cases(request: Request) -> Any:
        return website.case_list(request.state.owner_id)

    @app.get("/api/cases/{case_id}/thumbnail")
    async def thumbnail(case_id: str) -> FileResponse:
        case = website.cases.get(case_id)
        if not case or not case.get("thumbnail"):
            raise HTTPException(404)
        path = website.local_path(case["thumbnail"])
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png")

    @app.get("/api/runs")
    async def runs(request: Request) -> Any:
        return sorted((j for j in website.jobs.values() if website.owns(j, request.state.owner_id)), key=lambda j: j["started_at"], reverse=True)[:100]

    @app.post("/api/runs", status_code=201)
    async def start(request: Request) -> Any:
        import json
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > 100000:
                raise HTTPException(413, "调查请求过大")
            data.extend(chunk)
        try:
            body = json.loads(data)
        except ValueError as exc:
            raise HTTPException(400, "调查请求不是有效 JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "调查请求应为对象")
        return await website.start(body, request.state.owner_id)

    @app.post("/api/runs/{run_id}/stop")
    async def stop(run_id: str, request: Request) -> Any:
        website.resolve_run(run_id, request.state.owner_id)
        task = website.tasks.get(run_id)
        if not task or task.done():
            raise HTTPException(409, "这项调查已经结束")
        website.jobs[run_id]["status"] = "stopping"
        task.cancel()
        return {"status": "stopping"}

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    async def resume(run_id: str, request: Request) -> Any:
        return await website.resume(run_id, request.state.owner_id)

    @app.get("/api/runs/{run_id}")
    async def snapshot(run_id: str, request: Request, after: int = 0) -> Any:
        if after < 0:
            raise HTTPException(400)
        meta, directory = website.resolve_run(run_id, request.state.owner_id)
        snapshot = await asyncio.to_thread(InvestigationReader(directory).snapshot, after)
        snapshot["error"] = None if meta["status"] in {"running", "stopping"} else snapshot.get("error") or meta.get("error")
        verification = read_json(directory / "content-verification.json") or read_json(directory.parent / "review.json")
        return {**meta, **snapshot, "can_resume": website.can_resume(meta, directory),
                "verification": ({k: verification.get(k) for k in ("flow_passed", "content_verdict", "verified_points", "remaining_issues")} if verification else None)}

    @app.get("/api/runs/{run_id}/events/{event_id}")
    async def event(run_id: str, event_id: int, request: Request) -> Any:
        _, directory = website.resolve_run(run_id, request.state.owner_id)
        value = await asyncio.to_thread(InvestigationReader(directory).event, event_id)
        if value is None:
            raise HTTPException(404)
        return value

    @app.get("/api/runs/{run_id}/evidence/{evidence_id}")
    async def evidence(run_id: str, evidence_id: str, request: Request) -> Any:
        if not re.fullmatch(r"E[1-9][0-9]*", evidence_id):
            raise HTTPException(404)
        _, directory = website.resolve_run(run_id, request.state.owner_id)
        value = await asyncio.to_thread(InvestigationReader(directory).evidence, evidence_id)
        if value is None:
            raise HTTPException(404)
        return value

    @app.get("/api/runs/{run_id}/evidence/{evidence_id}/file")
    async def evidence_file(run_id: str, evidence_id: str, request: Request) -> FileResponse:
        if not re.fullmatch(r"E[1-9][0-9]*", evidence_id):
            raise HTTPException(404)
        _, directory = website.resolve_run(run_id, request.state.owner_id)
        if not (directory / "investigation.sqlite3").is_file():
            raise HTTPException(404)
        def artifact() -> Any:
            store = GlobalStore(directory / "investigation.sqlite3", directory / "artifacts")
            e = store.evidence(evidence_id)
            return store.artifact(e.artifact_ref) if e.artifact_ref else None
        try:
            material = await asyncio.to_thread(artifact)
        except KeyError as exc:
            raise HTTPException(404) from exc
        if material is None:
            raise HTTPException(404)
        safe = material.media_type in {"image/png", "image/jpeg", "image/webp", "image/gif", "application/pdf"} or material.media_type.startswith(("video/", "audio/"))
        return FileResponse(material.path, media_type=material.media_type if safe else "application/octet-stream",
            filename=f"{evidence_id}{material.path.suffix}", content_disposition_type="inline" if safe else "attachment")

    @app.post("/api/uploads", status_code=201)
    async def upload(request: Request, name: str) -> Any:
        async with website.upload_lock:
            return await save_upload(request, name)

    async def save_upload(request: Request, name: str) -> Any:
        if not name or len(name) > 180 or Path(name).name != name or any(c in name for c in '/\\:\x00') or Path(name).suffix.lower() not in MEDIA_SUFFIXES:
            raise HTTPException(400, "请选择图片、PDF、视频或音频文件")
        owner_id = request.state.owner_id
        stored_size = 0
        if trial:
            for metadata in (website.root / "uploads").glob("*/metadata.json"):
                info = read_json(metadata, {})
                if info.get("owner_id") == owner_id:
                    stored_size += info.get("size", 0)
        identifier = uuid.uuid4().hex
        directory = website.root / "uploads" / identifier
        directory.mkdir(parents=True)
        path = directory / name
        size = 0
        try:
            with path.open("xb") as target:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "每份材料不能超过64 MB")
                    if trial and stored_size + size > 8 * MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "试用材料空间已满，请联系邀请人处理")
                    target.write(chunk)
            if not size:
                raise HTTPException(400, "文件为空")
            write_json(directory / "metadata.json", {"filename": name, "size": size, "owner_id": owner_id})
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"id": identifier, "name": name, "size": size}

    @app.get("/api/training")
    async def training_overview() -> Any:
        return await asyncio.to_thread(training.overview)

    @app.get("/api/training/samples/{sample_id}")
    async def training_sample(sample_id: str) -> Any:
        return await asyncio.to_thread(training.sample, sample_id)

    @app.get("/api/training/samples/{sample_id}/steps/{step_id}")
    async def training_step(sample_id: str, step_id: str) -> Any:
        return await asyncio.to_thread(training.step, sample_id, step_id)

    @app.get("/api/training/samples/{sample_id}/steps/{step_id}/images/{image_index}")
    async def training_image(sample_id: str, step_id: str, image_index: int) -> FileResponse:
        path = await asyncio.to_thread(training.image_path, sample_id, step_id, image_index)
        return FileResponse(path, media_type="image/png")

    @app.get("/api/training/documents/{name}")
    async def training_document(name: str) -> FileResponse:
        return FileResponse(training.document_path(name), media_type="text/plain; charset=utf-8")

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")

    @app.get("/")
    async def index(request: Request) -> FileResponse:
        return FileResponse(ASSETS / ("login.html" if access and not request.state.user else "index.html"))

    return app


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="ForenSearcher 本机调查网站")
    parser.add_argument("--project", default=".")
    parser.add_argument("--harness-root", default=os.environ.get("FOREN_HARNESS_ROOT"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="在浏览器中打开本机网站")
    parser.add_argument("--trial", action="store_true", help="启用邀请码登录和用户数据隔离")
    parser.add_argument("--public-origin", help="HTTPS 反向代理的网站地址；必须同时启用 --trial")
    parser.add_argument("--daily-runs", type=int, default=3, help="每位试用者24小时内的调查次数，默认3次")
    admin = parser.add_mutually_exclusive_group()
    admin.add_argument("--invite", metavar="NAME", help="生成一位试用者的邀请码后退出")
    admin.add_argument("--revoke", metavar="USER_ID", help="撤销试用者的邀请及所有会话后退出")
    parser.add_argument("--invite-days", type=int, default=7)
    args = parser.parse_args()
    if args.invite or args.revoke:
        import json
        access = TrialAccess(Path(args.project).resolve() / ".local/trial-access.sqlite3")
        print(json.dumps(access.invite(args.invite, args.invite_days) if args.invite else {"revoked": access.revoke(args.revoke)}, ensure_ascii=False))
        return
    app = create_app(project=args.project, harness_root=args.harness_root, config=args.config,
                     trial=args.trial, public_origin=args.public_origin, daily_runs=args.daily_runs)
    if args.open:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
