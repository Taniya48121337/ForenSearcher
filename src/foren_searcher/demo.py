"""ForenSearcher 论文能力 Demo 的命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from .inspect.factory import create_inspect_service
from .inspect.contracts import inspect_action_schema
from .harness1 import harness1_workspace_type
from .hi.workspace import WorkspaceHarness
from .local_models import TransformersJsonGenerator, create_demo_policies
from .store.records import MemoryPolicy
from .gia import InitialMedia, InvestigationRuntime
from .config import load_config
from .openrouter_settings import DEFAULT_CONFIG, DEFAULT_MODEL, key_configured, read_settings
from .search.factory import (
    create_free_search_service,
    create_searxng_search_service,
    create_serpapi_search_service,
)
from .store import GlobalStore


SearchChoice = Literal["auto", "ddgs", "serpapi", "searxng"]


class DemoConfigurationError(RuntimeError):
    """Demo 所需的本地模型、搜索服务或输入配置不完整。"""


class DemoRunError(RuntimeError):
    """运行未完成，调查记录和失败结果已保存。"""

    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path
        super().__init__(f"Demo did not finish; details saved to {result_path}")


@dataclass(frozen=True, slots=True)
class DemoPaths:
    """一次 Demo 调查的持久化位置。"""

    root: Path
    database: Path
    artifacts: Path
    result: Path
    report: Path
    trajectory: Path


_INSPECT_MODEL_DIRECTORIES = {
    "FOREN_WHISPER_MODEL": "faster-whisper-small",
    "FOREN_CLIP_MODEL": "clip-vit-base-patch32",
    "FOREN_GROUNDING_MODEL": "grounding-dino-tiny",
    "FOREN_VLM_MODEL": "Qwen3-VL-2B-Instruct",
    "FOREN_AUDIO_MODEL": "Qwen2.5-Omni-3B",
}


def _demo_memory_policy(context_length: int = 0, *, state_renderer: str | None = None) -> MemoryPolicy:
    if state_renderer is None:
        state_renderer = load_config().runtime.state_renderer
    return MemoryPolicy(
        state_renderer=state_renderer,
        max_model_view_chars=80_000 if context_length >= 131_072 else 40_000,
        max_history_entries=16,
        max_read_chars=12_000 if context_length >= 131_072 else 6_000,
        max_observation_chars=20_000 if context_length >= 131_072 else 10_000,
    )


def _configure_local_inspect_models(model_root: str | Path) -> None:
    """自动使用下载器约定目录，同时尊重部署者显式设置的环境变量。"""

    root = Path(model_root).resolve() / "inspect"
    for variable, directory in _INSPECT_MODEL_DIRECTORIES.items():
        candidate = root / directory
        if not os.environ.get(variable) and candidate.is_dir():
            os.environ[variable] = str(candidate)


def _demo_paths(output_dir: str | Path | None, *, resume: bool = False) -> DemoPaths:
    """生成不覆盖旧调查的默认目录，或使用调用方明确给出的空目录。"""

    root = (
        Path(output_dir)
        if output_dir is not None
        else Path("outputs")
        / "demo"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    ).resolve()
    if resume and (output_dir is None or not (root / "investigation.sqlite3").is_file()
                   or not (root / "result.json").is_file()):
        raise DemoConfigurationError("resume requires an existing investigation and saved result")
    if not resume and root.exists() and (not root.is_dir() or any(p.name != ".run.lock" for p in root.iterdir())):
        raise DemoConfigurationError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return DemoPaths(
        root=root,
        database=root / "investigation.sqlite3",
        artifacts=root / "artifacts",
        result=root / "result.json",
        report=root / "report.md",
        trajectory=root / "trajectory.jsonl",
    )


@contextmanager
def _demo_run_lock(root: Path) -> Iterator[None]:
    """同一调查只由一个进程推进；操作系统在进程退出时释放锁。"""
    with (root / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DemoConfigurationError("this investigation is already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _search_configuration(choice: SearchChoice) -> tuple[str, str]:
    """解析搜索后端，返回类型和不含密钥的配置值。"""

    serpapi_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    searxng_url = os.environ.get("SEARXNG_BASE_URL", "").strip()
    selected = choice
    if selected == "auto":
        selected = "searxng" if searxng_url else "ddgs"
    if selected == "ddgs":
        return "ddgs", ""
    if selected == "serpapi":
        if not serpapi_key:
            raise DemoConfigurationError("SERPAPI_API_KEY is required for SerpAPI")
        return "serpapi", serpapi_key
    if selected == "searxng":
        if not searxng_url:
            raise DemoConfigurationError("SEARXNG_BASE_URL is required for SearXNG")
        return "searxng", searxng_url
    raise DemoConfigurationError(
        "configure SERPAPI_API_KEY or SEARXNG_BASE_URL before running the paper demo"
    )


def _create_search(store: GlobalStore, choice: SearchChoice, dynamic_pages: bool) -> Any:
    """按部署配置创建 SEARCH，不把密钥写入 Store 或输出。"""

    kind, value = _search_configuration(choice)
    if kind == "serpapi":
        return create_serpapi_search_service(
            store=store,
            api_key=value,
            enable_dynamic_pages=dynamic_pages,
        )
    if kind == "searxng" and os.environ.get("FOREN_REVERSE_IMAGE", "yandex") == "none":
        return create_searxng_search_service(
            store=store,
            base_url=value,
            enable_dynamic_pages=dynamic_pages,
        )
    reverse = os.environ.get("FOREN_REVERSE_IMAGE", "yandex")
    if reverse not in {"yandex", "iqdb-artwork", "iqdb-3d", "none"}:
        raise DemoConfigurationError(
            "FOREN_REVERSE_IMAGE must be yandex, iqdb-artwork, iqdb-3d, or none"
        )
    return create_free_search_service(
        store=store,
        searxng_base_url=value or None,
        reverse_engine=reverse,  # type: ignore[arg-type]
        enable_dynamic_pages=dynamic_pages,
    )


def _media_description(path_value: str | Path, index: int) -> InitialMedia:
    """把受支持的本地媒体文件转换为 Runtime 的初始 Evidence 输入。"""

    try:
        path = Path(path_value).resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise DemoConfigurationError(f"media must be a non-empty file: {path}")
    except OSError as exc:
        raise DemoConfigurationError(f"media cannot be opened: {path_value}") from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    suffix = path.suffix.lower()
    if media_type.startswith("image/"):
        modalities = ("image",)
    elif media_type.startswith("video/") or suffix in {".mkv", ".webm", ".avi", ".mov"}:
        media_type = media_type if media_type.startswith("video/") else "video/x-matroska"
        modalities = ("video",)
    elif media_type.startswith("audio/") or suffix in {".wav", ".flac", ".m4a"}:
        media_type = media_type if media_type.startswith("audio/") else "audio/wav"
        modalities = ("audio",)
    elif media_type == "application/pdf" or suffix == ".pdf":
        media_type = "application/pdf"
        modalities = ("text",)
    else:
        raise DemoConfigurationError(f"unsupported media type for {path.name}: {media_type}")
    return InitialMedia(
        source_ref=f"I{index}",
        media_type=media_type,
        modalities=modalities,
        label=path.name,
        original_filename=path.name,
        path=path,
    )


def _model_files_ready(directory: Path) -> bool:
    """识别完整权重文件集合，避免把下载中的目录当作可加载模型。"""

    if not all((directory / name).is_file() for name in ("config.json", "tokenizer.json", "tokenizer_config.json")):
        return False
    index = directory / "model.safetensors.index.json"
    try:
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        return bool(names) and all(isinstance(name, str) and Path(name).name == name
            and (directory / name).is_file() and (directory / name).stat().st_size > 0 for name in names)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def capability_report(
    *, model_root: str | Path = ".models", search: SearchChoice = "auto",
    harness_root: str | Path | None = None,
    model_backend: Literal["local", "openrouter"] = "openrouter",
    openrouter_config: str | Path = DEFAULT_CONFIG,
    student_model: str | None = None,
) -> dict[str, Any]:
    """返回不含密钥的可运行能力报告。``student_model`` 缺省取 configs/*.toml 里的 [models].student。"""

    root = Path(model_root).resolve()
    student_model = student_model or load_config().models.student
    _configure_local_inspect_models(root)
    search_ready = True
    search_provider: str | None = None
    search_error: str | None = None
    try:
        search_provider, _secret = _search_configuration(search)
    except DemoConfigurationError as exc:
        search_ready = False
        search_error = str(exc)
    inspect_models = {
        "asr": os.environ.get("FOREN_WHISPER_MODEL"),
        "frame_semantics": os.environ.get("FOREN_CLIP_MODEL"),
        "region_grounding": os.environ.get("FOREN_GROUNDING_MODEL"),
        "visual_qa": os.environ.get("FOREN_VLM_MODEL"),
        "audio_qa": os.environ.get("FOREN_AUDIO_MODEL"),
    }
    model_directory = root / student_model
    model_ready = _model_files_ready(model_directory)
    if model_backend == "openrouter":
        model_ready = key_configured(openrouter_config)
        settings = read_settings(openrouter_config)
        inspect_models.pop("visual_qa")
        inspect_models.pop("audio_qa")
    # Harness-1 已于 2026-09-15 从运行时剥离：不再是运行前提，只在显式指定时作为可选的兼容适配层。
    runtime_root = harness_root or os.environ.get("FOREN_HARNESS_ROOT")
    harness_ready = bool(runtime_root and (Path(runtime_root) / "harness/runtime/tool_executor.py").is_file())
    text_ready = model_ready and search_ready
    semantic_ready = all(bool(value) for value in inspect_models.values())
    return {
        "configured_for_text_demo": text_ready,
        "configured_for_paper_demo": text_ready and semantic_ready,
        "validation": "configuration_only",
        "model_backend": model_backend,
        "harness1": {"configured": harness_ready, "required": False, "deprecated": True,
                     "source_root": str(runtime_root) if runtime_root else None},
        "models": {
            "gia": model_ready,
            "hi": model_ready,
            "student": student_model,
            **({"model": settings.get("model", DEFAULT_MODEL), "hi_direct_images": True} if model_backend == "openrouter" else {}),
        },
        "search": {
            "ready": search_ready,
            "provider": search_provider,
            "error": search_error,
            "validation": "configuration_only",
        },
        "inspect_semantic_models": {
            name: {"configured": bool(value), "model": value}
            for name, value in inspect_models.items()
        },
    }


def _evidence_excerpt(content: Any) -> str:
    """从结构化分析中提取可读原文，保持事实内容不变。"""

    text = str(content)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        if isinstance(value.get("answer"), str):
            text = value["answer"]
        elif isinstance(value.get("results"), list):
            text = " ".join(item["text"] for page in value["results"] if isinstance(page, dict)
                for item in page.get("items", []) if isinstance(item, dict) and isinstance(item.get("text"), str)) or text
        elif isinstance(value.get("segments"), list):
            text = " ".join(segment["text"] for segment in value["segments"]
                if isinstance(segment, dict) and isinstance(segment.get("text"), str)) or text
    text = " ".join(text.split())
    return text[:600] + ("…" if len(text) > 600 else "")


def _markdown_report(task: str, result: dict[str, Any]) -> str:
    """把最终决定和引用来源导出为可直接展示的 Markdown。"""

    lines = ["# ForenSearcher 调查结果", "", "## 任务", "", task, "", "## 结论", ""]
    if result.get("status") == "failed":
        lines.extend(["调查未完成，尚未形成最终结论。", "", f"运行错误：{result['error']['type']}",
                      "", f"调查轨迹：`{result['audit']['trajectory']}`", ""])
        return "\n".join(lines)
    decision = result["decision"]
    lines.extend([str(decision["answer"]), "", "## 引用 Evidence", ""])
    evidence = result.get("cited_evidence", [])
    if not evidence:
        lines.append("本次结论没有可引用的 Evidence。")
    for item in evidence:
        source = item.get("source_ref") or "派生证据"
        locator = item.get("locator") or "未提供位置"
        label = item["label"]
        source_url = item.get("source_url")
        heading = (
            f"- {item['evidence_id']} — [{label}]({source_url})"
            if source_url
            else f"- {item['evidence_id']} — {label}"
        )
        excerpt = _evidence_excerpt(item.get("content", ""))
        artifact_path = item.get("artifact_path")
        lines.extend(
            [
                heading,
                f"  - 来源：{source}",
                f"  - 位置：{locator}",
                f"  - 取得方式：{item['acquired_by']}",
                *([f"  - [查看材料](<{artifact_path}>)"] if artifact_path else []),
                *[f"  - [{parent['evidence_id']}：{parent['label']}](<{parent['path']}>)"
                  for parent in item.get("supporting_artifacts", [])],
                *([f"  - 摘要：{excerpt}"] if excerpt else []),
            ]
        )
    reports = result.get("workspace_reports", [])
    if reports:
        lines.extend(["", "## Workspace 报告", ""])
        for report in reports:
            lines.append(f"- {report['task_id']}：{report['status']}")
            if report.get("status_reason"):
                lines.append(f"  - 原因：{report['status_reason']}")
            for finding in report.get("findings", []):
                citations = ", ".join(finding.get("evidence_ids", []))
                identifier = str(finding.get("finding_id", ""))
                lines.append(
                    "  - " + (f"{identifier}：" if identifier else "") + finding["statement"]
                    + (f"（{citations}）" if citations else "")
                )
                for limitation in finding.get("limitations", []):
                    lines.append(f"    - 限定：{limitation}")
    audit = result.get("audit")
    if isinstance(audit, dict):
        lines.extend(
            [
                "",
                "## 审计产物",
                "",
                f"- Evidence：{audit.get('evidence_count', 0)} 条",
                f"- Candidate Source：{audit.get('candidate_count', 0)} 条",
                f"- 轨迹：{audit.get('trajectory', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


async def run_demo(
    *,
    task: str,
    claims: tuple[str, ...] = (),
    media_paths: tuple[str, ...] = (),
    model_root: str | Path = ".models",
    output_dir: str | Path | None = None,
    search: SearchChoice = "auto",
    dynamic_pages: bool = False,
    max_rounds: int = 4,
    strict_semantics: bool = True,
    harness_root: str | Path | None = None,
    model_backend: Literal["local", "openrouter"] = "openrouter",
    openrouter_config: str | Path = DEFAULT_CONFIG,
    resume: bool = False,
) -> dict[str, Any]:
    """运行一次隔离调查，保存 Store、Artifact、JSON 和 Markdown 结果。"""

    if not isinstance(task, str) or not task.strip():
        raise DemoConfigurationError("task must be non-empty text")
    if resume and (claims or media_paths):
        raise DemoConfigurationError("resume reuses the stored materials; do not supply new claims or media")
    paths = _demo_paths(output_dir, resume=resume)
    with _demo_run_lock(paths.root):
        _demo_paths(paths.root, resume=resume)
        return await _run_demo(task=task, claims=claims, media_paths=media_paths, model_root=model_root,
            paths=paths, search=search, dynamic_pages=dynamic_pages, max_rounds=max_rounds,
            strict_semantics=strict_semantics, harness_root=harness_root,
            model_backend=model_backend, openrouter_config=openrouter_config, resume=resume)


async def _run_demo(*, task: str, claims: tuple[str, ...], media_paths: tuple[str, ...],
                    model_root: str | Path, paths: DemoPaths, search: SearchChoice,
                    dynamic_pages: bool, max_rounds: int, strict_semantics: bool,
                    harness_root: str | Path | None, model_backend: Literal["local", "openrouter"],
                    openrouter_config: str | Path, resume: bool) -> dict[str, Any]:
    store: GlobalStore
    previous: dict[str, Any] = {}
    originals: tuple[str, ...] = ()
    if resume:
        store = GlobalStore(paths.database, paths.artifacts)
        try:
            metadata = store.investigation_meta()
            previous = json.loads(paths.result.read_text(encoding="utf-8"))
        except (KeyError, ValueError) as exc:
            raise DemoConfigurationError("stored investigation cannot be resumed") from exc
        if not isinstance(previous, dict):
            raise DemoConfigurationError("stored result must be an object")
        if metadata["task"] != task.strip():
            raise DemoConfigurationError("resume task does not match the stored investigation")
        originals = tuple(metadata["original_material_ids"])
        try:
            final = store.final_result()
        except KeyError:
            final = None
        if final is not None and previous.get("decision") == final:
            return previous
    if not isinstance(task, str) or not task.strip():
        raise DemoConfigurationError("task must be non-empty text")
    report = capability_report(model_root=model_root, search=search, harness_root=harness_root,
                               model_backend=model_backend, openrouter_config=openrouter_config)
    if not report["configured_for_text_demo"]:
        raise DemoConfigurationError(json.dumps(report, ensure_ascii=False))
    media = tuple(
        _media_description(value, index) for index, value in enumerate(media_paths)
    )
    if strict_semantics and media:
        configured = report["inspect_semantic_models"]
        required: set[str] = set()
        if model_backend == "openrouter":
            if any("audio" in item.modalities for item in media):
                required.add("asr")
        elif any("image" in item.modalities for item in media):
            required.update({"visual_qa", "region_grounding"})
        if model_backend != "openrouter" and any(item.media_type == "application/pdf" for item in media):
            required.add("visual_qa")
        if model_backend != "openrouter" and any("video" in item.modalities for item in media):
            required.update({"frame_semantics", "visual_qa", "asr", "audio_qa"})
        if model_backend != "openrouter" and any("audio" in item.modalities for item in media):
            required.update({"asr", "audio_qa"})
        missing = sorted(
            name for name in required if not configured[name]["configured"]
        )
        if missing:
            raise DemoConfigurationError(
                "semantic INSPECT models are required for this media input: "
                + ", ".join(missing)
            )
    # 默认直接用本项目的 WorkspaceHarness；只有显式给了 Harness-1 目录才走已弃用的兼容适配层。
    runtime_root = harness_root or os.environ.get("FOREN_HARNESS_ROOT")
    workspace_type: type[WorkspaceHarness] = harness1_workspace_type(runtime_root) if runtime_root else WorkspaceHarness
    if not resume:
        store = GlobalStore(paths.database, paths.artifacts)
    search_service = None
    inspect_service = None
    policies: Any = None
    remote_client: Any = None
    failure: dict[str, Any] | None = None
    def remote_usage() -> dict[str, Any]:
        return remote_client.usage_report()
    def record_generation(value: dict[str, Any]) -> None:
        workspace_id = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace_id else "GIA",
            actor_id=workspace_id or "GIA", event_type="raw_model_output", payload=value)
    claim_materials = {
        f"C{index}": {"claim": value}
        for index, value in enumerate(claims)
    }
    if resume:
        store.record_event(actor_role="HARNESS", actor_id="HARNESS", event_type="investigation_resume",
            payload={"previous_error": previous.get("error"), "previous_usage": previous.get("model_usage")})
    try:
        search_service = _create_search(store, search, dynamic_pages)
        if model_backend == "openrouter":
            from .openrouter import OpenRouterClient, OpenRouterPolicyBundle
            remote_client = OpenRouterClient(openrouter_config)
            if resume and previous.get("model_usage"):
                try:
                    remote_client.restore_usage(previous["model_usage"])
                except ValueError as exc:
                    raise DemoConfigurationError(str(exc)) from exc
            operations = frozenset(branch["properties"]["request"]["properties"]["operation"]["const"]
                for branch in inspect_action_schema()["oneOf"]) - {"visual_qa", "audio_qa"}
            inspect_service = create_inspect_service(store=store, enabled_operations=operations)
            policies = OpenRouterPolicyBundle(remote_client, store=store, image_policy=inspect_service.policy)
        else:
            inspect_service = create_inspect_service(store=store)
            policies = create_demo_policies(model_root, model_name=load_config().models.student, store=store, image_policy=inspect_service.policy)
        policies.gia_model.generation_observer = record_generation
        policies.hi_model.generation_observer = record_generation
        await policies.preload()
        if (model_backend == "local" and isinstance(policies.gia_model, TransformersJsonGenerator)
                and isinstance(policies.hi_model, TransformersJsonGenerator)
                and (policies.gia_model.uses_cuda or policies.hi_model.uses_cuda)):
            # 调查尚未开始，统一模型线程锁；CPU 媒体操作仍可独立执行。
            policies.hi_model._lock = policies.gia_model._lock
            inspect_service.dispatcher.coordinate_local_models(policies.gia_model._lock)
        if model_backend == "local" and getattr(policies.hi_model, "supports_image_input", False):
            local_operations = inspect_service.enabled_operations
            if local_operations is None:
                local_operations = frozenset(branch["properties"]["request"]["properties"]["operation"]["const"]
                    for branch in inspect_action_schema()["oneOf"])
            inspect_service.enabled_operations = local_operations - {"visual_qa"}
        async with InvestigationRuntime(
            store=store,
            gia_policy=policies.gia_policy,
            hi_policy=policies.hi_policy,
            search_service=search_service,
            inspect_service=inspect_service,
            max_rounds=max_rounds,
            max_concurrency=4,
            gia_turn_timeout_s=300,
            workspace_harness_type=workspace_type,
            memory_policy=_demo_memory_policy(
                (remote_client.metadata.get("context_length") or 0) if remote_client is not None else 0
            ),
        ) as runtime:
            if resume:
                investigation = await runtime.run(task=task.strip(), original_material_ids=originals)
            else:
                investigation = await runtime.run_new(task=task.strip(), claim_materials=claim_materials, media=media)
    except DemoConfigurationError:
        raise
    except BaseException as exc:
        store.record_event(actor_role="HARNESS", actor_id="HARNESS", event_type="investigation_failed",
            payload={"error": {"type": type(exc).__name__, "message": str(exc)[:2000]}})
        store.export_trajectory(paths.trajectory)
        failure = {"task":task.strip(), "status":"failed", "decision":None,
            "error":{"type":type(exc).__name__, "message":str(exc)[:2000]},
            "output_dir":str(paths.root),
            "workspace_reports":[item.to_dict() for item in store.workspace_report_records()],
            "audit":{"trajectory":str(paths.trajectory)}}
        if remote_client is not None:
            failure["model_usage"] = remote_usage()
        elif previous.get("model_usage"):
            failure["model_usage"] = previous["model_usage"]
        paths.result.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.report.write_text(_markdown_report(task.strip(), failure), encoding="utf-8")
        if isinstance(exc, Exception):
            raise DemoRunError(paths.result) from exc
        raise
    finally:
        try:
            if inspect_service is not None and not getattr(inspect_service, "_closed", False):
                await inspect_service.aclose()
        finally:
            try:
                if policies is not None:
                    await policies.aclose()
                elif remote_client is not None:
                    await remote_client.aclose()
            finally:
                if search_service is not None and not getattr(search_service, "_closed", False):
                    await search_service.aclose()
                if failure is not None and remote_client is not None:
                    failure["model_usage"] = remote_usage()
                    paths.result.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
    investigation_value = investigation.to_dict()
    candidates = list(store.candidate_records())
    candidate_by_id = {
        item["candidate_id"]: item for item in candidates
    }
    cited_evidence = []
    for item in investigation_value["cited_evidence"]:
        enriched = dict(item)
        if enriched.get("artifact_ref") is not None:
            enriched["artifact_path"] = store.artifact_path(enriched["artifact_ref"]).as_posix()
        supporting = []
        for parent_id in enriched.get("derived_from", []):
            parent = store.evidence(parent_id)
            if parent.artifact_ref is not None:
                supporting.append({"evidence_id":parent_id, "label":parent.label,
                    "path":store.artifact_path(parent.artifact_ref).as_posix()})
        if supporting:
            enriched["supporting_artifacts"] = supporting
        source_ref = enriched.get("source_ref")
        candidate = candidate_by_id.get(source_ref)
        if candidate is not None:
            enriched["source_url"] = candidate.get("final_url") or candidate.get(
                "source_url"
            )
        cited_evidence.append(enriched)
    store.export_trajectory(paths.trajectory)
    reports = [item.to_dict() for item in store.workspace_report_records()]
    evidence_cards = [item.card().to_dict() for item in store.evidence_records()]
    value = {
        **investigation_value,
        "cited_evidence": cited_evidence,
        "task": task.strip(),
        "runtime": ({"base": "Harness-1 (deprecated adapter)", "source_root": str(Path(runtime_root).resolve())}
                    if runtime_root else {"base": "WorkspaceHarness", "source_root": None}),
        "output_dir": str(paths.root),
        "workspace_reports": reports,
        "candidate_sources": candidates,
        "evidence_index": evidence_cards,
        "audit": {
            "trajectory": str(paths.trajectory),
            "evidence_count": len(evidence_cards),
            "candidate_count": len(candidates),
            "workspace_report_count": len(reports),
        },
        "information_boundaries": {
            "shared_between_workspaces": [
                "committed Evidence cards, IDs and full records through READ_RECORD",
                "Candidate provenance linked to committed Evidence",
            ],
            "workspace_private": [
                "working-memory history and focus",
                "unopened Candidate Sources discovered by that Workspace",
                "own Findings and Report, also visible to GIA",
            ],
            "gia_global_only": [
                "combined access to every Workspace Report and Finding",
            ],
            "private_audit_artifact": str(paths.trajectory),
        },
        "store_health": store.healthcheck().to_dict(),
        "capabilities": report,
    }
    if remote_client is not None:
        value["model_usage"] = remote_usage()
    paths.result.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.report.write_text(_markdown_report(task.strip(), value), encoding="utf-8")
    return value


def _main() -> None:
    """解析 check/run 子命令并输出机器可读 JSON。"""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        # Windows 默认 GBK 无法表示任意模型/网页 Unicode；结果文件仍保持 UTF-8。
        reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description="Run the ForenSearcher paper demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check local demo capabilities")
    run = subparsers.add_parser("run", help="run one isolated investigation")
    for command in (check, run):
        command.add_argument("--model-root", default=".models")
        command.add_argument("--harness-root")
        command.add_argument("--model-backend", choices=("local", "openrouter"), default="openrouter")
        command.add_argument("--openrouter-config", default=str(DEFAULT_CONFIG))
        command.add_argument("--search", choices=("auto", "ddgs", "serpapi", "searxng"), default="auto")
    run.add_argument("--task", required=True)
    run.add_argument("--claim", action="append", default=[])
    run.add_argument("--media", action="append", default=[])
    run.add_argument("--output-dir")
    run.add_argument("--resume", action="store_true", help="continue the saved investigation without re-ingesting inputs")
    run.add_argument("--dynamic-pages", action="store_true")
    run.add_argument("--max-rounds", type=int, default=4)
    run.add_argument(
        "--allow-limited-inspect",
        action="store_true",
        help="allow honest non-semantic fallback operations when semantic models are absent",
    )
    args = parser.parse_args()
    if args.command == "check":
        value = capability_report(model_root=args.model_root, search=args.search, harness_root=args.harness_root,
                                  model_backend=args.model_backend, openrouter_config=args.openrouter_config)
    else:
        value = asyncio.run(
            run_demo(
                task=args.task,
                claims=tuple(args.claim),
                media_paths=tuple(args.media),
                model_root=args.model_root,
                output_dir=args.output_dir,
                search=args.search,
                dynamic_pages=args.dynamic_pages,
                max_rounds=args.max_rounds,
                strict_semantics=not args.allow_limited_inspect,
                harness_root=args.harness_root,
                model_backend=args.model_backend,
                openrouter_config=args.openrouter_config,
                resume=args.resume,
            )
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    """以清晰的退出码和已保存结果报告运行失败。"""

    try:
        _main()
    except DemoRunError as exc:
        print(exc.result_path.read_text(encoding="utf-8"))
        raise SystemExit(1) from None
    except DemoConfigurationError as exc:
        print(json.dumps({"status":"configuration_error", "message":str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
