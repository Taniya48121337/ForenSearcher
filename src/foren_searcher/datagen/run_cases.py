"""批量跑调查：一批案卷 → 逐个跑教师 bridge run → 生成 execution_links → 记成本 → （可选）接核验与导出。

    python -m foren_searcher.datagen.run_cases --dry-run                         # 只列计划，不跑
    python -m foren_searcher.datagen.run_cases --round r0 --endpoint http://服务器:8000/v1   # 服务器 122B 教师
    python -m foren_searcher.datagen.run_cases --ids true:4124903 --teacher codex          # 本机 Codex CLI 教师
    python -m foren_searcher.datagen.run_cases ... --then-verify --judge-endpoint http://服务器:8001/v1

输入是 experiments/sft_data/cases/*.json（案卷）：任务文字、媒体、来源策略都从案卷来，不再手写 selection 清单。
每个 (案例, 第 k 次采样) 一个 run，名字 <dataset>_<source_id>_<轮次>_<k>，放在 experiments/sft_data/runs/ 下：
- 已完成（有 result.json + execution_links.json）→ 跳过；只跑了一半 → 默认跳过并报告，--retry-failed 时把旧目录改名为
  <run>.failed-<时间> 后重跑（不删任何东西）。
- 每个 run 在独立子进程里跑（一个 run 崩了不影响后面的），日志在 experiments/sft_data/batch_logs/<run>.log。
- 跑完立刻写 execution_links.json（runlog 需要它），并把单 run 成本（墙钟、教师调用次数、输入/输出 token、附图数）
  追加到 experiments/sft_data/run_manifests/<轮次>.jsonl —— 这就是"成本实测"要的数。

教师：
- endpoint（默认）：OpenAI 兼容服务（服务器 vLLM 上的 Qwen3.5-122B）。地址 --endpoint > FOREN_TEACHER_ENDPOINT > configs 的
  models.teacher_endpoint；服务上只有一个模型时 --model 可省略。教师指令按角色取 specs/<hi|gia>@版本/teacher_instruction.md。
- codex：本机 Codex CLI（试点时期的 gpt-6-astra，走 ChatGPT 订阅），只串行。

媒体按案卷顺序标成"材料 A / 材料 B …"（与试点 run 一致；核验器按这个标签区分两份材料）。
来源策略：unrestricted → 开网页搜索；provided_media → 不搜索；exclude_original → 按 excluded_keys 过滤搜索结果和打开的页面；
source_bound → 用试点的 SourceBoundTextProvider。
检索护栏（阶段 04）：所有会搜索的模式外面再套一层 search.retrieval_guard —— 核查站点黑名单（--blocklist exclude|mark|off，
默认 exclude）与按案卷 date_cutoff 截断（--date-cutoff 可临时覆盖；抽不到日期的来源放行并打 date_unknown）。
拒绝与打标都记在 run 目录的 retrieval_guard.jsonl，汇总写进 run_config["retrieval_guard"] 与 result 侧的 guard_summary.json。

环境固化（docs/环境固化_录制回放与缓存.md）：
- 检索录制回放 --search-mode live|record|replay|replay_or_record（默认 live）+ --snapshot-root：快照按 case_id 共享，
  套在最里层 guard(filter(replay(base)))；INSPECT 受控获取的视频文件也一并录制回放。汇总写进 run_config["search_snapshot"]。
- 工具结果缓存 --tool-cache on|off|refresh（默认 off）+ --tool-cache-root：缓存 INSPECT processor 层的纯函数结果，
  E 编号仍由 Service 正常分配。汇总写进 run_config["tool_cache"]。
- 规格 harness_params 生效：--hi-spec / --gia-spec 选中的规格覆盖运行时配置（--harness-params apply，默认）；
  record 只记录不覆盖（旧行为，对照用）。映射表见 datagen/harness_params.py，记录写进 run_config["harness_params"]。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cases import Case, CaseError

SFT = Path("experiments") / "sft_data"
DEFAULT_CASES = SFT / "cases"
DEFAULT_RUNS = SFT / "runs"
MANIFESTS = SFT / "run_manifests"
LOGS = SFT / "batch_logs"
DEFAULT_SPECS = ("hi@0.4.0", "gia@0.5.0")     # 09-30：运行时提示的格式示例带 rationale，与这两版逐字一致
SEARCH_MODES = {"unrestricted", "exclude_original", "source_bound"}
INSPECT_OPERATIONS = ("get_frames", "get_pages", "crop", "ocr", "align_media")
DEFAULT_SNAPSHOT_ROOT = SFT / "search_snapshots"
DEFAULT_TOOL_CACHE_ROOT = SFT / "tool_cache"


# ---------------------------------------------------------------- 计划

@dataclass
class Planned:
    case_file: str
    case_id: str
    split: str
    sample: int
    run_name: str
    status: str            # todo | done | incomplete


def _safe(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "-", text).strip("-") or "x"


def run_name_for(case: Case, round_tag: str, sample: int) -> str:
    return f"{_safe(case.dataset)}_{_safe(case.source_id)}_{_safe(round_tag)}_{sample}"


def run_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "todo"
    if (run_dir / "result.json").is_file() and (run_dir / "execution_links.json").is_file():
        return "done"
    return "incomplete"


def plan(cases_dir: str | Path, *, runs_dir: str | Path, round_tag: str, samples: int = 1, split: str = "dev_train",
         ids: list[str] | None = None, limit: int = 0) -> list[Planned]:
    """列出要跑的 (案例, 采样)。split='all' 不筛；ids 可以是 case_id（true:4124903）或文件名（true_4124903）。"""

    root, runs = Path(cases_dir), Path(runs_dir)
    wanted = set(ids or ())
    chosen: list[tuple[Path, Case]] = []
    for path in sorted(root.glob("*.json")):
        try:
            case = Case.load(path)
        except (CaseError, ValueError, KeyError) as exc:
            raise SystemExit(f"案卷读不了：{path}（{exc}）") from exc
        if wanted and case.case_id not in wanted and path.stem not in wanted:
            continue
        if not wanted and split != "all" and case.effective_split != split:
            continue
        chosen.append((path, case))
    missing = wanted - {c.case_id for _, c in chosen} - {p.stem for p, _ in chosen}
    if missing:
        raise SystemExit(f"找不到这些案卷：{sorted(missing)}")
    if limit:
        chosen = chosen[:limit]
    out: list[Planned] = []
    for path, case in chosen:
        for k in range(1, samples + 1):
            name = run_name_for(case, round_tag, k)
            out.append(Planned(str(path), case.case_id, case.effective_split, k, name, run_status(runs / name)))
    return out


# ---------------------------------------------------------------- 单个 run 的组装

def media_inputs(case: Case, case_file: Path) -> tuple[Any, ...]:
    """案卷媒体 → InitialMedia。第 i 个媒体标成"材料 A/B/…"，文件名 A.mp4 / B.jpg …（与试点 run 相同）。"""

    from ..gia import InitialMedia

    out = []
    for index, item in enumerate(case.media):
        path = (case_file.parent / item.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{case.case_id} 的媒体 {item.ref} 不存在：{path}")
        letter = chr(ord("A") + index)
        kind = item.media_type.split("/", 1)[0]
        modality = {"video": "video", "image": "image", "audio": "audio"}.get(kind, "document")
        out.append(InitialMedia(source_ref=f"I{index}", media_type=item.media_type, modalities=(modality,),
                                label=f"材料 {letter}", original_filename=letter + path.suffix.lower(), path=path))
    return tuple(out)


class KeyExclusion:
    """按案卷 excluded_keys 过滤来源（接口与试点 source_policy.ArticleExclusion 相同，可直接套 FilteredProvider/Opener）。"""

    def __init__(self, keys: tuple[str, ...], log_path: Path) -> None:
        from .cases import _normalize_url

        # 存档地址（web.archive.org/web/<时间>/原址）按原址判断，与试点 source_policy.canonical_url 一致
        self._normalize = lambda url: _normalize_url(re.sub(r"^(https?://)?web\.archive\.org/web/[^/]+/", "", url.strip(), flags=re.I))
        self.keys = {self._normalize(k) for k in keys if k}
        self.log_path = log_path

    def matches(self, url: str | None) -> bool:
        if not url:
            return False
        key = self._normalize(url)
        return any(key == k or key.startswith(k.rstrip("/") + "/") for k in self.keys)

    def record(self, stage: str, source_url: str | None, final_url: str | None = None) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "source_url": source_url, "final_url": final_url,
                                     "reason": "case_source_policy_excluded_key"}, ensure_ascii=False) + "\n")


def search_service_for(case: Case, store: Any, output: Path, *, blocklist: str = "exclude",
                       date_cutoff: str | None = None, recording: Any | None = None) -> tuple[Any, str, Any]:
    """返回 (SearchService 或 None, 来源策略描述, RetrievalGuard 或 None)。

    包装顺序（由内到外）：真实 provider/opener → 录制回放（recording，可选）→ 案卷过滤 → 护栏。
    回放在最里层，所以快照里是原始世界，黑名单与日期截断在回放时照常生效。
    """

    mode = case.source_policy.mode
    if mode not in SEARCH_MODES:
        return None, "provided_media", None
    from ..search.factory import create_free_search_service
    from ..search.recording import apply_search_recording
    from ..search.retrieval_guard import RetrievalGuard, apply_retrieval_guard

    search = create_free_search_service(store=store, reverse_engine="none")
    apply_search_recording(search, recording)
    cutoff = date_cutoff if date_cutoff is not None else case.source_policy.date_cutoff
    guard = RetrievalGuard(blocklist=blocklist, date_cutoff=cutoff, log_path=output / "retrieval_guard.jsonl")  # type: ignore[arg-type]
    search, desc = _case_search(case, search, output)
    apply_retrieval_guard(search, guard)
    return search, desc, guard


def search_recording_for(case: Case, output: Path, run_name: str, *, search_mode: str = "live",
                         snapshot_root: str | Path = DEFAULT_SNAPSHOT_ROOT) -> Any | None:
    """--search-mode 不是 live 时建一个录制回放会话（快照按 case_id 共享，未命中日志写在 run 目录）。"""

    if search_mode == "live":
        return None
    from ..search.recording import SearchRecording

    return SearchRecording(search_mode, Path(snapshot_root), case.case_id, run_name=run_name,  # type: ignore[arg-type]
                           log_path=output / "search_snapshot.jsonl")


def inspect_service_for(store: Any, *, recording: Any | None = None, tool_cache: str = "off",
                        tool_cache_root: str | Path = DEFAULT_TOOL_CACHE_ROOT, policy: Any | None = None) -> tuple[Any, Any, Any]:
    """返回 (InspectService, ToolCache 或 None, 需要 run 结束时关闭的 acquirer 或 None)。

    录制回放模式下视频获取也走快照；工具缓存包在 dispatcher 的 processor 上（不碰分配 E 编号的 Service 层）。
    """

    from ..inspect.factory import create_inspect_service
    from ..inspect.tool_cache import ToolCache, apply_tool_cache

    acquirer = None
    if recording is not None and recording.mode != "live":
        from ..inspect.acquisition import DirectMediaAcquirer
        from ..search.recording import RecordingAcquirer

        acquirer = RecordingAcquirer(DirectMediaAcquirer(), recording)
    inspect = create_inspect_service(store=store, policy=policy, enabled_operations=frozenset(INSPECT_OPERATIONS),
                                     acquirer=acquirer)
    cache = None if tool_cache == "off" else ToolCache(Path(tool_cache_root), tool_cache)  # type: ignore[arg-type]
    apply_tool_cache(inspect, cache)
    return inspect, cache, acquirer


def harness_settings_for(config: Any, specs_root: str | Path, hi_spec: str, gia_spec: str, *,
                         mode: str = "apply", overrides: list[str] | None = None) -> Any:
    """读 --hi-spec / --gia-spec 两份规格的 harness_params，覆盖运行时配置（见 datagen/harness_params.py）。
    overrides：--harness-override 的 键=值 列表（对照实验用，优先于规格）。"""

    from .harness_params import build_harness_settings, parse_overrides
    from .specs import Spec

    root = Path(specs_root)
    return build_harness_settings(config, Spec.load(root / hi_spec), Spec.load(root / gia_spec), mode=mode,  # type: ignore[arg-type]
                                  overrides=parse_overrides(overrides))


def _case_search(case: Case, search: Any, output: Path) -> tuple[Any, str]:
    mode = case.source_policy.mode
    if mode == "exclude_original":
        if not case.source_policy.excluded_keys:
            raise ValueError(f"{case.case_id}: source_policy=exclude_original 但 excluded_keys 为空")
        sys.path.insert(0, str(SFT.resolve()))
        from source_policy import FilteredOpener, FilteredProvider  # type: ignore  # 试点脚本，在 experiments/sft_data/

        exclusion = KeyExclusion(case.source_policy.excluded_keys, output / "source_exclusions.jsonl")
        search.provider = FilteredProvider(search.provider, exclusion)
        search.opener = FilteredOpener(search.opener, exclusion)
        return search, "exclude_case_keys"
    if mode == "source_bound":
        sys.path.insert(0, str(SFT.resolve()))
        from source_bound_search import SourceBoundTextProvider  # type: ignore

        search.provider = SourceBoundTextProvider(search.provider, capture_dir=output / "search_captures")
        return search, "source_bound"
    return search, "unrestricted_retrospective_web"


def make_teacher(kind: str, *, store: Any, output: Path, instructions: dict[str, str], endpoint: str | None,
                 model: str | None, thinking: bool) -> Any:
    if kind == "endpoint":
        from .teacher import EndpointTeacher

        assert endpoint and model
        return EndpointTeacher(store=store, output_dir=output / "requests", endpoint=endpoint, model=model,
                               instruction=instructions, thinking=thinking)
    sys.path.insert(0, str(SFT.resolve()))
    from cli_generator import CliGenerator  # type: ignore  # 试点的 Codex CLI 教师

    from .teacher import RoleInstructions

    class RoleCli(CliGenerator):   # Codex 串行，每次调用前按角色换上教师指令
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._roles = RoleInstructions(instructions)

        async def generate_json(self, *, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
            from .teacher import role_of

            self.instruction = self._roles.by_role.get(role_of(payload), "")
            return await super().generate_json(system_prompt=system_prompt, payload=payload)

    return RoleCli(store=store, output_dir=output / "requests")


async def run_one(case_file: str | Path, run_name: str, *, runs_dir: str | Path, round_tag: str, sample: int,
                  teacher: str, endpoint: str | None, model: str | None, thinking: bool,
                  specs_root: str | Path, hi_spec: str, gia_spec: str, blocklist: str = "exclude",
                  date_cutoff: str | None = None, search_mode: str = "live",
                  snapshot_root: str | Path = DEFAULT_SNAPSHOT_ROOT, tool_cache: str = "off",
                  tool_cache_root: str | Path = DEFAULT_TOOL_CACHE_ROOT, harness_params: str = "apply",
                  harness_overrides: list[str] | None = None) -> dict[str, Any]:
    from .. import GlobalStore
    from ..config import load_config
    from ..gia import InvestigationRuntime
    from ..hi.workspace import WorkspaceHarness
    from ..local_models import TransformersGiaPolicy, TransformersHiPolicy
    from .harness_params import mapping_table, settings_snapshot
    from .teacher import spec_instructions

    config = load_config()
    case_path = Path(case_file)
    case = Case.load(case_path)
    # 规格参数先解析：未知键、类型 / 范围不对、两份规格冲突都在建 run 目录之前报错
    harness = harness_settings_for(config, specs_root, hi_spec, gia_spec, mode=harness_params, overrides=harness_overrides)
    output = Path(runs_dir) / run_name
    output.mkdir(parents=True, exist_ok=False)
    media = media_inputs(case, case_path)
    instructions, spec_ids = spec_instructions(specs_root, hi_spec, gia_spec)
    store = GlobalStore(output / "store.sqlite3", output / "artifacts")
    generator = make_teacher(teacher, store=store, output=output, instructions=instructions, endpoint=endpoint,
                             model=model, thinking=thinking)
    if harness.mode == "apply" and hasattr(generator, "image_policy"):
        generator.image_policy = harness.inspect_policy      # 教师附图上限与 INSPECT 的 visual_qa 上限同源（image_input_limit）
    runtime_limits = harness.runtime

    def record_generation(value: dict[str, Any]) -> None:
        workspace = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace else "GIA", actor_id=workspace or "GIA",
                           event_type="raw_model_output", payload=value)

    generator.generation_observer = record_generation
    recording = search_recording_for(case, output, run_name, search_mode=search_mode, snapshot_root=snapshot_root)
    search, source_desc, guard = search_service_for(case, store, output, blocklist=blocklist, date_cutoff=date_cutoff,
                                                    recording=recording)
    inspect, cache, acquirer = inspect_service_for(
        store, recording=recording, tool_cache=tool_cache, tool_cache_root=tool_cache_root,
        policy=harness.inspect_policy if harness.mode == "apply" else None)
    concurrency = 1 if teacher == "codex" else runtime_limits.max_concurrency
    runtime = InvestigationRuntime(
        store=store, gia_policy=TransformersGiaPolicy(generator), hi_policy=TransformersHiPolicy(generator),
        inspect_service=inspect, search_service=search, max_rounds=runtime_limits.max_rounds,
        memory_policy=harness.memory_policy if harness.mode == "apply" else None,
        max_gia_turns=runtime_limits.max_gia_turns, gia_turn_timeout_s=runtime_limits.gia_turn_timeout_s,
        max_workspace_turns=runtime_limits.max_workspace_turns, workspace_timeout_s=runtime_limits.workspace_timeout_s,
        max_concurrency=concurrency, max_workspaces_per_round=runtime_limits.max_workspaces_per_round,
        report_review_delay_s=None, workspace_harness_type=WorkspaceHarness)
    run_config = {
        "dataset": case.dataset, "case": case.source_id, "case_id": case.case_id, "case_file": str(case_path.resolve()),
        "split": case.effective_split, "round": round_tag, "sample": sample, "mode": "bridge",
        "gia_output_source": "openai_compatible" if teacher == "endpoint" else "codex_cli",
        "task": case.task_text, "image_binding": "claim_materials",
        "media_labels": {m.source_ref: m.label for m in media},
        "evidence_source_policy": source_desc, "source_policy": asdict(case.source_policy),
        "retrieval_guard": guard.summary() if guard else None,
        "date_cutoff_enforced": bool(guard and guard.date_cutoff) if case.source_policy.date_cutoff or date_cutoff else None,
        "configured_model": getattr(generator, "model_config", None), "teacher_specs": spec_ids,
        "instruction": instructions, "search": "ddgs_text" if search else None,
        "inspect_operations": list(INSPECT_OPERATIONS), "harness_root": None,
        "config_profile": config.profile, "config_source": str(config.source) if config.source else None,
        "runtime_limits": asdict(runtime_limits) | {"max_concurrency": concurrency},
        "driver": "foren_searcher.datagen.run_cases",
        # 环境固化：录制回放、工具缓存（计数在 run 结束时再写一次，见下方 finally）
        "search_mode": search_mode,
        "search_snapshot": recording.summary() if recording is not None else {"search_mode": "live"},
        "tool_cache": cache.summary() if cache is not None else {"mode": "off"},
        # 规格 harness_params：键 → 值 → 来源规格；与默认配置（configs 的 [runtime] + 各 Policy 默认）的差异
        "harness_params_applied": harness.applied,
        "harness_overrides": list(harness_overrides or []),
        "harness_params": harness.record(),
        "harness_param_map": mapping_table(),
        "effective_settings": settings_snapshot(harness),
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        async with runtime:
            result = await runtime.run_new(task=case.task_text, claim_materials={"C0": {"task": case.task_text}}, media=media)
            payload = asdict(result)
            (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    finally:
        if acquirer is not None:
            await acquirer.aclose()
        if recording is not None or cache is not None:
            _finish_environment_records(output, run_config, recording, cache)
        if guard is not None:
            (output / "guard_summary.json").write_text(json.dumps(guard.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        store.export_trajectory(output / "trajectory.jsonl")
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    from .links import write_execution_links

    summary = write_execution_links(output)
    return {"run": run_name, "decision": payload.get("decision"), "links": summary["statuses"]}


def _finish_environment_records(output: Path, run_config: dict[str, Any], recording: Any | None, cache: Any | None) -> None:
    """run 结束后把录制回放 / 工具缓存的最终计数补进 run_config.json，并写快照目录里的 usage/<run>.json。"""

    if recording is not None:
        run_config["search_snapshot"] = recording.summary()
        recording.write_usage()
    if cache is not None:
        run_config["tool_cache"] = cache.summary()
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 成本统计

def run_stats(run_dir: str | Path) -> dict[str, Any]:
    """从一个 run 目录读出成本：教师调用数、失败数、输入/输出 token、附图数、教师耗时、结果与报告数。"""

    root = Path(run_dir)
    requests = sorted((root / "requests").glob("request_*")) if (root / "requests").is_dir() else []
    stats: dict[str, Any] = {"teacher_calls": len(requests), "teacher_failed": 0, "input_tokens": 0, "output_tokens": 0,
                             "images": 0, "teacher_seconds": 0.0, "roles": {"HI": 0, "GIA": 0}}
    for request in requests:
        try:
            transport = json.loads((request / "transport.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["teacher_failed"] += 1
            continue
        stats["teacher_failed"] += int(transport.get("returncode") != 0)
        usage = transport.get("usage") or {}
        stats["input_tokens"] += int(usage.get("input_tokens") or 0)
        stats["output_tokens"] += int(usage.get("output_tokens") or 0)
        stats["images"] += int(transport.get("images") or 0)
        stats["teacher_seconds"] += float(transport.get("elapsed_s") or 0.0)
        try:
            role = "HI" if json.loads((request / "request.json").read_text(encoding="utf-8"))["payload"].get("workspace_id") else "GIA"
            stats["roles"][role] += 1
        except (OSError, ValueError, KeyError):
            pass
    stats["teacher_seconds"] = round(stats["teacher_seconds"], 1)
    config_file = root / "run_config.json"
    if config_file.is_file():
        try:
            run_config = json.loads(config_file.read_text(encoding="utf-8"))
        except ValueError:
            run_config = {}
        snapshot = run_config.get("search_snapshot") or {}
        if snapshot.get("search_mode") not in (None, "live"):
            stats["search_snapshot"] = {"mode": snapshot["search_mode"], **(snapshot.get("counts") or {})}
        cache = run_config.get("tool_cache") or {}
        if cache.get("mode") not in (None, "off"):
            stats["tool_cache"] = {"mode": cache["mode"], **(cache.get("counts") or {})}
    result_file = root / "result.json"
    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            stats["decision"] = result.get("decision")
        except ValueError:
            pass
    trajectory = root / "trajectory.jsonl"
    if trajectory.is_file():
        reports = 0
        for line in trajectory.read_text(encoding="utf-8").splitlines():
            if '"workspace_report"' in line:
                reports += 1
        stats["workspace_reports"] = reports
    return stats


# ---------------------------------------------------------------- 批量

def _teacher_target(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """endpoint 教师：解析地址与模型名，服务连不上就在开跑前退出。"""

    if args.teacher != "endpoint":
        return None, None
    from ..config import load_config
    from .teacher import TEACHER_ENDPOINT_ENV, TEACHER_MODEL_ENV
    from .verify.judges import LocalJudgeUnavailable, list_local_models

    endpoint = args.endpoint or os.environ.get(TEACHER_ENDPOINT_ENV) or load_config().models.teacher_endpoint
    if not endpoint:
        raise SystemExit(f"没有教师服务地址：用 --endpoint，或设 {TEACHER_ENDPOINT_ENV}，或在 configs 的 models.teacher_endpoint 里写"
                         "（服务器 configs/server.toml 已写 http://127.0.0.1:8000/v1）。本机用 Codex 教师加 --teacher codex。")
    try:
        served = list_local_models(endpoint)
    except LocalJudgeUnavailable as exc:
        raise SystemExit(f"!!! 教师服务连不上：{exc}") from exc
    model = args.model or os.environ.get(TEACHER_MODEL_ENV)
    if model and served and model not in served:
        raise SystemExit(f"教师服务 {endpoint} 上没有模型 {model!r}；现有 {served}")
    if not model:
        if len(served) != 1:
            raise SystemExit(f"教师服务上有 {len(served)} 个模型 {served}，请用 --model 指定")
        model = served[0]
    return endpoint.rstrip("/"), model


def _worker_command(args: argparse.Namespace, item: Planned, endpoint: str | None, model: str | None) -> list[str]:
    command = [sys.executable, "-m", "foren_searcher.datagen.run_cases", "--one", item.case_file, "--run-name", item.run_name,
               "--sample", str(item.sample), "--round", args.round, "--runs-dir", str(args.runs_dir),
               "--teacher", args.teacher, "--specs", str(args.specs), "--hi-spec", args.hi_spec, "--gia-spec", args.gia_spec]
    if endpoint:
        command += ["--endpoint", endpoint, "--model", model or ""]
    if args.no_think:
        command.append("--no-think")
    command += ["--blocklist", args.blocklist]
    if args.date_cutoff:
        command += ["--date-cutoff", args.date_cutoff]
    command += ["--search-mode", args.search_mode, "--snapshot-root", str(args.snapshot_root),
                "--tool-cache", args.tool_cache, "--tool-cache-root", str(args.tool_cache_root),
                "--harness-params", args.harness_params]
    if getattr(args, "harness_override", None):
        command += ["--harness-override", *args.harness_override]
    return command


def _run_batch(args: argparse.Namespace, todo: list[Planned], endpoint: str | None, model: str | None) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    LOGS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = MANIFESTS / f"{_safe(args.round)}.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, ["src", os.environ.get("PYTHONPATH")])),
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    results: list[dict[str, Any]] = []

    def one(index_item: tuple[int, Planned]) -> dict[str, Any]:
        index, item = index_item
        log = LOGS / f"{item.run_name}.log"
        print(f"[{index}/{len(todo)}] 开跑 {item.run_name}（{item.case_id}，日志 {log}）", flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.call(_worker_command(args, item, endpoint, model), stdout=stream, stderr=subprocess.STDOUT, env=env)
        wall = round(time.monotonic() - started, 1)
        run_dir = Path(args.runs_dir) / item.run_name
        row = {"run": item.run_name, "case_id": item.case_id, "split": item.split, "round": args.round,
               "sample": item.sample, "teacher": args.teacher, "teacher_model": model or "codex", "returncode": code,
               "blocklist": args.blocklist, "date_cutoff": args.date_cutoff, "search_mode": args.search_mode,
               "tool_cache": args.tool_cache, "harness_params": args.harness_params,
               "status": run_status(run_dir), "wall_seconds": wall, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **run_stats(run_dir)}
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        mark = "完成" if row["status"] == "done" else f"失败（退出码 {code}，看日志 {log}）"
        print(f"[{index}/{len(todo)}] {item.run_name}：{mark}；{wall:.0f}s，教师 {row['teacher_calls']} 次，"
              f"输入 {row['input_tokens']:,} / 输出 {row['output_tokens']:,} token", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(one, enumerate(todo, 1)))
    print(f"\n成本记录追加到 {manifest}", flush=True)
    return results


def _summarize(rows: list[dict[str, Any]]) -> None:
    done = [r for r in rows if r["status"] == "done"]
    print(f"\n本批 {len(rows)} 个 run：完成 {len(done)}，失败 {len(rows) - len(done)}", flush=True)
    if done:
        n = len(done)
        avg = lambda key: sum(float(r.get(key) or 0) for r in done) / n   # noqa: E731
        print(f"  每个 run 平均：墙钟 {avg('wall_seconds') / 60:.1f} 分钟，教师调用 {avg('teacher_calls'):.0f} 次，"
              f"输入 {avg('input_tokens'):,.0f} token，输出 {avg('output_tokens'):,.0f} token，附图 {avg('images'):.0f} 张", flush=True)


def _then_verify(args: argparse.Namespace, run_names: list[str]) -> int:
    command = [sys.executable, str(Path("scripts") / "complete_verify_local.py"), "--runs", *run_names,
               "--hi-spec", args.hi_spec, "--gia-spec", args.gia_spec, "--round", args.round]
    if args.judge == "openrouter":
        command += ["--judge", "openrouter", "--yes-spend"]
    elif args.judge == "claude-cli":
        command += ["--judge", "claude-cli", "--skip", "4"]      # 订阅额度：召回校准最费调用，看数据时不需要
    if args.judge_endpoint:
        command += ["--endpoint", args.judge_endpoint]
    if args.judge_model:
        command += ["--model", args.judge_model]
    print("\n接着核验与导出：python " + " ".join(command[1:]), flush=True)
    return subprocess.call(command)


def _report_harness_params(args: argparse.Namespace) -> None:
    """开跑前在主进程里解析一次规格参数：有错就在这里停，不要让每个子进程各报一次；有差异就打印出来。"""

    from ..config import load_config
    from .harness_params import HarnessParamError
    from .specs import SpecError

    try:
        settings = harness_settings_for(load_config(), args.specs, args.hi_spec, args.gia_spec, mode=args.harness_params,
                                        overrides=getattr(args, "harness_override", None))
    except (HarnessParamError, SpecError, OSError) as exc:
        raise SystemExit(f"规格 harness_params 不可用：{exc}") from exc
    verb = "覆盖" if settings.mode == "apply" else "只记录、不覆盖"
    if settings.differences:
        print(f"harness_params（{verb}）与默认配置不同的项：", flush=True)
        for key, item in settings.differences.items():
            print(f"  {key}: {item['default']!r} → {item['spec']!r}（{item['target']}，来自 {item['source']}）", flush=True)
    else:
        print(f"harness_params（{verb}）与默认配置一致", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量跑调查：案卷 → 教师 bridge run → execution_links → 成本记录")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="案卷目录")
    parser.add_argument("--split", default="dev_train", help="只跑这个划分（all = 不筛）；给了 --ids 时忽略")
    parser.add_argument("--ids", nargs="+", default=None, help="只跑这些案例（case_id 或案卷文件名）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个案例（先小规模试）")
    parser.add_argument("--round", default="r0", help="轮次标签，进 run 名与成本记录文件名")
    parser.add_argument("--samples", type=int, default=1, help="每个案例跑几次（学生自采时用 4）")
    parser.add_argument("--teacher", default="endpoint", choices=["endpoint", "codex"])
    parser.add_argument("--endpoint", default=None, help="教师服务地址（OpenAI 兼容，/v1 结尾）")
    parser.add_argument("--model", default=None, help="教师服务上的模型名")
    parser.add_argument("--no-think", action="store_true", help="关掉教师思考（默认开）")
    parser.add_argument("--specs", default="specs", help="规格根目录（取教师指令）")
    parser.add_argument("--hi-spec", default=DEFAULT_SPECS[0])
    parser.add_argument("--gia-spec", default=DEFAULT_SPECS[1])
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--jobs", type=int, default=1, help="同时跑几个 run（服务器 vLLM 可开 2–4；Codex 只能 1）")
    parser.add_argument("--blocklist", default="exclude", choices=["exclude", "mark", "off"],
                        help="核查站点黑名单：exclude 丢弃（产数据默认）/ mark 只打标（测试对照）/ off")
    parser.add_argument("--date-cutoff", default=None, help="临时覆盖案卷的 date_cutoff（YYYY-MM-DD）")
    parser.add_argument("--search-mode", default="live", choices=["live", "record", "replay", "replay_or_record"],
                        help="检索录制回放：live 直连（默认）/ record 联网并录 / replay 只读快照（未命中报 replay_miss）"
                             " / replay_or_record 命中回放、未命中联网补录")
    parser.add_argument("--snapshot-root", default=str(DEFAULT_SNAPSHOT_ROOT), help="检索快照根目录（按 case_id 分子目录）")
    parser.add_argument("--tool-cache", default="off", choices=["on", "off", "refresh"],
                        help="INSPECT 工具结果缓存：off（默认）/ on 命中复用 / refresh 一律重算并覆盖")
    parser.add_argument("--tool-cache-root", default=str(DEFAULT_TOOL_CACHE_ROOT), help="工具缓存目录")
    parser.add_argument("--harness-params", default="apply", choices=["apply", "record"],
                        help="规格 harness_params：apply 覆盖运行时配置（默认）/ record 只记录不生效（旧行为，对照用）")
    parser.add_argument("--harness-override", nargs="*", default=None, metavar="键=值",
                        help="对照实验用的临时参数覆盖（优先于规格，记进 run_config.harness_overrides），例如 image_input_limit=16 search_excerpt_chars=600")
    parser.add_argument("--retry-failed", action="store_true", help="只跑了一半的 run：旧目录改名 .failed-<时间> 后重跑")
    parser.add_argument("--dry-run", action="store_true", help="只列计划，不跑")
    parser.add_argument("--then-verify", action="store_true", help="跑完后对本批完成的 run 调 scripts/complete_verify_local.py")
    parser.add_argument("--judge", default="local", choices=["local", "openrouter", "claude-cli"],
                        help="--then-verify 用哪个核验器：local（默认）/ openrouter（付费，跳过确认）/ claude-cli（本机 Claude Code，订阅额度）")
    parser.add_argument("--judge-endpoint", default=None, help="--then-verify 用的本地核验服务地址")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--one", default=None, help=argparse.SUPPRESS)       # 子进程：跑单个 run
    parser.add_argument("--run-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.one:
        outcome = asyncio.run(run_one(args.one, args.run_name, runs_dir=args.runs_dir, round_tag=args.round, sample=args.sample,
                                      teacher=args.teacher, endpoint=args.endpoint, model=args.model, thinking=not args.no_think,
                                      specs_root=args.specs, hi_spec=args.hi_spec, gia_spec=args.gia_spec,
                                      blocklist=args.blocklist, date_cutoff=args.date_cutoff,
                                      search_mode=args.search_mode, snapshot_root=args.snapshot_root,
                                      tool_cache=args.tool_cache, tool_cache_root=args.tool_cache_root,
                                      harness_params=args.harness_params, harness_overrides=args.harness_override))
        print(json.dumps(outcome, ensure_ascii=False, default=str), flush=True)
        return 0

    if args.teacher == "codex" and args.jobs > 1:
        raise SystemExit("Codex CLI 教师只能串行（--jobs 1）")
    planned = plan(args.cases, runs_dir=args.runs_dir, round_tag=args.round, samples=args.samples, split=args.split,
                   ids=args.ids, limit=args.limit)
    counts = {s: sum(p.status == s for p in planned) for s in ("todo", "done", "incomplete")}
    print(f"计划：{len(planned)} 个 run（待跑 {counts['todo']}，已完成 {counts['done']}，半截 {counts['incomplete']}）；"
          f"教师 {args.teacher}；轮次 {args.round}", flush=True)
    for p in planned:
        print(f"  {p.status:10s} {p.run_name:45s} {p.case_id}（{p.split}）", flush=True)
    if not planned:
        print("没有符合条件的案卷。", flush=True)
        return 1
    _report_harness_params(args)
    if args.dry_run:
        return 0
    todo = [p for p in planned if p.status == "todo"]
    for p in planned:
        if p.status == "incomplete":
            if args.retry_failed:
                old = Path(args.runs_dir) / p.run_name
                old.rename(old.with_name(f"{p.run_name}.failed-{time.strftime('%Y%m%d-%H%M%S')}"))
                todo.append(p)
            else:
                print(f"  跳过半截的 {p.run_name}（要重跑加 --retry-failed，旧目录会改名保留）", flush=True)
    if not todo:
        print("没有要跑的 run。", flush=True)
        rows: list[dict[str, Any]] = []
    else:
        endpoint, model = _teacher_target(args)
        if endpoint:
            print(f"教师：{model} @ {endpoint}（思考{'关' if args.no_think else '开'}）；并行 {args.jobs}", flush=True)
        rows = _run_batch(args, todo, endpoint, model)
        _summarize(rows)
    if args.then_verify:
        finished = [p.run_name for p in planned if run_status(Path(args.runs_dir) / p.run_name) == "done"]
        if not finished:
            print("没有已完成的 run 可核验。", flush=True)
            return 1
        return _then_verify(args, finished)
    return 0 if all(r["status"] == "done" for r in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
