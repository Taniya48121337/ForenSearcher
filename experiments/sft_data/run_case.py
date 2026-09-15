"""Run a sampled case through the two-role investigation runtime.

路径、运行限制来自 configs/*.toml（FOREN_CONFIG 选择）；不再依赖 Harness-1（DESIGN.md 第 7 节）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cli_generator import CliGenerator
from source_policy import article_key, exclude_original_article
from source_bound_search import PARSER_VERSION, SourceBoundTextProvider
from foren_searcher import GlobalStore
from foren_searcher.config import load_config
from foren_searcher.hi.workspace import WorkspaceHarness
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import TransformersGiaPolicy, TransformersHiPolicy
from foren_searcher.gia import InitialMedia, InvestigationRuntime
from foren_searcher.search.factory import create_free_search_service


CONFIG = load_config()


def _videosham_root() -> Path:
    """VideoSham 原始数据目录：优先 configs 里的 datasets 目录，其次旧的归档位置。"""

    candidates = [CONFIG.resolve(CONFIG.paths.datasets) / "videosham",
                  PROJECT / "docs" / "01_早期探索_已放弃" / "claimunit_video_sham_1004" / "datasets" / "videosham",
                  PROJECT / "claimunit_video_sham_1004" / "datasets" / "videosham"]
    for candidate in candidates:
        if (candidate / "VideoSourceandEdits.csv").is_file():
            return candidate
    raise FileNotFoundError("VideoSham 数据未找到；请把它放到 configs 里 [paths].datasets 指向的目录下的 videosham/，候选位置: "
                            + "; ".join(str(c) for c in candidates))


def prepare_videosham(case_id: str, usage: str = "development") -> tuple[Path, dict]:
    source_root = _videosham_root()
    rows = list(csv.DictReader((source_root / "VideoSourceandEdits.csv").open(encoding="utf-8-sig")))
    row = next(row for row in rows if row["Video-Name"] == case_id + ".mp4")
    case_dir = ROOT / "data" / "cases" / ("videosham_" + case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    inputs = case_dir / "inputs"
    inputs.mkdir(exist_ok=True)
    # Neutral asset names remove the source/processed label from model-visible metadata.
    for filename, kind, suffix in [("A.mp4", "edited", "processed"), ("B.mp4", "source", "unedited")]:
        destination = inputs / filename
        if not destination.exists():
            shutil.copyfile(source_root / "media" / kind / (case_id + "_" + suffix + ".mp4"), destination)
    reference = {"dataset": "VideoSham", "source_record": row,
                 "input_assignment": {"A.mp4": "processed", "B.mp4": "unedited"},
                 "reference_status": "official_edit_description_requires_media_confirmation",
                 "usage": usage, "condition": "provided_candidate_pair"}
    (case_dir / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_dir, reference


def prepare_case(dataset: str, case_id: str, question_index: int | None = None,
                 usage: str = "development"):
    if dataset == "videosham":
        case_dir, reference = prepare_videosham(case_id, usage)
        task = ("比较给定视频 A 与候选视频 B，调查两者可由材料确认的差异及其对画面含义的影响。"
                "引用具体证据，说明差异所在位置和未能确定的内容。给定候选不代表已经确认其来源身份；"
                "不要把未检查的区域或模态视为没有差异。")
    else:
        case_dir = ROOT / "data" / dataset / case_id
        reference = json.loads((case_dir / "reference.json").read_text(encoding="utf-8"))
        if dataset == "qvhighlights":
            task = "在给定视频中定位符合以下描述的片段，给出秒数范围及证据；说明定位的不确定性：" + reference["query"]
        elif dataset == "vcsl":
            task = "在给定视频 A 与 B 中查找相同或经变换的对应片段，给出两侧时间范围及证据。不要把内容对应推断成原创归属。输入仅包含视频轨，不评估音频。"
        else:
            claim = reference.get("claim", reference.get("claim_text"))
            task = "核查以下声明，必要时检查给定媒体并搜索外部证据。依据实际证据回答，说明结论范围与不确定性：" + claim
            date = reference.get("claim_date", reference.get("date"))
            if date:
                task += "\n声明日期（任务元数据）：" + date
            if question_index is not None:
                question = reference["questions"][question_index]["question"]
                task = "完成以下局部调查问题，给出直接支持回答的证据，并准确说明来源与适用范围。证据类型取决于问题本身：问题要求核验原站、原件或最早来源时需满足该要求；概括报道的说法时保留转述关系，不自动追加原站核验：" + question + "\n背景声明：" + claim
                if date:
                    task += "\n声明日期（任务元数据）：" + date
    return case_dir, reference, task


class LocalTaskDispatcher:
    """Provide an externally assigned local task; its outputs are not GIA SFT targets."""
    async def decide(self, *, model_view, history, actions):
        if not model_view.get("latest_reports"):
            materials = [item.get("evidence_id", item.get("reference_id"))
                         for item in model_view.get("original_materials", [])]
            materials = [item for item in materials if item]
            return {"decision": "INVESTIGATE", "tasks": [{"task": model_view["task"], "materials": materials}]}
        return {"decision": "FINISH", "answer": "本次局部任务执行结束；结果保存在工作区报告中。"}


async def run(args: argparse.Namespace) -> None:
    case_dir, reference, task = prepare_case(args.dataset, args.case, args.question_index, args.usage)
    task_file = getattr(args, "task_file", None)
    if task_file:
        task = Path(task_file).read_text(encoding="utf-8").strip()
        if not task:
            raise ValueError("The task file must contain a non-empty task")
    output = ROOT / "runs" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    store = GlobalStore(output / "store.sqlite3", output / "artifacts")
    instruction = Path(args.instruction_file).read_text(encoding="utf-8") if args.instruction_file else args.instruction
    generator = CliGenerator(store=store, output_dir=output / "requests", instruction=instruction)
    def record_generation(value: dict) -> None:
        workspace = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace else "GIA", actor_id=workspace or "GIA",
                           event_type="raw_model_output", payload=value)
    generator.generation_observer = record_generation
    inspect = create_inspect_service(store=store, enabled_operations=frozenset({"get_frames", "get_pages", "crop", "ocr", "align_media"}))
    harness_root = CONFIG.harness_root  # 默认 None：直接用本项目的 WorkspaceHarness
    if harness_root is not None:
        from foren_searcher.harness1 import harness1_workspace_type
        workspace_type = harness1_workspace_type(harness_root)
    else:
        workspace_type = WorkspaceHarness
    search = create_free_search_service(store=store, reverse_engine="none") if args.dataset in {"true", "averitec", "averimatec"} else None
    source_policy = "unrestricted_retrospective_web" if search else "provided_media"
    applied_source_keys, applied_aliases = [], []
    if args.source_bound_search:
        if search is None:
            raise ValueError("Source-bound search requires a web-search dataset")
        search.provider = SourceBoundTextProvider(search.provider, capture_dir=output / "search_captures")
    if args.exclude_original_article:
        if args.dataset not in {"averitec", "averimatec"}:
            raise ValueError("Original-article exclusion is defined for the two fact-check datasets")
        article = reference.get("fact_checking_article", reference.get("article"))
        exclude_original_article(search, article, output / "source_exclusions.jsonl")
        applied_source_keys = sorted(search.provider.exclusion.keys)
        applied_aliases = [group for group in json.loads((ROOT / "source_article_aliases.json").read_text(encoding="utf-8"))
                           if article_key(group["reference_url"]) in applied_source_keys]
        source_policy = "exclude_known_original_article; publication_dates_not_filtered"
        task += "\n本次要求独立取得支持回答的资料；该案例原事实核查文章不作为证据来源。搜索环境排除其已知地址及跳转目标，但不保证其他网页的时间或出处正确，仍须按问题核对。"
    if args.source_hint:
        if search is None:
            raise ValueError("A source hint requires an available search service")
        task += ("\n提供的候选来源：" + args.source_hint +
                 "\n该地址由任务提供，尚未成为已核实证据。实际打开并核对相关正文、出处及时间范围；存档请求日期与最终取得的存档日期可能不同。")
    runtime = InvestigationRuntime(store=store, gia_policy=LocalTaskDispatcher() if args.mode == "hi" else TransformersGiaPolicy(generator),
        hi_policy=TransformersHiPolicy(generator), inspect_service=inspect, search_service=search,
        max_rounds=CONFIG.runtime.max_rounds, max_gia_turns=CONFIG.runtime.max_gia_turns,
        gia_turn_timeout_s=CONFIG.runtime.gia_turn_timeout_s,
        max_workspace_turns=CONFIG.runtime.max_workspace_turns, workspace_timeout_s=CONFIG.runtime.workspace_timeout_s,
        max_concurrency=1,  # 教师是单个 CLI 会话，串行执行
        max_workspaces_per_round=CONFIG.runtime.max_workspaces_per_round, report_review_delay_s=None,
        workspace_harness_type=workspace_type)
    inputs_dir = case_dir / "inputs"
    image_binding = "claim_materials"
    if args.question_images:
        if args.dataset != "averimatec" or args.question_index is None:
            raise ValueError("Question-image mode requires an AVerImaTeC local question")
        inputs_dir = case_dir / "question_inputs" / f"q{args.question_index}"
        if not (inputs_dir / "mapping.json").is_file():
            raise FileNotFoundError("Fetch official question image inputs before running this mode")
        image_binding = "official_question_input_images"
        task += "\n附带图片是该局部问题的官方输入图片，问题中的图像指代以这些附件为准。"
    files = sorted(inputs_dir.glob("*")) if inputs_dir.is_dir() else []
    files = [p for p in files if p.suffix.lower() in {".mp4", ".jpg", ".png"}]
    media = tuple(InitialMedia(source_ref=f"I{i}", media_type="video/mp4" if p.suffix == ".mp4" else "image/png" if p.suffix == ".png" else "image/jpeg",
        modalities=("video",) if p.suffix == ".mp4" else ("image",), label=f"材料 {p.stem}", original_filename=p.name, path=p)
        for i, p in enumerate(files))
    (output / "run_config.json").write_text(json.dumps({"dataset": args.dataset, "case": args.case,
        "mode": args.mode, "gia_output_source": "scripted_dispatcher" if args.mode == "hi" else "codex_cli",
        "question_index": args.question_index,
        "task": task,
        "task_file": str(Path(task_file).resolve()) if task_file else None,
        "image_binding": image_binding, "usage": args.usage,
        "evidence_source_policy": source_policy,
        "source_condition": "provided_source_url" if args.source_hint else "source_discovery" if search else "provided_media",
        "provided_source_url": args.source_hint,
        "source_exclusion_keys": applied_source_keys,
        "source_exclusion_aliases": applied_aliases,
        "text_search_parser": PARSER_VERSION if args.source_bound_search else "installed_ddgs",
        "configured_model": generator.model_config,
        "instruction": instruction, "search": "ddgs_text" if search else None,
        "inspect_operations": ["get_frames", "get_pages", "crop", "ocr", "align_media"],
        "harness_root": str(harness_root) if harness_root else None,
        "config_profile": CONFIG.profile, "config_source": str(CONFIG.source) if CONFIG.source else None,
        "runtime_limits": asdict(CONFIG.runtime)}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        async with runtime:
            result = await runtime.run_new(task=task, claim_materials={"C0": {"task": task}}, media=media)
            payload = asdict(result)
            (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"run": args.run_name, "runtime_returned": True,
                              "decision": payload.get("decision")}, ensure_ascii=False, default=str), flush=True)
    finally:
        store.export_trajectory(output / "trajectory.jsonl")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="1028")
    parser.add_argument("--dataset", choices=["videosham", "true", "averitec", "averimatec", "qvhighlights", "vcsl"], default="videosham")
    parser.add_argument("--mode", choices=["bridge", "hi"], default="bridge")
    parser.add_argument("--question-index", type=int)
    parser.add_argument("--task-file", help="UTF-8 task text for an explicit task on the selected case materials")
    parser.add_argument("--question-images", action="store_true")
    parser.add_argument("--exclude-original-article", action="store_true")
    parser.add_argument("--source-bound-search", action="store_true")
    parser.add_argument("--source-hint")
    parser.add_argument("--usage", default="development", choices=["development", "construction_validation"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--instruction-file")
    asyncio.run(run(parser.parse_args()))
