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
- SEARCH 查询守卫（10-05，search/query_guard.py，--query-guard on 默认）：查询文字里出现本次调查实际存在的 E/R/F/W/C 编号
  或"Material A / 材料A / 给定视频"这类代称时，在调用搜索后端前拒收（INVALID_INPUT），反馈怎么改写。拒收记在 run 目录的
  query_guard.jsonl，计数写进 run_config["query_guard"]。off 与以前完全相同。
- 视频搜索（10-05 晚，--video-search ytdlp 默认）：文字 → 视频的 SEARCH 先走 yt-dlp 的 YouTube 搜索，失败或空结果再退回 DDGS；
  其余请求不变。原因：evidb2 冒烟里 DDGS 视频搜索 21 次全部 DDGSException。走向计数写进 run_config["video_search"]。
  ddgs = 10-05 以前的行为。
- --instruction-mode student（10-04）：不拼教师指令，模型只看学生提示（= 部署时学生看到的系统提示）。
  用于"未训练底座放进架构"的基线（例如 EVID-Bench baseline）；默认 teacher 与以前完全相同。

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
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .cases import Case, CaseError

SFT = Path("experiments") / "sft_data"
DEFAULT_CASES = SFT / "cases"
DEFAULT_RUNS = SFT / "runs"
MANIFESTS = SFT / "run_manifests"
LOGS = SFT / "batch_logs"
DEFAULT_SPECS = ("hi@0.5.0", "gia@0.6.0")     # 10-05 晚：hi@0.5.0 = 0.4.2 + 学生提示加"查询不写内部编号 / 出处类任务先找原件"；gia@0.6.0 = 0.5.0 + "出处类任务至少一项找原始素材"。运行时提示已同步，新跑的 run 实际用的就是这两版
SEARCH_MODES = {"unrestricted", "exclude_original", "source_bound"}
INSPECT_OPERATIONS = ("get_frames", "get_pages", "crop", "ocr", "align_media")     # 不需要模型权重的基础操作
# 09-30：r0c 发现语音转写、按文字找帧、图内定位从没开过（.models/inspect 里有权重）。--inspect-ops auto
# 在"权重目录在 + 依赖包装了"时加开；base 只开基础五项（与 r0c 及以前的 run 可比）。
# 操作 → (环境变量, .models/inspect 下的约定目录, 需要的包)；目录名与 demo._INSPECT_MODEL_DIRECTORIES 一致。
MODEL_INSPECT_OPERATIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "asr": ("FOREN_WHISPER_MODEL", "faster-whisper-small", ("faster_whisper",)),
    "locate_frames": ("FOREN_CLIP_MODEL", "clip-vit-base-patch32", ("transformers", "torch")),
    "locate_regions": ("FOREN_GROUNDING_MODEL", "grounding-dino-tiny", ("transformers", "torch")),
}
INSPECT_OPS_CHOICES = ("auto", "base")
# 10-12：DETECT 已并入 SEARCH 与 INSPECT（规格 inspect_merged_catalog，hi@0.8.0 / gia@0.9.0）。--detect on 只保留一个版本
# 作为弃用别名 = --harness-override inspect_merged_catalog=true；--detect-config 不再读取。
DETECT_MODES = ("off", "on")
AUTO_TOOLS_MODES = ("off", "on", "compare")   # 10-14 compare：只自动比对，入库时不跑 video_structure
DEFAULT_DETECT_CONFIG = Path("configs") / "detect.evid.json"
# 10-12：以图搜图默认开。auto = 所有有 key 的引擎 + Yandex（免 key）同时搜、结果合并（search/providers/multi_reverse.py）；
# all = 四个都要（缺 key 报错）；off = 10-12 以前；也可逗号列出 gcv,lens,tineye,yandex 中的几个。
IMAGE_ENGINES = ("gcv", "lens", "tineye", "yandex")
IMAGE_SEARCH_MODES = ("auto", "all", "off", *IMAGE_ENGINES)
VIDEO_SEARCH_MODES = ("ytdlp", "ddgs")       # 文字 → 视频：ytdlp = YouTube 搜索优先、DDGS 兜底（默认）；ddgs = 10-05 以前
QUERY_GUARD_MODES = ("on", "off")           # SEARCH 查询守卫（search/query_guard.py）；off = 10-05 以前的行为
RUNTIME_GUARD_MODES = ("off", "eval", "eval_plus")   # 评测兜底（policies.RuntimeGuards）；off = 默认（SFT 数据生成）；
                                                     # eval = 方案第 1 步（不改模型输入）；eval_plus = 再加第 2 步（改输入，须起新规格）
INSTRUCTION_MODES = ("teacher", "student")   # teacher：学生提示 + 教师指令（产数据，默认）；student：只给学生提示（基线）


def resolve_inspect_operations(mode: str = "auto", model_root: str | Path | None = None) -> tuple[tuple[str, ...], dict[str, Any]]:
    """返回 (启用的 INSPECT 操作, 说明)。auto：环境变量没设时按约定目录补上；权重目录存在且包可导入才开。

    说明记进 run_config.inspect_operations_detail：每个模型操作开没开、没开的原因（缺权重 / 缺包）。
    """

    import importlib.util

    if mode not in INSPECT_OPS_CHOICES:
        raise ValueError(f"--inspect-ops 只能是 {INSPECT_OPS_CHOICES}：{mode!r}")
    enabled = list(INSPECT_OPERATIONS)
    detail: dict[str, Any] = {"mode": mode, "model_ops": {}}
    if mode == "base":
        return tuple(enabled), detail
    root = Path(model_root) / "inspect" if model_root is not None else None
    for operation, (variable, directory, packages) in MODEL_INSPECT_OPERATIONS.items():
        if not os.environ.get(variable) and root is not None and (root / directory).is_dir():
            os.environ[variable] = str((root / directory).resolve())
        weights = os.environ.get(variable)
        missing_pkgs = [p for p in packages if importlib.util.find_spec(p) is None]
        if not weights or not Path(weights).exists():
            detail["model_ops"][operation] = {"enabled": False, "reason": f"缺权重（{variable} 未设，且 {root / directory if root else '?'} 不存在）"}
        elif missing_pkgs:
            detail["model_ops"][operation] = {"enabled": False, "reason": f"缺包 {', '.join(missing_pkgs)}", "weights": weights}
        else:
            detail["model_ops"][operation] = {"enabled": True, "weights": weights}
            enabled.append(operation)
    return tuple(enabled), detail
def resolve_image_engines(mode: str = "auto", root: str | Path | None = None) -> list[str]:
    """--image-search → 实际用的引擎列表（顺序 gcv, lens, tineye, yandex）。

    auto：Google Cloud Vision（GOOGLE_VISION_API_KEY / .local/google_vision_key.txt）、Google Lens（SERPAPI_API_KEY 或
    SERPER_API_KEY / .local/serpapi_key.txt、serper_key.txt）、TinEye（TINEYE_API_KEY / .local/tineye_key.txt）有 key 的
    都开，Yandex 不要 key、总是开。all 与逗号列表：列出的都要开，缺 key 在建 run 目录之前报错。"""

    from ..search.providers.keys import find_key

    root = Path(root) if root is not None else Path.cwd()
    have = {"gcv": bool(find_key("gcv", root)), "lens": bool(find_key("serpapi", root) or find_key("serper", root)),
            "tineye": bool(find_key("tineye", root)), "yandex": True}
    if mode == "off":
        return []
    if mode == "auto":
        return [e for e in IMAGE_ENGINES if have[e]]
    wanted = list(IMAGE_ENGINES) if mode == "all" else [x.strip() for x in mode.split(",") if x.strip()]
    unknown = [e for e in wanted if e not in IMAGE_ENGINES]
    if unknown:
        raise ValueError(f"--image-search 里不认识的引擎 {unknown}；可选 {IMAGE_ENGINES}")
    missing = [e for e in wanted if not have[e]]
    if missing:
        raise ValueError(f"--image-search 要的引擎缺 API key：{missing}（见 search/providers/keys.py 的环境变量与 .local 文件名）")
    return [e for e in IMAGE_ENGINES if e in wanted]


def resolve_image_search_engine(mode: str = "auto", root: str | Path | None = None) -> str:
    """旧接口（10-12 上午）：引擎列表连成一个字符串，如 "gcv+yandex"；off / 空 = "off"。"""

    return "+".join(resolve_image_engines(mode, root)) or "off"


def image_engine_providers(names: list[str], root: str | Path | None, timeout_s: float) -> list[tuple[str, Any]]:
    from ..search.providers.keys import find_key, require_key

    root = Path(root) if root is not None else Path.cwd()
    out: list[tuple[str, Any]] = []
    for name in names:
        if name == "gcv":
            from ..search.providers.google_vision import GoogleVisionWebProvider

            out.append((name, GoogleVisionWebProvider(require_key("gcv", root), timeout_s)))
        elif name == "lens":
            from ..search.providers.google_lens import GoogleLensProvider

            out.append((name, GoogleLensProvider(serpapi_key=find_key("serpapi", root), serper_key=find_key("serper", root),
                                                 timeout_s=max(timeout_s, 45.0))))
        elif name == "tineye":
            from ..search.providers.tineye import TinEyeProvider

            out.append((name, TinEyeProvider(require_key("tineye", root), max(timeout_s, 45.0))))
        elif name == "yandex":
            from ..search.providers.yandex_reverse import YandexReverseImageProvider

            out.append((name, YandexReverseImageProvider(timeout_s)))
    return out


def with_spec_operations(operations: tuple[str, ...], detail: dict[str, Any], harness: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """10-04：规格 harness_params 开启的可选 INSPECT 操作（目前只有 inspect_diff_media → diff_media）。

    只在 --harness-params apply 时生效；record 模式与没写该键的旧规格（hi@0.4.1 及以前）操作集不变。
    说明写进 run_config.inspect_operations_detail.spec_ops。
    """

    switches = getattr(harness, "inspect_operations", None)
    applied = switches is not None and getattr(harness, "mode", "apply") == "apply"
    if applied and getattr(switches, "merged_catalog", False):
        return merged_catalog_operations(operations, detail)
    extra = switches.operations() if applied else ()
    detail = {**detail, "spec_ops": {name: {"enabled": True, "source": "harness_params.inspect_diff_media"} for name in extra}}
    return tuple(dict.fromkeys((*operations, *extra))), detail


def merged_catalog_operations(operations: tuple[str, ...], detail: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    """10-12 合并目录（harness_params.inspect_merged_catalog）：HI 看到的 INSPECT 操作换成六项。

    get_frames / ocr / video_structure / compare_media 不需要额外权重；asr 要 whisper；locate 在 CLIP（视频）或
    Grounding DINO（图片）任一可用时开放，另一种媒体会得到 BACKEND_UNAVAILABLE。旧操作名由服务内部翻译，不进目录。
    """

    from ..inspect.contracts import MERGED_OPERATIONS

    have = set(operations)
    enabled = ["get_frames", "ocr", "video_structure", "compare_media"]
    if "asr" in have:
        enabled.insert(2, "asr")
    if have & {"locate_frames", "locate_regions"}:
        enabled.insert(-2, "locate")
    assert set(enabled) <= MERGED_OPERATIONS
    detail = {**detail, "merged_catalog": {
        "source": "harness_params.inspect_merged_catalog", "operations": enabled,
        "locate_media": [kind for kind, op in (("video", "locate_frames"), ("image", "locate_regions")) if op in have],
        "not_enabled": sorted(MERGED_OPERATIONS - set(enabled))}}
    return tuple(enabled), detail


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

def media_inputs(case: Case, case_file: Path, language: str = "zh") -> tuple[Any, ...]:
    """案卷媒体 → InitialMedia。第 i 个媒体标成"材料 A/B/…"，文件名 A.mp4 / B.jpg …（与试点 run 相同）。

    language="en"（规格 harness_params.language，10-05 起）时标成 "Material A/B/…"，模型输入全英文。"""

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
                                label=(f"Material {letter}" if language == "en" else f"材料 {letter}"), original_filename=letter + path.suffix.lower(), path=path))
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
        # 带通配（* ? [）的键按 glob 判（10-11 分叉试点 V1 用案卷 url_pattern 原样屏蔽）；其余仍按"整段路径前缀"判，旧案卷行为不变
        return any(fnmatch.fnmatch(key, k) if any(c in k for c in "*?[") else (key == k or key.startswith(k.rstrip("/") + "/"))
                   for k in self.keys)

    def record(self, stage: str, source_url: str | None, final_url: str | None = None) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "source_url": source_url, "final_url": final_url,
                                     "reason": "case_source_policy_excluded_key"}, ensure_ascii=False) + "\n")


def search_service_for(case: Case, store: Any, output: Path, *, blocklist: str = "exclude",
                       date_cutoff: str | None = None, recording: Any | None = None,
                       query_guard: Any | None = None, video_search: str = "ytdlp",
                       search_arm: frozenset[str] = frozenset(),
                       image_search: str = "auto") -> tuple[Any, str, Any]:
    """返回 (SearchService 或 None, 来源策略描述, RetrievalGuard 或 None)。

    image_search（10-12 起默认 auto = 以图搜图打开，见 resolve_image_engines）：多个引擎同时搜、结果合并；off = 10-12
    以前的行为（image / text_image 输入不出现在工具目录）。以图搜图会把调查里的图片（帧、截图）发给对应搜索引擎。
    --search-arm 的 imageclip / gcvimage 优先于本开关。

    包装顺序（由内到外）：真实 provider/opener → 录制回放（recording，可选）→ 案卷过滤 → 护栏。
    回放在最里层，所以快照里是原始世界，黑名单与日期截断在回放时照常生效。
    """

    mode = case.source_policy.mode
    if mode not in SEARCH_MODES:
        return None, "provided_media", None
    from ..search.factory import create_free_search_service
    from ..search.recording import apply_search_recording
    from ..search.retrieval_guard import RetrievalGuard, apply_retrieval_guard

    engines = ([] if search_arm & {"imageclip", "gcvimage"} else ["yandex"] if "yandeximage" in search_arm
               else resolve_image_engines(image_search))
    search = create_free_search_service(store=store, reverse_engine="none")
    search.image_search_engine = ("gcvimage" if "gcvimage" in search_arm else "imageclip" if "imageclip" in search_arm
                                  else "+".join(engines) or "off")
    if video_search == "ytdlp":          # 放在录制回放之内：快照里记的是实际取得的候选
        from ..search.providers.video_fallback import VideoSearchFallbackProvider
        from ..search.providers.ytdlp import YtDlpSearchProvider

        primary: Any = YtDlpSearchProvider()
        if "fanout5" in search_arm:      # 10-06 找原片实验 A4：程序扇出（只作消融）
            from ..search.arms import FanoutVideoProvider

            primary = FanoutVideoProvider(primary)
        search.provider = VideoSearchFallbackProvider(primary, search.provider, explain_empty="guardlen" in search_arm)
    if search_arm:
        from ..search.arms import video_policy_for

        search.video_policy = video_policy_for(search.policy, search_arm)
    if "imageclip" in search_arm:        # 10-06 找原片实验 A3：本机以图搜视频（也在录制回放之内）
        from ..search.providers.thumbnail_clip import ImageQueryRouter, ThumbnailClipProvider
        from ..search.providers.ytdlp import YtDlpSearchProvider

        search.provider = ImageQueryRouter(search.provider, ThumbnailClipProvider(YtDlpSearchProvider(), store))
        search.local_image_search = True
    if "gcvimage" in search_arm:         # 10-12 以图搜图：Google Cloud Vision Web Detection（会把调查图片发给 Google；也在录制回放之内）
        from ..search.providers.google_vision import GoogleVisionWebProvider, resolve_api_key
        from ..search.providers.thumbnail_clip import ImageQueryRouter, text_also_note

        if "imageclip" in search_arm:
            raise ValueError("--search-arm 里 gcvimage 与 imageclip 不能同时开")
        search.provider = ImageQueryRouter(search.provider, GoogleVisionWebProvider(resolve_api_key(Path.cwd()),
                                                                                  search.policy.provider_timeout_s),
                                           text_also=True)
        search.text_image_wide = True
        search._allow_image_uploads = True
        search.image_search_note = text_also_note(GoogleVisionWebProvider.NOTE)
    if engines:                          # 10-12 多引擎以图搜图（也在录制回放之内）
        from ..search.providers.multi_reverse import MultiReverseImageProvider
        from ..search.providers.thumbnail_clip import ImageQueryRouter, text_also_note

        timeout = getattr(getattr(search, "policy", None), "provider_timeout_s", 30.0)
        multi = MultiReverseImageProvider(image_engine_providers(engines, Path.cwd(), timeout))
        # 10-13：text_image 的文字同时走文字搜索（evidM27a：只走以图搜图时 YouTube 文字搜索整个丢了）
        search.provider = ImageQueryRouter(search.provider, multi, text_also=True)
        search.text_image_wide = True
        search._allow_image_uploads = True
        search.image_search_note = text_also_note(multi.note())
    apply_search_recording(search, recording)
    cutoff = date_cutoff if date_cutoff is not None else case.source_policy.date_cutoff
    guard = RetrievalGuard(blocklist=blocklist, date_cutoff=cutoff, log_path=output / "retrieval_guard.jsonl")  # type: ignore[arg-type]
    search, desc = _case_search(case, search, output)
    apply_retrieval_guard(search, guard)
    from ..search.query_guard import apply_query_guard

    apply_query_guard(search, query_guard)       # 最外层：拒收的查询不联网、不进录制快照
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
                        tool_cache_root: str | Path = DEFAULT_TOOL_CACHE_ROOT, policy: Any | None = None,
                        operations: tuple[str, ...] = INSPECT_OPERATIONS) -> tuple[Any, Any, Any]:
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
    inspect = create_inspect_service(store=store, policy=policy, enabled_operations=frozenset(operations),
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
                 model: str | None, thinking: bool, render_inputs: bool = False) -> Any:
    if kind == "endpoint":
        from .teacher import EndpointTeacher

        assert endpoint and model
        return EndpointTeacher(store=store, output_dir=output / "requests", endpoint=endpoint, model=model,
                               instruction=instructions, thinking=thinking, render_inputs=render_inputs)
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


# ---------------------------------------------------------------------------- r0 冻结（10-15）：工具层 hash + 冻结守卫
# 工具层 = 决定模型看到的工具行为和输出措辞的源文件：hi/workspace.py（观察行、自动比对）、inspect/ 全部、search/ 全部、
# state_render.py（卡片与历史步渲染）。对这些文件的内容（包内相对路径 + 换行统一成 LF，Windows / 服务器检出一致）算 sha256，
# 写进 run_config.tool_layer_hash。specs/<id>_freeze.json 记冻结值；run 的 hi / gia 规格与某个冻结文件一致时，逐项核对
# 规格 hash、渲染档、附图上限、GIA 定时复核、工具层 hash 等，不一致就在 stderr 报警、run_config.freeze.ok=false（不中断 run；
# walker_run 的预检在开跑前比一次，FOREN_FREEZE_STRICT=1 时不开跑）。见 claude/r0冻结口径_20261015.md。
TOOL_LAYER_PATHS = ("hi/workspace.py", "inspect", "search", "state_render.py")
# 紧挨工具层、也带工具目录 / 结果标签文字的文件：只记各自 hash 供排查漂移，不进冻结值
TOOL_LAYER_ADJACENT = ("prompts_en.py", "policies.py", "local_models.py", "gia/global_view.py")
FREEZE_GLOB = "*_freeze.json"


def _layer_files(root: Path, entries: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for entry in entries:
        path = root / entry
        if path.is_dir():
            files.update(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)
        elif path.is_file():
            files.add(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _hash_files(root: Path, files: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return "sha256:" + digest.hexdigest()


def tool_layer_hash(package_root: str | Path | None = None) -> dict[str, Any]:
    """工具层 hash：{"hash", "files", "paths", "groups"（每项单独的 hash）, "adjacent"（不进冻结值）}。"""

    root = Path(package_root) if package_root else Path(__file__).resolve().parents[1]
    files = _layer_files(root, TOOL_LAYER_PATHS)
    return {"hash": _hash_files(root, files), "files": len(files), "paths": list(TOOL_LAYER_PATHS),
            "line_endings": "crlf_to_lf",
            "groups": {entry: _hash_files(root, _layer_files(root, (entry,))) for entry in TOOL_LAYER_PATHS},
            "adjacent": {entry: _hash_files(root, _layer_files(root, (entry,))) for entry in TOOL_LAYER_ADJACENT
                         if (root / entry).exists()}}


def load_freeze(specs_root: str | Path, hi_spec: str, gia_spec: str) -> dict[str, Any] | None:
    """specs/ 下与这一对规格一致的冻结文件（specs.hi 与 specs.gia 都相同）；没有返回 None。"""

    for path in sorted(Path(specs_root).glob(FREEZE_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        specs = data.get("specs") or {}
        if specs.get("hi") == hi_spec and specs.get("gia") == gia_spec:
            return {**data, "_file": path.as_posix()}
    return None


def freeze_check(freeze: dict[str, Any] | None, actual: dict[str, Any]) -> dict[str, Any]:
    """按冻结文件逐项核对：spec_hash.hi / spec_hash.gia / tool_layer_hash / checks 里的每个键。返回写进 run_config.freeze 的记录。"""

    if not freeze:
        return {"applies": False}
    expected: dict[str, Any] = {f"spec_hash.{role}": value for role, value in (freeze.get("spec_hash") or {}).items()}
    expected["tool_layer_hash"] = (freeze.get("tool_layer") or {}).get("hash")
    expected.update(freeze.get("checks") or {})
    items = {key: {"expected": value, "actual": actual.get(key), "ok": actual.get(key) == value}
             for key, value in expected.items()}
    failed = [key for key, item in items.items() if not item["ok"]]
    return {"applies": True, "freeze_id": freeze.get("freeze_id"), "file": freeze.get("_file"), "ok": not failed,
            "failed": failed, "items": items}


async def run_one(case_file: str | Path, run_name: str, *, runs_dir: str | Path, round_tag: str, sample: int,
                  teacher: str, endpoint: str | None, model: str | None, thinking: bool,
                  specs_root: str | Path, hi_spec: str, gia_spec: str, blocklist: str = "exclude",
                  date_cutoff: str | None = None, search_mode: str = "live",
                  snapshot_root: str | Path = DEFAULT_SNAPSHOT_ROOT, tool_cache: str = "off",
                  tool_cache_root: str | Path = DEFAULT_TOOL_CACHE_ROOT, harness_params: str = "apply",
                  harness_overrides: list[str] | None = None, inspect_ops: str = "auto",
                  instruction_mode: str = "teacher", query_guard: str = "on",
                  video_search: str = "ytdlp", runtime_guards: str = "off",
                  state_renderer: str | None = None, max_rounds: int | None = None,
                  search_arm: str | None = None, detect: str = "off",
                  detect_config: str | Path | None = None, image_search: str = "auto",
                  auto_tools: str = "off") -> dict[str, Any]:
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
    # P0.1（10-05）：--state-renderer 是评测档的强制渲染档位，作为一条 override 优先于规格 harness_params
    # （hi@0.5.0 / gia@0.6.0 写的是 legacy，EVID 12 轮循环里会顶过 32K），run_config 单独记一笔
    overrides = list(harness_overrides or []) + ([f"state_renderer={state_renderer}"] if state_renderer else [])
    if detect not in DETECT_MODES:
        raise ValueError(f"--detect 只能是 {DETECT_MODES}：{detect!r}")
    resolve_image_engines(image_search)        # 10-12：引擎名不对 / 缺 key 在建 run 目录之前报错
    if detect == "on":          # 10-12 弃用别名：DETECT 已并入 INSPECT compare_media / video_structure 与 SEARCH content_match
        print("警告：--detect on 已弃用，按 --harness-override inspect_merged_catalog=true 运行"
              "（请改用 --hi-spec hi@0.8.0 --gia-spec gia@0.9.0）；--detect-config 不再读取", file=sys.stderr, flush=True)
        overrides.append("inspect_merged_catalog=true")
    # 规格参数先解析：未知键、类型 / 范围不对、两份规格冲突都在建 run 目录之前报错
    harness = harness_settings_for(config, specs_root, hi_spec, gia_spec, mode=harness_params, overrides=overrides)
    language = harness.input_features.language if harness.mode == "apply" else "zh"
    # 10-07：英文规格交给模型英文任务文字（案卷 task_text_en）；缺了在建 run 目录之前报错
    task_text = case.task_for(language)
    output = Path(runs_dir) / run_name
    output.mkdir(parents=True, exist_ok=False)
    media = media_inputs(case, case_path, language)
    instructions, spec_ids = spec_instructions(specs_root, hi_spec, gia_spec)
    if instruction_mode not in INSTRUCTION_MODES:
        raise ValueError(f"--instruction-mode 只能是 {INSTRUCTION_MODES}：{instruction_mode!r}")
    if instruction_mode == "student":          # 基线：只给学生提示，不拼教师指令
        instructions = {role: "" for role in instructions}
    store = GlobalStore(output / "store.sqlite3", output / "artifacts")
    # 10-05（教师输入 = 学生输入）：两种教师都按规格 / --state-renderer 的档位渲染输入，教师模式也一样——以前教师模式
    # 看 legacy 全量视图、导出再给学生渲染，r0e 的 35 条报告监督里 17 条用了学生看不到的内容。档位是 legacy 时不变
    # （hi@0.5.0 / gia@0.6.0 的 r0e 不受影响），hi@0.6.0 / gia@0.7.0 起是 compact_16k。Codex 教师见 cli_generator。
    render_inputs = True
    generator = make_teacher(teacher, store=store, output=output, instructions=instructions, endpoint=endpoint,
                             model=model, thinking=thinking, render_inputs=render_inputs)
    if harness.mode == "apply" and hasattr(generator, "image_policy"):
        generator.image_policy = harness.inspect_policy      # 教师附图上限与 INSPECT 的 visual_qa 上限同源（image_input_limit）
    runtime_limits = harness.runtime
    if max_rounds is not None:
        # P1.7（10-05）：EVID 配置 max_rounds 2 → 3；P0 的 A/B 用 --max-rounds 2 固定与 evidb4 可比
        if max_rounds < 1:
            raise ValueError(f"--max-rounds 应为正整数：{max_rounds}")
        runtime_limits = replace(runtime_limits, max_rounds=max_rounds)

    def record_generation(value: dict[str, Any]) -> None:
        workspace = value.get("workspace_id")
        store.record_event(actor_role="HI" if workspace else "GIA", actor_id=workspace or "GIA",
                           event_type="raw_model_output", payload=value)

    generator.generation_observer = record_generation
    recording = search_recording_for(case, output, run_name, search_mode=search_mode, snapshot_root=snapshot_root)
    if query_guard not in QUERY_GUARD_MODES:
        raise ValueError(f"--query-guard 只能是 {QUERY_GUARD_MODES}：{query_guard!r}")
    if runtime_guards not in RUNTIME_GUARD_MODES:
        raise ValueError(f"--runtime-guards 只能是 {RUNTIME_GUARD_MODES}：{runtime_guards!r}")
    from ..policies import EVAL_GUARDS, EVAL_PLUS_GUARDS, RuntimeGuards, runtime_guard_scope

    guards = {"eval": EVAL_GUARDS, "eval_plus": EVAL_PLUS_GUARDS}.get(runtime_guards)
    # P1（10-05）：规格 harness_params 打开的输入补全特性并进同一个 scope（旧规格全关 → 与以前完全相同）
    from ..policies import merge_input_features

    input_features = harness.input_features.enabled() if harness.mode == "apply" else {}
    guards = merge_input_features(guards, input_features)
    if auto_tools not in AUTO_TOOLS_MODES:
        raise ValueError(f"--auto-tools 只能是 {AUTO_TOOLS_MODES}：{auto_tools!r}")
    if auto_tools in {"on", "compare"}:
        # 10-14 评测协议：合并目录上恢复 DETECT 的自动比对 / 结构分析（policies.RuntimeGuards.auto_*）。
        # compare：入库时的 video_structure 让 HI 先读结构表、不去找原片（evidM27h video_180 / 209 零次 SEARCH），只留自动比对
        if not (harness.mode == "apply" and harness.inspect_operations.merged_catalog):
            raise ValueError(f"--auto-tools {auto_tools} 需要合并目录规格（hi@0.8.0 / gia@0.9.0 起，inspect_merged_catalog=true）")
        guards = replace(guards or RuntimeGuards(), auto_compare_on_match=True, auto_video_structure=auto_tools == "on")
    from ..search.query_guard import QueryGuard

    from ..search.arms import parse_search_arm

    arm = parse_search_arm(search_arm)
    if "guardlen" in arm and query_guard != "on":
        raise ValueError("--search-arm guardlen 需要 --query-guard on")
    qguard = QueryGuard(store.has_reference, log_path=output / "query_guard.jsonl",
                        video_rules="guardlen" in arm) if query_guard == "on" else None
    if video_search not in VIDEO_SEARCH_MODES:
        raise ValueError(f"--video-search 只能是 {VIDEO_SEARCH_MODES}：{video_search!r}")
    search, source_desc, guard = search_service_for(case, store, output, blocklist=blocklist, date_cutoff=date_cutoff,
                                                    recording=recording, query_guard=qguard, video_search=video_search,
                                                    search_arm=arm, image_search=image_search)
    if search is not None and harness.mode == "apply" and getattr(harness.input_features, "search_dates", False):
        search.date_lookup = True          # 10-13 规格 search_dates（hi@0.9.0 起）：SEARCH 的日期查询进 HI 工具目录
    image_router = _find_provider(search, "ImageQueryRouter")
    video_router = _find_video_router(search)
    operations, operations_detail = resolve_inspect_operations(inspect_ops, config.model_root)
    operations, operations_detail = with_spec_operations(operations, operations_detail, harness)
    inspect, cache, acquirer = inspect_service_for(
        store, recording=recording, tool_cache=tool_cache, tool_cache_root=tool_cache_root,
        policy=harness.inspect_policy if harness.mode == "apply" else None, operations=operations)
    compact = harness.mode == "apply" and bool(getattr(harness.input_features, "compact_tool_params", False))
    if compact:
        # 10-13 规格 compact_tool_params（hi@0.10.0 起）：compare_media 无 mode（画面对不上时只比声音）、打开视频自动带字幕
        inspect.compact_params = True
        if search is not None:
            search.auto_subtitles = True
    concurrency = 1 if teacher == "codex" else runtime_limits.max_concurrency
    # 10-12 合并目录（DETECT 并入）：固定描述模型、人脸身份依赖、SEARCH 视频结果与材料视频的画面比对；GIA / HI 提示随开关切换
    merged = is_merged_harness(harness)
    inspect_summary: dict[str, Any] = {"merged_catalog": merged}
    matcher = None
    if merged:
        guards = replace(guards or RuntimeGuards(), merged_catalog=True)
        inspect_summary.update(configure_merged_inspect(inspect, generator, config.model_root))
        if search is not None:
            matcher = source_matcher_for(store, inspect, output)
            search.matcher = matcher
    runtime_extras = {"workspace_harness_type": WorkspaceHarness}
    runtime = InvestigationRuntime(
        store=store, gia_policy=TransformersGiaPolicy(generator), hi_policy=TransformersHiPolicy(generator),
        inspect_service=inspect, search_service=search, max_rounds=runtime_limits.max_rounds,
        memory_policy=harness.memory_policy if harness.mode == "apply" else None,
        max_gia_turns=runtime_limits.max_gia_turns, gia_turn_timeout_s=runtime_limits.gia_turn_timeout_s,
        max_workspace_turns=runtime_limits.max_workspace_turns, workspace_timeout_s=runtime_limits.workspace_timeout_s,
        max_concurrency=concurrency, max_workspaces_per_round=runtime_limits.max_workspaces_per_round,
        report_review_delay_s=None, **runtime_extras)
    # r0 冻结（10-15）：规格全 hash、渲染档、附图上限、GIA 定时复核（上面写死 None）、工具层 hash，与 specs/*_freeze.json 逐项核对
    from .specs import Spec as _FrozenSpec

    layer = tool_layer_hash()
    configured = getattr(generator, "model_config", None)
    teacher_model = ((configured.get("model") if isinstance(configured, dict) else None) or model
                     or (os.environ.get("FOREN_CODEX_MODEL") if teacher == "codex" else None))
    frozen_actual = {
        "spec_hash.hi": _FrozenSpec.load(Path(specs_root) / hi_spec).hash(),
        "spec_hash.gia": _FrozenSpec.load(Path(specs_root) / gia_spec).hash(),
        "tool_layer_hash": layer["hash"],
        "state_renderer": getattr(harness.memory_policy, "state_renderer", None) if harness.mode == "apply" else None,
        "image_input_limit": getattr(harness.inspect_policy, "maximum_visual_qa_images", None) if harness.mode == "apply" else None,
        "gia_report_review_delay_s": getattr(runtime, "report_review_delay_s", None),
        "harness_params_mode": harness.mode,
        "harness_overrides": list(harness_overrides or []),
        "state_renderer_override": state_renderer,
        "auto_tools": auto_tools,
        "instruction_mode": instruction_mode,
    }
    freeze = freeze_check(load_freeze(specs_root, hi_spec, gia_spec), frozen_actual)
    run_config = {
        "dataset": case.dataset, "case": case.source_id, "case_id": case.case_id, "case_file": str(case_path.resolve()),
        "split": case.effective_split, "round": round_tag, "sample": sample, "mode": "bridge",
        "gia_output_source": "openai_compatible" if teacher == "endpoint" else "codex_cli",
        "task": task_text, "task_language": language, "image_binding": "claim_materials",
        "media_labels": {m.source_ref: m.label for m in media},
        "evidence_source_policy": source_desc, "source_policy": asdict(case.source_policy),
        "retrieval_guard": guard.summary() if guard else None,
        "date_cutoff_enforced": bool(guard and guard.date_cutoff) if case.source_policy.date_cutoff or date_cutoff else None,
        "configured_model": getattr(generator, "model_config", None), "teacher_specs": spec_ids,
        "instruction": instructions, "instruction_mode": instruction_mode,
        "query_guard": qguard.summary() if qguard is not None and search is not None else {"enabled": False},
        "video_search": video_router.summary() if video_router is not None else {"mode": video_search if search else None}, "search": "ddgs_text" if search else None,
        # 10-06 找原片实验的检索臂（search/arms.py）；空 = A0（与以前完全相同）
        "search_arm": {"features": sorted(arm),
                       "video_policy": asdict(search.video_policy) if search is not None and getattr(search, "video_policy", None) else None,
                       "local_image_search": bool(getattr(search, "local_image_search", False))} if arm else {"features": []},
        "runtime_guards": {"mode": runtime_guards, **({"guards": guards.to_dict()} if guards else {}),
                           "input_features": input_features, "state_renderer_override": state_renderer,
                           "teacher_renders_state": harness.memory_policy.state_renderer != "legacy"
                           if harness.mode == "apply" and harness.memory_policy is not None else bool(state_renderer),
                           "max_rounds_override": max_rounds},
        "inspect_operations": list(operations), "inspect_operations_detail": operations_detail, "harness_root": None,
        "inspect": inspect_summary,
        "image_search": image_search_summary(search, image_search),
        "search_dates": bool(getattr(search, "date_lookup", False)),
        "auto_tools": auto_tools,
        "compact_tool_params": compact,
        "search_match": matcher.summary() if matcher is not None else {"enabled": False},
        "detect": {"enabled": False, "alias_of": "inspect_merged_catalog"} if detect == "on" else {"enabled": False},
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
        # r0 冻结（10-15）
        "teacher_model": teacher_model,
        "spec_hash": {"hi": frozen_actual["spec_hash.hi"], "gia": frozen_actual["spec_hash.gia"]},
        "render_profile": frozen_actual["state_renderer"],
        "image_input_limit": frozen_actual["image_input_limit"],
        "gia_report_review_delay_s": frozen_actual["gia_report_review_delay_s"],
        "tool_layer_hash": layer["hash"],
        "tool_layer": layer,
        "freeze": freeze,
    }
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    if freeze.get("applies") and not freeze.get("ok"):
        detail = "；".join(f"{key}: 应为 {freeze['items'][key]['expected']!r}，实际 {freeze['items'][key]['actual']!r}"
                          for key in freeze["failed"])
        print(f"警告：{run_name} 与冻结口径 {freeze.get('freeze_id')}（{freeze.get('file')}）不一致——{detail}",
              file=sys.stderr, flush=True)
    guard_box: dict[str, Any] = {}
    try:
        with runtime_guard_scope(guards) as guard_state:
            guard_box["state"] = guard_state
            async with runtime:
                result = await runtime.run_new(task=task_text, claim_materials={"C0": {"task": task_text}}, media=media)
                payload = asdict(result)
                if guard_state is not None:
                    # 兜底打过的标（finish_without_citation / forced_finish_* …）跟结果放在一起，评分与导出都能看到
                    payload["contract_flags"] = list(guard_state.contract_flags)
                (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    finally:
        if merged:
            run_config["inspect"] = {**run_config["inspect"], "calls": merged_inspect_calls(store),
                                     **merged_inspect_runtime(inspect)}
            if matcher is not None:
                run_config["search_match"] = matcher.summary()
            (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        if guard_box.get("state") is not None:
            run_config["runtime_guards"] = {**run_config["runtime_guards"], "mode": runtime_guards,
                                            **guard_box["state"].summary()}
            (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        if acquirer is not None:
            await acquirer.aclose()
        if recording is not None or cache is not None:
            _finish_environment_records(output, run_config, recording, cache)
        if video_router is not None:
            run_config["video_search"] = video_router.summary()
        if image_router is not None:
            usage = image_router.image_provider.summary()
            run_config["search_arm"]["image_search"] = usage
            run_config["image_search"] = {**run_config.get("image_search", {}), "usage": usage,
                                          "router": dict(getattr(image_router, "stats", {}) or {}),
                                          "text_image_also_text_search": bool(getattr(image_router, "text_also", False))}
        if (qguard is not None and search is not None) or video_router is not None:
            if qguard is not None and search is not None:
                run_config["query_guard"] = qguard.summary()
            (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        if guard is not None:
            (output / "guard_summary.json").write_text(json.dumps(guard.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        store.export_trajectory(output / "trajectory.jsonl")
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    from .links import write_execution_links

    summary = write_execution_links(output)
    return {"run": run_name, "decision": payload.get("decision"), "links": summary["statuses"]}


def is_merged_harness(harness: Any) -> bool:
    switches = getattr(harness, "inspect_operations", None)
    return bool(getattr(harness, "mode", "apply") == "apply" and getattr(switches, "merged_catalog", False))


def configure_merged_inspect(inspect: Any, generator: Any, model_root: str | Path | None) -> dict[str, Any]:
    """合并目录的 INSPECT 后端：固定描述模型（inspect/describe.py）与人脸身份依赖路径。返回写进 run_config.inspect 的说明。"""

    from ..inspect.describe import EndpointDescriber

    backend = getattr(getattr(inspect, "dispatcher", None), "probe_backend", None)
    pylib = add_face_pylib(model_root)
    describer = EndpointDescriber.resolve(getattr(generator, "model_config", None))
    if backend is not None:
        backend.describer = describer
    faces = None
    if backend is not None and hasattr(backend, "_face_identity"):
        try:
            faces = backend._face_identity() is not None
        except Exception:  # noqa: BLE001 - optional dependency
            faces = False
    return {"describer": describer.identity() if describer is not None else None, "face_identity": faces,
            "face_pylib": pylib}


def add_face_pylib(model_root: str | Path | None) -> str | None:
    """facenet-pytorch 等用 pip --target 装在 <models>/deepfake/pylib（原 detect.v2.json 的 pylib）；在就加进 sys.path。"""

    candidates = [os.environ.get("FOREN_PYLIB")]
    if model_root is not None:
        candidates.append(str(Path(model_root) / "deepfake" / "pylib"))
    for entry in candidates:
        if entry and Path(entry).is_dir():
            path = str(Path(entry).resolve())
            if path not in sys.path:
                sys.path.append(path)
            return path
    return None


def source_matcher_for(store: Any, inspect: Any, output: Path) -> Any:
    """SEARCH 视频结果 → 与材料视频的画面比对（search/source_match.py）。下载走 INSPECT 的受控获取（低清、纯画面，
    录制回放照常生效），对齐用 inspect.source_compare.SourceComparer.match。"""

    from ..inspect.source_compare import SourceComparer
    from ..search.source_match import SourceMatcher
    from ..store.resolver import EvidenceResolver

    comparer = SourceComparer()
    resolver = EvidenceResolver(store)
    folder = output / "source_match"

    async def fetch(candidate_id: str, source_url: str, page_evidence_id: str) -> Path:
        return await inspect.fetch_picture(candidate_id, page_evidence_id, folder / f"{candidate_id}.media")

    def align(path_a: Path, path_b: Path) -> dict:
        return comparer.match(path_a, path_b)

    def material_path(evidence_id: str) -> Path | None:
        resolved = resolver.resolve(evidence_id)
        return resolved.path if resolved.media_kind == "video" else None

    return SourceMatcher(store, fetch=fetch, align=align, material_path=material_path)


def image_search_summary(search: Any, mode: str) -> dict[str, Any]:
    if search is None:
        return {"mode": mode, "engine": None}
    engine = getattr(search, "image_search_engine", None)
    return {"mode": mode, "engine": engine, "uploads_images": bool(getattr(search, "allows_image_uploads", False))
            and engine not in {None, "off", "imageclip"}}


def merged_inspect_runtime(inspect: Any) -> dict[str, Any]:
    backend = getattr(getattr(inspect, "dispatcher", None), "probe_backend", None)
    describer = getattr(backend, "describer", None)
    return {"describe": {"calls": len(getattr(describer, "calls", []) or []),
                         "errors": list(getattr(backend, "describe_errors", []) or [])[:10]}}


def merged_inspect_calls(store: Any) -> dict[str, Any]:
    """本 run 里 compare_media / video_structure 的结果 E 数与比对图数（原 _detect_calls），进 run_config.inspect.calls。"""

    counts: dict[str, Any] = {"compare_media": 0, "video_structure": 0, "images": 0}
    try:
        records = store.evidence_records()
    except Exception:  # noqa: BLE001 - bookkeeping only
        return counts
    for record in records:
        acquired = getattr(record, "acquired_by", None) or ""
        if acquired not in {"INSPECT/compare_media", "INSPECT/video_structure"}:
            continue
        if "image" in (getattr(record, "modalities", None) or ()):
            counts["images"] += 1
        else:
            counts[acquired.split("/", 1)[1]] += 1
    return counts


def _find_provider(search: Any, class_name: str) -> Any | None:
    """沿 provider 包装链按类名找一层（10-06：取以图搜视频的计数）。"""

    node, seen = getattr(search, "provider", None), 0
    while node is not None and seen < 12:
        if type(node).__name__ == class_name:
            return node
        node, seen = getattr(node, "provider", None) or getattr(node, "_provider", None) or getattr(node, "inner", None), seen + 1
    return None


def _find_video_router(search: Any) -> Any | None:
    """沿 provider 包装链（守卫 / 过滤 / 录制都有 .provider）找到视频路由，取它的计数。"""

    from ..search.providers.video_fallback import VideoSearchFallbackProvider

    node, seen = getattr(search, "provider", None), 0
    while node is not None and seen < 12:
        if isinstance(node, VideoSearchFallbackProvider):
            return node
        node, seen = getattr(node, "provider", None) or getattr(node, "_provider", None) or getattr(node, "inner", None), seen + 1
    return None


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
    if model:
        from .teacher import parse_role_models

        try:
            names = sorted(set(parse_role_models(model).values()))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        absent = [name for name in names if served and name not in served]
        if absent:
            raise SystemExit(f"教师服务 {endpoint} 上没有模型 {absent}；现有 {served}"
                             "（学生 LoRA 要先挂上：bash server/12_add_lora.sh <名字> <目录>）")
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
                "--harness-params", args.harness_params, "--inspect-ops", getattr(args, "inspect_ops", "auto")]
    if getattr(args, "instruction_mode", "teacher") != "teacher":
        command += ["--instruction-mode", args.instruction_mode]
    if getattr(args, "query_guard", "on") != "on":
        command += ["--query-guard", args.query_guard]
    if getattr(args, "video_search", "ytdlp") != "ytdlp":
        command += ["--video-search", args.video_search]
    if getattr(args, "runtime_guards", "off") != "off":
        command += ["--runtime-guards", args.runtime_guards]
    if getattr(args, "state_renderer", None):
        command += ["--state-renderer", args.state_renderer]
    if getattr(args, "max_rounds", None) is not None:
        command += ["--max-rounds", str(args.max_rounds)]
    if getattr(args, "search_arm", None):
        command += ["--search-arm", args.search_arm]
    if getattr(args, "harness_override", None):
        command += ["--harness-override", *args.harness_override]
    if getattr(args, "detect", "off") != "off":
        command += ["--detect", args.detect, "--detect-config", str(args.detect_config)]
    if getattr(args, "image_search", "auto") != "auto":
        command += ["--image-search", args.image_search]
    if getattr(args, "auto_tools", "off") != "off":
        command += ["--auto-tools", args.auto_tools]
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
               "inspect_ops": getattr(args, "inspect_ops", "auto"),
               "instruction_mode": getattr(args, "instruction_mode", "teacher"),
               "query_guard": getattr(args, "query_guard", "on"),
               "video_search": getattr(args, "video_search", "ytdlp"),
               "runtime_guards": getattr(args, "runtime_guards", "off"),
               "state_renderer": getattr(args, "state_renderer", None), "max_rounds": getattr(args, "max_rounds", None),
               "search_arm": getattr(args, "search_arm", None), "detect": getattr(args, "detect", "off"),
               "image_search": getattr(args, "image_search", "auto"),
               "auto_tools": getattr(args, "auto_tools", "off"),
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
        overrides = list(getattr(args, "harness_override", None) or [])
        if getattr(args, "state_renderer", None):
            overrides.append(f"state_renderer={args.state_renderer}")
        settings = harness_settings_for(load_config(), args.specs, args.hi_spec, args.gia_spec, mode=args.harness_params,
                                        overrides=overrides)
    except (HarnessParamError, SpecError, OSError) as exc:
        raise SystemExit(f"规格 harness_params 不可用：{exc}") from exc
    verb = "覆盖" if settings.mode == "apply" else "只记录、不覆盖"
    if settings.differences:
        print(f"harness_params（{verb}）与默认配置不同的项：", flush=True)
        for key, item in settings.differences.items():
            print(f"  {key}: {item['default']!r} → {item['spec']!r}（{item['target']}，来自 {item['source']}）", flush=True)
    else:
        print(f"harness_params（{verb}）与默认配置一致", flush=True)


def _report_inspect_ops(args: argparse.Namespace) -> None:
    """开跑前打印一次本批会开哪些 INSPECT 操作、哪些模型操作没开及原因（子进程按同样规则再解析一次）。"""

    from ..config import load_config

    operations, detail = resolve_inspect_operations(args.inspect_ops, load_config().model_root)
    print(f"INSPECT 操作（{args.inspect_ops}）：{', '.join(operations)}", flush=True)
    for operation, item in detail.get("model_ops", {}).items():
        if not item["enabled"]:
            print(f"  未开 {operation}：{item['reason']}", flush=True)


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
    parser.add_argument("--model", default=None, help="服务上的模型名；按角色分开写 HI=hi_r1,GIA=gia_r1（学生两套 LoRA）")
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
    parser.add_argument("--inspect-ops", default="auto", choices=list(INSPECT_OPS_CHOICES),
                        help="INSPECT 操作集：auto（默认）= 基础五项 + 本机有权重且装了包的模型操作（语音转写 / 按文字找帧 / 图内定位）"
                             "；base = 只开基础五项（与 r0c 及以前可比）")
    parser.add_argument("--instruction-mode", default="teacher", choices=list(INSTRUCTION_MODES),
                        help="teacher（默认）= 学生提示 + 规格里的教师指令；student = 只给学生提示（未训练底座的基线用）")
    parser.add_argument("--video-search", default="ytdlp", choices=list(VIDEO_SEARCH_MODES),
                        help="文字搜视频：ytdlp（默认）= 先查 YouTube（yt-dlp），失败或空结果再用 DDGS；ddgs = 只用 DDGS（10-05 以前）")
    parser.add_argument("--query-guard", default="on", choices=list(QUERY_GUARD_MODES),
                        help="on（默认）= SEARCH 拒收含本次调查内部编号（E1/R3…）或\"材料A\"等代称的查询并提示改写；off = 不检查（10-05 以前）")
    parser.add_argument("--runtime-guards", default="off", choices=list(RUNTIME_GUARD_MODES),
                        help="off（默认）= 原行为；eval = 评测兜底（EVID 基线）：FINISH 不引 E 第 3 次接受并打标、工作区撞上限"
                             "自动 INCOMPLETE 报告、重复动作短路、GIA 耗尽时用草稿收尾、GIA 材料帧均匀抽、教师 JSON 模式；"
                             "eval_plus = 再加：GIA/HI 用完预算时只给 FINISH/报告、GIA 看得到任务数上限、搜索结果给出打开为视频与对齐的调用"
                             "（改模型输入）")
    parser.add_argument("--state-renderer", default=None, choices=["legacy", "compact_8k", "compact_16k"],
                        help="评测档强制的输入渲染档位（优先于规格 harness_params.state_renderer，记进 run_config.runtime_guards."
                             "state_renderer_override）；给了它时端点教师也按档位渲染输入。EVID 启动脚本默认 compact_16k")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="覆盖配置 [runtime].max_rounds（记进 run_config.runtime_guards.max_rounds_override）")
    parser.add_argument("--search-arm", default=None,
                        help="10-06 找原片实验的检索臂（逗号分隔，search/arms.py）：results10 视频每条 10 个 R / guardlen 视频查询"
                             "限长与去时刻戳并说明零结果 / imageclip 本机以图搜视频（CLIP 缩略图重排）/ fanout5 程序扇出 5 条改写"
                             "（只作消融）。不给 = A0，与以前完全相同")
    parser.add_argument("--detect", default="off", choices=list(DETECT_MODES),
                        help="已弃用（10-12 DETECT 并入 INSPECT / SEARCH）：on = --harness-override inspect_merged_catalog=true；"
                             "请改用 --hi-spec hi@0.8.0 --gia-spec gia@0.9.0")
    parser.add_argument("--detect-config", default=str(DEFAULT_DETECT_CONFIG), help="已弃用，不再读取")
    parser.add_argument("--image-search", default="auto",
                        help="以图搜图（10-12 起默认开，多引擎同时搜、结果合并）：auto = 有 key 的都开（Google Cloud Vision："
                             "GOOGLE_VISION_API_KEY 或 .local/google_vision_key.txt；Google Lens：SERPAPI_API_KEY / SERPER_API_KEY 或 "
                             ".local/serpapi_key.txt、serper_key.txt；TinEye：TINEYE_API_KEY 或 .local/tineye_key.txt）+ Yandex（免 key）；"
                             "all = 四个都要；也可逗号列出 gcv,lens,tineye,yandex；off = 10-12 以前（不开放 image 输入）。"
                             "会把调查里的图片发给这些搜索引擎")
    parser.add_argument("--auto-tools", default="off", choices=list(AUTO_TOOLS_MODES),
                        help="10-14 评测协议（只用于合并目录规格）：on = SEARCH 的画面比对判为原片时自动跑 compare_media、"
                             "登记材料后自动跑 video_structure（恢复 DETECT 的自动流程，改模型输入）；off（默认）= HI 自己决定")
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
                                      harness_params=args.harness_params, harness_overrides=args.harness_override,
                                      inspect_ops=args.inspect_ops, instruction_mode=args.instruction_mode,
                                      query_guard=args.query_guard, video_search=args.video_search,
                                      runtime_guards=args.runtime_guards, state_renderer=args.state_renderer,
                                      max_rounds=args.max_rounds, search_arm=args.search_arm,
                                      detect=args.detect, detect_config=args.detect_config,
                                      image_search=args.image_search, auto_tools=args.auto_tools))
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
    _report_inspect_ops(args)
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
