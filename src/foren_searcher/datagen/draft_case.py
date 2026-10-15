"""案卷自动起草：从一个跑完的 run 目录起草 case.json，供人审核（设计决定 2026-09-24"第一次训练前必须完成"第 4 条）。

用法：
    python -m foren_searcher.datagen.draft_case --runs experiments/sft_data/runs/true_4124903_r1 \\
        --out experiments/sft_data/cases_draft [--verdicts ...] [--units ...] [--labels labels.json] [--cases experiments/sft_data/cases]

输出：<out>/<name>.json（能被 Case.load 加载的案卷草稿）+ <out>/<name>.review.md（人读的审核单：generated_from /
generated_at / needs_review / notes 逐条，末尾附原 JSON 块）。Case.from_dict 拒绝未知字段，所以 draft 块不写进案卷本体；
审核单是 .md，草稿目录可以直接 load_cases()。
<name> 默认 <dataset>_<source_id>；同一案例已有别的 run 起草的草稿（其 review.md 的 generated_from 不同）时不覆盖，
改写成 <dataset>_<source_id>.from_<run>.json，并在审核单里提示"已有另一 run 的草稿"——此时两份 case_id 相同，
load_cases() 会报重复，审核时须二选一（或合并）后删掉另一份。同一 run 重跑则原地覆盖。

起草规则（纯程序、不调模型；所有文字取自 run 或数据集标注，模板只补连接词与审核提示）：
0. 任务类型分两套：
   - 声明核查（claim）：TRUE 等有"核查以下声明：…"的数据集。
   - 媒体对照（media）：videosham（比较视频 A 与候选 B 的差异 / 找某人物可见区间），或任务里没有声明、却在比较 A/B 的 run。
     没有外部来源，不套"外部可靠来源"模板；source_policy 写 provided_media（run 没配搜索或 evidence_source_policy=provided_media）。
1. task_text / dataset / source_id / case_id / family_key / media 直接从 run_config 与 store 取：task_text 是 GIA 首个
   请求 global_view.task；media 是 acquired_by=HARNESS/ingest 的 E（ref=source_ref，path 相对输出目录指向 run 的 artifact）；
   claims 取 task_text 里"：/:"之后的声明原文；split 默认 dev_train。
   task_kind：claim 按数据集猜（true → claim_check …，猜不到记 notes）；media 类任务文字含"时间区间/起止/定位"→ temporal_locate，
   否则 video_diff（cases.TaskKind 里没有专门的"差异定位"值，4126 那种"找可见区间"取 temporal_locate，notes 注明）。
2. requirements（2–5 条，REQ 编号，must_cite=False 的排最后）：
   - claim：把 GIA 首轮 INVESTIGATE 的每个任务切成分句（，。；？：），声明原文先替换成占位符，去掉套话
     （必要时…/依据实际证据…/核查声明"…"/检查给定视频 这类只有动作没有对象的分句），按三类归并：
     外部记载（external）→ 媒体内容（media）→ 来源/日期（provenance）；分句本身归不了类时继承它所在任务的主类
     （任务整体文字的归类）。每类一条；来源/日期类 must_cite=False。
     首轮任务只复述了总任务（validation run 常见）时抽不出分句：按总任务指令（"检查给定媒体""搜索外部证据"）写
     "声明绑定模板"（中性模板 + 声明原文），notes 与 draft.requirement_origin 标 claim_template；都没有才用纯中性模板（neutral）。
   - media：分句来源 = 总任务文字（优先）+ 首轮任务里不含数字/E 编号的分句（含数字的多是上一轮的观察，不能当要求）。
     按五类归并：差异位置（location：位置/时间/区间/起止/对应关系）→ 画面区域（region）→ 声音（audio）→
     对含义的影响（meaning）→ 未能确定的内容（uncertain：未能确定/未检查/精度/不代表…，must_cite=False）。
     一个分句含两类（"说明差异所在位置和未能确定的内容"）时在 和/及/并 处拆开，后半补上前半的动词。
     总任务里归不了类的分句（"比较给定视频 A 与候选视频 B"）并到下一分句前面；首轮任务里的直接丢掉。每类最多 3 个分句。
   最终答案按引用括号（（E1）/(E1)/[E1]/【E1】，相邻括号 [E3][E6] 合并）切句，同样归类，只用来检查"答案回应了
   哪些要求"——答案句是结论，不改写成要求（避免把答案泄进要求）。
3. required_evidence：
   - 基线 report 判定先过 verify.calibrate.recalibrate（verifier@0.1.0 的原判有误报），再算"干净 Finding"：不在重算后的
     tainted_findings 里；(F,E) 对判 contradicted/rejected 且未被校准撤销的不算。
   - 第一级：被干净 Finding 引用且答案也引用的 E。第一级一条都没有时**来源降级**：改取答案引用的 E（审核单注明）。
   - 补齐：claim 类要至少 1 条文本来源 + 帧（run 抽过帧时）。缺文本来源 → 取答案引用的文本 E（降级）→ 干净 Finding
     引用的文本 E；缺帧 → 答案引用的帧（降级）→ 干净 Finding 引用的帧 → run 里任意帧（标 run_only）。media 类要至少 1 条派生证据，
     缺了同样逐级降级。每一项的来源级别写进 draft.trace[key].origin。
   - 有 source_url 的按域名+路径生成 url_pattern（query 去掉；YouTube 等把视频号参数保留、问号写成 [?]；X/Twitter 写成
     *<账号>/status/<编号公共前缀>* 以覆盖 twitter.com 与 x.com）；派生 E（帧、裁剪、对齐、媒体属性…）按 acquired_by 归成一项。
     key 按站点命名（x/twitter → <账号>_tweets，youtube → youtube_upload，get_frames → frames，crop → crops，
     align_media → alignment，diff_media → pixel_diff，media_profile → media_profile，其余取二级域名）。
   - supports：先按"答案里引用该 E 的句子"归类到 REQ；答案没引用的（补齐项）按类型给默认类（文本 → external，
     派生 → claim 的 media / media 类的 location）。再用 verdicts_units 里 plan 单元 extra.gap.judge_targets（工作区 W1-k ↔
     任务 Tk）交叉核对——judge 的 REQ 编号属于判定时所用的案卷，需 --cases 里有该案卷才能按类反查；两者有交集取交集，
     否则取答案句归类并写 notes。must_cite 默认 True（只支撑 must_cite=False 要求的项为 False）。
   - required_evidence_rule：每条 must_cite 要求的证据若多于一项写 any_one_of，否则 all。
4. checks（只在 claim 类写；媒体对照没有数据集标注时一条都不写）：
   label_expected 只取数据集标签（--labels 文件或 run_config 里的 label 字段），找不到就不写并记 notes——不从答案反推；
   entity_in_evidence 取答案里的地点类实体（…桥/广场/街/路…，不含任务原文已有的），且须在答案判定的已核实片段
   或某条 Finding 原文里出现；
   date_in_range 保守：取答案里出现最多的日期，只有它 (a) 出现在答案首段（结论段，是声明所指的事件日期而不是历史背景）且
   (b) 能在答案所引某条文本来源正文里找到（2016-07-14 / July 14, 2016 / 14 July 2016 / 14 juillet 2016 / 20160714 …）时才写 ±1 天；
   拿不准就不写，并在 notes 里说明是哪一条没满足。
5. 审核单 <name>.review.md：draft 块 {"generated_from", "generated_at", "notes": [...], "needs_review": true,
   "requirement_origin": {REQ: task|claim_template|neutral}, "trace": {key: {evidence_ids, origin, …}}}
   先逐条列成 Markdown，末尾附原 JSON 块（```json … ```，read_review() 读回）。
6. 不引入 run 里没出现过的事实：描述用 E 的 label 原文，实体/日期须在答案原文里，要求文字来自任务原文（模板只含套话与声明原文）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit

from .cases import Case, Check, CaseError, Media, RequiredEvidence, Requirement, SourcePolicy
from .runlog import RunLog, load_run

DEFAULT_VERDICTS = Path("experiments/sft_data/verdicts_r0_judged.jsonl")
DEFAULT_UNITS = Path("experiments/sft_data/verdicts_units_judged.jsonl")
DEFAULT_CASES = Path("experiments/sft_data/cases")

TASK_KINDS: dict[str, str] = {"true": "claim_check", "averimatec": "claim_check", "mmfakebench": "claim_check",
                              "newsclippings": "claim_check", "evid": "local_question", "evidbench": "local_question"}
MEDIA_DATASETS = frozenset({"videosham"})

# ---- 声明核查（claim）三类
CATEGORIES = ("external", "media", "provenance")
_CAT_PATTERNS: dict[str, re.Pattern[str]] = {
    "external": re.compile(r"记载|报道|官方|可靠来源|外部|通报|原因|起因|因果|背景|可靠证据|辟谣|误传|定性"),
    "provenance": re.compile(r"上传|日期|出处|拍摄|最早|发布|来源与|的来源|原始来源|来历|(视频|媒体|素材)来源|身份与来源|保管|归属"),
    "media": re.compile(r"视频本身|媒体本身|画面|图像|帧|图片|材料\s*[A-Z]|场景|显示|呈现|视听|可辨认"),
}
_CAT_ORDER = ("external", "provenance", "media")
_CAT_FALLBACK_TEXT = {       # 任务里抽不到时的中性要求（只含套话，不含事实）
    "external": "外部可靠来源对声明所述事件的记载",
    "media": "给定媒体实际显示了什么，能否支持声明",
    "provenance": "给定媒体的公开来源与日期",
}
_CLAIM_TEMPLATE_TEXT = {     # 声明绑定模板：中性模板 + 声明原文（首轮任务只复述总任务时用）
    "external": "外部可靠来源对声明“{claim}”所述内容的记载",
    "media": "给定媒体实际显示了什么，能否支持声明“{claim}”",
    "provenance": "给定媒体的公开来源与日期",
}
_BOILERPLATE = re.compile(r"必要时|依据实际|结论范围|不确定性|注意证据是否|^核查声明|^核查以下声明")
_GENERIC = re.compile(r"^(并|再)?(检查|搜索|查看|核查)[^，。]{0,8}(视频|媒体|证据|材料)$")
_CLAIM_PREFIX = re.compile(r"^[^“〔，。]{0,12}声明[：:]?$")
_LEADING = re.compile(r"^(以及|并且|并|同时|重点|另外|此外)+")

# ---- 媒体对照（media）五类
MEDIA_CATEGORIES = ("location", "region", "audio", "meaning", "uncertain")
_MEDIA_PATTERNS: dict[str, re.Pattern[str]] = {
    "location": re.compile(r"位置|时间|时刻|区间|起止|边界|对应关系|片段|时序|编排|定位|所在"),
    "region": re.compile(r"区域|部位|画面中的|[左右][上下侧缘]|底边|顶边|边缘|标题区"),
    "audio": re.compile(r"音频|音轨|声音|言语|音画|听"),
    "meaning": re.compile(r"含义|意义|理解|语境|传达|情境"),
    "uncertain": re.compile(r"未能确定|不确定|未检查|无法|精度|限制|不代表|不要把|不将|不能"),
}
_MEDIA_ORDER = ("uncertain", "audio", "region", "meaning", "location")    # 平局取靠前的
_MEDIA_FALLBACK_TEXT = {
    "location": "说明两份材料差异所在的位置",
    "uncertain": "说明未检查或未能确定的内容",
}
_MEDIA_BOILERPLATE = re.compile(r"^引用具体证据$|^依据实际|^直接(比较|对照)[^，。]{0,20}$")
_VERB = re.compile(r"^(说明|引用|核查|核验|调查|区分|明确|核实|比较|确定|给出|报告)")

_CITE = re.compile(r"[（(\[【]\s*(E\d+(?:\s*[、,，]\s*E\d+)*)\s*[)）\]】]")
_E = re.compile(r"E\d+")
_DATE_CN = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ENTITY = re.compile(r"[一-鿿]{1,5}?(?:大桥|桥|广场|大街|街|大道|路|机场|车站|港|岛|体育场|大学|医院)")
_ENTITY_LEAD = re.compile(r"^[上在的于从到和与及被将把即是为]+")
_ID_PARAMS = ("v", "id", "p", "story_fbid", "fbid")
_SOCIAL = {"x.com", "twitter.com", "mobile.twitter.com"}
_MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December")
_MONTHS_FR = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
              "novembre", "décembre")


@dataclass
class _Segment:
    text: str
    evidence_ids: list[str]
    category: str


@dataclass
class _Pick:
    key: str
    evidence_ids: list[str] = field(default_factory=list)
    url_pattern: str | None = None
    acquired_by: str | None = None
    description: str = ""
    categories: set[str] = field(default_factory=set)
    workspaces: set[str] = field(default_factory=set)
    origin: str = "clean∩answer"


# ------------------------------------------------------------------ 文本工具

def _classify_with(text: str, patterns: dict[str, re.Pattern[str]], order: tuple[str, ...]) -> str | None:
    bare = re.sub(r"“[^”]*”|「[^」]*」|\"[^\"]*\"", "", text)
    scores = {cat: len(patterns[cat].findall(bare)) for cat in order}
    best = max(scores.values())
    if best == 0:
        return None
    return next(cat for cat in order if scores[cat] == best)


def classify(text: str) -> str | None:
    """声明核查三类：按关键词命中数取最多的一类；平局按 external > provenance > media。

    引号里的文字（多是声明原文的转述）先去掉，不参与归类——"国庆日期间"里的"日期"曾把媒体类分句误判成来源/日期类。
    """

    return _classify_with(text, _CAT_PATTERNS, _CAT_ORDER)


def classify_media(text: str) -> str | None:
    """媒体对照五类（location/region/audio/meaning/uncertain）；平局按 uncertain > audio > region > meaning > location。"""

    return _classify_with(text, _MEDIA_PATTERNS, _MEDIA_ORDER)


def classify_media_answer(text: str) -> str | None:
    """答案句归类：答案几乎句句带"不能/无法"，所以不归到 uncertain（它 must_cite=False，也不需要证据）。"""

    return _classify_with(text, {k: v for k, v in _MEDIA_PATTERNS.items() if k != "uncertain"},
                          tuple(c for c in _MEDIA_ORDER if c != "uncertain"))


def split_clauses(text: str, seps: str = "，。；;,") -> list[str]:
    """按 seps（默认 ，。；;,）切分句，引号“…”/「…」/《…》/〔…〕内部不切。"""

    parts: list[str] = []
    buf = ""
    depth = 0
    for ch in text:
        if ch in "“「《〔":
            depth += 1
        elif ch in "”」》〕":
            depth = max(0, depth - 1)
        if depth == 0 and ch in seps:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def answer_segments(answer: str, classifier: Callable[[str], str | None] = classify,
                    default: str = "external") -> list[_Segment]:
    """最终答案按引用括号切句：每段 = 上一个引用括号之后到这个括号为止的文字 + 括号里的 E。

    括号认（）/()/[]/【】；相邻括号（"…姿态。[E3][E6][E10]"）之间没有文字时并进上一段。
    """

    segments: list[_Segment] = []
    last = 0
    for m in _CITE.finditer(answer):
        text = answer[last:m.start()].strip()
        ids = _E.findall(m.group(1))
        last = m.end()
        if not text and segments:
            segments[-1].evidence_ids.extend(e for e in ids if e not in segments[-1].evidence_ids)
            continue
        # 分类只看最后一句（一段可能以"结论：…"开头再接具体事实）
        sentences = [s for s in re.split(r"[。！？\n]", text) if s.strip()]
        cat = None
        for s in reversed(sentences):
            cat = classifier(s)
            if cat:
                break
        segments.append(_Segment(text=text, evidence_ids=ids, category=cat or default))
    return segments


def _normalize(url: str) -> tuple[str, str, list[tuple[str, str]]]:
    parts = urlsplit(url.strip())
    host = re.sub(r"^(www|mobile|m)\.", "", parts.netloc.lower())
    path = parts.path.rstrip("/")
    return host, path, parse_qsl(parts.query)


def _glob_escape(value: str) -> str:
    return re.sub(r"([*?\[\]])", r"[\1]", value)


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    return os.path.commonprefix(values)


def url_key_and_pattern(urls: list[str]) -> tuple[str, str]:
    """同一站点/账号的一组 URL → (key, url_pattern)。输入须同 key（调用方先按 site_key 分组）。"""

    host, path, query = _normalize(urls[0])
    if host in _SOCIAL:
        account = path.strip("/").split("/")[0]
        ids = [(_normalize(u)[1].strip("/").split("/") + ["", "", ""])[2] for u in urls]
        prefix = _common_prefix(ids) if len(ids) > 1 else ids[0]
        return f"{_slug(account)}_tweets", f"*{_glob_escape(account.lower())}/status/{_glob_escape(prefix)}*"
    keep = [(k, v) for k, v in query if k in _ID_PARAMS]
    base = host + _glob_escape(path.lower())
    if len(urls) > 1:
        base = _glob_escape(_common_prefix([_normalize(u)[0] + _normalize(u)[1].lower() for u in urls]))
        keep = []
    if keep:
        base += "[?]" + "&".join(f"{k}={_glob_escape(v)}" for k, v in keep)
    return site_key(urls[0]), base + "*"


def site_key(url: str) -> str:
    host, path, _ = _normalize(url)
    if host in _SOCIAL:
        return f"{_slug(path.strip('/').split('/')[0])}_tweets"
    labels = [x for x in host.split(".") if x]
    if "youtube" in labels or host == "youtu.be":
        return "youtube_upload"
    name = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "source")
    if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "gov", "net", "ac"}:
        name = labels[-3]
    return _slug(name)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "source"


def _derived_key(acquired_by: str) -> str:
    op = acquired_by.split("/")[-1]
    return {"get_frames": "frames", "crop": "crops", "ocr": "ocr_text", "get_pages": "pages",
            "align_media": "alignment", "diff_media": "pixel_diff"}.get(op, _slug(op))


def is_media_task(dataset: str, task_text: str, claims: list[str]) -> bool:
    """媒体对照任务：videosham，或没有声明、却在比较/对照视频 A 与 B 的任务。"""

    if dataset in MEDIA_DATASETS:
        return True
    return not claims and bool(re.search(r"比较|对照|可见而|区间", task_text)) and bool(re.search(r"\bA\b", task_text)) \
        and bool(re.search(r"\bB\b", task_text))


def media_task_kind(task_text: str) -> str:
    return "temporal_locate" if re.search(r"时间区间|起止|定位", task_text) else "video_diff"


# ------------------------------------------------------------------ 判定文件

def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _report_verdicts(rows: list[dict[str, Any]], run: str) -> dict[str, dict[str, Any]]:
    return {str(r["subject"].get("workspace_id")): r for r in rows
            if (r.get("subject") or {}).get("run") == run and r["subject"].get("workspace_id")
            and (r.get("unit") or "report") == "report"}


def calibrated_report(row: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]], list[str]]:
    """一条 report 判定过 recalibrate 后：(tainted Finding, 仍算反驳的 (F,E) 对, 被校准撤销污染的 Finding)。

    judgments 里的 contradicted/rejected 是逐对原判；被 R4 等规则撤销的反驳（calibration.applied[].dismissed）不再算坏对。
    """

    from .verify.calibrate import recalibrate
    from .verify.verdict import Verdict

    before = set(row.get("tainted_findings") or [])
    verdict = recalibrate(Verdict.from_dict(row))
    tainted = set(verdict.tainted_findings)
    dismissed: set[tuple[str, str]] = set()
    for item in (verdict.extra.get("calibration") or {}).get("applied") or []:
        gone = item.get("dismissed") or {}
        for eid in gone.get("evidence_ids") or []:
            dismissed.add((str(item.get("finding")), str(eid)))
    bad = {(str(j.get("finding_id")), str(j.get("evidence_id"))) for j in verdict.judgments
           if j.get("verdict") in ("contradicted", "rejected")} - dismissed
    return tainted, bad, sorted(before - tainted, key=lambda f: int(f[1:]) if f[1:].isdigit() else 0)


def _judge_targets(rows: list[dict[str, Any]], run: str) -> dict[int, dict[str, list[str]]]:
    """plan 单元的 judge_targets，按轮次索引：{round: {T1: [REQ…]}}。"""

    out: dict[int, dict[str, list[str]]] = {}
    for r in rows:
        subject = r.get("subject") or {}
        if subject.get("run") != run or r.get("unit") != "plan":
            continue
        gap = (r.get("extra") or {}).get("gap") or {}
        if gap.get("judge_targets"):
            out[int(subject.get("round") or 1)] = {k: list(v) for k, v in gap["judge_targets"].items()}
    return out


def _answer_errors(rows: list[dict[str, Any]], run: str) -> list[dict[str, Any]]:
    """answer 单元校准后（recalibrate，R4 等规则）仍剩的错误——判定文件存的是原判，直接读会把已撤销的反驳当成错。"""

    from .verify.calibrate import recalibrate
    from .verify.verdict import Verdict

    out: list[dict[str, Any]] = []
    for r in rows:
        if (r.get("subject") or {}).get("run") != run or r.get("unit") != "answer":
            continue
        out.extend(e.to_dict() for e in recalibrate(Verdict.from_dict(r)).errors)
    return out


def _load_labels(path: Path | None) -> dict[str, str]:
    """数据集标签文件：JSON 对象 {case_id: label} 或 JSONL（每行含 case_id 与 label）。"""

    if path is None or not Path(path).is_file():
        return {}
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("{") and "\n{" not in text:
        data = json.loads(text)
        return {str(k): str(v) for k, v in data.items()}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if "case_id" in row and "label" in row:
                labels[str(row["case_id"])] = str(row["label"])
    return labels


# ------------------------------------------------------------------ requirements

def _first_plan_tasks(log: RunLog) -> list[str] | None:
    plans = [d for d in log.decisions if d.role == "GIA" and d.kind == "plan"]
    if not plans:
        return None
    return [str(t.get("task") if isinstance(t, dict) else t) for t in (plans[0].output.get("tasks") or [])]


def _finish_requirements(by_cat: dict[str, list[str]], order: tuple[str, ...], must: Callable[[str], bool],
                         fill: list[tuple[str, str, str]], notes: list[str]
                         ) -> tuple[list[Requirement], dict[str, str], dict[str, str]]:
    """by_cat（类 → 分句）→ 要求；不足 2 条时按 fill（类, 文字, origin）补。返回 (要求, 类→REQ, REQ→origin)。"""

    reqs: list[Requirement] = []
    cat_to_req: dict[str, str] = {}
    origin: dict[str, str] = {}
    for cat in order:
        clauses = by_cat.get(cat) or []
        if not clauses:
            continue
        text = "；".join(clauses)
        if cat == "provenance":
            text += "（能确认则给出，不能确认须明确说明）"
        rid = f"REQ{len(reqs) + 1}"
        reqs.append(Requirement(id=rid, text=text, must_cite=must(cat)))
        cat_to_req[cat] = rid
        origin[rid] = "task"
    for cat, text, how in fill:
        if len(reqs) >= 2:
            break
        if cat in cat_to_req:
            continue
        rid = f"REQ{len(reqs) + 1}"
        reqs.append(Requirement(id=rid, text=text, must_cite=must(cat)))
        cat_to_req[cat] = rid
        origin[rid] = how
        what = "声明绑定模板（中性模板 + 声明原文）" if how == "claim_template" else "中性模板"
        notes.append(f"{rid} 是{what}（任务文字里抽不到 {cat} 类分句），需人工改写")
    # 统一重新编号，让 must_cite=False 的排在最后
    ordered = sorted(reqs, key=lambda r: (not r.must_cite, int(r.id[3:])))
    renum = {r.id: f"REQ{i + 1}" for i, r in enumerate(ordered)}
    reqs = [Requirement(id=renum[r.id], text=r.text, must_cite=r.must_cite) for r in ordered]
    return reqs, {c: renum[r] for c, r in cat_to_req.items()}, {renum[r]: o for r, o in origin.items()}


def _claim_clauses(task: str, claim: str | None) -> tuple[list[str], str | None]:
    """一个首轮任务 → (去掉套话后的分句, 任务主类)。"""

    text = task
    if claim:
        for variant in (f"“{claim}”", f"\"{claim}\"", claim):
            text = text.replace(variant, "〔声明〕")
    kept: list[str] = []
    signal: list[str] = []          # 任务主类的依据：非套话分句，含"通过外部证据核查声明"这类只有动作的前缀
    for clause in split_clauses(text, "，。；;？?！：:"):
        if _BOILERPLATE.search(clause) and "〔声明〕" not in clause:
            continue
        clause = re.sub(r"^〔声明〕\s*", "", clause).strip()
        signal.append(clause)
        if not clause or _CLAIM_PREFIX.match(clause) or re.match(r"^核查(以下)?声明", clause):
            continue
        clause = clause.replace("〔声明〕", "声明")
        clause = _LEADING.sub("", clause)
        clause = re.sub(r"^判断其", "判断给定媒体", clause)
        clause = re.sub(r"^(重点)?查证", "", clause)
        if clause and not _GENERIC.match(clause) and not _BOILERPLATE.search(clause):
            kept.append(clause)
    task_cat = classify("，".join(signal)) or classify(task)
    return kept, task_cat


def _claim_requirements(log: RunLog, task_text: str, claim: str | None, notes: list[str]
                        ) -> tuple[list[Requirement], dict[str, str], dict[str, str], dict[str, str]]:
    """返回 (requirements, 类别→REQ id, 任务 T 编号→主类别串, REQ→origin)。"""

    tasks = _first_plan_tasks(log)
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    task_cats: dict[str, str] = {}
    if tasks is None:
        notes.append("run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写")
    for i, task in enumerate(tasks or []):
        clauses, task_cat = _claim_clauses(task, claim)
        cats = set()
        for clause in clauses:
            cat = classify(clause) or task_cat
            if cat is None:
                continue
            cats.add(cat)
            if clause not in by_cat[cat]:
                by_cat[cat].append(clause)
        task_cats[f"T{i + 1}"] = ",".join(sorted(cats))
    # 首轮任务只复述总任务：按总任务的指令（检查给定媒体 / 搜索外部证据）写声明绑定模板
    wants = [c for c, pat in (("external", r"外部|搜索"), ("media", r"媒体|视频|画面")) if re.search(pat, task_text)]
    fill: list[tuple[str, str, str]] = []
    for cat in wants + [c for c in CATEGORIES if c not in wants]:
        if claim and cat != "provenance":
            fill.append((cat, _CLAIM_TEMPLATE_TEXT[cat].format(claim=claim), "claim_template"))
        else:
            fill.append((cat, _CAT_FALLBACK_TEXT[cat], "neutral"))
    if not any(by_cat.values()) and tasks:
        notes.append("首轮 GIA 任务只复述了总任务（没有具体分句），requirements 只能按总任务指令写模板")
    reqs, cat_to_req, origin = _finish_requirements(by_cat, CATEGORIES, lambda c: c != "provenance", fill, notes)
    return reqs, cat_to_req, task_cats, origin


def _subsplit_media(clause: str) -> list[str]:
    """一个分句含两类要求时在 和/及/并/以及 处拆开；后半没有动词时补上前半的动词。"""

    parts = [p.strip() for p in re.split(r"以及|和|及|并", clause) if p.strip()]
    if len(parts) < 2 or any(len(p) < 3 for p in parts):
        return [clause]
    cats = [classify_media(p) for p in parts]
    if None in cats or len(set(cats)) < 2:
        return [clause]
    verb = _VERB.match(parts[0])
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p if _VERB.match(p) or not verb else verb.group(1) + p)
    return out


def _media_requirements(log: RunLog, task_text: str, notes: list[str]
                        ) -> tuple[list[Requirement], dict[str, str], dict[str, str], dict[str, str]]:
    by_cat: dict[str, list[str]] = {c: [] for c in MEDIA_CATEGORIES}
    task_cats: dict[str, str] = {}

    def add(clause: str, *, from_task: bool, pending: list[str]) -> str | None:
        found: str | None = None
        for piece in _subsplit_media(clause):
            cat = classify_media(piece)
            if cat is None:
                continue
            piece = _LEADING.sub("", piece)
            if pending:
                piece = "，".join(pending + [piece])
                pending.clear()
            found = found or cat
            if any(piece in old or old in piece for old in by_cat[cat]):
                continue                    # 与已有分句重复（互为子串）
            if from_task or len(by_cat[cat]) < 3:
                by_cat[cat].append(piece)
        return found

    pending: list[str] = []
    for clause in split_clauses(task_text, "，。；;？?！"):
        if _MEDIA_BOILERPLATE.search(clause):
            continue
        if add(clause, from_task=True, pending=pending) is None:
            pending.append(clause)          # 总任务里归不了类的分句并到下一分句前面
    tasks = _first_plan_tasks(log)
    if tasks is None:
        notes.append("run 里没有 GIA INVESTIGATE 决策，要求只取自总任务文字")
    skipped = 0
    for i, task in enumerate(tasks or []):
        cats = set()
        for clause in split_clauses(task, "，。；;？?！"):
            if _MEDIA_BOILERPLATE.search(clause):
                continue
            if re.search(r"\d|E\d+", clause):
                skipped += 1                 # 含数字/E 编号：多是上一轮的观察，不能当要求
                continue
            cat = add(clause, from_task=False, pending=[])
            if cat:
                cats.add(cat)
        task_cats[f"T{i + 1}"] = ",".join(sorted(cats))
    if skipped:
        notes.append(f"首轮 GIA 任务里 {skipped} 个分句含时间/编号（上一轮观察），没有写进 requirements")
    fill = [(c, t, "neutral") for c, t in _MEDIA_FALLBACK_TEXT.items()]
    reqs, cat_to_req, origin = _finish_requirements(by_cat, MEDIA_CATEGORIES, lambda c: c != "uncertain", fill, notes)
    return reqs, cat_to_req, task_cats, origin


# ------------------------------------------------------------------ required_evidence

def _clean_citations(log: RunLog, reports: dict[str, dict[str, Any]], notes: list[str]) -> dict[str, set[str]]:
    """E → 以干净 Finding 引用它的工作区集合（基线判定先 recalibrate）。"""

    cited: dict[str, set[str]] = {}
    for report in log.store.workspace_report_records():
        row = reports.get(report.task_id)
        if row is None:
            notes.append(f"{report.task_id} 没有报告判定，按全部 Finding 干净处理")
            tainted: set[str] = set()
            bad: set[tuple[str, str]] = set()
        else:
            tainted, bad, lifted = calibrated_report(row)
            if lifted:
                notes.append(f"{report.task_id}：recalibrate 撤销了 {'、'.join(lifted)} 的污染（verifier@0.1.0 原判误报）")
        for finding in report.findings:
            if finding.finding_id in tainted:
                continue
            for eid in finding.evidence_ids:
                if (finding.finding_id, eid) not in bad:
                    cited.setdefault(eid, set()).add(report.task_id)
    return cited


def _finding_workspaces(log: RunLog) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for report in log.store.workspace_report_records():
        for finding in report.findings:
            for eid in finding.evidence_ids:
                out.setdefault(eid, set()).add(report.task_id)
    return out


def _kind(log: RunLog, eid: str) -> str:
    """text（有 source_url）| frames（抽帧，含 align_media 抽出的帧不算）| derived | other。"""

    try:
        text = log.lookup.get(eid)
        ev = log.store.evidence(eid)
    except Exception:
        return "other"
    if text.source_url:
        return "text"
    if ev.derived_from and ev.acquired_by:
        return "frames" if ev.acquired_by.startswith("INSPECT/get_frames") else "derived"
    return "other"


def _choose_evidence(log: RunLog, answer_ids: list[str], clean: dict[str, set[str]], *, media_task: bool,
                     notes: list[str]) -> dict[str, str]:
    """挑证据：E → origin（clean∩answer / answer_only / clean_finding_only / run_only）。"""

    chosen: dict[str, str] = {e: "clean∩answer" for e in answer_ids if e in clean}
    if answer_ids and not chosen:
        chosen = {e: "answer_only" for e in answer_ids}
        notes.insert(0, "来源降级：答案所引的 E 没有一条被干净 Finding 引用（recalibrate 后相关 Finding 仍全部污染），"
                        f"required_evidence 改取答案引用的 E（{'、'.join(answer_ids)}）；这些 E 只经过被判有问题的 Finding，审核时逐条核对")
    all_ids = [ev.evidence_id for ev in log.store.evidence_records()]

    def top_up(kind: str, label: str) -> None:
        if any(_kind(log, e) == kind for e in chosen):
            return
        from_answer = [e for e in answer_ids if _kind(log, e) == kind]
        if from_answer:
            chosen.update({e: "answer_only" for e in from_answer})
            notes.append(f"来源降级：{label}只有答案引用、没有干净 Finding 引用，取答案引用的 {'、'.join(from_answer)}")
            return
        from_clean = [e for e in clean if _kind(log, e) == kind]
        if from_clean:
            chosen.update({e: "clean_finding_only" for e in from_clean})
            notes.append(f"答案没有引用{label}，取干净 Finding 引用的 {'、'.join(from_clean)}（答案未引用，审核时确认是否必需）")
            return
        from_run = [e for e in all_ids if _kind(log, e) == kind]
        if from_run and kind == "frames":
            chosen.update({e: "run_only" for e in from_run[:2]})
            notes.append(f"run 抽过帧，但答案与干净 Finding 都没引用帧；按 acquired_by 写 frames 一项（示例 {'、'.join(from_run[:2])}），审核时确认")
        elif kind == "text":
            notes.append("run 里没有可用的文本来源（没有带 source_url 的 E 被答案或干净 Finding 引用"
                         f"{'，run 也没有做过 SEARCH/open' if not from_run else ''}），required_evidence 缺文本来源项，审核时补或确认本案不需要")

    if media_task:
        if not any(_kind(log, e) in ("frames", "derived") for e in chosen):
            derived = [e for e in answer_ids if _kind(log, e) in ("frames", "derived")]
            derived = derived or [e for e in clean if _kind(log, e) in ("frames", "derived")]
            chosen.update({e: "answer_only" if e in answer_ids else "clean_finding_only" for e in derived})
            if derived:
                notes.append(f"来源降级：派生证据改取 {'、'.join(derived)}")
    else:
        top_up("text", "文本来源")
        if any(_kind(log, e) == "frames" for e in all_ids):
            top_up("frames", "帧")
    return chosen


# ------------------------------------------------------------------ checks

def _media(log: RunLog, out_dir: Path, notes: list[str]) -> list[Media]:
    media: list[Media] = []
    for ev in log.store.evidence_records():
        if ev.acquired_by != "HARNESS/ingest" or not ev.source_ref or not ev.artifact_ref:
            continue
        try:
            meta = log.store.artifact(ev.artifact_ref)
            target, media_type = meta.path.resolve(), meta.media_type
        except Exception:
            # 文件没随仓库入库（或校验失败）：仍从 store 的 artifacts 表取登记路径与类型，写进 notes 让人补文件
            row = _artifact_row(log, ev.artifact_ref)
            if row is None:
                notes.append(f"media {ev.source_ref}（{ev.evidence_id}）的 artifact {ev.artifact_ref} 在 store 里查不到，未写入 media")
                continue
            target, media_type = (Path(log.run_dir) / "artifacts" / row[0]).resolve(), row[1]
            notes.append(f"media {ev.source_ref}（{ev.evidence_id}）的文件 {row[0]} 不在 run/artifacts 里（未入库），路径按 store 登记写入")
        path = Path(os.path.relpath(target, out_dir.resolve()))
        media.append(Media(ref=ev.source_ref, path=path.as_posix(), media_type=media_type))
    return media


def _artifact_row(log: RunLog, artifact_ref: str) -> tuple[str, str] | None:
    m = re.match(r"^A(\d+)$", artifact_ref)
    db = Path(log.run_dir) / "store.sqlite3"
    if not m or not db.is_file():
        return None
    connection = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        row = connection.execute("SELECT relative_path, media_type FROM artifacts WHERE id=?", (int(m.group(1)),)).fetchone()
    finally:
        connection.close()
    return (str(row[0]), str(row[1])) if row else None


def _claim(task_text: str) -> list[str]:
    for sep in ("：", ":"):
        head, found, tail = task_text.rpartition(sep)
        if found and tail.strip() and ("声明" in head or "claim" in head.lower()):
            return [tail.strip()]
    return []


def _dates(answer: str) -> list[_dt.date]:
    out = []
    for m in list(_DATE_CN.finditer(answer)) + list(_DATE_ISO.finditer(answer)):
        try:
            out.append(_dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    return out


def date_in_text(day: _dt.date, text: str) -> bool:
    """来源正文里是否写着这一天（ISO / 紧凑 / 中文 / 英文两种顺序 / 法文 / 日月年数字）。"""

    m_en, m_fr = _MONTHS_EN[day.month - 1], _MONTHS_FR[day.month - 1]
    d, mo, y = day.day, day.month, day.year
    patterns = [
        rf"{y}-{mo:02d}-{d:02d}", rf"\b{y}{mo:02d}{d:02d}\b", rf"{y}年0?{mo}月0?{d}日",
        rf"\b{m_en}\.?\s+0?{d}(st|nd|rd|th)?,?\s+{y}", rf"\b{m_en[:3]}\.?\s+0?{d},?\s+{y}",
        rf"\b0?{d}(st|nd|rd|th)?\s+{m_en}\.?,?\s+{y}", rf"\b0?{d}\s+{m_en[:3]}\.?\s+{y}",
        rf"\b0?{d}(er)?\s+{m_fr}\s+{y}", rf"\b0?{d}/0?{mo}/{y}\b", rf"\b0?{mo}/0?{d}/{y}\b",
    ]
    return any(re.search(p, text, flags=re.I) for p in patterns)


def date_check(answer: str, sources: dict[str, str], notes: list[str]) -> Check | None:
    """保守的 date_in_range：答案里出现最多的日期须 (a) 在答案首段（结论段）且 (b) 见于所引文本来源正文，才写 ±1 天。

    sources：答案所引的文本 E → 正文。拿不准就不写，并往 notes 里写原因。
    """

    dates = _dates(answer)
    if not dates:
        return None
    day = Counter(dates).most_common(1)[0][0]
    first = next((p for p in re.split(r"\n\s*\n|\n", answer) if p.strip()), "")
    if day not in _dates(first):
        notes.append(f"date_in_range 未写：答案日期 {day.isoformat()} 不在首段结论里，可能是历史背景而不是声明所指事件的日期（拿不准不写）")
        return None
    hits = [eid for eid, body in sources.items() if date_in_text(day, body)]
    if not hits:
        notes.append(f"date_in_range 未写：答案日期 {day.isoformat()} 在答案所引文本来源正文里找不到（拿不准不写）")
        return None
    return Check(kind="date_in_range", params={"min": (day - _dt.timedelta(days=1)).isoformat(),
                                               "max": (day + _dt.timedelta(days=1)).isoformat()},
                 note=f"答案首段结论里的事件日期 ±1 天；见于来源 {'、'.join(hits)} 正文")


def _entities(answer: str, finding_texts: list[str], task_text: str) -> list[str]:
    found: list[str] = []
    for m in _ENTITY.finditer(answer):
        name = _ENTITY_LEAD.sub("", m.group(0))
        if len(name) < 2 or name in found or name in task_text:
            continue
        if any(name in t for t in finding_texts):
            found.append(name)
    return found


# ------------------------------------------------------------------ 起草

def draft_case(run_dir: str | Path, *, verdicts_path: str | Path | None = DEFAULT_VERDICTS,
               units_path: str | Path | None = DEFAULT_UNITS, out_dir: str | Path,
               labels_path: str | Path | None = None, cases_dir: str | Path | None = DEFAULT_CASES,
               today: str | None = None) -> Path:
    """起草一份案卷，写到 out_dir/<dataset>_<source_id>.json（+ .review.md 审核单；已有别的 run 的草稿时写 .from_<run>.json），返回案卷路径。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = load_run(run_dir)
    try:
        return _draft(log, verdicts=Path(verdicts_path) if verdicts_path else None,
                      units=Path(units_path) if units_path else None, out=out,
                      labels=Path(labels_path) if labels_path else None,
                      cases_dir=Path(cases_dir) if cases_dir else None, today=today)
    finally:
        log.close()


def _draft(log: RunLog, *, verdicts: Path | None, units: Path | None, out: Path, labels: Path | None,
           cases_dir: Path | None, today: str | None) -> Path:
    notes: list[str] = []
    dataset = str(log.config.get("dataset") or "").lower()
    source_id = str(log.config.get("case") or "")
    if not dataset or not source_id:
        raise CaseError(f"{log.name}: run_config 缺 dataset/case，无法起草")
    case_id = f"{dataset}:{source_id}"
    gia = [d for d in log.decisions if d.role == "GIA"]
    task_text = str((gia[0].view.get("task") if gia else "") or log.config.get("task") or log.config.get("instruction") or "").strip()
    if not task_text:
        raise CaseError(f"{log.name}: 找不到任务文字（GIA global_view.task）")
    claims = _claim(task_text)
    media_task = is_media_task(dataset, task_text, claims)
    if media_task:
        task_kind = media_task_kind(task_text)
        if task_kind == "temporal_locate":
            notes.append("task_kind 取 temporal_locate：任务是找 A/B 差异（人物可见性）的时间区间；TaskKind 没有\"差异区间定位\"专值，"
                         "video_diff 也说得通，审核时二选一")
    else:
        guessed = TASK_KINDS.get(dataset)
        if guessed is None:
            notes.append(f"数据集 {dataset} 的 task_kind 猜不出，暂填 claim_check")
        task_kind = guessed or "claim_check"
    answers = [d for d in gia if d.kind == "answer"]
    answer = str(answers[-1].output.get("answer") or "") if answers else ""
    if not answer:
        notes.append("run 没有 FINISH 答案：required_evidence / checks 无从起草")

    if media_task:
        reqs, cat_to_req, task_cats, req_origin = _media_requirements(log, task_text, notes)
        segments = answer_segments(answer, classify_media_answer, "location")
        order: tuple[str, ...] = MEDIA_CATEGORIES
    else:
        reqs, cat_to_req, task_cats, req_origin = _claim_requirements(log, task_text, claims[0] if claims else None, notes)
        segments = answer_segments(answer)
        order = CATEGORIES
    if answer and not segments:
        notes.append("最终答案里没有认出任何引用括号（（E1）/[E1]/【E1】），答案句归类与 supports 无从起草")
    answered = {s.category for s in segments}
    must_ids = {r.id for r in reqs if r.must_cite}
    for cat, rid in cat_to_req.items():
        if cat not in answered and rid in must_ids:
            notes.append(f"{rid}（{cat}）在最终答案里没有带引用的句子回应")
    for seg in segments:
        if seg.category not in cat_to_req:
            notes.append(f"答案句（{seg.category}）没有对应的任务要求：{seg.text[:40]}…（{'、'.join(seg.evidence_ids)}）")

    reports = _report_verdicts(_load_jsonl(verdicts), log.name)
    if not reports:
        notes.append(f"判定文件里没有 {log.name} 的报告判定，干净 Finding 无从判断")
    clean = _clean_citations(log, reports, notes)

    # 答案里每个 E 的归类（一个 E 可被多句引用）
    e_cats: dict[str, set[str]] = {}
    for seg in segments:
        for eid in seg.evidence_ids:
            e_cats.setdefault(eid, set()).add(seg.category)
    answer_ids = list(dict.fromkeys(eid for seg in segments for eid in seg.evidence_ids))
    chosen = _choose_evidence(log, answer_ids, clean, media_task=media_task, notes=notes)
    dropped = [e for e in answer_ids if e not in chosen]
    if dropped:
        notes.append(f"答案引用但没有被干净 Finding 引用、未入选的 E：{'、'.join(dropped)}（判定为有问题的 Finding 才引用它们，审核时看要不要补）")
    any_ws = _finding_workspaces(log)

    # 归组：派生 E 按 acquired_by，外部 E 按站点
    picks: dict[str, _Pick] = {}
    urls: dict[str, list[str]] = {}
    rank = {"clean∩answer": 0, "clean_finding_only": 1, "answer_only": 2, "run_only": 3}
    for eid, how in chosen.items():
        text = log.lookup.get(eid)
        ev = log.store.evidence(eid)
        acquired = ev.acquired_by or ""
        if text.source_url:
            key = site_key(text.source_url)
            pick = picks.setdefault(key, _Pick(key=key, origin=how))
            urls.setdefault(key, [])
            if text.source_url not in urls[key]:
                urls[key].append(text.source_url)
            if not pick.description:
                label = text.label.strip()
                pick.description = (label[:120] + "…" if len(label) > 120 else label) or text.source_url
        elif ev.derived_from and acquired:
            op = "/".join(acquired.split("/")[:2])
            key = _derived_key(op)
            pick = picks.setdefault(key, _Pick(key=key, acquired_by=op, origin=how))
        else:
            notes.append(f"{eid} 既无 source_url 也不是派生证据（{acquired}），跳过")
            continue
        if rank[how] < rank[pick.origin]:
            pick.origin = how
        pick.evidence_ids.append(eid)
        cats = e_cats.get(eid, set())
        if not cats:     # 补齐项（答案没引用）：按类型给默认类
            cats = {"external" if text.source_url else "media"} if not media_task else {"location"}
        pick.categories |= cats
        pick.workspaces |= clean.get(eid) or any_ws.get(eid, set())
    for key, pick in picks.items():
        if key in urls:
            _, pick.url_pattern = url_key_and_pattern(urls[key])
        if pick.acquired_by and not pick.description:
            parents = sorted({p for e in pick.evidence_ids for p in log.store.evidence(e).derived_from})
            pick.description = f"从给定媒体 {'、'.join(parents)} 经 {pick.acquired_by} 取得的派生证据（{'、'.join(pick.evidence_ids)}）"
        else:
            pick.description = f"{pick.description}（{'、'.join(pick.evidence_ids)}）"

    # judge_targets 交叉核对：judge 的 REQ 编号属于判定时的案卷，需要那份案卷的要求文字才能按类反查（只对 claim 三类）
    judge = _judge_targets(_load_jsonl(units), log.name).get(1, {})
    judge_cat: dict[str, set[str]] = {}
    ref_case_path = cases_dir / f"{dataset}_{source_id}.json" if cases_dir else None
    if judge and not media_task and ref_case_path is not None and ref_case_path.is_file():
        ref = Case.load(ref_case_path)
        ref_cat = {r.id: (classify(r.text) or "external") for r in ref.requirements}
        for task_id, targets in judge.items():
            judge_cat[f"W1-{task_id[1:]}"] = {ref_cat[t] for t in targets if t in ref_cat}
    elif judge:
        notes.append("有 judge_targets 但找不到判定时所用案卷，supports 只按答案句归类")
    else:
        notes.append("units 判定里没有首轮 judge_targets，supports 只按答案句归类")

    evidence: list[RequiredEvidence] = []
    trace: dict[str, Any] = {}
    for pick in picks.values():
        from_answer = {c for c in pick.categories if c in cat_to_req}
        if not from_answer:    # 归到的类没有对应要求：退到同类型的主要求
            fallback = [c for c in (("location",) if media_task else ("media", "external") if pick.acquired_by else ("external",))
                        if c in cat_to_req]
            from_answer = set(fallback[:1])
        from_judge: set[str] = set()
        for ws in pick.workspaces:
            from_judge |= judge_cat.get(ws, set())
        cats = (from_answer & from_judge) if (from_answer & from_judge) else from_answer
        if judge_cat and from_answer and not (from_answer & from_judge):
            notes.append(f"{pick.key}：答案句归类 {sorted(from_answer)} 与 judge_targets 反查 {sorted(from_judge & set(cat_to_req))} 不一致，按答案句")
        supports = tuple(sorted({cat_to_req[c] for c in cats}, key=lambda r: int(r[3:])))
        must = not supports or any(r.must_cite for r in reqs if r.id in supports)
        evidence.append(RequiredEvidence(key=pick.key, description=pick.description, must_cite=must, supports=supports,
                                         url_pattern=pick.url_pattern, acquired_by=pick.acquired_by))
        trace[pick.key] = {"evidence_ids": pick.evidence_ids, "origin": pick.origin, "workspaces": sorted(pick.workspaces),
                           "answer_categories": sorted(pick.categories), "judge_categories": sorted(from_judge)}
    evidence.sort(key=lambda r: (min((int(s[3:]) for s in r.supports), default=99), r.key))
    if not evidence:
        notes.append("required_evidence 为空：答案与干净 Finding 都没有可用的 E")

    groups: list[str] = []
    for req in reqs:
        if not req.must_cite:
            continue
        keys = [r.key for r in evidence if r.supports and r.supports[0] == req.id]
        if len(keys) > 1:
            groups.append("any_one_of:" + ",".join(keys))
        elif keys:
            groups.append("all:" + keys[0])
    rule = "; ".join(groups) if groups else "all"

    # checks
    checks: list[Check] = []
    unit_rows = _load_jsonl(units)
    dataset_label = _load_labels(labels).get(case_id) or log.config.get("label")
    if dataset_label:
        checks.append(Check(kind="label_expected", params={"label": str(dataset_label)}, note="数据集标签"))
    else:
        notes.append("没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推")
    if not media_task:
        # 实体须有旁证：答案判定里被来源原文逐句核实过（verified）的片段，或任一工作区 Finding 原文
        verified_parts = [str(part.get("part") or "") for r in unit_rows
                          if (r.get("subject") or {}).get("run") == log.name and r.get("unit") == "answer"
                          for j in r.get("judgments") or [] if j.get("verdict") == "supported"
                          for part in j.get("supported") or [] if part.get("verified")]
        finding_texts = [f.statement for r in log.store.workspace_report_records() for f in r.findings]
        for name in _entities(answer, verified_parts + finding_texts, task_text)[:2]:
            where = "答案判定已核实片段" if any(name in t for t in verified_parts) else "Finding 原文"
            checks.append(Check(kind="entity_in_evidence", params={"text": name},
                                note=f"答案中的地点实体（旁证：{where}）；所引 E 可能是外文原文，程序只查结论出现，需人工确认"))
        sources = {e: log.lookup.get(e).content for e in answer_ids if _kind(log, e) == "text"}
        check = date_check(answer, sources, notes)
        if check is not None:
            checks.append(check)
    elif _dates(answer):
        notes.append("媒体对照任务不写 date_in_range（答案里的日期不是事件日期）")
    for err in _answer_errors(unit_rows, log.name):
        if err.get("severity") == "critical":
            notes.append(f"答案判定有 critical 错误 {err.get('type')}：{err.get('detail')}（涉及 {'、'.join(err.get('evidence_ids') or [])}）——草稿的实体/日期来自这份答案，审核时核对")

    policy = SourcePolicy()
    if media_task:
        if log.config.get("evidence_source_policy") == "provided_media" or not log.config.get("search"):
            policy = SourcePolicy(mode="provided_media")
            notes.append("source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认")
    origin_note = "、".join(f"{k}={v}" for k, v in req_origin.items() if v != "task")
    case = Case(
        case_id=case_id, dataset=dataset, source_id=source_id, split="dev_train", task_kind=task_kind,  # type: ignore[arg-type]
        task_text=task_text, family_key=case_id, media=tuple(_media(log, out, notes)), claims=tuple(claims),
        source_policy=policy, requirements=tuple(reqs), required_evidence=tuple(evidence),
        required_evidence_rule=rule, checks=tuple(checks), reference_path=str(labels) if labels and dataset_label else None,
        notes=(f"草稿：由 draft_case 从 {log.name} 起草，needs_review；requirements 取自{'总任务与' if media_task else ''}首轮任务文字"
               f"{f'（模板：{origin_note}）' if origin_note else ''}，required_evidence 取自干净 Finding ∩ 答案引用"
               f"{'（有降级项，见审核单 trace.origin）' if any(p.origin != 'clean∩answer' for p in picks.values()) else ''}。"),
    )
    name = f"{dataset}_{source_id}"
    other = _existing_run(out, name)
    if (out / f"{name}.json").is_file() and other != log.name:
        notes.insert(0, f"已有另一 run 的草稿：{name}.json（generated_from={other or '未知'}）。两份 case_id 相同，"
                        f"load_cases() 会报重复，审核时二选一或合并后删掉另一份")
        name = f"{name}.from_{_slug(log.name)}"
    path = case.save(out / f"{name}.json")
    draft = {"generated_from": log.name, "generated_at": today or _dt.date.today().isoformat(),
             "notes": notes, "needs_review": True, "task_family": "media" if media_task else "claim",
             "requirement_origin": req_origin, "task_categories": task_cats, "trace": trace}
    (out / f"{name}.review.md").write_text(_render_review(name, draft), encoding="utf-8")
    return path


def _render_review(name: str, draft: dict[str, Any]) -> str:
    lines = [f"# 审核单：{name}.json", "",
             f"- generated_from: `{draft['generated_from']}`",
             f"- generated_at: {draft['generated_at']}",
             f"- needs_review: {'true' if draft['needs_review'] else 'false'}",
             f"- task_family: {draft.get('task_family', 'claim')}",
             f"- requirement_origin: {', '.join(f'{k}={v}' for k, v in (draft.get('requirement_origin') or {}).items())}", "",
             "## notes", ""]
    lines += [f"{i}. {n}" for i, n in enumerate(draft["notes"], 1)] or ["（无）"]
    lines += ["", "## 证据来源（trace）", ""]
    for key, item in draft["trace"].items():
        lines.append(f"- `{key}`：{'、'.join(item['evidence_ids'])}（{item.get('origin', 'clean∩answer')}；"
                     f"工作区 {'、'.join(item['workspaces']) or '—'}）")
    lines += ["", "## 原 JSON", "", "```json", json.dumps({"draft": draft}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def read_review(path: str | Path) -> dict[str, Any]:
    """读回审核单末尾的 JSON 块，返回 draft 字典。"""

    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"```json\n(.*?)\n```", text, flags=re.S)
    if not m:
        raise ValueError(f"{path} 里没有 JSON 块")
    return dict(json.loads(m.group(1))["draft"])


def _existing_run(out: Path, name: str) -> str | None:
    review = out / f"{name}.review.md"
    if not review.is_file():
        return None
    try:
        return str(read_review(review).get("generated_from") or "") or None
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 run 目录起草案卷（供人审核）")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verdicts", default=str(DEFAULT_VERDICTS))
    parser.add_argument("--units", default=str(DEFAULT_UNITS))
    parser.add_argument("--labels", default=None, help="数据集标签：JSON {case_id: label} 或 JSONL（case_id, label）")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="已有案卷目录（反查 judge_targets 的 REQ 编号用）")
    args = parser.parse_args(argv)
    for run in args.runs:
        path = draft_case(run, verdicts_path=args.verdicts, units_path=args.units, out_dir=args.out,
                          labels_path=args.labels, cases_dir=args.cases)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
