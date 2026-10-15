"""核验器精确率：四层自动估计 + 定向人工批次（2026-10-02 方案《核验精确率免人工方案》的实现）。

大白话：召回由注入实验覆盖（calibrate_recall）；这里回答"核验器说错的时候，有多少是真错"。四层：

- 层 A（answers，不花钱 / 少量调用）：答案单元的案卷标签（label_expected）。核验器对标签的判定（label_judged）与
  程序层子串匹配、第二评委的同题判定对照；三方不一致 → gold_conflict。
- 层 B（seeded）：构造"必然有据"的 (句, E) 对，看核验器误拒多少。B1 = 来源原句（程序抽取，quote_in_source 保证在原文里）；
  B2 = 改写模型把原句译写成中文结论句（与真实报告同分布：英文来源、中文结论），程序过滤：必须带回原句、原句回查得到、
  数字集合不变。期望 verdict=supported，误拒率 = (not_found ∪ contradicted) / 全部，是精确率的**上界**侧证据（分布偏干净）。
- 层 C（counterfactual）：只换 EvidenceText.content（或图片），不改 id、不加 payload 字段。
  C1 修复测试（文本对，verdict ∈ {not_found, contradicted} 且 rejected 为空）：程序先找与待核句最像的来源句；
  覆盖 ≥ 0.6 直接记 verbatim_but_unsupported（不调模型）；否则把来源换成"该句 ± 2 句"（≤ 1500 字）重判：
  仍判否 → repair_failed（"给了最相关的局部仍然判否"，送人工）；改判 supported → repair_fixed（原判是长文淹没）。
  C2 替换测试（supported 对抽样）：把来源换成另一个案例的同模态证据，仍判 supported → rubber_stamp。
  图片替换时去掉 media_note（程序记录的取帧时间，按规则可以单独支撑"第几秒"部分），只测像素。
- 层 D（agreement）：第二评委对同一批 Finding（含单元内全部 (句, E) 对 + 覆盖汇总）整条重判，Finding 级"是否有实锤错误"
  一致率与 κ；分歧 → disagree。扰动稳定性：文本来源前拼一段无关段落，Sonnet 整条重判，翻供 → unstable。

输出 precision_<tag>.json / .md：layers.{answers, seeded, counterfactual, agreement, stability} + hard_items[]。
hard_items 以"单元"（报告 Finding 或答案句）为粒度汇总各层原因；rubber_stamp 以 (单元, 替换后的证据) 为粒度单列。
`_gold_sample.py --from-hard` 按方案 §2 从这里抽人工批次。

所有模型调用经 CachedJudge（缓存键 = 系统提示 + payload + 图片字节）：重跑自动跳过已完成的调用（断点续跑），
replay 步骤用原 Sonnet 缓存逐对复现 r0c 的判定，证明重建的 (句, E) 与当时逐字相同。
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...store import GlobalStore
from ..errors import Severity
from .anchored import (JsonJudge, Judgment, _call_checked, _obj, _squash, _verify_finding, judge_pair, quote_in_source)
from .context import EvidenceText, StoreLookup
from .judges import CachedJudge, ClaudeCliJudge
from .label_judged import LABEL_SCHEMA, LABEL_SYSTEM_PROMPT
from .units import label_hit

FIRM = ("major", "critical")
WINDOW_CHARS = 1_500
VERBATIM_COVERAGE = 0.6
IRRELEVANT_PARAGRAPH = (
    "【附注：以下段落与本条来源无关，系版面拼接残留】本市图书馆将于下月起延长周末开放时间，周六、周日开放至晚上九点；"
    "读者可凭借阅证在自助机上办理续借，逾期图书每日收取少量滞纳金。馆内二层新设的儿童阅览区提供绘本与拼图，"
    "家长需全程陪同。停车场入口改至东侧辅路，原北门入口暂时封闭施工。\n\n")


# ---------------------------------------------------------------------------------------------------------------- 数据


@dataclass
class Unit:
    """核验的一个判断单元：报告里的一条 Finding，或答案里的一句。"""

    key: str                         # run|report:W1-1|F3  /  run|answer:request_0026|A4
    kind: str                        # report | answer
    run: str
    dataset: str
    case_id: str | None
    scope: str                       # 工作区 W1-1 / 答案 request
    finding_id: str                  # F3 / A4
    statement: str
    task: str | None
    texts: list[EvidenceText]        # 核验器当时判的那些 E（顺序同原 judgments）
    orig_judgments: list[dict[str, Any]]
    orig_errors: list[dict[str, Any]]

    @property
    def firm_types(self) -> list[str]:
        return [e["type"] for e in self.orig_errors if e.get("severity") in FIRM]

    @property
    def flagged(self) -> bool:
        return bool(self.firm_types)

    @property
    def has_image(self) -> bool:
        return any(t.is_image for t in self.texts)

    @property
    def has_video_frame(self) -> bool:
        return any(t.is_image and t.media_note for t in self.texts)

    def pair_key(self, evidence_id: str) -> str:
        return f"{self.key}|{evidence_id}"


@dataclass
class RunCtx:
    name: str
    config: dict[str, Any]
    store: GlobalStore
    lookup: StoreLookup
    scratch: Path

    @property
    def dataset(self) -> str:
        return str(self.config.get("dataset") or self.name.split("_")[0]).lower()

    @property
    def case_id(self) -> str | None:
        return f"{self.dataset}:{self.config.get('case')}" if self.config.get("case") else None

    def close(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def open_run(run_dir: Path) -> RunCtx:
    scratch = Path(tempfile.mkdtemp(prefix="calib_prec_"))
    shutil.copy2(run_dir / "store.sqlite3", scratch / "store.sqlite3")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").is_file() else {}
    artifacts = (run_dir / config["artifacts_from"]).resolve() if config.get("artifacts_from") else run_dir / "artifacts"
    store = GlobalStore(scratch / "store.sqlite3", artifacts)
    return RunCtx(run_dir.name, config, store, StoreLookup(store), scratch)


def _by_finding(rows: list[dict[str, Any]], field_name: str = "finding_id") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(field_name))].append(row)
    return out


def load_units(report_verdicts: str | Path, unit_verdicts: str | Path | None, runs_root: str | Path) -> tuple[list[Unit], dict[str, RunCtx]]:
    """从 verdict 行的 judgments[] 重建 (待核句, EvidenceText) 对：报告取 store 里的 Finding 原文，答案取 anchored.sentences。"""

    root = Path(runs_root)
    runs: dict[str, RunCtx] = {}

    def ctx(run: str) -> RunCtx:
        if run not in runs:
            runs[run] = open_run(root / run)
        return runs[run]

    units: list[Unit] = []
    for row in load_jsonl(report_verdicts):
        if row.get("unit") != "report":
            continue
        subj = row["subject"]
        c = ctx(subj["run"])
        ws = str(subj.get("workspace_id"))
        findings = {}
        task = ""
        for rep in c.store.workspace_report_records():
            if rep.task_id == ws:
                findings = {str(f.get("finding_id")): f for f in rep.to_dict().get("findings") or []}
                try:
                    task = str(c.store.workspace_task_record(ws).get("task") or "")
                except KeyError:
                    task = ""
        judged = _by_finding(row.get("judgments") or [])
        errs = _by_finding([{**e, "finding_id": (e.get("extra") or {}).get("finding")} for e in row.get("errors") or []])
        for fid, js in judged.items():
            f = findings.get(fid)
            if f is None:
                continue
            texts = [c.lookup.get(j["evidence_id"]) for j in js]
            units.append(Unit(key=f"{c.name}|report:{ws}|{fid}", kind="report", run=c.name, dataset=c.dataset, case_id=c.case_id,
                              scope=ws, finding_id=fid, statement=str(f.get("statement") or ""), task=task or None, texts=texts,
                              orig_judgments=js, orig_errors=errs.get(fid, [])))
    for row in load_jsonl(unit_verdicts) if unit_verdicts else []:
        if row.get("unit") != "answer":
            continue
        subj = row["subject"]
        c = ctx(subj["run"])
        sentences = {s["id"]: s for s in ((row.get("extra") or {}).get("anchored") or {}).get("sentences") or []}
        judged = _by_finding(row.get("judgments") or [])
        errs = _by_finding([{**e, "finding_id": (e.get("extra") or {}).get("sentence") or (e.get("extra") or {}).get("finding")}
                            for e in row.get("errors") or []])
        for sid, js in judged.items():
            s = sentences.get(sid)
            if s is None:
                continue
            texts = [c.lookup.get(j["evidence_id"]) for j in js]
            units.append(Unit(key=f"{c.name}|answer:{subj.get('request')}|{sid}", kind="answer", run=c.name, dataset=c.dataset,
                              case_id=c.case_id, scope=str(subj.get("request")), finding_id=sid, statement=str(s["text"]),
                              task=str(c.config.get("task") or "") or None, texts=texts, orig_judgments=js,
                              orig_errors=errs.get(sid, [])))
    return units, runs


# ------------------------------------------------------------------------------------------------------------ 评委


class CacheMiss(RuntimeError):
    status_code = 400          # anchored._retryable：400 不重试


class _NoCall:
    name = "replay-only"

    async def judge(self, **_: Any) -> dict[str, Any]:
        raise CacheMiss("replay: 缓存未命中")


def replay_judge(cache_path: str | Path) -> CachedJudge:
    return CachedJudge(_NoCall(), cache_path)   # type: ignore[arg-type]


def cli_judge(model: str, cache_path: str | Path, timeout: float = 600.0) -> CachedJudge:
    inner = ClaudeCliJudge(model=model, timeout=timeout)
    inner.resolve()
    return CachedJudge(inner, cache_path, progress=True)


def make_judge(kind: str, model: str, cache_path: str | Path, *, endpoint: str | None = None) -> JsonJudge:
    """claude-cli（默认，走订阅额度）或 local（本机 OpenAI 兼容服务，如 Ollama 上的 Qwen3.5-27B；缓存按模型分文件）。"""

    if kind == "claude-cli":
        return cli_judge(model, cache_path)
    from .judges import build_judge
    judge = build_judge(kind, model=model, cache_path=cache_path, endpoint=endpoint)
    if judge is None:
        raise SystemExit(f"评委 {kind}:{model} 建不起来")
    return judge


def _rejects(verdict: str) -> bool:
    return verdict in ("not_found", "contradicted")


# ------------------------------------------------------------------------------------------------------- 分句与定位

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=[.])\s+(?=[A-Z\"“(])|\n+")
_LATIN = re.compile(r"[A-Za-z][A-Za-z'’\-]{2,}")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "was", "were", "are", "has", "have", "had", "not", "but",
         "its", "his", "her", "their", "they", "you", "your", "our", "into", "about", "said", "says", "also", "been", "will"}


def split_sentences(content: str) -> list[tuple[int, int]]:
    """来源正文分句，返回 (起, 止) 字符区间（去掉首尾空白，空句丢弃）。"""

    spans, start = [], 0
    for m in _SENT_SPLIT.finditer(content):
        end = m.start() if m.group(0).strip() == "" else m.end()
        if m.start() == m.end():
            end = m.end()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(content):
        spans.append((start, len(content)))
    out = []
    for a, b in spans:
        seg = content[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        if b - trail > a + lead:
            out.append((a + lead, b - trail))
    return out


def char_coverage(statement: str, sentence: str) -> float:
    """待核句被这句来源覆盖的比例（与 quote_in_source 同口径：去空白、统一引号，只算 ≥4 字符的连续片段）。"""

    s, c = _squash(statement), _squash(sentence)
    if not s or not c:
        return 0.0
    m = difflib.SequenceMatcher(None, c, s, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks() if b.size >= 4) / len(s)


def anchor_tokens(statement: str) -> set[str]:
    """跨语言定位用的锚：数字、拉丁词（≥3 字母、去停用词）。中文结论句引英文来源时，人名地名数字通常原样保留。"""

    words = {w.lower().strip("'’-") for w in _LATIN.findall(statement)} - _STOP
    nums = {n.replace(",", "") for n in _NUM.findall(statement) if len(n.replace(",", "").replace(".", "")) >= 2}
    return words | nums


def token_score(anchors: set[str], sentence: str) -> float:
    if not anchors:
        return 0.0
    low = sentence.lower().replace(",", "")
    return sum(1 for a in anchors if a in low) / len(anchors)


@dataclass
class Located:
    index: int
    span: tuple[int, int]
    coverage: float                 # 字面覆盖
    tokens: float                   # 锚词命中率
    window: str

    @property
    def score(self) -> float:
        return max(self.coverage, self.tokens)


def locate(statement: str, content: str, *, context: int = 2, limit: int = WINDOW_CHARS) -> Located | None:
    spans = split_sentences(content)
    if not spans:
        return None
    anchors = anchor_tokens(statement)
    best: Located | None = None
    for i, (a, b) in enumerate(spans):
        sent = content[a:b]
        cov = char_coverage(statement, sent) if len(sent) < 4000 else 0.0
        tok = token_score(anchors, sent)
        cand = Located(i, (a, b), cov, tok, "")
        if best is None or (cand.score, cand.coverage) > (best.score, best.coverage):
            best = cand
    if best is None or best.score <= 0:
        return None
    lo, hi = max(0, best.index - context), min(len(spans), best.index + context + 1)
    window = content[spans[lo][0]:spans[hi - 1][1]]
    if len(window) > limit:
        a, b = best.span
        mid = (a + b) // 2
        start = max(0, min(mid - limit // 2, len(content) - limit))
        window = content[start:start + limit]
    best.window = window
    return best


def window_at(content: str, quote: str, *, context: int = 2, limit: int = WINDOW_CHARS, how: str = "quote") -> Located | None:
    """已知来源里的一段原文（反驳引语 / 定位模型摘出的句子）：以它所在的句子为中心取 ± context 句。"""

    q = quote.strip()
    if not q or not quote_in_source(q, content):
        return None
    spans = split_sentences(content)
    probe = q[:40]
    pos = content.find(probe)
    if pos < 0:
        best_i = max(range(len(spans)), key=lambda i: char_coverage(q, content[spans[i][0]:spans[i][1]]), default=None)
    else:
        best_i = next((i for i, (a, b) in enumerate(spans) if a <= pos < b), None)
    if best_i is None:
        return None
    lo, hi = max(0, best_i - context), min(len(spans), best_i + context + 1)
    window = content[spans[lo][0]:spans[hi - 1][1]]
    if len(window) > limit:
        a, b = spans[best_i]
        start = max(0, min((a + b) // 2 - limit // 2, len(content) - limit))
        window = content[start:start + limit]
    loc = Located(best_i, spans[best_i], 0.0, 0.0, window)
    loc.how = how      # type: ignore[attr-defined]
    return loc


LOCATE_SYSTEM = """给你一句调查结论（statement，可能是中文）和一篇来源正文（source，可能是英文）。
找出来源里与这句结论**最相关**的一句原文（讲同一件事、同一个数字或同一个人的那句），逐字照抄，不翻译、不改写。
只是定位，不判断它支不支持。来源里实在没有相关的句子时 sentence 写空串。输出 JSON：{"sentence":"..."}"""
LOCATE_SCHEMA = _obj({"sentence": {"type": "string"}})


async def model_locate(locator: JsonJudge, statement: str, content: str) -> Located | None:
    raw, _ = await _call_checked(locator, LOCATE_SYSTEM, {"statement": statement, "source": content[:60_000]}, [], LOCATE_SCHEMA)
    if not raw:
        return None
    return window_at(content, str(raw.get("sentence") or ""), how="model")


# -------------------------------------------------------------------------------------------------------------- 层 A


async def layer_answers(rows: list[dict[str, Any]], cases: dict[str, Any], runs: dict[str, RunCtx], second: JsonJudge | None,
                        say: Callable[[str], None]) -> dict[str, Any]:
    """答案单元的标签：核验器（Sonnet label_judged）× 程序层子串 × 第二评委同题。三方一致 = 准金标；不一致 → gold_conflict。"""

    items = []
    for row in rows:
        if row.get("unit") != "answer":
            continue
        subj = row["subject"]
        case = cases.get(subj.get("case_id") or "")
        checks = [c for c in (case.checks if case else ()) if c.kind == "label_expected"]
        if not checks:
            continue
        check = checks[0]
        label = str(check.params["label"])
        aliases = [str(a) for a in check.params.get("aliases") or []]
        sentences = ((row.get("extra") or {}).get("anchored") or {}).get("sentences") or []
        answer = "".join(s["text"] for s in sentences)
        verr = [e for e in row.get("errors") or [] if (e.get("extra") or {}).get("check") == "label_expected"]
        verifier = "matches"
        if verr:
            verifier = (verr[0].get("extra") or {}).get("label_judgment") or ("absent" if verr[0]["type"] == "omitted_required_fact" else "contradicts")
        program = "matches" if any(label_hit(f, answer) for f in [label, *aliases]) else "absent"
        second_status, second_notes = None, ""
        if second is not None:
            task = str(runs[subj["run"]].config.get("task") or case.task_text or "")
            raw, failure = await _call_checked(second, LABEL_SYSTEM_PROMPT,
                                               {"task": task[:4000], "answer": answer[:12_000], "label": label, "aliases": aliases},
                                               [], LABEL_SCHEMA)
            if raw:
                second_status, second_notes = raw.get("status"), str(raw.get("notes") or "")
        statuses = {verifier, second_status} - {None}
        items.append({"run": subj["run"], "request": subj.get("request"), "case_id": subj.get("case_id"), "label": label,
                      "verifier": verifier, "program_substring": program, "second": second_status, "second_notes": second_notes,
                      "verifier_bad": verifier != "matches", "conflict": len(statuses) > 1,
                      "conclusion": sentences[0]["text"] if sentences else ""})
        say(f"[A] {subj['run']} 标签 {label!r}：核验器 {verifier} / 子串 {program} / 第二评委 {second_status}")
    agree = [x for x in items if not x["conflict"] and x["second"] is not None]
    return {"n": len(items), "verifier_bad": sum(x["verifier_bad"] for x in items), "conflicts": sum(x["conflict"] for x in items),
            "agree_with_second": len(agree), "items": items,
            "note": "无人工金标时，'第二评委同判'只是准金标；conflict 条目进 hard_items（gold_conflict）。"}


# -------------------------------------------------------------------------------------------------------------- 层 B

_HAS_ENTITY = re.compile(r"\d|[A-Z][a-z]+")


def seed_sentences(ctx: RunCtx, *, per_run: int, rng: random.Random, min_chars: int = 30, max_chars: int = 400) -> list[tuple[EvidenceText, str]]:
    """B1：从 run 的文本证据（外部网页等非派生文本）抽完整句子，≥30 字、含数字或专名；程序保证 quote_in_source。"""

    pool: list[tuple[EvidenceText, str]] = []
    for rec in ctx.store.evidence_records():
        eid = str(rec.evidence_id)
        try:
            text = ctx.lookup.get(eid)
        except KeyError:
            continue
        if not text.is_text or text.derived_from or len(text.content) < 200:
            continue
        for a, b in split_sentences(text.content):
            sent = text.content[a:b].strip()
            if not (min_chars <= len(sent) <= max_chars) or not _HAS_ENTITY.search(sent):
                continue
            if sent.count("|") > 2 or sent.count("http") or sent.lower().startswith(("cookie", "subscribe", "sign up", "advertis")):
                continue
            if len(re.findall(r"[A-Za-z一-鿿]", sent)) < 0.6 * len(sent):
                continue
            light = re.sub(r"\s*[（(][^（）()]{0,40}[)）]", "", sent).strip().strip("\"“”")
            stmt = light if light != sent and quote_in_source(light, text.content) else sent
            if quote_in_source(stmt, text.content):
                pool.append((text, stmt))
    rng.shuffle(pool)
    out, per_e = [], Counter()
    for text, stmt in pool:
        if per_e[text.evidence_id] >= 3:
            continue
        out.append((text, stmt))
        per_e[text.evidence_id] += 1
        if len(out) >= per_run:
            break
    return out


PARAPHRASE_SYSTEM = """你把一句英文（或中文）来源原句改写成一句中文调查结论，像调查员在报告里写的那样。
硬性要求：只能改写和翻译，不得增加、删除或改变任何事实；数字、日期、人名、机构名必须保留（人名机构名可保留英文原文）；
不得加入“据报道”以外的评价或推断。输出 JSON：{"original":"逐字照抄给你的原句","statement":"改写后的中文结论句"}"""
PARAPHRASE_SCHEMA = _obj({"original": {"type": "string"}, "statement": {"type": "string"}})


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(s)}


async def paraphrase(writer: JsonJudge, text: EvidenceText, sentence: str) -> tuple[str | None, str]:
    """B2：改写 + 程序过滤（带回原句、原句回查得到、数字集合不变）。返回 (改写句或 None, 过滤原因)。"""

    raw, failure = await _call_checked(writer, PARAPHRASE_SYSTEM, {"sentence": sentence}, [], PARAPHRASE_SCHEMA)
    if raw is None:
        return None, f"call_failed: {failure.last_error}"
    original, stmt = str(raw.get("original") or ""), str(raw.get("statement") or "").strip()
    if not quote_in_source(original, text.content):
        return None, "original_not_in_source"
    if not stmt or len(stmt) < 8:
        return None, "empty"
    orig_nums, new_nums = _numbers(sentence), _numbers(stmt)
    extra = {x for x in new_nums - orig_nums if not (x.isdigit() and 1 <= int(x) <= 12)}   # 改写成中文日期常多出月份数
    if not orig_nums <= new_nums or extra:
        return None, "numbers_changed"
    return stmt, "ok"


async def layer_seeded(runs: dict[str, RunCtx], judge: JsonJudge, writer: JsonJudge | None, *, per_run: int, seed: int,
                       say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    jobs: list[tuple[str, RunCtx, EvidenceText, str, str]] = []
    filtered = Counter()
    picks: dict[str, list[tuple[EvidenceText, str]]] = {}
    for name in sorted(runs):
        picks[name] = seed_sentences(runs[name], per_run=per_run, rng=rng)
        for text, stmt in picks[name]:
            jobs.append(("B1", runs[name], text, stmt, stmt))
    if writer is not None:
        async def para(name: str, text: EvidenceText, stmt: str) -> tuple[str, EvidenceText, str, str | None, str]:
            out, why = await paraphrase(writer, text, stmt)
            return name, text, stmt, out, why
        results = await asyncio.gather(*(para(n, t, s) for n, ps in picks.items() for t, s in ps))
        for name, text, stmt, out, why in results:
            filtered[why] += 1
            if out:
                jobs.append(("B2", runs[name], text, out, stmt))

    async def one(tier: str, ctx: RunCtx, text: EvidenceText, stmt: str, origin: str) -> dict[str, Any]:
        j = await judge_pair(judge, "B", stmt, text)
        return {"tier": tier, "run": ctx.name, "dataset": ctx.dataset, "evidence_id": text.evidence_id, "statement": stmt,
                "source_sentence": origin, "content_chars": len(text.content), "verdict": j.verdict,
                "rejected": [p.to_dict() for p in j.rejected], "notes": j.notes}

    rows = list(await asyncio.gather(*(one(*job) for job in jobs)))
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in ("B1", "B2"):
        rs = [r for r in rows if r["tier"] == tier and r["verdict"] != "failed"]
        bad = [r for r in rs if _rejects(r["verdict"])]
        by_tier[tier] = {"n": len(rs), "false_reject": len(bad), "rate": round(len(bad) / len(rs), 3) if rs else None,
                         "failed": sum(1 for r in rows if r["tier"] == tier and r["verdict"] == "failed")}
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["verdict"] == "failed":
            continue
        b = "<10k" if r["content_chars"] < 10_000 else "10k-50k" if r["content_chars"] < 50_000 else "≥50k"
        buckets[f"{r['tier']} {b}"][0] += 1
        buckets[f"{r['tier']} {b}"][1] += _rejects(r["verdict"])
    usable = [r for r in rows if r["verdict"] != "failed"]
    total_bad = sum(_rejects(r["verdict"]) for r in usable)
    say(f"[B] B1 {by_tier['B1']} B2 {by_tier['B2']}；改写过滤 {dict(filtered)}")
    return ({"n": len(usable), "by_tier": by_tier, "by_length": {k: {"n": v[0], "false_reject": v[1]} for k, v in sorted(buckets.items())},
             "by_modality": {"text": {"n": len(usable), "false_reject": total_bad}},
             "false_reject_rate": round(total_bad / len(usable), 3) if usable else None, "paraphrase_filter": dict(filtered),
             "image_seeds": "未做：r0c 的 store 里没有 OCR 证据（图片转文字），按方案程序取整行文字当 statement 的前提不成立"},
            rows)


# -------------------------------------------------------------------------------------------------------------- 层 C


@dataclass
class PairRef:
    unit: Unit
    index: int

    @property
    def text(self) -> EvidenceText:
        return self.unit.texts[self.index]

    @property
    def orig(self) -> dict[str, Any]:
        return self.unit.orig_judgments[self.index]

    @property
    def key(self) -> str:
        return self.unit.pair_key(self.text.evidence_id)


def all_pairs(units: list[Unit]) -> list[PairRef]:
    return [PairRef(u, i) for u in units for i in range(len(u.texts))]


async def replay(pairs: list[PairRef], cache: JsonJudge) -> dict[str, Any]:
    """用原评委缓存逐对复现：命中且 verdict 相同 = 重建的 (句, E) 与当时逐字一致。"""

    async def one(p: PairRef) -> tuple[str, str]:
        j = await judge_pair(cache, p.unit.finding_id, p.unit.statement, p.text, task=p.unit.task)
        return j.verdict, p.orig["verdict"]

    out = await asyncio.gather(*(one(p) for p in pairs))
    hit = sum(1 for got, _ in out if got not in ("failed", "needs_pixels"))
    same = sum(1 for got, want in out if got == want)
    return {"pairs": len(pairs), "cache_hit": hit, "same_verdict": same,
            "mismatch": [p.key for p, (got, want) in zip(pairs, out) if got != "failed" and got != want][:20]}


async def layer_repair(pairs: list[PairRef], judge: JsonJudge | None, say: Callable[[str], None], *,
                       locator: JsonJudge | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """C1。定位顺序：程序（同语言字面覆盖 / 数字与拉丁词锚）→ 原判的反驳引语所在句 → 定位模型摘句（程序回查）。"""

    objs = [p for p in pairs if p.text.is_text and _rejects(p.orig["verdict"]) and not p.orig.get("rejected")]
    rows: list[dict[str, Any]] = []
    calls: list[tuple[PairRef, Located]] = []
    located: list[tuple[Located | None, str]] = []
    for p in objs:
        loc = locate(p.unit.statement, p.text.content)
        how = "program"
        if loc is not None and loc.coverage < VERBATIM_COVERAGE and loc.tokens < 0.5:
            loc = None                       # 锚太弱：不当定位
        if loc is None:
            for part in p.orig.get("contradicted") or []:
                loc = window_at(p.text.content, part.get("quote") or "", how="contradicted_quote")
                if loc is not None:
                    how = "contradicted_quote"
                    break
        located.append((loc, how))
    if locator is not None:
        todo = [i for i, (loc, _) in enumerate(located) if loc is None]
        found = await asyncio.gather(*(model_locate(locator, objs[i].unit.statement, objs[i].text.content) for i in todo))
        for i, loc in zip(todo, found):
            located[i] = (loc, "model" if loc is not None else located[i][1])
    for p, (loc, how) in zip(objs, located):
        row = {"key": p.key, "unit": p.unit.key, "evidence_id": p.text.evidence_id, "orig": p.orig["verdict"],
               "content_chars": len(p.text.content), "unit_flagged": p.unit.flagged, "located_by": how if loc else None}
        if loc is None:
            row.update(outcome="unlocatable")
        elif loc.coverage >= VERBATIM_COVERAGE:
            row.update(outcome="verbatim_but_unsupported", coverage=round(loc.coverage, 3), sentence=p.text.content[loc.span[0]:loc.span[1]][:400])
        elif _squash(loc.window) == _squash(p.text.content):
            row.update(outcome="no_shorter_window", window_chars=len(loc.window))
        else:
            row.update(coverage=round(loc.coverage, 3), tokens=round(loc.tokens, 3), window_chars=len(loc.window), outcome="pending")
            calls.append((p, loc))
        rows.append(row)
    if judge is not None and calls:
        async def one(p: PairRef, loc: Located) -> Judgment:
            return await judge_pair(judge, p.unit.finding_id, p.unit.statement, replace(p.text, content=loc.window), task=p.unit.task)
        results = await asyncio.gather(*(one(p, loc) for p, loc in calls))
        by_key = {r["key"]: r for r in rows}
        for (p, loc), j in zip(calls, results):
            r = by_key[p.key]
            r["repaired_verdict"] = j.verdict
            r["outcome"] = "failed" if j.verdict == "failed" else "repair_fixed" if j.verdict == "supported" else "repair_failed"
            r["window"] = loc.window[:600]
            r["repaired_supported"] = [x.to_dict() for x in j.supported]
    c = Counter(r["outcome"] for r in rows)
    n = len(rows)
    say(f"[C1] 文本否判对 {n}：{dict(c)}")
    fp_upper = c["verbatim_but_unsupported"] + c["repair_failed"]
    return ({"n": n, **{k: c[k] for k in ("verbatim_but_unsupported", "repair_failed", "repair_fixed", "unlocatable", "no_shorter_window",
                                          "pending", "failed")},
             "located_by": dict(Counter(r.get("located_by") for r in rows if r.get("located_by"))),
             "fp_rate_upper": round(fp_upper / n, 3) if n else None,
             "pair_fn_rate_repair_fixed": round(c["repair_fixed"] / n, 3) if n else None,
             "image_pairs_rejected": sum(1 for p in pairs if p.text.is_image and _rejects(p.orig["verdict"])),
             "note": "fp_rate_upper 把 repair_failed 全算作误报，是上界；repair_fixed = 原判在长文里没找到、给局部就找到（对级假阴性）。"
                     "图片否判对无法程序定位，交给层 D 与人工。"}, rows)


def _donor_pool(units: list[Unit]) -> tuple[list[tuple[str, EvidenceText]], list[tuple[str, EvidenceText]]]:
    texts, images, seen = [], [], set()
    for u in units:
        for t in u.texts:
            k = (u.case_id, t.evidence_id, u.run)
            if k in seen:
                continue
            seen.add(k)
            if t.is_image and t.image_path:
                images.append((u.case_id or u.run, t))
            elif t.is_text and len(t.content) >= 200:
                texts.append((u.case_id or u.run, t))
    return texts, images


def pick_donor(p: PairRef, texts: list[tuple[str, EvidenceText]], images: list[tuple[str, EvidenceText]], rng: random.Random) -> EvidenceText | None:
    case = p.unit.case_id or p.unit.run
    if p.text.is_image:
        cands = [t for c, t in images if c != case]
        if not cands:
            return None
        donor = rng.choice(cands)
        return replace(p.text, image_path=donor.image_path, media_note="")
    n = len(p.text.content)
    pool = [(c, t) for c, t in texts if c != case]
    if p.text.derived_from or str(p.text.acquired_by or "").startswith("INSPECT"):
        # 工具的结构化输出（对齐结果、媒体属性）换成同一工具的另一份输出不是有效反事实（两份空对齐结果说的是同一件事）
        pool = [(c, t) for c, t in pool if t.acquired_by != p.text.acquired_by] or pool
    cands = [t for c, t in pool if 0.5 <= len(t.content) / max(n, 1) <= 2.0] or [t for _, t in pool]
    if not cands:
        return None
    donor = min(rng.sample(cands, min(5, len(cands))), key=lambda t: abs(len(t.content) - n))
    return replace(p.text, content=donor.content)


def stratified_sample(pairs: list[PairRef], n: int, seed: int) -> list[PairRef]:
    groups: dict[tuple[str, str], list[PairRef]] = defaultdict(list)
    for p in sorted(pairs, key=lambda p: hashlib.sha1(f"{seed}|{p.key}".encode()).hexdigest()):
        groups[("image" if p.text.is_image else "text", p.unit.dataset)].append(p)
    out: list[PairRef] = []
    while len(out) < n and any(groups.values()):
        for g in sorted(groups):
            if groups[g] and len(out) < n:
                out.append(groups[g].pop(0))
    return out


async def layer_swap(pairs: list[PairRef], units: list[Unit], judge: JsonJudge | None, *, n: int, seed: int,
                     say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    texts, images = _donor_pool(units)
    sample = stratified_sample([p for p in pairs if p.orig["verdict"] == "supported"], n, seed)
    rows = []
    jobs = []
    for p in sample:
        donor = pick_donor(p, texts, images, rng)
        if donor is None:
            continue
        jobs.append((p, donor))
    if judge is None:
        return {"n": len(jobs), "pending": len(jobs)}, []

    async def one(p: PairRef, donor: EvidenceText) -> Judgment:
        return await judge_pair(judge, p.unit.finding_id, p.unit.statement, donor, task=p.unit.task)

    results = await asyncio.gather(*(one(p, d) for p, d in jobs))
    for (p, donor), j in zip(jobs, results):
        stamp = j.verdict == "supported"
        donor_src = next((t for _, t in texts if t.content == donor.content), None) if not p.text.is_image else None
        rows.append({"stamp_kind": stamp_kind(p.text, donor_src, j) if stamp else None,"key": p.key, "unit": p.unit.key, "evidence_id": p.text.evidence_id, "modality": "image" if p.text.is_image else "text",
                     "dataset": p.unit.dataset, "donor": str(donor.image_path.name) if p.text.is_image and donor.image_path else f"{len(donor.content)} chars",
                     "donor_image": str(donor.image_path) if p.text.is_image and donor.image_path else None,
                     "donor_text_head": donor.content[:300] if not p.text.is_image else None,
                     "verdict": j.verdict, "rubber_stamp": stamp, "supported": [x.to_dict() for x in j.supported]})
    return summarize_swap(rows, say), rows


def summarize_swap(rows: list[dict[str, Any]], say: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    ok = [r for r in rows if r["verdict"] != "failed"]
    by_mod = {m: {"n": sum(1 for r in ok if r["modality"] == m), "rubber_stamp": sum(1 for r in ok if r["modality"] == m and r["rubber_stamp"]),
                  "real": sum(1 for r in ok if r["modality"] == m and r.get("stamp_kind") == "real")} for m in ("text", "image")}
    stamps = sum(r["rubber_stamp"] for r in ok)
    kinds = Counter(r.get("stamp_kind") for r in ok if r["rubber_stamp"])
    say(f"[C2] 替换 {len(ok)} 对：判支持 {stamps}（{dict(kinds)}；{by_mod}）")
    return {"n": len(ok), "rubber_stamp": stamps, "rate": round(stamps / len(ok), 3) if ok else None,
            "rubber_stamp_real": kinds["real"], "rate_real": round(kinds["real"] / len(ok), 3) if ok else None,
            "stamp_kinds": dict(kinds), "by_modality": by_mod, "failed": len(rows) - len(ok),
            "note": "absence_only = 支持的部分全是'未见/没有'类否定断言（任何无关画面都'支持'，payload 里的来源归属没换）；"
                    "same_tool_template = 原证据与替身都是同一工具的结构化输出（如两份空的对齐结果），不是有效反事实。只有 real 算盖章。"}


_ABSENCE = re.compile(r"未见|没有|不见|未出现|并无|看不到|不可见|未显示|未发现|无(?:任何|明显)|不存在|空档|未返回|未建立")


def stamp_kind(orig: EvidenceText, donor: EvidenceText | None, j: Judgment) -> str:
    """替换后仍判支持的原因分类：absence_only / same_tool_template / real。"""

    parts = [x.part for x in j.supported]
    if parts and all(_ABSENCE.search(x) for x in parts):
        return "absence_only"
    if orig.is_text and (orig.derived_from or str(orig.acquired_by or "").startswith("INSPECT")) and donor is not None \
            and donor.acquired_by == orig.acquired_by:
        return "same_tool_template"
    return "real"


# -------------------------------------------------------------------------------------------------------------- 层 D


def finding_flag(errors: list[Any]) -> bool:
    return any((e.severity.value if hasattr(e, "severity") else e.get("severity")) in FIRM for e in errors)


async def rejudge_unit(u: Unit, judge: JsonJudge, *, perturb: bool = False) -> dict[str, Any]:
    texts = [replace(t, content=IRRELEVANT_PARAGRAPH + t.content) if perturb and t.is_text else t for t in u.texts]
    outcome = await _verify_finding(u.finding_id, u.statement, texts, judge, task=u.task)
    return {"flag": finding_flag(outcome.errors), "types": [e.type.value for e in outcome.errors if e.severity.value in FIRM],
            "pair_verdicts": {j.evidence_id: j.verdict for j in outcome.judgments}, "pending": len(outcome.pending),
            "unsupported_parts": (outcome.coverage or {}).get("unsupported_parts")}


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


async def layer_agreement(units: list[Unit], second: JsonJudge, *, n_random: int, seed: int,
                          say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """第二评委整条重判：对象 = 有实锤错误或有否判对的单元（全部）+ 其余单元随机 n_random 条。"""

    hot = [u for u in units if u.flagged or any(_rejects(j["verdict"]) for j in u.orig_judgments)]
    hot_keys = {u.key for u in hot}
    cold = sorted((u for u in units if u.key not in hot_keys), key=lambda u: hashlib.sha1(f"{seed}|{u.key}".encode()).hexdigest())[:n_random]
    target = hot + cold
    results = await asyncio.gather(*(rejudge_unit(u, second) for u in target))
    rows = []
    pair_rows: list[tuple[bool, bool]] = []
    for u, r in zip(target, results):
        if r["pending"]:
            rows.append({"unit": u.key, "status": "failed"})
            continue
        disagree = r["flag"] != u.flagged
        rows.append({"unit": u.key, "kind": u.kind, "dataset": u.dataset, "image": u.has_image, "group": "hot" if u.key in hot_keys else "random",
                     "sonnet_flag": u.flagged, "sonnet_types": u.firm_types, "second_flag": r["flag"], "second_types": r["types"],
                     "disagree": disagree, "second_pairs": r["pair_verdicts"], "second_unsupported_parts": r["unsupported_parts"]})
        for j in u.orig_judgments:
            sv = r["pair_verdicts"].get(j["evidence_id"])
            if sv in ("supported", "not_found", "contradicted"):
                pair_rows.append((_rejects(j["verdict"]), _rejects(sv)))
    ok = [r for r in rows if r.get("status") != "failed"]
    fpairs = [(r["sonnet_flag"], r["second_flag"]) for r in ok]
    flagged = [r for r in ok if r["sonnet_flag"]]
    confirmed = sum(1 for r in flagged if r["second_flag"])
    say(f"[D] 第二评委重判 {len(ok)} 单元：分歧 {sum(r['disagree'] for r in ok)}；Sonnet 标错 {len(flagged)} 中第二评委同意 {confirmed}")
    return ({"n": len(ok), "failed": len(rows) - len(ok), "agree": sum(not r["disagree"] for r in ok),
             "kappa": kappa(fpairs), "pair_level": {"n": len(pair_rows), "agree": sum(a == b for a, b in pair_rows), "kappa": kappa(pair_rows)},
             "sonnet_flagged": len(flagged), "second_confirms_flagged": confirmed,
             "precision_vs_second": round(confirmed / len(flagged), 3) if flagged else None,
             "second_flagged_sonnet_missed": sum(1 for r in ok if r["second_flag"] and not r["sonnet_flag"]),
             "by_group": {g: {"n": sum(1 for r in ok if r["group"] == g), "disagree": sum(1 for r in ok if r["group"] == g and r["disagree"])}
                          for g in ("hot", "random")},
             "disagreements": [r["unit"] for r in ok if r["disagree"]]}, rows)


async def layer_stability(units: list[Unit], judge: JsonJudge, *, n: int, seed: int,
                          say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sonnet 自身扰动稳定性：只对含文本来源的单元（图片对的 content 不进模型）。"""

    cands = sorted((u for u in units if any(t.is_text for t in u.texts)), key=lambda u: hashlib.sha1(f"{seed}|{u.key}".encode()).hexdigest())
    hot = [u for u in cands if u.flagged]
    target = hot + [u for u in cands if not u.flagged][: max(0, n - len(hot))]
    results = await asyncio.gather(*(rejudge_unit(u, judge, perturb=True) for u in target))
    rows = []
    for u, r in zip(target, results):
        if r["pending"]:
            continue
        flips = {}
        for t, j in zip(u.texts, u.orig_judgments):
            new = r["pair_verdicts"].get(t.evidence_id)
            if t.is_text and new in ("supported", "not_found", "contradicted") and _rejects(new) != _rejects(j["verdict"]):
                flips[t.evidence_id] = [j["verdict"], new]
        rows.append({"unit": u.key, "sonnet_flag": u.flagged, "perturbed_flag": r["flag"], "unstable": r["flag"] != u.flagged,
                     "pair_flips": flips, "perturbed_types": r["types"]})
    unstable = sum(r["unstable"] for r in rows)
    say(f"[D'] 扰动重判 {len(rows)} 单元：翻供 {unstable}")
    return {"n": len(rows), "unstable": unstable, "rate": round(unstable / len(rows), 3) if rows else None,
            "flagged_n": sum(r["sonnet_flag"] for r in rows), "flagged_unstable": sum(r["unstable"] for r in rows if r["sonnet_flag"]),
            "pair_flips": sum(len(r["pair_flips"]) for r in rows)}, rows


async def layer_retest(units: list[Unit], fresh: JsonJudge, target_keys: list[str], say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """评委自身重采样：同一评委、同一 payload、**新缓存文件**（= 不命中原缓存的独立重判）。
    用来区分"扰动翻供"与"本来就会抖"，也给第二评委的分歧一个同评委基线。"""

    index = {u.key: u for u in units}
    target = [index[k] for k in target_keys if k in index]
    results = await asyncio.gather(*(rejudge_unit(u, fresh) for u in target))
    rows = []
    for u, r in zip(target, results):
        if r["pending"]:
            continue
        rows.append({"unit": u.key, "sonnet_flag": u.flagged, "retest_flag": r["flag"], "retest_types": r["types"],
                     "flip": r["flag"] != u.flagged, "pairs": r["pair_verdicts"]})
    flagged = [r for r in rows if r["sonnet_flag"]]
    say(f"[R] 重采样 {len(rows)} 单元：翻转 {sum(r['flip'] for r in rows)}；原标错 {len(flagged)} 中重判仍标错 {sum(r['retest_flag'] for r in flagged)}")
    return {"n": len(rows), "flip": sum(r["flip"] for r in rows), "kappa": kappa([(r["sonnet_flag"], r["retest_flag"]) for r in rows]),
            "flagged": len(flagged), "flagged_kept": sum(r["retest_flag"] for r in flagged),
            "new_flags": sum(1 for r in rows if r["retest_flag"] and not r["sonnet_flag"])}, rows


FAKE_CLAUSES = ("现场共有 37 人受伤", "事发时间为当地时间凌晨 2 点 40 分", "涉事车辆车牌尾号为 4471",
                "现场负责人姓名为 Laurent Mercier", "该说法此前已被路透社在 2014 年核实", "画面左上角可见 KX-209 字样的标牌")
PLAIN_FAKES = ("警方随后逮捕了涉事司机", "当地政府事后撤回了这份声明", "画面中的人物是一名现役警察", "视频发布者已删除原帖并公开道歉")
_NUM2 = re.compile(r"(?<![\d.])(\d{2,4})(?![\d])")


def injections(u: Unit, rng: random.Random) -> list[tuple[str, str]]:
    """召回守门：对干净单元造已知错误（事先知道必须报实锤错）。hedged / limitation_wrap 专测 0.3.4 限制句规则会不会被
    "限制语包着编造事实"绕过。"""

    fake = rng.choice(FAKE_CLAUSES)
    base = u.statement.rstrip("。；;.! ")
    plain = rng.choice(PLAIN_FAKES)
    out = [("append_fact", f"{base}；{fake}。"),
           ("hedged_fact", f"{base}；其余细节尚不能确定，但{fake}。"),
           ("limitation_wrap", f"受材料所限，本次仅能确认：{fake}；{base}。"),
           ("hedged_plain", f"{base}；现有材料尚不能确认全部经过，但{plain}。"),
           ("limitation_plain", f"{base}；本次未能核验其他来源，{plain}。")]
    m = _NUM2.search(u.statement)
    if m:
        n = int(m.group(1))
        new = str(n + rng.choice((3, 7, 11)) if n < 1900 else n + rng.choice((-3, 2, 5)))
        out.append(("alter_number", u.statement[:m.start(1)] + new + u.statement[m.end(1):]))
    return out


async def layer_inject(units: list[Unit], judge: JsonJudge, *, n: int, seed: int, say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    clean = [u for u in units if not u.flagged and not any(_rejects(j["verdict"]) for j in u.orig_judgments) and len(u.statement) >= 12]
    clean.sort(key=lambda u: hashlib.sha1(f"{seed}|inject|{u.key}".encode()).hexdigest())
    pick = [u for u in clean if u.has_image][: n // 2]
    pick += [u for u in clean if not u.has_image][: n - len(pick)]
    jobs = [(u, kind, stmt) for u in pick for kind, stmt in injections(u, rng)]

    async def one(u: Unit, stmt: str) -> Any:
        return await _verify_finding(u.finding_id, stmt, u.texts, judge, task=u.task)

    outs = await asyncio.gather(*(one(u, stmt) for u, _, stmt in jobs))
    rows = []
    for (u, kind, stmt), out in zip(jobs, outs):
        if out.pending:
            continue
        firm = [e.type.value for e in out.errors if e.severity.value in FIRM]
        rows.append({"unit": u.key, "kind": kind, "image": u.has_image, "statement": stmt, "detected": bool(firm), "types": firm,
                     "downgraded": [e.type.value for e in out.errors if e.extra.get("unconfirmed")]})
    by_kind = {k: {"n": sum(1 for r in rows if r["kind"] == k), "detected": sum(1 for r in rows if r["kind"] == k and r["detected"])}
               for k in ("append_fact", "hedged_fact", "limitation_wrap", "hedged_plain", "limitation_plain", "alter_number")}
    for v in by_kind.values():
        v["recall"] = round(v["detected"] / v["n"], 3) if v["n"] else None
    hit = sum(r["detected"] for r in rows)
    say(f"[I] 注入 {len(rows)}：抓到 {hit}；{by_kind}")
    return {"n": len(rows), "detected": hit, "recall": round(hit / len(rows), 3) if rows else None, "by_kind": by_kind,
            "by_modality": {m: {"n": sum(1 for r in rows if r["image"] == (m == "image")),
                                "detected": sum(1 for r in rows if r["image"] == (m == "image") and r["detected"])} for m in ("text", "image")},
            "downgraded_by_confirm": sum(1 for r in rows if r["downgraded"] and not r["detected"])}, rows


# --------------------------------------------------------------------------------------------------------- 汇总


def hard_items(units: list[Unit], repair_rows: list[dict[str, Any]], swap_rows: list[dict[str, Any]], agree_rows: list[dict[str, Any]],
               stable_rows: list[dict[str, Any]], answers: dict[str, Any], retest_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    by_unit: dict[str, dict[str, Any]] = {}
    index = {u.key: u for u in units}

    def add(unit_key: str, reason: str, detail: Any = None) -> None:
        u = index[unit_key]
        item = by_unit.setdefault(unit_key, {"kind": "unit", "unit": unit_key, "unit_kind": u.kind, "run": u.run, "dataset": u.dataset,
                                              "case_id": u.case_id, "scope": u.scope, "finding_id": u.finding_id,
                                              "evidence_ids": [t.evidence_id for t in u.texts], "video_frames": u.has_video_frame,
                                              "image": u.has_image, "verifier_flagged": u.flagged, "verifier_types": u.firm_types,
                                              "reasons": [], "details": {}})
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        if detail is not None:
            item["details"].setdefault(reason, []).append(detail)

    for u in units:
        if u.flagged:
            add(u.key, "firm_error", u.firm_types)
    for r in repair_rows:
        if r["outcome"] in ("verbatim_but_unsupported", "repair_failed", "repair_fixed"):
            add(r["unit"], r["outcome"], {"evidence_id": r["evidence_id"], "coverage": r.get("coverage")})
    for r in agree_rows:
        if r.get("disagree"):
            add(r["unit"], "disagree", {"second_types": r["second_types"], "sonnet_types": r["sonnet_types"]})
    for r in stable_rows:
        if r["unstable"]:
            add(r["unit"], "unstable", r["perturbed_types"])
    for r in retest_rows or []:
        if r["flip"]:
            add(r["unit"], "retest_flip", r["retest_types"])
    for a in answers.get("items") or []:
        if a["conflict"]:
            for u in units:
                if u.kind == "answer" and u.run == a["run"] and u.scope == a["request"] and u.finding_id == "A1":
                    add(u.key, "gold_conflict", {"label": a["label"], "verifier": a["verifier"], "second": a["second"]})
    out = list(by_unit.values())
    for r in swap_rows:
        if r["rubber_stamp"] and r.get("stamp_kind", "real") == "real":
            u = index[r["unit"]]
            out.append({"kind": "swap", "unit": r["unit"], "unit_kind": u.kind, "run": u.run, "dataset": u.dataset, "case_id": u.case_id,
                        "scope": u.scope, "finding_id": u.finding_id, "evidence_ids": [r["evidence_id"]], "video_frames": r["modality"] == "image",
                        "image": r["modality"] == "image", "verifier_flagged": False, "verifier_types": [], "reasons": ["rubber_stamp"],
                        "details": {"rubber_stamp": [{"donor_image": r["donor_image"], "donor_text_head": r["donor_text_head"],
                                                      "evidence_id": r["evidence_id"], "supported": r["supported"]}]}})
    return out


def controls(units: list[Unit], hard: list[dict[str, Any]], repair_rows: list[dict[str, Any]], agree_rows: list[dict[str, Any]]) -> list[str]:
    """四层全通过：无实锤错、无否判对、第二评委重判过且同意。"""

    bad = {h["unit"] for h in hard}
    agreed = {r["unit"] for r in agree_rows if r.get("disagree") is False}
    return [u.key for u in units if u.key not in bad and u.key in agreed and not u.flagged
            and not any(_rejects(j["verdict"]) for j in u.orig_judgments)]


def precision_auto(units: list[Unit], rows: dict[str, Any]) -> dict[str, Any]:
    """核验器标错单元的自动精确率区间：下界 = 第二评委与同评委重采样都仍判错；上界 = 任一仍判错。没有人工时只能给区间。"""

    ag = {r["unit"]: r for r in rows.get("agreement") or [] if "second_flag" in r}
    rt = {r["unit"]: r for r in rows.get("retest") or []}
    st = {r["unit"]: r for r in rows.get("stability") or []}
    rep: dict[str, list[str]] = defaultdict(list)
    for r in rows.get("repair") or []:
        rep[r["unit"]].append(r["outcome"])
    table = []
    for u in units:
        if not u.flagged:
            continue
        second = ag.get(u.key, {}).get("second_flag")
        retest = rt.get(u.key, {}).get("retest_flag")
        table.append({"unit": u.key, "types": u.firm_types, "image": u.has_image, "second": second, "retest": retest,
                      "perturbed": st.get(u.key, {}).get("perturbed_flag"), "repair": rep.get(u.key, [])})
    n = len(table)
    both = sum(1 for t in table if t["second"] and t["retest"])
    either = sum(1 for t in table if t["second"] or t["retest"])
    second_only = sum(1 for t in table if t["second"])
    return {"flagged": n, "second_confirms": second_only, "retest_keeps": sum(1 for t in table if t["retest"]), "both": both, "either": either,
            "interval": [round(both / n, 3), round(either / n, 3)] if n else None, "table": table,
            "note": "区间不是置信区间：下界把'第二评委或重采样任一不认'都当误报，上界把'任一仍认'都当真错。真实值以人工 v1' 为准。"}


def write_md(path: Path, report: dict[str, Any]) -> None:
    L = report["layers"]
    lines = [f"# 核验器精确率四层估计：{report['meta']['tag']}", "",
             f"输入：{report['meta']['report_verdicts']} + {report['meta']['unit_verdicts']}；评委 {report['meta']['judge']}，第二评委 "
             f"{report['meta']['second']}，改写 {report['meta']['writer']}。", "",
             f"重建核对：{L['replay']['pairs']} 对中 {L['replay']['cache_hit']} 对命中原评委缓存，{L['replay']['same_verdict']} 对判定与 r0c 相同。", ""]
    a = L.get("answers") or {}
    lines += ["## 层 A · 答案标签", "", f"n={a.get('n')}；核验器判标签不符 {a.get('verifier_bad')}；与第二评委冲突 {a.get('conflicts')}。", "",
              "| run | 标签 | 核验器 | 子串 | 第二评委 |", "|---|---|---|---|---|"]
    for x in a.get("items") or []:
        lines.append(f"| {x['run']} | {x['label']} | {x['verifier']} | {x['program_substring']} | {x['second']} |")
    b = L.get("seeded") or {}
    lines += ["", "## 层 B · 构造正例", "", f"误拒率 {b.get('false_reject_rate')}（n={b.get('n')}）；分档 {json.dumps(b.get('by_tier'), ensure_ascii=False)}", "",
              f"按来源长度：{json.dumps(b.get('by_length'), ensure_ascii=False)}；改写过滤 {json.dumps(b.get('paraphrase_filter'), ensure_ascii=False)}", "",
              "B2 的被拒条目逐条附在 rows.json（seeded）里：改写模型自己改错了事实（加了来源没有的断言、把年份当数量）时，被拒是**对的**，"
              "不应计入误拒；看完再定误拒率。", ""]
    c1, c2 = (L.get("counterfactual") or {}).get("repair") or {}, (L.get("counterfactual") or {}).get("swap") or {}
    lines += ["## 层 C · 反事实", "", f"C1 修复（文本否判对 n={c1.get('n')}）：verbatim_but_unsupported {c1.get('verbatim_but_unsupported')}、"
              f"repair_failed {c1.get('repair_failed')}、repair_fixed {c1.get('repair_fixed')}、无法定位 {c1.get('unlocatable')}；误报率上界 {c1.get('fp_rate_upper')}。"
              f"图片否判对 {c1.get('image_pairs_rejected')} 条交层 D。", "",
              f"C2 替换（n={c2.get('n')}）：替换后仍判支持 {c2.get('rubber_stamp')}（{c2.get('rate')}），其中真盖章 {c2.get('rubber_stamp_real')}"
              f"（{c2.get('rate_real')}）；分类 {json.dumps(c2.get('stamp_kinds'), ensure_ascii=False)}；{json.dumps(c2.get('by_modality'), ensure_ascii=False)}", ""]
    d, s = L.get("agreement") or {}, L.get("stability") or {}
    lines += ["## 层 D · 双评委与稳定性", "", f"第二评委整条重判 {d.get('n')} 单元：一致 {d.get('agree')}，κ={d.get('kappa')}；"
              f"Sonnet 标错 {d.get('sonnet_flagged')} 条中第二评委同意 {d.get('second_confirms_flagged')}（{d.get('precision_vs_second')}）；"
              f"第二评委标错而 Sonnet 放过 {d.get('second_flagged_sonnet_missed')}。对级一致 {json.dumps(d.get('pair_level'), ensure_ascii=False)}。", "",
              f"扰动稳定性：{s.get('n')} 单元翻供 {s.get('unstable')}（{s.get('rate')}）；其中 Sonnet 标错 {s.get('flagged_n')} 条翻供 {s.get('flagged_unstable')}。", ""]
    rt = L.get("retest") or {}
    if rt:
        lines += [f"同评委重采样（新缓存、原样重判）：{rt.get('n')} 单元翻转 {rt.get('flip')}，κ={rt.get('kappa')}；"
                  f"原标错 {rt.get('flagged')} 条中重判仍标错 {rt.get('flagged_kept')}，新增标错 {rt.get('new_flags')}。", ""]
    inj = L.get("inject") or {}
    if inj:
        lines += ["## 召回守门（注入已知错误）", "", f"注入 {inj['n']} 条，抓到 {inj['detected']}（{inj['recall']}）；分类型 "
                  f"{json.dumps(inj['by_kind'], ensure_ascii=False)}；分模态 {json.dumps(inj['by_modality'], ensure_ascii=False)}；"
                  f"因复核未复现而漏掉 {inj['downgraded_by_confirm']}。", ""]
    pa = report.get("precision_auto") or {}
    if pa:
        lines += ["## 标错单元的自动精确率区间", "",
                  f"核验器（Sonnet）标了实锤错误的单元 {pa['flagged']} 个：第二评委仍判错 {pa['second_confirms']}，同评委重采样仍判错 {pa['retest_keeps']}，"
                  f"两者都仍判错 {pa['both']}，任一仍判错 {pa['either']} → 精确率自动区间 {pa['interval']}（门槛 0.85）。{pa['note']}", "",
                  "| 单元 | 错误类型 | 图 | 第二评委 | 重采样 | 扰动后 | C1 |", "|---|---|---|---|---|---|---|"]
        mark = {True: "仍错", False: "不错", None: "—"}
        for t in pa["table"]:
            lines.append(f"| {t['unit'].replace('|', ' · ')} | {','.join(t['types'])} | {'是' if t['image'] else ''} | {mark[t['second']]} | {mark[t['retest']]} | "
                         f"{mark[t['perturbed']]} | {','.join(t['repair'])} |")
        lines.append("")
    h = report["hard_items"]
    rc = Counter(r for x in h for r in x["reasons"])
    lines += ["## 疑难件", "", f"共 {len(h)} 条（单元 {sum(1 for x in h if x['kind']=='unit')}，替换盖章 {sum(1 for x in h if x['kind']=='swap')}）；"
              f"原因 {dict(rc)}；含视频帧 {sum(1 for x in h if x['video_frames'])}。对照池（四层全通过）{len(report['controls'])} 条。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------- CLI


async def main_async(args: argparse.Namespace) -> int:
    from ..cases import load_cases
    from . import anchored as _anchored

    if args.verifier == "0.3.3":
        _anchored.FEATURES.update(limitation_rule=False, confirm_firm=False)

    def say(line: str) -> None:
        print(line, flush=True)

    units, runs = load_units(args.report_verdicts, args.unit_verdicts, args.runs)
    pairs = all_pairs(units)
    cases = load_cases(args.cases) if args.cases else {}
    say(f"单元 {len(units)}（报告 Finding {sum(u.kind == 'report' for u in units)}，答案句 {sum(u.kind == 'answer' for u in units)}）；"
        f"(句, E) 对 {len(pairs)}；run {len(runs)}")
    out = Path(args.out)
    state = out.with_suffix(".rows.json")
    report: dict[str, Any] = {"meta": {"tag": out.stem, "report_verdicts": Path(args.report_verdicts).name,
                                       "unit_verdicts": Path(args.unit_verdicts).name if args.unit_verdicts else None,
                                       "judge": args.judge_model, "second": args.second_model, "writer": args.writer_model,
                                       "seed": args.seed, "verifier": args.verifier}, "layers": {}}
    rows: dict[str, Any] = {}
    try:
        report["layers"]["replay"] = await replay(pairs, replay_judge(args.replay_cache))
        say(f"[replay] {report['layers']['replay']}")
        judge = second = writer = None
        if not args.no_judge:
            judge = cli_judge(args.judge_model, args.cache)
            second = make_judge(args.second_kind, args.second_model, args.second_cache, endpoint=args.endpoint) if args.second_model else None
            writer = make_judge(args.writer_kind, args.writer_model, args.writer_cache, endpoint=args.endpoint) if args.writer_model else None
        repair, rows["repair"] = await layer_repair(pairs, judge, say, locator=writer)
        if args.no_judge:
            swap_n = len(stratified_sample([p for p in pairs if p.orig["verdict"] == "supported"], args.swap, args.seed))
            seeds = sum(len(seed_sentences(c, per_run=args.seeds_per_run, rng=random.Random(args.seed))) for c in runs.values())
            hot = sum(1 for u in units if u.flagged or any(_rejects(j["verdict"]) for j in u.orig_judgments))
            est = {"C1": repair["pending"], "C2": swap_n, "B1": seeds, "B2_write": seeds, "B2_judge": seeds,
                   "D_units": hot + args.agree_random, "D_calls≈": sum(len(u.texts) + 1 for u in units if u.flagged or any(_rejects(j["verdict"]) for j in u.orig_judgments)) + args.agree_random * 3,
                   "stability_units": args.stability}
            report["layers"]["counterfactual"] = {"repair": repair}
            report["estimate"] = est
            say(f"[no-judge] 预计调用：{est}")
            out.with_suffix(".nojudge.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            return 0
        assert judge is not None
        report["layers"]["answers"] = await layer_answers(load_jsonl(args.unit_verdicts) if args.unit_verdicts else [], cases, runs, second, say)
        report["layers"]["seeded"], rows["seeded"] = await layer_seeded(runs, judge, writer, per_run=args.seeds_per_run, seed=args.seed, say=say)
        swap, rows["swap"] = await layer_swap(pairs, units, judge, n=args.swap, seed=args.seed, say=say)
        report["layers"]["counterfactual"] = {"repair": repair, "swap": swap}
        if second is not None:
            report["layers"]["agreement"], rows["agreement"] = await layer_agreement(units, second, n_random=args.agree_random, seed=args.seed, say=say)
        else:
            rows["agreement"] = []
        report["layers"]["stability"], rows["stability"] = await layer_stability(units, judge, n=args.stability, seed=args.seed, say=say)
        rows["retest"] = []
        if args.retest_cache:
            fresh = cli_judge(args.judge_model, args.retest_cache)
            targets = [r["unit"] for r in rows["agreement"]] or [u.key for u in units if u.flagged]
            report["layers"]["retest"], rows["retest"] = await layer_retest(units, fresh, targets, say)
        if args.inject:
            report["layers"]["inject"], rows["inject"] = await layer_inject(units, judge, n=args.inject, seed=args.seed, say=say)
        report["precision_auto"] = precision_auto(units, rows)
        hard = hard_items(units, rows["repair"], rows["swap"], rows["agreement"], rows["stability"], report["layers"]["answers"],
                          rows["retest"])
        report["hard_items"] = hard
        report["controls"] = controls(units, hard, rows["repair"], rows["agreement"])
        report["units"] = [{"unit": u.key, "kind": u.kind, "run": u.run, "dataset": u.dataset, "flagged": u.flagged, "types": u.firm_types,
                            "image": u.has_image, "video_frames": u.has_video_frame, "n_evidence": len(u.texts)} for u in units]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        state.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        write_md(out.with_suffix(".md"), report)
        say(f"写入 {out}、{out.with_suffix('.md')}、{state}")
        for j in (judge, second, writer):
            if j is not None:
                say(j.summary())
    finally:
        for c in runs.values():
            c.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="核验器精确率四层估计（层 A/B/C/D）+ 疑难件")
    p.add_argument("--report-verdicts", required=True)
    p.add_argument("--unit-verdicts")
    p.add_argument("--runs", required=True, help="runs 根目录")
    p.add_argument("--cases", help="案卷目录（层 A）")
    p.add_argument("--out", required=True, help="precision_<tag>.json")
    p.add_argument("--replay-cache", required=True, help="原评委缓存（重建核对，只读）")
    p.add_argument("--cache", required=True, help="主评委缓存（层 B/C/稳定性的新调用写这里；可以就是原缓存）")
    p.add_argument("--judge-model", default="sonnet")
    p.add_argument("--second-model", default=None, help="第二评委（claude CLI 模型别名）；不给则跳过层 D 双评委")
    p.add_argument("--second-cache")
    p.add_argument("--second-kind", default="claude-cli", choices=("claude-cli", "local", "openrouter"),
                   help="第二评委来源：claude-cli（同家族，分歧率偏乐观）/ local（本机 Ollama 等，不同家族，方案首选）/ openrouter（花钱）")
    p.add_argument("--writer-kind", default="claude-cli", choices=("claude-cli", "local", "openrouter"))
    p.add_argument("--endpoint", default=None, help="local 评委的 OpenAI 兼容地址（如 http://127.0.0.1:11434/v1）")
    p.add_argument("--writer-model", default=None, help="B2 改写模型；不给则只做 B1")
    p.add_argument("--writer-cache")
    p.add_argument("--verifier", default="0.3.4", choices=("0.3.3", "0.3.4"), help="0.3.3 = 关掉 anchored.FEATURES（限制句规则、实锤复核）")
    p.add_argument("--inject", type=int, default=0, help="召回守门：抽这么多个干净单元做注入（0 = 不做）")
    p.add_argument("--retest-cache", default=None, help="主评委重采样用的新缓存文件（与 --cache 不同）；给了才跑重采样对照")
    p.add_argument("--seeds-per-run", type=int, default=10)
    p.add_argument("--swap", type=int, default=150)
    p.add_argument("--agree-random", type=int, default=40)
    p.add_argument("--stability", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--no-judge", action="store_true", help="只跑程序层并打印预计调用数")
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Located", "PairRef", "Unit", "anchor_tokens", "char_coverage", "hard_items", "load_units", "locate", "pick_donor",
           "seed_sentences", "split_sentences", "stratified_sample"]
