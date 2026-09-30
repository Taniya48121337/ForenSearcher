"""plan / answer 单元的召回率校准：往 GIA 的答案与派工计划里注入已知错误，再跑同一套单元核验，看能不能抓回来。

大白话：calibrate_recall.py 只改工作区报告；GIA 的两个单元（FINISH 答案、INVESTIGATE 计划）一直没有召回数。
路由的入场条件要每类错误的召回 ρ_c（S4：学生会往核验器看不见的地方钻），所以这里把"注入 → 核验 → 看是否
命中期望类型、与同条件对照比"扩到 answer / plan：

answer（改 decision.output["answer"] 的文字，核验走 units.check_answer + answer_anchored 逐句锚定 + calibrate.recalibrate）：
- drop_citation          删掉一组引用 → 期望 unsupported_assertion / wrong_citation。注意"uncited"不是错误类型：
                          删的是段末那组引用时，那几句变成 uncited，answer_anchored 按设计不报错（结构性漏报，行里标 becomes_uncited）；
                          删的是段中的引用时，那几句被"回溯覆盖"到下一组引用上，才可能被锚定层判无支持。
- add_unsupported_clause 给某组句子加一句所引 E 都不支持的数字 / 实体 → unsupported_assertion（或 wrong_citation）。
- swap_citation          把一组引用换成答案别处没引、内容也不相干的文本 E → wrong_citation / unsupported_assertion。
- fabricate_citation     往一组引用里加一个不存在的 E → fabricated_evidence_id（程序层）。
- flip_label             结论标签翻转（不实 ↔ 属实 等）→ label_expected 的 omitted_required_fact（程序层）。
- negate_label           "不实"改成"并非不实"：标签字符串还在，意思反了 → 期望同上；程序层是子串匹配，抓不到（结构性漏报）。
- drop_required_block    删掉覆盖某个 must_cite 要求的全部句子 → premature_finish / omitted_required_fact。
                          注意：premature_finish 是按**报告**里干净 Finding 的覆盖算的，与答案文字无关；answer 单元没有
                          "答案有没有写到每个必需要求"的充分性检查，所以这一类在现有核验器下注定漏报（行里照记）。

plan（改 decision.output["tasks"]，核验走 units.check_plan + plan_judged）：
- smuggle_premise                往任务文字里塞一个案卷没给、此前也没核实的事实前提 → premise_smuggling（只有模型能判）。
- duplicate_sibling              复制一个兄弟任务 → overlapping_sibling_tasks。注意该类型只由 hindsight 按"两个工作区交回了
                                  同类 Finding"判，计划阶段的文字重复程序层与 plan 模型评判都看不见（plan 提示词不问重叠）。
- retarget_covered               合成下一轮计划：只派已被覆盖的要求，同时留一项 must_cite 缺口不派 → task_not_gap_driven（只有模型能判）。
                                  条件是合成的（round / covered_before 写进 condition），对照在同一条件下跑原任务。
- sibling_dependency             任务文字写"待 W{r}-{j} 的结果确认后……" → dependent_sibling_tasks（程序层正则）。
- sibling_dependency_paraphrase  换个说法写依赖（"在另一名调查员查明……之后，以其查明的……为准"）→ 同上；正则抓不到，
                                  plan 提示词也不问依赖，结构性漏报。
- fabricate_material             materials 里加一个视图里不存在的编号 → illegal_action（程序层）。

期望类型只用 errors.py 里已有的枚举。命中 = 注入后出现期望类型的非 flag 错误，且它不在同条件对照判定里（按 unit_error_key 比）；
plan 的模型错误还要求落在被改的那个任务上。输出格式与 calibrate_recall 一致（rows / by_kind / recall / inconclusive / controls），
另加 by_unit，以及 rejected（处置比同条件对照更差）：命中 ≠ 拦下——plan 单元的错误类型都是 major / minor，单独一条不会让
decide() 判 negative，export 仍把它收进 accepted；rejected 量的是"检出之后真的挡住了没有"。

用法（本机；--judge none 只跑程序层，local 连本地核验服务，openrouter 走付费 API）：
    python -m foren_searcher.datagen.verify.calibrate_units_recall --runs experiments/sft_data/runs/true_4124903_r1 \
        --verdicts experiments/sft_data/verdicts_r0_judged.jsonl --judge none \
        --out experiments/sft_data/calibration/units_recall_r0.none.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..cases import Case, load_cases
from ..errors import ErrorRecord, ErrorType
from ..runlog import load_run
from .anchored import BUDGET_EXIT_CODE, JsonJudge, JudgeBudgetExhausted, _squash, budget_message
from .answer_anchored import _CITE_GROUP, _ID, anchor_answer, apply_answer_anchoring
from .calibrate import recalibrate
from .calibrate_recall import _FAKE_CLAUSES, _recall, is_inconclusive, pending_reasons
from .hindsight import UsefulEvidence
from .judges import add_judge_arguments, judge_from_args, judge_summary
from .plan_judged import apply_plan_judgment, judge_plan
from .units import check_answer, check_plan, verify_units
from .verdict import Verdict

if TYPE_CHECKING:
    from ..runlog import Decision, RunLog

# 塞进任务文字的"未经证实的前提"：都是可核的具体事实（时间、机构、伤亡、定性），案卷与报告里都不会有
FAKE_PREMISES = ("该视频由一名游客于当晚 22 点 35 分在战神广场拍摄", "起火车辆隶属于烟花承包商 Pyro-Est 公司",
                 "现场已有 37 人受伤送医", "法国内政部已于次日将此事定性为纵火")
# 标签翻转表：键 → 反义（只换整词；"不实"在"属实"之前，避免"不属实"被换两次）
LABEL_FLIPS = {"不实": "属实", "属实": "不实", "虚假": "真实", "真实": "虚假", "False": "True", "True": "False",
               "false": "true", "true": "false"}
PARAPHRASED_DEPENDENCY = "在另一名调查员查明视频的拍摄地点之后，以其查明的地点为准继续核查。"


@dataclass
class UnitInjection:
    kind: str
    unit: str                                   # answer | plan
    request: str
    target: str | None                          # answer：块号 B{i}；plan：任务 T{i}
    expected: tuple[ErrorType, ...]
    detail: str
    output: dict[str, Any]                      # 改过的 decision.output
    condition: dict[str, Any] = field(default_factory=dict)   # plan：{"round_index", "covered_before"}；空 = 真实条件
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "unit": self.unit, "request": self.request, "target": self.target,
                "expected": [e.value for e in self.expected], "detail": self.detail, "condition": self.condition, **self.extra}


# --------------------------------------------------------------------------- answer：按引用组切块

@dataclass
class Block:
    """答案里被同一组引用覆盖的一段文字：[text_start, group_start) 是句子，[group_start, group_end) 是引用组。
    与 answer_anchored.split_answer 的作用域一致：一组引用覆盖从上一组引用（或段首）到它为止的全部句子。"""

    index: int
    text_start: int
    group_start: int
    group_end: int
    ids: list[str]
    text: str
    last_in_paragraph: bool

    @property
    def block_id(self) -> str:
        return f"B{self.index}"


def answer_blocks(answer: str) -> list[Block]:
    out: list[Block] = []
    prev_end = 0
    matches = list(_CITE_GROUP.finditer(answer))
    for k, m in enumerate(matches):
        para_start = answer.rfind("\n", 0, m.start()) + 1
        start = max(prev_end, para_start)
        text = answer[start:m.start()]
        nxt = matches[k + 1].start() if k + 1 < len(matches) else len(answer)
        para_end = answer.find("\n", m.end())
        para_end = len(answer) if para_end < 0 else para_end
        # 本组之后、段末之前再没有引用组 → 它是段内最后一组
        last = nxt >= para_end
        out.append(Block(index=len(out) + 1, text_start=start, group_start=m.start(), group_end=m.end(), ids=_ID.findall(m.group(1)),
                         text=text, last_in_paragraph=last))
        prev_end = m.end()
    return [b for b in out if b.text.strip()]


def _finding_evidence(log: RunLog, useful: UsefulEvidence) -> dict[str, list[str]]:
    table = {fid: list(row["evidence_ids"]) for fid, row in useful.clean_findings.items()}
    for rec in log.store.workspace_report_records():
        for finding in rec.to_dict().get("findings") or []:
            table.setdefault(str(finding.get("finding_id")), [x for x in finding.get("evidence_ids") or [] if isinstance(x, str)])
    return table


def _block_evidence(block: Block, finding_evidence: dict[str, list[str]]) -> list[str]:
    out = [i for i in block.ids if i.startswith("E")]
    for f in (i for i in block.ids if i.startswith("F")):
        out += [e for e in finding_evidence.get(f, []) if e not in out]
    return out


def _label_forms(case: Case | None) -> list[str]:
    forms: list[str] = []
    for check in case.checks if case else ():
        if check.kind == "label_expected":
            forms += [str(check.params["label"]), *(str(a) for a in check.params.get("aliases") or [])]
    return [f for f in forms if f]


def _entity_needles(case: Case | None) -> list[str]:
    return [str(c.params["text"]) for c in (case.checks if case else ()) if c.kind == "entity_in_evidence"]


@dataclass
class AnswerContext:
    decision: Decision
    log: RunLog
    case: Case | None
    useful: UsefulEvidence
    finding_evidence: dict[str, list[str]]


AnswerMutator = Callable[[AnswerContext, str, list[Block], random.Random], "UnitInjection | None"]


def _answer_injection(kind: str, ctx: AnswerContext, target: Block | None, expected: tuple[ErrorType, ...], detail: str,
                      answer: str, **extra: Any) -> UnitInjection:
    output = {**copy.deepcopy(ctx.decision.output), "answer": answer}
    return UnitInjection(kind, "answer", ctx.decision.request_id, target.block_id if target else None, expected, detail, output, extra=extra)


def drop_citation(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    cands = [b for b in blocks if any(i.startswith("E") for i in b.ids)]
    if not cands:
        return None
    b = rng.choice(cands)
    mutated = answer[:b.group_start] + answer[b.group_end:]
    return _answer_injection("drop_citation", ctx, b, (ErrorType.UNSUPPORTED_ASSERTION, ErrorType.WRONG_CITATION),
                             f"删去 {b.block_id} 的引用 {answer[b.group_start:b.group_end].strip()}", mutated,
                             becomes_uncited=b.last_in_paragraph, removed_ids=b.ids)


def add_unsupported_clause(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    cands = [b for b in blocks if _block_evidence(b, ctx.finding_evidence)]
    if not cands:
        return None
    b = rng.choice(cands)
    cited = _squash("\n".join(ctx.log.lookup.get(e).content or "" for e in _block_evidence(b, ctx.finding_evidence)
                              if ctx.log.lookup.has(e) and ctx.log.lookup.get(e).is_text))
    clauses = [c for c in _FAKE_CLAUSES if _squash(c)[-6:] not in cited]
    if not clauses:
        return None
    clause = rng.choice(clauses)
    body = answer[b.text_start:b.group_start].rstrip()
    tail_ws = answer[b.text_start:b.group_start][len(body):]
    body = (body[:-1] if body.endswith(("。", ".", "；", ";")) else body) + "；" + clause + "。"
    mutated = answer[:b.text_start] + body + tail_ws + answer[b.group_start:]
    return _answer_injection("add_unsupported_clause", ctx, b, (ErrorType.UNSUPPORTED_ASSERTION, ErrorType.WRONG_CITATION),
                             f"{b.block_id} 句末追加 “{clause}”", mutated, clause=clause)


def swap_citation(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    cited_anywhere = {e for b in blocks for e in _block_evidence(b, ctx.finding_evidence)}
    pool = sorted((e for e in ctx.decision.visible_ids if e not in cited_anywhere and ctx.log.lookup.has(e) and ctx.log.lookup.get(e).is_text),
                  key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    cands = [b for b in blocks if any(i.startswith("E") for i in b.ids)]
    rng.shuffle(cands)
    for b in cands:
        head = _squash(b.text)[:12]
        usable = [e for e in pool if head and head not in _squash(ctx.log.lookup.get(e).content or "")]
        if not usable:
            continue
        new_ids = rng.sample(usable, k=min(2, len(usable)))
        group = "（" + "、".join(new_ids) + "）"
        mutated = answer[:b.group_start] + group + answer[b.group_end:]
        return _answer_injection("swap_citation", ctx, b, (ErrorType.WRONG_CITATION, ErrorType.UNSUPPORTED_ASSERTION),
                                 f"{b.block_id} 引用 {b.ids} → {new_ids}", mutated, before=b.ids, after=new_ids)
    return None


def fabricate_citation(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    if not blocks:
        return None
    b = rng.choice(blocks)
    fake = "E9%03d" % rng.randint(0, 999)
    while ctx.log.lookup.has(fake):
        fake = "E9%03d" % rng.randint(0, 999)
    group = "（" + "、".join([*b.ids, fake]) + "）"
    mutated = answer[:b.group_start] + group + answer[b.group_end:]
    return _answer_injection("fabricate_citation", ctx, b, (ErrorType.FABRICATED_EVIDENCE_ID,), f"{b.block_id} 引用追加不存在的 {fake}",
                             mutated, fake=fake)


def _flip_labels(text: str, forms: list[str]) -> str:
    keys = [k for k in sorted(LABEL_FLIPS, key=len, reverse=True) if k in forms or LABEL_FLIPS[k] in forms]
    if not keys:
        return text
    return re.sub("|".join(re.escape(k) for k in keys), lambda m: LABEL_FLIPS[m.group(0)], text)


def flip_label(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    forms = _label_forms(ctx.case)
    if not forms or not any(f.lower() in answer.lower() for f in forms):
        return None
    mutated = _flip_labels(answer, forms)
    if mutated == answer or any(f.lower() in mutated.lower() for f in forms):
        return None          # 翻不干净（比如标签词没有反义项）就不注入，免得"期望命中"本身不成立
    return _answer_injection("flip_label", ctx, None, (ErrorType.OMITTED_REQUIRED_FACT,), f"结论标签翻转（{'/'.join(forms)} → 反义）", mutated)


def negate_label(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    forms = _label_forms(ctx.case)
    hits = [(answer.find(f), f) for f in forms if f and answer.find(f) >= 0]
    if not hits:
        return None
    pos, form = min(hits)
    mutated = answer[:pos] + "并非" + answer[pos:]
    return _answer_injection("negate_label", ctx, None, (ErrorType.OMITTED_REQUIRED_FACT, ErrorType.WRONG_CITATION, ErrorType.UNSUPPORTED_ASSERTION),
                             f"“{form}” → “并非{form}”（标签字符串仍在，结论反了）", mutated, form=form)


def drop_required_block(ctx: AnswerContext, answer: str, blocks: list[Block], rng: random.Random) -> UnitInjection | None:
    case = ctx.case
    if case is None or not case.must_cite_requirements:
        return None
    guard = [x.lower() for x in _label_forms(case) + _entity_needles(case)]
    reqs = list(case.must_cite_requirements)
    rng.shuffle(reqs)
    for req in reqs:
        hit = [b for b in blocks if any(req in ctx.useful.requirements_of_evidence(case, e) for e in _block_evidence(b, ctx.finding_evidence))]
        if not hit or any(g in b.text.lower() for b in hit for g in guard):
            continue          # 删了会连带删掉标签 / 必需实体：程序层会因为别的原因报错，测不到本来要测的东西
        mutated = answer
        for b in sorted(hit, key=lambda x: x.text_start, reverse=True):
            mutated = mutated[:b.text_start] + mutated[b.group_end:]
        return _answer_injection("drop_required_block", ctx, hit[0], (ErrorType.PREMATURE_FINISH, ErrorType.OMITTED_REQUIRED_FACT),
                                 f"删去覆盖 {req} 的全部句子 {[b.block_id for b in hit]}", mutated, requirement=req,
                                 dropped=[b.block_id for b in hit])
    return None


ANSWER_MUTATORS: dict[str, AnswerMutator] = {
    "drop_citation": drop_citation, "add_unsupported_clause": add_unsupported_clause, "swap_citation": swap_citation,
    "fabricate_citation": fabricate_citation, "flip_label": flip_label, "negate_label": negate_label,
    "drop_required_block": drop_required_block,
}


# --------------------------------------------------------------------------- plan

@dataclass
class PlanContext:
    decision: Decision
    log: RunLog
    case: Case | None
    covered_before: set[str]
    covered_all: set[str]                        # 全部真实轮次结束后已覆盖的要求（hindsight）
    last_round: int


PlanMutator = Callable[[PlanContext, random.Random], "UnitInjection | None"]


def _tasks(decision: Decision) -> list[dict[str, Any]]:
    raw = decision.output.get("tasks")
    return [dict(t) for t in raw if isinstance(t, dict)] if isinstance(raw, list) else []


def _plan_injection(kind: str, ctx: PlanContext, target: int | None, expected: tuple[ErrorType, ...], detail: str,
                    tasks: list[dict[str, Any]], condition: dict[str, Any] | None = None, **extra: Any) -> UnitInjection:
    output = {**copy.deepcopy(ctx.decision.output), "tasks": tasks}
    return UnitInjection(kind, "plan", ctx.decision.request_id, f"T{target + 1}" if target is not None else None, expected, detail, output,
                         condition=condition or {}, extra=extra)


def _prior_text(ctx: PlanContext) -> str:
    parts = [ctx.case.task_text if ctx.case else "", *(ctx.case.claims if ctx.case else ())]
    parts += [str(t.get("task") or "") for t in _tasks(ctx.decision)]
    for rec in ctx.log.store.workspace_report_records():
        parts += [str(f.get("statement") or "") for f in rec.to_dict().get("findings") or []]
    return _squash("\n".join(parts))


def smuggle_premise(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    tasks = _tasks(ctx.decision)
    if not tasks:
        return None
    seen = _prior_text(ctx)
    premises = [p for p in FAKE_PREMISES if _squash(p) not in seen]
    if not premises:
        return None
    i = rng.randrange(len(tasks))
    premise = rng.choice(premises)
    tasks[i]["task"] = f"已知{premise}。" + str(tasks[i].get("task") or "")
    return _plan_injection("smuggle_premise", ctx, i, (ErrorType.PREMISE_SMUGGLING,), f"T{i + 1} 开头塞入前提 “{premise}”", tasks, premise=premise)


def duplicate_sibling(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    tasks = _tasks(ctx.decision)
    if not tasks:
        return None
    i = rng.randrange(len(tasks))
    tasks.append(copy.deepcopy(tasks[i]))
    return _plan_injection("duplicate_sibling", ctx, len(tasks) - 1, (ErrorType.OVERLAPPING_SIBLING_TASKS,),
                           f"复制 T{i + 1} 为 T{len(tasks)}", tasks, copied_from=f"T{i + 1}")


def retarget_covered(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    """合成第 last_round+1 轮：只派一个针对已覆盖要求 r 的任务；covered_before 去掉另一项 must_cite 要求 u（留作真缺口），
    这样 unnecessary_round（must ⊆ covered_before）不会顺带触发，测的只是 task_not_gap_driven。"""

    case = ctx.case
    if case is None or not case.requirements or not ctx.covered_all:
        return None
    covered = sorted(ctx.covered_all)
    r = rng.choice(covered)
    gaps = [u for u in case.must_cite_requirements if u != r]
    if not gaps:
        return None
    u = rng.choice(sorted(gaps))
    covered_before = sorted(set(covered) - {u})
    req = next(x for x in case.requirements if x.id == r)
    materials = list((_tasks(ctx.decision) or [{}])[0].get("materials") or [])
    tasks = [{"task": f"请再次核查：{req.text}。给出所引证据。", "materials": materials}]
    return _plan_injection("retarget_covered", ctx, 0, (ErrorType.TASK_NOT_GAP_DRIVEN,),
                           f"合成第 {ctx.last_round + 1} 轮：只派已覆盖的 {r}，缺口 {u} 不派", tasks,
                           condition={"round_index": ctx.last_round + 1, "covered_before": covered_before}, retarget=r, gap=u)


def sibling_dependency(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    tasks = _tasks(ctx.decision)
    if len(tasks) < 2:
        return None
    i = rng.randrange(len(tasks))
    j = rng.choice([k for k in range(len(tasks)) if k != i])
    round_index = int(ctx.decision.round_index or 1)
    tasks[i]["task"] = str(tasks[i].get("task") or "") + f"待 W{round_index}-{j + 1} 的结果确认后，再根据其结论继续核查。"
    return _plan_injection("sibling_dependency", ctx, i, (ErrorType.DEPENDENT_SIBLING_TASKS,), f"T{i + 1} 依赖 T{j + 1} 的结果（显式写法）",
                           tasks, depends_on=f"T{j + 1}")


def sibling_dependency_paraphrase(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    tasks = _tasks(ctx.decision)
    if len(tasks) < 2:
        return None
    i = rng.randrange(len(tasks))
    tasks[i]["task"] = str(tasks[i].get("task") or "") + PARAPHRASED_DEPENDENCY
    return _plan_injection("sibling_dependency_paraphrase", ctx, i, (ErrorType.DEPENDENT_SIBLING_TASKS,),
                           f"T{i + 1} 依赖兄弟任务的结果（换说法，不含 W 编号 / “另一任务”）", tasks)


def fabricate_material(ctx: PlanContext, rng: random.Random) -> UnitInjection | None:
    tasks = _tasks(ctx.decision)
    if not tasks:
        return None
    i = rng.randrange(len(tasks))
    fake = "E9%03d" % rng.randint(0, 999)
    while ctx.log.lookup.has(fake):
        fake = "E9%03d" % rng.randint(0, 999)
    tasks[i]["materials"] = [*(tasks[i].get("materials") or []), fake]
    return _plan_injection("fabricate_material", ctx, i, (ErrorType.ILLEGAL_ACTION,), f"T{i + 1} materials 追加不存在的 {fake}", tasks, fake=fake)


PLAN_MUTATORS: dict[str, PlanMutator] = {
    "smuggle_premise": smuggle_premise, "duplicate_sibling": duplicate_sibling, "retarget_covered": retarget_covered,
    "sibling_dependency": sibling_dependency, "sibling_dependency_paraphrase": sibling_dependency_paraphrase,
    "fabricate_material": fabricate_material,
}
KINDS = [*ANSWER_MUTATORS, *PLAN_MUTATORS]
_RANK = {"positive": 0, "uncertain": 1, "negative": 2}


# --------------------------------------------------------------------------- 单元核验（与 complete_units 同一条链）

@dataclass
class RunUnits:
    """一个 run 上跑单元核验需要的东西（只算一次）。"""

    log: RunLog
    case: Case | None
    reports: list[Verdict]
    useful: UsefulEvidence
    covered_before: dict[str, set[str]]         # plan request → 本轮前已覆盖
    covered_all: set[str]
    last_round: int


def prepare_run(log: RunLog, case: Case | None, base_verdicts: list[Verdict]) -> RunUnits:
    reports = [recalibrate(v) for v in base_verdicts if v.subject.get("run") == log.name and v.unit == "report"]
    units, useful = verify_units(log, case, reports)
    covered_before = {str(u.subject.get("request")): set((u.extra.get("gap") or {}).get("covered_before") or []) for u in units if u.unit == "plan"}
    covered_all: set[str] = set()
    if case is not None:
        for reqs in useful.coverage_by_workspace(case).values():
            covered_all |= reqs
    rounds = [int(d.round_index or 1) for d in log.decisions if d.role == "GIA" and d.kind == "plan"]
    return RunUnits(log, case, reports, useful, covered_before, covered_all, max(rounds) if rounds else 1)


async def verify_unit(ru: RunUnits, decision: Decision, judge: JsonJudge | None, *, judge_name: str | None,
                      condition: dict[str, Any] | None = None) -> Verdict:
    """对（可能改过的）GIA 决定跑与正式流程相同的单元核验：程序层 +（有 judge 时）模型层 + 规则校准。"""

    base = {"run": ru.log.name, "case_id": ru.log.case_id, "teacher": ru.log.teacher_model}
    if decision.kind == "answer":
        verdict = check_answer(decision, ru.log, ru.case, ru.useful, ru.reports, subject=base)
        if judge is not None:
            sentences, result = await anchor_answer(decision, ru.log, ru.useful, judge)
            apply_answer_anchoring(verdict, sentences, result, judge_model=judge_name)
        return recalibrate(verdict)
    condition = condition or {}
    if condition.get("round_index") is not None:
        decision = dataclasses.replace(decision, round_index=int(condition["round_index"]))
    covered = set(condition["covered_before"]) if condition.get("covered_before") is not None else ru.covered_before.get(decision.request_id, set())
    verdict = check_plan(decision, ru.log, ru.case, ru.useful, covered_before=covered, subject=base)
    if judge is not None:
        raw = await judge_plan(decision, ru.case, covered, judge, ru.log)
        apply_plan_judgment(verdict, raw, case=ru.case, judge_model=judge_name)
    return verdict


def mutate(decision: Decision, output: dict[str, Any]) -> Decision:
    return dataclasses.replace(decision, output=output)


# --------------------------------------------------------------------------- 打分

def unit_error_key(e: ErrorRecord) -> tuple[Any, ...]:
    """单元错误的比对键。程序层错误按 (类型, detail)（detail 由输入确定，任务编号 / E 编号都在里面）；模型错误按
    (类型, span, 所引 E, 任务)——答案句编号 A{n} 会因删句 / 插句整体平移，不能进键。"""

    if e.method == "exec":
        return e.type, e.detail
    return e.type, e.span or "", tuple(e.evidence_ids), str(e.extra.get("task") or "")


def unit_new_errors(verdict: Verdict, control: Verdict | None) -> list[ErrorRecord]:
    known = {unit_error_key(e) for e in control.errors} if control else set()
    return [e for e in verdict.errors if e.extra.get("confidence") != "flag" and unit_error_key(e) not in known]


def unit_detected(injection: UnitInjection, verdict: Verdict, control: Verdict | None) -> tuple[bool, list[str]]:
    hits: list[str] = []
    for e in unit_new_errors(verdict, control):
        if e.type not in injection.expected:
            continue
        task = e.extra.get("task")
        if injection.unit == "plan" and task and injection.target and task != injection.target:
            continue
        hits.append(e.type.value)
    return bool(hits), sorted(set(hits))


def new_flags(verdict: Verdict, control: Verdict | None, expected: tuple[ErrorType, ...]) -> list[str]:
    """注入后新出现、类型在期望内、但只是 flag（不计命中）的错误——judge=none 时程序层"看见了但不敢定"的那部分。"""

    known = {unit_error_key(e) for e in control.errors} if control else set()
    return sorted({e.type.value for e in verdict.errors if e.extra.get("confidence") == "flag" and e.type in expected and unit_error_key(e) not in known})


def plan_unit_injections(ru: RunUnits, *, kinds: list[str], per_kind: int, seed: int) -> list[UnitInjection]:
    rng = random.Random(seed)
    out: list[UnitInjection] = []
    finding_evidence = _finding_evidence(ru.log, ru.useful)
    for decision in ru.log.decisions:
        if decision.role != "GIA" or decision.kind not in ("answer", "plan"):
            continue
        for kind in kinds:
            made, attempts = 0, 0
            seen: set[str] = set()
            while made < per_kind and attempts < per_kind * 4:
                attempts += 1
                inj: UnitInjection | None = None
                if decision.kind == "answer" and kind in ANSWER_MUTATORS:
                    answer = str(decision.output.get("answer") or "")
                    ctx = AnswerContext(decision, ru.log, ru.case, ru.useful, finding_evidence)
                    inj = ANSWER_MUTATORS[kind](ctx, answer, answer_blocks(answer), rng)
                elif decision.kind == "plan" and kind in PLAN_MUTATORS:
                    pctx = PlanContext(decision, ru.log, ru.case, ru.covered_before.get(decision.request_id, set()), ru.covered_all, ru.last_round)
                    inj = PLAN_MUTATORS[kind](pctx, rng)
                if inj is None:
                    break
                sig = json.dumps(inj.output, ensure_ascii=False, sort_keys=True) + json.dumps(inj.condition, sort_keys=True)
                if sig in seen:
                    continue          # 同一决定上只有一种改法的注入（flip_label 等）不重复计
                seen.add(sig)
                out.append(inj)
                made += 1
    return out


async def run_units_calibration(run_dirs: list[Path], cases: dict[str, Case], base_verdicts: list[Verdict], judge: JsonJudge | None, *,
                                judge_name: str | None, kinds: list[str], per_kind: int, seed: int,
                                say: Callable[[str], None] = print) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    controls: dict[str, dict[str, Any]] = {}
    aborted: str | None = None
    abort_message = ""
    try:
        for run_dir in run_dirs:
            await _calibrate_run(run_dir, cases, base_verdicts, judge, rows, controls, judge_name=judge_name, kinds=kinds,
                                 per_kind=per_kind, seed=seed, say=say)
    except JudgeBudgetExhausted as exc:
        aborted = "budget_exhausted"
        say(budget_message(exc))
        abort_message = str(exc)
    summary = summarize_units(rows, judge_name=judge_name, kinds=kinds)
    summary["baseline"] = "control: 同一 GIA 决定未改原样、同条件（round / covered_before）/ 同 judge 下的单元判定"
    summary["controls"] = controls
    if aborted:
        summary["aborted"] = aborted
        summary["aborted_message"] = abort_message
    return summary


async def _calibrate_run(run_dir: Path, cases: dict[str, Case], base_verdicts: list[Verdict], judge: JsonJudge | None,
                         rows: list[dict[str, Any]], controls: dict[str, dict[str, Any]], *, judge_name: str | None, kinds: list[str],
                         per_kind: int, seed: int, say: Callable[[str], None]) -> None:
    log = load_run(run_dir)
    try:
        if log.origin == "D":
            say(f"[{log.name}] 离线换题（来源 D），跳过单元召回注入")
            return
        ru = prepare_run(log, cases.get(log.case_id or ""), base_verdicts)
        injections = plan_unit_injections(ru, kinds=kinds, per_kind=per_kind, seed=seed)
        say(f"[{log.name}] {len(injections)} 处单元注入")
        by_request = {d.request_id: d for d in log.decisions}
        control_cache: dict[str, Verdict] = {}
        for inj in injections:
            decision = by_request[inj.request]
            ckey = inj.request + json.dumps(inj.condition, sort_keys=True)
            if ckey not in control_cache:
                control_cache[ckey] = await verify_unit(ru, decision, judge, judge_name=judge_name, condition=inj.condition)
                c = control_cache[ckey]
                controls[f"{log.name}/{inj.request}/{json.dumps(inj.condition, sort_keys=True)}"] = {
                    "run": log.name, "request": inj.request, "unit": inj.unit, "condition": inj.condition, "disposition": c.disposition,
                    "control_error_types": sorted({e.type.value for e in c.errors if e.extra.get("confidence") != "flag"})}
            control = control_cache[ckey]
            verdict = await verify_unit(ru, mutate(decision, inj.output), judge, judge_name=judge_name, condition=inj.condition)
            ok, hits = unit_detected(inj, verdict, control)
            new_types = sorted({e.type.value for e in unit_new_errors(verdict, control)})
            pending = list(verdict.pending) + ([] if judge is not None else [{"reason": "no_judge", "note": "模型层未运行"}])
            reasons, last_error = pending_reasons(pending)
            rows.append({**inj.to_dict(), "run": log.name, "detected": ok, "hits": hits, "new_error_types": new_types,
                         "rejected": _RANK.get(verdict.disposition, 1) > _RANK.get(control.disposition, 1),
                         "new_flags": new_flags(verdict, control, inj.expected),
                         "disposition": verdict.disposition, "control_disposition": control.disposition,
                         "pending": len(pending), "pending_reasons": reasons, "pending_last_error": last_error,
                         "control_error_types": sorted({e.type.value for e in control.errors if e.extra.get("confidence") != "flag"}),
                         "errors": [{"type": e.type.value, "severity": e.severity.value, "method": e.method,
                                     "confidence": e.extra.get("confidence"), "span": (e.span or "")[:80], "detail": e.detail[:160]}
                                    for e in verdict.errors]})
            mark = "命中" if ok else ("未定" if pending else "漏报")
            say(f"  {inj.unit:6s} {inj.kind:30s} {inj.target or '-':4s} {mark} {hits or new_types} — {inj.detail}")
    finally:
        log.close()


def summarize_units(rows: list[dict[str, Any]], *, judge_name: str | None, kinds: list[str]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        recall, inconclusive, recall_conclusive = _recall(sub)
        by_kind[kind] = {"unit": sub[0]["unit"] if sub else ("answer" if kind in ANSWER_MUTATORS else "plan"),
                         "n": len(sub), "detected": sum(1 for r in sub if r["detected"]),
                         "recall": recall, "inconclusive": inconclusive, "recall_conclusive": recall_conclusive,
                         "flag_only": sum(1 for r in sub if not r["detected"] and r.get("new_flags")),
                         "rejected": sum(1 for r in sub if r.get("rejected")),
                         "misses": [{"run": r["run"], "request": r["request"], "target": r["target"], "detail": r["detail"],
                                     "got": r["new_error_types"], "inconclusive": is_inconclusive(r)} for r in sub if not r["detected"]]}
    by_unit: dict[str, dict[str, Any]] = {}
    for unit in ("answer", "plan"):
        sub = [r for r in rows if r["unit"] == unit]
        recall, inconclusive, recall_conclusive = _recall(sub)
        by_unit[unit] = {"n": len(sub), "detected": sum(1 for r in sub if r["detected"]), "recall": recall,
                         "inconclusive": inconclusive, "recall_conclusive": recall_conclusive}
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        for t in r["expected"]:
            slot = by_type.setdefault(t, {"expected_in": 0, "hit": 0})
            slot["expected_in"] += 1
            if t in r["hits"]:
                slot["hit"] += 1
    recall, inconclusive, recall_conclusive = _recall(rows)
    return {"judge_model": judge_name, "layers": "exec+hindsight+judge(answer 逐句锚定 / plan 评判)+calibrate" if judge_name else "exec+hindsight only",
            "n_injections": len(rows), "recall_overall": recall, "inconclusive": inconclusive, "recall_conclusive": recall_conclusive,
            "note_inconclusive": "inconclusive = 未命中且有 pending（judge=none 时模型层未运行也算）；recall_conclusive 的分母去掉这些",
            "by_kind": by_kind, "by_unit": by_unit, "by_expected_type": by_type,
            "false_new_errors": [{"run": r["run"], "request": r["request"], "kind": r["kind"], "types": [t for t in r["new_error_types"] if t not in r["expected"]]}
                                 for r in rows if [t for t in r["new_error_types"] if t not in r["expected"]]],
            "rows": rows}


def load_verdicts(path: str | Path | None) -> list[Verdict]:
    if not path or not Path(path).is_file():
        return []
    return [Verdict.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="plan / answer 单元的召回率校准（注入已知错误再核验）")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cases", default="experiments/sft_data/cases")
    parser.add_argument("--verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl", help="报告级判定（算干净 Finding / 覆盖用）")
    add_judge_arguments(parser, default="none")
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl")
    parser.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    parser.add_argument("--per-kind", type=int, default=2, help="每个 GIA 决定每种注入最多几处")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    judge = judge_from_args(args, cache_path=args.cache or None)
    judge_name = getattr(judge, "name", None) if judge is not None else None
    cases = load_cases(args.cases) if Path(args.cases).is_dir() else {}
    summary = asyncio.run(run_units_calibration([Path(r) for r in args.runs], cases, load_verdicts(args.verdicts), judge, judge_name=judge_name,
                                                kinds=args.kinds, per_kind=args.per_kind, seed=args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"单元召回 {summary['recall_overall']}（{summary['n_injections']} 处注入）；inconclusive {summary['inconclusive']}，"
          f"conclusive 召回 {summary['recall_conclusive']} → {out}")
    for kind, row in summary["by_kind"].items():
        print(f"  {row['unit']:6s} {kind:30s} {row['detected']}/{row['n']}  inconclusive {row['inconclusive']}  flag_only {row['flag_only']}")
    line = judge_summary(judge)
    if line:
        print(line, flush=True)
    if summary.get("aborted"):
        print(budget_message(summary.get("aborted_message") or "budget exhausted"), flush=True)
        return BUDGET_EXIT_CODE
    return 0


__all__ = ["ANSWER_MUTATORS", "FAKE_PREMISES", "KINDS", "PLAN_MUTATORS", "UnitInjection", "answer_blocks", "load_verdicts", "main",
           "plan_unit_injections", "prepare_run", "run_units_calibration", "summarize_units", "unit_detected", "unit_error_key",
           "unit_new_errors", "verify_unit"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
