"""核验结论 Verdict 与三层编排 verify_report()。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..cases import Case
from ..errors import SPECS, ErrorRecord, Severity, tally
from .anchored import JsonJudge, verify_anchored
from .context import EvidenceLookup
from .exec_checks import exec_checks
from .sufficiency import verify_sufficiency

Disposition = Literal["positive", "negative", "uncertain"]
VERIFIER_VERSION = "verifier@0.4.0"
# 0.3.6 → 0.4.0（2026-10-17，r1 预演之一，依据 r0g2 / r0a 金标与 r0g 充分性理由表，见 claude/r1核验器修正_20261017.md）：
#   0. 缓存键带核验器版本（judges.cache_namespace），0.4.0 的题一律不命中 0.3.x 的旧回答；
#   1. 充分性层不给 GT（sufficiency.py）：标注来源 / 网址指定的参考来源不进清单，改给案卷要求（REQ + must_cite）与证据索引；
#   2. 充分性层：failed 页、任务没要求的精度（B 族逐段帧精确边界 / 成对帧）只算 limitation，不算缺口；
#   3. 像素检查只对句子所指的那张图（anchored.py：图片对带同句其他图的索引；程序规则 tool_measurement / other_image_referent /
#      cited_unrelated_image）；
#   4. 第二层引语回查认省略号分段与短词（OCR 的 "ISSF" "2024"）；覆盖汇总带任务文字（标题、posted 日期、声明原文可作依据）；
#   5. sloppy_numeric：写明"页面声明的发布日期"且与 dates 卡一致的日期，不按页面正文另一个日期判错（declared_date）；
#   6. attribution_drift 看来源支持的那一部分本身有没有写成转述（r0a F001）。
# 0.3.5 → 0.3.6（2026-10-05，v1p2_hard 金标采纳后默认打开，见 claude/程序规则与逐帧差异对照_20261004.md）：
#   ① 第二层程序规则 task_given / back_reference / verdict_sentence（anchored.PROGRAM_RULES）；
#   ② 推理核对（inference.py）：实锤"无支持"片段若能由同单元已核实句子直接推出、或只是被反驳的引述，降为 minor；
#   ③ 语义覆盖（semantic_coverage.py）：答案 premature_finish 与报告"必需证据没取到"类错误，要求若已由已核验片段建立则不再计。
#   都只把错误降为 minor 并留痕（extra.program_rule），不删错误。FOREN_VERIFIER_CANDIDATES=none 可回到 0.3.5 行为。
# 0.3.4 → 0.3.5（2026-10-04，r0cn 的 averimatec_699 发现）：第一、三层缺项去重按证据组算。第一层对 any_one_of 组只记一条
#   "必需证据 a / b / c 未被任何 Finding 引用"（extra.keys 是整组），旧去重只认组里第一个 key，第三层对同组其余 key 各记一条
#   "任务要求 'b' 未被回答"（major），一组没取到就变成 2–5 个 major → negative。同样只缺一组的工作区，第三层碰巧没逐项列出的
#   就是 positive（r0c 的 true_20361105 W1-1/W1-2 负、W2-1 正）。只改合并规则，不调模型；第二层（anchored.FEATURES）不变。
# 0.3.3 → 0.3.4（2026-10-02，四层精确率实测 precision_r0c 发现）：第二层两处，见 anchored.FEATURES——
#   1. 交代调查范围 / 局限的句子不再被覆盖汇总记成无据断言（句中夹带的具体事实照常核）；只对命中 _LIMITATION 的句子换提示词；
#   2. 模型层 major/critical 错误独立复核一遍，没复现的降为 minor（extra.unconfirmed）。只给出了实锤错误的 Finding 多花调用。
#   把 anchored.FEATURES 两项都关掉 = 0.3.3 的行为（缓存键也与 0.3.3 逐字相同）。
# 0.3.2 → 0.3.3（2026-09-30，r0c 用 Sonnet 核验后发现的四处）：
#   1. 视频帧的图片对带 media_note（取帧时间、单帧时长、视频总长、首尾帧），"结尾帧""视频共 6 秒"不再因核验模型不知道总长被判无支持；
#   2. 来自输入材料的图片对带任务文字（只用来理解"材料A""给定视频"的指代），4126 里两段视频指代弄反的误判；
#   3. 一句话引了两张以上图片时，覆盖汇总附上这些图片（最多 8 张），跨帧比较（前后相同、B 比 A 多一段）不再必然判无支持；
#   4. answer 单元的案卷标签改为语义判定（label_judged.py）：等价说法算命中，反向结论记 critical；程序层子串匹配排除否定前缀
#      （"属实"不再命中"不属实""部分属实"）。
#   只在 payload 带新字段时换提示词：外部网页的文本对、单图汇总与旧版逐字相同，缓存照常命中；视频帧对与多图汇总会重判。
# 0.3.1 → 0.3.2（2026-09-30，Claude Sonnet 5.5 重判 r0b 时发现）：覆盖汇总的提示词让模型"拿不准的部分也列出来，但 grounded 只在确实
#   缺少支持时才为 false"，代码却把列出的每一部分都记成 critical 的 unsupported_assertion。模型列了部分但 grounded=true 时，
#   这些部分改记 minor（extra.uncertain=True），不计入处置。r0b 上 158 次覆盖汇总里 13 次属于这种情况，涉及 7 条错误、4 份报告。
# 0.3.0 → 0.3.1（2026-09-29，第二批案卷审稿发现、负责人 09-29 让本机完成）：组内全部 must_cite=False 的必需证据组 = 可选证据。
#   以前 evidence_groups() 把规则没提到的 key 各自当 all 组，exec_checks 对没满足的组一律记 hard omitted（MAJOR），两条就判
#   negative——media_profile / alignment / 旁证报道这类帮助性材料也成了必需。现在：exec_checks 与 sufficiency 对可选组只记
#   severity=minor 的 omitted（extra.optional=True），不计入处置、不转 uncertain、不污染 Finding；must_cite=True 的组照旧。
#   影响 21 个组（33 个案卷里 18 个案卷），旧判定的 negative 数会下降；规格 spec.json 里的 verifier 字段记的是当时产数据用的版本，不改。
# 0.2.0 → 0.3.0（2026-09-24，依据 09-23 召回实验）：充分性按工作区收窄——verify_report 接 requirement_ids（scope.py 算出的
#   本工作区任务对准的 REQ），第一层按范围用 url_pattern / acquired_by 程序核对必需证据（any_one_of 分组），第三层只把范围内的
#   必需证据交给模型；不再把整案清单压给单个工作区。
#   同版追加（2026-09-24，09-24 收窄测试发现）：第一层 exec_checks 不再对工作区报告跑 label_expected，整案标签只在 answer 单元核。
#   同版追加（2026-09-28，第二批案卷审稿发现）：interval_iou 也按 check.requirement 收窄，只查对准该 REQ 的工作区。
# 0.1.0 → 0.2.0（2026-09-21，依据 true:4124903 示例的五条修订）：
#   1. minor 错误不再传播污染（tainted 只看 critical / major）；
#   2. Verdict 增加 unit 字段：report | step | plan | answer；
#   3. calibrate.py 提供对已存判定的规则重算（scope 多证据豁免、关系句引用不全降级）。


@dataclass
class Verdict:
    subject: dict[str, Any]                       # {case_id, run, workspace_id, request, ...} 指向被核的那份报告
    unit: str = "report"                          # report | step | plan | answer（v2 §2 的四个核验单元）
    errors: list[ErrorRecord] = field(default_factory=list)
    disposition: Disposition = "uncertain"
    tainted_findings: list[str] = field(default_factory=list)   # 含传播型错误的 Finding
    pending: list[dict[str, Any]] = field(default_factory=list)  # 未能判定的项（缺模型、缺像素）
    layers_run: list[str] = field(default_factory=list)
    judgments: list[dict[str, Any]] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    verifier: str = VERIFIER_VERSION
    judge_model: str | None = None
    spec_hash: str | None = None
    state_renderer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)     # 单元专属信息：clue_gain / gap / tainted_input / calibration …

    @property
    def counts(self) -> dict[str, int]:
        return tally(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "unit": self.unit, "disposition": self.disposition, "errors": [e.to_dict() for e in self.errors],
            "tally": {k: v for k, v in self.counts.items() if v}, "tainted_findings": self.tainted_findings,
            "pending": self.pending, "layers_run": self.layers_run, "judgments": self.judgments, "coverage": self.coverage,
            "sufficiency": self.sufficiency, "verifier": self.verifier, "judge_model": self.judge_model,
            "spec_hash": self.spec_hash, "state_renderer": self.state_renderer, **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Verdict":
        return cls(subject=dict(value["subject"]), unit=str(value.get("unit") or "report"),
                   errors=[ErrorRecord.from_dict(e) for e in value.get("errors", [])],
                   disposition=value.get("disposition", "uncertain"), tainted_findings=list(value.get("tainted_findings", [])),
                   pending=list(value.get("pending", [])), layers_run=list(value.get("layers_run", [])),
                   judgments=list(value.get("judgments", [])), coverage=list(value.get("coverage", [])),
                   sufficiency=dict(value.get("sufficiency", {})),
                   verifier=value.get("verifier", VERIFIER_VERSION), judge_model=value.get("judge_model"),
                   spec_hash=value.get("spec_hash"), state_renderer=value.get("state_renderer"), extra=dict(value.get("extra", {})))


def merge_sufficiency_errors(errors: list[ErrorRecord], extra: list[ErrorRecord]) -> list[ErrorRecord]:
    """第三层错误并进第一、二层：状态级错误按类型去重，缺项按 (类型, key) 去重。

    verifier@0.3.5：第一层的必需证据错误带 extra.keys（整组 key），组里每个 key 都算已记过——any_one_of 组没取到只算一条缺项。
    """

    def key_of(e: ErrorRecord) -> tuple[Any, ...]:
        if SPECS[e.type].level.value == "status":
            return (e.type,)
        return (e.type, e.extra.get("key") or e.extra.get("requirement"))

    out = list(errors)
    seen = {key_of(e) for e in out}
    for e in out:
        for k in e.extra.get("keys") or ():
            seen.add((e.type, k))
    for e in extra:
        if key_of(e) not in seen:
            out.append(e)
            seen.add(key_of(e))
    return out


def decide(errors: list[ErrorRecord], pending: list[dict[str, Any]]) -> Disposition:
    """处置规则：任一 critical 定错 → negative；≥2 个 major 定错 → negative；有未决项或仅有 flag → uncertain；否则 positive。"""

    firm = [e for e in errors if e.extra.get("confidence") != "flag"]
    if any(e.severity is Severity.CRITICAL for e in firm):
        return "negative"
    if sum(1 for e in firm if e.severity is Severity.MAJOR) >= 2:
        return "negative"
    if pending or any(e.extra.get("confidence") == "flag" for e in errors):
        return "uncertain"
    return "positive"


def tainted(errors: list[ErrorRecord]) -> list[str]:
    """含传播型错误的 Finding。0.2.0 起 minor 不传播：一条 scope_overreach(minor) 不该把整条 Finding 拖成污染。"""

    out: list[str] = []
    for e in errors:
        if e.extra.get("confidence") == "flag" or not SPECS[e.type].propagates or e.severity is Severity.MINOR:
            continue
        finding = e.extra.get("finding")
        if isinstance(finding, str) and finding not in out:
            out.append(finding)
    return out


async def verify_report(report: dict[str, Any], lookup: EvidenceLookup, *, subject: dict[str, Any], case: Case | None = None,
                        task_text: str | None = None, judge: JsonJudge | None = None, visible_ids: set[str] | None = None,
                        judge_model: str | None = None, spec_hash: str | None = None, state_renderer: str | None = None,
                        requirement_ids: set[str] | None = None) -> Verdict:
    """三层顺序执行：exec → anchored → sufficiency。没有 judge 就只跑第一层，flag 留着，处置为 uncertain。

    requirement_ids：这份报告所属工作区对准的案卷要求（scope.requirements_for_workspace）；None = 不收窄（整案）。
    """

    verdict = Verdict(subject=dict(subject), judge_model=judge_model, spec_hash=spec_hash, state_renderer=state_renderer)
    if requirement_ids is not None:
        verdict.extra["scope"] = sorted(requirement_ids)
    first = exec_checks(report, lookup, case=case, visible_ids=visible_ids, requirement_ids=requirement_ids)
    verdict.layers_run.append("exec")
    errors: list[ErrorRecord] = list(first)
    if judge is not None:
        skip = {eid for e in first for eid in e.evidence_ids if e.type.value == "fabricated_evidence_id"}
        text = task_text if task_text is not None else (case.task_text if case else None)
        anchored = await verify_anchored(report, lookup, judge, skip_evidence_ids=skip, task=text)
        verdict.layers_run.append("anchored")
        judged = {j.finding_id for j in anchored.judgments}
        # 第二层看过的 Finding，第一层的 flag 由模型判定取代
        errors = [e for e in errors if not (e.extra.get("confidence") == "flag" and e.extra.get("finding") in judged)]
        errors.extend(anchored.errors)
        verdict.pending.extend(anchored.pending)
        verdict.judgments = [j.to_dict() for j in anchored.judgments]
        verdict.coverage = anchored.coverage
        if text:
            sufficiency = await verify_sufficiency(report, text, judge, case=case, requirement_ids=requirement_ids)
            verdict.layers_run.append("sufficiency")
            errors = merge_sufficiency_errors(errors, sufficiency.errors)
            verdict.sufficiency = sufficiency.raw
    else:
        verdict.pending.append({"reason": "no_judge", "note": "第二、三层未运行"})
    verdict.errors = errors
    verdict.tainted_findings = tainted(errors)
    verdict.disposition = decide(errors, verdict.pending)
    if judge is not None and case is not None:
        from .semantic_coverage import apply_report_semantic_coverage    # 候选，默认关闭；循环引用所以局部导入
        await apply_report_semantic_coverage(verdict, case, judge)
    return verdict


def write_verdicts(path: str | Path, verdicts: list[Verdict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
    return target


def read_verdicts(path: str | Path) -> list[Verdict]:
    return [Verdict.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["VERIFIER_VERSION", "Disposition", "JsonJudge", "Verdict", "decide", "read_verdicts", "tainted", "verify_report", "write_verdicts"]
