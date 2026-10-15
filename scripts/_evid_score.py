"""EVID-Bench 评分（2026-10-04）：ForenSearcher 的 FINISH 答案 → 伪造点 → 官方三评委逐 GT 投票。

    python scripts/_evid_score.py --round evidb0                       # 默认三个本机 Ollama 评委
    python scripts/_evid_score.py --round evidb0 --parse-only          # 不调评委，只看答案能否切成点、run 完成情况
    python scripts/_evid_score.py --round evidb0 --judges claude-cli:sonnet local:gemma3-12b-16k:latest

协议照搬官方仓库 yutao1024/EVID-Bench（commit 6f4237b，src/agent/judge.py 与 prompts.py 的 PROMPT_JUDGE_POINTS）：
- 每个评委看 GT 3–5 点与预测点，一对一最优匹配，逐 GT 给 verdict 0/1；
- 每个 GT 点按评委多数票定通过：pass_votes >= 评委数 / 2（官方写法；三评委即 ≥2，两评委即 ≥1）；
- 点级准确率 = 通过的 GT 点 / 全部 GT 点；视频级准确率 = 全部 GT 点都通过的视频 / 全部视频。
与官方的差别（写进汇总，别拿去和论文表格直接比）：
1. 评委：论文用 GPT-5.4 / Gemini-3.1-Pro / Claude Sonnet 4.6；这里默认三个本机 Ollama 模型。
   其中 qwen3vl-8b 同时是被测模型（自评），汇总里单列每个评委的分数，便于看偏差。
2. 提示词：官方把 GT / PRED 两段直接拼进提示；这里评委接口固定把数据放在 user 消息里，所以系统提示是官方提示
   去掉这两段、改成"两段列表在 user 消息里"，评分规则与输出 JSON 格式逐字不变。
3. 预测点来自 GIA 的 FINISH 答案文本：按编号列表切分（案卷任务要求编号列表），切不出就按空行分段，再不行整段算一点
   （一对一匹配下整段只能匹配一个 GT，等于格式不对就吃亏；parse_mode 逐条记录）。

分母一律是全部计划样本：没跑完 / 没有 FINISH 的样本按 0 分计，同时另报"只算完成的 run"的口径。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evid_behavior import (DEFAULT_LOGS, answer_language, behavior_markdown, funnel_markdown, run_behavior,  # noqa: E402
                            source_funnel, summarize_behavior, summarize_funnel)

DEFAULT_DATASET = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench"
DEFAULT_RUNS = ROOT / "experiments" / "evid" / "runs"
DEFAULT_CASES = ROOT / "experiments" / "evid" / "cases"
DEFAULT_OUT = ROOT / "experiments" / "evid" / "scores"
DEFAULT_MANIFESTS = ROOT / "experiments" / "sft_data" / "run_manifests"
DEFAULT_JUDGES = ("local:gemma3-12b-16k:latest", "local:qwen3-8b-16k:latest", "local:qwen3vl-8b-32k:latest")
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
OFFICIAL_COMMIT = "6f4237b0941112da546d78cd660e3a71acf2c48d"

# 官方 PROMPT_JUDGE_POINTS，评分规则与输出格式逐字照抄；只把末尾 GT / PREDICTION 两段换成"在 user 消息里"。
JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator for video forgery analysis. You will be given a ground-truth (GT) list of 3-5 forgery points and a predicted list of 3-5 forgery points. For each GT point, decide whether any prediction matches it.

Important framing:
- GT points and predictions may be written at different abstraction levels.
- Some items are narrative-level accusations: the edit constructs a false moral, political, social, or causal interpretation.
- Some items are event-level findings: the edit splices, reorders, re-contextualizes, or rebinds specific scenes, audio, or actions.
- A good prediction may express the same core misleading point either directly at the narrative level or indirectly through concrete event evidence.

Scoring rules (important; follow strictly):
1. Each GT point can be matched at most once. Use a one-to-one best assignment.
2. Evaluate every GT/pred pair on two dimensions:
   - Dimension A: manipulation method. Examples: splicing unrelated clips, reordering scenes, removing narration, binding unrelated audio, selective omission, context switching.
   - Dimension B: misleading point / false reframing. Examples: portraying someone as rule-breaking, constructing a negative personality arc, implying opportunistic recruitment, or creating a moral-judgment framing.
3. Dimension B is primary, but judge it flexibly across abstraction levels:
   - Direct narrative match: If the prediction states substantially the same false reframing or accusation as the GT, mark `matched_dim_misleading=true` and `verdict=1`.
   - Event-to-narrative support: If the GT is narrative-level, and the prediction identifies the concrete edited event(s) that materially support that same narrative reframing, you may still mark `matched_dim_misleading=true` and `verdict=1` even if the prediction is more concrete and less interpretive.
4. "Roughly consistent" allows paraphrase, synonymy, wording differences, partial compression of multiple scenes into one point, and different abstraction levels, as long as the core misleading implication is aligned.
5. Be conservative about over-crediting:
   - Do not give credit just because both items mention the same event.
   - Do not give credit just because both items describe editing manipulation in general.
   - Do give credit when the prediction captures the same accusation, reframing, or a concrete event-level mechanism that clearly underpins that accusation.
6. Report `matched_dim_method` truthfully even when `verdict=1` is driven mainly by Dimension B.
7. Every GT point should receive the best available predicted match. If multiple GT points compete for the same prediction, choose the assignment that gives the best overall scoring.
8. Output strict JSON only. Do not use markdown fences.

The GT list (3-5 points, labelled GT-1, GT-2, ...) and the PREDICTION list (labelled PRED-1, PRED-2, ...) are given in the user message.

Output JSON:
{
  "hits": <integer from 0 to number_of_gt_points>,
  "score": <hits / number_of_gt_points as a float>,
  "matches": [
    {
      "gt_idx": <0-based GT index>,
      "pred_idx": <0-based prediction index or null if no match>,
      "verdict": <0|1>,
      "matched_dim_method": <true|false>,
      "matched_dim_misleading": <true|false>,
      "reason": "One concise English sentence explaining whether the method matches, whether the misleading point matches, and the key mismatch if any."
    }
  ],
  "comment": "1-2 concise English sentences summarizing the biggest strength and biggest weakness of the predictions."
}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "hits": {"type": "integer"}, "score": {"type": "number"},
        "matches": {"type": "array", "items": {"type": "object", "properties": {
            "gt_idx": {"type": "integer"}, "pred_idx": {"type": ["integer", "null"]}, "verdict": {"type": "integer"},
            "matched_dim_method": {"type": "boolean"}, "matched_dim_misleading": {"type": "boolean"},
            "reason": {"type": "string"}}, "required": ["gt_idx", "pred_idx", "verdict", "reason"]}},
        "comment": {"type": "string"},
    },
    "required": ["hits", "matches"],
}


# ---------------------------------------------------------------- 答案 → 预测点

_CITATION = re.compile(r"\s*[\[(（]\s*(?:[EFR]\d+)(?:\s*[,，;；/、]\s*(?:and\s+)?[EFR]\d+)*\s*[\])）]")
_NUMBER_LINE = re.compile(r"^[ \t]*(?:\*\*)?(\d{1,2})\s*[.)、:：](?:\*\*)?\s+", re.M)   # 行首编号（优先）
# 一行里连写的 1) … 2) …；前面可以是空白或句读（10-05 全量：中文答案常写成 "…；2. …"，旧规则整段算一点）
_NUMBER_INLINE = re.compile(r"(?:(?<=[\s；;。，,！!？?])|^)(\d{1,2})\s*[.)、:：]\s*(?=\D)", re.M)


def strip_citations(text: str) -> str:
    """去掉 [E12] / (E12, E13) 这类证据编号：评委比的是伪造点内容，编号只是噪声。"""

    return re.sub(r"[ \t]{2,}", " ", _CITATION.sub("", text)).strip()


def split_points(answer: str) -> tuple[list[str], str]:
    """FINISH 答案 → (预测点列表, parse_mode)。

    numbered：找 1. 2. 3. … 连续递增的编号（行首或空白之后），按编号切；编号前的引言丢掉。
    paragraphs：没有编号时按空行分段（≥2 段）。
    whole：都不行就整段算一点。empty：答案为空。
    """

    text = (answer or "").strip()
    if not text:
        return [], "empty"
    starts: list[tuple[int, int]] = []          # (编号开始位置, 正文开始位置)
    for pattern in (_NUMBER_LINE, _NUMBER_INLINE):   # 先认行首编号，正文里的 "at 2. 5 s" 不会被当成第 2 点
        starts, expected = [], 1
        for match in pattern.finditer(text):
            if int(match.group(1)) == expected:
                starts.append((match.start(), match.end()))
                expected += 1
        if len(starts) >= 2:
            break
    if len(starts) >= 2:
        points = []
        for i, (_, body_start) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            point = strip_citations(" ".join(text[body_start:end].split()))
            if point:
                points.append(point)
        return points, "numbered"
    paragraphs = [strip_citations(" ".join(p.split())) for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs, "paragraphs"
    return [strip_citations(" ".join(text.split()))], "whole"


def numbered_block(points: list[str], label: str) -> str:
    return "\n".join(f"[{label}-{i + 1}] {p}" for i, p in enumerate(points))


# ---------------------------------------------------------------- run 目录

def run_name(sample_id: str, round_tag: str, sample: int = 1) -> str:
    from foren_searcher.datagen.run_cases import _safe

    return f"evid_{_safe(sample_id)}_{_safe(round_tag)}_{sample}"


def read_run(run_dir: Path, sources: list[str]) -> dict[str, Any]:
    """一个 run 的状态、答案、是否碰到过原始素材（source YouTube ID 出现在轨迹 / 引用证据里）。"""

    info: dict[str, Any] = {"run": run_dir.name, "status": "missing", "decision": None, "answer": "",
                            "source_in_trajectory": [], "source_in_cited_evidence": [],
                            "rejected_finish_answer": "", "rejected_finish_request": None, "contract_flags": []}
    if not run_dir.is_dir():
        return info
    result_file = run_dir / "result.json"
    done = result_file.is_file() and (run_dir / "execution_links.json").is_file()
    info["status"] = "done" if done else "incomplete"
    cited_text = ""
    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            decision = result.get("decision") or {}
            info["decision"] = decision.get("decision")
            info["answer"] = str(decision.get("answer") or "")
            info["contract_flags"] = list(result.get("contract_flags") or [])     # 评测兜底打的标（--runtime-guards eval）
            cited_text = json.dumps(result.get("cited_evidence") or [], ensure_ascii=False)
        except ValueError:
            info["status"] = "incomplete"
    trajectory = run_dir / "trajectory.jsonl"
    traj_text = trajectory.read_text(encoding="utf-8", errors="replace") if trajectory.is_file() else ""
    info["source_in_trajectory"] = [s for s in sources if s in traj_text]
    info["source_in_cited_evidence"] = [s for s in sources if s in cited_text]
    if info["decision"] != "FINISH":
        info["rejected_finish_answer"], info["rejected_finish_request"] = last_rejected_finish(run_dir)
    return info


def _answer_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def last_rejected_finish(run_dir: Path) -> tuple[str, str | None]:
    """没走完的 run 里 GIA 最后一次给出的 FINISH 答案（被框架格式检查拒收的那份）。

    10-05 全量：约 1/4 的 run 因 GIA 输出修复次数用完而失败，其中一半是"FINISH 答案没写 E 编号"——答案本身已经写了，
    只是不合框架的引用格式。EVID 官方评分只看答案文字。10-05 起正式口径把这份答案算进来（"系统最后写出的答案"）；
    只算框架接受的 FINISH 的严格口径另报一行（passed_gt_strict）。
    GIA 的回答带 decision 字段，HI 的不带，所以只看 response.json 就能分角色。
    """

    requests = run_dir / "requests"
    if not requests.is_dir():
        return "", None
    found: tuple[str, str | None] = ("", None)
    for directory in sorted(requests.glob("request_*")):
        response = directory / "response.json"
        if not response.is_file():
            continue
        raw = response.read_text(encoding="utf-8", errors="replace")
        if '"decision"' not in raw or "FINISH" not in raw:
            continue
        try:
            value = json.loads(raw)
        except ValueError:
            try:
                from foren_searcher.local_models import extract_json_object

                value = extract_json_object(raw)
            except Exception:  # noqa: BLE001  截断或坏 JSON 直接跳过
                continue
        if isinstance(value, dict) and value.get("decision") == "FINISH":
            answer = _answer_text(value.get("answer")).strip()
            if answer:
                found = (answer, directory.name)
    return found


# ---------------------------------------------------------------- 评委

def parse_judge(value: dict[str, Any], n_gt: int) -> list[dict[str, Any]]:
    """评委 JSON → 逐 GT 的 {verdict, pred_idx, reason}（官方 _judge_points_single 的解析：同一 gt_idx 取第一条）。"""

    by_gt: dict[int, dict[str, Any]] = {}
    for item in value.get("matches") or []:
        if not isinstance(item, dict):
            continue
        try:
            gt_idx = int(item.get("gt_idx", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= gt_idx < n_gt or gt_idx in by_gt:
            continue
        try:
            verdict = 1 if int(item.get("verdict", 0)) >= 1 else 0
        except (TypeError, ValueError):
            verdict = 0
        pred = item.get("pred_idx")
        by_gt[gt_idx] = {"verdict": verdict, "pred_idx": pred if isinstance(pred, int) else None,
                         "matched_dim_method": bool(item.get("matched_dim_method", False)),
                         "matched_dim_misleading": bool(item.get("matched_dim_misleading", False)),
                         "reason": str(item.get("reason") or "").strip()}
    return [by_gt.get(i, {"verdict": 0, "pred_idx": None, "reason": "", "missing": True}) for i in range(n_gt)]


def vote(per_judge: dict[str, list[dict[str, Any]]], n_gt: int) -> list[dict[str, Any]]:
    """官方多数票：pass_votes >= 评委数 / 2。失败的评委按全 0 票计（官方同样如此）。"""

    threshold = len(per_judge) / 2.0
    out = []
    for i in range(n_gt):
        votes = {name: rows[i]["verdict"] if i < len(rows) else 0 for name, rows in per_judge.items()}
        passes = sum(votes.values())
        out.append({"gt_idx": i, "pass_votes": passes, "judge_count": len(per_judge), "passed": passes >= threshold,
                    "votes": votes})
    return out


def build_judges(specs: list[str], endpoint: str, cache_path: Path) -> dict[str, Any]:
    from foren_searcher.datagen.verify.judges import build_judge

    judges: dict[str, Any] = {}
    for spec in specs:
        kind, _, model = spec.partition(":")
        if kind not in ("local", "claude-cli") or not model:
            raise SystemExit(f"--judges 的写法是 local:<Ollama 模型名> 或 claude-cli:<sonnet|opus>，得到 {spec!r}")
        judge = build_judge(kind, model=model, cache_path=cache_path, endpoint=endpoint if kind == "local" else None)
        inner = getattr(judge, "inner", None)
        if inner is not None and hasattr(inner, "max_tokens"):
            inner.max_tokens = 8192       # qwen3:8b 会先输出思考再给 JSON，默认 4096 偶尔被截断
        judges[spec] = judge
    return judges


async def judge_one(judge: Any, gt: list[str], preds: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    payload = {"GT (3-5 points)": numbered_block(gt, "GT"), "PREDICTION (3-5 points)": numbered_block(preds, "PRED")}
    try:
        return await judge.judge(system_prompt=JUDGE_SYSTEM_PROMPT, payload=payload, images=[], schema=JUDGE_SCHEMA), None
    except Exception as exc:  # noqa: BLE001  评委失败不中断整批：记下来，按 0 票计
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"


# ---------------------------------------------------------------- 汇总

def summarize(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, Any]:
    def acc(subset: list[dict[str, Any]], key: str = "passed_gt") -> dict[str, Any]:
        value = lambda r: r.get(key, r["passed_gt"])  # noqa: E731
        n_gt = sum(r["n_gt"] for r in subset)
        passed = sum(value(r) for r in subset)
        videos = len(subset)
        full = sum(1 for r in subset if r["n_gt"] and value(r) == r["n_gt"])
        return {"videos": videos, "gt_points": n_gt, "passed_points": passed,
                "point_acc": round(passed / n_gt, 4) if n_gt else None,
                "video_acc": round(full / videos, 4) if videos else None,
                "mean_video_score": round(sum(value(r) / r["n_gt"] for r in subset if r["n_gt"]) / videos, 4) if videos else None}

    finished = [r for r in rows if r["decision"] == "FINISH"]
    attempted = [r for r in rows if r["status"] != "missing"]     # 已经开跑过的（含崩溃），全量跑到一半时看这一行
    per_judge = {}
    for name in judges:
        hits = sum(sum(v["votes"].get(name, 0) for v in r.get("gt_votes", [])) for r in rows)
        total = sum(r["n_gt"] for r in rows)
        failed = sum(1 for r in rows if name in (r.get("judge_errors") or {}))
        per_judge[name] = {"point_acc_alone": round(hits / total, 4) if total else None, "failed_calls": failed}
    by = lambda key: {k: acc([r for r in rows if r[key] == k]) for k in sorted({r[key] for r in rows})}  # noqa: E731
    return {
        "all_planned": acc(rows),
        "attempted": acc(attempted),
        "attempted_strict": acc(attempted, "passed_gt_strict"),
        "rejected_finish_used": sum(1 for r in rows if r.get("answer_source") == "rejected_finish"),
        "contract_flags": dict(Counter(flag for r in rows for flag in r.get("contract_flags") or [])),
        "finished_only": acc(finished),
        "run_status": dict(Counter(r["status"] for r in rows)),
        "decision": dict(Counter(str(r["decision"]) for r in rows)),
        "parse_mode": dict(Counter(r["parse_mode"] for r in rows)),
        "pred_points": dict(Counter(len(r["pred_points"]) for r in finished)),
        "source_in_trajectory": sum(1 for r in rows if r["source_in_trajectory"]),
        "source_in_cited_evidence": sum(1 for r in rows if r["source_in_cited_evidence"]),
        "answer_language": dict(Counter(r.get("answer_language", "none") for r in rows if r.get("answer_source") != "none")),
        "behavior": summarize_behavior([r["behavior"] for r in rows if r.get("behavior")]),
        # 10-06 找原片实验计划：①进池 ②下载 ③比对（按 GT source ID 查 store）；没开跑的样本不计
        "funnel": summarize_funnel([r["funnel"] for r in rows if r.get("funnel")]),
        "by_category": by("category"),
        "by_type": by("type"),
        "per_judge": per_judge,
    }


def markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    a, t, f = summary["all_planned"], summary["attempted"], summary["finished_only"]
    lines = [f"# EVID-Bench 评分：{meta['round']}", "",
             f"- 评委：{', '.join(meta['judges']) or '（--parse-only，未调评委）'}；多数票 pass_votes ≥ 评委数/2",
             f"- 已降级出 test、不计分的样本：{', '.join(meta.get('excluded_used_for_revision') or []) or '无'}",
             f"- 样本：{a['videos']} 个；run 状态 {summary['run_status']}；GIA 决定 {summary['decision']}",
             f"- 答案切分：{summary['parse_mode']}；FINISH 答案的点数分布 {summary['pred_points']}",
             f"- 原始素材 ID 出现在轨迹里 {summary['source_in_trajectory']} 个，出现在最终引用证据里 {summary['source_in_cited_evidence']} 个",
             f"- 答案语言（任务要求英文）：{summary.get('answer_language') or '无'}；非英文 "
             f"{sum(v for k, v in (summary.get('answer_language') or {}).items() if k not in ('en', 'none'))} 个（评委照常按内容评）",
             f"- 评测兜底打标（样本数）：{summary.get('contract_flags') or '无'}（finish_without_citation = 答案没引 E 但被接受；"
             "forced_* = GIA 耗尽后由运行时收尾）",
             "", "| 口径 | 视频数 | GT 点 | 通过 | 点级准确率 | 视频级准确率 |", "|---|---|---|---|---|---|",
             f"| 全部计划样本（未完成按 0） | {a['videos']} | {a['gt_points']} | {a['passed_points']} | {pct(a['point_acc'])} | {pct(a['video_acc'])} |",
             f"| 已开跑的样本（没写出答案的按 0；全量没跑完时看这行） | {t['videos']} | {t['gt_points']} | {t['passed_points']} | {pct(t['point_acc'])} | {pct(t['video_acc'])} |",
             f"| 只算有 FINISH 的 run | {f['videos']} | {f['gt_points']} | {f['passed_points']} | {pct(f['point_acc'])} | {pct(f['video_acc'])} |",
             (lambda r: f"| 严格口径：已开跑，只算框架接受的 FINISH（被拒收的 {summary['rejected_finish_used']} 个按 0） | "
                        f"{r['videos']} | {r['gt_points']} | {r['passed_points']} | {pct(r['point_acc'])} | {pct(r['video_acc'])} |")(
                 summary["attempted_strict"]),
             "", f"上面前两行含 GIA 写出但被框架格式检查拒收的 FINISH 答案（{summary['rejected_finish_used']} 个，多为没写 E 编号）；"
                 "EVID 只评答案文字，这类答案按官方协议照常评分。",
             "", "## 按类别", "", "| 类别 | 视频数 | 点级 | 视频级 |", "|---|---|---|---|"]
    for name, v in summary["by_category"].items():
        lines.append(f"| {name} | {v['videos']} | {pct(v['point_acc'])} | {pct(v['video_acc'])} |")
    lines += ["", "## 按伪造类型", "", "| 类型 | 视频数 | 点级 | 视频级 |", "|---|---|---|---|"]
    for name, v in summary["by_type"].items():
        lines.append(f"| {name} | {v['videos']} | {pct(v['point_acc'])} | {pct(v['video_acc'])} |")
    if summary["per_judge"]:
        lines += ["", "## 单个评委（不投票，单看它判通过的 GT 点比例）", "", "| 评委 | 点级 | 调用失败的样本 |", "|---|---|---|"]
        for name, v in summary["per_judge"].items():
            lines.append(f"| {name} | {pct(v['point_acc_alone'])} | {v['failed_calls']} |")
    behavior_rows = meta.get("behavior_rows") or []
    if summary.get("behavior") and behavior_rows:
        lines += ["", *behavior_markdown(summary["behavior"], behavior_rows,
                                         title="## 行为面板（P0.3：是\"没看到\"还是\"看到了没答对\"）")]
    funnel_rows = [(r["sample_id"], r["funnel"]) for r in meta.get("funnel_rows") or []]
    if summary.get("funnel") and funnel_rows:
        lines += ["", *funnel_markdown(summary["funnel"], [f for _, f in funnel_rows], [n for n, _ in funnel_rows])]
    lines += ["", "## 与论文口径的差别", "",
              "- 论文评委是 GPT-5.4 / Gemini-3.1-Pro / Claude Sonnet 4.6；这里是上面列的评委。",
              "- 被测模型若也在评委里就是自评，看单个评委那一栏判断偏差。",
              f"- 评分提示词取自官方仓库 commit {OFFICIAL_COMMIT[:7]}，GT / 预测两段改放在 user 消息里，规则与输出格式不变。",
              "- 论文参考（他们的 agent，官方评委）：GPT-5.5 点级 61.43% / 视频级 43.24%；Qwen3-VL-235B-A22B 点级 48.80%。"]
    return "\n".join(lines) + "\n"


def revised_ids(cases_dir: Path) -> set[str]:
    """案卷里 used_for_revision 非空的样本（看过轨迹并据此改过规格，cases.py 规则下已不算 test）。"""

    out: set[str] = set()
    for path in sorted(cases_dir.glob("evid_*.json")) if cases_dir.is_dir() else ():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if data.get("used_for_revision"):
            out.add(str(data.get("source_id")))
    return out


# ---------------------------------------------------------------- 主流程

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EVID-Bench 评分：官方三评委逐 GT 投票")
    parser.add_argument("--round", required=True, help="run_cases 的轮次标签（run 名 evid_<id>_<轮次>_1）")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--ids", nargs="+", default=None, help="只评这些样本（video_010 …）；默认全部 222 个")
    parser.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES),
                        help="local:<Ollama 模型名> 或 claude-cli:<sonnet|opus>，默认三个本机模型")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="local 评委的 OpenAI 兼容地址（Ollama 默认 11434）")
    parser.add_argument("--parse-only", action="store_true", help="不调评委：只出 run 完成情况与答案切分")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS), help="批量日志目录（行为面板读崩溃原因）")
    parser.add_argument("--cases", default=str(DEFAULT_CASES),
                        help="案卷目录：不给 --ids 时，used_for_revision 非空（已降级出 test）的样本不计分")
    args = parser.parse_args(argv)

    entries = json.loads((Path(args.dataset_root) / "evid_bench.json").read_text(encoding="utf-8"))
    excluded: list[str] = []
    if not args.ids:              # 正式口径：只算仍在 test 里的样本
        revised = revised_ids(Path(args.cases))
        excluded = sorted(e["id"] for e in entries if e["id"] in revised)
        entries = [e for e in entries if e["id"] not in revised]
    if args.ids:
        wanted = set(args.ids)
        entries = [e for e in entries if e["id"] in wanted]
        if missing := wanted - {e["id"] for e in entries}:
            raise SystemExit(f"evid_bench.json 里没有：{sorted(missing)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_specs = [] if args.parse_only else list(args.judges)
    judges = build_judges(judge_specs, args.endpoint, out_dir / "judge_cache.jsonl") if judge_specs else {}
    subset = ".subset" if args.ids else ""        # 指定样本的结果另存，不覆盖全量汇总
    tag = "parse-only" if args.parse_only else "+".join(re.sub(r"[^0-9a-zA-Z.]+", "-", s.split(":", 1)[1].split(":")[0]).lower()
                                                     for s in judge_specs)
    tag += subset
    rows_path = out_dir / f"{args.round}.{tag}.jsonl"
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, entry in enumerate(entries, 1):
            gt = [g for g in entry.get("groundtruth") or [] if str(g).strip()]
            task = entry.get("task") or {}
            info = read_run(Path(args.runs_dir) / run_name(entry["id"], args.round), list(entry.get("source") or []))
            source = "final"
            if info["decision"] == "FINISH":
                preds, mode = split_points(info["answer"])
            elif info["rejected_finish_answer"]:
                preds, mode = split_points(info["rejected_finish_answer"])
                source = "rejected_finish"
            else:
                preds, mode = [], "no_finish"
            used_answer = info["answer"] if source == "final" else info["rejected_finish_answer"]
            run_dir = Path(args.runs_dir) / run_name(entry["id"], args.round)
            row: dict[str, Any] = {"sample_id": entry["id"], "category": task.get("category"), "type": task.get("type"),
                                   "topic": entry.get("topic"), "n_gt": len(gt), **info, "parse_mode": mode,
                                   "answer_source": source if preds else "none",
                                   # P0.4（10-05）：任务要求英文作答；evidb5 的 027 写成了中文——评委仍按内容评，这里单独计数
                                   "answer_language": answer_language(used_answer) if preds else "none",
                                   # P0.3（10-05）：行为面板（不调评委，读轨迹与 transport；没开跑的样本为空）
                                   "behavior": run_behavior(run_dir, Path(args.logs_dir)) if run_dir.is_dir() else {},
                                   "funnel": source_funnel(run_dir, list(entry.get("source") or [])) if run_dir.is_dir() else {},
                                   "pred_points": preds, "passed_gt": 0, "passed_gt_strict": 0,
                                   "gt_votes": [], "judge_errors": {}}
            if judges and preds:
                per_judge: dict[str, list[dict[str, Any]]] = {}
                raw: dict[str, Any] = {}
                for name, judge in judges.items():
                    value, error = asyncio.run(judge_one(judge, gt, preds))
                    if error:
                        row["judge_errors"][name] = error
                        per_judge[name] = [{"verdict": 0, "pred_idx": None, "reason": "", "missing": True}] * len(gt)
                    else:
                        per_judge[name] = parse_judge(value or {}, len(gt))
                        raw[name] = {"hits_claimed": (value or {}).get("hits"), "comment": (value or {}).get("comment")}
                row["gt_votes"] = vote(per_judge, len(gt))
                row["judge_detail"] = {name: rows_ for name, rows_ in per_judge.items()}
                row["judge_raw"] = raw
                passed = sum(1 for v in row["gt_votes"] if v["passed"])
                row["passed_gt"] = passed                                    # 正式：系统最后写出的答案都评
                row["passed_gt_strict"] = passed if source == "final" else 0 # 严格：只算框架接受的 FINISH
            elif judges:
                row["gt_votes"] = vote({name: [{"verdict": 0}] * len(gt) for name in judges}, len(gt))
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{index}/{len(entries)}] {entry['id']}：{info['status']} / {info['decision']} / {mode} "
                  f"{len(preds)} 点 → 通过 {row['passed_gt']}/{len(gt)}"
                  + ("（答案来自被框架拒收的 FINISH）" if source == "rejected_finish" else "")
                  + (f"；评委失败 {list(row['judge_errors'])}" if row["judge_errors"] else ""), flush=True)
    for judge in judges.values():
        close = getattr(judge, "close", None)
        if callable(close):
            close()
    summary = summarize(rows, judge_specs)
    meta = {"round": args.round, "judges": judge_specs, "endpoint": args.endpoint, "official_commit": OFFICIAL_COMMIT,
            "samples": len(rows), "seconds": round(time.monotonic() - started, 1), "rows": str(rows_path),
            "excluded_used_for_revision": excluded}
    md_meta = {**meta, "behavior_rows": [r["behavior"] for r in rows if r.get("behavior")],
               "funnel_rows": [{"sample_id": r["sample_id"], "funnel": r["funnel"]} for r in rows if r.get("funnel")]}
    (out_dir / f"{args.round}.{tag}.summary.json").write_text(json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2),
                                                              encoding="utf-8")
    md = out_dir / f"{args.round}.{tag}.md"
    md.write_text(markdown(summary, md_meta), encoding="utf-8")
    a = summary["all_planned"]
    t = summary["attempted"]
    r = summary["attempted_strict"]
    b = summary["behavior"]
    if b.get("runs"):
        print(f"\n行为面板：覆盖率中位 {b['coverage_median']}，候选下载 {b['video_downloads_total']}，GIA 末次零原件帧 "
              f"{b['gia_last_zero_original_frames']} 个，重复率 {b['repeat_rate']}，减图请求 {b['context_retry_requests']}，"
              f"被拒率 HI {b['hi_rejection_rate']} / GIA {b['gia_rejection_rate']}，HI 报告自写 / 代交 {b.get('hi_self_reports')} / "
              f"{b.get('hi_auto_reports')}，崩溃 {b['crashes'] or '无'}；"
              f"答案语言 {summary['answer_language']}", flush=True)
    fu = summary.get("funnel") or {}
    if fu.get("runs"):
        print(f"找原片漏斗：进池 {fu['source_in_pool']}/{fu['runs_with_store']}，下载 {fu['source_downloaded']}，"
              f"比对 {fu['source_compared']}；途径 {fu['source_found_by'] or '无'}；视频搜索 {fu['video_searches']} 次，"
              f"零结果 {fu['video_searches_empty']} 次", flush=True)
    print(f"\n严格口径（被拒收的 FINISH {summary['rejected_finish_used']} 个按 0）：点级 {r['point_acc']}，视频级 {r['video_acc']}", flush=True)
    print(f"已开跑 {t['videos']} 个：点级 {t['point_acc']}，视频级 {t['video_acc']}（崩溃按 0）；"
          f"全部 {a['videos']} 个（没跑的也按 0）：{a['point_acc']} / {a['video_acc']}；"
          f"只算 FINISH：{summary['finished_only']['point_acc']} / {summary['finished_only']['video_acc']}\n汇总 {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
