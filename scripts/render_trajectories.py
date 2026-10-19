"""把 walker run 渲染成可读轨迹（对话 6 用）：每一步 = 状态（任务 / 上一步观察 / 附图编号）+ 决策（动作、参数、理由或报告 / 计划 / 答案）
+ 观察（下一次同一角色请求里的 latest_observation）+ 教师槽记录（槽、后端、模型）。

    python scripts/render_trajectories.py --runs <run 名> ... --out experiments/sft_data/r1a_review
（10-17：从 experiments/sft_data/r0a_review/ 拷进 scripts/，内容不变，只是让 r1a 的通读样本出自入库的脚本。）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3] if (Path(__file__).resolve().parents[2] / "sft_data").is_dir() else Path.cwd()
RUNS = Path("experiments") / "sft_data" / "runs"
OBS_LIMIT = 1600
TEXT_LIMIT = 2400


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cut(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" …[+{len(text) - limit} chars]"


def fence(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def images_of(payload: dict[str, Any]) -> list[str]:
    view = payload.get("model_view") or payload.get("global_view") or {}
    out = []
    for item in view.get("image_inputs") or []:
        loc = item.get("locator")
        try:
            loc_d = json.loads(loc) if isinstance(loc, str) else (loc or {})
        except ValueError:
            loc_d = {}
        where = ""
        if loc_d.get("time_s") is not None:
            where = f"@{loc_d['time_s']}s"
        elif loc_d.get("page") is not None:
            where = f"p.{loc_d['page']}"
        elif loc_d.get("scope"):
            where = str(loc_d["scope"])
        out.append(f"{item.get('evidence_id')}{('(' + where + ')') if where else ''}")
    return out


def observation_text(view: dict[str, Any]) -> str:
    """10-15 对话 5''：SEARCH 的结果行之外，把学生同样看得到的 search_title / search_snippet 渲染出来；READ_RECORD 渲染记录的
    标签与正文前 300 字。以前只有 "[R3] url; page=ready(E4)" 与 "offset=0"，人工通读会误判"理由用了看不到的信息"
    （D2：7 条 hidden_knowledge 指控里 5 条是这个假象）。"""

    latest = view.get("latest_observation") or {}
    items = latest.get("items") or []
    if not items:
        return ""
    lines = [str(x) for x in items]
    details = latest.get("details") if isinstance(latest.get("details"), dict) else {}
    if latest.get("action") == "SEARCH":
        for result in details.get("results") or []:
            if not isinstance(result, dict):
                continue
            title, snippet = result.get("search_title"), result.get("search_snippet")
            if title or snippet:
                lines.append(f"  {result.get('candidate_id')}: {cut(title or '', 140)}" + (f" — {cut(snippet, 220)}" if snippet else ""))
    elif latest.get("action") == "READ_RECORD" and (details.get("label") or details.get("content")):
        lines.append(f"  {details.get('evidence_id')} label: {cut(details.get('label') or '', 200)}")
        if details.get("content"):
            lines.append("  content: " + cut(" ".join(str(details["content"]).split()), 300))
    text = cut("\n".join(lines), OBS_LIMIT + 1600)
    extra = inspect_excerpt(details.get("result")) if latest.get("action") == "INSPECT" else ""
    return text + ("\n" + extra if extra else "")


TRANSCRIPT_LIMIT = 3000
SUPPORT_LIMIT = 1400       # compact_16k 已把每条支撑记录截到约 1.3 千字：印全


def inspect_excerpt(result: Any) -> str:
    """10-18（D2 r1a 的 3 处 hidden_knowledge 假象）：INSPECT 的结果正文学生看得到，以前只印 items 一句。
    asr → 分段转写（时间 + 文字）；ocr → 识别出的文字与出现时段（结果里有 texts 时）。按模型实际收到的 rendered_payload 印。"""

    if not isinstance(result, dict):
        return ""
    lines: list[str] = []
    segments = result.get("segments")
    if isinstance(segments, list) and segments and isinstance(segments[0], dict) and "text" in segments[0]:
        lines.append(f"  transcript ({len(segments)} segments, language {result.get('language')}):")
        used = shown = 0
        for seg in segments:
            # compact_16k 截短时末尾是一句说明字符串（不是分段），照原样印
            row = (f"    [{seg.get('start_s')}–{seg.get('end_s')} s] {' '.join(str(seg.get('text') or '').split())}"
                   if isinstance(seg, dict) else f"    {cut(str(seg), 300)}")
            if used + len(row) > TRANSCRIPT_LIMIT:
                lines.append(f"    …[+{len(segments) - shown} more segments]")
                break
            lines.append(row)
            used += len(row)
            shown += 1
    texts = (result.get("summary") or {}).get("texts") if isinstance(result.get("summary"), dict) else None
    if isinstance(texts, list) and texts:
        lines.append("  OCR texts: " + cut("; ".join(
            f"\"{t.get('text')}\" ({t.get('first_s')}–{t.get('last_s')} s, {t.get('frames')} frames)"
            for t in texts if isinstance(t, dict)), 1600))
    return "\n".join(lines)


def support_excerpt(report: dict[str, Any]) -> list[str]:
    """10-18：GIA 的 latest_reports[i].supporting_records（报告所引 E / Finding 的正文）随报告一起给到 GIA，不用 READ_RECORD；
    以前只印 Finding 个数，答案里出现的页面原话像是"GIA 没读过"。按 rendered_payload（已截到约 1.3 千字）印。"""

    out = []
    for rec in report.get("supporting_records") or []:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("evidence_id") or rec.get("finding_id") or rec.get("record_id")
        body = rec.get("content") or rec.get("statement") or rec.get("content_notice") or ""
        out.append(f"    {rid} [{cut(rec.get('label') or rec.get('record_type') or '', 120)}]: "
                   f"{cut(' '.join(str(body).split()), SUPPORT_LIMIT)}")
    return out


def decision_text(resp: dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return "(no response)"
    if "findings" in resp:                       # HI report
        lines = [f"REPORT status={resp.get('status')}" + (f" reason={resp.get('status_reason')}" if resp.get("status_reason") else "")]
        for i, f in enumerate(resp.get("findings") or [], 1):
            lines.append(f"  F{i}. {cut(f.get('statement', ''), 900)}  [cites {', '.join(f.get('evidence_ids') or [])}]")
            for lim in f.get("limitations") or []:
                lines.append(f"      limitation: {cut(lim, 300)}")
        return "\n".join(lines)
    if resp.get("decision") in ("INVESTIGATE", "FINISH"):
        lines = [f"GIA {resp['decision']}"]
        for i, t in enumerate(resp.get("tasks") or [], 1):
            lines.append(f"  T{i}. {cut(t.get('task', ''), 900)}  [materials {', '.join(t.get('materials') or [])}]")
        if resp.get("answer"):
            lines.append("  ANSWER: " + cut(resp["answer"], TEXT_LIMIT))
        if resp.get("rationale"):
            lines.append("  rationale: " + cut(resp["rationale"], 600))
        return "\n".join(lines)
    action = resp.get("action")
    args = json.dumps(resp.get("arguments"), ensure_ascii=False)
    lines = [f"{action} {cut(args, 700)}"]
    if resp.get("focus_evidence_ids"):
        lines.append(f"  focus: {', '.join(resp['focus_evidence_ids'])}")
    if resp.get("rationale"):
        lines.append("  rationale: " + cut(resp["rationale"], 900))
    return "\n".join(lines)


def render(run_dir: Path) -> str:
    cfg = load(run_dir / "run_config.json") or {}
    import sys
    sys.path.insert(0, "src")
    from foren_searcher.datagen.run_cases import run_status
    status = run_status(run_dir)
    walker = cfg.get("walker") or {}
    reqs = sorted((run_dir / "requests").glob("request_*"))
    rows = []
    for d in reqs:
        req, resp, tr = load(d / "request.json") or {}, load(d / "response.json"), load(d / "transport.json") or {}
        payload = req.get("payload") or {}
        role = payload.get("workspace_id") or "GIA"
        rendered = req.get("rendered_payload")
        if isinstance(rendered, str):
            try:
                rendered = json.loads(rendered)
            except ValueError:
                rendered = None
        # 10-18：观察与 GIA 的支撑记录按模型实际收到的渲染后状态印（compact_16k 会截短正文）；没有渲染版时退回原始 payload
        seen = rendered if isinstance(rendered, dict) and (rendered.get("model_view") or rendered.get("global_view")) else payload
        rows.append({"dir": d.name, "role": role, "payload": payload, "seen": seen, "resp": resp, "tr": tr})
    head = [f"# {run_dir.name}", "",
            f"- case: `{cfg.get('case_id') or cfg.get('case')}`；status: {status}；specs: {walker.get('specs')}",
            f"- teacher: {cfg.get('teacher_model')}（role {walker.get('teacher_role')}）；judge planned: {walker.get('judge_planned')}",
            f"- requests: {len(rows)}", ""]
    seen_task: set[str] = set()
    shown_obs: set[str] = set()          # 上一步已在 Observation 里展示过的角色，State 里不再重复
    body = []
    for i, row in enumerate(rows):
        payload, role = row["payload"], row["role"]
        view = payload.get("model_view") or payload.get("global_view") or {}
        oracle = (row["tr"].get("oracle") or {})
        rc = view.get("runtime_constraints") or {}
        body.append(f"## {i + 1}. {role} · turn {rc.get('turn_index')}/{rc.get('max_turns')}"
                    + (f" · round {rc.get('round_index')}" if rc.get("round_index") else "")
                    + f" · rule `{oracle.get('rule')}` · branch `{oracle.get('branch')}` · {row['dir']}")
        body.append("")
        body.append("**State**")
        body.append("")
        if role not in seen_task:
            seen_task.add(role)
            body.append("- task: " + cut(view.get("task") or "", 1200))
        seen_view = row["seen"].get("model_view") or row["seen"].get("global_view") or view
        obs = observation_text(seen_view) if role not in shown_obs else ""
        if obs:
            body.append("- latest observation (result of this workspace's previous action):")
            body.append(fence(obs))
        if role == "GIA":
            reports = view.get("latest_reports") or []
            if reports:
                # 10-15 对话 5''：latest_reports[i] 的真实结构是 {task, materials, run_status, report: {task_id, status,
                # finding_ids, …}}；以前读 r.task_id / r.status / r.findings，GIA 每轮都显示 "None None (0 findings)"
                def _rep(r: dict[str, Any]) -> dict[str, Any]:
                    return r.get("report") if isinstance(r.get("report"), dict) else r

                body.append(f"- latest reports: {len(reports)} → " + "; ".join(
                    f"{_rep(r).get('task_id')} {_rep(r).get('status')} "
                    f"({len(_rep(r).get('finding_ids') or _rep(r).get('findings') or [])} findings)"
                    + (f" — reason: {cut(_rep(r).get('status_reason'), 300)}" if _rep(r).get("status_reason") else "")
                    for r in reports if isinstance(r, dict)))
                for r in seen_view.get("latest_reports") or []:
                    if isinstance(r, dict) and r.get("supporting_records"):
                        body.append(f"  - supporting records of {_rep(r).get('task_id')} (given to GIA with the report, as rendered):")
                        body.append(fence("\n".join(support_excerpt(r))))
        imgs = images_of(payload)
        body.append(f"- attached images ({len(imgs)}): {', '.join(imgs) if imgs else 'none'}")
        if view.get("focus_evidence_ids"):
            body.append(f"- focus kept: {', '.join(view['focus_evidence_ids'])}")
        body.append("")
        body.append("**Decision**")
        body.append("")
        revise = load(run_dir / "requests" / row["dir"] / "slot_S5_revise.json")
        resp = row["resp"]
        # 10-19：S5_answer_revise 会把 response.json 改写成新答案；Decision 一栏要显示核验前的原答案（revise.old），
        # 否则它与下面的 REVISED ANSWER 是同一段文字，通读会误报"重写与原答案逐字相同"
        if isinstance(revise, dict) and revise.get("changed") and isinstance(resp, dict) and revise.get("old"):
            resp = {**resp, "answer": revise["old"]}
            body.append("- Decision below shows the **original answer before verification** (from slot_S5_revise.json `old`).")
        body.append(fence(decision_text(resp)))
        if isinstance(revise, dict) and revise.get("changed"):
            body.append("- **answer rewritten after verification** (support-only S5_answer_revise; this is the exported version). "
                        "Verifier issues fed back: " + "; ".join(f"{e.get('type')}/{e.get('severity')}" for e in revise.get("errors") or []))
            body.append(fence("REVISED ANSWER: " + cut(revise.get("new") or "", TEXT_LIMIT)))
        slots = row["tr"].get("slots") or []
        if slots:
            body.append("- teacher slots: " + "; ".join(
                f"{s.get('name')} → {s.get('backend')}:{s.get('model')}" + (f" (served {s.get('model_served')})" if s.get("model_served") else "")
                for s in slots))
        nxt = next((r for r in rows[i + 1:] if r["role"] == role), None)
        if nxt is not None:
            nobs = observation_text(nxt["seen"].get("model_view") or nxt["seen"].get("global_view") or {})
            if nobs and role != "GIA":
                shown_obs.add(role)
                body.append("")
                body.append("**Observation** (what the tool returned, as the next state shows it)")
                body.append("")
                body.append(fence(nobs))
        body.append("")
    return "\n".join(head + body) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = ["# Readable trajectories", ""]
    for name in args.runs:
        run_dir = RUNS / name
        text = render(run_dir)
        (out / f"{name}.md").write_text(text, encoding="utf-8")
        index.append(f"- [{name}]({name}.md)")
        print("wrote", out / f"{name}.md", len(text))
    (out / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
