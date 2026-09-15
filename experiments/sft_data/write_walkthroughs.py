"""Render readable examples directly from actual requests, observations and reviews."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def compact(value, limit=300):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def render_run(run):
    links = read(run / "execution_links.json", [])
    assessment = read(run / "selected_assessment.json", {}).get("assessment", "assessment_r2")
    if not links or not (run / assessment / "judgment.json").exists():
        return None
    verdict = extract_json_object((run / assessment / "judgment.json").read_text(encoding="utf-8"))
    targets = {item["request_id"]: item for item in verdict["target_findings"]}
    events = {event["id"]: event for event in [json.loads(line) for line in (run / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]}
    first = read(run / "requests" / links[0]["request_id"] / "request.json")
    payload = first["payload"]
    task = payload.get("global_view", payload.get("model_view", {})).get("task", "见原始请求")
    config = read(run / "run_config.json", {"dataset": "videosham", "case": "1028", "mode": "bridge"})
    lines = [f"# {config['dataset']} / {config['case']}：{run.name}", "", "## 任务", "", task, "",
             "## 逐步轨迹", "", "动作与观察来自真实执行记录。评分列表示该输出是否适合作为正向模仿目标；排除的步骤仍可作为后续真实历史。", "",
             "| 请求 | 角色 | 教师输出 | 实际反馈 | 目标评分 | 处置 |", "|---|---|---|---|---:|---|"]
    for link in links:
        rid = link["request_id"]
        target = targets[rid]
        output = link["raw_output"]
        if "action" in output:
            action = output["action"] + " " + compact(output.get("arguments", {}), 180)
        elif output.get("decision") == "INVESTIGATE":
            action = "INVESTIGATE：" + compact(output.get("tasks", []), 220)
        elif output.get("decision") == "FINISH":
            action = "FINISH：" + compact(output.get("answer", ""), 120)
        else:
            action = "局部报告 " + output.get("status", "")
        observation = events.get(link.get("observation_event_id"), {}).get("payload", {}).get("observation", {})
        feedback = compact(observation.get("items", [])[:2], 180) if observation else link["status"]
        lines.append(f"| [{rid}](../runs/{run.name}/requests/{rid}/request.json) | {link['actor']} | {compact(action, 250)} | {feedback} | {target['target_rating']} | {target['disposition']} |")
    lines += ["", "## 报告与最终决定", ""]
    for link in links:
        output = link["raw_output"]
        if "findings" not in output and output.get("decision") != "FINISH":
            continue
        rid = link["request_id"]
        lines += [f"### {link['actor']} / {rid}", ""]
        if "findings" in output:
            lines += ["报告状态：" + output.get("status", ""), ""]
            if output.get("status_reason"):
                lines += [output["status_reason"], ""]
            for finding in output["findings"]:
                lines += [finding["statement"], "", "证据：" + "、".join(finding.get("evidence_ids", [])), ""]
                for limitation in finding.get("limitations", []):
                    lines += ["- " + limitation]
                lines += [""]
        else:
            lines += [output["answer"], ""]
        target = targets[rid]
        lines += [f"目标评分：{target['target_rating']}/4；处置：{target['disposition']}。", "", target["reason"], ""]
    lines += ["## 筛选理由", ""]
    for target in verdict["target_findings"]:
        if target["disposition"] != "accept_candidate":
            lines += [f"- {target['request_id']}：{target['reason']}"]
    lines += ["", "## 原始记录", "",
              f"- [完整时序记录](../runs/{run.name}/trajectory.jsonl)",
              f"- [完整逐目标评审](../runs/{run.name}/{assessment}/judgment.json)",
              f"- [候选数据与排除记录](../runs/{run.name}/{assessment}/sft_candidates/)", ""]
    return config, "\n".join(lines)


def render_offline(source):
    assessment_name = read(source / "selected_assessment.json", {}).get("assessment", "assessment_r2")
    assessment = source / assessment_name / "judgment.json"
    if not assessment.is_file():
        return None
    verdict = extract_json_object(assessment.read_text(encoding="utf-8"))
    lines = [f"# GIA 离线状态：{source.name}", "",
             "以下为教师实际生成并独立会话评审的单步决定。输入来自已取得的真实证据；新提议的调查任务没有在这些离线状态中执行。", ""]
    for target in verdict["states"]:
        state = source / target["id"]
        directory = next((state / "requests").glob("request_*"))
        request = read(directory / "request.json")
        view = request["payload"].get("global_view", request["payload"].get("model_view", {}))
        decision = read(state / "decision.json")
        history = request["payload"].get("memory_history", [])
        if history:
            lines += ["已执行的记忆读取：", ""]
            for step in history:
                lines.append("- `" + json.dumps(step["output"], ensure_ascii=False) + "`")
            lines += ["", f"[实际读取返回](../gia_decisions/{source.name}/{target['id']}/executed_history.json)", ""]
        lines += [f"## {target['id']}", "", "当前问题：" + view.get("task", "见原始输入"), "",
                  "### 当前局部报告", "", "```json", json.dumps(view.get("latest_reports", []), ensure_ascii=False, indent=2),
                  "```", "", "### 实际 GIA 输出", "", "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", "",
                  f"目标评分：{target['target_rating']}/4；处置：{target['disposition']}。", ""]
        for dimension in target.get("dimensions", []):
            if dimension["id"] in {"grounding", "task", "decision", "uncertainty"}:
                lines += [f"- {dimension['id']}：{dimension['reason']}"]
        lines += ["", f"[原始输入与图片清单](../gia_decisions/{source.name}/{target['id']}/requests/{directory.name}/request.json)", ""]
    return "\n".join(lines)


def render_terminal(source):
    assessment = read(source / "selected_assessment.json", {}).get("assessment", "assessment_r2")
    summary = read(source / assessment / "sft_candidates/summary.json")
    if not summary:
        return None
    provenance = read(source / "provenance.json")
    request = read(source / "requests/request_0001/request.json")
    output = read(source / "decision.json")
    target = summary["target_assessment"]
    view = request["payload"]["model_view"]
    task = view.get("task", "见原始模型输入中的局部任务。")
    lines = [f"# HI 局部报告：{source.name}", "", "## 局部任务", "", task, "",
             "输入取自真实执行中的报告提交前状态，保留该次实际图片与已取得证据。教师实际生成这份报告，未执行新工具，也未将报告拼接到原有后续 GIA 决定中。", "",
             "## 实际报告", "", "```json", json.dumps(output, ensure_ascii=False, indent=2), "```", "",
             "## 逐目标评审", "", f"评分：{target['target_rating']}/4；处置：{target['disposition']}。", "", target["reason"], "",
             "## 原始记录", "",
             f"- [原始执行与输入](../runs/{provenance['source_run']}/requests/{provenance['source_request']}/request.json)",
             f"- [新报告的实际输入](../hi_terminal_repairs/{source.name}/requests/request_0001/request.json)",
             f"- [独立会话评审](../hi_terminal_repairs/{source.name}/{assessment}/judgment.json)", ""]
    return "\n".join(lines)


def main():
    destination = ROOT / "examples"
    destination.mkdir(exist_ok=True)
    entries = []
    for run in sorted((ROOT / "runs").iterdir()):
        if not run.is_dir():
            continue
        result = render_run(run)
        if result:
            config, document = result
            (destination / (run.name + ".md")).write_text(document, encoding="utf-8")
            entries.append((config["dataset"], str(config["case"]), run.name))
    lines = ["# 实际轨迹样例", "", "每个样例包含原问题、逐步行动与实际反馈、局部报告、最终决定及逐目标评分。本文档按原始记录生成，不补写未发生的动作或观察。", "",
             "自动评审候选仍需按研究需要抽查；完整任务成功、局部报告可用、单个动作可用分别判断。", "",
             "| 数据集 | 案例 | 运行与评分 |", "|---|---|---|"]
    for dataset, case, name in entries:
        lines.append(f"| {dataset} | {case} | [{name}](examples/{name}.md) |")
    lines += ["", "## 独立 GIA 决策样例", "", "局部报告的事实与现有证据保留，当前问题或证据状态变化。每个输出只表示当时的下一步决定。", ""]
    offline_count = 0
    for source in sorted((ROOT / "gia_decisions").iterdir()):
        if source.is_dir():
            document = render_offline(source)
            if document:
                filename = "gia_" + source.name + ".md"
                (destination / filename).write_text(document, encoding="utf-8")
                lines.append(f"- [{source.name}](examples/{filename})")
                offline_count += 1
    lines += ["", "## HI 报告生成样例", "", "这些样例只监督从已有状态生成准确的局部报告；没有新增工具执行。", ""]
    terminal_count = 0
    for source in sorted((ROOT / "hi_terminal_repairs").iterdir()):
        if source.is_dir():
            document = render_terminal(source)
            if document:
                filename = "hi_report_" + source.name + ".md"
                (destination / filename).write_text(document, encoding="utf-8")
                lines.append(f"- [{source.name}](examples/{filename})")
                terminal_count += 1
    lines += ["", "[汇总结果](试验结果.md) · [构造步骤](方法与数据质量标准.md) · [当前质量定义](rubric_r6.md)", ""]
    (ROOT / "样例轨迹.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"reviewed_example_documents": len(entries), "offline_example_documents": offline_count,
                      "terminal_report_documents": terminal_count}))


if __name__ == "__main__":
    main()
