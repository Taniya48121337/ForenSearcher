"""Summarize observed runs, reviewed targets and separately constructed GIA states."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def collect():
    runs = []
    for run in sorted((ROOT / "runs").iterdir()):
        if not run.is_dir():
            continue
        config = read(run / "run_config.json", {})
        if not config and not run.name.startswith("videosham_1028"):
            continue
        trajectory = run / "trajectory.jsonl"
        events = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines()] if trajectory.exists() else []
        outcomes = [{"actor": e["actor_id"], **e["payload"]} for e in events if e["event_type"] == "workspace_outcome"]
        reports = [e for e in events if e["event_type"] == "workspace_report"]
        usage = Counter()
        requests, completed_calls = list((run / "requests").glob("request_*")), 0
        for request in requests:
            transport = read(request / "transport.json", {})
            if transport.get("returncode") == 0:
                completed_calls += 1
            for key, value in (transport.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
        assessment = read(run / "selected_assessment.json", {}).get("assessment", "assessment_r2")
        summary = read(run / assessment / "sft_candidates/summary.json", {})
        runs.append({"run": run.name, "dataset": config.get("dataset", "videosham"),
            "case": str(config.get("case", "1028")), "mode": config.get("mode", "bridge"),
            "usage_split": config.get("usage", "construction_validation" if "validation" in run.name else "development"),
            "runtime_returned_result": (run / "result.json").exists(), "trajectory_exported": trajectory.exists(),
            "outcomes": outcomes, "committed_reports": len(reports),
            "reported_statuses": dict(Counter(e["payload"].get("report", e["payload"]).get("status", "unspecified") for e in reports)),
            "cli_requests_started": len(requests), "cli_calls_returned_success": completed_calls,
            "recorded_cli_usage": dict(usage), "execution": read(run / "execution_summary.json"),
            "assessment": assessment, "scores": read(run / assessment / "scores.json"), "candidate_counts": summary.get("counts"),
            "excluded_targets": summary.get("excluded")})
    gia = []
    for folder in sorted((ROOT / "gia_decisions").iterdir()):
        if not folder.is_dir():
            continue
        assessment = read(folder / "selected_assessment.json", {}).get("assessment", "assessment_r2")
        summary = read(folder / assessment / "sft_candidates/summary.json")
        references = [read(state / "reference.json", {}) for state in sorted(folder.glob("state_*"))]
        gia.append({"construction": folder.name, "states": len(references), "summary": summary,
                    "source_cases": sorted({(item.get("provenance", {}).get("dataset", "unknown"),
                                             item.get("provenance", {}).get("case", "unknown")) for item in references})})
    calibration = [{"name": folder.name, "metrics": read(folder / "metrics.json")}
                   for folder in sorted((ROOT / "rubric_calibration").iterdir()) if folder.is_dir() and (folder / "metrics.json").exists()]
    repairs_root = ROOT / "hi_terminal_repairs"
    repairs = [{"name": folder.name, "provenance": read(folder / "provenance.json"),
                "has_report": (folder / "decision.json").is_file(),
                "summary": read(folder / read(folder / "selected_assessment.json", {}).get("assessment", "assessment_r2") / "sft_candidates/summary.json")}
               for folder in sorted(repairs_root.iterdir()) if folder.is_dir()] if repairs_root.is_dir() else []
    corpus = read(ROOT / "corpus/manifest.json", {})
    return {"generated_at": datetime.now().astimezone().isoformat(), "runs": runs, "gia_offline": gia,
            "corpus_files": corpus.get("files", []), "corpus_exclusions": corpus.get("curation_exclusions", []),
            "rubric_diagnostics": calibration, "hi_terminal_repairs": repairs,
            "scope": "data_construction_pilot; automated candidate selection; no student weight training or independent human certification"}


def render(data):
    lines = ["# SFT 数据构造试验结果", "", "统计时间：" + data["generated_at"], "",
             "本结果区分真实工具轨迹、离线 GIA 决定与评分诊断。评分为独立会话的自动评审；未进行学生权重训练或独立人工认证。", "",
             "## 案例覆盖", "", "案例数按数据集原始编号去重，包含已发起但未结束的运行；重试和状态变体不增加案例数。不同编号也不保证语义事件独立。", "",
             "| 数据集 | 开发案例 | 保留案例 | 已发起运行 | 有逐目标评审与导出的运行 |", "|---|---:|---:|---:|---:|"]
    for dataset in sorted({run["dataset"] for run in data["runs"]}):
        selected = [run for run in data["runs"] if run["dataset"] == dataset]
        development = {run["case"] for run in selected if run["usage_split"] == "development"}
        validation = {run["case"] for run in selected if run["usage_split"] == "construction_validation"}
        lines.append(f"| {dataset} | {len(development)} | {len(validation)} | {len(selected)} | {sum(run['candidate_counts'] is not None for run in selected)} |")
    lines += ["", "## 最终候选包", "", "按案例用途分开保存。开发文件供后续训练准备，构造保留文件不能混入开发训练。以下为决策行数，相同案例可贡献多行。", "",
              "| 文件 | 角色 | 决策行 | 来源案例家族 |", "|---|---|---:|---:|"]
    for item in data["corpus_files"]:
        lines.append(f"| [{item['file']}](corpus/{item['file']}) | {item['role']} | {item['rows']} | {item['case_families']} |")
    if data["corpus_exclusions"]:
        lines += ["", "下方逐运行表保留原自动评审候选数；最终候选包另外排除已知存在实质任务或评分分歧的目标，详见 candidate_exclusions.json。"]
    lines += ["",
             "## 真实执行与逐步筛选", "",
             "表中的报告数是已提交数量，不等于完整任务成功。候选数来自逐目标评审和实际执行对应检查。未评审的项目不计为通过。", "",
             "| 数据集 / 案例 | 运行 | 语料用途 | 已提交报告 | HI 候选 | GIA 候选 | 排除目标 |", "|---|---|---|---:|---:|---:|---:|"]
    for run in data["runs"]:
        counts = run["candidate_counts"] or {}
        lines.append(f"| {run['dataset']} / {run['case']} | [{run['run']}](runs/{run['run']}/) | {run['usage_split']} | {run['committed_reports']} | {counts.get('HI', '未导出')} | {counts.get('GIA', '未导出')} | {run['excluded_targets'] if run['excluded_targets'] is not None else '—'} |")
    lines += ["", "## 自动评分", "", "各角色分数独立报告；高均分不覆盖错误步骤，也不把 INCOMPLETE 报告变成任务成功。", "",
              "| 运行 | 角色 | 已评分维度归一化分 | 评分覆盖率 | 关键错误数 |", "|---|---|---:|---:|---:|"]
    for run in data["runs"]:
        for score in run["scores"] or []:
            if score.get("score_on_assessed") is None:
                continue
            coverage = score.get("coverage")
            coverage_text = f"{coverage:.0%}" if coverage is not None else "—"
            lines.append(f"| {run['run']} | {score['actor']} | {score['score_on_assessed']:.2f} | {coverage_text} | {len(score['critical_errors'])} |")
    lines += ["", "## 独立 GIA 状态", "", "同一案例的状态变体不计为新增独立案例，所有新提议的工具行动均未在这些离线样本中执行。", "",
              "| 构造批次 | 状态数 | GIA 候选 | 排除状态 | 来源案例 |", "|---|---:|---:|---:|---|"]
    for gia in data["gia_offline"]:
        summary = gia["summary"] or {}
        lines.append(f"| [{gia['construction']}](gia_decisions/{gia['construction']}/) | {gia['states']} | {summary.get('GIA_offline_candidates', '待评审')} | {summary.get('excluded_states', '—')} | {', '.join('/'.join(case) for case in gia['source_cases'])} |")
    lines += ["", "## HI 终态报告重新生成", "",
              "输入来自真实轨迹的报告提交前状态，图片保持一致。新报告未接入原有后续 GIA 决策，不计为新增完整执行轨迹。", "",
              "| 构造 | 来源 | 用途 | 候选报告 | 目标评分 |", "|---|---|---|---:|---:|"]
    for repair in data.get("hi_terminal_repairs", []):
        provenance, summary = repair["provenance"] or {}, repair["summary"] or {}
        target = summary.get("target_assessment", {})
        source = provenance.get("source_run", "—") + "/" + provenance.get("source_request", "")
        status = "未导出" if repair["has_report"] else "无报告目标"
        lines.append(f"| [{repair['name']}](hi_terminal_repairs/{repair['name']}/) | {source} | {provenance.get('usage_split', '—')} | {summary.get('HI_terminal_candidates', status)} | {target.get('target_rating', '—')} |")
    lines += ["", "## Rubric 诊断", "", "这些正负例用于测量评分行为，不是实际执行轨迹，也不进入正向 SFT。开发诊断不能证明新案例泛化。", "",
              "| 诊断 | 样本数 | 可用 / 不收录 | 错误接受 | 有效样本误拒 | 关键未知 |", "|---|---:|---|---:|---:|---:|"]
    for item in data["rubric_diagnostics"]:
        m = item["metrics"]
        lines.append(f"| {item['name']} | {m['total']} | {m['positive']} / {m['negative']} | {len(m['false_accept_ids'])} | {len(m['false_reject_ids'])} | {len(m['critical_unknown_ids'])} |")
    lines += ["", "scope 与 source 各组诊断的 4 个不收录样本中，3 个有已知错误，1 个缺少关键评审材料；缺件不等于已证实错误。旧、新 rubric 均正确区分这组简单诊断，不能据此宣称分类准确率提高。", "", "## 解释范围", "",
              "- 运行、工作区、决策行与原始案例分别计数；不同构造批次可能来自同一案例。",
              "- 当前模型评分与实际数据集指标分别保存，不能互相替代。QVHighlights 区间和 VCSL 原坐标指标见 temporal_results.json；边界不确定性的补充分析见 temporal_comparison.json。",
              "- VCSL 的原标注和额外可见对应分别核查；局部匹配发现不等于完整官方定位成功。",
              "- 语料迁移验证使用各数据集另选的来源案例；样本量小，且部分素材来自既有公开子集。",
              "- 构造保留划分对应 rubric_r2 的首轮检验。AVeriTeC 392、1707 后来用于任务与数量口径修改，AVerImaTeC 660 后来用于来源变体与评审细则修改；它们不能作为相应修改的未见检验例，仍不混入开发训练语料。",
              "- VCSL 的短视频样例包含相近动画的改编，来源 URL 不同不足以证明内容分布广泛。当前结果不能概括整个 VCSL 数据集。",
              "- 未导出终态轨迹的运行不推断为完成或仍在运行；其当前进程状态需另行检查。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    result = collect()
    (ROOT / "pilot_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "试验结果.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"runs": len(result["runs"]), "reviewed_runs": sum(r["candidate_counts"] is not None for r in result["runs"]),
                      "offline_constructions": len(result["gia_offline"])}))
