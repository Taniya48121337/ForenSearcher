"""Package reviewed per-role candidates and their actual pixels by existing usage split."""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def source_keys(dataset, case):
    folder = ROOT / "data" / dataset / case
    if dataset == "videosham":
        folder = ROOT / "data/cases" / ("videosham_" + case)
    reference = read(folder / "reference.json", {})
    if dataset == "videosham":
        return [reference.get("source_record", {}).get("Name/Id in source dataset", "").strip()]
    if dataset == "true":
        return [reference.get("video_information", {}).get("video_url", "")]
    if dataset == "qvhighlights":
        vid = reference.get("vid", "")
        return ["youtube:" + vid.rsplit("_", 2)[0]] if vid else []
    if dataset == "vcsl":
        return reference.get("urls", [])
    return [reference.get("article", reference.get("fact_checking_article", ""))]


def input_files():
    for source in sorted((ROOT / "runs").iterdir()):
        config = read(source / "run_config.json", {})
        split = config.get("usage", "construction_validation" if "validation" in source.name else "development")
        origin = "bridge" if config.get("mode", "bridge") == "bridge" else "hi_independent"
        assessment = read(source / "selected_assessment.json", {}).get("assessment", "assessment_r2")
        for path in sorted((source / assessment / "sft_candidates").glob("*.jsonl")):
            yield path, split, origin
    for source in sorted((ROOT / "gia_decisions").iterdir()):
        assessment = read(source / "selected_assessment.json", {}).get("assessment", "assessment_r2")
        path = source / assessment / "sft_candidates/gia.jsonl"
        if path.is_file():
            reference = read(source / "state_01/reference.json", {})
            yield path, reference.get("provenance", {}).get("usage_split", "development"), "gia_offline"
    repairs = ROOT / "hi_terminal_repairs"
    if repairs.is_dir():
        for source in sorted(repairs.iterdir()):
            assessment = read(source / "selected_assessment.json", {}).get("assessment", "assessment_r2")
            path = source / assessment / "sft_candidates/hi.jsonl"
            provenance = read(source / "provenance.json", {})
            if path.is_file():
                yield path, provenance["usage_split"], "hi_terminal_replay"


def main():
    destination = ROOT / "corpus"
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_manifest_path = destination / "image_manifest.json"
    image_manifest = read(image_manifest_path, {})
    groups, sources, identifiers = defaultdict(list), [], set()
    exclusions = read(ROOT / "candidate_exclusions.json", [])
    excluded_ids = {(item["run"], item["request"]) for item in exclusions}
    applied_exclusions = []
    family_splits, source_splits = defaultdict(set), defaultdict(set)
    used_images = set()
    for path, split, origin in input_files():
        input_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            metadata = row["metadata"]
            if (metadata.get("run"), metadata.get("request")) in excluded_ids:
                applied_exclusions.append({"run": metadata["run"], "request": metadata["request"],
                                           "candidate_file": path.relative_to(ROOT).as_posix()})
                continue
            dataset, case = metadata["dataset"].lower(), str(metadata["case"])
            role = metadata["role"].lower()
            family = dataset + ":" + case
            identifier = (metadata.get("run", metadata.get("construction")),
                          metadata.get("request", metadata.get("state")), role)
            if identifier in identifiers:
                raise ValueError(f"Duplicate decision: {identifier}")
            identifiers.add(identifier)
            if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
                raise ValueError(f"Unexpected role sequence in {path}")
            if metadata.get("loss_scope") != "assistant_output_only":
                raise ValueError(f"Missing assistant loss scope in {path}")
            if row["messages"][1]["content"].count("<image>") != len(row["images"]):
                raise ValueError(f"Image placeholder count differs from attached pixels in {path}")
            packaged_images = []
            for image in row["images"]:
                source = Path(image).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                key = str(source)
                if key not in image_manifest:
                    image_manifest[key] = f"images/image_{len(image_manifest) + 1:06d}.png"
                relative = image_manifest[key]
                target = destination / relative
                if not target.is_file():
                    shutil.copyfile(source, target)
                packaged_images.append(relative)
                used_images.add(relative)
            row["images"] = packaged_images
            keys = [key for key in source_keys(dataset, case) if key]
            row["metadata"] = {**metadata, "usage_split": split, "origin": origin,
                               "case_family": family, "known_source_keys": keys,
                               "candidate_file": path.relative_to(ROOT).as_posix()}
            if dataset in {"averitec", "averimatec", "true"}:
                run_name = metadata.get("run", metadata.get("source_run"))
                run_config = read(ROOT / "runs" / str(run_name) / "run_config.json", {})
                row["metadata"]["evidence_source_policy"] = run_config.get(
                    "evidence_source_policy", "unrestricted_retrospective_web")
                row["metadata"]["text_search_parser"] = run_config.get("text_search_parser", "installed_ddgs")
                row["metadata"]["source_condition"] = run_config.get("source_condition", "source_discovery")
                row["metadata"]["provided_source_url"] = run_config.get("provided_source_url")
            family_splits[family].add(split)
            for key in keys:
                source_splits[key].add(split)
            groups[(split, role)].append(row)
            input_count += 1
        sources.append({"file": path.relative_to(ROOT).as_posix(), "rows": input_count,
                        "usage_split": split, "origin": origin})
    collisions = {"case_families": {k: sorted(v) for k, v in family_splits.items() if len(v) > 1},
                  "known_sources": {k: sorted(v) for k, v in source_splits.items() if len(v) > 1}}
    if any(collisions.values()):
        raise ValueError(f"Development and validation share known case/material sources: {collisions}")
    counts = []
    for (split, role), rows in sorted(groups.items()):
        path = destination / f"{role}_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        counts.append({"file": path.name, "role": role.upper(), "usage_split": split, "rows": len(rows),
                       "case_families": len({r["metadata"]["case_family"] for r in rows}),
                       "origins": dict(Counter(r["metadata"]["origin"] for r in rows)),
                       "target_ratings": dict(Counter(r["metadata"]["target_rating"] for r in rows))})
    image_manifest_path.write_text(json.dumps(image_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"files": counts, "images_used": len(used_images), "sources": sources,
               "curation_exclusions": exclusions, "applied_exclusions": applied_exclusions,
               "split_check": "no_overlap_in_case_ids_or_known_source_keys; event-level_semantic_deduplication_not_established",
               "quality_scope": "automated_candidates; not_independent_human_certification",
               "training_status": "student_weight_training_not_run",
               "processing_checks": [{"file": path.relative_to(ROOT).as_posix(),
                                      "model": read(path)["model"],
                                      "rows": sum(item["processed"] for item in read(path)["files"])}
                                     for path in sorted(ROOT.glob("student_inputs_*/summary.json"))],
               "evidence_source_audit": "../evidence_source_audit.json",
               "image_paths_relative_to": "corpus"}
    (destination / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# 分角色 SFT 候选语料", "",
            "每行只监督最后一条 assistant 输出。system、user、工具观察及其中出现的其他角色内容不计算目标损失。图片路径相对于本目录，须由学生模型的多模态模板逐一对应 `<image>`。", "",
            "本地图片包中可有指向原始请求图片的同卷硬链接，供不可变训练素材复用存储；不要原地编辑图片。复制本目录到其他文件系统时，图片内容随普通文件复制。具体批次的存储记录保存在运行目录 reference_review/packaged_image_storage.json。", "",
            "development 文件供开发与后续训练准备；construction_validation 文件是数据构造方法的保留检查样本，不能混入开发训练。它们不是官方测试集。", "",
            "origin=hi_independent 为独立 HI 真实执行，origin=gia_offline 为离线 GIA 单步决定，origin=bridge 为真实协作中的对应角色输出。离线决定中的补查提议未被执行。", "",
            "origin=hi_terminal_replay 为从已记录的 HI 报告提交前状态重新生成的报告。它保留实际输入和图片，前序工具记录真实；没有新增工具执行，也没有复用原报告之后的 GIA 决定。", "",
            "网页任务的 source_condition 区分来源发现与 provided_source_url。后者的候选地址已在任务中提供，仍须实际打开并核验；它训练来源阅读与取证，不能计为自主搜索发现成功。provided_source_url 保存实际提供的地址。", "",
            "source_condition=recorded_plan_execution_with_provided_evidence 表示从已记录的GIA计划启动新调查，并提供先前取得的真实来源正文或媒体证据。首项计划只用于启动执行，不重复导出为新模型目标；新HI行动、报告和之后的实际GIA决定分别评审。plan_origin.json记录原始计划、证据编号对应及来源。此条件不计为从零发现来源，也不代表恢复了原调查的全部历史。", "",
            "候选经过逐目标自动评审和真实输入、输出、图片对应检查；已知的任务含义、事实范围或评分分歧记录在manifest.json的curation_exclusions中，相应目标不收录。没有独立人工认证或学生权重训练。实际Qwen3.5-9B分词与图片处理批次见manifest.json的processing_checks；每批记录其覆盖行数，旧批次不代表后加样本已检查。processing_coverage.json按当前保留决策编号匹配已完成处理记录，不将已排除目标计为当前覆盖。四个完整输入在本机RTX 5090上完成前向损失计算，不能据此推定A100的长上下文训练容量。相同案例的多个运行和状态变体属于同一家族，不能按行随机拆分训练与测试。", "",
            "网页取证样本的 evidence_source_policy 区分 unrestricted_retrospective_web 与 exclude_known_original_article; publication_dates_not_filtered。前者允许回顾性取证，后者在工具返回前排除已知原事实核查文章及明确登记的别名；两者均未统一限制来源必须早于声明日期，不能视为符合 AVeriTeC / AVerImaTeC 原始评测取证规则的独立调查结果。来源比对见 [来源检查](../evidence_source_audit.json)；没有匹配已知文章 URL，不表示已排除所有镜像、转载或时间泄漏。", "",
            "| 文件 | 角色 | 决策行 | 来源案例家族 |", "|---|---|---:|---:|"]
    for item in counts:
        text.append(f"| [{item['file']}]({item['file']}) | {item['role']} | {item['rows']} | {item['case_families']} |")
    text += ["", "manifest.json 记录来源与角色构成；image_manifest.json 将原始图片路径对应到随包图片。已检查案例编号及已知素材来源不跨开发与保留划分，尚不能排除跨数据集的语义事件重叠。", ""]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"files": counts, "images_used": len(used_images)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
