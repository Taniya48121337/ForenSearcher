"""Read the selected training experiments without running models or changing their data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException


CATALOG: list[dict[str, Any]] = [
    {"id": "video-intervals", "directory": "runs/videosham_4126_split_intervals_r7", "title": "人物差异，究竟持续了多久？", "description": "GIA 分段安排调查，HI 成对抽帧核查。沿着 106 次输出，查看边界证据与未解决的空档。", "tags": ["真实协作", "边界核查", "证据保留"], "kind": "video"},
    {"id": "gia-evidence-states", "directory": "gia_decisions/videosham_4126_task_states_r1", "title": "同样的证据，为什么决定不同？", "description": "保留同一份真实报告，切换四个问题，比较 GIA 何时回答、何时补查。", "tags": ["GIA 独立", "四种问题", "停止判断"], "kind": "contrast"},
    {"id": "video-local-intervals", "directory": "runs/videosham_4126_full_intervals_r7", "title": "找到了差异，也找全了吗？", "description": "一次局部视频调查：已确认的帧、关键证据的流失，以及准确的未完成报告。", "tags": ["HI 独立", "逐步评审", "未完成报告"], "kind": "video"},
    {"id": "video-scene-difference", "directory": "runs/videosham_0069_r1", "title": "两个视频，画面含义有何变化？", "description": "从全局分工到局部取证，对照画面差异、具体证据与结论范围。", "tags": ["真实协作", "画面比较"], "kind": "video"},
    {"id": "video-moment", "directory": "runs/qvhighlights_9167_r2", "title": "定位机库里并排站立的人群", "description": "根据自然语言描述寻找视频片段，检查命中位置和完整时间范围的区别。", "tags": ["HI 独立", "时间定位"], "kind": "video"},
    {"id": "video-correspondence", "directory": "runs/vcsl_pair_4_r1", "title": "两段视频，哪些片段对应？", "description": "核对两侧时间范围及画面对应关系，保留对原创归属和音频的判断边界。", "tags": ["HI 独立", "跨视频对应"], "kind": "video"},
    {"id": "speech-context", "directory": "runs/true_20361105_visibility_validation", "title": "一句引语，能支持整项声明吗？", "description": "结合媒体与网页调查 Frederica Wilson 的言论，核对原话、语境和未验证内容。", "tags": ["真实协作", "引语核查"], "kind": "web"},
    {"id": "mask-history", "directory": "runs/true_39132300_validation_r3", "title": "面具和电影，谁先出现？", "description": "检索 Ghostface 面具的历史来源，核对电影上映前是否已有销售记录。", "tags": ["真实协作", "来源时间"], "kind": "web"},
    {"id": "dog-source", "directory": "runs/averimatec_652_q1_visibility_validation", "title": "流浪犬的故事，来源说了什么？", "description": "从问题图片与报道入手，查明 Southlake DPS 的处置说法，保留来源和适用范围。", "tags": ["HI 独立", "图文取证"], "kind": "image"},
    {"id": "archived-source", "directory": "runs/averimatec_63_q0_archived_source_r1", "title": "打开存档，核对机构原文", "description": "使用给定的存档地址读取机构资料，核对原文内容与存档时间限制。", "tags": ["HI 独立", "给定来源", "网页存档"], "kind": "web"},
    {"id": "sea-level", "directory": "runs/averimatec_423_q0_validation_r3", "title": "海平面变化，概念先说清楚", "description": "阅读图文材料与来源，核对术语定义及其能回答的问题范围。", "tags": ["HI 独立", "图文阅读"], "kind": "image"},
    {"id": "university-budget", "directory": "runs/averitec_2694_budget_plan_execution_r1", "title": "批准预算，等于实际支出吗？", "description": "两所大学的建设支出调查：读报告、核对阶段和金额，区分预算、合同与实际支出。", "tags": ["真实协作", "数字口径", "长文档"], "kind": "web"},
]

# 第 0 轮三层核验的汇总（按优先顺序取第一个存在的）。summary49 是正式口径：49 个 run / 80 份报告，
# 第 50 个 run 未跑完未计入；jsonl.summary 是 reverify 自动写出的全量汇总，作为兜底。
VERIFICATION_SUMMARIES = ("verdicts_r0_judged.summary49.json", "verdicts_r0_judged.jsonl.summary.json")

DATASETS = {"videosham": "VideoSham", "qvhighlights": "QVHighlights", "vcsl": "VCSL", "true": "TRUE", "averitec": "AVeriTeC", "averimatec": "AVerImaTeC"}


def read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def view_of(request: dict[str, Any]) -> dict[str, Any]:
    payload = request.get("payload", {})
    return payload.get("model_view", payload.get("global_view", {}))


class TrainingReader:
    def __init__(self, project: Path) -> None:
        self.root = project / "experiments/sft_data"
        self.definitions = {entry["id"]: entry for entry in CATALOG}
        self._corpus_stamp: tuple[Any, ...] = ()
        self._candidates: dict[tuple[str, str], dict[str, Any]] = {}
        self._samples: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        self._events: dict[str, tuple[int, dict[int, dict[str, Any]]]] = {}

    def _refresh_corpus(self) -> None:
        paths = sorted((self.root / "corpus").glob("*.jsonl"))
        stamp = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if stamp == self._corpus_stamp:
            return
        candidates = {}
        for path in paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    meta = row["metadata"]
                    key = (meta.get("run", meta.get("construction", "")), meta.get("request", meta.get("state", "")))
                    candidates[key] = {k: meta.get(k) for k in ("role", "dataset", "case", "usage_split", "origin", "target_rating", "assessment", "case_family")}
        self._candidates = candidates
        self._corpus_stamp = stamp
        self._samples.clear()

    def _directory(self, sample_id: str) -> tuple[dict[str, Any], Path]:
        definition = self.definitions.get(sample_id)
        if definition is None:
            raise HTTPException(404, "样本不存在")
        directory = (self.root / definition["directory"]).resolve()
        if not directory.is_relative_to(self.root.resolve()) or not directory.is_dir():
            raise HTTPException(404, "样本材料尚未就绪")
        return definition, directory

    def _assessment(self, directory: Path) -> tuple[Path, str]:
        selected = read(directory / "selected_assessment.json", {})
        name = selected.get("assessment", "assessment_r2")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise HTTPException(404, "评审记录不存在")
        rubric = selected.get("rubric_file") or ("rubric_r6.md" if "r6" in name else "rubric_r5.md" if "r5" in name else "rubric_r4.md" if "r4" in name else "rubric_r3.md" if "r3" in name else "rubric_r2.md")
        return path, rubric.removeprefix("rubric_").removesuffix(".md")

    def _load_sample(self, sample_id: str) -> dict[str, Any]:
        self._refresh_corpus()
        definition, directory = self._directory(sample_id)
        assessment, version = self._assessment(directory)
        dependencies = [directory / "run_config.json", directory / "execution_links.json", assessment / "judgment.json", directory / "selected_assessment.json"]
        stamp = tuple((p.name, p.stat().st_mtime_ns) for p in dependencies if p.exists())
        cached = self._samples.get(sample_id)
        if cached and cached[0] == stamp:
            return cached[1]
        judgment = read(assessment / "judgment.json", {})
        config = read(directory / "run_config.json", {})
        offline = definition["kind"] == "contrast"
        steps = []
        if offline:
            targets = {t["id"]: t for t in judgment.get("states", [])}
            for state in sorted(directory.glob("state_*")):
                req_path = state / "requests/request_0001/request.json"
                if not req_path.exists():
                    continue
                request = read(req_path)
                output = read(state / "decision.json", read(req_path.parent / "response.json", {}))
                # decision.json may include the response together with construction metadata.
                if "decision" not in output and "output" in output:
                    output = output["output"]
                ref = read(state / "reference.json", {})
                config = {**ref.get("provenance", {}), "mode": "offline"}
                steps.append(self._step_summary(directory, state.name, request, output, targets.get(state.name, {}), "GIA", "GIA", "offline_decision", req_path.parent, None))
        else:
            targets = {t["request_id"]: t for t in judgment.get("target_findings", [])}
            for link in read(directory / "execution_links.json", []):
                request_id = link["request_id"]
                if not re.fullmatch(r"request_[0-9]+", request_id):
                    continue
                req_dir = directory / "requests" / request_id
                if not (req_dir / "request.json").exists():
                    continue
                request = read(req_dir / "request.json")
                steps.append(self._step_summary(directory, request_id, request, link.get("raw_output", {}), targets.get(request_id, {}), link.get("role", "HI"), link.get("actor", "HI"), link.get("status", "unknown"), req_dir, link))
        if not steps:
            raise HTTPException(404, "没有可展示的决策记录")
        candidates = [s for s in steps if s["included"]]
        roles = sorted({s["role"] for s in steps})
        first_image = next((s for s in steps if s["image_count"]), None)
        dataset = config.get("dataset", candidates[0]["candidate"].get("dataset") if candidates else "")
        config_task = config.get("task")
        if not config_task:
            first_request = read(steps[0]["_request_dir"] / "request.json")
            first_view = view_of(first_request)
            config_task = first_view.get("task", "")
        dimensions = judgment.get("roles", [])
        summary = {k: definition[k] for k in ("id", "title", "description", "tags", "kind")}
        summary.update(dataset=DATASETS.get(dataset, dataset), case=str(config.get("case", "")),
                       mode="gia_offline" if offline else "bridge" if config.get("mode", "bridge") == "bridge" else "hi_independent",
                       roles=roles, step_count=len(steps), included_count=len(candidates), excluded_count=len(steps)-len(candidates),
                       usage_split=(candidates[0]["candidate"].get("usage_split") if candidates else config.get("usage", "development")),
                       rubric_version=version, task=config_task, source_condition=config.get("source_condition", "provided_media" if definition["kind"] == "video" else "source_discovery"),
                       evidence_source_policy=config.get("evidence_source_policy"),
                       thumbnail=f"/api/training/samples/{sample_id}/steps/{first_image['id']}/images/0" if first_image else None,
                       source=definition["directory"])
        result = {**summary, "steps": steps, "role_assessments": dimensions, "_directory": directory}
        self._samples[sample_id] = stamp, result
        return result

    def _step_summary(self, directory: Path, request_id: str, request: dict[str, Any], output: dict[str, Any], target: dict[str, Any], role: str, actor: str, execution: str, req_dir: Path, link: Any) -> dict[str, Any]:
        candidate = self._candidates.get((directory.name, request_id))
        action = output.get("action", output.get("decision", "REPORT"))
        view = view_of(request)
        operation = output.get("arguments", {}).get("request", {}).get("operation")
        workspace_round = re.fullmatch(r"W([0-9]+)-[0-9]+", actor)
        round_index = (int(workspace_round[1]) if role == "HI" and workspace_round
                       else view.get("runtime_constraints", {}).get("round_index"))
        arguments = output.get("arguments", {})
        preview = (output.get("answer") or output.get("status_reason") or arguments.get("query")
                   or arguments.get("source") or arguments.get("record_id") or "")
        return {"id": request_id, "role": role, "actor": actor, "action": action, "operation": operation,
                "round_index": round_index, "preview": str(preview)[:320], "task_count": len(output.get("tasks", [])),
                "task": view.get("task", ""), "image_count": len(request.get("image_evidence_ids", [])),
                "rating": target.get("target_rating"), "disposition": target.get("disposition", "unknown"),
                "critical_unknown": target.get("critical_unknown"), "reason": target.get("reason", "暂无逐决策评审"),
                "included": candidate is not None, "candidate": candidate, "execution": execution,
                "report_status": output.get("status"), "_output": output, "_request_dir": req_dir,
                "_target": target, "_link": link}

    @staticmethod
    def _public(value: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in value.items() if not k.startswith("_")}

    def overview(self) -> dict[str, Any]:
        self._refresh_corpus()
        manifest = read(self.root / "corpus/manifest.json", {})
        files = manifest.get("files", [])
        samples = []
        for sample_id in self.definitions:
            try:
                value = self._load_sample(sample_id)
                samples.append({k: v for k, v in self._public(value).items() if k not in {"steps", "role_assessments"}})
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
        return {"stats": {"total": sum(f["rows"] for f in files),
                          "development": sum(f["rows"] for f in files if f["usage_split"] == "development"),
                          "validation": sum(f["rows"] for f in files if f["usage_split"] == "construction_validation"),
                          "gia": sum(f["rows"] for f in files if f["role"] == "GIA"),
                          "hi": sum(f["rows"] for f in files if f["role"] == "HI"),
                          "images": manifest.get("images_used", 0),
                          "datasets": len({str(v.get("dataset", "")).lower() for v in self._candidates.values()})},
                "verification": self.verification(),
                "files": files, "samples": samples, "rubric_version": "r6",
                "training_status": manifest.get("training_status"), "quality_scope": manifest.get("quality_scope")}

    def verification(self) -> dict[str, Any]:
        """第 0 轮三层核验的规模：页面上的"已核验调查 / 报告 / 证据对照 / 数据来源"只从这里取数。"""
        for name in VERIFICATION_SUMMARIES:
            data = read(self.root / name)
            if not isinstance(data, dict) or not data.get("reports"):
                continue
            datasets = [key for key in data.get("by_dataset", {}) if key != "unknown"]
            return {"runs": data.get("runs"), "reports": data["reports"], "pairs": data.get("pairs"),
                    "findings": data.get("findings"), "datasets": len(datasets) or None,
                    "verifier": data.get("verifier"), "source": name}
        return {}

    def sample(self, sample_id: str) -> dict[str, Any]:
        value = self._load_sample(sample_id)
        return {**self._public(value), "steps": [self._public(step) for step in value["steps"]]}

    def _step(self, sample_id: str, step_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        sample = self._load_sample(sample_id)
        step = next((step for step in sample["steps"] if step["id"] == step_id), None)
        if step is None:
            raise HTTPException(404, "决策记录不存在")
        return sample, step

    def step(self, sample_id: str, step_id: str) -> dict[str, Any]:
        sample, step = self._step(sample_id, step_id)
        request = read(step["_request_dir"] / "request.json")
        notes = {}
        for note in request.get("image_notes", []):
            if isinstance(note, str) and note.startswith("{"):
                try:
                    parsed = json.loads(note)
                    notes[parsed.get("evidence_id")] = parsed
                except ValueError:
                    pass
        images = [{"evidence_id": eid, "url": f"/api/training/samples/{sample_id}/steps/{step_id}/images/{i}", "note": notes.get(eid, {})}
                  for i, eid in enumerate(request.get("image_evidence_ids", []))]
        observation = None
        link = step["_link"]
        if link and link.get("observation_event_id"):
            path = sample["_directory"] / "trajectory.jsonl"
            if path.is_file():
                stamp = path.stat().st_mtime_ns
                cached = self._events.get(sample_id)
                if not cached or cached[0] != stamp:
                    events = {}
                    with path.open(encoding="utf-8") as source:
                        for line in source:
                            event = json.loads(line)
                            if event.get("event_type") == "observation":
                                events[event["id"]] = event["payload"].get("observation", event["payload"])
                    self._events[sample_id] = (stamp, events)
                observation = self._events[sample_id][1].get(link["observation_event_id"])
        return {**self._public(step), "payload": request.get("payload", {}), "system_prompt": request.get("system_prompt", ""),
                "image_notes": request.get("image_notes", []), "images": images, "output": step["_output"],
                "observation": observation, "assessment": step["_target"], "rubric_version": sample["rubric_version"],
                "assessment_scope": "state" if sample["mode"] == "gia_offline" else "actor",
                "dimensions": step["_target"].get("dimensions", next((r.get("dimensions", []) for r in sample["role_assessments"] if r.get("actor") == step["actor"]), []))}

    def image_path(self, sample_id: str, step_id: str, index: int) -> Path:
        _, step = self._step(sample_id, step_id)
        if index < 0 or index >= step["image_count"]:
            raise HTTPException(404, "图片不存在")
        path = (step["_request_dir"] / "actor" / f"image_{index + 1:02d}.png").resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise HTTPException(404, "图片尚未就绪")
        return path

    def document_path(self, name: str) -> Path:
        names = {"method": "方法与数据质量标准.md", "results": "构造方法实测分析.md", "rubric": "rubric_r6.md"}
        if name not in names or not (self.root / names[name]).is_file():
            raise HTTPException(404, "文档不存在")
        return self.root / names[name]
