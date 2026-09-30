"""读取一个 run 目录（trajectory.jsonl + execution_links.json + requests/ + store.sqlite3）并整理成"决策单元"。

大白话：一个 run 里发生过的每一次模型决策（GIA 派工 / GIA 结束 / HI 每一步 / HI 报告），连同它当时看到的
桌面（request.json 里的 payload）、Harness 的执行结果（observation 事件）、附了哪些图，各自装进一个 Decision。
核验器的 step / plan / answer 三个单元和 export 都从这里取数，不各自再去翻 trajectory。

request.json 是教师**实际收到**的请求（system_prompt、payload、image_evidence_ids、image_notes、actor/image_NN.png），
比 trajectory 里的 model_input 事件多了附图与说明；execution_links.json 把每个 request 对到 raw/accepted 事件与执行状态。
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store import GlobalStore
from .verify.context import StoreLookup

_E = re.compile(r"\bE\d+\b")


@dataclass
class Decision:
    request_id: str                     # request_0007
    role: str                           # HI | GIA
    actor_id: str                       # W1-1 | GIA
    turn: int                           # HI: turn_index；GIA: 该轮内的 turn_index
    round_index: int | None             # GIA 才有
    kind: str                           # step | report | plan | answer | memory（GIA 的 READ_RECORD/QUERY_MEMORY）
    output: dict[str, Any]              # 教师的原始输出（raw_output）
    status: str                         # executed | report_committed | gia_decision_accepted | rejected …
    system_prompt: str
    payload: dict[str, Any]             # 教师看到的 payload（未渲染）
    image_evidence_ids: list[str]
    image_notes: list[str]
    image_paths: list[Path]
    raw_event_id: int | None = None
    observation: dict[str, Any] | None = None   # 这步动作的 Harness 观察（HI step 才有）
    visible_ids: set[str] = field(default_factory=set)         # 该时刻 payload 里出现过的全部 E 编号（含附图）
    content_observed_before: set[str] = field(default_factory=set)  # HI：此前已看过正文的 E（model_view.content_observed_evidence_ids）
    student_system_prompt: str | None = None   # 去掉教师指令后的运行时 system（学生部署时看到的那份）；None = 与 system_prompt 相同

    @property
    def student_system(self) -> str:
        return self.system_prompt if self.student_system_prompt is None else self.student_system_prompt

    @property
    def view(self) -> dict[str, Any]:
        value = self.payload.get("model_view") if self.role == "HI" else self.payload.get("global_view")
        return value if isinstance(value, dict) else {}

    @property
    def action(self) -> str | None:
        if self.kind in ("step", "memory"):
            return self.output.get("action")
        if self.kind == "plan":
            return "INVESTIGATE"
        if self.kind == "answer":
            return "FINISH"
        return self.output.get("status")


@dataclass
class RunLog:
    run_dir: Path
    config: dict[str, Any]
    events: list[dict[str, Any]]
    decisions: list[Decision]
    store: GlobalStore
    lookup: StoreLookup
    _scratch: Path

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def origin(self) -> str:
        """数据来源：A 真实轨迹（默认）/ D 离线换题（datagen.offline_gia）。"""

        return str(self.config.get("origin") or "A")

    @property
    def case_id(self) -> str | None:
        """案卷检查用的 case_id。离线换题（D）换了总任务，原案卷的要求与标签不适用 → None。"""

        if self.origin == "D":
            return None
        return self.family_case_id

    @property
    def family_case_id(self) -> str | None:
        """划分 / family 用的 case_id：D 继承源案例（同一素材不能跨划分）。"""

        dataset = str(self.config.get("dataset") or "").lower()
        case = self.config.get("case")
        return f"{dataset}:{case}" if dataset and case else None

    @property
    def teacher_model(self) -> str | None:
        if self.config.get("gia_output_source") == "scripted_dispatcher":
            return None
        return (self.config.get("configured_model") or {}).get("model")

    def gia_is_model(self) -> bool:
        return self.config.get("gia_output_source") != "scripted_dispatcher"

    def workspaces(self) -> list[str]:
        seen: list[str] = []
        for d in self.decisions:
            if d.role == "HI" and d.actor_id not in seen:
                seen.append(d.actor_id)
        return seen

    def steps(self, workspace: str) -> list[Decision]:
        return [d for d in self.decisions if d.role == "HI" and d.actor_id == workspace]

    def report_decision(self, workspace: str) -> Decision | None:
        for d in self.steps(workspace):
            if d.kind == "report":
                return d
        return None

    def event(self, event_id: int) -> dict[str, Any] | None:
        for e in self.events:
            if e.get("id") == event_id:
                return e
        return None

    def workspace_round(self, workspace: str) -> int:
        m = re.match(r"^W(\d+)-", workspace)
        return int(m.group(1)) if m else 1

    def round_workspaces(self, round_index: int) -> list[str]:
        return [w for w in self.workspaces() if self.workspace_round(w) == round_index]

    def evidence_first_seen(self) -> dict[str, tuple[str, int]]:
        """每个 E 第一次出现在哪个 (actor, turn) 的观察里（判断 SEARCH 结果是"新产生"还是"重新浮现"）。"""

        first: dict[str, tuple[str, int]] = {}
        for e in self.events:
            if e.get("event_type") not in ("observation", "investigation_run"):
                continue
            actor = str(e.get("actor_id"))
            turn = int((e.get("payload") or {}).get("turn_index") or 0)
            for eid in sorted(set(_E.findall(json.dumps(e.get("payload"), ensure_ascii=False))), key=lambda x: int(x[1:])):
                first.setdefault(eid, (actor, turn))
        return first

    def close(self) -> None:
        shutil.rmtree(self._scratch, ignore_errors=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ids_in(value: Any) -> set[str]:
    return set(_E.findall(json.dumps(value, ensure_ascii=False, default=str)))


def student_system_prompt(request: dict[str, Any], config: dict[str, Any], role: str) -> str:
    """教师实际收到的 system = 运行时学生提示 + "\n" + 教师指令。取回学生那一段：

    - 新 run（EndpointTeacher）：request.json 直接记 student_system_prompt；
    - 旧 run：run_config.instruction 是字符串（两个角色共用）或 {"HI": ..., "GIA": ...}，在末尾找到就去掉；
    - 找不到就原样返回（说明当时没拼指令，或指令被改过——那一行照实算不一致）。
    """

    if isinstance(request.get("student_system_prompt"), str):
        return request["student_system_prompt"]
    text = request.get("system_prompt") or ""
    instruction = config.get("instruction")
    if isinstance(instruction, dict):
        instruction = instruction.get(role)
    if isinstance(instruction, str) and instruction:
        for suffix in ("\n" + instruction, "\n" + instruction.strip()):
            if text.endswith(suffix):
                return text[: -len(suffix)]
    return text


def load_run(run_dir: str | Path) -> RunLog:
    root = Path(run_dir)
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8")) if (root / "run_config.json").is_file() else {}
    events = _load_jsonl(root / "trajectory.jsonl")
    links = json.loads((root / "execution_links.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in events}
    # Store 只读：复制一份再开（GlobalStore 打开会建表 + WAL，会动原文件）
    scratch = Path(tempfile.mkdtemp(prefix="runlog_"))
    shutil.copy2(root / "store.sqlite3", scratch / "store.sqlite3")
    # 离线换题的 run 不复制源 run 的媒体（几十 MB），run_config.artifacts_from 指向源 run 的 artifacts 目录
    artifacts = (root / config["artifacts_from"]).resolve() if config.get("artifacts_from") else root / "artifacts"
    store = GlobalStore(scratch / "store.sqlite3", artifacts)
    lookup = StoreLookup(store)

    decisions: list[Decision] = []
    for link in links:
        request_id = link.get("request_id")
        if not request_id:
            continue
        rdir = root / "requests" / request_id
        request = json.loads((rdir / "request.json").read_text(encoding="utf-8"))
        payload = request["payload"]
        role = "HI" if payload.get("workspace_id") else "GIA"
        raw_event = by_id.get(link.get("raw_event_id"))
        raw_payload = (raw_event or {}).get("payload") or {}
        output = link.get("raw_output") or {}
        if role == "HI":
            kind = "report" if "status" in output and "findings" in output else "step"
            actor = str(payload["workspace_id"])
            turn = int(raw_payload.get("turn_index") or (payload.get("model_view") or {}).get("runtime_constraints", {}).get("turn_index") or 0)
            round_index = None
        else:
            decision = output.get("decision")
            kind = "plan" if decision == "INVESTIGATE" else "answer" if decision == "FINISH" else "memory"
            actor = "GIA"
            turn = int(raw_payload.get("turn_index") or 0)
            round_index = raw_payload.get("round_index")
            if round_index is None:
                round_index = (payload.get("global_view") or {}).get("runtime_constraints", {}).get("round_index")
        observation = None
        if link.get("observation_event_id") in by_id:
            observation = (by_id[link["observation_event_id"]].get("payload") or {}).get("observation")
        image_ids = list(request.get("image_evidence_ids") or [])
        image_paths = [rdir / "actor" / f"image_{i + 1:02d}.png" for i in range(len(image_ids))]
        view = payload.get("model_view") if role == "HI" else payload.get("global_view")
        visible = _ids_in(view) | set(image_ids)
        observed = set()
        if role == "HI" and isinstance(view, dict):
            observed = {e for e in (view.get("content_observed_evidence_ids") or []) if isinstance(e, str)}
        decisions.append(Decision(
            request_id=request_id, role=role, actor_id=actor, turn=turn, round_index=round_index, kind=kind,
            output=output, status=str(link.get("status")), system_prompt=request["system_prompt"], payload=payload,
            image_evidence_ids=image_ids, image_notes=list(request.get("image_notes") or []), image_paths=image_paths,
            raw_event_id=link.get("raw_event_id"), observation=observation, visible_ids=visible,
            content_observed_before=observed, student_system_prompt=student_system_prompt(request, config, role),
        ))
    decisions.sort(key=lambda d: int(d.request_id.split("_")[-1]))
    return RunLog(run_dir=root, config=config, events=events, decisions=decisions, store=store, lookup=lookup, _scratch=scratch)


__all__ = ["Decision", "RunLog", "load_run", "student_system_prompt"]
