"""可见策略 walker（2026-10-12，《轨迹生成计划：walker（程序走、教师填），以能力覆盖为纲》D1）。

程序按状态表决定 HI / GIA 的每一步；只读 generate_json 拿到的 payload（= 学生那份视图），案卷、标注、PFT 一律不传进来。
语言部分（报告 Finding、最终答案、第 2 轮任务文字）不在这里写：这里只给出"该交报告 / 该 FINISH / 该派第 2 轮"
以及程序定下的 status、允许引用的 E、写作要点，由 OracleGenerator 调教师槽（S4 / S5 / S2）填。

本文件只做 D1 范围：
- B 族（给定媒体，不搜索）：两段视频的比对 / 对应（videosham、vcsl）与单段视频按描述定位（qvhighlights）；
- 通用规则：remaining_turns=0 交报告、失败不重发、同参数不重复、比对阶段两侧各留一半容量；
- GIA：第 1 轮派任务、全部报告后亲眼看引用图、INCOMPLETE 且可再开一轮时派第 2 轮、否则 FINISH。
D2（10-13）：A 族（找原片：OCR / ASR → S1 槽写查询 → 以图搜 → content_match 对上的候选 compare_media）与
C 族（核查声明：ASR / OCR → S3 槽在"再搜 / 读一页 / 交报告"里选，程序限预算）；B 族补了横向变化区域的 OCR。

决策函数是纯函数：decide_hi(payload, memory) / decide_gia(payload, memory) → Decision。memory 是同一工作区 / GIA
自己以前发过的决定（walker 自己的历史，学生也能从 history 看到），用于"已发过的不再发"。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------- 结果对象


@dataclass
class Decision:
    """walker 的一步。kind=action：output 就是最终 JSON；kind=report / finish / investigate_slot：
    output 是骨架（程序定的字段），slot 说明要教师填什么。"""

    kind: str                                   # action | report | finish | investigate_slot
    rule: str                                   # 命中的状态表规则名（导出按分支计数用）
    output: dict[str, Any]
    branch: str = ""                            # §2 覆盖矩阵的分支名
    slot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rule": self.rule, "branch": self.branch, "output": self.output, "slot": self.slot}


# ---------------------------------------------------------------------------- 读可见状态

_PROFILE = re.compile(r"media profile:\s*(\w+),\s*([\d.]+)\s*s,\s*(\d+)\D(\d+),\s*([\d.]+)\s*fps", re.I)
_AUDIO = re.compile(r"(\d+)\s+audio track", re.I)
_EID = re.compile(r"^E[1-9][0-9]*$")
_NO_AUDIO_TASK = re.compile(r"do not assess audio|only the video track|video track only", re.I)
_BOUNDARY_TASK = re.compile(r"time ranges?|ranges? on both sides|start and end|boundar|first and last|where (?:it|they|each) (?:starts?|begins?|ends?)"
                            r"|from when|how long", re.I)
_VISUAL_TASK = re.compile(r"\b(?:visible|visibility|appears? (?:in|on) (?:the )?(?:frame|picture|screen))\b", re.I)
_SOUND_WORDS = re.compile(r"\b(?:audio|sound|speech|spoken|says?|said|voice|narrat\w*|music)\b", re.I)


def _visual_only(task: str) -> bool:
    return bool(_VISUAL_TASK.search(task)) and not _SOUND_WORDS.search(task)


_COPY_TASK = re.compile(r"segments?\s+(?:of\s+\w+\s+and\s+\w+\s+)?that\s+correspond|correspond(?:ing)?\s+to\s+each\s+other|copied\s+segment", re.I)
_LOCATE_DESC = re.compile(r"(?:locate|find)[^:]*segment[^:]*:\s*(.+)$", re.I | re.S)


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _r(value: float, nd: int = 3) -> float:
    return float(round(value, nd))


class HiState:
    """从一份 HI payload 里取 walker 要的东西（只读）。"""

    def __init__(self, payload: dict[str, Any], memory: dict[str, Any] | None = None, rendered: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.memory = memory if memory is not None else {}
        # 学生实际看到的文字（按规格渲染后的状态）：判断"某个数现在还看得见吗"用它
        self.visible_text = json.dumps(rendered if rendered is not None else payload, ensure_ascii=False)
        self.view: dict[str, Any] = payload.get("model_view") or {}
        self.task: str = str(self.view.get("task") or "")
        self.task_id: str = str(self.view.get("task_id") or payload.get("workspace_id") or "")
        rc = self.view.get("runtime_constraints") or {}
        self.remaining = int(rc.get("remaining_turns") if rc.get("remaining_turns") is not None else 99)
        self.limit = int(self.view.get("image_input_limit") or 8)
        self.actions = {a.get("name") for a in payload.get("available_actions") or [] if isinstance(a, dict)}
        self.inspect_ops = self._inspect_ops(payload)
        # 卡片：materials + evidence_view + 各次观察里带的 evidence
        self.cards: dict[str, dict[str, Any]] = {}
        for card in (self.view.get("materials") or {}).values():
            self._add_card(card)
        for card in self.view.get("evidence_view") or []:
            self._add_card(card)
        self.observations = self._observations()
        for obs in self.observations:
            for card in obs.get("evidence") or []:
                self._add_card(card)
        # r1（10-17）：SEARCH 结果材料里的 E（页面、页内主图、缩略图）也是学生看过的卡片。以前只认 evidence_view 与 INSPECT 观察的
        # evidence：卡片清单满了（max_evidence_cards=12）之后，第 2 轮打开再读的页面不在卡片里，S4 的可引 E 清单里没有它——
        # r1s2 l208 W2-1 读到 geologyscience 页（写着 Peru）却"按 evidence-ID 限制不能引"
        if "SEARCH" in self.actions:
            for search in _search_obs(self):
                for result in (search.get("details") or {}).get("results") or []:
                    for material in result.get("materials") or []:
                        if material.get("status") in ("ready", "partial"):
                            for card in material.get("evidence") or []:
                                self._add_card(card)
        # 卡片清单有上限（max_evidence_cards），早先看到的卡片存进 memory，后面照样认得这些 E
        known = self.memory.setdefault("cards", {})
        for eid, card in self.cards.items():
            known[eid] = {**known.get(eid, {}), **card}
        for eid, card in known.items():
            self.cards.setdefault(eid, card)
        self.observed = set(self.view.get("content_observed_evidence_ids") or [])
        self.attached = [item.get("evidence_id") for item in self.view.get("image_inputs") or [] if isinstance(item, dict)]
        self.omitted = [item.get("evidence_id") if isinstance(item, dict) else item
                        for item in self.view.get("image_inputs_omitted") or []]
        self.invalid = payload.get("validation_error") or payload.get("previous_invalid_output")

    def compare_params(self, **params: Any) -> dict[str, Any]:
        """compare_media 的参数：工具目录里没有 mode（compact_tool_params，hi@0.10.0 起）就不写 mode。"""

        if not self._compare_has_mode():
            params.pop("mode", None)
        return params

    def _compare_has_mode(self) -> bool:
        for action in self.payload.get("available_actions") or []:
            if not isinstance(action, dict) or action.get("name") != "INSPECT":
                continue
            for branch in (action.get("schema") or {}).get("oneOf") or []:
                try:
                    request = branch["properties"]["request"]["properties"]
                    if request["operation"]["const"] == "compare_media":
                        return "mode" in (request["params"].get("properties") or {})
                except (KeyError, TypeError, AttributeError):
                    continue
        return True

    @staticmethod
    def _inspect_ops(payload: dict[str, Any]) -> set[str]:
        ops: set[str] = set()
        for action in payload.get("available_actions") or []:
            if not isinstance(action, dict) or action.get("name") != "INSPECT":
                continue
            for branch in (action.get("schema") or {}).get("oneOf") or []:
                try:
                    ops.add(branch["properties"]["request"]["properties"]["operation"]["const"])
                except (KeyError, TypeError):
                    continue
        return ops

    def _add_card(self, card: Any) -> None:
        if isinstance(card, dict) and card.get("evidence_id"):
            merged = dict(self.cards.get(card["evidence_id"]) or {})
            merged.update({k: v for k, v in card.items() if v is not None})
            self.cards[card["evidence_id"]] = merged

    def _observations(self) -> list[dict[str, Any]]:
        """history 里每一步的 (请求, 观察)。最后一条的 observation 是 null，用 latest_observation。

        观察太长时状态里只有截断的 preview（学生看到的也是它）：从 preview 里尽量读出 status、结果 E、新 E 与完整的
        段 / 候选对象；读到的结果按执行顺序存进 memory["parsed"]，之后的状态里同一步被截得更短时沿用当时读到的。"""

        history = list(self.view.get("history") or [])
        latest = self.view.get("latest_observation")
        parsed_cache: dict[str, Any] = self.memory.setdefault("parsed", {})
        out: list[dict[str, Any]] = []
        for index, item in enumerate(history):
            observation = item.get("observation")
            if observation is None and index == len(history) - 1:
                observation = latest
            request = item.get("request") or {}
            inner = request.get("request") or {}
            key = f"{index}:{item.get('action')}:{json.dumps(request, sort_keys=True, ensure_ascii=False)}"
            parsed = _parse_observation(observation or {}, inner.get("operation"))
            cached = parsed_cache.get(key)
            if cached and _richness(cached) >= _richness(parsed):
                parsed = cached
            elif observation is not None:
                parsed_cache[key] = parsed
            out.append({"action": item.get("action"), "targets": request.get("targets") or [],
                        "operation": inner.get("operation") or parsed.get("operation"),
                        "params": inner.get("params") or {}, "request": request, **parsed})
        # 历史有条数上限（max_history_entries）：早先的步会整个掉出视图。walker 记得自己走过的步（obs_log），
        # 掉出去的补在前面；交报告前要引用的结果看不到了，由 _before_report 先 QUERY_MEMORY 查回来
        log: list[dict[str, Any]] = self.memory.setdefault("obs_log", [])
        keys = [f"{item.get('action')}:{json.dumps(item.get('request') or {}, sort_keys=True, ensure_ascii=False)}"
                for item in history]
        logged = {entry["k"] for entry in log}
        for key, obs in zip(keys, out):
            if key not in logged:
                log.append({"k": key, "obs": obs})
                logged.add(key)
            else:
                for entry in log:
                    if entry["k"] == key and _richness(obs) >= _richness(entry["obs"]):
                        entry["obs"] = obs
        current = set(keys)
        dropped = [entry["obs"] for entry in log if entry["k"] not in current]
        return dropped + out if dropped else out

    # -- 便捷
    def videos(self) -> list[str]:
        mats = self.view.get("materials") or {}
        ids = [card.get("evidence_id") for card in mats.values()
               if isinstance(card, dict) and "video" in (card.get("modalities") or []) and not card.get("derived_from")]
        return sorted((i for i in ids if i), key=lambda e: int(e[1:]))

    def profile(self, video: str) -> dict[str, Any]:
        """media_profile 卡片（derived_from=[video]）里的时长、帧率、音轨数；没有就用材料卡片的 duration_s。"""

        cached = self.memory.setdefault("profiles", {}).get(video)
        if cached and cached.get("fps"):
            return cached
        info: dict[str, Any] = {"duration": _num((self.cards.get(video) or {}).get("duration_s")), "fps": None, "audio": None}
        for card in self.cards.values():
            if card.get("acquired_by") != "INSPECT/media_profile" or video not in (card.get("derived_from") or []):
                continue
            label = str(card.get("label") or "")
            m = _PROFILE.search(label)
            if m:
                info["duration"] = info["duration"] or _num(m.group(2))
                info["fps"] = _num(m.group(5))
                info["w"], info["h"] = _num(m.group(3)), _num(m.group(4))
            a = _AUDIO.search(label)
            if a:
                info["audio"] = int(a.group(1))
            elif "no audio" in label.lower():
                info["audio"] = 0
        self.memory["profiles"][video] = info
        return info

    def last(self, operation: str, **match: Any) -> dict[str, Any] | None:
        for obs in reversed(self.observations):
            if obs["operation"] != operation:
                continue
            if all(obs["params"].get(k) == v for k, v in match.items()):
                return obs
        return None

    def seen(self, evidence_id: str) -> bool:
        return evidence_id in self.observed or evidence_id in self.attached

    def label(self, evidence_id: str) -> str:
        return str((self.cards.get(evidence_id) or {}).get("label") or "")


# ---------------------------------------------------------------------------- 观察解析（含截断的 preview）

_RESULT_EVIDENCE = re.compile(r"Result Evidence:\s*((?:E[1-9][0-9]*(?:,\s*)?)+)")
_RETURNED = re.compile(r"returned\s+(\w+)")


def _balanced_objects(text: str, start_key: str) -> list[dict[str, Any]]:
    """在一段（可能被截断的）JSON 文字里找所有以 {"start_key" 开头、括号完整的对象。"""

    out: list[dict[str, Any]] = []
    needle = '{"' + start_key + '"'
    pos = text.find(needle)
    while pos != -1:
        depth, i, in_str, esc = 0, pos, False, False
        end = -1
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end == -1:
            break
        try:
            out.append(json.loads(text[pos:end + 1]))
        except ValueError:
            pass
        pos = text.find(needle, end + 1)
    return out


def _richness(parsed: dict[str, Any]) -> int:
    result = parsed.get("result") or {}
    return (len(result.get("segments") or []) + len(result.get("matches") or []) + len(result.get("frames") or [])
            + len(result.get("candidates") or []) + len(result.get("images") or []) + (1 if parsed.get("status") else 0))


def _parse_observation(observation: dict[str, Any], operation: str | None) -> dict[str, Any]:
    details = observation.get("details") or {}
    result = details.get("result")
    parsed: dict[str, Any] = {"status": details.get("status"), "result": result if isinstance(result, dict) else {},
                              "result_evidence_id": details.get("result_evidence_id"),
                              "evidence": details.get("evidence") or [],
                              "created": ((details.get("execution") or {}).get("created_evidence_ids") or []),
                              "operation": details.get("operation") or operation, "truncated": False}
    text = ""
    if observation.get("preview"):
        text = str(observation["preview"])
    elif isinstance(result, str):
        text = result
    if not text:
        return parsed
    parsed["truncated"] = True
    items_text = text[:2000]
    if not parsed["status"]:
        m = _RETURNED.search(items_text) or re.search(r'"status":\s*"(\w+)"', text)
        parsed["status"] = m.group(1) if m else None
    m = _RESULT_EVIDENCE.search(items_text)
    if m:
        ids = [x.strip() for x in m.group(1).split(",") if x.strip()]
        parsed["result_evidence_id"] = parsed["result_evidence_id"] or (ids[0] if ids else None)
        parsed["created"] = parsed["created"] or ids
    res = dict(parsed["result"])
    if not res.get("segments"):
        segs = _balanced_objects(text, "start_frame_a")
        if segs:
            res["segments"] = segs
    if not res.get("matches"):
        matches = _balanced_objects(text, "range_a_s")
        if matches:
            res["matches"] = [m for m in matches if "range_b_s" in m]
    if not res.get("images"):
        imgs = [o for o in _balanced_objects(text, "evidence_id") if "segment_index" in o]
        if imgs:
            res["images"] = imgs
    if not res.get("frame_rate"):
        m = re.search(r'"frame_rate":\s*\{"a":\s*([\d.]+),\s*"b":\s*([\d.]+)\}', text)
        if m:
            res["frame_rate"] = {"a": float(m.group(1)), "b": float(m.group(2))}
    if not res.get("frames"):
        frames = [o for o in _balanced_objects(text, "evidence_id") if "time_s" in o and "segment_index" not in o and "score" not in o]
        if frames:
            res["frames"] = frames
    if not res.get("candidates"):
        cands = [o for o in _balanced_objects(text, "evidence_id") if "score" in o]
        if cands:
            res["candidates"] = cands
    summary = re.search(r'"summary":\s*"((?:[^"\\]|\\.)*)"', text)
    if summary and not res.get("summary"):
        res["summary"] = summary.group(1)
    parsed["result"] = res
    return parsed


def _existing(state: HiState, ids: list[Any], cap: int) -> list[str]:
    out: list[str] = []
    for item in ids:
        if isinstance(item, str) and _EID.match(item) and item in state.cards and item not in out:
            if "image" in ((state.cards.get(item) or {}).get("modalities") or ["image"]):
                out.append(item)
        if len(out) >= cap:
            break
    return out


def _inspect(targets: list[str], operation: str, params: dict[str, Any], focus: list[str], rationale: str) -> dict[str, Any]:
    return {"action": "INSPECT", "arguments": {"targets": targets, "request": {"operation": operation, "params": params}},
            "focus_evidence_ids": focus, "rationale": rationale}


def _read(record_id: str, focus: list[str], rationale: str) -> dict[str, Any]:
    return {"action": "READ_RECORD", "arguments": {"record_id": record_id}, "focus_evidence_ids": focus, "rationale": rationale}


# ---------------------------------------------------------------------------- HI：B 族


def _issued(memory: dict[str, Any]) -> set[str]:
    return set(memory.setdefault("issued", []))


def _mark(memory: dict[str, Any], key: str) -> None:
    memory.setdefault("issued", []).append(key)


def _best_match(align: dict[str, Any] | None) -> dict[str, Any] | None:
    if not align or align.get("status") != "ready":
        return None
    matches = (align.get("result") or {}).get("matches") or []
    return matches[0] if matches else None


def _events_from_diff(diff: dict[str, Any], keep: int = 2) -> list[dict[str, Any]]:
    """diff_media 段 → 要核边界的事件。按差异帧数取最多 keep 段，再按时间排序。"""

    result = diff.get("result") or {}
    segs = [s for s in result.get("segments") or [] if isinstance(s, dict)]
    segs = sorted(segs, key=lambda s: (-(s.get("differing_frames") or 0), s.get("start_s") or 0))[:keep]
    fps_a = _num(((result.get("frame_rate") or {}).get("a")), 30.0) or 30.0
    fps_b = _num(((result.get("frame_rate") or {}).get("b")), fps_a) or fps_a
    images = result.get("images") or []
    events = []
    for seg in sorted(segs, key=lambda s: s.get("start_s") or 0):
        index = (result.get("segments") or []).index(seg)
        events.append({
            "kind": "diff", "index": index,
            "a": [_num(seg.get("first_diff_s"), _num(seg.get("start_s"))), _num(seg.get("last_diff_s")), _num(seg.get("end_s"))],
            "b": [_num(seg.get("first_diff_b_s"), _num(seg.get("start_b_s"))), _num(seg.get("last_diff_b_s")), _num(seg.get("end_b_s"))],
            "dt_a": 1.0 / fps_a, "dt_b": 1.0 / fps_b, "bbox": seg.get("union_bbox"),
            "peak_area": seg.get("peak_area"), "end_reason": seg.get("end_reason"),
            "images": [img.get("evidence_id") for img in images if img.get("segment_index") == index and img.get("evidence_id")],
        })
    return events


def _events_from_align(align: dict[str, Any], keep: int = 2) -> list[dict[str, Any]]:
    result = align.get("result") or {}
    matches = [m for m in result.get("matches") or [] if isinstance(m, dict)][:keep]
    events = []
    for index, match in enumerate(matches):
        ra, rb = match.get("range_a_s") or [None, None], match.get("range_b_s") or [None, None]
        anchors = match.get("anchors") or []
        images = [x for a in anchors for x in (a.get("image_a_evidence_id"), a.get("image_b_evidence_id")) if x]
        events.append({"kind": "align", "index": index, "a": [_num(ra[0]), None, _num(ra[1])], "b": [_num(rb[0]), None, _num(rb[1])],
                       "dt_a": 1.0, "dt_b": 1.0, "bbox": None, "images": images,
                       "mapping": match.get("time_mapping") or {}})
    return events


def _boundary_times(start: float | None, last: float | None, end: float | None, dt: float, duration: float | None,
                    coarse: bool) -> list[float]:
    """一段的边界两侧取帧：[起点前一帧, 起点, 最后差异帧, 终点（右开，第一个无差异帧）]。
    align 事件（coarse）没有逐帧边界：起点 ±1 秒、终点 ±1 秒。"""

    times: list[float] = []
    if coarse:
        cand = [(start or 0) - 1.0, start, (end or 0) - 1.0, end]
    else:
        cand = [None if start is None else start - dt, start, last, end]
    limit = (duration - min(dt, 0.05)) if duration else None
    for value in cand:
        if value is None:
            continue
        value = max(0.0, value)
        if limit is not None and value > limit:
            value = limit
        value = _r(value, 3)
        if value not in times:
            times.append(value)
    return sorted(times)


def _frames_of(obs: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not obs or obs.get("status") != "ready":
        return []
    return [(f["evidence_id"], float(f.get("time_s") or 0.0)) for f in (obs.get("result") or {}).get("frames") or []
            if isinstance(f, dict) and f.get("evidence_id")]


def _nearest(frames: list[tuple[str, float]], t: float | None) -> str | None:
    if not frames or t is None:
        return None
    return min(frames, key=lambda f: abs(f[1] - t))[0]


def _status_of(state: HiState, memory: dict[str, Any], *, chain: dict[str, bool], missing: list[str]) -> tuple[str, str | None]:
    if all(chain.values()) and not missing:
        return "COMPLETED", None
    pieces = list(missing)
    if not all(chain.values()):
        pieces.append("steps not completed: " + ", ".join(k for k, v in chain.items() if not v))
    return "INCOMPLETE", "; ".join(pieces)


def _report(state: HiState, memory: dict[str, Any], rule: str, *, status: str, reason: str | None,
            notes: list[str], branch: str) -> Decision:
    allowed = sorted({e for e in state.observed | set(state.attached) if e in state.cards}, key=lambda e: int(e[1:]))
    skeleton: dict[str, Any] = {"task_id": state.task_id, "status": status}
    if reason:
        skeleton["status_reason"] = reason
    family = task_family(state.task, "SEARCH" in state.actions)
    if not state.task_id.startswith("W1-") and memory.get("run_family") in ("E", "L", "C") and "SEARCH" in state.actions:
        family = str(memory["run_family"])          # r1：第 2 轮沿用第 1 轮的族（见 decide_hi）
    # 10-15 对话 5''（追加 3 / 4）：可引 E 的来源与模态（S4 程序校验用：limitations 不得说已有的材料"不可用"、结论要引建立它的
    # 工具记录）；检索候选页面 failed 的计数（≥ 1/3 时在 limitations 末尾自动加一句，不占教师槽）
    kinds = {e: {"acquired_by": (state.cards.get(e) or {}).get("acquired_by"),
                 "image": "image" in ((state.cards.get(e) or {}).get("modalities") or [])} for e in allowed}
    return Decision("report", rule, skeleton, branch=branch,
                    slot={"name": "S4_report", "status": status, "status_reason": reason, "allowed_evidence_ids": allowed,
                          "notes": notes, "task_parts": task_parts(state.task), "family": family, "evidence_kinds": kinds,
                          "auto_limitations": _failed_page_limitation(state),
                          "followup": _followup_hints(state) if family in ("E", "L", "C") else {}})


def _claim_of(task: str) -> str:
    m = re.search(r'claim:\s*"(.*)"\s*\(claim date', task, re.S)
    return m.group(1) if m else task.rsplit(":", 1)[-1].strip()


def _followup_hints(state: HiState) -> dict[str, Any]:
    """10-16 r0+：交报告时记下第 2 轮还能做什么，供循环内充分性检查判"报告没答完"时写可修缺口（inloop_status_reason）。
    unread：已取得、没读、不是社交平台的页面——以图搜的结果在前（它们载有这张图），其后按标题 / 摘要命中声明的词数；
    refs：L 族以文字搜到的参考图；image：给定图片。只用学生看得到的检索结果。"""

    if "SEARCH" not in state.actions:
        return {}
    searches = _search_obs(state)
    read_ids = {str((r.get("request") or {}).get("record_id")) for r in _read_records(state)}
    reverse = {p["evidence_id"] for p in _search_pages([s for s in searches if (s.get("request") or {}).get("input_type") == "image"])}
    words = _claim_words(_claim_of(state.task))
    pages = [p for p in _search_pages(searches) if p["evidence_id"] not in read_ids and p.get("url") and not _social(p.get("url"))]
    pages.sort(key=lambda p: (p["evidence_id"] not in reverse, -len(words & _claim_words(_page_text(p)))))
    images = _material_images(state)
    return {"unread": [{"evidence_id": p["evidence_id"], "url": p["url"], "reverse": p["evidence_id"] in reverse}
                       for p in pages[:6]],
            "refs": [r["evidence_id"] for r in ranked_refs(searches)][:3], "image": images[0] if images else None}


_LOCATION_POINT = re.compile(r"compar|reference|same (?:place|building|location|site|landmark)|landmark|architect|"
                             r"where (?:the )?(?:picture|photo|photograph|image) was (?:actually )?taken|actual location|"
                             r"shows the (?:claimed|named)|place the claim names", re.I)


def _strip_tracking(url: str) -> str:
    """去掉以图搜结果网址上的 utm_* 跟踪参数（?utm_medium=organic&utm_source=yandexsmartcamera），其余参数保留。"""

    base, _, query = url.partition("?")
    keep = [q for q in query.split("&") if q and not q.lower().startswith("utm_")]
    return base + ("?" + "&".join(keep) if keep else "")


def inloop_actions(items: list[str], hints: dict[str, Any], family: str) -> list[str]:
    """r0+：报告没答完时，walker 按自己的状态写出第 2 轮能做的事（GIA 的 _search_gap 认这几种说法）：
    未读页面的网址 → 打开再读；L 族参考图 → 逐项并排比对；都没有 → 再搜一次来源。B / A 族不写。"""

    if family not in ("E", "L", "C"):
        return []
    out = []
    urls = [_strip_tracking(str(p["url"])) for p in hints.get("unread") or [] if p.get("url")][:3]
    if urls:
        out.append("candidate pages that were found but not read: " + ", ".join(urls))
    if family == "L" and hints.get("refs") and hints.get("image") and any(_LOCATION_POINT.search(i) for i in items):
        out.append(f"the reference photos {', '.join(hints['refs'])} of the claimed place were not compared feature by "
                   f"feature with {hints['image']}")
    if not out:
        out.append("a further search for sources can address the points above")
    return out


def has_followup_clause(reason: str) -> bool:
    return any(_search_gap(c) for c in _gap_clauses(reason or ""))


def inloop_status_reason(items: list[str], hints: dict[str, Any], family: str) -> str:
    """循环内充分性检查把 COMPLETED 改成 INCOMPLETE 时的 status_reason：充分性判定的 missing（build_reason 过滤后的英文条目）
    + walker 写的可修缺口（inloop_actions）。"""

    items = [i for i in items if i]
    return "; ".join(items + inloop_actions(items, hints, family))


_PAGE_STATUS = re.compile(r"^\[(R\d+)\] (\S+?);? page=(ready|partial|failed)\b")


def _failed_page_limitation(state: HiState) -> list[str]:
    """10-15 对话 5''（追加 4）：检索候选里 page=failed 的占 ≥ 1/3（且至少 3 个候选有页面状态）→ 报告 limitations 末尾自动
    加一句模板（不占教师槽）。r0a e12：9 个反搜命中 5 个 failed，报告仍称 DW 是 "the earliest publication located"。"""

    if "SEARCH" not in state.actions:
        return []
    status: dict[str, str] = {}
    for search in _search_obs(state):
        lines = [str(x) for x in search.get("items") or []]
        for line in lines:
            m = _PAGE_STATUS.match(line)
            if m:
                status.setdefault(m.group(2).rstrip(";"), m.group(3))
        if not lines:
            for result in (search.get("details") or {}).get("results") or []:
                for material in result.get("materials") or []:
                    if material.get("kind") == "page" and material.get("status") in ("ready", "partial", "failed"):
                        status.setdefault(str(result.get("source_url")), str(material["status"]))
    total = len(status)
    failed = sum(1 for v in status.values() if v == "failed")
    if total >= 3 and failed * 3 >= total:
        return [f"{failed} of {total} candidate pages failed to load and were not examined; an earlier or other copy may "
                "exist among them."]
    return []


_PART_SPLIT = re.compile(r"(?<=[.;])\s+|,\s+and\s+(?=(?:state|give|check|cite|investigate|find|determine|describe|say|explain)\b)", re.I)


def task_parts(task: str) -> list[str]:
    """把任务文字拆成要逐条回应的子要求（只用学生看得到的任务文字）。纯限定语（"do not …"、"Being given … does not mean …"）
    不单列成一条，留给 S4 当作写作约束。"""

    parts: list[str] = []
    for chunk in _PART_SPLIT.split(task.strip()):
        chunk = chunk.strip().rstrip(".;").strip()
        if not chunk:
            continue
        lowered = chunk.lower()
        if lowered.startswith(("do not", "being given", "don't", "the input contains only",
                                "the given image circulated", "the given video circulated")):
            continue          # 前提句（"给定图片随以下声明流传…"）不是要回答的一项；声明原文由下一项引用
        for sub in re.split(r"\s+and\s+(?=(?:how|what|where|whether|which|when)\b)", chunk):
            sub = sub.strip()
            if len(sub) > 12:
                parts.append(sub)
    return parts[:6]


def decide_hi(payload: dict[str, Any], memory: dict[str, Any], rendered: dict[str, Any] | None = None) -> Decision:
    state = HiState(payload, memory, rendered)
    issued = _issued(memory)
    videos = state.videos()

    # 通用：上一次输出被拒（修复回合）——walker 的输出按 schema 生成，不该出现；出现就退到最保守的一步：交报告
    if state.invalid and "repair" not in issued:
        _mark(memory, "repair")
        return _report(state, memory, "G0_invalid_output_fallback", status="INCOMPLETE",
                       reason="the previous action was rejected by the interface; reporting what has been observed",
                       notes=[f"validation feedback: {str(state.invalid)[:300]}"], branch="report_incomplete")

    family = task_family(state.task, "SEARCH" in state.actions)
    # r1（10-17）：第 2 轮的任务文字是 S2 写的，族词可能不在里面（r0h e133 W2-1 "…as claimed. Scope: open and read <网址>"
    # 判成 B 族，走到 G9 直接交 INCOMPLETE）。检索类案例的第 2 轮沿用第 1 轮的族（oracle_generator 记在 memory["run_family"]）
    if not state.task_id.startswith("W1-") and "SEARCH" in state.actions and memory.get("run_family") in ("E", "L", "C"):
        family = str(memory["run_family"])
    # 第 2 轮及以后的工作区：任务里写明了要补的时间窗 / 音轨 → 跟进模式（不重走整条比对链）
    if not state.task_id.startswith("W1-") and family == "B":
        windows, audio, anchors = followup_scope(state.task)
        if windows or audio or anchors:
            return _decide_followup(state, memory, windows, audio, anchors)
    # 10-15 对话 5''（追加 2）：检索类第 2 轮——任务的 Scope 写明了要打开再读的网址 / 要并排看的参考图，只补这些
    if not state.task_id.startswith("W1-") and family in ("E", "L", "C") and _SEARCH_SCOPE.search(state.task):
        return _before_report(state, memory, _decide_search_followup(state, memory, family))
    if family == "L" and _material_images(state):
        return _before_report(state, memory, _decide_place_check(state, memory))
    if family == "E":
        return _before_report(state, memory, _decide_image_origin(state, memory))
    if family == "A" and len(videos) == 1:
        return _before_report(state, memory, _decide_find_original(state, memory, videos[0]))
    if family == "C":
        return _before_report(state, memory, _decide_claim(state, memory))
    if len(videos) >= 2:
        if "compare_media" in state.inspect_ops:
            return _before_report(state, memory, _decide_pair_merged(state, memory, videos[:2]))
        return _decide_pair(state, memory, videos[:2])
    if len(videos) == 1 and ({"locate_frames", "locate"} & state.inspect_ops) and _LOCATE_DESC.search(state.task):
        return _decide_locate(state, memory, videos[0])
    return _report(state, memory, "G9_unsupported_family", status="INCOMPLETE",
                   reason="this walker version only covers given-media comparison and description localization",
                   notes=["walker D1 does not implement search / claim-checking families"], branch="report_incomplete")


_OBSERVED_LIST = re.compile(r'"content_observed_evidence_ids"\s*:\s*\[[^\]]*\]')
_VISIBLE_EID = re.compile(r"\bE[1-9][0-9]*\b")
_QM_SKIP = ("INSPECT/media_profile",)


def _hidden_evidence(state: HiState, ids: list[str]) -> list[str]:
    """要引用的 E 里，学生现在看不到卡片、也不在任何可见观察里的（只在"已看过的 E 编号"清单里剩个编号）。
    卡片清单有上限（max_evidence_cards）、早先的观察会被压缩，长一点的工作区里早先的证据会这样掉出视图。"""

    text = _OBSERVED_LIST.sub("", state.visible_text)
    visible = set(_VISIBLE_EID.findall(text))
    return [e for e in ids if e not in visible and (state.cards.get(e) or {}).get("acquired_by") not in _QM_SKIP]


def _before_report(state: HiState, memory: dict[str, Any], decision: Decision) -> Decision:
    """交报告前：报告要用的证据有的已掉出视图（只剩编号），先 QUERY_MEMORY 按它的标签查回来再写（每个工作区一次）。
    只在确实看不到时才查——卡片都在视图里就直接交（V9 变体 max_evidence_cards=6 时更常出现）。"""

    if decision.kind != "report" or decision.rule in ("G0_invalid_output_fallback", "G1_last_turn_report"):
        return decision
    issued = _issued(memory)
    if "QUERY_MEMORY" not in state.actions or "memory_before_report" in issued or state.remaining < 2:
        return decision
    if just_reviewed(state, memory):
        return decision          # r1：上一步刚把帧 / 对照图放回附图（E7 / C9v），这一轮必须交报告，不插 QUERY_MEMORY
    hidden = _hidden_evidence(state, list(decision.slot.get("allowed_evidence_ids") or []))
    # 只为文字结果（比对报告、转写、OCR、页面）查：掉出视图的帧 / 图片在比对结果与焦点里另有出处，查回来也说不清它是什么
    hidden = [e for e in hidden if "image" not in ((state.cards.get(e) or {}).get("modalities") or [])]
    if not hidden:
        return decision
    pick = hidden[0]
    label = " ".join(state.label(pick).split())[:120] or pick
    _mark(memory, "memory_before_report")
    memory["memory_hidden"] = hidden
    return Decision("action", "Q1_query_memory_before_report", {"action": "QUERY_MEMORY",
                    "arguments": {"query": label, "record_types": ["evidence"]}, "focus_evidence_ids": [],
                    "rationale": f"{', '.join(hidden[:4])} {'are' if len(hidden) > 1 else 'is'} no longer shown in the current view; "
                                 f"look {'them' if len(hidden) > 1 else 'it'} up before citing in the report."},
                    branch="QUERY_MEMORY (HI)")


# ---- 两段视频：比对（videosham）/ 对应（vcsl）


def _decide_pair(state: HiState, memory: dict[str, Any], pair: list[str]) -> Decision:
    issued = _issued(memory)
    a, b = pair
    pa, pb = state.profile(a), state.profile(b)
    copy_task = bool(_COPY_TASK.search(state.task))
    align = state.last("align_media", mode="visual")
    diff = state.last("diff_media")
    match = _best_match(align)
    notes: list[str] = []

    def chain() -> dict[str, bool]:
        out = {"align_media": align is not None and align.get("status") == "ready"}
        if not (copy_task and _weak_align(align, pa, pb)):
            out["diff_media"] = diff is not None and diff.get("status") == "ready"
        return out | {
                "frames_compared": any(k.startswith("frames_A_") or k in ("uniform_B", "review_pairs") for k in _issued(memory))}

    # 通用 0：最后一轮只交报告
    if state.remaining <= 0:
        miss = _missing_parts(state, memory, pa, pb, memory.get("uncovered") or [], copy_task=copy_task,
                              pairs=_locate_pairs(state, a) if copy_task and _weak_align(align, pa, pb) else None)
        status, reason = _status_of(state, memory, chain=chain(), missing=miss)
        if status == "INCOMPLETE":
            reason = "turn budget exhausted: " + (reason or "the comparison did not finish")
        return _report(state, memory, "G1_last_turn_report", status=status, reason=reason,
                       notes=_pair_notes(state, align, diff, pa, pb, memory), branch="report_last_turn")

    # 1 对齐
    if "align_visual" not in issued:
        _mark(memory, "align_visual")
        return Decision("action", "B1_align", _inspect([a, b], "align_media", {"mode": "visual", "top_k": 3}, [],
                        f"Establish the temporal correspondence between {a} and {b} before measuring where they differ."),
                        branch="INSPECT align_media")

    # 2 逐帧差分（带 align 的 mapping）
    if "diff" not in issued and "diff_media" in state.inspect_ops and not (copy_task and _weak_align(align, pa, pb)):
        _mark(memory, "diff")
        params: dict[str, Any] = {}
        mapping = (match or {}).get("time_mapping") or {}
        scale, offset = _num(mapping.get("scale"), 1.0), _num(mapping.get("offset_s"), 0.0)
        branch = "INSPECT diff_media"
        if match and (abs((scale or 1.0) - 1.0) > 1e-3 or abs(offset or 0.0) > 1e-3):
            # 渲染视图里数值保留 3 位小数；抄学生看得到的那个数
            params["mapping"] = {"scale": _r(scale or 1.0, 3), "offset_s": _r(offset or 0.0, 3)}
            branch = "INSPECT diff_media(mapping)"
        if copy_task and match and match.get("range_a_s"):
            params["range_a_s"] = [_r(float(x), 3) for x in match["range_a_s"]]
        # focus 留空：差分会产出每段 3 张对照图（首 / 峰值 / 末），容量全部留给它们
        focus: list[str] = []
        why = (f"Align result {align.get('result_evidence_id') if align else ''} gives a time mapping; compute frame-by-frame where "
               f"{a} and {b} differ under it." if match else f"No alignment candidate; compare {a} and {b} frame by frame on the identity mapping.")
        return Decision("action", "B2_diff", _inspect([a, b], "diff_media", params, focus, why.replace("  ", " ")), branch=branch)

    weak_align = _weak_align(align, pa, pb)
    if copy_task and weak_align:
        events = []
    elif copy_task:
        events = _events_from_align(align)
    elif diff and diff.get("status") == "ready":
        events = _events_from_diff(diff)
    else:
        events = _events_from_align(align) if align and not weak_align else []

    # 3 差分图没附上 → READ_RECORD 带 focus 回看（H6）
    if diff and diff.get("status") == "ready" and "review_diff" not in issued:
        diff_images = [img.get("evidence_id") for img in (diff.get("result") or {}).get("images") or [] if img.get("evidence_id")]
        unseen = [e for e in diff_images if not state.seen(e)]
        if unseen and diff.get("result_evidence_id"):
            _mark(memory, "review_diff")
            focus = _existing(state, unseen, state.limit)
            return Decision("action", "B3_review_diff_images", _read(diff["result_evidence_id"], focus,
                            f"Difference images {', '.join(focus)} were not attached; view them before checking boundaries."),
                            branch="READ_RECORD focus")

    # 3' 对应类任务、对齐结果弱（无候选或覆盖太短）：A 侧均匀取帧，逐帧以图在 B 里找候选，最后成对回看
    if copy_task and weak_align:
        if "uniform_A" not in issued and state.remaining >= 2:
            _mark(memory, "uniform_A")
            return Decision("action", "B8_uniform_frames_A", _inspect([a], "get_frames", {"count": state.limit // 2}, [],
                            f"Alignment gave no candidate covering enough of the videos; sample {a} evenly and look each frame up in {b}."),
                            branch="INSPECT get_frames count")
        a_obs = next((o for o in reversed(state.observations) if o["operation"] == "get_frames" and o["targets"] == [a]
                      and "count" in o["params"]), None)
        a_frames = _frames_of(a_obs)
        if "locate_frames" in state.inspect_ops:
            for eid, t in a_frames:
                if f"locate_{eid}" in issued or state.remaining < 2:
                    continue
                _mark(memory, f"locate_{eid}")
                return Decision("action", "B8_locate_in_B", _inspect([b], "locate_frames",
                                {"query": {"type": "image", "evidence_id": eid}, "top_k": 3}, _existing(state, [eid], 1),
                                f"Find where the content of {eid} ({a} at {t:.1f} s) appears in {b}; candidates still need a visual check."),
                                branch="INSPECT locate_frames(image)")
            pairs: list[str] = []
            last_locate = None
            for obs in state.observations:
                if obs["operation"] != "locate_frames" or obs.get("status") != "ready":
                    continue
                query = ((obs["params"].get("query") or {}).get("evidence_id"))
                cands = sorted((obs.get("result") or {}).get("candidates") or [], key=lambda c: -float(c.get("score") or 0))
                if query and cands:
                    pairs += [query, cands[0].get("evidence_id")]
                last_locate = obs
            if pairs and "review_pairs" not in issued and state.remaining >= 1:
                _mark(memory, "review_pairs")
                record = (last_locate or {}).get("result_evidence_id") or pairs[0]
                focus = _existing(state, pairs, state.limit)
                return Decision("action", "B8_review_pairs", _read(record, focus,
                                "View each A frame next to its best-ranked B candidate together to judge which parts correspond."),
                                branch="READ_RECORD focus")
        elif "uniform_B" not in issued and state.remaining >= 1:
            _mark(memory, "uniform_B")
            focus = _existing(state, [e for e, _ in a_frames], state.limit // 2)
            return Decision("action", "B8_uniform_frames_B", _inspect([b], "get_frames", {"count": state.limit // 2}, focus,
                            f"Sample {b} evenly and keep {', '.join(focus)} in view to judge which parts correspond."),
                            branch="INSPECT get_frames count")

    # 以下按优先级：首段边界 → 未覆盖区间 → 语音转写 → 其余段边界 → 放大 → 音轨对齐 → 报告。
    # 抄进参数的数只取学生此刻看得见的（渲染视图保留 3 位小数）；另一侧的时刻用可见的 mapping 换算。
    mapping = _visible_mapping(state)
    plan = memory.setdefault("plan", {})

    # 4 前两段边界两侧（diff 刚出来、段的起止在当前视图里；被截断就先分页重读）。第二段紧跟第一段做：
    #   隔几步再做时，重读出来的数已被挤出视图，只能放弃这一段（wk3 4126 的第二段就是这样丢的）
    for ev in events[:2]:
        step = _event_step(state, memory, ev, a, b, pa, pb, mapping, source_for(ev, diff, align), plan)
        if step is not None:
            return step

    # 5 没被比对覆盖到的区间（两侧时长不同、或只比了对应段）：在较长那段的空白处均匀取帧
    uncovered = [] if (copy_task and weak_align) else _uncovered(pa, pb, mapping, diff, copy_task)
    memory["uncovered"] = uncovered
    for side, lo, hi in uncovered:
        key = f"uncovered_{side}_{lo:.3f}"
        if key in issued or state.remaining < 2:
            continue
        _mark(memory, key)
        target = a if side == "A" else b
        count = 4 if hi - lo >= 4 else 2
        dur = (pa if side == "A" else pb).get("duration")
        top = _r(min(hi, dur - 0.05), 3) if dur else hi          # 区间写到时长为止；取帧的右端留半帧余量
        return Decision("action", "B9_uncovered_range", _inspect([target], "get_frames", {"range_s": [lo, top], "count": count}, [],
                        f"{side} {lo:.2f}-{hi:.2f} s has no counterpart in the comparison so far; sample it so it is not treated as unchanged."),
                        branch="INSPECT get_frames range")

    # 6 语音：任务没说只看画面时，有音轨的一侧各转写一次（只有对齐不够回答"声音是否不同"）
    if not _NO_AUDIO_TASK.search(state.task) and "asr" in state.inspect_ops:
        for side, video, prof in (("A", a, pa), ("B", b, pb)):
            if (prof.get("audio") or 0) > 0 and f"asr_{side}" not in issued and state.remaining >= 2:
                _mark(memory, f"asr_{side}")
                return Decision("action", "B10_asr", _inspect([video], "asr", {}, _key_images(state, memory, events, a, b),
                                f"{video} has an audio track; transcribe it to check whether the spoken content differs."),
                                branch="INSPECT asr")

    # 8 差异框小 → 放大（A、B 各一张，同一框）
    small = next((ev for ev in events if ev["kind"] == "diff" and _bbox_area(ev.get("bbox")) is not None
                  and _bbox_area(ev["bbox"]) < 0.15 and f"frames_A_diff{ev['index']}" in _issued(memory)), None)
    if small and "crop" in state.inspect_ops and state.remaining >= 2:
        key = f"diff{small['index']}"
        box = _pad_box(small["bbox"])
        b_frames = _frames_of(_frames_obs(state, b, plan.get(f"B_{key}")))
        a_frames = _frames_of(_frames_obs(state, a, plan.get(f"A_{key}")))
        t_b = small["b"][1] if small["b"][1] is not None else small["b"][0]
        target_b = _nearest(b_frames, t_b)
        target_a = _nearest(a_frames, _inv(mapping, t_b) if t_b is not None else None)
        if target_b and "crop_B" not in _issued(memory):
            _mark(memory, "crop_B")
            focus = _existing(state, [target_a] if target_a else [], 1) + _existing(state, small["images"], 2)
            return Decision("action", "B5_crop_B", _inspect([target_b], "crop", {"box": box}, focus,
                            f"The difference box covers under 15% of the picture; enlarge it in {target_b} to see what differs."),
                            branch="INSPECT crop")
        if target_a and "crop_B" in _issued(memory) and "crop_A" not in _issued(memory):
            _mark(memory, "crop_A")
            crop_b = state.last("crop")
            focus = _existing(state, (crop_b or {}).get("created", []) + [target_b] + small["images"], state.limit // 2)
            return Decision("action", "B5_crop_A", _inspect([target_a], "crop", {"box": box}, focus,
                            f"Enlarge the same region in {target_a} to compare with the B-side crop."),
                            branch="INSPECT crop")

    # 9 音轨对齐（两侧都有音轨、还有预算）
    if ("audio" not in _issued(memory) and not _NO_AUDIO_TASK.search(state.task)
            and (pa.get("audio") or 0) > 0 and (pb.get("audio") or 0) > 0 and state.remaining >= 1):
        _mark(memory, "audio")
        focus = _key_images(state, memory, events, a, b)
        return Decision("action", "B6_align_audio", _inspect([a, b], "align_media", {"mode": "audio", "top_k": 3}, focus,
                        f"Both {a} and {b} have an audio track; check whether the sound corresponds in time before reporting."),
                        branch="INSPECT align_media(audio)")

    # 10 走完 → 报告。没看的区间、该转写没转写的音轨，都让 status 落到 INCOMPLETE
    missing = _missing_parts(state, memory, pa, pb, uncovered, copy_task=copy_task,
                             pairs=_locate_pairs(state, a) if copy_task and weak_align else None)
    status, reason = _status_of(state, memory, chain=chain(), missing=missing)
    return _report(state, memory, "B7_report", status=status, reason=reason,
                   notes=_pair_notes(state, align, diff, pa, pb, memory),
                   branch="report_completed" if status == "COMPLETED" else "report_incomplete")


# ---- 两段视频（合并目录，hi@0.8.0 起）：compare_media 一次给出对应段、删减、插入、变速、区域差异、声音

def _compare_obs(state: HiState) -> dict[str, Any] | None:
    """整段比对（参数里没有 range_a_s）的最近一次结果。"""

    for obs in reversed(state.observations):
        if obs["operation"] == "compare_media" and not (obs["params"] or {}).get("range_a_s"):
            return obs
    return None


def _cmp_mapped(res: dict[str, Any], t: float) -> float | None:
    """A 时刻 → B 时刻：用含该时刻的对应段的映射（可见的 scale / offset_s）。"""

    for c in res.get("correspondences") or []:
        ra = c.get("a_range_s") or [None, None]
        m = c.get("mapping") or {}
        if None in ra or _num(m.get("scale")) is None or _num(m.get("offset_s")) is None:
            continue
        if float(ra[0]) - 0.25 <= t <= float(ra[1]) + 0.25:
            return float(m["scale"]) * t + float(m["offset_s"])
    return None


def _events_from_compare(res: dict[str, Any], pa: dict[str, Any], pb: dict[str, Any], copy_task: bool) -> list[dict[str, Any]]:
    """compare_media 结果 → 要两侧取帧核看的事件（最多 2 个），按对任务的重要性排。每个事件给出 A、B 两侧的取帧时刻。"""

    da, db = pa.get("duration"), pb.get("duration")
    dt_a = 1.0 / pa["fps"] if pa.get("fps") else 0.04
    images = [i.get("evidence_id") for i in res.get("images") or [] if isinstance(i, dict) and i.get("evidence_id")]
    img_kind = {i.get("evidence_id"): (i.get("kind"), i.get("index")) for i in res.get("images") or [] if isinstance(i, dict)}

    def imgs(kind: str, index: int) -> list[str]:
        return [e for e in images if img_kind.get(e) == (kind, index)]

    events: list[dict[str, Any]] = []
    corr = [c for c in res.get("correspondences") or [] if isinstance(c, dict) and c.get("a_range_s") and c.get("b_range_s")]
    if copy_task:
        for i, c in enumerate(sorted(corr, key=lambda c: -(float(c["a_range_s"][1]) - float(c["a_range_s"][0])))[:2]):
            a0, a1 = (float(x) for x in c["a_range_s"])
            b0, b1 = (float(x) for x in c["b_range_s"])
            events.append({"key": f"corr{i}", "what": f"the corresponding span A {a0:.1f}-{a1:.1f} s ↔ B {b0:.1f}-{b1:.1f} s",
                           # 偏移取两位小数（±0.45 / 0.25）：整一位小数的派生时刻常与原始结果里的别的数巧合，被导出当成隐藏信息
                           "a_times": [a0 - 0.45, a0 + 0.25, a1 - 0.25, a1 + 0.45], "b_times": [b0 - 0.45, b0 + 0.25, b1 - 0.25, b1 + 0.45],
                           "images": [], "box": None, "basis": [a0, a1, b0, b1]})
        return events
    fd = res.get("frame_diff") if isinstance(res.get("frame_diff"), dict) else None
    if fd:
        for ev in _events_from_diff(fd):
            a0, alast, a1 = ev["a"]
            b0, blast, b1 = ev["b"]
            if a0 is None or b0 is None:
                continue
            events.append({"key": f"fd{ev['index']}", "what": f"frame-exact difference segment {ev['index'] + 1}",
                           # 段尾用 end - 1 帧（最后差异帧的时刻只在原始结果里，学生视图的摘要行只有起止）
                           "a_times": [a0 - ev["dt_a"], a0] + ([a1 - ev["dt_a"], a1] if a1 is not None else []),
                           "b_times": [b0 - ev["dt_b"], b0] + ([b1 - ev["dt_b"], b1] if b1 is not None else []),
                           "images": ev["images"], "box": None, "basis": [a0, a1, b0, b1]})
    for i, cut in enumerate(res.get("b_cuts") or []):
        at, miss = _num(cut.get("a_at_s")), cut.get("b_missing_s") or [None, None]
        if at is None or None in miss:
            continue
        b0, b1 = float(miss[0]), float(miss[1])
        events.append({"key": f"cut{i}", "what": f"the B span {b0:.1f}-{b1:.1f} s that A skips at {at:.1f} s",
                       "a_times": [at - 0.2, at + 0.2], "b_times": [b0 - 0.2, b0 + 0.3, b1 - 0.3, b1 + 0.2], "images": [], "box": None,
                       "basis": [at, b0, b1]})
    for i, seg in enumerate(res.get("segments_not_in_b") or []):
        ra = seg.get("a_range_s") or [None, None]
        if None in ra:
            continue
        a0, a1 = float(ra[0]), float(ra[1])
        bt = [float(x) for x in (seg.get("b_before_s"), seg.get("b_after_s")) if _num(x) is not None]
        events.append({"key": f"ins{i}", "what": f"A {a0:.1f}-{a1:.1f} s, which has no counterpart in B",
                       "a_times": [a0 - 0.3, a0 + 0.3, a1 - 0.3, a1 + 0.3], "b_times": bt, "images": imgs("segment", i), "box": None,
                       "basis": [a0, a1] + bt})
    for i, c in enumerate(corr):
        kind = str(c.get("kind") or "normal")
        if kind == "normal" and not c.get("picture_differs") and not c.get("mirrored"):
            continue
        a0, a1 = (float(x) for x in c["a_range_s"])
        m = c.get("mapping") or {}
        scale, off = _num(m.get("scale"), 1.0), _num(m.get("offset_s"), 0.0)
        mid = (a0 + a1) / 2
        if kind in ("slowed", "frozen"):
            a_times = [mid + k * dt_a for k in range(4)]          # 相邻 4 帧：慢放 / 定格时它们几乎相同
            b_times = [scale * mid + off, scale * (mid + 1.0) + off]
        else:
            a_times = [a0 + 0.3, mid, a1 - 0.3]
            b_times = [scale * t + off for t in a_times]
        b_rng = [float(x) for x in c["b_range_s"]]
        if kind in ("slowed", "frozen"):
            b_times = [b_rng[0] + (b_rng[1] - b_rng[0]) * f for f in (0.4, 0.6)]   # B 侧取可见区间内的时刻，不用映射式里的数
        else:
            b_times = [b_rng[0] + 0.3, (b_rng[0] + b_rng[1]) / 2, b_rng[1] - 0.3] if kind != "reversed" else [b_rng[1] - 0.3, (b_rng[0] + b_rng[1]) / 2, b_rng[0] + 0.3]
        events.append({"key": f"corr{i}", "what": f"A {a0:.1f}-{a1:.1f} s, reported as {kind}{' (mirrored)' if c.get('mirrored') else ''}",
                       "a_times": a_times, "b_times": b_times, "images": [], "box": None, "basis": [a0, a1] + b_rng})
    same_length = bool(da and db and abs(da - db) < 0.1)
    for i, reg in enumerate(res.get("changed_regions") or []):
        ra = reg.get("a_range_s") or [None, None]
        box = reg.get("box")
        if None in ra or not box or len(box) != 4:
            continue
        t = _r((float(ra[0]) + float(ra[1])) / 2, 2)         # 区间中点（摘要行里写的是区间与框，峰值时刻不在学生视图里）
        box2 = [_r(float(box[0]), 2), _r(float(box[1]), 2), _r(float(box[2]), 2), _r(float(box[3]), 2)]
        # 10-15 对话 5''（追加 2(c)）：区间两端外侧各取一对 A / B 帧（以前只取中点——r0a videosham_4126 的 A 3.3–4.3 s 从没在
        # 区间外两侧取过配对帧，边界因此定不下来）
        outside = [_r(float(ra[0]) - 0.3, 2), t, _r(float(ra[1]) + 0.3, 2)]
        events.append({"key": f"reg{i}", "what": f"the region that differs around A {t:.2f} s and just outside its ends",
                       "a_times": outside, "b_times": outside if same_length else [], "images": imgs("region", i),
                       "box": _pad_box(box2, 0.025), "basis": [float(ra[0]), float(ra[1])] + box2})
    return events[:2]


_MATCHED_IN_B = re.compile(r"matched footage lies in B (\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?) s")


def _visible_events(state: HiState, events: list[dict[str, Any]], res: dict[str, Any], pa: dict[str, Any],
                    pb: dict[str, Any], copy_task: bool) -> list[dict[str, Any]]:
    """只留下基准数（区间起止、框）在学生视图里看得见的事件；对应类任务一个都看不见时，用摘要行
    "the matched footage lies in B x-y s" 配 A 全长做一个粗事件。"""

    keep = [ev for ev in events if _visible(state, [v for v in ev.get("basis") or [] if v is not None])]
    if keep or not copy_task:
        return keep
    m = next((m for line in res.get("summary_lines") or [] for m in [_MATCHED_IN_B.search(str(line))] if m), None)
    da = pa.get("duration")
    if not m or not da or not _visible(state, [float(m.group(1)), float(m.group(2))]):
        return []
    b0, b1 = float(m.group(1)), float(m.group(2))
    return [{"key": "matchedB", "what": f"the matched footage (B {b0:.0f}-{b1:.0f} s)", "a_times": [0.3, da - 0.5],
             "b_times": [b0 - 0.5, b0 + 0.3, b1 - 0.3, b1 + 0.5], "images": [], "box": None, "basis": [b0, b1]}]


def _decide_pair_merged(state: HiState, memory: dict[str, Any], pair: list[str]) -> Decision:
    issued = _issued(memory)
    a, b = pair
    pa, pb = state.profile(a), state.profile(b)
    copy_task = bool(_COPY_TASK.search(state.task))
    cmp = _compare_obs(state)
    res = (cmp or {}).get("result") or {}
    ready = bool(cmp and cmp.get("status") == "ready")
    # 10-14 晚：对应类任务（VCSL）只要比对给了对应段就算对上——source_match 是"B 是否是 A 的来源"的判断，部分复制时可能不成立
    matched = ready and bool(res.get("correspondences")) and (bool(res.get("source_match")) or copy_task)
    loc_op = "locate_frames" if "locate_frames" in state.inspect_ops else "locate"

    def notes() -> list[str]:
        out = _pair_notes(state, None, None, pa, pb, memory, compare=True) + _ocr_notes(state)
        if ready:
            out.append("compare_media span boundaries are accurate to about 0.5 s; frame-exact difference segments (frame_diff) "
                       "are half-open [start, end); a boundary bracketed by an inside frame that shows it and an outside frame "
                       "that does not is answered to that bracket")
            if not matched:
                out.append("compare_media found no consistent correspondence between A and B; any comparison rests on the "
                           "sampled frames")
                ut = memory.get("uniform_times") or {}
                if ut.get("A") and ut.get("B"):
                    out.append(f"A frames at {ut['A']} s and B frames at {ut['B']} s were taken at the same relative positions "
                               f"(1/8, 3/8, 5/8, 7/8 of each video); describe each pair side by side")
        return out

    def status_and_report(rule: str, last_turn: bool = False) -> Decision:
        missing = _missing_parts(state, memory, pa, pb, [], copy_task=copy_task,
                                 pairs=_locate_pairs(state, a) if copy_task and not matched else None)
        if ready and not matched and not copy_task:
            missing.append("Material A and Material B: no consistent frame-level correspondence was found, so time-aligned "
                           "differences cannot be measured (sampled frames only)")
        events = _visible_events(state, _events_from_compare(res, pa, pb, copy_task), res, pa, pb, copy_task) if matched else []
        chain = {"compare_media": ready}
        if events or not matched:
            chain["frames_compared"] = any(k.startswith(("cmpB_", "cmpA_", "uniform_", "review_pairs")) for k in _issued(memory))
        status, reason = _status_of(state, memory, chain=chain, missing=missing)
        if last_turn and status == "INCOMPLETE":
            reason = "turn budget exhausted: " + (reason or "the comparison did not finish")
        return _report(state, memory, rule, status=status, reason=reason, notes=notes(),
                       branch="report_last_turn" if last_turn else ("report_completed" if status == "COMPLETED" else "report_incomplete"))

    if state.remaining <= 0:
        return status_and_report("G1_last_turn_report", last_turn=True)

    # M1 一次比对（画面 + 声音）
    if "compare" not in issued:
        _mark(memory, "compare")
        return Decision("action", "M1_compare_media", _inspect([a, b], "compare_media", state.compare_params(mode="auto"), [],
                        f"Compare {a} with {b} in one pass: which spans correspond, what is cut, inserted, re-timed or changed, "
                        f"and whether the sound matches."), branch="INSPECT compare_media")

    # M2 比对图没附上 → READ_RECORD 带 focus 回看
    if ready and "review_cmp" not in issued and cmp.get("result_evidence_id"):
        pics = [i.get("evidence_id") for i in res.get("images") or [] if isinstance(i, dict) and i.get("evidence_id")]
        fd = res.get("frame_diff") if isinstance(res.get("frame_diff"), dict) else {}
        pics += [i.get("evidence_id") for i in fd.get("images") or [] if isinstance(i, dict) and i.get("evidence_id")]
        unseen = [e for e in pics if not state.seen(e)]
        if unseen:
            _mark(memory, "review_cmp")
            focus = _existing(state, unseen, state.limit)
            if focus:
                return Decision("action", "M2_review_compare_images", _read(cmp["result_evidence_id"], focus,
                                f"The comparison images {', '.join(focus)} were not attached; view them before checking boundaries."),
                                branch="READ_RECORD focus")

    # M1b 等长的短视频、比对没给逐帧差异：限定 A 全长再比一次，拿到逐帧精确的差异段（4126 这类"何时可见"的题）
    da, db = pa.get("duration"), pb.get("duration")
    # 10-15 对话 5''（追加 5）：整段比对一个对应段都没对上（0%）时不缩小范围重比（同一引擎缩窗也对不上）
    if (ready and matched and not copy_task and da and db and abs(da - db) < 0.1 and da <= 30 and not res.get("frame_diff")
            and "compare_exact" not in issued and state.remaining >= 3):
        _mark(memory, "compare_exact")
        top = _r(da - 0.01, 3)
        return Decision("action", "M1b_compare_frame_exact", _inspect([a, b], "compare_media", state.compare_params(mode="visual", range_a_s=[0.0, top]),
                        [], f"The two videos have the same length; compare A 0-{top:.2f} s frame by frame to get exact difference "
                            f"boundaries."), branch="INSPECT compare_media(range)")
    exact = next((o for o in reversed(state.observations) if o["operation"] == "compare_media"
                  and (o["params"] or {}).get("range_a_s") and o.get("status") == "ready"), None)
    if exact and (exact.get("result") or {}).get("frame_diff") and not res.get("frame_diff"):
        res = {**res, "frame_diff": exact["result"]["frame_diff"]}

    # M3 两侧取帧核看（最多两个事件）。只用学生视图里看得到的数（比对结果在历史里只留摘要行）
    if matched:
        visible_events = _visible_events(state, _events_from_compare(res, pa, pb, copy_task), res, pa, pb, copy_task)
        for ev in visible_events:
            step = _merged_event_step(state, memory, ev, a, b, pa, pb)
            if step is not None:
                return step
        # M3c 同一时刻画面不同的横向长条区域（字幕、标题条、标牌的形状）：两侧放大图各 OCR 一次（10-13）。
        # 被替换 / 添加的文字要逐字说出；放大图里没有文字时 OCR 结果为空，报告不提文字
        step = _region_ocr_step(state, memory, visible_events, a, b)
        if step is not None:
            return step
    elif ready:
        # M5 比对没对上：对应类任务以图找图；差异类任务两侧均匀取帧，成对看
        if copy_task and loc_op in state.inspect_ops:
            if "uniform_A" not in issued and state.remaining >= 2:
                _mark(memory, "uniform_A")
                return Decision("action", "B8_uniform_frames_A", _inspect([a], "get_frames", {"count": state.limit // 2}, [],
                                f"The comparison found no consistent correspondence; sample {a} evenly and look each frame up in {b}."),
                                branch="INSPECT get_frames count")
            a_obs = next((o for o in reversed(state.observations) if o["operation"] == "get_frames" and o["targets"] == [a]
                          and "count" in o["params"]), None)
            for eid, t in _frames_of(a_obs):
                if f"locate_{eid}" in issued or state.remaining < 2:
                    continue
                _mark(memory, f"locate_{eid}")
                return Decision("action", "B8_locate_in_B", _inspect([b], loc_op,
                                {"query": {"type": "image", "evidence_id": eid}, "top_k": 3}, _existing(state, [eid], 1),
                                f"Find where the content of {eid} ({a} at {t:.1f} s) appears in {b}; candidates still need a visual check."),
                                branch=f"INSPECT {loc_op}(image)")
            pairs: list[str] = []
            last_locate = None
            for obs in state.observations:
                if obs["operation"] != loc_op or obs.get("status") != "ready":
                    continue
                query = ((obs["params"].get("query") or {}).get("evidence_id"))
                cands = sorted((obs.get("result") or {}).get("candidates") or [], key=lambda c: -float(c.get("score") or 0))
                if query and cands:
                    pairs += [query, cands[0].get("evidence_id")]
                last_locate = obs
            if pairs and "review_pairs" not in issued and state.remaining >= 1:
                _mark(memory, "review_pairs")
                focus = _existing(state, pairs, state.limit)
                return Decision("action", "B8_review_pairs", _read((last_locate or {}).get("result_evidence_id") or pairs[0], focus,
                                "View each A frame next to its best-ranked B candidate together to judge which parts correspond."),
                                branch="READ_RECORD focus")
        else:
            for side, video, prof in (("A", a, pa), ("B", b, pb)):
                key = f"uniform_{side}"
                if key in issued or state.remaining < 2:
                    continue
                _mark(memory, key)
                d = prof.get("duration")
                params: dict[str, Any] = {"times_s": [_r(d * k / 4 + d / 8, 3) for k in range(4)]} if d else {"count": 4}
                focus = _existing(state, [e for e, _ in _frames_of(_frames_obs(state, a, (memory.get("uniform_times") or {}).get("A")))], 4)
                memory.setdefault("uniform_times", {})[side] = params.get("times_s")
                return Decision("action", "M5_uniform_frames", _inspect([video], "get_frames", params, focus if side == "B" else [],
                                f"The comparison found no consistent correspondence; sample {video} at the same relative positions "
                                f"to compare the two pictures directly."), branch="INSPECT get_frames times_s" if d else "INSPECT get_frames count")

    # M4 语音：任务没说只看画面时，有音轨的一侧各转写一次。10-15 对话 5''（追加 5）：纯视觉任务（问"何时可见"、不提声音）
    # 不自动跑 ASR（r0a videosham_4126 跑了两次）
    if not _NO_AUDIO_TASK.search(state.task) and not _visual_only(state.task) and "asr" in state.inspect_ops:
        for side, video, prof in (("A", a, pa), ("B", b, pb)):
            if (prof.get("audio") or 0) > 0 and f"asr_{side}" not in issued and state.remaining >= 2:
                _mark(memory, f"asr_{side}")
                return Decision("action", "B10_asr", _inspect([video], "asr", {}, _merged_key_images(state, memory, a, b),
                                f"{video} has an audio track; transcribe it to check whether the spoken content differs."),
                                branch="INSPECT asr")
    return status_and_report("M7_report")


def _merged_event_step(state: HiState, memory: dict[str, Any], ev: dict[str, Any], a: str, b: str,
                       pa: dict[str, Any], pb: dict[str, Any]) -> Decision | None:
    issued = _issued(memory)
    key = ev["key"]
    plan = memory.setdefault("plan", {})
    if f"cmpB_{key}" not in issued and ev["b_times"] and state.remaining >= 2:
        _mark(memory, f"cmpB_{key}")
        times = _clip_times(ev["b_times"], pb.get("duration"), 0.05)
        if times:
            plan[f"B_{key}"] = times
            params: dict[str, Any] = {"times_s": times}
            if ev.get("box"):
                params["box"] = ev["box"]
            return Decision("action", "M3_frames_B", _inspect([b], "get_frames", params, _existing(state, ev["images"], state.limit // 2),
                            f"Look at {b} around {ev['what']}" + (" (enlarged region)" if ev.get("box") else "") + "."),
                            branch="INSPECT get_frames times_s" + ("+box" if ev.get("box") else ""))
    if f"cmpA_{key}" not in issued and ev["a_times"] and state.remaining >= 1:
        _mark(memory, f"cmpA_{key}")
        times = _clip_times(ev["a_times"], pa.get("duration"), 0.05)
        if times:
            plan[f"A_{key}"] = times
            params = {"times_s": times}
            if ev.get("box"):
                params["box"] = ev["box"]
            b_frames = _frames_of(_frames_obs(state, b, plan.get(f"B_{key}")))
            focus = _existing(state, [e for e, _ in b_frames], state.limit // 2)
            return Decision("action", "M3_frames_A", _inspect([a], "get_frames", params, focus,
                            f"Look at {a} around {ev['what']}, next to the B frames."),
                            branch="INSPECT get_frames times_s" + ("+box" if ev.get("box") else ""))
    return None


def _region_ocr_step(state: HiState, memory: dict[str, Any], events: list[dict[str, Any]], a: str, b: str) -> Decision | None:
    if "ocr" not in state.inspect_ops:
        return None
    issued = _issued(memory)
    plan = memory.get("plan") or {}
    prof = state.profile(a)
    for ev in events:
        if not str(ev.get("key", "")).startswith("reg") or not ev.get("box"):
            continue
        # 只对横向的长条区域读字：框的像素宽高比 ≥ 1.6；竖条（人、物）不读
        x1, y1, x2, y2 = (float(v) for v in ev["basis"][-4:]) if len(ev.get("basis") or []) >= 6 else ev["box"]
        aspect = ((x2 - x1) * (prof.get("w") or 16.0)) / max(1e-6, (y2 - y1) * (prof.get("h") or 9.0))
        if aspect < 1.6:
            continue
        for side, video in (("A", a), ("B", b)):
            key = f"ocr{side}_{ev['key']}"
            times = plan.get(f"{side}_{ev['key']}")
            if not times or key in issued or state.remaining < 2:
                continue
            frames = _frames_of(_frames_obs(state, video, times))
            if not frames:
                continue
            _mark(memory, key)
            eid = frames[0][0]
            other = plan.get(f"{'B' if side == 'A' else 'A'}_{ev['key']}")
            keep = [e for e, _ in _frames_of(_frames_obs(state, b if side == "A" else a, other))][:1] if other else []
            return Decision("action", "M3c_ocr_region", _inspect([eid], "ocr", {}, _existing(state, [eid] + keep, 2),
                            f"Read any writing inside the enlarged differing region of {video} ({eid}), so that replaced or "
                            f"added text can be quoted exactly."), branch="INSPECT ocr(image)")
    return None


def _asr_notes(state: HiState) -> list[str]:
    """语音转写的摘录（walker 当时从观察里读到、存在 memory 的；学生当时看得到）。C 族报告要写视频本身说了什么。"""

    out = []
    for o in state.observations:
        if o["operation"] != "asr" or o.get("status") != "ready":
            continue
        segs = (o.get("result") or {}).get("segments") or []
        text = " ".join(str(x.get("text") or "").strip() for x in segs if isinstance(x, dict))
        if not text:
            continue
        rid = o.get("result_evidence_id") or "?"
        # 10-14 晚：不再把转写原文抄进提示——学生这一步的输入里没有原文时，教师照着提示写出来的词就是"凭空写出"
        # （导出 hidden_context 扣下的 Wi-Fi 等）。原文要写进报告，得先在状态里（_before_report 会先 QUERY_MEMORY 取回）
        out.append(f"speech transcript {rid} of {', '.join(o.get('targets') or [])}: one finding must say what the given video "
                   f"itself says and shows, quoting only what {rid} shows in this state (and the OCR record for on-screen text), "
                   f"and whether that matches the claim")
    return out


def _ocr_notes(state: HiState) -> list[str]:
    ocr = [o for o in state.observations if o["operation"] == "ocr" and o.get("status") == "ready"]
    if not ocr:
        return []
    ids = ", ".join(str(o.get("result_evidence_id") or "?") for o in ocr)
    # 10-15（A7）：一次 OCR = 一个记录；观察行列出出现最多的文字（视频带首末时刻）和总数，逐框结果在记录里
    return [f"OCR records ({ids}) hold the text recognised in the images or video they were run on; the observation line of "
            f"each lists its most frequent texts (with first and last times for a video) and the counts. Quote recognised "
            f"text exactly as it appears in this state and cite that OCR record; where an OCR record shows no text, do not "
            f"state any text for that image; OCR can miss characters, so do not treat absent text as proof that none is there"]


def _merged_key_images(state: HiState, memory: dict[str, Any], a: str, b: str) -> list[str]:
    keys: list[str] = []
    plan = memory.get("plan") or {}
    for name, times in plan.items():
        if not name.startswith(("A_", "B_")):
            continue
        video = a if name.startswith("A_") else b
        keys += [e for e, _ in _frames_of(_frames_obs(state, video, times))][:2]
    cmp = _compare_obs(state)
    keys += [i.get("evidence_id") for i in ((cmp or {}).get("result") or {}).get("images") or [] if isinstance(i, dict)][:2]
    return _existing(state, keys, state.limit)


def source_for(ev: dict[str, Any], diff: dict[str, Any] | None, align: dict[str, Any] | None) -> dict[str, Any] | None:
    return diff if ev["kind"] == "diff" else align


def _visible_numbers(state: HiState) -> set[float]:
    """学生此刻视图里出现的所有数（按 3 位与 2 位小数取整后的集合），缓存在 state 上。"""

    cached = getattr(state, "_numbers", None)
    if cached is None:
        cached = set()
        for token in re.findall(r"-?\d+(?:\.\d+)?", state.visible_text):
            try:
                value = float(token)
            except ValueError:
                continue
            for v in (value, abs(value)):              # "1.4-4.2" 里的 "-4.2" 是区间连字符，不是负号
                cached.add(round(v, 3))
                cached.add(round(v, 2))
        state._numbers = cached  # type: ignore[attr-defined]
    return cached


def _visible(state: HiState, values: list[float | None]) -> bool:
    numbers = _visible_numbers(state)
    return all(v is None or round(v, 3) in numbers or round(v, 2) in numbers for v in values)


def _visible_mapping(state: HiState) -> dict[str, float]:
    """学生可见的 A→B 时间映射：取 diff_media 请求里自己写的 mapping（history 里可见）；没有 = 恒等。"""

    for obs in reversed(state.observations):
        if obs["operation"] == "diff_media":
            m = obs["params"].get("mapping") or {}
            return {"scale": float(m.get("scale", 1.0) or 1.0), "offset_s": float(m.get("offset_s", 0.0) or 0.0)}
    return {"scale": 1.0, "offset_s": 0.0}


def _fwd(mapping: dict[str, float], t: float | None) -> float | None:
    return None if t is None else mapping["scale"] * t + mapping["offset_s"]


def _inv(mapping: dict[str, float], t: float | None) -> float | None:
    return None if t is None else (t - mapping["offset_s"]) / (mapping["scale"] or 1.0)


def _clip_times(values: list[float | None], duration: float | None, dt: float) -> list[float]:
    limit = (duration - min(dt, 0.05)) if duration else None
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        value = max(0.0, value)
        if limit is not None and value > limit:
            value = limit
        value = _r(value, 3)
        if value not in out:
            out.append(value)
    return sorted(out)


def _next_offset(state: HiState, record_id: str) -> int | None:
    """最近一次 READ_RECORD 同一记录的 next_offset（观察条目 "READ_RECORD E38; offset=0; next_offset=6000"）。"""

    latest = state.view.get("latest_observation") or {}
    items = " ".join(str(x) for x in latest.get("items") or []) + " " + str(latest.get("preview") or "")[:300]
    m = re.search(r"READ_RECORD\s+" + re.escape(record_id) + r";\s*offset=(\d+);\s*next_offset=(\w+)", items)
    if not m:
        return None
    return int(m.group(2)) if m.group(2).isdigit() else -1


def _event_step(state: HiState, memory: dict[str, Any], ev: dict[str, Any], a: str, b: str, pa: dict[str, Any],
                pb: dict[str, Any], mapping: dict[str, float], source: dict[str, Any] | None,
                plan: dict[str, Any]) -> Decision | None:
    issued = _issued(memory)
    key = ev.get("key") or f"{ev['kind']}{ev['index']}"
    if f"skip_{key}" in issued:
        return None
    coarse = ev["kind"] == "align"
    if f"frames_B_{key}" not in issued:
        if state.remaining < 2:
            return None
        b0, b_last, b1 = ev["b"]
        if coarse and not ev.get("key"):
            # 对应题的首个候选：diff 请求里自己写过 range_a_s 与 mapping（history 可见），B 侧由它换算，不再抄对齐结果里的数
            diff_obs = next((o for o in reversed(state.observations) if o["operation"] == "diff_media"), None)
            ra = ((diff_obs or {}).get("params") or {}).get("range_a_s")
            if ra and ev["index"] == 0:
                b0, b1 = _fwd(mapping, float(ra[0])), _fwd(mapping, float(ra[1]))
        if not _visible(state, [b0, b1]) and not (coarse and ev["index"] == 0 and b0 is not None and b0 != ev["b"][0]):
            # 起止不在学生此刻的视图里：分页重读结果记录（最多 3 页），还看不到就放弃这一段
            step = _reread_step(state, memory, ev, source, min_remaining=3)
            if step is not None:
                return step
            _mark(memory, f"skip_{key}")
            return None
        if coarse:
            times = _clip_times([(b0 or 0) - 1.0, b0, (b1 or 0) - 1.0, b1], pb["duration"], 1.0)
        else:
            times = _clip_times([None if b0 is None else b0 - ev["dt_b"], b0, b_last, b1], pb["duration"], ev["dt_b"])
        if not times:
            _mark(memory, f"skip_{key}")
            return None
        _mark(memory, f"frames_B_{key}")
        plan[f"B_{key}"] = times
        focus = _existing(state, ev["images"], state.limit // 2)
        return Decision("action", "B4_boundary_frames_B", _inspect([b], "get_frames", {"times_s": times}, focus,
                        _why_frames(ev, b, side="B")), branch="INSPECT get_frames times_s")
    if f"frames_A_{key}" not in issued and state.remaining >= 1:
        b_times = plan.get(f"B_{key}") or []
        ev_map = mapping
        if coarse and ev.get("mapping"):
            m = ev["mapping"]
            ev_map = {"scale": _r(float(m.get("scale") or 1.0), 3), "offset_s": _r(float(m.get("offset_s") or 0.0), 3)}
        times = _clip_times([_inv(ev_map, t) for t in b_times], pa["duration"], ev["dt_a"])
        if not times:
            _mark(memory, f"frames_A_{key}")
            return None
        _mark(memory, f"frames_A_{key}")
        plan[f"A_{key}"] = times
        b_obs = _frames_obs(state, b, b_times)
        focus = _existing(state, [e for e, _ in _frames_of(b_obs)], state.limit // 2)
        return Decision("action", "B4_boundary_frames_A", _inspect([a], "get_frames", {"times_s": times}, focus,
                        _why_frames(ev, a, side="A", paired=focus)), branch="INSPECT get_frames times_s")
    return None


def _reread_step(state: HiState, memory: dict[str, Any], ev: dict[str, Any], source: dict[str, Any] | None, *,
                 min_remaining: int) -> Decision | None:
    """结果记录在视图里被截断、这一段的起止看不到：READ_RECORD 分页重读（每段最多 3 页）。"""

    key = ev.get("key") or f"{ev['kind']}{ev['index']}"
    record = (source or {}).get("result_evidence_id")
    if not record:
        return None
    pages = memory.setdefault("reread_pages", {}).get(key, 0)
    # 同一记录的页只读一次（几段共用一个记录时）；下一页的 offset 取上次读这条记录时观察里写的 next_offset
    seen_next = _next_offset(state, record)
    if seen_next is not None:
        memory.setdefault("record_next", {})[record] = seen_next
    read = memory.setdefault("record_pages", {}).setdefault(record, [])
    nxt = 0 if 0 not in read else memory.get("record_next", {}).get(record)
    if nxt is None or nxt == -1 or nxt in read or pages >= 3 or state.remaining < min_remaining:
        return None
    read.append(nxt)
    memory["reread_pages"][key] = pages + 1
    _mark(memory, f"reread_{key}_{pages}")
    args: dict[str, Any] = {"record_id": record}
    if nxt:
        args["offset"] = nxt
    return Decision("action", "B4a_reread_result" if not nxt else "B4b_reread_next_page",
                    {"action": "READ_RECORD", "arguments": args, "focus_evidence_ids": [],
                     "rationale": f"The {(source or {}).get('operation') or 'tool'} output is cut short in the current view; read "
                                  f"{record}{' from offset ' + str(nxt) if nxt else ''} for the times of {_what(ev)}."},
                    branch="READ_RECORD offset>0" if nxt else "READ_RECORD text")


def _event_bounds_visible(state: HiState, ev: dict[str, Any]) -> bool:
    return _visible(state, [ev["b"][0], ev["b"][2]]) and _visible(state, [ev["a"][0], ev["a"][2]])


def _uncovered(pa: dict[str, Any], pb: dict[str, Any], mapping: dict[str, float], diff: dict[str, Any] | None,
               copy_task: bool) -> list[tuple[str, float, float]]:
    """按可见的时长与 mapping 算两侧没被比对覆盖的区间（≥ 1 秒且 ≥ 该侧时长 10%），最多取 2 段（按长短）。"""

    da, db = pa.get("duration"), pb.get("duration")
    if not da or not db:
        return []
    a_lo, a_hi = 0.0, da
    if diff is not None:
        ra = (diff.get("params") or {}).get("range_a_s")
        if ra:
            a_lo, a_hi = float(ra[0]), float(ra[1])
    # A 里映射后落在 B 范围内的部分 = 实际比对过的 A 区间
    lo = max(a_lo, _inv(mapping, 0.0) or 0.0)
    hi = min(a_hi, _inv(mapping, db) or da)
    if mapping["scale"] < 0:
        lo, hi = a_lo, a_hi
    covered_a = (min(lo, hi), max(lo, hi))
    covered_b = (max(0.0, _fwd(mapping, covered_a[0]) or 0.0), min(db, _fwd(mapping, covered_a[1]) or db))
    gaps: list[tuple[str, float, float]] = []
    for side, (c0, c1), dur in (("A", covered_a, da), ("B", covered_b, db)):
        for g0, g1 in ((0.0, c0), (c1, dur)):
            if min(g1, dur - 0.05) - g0 >= max(1.0, 0.1 * dur):
                gaps.append((side, _r(g0, 3), _r(g1, 3)))
    gaps.sort(key=lambda g: -(g[2] - g[1]))
    return gaps[:2]


def _locate_pairs(state: HiState, a: str) -> list[tuple[float, float, float]]:
    """以图找图得到的 (A 帧时刻, B 最佳候选时刻, 分数)，按分数降序（数取视图里的 3 位小数）。"""

    a_times = {e: t for o in state.observations if o["operation"] == "get_frames" and o["targets"] == [a] for e, t in _frames_of(o)}
    out = []
    for obs in state.observations:
        if obs["operation"] not in ("locate_frames", "locate") or obs.get("status") != "ready":
            continue
        q = ((obs["params"].get("query") or {}).get("evidence_id"))
        cands = sorted((obs.get("result") or {}).get("candidates") or [], key=lambda c: -float(c.get("score") or 0))
        if q in a_times and cands and _num(cands[0].get("time_s")) is not None:
            out.append((_r(a_times[q], 3), _r(float(cands[0]["time_s"]), 3), float(cands[0].get("score") or 0)))
    return sorted(out, key=lambda x: -x[2])


def _missing_parts(state: HiState, memory: dict[str, Any], pa: dict[str, Any], pb: dict[str, Any],
                   uncovered: list[tuple[str, float, float]], *, copy_task: bool = False,
                   pairs: list[tuple[float, float, float]] | None = None) -> list[str]:
    """报告的缺口（status_reason 的各段，格式与 S4 要求的一致，GIA 按段分类）：
    - 能靠更多查看补上的：没看的区间、没转写的音轨；
    - 补不了的（材料本身决定）：一侧多出来、另一侧没有对应的区间（只能取样看，量不出差异）；只有一侧有音轨（声音没法比）。
    后一类也让 status 落到 INCOMPLETE——wk3 的 0069 把它们当 COMPLETED 报，被判 completed_without_finishing。"""

    issued = _issued(memory)
    missing = []
    other = {"A": "B", "B": "A"}
    # 10-15（B6）：只有采样对应点时，任务要逐段的时间范围 / 起止才算缺口；只问"有没有对应"的任务不列
    for ta, tb, _ in (pairs or [])[:2] if _BOUNDARY_TASK.search(state.task) else []:
        missing.append(f"correspondence A {ta:.3f} s ↔ B {tb:.3f} s: start and end boundaries not established (sampled frames only)")
    for side, lo, hi in uncovered:
        if copy_task:
            # 对应类任务：整段对齐没对上的区间不等于"没有对应"，可以限定范围再对齐（第 2 轮能补）
            missing.append(f"Material {side} {lo:.3f}–{hi:.3f} s: no corresponding clip found by the whole-video alignment, so "
                           f"correspondence there is not established")
        elif f"uncovered_{side}_{lo:.3f}" not in issued:
            missing.append(f"Material {side} {lo:.3f}–{hi:.3f} s: not examined")
        else:
            missing.append(f"Material {side} {lo:.3f}–{hi:.3f} s: no counterpart in Material {other[side]}, so differences there "
                           f"cannot be measured (sampled frames only)")
    # 10-16 对话 5''：纯视觉任务（问"何时可见"、不提声音）不跑 ASR，也不把"音轨没转写 / 一侧没音轨"列为缺口——r0g 第一遍
    # videosham_4126 第 1 轮因此判 INCOMPLETE，第 2 轮去转写两侧（都没有语音）
    if not _NO_AUDIO_TASK.search(state.task) and not _visual_only(state.task):
        audio = {"A": pa.get("audio"), "B": pb.get("audio")}
        for side in ("A", "B"):
            if (audio[side] or 0) > 0 and audio[other[side]] == 0:
                missing.append(f"audio of Material {side}: Material {other[side]} has no audio track, so the sound cannot be compared")
        if "asr" in state.inspect_ops:
            for side in ("A", "B"):
                if (audio[side] or 0) > 0 and f"asr_{side}" not in issued:
                    missing.append(f"audio of Material {side}: not transcribed")
    return missing


def _weak_align(align: dict[str, Any] | None, pa: dict[str, Any], pb: dict[str, Any]) -> bool:
    """对齐不可用：失败、没有候选，或最好的候选覆盖不到较短一侧的 25%。"""

    match = _best_match(align)
    if not match:
        return True
    durations = [x for x in (pa.get("duration"), pb.get("duration")) if x]
    if not durations:
        return False
    ra = match.get("range_a_s") or [0, 0]
    rb = match.get("range_b_s") or [0, 0]
    span = min(float(ra[1]) - float(ra[0]), float(rb[1]) - float(rb[0]))
    return span < 0.25 * min(durations)


def _what(ev: dict[str, Any]) -> str:
    return f"difference segment {ev['index'] + 1}" if ev["kind"] == "diff" else f"alignment candidate {ev['index'] + 1}"


def _event_visible(state: HiState, ev: dict[str, Any]) -> bool:
    """这段的起点时刻是否出现在学生此刻看到的文字里（摘要里的 "A 0.200–1.367 s"、或段对象里的数）。"""

    start = ev["a"][0]
    if start is None:
        return True
    text = state.visible_text
    tokens = [f"{start:.3f}", f": {_r(start, 3)},", f":{_r(start, 3)},", f"{start:.2f} s"]
    return any(t in text for t in tokens)


def _frames_obs(state: HiState, video: str, times: list[float] | None) -> dict[str, Any] | None:
    for obs in reversed(state.observations):
        if obs["operation"] == "get_frames" and obs["targets"] == [video] and (times is None or obs["params"].get("times_s") == times):
            return obs
    return None


def _bbox_area(box: Any) -> float | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _pad_box(box: list[float], pad: float = 0.05) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, y1, x2, y2 = max(0.0, x1 - pad), max(0.0, y1 - pad), min(1.0, x2 + pad), min(1.0, y2 + pad)
    if x2 - x1 < 0.1:
        c = (x1 + x2) / 2
        x1, x2 = max(0.0, c - 0.05), min(1.0, c + 0.05)
    if y2 - y1 < 0.1:
        c = (y1 + y2) / 2
        y1, y2 = max(0.0, c - 0.05), min(1.0, c + 0.05)
    return [_r(x1, 3), _r(y1, 3), _r(x2, 3), _r(y2, 3)]


def _key_images(state: HiState, memory: dict[str, Any], events: list[dict[str, Any]], a: str, b: str) -> list[str]:
    """报告前最后一步的 focus：首段的 A/B 边界帧（成对）+ 放大图 + 差分峰值图，总数 ≤ 容量。"""

    keys: list[str] = []
    crops = [o for o in state.observations if o["operation"] == "crop" and o.get("status") == "ready"]
    for obs in crops[-2:]:
        keys += obs.get("created") or []
    for ev in events[:2]:
        key = f"{ev['kind']}{ev['index']}"
        bf = _frames_of(_frames_obs(state, b, memory.get("plan", {}).get(f"B_{key}")))
        af = _frames_of(_frames_obs(state, a, memory.get("plan", {}).get(f"A_{key}")))
        keys += [e for e, _ in bf[:2]] + [e for e, _ in af[:2]]
        keys += ev.get("images", [])[1:2]
    return _existing(state, keys, state.limit)


def _why_frames(ev: dict[str, Any], video: str, *, side: str, paired: list[str] | None = None) -> str:
    what = _what(ev)
    text = f"Check the start and end boundaries of {what} on the {side} side ({video}) with frames just before and after it."
    if paired:
        text += f" Keep {', '.join(paired[:4])} in view to compare both sides."
    return text


def _pair_notes(state: HiState, align: dict[str, Any] | None, diff: dict[str, Any] | None,
                pa: dict[str, Any], pb: dict[str, Any], memory: dict[str, Any] | None = None, compare: bool = False) -> list[str]:
    notes = []
    memory = memory or {}
    issued = _issued(memory)
    for side, lo, hi in memory.get("uncovered") or []:
        done = f"uncovered_{side}_{lo:.3f}" in issued
        notes.append(f"{side} {lo:.2f}-{hi:.2f} s lies outside the compared / aligned range; "
                     + ("it was sampled with get_frames (sampled frames only)" if done else "it was NOT examined"))
    for side, prof in (("A", pa), ("B", pb)):
        if _NO_AUDIO_TASK.search(state.task):
            continue
        if prof.get("audio") == 0:
            notes.append(f"{side} has no audio track (media profile)")
        elif (prof.get("audio") or 0) > 0:
            notes.append(f"{side} audio " + ("was transcribed with asr; report what the transcripts show, and that a failed or empty "
                                             "transcription is not proof of silence" if f"asr_{side}" in issued else "was NOT transcribed"))
    if pa.get("fps"):
        notes.append(f"frame-level precision: A is {pa['fps']:.3f} fps (one frame = {1 / pa['fps']:.4f} s); diff segments are half-open [start, end)")
    if compare:
        return notes
    if align is None or align.get("status") != "ready":
        notes.append("visual alignment did not return a usable result")
    if diff is None:
        notes.append("frame-by-frame difference was not computed")
    elif diff.get("status") != "ready":
        notes.append("diff_media failed")
    audio = state.last("align_media", mode="audio")
    if audio is None:
        notes.append("audio was not compared (not attempted)")
    elif audio.get("status") != "ready":
        notes.append("audio alignment did not return a result; audio is unverified, not 'no difference'")
    return notes


# ---- 单段视频按描述定位（qvhighlights）


def _decide_locate(state: HiState, memory: dict[str, Any], video: str) -> Decision:
    issued = _issued(memory)
    prof = state.profile(video)
    duration = prof.get("duration")
    desc = (_LOCATE_DESC.search(state.task).group(1) if _LOCATE_DESC.search(state.task) else state.task).strip().rstrip(".")
    loc_op = "locate_frames" if "locate_frames" in state.inspect_ops else "locate"
    loc = state.last(loc_op)
    notes = []
    if prof.get("fps"):
        notes.append(f"video frame rate {prof['fps']:.3f} fps; boundaries are bracketed by the sampled frames, not exact cut times")

    if state.remaining <= 0:
        ok = loc is not None and loc.get("status") == "ready" and "probe1" in issued
        return _report(state, memory, "G1_last_turn_report", status="COMPLETED" if ok else "INCOMPLETE",
                       reason=None if ok else "turn budget exhausted before the boundaries were checked",
                       notes=notes + _locate_notes(memory), branch="report_last_turn")

    if "locate" not in issued:
        _mark(memory, "locate")
        return Decision("action", "Q1_locate_frames", _inspect([video], loc_op,
                        {"query": {"type": "text", "text": desc[:300]}, "top_k": 6}, [],
                        f"Look for frames in {video} that match the description, then check the extent around them."),
                        branch=f"INSPECT {loc_op}")

    cands = sorted([(c["evidence_id"], float(c.get("time_s") or 0.0), float(c.get("score") or 0.0))
                    for c in ((loc or {}).get("result") or {}).get("candidates") or [] if c.get("evidence_id")],
                   key=lambda x: x[1])
    if not cands:
        return _report(state, memory, "Q9_no_candidates", status="INCOMPLETE",
                       reason="locate_frames returned no candidate frames", notes=notes, branch="report_incomplete")
    cluster = _main_cluster(cands)
    t0, t1 = cluster[0][1], cluster[-1][1]
    memory["cluster"] = [t0, t1]

    def clip(values: list[float]) -> list[float]:
        top = (duration - 0.05) if duration else None
        out = []
        for v in values:
            v = max(0.0, v if top is None else min(v, top))
            v = _r(v, 2)
            if v not in out:
                out.append(v)
        return sorted(out)

    # 合并目录（hi@0.8.0）：镜头切分给出硬切时刻，边界就夹在切点两侧（比 ±3 s 取样准）
    if "video_structure" in state.inspect_ops and "structure" not in issued and state.remaining >= 2:
        _mark(memory, "structure")
        top = (duration - 0.01) if duration else t1 + 20
        rng = [_r(max(0.0, t0 - 20), 2), _r(min(top, t1 + 20), 2)]
        memory["structure_range"] = rng
        return Decision("action", "Q1b_video_structure", _inspect([video], "video_structure", {"range_s": rng}, [],
                        f"Split {rng[0]:.0f}-{rng[1]:.0f} s into shots, so the segment's start and end can be placed at cuts."),
                        branch="INSPECT video_structure")
    vs = state.last("video_structure")
    shots = _shots(vs)
    if shots and "structure_read" not in issued and vs.get("result_evidence_id") and state.remaining >= 2 \
            and not _visible(state, [shots[0][1]]):
        # 镜头表只在结果记录里（历史只留摘要行）：先读记录，切点时刻出现在视图里再用
        _mark(memory, "structure_read")
        return Decision("action", "Q1c_read_structure", {"action": "READ_RECORD", "arguments": {"record_id": vs["result_evidence_id"]},
                        "focus_evidence_ids": [], "rationale": "Read the shot list to get the cut times around the candidates."},
                        branch="READ_RECORD text")
    if shots and "probe1" not in issued and state.remaining >= 1:
        first = next((sh for sh in shots if sh[0] <= t0 < sh[1]), None)
        last = next((sh for sh in shots if sh[0] <= t1 < sh[1]), None)
        if first and last and _visible(state, [first[0], last[1]]):
            _mark(memory, "probe1")
            _mark(memory, "probe2")
            s0, e1 = first[0], last[1]
            memory["shot_bounds"] = [s0, e1]
            times = clip([s0 - 3, s0 - 0.5, s0 + 0.3, e1 - 0.3, e1 + 0.5, e1 + 3])
            memory["probe1"] = times
            focus = _existing(state, [c[0] for c in sorted(cluster, key=lambda c: -c[2])], 2)
            return Decision("action", "Q2_probe_shot_cuts", _inspect([video], "get_frames", {"times_s": times}, focus,
                            f"The candidates fall in the shots from the cut at {s0:.2f} s to the cut at {e1:.2f} s; look just inside "
                            f"and outside both cuts, and one shot further out."), branch="INSPECT get_frames times_s")
    if "probe1" not in issued and state.remaining >= 1:
        _mark(memory, "probe1")
        times = clip([t0 - 6, t0 - 3, t1 + 3, t1 + 6])
        memory["probe1"] = times
        focus = _existing(state, [c[0] for c in sorted(cluster, key=lambda c: -c[2])], state.limit // 2)
        return Decision("action", "Q2_probe_boundaries", _inspect([video], "get_frames", {"times_s": times}, focus,
                        f"Candidates {', '.join(c[0] for c in cluster[:4])} cluster at {t0:.1f}-{t1:.1f} s; sample frames on both sides to bracket the segment."),
                        branch="INSPECT get_frames times_s")
    if "probe2" not in issued and state.remaining >= 1:
        _mark(memory, "probe2")
        times = clip([t0 - 1.5, t0 - 0.5, t1 + 0.5, t1 + 1.5])
        memory["probe2"] = times
        p1 = _frames_of(_frames_obs(state, video, memory.get("probe1")))
        edge = [c[0] for c in (cluster[0], cluster[-1])]
        # 报告那一步能看到的图 = 本步新帧 + focus：留第一轮的边界外侧帧，两轮合起来把起止各夹在 4 帧之间
        focus = _existing(state, [e for e, _ in p1] + edge, state.limit // 2)
        return Decision("action", "Q3_refine_boundaries", _inspect([video], "get_frames", {"times_s": times}, focus,
                        f"Narrow the start near {t0:.1f} s and the end near {t1:.1f} s with frames closer to the candidate edges."),
                        branch="INSPECT get_frames times_s")
    return _report(state, memory, "Q4_report", status="COMPLETED", reason=None, notes=notes + _locate_notes(memory),
                   branch="report_completed")


def _shots(obs: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not obs or obs.get("status") != "ready":
        return []
    out = []
    for sh in (obs.get("result") or {}).get("shots") or []:
        rng = sh.get("range_s") if isinstance(sh, dict) else None
        if rng and len(rng) == 2 and _num(rng[0]) is not None and _num(rng[1]) is not None:
            out.append((float(rng[0]), float(rng[1])))
    return out


def _main_cluster(cands: list[tuple[str, float, float]], gap: float = 15.0) -> list[tuple[str, float, float]]:
    clusters: list[list[tuple[str, float, float]]] = []
    for item in cands:
        if clusters and item[1] - clusters[-1][-1][1] <= gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return max(clusters, key=lambda c: (sum(x[2] for x in c), len(c)))


def _locate_notes(memory: dict[str, Any]) -> list[str]:
    notes = []
    if memory.get("cluster"):
        t0, t1 = memory["cluster"]
        notes.append(f"frame-retrieval candidates cluster between {t0:.2f} s and {t1:.2f} s (retrieval scores rank only)")
    if memory.get("shot_bounds"):
        s0, e1 = memory["shot_bounds"]
        notes.append("the probe frames were taken just inside and just outside the hard cuts that video_structure found around "
                     "the candidates; a boundary is bracketed by an inside frame that matches the description and an outside "
                     "frame that does not (give the two probe times, not the cut time itself)")
    if memory.get("probe1"):
        notes.append(f"boundary probes at {memory['probe1']} s")
    if memory.get("probe2"):
        notes.append(f"refined probes at {memory['probe2']} s; precision is limited to the gap between adjacent sampled frames")
    return notes


# ---------------------------------------------------------------------------- GIA


# ---------------------------------------------------------------------------- HI：A 族（找原片）/ C 族（核查声明）（D2，10-13）

_FIND_ORIGINAL = re.compile(r"\boriginal (?:source|version|video|upload)|find (?:its|the) (?:original|source)|"
                            r"where (?:it|the video) (?:came|comes) from|provenance", re.I)
_CLAIM_TASK = re.compile(r"\bfact-check|\bclaim\b", re.I)
_MAX_TEXT_QUERIES = 2          # A 族：文字查询最多 2 条，之后以图搜一次
_FULL_MATCH = 0.6              # A 族：最好的候选画面对上不到 60% 时（多半是剪辑过的另一个上传），先用掉剩下的查询再定
_MAX_CLAIM_SEARCHES = 3        # C 族：最多 3 条查询、5 次阅读
_MAX_CLAIM_READS = 5


_EARLIEST = re.compile(r"\bfirst (?:published|posted|appeared|uploaded)|earliest (?:publication|copy|source|appearance)", re.I)
_MAX_DATE_LOOKUPS = 5
# L 族（10-13）：给定照片随声明流传，问它是否真拍于声明说的地点 / 实际拍于哪里（AVerImaTeC "Where was this image taken?"）
_PLACE_CHECK = re.compile(r"\b(?:really|actually) shows? the place|where (?:the|this|it) (?:picture|image|photo )?was (?:actually )?taken", re.I)


def task_family(task: str, has_search: bool) -> str:
    """A（找原片）/ L（照片拍摄地，10-13）/ E（图片最早出处，10-13）/ C（核查声明）/ B（给定媒体）。只看学生可见的任务文字与动作清单。"""

    if has_search and _PLACE_CHECK.search(task):
        return "L"
    if has_search and _EARLIEST.search(task):
        return "E"
    if has_search and _FIND_ORIGINAL.search(task):
        return "A"
    if has_search and _CLAIM_TASK.search(task):
        return "C"
    return "B"


_ITEM_RESULT = re.compile(r'"(\[R\d+\] [^"]*)"')
_ITEM_OTHER = re.compile(r'"((?:content_match|dates|full_comparison|SEARCH [A-Z_]+)[^"]*)"')
_CM_RUN = re.compile(r"(E\d+) ([\d.]+)-([\d.]+) s = this video ([\d.]+)-([\d.]+) s")


def _items_from_preview(preview: str) -> list[str]:
    """截断的观察 preview（JSON 文本的开头）→ 其中完整出现的结果行（学生看到的也是这些）。"""

    head = preview.split('"details"', 1)[0]
    found = [(m.start(), m.group(1)) for m in _ITEM_RESULT.finditer(head)]
    found += [(m.start(), m.group(1)) for m in _ITEM_OTHER.finditer(head)]
    return [text.replace('\\"', '"') for _, text in sorted(found)]


def _results_from_items(items: list[str]) -> list[dict[str, Any]]:
    """结果行 → 与 details.results 同形的最小结果（候选、网址、页面 E、画面比对）。"""

    results: dict[str, dict[str, Any]] = {}
    for line in items:
        m = re.match(r"\[(R\d+)\] (\S+); (.*)$", line)
        if m:
            page = re.search(r"\bpage=(?:ready|partial)\((E\d+)", m.group(3))
            # r1：以图搜 output_type=image 的结果行还有 image= / thumbnail=（这个网页上那张图）
            mats = [{"kind": k, "status": "ready", "evidence": [{"evidence_id": e, "modalities": ["image"]}]}
                    for k, e in re.findall(r"\b(image|thumbnail)=(?:ready|partial)\((E\d+)", m.group(3))]
            results.setdefault(m.group(1), {"candidate_id": m.group(1), "source_url": m.group(2).rstrip(";"),
                                            "page_evidence_id": page.group(1) if page else None, "materials": mats})
            continue
        m = re.match(r"content_match\[(R\d+)\]: (.*)$", line)
        if not m or m.group(1) not in results:
            continue
        text = m.group(2)
        pct = re.search(r"(\d+)% of", text)
        mat = re.search(r"\b(?:of|part of) (E\d+)", text)
        status = ("source" if "picture samples are found in this video" in text else
                  "partial" if text.startswith("part of") else
                  "none" if "was not found" in text else "download_failed")
        runs = [{"a_range_s": [float(r[1]), float(r[2])], "b_range_s": [float(r[3]), float(r[4])], "kind": "normal",
                 "mirrored": False} for r in _CM_RUN.findall(text)]
        results[m.group(1)]["content_match"] = {"material": mat.group(1) if mat else None, "status": status,
                                                "fraction": int(pct.group(1)) / 100 if pct else None, "runs": runs}
    return list(results.values())


def _search_obs(state: HiState) -> list[dict[str, Any]]:
    """各次 SEARCH 的 (请求, 结果行, 结果明细)。旧的观察在之后的状态里可能被压缩掉，读到时存进 memory。"""

    cache: dict[str, Any] = state.memory.setdefault("search_cache", {})
    history = list(state.view.get("history") or [])
    latest = state.view.get("latest_observation")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if item.get("action") != "SEARCH":
            continue
        observation = item.get("observation")
        if observation is None and index == len(history) - 1:
            observation = latest
        key = f"{index}:{json.dumps(item.get('request') or {}, sort_keys=True, ensure_ascii=False)}"
        if isinstance(observation, dict) and (observation.get("details") or observation.get("items")):
            cache[key] = {"request": item.get("request") or {}, "items": [str(x) for x in observation.get("items") or []],
                          "details": observation.get("details") or {}}
        elif isinstance(observation, dict) and observation.get("preview") and key not in cache:
            # 观察太长，状态里只有截断的 preview：从里面完整的结果行重建（10-14，wk9 1KkAwCyj7nI）
            items = _items_from_preview(str(observation["preview"]))
            if items:
                failed = any(x.startswith("SEARCH ") and "ERROR" in x for x in items)
                cache[key] = {"request": item.get("request") or {}, "items": items, "from_preview": True,
                              "details": {"results": _results_from_items(items),
                                          "stop_reason": "error" if failed else "target_reached"}}
        out.append(cache.get(key) or {"request": item.get("request") or {}, "items": [], "details": {}})
    return out


def _page_of(result: dict[str, Any]) -> str | None:
    if result.get("page_evidence_id"):
        return str(result["page_evidence_id"])
    for material in result.get("materials") or []:
        if material.get("kind") == "page" and material.get("status") in ("ready", "partial"):
            for card in material.get("evidence") or []:
                if card.get("evidence_id"):
                    return str(card["evidence_id"])
    return None


def _image_of(result: dict[str, Any]) -> tuple[str | None, str | None]:
    """r1（10-17）：以图搜（output_type=image）的结果里，这个网页上的那张图（engine 报的原图，kind=image）；没取到就退到
    缩略图（kind=thumbnail）。返回 (E, 种类)。"""

    for kind in ("image", "thumbnail"):
        for material in result.get("materials") or []:
            if material.get("kind") == kind and material.get("status") in ("ready", "partial"):
                for card in material.get("evidence") or []:
                    if card.get("evidence_id") and "image" in (card.get("modalities") or ["image"]):
                        return str(card["evidence_id"]), kind
    return None, None


# r1：search 层（reverse_normalize）写进摘要的缩略图自核结论，学生看得到的就是这句
_THUMB_VERDICT = re.compile(r"Thumbnail compared with the query image:\s*(near-identical|partial match|no match)", re.I)


def thumb_verdict(snippet: Any) -> str | None:
    """摘要里的缩略图哈希核对 → same / similar / different；没核对过 → None。"""

    m = _THUMB_VERDICT.search(str(snippet or ""))
    if not m:
        return None
    return {"near-identical": "same", "partial match": "similar", "no match": "different"}[m.group(1).lower()]


def _search_pages(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """检索结果里已取得的页面 E（按出现顺序，去重）。r1：带上这个网页上那张图的 E（以图搜 output_type=image 时）与
    缩略图哈希核对结论（verified，只用来排序，不用来排除候选）。"""

    pages: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_candidate: dict[str, dict[str, Any]] = {}
    for search in searches:
        for result in (search.get("details") or {}).get("results") or []:
            page = _page_of(result)
            cand = result.get("candidate_id")
            if not page and cand in by_candidate:
                page = by_candidate[cand]["evidence_id"]          # r1：SEARCH source=R output_type=image 的结果里 page 是缓存的
            if page and page in seen:
                # r1：之后对同一页 SEARCH source=R output_type=image 取到了页内主图——补进先前那条
                image, image_kind = _image_of(result)
                entry = next(p for p in pages if p["evidence_id"] == page)
                if image and not entry.get("image"):
                    entry.update(image=image, image_kind=image_kind)
                continue
            if page and page not in seen:
                seen.add(page)
                image, image_kind = _image_of(result)
                pages.append({"evidence_id": page, "candidate": result.get("candidate_id"), "url": result.get("source_url"),
                              "title": result.get("search_title"), "snippet": result.get("search_snippet"),
                              "content_match": result.get("content_match") or {}, "image": image, "image_kind": image_kind,
                              "verified": thumb_verdict(result.get("search_snippet"))})
                if cand:
                    by_candidate[str(cand)] = pages[-1]
    return pages


# ---- 10-15 对话 5''（追加 2 / 5）：检索类（E / L / C）交报告前的程序前置条件与选页启发式
_CLAIM_DATE_IN_TASK = re.compile(r"\((?:claim date|posted):\s*(\d{4}-\d{2}-\d{2})\)")
_KW_STOP = frozenset(("that this with from were have been which there their does about into than them they these those such also "
                      "image picture photo photograph video shows show showing claim claims circulated following given said says "
                      "after before during while where when what being over under more most some many very just only other "
                      "people person time year years date news report reports").split())


def _claim_words(claim: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]{4,}", claim.lower()) if w not in _KW_STOP}


def _claim_date(task: str) -> str | None:
    m = _CLAIM_DATE_IN_TASK.search(task)
    return m.group(1) if m else None


def _page_text(page: dict[str, Any]) -> str:
    url = str(page.get("url") or "")
    slug = re.sub(r"[^a-z0-9]+", " ", url.lower().split("://", 1)[-1].split("?", 1)[0])
    return f"{page.get('title') or ''} {page.get('snippet') or ''} {slug}"


def _must_read(pages: list[dict[str, Any]], read_ids: set[str], claim: str, *, dated: dict[str, str] | None = None,
               claim_date: str | None = None) -> list[dict[str, Any]]:
    """还没读、页面已取得（page=ready）、且标题 / 摘要 / 网址命中声明关键词（≥2 个且不少于 1/3；短声明 ≥1 个）的页；E 族另加：日期查询给出
    的日期 ≤ 声明日期的页。r0a e2：第 10 步搜到 2017 年 Kheer Bhawani 的原始报道（page=ready）就直接交报告。"""

    words = _claim_words(claim)
    need = max(2, -(-len(words) // 3)) if len(words) >= 4 else 1        # ≥2 个且不少于声明关键词的 1/3
    out = []
    for p in pages:
        if p["evidence_id"] in read_ids:
            continue
        hits = len(words & _claim_words(_page_text(p)))
        early = bool(dated and claim_date and p.get("candidate") in dated and dated[p["candidate"]] <= claim_date)
        if hits >= need or early:
            out.append({**p, "why": "dated on or before the claim date" if early else "its title matches the claim"})
    return out


def _open_gap_clause(must: list[dict[str, Any]]) -> str:
    """交报告时还有未读命中页：status_reason 里写成"能补"的缺口（GIA 据此派第 2 轮，第 2 轮按网址打开再读）。"""

    urls = [str(p.get("url")) for p in must if p.get("url")][:3]
    return ("candidate pages that match the claim were not examined: " + ", ".join(urls)) if urls else ""


def _query_text(request: dict[str, Any]) -> str:
    query = request.get("query")
    if isinstance(query, dict):
        return str(query.get("text") or query.get("evidence_id") or "")
    return str(query or "")


def _image_search_allowed(state: HiState) -> bool:
    for action in state.payload.get("available_actions") or []:
        if isinstance(action, dict) and action.get("name") == "SEARCH":
            for branch in (action.get("schema") or {}).get("oneOf") or []:
                if ((branch.get("properties") or {}).get("input_type") or {}).get("const") == "image":
                    return True
    return False


def _search_failed(search: dict[str, Any]) -> bool:
    details = search.get("details") or {}
    return details.get("stop_reason") in ("error", "timeout") or not (details.get("results") or [])


def _provider_error(search: dict[str, Any]) -> bool:
    """检索服务本身出错（不是没结果）：wk9 4u12EqPAHqU 第 1 条查询 DDGSException。"""

    return any(str(x).startswith("SEARCH ") and ("PROVIDER_ERROR" in str(x) or "TIMEOUT" in str(x))
               for x in search.get("items") or [])


def _query_searches(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """真正的查询（有 input_type）；打开 / 日期查询（source=…）不算。"""

    return [s for s in searches if (s.get("request") or {}).get("input_type")]


def _dates_allowed(state: HiState) -> bool:
    for action in state.payload.get("available_actions") or []:
        if isinstance(action, dict) and action.get("name") == "SEARCH":
            for branch in (action.get("schema") or {}).get("oneOf") or []:
                inc = ((branch.get("properties") or {}).get("include") or {}).get("items") or {}
                if inc.get("const") == "dates":
                    return True
    return False


def _dated(searches: list[dict[str, Any]]) -> dict[str, str]:
    """已做过日期查询的候选 R → 观察里的日期摘要行。"""

    out: dict[str, str] = {}
    for search in searches:
        req = search.get("request") or {}
        if list(req.get("include") or []) != ["dates"]:
            continue
        line = next((x for x in search.get("items") or [] if str(x).startswith("dates[")), "")
        out[str(req.get("source"))] = line or "(no dates line)"
    return out


def _dates_action(source: str, rule: str, why: str, branch: str = "SEARCH dates") -> Decision:
    return Decision("action", rule, {"action": "SEARCH", "arguments": {"source": source, "output_type": "text",
                                                                        "include": ["dates"]},
                                     "focus_evidence_ids": [], "rationale": why}, branch=branch)


_PDF_NO_TEXT = re.compile(r"no extractable text layer|text extraction failed|pages_with_text=0", re.I)


def _pdf_pages_step(state: HiState, memory: dict[str, Any], read_ids: set[str]) -> Decision | None:
    """读到的是没有文字层的 PDF（扫描件）：渲染前两页再看（10-13，get_frames pages）。"""

    if "get_frames" not in state.inspect_ops:
        return None
    issued = _issued(memory)
    for eid in sorted(read_ids):
        card = state.cards.get(eid) or {}
        label = str(card.get("label") or "") + " " + str(card.get("locator") or "")
        text = json.dumps(state.view.get("latest_observation") or {}, ensure_ascii=False)[:3000]
        if f"pdfpages_{eid}" in issued or not ("pdf" in label.lower() or eid in text) or not _PDF_NO_TEXT.search(label + " " + text):
            continue
        _mark(memory, f"pdfpages_{eid}")
        return Decision("action", "C6_pdf_pages", _inspect([eid], "get_frames", {"pages": [1, 2]}, [],
                        f"{eid} is a PDF without a text layer; render its first pages to read them."),
                        branch="INSPECT get_frames pages")
    return None


def _view_ids(state: HiState) -> set[str]:
    """学生视图里此刻有卡片的 E（材料、证据卡、附图）。"""

    ids = {str(c.get("evidence_id")) for c in (state.view.get("materials") or {}).values() if isinstance(c, dict)}
    ids |= {str(c.get("evidence_id")) for c in state.view.get("evidence_view") or [] if isinstance(c, dict)}
    return ids | {str(e) for e in state.attached if e}


def _visible_frames(state: HiState, video: str) -> list[str]:
    shown = _view_ids(state)
    frames = [e for e in _material_frames(state, video) if e in shown]
    attached = [e for e in frames if e in state.attached]
    return attached or frames


def _material_frames(state: HiState, video: str) -> list[str]:
    out = []
    for eid, card in sorted(state.cards.items(), key=lambda kv: int(kv[0][1:])):
        if video in (card.get("derived_from") or []) and card.get("acquired_by") == "INSPECT/get_frames" \
                and "image" in (card.get("modalities") or []):
            out.append(eid)
    return out


def _search_notes(searches: list[dict[str, Any]], found: dict[str, Any] | None) -> list[str]:
    notes = []
    for i, search in enumerate(searches, 1):
        req = search.get("request") or {}
        details = search.get("details") or {}
        results = details.get("results") or []
        matched = [r for r in results if (r.get("content_match") or {}).get("status") in ("source", "partial")]
        notes.append(f"search {i} ({req.get('input_type')} → {req.get('output_type')}, query {_query_text(req)!r}): "
                     f"{len(results)} results"
                     + (f", stop_reason {details.get('stop_reason')}" if details.get("stop_reason") not in (None, "target_reached") else "")
                     + (f"; picture matched in {', '.join(str(r.get('candidate_id')) for r in matched)}" if matched
                        else "; no result's picture matched"))
    if found:
        notes.append(f"candidate {found['candidate']} ({found['url']}) is the video whose picture matched; its page E is "
                     f"{found['page']} and compare_media used it as B. Identify it as the source video found: its title and "
                     f"address citing {found['page']}, the picture match citing the comparison result; whether it is the earliest "
                     f"upload is a limitation, not a reason to leave the source unidentified")
    else:
        notes.append("no searched video's picture matched the material; the original source was not found, so changes relative "
                     "to an original cannot be stated; report only what the material itself shows")
    return notes


def _decide_find_original(state: HiState, memory: dict[str, Any], video: str) -> Decision:
    """A 族：OCR / ASR 取线索 → 文字查询（S1 槽写查询词）≤2 条 → 以图搜一次 → 画面对上的候选用 compare_media 比对
    （之后与 B 族两段比对同一套规则）→ 报告；都没对上 → INCOMPLETE。"""

    issued = _issued(memory)
    searches = _search_obs(state)
    found = memory.get("found")
    if not found:
        best = None
        for page in _search_pages(searches):
            cm = page["content_match"]
            if cm.get("status") in ("source", "partial") and (cm.get("material") in (None, video)):
                score = float(cm.get("fraction") or 0.0) + (1.0 if cm.get("status") == "source" else 0.0)
                if best is None or score > best[0]:
                    best = (score, page)
        n_queries = len([s for s in _query_searches(searches) if not _provider_error(s)])
        if best and float(best[1]["content_match"].get("fraction") or 0.0) < _FULL_MATCH and n_queries < _MAX_TEXT_QUERIES \
                and state.remaining >= 6 and "partial_more_search" not in issued:
            # 只对上一部分（wk9 3GwQdrhp__k：另一个上传只对上 34%，原片本身没搜到）：先再搜一次，可能有完整的原片
            _mark(memory, "partial_more_search")
            best = None
        if best:
            page = best[1]
            found = {"candidate": page["candidate"], "url": page["url"], "page": page["evidence_id"],
                     "status": page["content_match"].get("status"), "fraction": page["content_match"].get("fraction")}
            memory["found"] = found
    if found:
        decision = _decide_pair_merged(state, memory, [video, found["page"]])
        if decision.kind == "report":
            # 找到并比对完原片：交报告前查一次原片的发布日期（上传日期 / 最早存档），回答"谁先发布"（10-13）
            if (_dates_allowed(state) and found.get("candidate") and "dates_found" not in issued
                    and state.remaining >= 2):
                _mark(memory, "dates_found")
                return _dates_action(found["candidate"], "A6_dates_of_original",
                                     f"{found['candidate']} matches the material; check when it was uploaded or first archived.")
            dated = _dated(searches)
            notes = _search_notes(searches, found) + list(decision.slot.get("notes") or [])
            if dated:
                notes.append("date lookups: " + "; ".join(f"{k}: {v}" for k, v in dated.items())
                             + " — an upload or archive date says when that copy was online, not that no earlier copy exists")
            decision.slot["notes"] = notes
        return decision

    def not_found(rule: str, last_turn: bool = False) -> Decision:
        reason = ("original source not found: no searched video's picture matched Material A"
                  + (f" ({len(searches)} searches)" if searches else " (no search was made)")
                  + "; differences from an original cannot be determined")
        if last_turn:
            reason = "turn budget exhausted: " + reason
        return _report(state, memory, rule, status="INCOMPLETE", reason=reason,
                       notes=_search_notes(searches, None) + _ocr_notes(state),
                       branch="report_last_turn" if last_turn else "report_incomplete")

    if state.remaining <= 0:
        return not_found("G1_last_turn_report", last_turn=True)
    prof = state.profile(video)
    # A1 画面文字（台标、字幕、标题条）：查询词的主要线索
    if "ocr" in state.inspect_ops and "ocr_material" not in issued and state.remaining >= 3:
        _mark(memory, "ocr_material")
        return Decision("action", "A1_ocr_material", _inspect([video], "ocr", {}, _existing(state, state.attached, state.limit),
                        f"Read the on-screen text of {video} (captions, logos, name straps) to get searchable clues for its original."),
                        branch="INSPECT ocr(video)")
    # A2 语音：说出的人名、地名、事件
    if "asr" in state.inspect_ops and (prof.get("audio") or 0) > 0 and "asr_A" not in issued and state.remaining >= 3:
        _mark(memory, "asr_A")
        return Decision("action", "A2_asr_material", _inspect([video], "asr", {}, _existing(state, state.attached, state.limit),
                        f"Transcribe {video}'s speech; names, places and quoted words are clues for finding the original."),
                        branch="INSPECT asr")
    queries = _query_searches(searches)
    # 检索服务出错的查询最多补 1 次，不占名额
    errored = [s for s in queries if _provider_error(s)][:1]
    queries = [s for s in queries if s not in errored]
    text_searches = [s for s in queries if (s.get("request") or {}).get("input_type") == "text"]
    image_searches = [s for s in queries if (s.get("request") or {}).get("input_type") in ("image", "text_image")]
    # A3 查询（S1 槽按可见线索写查询词）：第 1 条纯文字；第 2 条在能以图搜时用"文字 + 中段一帧"（text_image 同时跑文字搜索与以图搜图）
    if len(queries) < _MAX_TEXT_QUERIES and state.remaining >= 2:
        previous = [{"query": _query_text(s.get("request") or {}),
                     "outcome": "no result" if _search_failed(s) else "results, but no result's picture matched the material"}
                    for s in queries]
        frames = _visible_frames(state, video)
        pair = bool(text_searches) and not image_searches and _image_search_allowed(state) and bool(frames)
        slot = {"name": "S1_query", "purpose": "original", "input_type": "text_image" if pair else "text",
                "output_type": "video", "previous": previous, "focus": _existing(state, state.attached, 2)}
        if pair:
            slot["evidence_id"] = frames[len(frames) // 2]
        return Decision("query_slot", "A3_search_original_text_image" if pair else "A3_search_original_text",
                        {"action": "SEARCH"}, branch="SEARCH text_image video" if pair else "SEARCH text video", slot=slot)
    # A4 以图搜：取材料中段的一帧
    if not image_searches and _image_search_allowed(state) and state.remaining >= 2:
        frames = _visible_frames(state, video)
        if frames:
            pick = frames[len(frames) // 2]
            _mark(memory, "image_search")
            return Decision("action", "A4_search_original_image", {
                "action": "SEARCH", "arguments": {"input_type": "image", "query": pick, "output_type": "video"},
                "focus_evidence_ids": [pick],
                "rationale": f"Text queries found no video whose picture matches {video}; search with the frame {pick} itself."},
                branch="SEARCH image video")
    return not_found("A5_original_not_found")


def _material_images(state: HiState) -> list[str]:
    mats = state.view.get("materials") or {}
    ids = [card.get("evidence_id") for card in mats.values() if isinstance(card, dict)
           and "image" in (card.get("modalities") or []) and not card.get("derived_from")]
    return sorted((i for i in ids if i), key=lambda e: int(e[1:]))


_NO_DATE_HOSTS = ("pinterest.", "instagram.com", "facebook.com", "vk.com", "vk.ru", "reddit.com", "x.com", "twitter.com",
                  "tiktok.com", "threads.net", "t.me", "shazam.com", "soundcloud.com")
_ISO_DAY = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _social(url: Any) -> bool:
    host = str(url or "").split("/")[2].lower() if "://" in str(url or "") else ""
    return any(h in host for h in _NO_DATE_HOSTS)


def _earliest_date(line: str) -> str | None:
    days = _ISO_DAY.findall(str(line or ""))
    return min(days) if days else None


# ---- r1（10-17）E 族："网页上的图是不是同一张"必须比对。r0g / r1v 的 E 可回答报告全被核验判"没确认网页上的图就是给定那张"：
# 以前以图搜只取页面文字，walker 读页、查日期，从不把页面上的图拿来与 E1 比。现在以图搜用 output_type=image（每个结果带
# 这个网页上的那张图 + 页面 + 缩略图），读过的页面就用 compare_media 图片模式把它的图与 E1 比对（对齐后的逐像素差异 + 三栏
# 对照图），比对结果是证据记录；没比对过的页面 S4 只能写"候选"。
_MAX_IMAGE_COMPARES = 3
_MAX_L_COMPARES = 2       # r1（负责人定）：L 族每个工作区比对 ≤ 2 次（含取页内主图），给文字搜和读页留回合
_SAME_AREA = 0.02          # 对齐后差异面积 ≤ 2%：同一张图，没有可见的像素差异
_EDITED_AREA = 0.35        # ≤ 35%：同一张图，有局部差异（裁切 / 加框 / 贴字 / 局部改动，差异框指出在哪）；更大：不是同一张
_AREA = re.compile(r"(?:changed area|measured)\s+([\d.]+)\s*%", re.I)


def _compare_class(obs: dict[str, Any]) -> tuple[str, float | None]:
    """compare_media 图片对的结果 → same / edited / different / failed，及差异面积（0–1）。"""

    if obs.get("status") not in (None, "ready"):
        return "failed", None
    result = obs.get("result") or {}
    comparison = (result.get("frame_diff") or {}).get("image_comparison") or result.get("image_comparison") or {}
    area = _num(comparison.get("changed_area"))
    if area is None:
        m = _AREA.search(" ".join(str(x) for x in result.get("summary_lines") or []) + " " + str(result.get("summary") or ""))
        area = float(m.group(1)) / 100 if m else None
    if area is None:
        return ("failed", None) if obs.get("status") != "ready" else ("unknown", None)
    registration = (result.get("frame_diff") or {}).get("registration") or {}
    if registration.get("requested") and registration.get("applied") is False and area > 0.10:
        return "different", area          # 特征点对不上（配准没做成）、差异又大：不是同一张
    return ("same" if area <= _SAME_AREA else "edited" if area <= _EDITED_AREA else "different"), area


def image_compares(state: HiState, image: str | None) -> dict[str, dict[str, Any]]:
    """给定图片与别的图做过的 compare_media（图片对）：对方 E → {class, area, record, composite, summary}。"""

    out: dict[str, dict[str, Any]] = {}
    if not image:
        return out
    for obs in state.observations:
        targets = [str(t) for t in obs.get("targets") or []]
        if obs.get("operation") != "compare_media" or len(targets) != 2 or image not in targets:
            continue
        other = targets[1] if targets[0] == image else targets[0]
        cls, area = _compare_class(obs)
        created = [str(e) for e in obs.get("created") or [] if e != obs.get("result_evidence_id")]
        composite = [e for e in created if "image" in ((state.cards.get(e) or {}).get("modalities") or ["image"])]
        summary = " ".join(str(x) for x in (obs.get("result") or {}).get("summary_lines") or []) or str((obs.get("result") or {}).get("summary") or "")
        out[other] = {"class": cls, "area": area, "record": obs.get("result_evidence_id"), "composite": composite[:1],
                      "summary": summary[:300]}
    return out


_CLASS_TEXT = {"same": "the same picture, no pixel difference above the noise threshold after alignment",
               "edited": "the same picture after alignment, with pixel differences over part of it — look at the comparison "
                         "image: differences spread over the whole picture come from resizing or recompression, a compact box "
                         "from a crop, a border, added text or a local edit; say which you see and where",
               "different": "not the same picture (most of the aligned picture differs)",
               "failed": "the comparison could not be made", "unknown": "no measured result in the state"}


def _compare_page_step(state: HiState, memory: dict[str, Any], image: str | None, pages: list[dict[str, Any]],
                       read_ids: set[str], compares: dict[str, dict[str, Any]],
                       dated_pages: list[tuple[str, dict[str, Any]]], cap: int = 3, family: str = "E") -> Decision | None:
    """E6 / L6：读过、带这个网页上那张图（或缩略图）、还没比过的页面 → compare_media(E1, 那张图)。有日期的页面按日期先比，
    每个工作区最多比 cap 次（E 3、L 2）。r1（负责人定 (a)）：读过的页面没有图（文字搜到的）→ 先 SEARCH source=R
    output_type=image 取页内主图（E6a / L6a），下一轮再比；打开也算进 cap。"""

    if not image or "compare_media" not in state.inspect_ops or state.remaining < 2:
        return None
    issued = _issued(memory)
    done = {k for k in compares} | {k[4:] for k in issued if k.startswith("cmp_")}
    opened = {k[5:] for k in issued if k.startswith("pimg_")}
    order = [pg["evidence_id"] for _, pg in dated_pages]

    def rank(p: dict[str, Any]) -> tuple[bool, bool, int]:
        # r1：特意取了页内主图的页面（文字搜到、写着拍摄地 / 出处的那页）先比——r1s2 l623 取到了 BuzzFeed 的图，比对名额却先给了
        # 以图搜的副本，BuzzFeed 那张没比上
        return (p.get("candidate") not in opened, p["evidence_id"] not in order,
                order.index(p["evidence_id"]) if p["evidence_id"] in order else 0)

    todo = sorted((p for p in pages if p["evidence_id"] in read_ids and p.get("image") and p["image"] not in done
                   and p["image"] != image), key=rank)
    if len(done) >= cap:
        return None
    if not todo:
        pending = [p for p in pages if p.get("candidate") in opened and not p.get("image")]
        need = sorted((p for p in pages if p["evidence_id"] in read_ids and not p.get("image") and p.get("candidate")
                       and p["candidate"] not in opened and not _social(p.get("url"))), key=rank)
        if not need or len(done) + len(pending) >= cap or state.remaining < 4 or "SEARCH" not in state.actions:
            return None
        page = need[0]
        _mark(memory, f"pimg_{page['candidate']}")
        return Decision("action", f"{family}6a_open_page_image", {"action": "SEARCH", "arguments": {
            "source": page["candidate"], "output_type": "image"}, "focus_evidence_ids": [],
            "rationale": f"{page['evidence_id']} was read; fetch the main picture of that page ({page['candidate']}) to compare it "
                         f"with {image}."}, branch="SEARCH open image")
    page = todo[0]
    _mark(memory, f"cmp_{page['image']}")
    what = "thumbnail" if page.get("image_kind") == "thumbnail" else "copy of the picture"
    return Decision("action", "E6_compare_page_image", _inspect([image, page["image"]], "compare_media", state.compare_params(),
                    [image, page["image"]],
                    f"{page['evidence_id']} was read; compare its {what} {page['image']} with {image} to establish whether the "
                    f"page carries the same picture and whether it was altered."), branch="INSPECT compare_media(image)")


def _image_match_notes(state: HiState, image: str | None, pages: list[dict[str, Any]], read_ids: set[str],
                       compares: dict[str, dict[str, Any]]) -> list[str]:
    """S4 的说明：每个读过的页面，它的图与 E1 比对了没有、结果如何；"最早来源"只给比对过、对上了的页面。"""

    if not image:
        return []
    lines = []
    for p in pages:
        if p["evidence_id"] not in read_ids:
            continue
        c = compares.get(str(p.get("image"))) if p.get("image") else None
        if c:
            area = f", changed area {c['area']:.1%}" if c.get("area") is not None else ""
            lines.append(f"{p['evidence_id']} ({p.get('candidate')}): its {'thumbnail' if p.get('image_kind') == 'thumbnail' else 'picture'} "
                         f"{p['image']} was compared with {image} by compare_media (record {c.get('record') or '?'}"
                         f"{', comparison image ' + c['composite'][0] if c.get('composite') else ''}): {_CLASS_TEXT[c['class']]}{area}")
        else:
            lines.append(f"{p['evidence_id']} ({p.get('candidate')}): its picture was not compared with {image}"
                         + ("" if p.get("image") else " (no copy of the picture was obtained from it)"))
    if not lines:
        return []
    return ["picture comparisons for the pages that were read: " + "; ".join(lines),
            "call a page a copy of the given picture, its earliest copy or its original source only when its picture was "
            "compared with the given picture and matched; a page whose picture was not compared, or did not match, is a "
            "candidate only — write 'candidate' and say why it is not confirmed",
            "a near-identical match or a small changed area does not show the copy is unedited: state where compare_media "
            "found local differences; a thumbnail hash 'no match' can come from cropping, a border or a local edit and does "
            "not show that the page lacks the picture",
            "write one finding that states the outcome of each comparison (same picture / same picture with local differences "
            "/ not the same picture / could not be compared, and why), citing the compare_media record"]


def _compare_required(image: str | None, pages: list[dict[str, Any]], read_ids: set[str],
                      compares: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """S4 合同校验：每条比对记录都要被某个 Finding 引用并写出结论（oracle_generator.s4_contract_problems 用）。"""

    out = []
    for p in pages:
        c = compares.get(str(p.get("image"))) if p.get("image") and p["evidence_id"] in read_ids else None
        if c and c.get("record"):
            out.append({"ids": [c["record"]], "words": r"same|match|differ|compar|identical|alter|edit|crop|border",
                        "message": f"no finding cites the compare_media record {c['record']} ({image} vs the picture on "
                                   f"{p['evidence_id']}); write one finding stating that comparison's outcome and cite {c['record']}"})
    return out


def _view_compare_before_report(state: HiState, memory: dict[str, Any], image: str | None, pages: list[dict[str, Any]],
                                read_ids: set[str], compares: dict[str, dict[str, Any]]) -> Decision | None:
    """E7（同 L3b）：要交报告时比对过的图与三栏对照图不在附图里 → 先读一次对照图（E1 与那张图作焦点），下一轮交报告。
    每个工作区一次（按历史里有没有读过对照图判断）。"""

    if not image or state.remaining < 2:
        return None
    shown = []
    for p in pages:
        c = compares.get(str(p.get("image"))) if p.get("image") and p["evidence_id"] in read_ids else None
        if c and c.get("composite") and c["class"] in ("same", "edited", "different"):
            shown.append((p["image"], c["composite"][0]))
    if not shown:
        return None
    viewed = any(h.get("action") == "READ_RECORD" and (h.get("request") or {}).get("record_id") in {c for _, c in shown}
                 for h in state.view.get("history") or [])
    attached = all(e in state.attached for pair in shown[:2] for e in pair)
    if viewed or attached:
        return None
    focus = list(dict.fromkeys([image] + [e for pair in shown[:2] for e in pair]))[:state.limit]
    memory["review_record"] = shown[0][1]
    return _substituted(Decision("action", "E7_view_comparison_before_report", _read(shown[0][1], focus,
                        f"Before reporting, view the comparison image {shown[0][1]} with {image} and the page's picture "
                        f"{shown[0][0]} to state whether they are the same picture."), branch="READ_RECORD focus"))


def just_reviewed(state: HiState, memory: dict[str, Any]) -> bool:
    """上一步就是交报告前的补看（E7 / C9v）：这一轮直接交报告，别让 S3 再选读页（附图会被换掉）。"""

    history = state.view.get("history") or []
    record = memory.get("review_record")
    return bool(record and history and history[-1].get("action") == "READ_RECORD"
                and (history[-1].get("request") or {}).get("record_id") == record)


def _decide_image_origin(state: HiState, memory: dict[str, Any]) -> Decision:
    """E 族（10-13，AVerImaTeC 类"图片最早在哪里、何时发布"）：E1 读图里的文字 → E2 以图搜（反向图片，结果是页面）
    → E3 对排在前面的 3 个结果查日期（页面声明 / 上传 / 最早存档）→ 之后 S3 槽在"读一页 / 再查一个日期 / 再搜一条 /
    交报告"里选。报告写能确认的最早一份及其日期（只能说"不晚于"），再对照声明的场合。"""

    issued = _issued(memory)
    images = _material_images(state)
    searches = _search_obs(state)
    queries = _query_searches(searches)
    dated = _dated(searches)
    pages = _search_pages(searches)
    reads = _read_records(state)
    read_ids = {str((r.get("request") or {}).get("record_id")) for r in reads}

    by_cand = {p.get("candidate"): p for p in pages if p.get("candidate")}
    dated_found = {c: _earliest_date(v) for c, v in dated.items() if _earliest_date(v)}
    # 有日期的结果按日期排（最早在前），读它的页面看它说图上是什么
    dated_pages = sorted(((d, by_cand[c]) for c, d in dated_found.items() if c in by_cand), key=lambda x: x[0])
    # 10-15 对话 5''（追加 5）：找到最早副本（读过的有日期页里最早的那个）之后，比它晚的转载最多再读 1 页
    m_claim = re.search(r'claim:\s*"(.*)"\s*\(claim date', state.task, re.S)
    claim = m_claim.group(1) if m_claim else state.task.rsplit(":", 1)[-1]
    claim_date = _claim_date(state.task)
    read_dates = [d for d, pg in dated_pages if pg["evidence_id"] in read_ids]
    later_ids: set[str] = set()
    if read_dates:
        earliest = min(read_dates)
        later = {pg["evidence_id"] for d, pg in dated_pages if d > earliest}
        if len(later & read_ids) >= 1:
            later_ids = later - read_ids
    # 追加 2(a)：还有未读的命中页（标题 / 摘要命中声明，或日期 ≤ 声明日期）时不许交报告
    must = [p for p in _must_read(pages, read_ids, claim, dated=dated_found, claim_date=claim_date)
            if p["evidence_id"] not in later_ids]
    image = images[0] if images else None
    compares = image_compares(state, image)

    def match_of(page: dict[str, Any]) -> dict[str, Any] | None:
        return compares.get(str(page.get("image"))) if page.get("image") else None

    def report(rule: str, last_turn: bool = False) -> Decision:
        read_dated = [pg for _, pg in dated_pages if pg["evidence_id"] in read_ids]
        # r1：只有画面与 E1 比对过、且是同一张（含局部差异）的有日期页面，才能当"最早副本"写 COMPLETED
        matched = [pg for pg in read_dated if (match_of(pg) or {}).get("class") in ("same", "edited")]
        if matched:
            status, reason = "COMPLETED", None
        elif read_dated:
            names = ", ".join(pg["evidence_id"] for pg in read_dated[:3])
            if any(match_of(pg) for pg in read_dated):
                reason = (f"the pictures on the dated pages that were read ({names}) were not shown by compare_media to be the "
                          f"given picture {image}, so none of them is established as a copy and the earliest publication is not "
                          f"established")
            else:
                reason = (f"the pictures on the dated pages that were read ({names}) were not compared with the given picture "
                          f"{image} (no copy of the picture was obtained from them), so they remain candidates and the earliest "
                          f"publication is not established")
            status = "INCOMPLETE"
        elif dated_found:
            status, reason = "INCOMPLETE", ("dated copies were found but none of their pages was read, so what they show "
                                            "and say is not established")
        else:
            status, reason = "INCOMPLETE", ("no date of any copy of the image was established"
                                            if queries else "no search was made")
        if status == "INCOMPLETE" and _open_gap_clause(must):
            reason = f"{reason}; {_open_gap_clause(must)}"
        if last_turn and status == "INCOMPLETE":
            reason = "turn budget exhausted: " + str(reason)
        notes = ["date lookups: " + ("; ".join(f"{k}: {v}" for k, v in dated.items()) or "none"),
                 "dated results and their pages (earliest first): " + ("; ".join(
                     f"{pg.get('candidate')} {d} page {pg['evidence_id']} ({'read' if pg['evidence_id'] in read_ids else 'not read'})"
                     for d, pg in dated_pages) or "none"),
                 "cite the date record (the E named in the dates line) for a date and what kind of date it is; cite the page "
                 "for what that page says the picture shows; a page that was not read supports nothing beyond its title",
                 "pages read: " + (", ".join(sorted(read_ids)) or "none"),
                 "the earliest date found bounds the image's first publication from above ('no later than'); a page's own "
                 "date may be later than the picture's first appearance, and an archive capture only shows it was online "
                 "by then; name the earliest copy found, its date and what that copy says the picture shows, then compare "
                 "that with the claim's place, time and event"] + _image_match_notes(state, image, pages, read_ids, compares) \
            + _ocr_notes(state)
        decision = _report(state, memory, rule, status=status, reason=reason, notes=notes,
                           branch="report_last_turn" if last_turn else ("report_completed" if status == "COMPLETED" else "report_incomplete"))
        decision.slot["required_findings"] = _compare_required(image, pages, read_ids, compares)
        return decision

    def report_now(rule: str) -> Decision:
        return _view_compare_before_report(state, memory, image, pages, read_ids, compares) or report(rule)

    if state.remaining <= 0:
        return report("G1_last_turn_report", last_turn=True)
    # E1 图里的文字（水印、字幕、标牌）
    if image and "ocr" in state.inspect_ops and "ocr_image" not in issued and state.remaining >= 4:
        _mark(memory, "ocr_image")
        return Decision("action", "E1_ocr_image", _inspect([image], "ocr", {}, [image],
                        f"Read any writing in {image} (watermark, caption, signs); it can identify where the picture comes from."),
                        branch="INSPECT ocr(image)")
    # E2 以图搜：找出现过这张图的网页。r1：output_type=image——每个结果另带这个网页上的那张图（之后与 E1 比对）
    if not queries and image and _image_search_allowed(state) and state.remaining >= 3:
        return Decision("action", "E2_reverse_image", {"action": "SEARCH", "arguments": {
            "input_type": "image", "query": image, "output_type": "image"}, "focus_evidence_ids": [image],
            "rationale": f"Find web pages that carry the same picture as {image}, with each page's copy of the picture; their "
                         f"dates bound when it first appeared."},
            branch="SEARCH image image")
    # E3 前 3 个结果查日期（社交 / 图片收藏站不自动查：页面不声明日期，存档也少；wk8 查了 15 次全是 no date found）。
    # r1：缩略图哈希判 no match 的排后（不排除：e107 的真来源排第 1 却被判 no match——加框贴旗的副本哈希对不上）
    if _dates_allowed(state) and len(dated) < 3 and state.remaining >= 3:
        for page in sorted(pages, key=lambda p: p.get("verified") == "different"):
            cand = page.get("candidate")
            if cand and cand not in dated and not _social(page.get("url")):
                return _dates_action(cand, "E3_dates_top_results",
                                     f"{cand} carries a matching picture; check when that page was published or first archived.")
    step = _pdf_pages_step(state, memory, read_ids)
    if step is not None and state.remaining >= 2:
        return step
    # E6（r1）读过的页面：把它上面那张图与 E1 比对（compare_media 图片模式：对齐后逐像素差异 + 三栏对照图）。有日期的先比
    if just_reviewed(state, memory):
        return report("E9_report")
    step = _compare_page_step(state, memory, image, pages, read_ids, compares, dated_pages)
    if step is not None:
        return step
    # E5 交报告前先读最早的有日期页面（wk8 averimatec_e2：找到 2016-06-12 的两页却一页没读，报告只能写"未确定"）
    first_dated = next((pg for _, pg in dated_pages if pg["evidence_id"] not in read_ids), None)
    if first_dated and not any(pg["evidence_id"] in read_ids for _, pg in dated_pages) and state.remaining >= 2 \
            and f"read_dated_{first_dated['evidence_id']}" not in issued:
        _mark(memory, f"read_dated_{first_dated['evidence_id']}")
        return Decision("action", "E5_read_earliest_dated", {"action": "READ_RECORD", "arguments": {
            "record_id": first_dated["evidence_id"]}, "focus_evidence_ids": [],
            "rationale": f"{first_dated.get('candidate')} is the earliest dated result; read its page {first_dated['evidence_id']} "
                         f"to see what it says the picture shows."}, branch="READ_RECORD text")
    dated_ids = [pg["evidence_id"] for _, pg in dated_pages]
    # 追加 5：日期 ≤ 声明日期的页优先，其后按日期先后；最早副本找到后，后期转载只留 1 页的读额度
    early_ids = {pg["evidence_id"] for d, pg in dated_pages if claim_date and d <= claim_date}
    # r1：缩略图哈希 near-identical 的先读、no match 的最后读（降权，不排除）
    unread = sorted((p for p in pages if p["evidence_id"] not in read_ids and p["evidence_id"] not in later_ids),
                    key=lambda p: (p["evidence_id"] not in early_ids, p["evidence_id"] not in dated_ids,
                                   {"same": 0, "different": 2}.get(str(p.get("verified")), 1),
                                   dated_ids.index(p["evidence_id"]) if p["evidence_id"] in dated_ids else 0))
    undated = [p for p in pages if p.get("candidate") and p["candidate"] not in dated and not _social(p.get("url"))]
    choices: list[str] = []
    if len(queries) < _MAX_CLAIM_SEARCHES:
        choices.append("search")
    if unread and len(reads) < 4:
        choices.append("read")
    if _dates_allowed(state) and undated and len(dated) < _MAX_DATE_LOOKUPS:
        choices.append("dates")
    if dated or reads:
        choices.append("report")
    if must and state.remaining >= 2 and len(reads) < 6:
        # 追加 2(a)：命中页没读完不许交报告；读额度为它们放宽 2 页；命中页排在未读清单最前
        choices = [c for c in choices if c != "report"]
        if "read" not in choices:
            choices.append("read")
        must_ids = [p["evidence_id"] for p in must]
        unread = [p for p in must] + [p for p in unread if p["evidence_id"] not in must_ids]
    if state.remaining <= 1 or not choices or choices == ["report"]:
        return report_now("E9_report")
    if state.remaining == 2:
        # 只剩两轮：比对结果的三栏图不在附图里时，这一轮先补看、下一轮交报告（同 L3b）
        step = _view_compare_before_report(state, memory, image, pages, read_ids, compares)
        if step is not None:
            return step
    return Decision("next_slot", "E4_next_step", {}, branch="S3",
                    slot={"name": "S3_next", "choices": choices,
                          "unread": [{k: p.get(k) for k in ("evidence_id", "title", "url")} for p in unread[:8]],
                          "undated": [{"candidate": p["candidate"], "title": p.get("title"), "url": p.get("url")}
                                      for p in undated[:8]],
                          "offsets": {}, "read": sorted(read_ids),
                          "previous": [_query_text(s.get("request") or {}) for s in queries],
                          "claim": state.task.rsplit(":", 1)[-1].strip()[:400],
                          "report_decision": (report_now if "report" in choices else report)("E9_report").to_dict()})


def _search_images(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """文字搜图片的结果里已取得的图片 E（按出现顺序，去重）。"""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for search in searches:
        if (search.get("request") or {}).get("output_type") != "image":
            continue
        for result in (search.get("details") or {}).get("results") or []:
            for material in result.get("materials") or []:
                if material.get("kind") != "image" or material.get("status") not in ("ready", "partial"):
                    continue
                for card in material.get("evidence") or []:
                    eid = card.get("evidence_id")
                    if eid and eid not in seen:
                        seen.add(eid)
                        out.append({"evidence_id": eid, "candidate": result.get("candidate_id"), "url": result.get("source_url"),
                                    "title": result.get("search_title")})
    return out


def ranked_refs(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """r1（10-17）：参考图按标题 / 网址与参考图查询词的重合排序，有相关的就去掉完全无关的。r1s l623：查询
    "U.S. Capitol building interior" 的前 3 张是拉链零件、字母 U 剪贴画（E4 / E7 / E11），"State Capitol Buildings Interior"
    排在后面，L3 拿无关图去比，S4 只能写"参考图里没有建筑特征"。"""

    out: list[dict[str, Any]] = []
    for search in searches:
        req = search.get("request") or {}
        if req.get("output_type") != "image" or req.get("input_type") != "text":
            continue
        words = _claim_words(_query_text(req))
        for ref in _search_images([search]):
            score = len(words & _claim_words(f"{ref.get('title') or ''} {_page_text(ref)}"))
            out.append({**ref, "score": score})
    if any(r["score"] > 0 for r in out):
        out = [r for r in out if r["score"] > 0]
    seen: set[str] = set()
    ranked = []
    for r in sorted(out, key=lambda r: -r["score"]):
        if r["evidence_id"] not in seen:
            seen.add(r["evidence_id"])
            ranked.append(r)
    return ranked


_L_VERDICT_WORDS =r"same place|different place|match|differ|consistent|inconsistent|cannot tell|can't tell|not the same|resembl"
_L_CANNOT_WORDS = r"could not be compared|cannot be compared|not compared|no reference|could not be confirmed|not be confirmed|" \
                  r"comparison"


def _decide_place_check(state: HiState, memory: dict[str, Any]) -> Decision:
    """L 族（10-13，AVerImaTeC "Where was this image taken?"）：给定照片随声明流传，声明说它拍的是某个地点 / 地标。
    L1 读图里的文字 → L2 以文字搜图片（S1 槽写声明所说地点的名字）取参考照片 → L3 参考照片与给定照片并排看
    → L4 以图搜（反向图片）找载有这张图的网页、看它们说拍于哪里 → S3 槽（读页 / 再搜 / 交报告）。
    以文字搜图片在这里有确定的用途：拿到声明所说地点的照片，才能当面比对给定照片是不是那里。"""

    issued = _issued(memory)
    images = _material_images(state)
    image = images[0]
    searches = _search_obs(state)
    queries = _query_searches(searches)
    ref_searches = [s for s in queries if (s.get("request") or {}).get("output_type") == "image"]
    reverse = [s for s in queries if (s.get("request") or {}).get("input_type") == "image"]
    refs = ranked_refs(searches)          # r1：相关的参考图在前，无关的去掉
    pages = _search_pages(searches)
    reads = _read_records(state)
    read_ids = {str((r.get("request") or {}).get("record_id")) for r in reads}
    m = re.search(r'claim:\s*"(.*)"\s*\(claim date', state.task, re.S)
    claim = m.group(1) if m else state.task.rsplit(":", 1)[-1].strip()
    # 10-15 对话 5''（追加 2）：未读命中页；参考图与给定照片的并排比对是交报告前的必经动作
    must = _must_read(pages, read_ids, claim)
    # r0+（10-16，L 族 0 份进 accepted 的排查）：以图搜到的页面载有这张图，是"来源说它拍于哪里"（REQ2）的直接证据——r0g l208
    # 以图搜的 geologyscience 页（写着 Peru）没读，S3 去读了声明地点 Joshimath 的新闻。前 2 个没读的以图搜结果页先读
    reverse_pages = [p for p in _search_pages(reverse) if not _social(p.get("url"))]
    read_reverse = [p for p in reverse_pages if p["evidence_id"] in read_ids]
    # r1（负责人定 (c)）：以图搜之后必做的那条文字搜（L4b）——它的第 1 个结果页也必读；有它时以图搜结果页只必读 1 个
    place_searches = [s for s in queries if (s.get("request") or {}).get("input_type") == "text"
                      and (s.get("request") or {}).get("output_type") == "text"]
    place_pages = [p for p in _search_pages(place_searches[:1]) if not _social(p.get("url"))]
    quota = 1 if place_pages else 2
    if len(read_reverse) < quota:
        first = [{**p, "why": "it carries the picture (reverse image search)"} for p in reverse_pages
                 if p["evidence_id"] not in read_ids][:quota - len(read_reverse)]
        must = first + [p for p in must if p["evidence_id"] not in {f["evidence_id"] for f in first}]
    if place_pages and not any(p["evidence_id"] in read_ids for p in place_pages):
        top = {**place_pages[0], "why": "top result of the text search for where the picture was taken"}
        must = [m for m in must if m["evidence_id"] != top["evidence_id"]]
        must.insert(len([m for m in must if "reverse image search" in str(m.get("why"))]), top)
    side = _existing(state, [r["evidence_id"] for r in refs], max(1, min(3, state.limit - 1)))
    refs_shown = bool(side) and all(e in state.attached for e in [image, *side])
    compares = image_compares(state, image)

    def report(rule: str, last_turn: bool = False) -> Decision:
        compared = "compare_refs" in issued
        if compared and (reads or pages):
            status, reason = "COMPLETED", None
        elif not refs:
            status, reason = "INCOMPLETE", ("no reference photo of the place named in the claim was found"
                                            if ref_searches else "no reference photos were searched")
        elif not compared:
            status, reason = "INCOMPLETE", (f"the reference photos {', '.join(side or [r['evidence_id'] for r in refs[:3]])} of "
                                            f"the claimed place were not viewed side by side with {image}")
        else:
            status, reason = "INCOMPLETE", "no page that carries the picture was read, so where it was actually taken is not established"
        if status == "INCOMPLETE" and _open_gap_clause(must):
            reason = f"{reason}; {_open_gap_clause(must)}"
        if last_turn and status == "INCOMPLETE":
            reason = "turn budget exhausted: " + str(reason)
        notes = [f"reference photos of the place named in the claim (text→image search): "
                 + (", ".join(f"{r['evidence_id']} ({str(r.get('title') or r.get('url') or '')[:60]})" for r in refs[:4]) or "none"),
                 "compare the given picture with the reference photos feature by feature (buildings, landmarks, terrain, "
                 "signs, layout) and say plainly whether they show the same place; a similar kind of scene is not the same place",
                 # r0+：r0g 的 4 份可回答 L 报告没有一条写比对结果（S4 交报告时看不到参考图，比对只在中间一步做过）
                 (f"{image} and the reference photos {', '.join(side)} are attached to this turn: write one finding that states "
                  f"the outcome of that side-by-side comparison — match (the same place), no match (a different place), or "
                  f"cannot tell (and why: angle, crop, too few distinctive features) — naming the features that match or "
                  f"differ, citing {image} and the reference photo IDs" if refs_shown else
                  (f"the reference photos {', '.join(side)} are not attached to this turn; do not describe what you cannot see: "
                   f"write one finding that says the comparison with them could not be confirmed in this report and why"
                   if side else
                   "no reference photo of the claimed place was obtained: write one finding that says the given picture could "
                   "not be compared with the claimed place and why (no reference photo was found)")),
                 "pages read: " + (", ".join(sorted(read_ids)) or "none"),
                 "say where the pages that carry the picture say it was taken, citing them; if they disagree or none says, "
                 "state that the place is not established"] + _image_match_notes(state, image, pages, read_ids, compares) \
            + _ocr_notes(state)
        decision = _report(state, memory, rule, status=status, reason=reason, notes=notes,
                           branch="report_last_turn" if last_turn else ("report_completed" if status == "COMPLETED" else "report_incomplete"))
        # r1：S4 报告必须写参考图比对的结论（匹配 / 不匹配 / 无法比对及原因），程序校验（s4_contract_problems）
        decision.slot["required_findings"] = [
            {"ids": list(side) if refs_shown else [], "words": _L_VERDICT_WORDS if refs_shown else _L_CANNOT_WORDS,
             "message": (f"no finding states the outcome of comparing {image} with the reference photos {', '.join(side)} "
                         f"(match / no match / cannot tell and why), citing them" if refs_shown else
                         "no finding says whether the given picture could be compared with the claimed place, and why not")}] \
            + _compare_required(image, pages, read_ids, compares)
        return decision

    if state.remaining <= 0:
        return report("G1_last_turn_report", last_turn=True)
    # L1 图里的文字（标牌、水印、字幕）
    if "ocr" in state.inspect_ops and "ocr_image" not in issued and state.remaining >= 5:
        _mark(memory, "ocr_image")
        return Decision("action", "L1_ocr_image", _inspect([image], "ocr", {}, [image],
                        f"Read any writing in {image} (signs, watermark, caption); it can show where the picture was taken."),
                        branch="INSPECT ocr(image)")
    # L2 以文字搜图片：声明所说地点的参考照片（S1 槽写查询词）
    if not ref_searches and state.remaining >= 4:
        return Decision("query_slot", "L2_reference_images", {"action": "SEARCH"}, branch="SEARCH text image",
                        slot={"name": "S1_query", "purpose": "reference_images", "input_type": "text", "output_type": "image",
                              "previous": [], "focus": [image], "claim": claim[:400]})
    # L3 参考照片与给定照片并排看。10-15 对话 5''：一定发一个真动作（以前两者已同时附上就只记一笔、不发动作——r0a l623
    # 因此从没并排看过，报告里也没有这一步）
    if side and "compare_refs" not in issued and state.remaining >= 3:
        _mark(memory, "compare_refs")
        return Decision("action", "L3_view_with_references", _read(side[0], [image, *side],
                        f"View {image} next to the reference photos {', '.join(side)} of the place the claim names, to compare "
                        f"landmarks and layout."), branch="READ_RECORD focus")
    # L4 以图搜：哪些网页载有这张图、它们说拍于哪里（第 2 轮跟进只补比对 / 读页时不再搜）
    if not reverse and _image_search_allowed(state) and state.remaining >= 3 and "skip_reverse" not in issued:
        # r1：output_type=image——每个结果另带这个网页上的那张图，读过的页面再与给定照片比对（同 E 族 E6）
        return Decision("action", "L4_reverse_image", {"action": "SEARCH", "arguments": {
            "input_type": "image", "query": image, "output_type": "image"}, "focus_evidence_ids": [image],
            "rationale": f"Find web pages that carry {image}, with each page's copy of it; they may say where it was actually taken."},
            branch="SEARCH image image")
    # L4b（r1，负责人定 (c)）：以图搜之后必做一次文字搜（OCR / 标题文字 + 地点线索），不交给 S3 选
    if reverse and not place_searches and "place_text" not in issued and state.remaining >= 5:
        _mark(memory, "place_text")
        return Decision("query_slot", "L4b_place_text_search", {"action": "SEARCH"}, branch="SEARCH text page",
                        slot={"name": "S1_query", "purpose": "place_text", "input_type": "text", "output_type": "text",
                              "previous": [_query_text(s.get("request") or {}) for s in ref_searches], "focus": [image],
                              "claim": claim[:400]})
    step = _pdf_pages_step(state, memory, read_ids)
    if step is not None and state.remaining >= 2:
        return step
    # L6（r1）：读过的以图搜页面——它上面那张图与给定照片比对，"这页说的拍摄地"才算这张照片的（r1s l623 第 1 轮核验：
    # "does not confirm that the photo in the article is the same image"）
    step = (_compare_page_step(state, memory, image, pages, read_ids, compares, [], cap=_MAX_L_COMPARES, family="L")
            if state.remaining >= 3 else None)
    if step is not None:
        if step.rule == "E6_compare_page_image":
            step.rule = "L6_compare_page_image"
        return step
    unread = [p for p in pages if p["evidence_id"] not in read_ids]
    text_queries = [s for s in queries if (s.get("request") or {}).get("output_type") == "text"
                    and (s.get("request") or {}).get("input_type") == "text"]
    choices: list[str] = []
    if len(text_queries) < 2:
        choices.append("search")
    if unread and len(reads) < 4:
        choices.append("read")
    if reads or (refs and "compare_refs" in issued):
        choices.append("report")
    if must and state.remaining >= 2 and len(reads) < 6:
        choices = [c for c in choices if c != "report"]
        if "read" not in choices:
            choices.append("read")
        must_ids = [p["evidence_id"] for p in must]
        unread = list(must) + [p for p in unread if p["evidence_id"] not in must_ids]
    # 追加 2(b)：要交报告时给定照片与参考图得在附图里并排（读页之后它们会掉出附图——r0a l101 第 4 步并排看过，第 11 步
    # 交报告时已看不到，报告反说 "reference photographs are not available"）。r0+（10-16）：以前在"可以交报告"刚成立时就补看
    # （r0g l208 / l623 第 6 步），之后又读了 5 页，交报告时参考图早已掉出附图。现在补看就是交报告前的最后一步：要交报告时
    # 先发 L3b，下一轮参考图在附图里就直接交报告
    # L3b 每个工作区只发一次：同记录同焦点的重复动作会被运行时合并、插一条中文 harness_event（r0g 第一遍 B8）。读给定照片
    # 本身只有 L3b 会做，所以按历史里有没有 READ_RECORD(给定照片) 判断
    reviewed = any(h.get("action") == "READ_RECORD" and (h.get("request") or {}).get("record_id") == image
                   for h in state.view.get("history") or [])
    if memory.get("l3b_offered") and reviewed and refs_shown and "compare_refs" in issued:
        return report("L9_report")

    def report_now(rule: str) -> Decision:
        return _l3b_before_report(state, memory, image, side, compared="compare_refs" in issued, refs_shown=refs_shown,
                                  reviewed=reviewed) or report(rule)

    if state.remaining <= 1 or not choices or choices == ["report"]:
        return report_now("L9_report")
    if state.remaining == 2:
        # 只剩两轮：这一轮补看参考图、下一轮交报告（r0h 冒烟 l623：读到第 11 步只剩 1 轮，交报告时参考图不在附图里）
        step = _l3b_before_report(state, memory, image, side, compared="compare_refs" in issued, refs_shown=refs_shown,
                                  reviewed=reviewed)
        if step is not None:
            return step
    return Decision("next_slot", "L5_next_step", {}, branch="S3",
                    slot={"name": "S3_next", "choices": choices,
                          "unread": [{k: p.get(k) for k in ("evidence_id", "title", "url")} for p in unread[:8]],
                          "undated": [], "offsets": {}, "read": sorted(read_ids),
                          "previous": [_query_text(s.get("request") or {}) for s in text_queries],
                          "claim": claim[:400],
                          "report_decision": (report_now if "report" in choices else report)("L9_report").to_dict()})


def _l3b_before_report(state: HiState, memory: dict[str, Any], image: str | None, side: list[str], *, compared: bool,
                       refs_shown: bool, reviewed: bool) -> Decision | None:
    """L 族要交报告时：参考图比对做过、但给定照片与参考图此刻不在附图里，且这个工作区还没补看过 → 先补看（L3b：读给定照片本身、
    参考图作焦点），下一轮参考图回到附图就交报告。S3 选"交报告"时换成这一步，标 substituted（理由由 S0 按实际动作写）。"""

    if not (image and side and compared) or refs_shown or reviewed or state.remaining < 2:
        return None
    memory["l3b_offered"] = 1
    return _substituted(Decision("action", "L3b_view_references_before_report", _read(image, [image, *side],
                        f"Before reporting, view {image} again next to the reference photos {', '.join(side)} to state "
                        f"whether they show the same place."), branch="READ_RECORD focus"))


_SEARCH_SCOPE = re.compile(r"Scope:[^\n]*(?:open and read https?://|next to reference photos|search for sources on the open points)", re.I)
_SCOPE_URL = re.compile(r"open and read (https?://[^\s;]+?)(?=[;]|\.\s|\.$|\s|$)", re.I)
_SCOPE_SEARCH = re.compile(r"search for sources on the open points", re.I)
_FOLLOWUP_READS = 2


def _decide_search_followup(state: HiState, memory: dict[str, Any], family: str) -> Decision:
    """检索类第 2 轮（追加 2）：第 1 轮报告写明了未读的命中页（网址）/ 没并排看的参考图，GIA 派的任务 Scope 里照列。
    这里只补这些：逐个网址 SEARCH source=网址 打开 → READ_RECORD 读；L 族要并排看参考图时走 L2（S1 写参考图查询）→ L3 并排看；
    然后交报告。不重走 OCR / 以图搜 / 日期查询。"""

    issued = _issued(memory)
    scope = state.task[state.task.rfind("Scope:"):]
    urls = [u.rstrip(".") for u in _SCOPE_URL.findall(scope)][:3]
    refs_wanted = family == "L" and bool(re.search(r"next to reference photos", scope, re.I))
    searches = _search_obs(state)
    reads = _read_records(state)
    read_ids = {str((r.get("request") or {}).get("record_id")) for r in reads}
    images = _material_images(state)
    image = images[0] if images else None
    opened: dict[str, str | None] = {}
    for search in searches:
        src = str((search.get("request") or {}).get("source") or "")
        if src in urls:
            pages = _search_pages([search])
            opened[src] = pages[0]["evidence_id"] if pages else None
    refs = ranked_refs(searches)          # r1：相关的参考图在前，无关的去掉
    side = _existing(state, [r["evidence_id"] for r in refs], max(1, min(3, state.limit - 1)))
    # r0+：Scope 写了"再搜来源"——S1 写一条文字查询（针对任务里列的未答点）→ 读命中声明最多的 2 页
    search_wanted = bool(_SCOPE_SEARCH.search(scope))
    claim = _claim_of(state.task)
    text_queries = [s for s in _query_searches(searches) if (s.get("request") or {}).get("input_type") == "text"
                    and (s.get("request") or {}).get("output_type") == "text" and not (s.get("request") or {}).get("source")]
    found = [p for p in _search_pages(text_queries) if not _social(p.get("url"))]
    found_read = [p for p in found if p["evidence_id"] in read_ids]

    def report(rule: str, last_turn: bool = False) -> Decision:
        unread = [u for u in urls if not opened.get(u) or opened[u] not in read_ids]
        missing = []
        if unread:
            missing.append("candidate pages that could not be opened or read: " + ", ".join(unread))
        if refs_wanted and "compare_refs" not in issued:
            missing.append("no reference photo of the claimed place was viewed side by side with the picture")
        if search_wanted and not found_read:
            missing.append("no page about the open points could be found and read" if text_queries else "no search was made")
        status = "INCOMPLETE" if missing and not (read_ids or "compare_refs" in issued) else "COMPLETED"
        reason = "; ".join(missing) if status == "INCOMPLETE" else None
        if last_turn and status == "INCOMPLETE":
            reason = "turn budget exhausted: " + str(reason)
        notes = ["this round only addresses what the first round left open (pages left unread, comparison with reference "
                 "photos, the open points named in the task); say what each page that was read states about the picture "
                 "(where / when it was published or taken) or about the claim, and whether that changes the first round's conclusion",
                 "pages read: " + (", ".join(f"{e} ({u})" for u, e in opened.items() if e in read_ids) or "none"),
                 "pages that could not be opened: " + (", ".join(u for u in urls if not opened.get(u)) or "none")]
        if search_wanted:
            notes.append("pages read from the new search: " + (", ".join(f"{p['evidence_id']} ({str(p.get('title') or p.get('url'))[:80]})"
                                                                         for p in found_read) or "none")
                         + "; answer each open point named in the task in its own finding, citing the page that settles it, "
                           "and say plainly which points stay open")
        if refs_wanted:
            notes.append("reference photos: " + (", ".join(side) or "none") + "; compare them with the given picture feature by "
                         "feature and say plainly whether they show the same place (match / no match / cannot tell, and why)")
        decision = _report(state, memory, rule, status=status, reason=reason, notes=notes + _ocr_notes(state),
                           branch="report_last_turn" if last_turn else ("report_completed" if status == "COMPLETED" else "report_incomplete"))
        if refs_wanted:
            decision.slot["required_findings"] = [{"ids": list(side), "words": _L_VERDICT_WORDS if side else _L_CANNOT_WORDS,
                                                   "message": "no finding states the outcome of the comparison with the reference "
                                                              "photos (match / no match / cannot tell and why)"}]
        return decision

    if state.remaining <= 0:
        return report("G1_last_turn_report", last_turn=True)
    for url in urls:
        if url not in opened and f"open_{url}" not in issued and state.remaining >= 2:
            _mark(memory, f"open_{url}")
            return Decision("action", "S2f_open_page", {"action": "SEARCH", "arguments": {"source": url, "output_type": "text"},
                            "focus_evidence_ids": [], "rationale": f"The first round found {url} but did not read it; open it."},
                            branch="SEARCH open page")
        page = opened.get(url)
        if page and page not in read_ids and state.remaining >= 2:
            return Decision("action", "S2f_read_page", {"action": "READ_RECORD", "arguments": {"record_id": page},
                            "focus_evidence_ids": [], "rationale": f"Read {page}, the page at {url}."}, branch="READ_RECORD text")
    if refs_wanted and image:
        if not [s for s in _query_searches(searches) if (s.get("request") or {}).get("output_type") == "image"] \
                and state.remaining >= 3:
            m = re.search(r'claim:\s*"(.*)"\s*\(claim date', state.task, re.S)
            claim = m.group(1) if m else state.task
            return Decision("query_slot", "L2_reference_images", {"action": "SEARCH"}, branch="SEARCH text image",
                            slot={"name": "S1_query", "purpose": "reference_images", "input_type": "text", "output_type": "image",
                                  "previous": [], "focus": [image], "claim": claim[:400]})
        if side and "compare_refs" not in issued and state.remaining >= 2:
            _mark(memory, "compare_refs")
            return Decision("action", "L3_view_with_references", _read(side[0], [image, *side],
                            f"View {image} next to the reference photos {', '.join(side)} of the place the claim names, to compare "
                            f"landmarks and layout."), branch="READ_RECORD focus")
    if search_wanted:
        if not text_queries and "followup_search" not in issued and state.remaining >= 3:
            _mark(memory, "followup_search")
            return Decision("query_slot", "S2f_search_open_points", {"action": "SEARCH"}, branch="SEARCH text page",
                            slot={"name": "S1_query", "purpose": "followup", "input_type": "text", "output_type": "text",
                                  "previous": [], "focus": [], "claim": claim[:400]})
        words = _claim_words(claim)
        todo = sorted((p for p in found if p["evidence_id"] not in read_ids),
                      key=lambda p: -len(words & _claim_words(_page_text(p))))
        if todo and len(found_read) < _FOLLOWUP_READS and state.remaining >= 2:
            page = todo[0]
            return Decision("action", "S2f_read_found_page", {"action": "READ_RECORD", "arguments": {"record_id": page["evidence_id"]},
                            "focus_evidence_ids": [], "rationale": f"Read {page['evidence_id']} "
                            f"({str(page.get('title') or page.get('url'))[:80]}); it may settle the open points."},
                            branch="READ_RECORD text")
    return report({"E": "E9_report", "L": "L9_report"}.get(family, "C9_report"))


def _read_records(state: HiState) -> list[dict[str, Any]]:
    return [o for o in state.observations if o.get("action") == "READ_RECORD"]


_VIDEO_LINK = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)[\w-]{6,}|youtu\.be/[\w-]{6,}|"
                         r"vimeo\.com/\d{5,}|dailymotion\.com/video/\w+)")


def _linked_videos(state: HiState) -> list[str]:
    """读过的页面正文里出现的视频链接（学生在 READ_RECORD 的观察里看得到的）。读到时存进 memory，之后被压缩掉也认得。"""

    found: list[str] = state.memory.setdefault("linked_videos", [])
    history = list(state.view.get("history") or [])
    for index, item in enumerate(history):
        if item.get("action") != "READ_RECORD":
            continue
        observation = item.get("observation")
        if observation is None and index == len(history) - 1:
            observation = state.view.get("latest_observation")
        for url in _VIDEO_LINK.findall(json.dumps(observation or {}, ensure_ascii=False)):
            if url not in found:
                found.append(url)
    return found


def _opened_videos(searches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """直接打开的视频来源（source=URL, output_type=video）：候选 R、页面 E、字幕 E（打开视频自动带字幕时）。"""

    out = []
    for search in searches:
        req = search.get("request") or {}
        if not req.get("source") or req.get("output_type") != "video":
            continue
        for result in (search.get("details") or {}).get("results") or []:
            subtitle = next((c.get("evidence_id") for m in result.get("materials") or []
                             if m.get("kind") == "subtitle" and m.get("status") in ("ready", "partial")
                             for c in m.get("evidence") or [] if c.get("evidence_id")), None)
            out.append({"source": req.get("source"), "candidate": result.get("candidate_id"), "page": _page_of(result),
                        "subtitle": subtitle, "url": result.get("source_url")})
    return out


def _linked_video_step(state: HiState, memory: dict[str, Any], video: str | None, read_ids: set[str]) -> Decision | None:
    """C 族：读到的页面里给出了声明所说视频的链接 → 打开它一次（工具目录没有字幕参数时打开视频自动带原语言字幕）
    → 字幕取到了就读 → 与材料视频比对一次（画面对不上且两边都有声音时比对自动只比声音）。"""

    issued = _issued(memory)
    opened = _opened_videos(_search_obs(state))
    if not opened:
        links = _linked_videos(state)
        if links and "open_linked" not in issued and state.remaining >= 4:
            _mark(memory, "open_linked")
            return Decision("action", "C8_open_linked_video", {"action": "SEARCH", "arguments": {
                "source": links[0], "output_type": "video"}, "focus_evidence_ids": [],
                "rationale": f"A page that was read links the video {links[0]}; open it to see what it shows and says."},
                branch="SEARCH open video")
        return None
    first = opened[0]
    if first.get("subtitle") and first["subtitle"] not in read_ids and "read_subtitle" not in issued and state.remaining >= 3:
        _mark(memory, "read_subtitle")
        return Decision("action", "C8b_read_subtitles", {"action": "READ_RECORD", "arguments": {"record_id": first["subtitle"]},
                        "focus_evidence_ids": [], "rationale": f"Read the subtitles {first['subtitle']} of the linked video."},
                        branch="READ_RECORD text")
    if (video and first.get("page") and "compare_media" in state.inspect_ops and "compare_linked" not in issued
            and state.remaining >= 3):
        _mark(memory, "compare_linked")
        return Decision("action", "C8c_compare_linked_video", _inspect([video, first["page"]], "compare_media",
                        state.compare_params(mode="auto"), [],
                        f"Compare {video} with the linked video {first['page']}: whether the given video is taken from it and "
                        f"what differs."), branch="INSPECT compare_media")
    return None


# ---- r1（10-17）C 族：画面类声明必须看帧、"视频说了什么"要看完整转写。verifier@0.4.0 第 2 处补把"声明说的是画面里发生的事、
# 报告只看了 OCR / 转写"判为缺口（r1v C 改写 17 → 24，新增 8 份里 3 份没看帧、5 份只引了转写第一句）。以前 walker 的 C 族只做
# ASR / OCR，交报告那一轮附图是空的（r0g groundlie360_v1959500：茶里的"虫子"一帧没看）。
_SPEECH_CLAIM = re.compile(r"\b(?:says?|said|saying|tells?|told|claims? that|admits?|announc\w*|declar\w*|remarks?|speech|"
                           r"interview|quot\w*|stated?|statement|vows?|promis\w*|warns?|urges?|calls? for|asks?|argues?|"
                           r"mentions?|confess\w*|insult\w*|joke\w*)\b|[\"“][^\"”]{8,}[\"”]|:\s*['\"“]", re.I)
_VISUAL_CLAIM = re.compile(r"\b(?:shows?|showing|shown|footage|filmed|caught|captured|seen|visible|appears?|watch|look at|"
                           r"in (?:the|this) (?:video|clip)|video of|moment|scene|crowd|fire|flood|explosion|attack\w*|"
                           r"crash\w*|stepping|walk\w*|smok\w*|dancing|fight\w*|beat\w*|rob\w*|worms?|insects?|"
                           r"inside|landscape|aerial|sky|street|building)\b", re.I)
_MAX_SCENE_FRAMES = 6


def claim_kinds(claim: str) -> set[str]:
    """声明讲的是什么：visual = 画面里发生的事（动作、事件、物体出现与否）；speech = 视频里有人说了什么。可以两者都是。
    只看声明文字；没有说话类的词时按画面类处理（REQ1 本来就问"画面、屏幕文字、语音"，多看一组帧不会错）。"""

    kinds: set[str] = set()
    if _VISUAL_CLAIM.search(claim) or not _SPEECH_CLAIM.search(claim):
        kinds.add("visual")
    if _SPEECH_CLAIM.search(claim):
        kinds.add("speech")
    return kinds


def _scene_frames(state: HiState, video: str) -> list[str]:
    """按场景采样的帧：按镜头取的帧（C2w get_frames）；没有就用 video_structure 每个镜头的中间帧图，再没有用 get_frames 取到的帧、
    入库关键帧。均匀取 ≤ 6 张。"""

    per_shot = [e for o in state.observations if o.get("operation") == "get_frames" and video in (o.get("targets") or [])
                and (o.get("params") or {}).get("times_s") for e, _ in _frames_of(o)]
    if per_shot:
        return _spread(per_shot, _MAX_SCENE_FRAMES)
    for obs in reversed(state.observations):
        if obs.get("operation") != "video_structure" or video not in (obs.get("targets") or []):
            continue
        ids = [str(i.get("evidence_id")) for i in (obs.get("result") or {}).get("images") or [] if isinstance(i, dict) and i.get("evidence_id")]
        ids = ids or [str(e) for e in obs.get("created") or [] if e != obs.get("result_evidence_id")]
        if ids:
            return _spread(ids, _MAX_SCENE_FRAMES)
    frames = [e for o in state.observations if o.get("operation") == "get_frames" and video in (o.get("targets") or [])
              for e, _ in _frames_of(o)]
    return _spread(frames or _material_frames(state, video), _MAX_SCENE_FRAMES)


def _scene_step(state: HiState, memory: dict[str, Any], video: str) -> Decision | None:
    """C2v：按场景采样取帧（video_structure：镜头切分 + 每个镜头的中间帧图）；没有这个操作时 get_frames 均匀取 6 帧。"""

    issued = _issued(memory)
    if "scene_frames" in issued and "scene_extra" not in issued and "get_frames" in state.inspect_ops and state.remaining >= 4:
        # 镜头少或镜头长（r1dev v1959500：166 s 只有 1 个镜头 → 只有 1 张中间帧）：按镜头取帧，长镜头约每 30 s 一张，≤ 6 张
        obs = next((o for o in reversed(state.observations) if o.get("operation") == "video_structure"
                    and video in (o.get("targets") or []) and o.get("status") == "ready"), None)
        shots = [s.get("range_s") for s in ((obs or {}).get("result") or {}).get("shots") or [] if isinstance(s, dict)]
        times: list[float] = []
        for rng in shots:
            if isinstance(rng, list) and len(rng) == 2 and _num(rng[0]) is not None and _num(rng[1]) is not None:
                a, b = float(rng[0]), float(rng[1])
                n = max(1, min(_MAX_SCENE_FRAMES, round((b - a) / 30.0)))
                times += [_r(a + (k + 0.5) * (b - a) / n, 2) for k in range(n)]
        times = [float(t) for t in _spread([str(t) for t in times], _MAX_SCENE_FRAMES)]
        images = ((obs or {}).get("result") or {}).get("images") or []
        if obs and len(times) > len(images):
            _mark(memory, "scene_extra")
            return Decision("action", "C2w_frames_per_shot", _inspect([video], "get_frames", {"times_s": times}, [],
                            f"{video} has {len(shots)} shot(s); take {len(times)} frames spread over its shots to see what "
                            f"the footage shows throughout."), branch="INSPECT get_frames")
    if "scene_frames" in issued or state.remaining < 4:
        return None
    if "video_structure" in state.inspect_ops:
        _mark(memory, "scene_frames")
        return Decision("action", "C2v_scene_frames", _inspect([video], "video_structure", {}, [],
                        f"Split {video} into its shots and take one frame per shot, to see what the footage itself shows."),
                        branch="INSPECT video_structure")
    if "get_frames" in state.inspect_ops:
        _mark(memory, "scene_frames")
        return Decision("action", "C2v_sample_frames", _inspect([video], "get_frames", {"count": _MAX_SCENE_FRAMES}, [],
                        f"Take {_MAX_SCENE_FRAMES} frames spread over {video} to see what the footage itself shows."),
                        branch="INSPECT get_frames")
    return None


def _asr_record(state: HiState, video: str) -> str | None:
    for o in state.observations:
        if o.get("operation") == "asr" and o.get("status") == "ready" and video in (o.get("targets") or []) \
                and o.get("result_evidence_id"):
            return str(o["result_evidence_id"])
    return None


def _claim_review_before_report(state: HiState, memory: dict[str, Any], video: str | None) -> Decision | None:
    """C9v（同 L3b）：交报告前把场景帧放回附图（焦点），并读一次完整转写（有语音时）——下一轮 S4 看着帧和整段转写写报告。
    每个工作区一次（按历史里有没有这一步判断）。"""

    if not video or state.remaining < 2:
        return None
    frames = _scene_frames(state, video)
    asr = _asr_record(state, video)
    if not frames and not asr:
        return None
    record = asr or frames[0]
    done = any(h.get("action") == "READ_RECORD" and (h.get("request") or {}).get("record_id") == record
               and set(h.get("focus_evidence_ids") or []) >= set(frames[:2])
               for h in state.view.get("history") or [])
    if done or (memory.get("claim_review") and all(e in state.attached for e in frames)):
        return None
    memory["claim_review"] = 1
    memory["review_record"] = record
    focus = frames[:max(1, state.limit)]
    what = (f"read the full transcript {asr} and view" if asr else "view") + f" the scene frames {', '.join(focus)}"
    return _substituted(Decision("action", "C9v_review_before_report", _read(record, focus,
                        f"Before reporting, {what} of {video}, to state what the video itself shows and says."),
                        branch="READ_RECORD focus"))


def _claim_media_notes(state: HiState, video: str | None, claim: str) -> tuple[list[str], list[dict[str, Any]]]:
    """S4 的说明与合同校验项：画面类声明 → 一条 Finding 写帧里有没有声明说的那件事（引帧）；有转写 → 一条 Finding 概括整段
    视频说了什么（不只第一句，引转写记录）。"""

    if not video:
        return [], []
    kinds = claim_kinds(claim)
    frames = [e for e in _scene_frames(state, video) if e in state.attached]
    asr = _asr_record(state, video)
    notes: list[str] = []
    required: list[dict[str, Any]] = []
    if frames:
        notes.append(f"scene frames {', '.join(frames)} of {video} (one per shot) are attached to this turn"
                     + (": the claim is about something visible in the footage, so one finding must say whether the frames "
                        "show it (what is visible, which frame, and what is not visible), citing the frame IDs; on-screen "
                        "text or the transcript alone does not answer what the picture shows" if "visual" in kinds else
                        ": one finding must say what the footage itself shows, citing the frame IDs"))
        required.append({"ids": frames, "words": r"show|visible|frame|appear|depict|see|footage|picture",
                         "message": f"no finding says what the scene frames ({', '.join(frames)}) show; write one, citing them"})
    elif "visual" in kinds:
        notes.append(f"no frames of {video} are attached to this turn: do not describe the footage beyond what the cited "
                     f"records state; say in limitations that the picture content was not examined")
    if asr:
        notes.append(f"the full speech transcript {asr} is in the state: one finding must say what the video says as a whole "
                     f"(its main points in order, not only its opening sentence), quoting the passages that bear on the claim "
                     f"and saying whether any passage states what the claim says, citing {asr}")
        required.append({"ids": [asr], "words": r"say|says|said|speak|speech|transcript|state|mention",
                         "message": f"no finding cites the speech transcript {asr} for what the video says as a whole; write one"})
    return notes, required


def _decide_claim(state: HiState, memory: dict[str, Any]) -> Decision:
    """C 族：（有视频时）先转写语音、读画面文字 → 每一步由 S3 槽在"再搜一条 / 读一页 / 交报告"里选，程序限预算。"""

    issued = _issued(memory)
    videos = state.videos()
    searches = _search_obs(state)
    queries = _query_searches(searches)
    dated = _dated(searches)
    pages = _search_pages(searches)
    reads = _read_records(state)
    read_ids = {str((r.get("request") or {}).get("record_id")) for r in reads}
    # 10-15 对话 5''（追加 2(a)）：标题 / 摘要命中声明的已取得页面没读完，不许交报告
    claim_text = re.sub(r"\s*\((?:posted|claim date):[^)]*\)\s*$", "", state.task.split(":", 1)[-1])
    must = _must_read(pages, read_ids, claim_text)

    def report(rule: str, last_turn: bool = False) -> Decision:
        read_pages = [p for p in pages if p["evidence_id"] in read_ids]
        if read_pages:
            status, reason = "COMPLETED", None
        else:
            status, reason = "INCOMPLETE", ("no external source was read" if searches else "no search was made")
        if status == "INCOMPLETE" and _open_gap_clause(must):
            reason = f"{reason}; {_open_gap_clause(must)}"
        if last_turn and status == "INCOMPLETE":
            reason = "turn budget exhausted: " + str(reason)
        notes = [f"searches made: {', '.join(repr(_query_text(s.get('request') or {})) for s in queries) or 'none'}",
                 "pages read: " + (", ".join(p["evidence_id"] + " (" + str(p.get("title") or p.get("url"))[:80] + ")"
                                             for p in read_pages) or "none"),
                 "a page that was found but not read is not evidence; cite only pages that were read and what they say",
                 "a claim is answered by what reliable sources record about the event itself; state plainly whether the claim "
                 "is accurate, misleading or false as far as the evidence shows, or that it could not be verified"] \
            + _asr_notes(state) + _ocr_notes(state)
        if dated:
            notes.append("date lookups: " + "; ".join(f"{k}: {v}" for k, v in dated.items())
                         + " — a page's date is when that page was published; an archive capture only bounds when it was online")
        for o in _opened_videos(searches)[:1]:
            notes.append(f"the linked video {o.get('url')} was opened ({o.get('page')}"
                         + (f", subtitles {o['subtitle']}" if o.get("subtitle") else ", no subtitles") + ")"
                         + ("; compare_media compared it with the given video — say whether the given video comes from it"
                            if "compare_linked" in _issued(memory) else ""))
        media_notes, required = _claim_media_notes(state, videos[0] if videos else None, claim_text)
        decision = _report(state, memory, rule, status=status, reason=reason, notes=notes + media_notes,
                           branch="report_last_turn" if last_turn else ("report_completed" if status == "COMPLETED" else "report_incomplete"))
        decision.slot["required_findings"] = required
        return decision

    def report_now(rule: str) -> Decision:
        return _claim_review_before_report(state, memory, videos[0] if videos else None) or report(rule)

    if state.remaining <= 0:
        return report("G1_last_turn_report", last_turn=True)
    if videos:
        video = videos[0]
        prof = state.profile(video)
        if "asr" in state.inspect_ops and (prof.get("audio") or 0) > 0 and "asr_A" not in issued and state.remaining >= 4:
            _mark(memory, "asr_A")
            return Decision("action", "C1_asr_material", _inspect([video], "asr", {}, _existing(state, state.attached, state.limit),
                            f"Transcribe what is said in {video}; the claim is checked against what the video actually says."),
                            branch="INSPECT asr")
        if "ocr" in state.inspect_ops and "ocr_material" not in issued and state.remaining >= 4:
            _mark(memory, "ocr_material")
            return Decision("action", "C2_ocr_material", _inspect([video], "ocr", {}, _existing(state, state.attached, state.limit),
                            f"Read the on-screen text of {video} (captions, name straps, logos) to compare it with the claim."),
                            branch="INSPECT ocr(video)")
        step = _scene_step(state, memory, video)
        if step is not None:
            return step
    step = _pdf_pages_step(state, memory, read_ids)
    if step is not None and state.remaining >= 2:
        return step
    if just_reviewed(state, memory):
        return report("C9_report")
    if reads:
        step = _linked_video_step(state, memory, videos[0] if videos else None, read_ids)
        if step is not None:
            return step
    choices: list[str] = []
    if len(queries) < _MAX_CLAIM_SEARCHES:
        choices.append("search")
    offsets: dict[str, int] = {}
    for record_id in read_ids:
        nxt = _next_offset(state, record_id)
        if nxt and nxt > 0:
            offsets[record_id] = nxt
    unread = [p for p in pages if p["evidence_id"] not in read_ids]
    if (unread or offsets) and len(reads) < _MAX_CLAIM_READS:
        choices.append("read")
    undated = [p for p in pages if p.get("candidate") and p["candidate"] not in dated]
    if _dates_allowed(state) and undated and len(dated) < _MAX_DATE_LOOKUPS:
        choices.append("dates")
    if reads:
        choices.append("report")
    if not queries:
        choices = ["search"]
    if must and state.remaining >= 2 and len(reads) < _MAX_CLAIM_READS + 2:
        choices = [c for c in choices if c != "report"]
        if "read" not in choices:
            choices.append("read")
        must_ids = [p["evidence_id"] for p in must]
        unread = list(must) + [p for p in unread if p["evidence_id"] not in must_ids]
    if state.remaining <= 1 or not choices or choices == ["report"]:
        return report_now("C9_report")
    if state.remaining == 2 and videos:
        # 只剩两轮：这一轮把帧放回附图、读整段转写，下一轮交报告
        step = _claim_review_before_report(state, memory, videos[0])
        if step is not None:
            return step
    return Decision("next_slot", "C3_next_step", {}, branch="S3",
                    slot={"name": "S3_next", "choices": choices,
                          "unread": [{k: p.get(k) for k in ("evidence_id", "title", "url")} for p in unread[:8]],
                          "undated": [{"candidate": p["candidate"], "title": p.get("title"), "url": p.get("url")}
                                      for p in undated[:8]],
                          "offsets": offsets, "read": sorted(read_ids),
                          "previous": [_query_text(s.get("request") or {}) for s in queries],
                          "claim": state.task.rsplit(":", 1)[-1].strip()[:400],
                          "report_decision": (report_now if "report" in choices else report)("C9_report").to_dict()})


def _substituted(decision: Decision) -> Decision:
    decision.slot = {**decision.slot, "substituted": True}
    return decision


def resolve_slot_choice(decision: Decision, choice: dict[str, Any]) -> Decision:
    """S1 / S3 槽的输出 → 真正的一步（SEARCH / READ_RECORD）或报告。不合法的选择退到最保守的一步。"""

    slot = decision.slot
    rationale = str(choice.get("rationale") or "")[:300]
    if slot.get("name") == "S1_query":
        query = " ".join(str(choice.get("query") or "").split())[:300]
        if not query:
            raise ValueError("S1 slot returned no query")
        if slot.get("input_type") == "text_image" and slot.get("evidence_id"):
            arguments: dict[str, Any] = {"input_type": "text_image", "query": {"evidence_id": slot["evidence_id"], "text": query},
                                         "output_type": slot.get("output_type") or "video"}
            focus = [slot["evidence_id"]]
        else:
            arguments = {"input_type": "text", "query": query, "output_type": slot.get("output_type") or "video"}
            focus = list(slot.get("focus") or [])
        return Decision("action", decision.rule, {"action": "SEARCH", "arguments": arguments, "focus_evidence_ids": focus,
            "rationale": rationale or "Search for the original with the clues read from the material."}, branch=decision.branch)
    if slot.get("name") == "S3_next":
        report = Decision(**slot["report_decision"])
        kind = str(choice.get("next") or "").lower()
        choices = slot.get("choices") or []
        unread = [p.get("evidence_id") for p in slot.get("unread") or []]
        offsets = slot.get("offsets") or {}
        if kind == "search" and "search" in choices and str(choice.get("query") or "").strip():
            query = " ".join(str(choice["query"]).split())[:300]
            return Decision("action", "C4_search", {"action": "SEARCH", "arguments": {
                "input_type": "text", "query": query, "output_type": "text"}, "focus_evidence_ids": [],
                "rationale": rationale or "Search for reports of the event the claim describes."},
                branch="SEARCH text page" if not slot.get("previous") else "SEARCH text page (next query)")
        if kind == "read" and "read" in choices:
            eid = str(choice.get("evidence_id") or "")
            if eid in offsets:
                return Decision("action", "C5_read_more", {"action": "READ_RECORD", "arguments": {
                    "record_id": eid, "offset": int(offsets[eid])}, "focus_evidence_ids": [],
                    "rationale": rationale or f"Continue reading {eid}."}, branch="READ_RECORD offset>0")
            if eid in unread:
                return Decision("action", "C5_read_page", {"action": "READ_RECORD", "arguments": {"record_id": eid},
                                "focus_evidence_ids": [], "rationale": rationale or f"Read {eid}."}, branch="READ_RECORD text")
        if kind == "dates" and "dates" in choices:
            undated = [p.get("candidate") for p in slot.get("undated") or []]
            cand = str(choice.get("candidate") or "")
            if cand in undated:
                return _dates_action(cand, "C7_dates", rationale or f"Check when {cand} was published or first archived.")
            if undated:
                # 10-15 对话 5''：选的候选不在清单里、换成第一个没查过的——S3 的理由讲的是它选的那个（r0a fmnv TIu R2 → R5），不沿用
                return _substituted(_dates_action(undated[0], "C7_dates", f"Check when {undated[0]} was published or first archived."))
        if kind == "report" and "report" in choices:
            return report
        # 选择不合法（编号不在未读清单、不许交报告时选了报告等）：退到"读第一篇未读"→"按声明原文搜"→ 报告。
        # 10-15 对话 5''：退选的动作不沿用 S3 的理由（它讲的是 S3 选的另一条记录：r0a fmnv RXR 读 E14 / E15 时理由讲 E11、
        # e17 读 E4 时理由讲 E6）；标 substituted，由 S0 按实际动作重写
        if "read" in choices and unread:
            return _substituted(Decision("action", "C5_read_page", {"action": "READ_RECORD", "arguments": {"record_id": unread[0]},
                                "focus_evidence_ids": [], "rationale": f"Read {unread[0]}."}, branch="READ_RECORD text"))
        if "report" in choices and "search" not in choices:
            return report
        if "search" in choices:
            claim = str(slot.get("claim") or "").split()
            query = " ".join(claim[:12]) or "news report"
            # 10-15 对话 5''：兜底搜索也由 S0 按实际动作写理由（r0g 第一遍这句模板在 accepted 里出现 6 次，门 B7 上限 5）
            return _substituted(Decision("action", "C4_search_fallback", {"action": "SEARCH", "arguments": {
                "input_type": "text", "query": query, "output_type": "text"}, "focus_evidence_ids": [],
                "rationale": "Search for reports of the event the claim describes."}, branch="SEARCH text page"))
        return report
    raise ValueError(f"unknown slot {slot.get('name')}")


class GiaState:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.view: dict[str, Any] = payload.get("global_view") or {}
        self.task = str(self.view.get("task") or "")
        rc = self.view.get("runtime_constraints") or {}
        self.round = int(rc.get("round_index") or 1)
        self.can_create = bool(rc.get("can_create_round"))
        self.remaining = int(rc.get("remaining_turns") if rc.get("remaining_turns") is not None else 99)
        self.max_tasks = int(rc.get("max_tasks_per_round") or 2)
        self.reports = [r for r in self.view.get("latest_reports") or [] if isinstance(r, dict)]
        self.attached = [i.get("evidence_id") for i in self.view.get("image_inputs") or [] if isinstance(i, dict)]
        self.cards: dict[str, dict[str, Any]] = {}
        for key in ("original_materials", "current_materials", "history_index"):
            for card in self.view.get(key) or []:
                if isinstance(card, dict) and card.get("evidence_id"):
                    self.cards.setdefault(card["evidence_id"], card)

    def originals(self) -> list[str]:
        out = []
        for card in self.view.get("original_materials") or []:
            rid = card.get("reference_id") or card.get("evidence_id")
            if rid:
                out.append(rid)
        return out

    def is_image(self, evidence_id: str) -> bool:
        card = self.cards.get(evidence_id) or {}
        mods = card.get("modalities")
        if mods:
            return "image" in mods
        label = str(card.get("label") or "").lower()
        return label.startswith(("frame of", "difference", "crop", "cropped")) or "comparison image" in label


def decide_gia(payload: dict[str, Any], memory: dict[str, Any]) -> Decision:
    state = GiaState(payload)
    issued = _issued(memory)
    reported = [r for r in state.reports if (r.get("report") or {}).get("status")]

    # 第 1 轮：一个工作区，任务 = 原任务（B 族给定媒体：总任务就是一个完整的本地任务）
    if not state.reports and state.round == 1 and "round1" not in issued:
        _mark(memory, "round1")
        why = ("Checking where the picture was taken means comparing it with photos of the claimed place and reading the "
               "pages that carry it, one chain of dependent steps; one workspace does it."
               if _PLACE_CHECK.search(state.task) else
               "Tracing the picture's earliest copies and their dates is one chain of dependent steps; one workspace does it."
               if _EARLIEST.search(state.task) else
               "Finding the original and comparing it with the material is one chain of dependent steps; one workspace does it."
               if _FIND_ORIGINAL.search(state.task) else
               "Checking the claim needs the material and external sources read together; one workspace does it."
               if _CLAIM_TASK.search(state.task) else
               "The given materials define one self-contained comparison task; one workspace can gather all of its evidence.")
        return Decision("action", "G1_round1", {"decision": "INVESTIGATE",
                        "tasks": [{"task": state.task, "materials": state.originals()}], "rationale": why},
                        branch="INVESTIGATE round1")

    if not reported:
        return Decision("finish", "G9_no_reports", {"decision": "FINISH"}, branch="FINISH",
                        slot={"name": "S5_answer", "notes": ["no workspace report is available; state what remains unknown"]})

    # 亲眼看：报告引用的图像 E 不在本次附图里 → READ_RECORD（每轮最多一次）
    key = f"look_r{state.round}"
    if key not in issued and state.remaining >= 1:
        cited: list[str] = []
        for item in reported:
            for f in (item.get("report") or {}).get("findings") or []:
                cited += [e for e in f.get("evidence_ids") or [] if e not in cited]
        missing = [e for e in cited if state.is_image(e) and e not in state.attached]
        if missing:
            _mark(memory, key)
            return Decision("action", "G4_look_at_cited_image", {"action": "READ_RECORD", "arguments": {"record_id": missing[0]},
                            "rationale": f"{missing[0]} is cited by the report but not attached; view it before answering."},
                            branch="GIA READ_RECORD")

    # 第 2 轮：读报告里写明的缺口（status_reason）。能靠更多查看补上的（某段时间没看 / 没确认、音轨没转写、预算用尽）
    # 且还能开新一轮 → 派一个针对缺口的任务；补不了的（材料本身没有、原件取不到、工具已失败）→ 直接 FINISH，答案写明不确定。
    # 已经跑完的轮次：看报告的工作区编号（W1-1 → 第 1 轮）；runtime_constraints.round_index 是"下一轮"的编号
    rounds_done = max((int(m.group(1)) for r in reported
                       for m in [re.match(r"W(\d+)-", str((r.get("report") or {}).get("task_id") or ""))] if m), default=1)
    durations = {}
    for card in state.view.get("original_materials") or []:
        m = re.search(r"Material\s+([AB])\b", str(card.get("label") or ""))
        if m and _num(card.get("duration_s")):
            durations[m.group(1).upper()] = float(card["duration_s"])
    gaps = classify_gaps(reported, durations, copy_task=bool(_COPY_TASK.search(state.task)))
    memory.setdefault("gaps", {})[f"r{state.round}"] = gaps
    family = ("L" if _PLACE_CHECK.search(state.task) else "E" if _EARLIEST.search(state.task)
              else "A" if _FIND_ORIGINAL.search(state.task) else "C" if _CLAIM_TASK.search(state.task) else "B")
    # 10-15 对话 5''（追加 2）：检索类（E / L / C）的"不可修"程序化——工作区还有未读的命中页 / 没并排看的参考图时，walker 在
    # status_reason 里写明（带网址 / E 编号），这类缺口能补：派第 2 轮按网址打开再读、并排看参考图。其余检索类缺口（原片没找到、
    # 没有来源）仍按"补不了"直接 FINISH。A 族不变
    if family in ("E", "L", "C"):
        for g in gaps:
            if g["kind"] == "unfixable" and not _SEARCH_GAP.search(g["reason"]):
                continue
            search_gap = _search_gap(g["reason"])
            if search_gap:
                g.update(search_gap)
        # r0+（10-16）：报告里写了 walker 能接手的跟进（未读页面 / 参考图比对 / 再搜来源）时，同一份报告里其余没答上的点
        # （充分性判定的 missing 条目等，"other"）也作为第 2 轮要补的点交给 S2；"N 个页面打不开"这类补不上的除外
        followable = {g["task_id"] for g in gaps if g["kind"] in _FOLLOWUP_KINDS}
        for g in gaps:
            if g["task_id"] in followable and g["kind"] == "other" and not g["budget"]:
                if _FAILED_PAGES.search(g["reason"]):
                    g["kind"] = "failed_pages"
                else:
                    g.update({"kind": "missing_point", "fixable": True})
    fixable = [g for g in gaps if g["fixable"] and (family == "B" or g["kind"] in (*_FOLLOWUP_KINDS, "missing_point"))]
    if family in ("E", "L", "C") and any(g["kind"] in _FOLLOWUP_KINDS for g in fixable) and state.can_create \
            and state.remaining >= 1 and "round2" not in issued and rounds_done == 1:
        _mark(memory, "round2")
        urls: list[str] = []
        for g in fixable:
            urls += [u for u in g.get("urls") or [] if u not in urls]
        refs = any(g["kind"] == "references" for g in fixable) and family == "L"
        parts = [f"open and read {u}" for u in urls[:3]]
        if refs:
            parts.append("view the given picture next to reference photos of the place the claim names")
        if any(g["kind"] == "search_more" for g in fixable):
            parts.append("search for sources on the open points listed above and read them")
        scope = "Scope: " + "; ".join(parts) + "." if parts else ""
        return Decision("investigate_slot", "G2_round2_search_gap", {"decision": "INVESTIGATE"},
                        branch="INVESTIGATE round2 " + "+".join(sorted({g["kind"] for g in fixable})),
                        slot={"name": "S2_task", "gaps": [g["reason"] for g in fixable if g["kind"] != "search_more"],
                              "materials": state.originals(),
                              "unfixable": [g["reason"] for g in gaps if not g["fixable"]], "scope": scope})
    if family != "B":
        fixable = []
    if fixable and state.can_create and state.remaining >= 1 and "round2" not in issued and rounds_done == 1:
        _mark(memory, "round2")
        anchors: list[tuple[float, float]] = []
        for g in fixable:
            anchors += [x for x in g["anchors"] if x not in anchors]
        anchors = anchors[:2]
        windows = [w for g in fixable for w in g["windows"]][:3 - len(anchors) if anchors else 3]
        audio = sorted({s for g in fixable for s in g["audio_sides"]})
        cited = []
        for g in fixable:
            cited += [e for e in g["cited"] if e not in cited]
        materials = state.originals() + [e for e in cited if state.is_image(e)][:6]
        kinds = sorted({g["kind"] for g in fixable})
        scope = _scope_line(windows, audio, fixable, anchors, bool(_COPY_TASK.search(state.task)))
        if scope and _NO_CORRESPONDENCE.search(json.dumps(reported, ensure_ascii=False)):
            # 追加 5：第 1 轮整段比对一个对应段都没对上——第 2 轮在锚点附近缩小范围再比大概率也是 0%（r0a vcsl_pair_1），跟进时跳过
            scope = scope.rstrip(".") + "; the whole-video comparison found no correspondence."
        return Decision("investigate_slot", "G2_round2_fixable_gap", {"decision": "INVESTIGATE"},
                        branch="INVESTIGATE round2 " + "+".join(kinds),
                        slot={"name": "S2_task", "gaps": [g["reason"] for g in fixable], "materials": materials,
                              "unfixable": [g["reason"] for g in gaps if not g["fixable"]], "scope": scope})

    # 第 2 轮之后：第 1 轮的报告不在 latest_reports 里，先用记忆查回来再写答案（G4 / QUERY_MEMORY）
    if rounds_done >= 2 and "memory_r1" not in issued and state.remaining >= 1:
        reported_ids = {str((r.get("report") or {}).get("task_id") or "") for r in reported}
        if not any(t.startswith("W1-") for t in reported_ids):
            _mark(memory, "memory_r1")
            query = " ".join(state.task.split()[:10])
            return Decision("action", "G5_query_round1_report", {"action": "QUERY_MEMORY",
                            "arguments": {"query": query, "record_types": ["report", "finding"]},
                            "rationale": "The first-round report is no longer in the current view; retrieve it so the answer combines both rounds."},
                            branch="GIA QUERY_MEMORY")

    notes = []
    unfixable = [g for g in gaps if not g["fixable"]]
    for g in unfixable:
        notes.append(f"a report left a gap that more inspection cannot close: {g['reason'][:300]} — state it as not determinable")
    if fixable and not state.can_create:
        notes.append("the reports still list open gaps but no further round is available; state them as unverified")
    if rounds_done >= 2:
        notes.append("combine the first-round findings (memory results in this state) with the second-round report")
    if family == "L":
        notes.append("say first whether the picture shows the place the claim names (on the basis of the comparison with "
                     "reference photos of that place), then where the pages that carry it say it was taken, giving the verdict "
                     "on the claim in plain words; if the place is not established, say so")
    elif family == "E":
        notes.append("say first where and when the picture was first published as far as the evidence shows (the earliest "
                     "copy found and its date, as 'no later than'), then whether the claim's place, time or event matches "
                     "that original context, giving the verdict on the claim in plain words")
    elif family == "C":
        notes.append("give the verdict on the claim in plain words (accurate / misleading / false, or not verifiable from the "
                     "evidence) in the first sentence, then the evidence for it; distinguish what the given video shows or says "
                     "from what external sources record; when the video itself or the sources read show the opposite of what the "
                     "claim says (e.g. vehicles deployed where the claim says withdrawn), the claim is false or misleading — "
                     "finding no source that repeats the claim is not the same as 'not verifiable'")
    elif family == "A":
        notes.append("say first whether an original was found (which video, and on what basis), then each difference with its "
                     "times in the given video and in the original; if no original was found, say so and do not describe "
                     "changes that would need it")
    rule = "G3_finish_unfixable_gap" if unfixable and not fixable else ("G3_finish_after_round2" if rounds_done >= 2 else "G3_finish")
    return Decision("finish", rule, {"decision": "FINISH"}, branch="FINISH", slot={"name": "S5_answer", "notes": notes})


# ---- 报告缺口的分类（GIA 只看报告里写出来的东西）

_SEARCH_GAP = re.compile(r"candidate pages that (?:match the claim were not examined|were found but not read)|reference photos .{0,80}"
                         r"were not (?:viewed side by side|compared feature by feature)|a further search for sources can address", re.I)
# r0+：只说"N 个候选页面打不开 / 没检查"的条目——靠更多工作补不上（r0g 理由表四节第 2 处）
_FAILED_PAGES = re.compile(r"\b\d+\s+(?:of|out of)\s+\d+\s+(?:candidate\s+)?pages?\b[^;]{0,60}\b(?:fail|could not|"
                           r"(?:were|was) (?:never |not )(?:be )?(?:opened|loaded|examined|checked))|failed to load", re.I)
_FOLLOWUP_KINDS = ("unread_pages", "references", "search_more")
_URL = re.compile(r"https?://[^\s,;]+")
_NO_CORRESPONDENCE = re.compile(r"no consistent (?:frame-level )?correspondence|found no (?:consistent )?correspondence|"
                                r"\b0% of (?:A|B|Material)", re.I)


def _search_gap(reason: str) -> dict[str, Any] | None:
    """walker 写进检索类报告 status_reason 的可修缺口（追加 2）：未读命中页（带网址）、参考图没并排看。"""

    if re.search(r"candidate pages that (?:match the claim were not examined|were found but not read)", reason, re.I):
        urls = [u.rstrip(".)") for u in _URL.findall(reason)]
        if urls:
            return {"kind": "unread_pages", "fixable": True, "urls": urls}
    if re.search(r"reference photos .{0,80}were not (?:viewed side by side|compared feature by feature)", reason, re.I):
        return {"kind": "references", "fixable": True, "urls": []}
    if re.search(r"a further search for sources can address", reason, re.I):     # r0+（inloop_actions）
        return {"kind": "search_more", "fixable": True, "urls": []}
    return None


_RESOLVED = re.compile(r"\b(?:match(?:es|ed)?|identical|the same|no (?:audible |audio |sound )?difference|consistent)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|no|never|cannot|could not|unverified|unclear)\b[^.;]{0,30}\b(?:match|identical|same|consistent)", re.I)


def _resolved_in_findings(clause: str, findings: list[dict[str, Any]]) -> bool:
    """追加 2：第 2 轮任务不针对报告自己已写明"已匹配 / 已解决"的点（r0a videosham_1028_v9：Finding 写了音频 8/8 匹配，
    status_reason 却列了"音频没核实"，第 2 轮就去追这个缺口）。现在只认音频：缺口讲音频 / 语音，而某条 Finding 里讲音频的句子
    写了匹配、没有否定。"""

    if not re.search(r"\b(?:audio|sound|speech|spoken)\b", clause, re.I):
        return False
    for f in findings:
        for sent in re.split(r"(?<=[.;])\s+", str(f.get("statement") or "")):
            if re.search(r"\b(?:audio|sound)\b", sent, re.I) and _RESOLVED.search(sent) and not _NEGATED.search(sent):
                return True
    return False

_SIDE_WINDOW = re.compile(r"(?:Material\s+)?\b([AB])\b(?:\s+at|\s+from|\s*:)?\s*(\d+(?:\.\d+)?)\s*(?:s|sec(?:onds)?)?\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", re.I)
# 对应锚点："A 5.900 s ↔ B 11.000 s"、"Material A at 15.349 s and Material B at 11.585 s"
_ANCHOR = re.compile(r"(?:Material\s+)?\bA\b\s*(?:at\s+)?(\d+(?:\.\d+)?)\s*s?\s*(?:↔|<->|<=>|vs\.?|versus|and|with|matches|against)\s*"
                     r"(?:Material\s+)?\bB\b\s*(?:at\s+)?(\d+(?:\.\d+)?)\s*s\b", re.I)
_EXTENT = re.compile(r"boundar|extent|continu|segment|start|end\b|range", re.I)
_FIXABLE = re.compile(r"not (?:been |yet |visually |independently |directly |fully )?(?:verified|confirmed|examined|checked|transcribed|viewed|sampled|inspected|established|determined)"
                      r"|unverified|unconfirmed|unexamined|unsettled|unresolved|budget|exhausted|ambiguous|uncertain", re.I)
_UNFIXABLE = re.compile(r"no audio track|has no audio|no counterpart|cannot be (?:compared|measured|obtained|established from)"
                        r"|could not be obtained|source identity|original source|authorship|production (?:order|history)"
                        r"|which (?:one|video|file) is (?:the )?original"
                        r"|not available|unavailable|does not exist|not provided|beyond the (?:end of the )?(?:clip|video)"
                        r"|tool failed|failed repeatedly", re.I)


_SIDE_INSTANT = re.compile(r"Material\s+([AB])\b(?:\s+at)?\s+(\d+(?:\.\d+)?)\s*s\b(?!\s*(?:–|—|-|to))", re.I)


def _windows_in(text: str) -> list[tuple[str, float, float]]:
    out = []
    for m in _SIDE_WINDOW.finditer(text):
        t0, t1 = float(m.group(2)), float(m.group(3))
        w = (m.group(1).upper(), _r(t0, 3), _r(t1, 3))
        if t1 > t0 and w not in out:
            out.append(w)
    return out


def _anchors_in(text: str) -> list[tuple[float, float]]:
    out = []
    for m in _ANCHOR.finditer(text):
        pair = (_r(float(m.group(1)), 3), _r(float(m.group(2)), 3))
        if pair not in out:
            out.append(pair)
    return out


def _gap_clauses(reason: str) -> list[str]:
    """status_reason 按 '; ' 分段（S4 要求的格式）；coverage 降级的前缀 "not every part ...: " 去掉。"""

    reason = re.sub(r"^not every part of the task is answered by the evidence:\s*", "", reason.strip())
    reason = re.sub(r"^turn budget exhausted:\s*", "turn budget exhausted; ", reason)
    return [c.strip() for c in re.split(r";\s+", reason) if c.strip()]


def windows_narrow(windows: list[tuple[str, float, float]], durations: dict[str, float] | None = None) -> bool:
    """时间窗是否是定点的：知道时长时，短于该侧时长的 80%；不知道时，不从 0 开始或短于 10 s。整段的"窗口"不是定点缺口。"""

    durations = durations or {}
    for side, t0, t1 in windows:
        d = durations.get(side)
        if (d and t1 - t0 < 0.8 * d) or (not d and (t0 > 0.5 or t1 - t0 < 10.0)):
            return True
    return False


def classify_gaps(reported: list[dict[str, Any]], durations: dict[str, float] | None = None,
                  copy_task: bool = True) -> list[dict[str, Any]]:
    """把 INCOMPLETE 报告的 status_reason 逐段分类（GIA 只看报告里写出来的东西）：
      extent  对应区间的起止没确定，有锚点 "A x s ↔ B y s"（或报告 Finding 里有） → 能补（锚点 ±10 s 限定范围对齐、两侧取边界帧）
      window  某侧某段时间没看 / 没确认                                       → 能补（该窗口逐帧差分或对齐、两侧取帧）
      audio   某侧音轨没转写                                                   → 能补（asr）
      budget  上一轮预算用完                                                   → 补不了（没写明哪段没做，跟进没有对应程序）
      其余，或写明材料本身决定的（没有对应、没有音轨、取不到、工具失败）     → 补不了"""

    gaps = []
    for item in reported:
        report = item.get("report") or {}
        if report.get("status") != "INCOMPLETE":
            continue
        cited = []
        for f in report.get("findings") or []:
            cited += [e for e in f.get("evidence_ids") or [] if e not in cited]
        finding_anchors: list[tuple[float, float]] = []
        for f in report.get("findings") or []:
            finding_anchors += [x for x in _anchors_in(str(f.get("statement") or "")) if x not in finding_anchors]
        for clause in _gap_clauses(str(report.get("status_reason") or "")):
            windows, anchors = _windows_in(clause), _anchors_in(clause)
            if not windows and not anchors:
                # 单个时刻（"Material B 2.300 s: ..."）：看它前后各 0.1 s
                windows = [(m.group(1).upper(), _r(max(0.0, float(m.group(2)) - 0.1), 3), _r(float(m.group(2)) + 0.1, 3))
                           for m in _SIDE_INSTANT.finditer(clause)][:1]
            if (copy_task and not anchors and _EXTENT.search(clause) and re.search(r"correspond", clause, re.I)
                    and not windows_narrow(windows, durations)):
                # 整段"对应的起止没定"、没写锚点：用报告 Finding 里写出的对应锚点（GIA 看得到的文字）
                anchors = finding_anchors[:2]
            audio_sides = sorted({s.upper() for s in re.findall(r"audio of (?:Material\s+)?([AB])\b", clause, re.I)}
                                 | ({"A", "B"} if re.search(r"(?:audio|speech|sound)[^.;]{0,40}not (?:been )?transcribed", clause, re.I)
                                    and not re.search(r"audio of (?:Material\s+)?[AB]\b", clause, re.I) else set()))
            budget = bool(re.search(r"budget|exhausted", clause, re.I))
            if _resolved_in_findings(clause, report.get("findings") or []):
                gaps.append({"task_id": report.get("task_id"), "reason": clause, "kind": "resolved_in_findings", "fixable": False,
                             "windows": [], "anchors": [], "audio_sides": [], "budget": budget, "cited": cited})
                continue
            if _UNFIXABLE.search(clause):
                kind, fixable = "unfixable", False
            elif anchors and (_EXTENT.search(clause) or _FIXABLE.search(clause)):
                kind, fixable = "extent", True
            elif windows and _FIXABLE.search(clause) and windows_narrow(windows, durations):
                kind, fixable = "window", True
            elif audio_sides and re.search(r"not (?:been )?transcribed|untranscribed", clause, re.I):
                kind, fixable = "audio", True
            elif budget:
                kind, fixable = "budget", False              # 没写明哪段没做，跟进没有对应的程序（wk4）
            else:
                kind, fixable = "other", False
            gaps.append({"task_id": report.get("task_id"), "reason": clause, "kind": kind, "fixable": fixable,
                         "windows": [x for x in windows if windows_narrow([x], durations)] if kind == "window" else [], "anchors": anchors if kind == "extent" else [],
                         "audio_sides": audio_sides if kind == "audio" else [], "budget": budget, "cited": cited})
    return gaps


def _scope_line(windows: list[tuple[str, float, float]], audio: list[str], gaps: list[dict[str, Any]],
                anchors: list[tuple[float, float]] | None = None, copy_task: bool = False) -> str:
    parts = [f"correspondence A {ta:.3f} s ↔ B {tb:.3f} s" for ta, tb in anchors or []]
    if copy_task and windows:
        parts.append("search for corresponding clips in " + " and ".join(f"Material {s} {t0:.3f}–{t1:.3f} s" for s, t0, t1 in windows))
    else:
        parts += [f"Material {s} {t0:.3f}–{t1:.3f} s" for s, t0, t1 in windows]
    parts += [f"transcribe the audio of Material {s}" for s in audio]
    if not parts and any(g["budget"] for g in gaps):
        parts.append("finish the steps the first round did not reach")
    return "Scope: " + "; ".join(parts) + "." if parts else ""


# ---- 第 2 轮跟进：任务里写明了要补的时间窗 / 锚点 / 音轨时，只补这些，不重走整条比对链


def followup_scope(task: str) -> tuple[list[tuple[str, float, float]], list[str], list[tuple[float, float]]]:
    scope = task[task.rfind("Scope:"):] if "Scope:" in task else task
    anchors = _anchors_in(scope)[:2]
    windows = _windows_in(re.sub(_ANCHOR, " ", scope))[:3]
    audio = sorted({s.upper() for s in re.findall(r"(?:transcribe|audio of)[^.;]{0,30}?Material\s+([AB])\b", scope, re.I)})
    return windows, audio, anchors


def _material_sides(state: HiState) -> dict[str, str]:
    out: dict[str, str] = {}
    for card in (state.view.get("materials") or {}).values():
        if not isinstance(card, dict) or "video" not in (card.get("modalities") or []) or card.get("derived_from"):
            continue
        m = re.search(r"Material\s+([AB])\b", str(card.get("label") or ""))
        if m and card.get("evidence_id"):
            out[m.group(1).upper()] = card["evidence_id"]
    return out


_EXTENT_WINDOW_S = 10.0       # 锚点两侧各看多少秒做窗口对齐


def _decide_followup(state: HiState, memory: dict[str, Any], windows: list[tuple[str, float, float]],
                     audio: list[str], anchors: list[tuple[float, float]] | None = None) -> Decision:
    """第 2 轮跟进：按缺口类型出程序（wk4）。
      锚点（对应段的起止没定）：锚点两侧 ±10 s 窗口对齐 → 取到与锚点一致的候选就在它的起止两侧各取 A/B 帧夹边界；
                               取不到就退到以图找图：A 在锚点 ±3 s 的帧，各在 B 的锚点附近 locate，看位置是否随之平移；
                               10-15（B6）：再按"里面还对得上 / 外面对不上"把每侧的起止夹出来（往外延伸或二分，预算够就
                               夹到 1 s），最后取 B 侧对应时刻的帧，和 A 的探针帧成对给报告看。
      时间窗（某段没看 / 没确认）：两侧时长相同（恒等映射）且窗口 > 1 s → 窗口内逐帧差分；否则取帧，另一侧同时刻也取。
      音轨：asr。
    覆盖某侧 80% 以上的"窗口"不是定点缺口（整段没确定），跳过——它由锚点程序处理。"""

    issued = _issued(memory)
    sides = _material_sides(state)
    given = [e for e in (state.view.get("materials") or {}) if isinstance(e, str) and e.startswith("E")
             and "image" in ((state.cards.get(e) or {}).get("modalities") or [])]
    anchors = anchors or []
    dur = {s: state.profile(v).get("duration") for s, v in sides.items()}
    if state.remaining <= 0:
        return _followup_report(state, memory, windows, audio, anchors, sides, dur, last_turn=True)

    a, b = sides.get("A"), sides.get("B")
    jobs = _extent_jobs(state, windows, anchors, dur)
    # 1 对应外延：锚点两侧 / 对应类任务里没对上的区间，都用限定范围的对齐
    if a and b and dur.get("A") and dur.get("B"):
        for job in jobs:
            if job["anchor"] and _inside_found(memory, job):
                continue
            step = _extent_step(state, memory, job, a, b, dur["A"], dur["B"], given, jobs)
            if step is not None:
                return step
        # 1b 窗口对齐对不上的锚点：以图找图把每侧起止夹出来，再取 B 侧对应帧
        step = _bracket_step(state, memory, a, b, dur["A"], dur["B"])
        if step is not None:
            return step
    if jobs:
        windows = []                                             # 对应类任务的时间窗已并入对齐作业

    # 2 时间窗
    same_length = bool(dur.get("A") and dur.get("B") and abs(dur["A"] - dur["B"]) < 0.1)
    for side, t0, t1 in windows:
        video = sides.get(side)
        d = dur.get(side)
        if not video or (d and t1 - t0 >= 0.8 * d):
            continue
        other_side = "A" if side == "B" else "B"
        key = f"fu_{side}_{t0:.3f}"
        if key not in issued and state.remaining >= 2:
            _mark(memory, key)
            diff_op = "diff_media" if "diff_media" in state.inspect_ops else "compare_media"
            if same_length and a and b and t1 - t0 > 1.0 and diff_op in state.inspect_ops:
                top = _r(min(t1, dur["A"] - 0.01), 3)
                memory.setdefault("followup", []).append(f"Material A/B {t0:.3f}-{top:.3f} s compared frame by frame ({diff_op})")
                _mark(memory, f"fu_{other_side}_{t0:.3f}")
                return Decision("action", "F1_followup_window_diff", _inspect([a, b], diff_op, {"range_a_s": [t0, top]}, [],
                                f"The task asks to settle {t0:.2f}-{top:.2f} s; compare both videos frame by frame over exactly "
                                f"that window to get where differences start and end."), branch=f"INSPECT {diff_op}(range)")
            params, how = _window_frames(state.profile(video).get("fps"), t0, t1, d)
            memory.setdefault("followup", []).append(f"Material {side} {t0:.3f}-{t1:.3f} s ({how})")
            memory.setdefault("fu_times", {})[side + f"{t0:.3f}"] = params.get("times_s")
            return Decision("action", "F1_followup_window", _inspect([video], "get_frames", params, _existing(state, given, 2),
                            f"The task asks to settle Material {side} {t0:.2f}-{t1:.2f} s; look at that window closely."),
                            branch="INSPECT get_frames times_s" if "times_s" in params else "INSPECT get_frames range")
        # 另一侧同时刻（只在两侧时长相同、映射为恒等时；否则换算不出来）
        okey = f"fu_{other_side}_{t0:.3f}"
        times = (memory.get("fu_times") or {}).get(side + f"{t0:.3f}")
        other = sides.get(other_side)
        if same_length and times and other and okey not in issued and state.remaining >= 2:
            _mark(memory, okey)
            mine = _frames_of(_frames_obs(state, video, times))
            memory.setdefault("followup", []).append(f"Material {other_side} at the same instants {times[0]:.3f}-{times[-1]:.3f} s")
            return Decision("action", "F1_followup_window_other_side", _inspect([other], "get_frames", {"times_s": times},
                            _existing(state, [e for e, _ in mine], state.limit // 2),
                            f"Both videos last the same; look at Material {other_side} at the same instants to compare both sides."),
                            branch="INSPECT get_frames times_s")

    # 3 音轨
    if "asr" in state.inspect_ops:
        for side in audio:
            video = sides.get(side)
            key = f"fu_asr_{side}"
            if not video or key in issued or state.remaining < 2:
                continue
            _mark(memory, key)
            memory.setdefault("followup", []).append(f"audio of Material {side} transcribed")
            return Decision("action", "F2_followup_asr", _inspect([video], "asr", {}, _existing(state, given, 2),
                            f"The task asks for the audio of Material {side}; transcribe it."), branch="INSPECT asr")

    # 4 报告前成对回看（取过两批以上帧时，报告那一步只附得上最后一批）
    frames = [e for o in state.observations if o["operation"] == "get_frames" for e, _ in _frames_of(o)]
    batches = len([o for o in state.observations if o["operation"] == "get_frames" and o.get("status") == "ready"])
    if batches > 1 and "fu_review" not in issued and frames and state.remaining >= 1:
        _mark(memory, "fu_review")
        picks = _review_picks(state, memory, frames)
        return Decision("action", "F3_followup_review", _read(picks[-1], picks,
                        "View the boundary frames of both sides together before reporting."), branch="READ_RECORD focus")
    return _followup_report(state, memory, windows, audio, anchors, sides, dur, last_turn=False)


def _window_frames(fps: float | None, t0: float, t1: float, duration: float | None) -> tuple[dict[str, Any], str]:
    top = min(t1, duration - 0.05) if duration else t1
    if fps and (top - t0) * fps <= 7:                         # 窗口不超过 8 帧：逐帧
        n = int(round((top - t0) * fps)) + 1
        return {"times_s": [_r(t0 + k / fps, 3) for k in range(max(1, n))][:8]}, "every frame"
    if top - t0 <= 1.0:                                       # 帧率看不到的短窗口：6 个等距时刻
        return {"times_s": sorted({_r(t0 + (top - t0) * k / 5, 3) for k in range(6)})}, "6 evenly spaced instants"
    return {"range_s": [_r(t0, 3), _r(top, 3)], "count": 6}, "6 evenly spaced frames"


def _range_matches(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """align_media 的 matches，或 compare_media 的 correspondences，统一成 range_a_s / range_b_s / time_mapping / score。"""

    res = obs.get("result") or {}
    if res.get("matches"):
        return [m for m in res["matches"] if isinstance(m, dict)]
    out = []
    for c in res.get("correspondences") or []:
        if isinstance(c, dict) and c.get("a_range_s") and c.get("b_range_s"):
            out.append({"range_a_s": c["a_range_s"], "range_b_s": c["b_range_s"], "time_mapping": c.get("mapping") or {},
                        "score": c.get("samples") or 0, "anchors": []})
    return out


def _consistent_match(align: dict[str, Any] | None, ta: float, tb: float) -> dict[str, Any] | None:
    """窗口对齐的候选里与锚点一致的那个：A 区间含 ta、B 区间含 tb（各留 1 s 余量），或映射把 ta 送到 tb 附近。"""

    if not align or align.get("status") != "ready":
        return None
    best = None
    for m in _range_matches(align):
        ra, rb = m.get("range_a_s") or [None, None], m.get("range_b_s") or [None, None]
        if None in ra or None in rb:
            continue
        inside = float(ra[0]) - 1.0 <= ta <= float(ra[1]) + 1.0 and float(rb[0]) - 1.0 <= tb <= float(rb[1]) + 1.0
        tm = m.get("time_mapping") or {}
        scale, offset = _num(tm.get("scale")), _num(tm.get("offset_s"))
        mapped = scale is not None and offset is not None and abs(scale * ta + offset - tb) <= 1.5
        if inside or mapped:
            if best is None or float(m.get("score") or 0) > float(best.get("score") or 0):
                best = m
    return best


_CORR_SCOPE = re.compile(r"search for corresponding clips", re.I)


def _extent_jobs(state: HiState, windows: list[tuple[str, float, float]], anchors: list[tuple[float, float]],
                 dur: dict[str, Any]) -> list[dict[str, Any]]:
    """对应外延作业：每个锚点一个（两侧各 ±10 s）；对应类任务里写明的"没对上"的时间窗（A、B 各取第一段配成一对，
    只给一侧就让另一侧全长）一个。覆盖某侧 80% 以上的窗口不算（整段对齐第 1 轮已做过）。"""

    jobs: list[dict[str, Any]] = []
    W = _EXTENT_WINDOW_S
    da, db = dur.get("A"), dur.get("B")
    if not da or not db:
        return jobs
    for ta, tb in anchors[:2]:
        if any(j["anchor"] and abs(j["anchor"][0] - ta) < 3.0 and abs(j["anchor"][1] - tb) < 3.0 for j in jobs):
            continue                                             # 与前一个锚点挨着：一次窗口对齐就覆盖了
        jobs.append({"id": f"x{len(jobs)}", "anchor": [ta, tb],
                     "wa": [_r(max(0.0, ta - W), 3), _r(min(da - 0.01, ta + W), 3)],
                     "wb": [_r(max(0.0, tb - W), 3), _r(min(db - 0.01, tb + W), 3)]})
    scope = state.task[state.task.rfind("Scope:"):] if "Scope:" in state.task else ""
    if _COPY_TASK.search(state.task) or _CORR_SCOPE.search(scope):
        narrow = [(s, t0, t1) for s, t0, t1 in windows if t1 - t0 < 0.8 * (da if s == "A" else db)]
        wa = next(([t0, min(t1, da - 0.01)] for s, t0, t1 in narrow if s == "A"), None)
        wb = next(([t0, min(t1, db - 0.01)] for s, t0, t1 in narrow if s == "B"), None)
        if (wa or wb) and len(jobs) < 2:
            jobs.append({"id": f"x{len(jobs)}", "anchor": None,
                         "wa": [_r(x, 3) for x in (wa or [0.0, da - 0.01])], "wb": [_r(x, 3) for x in (wb or [0.0, db - 0.01])]})
    return jobs


def _inside_found(memory: dict[str, Any], job: dict[str, Any]) -> bool:
    """这个锚点已落在前一个作业找到的匹配里（两侧都在区间内）。"""

    ta, tb = job["anchor"]
    for key, ext in (memory.get("extent") or {}).items():
        if key == job["id"] or not ext.get("found"):
            continue
        ra, rb = ext["match_a"], ext["match_b"]
        if ra[0] <= ta <= ra[1] and rb[0] <= tb <= rb[1]:
            return True
    return False


def _empty_compare_cover(state: HiState, wa: list[float]) -> list[float] | None:
    """这一轮已做过的限定范围 compare_media 里，A 范围盖住 wa、且一个对应段也没对上的那次的范围（同一引擎缩小范围也对不上）。"""

    for obs in state.observations:
        ra = (obs["params"] or {}).get("range_a_s")
        if (obs["operation"] == "compare_media" and ra and obs.get("status") == "ready"
                and not (obs.get("result") or {}).get("correspondences") and ra[0] <= wa[0] and ra[1] >= wa[1]):
            return ra
    return None


def _skip_compare(state: HiState, memory: dict[str, Any], job: dict[str, Any], ext: dict[str, Any]) -> bool:
    """锚点作业的窗口已被前一次空的限定范围比对盖住：记成已比（结果取那一次），不再发 F5。"""

    if not job["anchor"] or "align_media" in state.inspect_ops:
        return False
    if re.search(r"whole-video comparison found no correspondence", state.task, re.I):
        # 10-15 对话 5''（追加 5）：第 1 轮整段比对 0%，锚点附近缩窗再比大概率也是 0%（r0a vcsl_pair_1 两次都是）：直接以图找图
        _mark(memory, f"fu_align_{job['id']}")
        ext.update({"issued": True, "found": False, "compare_from": None, "skipped_no_correspondence": True})
        return True
    cover = _empty_compare_cover(state, job["wa"])
    if cover is None:
        return False
    _mark(memory, f"fu_align_{job['id']}")
    ext.update({"issued": True, "compare_from": cover})
    return True


def _extent_step(state: HiState, memory: dict[str, Any], job: dict[str, Any], a: str, b: str,
                 da: float, db: float, given: list[str], jobs: list[dict[str, Any]] | None = None) -> Decision | None:
    issued = _issued(memory)
    i = job["id"]
    ext = memory.setdefault("extent", {}).setdefault(i, {"anchor": job["anchor"], "window_a": job["wa"], "window_b": job["wb"]})
    wa, wb, anchor = job["wa"], job["wb"], job["anchor"]
    key = f"fu_align_{i}"
    if key not in issued and state.remaining >= 3 and _skip_compare(state, memory, job, ext):
        issued = _issued(memory)
    if key not in issued:
        if state.remaining < 3:
            return None
        _mark(memory, key)
        ext["issued"] = True
        if anchor:
            why = (f"Align only the neighbourhood of the correspondence A {anchor[0]:.3f} s ↔ B {anchor[1]:.3f} s (A {wa[0]:.1f}-"
                   f"{wa[1]:.1f} s, B {wb[0]:.1f}-{wb[1]:.1f} s) to see how far it extends on both sides.")
        else:
            why = (f"Align A {wa[0]:.3f}-{wa[1]:.3f} s against B {wb[0]:.3f}-{wb[1]:.3f} s only, to find a corresponding clip in the "
                   f"range the first round left unmatched.")
        if "align_media" in state.inspect_ops:
            return Decision("action", "F5_extent_align", _inspect([a, b], "align_media",
                            {"mode": "visual", "range_a_s": wa, "range_b_s": wb, "top_k": 3}, _existing(state, given, 2), why),
                            branch="INSPECT align_media(range)")
        return Decision("action", "F5_extent_compare", _inspect([a, b], "compare_media",
                        {"mode": "visual", "range_a_s": wa}, _existing(state, given, 2), why.replace(
                            f", B {wb[0]:.1f}-{wb[1]:.1f} s", "").replace(f" against B {wb[0]:.3f}-{wb[1]:.3f} s only", " only")),
                        branch="INSPECT compare_media(range)")
    align = next((o for o in reversed(state.observations) if o["operation"] in ("align_media", "compare_media")
                  and (o["params"] or {}).get("range_a_s") == (ext.get("compare_from") or wa)
                  and (o["operation"] == "compare_media" or (o["params"] or {}).get("range_b_s") == wb)), None)
    if align is None and not ext.get("skipped_no_correspondence"):
        return None
    match = _consistent_match(align, anchor[0], anchor[1]) if anchor else (_range_matches(align) or [None])[0]
    if match is not None and match.get("range_a_s") and match.get("range_b_s"):
        ra = [_r(float(x), 3) for x in match["range_a_s"]]
        rb = [_r(float(x), 3) for x in match["range_b_s"]]
        edge = (ra[0] <= wa[0] + 0.5 and wa[0] > 0) or (ra[1] >= wa[1] - 0.5 and wa[1] < da - 0.6)
        if align["operation"] == "align_media":           # compare_media 不限 B 的范围，B 侧没有窗口边
            edge = edge or (rb[0] <= wb[0] + 0.5 and wb[0] > 0) or (rb[1] >= wb[1] - 0.5 and wb[1] < db)
        ext.update({"match_a": ra, "match_b": rb, "edge": edge, "found": True})
        if f"fu_bB_{i}" not in issued and state.remaining >= 2:
            _mark(memory, f"fu_bB_{i}")
            times = _clip_times([rb[0] - 1.0, rb[0], rb[1] - 0.1, rb[1] + 1.0], db, 0.05)
            ext["frames_b"] = times
            images = [x for an in match.get("anchors") or [] for x in (an.get("image_a_evidence_id"), an.get("image_b_evidence_id")) if x]
            return Decision("action", "F6_extent_frames_B", _inspect([b], "get_frames", {"times_s": times},
                            _existing(state, images, state.limit // 2),
                            f"The windowed alignment puts the match at B {rb[0]:.3f}-{rb[1]:.3f} s; look just inside and outside "
                            f"both ends in {b}."), branch="INSPECT get_frames times_s")
        if f"fu_bA_{i}" not in issued and state.remaining >= 2:
            _mark(memory, f"fu_bA_{i}")
            times = _clip_times([ra[0] - 1.0, ra[0], ra[1] - 0.1, ra[1] + 1.0], da, 0.05)
            ext["frames_a"] = times
            mine = _frames_of(_frames_obs(state, b, ext.get("frames_b")))
            return Decision("action", "F6_extent_frames_A", _inspect([a], "get_frames", {"times_s": times},
                            _existing(state, [e for e, _ in mine], state.limit // 2),
                            f"Look at A {ra[0]:.3f}-{ra[1]:.3f} s just inside and outside both ends, next to the B frames."),
                            branch="INSPECT get_frames times_s")
        return None
    ext["found"] = False
    loc_op = "locate_frames" if "locate_frames" in state.inspect_ops else "locate"
    if not anchor or loc_op not in state.inspect_ops:
        return None
    # 锚点附近对齐不出一致的候选：以图找图（A 在锚点 ±3 s 的帧，各在 B 的锚点附近找，看位置是否随之平移）
    ta, tb = anchor
    if f"fu_locA_{i}" not in issued and state.remaining >= 3:
        # 10-15：后面的锚点窗口也被同一次空比对盖住时，它们的 ±3 s 探针帧一起取（同一段 A，省一步）
        group = [(ext, ta)]
        for other in jobs or []:
            if other["id"] == i or not other["anchor"] or f"fu_locA_{other['id']}" in issued or _inside_found(memory, other):
                continue
            oext = memory["extent"].setdefault(other["id"], {"anchor": other["anchor"], "window_a": other["wa"], "window_b": other["wb"]})
            if f"fu_align_{other['id']}" not in _issued(memory) and not _skip_compare(state, memory, other, oext):
                continue
            if oext.get("found") is None and oext.get("compare_from"):
                oext["found"] = False
            if oext.get("found") is False:
                _mark(memory, f"fu_locA_{other['id']}")
                group.append((oext, other["anchor"][0]))
        _mark(memory, f"fu_locA_{i}")
        times: list[float] = []
        for e, t0 in group:
            e["probe_own"] = _clip_times([t0 - 3.0, t0 + 3.0], da, 0.05)
            # 锚点那一帧也一起取（不 locate）：夹边界时它是"里面"那一帧，与 B 侧锚点帧成对给报告看（r0f vcsl_pair_1：
            # 第 2 轮没有锚点帧，S4 无法用里面 / 外面一对帧核边界）
            e["anchor_a"] = _clip_times([t0], da, 0.05)
            times += e["probe_own"] + e["anchor_a"]
        times = sorted(set(times))
        for e, _ in group:
            e["probe_a"] = times
        if len(group) > 1:
            listed = ", ".join(f"{t0:.3f}" for _, t0 in group)
            why = (f"The windowed comparison found no match consistent with the anchors A {listed} s; take A frames 3 s "
                   f"before and after each anchor to look them up in B.")
        else:
            why = (f"The windowed alignment found no match consistent with A {ta:.3f} s ↔ B {tb:.3f} s; take A frames 3 s "
                   f"before and after the anchor to look them up in B.")
        return Decision("action", "F7_extent_probe_frames_A", _inspect([a], "get_frames", {"times_s": times}, [], why),
                        branch="INSPECT get_frames times_s")
    own = ext.get("probe_own") or ext.get("probe_a") or []
    probes = [(e, t) for e, t in _frames_of(_frames_obs(state, a, ext.get("probe_a"))) if any(abs(t - x) < 0.25 for x in own)]
    for eid, t in probes:
        if f"fu_loc_{i}_{eid}" in issued or state.remaining < 2:
            continue
        _mark(memory, f"fu_loc_{i}_{eid}")
        return Decision("action", "F7_extent_locate", _inspect([b], loc_op,
                        {"query": {"type": "image", "evidence_id": eid}, "range_s": wb, "top_k": 3}, _existing(state, [eid], 1),
                        f"Find {eid} (A at {t:.3f} s) in B near {tb:.3f} s; if the correspondence continues it should appear near "
                        f"B {tb + (t - ta):.3f} s."), branch=f"INSPECT {loc_op}(image,range)")
    return None


_BRACKET_SETTLED_S = 3.0      # 每侧"里面对得上 / 外面对不上"的两个探针相距不超过这个，就算量出了起止（宽度写进 notes）
_BRACKET_TARGET_S = 1.0       # 预算够时继续二分到这个宽度


def _probe_points(state: HiState, memory: dict[str, Any], i: str, ext: dict[str, Any], a: str) -> list[dict[str, Any]]:
    """锚点作业 i 的以图找图探针（±3 s 的和之后夹边界的）：d = 探针 A 时刻 − 锚点 A 时刻；ok = 在 B 里找到的位置
    离"按锚点平移"的位置不超过 _consistent_tol(d)（B 窗口里一个候选也没有记为不一致）。"""

    ta, tb = ext["anchor"]
    batches = [(ext.get("probe_a"), ext.get("probe_own") or ext.get("probe_a") or [])]
    batches += [(bt["times"], [p["t"] for p in bt["probes"] if p["job"] == i]) for bt in memory.get("bracket_batches") or []]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for times, mine in batches:
        if not times or not mine:
            continue
        for eid, t in _frames_of(_frames_obs(state, a, times)):
            if eid in seen or not any(abs(t - x) < 0.25 for x in mine):
                continue
            obs = next((o for o in state.observations if o["operation"] in ("locate_frames", "locate") and o.get("status") == "ready"
                        and (o["params"] or {}).get("range_s") == ext["window_b"]
                        and ((o["params"].get("query") or {}).get("evidence_id")) == eid), None)
            if obs is None:
                continue
            seen.add(eid)
            cands = sorted((obs.get("result") or {}).get("candidates") or [], key=lambda c: -float(c.get("score") or 0))
            got = _num(cands[0].get("time_s")) if cands else None
            want = tb + (t - ta)
            out.append({"d": _r(t - ta, 3), "t": t, "eid": eid, "got": got, "want": want,
                        "ok": got is not None and abs(got - want) <= _consistent_tol(t - ta)})
    return sorted(out, key=lambda p: p["t"])


def _consistent_tol(d: float) -> float:
    """以图找图的位置离"按锚点平移"的位置多近算一致：至少 1 s，离锚点越远放得越宽（20%）。重剪辑素材有轻微变速，
    r0f vcsl_pair_1 里 A 15.967 s 落在 B 16.96 s（偏 1.12 s，探针离锚点 6 s），画面确是同一场景，固定 1 s 把它判成了不一致。"""

    return max(1.0, 0.2 * abs(d))


def _brackets(state: HiState, ext: dict[str, Any], points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """锚点两侧（side −1 = 起点、+1 = 终点）各一个夹区间：从锚点往外，最后一个一致的探针（锚点本身算 0）与其后第一个不一致的。
    cap = 这一侧还能探到多远（窗口 ±10 s、A / B 的首尾）；一致到 cap 且 cap 是视频首尾时，起止就是那个首尾（edge）。"""

    ta, tb = ext["anchor"]
    sides = _material_sides(state)
    da = state.profile(sides.get("A") or "").get("duration") or 0.0
    db = state.profile(sides.get("B") or "").get("duration") or 0.0
    out = []
    for side in (-1, 1):
        limits = ([(_EXTENT_WINDOW_S, None), (ta - 0.05, "start of Material A"), (tb, "start of Material B")] if side < 0 else
                  [(_EXTENT_WINDOW_S, None), (da - 0.05 - ta, "end of Material A"), (db - 0.05 - tb, "end of Material B")])
        cap, edge_name = min(limits, key=lambda x: x[0])
        inside, outside = 0.0, None
        for p in sorted((p for p in points if p["d"] * side > 0.05), key=lambda p: abs(p["d"])):
            if p["ok"]:
                inside = abs(p["d"])
            else:
                outside = abs(p["d"])
                break
        at_cap = outside is None and inside >= cap - 0.3
        out.append({"side": side, "inside": inside, "outside": outside, "cap": max(0.0, cap), "at_cap": at_cap,
                    "edge": edge_name if at_cap else None, "tried": sorted(abs(p["d"]) for p in points if p["d"] * side > 0.05)})
    return out


def _next_probe(br: dict[str, Any]) -> float | None:
    """这一侧下一个探针离锚点多远：还没出界就往外加倍（至少 +3 s），夹住了且宽于 1 s 就二分；都不用就 None。"""

    if br["outside"] is None:
        if br["at_cap"]:
            return None
        d = min(br["cap"], max(br["inside"] * 2, br["inside"] + 3.0))
    elif br["outside"] - br["inside"] > _BRACKET_TARGET_S:
        d = (br["inside"] + br["outside"]) / 2
    else:
        return None
    if d - br["inside"] < 0.3 or any(abs(d - x) < 0.1 for x in br["tried"]):
        return None
    return _r(d, 3)


def _bracket_step(state: HiState, memory: dict[str, Any], a: str, b: str, da: float, db: float) -> Decision | None:
    """窗口比对对不上的锚点（±3 s 探针已在 B 里找过）：
      F8_bracket_frames_A：按 _next_probe 一次取一批 A 帧（各锚点各侧一起；先补还没出界的一侧，再二分最宽的）；
      F8_bracket_locate：逐帧在 B 的锚点窗口里找；
      F8_bracket_frames_B：夹完后取 B 侧"按锚点平移"的对应时刻（每侧里面 / 外面各一），与 A 探针帧成对给报告看。
    预算：每批留出取帧、B 侧成对帧、报告各一步。"""

    loc_op = "locate_frames" if "locate_frames" in state.inspect_ops else "locate"
    exts = [(i, e) for i, e in sorted((memory.get("extent") or {}).items())
            if e.get("anchor") and e.get("issued") and e.get("found") is False and e.get("probe_a")]
    if not exts or loc_op not in state.inspect_ops:
        return None
    issued = _issued(memory)
    for bt in memory.get("bracket_batches") or []:
        for eid, t in _frames_of(_frames_obs(state, a, bt["times"])):
            probe = min(bt["probes"], key=lambda p: abs(p["t"] - t))
            if abs(probe["t"] - t) >= 0.25 or f"fu_bloc_{eid}" in issued or state.remaining < 2:
                continue
            ext = memory["extent"][probe["job"]]
            ta, tb = ext["anchor"]
            _mark(memory, f"fu_bloc_{eid}")
            return Decision("action", "F8_bracket_locate", _inspect([b], loc_op,
                            {"query": {"type": "image", "evidence_id": eid}, "range_s": ext["window_b"], "top_k": 3},
                            _existing(state, [eid], 1),
                            f"Find {eid} (A at {t:.3f} s) in B; if the correspondence A {ta:.3f} s ↔ B {tb:.3f} s still holds there, it "
                            f"should appear near B {tb + (t - ta):.3f} s."), branch=f"INSPECT {loc_op}(image,range)")
    a_name = _material_sides(state).get("A") or ""
    found = {i: (e, _probe_points(state, memory, i, e, a_name)) for i, e in exts}
    brackets = {i: _brackets(state, e, pts) for i, (e, pts) in found.items() if pts}
    pairs_pending = "fu_pairB" not in issued
    budget = state.remaining - 2 - (1 if pairs_pending else 0)
    wants = [(i, br, d) for i, brs in brackets.items() for br in brs for d in [_next_probe(br)] if d is not None]
    wants.sort(key=lambda w: (w[1]["outside"] is not None, -((w[1]["outside"] or 0) - w[1]["inside"]), w[0], w[1]["side"]))
    if wants and budget >= 1 and pairs_pending:              # B 侧成对帧取过后不再夹（成对帧要对应最终的区间）
        probes = []
        for i, br, d in wants[:min(budget, 4)]:
            t = _clip_times([memory["extent"][i]["anchor"][0] + br["side"] * d], da, 0.05)
            if t and not any(abs(p["t"] - t[0]) < 0.05 for p in probes):
                probes.append({"job": i, "side": br["side"], "t": t[0]})
        if probes:
            batches = memory.setdefault("bracket_batches", [])
            _mark(memory, f"fu_bracket_{len(batches)}")
            times = sorted({p["t"] for p in probes})
            batches.append({"times": times, "probes": probes})
            return Decision("action", "F8_bracket_frames_A", _inspect([a], "get_frames", {"times_s": times}, [],
                            "Take A frames between the last instants that still match and the first that no longer match (or further "
                            "out where every lookup still matched), to narrow down where each correspondence starts and ends."),
                            branch="INSPECT get_frames times_s")
    if pairs_pending and state.remaining >= 2:
        # 成对的先后（B 帧最多 8 张、回看最多 4 对）：每个锚点先锚点本身、再各侧的"外面"一帧，最后是不在锚点的"里面"一帧
        first: list[tuple[float, float, list[str]]] = []
        later: list[tuple[float, float, list[str]]] = []
        for i, brs in brackets.items():
            ext_i = memory["extent"][i]
            (ta, tb), pts = ext_i["anchor"], found[i][1]
            anchor_eids = [e for e, t in _frames_of(_frames_obs(state, a, ext_i.get("probe_a")))
                           if any(abs(t - x) < 0.05 for x in ext_i.get("anchor_a") or [])]
            if anchor_eids:
                first.append((_r(ta, 3), _r(tb, 3), anchor_eids[:1]))
            for br in brs:
                for dist, bucket in ((br["outside"], first), (br["inside"], later)):
                    if not dist:
                        continue
                    bucket.append((_r(ta + br["side"] * dist, 3), _r(tb + br["side"] * dist, 3),
                                   [p["eid"] for p in pts if abs(p["d"] - br["side"] * dist) < 0.25]))
        pairs: list[tuple[float, float]] = []
        a_frames: list[str] = []
        for t_a, t_b, eids in first + later:
            if (t_a, t_b) in pairs or not _clip_times([t_b], db, 0.05) or len({x for _, x in pairs} | {t_b}) > 8:
                continue
            pairs.append((t_a, t_b))
            a_frames += eids
        times = _clip_times(sorted({t_b for _, t_b in pairs}), db, 0.05)
        if times:
            _mark(memory, "fu_pairB")
            memory["bracket_pairs"] = [[t_a, t_b] for t_a, t_b in pairs]
            memory["bracket_pairs_b"] = times
            memory.setdefault("followup", []).append(f"Material B at the anchors' offsets {', '.join(f'{t:.3f}' for t in times)} s, "
                                                     f"paired with the A lookup frames")
            return Decision("action", "F8_bracket_frames_B", _inspect([b], "get_frames", {"times_s": times},
                            _existing(state, a_frames, state.limit // 2),
                            "Take B frames at the anchor's offset from the A lookup frames on both sides of each boundary, to view the "
                            "pairs side by side."), branch="INSPECT get_frames times_s")
    return None


def _extent_findings(state: HiState, memory: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    """锚点外延的结果 → (notes, 缺口段, 是否都确定了)。"""

    notes: list[str] = []
    gaps: list[str] = []
    settled = True
    for i, ext in sorted((memory.get("extent") or {}).items()):
        wa, wb = ext.get("window_a"), ext.get("window_b")
        if ext.get("anchor"):
            ta, tb = ext["anchor"]
            name = f"correspondence A {ta:.3f} s ↔ B {tb:.3f} s"
        else:
            ta = tb = None
            name = f"Material A {wa[0]:.3f}–{wa[1]:.3f} s against Material B {wb[0]:.3f}–{wb[1]:.3f} s"
        if not ext.get("issued"):
            settled = False
            gaps.append(f"{name}: not examined (turn budget)")
            continue
        if ext.get("found"):
            ra, rb = ext["match_a"], ext["match_b"]
            notes.append(f"{name}: windowed alignment (A {wa[0]:.3f}-{wa[1]:.3f} s, B {wb[0]:.3f}-{wb[1]:.3f} s) gives a matching "
                         f"clip A {ra[0]:.3f}-{ra[1]:.3f} s ↔ B {rb[0]:.3f}-{rb[1]:.3f} s (an estimate; the boundary frames show "
                         f"whether it holds)")
            if ext.get("edge"):
                settled = False
                gaps.append(f"{name}: the match reaches the edge of the examined window (A {wa[0]:.3f}-{wa[1]:.3f} s), so it may "
                            f"continue beyond it")
            if not (ext.get("frames_a") and ext.get("frames_b")):
                settled = False
                gaps.append(f"{name}: boundary frames not examined on both sides")
            continue
        if ta is None:
            settled = False
            notes.append(f"{name}: the range-limited alignment returned no matching clip")
            gaps.append(f"{name}: no corresponding clip found by alignment within these ranges; only sampled frames could show "
                        f"a correspondence there")
            continue
        points = _probe_points(state, memory, i, ext, _material_sides(state).get("A") or "")
        for p in points:
            where = f"located in B at {p['got']:.3f} s" if p["got"] is not None else "not found in B near there"
            notes.append(f"{name}: A at {p['t']:.3f} s {where} (expected about {p['want']:.3f} s if the "
                         f"correspondence continued) — {'consistent' if p['ok'] else 'not consistent'}")
        notes.append(f"{name}: windowed alignment found no match consistent with the anchor")
        if not points:
            settled = False
            gaps.append(f"{name}: no continuous correspondence could be measured around this anchor; only the sampled instants "
                        f"are supported")
            continue
        # 10-15（B6）：每侧按"里面还对得上 / 外面对不上"的一对探针给出起止所在的区间；夹不出的一侧才是缺口
        side_settled = True
        for br in _brackets(state, ext, points):
            s, word = br["side"], ("start" if br["side"] < 0 else "end")
            a_in, b_in = ta + s * br["inside"], tb + s * br["inside"]
            inside_what = "the anchor" if br["inside"] == 0 else "still matching"
            if br["outside"] is not None:
                a_out, b_out = ta + s * br["outside"], tb + s * br["outside"]
                width = br["outside"] - br["inside"]
                lo, hi = (f"A {a_out:.3f} s (no longer matching)", f"A {a_in:.3f} s ({inside_what})") if s < 0 else \
                         (f"A {a_in:.3f} s ({inside_what})", f"A {a_out:.3f} s (no longer matching)")
                notes.append(f"{name}: the {word} lies between {lo} and {hi}, i.e. between B {min(b_in, b_out):.3f} and "
                             f"{max(b_in, b_out):.3f} s at the anchor's offset; the image lookups bracket it to {width:.3f} s")
                if width > _BRACKET_SETTLED_S + 0.05:
                    side_settled = False
                    gaps.append(f"{name}: the {word} lies between A {min(a_in, a_out):.3f} and {max(a_in, a_out):.3f} s, not "
                                f"narrowed further (turn budget)")
            elif br["edge"]:
                notes.append(f"{name}: still matching at A {a_in:.3f} s ↔ B {b_in:.3f} s, at the {br['edge']}; the correspondence "
                             f"runs to that edge")
            else:
                side_settled = False
                gaps.append(f"{name}: the {word} was not reached — still matching at A {a_in:.3f} s ↔ B {b_in:.3f} s"
                            + (", the edge of the examined range" if br["at_cap"] else " (turn budget)"))
        settled = settled and side_settled
    return notes, gaps, settled


def _followup_report(state: HiState, memory: dict[str, Any], windows: list[tuple[str, float, float]], audio: list[str],
                     anchors: list[tuple[float, float]], sides: dict[str, str], dur: dict[str, Any], *, last_turn: bool) -> Decision:
    issued = _issued(memory)
    ext_notes, gaps, settled = _extent_findings(state, memory)
    jobs = _extent_jobs(state, windows, anchors, dur)
    for job in jobs:
        if job["id"] not in (memory.get("extent") or {}) and not (job["anchor"] and _inside_found(memory, job)):
            settled = False
            what = (f"correspondence A {job['anchor'][0]:.3f} s ↔ B {job['anchor'][1]:.3f} s" if job["anchor"]
                    else f"Material A {job['wa'][0]:.3f}–{job['wa'][1]:.3f} s against Material B {job['wb'][0]:.3f}–{job['wb'][1]:.3f} s")
            gaps.append(f"{what}: not examined (turn budget)")
    if jobs:
        windows = []
    for side, t0, t1 in windows:
        d = dur.get(side)
        if not sides.get(side) or (d and t1 - t0 >= 0.8 * d):
            continue
        if f"fu_{side}_{t0:.3f}" not in issued:
            settled = False
            gaps.append(f"Material {side} {t0:.3f}–{t1:.3f} s: not examined (turn budget)")
    for side in audio:
        if f"fu_asr_{side}" not in issued:
            settled = False
            gaps.append(f"audio of Material {side}: not transcribed")
    if not memory.get("extent") and not any(k.startswith("fu_") for k in issued):
        settled = False
        gaps.append("the follow-up task named no specific range, anchor or audio track that could be examined further")
    status = "COMPLETED" if settled and not gaps else "INCOMPLETE"
    notes = _followup_notes(memory) + ext_notes
    return _report(state, memory, "F9_last_turn_report" if last_turn else "F4_followup_report", status=status,
                   reason="; ".join(gaps) if gaps else None, notes=notes,
                   branch="report_completed" if status == "COMPLETED" else "report_incomplete")


def _review_picks(state: HiState, memory: dict[str, Any], frames: list[str]) -> list[str]:
    """回看：锚点外延的 A/B 边界帧按时间成对排；没有就均匀挑。"""

    a, b = _material_sides(state).get("A"), _material_sides(state).get("B")
    pairs: list[str] = []
    for ext in (memory.get("extent") or {}).values():
        fa = _frames_of(_frames_obs(state, a, ext.get("frames_a"))) if a else []
        fb = _frames_of(_frames_obs(state, b, ext.get("frames_b"))) if b else []
        for (ea, _), (eb, _) in zip(fa, fb):
            pairs += [ea, eb]
    if a and b and memory.get("bracket_pairs"):
        # 夹边界：A 探针帧与 B 侧按锚点平移的帧成对（10-15）
        fa = [(e, t) for o in state.observations if o["operation"] == "get_frames" and o["targets"] == [a] for e, t in _frames_of(o)]
        fb = _frames_of(_frames_obs(state, b, memory.get("bracket_pairs_b")))
        for t_a, t_b in memory["bracket_pairs"]:
            ea = next((e for e, t in fa if abs(t - t_a) < 0.05), None)
            eb = next((e for e, t in fb if abs(t - t_b) < 0.05), None)
            if ea and eb:
                pairs += [ea, eb]
    if not pairs and a and b:
        # 时间窗跟进：两侧同一时刻的帧配成对，按时间均匀挑
        fa = {round(t, 3): e for o in state.observations if o["operation"] == "get_frames" and o["targets"] == [a] for e, t in _frames_of(o)}
        fb = {round(t, 3): e for o in state.observations if o["operation"] == "get_frames" and o["targets"] == [b] for e, t in _frames_of(o)}
        common = sorted(set(fa) & set(fb))
        if common:
            n = max(1, state.limit // 2)
            chosen = common if len(common) <= n else [common[round(k * (len(common) - 1) / (n - 1))] for k in range(n)]
            for t in chosen:
                pairs += [fa[t], fb[t]]
    picks = _existing(state, pairs, state.limit)
    return picks if len(picks) >= 2 else _spread(frames, state.limit)


def _spread(ids: list[str], cap: int) -> list[str]:
    if len(ids) <= cap:
        return ids
    step = (len(ids) - 1) / (cap - 1)
    return [ids[round(i * step)] for i in range(cap)]


def _followup_notes(memory: dict[str, Any]) -> list[str]:
    notes = ["this is a follow-up round: answer only the correspondences / windows / audio named in the task, citing the new "
             "frames and tool results; say for each first-round gap whether it is now settled, and give the measured ranges"]
    notes += [f"examined: {x}" for x in memory.get("followup") or []]
    return notes


def summarize_decision(decision: Decision) -> str:
    return json.dumps({"rule": decision.rule, "branch": decision.branch, "kind": decision.kind}, ensure_ascii=False)
