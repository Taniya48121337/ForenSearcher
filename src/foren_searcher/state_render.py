"""状态渲染器：把送进模型的输入（payload）压到固定的字符预算内，同时不丢掉 E/R 编号。

大白话：以前每一轮送给模型的"桌面"平均 2.5 万字符、最长 7.6 万，其中大头是三样重复的东西——
每轮都原样附上的工具参数 schema（6.6K，永远一样）、历史里每一步搜索结果的完整快照（每步 6–9K，
其实模型只需要"当时搜了什么、拿到哪些 R/E"）、以及证据正文在报告附件里的整页复制。压缩的原则：

1. 编号不丢：所有 E、R、C、F 编号原样保留；模型要看正文可以 READ_RECORD。
2. 最新一步最详细：latest_observation 保留 details（正文按档位截断），历史步只留一行摘要 + 摘要行。
3. 常量不重复：available_actions 的 schema 换成一行签名（名字、必填参数、枚举值）。
4. 超预算时按"可恢复优先"丢：先丢最旧的历史步（保留焦点证据的读取步），再截更多正文，最后标 context_truncated。

渲染是纯函数，不改传入的对象；同一输入 + 同一档位 = 同一输出，训练导出时可以从记录的
model_input 事件重放。档位名写进 runtime_constraints.state_renderer 与规格的 harness_params，
以便每行训练数据都能追溯它是按哪档渲染的。

档位（预算是 user 消息的字符数，不含 system prompt；按试点语料 1 token ≈ 1.7 字符估算）：
    legacy        不压缩（旧行为）
    compact_8k    ≤ 11 000 字符（整行 ≈ 8K token）
    compact_16k   ≤ 24 000 字符（整行 ≈ 16K token）
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

RENDERER_VERSION = "1"


@dataclass(frozen=True)
class Tier:
    name: str
    budget_chars: int          # 整个 payload 序列化后的上限
    latest_text_chars: int     # latest_observation / 报告附件里单条正文的上限
    history_items: int         # 历史步的 items 摘要行上限
    history_excerpt_chars: int  # 历史步 READ_RECORD 正文摘录长度
    list_cap: int              # 结果列表、反馈列表的条数上限
    snippet_chars: int         # 搜索摘要、标签等短文本上限
    support_text_chars: int    # GIA 报告附件 supporting_records 单条正文上限；0 = 只留编号和标签，正文靠 READ_RECORD
    report_visual_summary: bool  # GIA 是否保留每份报告自带的 visual_observation_summary（全局的那份总保留）


TIERS: dict[str, Tier | None] = {
    "legacy": None,
    "compact_8k": Tier("compact_8k", 11_000, 3_000, 4, 160, 6, 160, 0, False),
    "compact_16k": Tier("compact_16k", 24_000, 6_000, 12, 240, 10, 240, 1_200, True),
}

TRUNCATION_MARK = "…[已截断 {n} 字符；READ_RECORD 可读全文]"
LIST_MARK = "…（另有 {n} 条未展示）"
_NOTICE_KEYS = ("content_notice",)   # 长段固定说明文字：压缩档只保留一个短标记
_DROP_KEYS = ("record_type", "artifact_ref", "production_method")   # 卡片里对决策无用的常量字段


class RenderError(ValueError):
    """渲染无法把输入压进预算（任务/材料/最新观察本身就超限）。chars / budget：超限时渲染后的字符数与预算。"""

    def __init__(self, message: str, *, chars: int | None = None, budget: int | None = None) -> None:
        super().__init__(message)
        self.chars = chars
        self.budget = budget


def tier_for(name: str | None) -> Tier | None:
    if name is None:
        return None
    if name not in TIERS:
        raise RenderError(f"未知的 state_renderer: {name!r}；可选 {', '.join(TIERS)}")
    return TIERS[name]


def payload_chars(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))


# --------------------------------------------------------------------------- 通用裁剪

def cut_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARK.format(n=len(text) - limit)


def cut_list(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    return [*items[:limit], LIST_MARK.format(n=len(items) - limit)]


def shrink(value: Any, *, text_chars: int, list_cap: int, snippet_chars: int) -> Any:
    """递归：长字符串截断、长列表截断、固定说明文字换短标记。编号类短字符串不受影响。"""

    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}") and len(value) < 400:
            try:
                return json.dumps(shrink(json.loads(value), text_chars=text_chars, list_cap=list_cap, snippet_chars=snippet_chars),
                                  ensure_ascii=False, separators=(",", ":"))
            except ValueError:
                pass
        return cut_text(value, text_chars)
    if isinstance(value, list):
        return [shrink(v, text_chars=text_chars, list_cap=list_cap, snippet_chars=snippet_chars)
                for v in cut_list(value, list_cap)]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in _DROP_KEYS or item is None or item == [] or item == {}:
                continue
            if key in _NOTICE_KEYS and isinstance(item, str):
                out[key] = "media_metadata_not_pixels" if "像素" in item else cut_text(item, snippet_chars)
            elif key in ("search_snippet", "label", "search_title", "message") and isinstance(item, str):
                out[key] = cut_text(item, snippet_chars)
            else:
                out[key] = shrink(item, text_chars=text_chars, list_cap=list_cap, snippet_chars=snippet_chars)
        return out
    return value


_MEDIA_PROFILE_BOILERPLATE = "；本记录未核验日期、地点或作者等来源元数据，不能据此判断这些信息不存在。"


def card_line(card: Any, tier: Tier) -> Any:
    """一张卡片压成一行字符串：编号 [模态] 标签 ← 来源/父证据 (获取方式)。非卡片原样返回。

    例：E12 [text] Fire near Eiffel Tower … ← R8 (SEARCH/open)
        E7 [image] 视频E1在14.133秒的帧 ← E1 (INSPECT/get_frames)
        R9 page: Eiffel Tower Fire 'Accidental' — https://… — snippet…
    """

    if not isinstance(card, dict):
        return card
    eid = card.get("evidence_id")
    cid = card.get("candidate_id")
    if isinstance(cid, str) and not eid:
        parts = [cid]
        if card.get("candidate_type"):
            parts[0] += f" {card['candidate_type']}:"
        if card.get("title"):
            parts.append(cut_text(str(card["title"]), tier.snippet_chars))
        if card.get("source_url"):
            parts.append(str(card["source_url"]))
        if card.get("snippet") and tier.support_text_chars > 0:
            parts.append(cut_text(str(card["snippet"]), tier.snippet_chars))
        return " — ".join(parts)
    if not isinstance(eid, str):
        return shrink(card, text_chars=tier.latest_text_chars, list_cap=64, snippet_chars=tier.snippet_chars)
    label = str(card.get("label") or "")
    if label.startswith("媒体属性：") and label.endswith(_MEDIA_PROFILE_BOILERPLATE):
        label = label[: -len(_MEDIA_PROFILE_BOILERPLATE)]
    modalities = card.get("modalities") or []
    line = eid
    if modalities:
        line += " [" + "/".join(str(m) for m in modalities) + "]"
    if label:
        line += " " + cut_text(label, tier.snippet_chars)
    origin = [str(x) for x in (card.get("derived_from") or []) if isinstance(x, str)]
    if isinstance(card.get("source_ref"), str) and card["source_ref"] not in origin:
        origin.append(card["source_ref"])
    if origin:
        line += " ← " + ",".join(origin)
    if isinstance(card.get("source_workspace"), str):
        line += f" @{card['source_workspace']}"
    if isinstance(card.get("acquired_by"), str) and tier.support_text_chars > 0:
        line += f" ({card['acquired_by']})"
    extra = {k: v for k, v in card.items() if k in ("retrieval", "content_offset", "next_offset") and v not in (None, 0)}
    if extra:
        line += " " + json.dumps(shrink(extra, text_chars=tier.snippet_chars, list_cap=8, snippet_chars=tier.snippet_chars),
                                 ensure_ascii=False, separators=(",", ":"))
    return line


def compact_cards(cards: Any, tier: Tier) -> Any:
    """证据卡片 / 候选卡片列表 → 每张一行；HI 的 materials 字典保留键（C0 的任务正文原样）。"""

    if isinstance(cards, dict):          # HI 的 materials 是 {"C0": card, "E1": card}
        return {key: (card_line(card, tier) if isinstance(card, dict) and "evidence_id" in card
                      else shrink(card, text_chars=tier.latest_text_chars, list_cap=64, snippet_chars=tier.snippet_chars))
                for key, card in cards.items()}
    if isinstance(cards, list):
        return [card_line(card, tier) for card in cards]
    return cards


# --------------------------------------------------------------------------- available_actions

def _schema_signature(schema: Any) -> str:
    """把 JSON schema 压成一行：必填键(类型/枚举)，可选键加 ?；oneOf 用 | 连接。"""

    if not isinstance(schema, dict):
        return ""
    if "oneOf" in schema:
        return " | ".join(_schema_signature(s) for s in schema["oneOf"])
    if "anyOf" in schema:
        return " | ".join(_schema_signature(s) for s in schema["anyOf"])
    if schema.get("type") == "object" or "properties" in schema:
        required = set(schema.get("required", ()))
        parts = []
        for key, sub in (schema.get("properties") or {}).items():
            parts.append(f"{key}{'' if key in required else '?'}:{_schema_signature(sub) or _type_name(sub)}")
        return "{" + ", ".join(parts) + "}"
    if schema.get("type") == "array":
        return f"[{_schema_signature(schema.get('items')) or _type_name(schema.get('items'))}]"
    if "enum" in schema:
        return "|".join(str(v) for v in schema["enum"])
    if "const" in schema:
        return str(schema["const"])
    return _type_name(schema)


def _type_name(schema: Any) -> str:
    if isinstance(schema, dict):
        t = schema.get("type")
        if isinstance(t, list):
            return "/".join(str(x) for x in t)
        if t:
            return str(t)
    return "any"


def compact_actions(actions: list[Any]) -> list[Any]:
    """schema → signature；INSPECT 的 operation_catalog 每项只留名字和一句用途。"""

    out = []
    for action in actions:
        if not isinstance(action, dict):
            out.append(action)
            continue
        item: dict[str, Any] = {k: v for k, v in action.items() if k not in ("schema", "operation_catalog")}
        if "schema" in action:
            item["signature"] = _schema_signature(action["schema"])
        catalog = action.get("operation_catalog")
        if isinstance(catalog, dict):
            item["operations"] = {
                name: (cut_text(str(spec.get("description", "")), 80) if isinstance(spec, dict) else "")
                for name, spec in catalog.items()
            }
        out.append(item)
    return out


# --------------------------------------------------------------------------- 观察

def _digest_observation(observation: Any, tier: Tier) -> Any:
    """历史步的观察：action + items 摘要行 + 最小 details 摘要；完整 details 只在最新一步保留。"""

    if not isinstance(observation, dict):
        return observation
    digest: dict[str, Any] = {}
    if "action" in observation:
        digest["action"] = observation["action"]
    items = observation.get("items")
    if isinstance(items, list):
        digest["items"] = [cut_text(str(i), tier.snippet_chars) for i in cut_list(items, tier.history_items)]
    details = observation.get("details")
    action = observation.get("action")
    if isinstance(details, dict):
        if action == "READ_RECORD":
            summary = {k: details.get(k) for k in ("evidence_id", "content_offset", "next_offset") if k in details}
            content = details.get("content")
            if isinstance(content, str):
                summary["excerpt"] = cut_text(content, tier.history_excerpt_chars)
            digest["details"] = summary
        elif action == "INSPECT":
            summary = {k: details.get(k) for k in ("operation", "status", "result_evidence_id") if k in details}
            feedback = details.get("feedback")
            if isinstance(feedback, list) and feedback:
                summary["feedback"] = [cut_text(str(f.get("message", f)), tier.snippet_chars) if isinstance(f, dict) else cut_text(str(f), tier.snippet_chars)
                                       for f in cut_list(feedback, 3)]
            result = details.get("result")
            if isinstance(result, (str, int, float)):
                summary["result"] = cut_text(str(result), tier.history_excerpt_chars)
            elif isinstance(result, (dict, list)):
                summary["result"] = cut_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), tier.history_excerpt_chars)
            digest["details"] = summary
        elif action == "SEARCH":
            summary = {k: details.get(k) for k in ("output_type", "stop_reason") if k in details}
            digest["details"] = summary
        elif action == "HARNESS_REJECTION":
            digest["details"] = shrink(details, text_chars=tier.snippet_chars * 2, list_cap=3, snippet_chars=tier.snippet_chars)
        # QUERY_MEMORY 等：items 已列出编号，details 不保留
    for key in ("record", "error", "preview"):
        if key in observation:
            digest[key] = shrink(observation[key], text_chars=tier.history_excerpt_chars, list_cap=3, snippet_chars=tier.snippet_chars)
    for key in ("truncated", "evidence_ids"):
        if key in observation:
            digest[key] = observation[key]
    return digest


def _compact_search_result(result: Any, tier: Tier) -> Any:
    """搜索结果：候选编号、标题、摘要、已产生的 E 编号、失败原因；materials 的状态机细节不保留。"""

    if not isinstance(result, dict):
        return result
    out = {k: result[k] for k in ("candidate_id", "source_url", "search_title", "search_snippet") if result.get(k)}
    evidence: list[str] = []
    errors: list[str] = []
    for material in result.get("materials") or []:
        if not isinstance(material, dict):
            continue
        for card in material.get("evidence") or []:
            if isinstance(card, dict) and isinstance(card.get("evidence_id"), str):
                evidence.append(f"{card['evidence_id']}:{material.get('kind', '?')}")
        error = material.get("error")
        if isinstance(error, dict) and material.get("status") == "failed":
            errors.append(f"{material.get('kind', '?')}:{error.get('code') or error.get('message') or 'failed'}")
    if evidence:
        out["evidence"] = evidence
    if errors:
        out["failed"] = errors
    return shrink(out, text_chars=tier.latest_text_chars, list_cap=tier.list_cap, snippet_chars=tier.snippet_chars)


def _latest_observation(observation: Any, tier: Tier) -> Any:
    if isinstance(observation, dict) and observation.get("action") == "SEARCH" and isinstance(observation.get("details"), dict):
        observation = dict(observation)
        details = dict(observation["details"])
        if isinstance(details.get("results"), list):
            details["results"] = [_compact_search_result(r, tier) for r in cut_list(details["results"], tier.list_cap)]
        observation["details"] = details
    return shrink(observation, text_chars=tier.latest_text_chars, list_cap=tier.list_cap, snippet_chars=tier.snippet_chars)


def _slim_visual_summary(view: dict[str, Any]) -> None:
    """视觉摘要只留已观察区间（未确认区间是它的补集，模型可自行推出）。"""

    summary = view.get("visual_observation_summary")
    if isinstance(summary, dict) and isinstance(summary.get("videos"), list):
        for video in summary["videos"]:
            if isinstance(video, dict):
                video.pop("unconfirmed_ranges_s", None)
                video.pop("unknown_duration_frame_times_s", None)


def _ids_only(container: dict[str, Any], key: str) -> None:
    items = container.get(key)
    if isinstance(items, list):
        container[key] = [item.get("evidence_id", item) if isinstance(item, dict) else item for item in items]


# --------------------------------------------------------------------------- HI

def _history_priority(entry: dict[str, Any], focus: set[str]) -> int:
    """丢历史时的保留优先级：读过焦点证据的步 > 报错/被拒的步 > 其他。数值越大越先丢。"""

    text = json.dumps({k: entry.get(k) for k in ("request", "observation")}, ensure_ascii=False, default=str)
    if focus and any(eid in text for eid in focus):
        return 0
    if entry.get("action") == "HARNESS_REJECTION":
        return 1
    return 2


def render_hi_payload(payload: dict[str, Any], renderer: str | None) -> dict[str, Any]:
    """HI 的 payload {available_actions, workspace_id, model_view} → 压缩副本。legacy 原样深拷贝。"""

    tier = tier_for(renderer)
    if tier is None:
        return copy.deepcopy(payload)
    out = copy.deepcopy(payload)
    if isinstance(out.get("available_actions"), list):
        out["available_actions"] = compact_actions(out["available_actions"])
    view = out.get("model_view")
    if not isinstance(view, dict):
        return out
    focus = {e for e in view.get("focus_evidence_ids", []) if isinstance(e, str)}
    history = view.get("history")
    if isinstance(history, list):
        view["history"] = [
            {**{k: v for k, v in entry.items() if k != "observation"},
             "observation": _digest_observation(entry.get("observation"), tier)}
            if isinstance(entry, dict) else entry
            for entry in history
        ]
    if "latest_observation" in view:
        view["latest_observation"] = _latest_observation(view["latest_observation"], tier)
    for key in ("evidence_view", "materials", "candidate_sources"):
        if key in view:
            view[key] = compact_cards(view[key], tier)
    for key in ("image_inputs", "visual_observation_summary", "alignment_image_pairs"):
        if key in view:
            view[key] = shrink(view[key], text_chars=tier.latest_text_chars, list_cap=64, snippet_chars=tier.snippet_chars)
    constraints = view.get("runtime_constraints")
    if isinstance(constraints, dict):
        constraints["state_renderer"] = tier.name
    _fit_hi(out, view, tier, focus)
    return out


def _fit_hi(payload: dict[str, Any], view: dict[str, Any], tier: Tier, focus: set[str]) -> None:
    truncated = False
    view["context_truncated"] = True     # 先占位，保证标记本身的长度计入预算；没截断再拿掉
    history: list[Any] = view["history"] if isinstance(view.get("history"), list) else []
    # 1) 逐条丢历史（最后一条保留，它是 latest_observation 的动作）
    while len(history) > 1 and payload_chars(payload) > tier.budget_chars:
        index = max(range(len(history) - 1), key=lambda i: (_history_priority(history[i], focus), -i))
        history.pop(index)
        truncated = True
    # 2) 候选来源
    candidates = view.get("candidate_sources") if isinstance(view.get("candidate_sources"), list) else []
    while candidates and payload_chars(payload) > tier.budget_chars:
        candidates.pop(0)
        truncated = True
    # 3) 最新观察正文再砍一半，最多三次
    limit = tier.latest_text_chars
    for _ in range(3):
        if payload_chars(payload) <= tier.budget_chars:
            break
        limit //= 2
        view["latest_observation"] = shrink(view.get("latest_observation"), text_chars=limit, list_cap=max(2, tier.list_cap // 2), snippet_chars=tier.snippet_chars // 2)
        truncated = True
    # 4) 证据卡片从末尾丢（保留焦点）
    cards = view.get("evidence_view") if isinstance(view.get("evidence_view"), list) else []
    while cards and payload_chars(payload) > tier.budget_chars:
        droppable = [i for i, c in enumerate(cards) if not (isinstance(c, dict) and c.get("evidence_id") in focus)]
        if not droppable:
            break
        cards.pop(droppable[-1])
        truncated = True
    # 5) 兜底：最新观察只留摘要行，附图只留编号
    if payload_chars(payload) > tier.budget_chars:
        view["latest_observation"] = shrink(view.get("latest_observation"), text_chars=200, list_cap=2, snippet_chars=80)
        _ids_only(view, "image_inputs")
        _slim_visual_summary(view)
        truncated = True
    if not truncated:
        view.pop("context_truncated", None)
    if payload_chars(payload) > tier.budget_chars:
        n = payload_chars(payload)
        raise RenderError(f"任务、材料与最新观察本身超过 {tier.name} 的预算（{n} > {tier.budget_chars} 字符）", chars=n, budget=tier.budget_chars)


# --------------------------------------------------------------------------- GIA

def _compact_supporting_record(record: Any, tier: Tier) -> Any:
    if not isinstance(record, dict):
        return record
    keep = ("evidence_id", "modalities", "label", "source_ref", "derived_from", "acquired_by", "locator",
            "content_offset", "next_offset")
    out = {k: record[k] for k in keep if k in record and record[k] not in (None, 0, [], "")}
    if "evidence_id" in record:
        out.setdefault("evidence_id", record["evidence_id"])
    if isinstance(record.get("label"), str):
        out["label"] = cut_text(record["label"], tier.snippet_chars)
    content = record.get("content")
    if isinstance(content, str) and tier.support_text_chars > 0:
        if "image" in (record.get("modalities") or []) and content.startswith("{"):
            out["content"] = "media_metadata_not_pixels"
        else:
            out["content"] = cut_text(content, tier.support_text_chars)
    if tier.support_text_chars <= 0:
        out.pop("content_offset", None)
        out.pop("next_offset", None)
        out.pop("acquired_by", None)
    elif isinstance(record.get("content_notice"), str):
        out["content_notice"] = "media_metadata_not_pixels" if "像素" in record["content_notice"] else cut_text(record["content_notice"], tier.snippet_chars)
    return shrink(out, text_chars=max(tier.support_text_chars, tier.snippet_chars) + len(TRUNCATION_MARK) + 8,
                  list_cap=64, snippet_chars=tier.snippet_chars)


def render_gia_payload(payload: dict[str, Any], renderer: str | None) -> dict[str, Any]:
    """GIA 的 payload {available_actions, memory_history, global_view} → 压缩副本。"""

    tier = tier_for(renderer)
    if tier is None:
        return copy.deepcopy(payload)
    out = copy.deepcopy(payload)
    if isinstance(out.get("available_actions"), list):
        out["available_actions"] = compact_actions(out["available_actions"])
    view = out.get("global_view")
    if not isinstance(view, dict):
        return out
    reports = view.get("latest_reports")
    if isinstance(reports, list):
        for report in reports:
            if not isinstance(report, dict):
                continue
            if isinstance(report.get("supporting_records"), list):
                report["supporting_records"] = [
                    card_line(r, tier) if tier.support_text_chars <= 0 else _compact_supporting_record(r, tier)
                    for r in report["supporting_records"]
                ]
            if isinstance(report.get("report"), dict):
                report["report"] = shrink(report["report"], text_chars=tier.latest_text_chars, list_cap=64, snippet_chars=tier.latest_text_chars)
            if isinstance(report.get("image_inputs"), list):
                report["image_inputs"] = cut_list(report["image_inputs"], tier.list_cap)
            if not tier.report_visual_summary:
                report.pop("visual_observation_summary", None)
                if isinstance(report.get("image_inputs"), list):
                    report["image_inputs"] = [
                        {k: v for k, v in item.items() if k in ("evidence_id", "locator", "source_ref")} if isinstance(item, dict) else item
                        for item in report["image_inputs"]
                    ]
    for key in ("current_materials", "original_materials"):
        if key in view:
            view[key] = compact_cards(view[key], tier)
    if "history_index" in view:
        view["history_index"] = compact_cards(view["history_index"], tier)
    for key in ("image_inputs", "visual_observation_summary"):
        if key in view:
            view[key] = shrink(view[key], text_chars=tier.latest_text_chars, list_cap=64, snippet_chars=tier.snippet_chars)
    history = out.get("memory_history")
    if isinstance(history, list):
        out["memory_history"] = [
            {**{k: v for k, v in step.items() if k != "observation"},
             "observation": _latest_observation(step.get("observation"), tier) if i == len(history) - 1
             else _digest_observation(step.get("observation"), tier)}
            if isinstance(step, dict) else step
            for i, step in enumerate(history)
        ]
    constraints = view.get("runtime_constraints")
    if isinstance(constraints, dict):
        constraints["state_renderer"] = tier.name
    _fit_gia(out, view, tier)
    return out


def _fit_gia(payload: dict[str, Any], view: dict[str, Any], tier: Tier) -> None:
    truncated = False
    view["context_truncated"] = True
    history: list[Any] = payload["memory_history"] if isinstance(payload.get("memory_history"), list) else []
    while len(history) > 1 and payload_chars(payload) > tier.budget_chars:
        history.pop(0)
        truncated = True
    limit = tier.support_text_chars
    for _ in range(3):
        if limit <= 1 or payload_chars(payload) <= tier.budget_chars:
            break     # support_text_chars=0 的档位附件已是单行卡片，再截只剩截断标记、连编号都丢
        limit //= 2
        for report in view.get("latest_reports") or []:
            if isinstance(report, dict) and isinstance(report.get("supporting_records"), list):
                report["supporting_records"] = shrink(report["supporting_records"], text_chars=limit, list_cap=tier.list_cap, snippet_chars=tier.snippet_chars)
        truncated = True
    if history and payload_chars(payload) > tier.budget_chars:
        history.clear()
        truncated = True
    # 兜底：报告附图只留编号、去掉各报告的视觉摘要、旧轮次索引只留最近 12 条
    if payload_chars(payload) > tier.budget_chars:
        for report in view.get("latest_reports") or []:
            if isinstance(report, dict):
                _ids_only(report, "image_inputs")
                report.pop("visual_observation_summary", None)
        index = view.get("history_index")
        if isinstance(index, list) and len(index) > 12:
            view["history_index"] = [LIST_MARK.format(n=len(index) - 12), *index[-12:]]
        truncated = True
    if payload_chars(payload) > tier.budget_chars:
        _ids_only(view, "image_inputs")
        _slim_visual_summary(view)
        truncated = True
    if payload_chars(payload) > tier.budget_chars:
        _shrink_reports(payload, view, tier)
        truncated = True
    if not truncated:
        view.pop("context_truncated", None)
    if payload_chars(payload) > tier.budget_chars:
        n = payload_chars(payload)
        raise RenderError(f"总任务、材料与最新报告本身超过 {tier.name} 的预算（{n} > {tier.budget_chars} 字符）", chars=n, budget=tier.budget_chars)


REPORT_STATEMENT_STEPS = (600, 400, 250, 150)   # 最后一级：Finding 陈述逐级收缩的上限
COLLAPSED_STATEMENT_CHARS = 80                  # 折叠后的旧报告每条 Finding 只留这么多陈述


def _report_findings(view: dict[str, Any]) -> list[tuple[dict[str, Any] | None, dict[str, Any]]]:
    """(所属报告, Finding)：latest_reports[*].report.findings 与 history_index 里较早轮次的 Finding。"""

    out: list[tuple[dict[str, Any] | None, dict[str, Any]]] = []
    for entry in view.get("latest_reports") or []:
        report = entry.get("report") if isinstance(entry, dict) else None
        if isinstance(report, dict):
            out.append((report, {}))
            out.extend((report, f) for f in report.get("findings") or [] if isinstance(f, dict))
    for item in view.get("history_index") or []:
        if isinstance(item, dict) and "finding_id" in item:
            out.append((None, item))
        elif isinstance(item, dict) and "finding_ids" in item:
            out.append((item, {}))
    return out


def _cap_report_text(view: dict[str, Any], chars: int) -> None:
    for report, finding in _report_findings(view):
        if not finding:
            if report is not None and isinstance(report.get("status_reason"), str):
                report["status_reason"] = cut_text(report["status_reason"], chars)
            continue
        if isinstance(finding.get("statement"), str):
            finding["statement"] = cut_text(finding["statement"], chars)
        if isinstance(finding.get("limitations"), list):
            finding["limitations"] = [cut_text(x, chars) if isinstance(x, str) else x for x in finding["limitations"]]


def _collapse_finding(finding: Any) -> Any:
    if not isinstance(finding, dict):
        return finding
    out: dict[str, Any] = {"finding_id": finding.get("finding_id")}
    if isinstance(finding.get("statement"), str):
        out["statement"] = cut_text(finding["statement"], COLLAPSED_STATEMENT_CHARS)
    if "evidence_ids" in finding:
        out["evidence_ids"] = finding["evidence_ids"]
    return out


def _collapse_report_entry(entry: dict[str, Any]) -> None:
    """latest_reports 里的一份报告 → 只留 finding_id + 陈述前 80 字 + evidence_ids；附件只留编号。"""

    report = entry.get("report")
    if isinstance(report, dict):
        entry["report"] = {k: v for k, v in report.items() if k in ("task_id", "status", "finding_ids")}
        entry["report"]["findings"] = [_collapse_finding(f) for f in report.get("findings") or []]
    if isinstance(entry.get("task"), str):
        entry["task"] = cut_text(entry["task"], COLLAPSED_STATEMENT_CHARS)
    records = entry.get("supporting_records")
    if isinstance(records, list):
        entry["supporting_records"] = [
            r.get("evidence_id", r) if isinstance(r, dict) else r.split(" ", 1)[0] if isinstance(r, str) and r[:1] == "E" else r
            for r in records]
    _ids_only(entry, "image_inputs")
    entry.pop("visual_observation_summary", None)


def _shrink_reports(payload: dict[str, Any], view: dict[str, Any], tier: Tier) -> None:
    """最后一级退化：收缩报告与较早 Finding 的陈述，再把较早的报告折叠成摘要；最新一份报告保留完整陈述。

    每一步都写 view["reports_truncated"] = {"statement_chars": N, "collapsed_reports": [...]}，让 GIA 知道看到的是截断版。
    """

    info: dict[str, Any] = {"statement_chars": None, "collapsed_reports": []}
    view["reports_truncated"] = info
    for chars in REPORT_STATEMENT_STEPS:
        if payload_chars(payload) <= tier.budget_chars:
            return
        info["statement_chars"] = chars
        _cap_report_text(view, chars)
    # 较早轮次（history_index 里的报告 / Finding，按时间从早到晚），再是 latest_reports 里除最后一份以外的报告
    history = view.get("history_index")
    if isinstance(history, list):
        for i, item in enumerate(history):
            if payload_chars(payload) <= tier.budget_chars:
                return
            if isinstance(item, dict) and "finding_id" in item:
                history[i] = {**_collapse_finding(item), **({"workspace_id": item["workspace_id"]} if "workspace_id" in item else {})}
                owner = item.get("workspace_id")
            elif isinstance(item, dict) and "finding_ids" in item:
                history[i] = {k: v for k, v in item.items() if k in ("task_id", "status", "finding_ids")}
                owner = item.get("task_id")
            else:
                continue
            if owner is not None and owner not in info["collapsed_reports"]:
                info["collapsed_reports"].append(owner)
    reports = view.get("latest_reports")
    if isinstance(reports, list):
        for entry in reports[:-1]:
            if payload_chars(payload) <= tier.budget_chars:
                return
            if not isinstance(entry, dict):
                continue
            _collapse_report_entry(entry)
            report = entry.get("report")
            owner = report.get("task_id") if isinstance(report, dict) else None
            if owner is not None and owner not in info["collapsed_reports"]:
                info["collapsed_reports"].append(owner)


def render_payload(payload: dict[str, Any], renderer: str | None) -> dict[str, Any]:
    """按 payload 形状自动选择 HI / GIA 渲染。"""

    if "global_view" in payload:
        return render_gia_payload(payload, renderer)
    return render_hi_payload(payload, renderer)


__all__ = ["RENDERER_VERSION", "RenderError", "TIERS", "Tier", "compact_actions", "payload_chars", "render_gia_payload",
           "render_hi_payload", "render_payload", "tier_for"]
