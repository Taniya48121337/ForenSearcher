"""EVID 行为面板（10-05 修正优化计划 P0.3 / P0.4）：从 run 目录读出"模型看到了什么、做了什么"，不调评委。

    python scripts/_evid_behavior.py --round evidb5                 # 该轮全部 run，打印汇总并写 scores/<轮次>.behavior.{jsonl,md}
    python scripts/_evid_behavior.py --round evidb4 --ids video_027 video_151

_evid_score.py 每评一个 run 也调 run_behavior()，汇总表里多出这几列。目的：以后每轮一眼看出是"没看到"还是"看到了没答对"。

每个 run 的字段（都从 trajectory.jsonl + requests/*/transport.json + 批量日志读，不改任何东西）：
① coverage          材料 A 覆盖率 = 实际附给模型看过的原件帧的最大时刻 / 原件时长（帧来自 HI 取帧或 P1.1 的入库关键帧）
   coverage_extracted 同上但只要求帧被取出来过（不要求附图给模型）
② candidate_opens   对候选来源动手的次数：INSPECT / READ_RECORD 目标是候选来源的 E，或 SEARCH source=R
③ video_downloads   候选视频真正下载的次数（INSPECT 对候选页面 E 触发的 acquisition，不含复用）
④ gia_last_images / gia_last_original_frames   GIA 最后一次决策实际附图数 / 其中原件派生帧数
⑤ repeat_rate       HI 动作里与本工作区之前某次动作逐字相同的比例
⑥ context_retry_requests / context_retry_steps   因上下文超限减图重发的请求数 / 减图次数；input_tokens_p95 / max
⑦ crash_reason      没有 result.json 时，批量日志里最后一个异常（类名: 前 120 字）；有结果但由运行时收尾时是 forced_* 标记
另：hi_rejected / gia_rejected（原始输出没通过角色校验的次数，P0.2 schema 解码的验收看它）、answer_language（P0.4，在 _evid_score.py）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "experiments" / "evid" / "runs"
DEFAULT_LOGS = ROOT / "experiments" / "sft_data" / "batch_logs"
DEFAULT_OUT = ROOT / "experiments" / "evid" / "scores"

# 中文标签（旧规格）与英文标签（language="en"，hi@0.7.0 / gia@0.8.0 起）都认
_FRAME_LABEL = re.compile(r"视频(E\d+)在([0-9.]+)秒的帧|frame of video (E\d+) at ([0-9.]+) s")
_PROFILE_LABEL = re.compile(r"媒体属性：video，([0-9.]+)秒|media profile: video, ([0-9.]+) s")
_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_LATIN = re.compile(r"[A-Za-z]")
_EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Limit|Overflow|Timeout|Exit|Interrupt)\w*)(?::\s*(.*))?$")


def answer_language(text: str) -> str:
    """粗分答案语言：汉字数多于英文词数（字母数 / 5）记 zh，否则记 en；空答案记 none。"""

    cjk = len(_CJK.findall(text or ""))
    latin = len(_LATIN.findall(text or ""))
    if not cjk and not latin:
        return "none"
    return "zh" if cjk > latin / 5 else "en"     # 一个汉字 ≈ 一个英文单词（约 5 个字母）；英文答案里引一句中文字幕仍记 en


def _events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trajectory.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            yield item
            stack.extend(child for child in item.values() if isinstance(child, (dict, list)))
        elif isinstance(item, list):
            stack.extend(child for child in item if isinstance(child, (dict, list)))


def _position(locator: Any) -> float | None:
    if isinstance(locator, str):
        try:
            locator = json.loads(locator)
        except ValueError:
            return None
    if isinstance(locator, dict):
        value = locator.get("time_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def crash_reason(run_dir: Path, logs_dir: Path | None) -> str | None:
    if (run_dir / "result.json").is_file():
        return None
    log = (logs_dir / f"{run_dir.name}.log") if logs_dir is not None else None
    if log is None or not log.is_file():
        return "no result.json (no log)"
    last = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EXCEPTION.match(line.strip())
        if match:
            last = match
    if last is None:
        return "no result.json"
    name = last.group(1).rsplit(".", 1)[-1]
    return f"{name}: {(last.group(2) or '').strip()[:120]}".rstrip(": ")


def run_behavior(run_dir: Path, logs_dir: Path | None = DEFAULT_LOGS) -> dict[str, Any]:
    run_dir = Path(run_dir)
    events = _events(run_dir)
    out: dict[str, Any] = {"run": run_dir.name, "has_trajectory": bool(events)}
    originals: list[str] = []
    for event in events:
        if event.get("event_type") == "investigation_run":
            originals = [x for x in event["payload"].get("original_material_ids", []) if str(x).startswith("E")]
            break
    original_set = set(originals)
    cards: dict[str, dict[str, Any]] = {}
    frame_time: dict[str, tuple[str, float]] = {}       # 帧 E → (原件 E, 时刻)
    durations: dict[str, float] = {}
    for event in events:
        for item in _walk(event.get("payload")):
            eid = item.get("evidence_id")
            if isinstance(eid, str) and eid.startswith("E"):
                if "acquired_by" in item or "derived_from" in item or "source_ref" in item:
                    merged = cards.setdefault(eid, {})
                    for key in ("acquired_by", "derived_from", "source_ref", "label", "modalities"):
                        if key in item and item[key] is not None:
                            merged.setdefault(key, item[key])
                parents = item.get("derived_from")
                position = _position(item.get("locator")) if "locator" in item else None
                if position is None and isinstance(item.get("time_s"), (int, float)):
                    position = float(item["time_s"])
                if position is not None and isinstance(parents, list) and len(parents) == 1 and parents[0] in original_set:
                    frame_time.setdefault(eid, (parents[0], position))
            if isinstance(item.get("duration_s"), (int, float)) and isinstance(eid, str) and eid in original_set:
                durations.setdefault(eid, float(item["duration_s"]))
    for eid, card in cards.items():
        label = str(card.get("label") or "")
        match = _FRAME_LABEL.search(label)
        if match:
            video, time_s = match.group(1) or match.group(3), match.group(2) or match.group(4)
            if video in original_set:
                frame_time.setdefault(eid, (video, float(time_s)))
        match = _PROFILE_LABEL.search(label)
        parents = card.get("derived_from") or []
        if match and len(parents) == 1 and parents[0] in original_set:
            durations.setdefault(parents[0], float(match.group(1) or match.group(2)))
    # get_frames 结果里帧只带 time_s，父件要看动作目标
    last_action: dict[str, dict[str, Any]] = {}
    shown: set[str] = set()
    gia_last: list[str] = []
    hi_actions = hi_repeats = 0
    seen_actions: dict[str, set[str]] = defaultdict(set)
    candidate_opens = 0
    downloads = 0
    downloaded: set[str] = set()     # 10-05 晚：观察里不带 acquisition 明细，按 INSPECT/acquire 的视频 E 去重计数
    reports = {"self": 0, "auto": 0}
    rejections = {"HI": 0, "GIA": 0}
    raw = {"HI": 0, "GIA": 0}
    accepted = {"HI": 0, "GIA": 0}
    ingest: list[dict[str, Any]] = []
    for event in events:
        kind, role, payload = event.get("event_type"), event.get("actor_role"), event.get("payload") or {}
        actor = event.get("actor_id")
        if kind == "raw_model_output" and role in raw:
            raw[role] += 1
            ids = [x for x in payload.get("image_evidence_ids") or [] if isinstance(x, str)]
            shown.update(ids)
            if role == "GIA":
                gia_last = ids
        elif kind == "model_output" and role in accepted and role != "HI":
            accepted[role] += 1
        elif kind == "model_output" and role == "HI":
            accepted["HI"] += 1
            output = payload.get("output") or {}
            if isinstance(output, dict) and output.get("action"):
                last_action[actor] = output
                hi_actions += 1
                fingerprint = json.dumps({"a": output.get("action"), "g": output.get("arguments")}, sort_keys=True,
                                         ensure_ascii=False)
                if fingerprint in seen_actions[actor]:
                    hi_repeats += 1
                seen_actions[actor].add(fingerprint)
                arguments = output.get("arguments") or {}
                targets: list[str] = []
                if output["action"] == "INSPECT":
                    targets = [t for t in arguments.get("targets") or [] if isinstance(t, str)]
                elif output["action"] == "READ_RECORD" and isinstance(arguments.get("record_id"), str):
                    targets = [arguments["record_id"]]
                elif output["action"] == "SEARCH" and str(arguments.get("source") or "").startswith("R"):
                    candidate_opens += 1
                if any(str(cards.get(t, {}).get("source_ref") or "").startswith("R") for t in targets):
                    candidate_opens += 1
        elif kind == "workspace_report" and role == "HI":
            auto = str(payload.get("status_reason") or "").startswith("auto_report")
            reports["auto" if auto else "self"] += 1
        elif kind == "observation":
            observation = payload.get("observation") or {}
            if observation.get("action") == "HARNESS_REJECTION" and role in rejections:
                rejections[role] += 1          # 策略重试也没救回来、交给运行时的拒收（每次占一个回合）
            details = observation.get("details") or {}
            if isinstance(details, dict):
                for item in details.get("acquisition") or []:
                    if isinstance(item, dict) and not item.get("reused"):
                        downloads += 1
                for card in details.get("evidence") or []:
                    if isinstance(card, dict) and card.get("acquired_by") == "INSPECT/acquire" \
                            and isinstance(card.get("evidence_id"), str):
                        downloaded.add(card["evidence_id"])
                result = details.get("result")
                frames = result.get("frames") if isinstance(result, dict) else None
                targets = ((last_action.get(actor) or {}).get("arguments") or {}).get("targets") or []
                if details.get("operation") == "get_frames" and isinstance(frames, list) and len(targets) == 1 \
                        and targets[0] in original_set:
                    for frame in frames:
                        if isinstance(frame, dict) and isinstance(frame.get("evidence_id"), str) \
                                and isinstance(frame.get("time_s"), (int, float)):
                            frame_time.setdefault(frame["evidence_id"], (targets[0], float(frame["time_s"])))
        elif kind == "ingest_frames":
            ingest.append(payload)
            source = payload.get("evidence_id")
            if isinstance(payload.get("duration_s"), (int, float)) and source:
                durations.setdefault(source, float(payload["duration_s"]))
            for eid, time_s in zip(payload.get("frame_evidence_ids") or [], payload.get("times_s") or []):
                if isinstance(eid, str) and source:
                    frame_time.setdefault(eid, (source, float(time_s)))
    # ① 覆盖率（按第一个有时长的视频原件）
    video = next((eid for eid in originals if eid in durations), originals[0] if originals else None)
    duration = durations.get(video) if video else None
    extracted = [t for eid, (src, t) in frame_time.items() if src == video]
    observed = [t for eid, (src, t) in frame_time.items() if src == video and eid in shown]
    out.update({
        "duration_s": duration,
        "max_observed_frame_s": max(observed) if observed else None,
        "coverage": round(min(1.0, max(observed) / duration), 3) if observed and duration else (0.0 if duration else None),
        "coverage_extracted": round(min(1.0, max(extracted) / duration), 3) if extracted and duration else (0.0 if duration else None),
        "original_frames_observed": len({eid for eid, (src, _) in frame_time.items() if src == video and eid in shown}),
        "ingest_frames": sum(len(p.get("frame_evidence_ids") or []) for p in ingest),
        "candidate_opens": candidate_opens,
        "video_downloads": max(downloads, len(downloaded)),
        "hi_self_reports": reports["self"], "hi_auto_reports": reports["auto"],
        "gia_last_images": len(gia_last),
        "gia_last_original_frames": sum(1 for eid in gia_last if frame_time.get(eid, (None,))[0] in original_set),
        "hi_actions": hi_actions,
        "hi_repeats": hi_repeats,
        "repeat_rate": round(hi_repeats / hi_actions, 3) if hi_actions else None,
        "hi_raw_outputs": raw["HI"], "gia_raw_outputs": raw["GIA"],
        # 被拒 = 原始输出里没通过角色校验的（含策略内重试）；P0.2 schema 解码的验收看 rejected / raw
        "hi_rejected": max(0, raw["HI"] - accepted["HI"]), "gia_rejected": max(0, raw["GIA"] - accepted["GIA"]),
        "hi_rejections": rejections["HI"], "gia_rejections": rejections["GIA"],
    })
    # ⑥ 上下文
    retry_requests = retry_steps = 0
    tokens: list[float] = []
    formats: dict[str, int] = defaultdict(int)
    for transport in sorted((run_dir / "requests").glob("request_*/transport.json")) if (run_dir / "requests").is_dir() else []:
        try:
            data = json.loads(transport.read_text(encoding="utf-8"))
        except ValueError:
            continue
        retries = data.get("context_retries") or []
        if retries:
            retry_requests += 1
            retry_steps += len(retries)
        value = (data.get("usage") or {}).get("input_tokens")
        if isinstance(value, (int, float)):
            tokens.append(float(value))
        formats[str(data.get("response_format") or ("json_object" if data.get("json_mode") else "none"))] += 1
    out.update({"context_retry_requests": retry_requests, "context_retry_steps": retry_steps,
                "input_tokens_p95": _percentile(tokens, 0.95), "input_tokens_max": max(tokens) if tokens else None,
                "response_formats": dict(formats)})
    # ⑦ 崩溃原因
    reason = crash_reason(run_dir, logs_dir)
    if reason is None:
        try:
            flags = json.loads((run_dir / "result.json").read_text(encoding="utf-8")).get("contract_flags") or []
        except ValueError:
            flags = []
        forced = [flag for flag in flags if str(flag).startswith("forced_")]
        reason = ",".join(forced) if forced else None
    out["crash_reason"] = reason
    return out


# ---------------------------------------------------------------- 找原片漏斗（10-06 找原片实验计划 ①②③）

_COMPARE_OPS = ("align_media", "diff_media", "compare_media")   # 10-13：合并目录的 compare_media 也算


def _store_rows(path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    import sqlite3

    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return connection.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def _query_words(text: str) -> int:
    """查询长度：拉丁词按空白切；连续汉字 / 假名每 2 个字算 1 个词。"""

    cjk = sum((len(run) + 1) // 2 for run in re.findall(r"[぀-ヿ㐀-鿿豈-﫿]+", text))
    latin = len(re.sub(r"[぀-ヿ㐀-鿿豈-﫿]+", " ", text).split())
    return cjk + latin


def source_funnel(run_dir: Path, sources: list[str]) -> dict[str, Any]:
    """原片（GT source 的 YouTube ID）在本 run 里走到了哪一步：① 进候选池 ② 视频被下载 ③ align/diff 对它跑过。

    全部从 store.sqlite3 读（只读打开）：candidates.platform_video_id / source_url 匹配 GT ID；
    candidate_evidence.material_kind='video' 的 E 就是下载下来的视频；inspect_executions 里 align_media / diff_media
    的 targets 含这个 E 算比对过。另记搜索次数、视频搜索空结果数、首次发现它的 provider / 排名 / 查询，给臂的归因用。
    """

    store = Path(run_dir) / "store.sqlite3"
    out: dict[str, Any] = {"has_store": store.is_file(), "sources": list(sources)}
    if not store.is_file():
        return out
    searches = _store_rows(store, "SELECT id, request_json, candidate_count, stop_reason FROM search_executions")
    kinds: dict[str, int] = defaultdict(int)
    video_empty = 0
    query_by_exec: dict[int, str] = {}
    video_words: list[int] = []
    for execution_id, request_json, candidate_count, _stop in searches:
        try:
            request = json.loads(request_json)
        except ValueError:
            continue
        if "source" in request:
            kinds["open"] += 1
            continue
        query = request.get("query")
        text = query if isinstance(query, str) else str((query or {}).get("text") or "")
        query_by_exec[int(execution_id)] = text if request.get("input_type") == "text" else \
            f"[{request.get('input_type')} {(query or {}).get('evidence_id') if isinstance(query, dict) else query}] {text}".strip()
        kinds[f"{request.get('input_type')}->{request.get('output_type')}"] += 1
        if request.get("output_type") == "video":
            if not candidate_count:
                video_empty += 1
            if text:
                video_words.append(_query_words(text))
    # 10-13：final_url 也算（以图搜图找到的新闻页，SEARCH 跟到了原片的 YouTube 页）
    pool = _store_rows(store, "SELECT id, source_url, platform_video_id, final_url FROM candidates") or [
        (*row, None) for row in _store_rows(store, "SELECT id, source_url, platform_video_id FROM candidates")]
    video_pool = {row[2] for row in pool if row[2]}
    hits = [row for row in pool if (row[2] and row[2] in sources)
            or any(s in str(row[1] or "") or s in str(row[3] or "") for s in sources)]
    out.update({
        "searches": dict(kinds),
        "video_searches": sum(v for k, v in kinds.items() if k.endswith("->video")),
        "video_searches_empty": video_empty,
        "video_query_words_median": statistics.median(video_words) if video_words else None,
        "pool_videos": len(video_pool),
        "source_in_pool": bool(hits),
        "source_downloaded": False, "source_compared": False,
        "source_found_by": None, "source_rank": None, "source_query": None, "source_hit_ids": [],
    })
    if not hits:
        return out
    candidate_ids = [int(row[0]) for row in hits]
    marks = ",".join("?" * len(candidate_ids))
    out["source_hit_ids"] = sorted({row[2] for row in hits if row[2]} | {s for s in sources for row in hits if s in str(row[1] or "") or s in str(row[3] or "")})
    discoveries = _store_rows(store, f"SELECT execution_id, provider, engine, provider_rank FROM candidate_discoveries "
                                     f"WHERE candidate_id IN ({marks}) ORDER BY id", tuple(candidate_ids))
    if discoveries:
        first = discoveries[0]
        out["source_found_by"] = f"{first[1]}/{first[2]}"
        ranks = [int(r[3]) for r in discoveries if isinstance(r[3], int)]
        out["source_rank"] = min(ranks) if ranks else None
        out["source_query"] = (query_by_exec.get(int(first[0])) or "")[:200]
        out["source_found_by_all"] = sorted({f"{r[1]}/{r[2]}" for r in discoveries})
    videos = {f"E{row[0]}" for row in _store_rows(
        store, f"SELECT evidence_id FROM candidate_evidence WHERE candidate_id IN ({marks}) AND material_kind='video'",
        tuple(candidate_ids))}
    out["source_downloaded"] = bool(videos)
    out["source_video_evidence"] = sorted(videos)
    # compare_media 的目标是候选的页面 E（INSPECT 内部再下载），页面 E 也算"对原片"
    pages = {f"E{row[0]}" for row in _store_rows(
        store, f"SELECT evidence_id FROM candidate_evidence WHERE candidate_id IN ({marks}) AND material_kind='page'",
        tuple(candidate_ids))}
    compared = viewed = 0
    for (request_json,) in _store_rows(store, "SELECT request_json FROM inspect_executions"):
        try:
            request = json.loads(request_json)
        except ValueError:
            continue
        targets = set(request.get("targets") or [])
        operation = (request.get("request") or {}).get("operation")
        if not (targets & videos or (operation == "compare_media" and targets & pages)):
            continue
        if operation in _COMPARE_OPS:
            compared += 1
        else:
            viewed += 1
    out["source_compared"] = compared > 0
    out["source_compare_calls"] = compared
    out["source_other_inspect_calls"] = viewed
    return out


def summarize_funnel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in rows if r.get("has_store")]
    found_by: dict[str, int] = defaultdict(int)
    for r in present:
        if r.get("source_in_pool"):
            found_by[str(r.get("source_found_by"))] += 1
    words = [r["video_query_words_median"] for r in present if isinstance(r.get("video_query_words_median"), (int, float))]
    return {
        "runs": len(rows), "runs_with_store": len(present),
        "source_in_pool": sum(1 for r in present if r.get("source_in_pool")),
        "source_downloaded": sum(1 for r in present if r.get("source_downloaded")),
        "source_compared": sum(1 for r in present if r.get("source_compared")),
        "source_found_by": dict(found_by),
        "source_rank_over_5": sum(1 for r in present if isinstance(r.get("source_rank"), int) and r["source_rank"] > 5),
        "video_searches": sum(r.get("video_searches") or 0 for r in present),
        "video_searches_empty": sum(r.get("video_searches_empty") or 0 for r in present),
        "video_searches_per_run_median": statistics.median([r.get("video_searches") or 0 for r in present]) if present else None,
        "video_query_words_median": statistics.median(words) if words else None,
        "pool_videos_median": statistics.median([r.get("pool_videos") or 0 for r in present]) if present else None,
    }


def funnel_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], names: list[str],
                    *, title: str = "## 找原片漏斗（GT source ID）") -> list[str]:
    s = summary
    n = s["runs_with_store"] or 0
    lines = [title, "",
             f"- ① 原片进候选池 **{s['source_in_pool']}/{n}**　② 原片视频被下载 **{s['source_downloaded']}/{n}**　"
             f"③ align/diff/compare_media 对原片跑过 **{s['source_compared']}/{n}**",
             f"- 进池途径：{s['source_found_by'] or '无'}；排名 > 5 才进池的 {s['source_rank_over_5']} 个",
             f"- 视频搜索 {s['video_searches']} 次（每 run 中位 {s['video_searches_per_run_median']}），其中**零结果 "
             f"{s['video_searches_empty']} 次**；视频查询词数中位 {s['video_query_words_median']}；每 run 候选视频数中位 "
             f"{s['pool_videos_median']}",
             "", "| run | 视频搜索（零结果） | 候选视频 | ① 进池（途径 / 排名） | ② 下载 | ③ 比对 | 发现它的查询 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in zip(names, rows):
        if not r.get("has_store"):
            lines.append(f"| {name} | — | — | — | — | — | 没有 store |")
            continue
        pool = f"是（{r.get('source_found_by')} / {r.get('source_rank')}）" if r.get("source_in_pool") else "否"
        query = str(r.get("source_query") or "").replace("|", "/")[:70]
        lines.append(f"| {name} | {r.get('video_searches')}（{r.get('video_searches_empty')}） | {r.get('pool_videos')} | {pool} | "
                     f"{'是' if r.get('source_downloaded') else '否'} | {'是' if r.get('source_compared') else '否'} | {query} |")
    return lines


def summarize_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in rows if r.get("has_trajectory")]

    def median(key: str) -> float | None:
        values = [r[key] for r in present if isinstance(r.get(key), (int, float))]
        return round(statistics.median(values), 3) if values else None

    hi_actions = sum(r.get("hi_actions") or 0 for r in present)
    hi_repeats = sum(r.get("hi_repeats") or 0 for r in present)
    crashes: dict[str, int] = defaultdict(int)
    for r in rows:                     # 崩溃的 run 可能连轨迹都没导出，按全部 run 计
        if r.get("crash_reason") and not str(r["crash_reason"]).startswith("forced_"):
            crashes[str(r["crash_reason"]).split(":", 1)[0]] += 1
    raw_hi = sum(r.get("hi_raw_outputs") or 0 for r in present)
    raw_gia = sum(r.get("gia_raw_outputs") or 0 for r in present)
    return {
        "runs": len(rows),
        "runs_with_trajectory": len(present),
        "coverage_median": median("coverage"),
        "coverage_extracted_median": median("coverage_extracted"),
        "coverage_below_25pct": sum(1 for r in present if isinstance(r.get("coverage"), (int, float)) and r["coverage"] < 0.25),
        "candidate_opens_total": sum(r.get("candidate_opens") or 0 for r in present),
        "runs_with_candidate_open": sum(1 for r in present if r.get("candidate_opens")),
        "video_downloads_total": sum(r.get("video_downloads") or 0 for r in present),
        "runs_with_download": sum(1 for r in present if r.get("video_downloads")),
        "gia_last_zero_original_frames": sum(1 for r in present if r.get("gia_raw_outputs") and not r.get("gia_last_original_frames")),
        "gia_last_original_frames_mean": round(statistics.mean([r.get("gia_last_original_frames") or 0 for r in present]), 2) if present else None,
        "repeat_rate": round(hi_repeats / hi_actions, 3) if hi_actions else None,
        "context_retry_requests": sum(r.get("context_retry_requests") or 0 for r in present),
        "runs_with_context_retry": sum(1 for r in present if r.get("context_retry_requests")),
        "input_tokens_p95_max": int(max((r["input_tokens_p95"] for r in present if r.get("input_tokens_p95")), default=0)) or None,
        "hi_rejection_rate": round(sum(r.get("hi_rejected") or 0 for r in present) / raw_hi, 3) if raw_hi else None,
        "gia_rejection_rate": round(sum(r.get("gia_rejected") or 0 for r in present) / raw_gia, 3) if raw_gia else None,
        "crashes": dict(crashes),
        "forced": sum(1 for r in present if str(r.get("crash_reason") or "").startswith("forced_")),
        "hi_self_reports": sum(r.get("hi_self_reports") or 0 for r in present),
        "hi_auto_reports": sum(r.get("hi_auto_reports") or 0 for r in present),
    }


def behavior_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], *, title: str = "## 行为面板") -> list[str]:
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    s = summary
    lines = [title, "",
             "| 指标 | 值 | 修正计划目标 |", "|---|---|---|",
             f"| ① 材料 A 覆盖率中位（附图看过的最大帧时刻 / 时长） | {pct(s['coverage_median'])}（只算取出来过：{pct(s['coverage_extracted_median'])}）；<25% 的 run {s['coverage_below_25pct']} 个 | > 60% |",
             f"| ② 对候选来源动手的次数 | {s['candidate_opens_total']}（{s['runs_with_candidate_open']} 个 run） | — |",
             f"| ③ 候选视频下载次数 | {s['video_downloads_total']}（{s['runs_with_download']} 个 run） | > 0 |",
             f"| ④ GIA 最后决策零原件帧的 run | {s['gia_last_zero_original_frames']}（平均原件帧 {s['gia_last_original_frames_mean']}） | 0 |",
             f"| ⑤ HI 重复调用率 | {pct(s['repeat_rate'])} | < 15% |",
             f"| ⑥ 上下文减图重发的请求 | {s['context_retry_requests']}（{s['runs_with_context_retry']} 个 run）；单 run input_tokens p95 最大 {s['input_tokens_p95_max']} | 0；p95 < 24k |",
             f"| ⑦ 崩溃（没有 result.json） | {s['crashes'] or '无'}；运行时收尾 forced_* {s['forced']} 个 | 崩溃 < 5% |",
             f"| HI 报告：自己写的 / 运行时代交的空报告 | {s.get('hi_self_reports')} / {s.get('hi_auto_reports')} | 代交 → 0（evidb7：1 / 26） |",
             f"| 输出被拒率（没通过角色校验的原始输出 / 全部原始输出） | HI {pct(s['hi_rejection_rate'])}，GIA {pct(s['gia_rejection_rate'])} | 个位数（evidb4：32% / 38%） |",
             "", "| run | 覆盖率 | 候选动手 | 下载 | GIA 末次附图（原件帧） | 重复率 | 减图请求 | p95 token | 崩溃 / 收尾 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("has_trajectory"):
            lines.append(f"| {r['run']} | — | — | — | — | — | — | — | {str(r.get('crash_reason') or '没有轨迹')[:80]} |")
            continue
        p95 = r.get("input_tokens_p95")
        lines.append(f"| {r['run']} | {pct(r.get('coverage'))} | {r.get('candidate_opens')} | {r.get('video_downloads')} | "
                     f"{r.get('gia_last_images')}（{r.get('gia_last_original_frames')}） | {pct(r.get('repeat_rate'))} | "
                     f"{r.get('context_retry_requests')} | {int(p95) if p95 else '—'} | {str(r.get('crash_reason') or '')[:80]} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EVID 行为面板（不调评委）")
    parser.add_argument("--round", required=True)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--ids", nargs="+", default=None, help="video_027 … ；默认该轮全部 run")
    args = parser.parse_args(argv)
    runs = Path(args.runs_dir)
    pattern = re.compile(rf"^evid_(video_\d+)_{re.escape(args.round)}_\d+$")
    dirs = sorted(d for d in runs.iterdir() if d.is_dir() and pattern.match(d.name)) if runs.is_dir() else []
    if args.ids:
        wanted = set(args.ids)
        dirs = [d for d in dirs if pattern.match(d.name).group(1) in wanted]
    rows = [run_behavior(d, Path(args.logs_dir)) for d in dirs]
    summary = summarize_behavior(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.round}.behavior" + (".subset" if args.ids else "")
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / f"{tag}.md").write_text("\n".join([f"# EVID 行为面板：{args.round}", "",
                                              *behavior_markdown(summary, rows, title="## 汇总")]) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {out / (tag + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
