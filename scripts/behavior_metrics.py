"""r0 评测的四个行为指标（2026-10-15，清单 E2 / E3 / E5）：输入 run 目录，输出

    ① 打开候选率   open_rate          原片进候选池后被模型打开的比例（EVID，按 GT 原片 ID 匹配）
    ② 比对率       compare_rate       模型打开过的候选（视频 / 图片）里，之后被 compare_media（或旧的 align/diff_media）比对过的比例
    ③ 重复调用率   repeat_rate        HI 动作里与本工作区之前某次动作逐字相同的比例
    ④ 格式拒率     format_reject_rate 原始模型输出里没通过角色校验的比例，HI、GIA 各算一个再取平均（两角色各一套 LoRA，
                                       各占一半权重；按次数合计会被 HI 的大量输出淹没：b12bf 合计 5.5%，GIA 单独 25%）

另附分数（--scores 给 _evid_score.py 的评分 jsonl 时）、检索快照命中 / 新录率、冻结核对、旧口径复算（与总表旧数对照）。

用法：

    python scripts/behavior_metrics.py experiments/evid/runs --round evidF27              # 一轮
    python scripts/behavior_metrics.py experiments/eval/r0/runs --round r0base-s1 --round r0base-s2 \\
        --eval-set experiments/eval/r0/eval_set.json --scores experiments/eval/r0/scores/r0base-s1.sonnet.subset.jsonl \\
        experiments/eval/r0/scores/r0base-s2.sonnet.subset.jsonl --out experiments/eval/r0/reports/r0base
    python scripts/behavior_metrics.py --from-score-jsonl experiments/evid/scores/evidb12bf.sonnet.subset.jsonl
                                                       # run 目录已删时：用评分文件里逐 run 存的行为字段重算（只有 ③④ 与旧漏斗）
    python scripts/behavior_metrics.py ... --criteria experiments/eval/r0/success_criteria.json --baseline <E2 的 .json>
                                                       # 训后：对照预先登记的成功标准逐条判定

定义细节（全部只读，不改 run）：
- "打开"= 模型自己的动作（轨迹 model_output）指向这个候选：INSPECT 的 targets、HI / GIA 的 READ_RECORD record_id、
  SEARCH source=R<n>。目标是候选的任一材料 E（page / cover / video / image）
  或 R<n>（= candidates.id）。SEARCH 自动打开前 5 个候选生成的 page E 不算"打开"。
- "比对"= 模型自己发出的 INSPECT compare_media / align_media / diff_media，targets 含该候选的任一材料 E。
  --auto-tools on 时运行时代跑的 compare_media（SEARCH 观察里的 auto_compare）不算，另列 auto_compare_runs。
  旧漏斗（_evid_behavior.source_funnel）只认 targets 含视频 E、或 compare_media 指向 page E；align_media 直接对 page E
  （内部会先下载）它不算——这是新旧"原片比对"数不同的原因（F27：旧 6，新 11）。
- 合计一律按次数汇总（micro）：分子分母先跨 run、跨种子相加再相除；每项都带 Wilson 95% 区间。
- ① 的分母小于 min_denominator（默认 5）时不下结论，看伴随指标 open_rate_runs（打开了原片的 run / 全部 EVID run）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
METRICS_VERSION = "behavior_metrics@1.0.0"
DEFAULT_EVID_BENCH = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "evid_bench.json"
COMPARE_OPS = ("compare_media", "align_media", "diff_media")
CANDIDATE_TYPES = ("video", "image")
MIN_DENOMINATOR = 5
_RUN_SUFFIX = re.compile(r"^(?P<case>.+)_(?P<round>[^_]+)_(?P<sample>\d+)$")


# ---------------------------------------------------------------- 小工具

def _sha256_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def ratio(k: int, n: int) -> dict[str, Any]:
    return {"value": round(k / n, 4) if n else None, "num": k, "den": n, "ci95": wilson(k, n)}


def role_mean(hi: dict[str, Any], gia: dict[str, Any]) -> dict[str, Any]:
    """两个角色的比率取平均；有一个角色没有输出时就用另一个。区间取两者区间端点的平均（近似，只作参考）。"""

    parts = [m for m in (hi, gia) if m.get("value") is not None]
    if not parts:
        return {"value": None, "num": None, "den": None, "ci95": None, "roles": {"HI": hi, "GIA": gia}}
    value = sum(m["num"] / m["den"] for m in parts) / len(parts)
    cis = [m["ci95"] for m in parts if m.get("ci95")]
    ci = (round(sum(c[0] for c in cis) / len(cis), 4), round(sum(c[1] for c in cis) / len(cis), 4)) if cis else None
    return {"value": round(value, 4), "num": sum(m["num"] for m in parts), "den": sum(m["den"] for m in parts),
            "ci95": ci, "macro_over_roles": [r for r, m in (("HI", hi), ("GIA", gia)) if m.get("value") is not None],
            "roles": {"HI": hi, "GIA": gia}}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _events(run_dir: Path) -> Iterable[dict[str, Any]]:
    path = run_dir / "trajectory.jsonl"
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def _rows(store: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(f"file:{store.as_posix()}?mode=ro", uri=True)
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def load_evid_sources(path: Path | None) -> dict[str, list[str]]:
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["id"]: list(entry.get("source") or []) for entry in data if isinstance(entry, dict) and "id" in entry}


def case_of(run_dir: Path, run_config: dict[str, Any] | None) -> tuple[str | None, str | None, int | None]:
    """(case_id, round, sample)：优先 run_config，其次 run 目录名 <case_key>_<round>_<sample>。"""

    case_id = (run_config or {}).get("case_id")
    round_tag = (run_config or {}).get("round")
    sample = (run_config or {}).get("sample")
    match = _RUN_SUFFIX.match(run_dir.name)
    if match:
        round_tag = round_tag or match.group("round")
        sample = sample if isinstance(sample, int) else int(match.group("sample"))
        if not case_id:
            key = match.group("case")
            case_id = re.sub(r"^evid_(video_\d+)$", r"evid:\1", key)
    return case_id, round_tag, sample


def evid_sample(case_id: str | None) -> str | None:
    match = re.match(r"^evid:(video_\d+)$", case_id or "")
    return match.group(1) if match else None


# ---------------------------------------------------------------- 单个 run

def run_metrics(run_dir: Path, sources: list[str] | None, *, candidate_types: tuple[str, ...] = CANDIDATE_TYPES
                ) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_config = _read_json(run_dir / "run_config.json") or {}
    case_id, round_tag, sample = case_of(run_dir, run_config)
    out: dict[str, Any] = {"run": run_dir.name, "case_id": case_id, "round": round_tag, "sample": sample,
                           "has_trajectory": (run_dir / "trajectory.jsonl").is_file(),
                           "has_store": (run_dir / "store.sqlite3").is_file(),
                           "has_result": (run_dir / "result.json").is_file(), "has_run_config": bool(run_config)}

    # ---- 轨迹：格式拒率、重复调用率、模型动作的目标
    raw = Counter()
    accepted = Counter()
    harness_rejections = Counter()
    hi_actions = hi_repeats = 0
    seen: dict[str, set[str]] = defaultdict(set)
    action_targets: set[str] = set()
    compare_targets: set[str] = set()
    model_compare_calls = 0
    auto_compares: list[list[str]] = []
    action_counts: Counter[str] = Counter()
    for event in _events(run_dir):
        kind, role, payload = event.get("event_type"), event.get("actor_role"), event.get("payload") or {}
        if kind == "raw_model_output" and role in ("HI", "GIA"):
            raw[role] += 1
        elif kind == "model_output" and role in ("HI", "GIA"):
            accepted[role] += 1
            output = payload.get("output") if isinstance(payload, dict) else None
            if not isinstance(output, dict) or not output.get("action"):
                continue
            action, arguments = output["action"], output.get("arguments") or {}
            action_counts[f"{role}/{action}"] += 1
            if role == "HI":
                hi_actions += 1
                fingerprint = json.dumps({"a": action, "g": arguments}, sort_keys=True, ensure_ascii=False)
                actor = str(event.get("actor_id"))
                if fingerprint in seen[actor]:
                    hi_repeats += 1
                seen[actor].add(fingerprint)
            if not isinstance(arguments, dict):
                continue
            if action == "INSPECT":
                targets = [t for t in arguments.get("targets") or [] if isinstance(t, str)]
                action_targets.update(targets)
                if (arguments.get("request") or {}).get("operation") in COMPARE_OPS:
                    model_compare_calls += 1
                    compare_targets.update(targets)
            elif action == "READ_RECORD" and isinstance(arguments.get("record_id"), str):
                action_targets.add(arguments["record_id"])
            elif action == "SEARCH" and str(arguments.get("source") or "").startswith("R"):
                action_targets.add(str(arguments["source"]))
        elif kind == "observation" and role in ("HI", "GIA"):
            observation = payload.get("observation") if isinstance(payload, dict) else None
            if isinstance(observation, dict) and observation.get("action") == "HARNESS_REJECTION":
                harness_rejections[role] += 1
            details = observation.get("details") if isinstance(observation, dict) else None
            auto = details.get("auto_compare") if isinstance(details, dict) else None
            for item in auto if isinstance(auto, list) else [auto] if isinstance(auto, dict) else []:
                request = item.get("request") if isinstance(item, dict) else None      # --auto-tools on：运行时代 HI 跑的比对
                if isinstance(request, dict):
                    auto_compares.append([t for t in request.get("targets") or [] if isinstance(t, str)])
    rejected = {role: max(0, raw[role] - accepted[role]) for role in ("HI", "GIA")}
    out.update({"hi_actions": hi_actions, "hi_repeats": hi_repeats,
                "raw_outputs": dict(raw), "accepted_outputs": dict(accepted), "rejected_outputs": rejected,
                "harness_rejections": dict(harness_rejections), "action_counts": dict(action_counts)})

    # ---- store：候选池、打开、下载、比对
    store = run_dir / "store.sqlite3"
    candidates: dict[int, tuple[Any, ...]] = {}
    evidence_to_candidates: dict[str, set[int]] = defaultdict(set)
    acquired: set[int] = set()
    if out["has_store"]:
        for row in _rows(store, "SELECT id, candidate_type, platform_video_id, source_url, final_url FROM candidates") or [
                (*r, None) for r in _rows(store, "SELECT id, candidate_type, platform_video_id, source_url FROM candidates")]:
            candidates[int(row[0])] = row
        for candidate_id, evidence_id, material_kind in _rows(
                store, "SELECT candidate_id, evidence_id, material_kind FROM candidate_evidence"):
            evidence_to_candidates[f"E{evidence_id}"].add(int(candidate_id))
            if material_kind == "video":
                acquired.add(int(candidate_id))
    # 打开、比对只认模型自己的输出（轨迹 model_output）。store.inspect_executions 不分来源：--auto-tools on 时运行时
    # 代跑的 compare_media 也记在 HI 的工作区 ID 下，用它会把自动比对算成模型行为。

    def to_candidates(targets: Iterable[str]) -> set[int]:
        found: set[int] = set()
        for target in targets:
            if re.fullmatch(r"R\d+", target) and int(target[1:]) in candidates:
                found.add(int(target[1:]))
            found |= evidence_to_candidates.get(target, set())
        return found

    opened_all = to_candidates(action_targets)
    compared = to_candidates(compare_targets)
    auto_compared = set().union(*[to_candidates(t) for t in auto_compares]) if auto_compares else set()
    pool = {cid for cid, row in candidates.items() if row[1] in candidate_types}
    opened = opened_all & pool
    out.update({"pool": len(pool), "opened": len(opened), "acquired": len(acquired & pool),
                "compared": len(compared & pool), "compared_of_opened": len(compared & opened),
                "compare_calls": model_compare_calls, "auto_compare_calls": len(auto_compares),
                "auto_compared": len(auto_compared & pool)})

    # ---- 原片（GT source ID，EVID）
    if sources:
        def is_source(row: tuple[Any, ...]) -> bool:
            return bool((row[2] and row[2] in sources)
                        or any(s in str(row[3] or "") or s in str(row[4] or "") for s in sources))

        source_candidates = {cid for cid, row in candidates.items() if is_source(row)}
        out.update({"gt_sources": list(sources),
                    "src_in_pool": bool(source_candidates),
                    "src_opened": bool(source_candidates & opened_all),
                    "src_acquired": bool(source_candidates & acquired),
                    "src_compared": bool(source_candidates & compared)})
    else:
        out.update({"gt_sources": None})

    # ---- run_config：检索快照、冻结、模型设置
    snapshot = run_config.get("search_snapshot") or {}
    counts = snapshot.get("counts") or {}
    freeze = run_config.get("freeze") or {}
    configured = run_config.get("configured_model") or {}
    out.update({
        "snapshot": {"mode": snapshot.get("search_mode") or run_config.get("search_mode"),
                     "root": snapshot.get("snapshot_root"), "hit": counts.get("hit", 0), "miss": counts.get("miss", 0),
                     "live_calls": counts.get("live_calls", 0)} if snapshot or run_config.get("search_mode") else None,
        "freeze": {"applies": freeze.get("applies"), "ok": freeze.get("ok"), "failed": freeze.get("failed") or []}
        if freeze else None,
        "specs": run_config.get("teacher_specs"),
        "model": configured.get("model"), "model_by_role": configured.get("model_by_role"),
        "temperature": configured.get("temperature"), "thinking": configured.get("thinking"),
        "instruction_mode": run_config.get("instruction_mode"),
        "tool_layer_hash": run_config.get("tool_layer_hash"),
    })
    result = _read_json(run_dir / "result.json") if out["has_result"] else None
    if isinstance(result, dict):
        decision = result.get("decision") or result.get("final_decision") or {}
        out["decision"] = decision.get("decision") if isinstance(decision, dict) else None
        out["contract_flags"] = [f for f in result.get("contract_flags") or [] if isinstance(f, str)]
    return out


# ---------------------------------------------------------------- 评分文件里的逐 run 字段（run 目录已删时）

def rows_from_score_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        behavior = entry.get("behavior") or {}
        funnel = entry.get("funnel") or {}
        run_name = str(behavior.get("run") or entry.get("run") or "")
        match = _RUN_SUFFIX.match(run_name)
        rows.append({
            "run": run_name, "case_id": f"evid:{entry.get('sample_id')}", "round": match.group("round") if match else None,
            "sample": int(match.group("sample")) if match else None, "from_score_jsonl": True,
            "has_trajectory": bool(behavior.get("has_trajectory")),
            "hi_actions": behavior.get("hi_actions") or 0, "hi_repeats": behavior.get("hi_repeats") or 0,
            "raw_outputs": {"HI": behavior.get("hi_raw_outputs") or 0, "GIA": behavior.get("gia_raw_outputs") or 0},
            "rejected_outputs": {"HI": behavior.get("hi_rejected") or 0, "GIA": behavior.get("gia_rejected") or 0},
            "legacy": {"source_in_pool": bool(funnel.get("source_in_pool")),
                       "source_downloaded": bool(funnel.get("source_downloaded")),
                       "source_compared": bool(funnel.get("source_compared")),
                       "candidate_opens": behavior.get("candidate_opens") or 0,
                       "video_downloads": behavior.get("video_downloads") or 0},
            "n_gt": entry.get("n_gt"), "passed_gt": entry.get("passed_gt"),
        })
    return rows


# ---------------------------------------------------------------- 旧口径复算（_evid_behavior，原样调用）

def legacy_block(run_dir: Path, sources: list[str] | None) -> dict[str, Any] | None:
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import _evid_behavior as legacy  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    behavior = legacy.run_behavior(run_dir, None)
    funnel = legacy.source_funnel(run_dir, list(sources or [])) if sources is not None else {}
    return {"repeat_rate": behavior.get("repeat_rate"), "hi_actions": behavior.get("hi_actions"),
            "hi_repeats": behavior.get("hi_repeats"),
            "hi_rejected": behavior.get("hi_rejected"), "hi_raw": behavior.get("hi_raw_outputs"),
            "gia_rejected": behavior.get("gia_rejected"), "gia_raw": behavior.get("gia_raw_outputs"),
            "candidate_opens": behavior.get("candidate_opens"), "video_downloads": behavior.get("video_downloads"),
            "source_in_pool": bool(funnel.get("source_in_pool")), "source_downloaded": bool(funnel.get("source_downloaded")),
            "source_compared": bool(funnel.get("source_compared"))}


# ---------------------------------------------------------------- 汇总

def summarize(rows: list[dict[str, Any]], *, min_denominator: int = MIN_DENOMINATOR) -> dict[str, Any]:
    present = [r for r in rows if r.get("has_trajectory")]
    evid = [r for r in present if r.get("gt_sources")]
    raw_runs = [r for r in present if not r.get("from_score_jsonl")]
    total = lambda key, rs=present: sum(int(r.get(key) or 0) for r in rs)  # noqa: E731
    count = lambda key, rs: sum(1 for r in rs if r.get(key))  # noqa: E731
    raw = {role: sum(int((r.get("raw_outputs") or {}).get(role) or 0) for r in present) for role in ("HI", "GIA")}
    rej = {role: sum(int((r.get("rejected_outputs") or {}).get(role) or 0) for r in present) for role in ("HI", "GIA")}
    in_pool = count("src_in_pool", evid)
    summary: dict[str, Any] = {
        "runs": len(rows), "runs_with_trajectory": len(present), "evid_runs": len(evid),
        "open_rate": {**ratio(count("src_opened", evid), in_pool),
                      "judgeable": in_pool >= min_denominator, "min_denominator": min_denominator}
        if raw_runs else None,
        "compare_rate": ratio(total("compared_of_opened", raw_runs), total("opened", raw_runs)) if raw_runs else None,
        "repeat_rate": ratio(total("hi_repeats"), total("hi_actions")),
        "format_reject_rate": role_mean(ratio(rej["HI"], raw["HI"]), ratio(rej["GIA"], raw["GIA"])),
        "companions": {
            "open_rate_runs": ratio(count("src_opened", evid), len(evid)) if raw_runs else None,
            "src_in_pool_runs": ratio(in_pool, len(evid)) if raw_runs else None,
            "compare_rate_source": ratio(count("src_compared", evid), count("src_opened", evid)) if raw_runs else None,
            "candidate_open_rate": ratio(total("opened", raw_runs), total("pool", raw_runs)) if raw_runs else None,
            "compare_rate_of_acquired": ratio(total("compared", raw_runs), total("acquired", raw_runs)) if raw_runs else None,
            "auto_compare_runs": ratio(sum(1 for r in raw_runs if r.get("auto_compare_calls")), len(raw_runs)) if raw_runs else None,
            "format_reject_rate_pooled": ratio(rej["HI"] + rej["GIA"], raw["HI"] + raw["GIA"]),
            "format_reject_rate_hi": ratio(rej["HI"], raw["HI"]),
            "format_reject_rate_gia": ratio(rej["GIA"], raw["GIA"]),
        },
    }
    snapshots = [r["snapshot"] for r in raw_runs if r.get("snapshot")]
    if snapshots:
        hit = sum(s["hit"] for s in snapshots)
        miss = sum(s["miss"] for s in snapshots)
        summary["snapshot"] = {"lookups": hit + miss, "hit": hit, "miss_recorded_live": miss,
                               "new_record_rate": ratio(miss, hit + miss),
                               "modes": dict(Counter(str(s["mode"]) for s in snapshots)),
                               "roots": dict(Counter(str(s["root"]) for s in snapshots))}
    freezes = [r["freeze"] for r in raw_runs if r.get("freeze")]
    if freezes:
        failed = Counter(item for f in freezes for item in f.get("failed") or [])
        summary["freeze"] = {"runs_checked": len(freezes), "runs_ok": sum(1 for f in freezes if f.get("ok")),
                             "failed_items": dict(failed)}
    configured = [r for r in raw_runs if r.get("has_run_config")]
    summary["settings"] = {
        "runs_with_run_config": len(configured),
        "models": dict(Counter(str(r.get("model")) for r in configured)),
        "temperatures": dict(Counter(str(r.get("temperature")) for r in configured)),
        "specs": dict(Counter(json.dumps(r.get("specs"), sort_keys=True) for r in configured)),
        "instruction_modes": dict(Counter(str(r.get("instruction_mode")) for r in configured)),
        "tool_layer_hashes": dict(Counter(str(r.get("tool_layer_hash")) for r in configured)),
    } if configured else None
    scored = [r for r in rows if isinstance(r.get("n_gt"), int) and r.get("n_gt")]
    if scored:
        points = sum(int(r.get("passed_gt") or 0) for r in scored)
        gt = sum(int(r["n_gt"]) for r in scored)
        summary["score"] = {"point_accuracy": ratio(points, gt),
                            "video_accuracy": ratio(sum(1 for r in scored if int(r.get("passed_gt") or 0) == int(r["n_gt"])),
                                                    len(scored)),
                            "runs_scored": len(scored)}
    legacy = [r["legacy"] for r in rows if r.get("legacy")]
    if legacy:
        lr = lambda key: sum(int(x.get(key) or 0) for x in legacy)  # noqa: E731
        lc = lambda key: sum(1 for x in legacy if x.get(key))  # noqa: E731
        summary["legacy"] = {
            "note": "旧口径（_evid_behavior 原样复算），与《EVID_baseline效果总表》对照",
            "source_in_pool": lc("source_in_pool"), "source_downloaded": lc("source_downloaded"),
            "source_compared": lc("source_compared"),
            "candidate_opens": lr("candidate_opens"), "runs_with_candidate_open": lc("candidate_opens"),
            "video_downloads": lr("video_downloads"), "runs_with_download": lc("video_downloads"),
            "repeat_rate": summary["repeat_rate"]["value"],
            "hi_rejection_rate": summary["companions"]["format_reject_rate_hi"]["value"],
            "gia_rejection_rate": summary["companions"]["format_reject_rate_gia"]["value"],
        }
    return summary


def attach_scores(rows: list[dict[str, Any]], score_files: list[Path]) -> int:
    by_run: dict[str, dict[str, Any]] = {}
    for path in score_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                name = (entry.get("behavior") or {}).get("run") or entry.get("run")
                if name:
                    by_run[str(name)] = entry
    attached = 0
    for row in rows:
        entry = by_run.get(row["run"])
        if entry is not None:
            row["n_gt"], row["passed_gt"] = entry.get("n_gt"), entry.get("passed_gt")
            attached += 1
    return attached


# ---------------------------------------------------------------- 成功标准判定

def _metric_value(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    if name in summary and isinstance(summary[name], dict):
        return summary[name]
    return (summary.get("companions") or {}).get(name)


def _meets(value: float, direction: str, target: float) -> bool:
    return value >= target - 1e-9 if direction == ">=" else value <= target + 1e-9


def _regressed(value: float | None, base: float | None, direction: str, tolerance: float) -> bool | None:
    """比 E2 基线差超过容差（百分点，绝对值）。缺任一边时返回 None。"""

    if value is None or base is None:
        return None
    return value < base - tolerance - 1e-9 if direction == ">=" else value > base + tolerance + 1e-9


def evaluate_criteria(criteria: dict[str, Any], groups: dict[str, dict[str, Any]],
                      baseline_groups: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """按预先登记的 success_criteria.json 逐条判定。每项：达标（目标数）且不比 E2 倒退超过容差 = 通过。"""

    items: list[dict[str, Any]] = []
    for item in criteria.get("behavior") or []:
        set_name = item.get("set") or "all"
        summary = groups.get(set_name) or {}
        base_summary = (baseline_groups or {}).get(set_name) or {}
        name, direction, target = item["metric"], item["direction"], float(item["target"])
        metric = _metric_value(summary, name) or {}
        used = {"metric": name, "direction": direction, "target": target, "value": metric.get("value"),
                "num": metric.get("num"), "den": metric.get("den")}
        status = "judged"
        if metric.get("value") is None or (metric.get("den") or 0) < int(item.get("min_denominator") or 0):
            fallback = item.get("fallback")
            if not fallback:
                status = "not_judgeable"
            else:
                fb = _metric_value(summary, fallback["metric"]) or {}
                used = {"metric": fallback["metric"], "direction": fallback["direction"], "target": float(fallback["target"]),
                        "value": fb.get("value"), "num": fb.get("num"), "den": fb.get("den")}
                status = "fallback" if fb.get("value") is not None else "not_judgeable"
        base_value = (_metric_value(base_summary, used["metric"]) or {}).get("value") if baseline_groups is not None else None
        meets = None if status == "not_judgeable" else _meets(used["value"], used["direction"], used["target"])
        regressed = _regressed(used["value"], base_value, used["direction"], float(item.get("regression_tolerance", 0.0)))
        items.append({"id": item.get("id"), "name": item.get("name"), "set": set_name, "status": status,
                      "primary_metric": name, "primary_value": metric.get("value"),
                      "primary_num": metric.get("num"), "primary_den": metric.get("den"),
                      "used_metric": used["metric"], "target": f"{used['direction']} {used['target']}",
                      "value": used["value"], "num": used["num"], "den": used["den"], "baseline": base_value,
                      "meets_target": meets, "regressed": regressed,
                      "pass": None if meets is None else bool(meets and not regressed)})
    score = None
    rule = criteria.get("score")
    if rule:
        set_name = rule.get("set") or "all"
        point = (((groups.get(set_name) or {}).get("score") or {}).get("point_accuracy") or {})
        base = ((((baseline_groups or {}).get(set_name) or {}).get("score") or {}).get("point_accuracy") or {})
        value, base_value = point.get("value"), base.get("value")
        score = {"id": rule.get("id", "S1"), "set": set_name, "value": value, "num": point.get("num"), "den": point.get("den"),
                 "baseline": base_value}
        if value is None:
            score.update({"status": "not_judgeable", "pass": None})
        else:
            checks = {"abs_floor": value >= float(rule["min_abs"]) - 1e-9}
            if base_value is not None:
                checks["gain"] = value - base_value >= float(rule["min_gain_vs_baseline"]) - 1e-9
            regressed = None if base_value is None else value < base_value - float(rule["regression_tolerance"]) - 1e-9
            score.update({"status": "judged", "checks": checks, "regressed": regressed,
                          "pass": all(checks.values()) and "gain" in checks,
                          "stretch_reached": value >= float(rule["stretch"]) - 1e-9})
    overall_rule = criteria.get("overall") or {}
    comparability: dict[str, Any] = {}
    if baseline_groups is not None:
        now = (groups.get("all") or {}).get("settings") or {}
        base = (baseline_groups.get("all") or {}).get("settings") or {}
        if not now or not base:
            comparability["settings_present"] = {"after": bool(now), "baseline": bool(base), "same": False}
        for key in ("tool_layer_hashes", "specs", "temperatures", "instruction_modes") if now and base else ():
            comparability[key] = {"after": sorted(now.get(key) or {}), "baseline": sorted(base.get(key) or {}),
                                  "same": sorted(now.get(key) or {}) == sorted(base.get(key) or {})
                                  and len(now.get(key) or {}) == 1}
        allowed = set(criteria.get("freeze_expected_failures") or [])
        for side, source in (("after", groups), ("baseline", baseline_groups)):
            failed = ((source.get("all") or {}).get("freeze") or {}).get("failed_items") or {}
            comparability[f"freeze_unexpected_failures_{side}"] = sorted(k for k in failed if k not in allowed)
    passed = sum(1 for i in items if i["pass"])
    any_regressed = any(i["regressed"] for i in items) or bool(score and score.get("regressed"))
    comparable = all(v["same"] for v in comparability.values() if isinstance(v, dict)) and \
        not any(v for k, v in comparability.items() if k.startswith("freeze_unexpected_failures"))
    if baseline_groups is not None and not comparable:
        overall = "not_comparable"
    elif items and passed == len(items) and score and score.get("pass"):
        overall = "success"
    elif passed >= int(overall_rule.get("diagnostic_min_behavior_pass", len(items))) and not any_regressed:
        overall = "diagnostic_success"
    else:
        overall = "fail"
    return {"items": items, "score": score,
            "overall": {"verdict": overall, "behavior_passed": passed, "behavior_total": len(items),
                        "any_regressed": any_regressed, "comparability": comparability, "labels": overall_rule.get("labels")}}


# ---------------------------------------------------------------- 发现 run、输出

def discover(paths: list[Path], rounds: list[str] | None, ids: list[str] | None) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        path = Path(path)
        candidates = [path] if (path / "trajectory.jsonl").is_file() or (path / "store.sqlite3").is_file() else (
            sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else [])
        for run_dir in candidates:
            if ".failed-" in run_dir.name:
                continue
            match = _RUN_SUFFIX.match(run_dir.name)
            if rounds and (not match or match.group("round") not in rounds):
                continue
            if ids:
                key = match.group("case") if match else run_dir.name
                if not any(key == i or key.endswith("_" + i) or key == re.sub(r"[^0-9A-Za-z_.-]+", "_", i) for i in ids):
                    continue
            found.append(run_dir)
    return found


def group_rows(rows: list[dict[str, Any]], eval_set: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    if eval_set:
        for name, spec in (eval_set.get("sets") or {}).items():
            wanted = {c["case_id"] for c in spec.get("cases") or []}
            groups[name] = [r for r in rows if r.get("case_id") in wanted]
    for row in rows:
        groups.setdefault(f"round:{row.get('round')}", []).append(row)
    return groups


def _pct(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("value") is None:
        return "—" if not metric else f"—（{metric.get('num')}/{metric.get('den')}）"
    if metric.get("macro_over_roles") is not None:
        hi, gia = (metric.get("roles") or {}).get("HI"), (metric.get("roles") or {}).get("GIA")
        parts = [f"{r} {100 * m['value']:.1f}%（{m['num']}/{m['den']}）" for r, m in (("HI", hi), ("GIA", gia))
                 if m and m.get("value") is not None]
        return f"{100 * metric['value']:.1f}%（{'，'.join(parts)}）"
    ci = metric.get("ci95")
    return f"{100 * metric['value']:.1f}%（{metric['num']}/{metric['den']}{'' if not ci else f'，95% 区间 {100 * ci[0]:.0f}–{100 * ci[1]:.0f}%'}）"


def markdown(title: str, groups: dict[str, dict[str, Any]], verdicts: list[dict[str, Any]] | None,
             meta: dict[str, Any]) -> str:
    lines = [f"# {title}", "", f"- 指标脚本：{meta['metrics_version']}（{meta['script_sha256']}）",
             f"- 输入：{meta['inputs']}", ""]
    names = [n for n in groups if not n.startswith("round:")] + [n for n in groups if n.startswith("round:")]
    lines += ["| 指标 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [("① 打开候选率（原片进池后被打开）", "open_rate"), ("② 比对率（打开过的候选里被比对的）", "compare_rate"),
            ("③ 重复调用率（HI）", "repeat_rate"), ("④ 格式拒率（HI、GIA 各自拒率的平均）", "format_reject_rate")]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(_pct(groups[n].get(key)) for n in names) + " |")
    for label, key in (("　伴随：打开了原片的 run / EVID run", "open_rate_runs"), ("　伴随：原片进池的 run", "src_in_pool_runs"),
                       ("　伴随：原片打开后被比对", "compare_rate_source"), ("　伴随：候选打开 / 进池（候选级）", "candidate_open_rate"),
                       ("　伴随：运行时自动比对过的 run（--auto-tools on，不计入②）", "auto_compare_runs"),
                       ("　伴随：格式拒率 HI+GIA 按次数合计", "format_reject_rate_pooled"),
                       ("　伴随：格式拒率 HI", "format_reject_rate_hi"), ("　伴随：格式拒率 GIA", "format_reject_rate_gia")):
        lines.append(f"| {label} | " + " | ".join(_pct((groups[n].get("companions") or {}).get(key)) for n in names) + " |")
    lines.append("| 分数：点级（Sonnet） | " + " | ".join(_pct((groups[n].get("score") or {}).get("point_accuracy"))
                                               for n in names) + " |")
    lines.append("| 检索快照新录率（未命中、联网补录） | " + " | ".join(
        _pct((groups[n].get("snapshot") or {}).get("new_record_rate")) for n in names) + " |")
    lines.append("| run 数（有轨迹 / 其中 EVID） | " + " | ".join(
        f"{groups[n]['runs']}（{groups[n]['runs_with_trajectory']} / {groups[n]['evid_runs']}）" for n in names) + " |")
    legacy = groups["all"].get("legacy")
    if legacy:
        lines += ["", "## 旧口径复算（与《EVID_baseline效果总表》对照）", "",
                  f"- 原片进池 / 下载 / 比对（旧定义）：{legacy['source_in_pool']} / {legacy['source_downloaded']} / "
                  f"{legacy['source_compared']}",
                  f"- 候选动手 {legacy['candidate_opens']} 次（{legacy['runs_with_candidate_open']} 个 run）；"
                  f"候选视频下载 {legacy['video_downloads']} 次（{legacy['runs_with_download']} 个 run）",
                  f"- 重复调用率 {legacy['repeat_rate']}；输出被拒率 HI {legacy['hi_rejection_rate']}、GIA {legacy['gia_rejection_rate']}"]
    freeze = groups["all"].get("freeze")
    settings = groups["all"].get("settings")
    if freeze or settings:
        lines += ["", "## 口径核对", ""]
        if freeze:
            lines.append(f"- 冻结核对：{freeze['runs_ok']}/{freeze['runs_checked']} 个 run ok；未过项 {freeze['failed_items'] or '无'}")
        if settings:
            lines.append(f"- 有 run_config 的 run：{settings['runs_with_run_config']} 个（下面各项只统计这些）")
            for key, label in (("models", "模型"), ("temperatures", "温度"), ("specs", "规格"),
                               ("instruction_modes", "指令模式"), ("tool_layer_hashes", "工具层 hash")):
                lines.append(f"- {label}：{settings[key]}")
    if verdicts:
        yes = {True: "是", False: "否", None: "—"}
        lines += ["", "## 成功标准判定（预先登记，见 success_criteria.json）", "",
                  "| 条目 | 指标（集合） | 目标 | 实测 | E2 基线 | 达标 | 比 E2 倒退 | 结论 |", "|---|---|---|---|---|---|---|---|"]
        for v in verdicts["items"]:
            used = v["used_metric"] + ("（分母不足，改看伴随指标）" if v["status"] == "fallback" else "")
            lines.append(f"| {v['id']} {v.get('name') or ''} | {used}（{v['set']}） | {v['target']} | {v['value']}（{v['num']}/{v['den']}） | "
                         f"{v['baseline']} | {yes[v['meets_target']]} | {yes[v['regressed']]} | "
                         f"{ {True: '通过', False: '未通过', None: '无法判定'}[v['pass']] } |")
        score = verdicts.get("score")
        if score:
            result = {True: "通过", False: "未通过", None: "无法判定"}[score["pass"]]
            if score.get("stretch_reached"):
                result += "；达到 27B 水平"
            lines.append(f"| {score['id']} 分数 | 点级（{score['set']}） | 见标准 | {score['value']}（{score.get('num')}/{score.get('den')}） | "
                         f"{score['baseline']} | {score.get('checks')} | {yes[score.get('regressed')]} | {result} |")
        o = verdicts["overall"]
        label = (o.get("labels") or {}).get(o["verdict"], o["verdict"])
        lines += ["", f"**总判定：{label}**（行为指标通过 {o['behavior_passed']}/{o['behavior_total']}；有无倒退：{yes[o['any_regressed']]}）"]
        for key, value in (o.get("comparability") or {}).items():
            lines.append(f"- 与 E2 同口径 · {key}：{value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="r0 评测的四个行为指标（打开候选率、比对率、重复调用率、格式拒率）")
    parser.add_argument("paths", nargs="*", type=Path, help="run 目录，或包含 run 目录的父目录")
    parser.add_argument("--round", dest="rounds", action="append", default=None, help="只算这些轮次（可多次给）")
    parser.add_argument("--ids", nargs="+", default=None, help="只算这些案例（video_002 / evid:video_002 / vcsl:pair_4）")
    parser.add_argument("--from-score-jsonl", nargs="+", type=Path, default=None,
                        help="run 目录已删时：用 _evid_score.py 评分 jsonl 里逐 run 存的行为字段（只能算 ③④ 与旧漏斗）")
    parser.add_argument("--scores", nargs="+", type=Path, default=None, help="_evid_score.py 的评分 jsonl，并入点级分数")
    parser.add_argument("--eval-set", type=Path, default=None, help="评测集清单（按 sets 分组汇总）")
    parser.add_argument("--evid-bench", type=Path, default=DEFAULT_EVID_BENCH, help="EVID GT（读 source 字段）")
    parser.add_argument("--candidate-types", nargs="+", default=list(CANDIDATE_TYPES))
    parser.add_argument("--min-denominator", type=int, default=MIN_DENOMINATOR)
    parser.add_argument("--no-legacy", action="store_true", help="不跑旧口径复算（_evid_behavior）")
    parser.add_argument("--criteria", type=Path, default=None, help="success_criteria.json：逐条判定")
    parser.add_argument("--baseline", type=Path, default=None, help="E2 基线的本脚本输出 .json（判定\"不比 E2 差\"与分数增益）")
    parser.add_argument("--out", type=Path, default=None, help="输出前缀：写 <out>.json 与 <out>.md")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    sources = load_evid_sources(args.evid_bench)
    rows: list[dict[str, Any]] = []
    for run_dir in discover(args.paths, args.rounds, args.ids):
        case_id, _, _ = case_of(run_dir, _read_json(run_dir / "run_config.json"))
        sample = evid_sample(case_id)
        gt = sources.get(sample) if sample else None
        row = run_metrics(run_dir, gt, candidate_types=tuple(args.candidate_types))
        if not args.no_legacy and sample:
            row["legacy"] = legacy_block(run_dir, gt or [])
        rows.append(row)
    for path in args.from_score_jsonl or []:
        rows.extend(rows_from_score_jsonl(path))
    if not rows:
        print("没有找到 run（检查路径、--round、--ids）", file=sys.stderr)
        return 2
    if args.scores:
        print(f"并入分数：{attach_scores(rows, args.scores)} 个 run", file=sys.stderr)

    eval_set = _read_json(args.eval_set) if args.eval_set else None
    grouped = group_rows(rows, eval_set)
    summaries = {name: summarize(rs, min_denominator=args.min_denominator) for name, rs in grouped.items() if rs}
    verdicts = None
    if args.criteria:
        criteria = _read_json(args.criteria) or {}
        baseline = (_read_json(args.baseline) or {}).get("groups") if args.baseline else None
        verdicts = evaluate_criteria(criteria, summaries, baseline)
    meta = {"metrics_version": METRICS_VERSION, "script_sha256": _sha256_file(Path(__file__)),
            "legacy_script_sha256": _sha256_file(ROOT / "scripts" / "_evid_behavior.py"),
            "inputs": [str(p) for p in args.paths] + [str(p) for p in args.from_score_jsonl or []],
            "rounds": args.rounds, "candidate_types": args.candidate_types, "compare_ops": list(COMPARE_OPS),
            "eval_set": str(args.eval_set) if args.eval_set else None,
            "criteria": str(args.criteria) if args.criteria else None,
            "criteria_sha256": _sha256_file(args.criteria) if args.criteria else None}
    report = {"meta": meta, "groups": summaries, "verdicts": verdicts, "runs": rows}
    title = args.title or f"r0 行为指标：{', '.join(args.rounds or []) or ', '.join(str(p) for p in args.paths)}"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(f"{args.out}.md").write_text(markdown(title, summaries, verdicts, meta), encoding="utf-8")
        print(f"写入 {args.out}.json / .md", file=sys.stderr)
    print(json.dumps({name: {k: s.get(k) for k in ("open_rate", "compare_rate", "repeat_rate", "format_reject_rate", "score")}
                      for name, s in summaries.items() if not name.startswith("round:") or len(summaries) <= 4},
                     ensure_ascii=False, indent=1))
    if verdicts:
        print(json.dumps(verdicts["overall"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
