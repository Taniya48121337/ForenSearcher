"""找原片实验的检索臂自检（10-06）：跑模型之前，用真实网络把每个臂的工具层走一遍，坏了就不浪费一整晚。

用与 run 完全相同的搜索链（run_cases.search_service_for + 查询守卫），查询是与 EVID 无关的中性词，不录快照、不进任何 run：

    1 results10   文字 → 视频，应给满 10 个 R
    2 guardlen    一条 20 词的长查询，应被拒收并提示"3-8 个词"
    3 imageclip   取第 1 步第 3 名视频的 YouTube 自动截帧（hq2.jpg）当查询图，text_image 同一句文字：
                  缩略图下载、CLIP 装载、重排要跑通（有 clip_score）；它被排到第 1 另记（重排确实在起作用）
    4 image-only  纯图片查询：重排已有候选，应有结果
    5 fanout5     文字 → 视频，应给出扇出合并后的 15 个 R

退出码：1–4 都通过为 0；否则 1（夜跑会跳过依赖这些工具的臂）。结果写 experiments/evid/scores/arm_selftest.json。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUERY = "NASA Artemis I rocket launch"
LONG = ("rocket on the launch pad at night with bright flames and smoke clouds while people watch from far away "
        "on a beach")


def _case(tmp: Path):
    from foren_searcher.datagen.cases import Case

    path = tmp / "case.json"
    path.write_text(json.dumps({"case_id": "selftest:arms", "dataset": "selftest", "source_id": "arms", "split": "dev",
                                "task_kind": "video_diff", "task_text": "self test", "family_key": "selftest:arms",
                                "media": [], "source_policy": {"mode": "unrestricted"}}), encoding="utf-8")
    return Case.load(path)


def _service(tmp: Path, arm: str):
    from foren_searcher.datagen.run_cases import search_service_for
    from foren_searcher.search.arms import parse_search_arm
    from foren_searcher.search.query_guard import QueryGuard
    from foren_searcher.store import GlobalStore

    out = tmp / arm.replace(",", "_")
    out.mkdir()
    store = GlobalStore(out / "store.sqlite3", out / "artifacts")
    features = parse_search_arm(arm)
    guard = QueryGuard(store.has_reference, video_rules="guardlen" in features)
    search, _, _ = search_service_for(_case(tmp), store, out, query_guard=guard, search_arm=features)
    return search, store


def _discoveries(store, execution: int | None = None) -> list[tuple]:
    with store._connect() as connection:
        sql = ("SELECT d.execution_id, c.platform_video_id, d.provider, d.engine, d.provider_rank, c.extra_json "
               "FROM candidate_discoveries d JOIN candidates c ON c.id = d.candidate_id")
        rows = [tuple(r) for r in connection.execute(sql + " ORDER BY d.id").fetchall()]
    last = max((r[0] for r in rows), default=None) if execution is None else execution
    return [r for r in rows if r[0] == last]


async def main() -> int:
    import httpx

    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    checks = report["checks"]

    def record(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": ok, "detail": detail}
        print(f"  [{'通过' if ok else '失败'}] {name}：{detail}", flush=True)

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        search, store = _service(tmp, "results10,guardlen,imageclip")
        try:
            response = await search.execute({"input_type": "text", "query": QUERY, "output_type": "video"}, workspace_id="W1-1")
            rows = _discoveries(store)
            record("1 results10", len(response.results) == 10 and not response.error,
                   f"{len(response.results)} 个 R，stop={response.stop_reason}，错误 {response.error and response.error.code}")
            response = await search.execute({"input_type": "text", "query": LONG, "output_type": "video"}, workspace_id="W1-1")
            message = response.error.message if response.error else ""
            record("2 guardlen", bool(response.error) and response.error.code == "INVALID_INPUT" and "3-8" in message,
                   message[:100] or "没有被拒收")
            target = rows[2][1] if len(rows) >= 3 else None
            image = None
            if target:
                async with httpx.AsyncClient(timeout=15) as client:
                    got = await client.get(f"https://i.ytimg.com/vi/{target}/hq2.jpg")
                if got.status_code == 200:
                    image = got.content
            if image is None:
                record("3 imageclip", False, f"取不到查询图（第 3 名 {target}）")
            else:
                evidence = store.record_evidence(
                    content="self-test frame", modalities=("image",), source_ref="V0", locator="complete",
                    acquired_by="INSPECT/get_frames", discovered_by="W1-1",
                    artifact_ref=store.save_artifact(image, media_type="image/jpeg", suffix=".jpg", role="source_material"))
                started = time.monotonic()
                response = await search.execute({"input_type": "text_image",
                                                 "query": {"evidence_id": evidence.evidence_id, "text": QUERY},
                                                 "output_type": "video"}, workspace_id="W1-1")
                ranked = _discoveries(store)
                top = ranked[0] if ranked else None
                from foren_searcher.datagen.run_cases import _find_provider

                router = _find_provider(search, "ImageQueryRouter")
                stats = router.image_provider.stats if router is not None else {}
                score = (stats.get("last_top") or [[None, None]])[0][1]       # 已在池里的候选，extra 不会被覆盖，分数看 provider 计数
                # 关键：缩略图下载 + CLIP 装载 + 重排跑通（有 clip_score）。查询图那条视频若在前 30 里应排第 1（分数≈1）；
                # YouTube 两次搜索结果偶有出入，它不在前 30 时不算失败，只记下来。
                ok = bool(top) and top[2] == "thumbnail_clip" and score is not None
                report["imageclip_rank_check"] = bool(top) and top[1] == target
                record("3 imageclip", ok, f"查询图来自 {target}（原第 3 名）→ 重排后第 1 = {top and top[1]}"
                       f"（{'就是它' if report['imageclip_rank_check'] else '不是它'}），provider "
                       f"{top and top[2]}，clip_score {score}；{time.monotonic() - started:.1f} 秒（含首次装 CLIP）"
                       + (f"；错误 {response.error.code}: {response.error.message[:80]}" if response.error else ""))
                response = await search.execute({"input_type": "image", "query": evidence.evidence_id, "output_type": "video"},
                                                workspace_id="W1-1")
                record("4 image-only", bool(response.results) and not response.error,
                       f"{len(response.results)} 个 R" + (f"；错误 {response.error.code}: {response.error.message[:100]}"
                                                         if response.error else ""))
        finally:
            await search.aclose()
        search, store = _service(tmp, "results10,guardlen,fanout5")
        try:
            response = await search.execute({"input_type": "text", "query": QUERY + " 2022 Kennedy", "output_type": "video"},
                                            workspace_id="W1-1")
            engines = {r[3] for r in _discoveries(store)}
            record("5 fanout5", len(response.results) >= 10 and "youtube_search_fanout" in engines,
                   f"{len(response.results)} 个 R，engine {sorted(engines)}"
                   + (f"；错误 {response.error.code}" if response.error else ""))
        finally:
            await search.aclose()
    critical = all(checks.get(name, {}).get("ok") for name in ("1 results10", "2 guardlen", "3 imageclip", "4 image-only"))
    report["critical_ok"] = critical
    report["fanout_ok"] = bool(checks.get("5 fanout5", {}).get("ok"))
    out = ROOT / "experiments" / "evid" / "scores" / "arm_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检索臂自检：{'全部关键项通过' if critical else '有关键项没通过'}；写入 {out}", flush=True)
    return 0 if critical else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
