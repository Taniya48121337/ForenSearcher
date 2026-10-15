"""找原片实验（claude/EVID_找原片实验计划_20261006.md）的检索臂与漏斗列。

都在 run_cases --search-arm 给了时才生效；不给（A0）时搜索链、动作目录、查询守卫与以前完全相同。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.search.arms import (
    NO_RESULTS, FanoutVideoProvider, fanout_queries, parse_search_arm, query_words, video_policy_for, video_query_problem,
)
from foren_searcher.search.contracts import (
    CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchQueryRequest,
)
from foren_searcher.search.policy import SearchPolicy

ROOT = Path(__file__).resolve().parents[1]


def _yt(video_id: str, rank: int = 1, title: str = "t") -> CandidateSource:
    return CandidateSource(f"https://www.youtube.com/watch?v={video_id}", title, None, "yt-dlp", "youtube_search", rank,
                           "video", platform="youtube", platform_video_id=video_id)


class _Videos:
    """按查询文字返回 n 个候选；记录每次调用。"""

    def __init__(self, n: int = 30, empty: set[str] | None = None) -> None:
        self.n, self.empty, self.calls = n, empty or set(), []

    async def search(self, request, *, offset, limit, image_path=None):
        self.calls.append((request.query, offset, limit))
        if request.query in self.empty:
            return CandidateBatch("yt-dlp", "youtube_search", ())
        key = abs(hash(request.query)) % 1000
        items = tuple(_yt(f"v{key:03d}x{i:02d}", i + 1) for i in range(self.n))
        return CandidateBatch("yt-dlp", "youtube_search", items[offset:offset + limit])


class _Opener:
    async def open(self, candidate, request):
        material = OpenedMaterial("page", "ready", "[1] source", b"<p>x</p>", "text/html", ".html", ("text",),
                                  "source page", "paragraph 1", "SEARCH/open/test",
                                  "source_metadata" if request.output_type == "video" else "source_excerpt")
        if request.output_type == "video":
            return OpenedSource(candidate.source_url, (material,), {"status": "formats_found", "reason": None},
                                {"status": "not_requested", "evidence_id": None, "error": None})
        return OpenedSource(candidate.source_url, (material,))


def test_parse_search_arm():
    assert parse_search_arm(None) == frozenset() and parse_search_arm("a0") == frozenset()
    assert parse_search_arm("results10, guardlen,imageclip") == {"results10", "guardlen", "imageclip"}
    with pytest.raises(ValueError):
        parse_search_arm("results10,magic")


@pytest.mark.parametrize("query,bad", [
    ("Nanfang'ao Bridge collapse 01:23:09 EDT", "date or time"),
    ("FORBIDDEN ZONE - NORTH 2026/05/20 camping tent forest", "date or time"),
    ("StreamTime LIVE Baltimore MD 2024-03-26", "date or time"),
    ("lamp with fringed shade, Chanel Fendi poster, hand with ring, sweet potato, honey drizzle, asparagus in pan", "too long"),
    ("一男子运动中喝下不明液体后竟晕倒 男子 绿色短裤 白色运动鞋 墙上Pulp Fiction海报", "too long"),
    ("Nanfang'ao Bridge collapse 2019", None),
    ("kurashimono journal", None),
    ("Nanfang'ao Bridge collapse October 1 2019 video", None),
])
def test_guardlen_rules(query, bad):
    problem = video_query_problem(query)
    assert (problem is None) == (bad is None)
    if bad:
        assert bad in problem and "3-8 distinctive words" in problem


def test_query_words_counts_cjk_pairs():
    assert query_words("kurashimono journal") == 2
    assert query_words("一男子运动中喝下") == 4


def test_query_guard_video_rules_only_when_enabled_and_only_for_video():
    from foren_searcher.search.query_guard import QueryGuard

    long = SearchQueryRequest("text", "a b c d e f g h i j k l m n", "video")
    QueryGuard().check(long)                                         # A0：照旧放行
    guard = QueryGuard(video_rules=True)
    guard.check(SearchQueryRequest("text", "a b c d e f g h i j k l m n", "text"))     # 网页搜索不管
    with pytest.raises(SearchFailure) as caught:
        guard.check(long)
    assert caught.value.error.code == "INVALID_INPUT" and "too long" in caught.value.error.message
    assert guard.summary()["rejected_video_rules"] == 1
    assert "video_rules" not in QueryGuard().summary()
    QueryGuard(lambda token: True, video_rules=True).check(SearchQueryRequest("image", "E11", "video"))   # 纯图片：E 编号不是查询文字
    with pytest.raises(SearchFailure):
        QueryGuard(lambda token: True).check(SearchQueryRequest("text_image", {"evidence_id": "E11", "text": "E3 clip"}, "video"))


def test_fanout_queries_rewrites_without_model():
    rewrites = fanout_queries('Nanfang\'ao Bridge (Old) collapse 01:23:09 EDT original footage of E1 "bridge falls"')
    assert len(rewrites) <= 5 and rewrites[0].startswith("Nanfang'ao")
    assert all("01:23:09" not in r and "E1" not in r for r in rewrites[1:])
    assert len({r.lower() for r in rewrites}) == len(rewrites)


@pytest.mark.asyncio
async def test_fanout_provider_merges_round_robin():
    inner = _Videos(n=10)
    provider = FanoutVideoProvider(inner)
    batch = await provider.search(SearchQueryRequest("text", "Nanfang'ao Bridge collapse 2019 video footage", "video"),
                                  offset=0, limit=15)
    assert len(batch.candidates) == 15 and len({c.platform_video_id for c in batch.candidates}) == 15
    assert [c.provider_rank for c in batch.candidates] == list(range(1, 16))
    firsts = {c.extra["fanout_query"] for c in batch.candidates[:len(inner.calls)]}
    assert len(firsts) == len(inner.calls) >= 3                     # 前几名来自不同改写，每条的第 1 名
    text = await provider.search(SearchQueryRequest("text", "x", "text"), offset=0, limit=5)
    assert len(text.candidates) == 5                                # 非视频查询原样透传


@pytest.mark.asyncio
async def test_explain_empty_replaces_ddgs_error_only_when_enabled():
    from foren_searcher.search.providers.video_fallback import VideoSearchFallbackProvider

    class _Broken:
        async def search(self, request, *, offset, limit, image_path=None):
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "Free search failed (DDGSException)"))

    request = SearchQueryRequest("text", "q", "video")
    plain = VideoSearchFallbackProvider(_Videos(empty={"q"}), _Broken())
    with pytest.raises(SearchFailure) as caught:
        await plain.search(request, offset=0, limit=5)
    assert caught.value.error.code == "PROVIDER_ERROR"              # A0 不变
    explained = VideoSearchFallbackProvider(_Videos(empty={"q"}), _Broken(), explain_empty=True)
    with pytest.raises(SearchFailure) as caught:
        await explained.search(request, offset=0, limit=5)
    assert caught.value.error.code == "NOT_FOUND" and caught.value.error.message == NO_RESULTS
    assert explained.summary()["video_empty_explained"] == 1


@pytest.mark.asyncio
async def test_video_policy_gives_ten_results_for_video_only():
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    service = SearchService(store=GlobalStore(), provider=_Videos(), opener=_Opener())
    assert service.video_policy is None
    service.video_policy = video_policy_for(service.policy, frozenset({"results10"}))
    video = await service.execute({"input_type": "text", "query": "abc", "output_type": "video"}, workspace_id="W1-1")
    assert len(video.results) == 10 and video.stop_reason == "target_reached"
    text = await service.execute({"input_type": "text", "query": "abc", "output_type": "text"}, workspace_id="W1-1")
    assert len(text.results) == 5
    assert video_policy_for(SearchPolicy(), frozenset({"guardlen"})) is None
    assert video_policy_for(SearchPolicy(), frozenset({"fanout5", "results10"})).maximum_candidates == 15


def _frame_store():
    from foren_searcher.store import GlobalStore

    sys.path.insert(0, str(ROOT / "tests"))
    from search_helpers import png_bytes

    store = GlobalStore()
    frame = store.record_evidence(content="frame", modalities=("image",), source_ref="V0", locator="complete",
                                  acquired_by="INSPECT/get_frames", discovered_by="W1-1",
                                  artifact_ref=store.save_artifact(png_bytes(), media_type="image/png", suffix=".png",
                                                                   role="source_material"))
    return store, frame.evidence_id


@pytest.mark.asyncio
async def test_thumbnail_clip_reranks_text_image_and_records_provider():
    from foren_searcher.search.providers.thumbnail_clip import ImageQueryRouter, ThumbnailClipProvider
    from foren_searcher.search.service import SearchService

    store, frame = _frame_store()
    text_provider = _Videos(n=30)

    async def downloader(ids):
        return {video_id: [b"thumb"] for video_id in ids}

    def scorer(query, thumbs):                     # 第 30 名最像
        return {video_id: (1.0 if video_id.endswith("x29") else 0.1) for video_id in thumbs}

    clip = ThumbnailClipProvider(text_provider, store, scorer=scorer, downloader=downloader)
    service = SearchService(store=store, provider=ImageQueryRouter(_Videos(), clip), opener=_Opener(),
                            allow_image_uploads=False)
    blocked = await service.execute({"input_type": "text_image", "query": {"evidence_id": frame, "text": "cats"},
                                     "output_type": "video"}, workspace_id="W1-1")
    assert blocked.error is not None and blocked.error.code == "ACCESS_RESTRICTED"       # 没开本机以图搜：照旧拒
    service.local_image_search = True
    response = await service.execute({"input_type": "text_image", "query": {"evidence_id": frame, "text": "cats"},
                                      "output_type": "video"}, workspace_id="W1-1")
    assert response.error is None and response.results
    assert text_provider.calls[0][2] == 30
    with store._connect() as connection:
        first = connection.execute("SELECT platform_video_id, provider, provider_rank, extra_json FROM candidates "
                                   "ORDER BY id LIMIT 1").fetchone()
    assert first[0].endswith("x29") and first[1] == "thumbnail_clip" and first[2] == 1
    assert json.loads(first[3])["text_rank"] == 30


@pytest.mark.asyncio
async def test_thumbnail_clip_image_only_uses_pool_or_explains():
    from foren_searcher.search.providers.thumbnail_clip import EMPTY_POOL, NOT_VIDEO, ThumbnailClipProvider

    store, _ = _frame_store()
    clip = ThumbnailClipProvider(_Videos(), store, scorer=lambda q, t: {k: 0.5 for k in t},
                                 downloader=lambda ids: asyncio.sleep(0, {i: [b"x"] for i in ids}))
    image = Path(__file__)
    with pytest.raises(SearchFailure) as caught:
        await clip.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=image)
    assert caught.value.error.message == EMPTY_POOL
    with pytest.raises(SearchFailure) as caught:
        await clip.search(SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=image)
    assert caught.value.error.message == NOT_VIDEO
    for i in range(3):
        store.get_or_create_candidate(identity_key=f"youtube:p{i}", source_url=f"https://www.youtube.com/watch?v=p{i}",
                                      search_title="t", search_snippet=None, provider="yt-dlp", engine="youtube_search",
                                      provider_rank=i + 1, discovered_execution_id=None, raw_response_ref=None,
                                      candidate_type="video", image_url=None, thumbnail_url=None, platform="youtube",
                                      platform_video_id=f"p{i}", extra={})
    batch = await clip.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=image)
    assert [c.platform_video_id for c in batch.candidates] == ["p0", "p1", "p2"]
    assert {c.provider for c in batch.candidates} == {"thumbnail_clip"}


def _harness(search):
    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.store import GlobalStore
    from foren_searcher.workspace import WorkspaceHarness

    return WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=GlobalStore(), search_service=search)  # type: ignore[arg-type]


def test_schema_offers_local_image_search_for_video_only():
    action = next(a for a in _harness(SimpleNamespace(allows_image_uploads=False, local_image_search=True))
                  .eligible_actions() if a["name"] == "SEARCH")
    branches = {b["properties"]["input_type"]["const"]: b for b in action["schema"]["oneOf"] if "input_type" in b["properties"]}
    assert set(branches) == {"text", "image", "text_image"}
    assert branches["image"]["properties"]["output_type"] == {"const": "video"}
    assert branches["text"]["properties"]["output_type"] == {"enum": ["text", "image", "video"]}
    assert "runs locally" in action["description"] and "not available" not in action["description"]
    a0 = next(a for a in _harness(SimpleNamespace(allows_image_uploads=False)).eligible_actions() if a["name"] == "SEARCH")
    assert "not available" in a0["description"]                       # A0 不变


def _case(tmp_path: Path):
    from foren_searcher.datagen.cases import Case

    path = tmp_path / "case.json"
    path.write_text(json.dumps({"case_id": "evid:video_001", "dataset": "evid", "source_id": "video_001", "split": "test",
                                "task_kind": "video_diff", "task_text": "Explain", "family_key": "evid:video_001",
                                "media": [], "source_policy": {"mode": "unrestricted"}}), encoding="utf-8")
    return Case.load(path)


def test_search_service_for_builds_arm_chain_and_a0_is_unchanged(tmp_path: Path):
    from foren_searcher.datagen.run_cases import _find_provider, _find_video_router, search_service_for
    from foren_searcher.store import GlobalStore

    case = _case(tmp_path)
    # 10-12 起以图搜图默认开（多引擎，ImageQueryRouter）；10-06 的 A0 = --image-search off
    a0, _, _ = search_service_for(case, GlobalStore(), tmp_path, image_search="off")
    assert a0.video_policy is None and a0.local_image_search is False
    assert _find_provider(a0, "ImageQueryRouter") is None and not _find_video_router(a0).explain_empty
    arm = parse_search_arm("results10,guardlen,imageclip,fanout5")
    search, _, _ = search_service_for(case, GlobalStore(), tmp_path, search_arm=arm)
    assert search.video_policy.target_sources == 15 and search.local_image_search is True
    assert _find_provider(search, "ImageQueryRouter") is not None
    router = _find_video_router(search)
    assert router.explain_empty and type(router.video_provider).__name__ == "FanoutVideoProvider"


def test_worker_command_passes_search_arm():
    from foren_searcher.datagen.run_cases import Planned, _worker_command

    args = SimpleNamespace(round="r", runs_dir="runs", teacher="endpoint", specs="specs", hi_spec="h", gia_spec="g",
                           no_think=False, blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s",
                           tool_cache="off", tool_cache_root="t", harness_params="apply", search_arm="results10,guardlen")
    item = Planned("c.json", "evid:video_001", "test", "run", 1, "todo")
    command = _worker_command(args, item, "http://x/v1", "m")
    assert command[command.index("--search-arm") + 1] == "results10,guardlen"
    args.search_arm = None
    assert "--search-arm" not in _worker_command(args, item, "http://x/v1", "m")


def _funnel_store(path: Path) -> None:
    from foren_searcher.store import GlobalStore

    store = GlobalStore(path / "store.sqlite3", path / "artifacts")
    del store
    connection = sqlite3.connect(path / "store.sqlite3")
    connection.execute("INSERT INTO search_executions(id, request_json, workspace_id, candidate_count) VALUES (1, ?, 'W1-1', 2)",
                       (json.dumps({"input_type": "text", "query": "kurashimono journal", "output_type": "video"}),))
    connection.execute("INSERT INTO search_executions(id, request_json, workspace_id, candidate_count) VALUES (2, ?, 'W1-1', 0)",
                       (json.dumps({"input_type": "text", "query": "a b c 2026/05/20", "output_type": "video"}),))
    for cid, vid in ((1, "OTHER"), (2, "G2e47y_flZE")):
        connection.execute("INSERT INTO candidates(id, identity_key, source_url, provider, engine, provider_rank, platform, "
                           "platform_video_id) VALUES (?, ?, ?, 'yt-dlp', 'youtube_search', ?, 'youtube', ?)",
                           (cid, f"k{cid}", f"https://www.youtube.com/watch?v={vid}", cid + 6, vid))
        connection.execute("INSERT INTO candidate_discoveries(execution_id, candidate_id, provider, engine, provider_rank) "
                           "VALUES (1, ?, 'yt-dlp', 'youtube_search', ?)", (cid, cid + 6))
    connection.execute("INSERT INTO evidence(id, modalities_json, label, content, derived_from_json, locator, acquired_by, "
                       "discovered_by, production_method) VALUES (7, '[]', 'v', '', '[]', '', 'INSPECT/acquire', 'W', 'x')")
    connection.execute("INSERT INTO candidate_evidence VALUES (2, 7, 'video', 'ready')")
    connection.execute("INSERT INTO inspect_executions(request_json, workspace_id) VALUES (?, 'W1-1')",
                       (json.dumps({"targets": ["E1", "E7"], "request": {"operation": "align_media", "params": {}}}),))
    connection.commit()
    connection.close()


def test_source_funnel_columns(tmp_path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _evid_behavior import funnel_markdown, source_funnel, summarize_funnel

    run = tmp_path / "evid_video_090_x_1"
    run.mkdir()
    _funnel_store(run)
    row = source_funnel(run, ["_C5bhsSvOLU", "G2e47y_flZE"])
    assert row["source_in_pool"] and row["source_downloaded"] and row["source_compared"]
    assert row["source_rank"] == 8 and row["source_found_by"] == "yt-dlp/youtube_search"
    assert row["source_query"] == "kurashimono journal"
    assert row["video_searches"] == 2 and row["video_searches_empty"] == 1
    miss = source_funnel(run, ["nothing"])
    assert not miss["source_in_pool"] and not miss["source_downloaded"]
    summary = summarize_funnel([row, miss, source_funnel(tmp_path / "missing", ["x"])])
    assert summary["source_in_pool"] == 1 and summary["runs_with_store"] == 2 and summary["source_rank_over_5"] == 1
    assert "1/2" in "\n".join(funnel_markdown(summary, [row, miss], ["a", "b"]))


# ---------------------------------------------------------------- 一夜编排（不跑模型、不联网：全部打桩）

def _night(monkeypatch, tmp_path: Path, *, selftest_ok: bool = True, a6_hits: int = 8, served=("qwen3vl-8b-bf16-32k:latest",),
           until: str | None = None):
    sys.path.insert(0, str(ROOT / "scripts"))
    import _evid_night as night

    monkeypatch.setattr(night, "ROOT", tmp_path)
    monkeypatch.setattr(night, "env_setup", lambda: None)
    monkeypatch.setattr(night.L, "ollama_models", lambda: list(served))
    monkeypatch.setattr(night.L, "bf16_preflight", lambda log: True)
    monkeypatch.setattr(night.L, "_ollama_ps", lambda log, label: None)
    monkeypatch.setattr(night.L, "evid27_probe", lambda log: True)
    if until:
        monkeypatch.setenv("FOREN_NIGHT_UNTIL", until)
    else:
        monkeypatch.delenv("FOREN_NIGHT_UNTIL", raising=False)
    for arm in night.ARMS:
        arm.status, arm.notes = "pending", []
    scores = tmp_path / night.SCORES
    scores.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(self, label, argv, *, show=False):
        calls.append(list(argv))
        joined = " ".join(map(str, argv))
        if "_evid_arm_selftest.py" in joined:
            (scores / "arm_selftest.json").write_text(json.dumps({"checks": {"1 results10": {"ok": selftest_ok, "detail": "x"}},
                                                                  "critical_ok": selftest_ok, "fanout_ok": True}))
            return 0 if selftest_ok else 1
        if "_evid_oracle_search.py" in joined:
            (scores / "A6_oracle.json").write_text(json.dumps({"summary": {"hit_at_10": a6_hits, "hit_at_5": 3, "samples": 20,
                                                                           "samples_with_available_source": 18}}))
        if "_evid_replay_old_queries.py" in joined:
            (scores / "A7_replay.json").write_text(json.dumps({"summary": {
                "new_hit": 8, "old_text_hit": 7, "old_any_hit": 8, "agree_with_old_text": 19, "with_old_log": 20,
                "new_empty": 30, "old_failures": 40}}))
        if "foren_searcher.datagen.run_cases" in joined:
            round_tag = argv[argv.index("--round") + 1]
            ids = argv[argv.index("--ids") + 1:]
            ids = ids[:next((i for i, x in enumerate(ids) if x.startswith("--")), len(ids))]
            for case in ids:
                run = tmp_path / night.run_dir(round_tag, case.split(":", 1)[1]).relative_to(night.ROOT)
                run.mkdir(parents=True, exist_ok=True)
                (run / "result.json").write_text("{}")
                (run / "execution_links.json").write_text("{}")
        if "_evid_score.py" in joined and "--judges" in argv:
            round_tag = argv[argv.index("--round") + 1]
            pool = {"evidsA9": 6, "evidsA4": 9, "evidsA5": 1}.get(round_tag, 0)
            (scores / f"{round_tag}.sonnet.subset.summary.json").write_text(json.dumps({
                "attempted": {"point_acc": 0.30 + pool / 100, "video_acc": 0.1},
                "funnel": {"source_in_pool": pool, "source_downloaded": pool // 2, "source_compared": 1,
                           "runs_with_store": 20, "source_found_by": {"thumbnail_clip/youtube_thumbnail_clip": pool} if pool else {},
                           "video_searches": 40, "video_searches_empty": 5}}))
        return 0

    monkeypatch.setattr(night.NightLog, "run", fake_run)
    return night, calls, scores


def test_night_runs_arms_in_order_with_gates(monkeypatch, tmp_path: Path):
    night, calls, scores = _night(monkeypatch, tmp_path)
    assert night.main() == 0
    status = {arm.name: arm.status for arm in night.ARMS}
    assert status == {"A9": "done", "A5": "done", "A4": "done", "A8": "skipped_gate",      # 本机没有 27B
                      "A8G": "skipped_gate", "A8T": "skipped_gate"}
    runs = [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]
    rounds = [c[c.index("--round") + 1] for c in runs]
    assert rounds[:2] == ["evidsA9", "evidsA9"] and len(runs[0][runs[0].index("--ids") + 1:]) >= 2   # 金丝雀 + 其余
    a9 = runs[0]
    assert a9[a9.index("--search-arm") + 1] == "results10,guardlen,imageclip"
    assert a9[a9.index("--snapshot-root") + 1].endswith("evidsA9") and "--instruction-mode" in a9
    a5 = next(c for c in runs if c[c.index("--round") + 1] == "evidsA5")
    assert "--instruction-mode" not in a5 and "--search-arm" not in a5                        # 教师指令、A0 工具
    report = next(scores.glob("night_source_*.md")).read_text(encoding="utf-8")
    assert "| A9 | evidsA9 | 20/20 | 6 |" in report and "达到计划目标" in report and "≥ 4/20" in report
    assert "A6 前 10 命中 8/20 ≥ 30%" in report
    calls.clear()                                       # 再双击：评过分的臂不再跑
    for arm in night.ARMS:
        arm.status, arm.notes = "pending", []
    assert night.main() == 0
    assert not [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]


def test_night_skips_tool_arms_when_selftest_fails_and_a4_when_a6_low(monkeypatch, tmp_path: Path):
    night, calls, _ = _night(monkeypatch, tmp_path, selftest_ok=False, a6_hits=2)
    night.main()
    status = {arm.name: arm.status for arm in night.ARMS}
    assert status["A9"] == "skipped_gate" and status["A4"] == "skipped_gate" and status["A5"] == "done"
    assert "A6 < 30%" in next(a for a in night.ARMS if a.name == "A4").notes[0]


def test_night_has_no_deadline_by_default(monkeypatch, tmp_path: Path):
    night, calls, _ = _night(monkeypatch, tmp_path)
    assert night.deadline() is None                        # 10-07：白天也能跑，不设截止就不按时间跳过

def test_night_does_not_start_arm_that_cannot_finish_before_deadline(monkeypatch, tmp_path: Path):
    import datetime as dt

    soon = (dt.datetime.now() + dt.timedelta(minutes=60)).strftime("%H:%M")
    night, calls, _ = _night(monkeypatch, tmp_path, until=soon)
    night.main()
    status = {arm.name: arm.status for arm in night.ARMS}
    assert status["A9"] == status["A5"] == "skipped_time" and status["A4"] == "skipped_gate"      # A4 依赖 A9
    assert not [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]


def test_night_arm_selection_and_dependency_from_an_earlier_night(monkeypatch, tmp_path: Path):
    night, calls, _ = _night(monkeypatch, tmp_path)
    monkeypatch.setenv("FOREN_NIGHT_ARMS", "A9")
    night.main()
    assert {a.name: a.status for a in night.ARMS} == {"A9": "done", "A5": "not_selected", "A4": "not_selected",
                                                      "A8": "not_selected", "A8G": "not_selected", "A8T": "not_selected"}
    for arm in night.ARMS:
        arm.status, arm.notes = "pending", []
    monkeypatch.setenv("FOREN_NIGHT_ARMS", "A4")           # 第二晚只跑 A4：A9 前一晚已完成并评分，依赖成立
    night.main()
    assert next(a for a in night.ARMS if a.name == "A4").status == "done"


def test_second_night_runs_only_new_27b_arms(monkeypatch, tmp_path: Path):
    """10-07：第一晚的 A9 / A5 / A4 / A8 已评过分；再双击只跑新加的 A8G、A8T（27B + 工具层），判读里与 A8 对照。"""

    night, calls, scores = _night(monkeypatch, tmp_path, served=("qwen3vl-8b-bf16-32k:latest", "qwen38-27b-32k:latest"))
    assert night.main() == 0
    assert {a.name: a.status for a in night.ARMS} == {n: "done" for n in ("A9", "A5", "A4", "A8", "A8G", "A8T")}
    for arm in night.ARMS:
        arm.status, arm.notes = "pending", []
    calls.clear()
    assert night.main() == 0
    assert not [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]       # 都评过分：不再跑
    for arm in night.ARMS:
        arm.status, arm.notes = "pending", []
    for tag in ("evidsA8G", "evidsA8T"):                   # 模拟"第一晚只有前四个臂"：删掉新臂的评分与 run
        (scores / f"{tag}.sonnet.subset.summary.json").unlink()
        for sample in night.IDS:
            run = tmp_path / night.run_dir(tag, sample).relative_to(night.ROOT)
            (run / "result.json").unlink()
    calls.clear()
    assert night.main() == 0
    runs = [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]
    rounds = {c[c.index("--round") + 1] for c in runs}
    assert rounds == {"evidsA8G", "evidsA8T"}
    a8g = next(c for c in runs if c[c.index("--round") + 1] == "evidsA8G")
    assert a8g[a8g.index("--model") + 1] == night.L.EVID27_MODEL and "--no-think" in a8g
    assert a8g[a8g.index("--search-arm") + 1] == "guardlen"
    a8t = next(c for c in runs if c[c.index("--round") + 1] == "evidsA8T")
    assert a8t[a8t.index("--search-arm") + 1] == "results10,guardlen,imageclip"
    report = next(scores.glob("night_source_*.md")).read_text(encoding="utf-8")
    assert "| A8G | evidsA8G | 20/20 |" in report and "A8G（27B + 查询限长）对 A8" in report


def test_english_spec_accepts_cases_whose_task_text_is_already_english():
    """10-07：造数链路改英文后，英文规格要求 task_text_en；EVID 的 task_text 本来就是纯英文，直接用，中文仍报错。"""

    from foren_searcher.datagen.cases import Case, CaseError

    evid = Case(case_id="evid:video_002", dataset="evid", source_id="video_002", split="test", task_kind="provenance",
                task_text="Material A is a video that has been manipulated.", family_key="evid_src:x")
    assert evid.task_for("en") == evid.task_text
    zh = Case(case_id="x:1", dataset="x", source_id="1", split="dev_train", task_kind="claim_check", task_text="核查 Material A",
              family_key="x:1")
    with pytest.raises(CaseError):
        zh.task_for("en")
