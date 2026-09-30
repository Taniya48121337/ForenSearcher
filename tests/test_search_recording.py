"""检索录制回放：录后回放逐字段相同、回放未命中报错、replay_or_record 只补录缺的、护栏在回放上照常拦截、
请求规范化稳定、run_cases 参数接线。全部用假 provider / 假 opener / 假 acquirer，不联网。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from foren_searcher.search.contracts import (
    CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource, OpenSourceRequest, SearchError, SearchFailure,
    SearchQueryRequest,
)
from foren_searcher.search.recording import (
    RecordingAcquirer, RecordingOpener, RecordingProvider, ReplayMiss, SearchRecording, apply_search_recording,
    case_key, open_request_view, request_key, search_request_view, snapshot_url,
)
from foren_searcher.search.retrieval_guard import GuardedOpener, GuardedProvider, RetrievalGuard, apply_retrieval_guard

REQ = SearchQueryRequest("text", "eiffel tower  lightning", "text")


def _cand(url: str, rank: int = 1, **extra) -> CandidateSource:
    return CandidateSource(url, f"title {rank}", f"snippet {rank}", "ddgs", "bing", rank, "page", extra=extra)


class FakeProvider:
    """按查询返回固定候选；calls 记录真正"联网"的次数。"""

    def __init__(self, urls: dict[str, list[str]] | None = None, failure: SearchFailure | None = None) -> None:
        self.urls = urls or {}
        self.failure = failure
        self.calls: list[tuple[str, int]] = []

    async def search(self, request, *, offset, limit, image_path=None):
        self.calls.append((str(request.query), offset))
        if self.failure is not None:
            raise self.failure
        urls = self.urls.get(" ".join(str(request.query).split()), ["https://example.org/a"])
        candidates = tuple(_cand(u, i + 1, publishedDate="2016-07-01", fetched_at=f"t{len(self.calls)}")
                           for i, u in enumerate(urls[offset:offset + limit]))
        return CandidateBatch("ddgs", "bing", candidates, {"results": [c.source_url for c in candidates],
                                                            "search_metadata": {"id": len(self.calls)}},
                              {"region": "wt-wt"})


class FakeOpener:
    def __init__(self, html: bytes = b"<html><meta property='article:published_time' content='2016-07-02'></html>",
                 failures: dict[str, SearchFailure] | None = None) -> None:
        self.html = html
        self.failures = failures or {}
        self.opened: list[str] = []

    async def open(self, candidate, request):
        self.opened.append(candidate.source_url)
        if candidate.source_url in self.failures:
            raise self.failures[candidate.source_url]
        page = OpenedMaterial("page", "ready", "正文 " + "x" * 3000, self.html, "text/html", ".html", ("text",),
                              "页面", "complete page", "SEARCH/open/http", "source_excerpt")
        failed = OpenedMaterial("document", "failed", error=SearchError("extract", "EXTRACT_FAILED", "no pdf"))
        return OpenedSource(candidate.source_url + "#final", (page, failed), None, None, ("https://youtu.be/abc",))


def _recording(tmp_path: Path, mode: str, run: str = "run1") -> SearchRecording:
    return SearchRecording(mode, tmp_path / "snap", "true:4124903", run_name=run, log_path=tmp_path / run / "search_snapshot.jsonl")  # type: ignore[arg-type]


def test_record_then_replay_is_field_identical_and_offline(tmp_path: Path) -> None:
    provider, opener = FakeProvider({"eiffel tower lightning": ["https://example.org/a", "https://example.org/b"]}), FakeOpener()
    rec = _recording(tmp_path, "record")
    live_batch = asyncio.run(RecordingProvider(provider, rec).search(REQ, offset=0, limit=10))
    live_open = asyncio.run(RecordingOpener(opener, rec).open(live_batch.candidates[0], REQ))
    assert rec.counts["recorded"] == 2 and rec.counts["live_calls"] == 2
    assert (tmp_path / "snap" / "true_4124903" / "index.jsonl").is_file()

    replay = _recording(tmp_path, "replay", "run2")
    dead_provider, dead_opener = FakeProvider(), FakeOpener()
    batch = asyncio.run(RecordingProvider(dead_provider, replay).search(REQ, offset=0, limit=10))
    opened = asyncio.run(RecordingOpener(dead_opener, replay).open(live_batch.candidates[0], REQ))
    assert dead_provider.calls == [] and dead_opener.opened == []          # 回放不联网
    assert batch == live_batch                                              # CandidateBatch 逐字段相同（含 raw_response）
    assert opened == live_open                                              # OpenedSource 逐字段相同（含字节与失败材料）
    assert opened.materials[0].artifact_bytes == opener.html and opened.materials[1].error.code == "EXTRACT_FAILED"
    assert replay.summary()["counts"]["hit"] == 2 and replay.summary()["hit_rate"] == 1.0
    usage = replay.write_usage()
    assert json.loads(usage.read_text(encoding="utf-8"))["counts"]["hit"] == 2


def test_replay_miss_raises_structured_failure_and_logs(tmp_path: Path) -> None:
    rec = _recording(tmp_path, "replay")
    provider, opener = FakeProvider(), FakeOpener()
    with pytest.raises(ReplayMiss) as caught:
        asyncio.run(RecordingProvider(provider, rec).search(REQ, offset=0, limit=10))
    assert isinstance(caught.value, SearchFailure) and caught.value.reason == "replay_miss"
    assert caught.value.error.code == "NOT_FOUND" and "replay_miss" in caught.value.error.message
    with pytest.raises(ReplayMiss):
        asyncio.run(RecordingOpener(opener, rec).open(_cand("https://example.org/zz"), REQ))
    assert provider.calls == [] and opener.opened == []
    log = [json.loads(line) for line in (tmp_path / "run1" / "search_snapshot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["kind"], row["event"]) for row in log] == [("search", "replay_miss"), ("open", "replay_miss")]
    assert rec.counts["miss"] == 2 and rec.counts["live_calls"] == 0


def test_replay_or_record_only_records_what_is_missing(tmp_path: Path) -> None:
    first = _recording(tmp_path, "record")
    asyncio.run(RecordingProvider(FakeProvider(), first).search(REQ, offset=0, limit=10))
    provider = FakeProvider()
    mixed = _recording(tmp_path, "replay_or_record", "run2")
    wrapped = RecordingProvider(provider, mixed)
    asyncio.run(wrapped.search(REQ, offset=0, limit=10))                                   # 命中
    other = SearchQueryRequest("text", "another query", "text")
    asyncio.run(wrapped.search(other, offset=0, limit=10))                                 # 未命中 → 联网补录
    assert provider.calls == [("another query", 0)]
    assert (mixed.counts["hit"], mixed.counts["miss"], mixed.counts["recorded"]) == (1, 1, 1)
    # 补录后的条目在纯回放里可用
    replay = _recording(tmp_path, "replay", "run3")
    asyncio.run(RecordingProvider(FakeProvider(), replay).search(other, offset=0, limit=10))
    assert replay.counts["hit"] == 1


def test_failures_are_replayed_and_transient_ones_retried_in_mixed_mode(tmp_path: Path) -> None:
    gone = SearchFailure(SearchError("open", "NOT_FOUND", "page deleted"))
    slow = SearchFailure(SearchError("open", "TIMEOUT", "timed out"))
    opener = FakeOpener(failures={"https://example.org/gone": gone, "https://example.org/slow": slow})
    rec = _recording(tmp_path, "record")
    for url in ("https://example.org/gone", "https://example.org/slow"):
        with pytest.raises(SearchFailure):
            asyncio.run(RecordingOpener(opener, rec).open(_cand(url), REQ))
    replay = _recording(tmp_path, "replay", "run2")
    for url, code in (("https://example.org/gone", "NOT_FOUND"), ("https://example.org/slow", "TIMEOUT")):
        with pytest.raises(SearchFailure) as caught:
            asyncio.run(RecordingOpener(FakeOpener(), replay).open(_cand(url), REQ))
        assert caught.value.error.code == code and not isinstance(caught.value, ReplayMiss)
    healthy = FakeOpener()
    mixed = _recording(tmp_path, "replay_or_record", "run3")
    with pytest.raises(SearchFailure):
        asyncio.run(RecordingOpener(healthy, mixed).open(_cand("https://example.org/gone"), REQ))
    asyncio.run(RecordingOpener(healthy, mixed).open(_cand("https://example.org/slow"), REQ))
    assert healthy.opened == ["https://example.org/slow"]            # 只有瞬时失败被重新联网


def test_request_keys_are_stable_across_runs() -> None:
    a = search_request_view(SearchQueryRequest("text", "  Eiffel　tower\nlightning ", "text"), offset=0, limit=10, image_path=None)
    b = search_request_view(SearchQueryRequest("text", "Eiffel tower lightning", "text"), offset=0, limit=10, image_path=None)
    c = search_request_view(SearchQueryRequest("text", "Eiffel tower lightning", "text"), offset=10, limit=10, image_path=None)
    assert request_key(a) == request_key(b) != request_key(c)
    assert request_key({"b": 1, "a": [1, 2]}) == request_key({"a": [1, 2], "b": 1})
    # 候选里的标题、摘要、排名、时间戳这类随响应变化的字段不影响打开的 key；追踪参数也不影响
    one = CandidateSource("https://Example.org:443/p?b=2&a=1&utm_source=x#frag", "t1", "s1", "ddgs", "bing", 1, "page",
                          extra={"fetched_at": "2026-09-29T10:00:00"})
    two = CandidateSource("https://example.org/p?a=1&b=2&fbclid=zz", "t2", "s2", "searx", "google", 7, "page",
                          extra={"fetched_at": "2026-09-30T11:00:00"})
    assert request_key(open_request_view(one, REQ)) == request_key(open_request_view(two, REQ))
    assert snapshot_url("https://www.youtube.com/watch?v=abc&si=RANDOM") == "https://www.youtube.com/watch?v=abc"
    subtitles = OpenSourceRequest("https://example.org/p?a=1&b=2", "video", ("subtitles",), "en")
    assert request_key(open_request_view(two, subtitles)) != request_key(open_request_view(two, REQ))
    assert case_key("true:4124903") == "true_4124903"


def test_image_query_key_uses_image_bytes_not_evidence_id(tmp_path: Path) -> None:
    image = tmp_path / "q.jpg"
    image.write_bytes(b"\xff\xd8\xff fake jpeg")
    e1 = search_request_view(SearchQueryRequest("image", "E1", "image"), offset=0, limit=5, image_path=image)
    e7 = search_request_view(SearchQueryRequest("image", "E7", "image"), offset=0, limit=5, image_path=image)
    assert request_key(e1) == request_key(e7) and e1["image_sha256"]


def test_guard_still_applies_on_replayed_results(tmp_path: Path) -> None:
    urls = {"eiffel tower lightning": ["https://www.snopes.com/fact-check/eiffel", "https://news.example.org/2016/story"]}
    rec = _recording(tmp_path, "record")
    raw = asyncio.run(RecordingProvider(FakeProvider(urls), rec).search(REQ, offset=0, limit=10))
    assert len(raw.candidates) == 2                                   # 快照里是原始世界，含核查站点
    asyncio.run(RecordingOpener(FakeOpener(), rec).open(raw.candidates[1], REQ))

    class Service:
        provider, opener = FakeProvider(), FakeOpener()

    replay = _recording(tmp_path, "replay", "run2")
    service = apply_search_recording(Service(), replay)
    guard = RetrievalGuard(blocklist="exclude", date_cutoff="2016-07-01", log_path=tmp_path / "guard.jsonl")  # type: ignore[arg-type]
    apply_retrieval_guard(service, guard)                             # guard(replay(...))
    assert isinstance(service.provider, GuardedProvider) and isinstance(service.provider.provider, RecordingProvider)
    batch = asyncio.run(service.provider.search(REQ, offset=0, limit=10))
    assert [c.source_url for c in batch.candidates] == ["https://news.example.org/2016/story"]
    with pytest.raises(SearchFailure) as caught:                     # 回放出来的页面发布于截止日之后 → 照样拒绝
        asyncio.run(service.opener.open(raw.candidates[1], REQ))
    assert caught.value.error.code == "ACCESS_RESTRICTED" and guard.counts["after_cutoff"] == 1
    assert replay.counts["hit"] == 2


def test_search_service_end_to_end_replay_matches_recording(tmp_path: Path) -> None:
    from foren_searcher.search.service import SearchService
    from foren_searcher.store import GlobalStore

    urls = {"eiffel tower lightning": ["https://example.org/a", "https://example.org/b"]}

    def run(mode: str, name: str, provider: FakeProvider, opener: FakeOpener) -> dict:
        store = GlobalStore(tmp_path / name / "s.sqlite3", tmp_path / name / "artifacts")
        service = SearchService(store=store, provider=provider, opener=opener)  # type: ignore[arg-type]
        apply_search_recording(service, _recording(tmp_path, mode, name))
        response = asyncio.run(service.execute(REQ.to_dict(), workspace_id="W1-1")).to_dict()
        contents = {card["evidence_id"]: store.evidence(card["evidence_id"]).content
                    for result in response["results"] for material in result["materials"] for card in material["evidence"]}
        return {"response": response, "contents": contents}

    recorded = run("record", "a", FakeProvider(urls), FakeOpener())
    replayed = run("replay", "b", FakeProvider(), FakeOpener())
    assert replayed == recorded and recorded["contents"]


def test_acquirer_records_and_replays_media_file(tmp_path: Path) -> None:
    from foren_searcher.inspect.acquisition import AcquiredMedia
    from foren_searcher.inspect.backends.base import BackendFailure

    class FakeAcquirer:
        def __init__(self) -> None:
            self.calls = 0

        async def acquire(self, source_url, *, destination, policy):
            self.calls += 1
            destination.write_bytes(b"\x00\x00\x00\x18ftypmp42 video bytes")
            return AcquiredMedia(destination, "video/mp4", ".mp4", "v.mp4", {"requested_url": source_url, "byte_size": 27})

    url = "https://cdn.example.org/v.mp4?sig=abc"
    live = FakeAcquirer()
    rec = _recording(tmp_path, "record")
    first = asyncio.run(RecordingAcquirer(live, rec).acquire(url, destination=tmp_path / "one.media", policy=None))
    replay = _recording(tmp_path, "replay", "run2")
    dead = FakeAcquirer()
    second = asyncio.run(RecordingAcquirer(dead, replay).acquire(url, destination=tmp_path / "two.media", policy=None))
    assert dead.calls == 0 and second.path.read_bytes() == first.path.read_bytes()
    assert (second.media_type, second.suffix, second.label, second.details) == (first.media_type, first.suffix, first.label, first.details)
    with pytest.raises(BackendFailure) as caught:
        asyncio.run(RecordingAcquirer(dead, replay).acquire("https://cdn.example.org/other.mp4", destination=tmp_path / "x", policy=None))
    assert "replay_miss" in str(caught.value) and dead.calls == 0


def test_snapshot_stats_reports_entries_and_run_hit_rates(tmp_path: Path) -> None:
    import importlib.util

    rec = _recording(tmp_path, "record")
    asyncio.run(RecordingProvider(FakeProvider(), rec).search(REQ, offset=0, limit=10))
    asyncio.run(RecordingProvider(FakeProvider(), rec).search(REQ, offset=10, limit=10))
    rec.write_usage()
    replay = _recording(tmp_path, "replay", "run2")
    asyncio.run(RecordingProvider(FakeProvider(), replay).search(REQ, offset=0, limit=10))
    with pytest.raises(ReplayMiss):
        asyncio.run(RecordingProvider(FakeProvider(), replay).search(REQ, offset=20, limit=10))
    replay.write_usage()
    spec = importlib.util.spec_from_file_location("snapshot_stats", Path(__file__).resolve().parents[1] / "scripts" / "snapshot_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    rows = module.collect(tmp_path / "snap")
    assert rows[0]["entries"] == 2 and rows[0]["search"] == 2 and rows[0]["bytes"] > 0
    rates = {run["run"]: run["hit_rate"] for run in rows[0]["runs"]}
    assert rates == {"run1": None, "run2": 0.5}
    text = module.render_markdown(rows)
    assert "| true_4124903 | 2 |" in text and "50.0%" in text


def test_rerecording_identical_content_does_not_grow_index(tmp_path: Path) -> None:
    class Drifting(FakeProvider):
        """每次响应的时间戳 / 请求 ID 都不同，但结果本身没变。"""

        def __init__(self, stamp: str) -> None:
            super().__init__()
            self.stamp = stamp

        async def search(self, request, *, offset, limit, image_path=None):
            batch = await super().search(request, offset=offset, limit=limit, image_path=image_path)
            candidates = tuple(CandidateSource(**{**asdict(c), "extra": {**c.extra, "fetched_at": self.stamp}}) for c in batch.candidates)
            return CandidateBatch(batch.provider, batch.engine, candidates,
                                  {**batch.raw_response, "search_metadata": {"id": self.stamp, "created_at": self.stamp}})

    rec = _recording(tmp_path, "record")
    asyncio.run(RecordingProvider(Drifting("2026-09-29T10:00"), rec).search(REQ, offset=0, limit=10))
    again = _recording(tmp_path, "record", "run2")
    asyncio.run(RecordingProvider(Drifting("2026-09-30T11:00"), again).search(REQ, offset=0, limit=10))
    index = tmp_path / "snap" / "true_4124903" / "index.jsonl"
    assert len(index.read_text(encoding="utf-8").splitlines()) == 1 and again.counts["recorded_unchanged"] == 1
    changed = _recording(tmp_path, "record", "run3")
    asyncio.run(RecordingProvider(FakeProvider({"eiffel tower lightning": ["https://example.org/new"]}), changed).search(REQ, offset=0, limit=10))
    assert len(index.read_text(encoding="utf-8").splitlines()) == 2 and changed.counts["recorded"] == 1
    drift = [json.loads(line) for line in (tmp_path / "run3" / "search_snapshot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert drift[0]["event"] == "drift"


# ---------------------------------------------------------------- run_cases 接线

def _case(mode: str = "unrestricted"):
    from foren_searcher.datagen.cases import Case, SourcePolicy

    return Case(case_id="true:1", dataset="true", source_id="1", split="dev_train", task_kind="claim_check", task_text="t",
                family_key="true:1", media=(), source_policy=SourcePolicy(mode=mode, date_cutoff="2016-07-20"))  # type: ignore[arg-type]


def test_run_cases_wraps_replay_inside_guard(tmp_path: Path, monkeypatch) -> None:
    from foren_searcher.datagen import run_cases
    import foren_searcher.search.factory as factory

    class Service:
        def __init__(self):
            self.provider, self.opener = FakeProvider(), FakeOpener()

    monkeypatch.setattr(factory, "create_free_search_service", lambda **kw: Service())
    assert run_cases.search_recording_for(_case(), tmp_path, "r1") is None                    # 默认 live 不包
    recording = run_cases.search_recording_for(_case(), tmp_path, "r1", search_mode="replay", snapshot_root=tmp_path / "snap")
    assert recording.directory == tmp_path / "snap" / "true_1" and recording.log_path == tmp_path / "search_snapshot.jsonl"
    service, desc, guard = run_cases.search_service_for(_case(), store=None, output=tmp_path, recording=recording)
    assert isinstance(service.provider, GuardedProvider) and isinstance(service.provider.provider, RecordingProvider)
    assert isinstance(service.opener, GuardedOpener) and isinstance(service.opener.opener, RecordingOpener)
    live, _, _ = run_cases.search_service_for(_case(), store=None, output=tmp_path)
    assert not isinstance(live.provider.provider, RecordingProvider)


def test_run_cases_cli_passes_environment_flags_to_worker(tmp_path: Path) -> None:
    import argparse
    from foren_searcher.datagen import run_cases

    ns = argparse.Namespace(round="r0", runs_dir=tmp_path, teacher="endpoint", specs="specs", hi_spec="hi@0.3.0",
                            gia_spec="gia@0.4.0", no_think=False, blocklist="exclude", date_cutoff=None,
                            search_mode="replay", snapshot_root=tmp_path / "snap", tool_cache="on",
                            tool_cache_root=tmp_path / "cache", harness_params="apply")
    item = run_cases.Planned("c.json", "true:1", "dev_train", 1, "true_1_r0_1", "todo")
    command = run_cases._worker_command(ns, item, "http://t/v1", "m")
    joined = " ".join(map(str, command))
    assert f"--search-mode replay --snapshot-root {tmp_path / 'snap'}" in joined
    assert f"--tool-cache on --tool-cache-root {tmp_path / 'cache'}" in joined and "--harness-params apply" in joined


def test_run_cases_argparse_defaults_keep_live_behaviour(monkeypatch) -> None:
    from foren_searcher.datagen import run_cases

    seen: dict = {}

    async def fake_run_one(*args, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(run_cases, "run_one", fake_run_one)
    assert run_cases.main(["--one", "c.json", "--run-name", "x"]) == 0
    assert seen["search_mode"] == "live" and seen["tool_cache"] == "off" and seen["harness_params"] == "apply"
    assert Path(seen["snapshot_root"]) == Path("experiments") / "sft_data" / "search_snapshots"
    assert Path(seen["tool_cache_root"]) == Path("experiments") / "sft_data" / "tool_cache"
    seen.clear()
    run_cases.main(["--one", "c.json", "--run-name", "x", "--search-mode", "replay_or_record", "--snapshot-root", "S"])
    assert seen["search_mode"] == "replay_or_record" and seen["snapshot_root"] == "S"
    with pytest.raises(SystemExit):
        run_cases.main(["--one", "c.json", "--run-name", "x", "--search-mode", "bogus"])
