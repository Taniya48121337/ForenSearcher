"""evidb7 冒烟（hi@0.7.0 / gia@0.8.0，10-05 晚）之后的两处修正。

1. 候选视频下载：直链带上平台元数据给的请求头；被拒（403）或不是视频时，YouTube 页面改由 yt-dlp 下载。
   冒烟里对候选页面 E 的 get_frames 5/5 被 HTTP 403，下载次数 0。
2. HI 重复用完：约束可见时先给一个只可交报告的回合，交不出再自动 INCOMPLETE。
   冒烟里 27 个工作区 26 个因重复用完而自动 INCOMPLETE，HI 一份自己的报告都没写。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_helpers import make_video  # noqa: E402

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceRunner
from foren_searcher.inspect import acquisition
from foren_searcher.inspect.acquisition import AcquiredMedia, DirectMediaAcquirer, ytdlp_download
from foren_searcher.inspect.backends.base import BackendFailure
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.policies import EVAL_GUARDS, merge_input_features, runtime_guard_scope
from foren_searcher.search.openers.http import SafeHttpClient
from foren_searcher.search.recording import acquire_request_view
from foren_searcher.store import GlobalStore

PUBLIC = "https://93.184.216.34/"


# ---------------------------------------------------------------- 1. 下载

def test_download_forwards_only_allowlisted_platform_headers(tmp_path: Path):
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, headers={"Content-Type": "video/mp4"}, content=b"x" * 10)

    async def go():
        http = SafeHttpClient(transport=httpx.MockTransport(handler))
        try:
            await http.download_to_file(PUBLIC + "v.mp4", destination=tmp_path / "v.mp4", maximum_bytes=100,
                                        extra_headers={"User-Agent": "Mozilla/5.0 test", "Accept-Language": "en-us",
                                                       "Cookie": "secret=1", "Authorization": "Bearer x"})
        finally:
            await http.aclose()

    asyncio.run(go())
    assert seen["user-agent"] == "Mozilla/5.0 test" and seen["accept-language"] == "en-us"
    assert "cookie" not in seen and "authorization" not in seen


class _Denied:
    def __init__(self, *args, **kwargs):
        self.policy = None

    async def download_to_file(self, *args, **kwargs):
        from foren_searcher.search.contracts import SearchError, SearchFailure

        raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED", "source denied access (HTTP 403)"))

    async def aclose(self):
        return None


def test_direct_403_falls_back_to_ytdlp_only_with_a_page_url(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(acquisition, "SafeHttpClient", _Denied)
    calls = []

    def fake_ytdlp(page_url, destination, policy):
        calls.append(page_url)
        destination.write_bytes(b"video")
        return AcquiredMedia(destination, "video/mp4", ".mp4", "title", {"method": "yt-dlp", "requested_url": page_url})

    monkeypatch.setattr(acquisition, "ytdlp_download", fake_ytdlp)
    acquirer = DirectMediaAcquirer()
    acquired = asyncio.run(acquirer.acquire("https://rr1.googlevideo.com/videoplayback?x=1", destination=tmp_path / "a.media",
                                            policy=InspectPolicy(), headers={"User-Agent": "u"},
                                            page_url="https://www.youtube.com/watch?v=abc"))
    assert calls == ["https://www.youtube.com/watch?v=abc"]
    assert acquired.details["method"] == "yt-dlp" and acquired.details["direct_error"].startswith("ACCESS_RESTRICTED")
    with pytest.raises(BackendFailure) as error:                        # 没有视频页：照旧报 403，不走 yt-dlp
        asyncio.run(acquirer.acquire("https://rr1.googlevideo.com/videoplayback?x=1", destination=tmp_path / "b.media",
                                     policy=InspectPolicy()))
    assert error.value.code == "ACCESS_RESTRICTED" and len(calls) == 1


def test_ytdlp_download_moves_a_real_video_and_rejects_other_hosts(tmp_path: Path):
    source = tmp_path / "src.mp4"
    make_video(source)
    seen = {}

    class FakeYDL:
        def __init__(self, options):
            seen.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            out = Path(seen["outtmpl"].replace("%(ext)s", "mp4"))
            out.write_bytes(source.read_bytes())
            return {"id": "abc", "title": "Original upload", "webpage_url": url, "format_id": "18", "duration": 3}

    destination = tmp_path / "x.media"
    acquired = ytdlp_download("https://www.youtube.com/watch?v=abc", destination, InspectPolicy(), factory=FakeYDL,
                              ffmpeg="ffmpeg")
    assert acquired.path == destination.resolve() and destination.is_file()
    assert acquired.media_type == "video/mp4" and acquired.details["method"] == "yt-dlp" and acquired.label == "Original upload"
    assert seen["format"].startswith("18/") and seen["max_filesize"] == InspectPolicy().maximum_download_bytes
    assert not (tmp_path / "x.media.ytdlp").exists()                     # 临时目录清掉了
    with pytest.raises(BackendFailure):
        ytdlp_download("https://evil.example.com/watch?v=abc", tmp_path / "y.media", InspectPolicy(), factory=FakeYDL,
                       ffmpeg="ffmpeg")

    class Blocked(FakeYDL):
        def extract_info(self, url, download):
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    with pytest.raises(BackendFailure) as error:
        ytdlp_download("https://www.youtube.com/watch?v=abc", tmp_path / "z.media", InspectPolicy(), factory=Blocked,
                       ffmpeg="ffmpeg")
    assert error.value.code == "ACCESS_RESTRICTED"


def _encode(path: Path, kind: str) -> None:
    import av
    import numpy as np

    with av.open(str(path), "w") as out:
        if kind in ("video", "av1"):
            stream = out.add_stream("mpeg4" if kind == "video" else "libsvtav1", rate=10)
            stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
            for index in range(20):
                image = np.full((96, 160, 3), index * 10, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    out.mux(packet)
        else:
            stream = out.add_stream("aac", rate=44100)
            stream.layout = "mono"
            for index in range(40):
                samples = np.zeros((1, 1024), dtype=np.float32)
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                frame.sample_rate, frame.pts = 44100, index * 1024
                for packet in stream.encode(frame):
                    out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_without_ffmpeg_separate_streams_are_combined_with_pyav(tmp_path: Path):
    import av

    video_src, audio_src = tmp_path / "v.mp4", tmp_path / "a.m4a"
    _encode(video_src, "video")
    _encode(audio_src, "audio")
    formats = []

    class SplitOnly:                                # 2026.08.19 之后的 YouTube：没有音画合一格式
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            selector = self.options["format"]
            formats.append(selector)
            assert "ffmpeg_location" not in self.options
            if selector.startswith("18/"):
                raise RuntimeError("ERROR: [youtube] abc: Requested format is not available. Use --list-formats")
            source, ext = (video_src, "mp4") if selector.startswith("bv*") else (audio_src, "m4a")
            Path(self.options["outtmpl"].replace("%(ext)s", ext)).write_bytes(source.read_bytes())
            return {"id": "abc", "title": "Split upload", "webpage_url": url, "duration": 2}

    acquired = ytdlp_download("https://www.youtube.com/watch?v=abc", tmp_path / "s.media", InspectPolicy(),
                              factory=SplitOnly, ffmpeg=False)
    assert formats[0].startswith("18/") and formats[1].startswith("bv*") and formats[2].startswith("ba")
    assert acquired.details["method"] == "yt-dlp+pyav" and acquired.suffix == ".mkv"
    with av.open(str(acquired.path)) as container:
        assert len(container.streams.video) == 1 and len(container.streams.audio) == 1


def test_service_passes_format_headers_and_page_url(tmp_path: Path):
    from types import SimpleNamespace

    from foren_searcher.inspect.service import InspectService

    meta = tmp_path / "meta.json"
    url = "https://rr1.googlevideo.com/videoplayback?itag=18"
    meta.write_text(json.dumps({"extractor_key": "Youtube", "webpage_url": "https://www.youtube.com/watch?v=abc",
                                "formats": [{"url": url, "format_id": "18", "ext": "mp4", "vcodec": "avc1", "acodec": "mp4a",
                                             "protocol": "https", "height": 360,
                                             "http_headers": {"User-Agent": "Mozilla/5.0 X"}}]}), encoding="utf-8")
    trigger = SimpleNamespace(artifact=SimpleNamespace(media_type="application/json", byte_size=meta.stat().st_size), path=meta)
    assert InspectService._candidate_media_url({"platform": "youtube"}, trigger) == url
    hints = InspectService._candidate_download_hints({}, trigger, url)
    assert hints == {"headers": {"User-Agent": "Mozilla/5.0 X"}, "page_url": "https://www.youtube.com/watch?v=abc"}
    assert acquire_request_view(url)["acquirer"] == 3                 # 10-05 / 10-07 前录下的 403 不再回放


# ---------------------------------------------------------------- 2. 重复用完先给只可交报告的回合

def _evidence(store: GlobalStore):
    return store.record_evidence(content="A saved source sentence", label="A saved source sentence", source_ref="C0",
                                 locator="paragraph:1", acquired_by="TEST/source", discovered_by="W0")


class _ReadThenReport:
    def __init__(self, evidence_id: str, report: bool = True) -> None:
        self.evidence_id, self.report, self.calls, self.actions_seen = evidence_id, report, 0, []

    async def decide(self, *, model_view, actions, **kwargs):
        self.calls += 1
        self.actions_seen.append((len(actions), model_view.get("runtime_constraints", {}).get("next_output")))
        if not actions and self.report:
            return {"task_id": "W1-1", "status": "INCOMPLETE",
                    "findings": [{"statement": "The source sentence was read.", "evidence_ids": [self.evidence_id],
                                  "limitations": []}],
                    "selected_evidence_ids": [self.evidence_id], "status_reason": "needs more sources"}
        return {"action": "READ_RECORD", "arguments": {"record_id": self.evidence_id}, "focus_evidence_ids": []}


def _run(store, policy, evidence_id, guards):
    harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Read", (evidence_id,)))
    with runtime_guard_scope(guards) as state:
        report = asyncio.run(WorkspaceRunner(harness=harness, policy=policy, max_turns=3).run())
    return report, state


def test_repeat_budget_then_report_only_turn_gets_the_hi_report():
    store = GlobalStore()
    evidence = _evidence(store)
    policy = _ReadThenReport(evidence.evidence_id)
    report, state = _run(store, policy, evidence.evidence_id, merge_input_features(EVAL_GUARDS, {"constraints_visible": True}))
    assert report.status == "INCOMPLETE" and not report.status_reason.startswith("auto_report:")
    assert len(report.findings) == 1                                    # HI 自己写的报告，GIA 收得到判断
    assert policy.actions_seen[-1][0] == 0 and "WORKSPACE REPORT only" in policy.actions_seen[-1][1]
    assert state.counters.get("report_only_after_repeats") == 1 and "auto_incomplete_report" not in state.counters


def test_report_only_turn_that_still_acts_falls_back_to_auto_incomplete():
    store = GlobalStore()
    evidence = _evidence(store)
    policy = _ReadThenReport(evidence.evidence_id, report=False)
    report, state = _run(store, policy, evidence.evidence_id, merge_input_features(EVAL_GUARDS, {"constraints_visible": True}))
    assert report.status_reason.startswith("auto_report:") and "report-only turn" in report.status_reason
    assert state.counters.get("auto_incomplete_report") == 1


def test_without_constraints_visible_the_old_auto_incomplete_is_unchanged():
    store = GlobalStore()
    evidence = _evidence(store)
    policy = _ReadThenReport(evidence.evidence_id)
    report, state = _run(store, policy, evidence.evidence_id, EVAL_GUARDS)
    assert policy.calls == 5 and report.status_reason.startswith("auto_report:")
    assert state.counters == {"repeated_action_short_circuit": 3, "auto_incomplete_report": 1}


def test_av1_video_stream_is_copied_without_an_encoder(tmp_path: Path):
    # 10-05 晚本机自检：YouTube 只给了 AV1 视频流，按解码器名 libdav1d 找编码器失败（UnknownCodecError）
    import av

    from foren_searcher.inspect.acquisition import remux_video_audio

    if "libsvtav1" not in av.codecs_available or "libdav1d" not in av.codecs_available:
        pytest.skip("this PyAV build cannot write and read AV1")
    video, audio, merged = tmp_path / "v.mp4", tmp_path / "a.m4a", tmp_path / "m.mkv"
    _encode(video, "av1")
    _encode(audio, "audio")
    remux_video_audio(video, audio, merged)
    with av.open(str(merged)) as container:
        assert container.streams.video[0].codec_context.name in {"libdav1d", "av1"}
        assert len(container.streams.audio) == 1
        assert sum(1 for _ in container.decode(video=0)) == 20


def test_behavior_panel_counts_acquired_videos_and_self_vs_auto_reports(tmp_path: Path):
    # evidb11：151 下载到一个候选视频，但观察里没有 acquisition 明细，面板曾显示 0；HI 报告自写 / 代交另列
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _evid_behavior import run_behavior, summarize_behavior

    run = tmp_path / "evid_video_151_evidb11_1"
    run.mkdir()
    card = {"evidence_id": "E39", "modalities": ["video"], "source_ref": "R6", "derived_from": [], "acquired_by": "INSPECT/acquire"}
    events = [
        {"event_type": "observation", "actor_role": "HI", "actor_id": "W1-1",
         "payload": {"observation": {"action": "INSPECT", "items": [], "details": {"operation": "get_frames", "evidence": [card]}}}},
        {"event_type": "observation", "actor_role": "HI", "actor_id": "W1-1",
         "payload": {"observation": {"action": "INSPECT", "items": [], "details": {"operation": "ocr", "evidence": [card]}}}},
        {"event_type": "workspace_report", "actor_role": "HI", "actor_id": "W1-1", "payload": {"status": "INCOMPLETE"}},
        {"event_type": "workspace_report", "actor_role": "HI", "actor_id": "W1-2",
         "payload": {"status": "INCOMPLETE", "status_reason": "auto_report: turn limit reached"}},
    ]
    (run / "trajectory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    row = run_behavior(run, tmp_path / "logs")
    assert row["video_downloads"] == 1                                   # 同一个 E39 出现两次只算一次
    assert (row["hi_self_reports"], row["hi_auto_reports"]) == (1, 1)
    summary = summarize_behavior([row])
    assert summary["video_downloads_total"] == 1 and summary["hi_auto_reports"] == 1


def test_video_ocr_passes_frame_bytes_that_rapidocr_accepts(tmp_path: Path, monkeypatch):
    # evidb11：视频 OCR 把帧包成 BytesIO 交给 RapidOCR，它只收 str / Path / bytes / ndarray，1123 帧全报 LoadImageError
    import io as _io
    import types

    from foren_searcher.inspect.backends.base import PreparedMedia
    from foren_searcher.inspect.backends.local import LocalInspectBackend
    from foren_searcher.inspect.contracts import OcrParams

    seen = []

    class FakeRapidOCR:
        def __call__(self, image):
            if isinstance(image, _io.BytesIO) or not isinstance(image, (str, Path, bytes)):
                raise type("LoadImageError", (Exception,), {})(f"{type(image)} is not supported!")
            seen.append(type(image))
            return [([[0, 0], [10, 0], [10, 5], [0, 5]], "HELLO", 0.9)], None

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=FakeRapidOCR))
    video = tmp_path / "v.mp4"
    _encode(video, "video")
    backend = LocalInspectBackend()
    profile = asyncio.run(backend.probe(video, "video"))
    outcome = asyncio.run(backend.ocr((PreparedMedia("E1", video, "video", profile),), OcrParams(range_s=(0.0, 1.0)),
                                      InspectPolicy()))
    assert seen and set(seen) == {bytes}
    assert not [issue for issue in outcome.issues if "OCR failed for one frame" in str(issue)]
    assert any(item["text"] == "HELLO" for row in outcome.result["results"] for item in row["items"])



def test_ytdlp_403_retries_other_clients_for_each_stream(tmp_path: Path):
    """10-07：默认客户端被 403 拒时换客户端重试；声音那一路也一样；记下用了哪个客户端。"""

    from foren_searcher.inspect import acquisition as acq

    video_src, audio_src = tmp_path / "v.mp4", tmp_path / "a.m4a"
    _encode(video_src, "video")
    _encode(audio_src, "audio")
    calls = []

    class Picky:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            selector = self.options["format"]
            client = (self.options.get("extractor_args") or {}).get("youtube", {}).get("player_client", ["default"])[0]
            calls.append((selector[:3], client))
            if selector.startswith("18/"):
                raise RuntimeError("ERROR: [youtube] abc: Requested format is not available.")
            if selector.startswith("ba") and client in ("default", "tv"):
                raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
            source, ext = (video_src, "mp4") if selector.startswith("bv*") else (audio_src, "m4a")
            Path(self.options["outtmpl"].replace("%(ext)s", ext)).write_bytes(source.read_bytes())
            return {"id": "abc", "title": "t", "webpage_url": url, "duration": 2}

    acquired = acq.ytdlp_download("https://www.youtube.com/watch?v=abc", tmp_path / "r.media", InspectPolicy(),
                                  factory=Picky, ffmpeg=False)
    assert acquired.details["method"] == "yt-dlp+pyav"                  # 声音换到第二个客户端拿到了
    assert acquired.details["ytdlp_client"] == {"video": "default", "audio": "mweb"}
    assert [r["client"] for r in acquired.details["ytdlp_retries"]] == ["tv", "mweb"]
    assert ("18/", "tv") not in calls                                   # "没有这个格式"不是 403，不换客户端重试

    class AlwaysForbidden(Picky):
        def extract_info(self, url, download):
            calls.append(("x", "x"))
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")

    calls.clear()
    with pytest.raises(BackendFailure) as error:
        acq.ytdlp_download("https://www.youtube.com/watch?v=abc", tmp_path / "q.media", InspectPolicy(),
                           factory=AlwaysForbidden, ffmpeg="ffmpeg")
    assert error.value.code == "ACCESS_RESTRICTED" and len(calls) == 1 + len(acq.YTDLP_RETRY_CLIENTS)
