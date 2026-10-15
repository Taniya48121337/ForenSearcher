"""检索后端自检（10-05）：跑调查之前各发一次"文字 → 视频"查询，看 yt-dlp（YouTube）与 DDGS 视频搜索能不能用。

只打印结果条数与前 3 个标题，不进任何 run；查询是与 EVID 无关的中性词。退出码恒为 0（自检失败不拦调查，只在日志里提示）。

10-05 晚加：候选视频下载自检。evidb7 冒烟里对候选页面 E 的 get_frames 5/5 被 HTTP 403（下载次数 0）。
这里对 yt-dlp 搜到的第一个 10 分钟以内的视频走一遍 INSPECT 的真实下载路径（直链加平台请求头，失败再由 yt-dlp 下载），
打印用的是哪条路、文件多大；下载到临时目录，用完删掉。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest  # noqa: E402
from foren_searcher.search.providers.ddgs import DdgsProvider  # noqa: E402
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider  # noqa: E402

QUERY = "NASA Artemis I rocket launch"
# yt-dlp 2026.08.19 之前默认用 YouTube 的 android_vr 客户端，2026-07 底起它的下载一律 HTTP 403
# （github.com/yt-dlp/yt-dlp/issues/17456）；本机 10-05 是 2026.07.04，下载自检失败。
YTDLP_MIN_VERSION = (2026, 8, 18)


def ytdlp_version() -> tuple[str, bool]:
    try:
        from yt_dlp.version import __version__ as version
    except Exception:  # noqa: BLE001
        return "未安装", False
    try:
        parts = tuple(int(x) for x in version.split(".")[:3])
    except ValueError:
        return version, True
    return version, parts >= YTDLP_MIN_VERSION


async def probe(name: str, provider) -> bool:
    request = SearchQueryRequest("text", QUERY, "video")
    try:
        batch = await provider.search(request, offset=0, limit=5)
    except SearchFailure as exc:
        print(f"  {name}：失败 {exc.error.code}：{exc.error.message}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}：失败 {type(exc).__name__}: {exc}")
        return False
    titles = [c.search_title or c.source_url for c in batch.candidates[:3]]
    print(f"  {name}：{len(batch.candidates)} 条；前 3 个：{titles}")
    return bool(batch.candidates)


async def probe_download() -> None:
    import json
    import tempfile
    from types import SimpleNamespace

    from foren_searcher.inspect.acquisition import DirectMediaAcquirer
    from foren_searcher.inspect.backends.base import BackendFailure
    from foren_searcher.inspect.policy import InspectPolicy
    from foren_searcher.inspect.service import InspectService
    from foren_searcher.search.contracts import OpenSourceRequest
    from foren_searcher.search.openers.http import SafeHttpClient
    from foren_searcher.search.openers.video import VideoSourceOpener

    print("候选视频下载自检（INSPECT 对候选页面 E 取帧时走的同一条路）：")
    try:
        batch = await YtDlpSearchProvider().search(SearchQueryRequest("text", QUERY, "video"), offset=0, limit=5)
    except Exception as exc:  # noqa: BLE001
        print(f"  跳过：yt-dlp 搜索失败（{type(exc).__name__}）")
        return
    http = SafeHttpClient()
    try:
        for candidate in batch.candidates:
            try:
                opened = await VideoSourceOpener(http).open(candidate, OpenSourceRequest(candidate.source_url, "video"))
            except Exception as exc:  # noqa: BLE001
                print(f"  打开 {candidate.source_url} 失败：{type(exc).__name__}")
                continue
            page = opened.materials[0]
            info = json.loads(page.artifact_bytes)
            if not isinstance(info.get("duration"), (int, float)) or info["duration"] > 600:
                continue
            with tempfile.TemporaryDirectory() as work:
                meta = Path(work) / "meta.json"
                meta.write_bytes(page.artifact_bytes)
                trigger = SimpleNamespace(artifact=SimpleNamespace(media_type="application/json",
                                                                   byte_size=meta.stat().st_size), path=meta)
                try:
                    url = InspectService._candidate_media_url({"platform": "youtube"}, trigger)
                except BackendFailure as exc:
                    url = ""
                    print(f"  元数据里没有可直下的音画合一格式（{exc}），直接试 yt-dlp")
                hints = {k: v for k, v in InspectService._candidate_download_hints({}, trigger, url).items() if v}
                if not url:
                    url = hints.get("page_url", candidate.source_url)
                acquirer = DirectMediaAcquirer()
                try:
                    acquired = await acquirer.acquire(url, destination=Path(work) / "probe.media", policy=InspectPolicy(), **hints)
                except BackendFailure as exc:
                    print(f"  !!! 下载失败 {exc.code}：{exc}")
                    hint = ("先双击 scripts\\_update_ytdlp.cmd 更新 yt-dlp（需 ≥ 2026.08.19），仍不行再检查网络 / 代理。"
                            if "403" in str(exc) else "把这几行日志发给 Claude。")
                    print(f"  !!! 候选原片拿不到，'找原片 → 比对'这条路走不通。{hint}")
                else:
                    method = acquired.details.get("method")
                    size = acquired.path.stat().st_size / 1024 / 1024
                    note = f"；直链先失败：{acquired.details['direct_error']}" if acquired.details.get("direct_error") else ""
                    clients = acquired.details.get("ytdlp_client") or {}
                    retried = acquired.details.get("ytdlp_retries") or []
                    note += (f"；yt-dlp 客户端 {clients}" if clients else "") + \
                        (f"；被 403 后换客户端重试 {len(retried)} 次" if retried else "")
                    print(f"  可以下载：{info.get('title')!r}（{info['duration']} 秒），走 {method}，{size:.1f} MB{note}")
                    if method == "yt-dlp (video only)":
                        print("  注意：声音一路换了所有客户端仍被拒，只下到画面（取帧、画面对齐照常；这个视频的语音转写会是空的）")
                finally:
                    await acquirer.aclose()
            return
        print("  跳过：前 5 个结果里没有 10 分钟以内、能打开的视频")
    finally:
        await http.aclose()


async def main() -> int:
    version, recent = ytdlp_version()
    print(f"yt-dlp 版本：{version}" + ("" if recent else
          "  !!! 太旧：2026.08.19 以前的版本下载 YouTube 视频一律 HTTP 403。双击 scripts\\_update_ytdlp.cmd 更新"))
    print(f"文字 → 视频自检，查询：{QUERY!r}")
    ok_yt = await probe("yt-dlp（YouTube 搜索，现默认）", YtDlpSearchProvider())
    ok_dd = await probe("DDGS 视频搜索（兜底）", DdgsProvider())
    if not ok_yt and not ok_dd:
        print("!!! 两条视频搜索都不可用：这次调查里 HI 找不到原片，结果只能反映'没有检索'的情况。检查网络 / 代理 / yt-dlp 版本。")
    elif not ok_yt:
        print("!!! yt-dlp 不可用，视频搜索会退回 DDGS（evidb2 里 DDGS 视频全部失败过）。可试 pip install -U yt-dlp。")
    if ok_yt:
        try:
            await probe_download()
        except Exception as exc:  # noqa: BLE001
            print(f"!!! 下载自检出错（不影响调查）：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001  自检本身出错也不拦调查
        print(f"!!! 自检脚本出错（不影响调查）：{type(exc).__name__}: {exc}")
    raise SystemExit(0)
