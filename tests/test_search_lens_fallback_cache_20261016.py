"""10-16 Google Lens：SerpAPI 额度用完自动换 Serper；结果按图片 sha256 存硬盘，重跑不再计费。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest
from foren_searcher.search.providers.base import QUERY_IMAGE_URL
from foren_searcher.search.providers.google_lens import GoogleLensProvider


def _req(output_type: str = "text") -> SearchQueryRequest:
    return SearchQueryRequest("image", "E3", output_type)


def _image(tmp_path: Path, n: int = 0) -> Path:
    path = tmp_path / f"q{n}.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg" + bytes([n]))
    return path


def _run(provider, image, url=None, output_type="text"):
    token = QUERY_IMAGE_URL.set(url)
    try:
        return asyncio.run(provider.search(_req(output_type), offset=0, limit=10, image_path=image))
    finally:
        QUERY_IMAGE_URL.reset(token)


def _handler(calls, *, serpapi_out=False, serper_out=False, serper_results=None):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "serpapi.com":
            if request.url.path == "/image":
                return httpx.Response(200, json={"image_id": "IMG1"})
            if serpapi_out:
                return httpx.Response(200, json={"error": "Your account has run out of searches."})
            return httpx.Response(200, json={"exact_matches": [{"title": "Exact", "link": "https://site.example/exact"}]})
        assert request.url.host == "google.serper.dev" and request.headers["X-API-KEY"] == "s"
        if serper_out:
            return httpx.Response(400, json={"message": "Not enough credits", "statusCode": 400})
        payload = json.loads(request.content)
        results = serper_results if serper_results is not None else [
            {"title": "Found", "link": "https://found.example/page", "imageUrl": "https://found.example/p.jpg"}]
        assert payload["url"].startswith("https://")
        return httpx.Response(200, json={"organic": results})
    return handler


def _provider(calls, tmp_path=None, **kw):
    handler_kw = {k: kw.pop(k) for k in ("serpapi_out", "serper_out", "serper_results") if k in kw}
    return GoogleLensProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(calls, **handler_kw))),
                              cache_dir=(tmp_path / "lens_cache") if tmp_path is not None else None, **kw)


def test_serpapi_quota_switches_to_serper_for_this_and_later_images(tmp_path):
    calls = []
    provider = _provider(calls, serpapi_key="k", serper_key="s", serpapi_out=True)
    batch = _run(provider, _image(tmp_path, 0), url="https://cdn.example/a.jpg")
    assert [c.source_url for c in batch.candidates] == ["https://found.example/page"]
    assert [r.url.host for r in calls] == ["serpapi.com", "serpapi.com", "google.serper.dev"]
    calls.clear()
    _run(provider, _image(tmp_path, 1), url="https://cdn.example/b.jpg")
    assert [r.url.host for r in calls] == ["google.serper.dev"]            # SerpAPI 不再试
    s = provider.summary()
    assert s["billed_by_backend"] == {"serper": 2} and "run out" in s["serpapi_disabled"]   # 报错的那次不计
    assert "disabled" not in s


def test_after_serpapi_quota_local_frames_are_skipped_but_provider_stays_on(tmp_path):
    calls = []
    provider = _provider(calls, serpapi_key="k", serper_key="s", serpapi_out=True)
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0))                                # 本机帧：没有公开地址
    assert exc.value.error.code == "INVALID_INPUT"
    batch = _run(provider, _image(tmp_path, 1), url="https://cdn.example/b.jpg")
    assert batch.candidates and "disabled" not in provider.summary()


def test_serper_out_of_credits_disables_the_provider(tmp_path):
    calls = []
    provider = _provider(calls, serper_key="s", serper_out=True)
    for n in range(2):
        with pytest.raises(SearchFailure) as exc:
            _run(provider, _image(tmp_path, n), url="https://cdn.example/a.jpg")
        assert exc.value.error.code == "QUOTA_EXCEEDED"
    assert len(calls) == 1 and "disabled" in provider.summary()


def test_both_backends_out_disables_the_provider(tmp_path):
    calls = []
    provider = _provider(calls, serpapi_key="k", serper_key="s", serpapi_out=True, serper_out=True)
    with pytest.raises(SearchFailure):
        _run(provider, _image(tmp_path, 0), url="https://cdn.example/a.jpg")
    calls.clear()
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 1), url="https://cdn.example/b.jpg")
    assert exc.value.error.code == "QUOTA_EXCEEDED" and calls == []


def test_disk_cache_reused_by_a_new_provider_without_billing(tmp_path):
    calls = []
    image = _image(tmp_path, 0)
    first = _provider(calls, tmp_path, serpapi_key="k")
    before = _run(first, image, output_type="video")
    assert first.summary()["billed_searches"] == 1
    files = list((tmp_path / "lens_cache").rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text(encoding="utf-8"))["backend"] == "serpapi"
    calls.clear()
    second = _provider(calls, tmp_path, serpapi_key="k")                   # 新 run：同一张图
    after = _run(second, image, output_type="video")
    assert calls == [] and second.summary()["disk_hits"] == 1 and second.summary()["billed_searches"] == 0
    assert [c.source_url for c in after.candidates] == [c.source_url for c in before.candidates]


def test_disk_cache_keeps_empty_results_but_not_errors(tmp_path):
    calls = []
    empty = _provider(calls, tmp_path, serper_key="s", serper_results=[])
    assert _run(empty, _image(tmp_path, 0), url="https://cdn.example/a.jpg").candidates == ()
    out = _provider(calls, tmp_path, serper_key="s", serper_out=True)
    with pytest.raises(SearchFailure):
        _run(out, _image(tmp_path, 1), url="https://cdn.example/b.jpg")
    assert len(list((tmp_path / "lens_cache").rglob("*.json"))) == 1       # 只有"没有结果"那张
    calls.clear()
    again = _provider(calls, tmp_path, serper_key="s")
    assert _run(again, _image(tmp_path, 0), url="https://cdn.example/a.jpg").candidates == () and calls == []


def test_corrupt_cache_file_is_ignored(tmp_path):
    calls = []
    image = _image(tmp_path, 0)
    _run(_provider(calls, tmp_path, serpapi_key="k"), image)
    path = next((tmp_path / "lens_cache").rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    calls.clear()
    batch = _run(_provider(calls, tmp_path, serpapi_key="k"), image)
    assert batch.candidates and len(calls) == 2                            # 重新上传 + 搜索


# ---------------------------------------------------------------- imgbb 图床（默认关，FOREN_LENS_IMAGE_HOST=imgbb 打开）


def _imgbb_handler(calls, *, imgbb_status=200):
    serper = _handler(calls, serpapi_out=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.imgbb.com":
            calls.append(request)
            assert request.url.params["key"] == "ib" and request.url.params["expiration"] == "600"
            assert b'name="image"' in request.content or b"image=" in request.content
            if imgbb_status != 200:
                return httpx.Response(imgbb_status, json={"error": {"message": "Invalid API v1 key."}})
            return httpx.Response(200, json={"data": {"url": "https://i.ibb.co/abc/query.jpg"}, "success": True})
        if request.url.host == "google.serper.dev":
            assert json.loads(request.content)["url"] in {"https://i.ibb.co/abc/query.jpg", "https://cdn.example/x.jpg"}
        return serper(request)
    return handler


def _lens(calls, tmp_path=None, **kw):
    status = kw.pop("imgbb_status", 200)
    return GoogleLensProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(_imgbb_handler(calls, imgbb_status=status))),
                              cache_dir=(tmp_path / "lens_cache") if tmp_path is not None else None, **kw)


def test_local_frame_goes_through_imgbb_then_serper(tmp_path):
    calls = []
    provider = _lens(calls, serpapi_key="k", serper_key="s", imgbb_key="ib")
    batch = _run(provider, _image(tmp_path, 0))                            # 本机帧：没有公开地址
    assert [c.source_url for c in batch.candidates] == ["https://found.example/page"]
    assert [r.url.host for r in calls] == ["serpapi.com", "serpapi.com", "api.imgbb.com", "google.serper.dev"]
    s = provider.summary()
    assert s["host_uploads"] == 1 and s["image_host"] == "imgbb (expires 600s)" and s["image_host_stats"]["uploads"] == 1


def test_web_image_with_public_url_is_not_uploaded(tmp_path):
    calls = []
    provider = _lens(calls, serper_key="s", imgbb_key="ib")
    token = QUERY_IMAGE_URL.set("https://i.ibb.co/abc/query.jpg")          # 已有公开地址
    try:
        asyncio.run(provider.search(_req(), offset=0, limit=5, image_path=_image(tmp_path, 0)))
    finally:
        QUERY_IMAGE_URL.reset(token)
    assert [r.url.host for r in calls] == ["google.serper.dev"] and provider.summary()["host_uploads"] == 0


def test_without_imgbb_key_the_host_stays_off(tmp_path):
    calls = []
    provider = _lens(calls, serper_key="s")
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0))
    assert exc.value.error.code == "INVALID_INPUT" and calls == [] and provider.summary()["image_host"] == "off"


def test_cached_local_frame_is_not_uploaded_again(tmp_path):
    calls = []
    image = _image(tmp_path, 0)
    _run(_lens(calls, tmp_path, serper_key="s", imgbb_key="ib"), image)
    calls.clear()
    again = _lens(calls, tmp_path, serper_key="s", imgbb_key="ib")
    assert _run(again, image).candidates and calls == [] and again.summary()["host_uploads"] == 0


def test_bad_imgbb_key_stops_uploads_but_not_lens(tmp_path):
    calls = []
    provider = _lens(calls, serper_key="s", imgbb_key="ib", imgbb_status=400)
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0))
    assert exc.value.error.code == "PROVIDER_ERROR" and "image_host_disabled" not in provider.summary()   # 400：下次还会试
    calls.clear()
    provider2 = _lens(calls, serper_key="s", imgbb_key="ib", imgbb_status=403)
    with pytest.raises(SearchFailure) as exc:
        _run(provider2, _image(tmp_path, 1))
    assert exc.value.error.code == "AUTH_ERROR" and "image_host_disabled" in provider2.summary()
    calls.clear()
    with pytest.raises(SearchFailure) as exc:
        _run(provider2, _image(tmp_path, 2))                              # 不再上传
    assert exc.value.error.code == "INVALID_INPUT" and calls == []
    batch = _run(provider2, _image(tmp_path, 3), url="https://cdn.example/x.jpg")   # 网络图片照常
    assert batch.candidates and "disabled" not in provider2.summary()


def test_resolve_host_defaults_off_and_clamps_ttl(monkeypatch, tmp_path):
    from foren_searcher.search.providers.image_host import resolve_host

    for name in ("FOREN_LENS_IMAGE_HOST", "FOREN_LENS_IMAGE_TTL_S", "IMGBB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_host(tmp_path) == (None, None, 600)
    monkeypatch.setenv("FOREN_LENS_IMAGE_HOST", "imgbb")
    with pytest.raises(ValueError, match="imgbb"):
        resolve_host(tmp_path)
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local" / "imgbb_key.txt").write_text("# key below\nib\n", encoding="utf-8")
    monkeypatch.setenv("FOREN_LENS_IMAGE_TTL_S", "3600")
    assert resolve_host(tmp_path) == ("imgbb", "ib", 600)
    monkeypatch.setenv("FOREN_LENS_IMAGE_TTL_S", "5")
    assert resolve_host(tmp_path)[2] == 60
    monkeypatch.setenv("FOREN_LENS_IMAGE_HOST", "litterbox")
    with pytest.raises(ValueError, match="不认识"):
        resolve_host(tmp_path)


def test_serpapi_429_run_out_on_upload_counts_as_quota_and_switches(tmp_path):
    """10-16 本机实测：SerpAPI 没额度时上传就返回 HTTP 429 "your account has run out of searches."。"""

    calls = []
    serper = _handler(calls)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "serpapi.com":
            calls.append(request)
            return httpx.Response(429, text='{\n  "error": "your account has run out of searches."\n}')
        return serper(request)

    provider = GoogleLensProvider(serpapi_key="k", serper_key="s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    batch = _run(provider, _image(tmp_path, 0), url="https://cdn.example/a.jpg")
    assert batch.candidates and [r.url.host for r in calls] == ["serpapi.com", "google.serper.dev"]
    assert "QUOTA_EXCEEDED" in provider.summary()["serpapi_disabled"]


def test_plain_429_is_still_rate_limited(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={"error": "Too many requests"})

    provider = GoogleLensProvider(serpapi_key="k", serper_key="s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0), url="https://cdn.example/a.jpg")
    assert exc.value.error.code == "RATE_LIMITED" and "serpapi_disabled" not in provider.summary()


def _imgbb_only(calls, statuses):
    """imgbb 依次回 statuses 里的状态码（200 给网址）。"""

    seq = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = seq.pop(0)
        if status == 200:
            return httpx.Response(200, json={"data": {"url": "https://i.ibb.co/abc/q.jpg"}})
        return httpx.Response(status, json={"error": {"message": "Internal upload error"}})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_imgbb_sends_a_file_first_then_retries_as_base64():
    from foren_searcher.search.providers.image_host import ImgbbHost

    calls = []
    host = ImgbbHost("ib", client=_imgbb_only(calls, [400, 200]))
    assert asyncio.run(host.upload(b"\xff\xd8jpeg")) == "https://i.ibb.co/abc/q.jpg"
    assert b'filename="query.jpg"' in calls[0].content and b"image=" in calls[1].content
    assert calls[1].headers["content-type"].startswith("application/x-www-form-urlencoded")
    assert host.stats["uploads"] == 1 and host.stats["file_errors"] == 1 and host.stats["errors"] == 0


def test_imgbb_gives_up_after_both_forms_fail_and_does_not_retry_bad_key():
    from foren_searcher.search.providers.image_host import ImgbbHost

    calls = []
    host = ImgbbHost("ib", client=_imgbb_only(calls, [400, 400]))
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(host.upload(b"\xff\xd8jpeg"))
    assert "(base64)" in exc.value.error.message and len(calls) == 2
    calls.clear()
    host = ImgbbHost("ib", client=_imgbb_only(calls, [403]))
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(host.upload(b"\xff\xd8jpeg"))
    assert exc.value.error.code == "AUTH_ERROR" and len(calls) == 1


# ---------------------------------------------------------------- 备用图床 litterbox（imgbb 失败才用；10-16 本机 imgbb 新图一律 400 code 111）

LITTER_URL = "https://litter.catbox.moe/ab12cd.jpg"


def _chain_handler(calls, *, imgbb_error=None, litter_status=200, litter_text=LITTER_URL):
    """imgbb_error=None 成功；否则 (HTTP 状态, error.code)。Serper 只接受 imgbb / litterbox 的网址。"""

    serper = _handler(calls, serpapi_out=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.imgbb.com":
            calls.append(request)
            if imgbb_error is None:
                return httpx.Response(200, json={"data": {"url": "https://i.ibb.co/abc/query.jpg"}})
            status, code = imgbb_error
            return httpx.Response(status, json={"status_code": status, "error": {"message": f"err {code}", "code": code}})
        if request.url.host == "litterbox.catbox.moe":
            calls.append(request)
            assert request.url.path == "/resources/internals/api.php"
            body = request.content
            assert b'name="reqtype"' in body and b"fileupload" in body and b'name="time"' in body and b"1h" in body
            assert b'name="fileToUpload"; filename="query.jpg"' in body
            return httpx.Response(litter_status, text=litter_text)
        if request.url.host == "google.serper.dev":
            assert json.loads(request.content)["url"] in {"https://i.ibb.co/abc/query.jpg", LITTER_URL}
        return serper(request)
    return handler


def _chain(calls, tmp_path=None, *, fallback="litterbox", **kw):
    handler_kw = {k: kw.pop(k) for k in ("imgbb_error", "litter_status", "litter_text") if k in kw}
    return GoogleLensProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(_chain_handler(calls, **handler_kw))),
                              cache_dir=(tmp_path / "lens_cache") if tmp_path is not None else None,
                              serper_key="s", imgbb_key="ib", image_fallback=fallback, **kw)


def _hosts(calls):
    return [r.url.host for r in calls]


def test_imgbb_ok_litterbox_not_used(tmp_path):
    calls = []
    provider = _chain(calls)
    assert _run(provider, _image(tmp_path, 0)).candidates
    assert _hosts(calls) == ["api.imgbb.com", "google.serper.dev"]
    s = provider.summary()
    assert s["image_host"] == "imgbb (expires 600s) → litterbox (expires 1h)"
    assert s["image_host_used"] == "imgbb" and s["host_uploads_by_host"] == {"imgbb": 1}
    assert s["image_host_stats"]["fallbacks"] == 0


def test_imgbb_internal_error_falls_back_to_litterbox(tmp_path):
    calls = []
    provider = _chain(calls, imgbb_error=(400, 111))
    batch = _run(provider, _image(tmp_path, 0))
    assert [c.source_url for c in batch.candidates] == ["https://found.example/page"]
    assert _hosts(calls) == ["api.imgbb.com", "api.imgbb.com", "litterbox.catbox.moe", "google.serper.dev"]
    s = provider.summary()
    assert s["image_host_used"] == "litterbox" and s["host_uploads_by_host"] == {"litterbox": 1}
    assert s["host_uploads"] == 1 and s["billed_by_backend"] == {"serper": 1}
    stats = s["image_host_stats"]
    assert stats["fallbacks"] == 1 and stats["by_host"]["imgbb"]["errors"] == 1
    calls.clear()                                     # 111 不是 key 问题：下一张还是先试 imgbb
    _run(provider, _image(tmp_path, 1))
    assert _hosts(calls)[0] == "api.imgbb.com"


def test_imgbb_bad_key_code_100_is_dropped_and_litterbox_used_directly(tmp_path):
    calls = []
    provider = _chain(calls, imgbb_error=(400, 100))
    _run(provider, _image(tmp_path, 0))
    assert _hosts(calls) == ["api.imgbb.com", "litterbox.catbox.moe", "google.serper.dev"]   # code 100 不再换 base64 重试
    calls.clear()
    _run(provider, _image(tmp_path, 1))
    assert _hosts(calls) == ["litterbox.catbox.moe", "google.serper.dev"]
    s = provider.summary()
    assert "imgbb" in s["image_host_stats"]["dropped"][0] and "image_host_disabled" not in s
    assert s["host_uploads_by_host"] == {"litterbox": 2}


def test_both_hosts_fail_reports_error_and_keeps_trying_next_image(tmp_path):
    calls = []
    provider = _chain(calls, imgbb_error=(400, 111), litter_status=500, litter_text="Internal error")
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0))
    assert exc.value.error.code == "PROVIDER_ERROR" and "litterbox" in exc.value.error.message
    assert "google.serper.dev" not in _hosts(calls) and provider.summary()["billed_searches"] == 0
    assert "image_host_disabled" not in provider.summary() and provider.summary()["image_host_used"] is None


def test_litterbox_200_without_url_is_an_error(tmp_path):
    calls = []
    provider = _chain(calls, imgbb_error=(400, 111), litter_text="No file uploaded")
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 0))
    assert "No file uploaded" in exc.value.error.message


def test_fallback_off_keeps_imgbb_only(tmp_path):
    calls = []
    provider = _chain(calls, fallback=None, imgbb_error=(400, 111))
    with pytest.raises(SearchFailure):
        _run(provider, _image(tmp_path, 0))
    assert "litterbox.catbox.moe" not in _hosts(calls) and provider.summary()["image_host"] == "imgbb (expires 600s)"


def test_imgbb_code_100_alone_counts_as_bad_key():
    from foren_searcher.search.providers.image_host import ImgbbHost

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "Invalid API v1 key.", "code": 100}})

    host = ImgbbHost("ib", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(SearchFailure) as exc:
        asyncio.run(host.upload(b"\xff\xd8jpeg"))
    assert exc.value.error.code == "AUTH_ERROR" and len(calls) == 1


def test_resolve_fallback_default_litterbox_only_when_host_on(monkeypatch):
    from foren_searcher.search.providers.image_host import resolve_fallback

    for name in ("FOREN_LENS_IMAGE_HOST", "FOREN_LENS_IMAGE_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_fallback() is None                                   # 图床默认关：备用也关
    monkeypatch.setenv("FOREN_LENS_IMAGE_FALLBACK", "litterbox")
    assert resolve_fallback() is None
    monkeypatch.setenv("FOREN_LENS_IMAGE_HOST", "imgbb")
    monkeypatch.delenv("FOREN_LENS_IMAGE_FALLBACK")
    assert resolve_fallback() == "litterbox"
    monkeypatch.setenv("FOREN_LENS_IMAGE_FALLBACK", "off")
    assert resolve_fallback() is None
    monkeypatch.setenv("FOREN_LENS_IMAGE_FALLBACK", "0x0")
    with pytest.raises(ValueError, match="不认识"):
        resolve_fallback()


# ---------------------------------------------------------------- 硬上限（10-16 第二轮：Serper 最多 20 次）


def test_max_searches_is_a_hard_cap_and_blocks_uploads(tmp_path):
    calls = []
    provider = _chain(calls, max_searches=2)
    for n in range(2):
        assert _run(provider, _image(tmp_path, n)).candidates
    calls.clear()
    with pytest.raises(SearchFailure) as exc:
        _run(provider, _image(tmp_path, 9))
    assert exc.value.error.code == "QUOTA_EXCEEDED" and "cap reached (2)" in exc.value.error.message
    assert calls == []                                   # 到上限：不上传、不请求
    s = provider.summary()
    assert s["search_requests"] == 2 and s["billed_by_backend"] == {"serper": 2} and s["max_searches"] == 2


def test_cache_hits_do_not_count_against_the_cap(tmp_path):
    calls = []
    image = _image(tmp_path, 0)
    _run(_chain(calls, tmp_path, max_searches=1), image)
    calls.clear()
    again = _chain(calls, tmp_path, max_searches=0)
    assert _run(again, image).candidates and calls == [] and again.summary()["search_requests"] == 0


def test_run_cases_reads_the_cap(monkeypatch):
    from foren_searcher.datagen.run_cases import lens_max_searches

    monkeypatch.delenv("FOREN_LENS_MAX_SEARCHES", raising=False)
    assert lens_max_searches() is None
    monkeypatch.setenv("FOREN_LENS_MAX_SEARCHES", "20")
    assert lens_max_searches() == 20
