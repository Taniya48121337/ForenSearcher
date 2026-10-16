"""10-16 多引擎以图搜图统一写法：去跟踪参数、引擎匹配类型、缩略图自核、日期类型、same 排前。

全部用 httpx.MockTransport 与本地生成的图片，不联网、不花额度。
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from foren_searcher.search.contracts import CandidateBatch, CandidateSource, SearchQueryRequest
from foren_searcher.search.openers.http import SafeHttpClient
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers import reverse_normalize as rn
from foren_searcher.search.providers import thumb_verify as tv
from foren_searcher.search.providers.base import QUERY_IMAGE_URL
from foren_searcher.search.providers.google_lens import GoogleLensProvider
from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider, merge_candidates


# ---------------------------------------------------------------- 本地生成的图片


def _picture(seed: int, size=(640, 360)) -> Image.Image:
    """低频随机图案（像照片那样有大块明暗），不同 seed 是不同的图。"""

    rng = np.random.default_rng(seed)
    grid = (rng.random((6, 10, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(grid, "RGB").resize(size, Image.Resampling.BICUBIC)
    noise = (rng.random((size[1], size[0], 3)) * 24 - 12)
    return Image.fromarray(np.clip(np.asarray(im, dtype=np.float32) + noise, 0, 255).astype(np.uint8), "RGB")


def _jpeg(im: Image.Image, q: int = 90, width: int | None = None) -> bytes:
    if width:
        im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


QUERY = _picture(1)
QUERY_BYTES = _jpeg(QUERY)


def _thumb_server(routes: dict[str, bytes | int], calls: list[str]) -> SafeHttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        body = routes.get(str(request.url), 404)
        if isinstance(body, int):
            return httpx.Response(body)
        return httpx.Response(200, content=body, headers={"content-type": "image/jpeg"})
    return SafeHttpClient(SearchPolicy(), allow_private_network=True, transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- 1. 去跟踪参数、跨引擎去重


def test_clean_url_drops_tracking_but_keeps_meaningful_params():
    assert rn.clean_url("https://news.example/a?id=123&utm_source=x&utm_medium=organic&fbclid=F") == "https://news.example/a?id=123"
    assert rn.clean_url("https://n.example/p?gclid=1&yclid=2&ref_src=twsrc&spm=a.b&article=9#top") == "https://n.example/p?article=9#top"
    assert rn.clean_url("https://www.youtube.com/watch?v=abcdefghijk&si=XYZ&feature=share") == "https://www.youtube.com/watch?v=abcdefghijk"
    assert rn.clean_url("https://shop.example/item?si=keep") == "https://shop.example/item?si=keep"   # 只在 YouTube 上去 si
    assert rn.clean_url("https://plain.example/x") == "https://plain.example/x"
    assert rn.identity("https://www.News.example/a/?utm_source=y") == rn.identity("https://news.example/a")


def _c(url, provider, rank, snippet=None, **extra):
    return CandidateSource(url, f"title {rank}", snippet, provider=provider, engine=provider, provider_rank=rank,
                           candidate_type="page", extra=extra)


def test_lens_clean_url_and_yandex_utm_url_merge_into_one():
    lists = {"lens": [_c("https://news.example/story?id=7", "google-lens", 1, match="visual", lens_backend="serper")],
             "yandex": [_c("https://www.news.example/story?id=7&utm_medium=organic&utm_source=yandexsmartcamera",
                           "yandex-reverse", 1, "A story about the fire.")]}
    slots = rn.merge_slots(lists, ["lens", "yandex"])
    assert len(slots) == 1 and slots[0]["found_by"] == ["lens", "yandex"]
    c = rn.build_candidate(slots[0])
    assert c.source_url == "https://news.example/story?id=7" and c.extra["found_by"] == ["lens", "yandex"]
    # 老写法（开关关）合不成一条
    assert len(merge_candidates(lists, ["lens", "yandex"])) == 2


def test_yandex_only_result_is_shown_without_tracking_params():
    slot = rn.merge_slots({"yandex": [_c("https://site.example/p?x=1&utm_source=yandexsmartcamera", "yandex-reverse", 1, "Text.")]},
                          ["yandex"])[0]
    c = rn.build_candidate(slot)
    assert c.source_url == "https://site.example/p?x=1"
    assert c.extra["url_tracking_removed_from"].endswith("utm_source=yandexsmartcamera")


# ---------------------------------------------------------------- 2. 不再用错标签


def test_engine_match_words():
    assert rn.engine_match("lens", _c("https://u.example/p", "l", 1, match="visual", lens_backend="serper")) == "unknown"
    assert rn.engine_match("lens", _c("https://u.example/p", "l", 1, match="visual", lens_backend="serpapi")) == "similar"
    assert rn.engine_match("lens", _c("https://u.example/p", "l", 1, match="exact", lens_backend="serpapi")) == "exact"
    assert rn.engine_match("yandex", _c("https://u.example/p", "y", 1)) == "unknown"
    assert rn.engine_match("tineye", _c("https://u.example/p", "t", 1, crawl_date="2012-01-01")) == "exact"
    for kind, word in (("full", "exact"), ("partial", "partial"), ("page", "page"), ("similar_thumbnail", "similar")):
        assert rn.engine_match("gcv", _c("https://u.example/p", "g", 1, match=kind)) == word
    assert rn.engine_match("gcv", _c("https://u.example/p", "g", 1, match="thumbnail", thumb_match="full")) == "exact"
    assert rn.engine_match("gcv", _c("https://u.example/p", "g", 1, match="thumbnail", thumb_match="partial")) == "partial"
    assert set(rn.MATCH_WORDS) == {"exact", "partial", "similar", "page", "unknown"}


def test_serper_and_yandex_results_say_match_type_not_reported():
    lists = {"lens": [_c("https://a.example/1", "google-lens", 1,
                         "Reverse image search (Google Lens): this page shows a visually similar image.",
                         match="visual", lens_backend="serper")],
             "yandex": [_c("https://a.example/1", "yandex-reverse", 1, "Page text here.")]}
    c = rn.build_candidate(rn.merge_slots(lists, ["lens", "yandex"])[0])
    assert "visually similar" not in c.search_snippet
    assert "Found by Google Lens (match type not reported), Yandex (match type not reported)." in c.search_snippet
    assert c.extra["engine_match"] == {"lens": "unknown", "yandex": "unknown"} and "match" not in c.extra


def test_lens_provider_records_backend_and_fetch_day(tmp_path):
    def handler(request):
        return httpx.Response(200, json={"organic": [{"title": "T", "link": "https://found.example/p", "date": "2 months ago",
                                                      "source": "Facebook"}]})
    provider = GoogleLensProvider(serper_key="s", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    image = tmp_path / "q.jpg"
    image.write_bytes(QUERY_BYTES)
    token = QUERY_IMAGE_URL.set("https://cdn.example/q.jpg")
    try:
        batch = asyncio.run(provider.search(SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=image))
    finally:
        QUERY_IMAGE_URL.reset(token)
    c = batch.candidates[0]
    assert c.extra["lens_backend"] == "serper" and len(c.extra["fetched_on"]) == 10 and c.extra["source"] == "Facebook"
    assert "visually similar" in c.search_snippet          # 开关关时 Lens 自己的摘要不变


# ---------------------------------------------------------------- 3. 缩略图自核


def test_hash_verdicts_on_generated_images():
    q = tv.fingerprint(QUERY_BYTES)
    same = tv.distances(q, tv.fingerprint(_jpeg(QUERY, q=40, width=120)))
    other = tv.distances(q, tv.fingerprint(_jpeg(_picture(2), width=120)))
    assert tv.verdict(same) == "same" and tv.verdict(other) == "different"
    w, h = QUERY.size
    cropped = tv.distances(q, tv.fingerprint(_jpeg(QUERY.crop((int(w * .03), int(h * .03), w - int(w * .03), h - int(h * .03))), 60, 150)))
    assert tv.verdict(cropped) == "same"


def test_letterboxed_thumbnail_still_matches():
    canvas = Image.new("RGB", (640, 480), (0, 0, 0))
    canvas.paste(QUERY, (0, 60))
    assert tv.verdict(tv.distances(tv.fingerprint(QUERY_BYTES), tv.fingerprint(_jpeg(canvas, 60, 160)))) == "same"


def test_verifier_results_cache_and_unchecked_cases(tmp_path):
    calls: list[str] = []
    routes = {"https://t.example/same.jpg": _jpeg(QUERY, 50, 120), "https://t.example/other.jpg": _jpeg(_picture(3), 50, 120),
              "https://t.example/html": b"<html>not an image</html>", "https://t.example/gone.jpg": 404}
    verifier = tv.ThumbnailVerifier(_thumb_server(routes, calls), top_n=10, cache_dir=tmp_path / "vc")
    urls = ["https://t.example/same.jpg", "https://t.example/other.jpg", "https://t.example/gone.jpg",
            "https://t.example/html", None]
    out = asyncio.run(verifier.verify(QUERY_BYTES, urls))
    assert [o["verified_match"] for o in out] == ["same", "different", "unchecked", "unchecked", "unchecked"]
    assert out[2]["reason"] == "not_found" and out[3]["reason"] == "not_an_image" and out[4]["reason"] == "no_thumbnail"
    assert 0 <= out[0]["dhash"] <= tv.SAME_DHASH and verifier.stats["verdicts"]["same"] == 1
    calls.clear()
    again = tv.ThumbnailVerifier(_thumb_server(routes, calls), top_n=10, cache_dir=tmp_path / "vc")
    out2 = asyncio.run(again.verify(QUERY_BYTES, urls[:2]))
    assert [o["verified_match"] for o in out2] == ["same", "different"] and calls == [] and again.stats["cache_hits"] == 2
    files = list((tmp_path / "vc").rglob("*.json"))
    assert len(files) == 2 and json.loads(files[0].read_text(encoding="utf-8"))["query_sha256"]


def test_verifier_blocks_private_addresses_and_oversized_files(tmp_path):
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/jpeg"})

    guarded = SafeHttpClient(SearchPolicy(), transport=httpx.MockTransport(handler))     # 默认：只许公网
    out = asyncio.run(tv.ThumbnailVerifier(guarded, max_bytes=1000).verify(QUERY_BYTES, ["http://127.0.0.1/x.jpg",
                                                                                          "http://10.0.0.5/y.jpg"]))
    assert [o["reason"] for o in out] == ["access_restricted", "access_restricted"] and calls == []
    open_client = SafeHttpClient(SearchPolicy(), allow_private_network=True, transport=httpx.MockTransport(handler))
    out = asyncio.run(tv.ThumbnailVerifier(open_client, max_bytes=1000).verify(QUERY_BYTES, ["https://big.example/z.jpg"]))
    assert out[0]["verified_match"] == "unchecked" and out[0]["reason"] == "unsupported_format"


def test_verifier_timeout_marks_unchecked():
    class Slow:
        async def fetch(self, url, *, maximum_bytes):
            await asyncio.sleep(5)

    out = asyncio.run(tv.ThumbnailVerifier(Slow(), timeout_s=0.05).verify(QUERY_BYTES, ["https://slow.example/a.jpg"]))
    assert out == [{"verified_match": "unchecked", "reason": "timeout"}]


class _Engine:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    async def search(self, request, *, offset, limit, image_path=None):
        self.calls += 1
        return CandidateBatch("x", "x", tuple(self.candidates[offset:offset + limit]), {}, {})

    def summary(self):
        return {"calls": self.calls}


def _engines():
    lens = _Engine([_c("https://a.example/1", "google-lens", 1, match="visual", lens_backend="serper",
                       lens_date="3 years ago", fetched_on="2026-10-16"),
                    _c("https://b.example/2", "google-lens", 2, match="visual", lens_backend="serper")])
    yandex = _Engine([_c("https://c.example/3?utm_source=yandexsmartcamera", "yandex-reverse", 1, "Yandex page text.")])
    from dataclasses import replace
    lens.candidates[0] = replace(lens.candidates[0], thumbnail_url="https://t.example/a.jpg")
    lens.candidates[1] = replace(lens.candidates[1], thumbnail_url="https://t.example/b.jpg")
    yandex.candidates[0] = replace(yandex.candidates[0], thumbnail_url="https://t.example/c.jpg")
    return [("lens", lens), ("yandex", yandex)]


def _search(provider, tmp_path):
    image = tmp_path / "q.jpg"
    image.write_bytes(QUERY_BYTES)
    return asyncio.run(provider.search(SearchQueryRequest("image", "E1", "text"), offset=0, limit=10, image_path=image))


def test_multi_provider_normalized_same_first_and_stats(tmp_path):
    calls: list[str] = []
    routes = {"https://t.example/a.jpg": _jpeg(_picture(7), 60, 120),       # 不同的图
              "https://t.example/b.jpg": _jpeg(QUERY, 60, 120),             # 同一张图
              "https://t.example/c.jpg": 404}
    verifier = tv.ThumbnailVerifier(_thumb_server(routes, calls), top_n=10)
    multi = MultiReverseImageProvider(_engines(), normalize=True, verifier=verifier)
    batch = _search(multi, tmp_path)
    urls = [c.source_url for c in batch.candidates]
    assert urls == ["https://b.example/2", "https://a.example/1", "https://c.example/3"]     # same 排最前，其余保持轮流顺序
    first = batch.candidates[0]
    assert first.search_snippet.startswith("Thumbnail compared with the query image: near-identical. Found by Google Lens")
    assert first.extra["verified_match"] == "same" and set(first.extra["verified_scores"]) == {"dhash", "phash"}
    second = batch.candidates[1]
    assert ("Thumbnail compared with the query image: no match (cropping, an added border or a local edit can also "
            "prevent a hash match).") in second.search_snippet
    assert second.extra["date"] == "2023" and second.extra["date_kind"] == "approx_relative"
    assert 'Approximate date (from "3 years ago" shown by Google Lens on 2026-10-16): about 2023.' in second.search_snippet
    third = batch.candidates[2]
    assert third.extra["verified_match"] == "unchecked" and "Thumbnail compared" not in third.search_snippet
    assert "Page text: Yandex page text." in third.search_snippet
    summary = multi.summary()
    assert summary["verify"]["verdicts"] == {"same": 1, "similar": 0, "different": 1, "unchecked": 1}
    assert summary["url_tracking_removed"] == 1 and summary["date_kinds"] == {"approx_relative": 1}
    assert batch.raw_response["verified"]["same"] == 1
    calls.clear()                                       # 翻页从合并结果里切，不再下载
    asyncio.run(multi.search(SearchQueryRequest("image", "E1", "text"), offset=1, limit=1, image_path=tmp_path / "q.jpg"))
    assert calls == []


def test_switch_off_is_the_old_behaviour(tmp_path, monkeypatch):
    monkeypatch.delenv("FOREN_REVERSE_NORMALIZE", raising=False)

    class Boom:
        top_n = 10
        stats = {}

        async def verify(self, *a, **k):
            raise AssertionError("verifier must not run when the switch is off")

    engines = _engines()
    multi = MultiReverseImageProvider(engines, verifier=Boom())
    batch = _search(multi, tmp_path)
    old = merge_candidates({n: e.candidates for n, e in engines}, ["lens", "yandex"])
    assert [(c.source_url, c.search_snippet, c.extra) for c in batch.candidates] == [(c.source_url, c.search_snippet, c.extra) for c in old]
    assert "normalize" not in multi.summary() and multi.verifier is None


def test_env_switch_and_run_cases_verifier(monkeypatch, tmp_path):
    from foren_searcher.datagen.run_cases import reverse_verifier

    monkeypatch.delenv("FOREN_REVERSE_NORMALIZE", raising=False)
    assert not rn.enabled() and reverse_verifier(None, tmp_path) is None
    monkeypatch.setenv("FOREN_REVERSE_NORMALIZE", "1")
    assert rn.enabled() and MultiReverseImageProvider([("yandex", _Engine([]))]).normalize
    v = reverse_verifier(None, tmp_path)
    assert v.top_n == 10 and v.stats["cache_dir"].endswith("thumb_verify_cache")
    monkeypatch.setenv("FOREN_REVERSE_VERIFY_TOP", "3")
    assert reverse_verifier(None, tmp_path).top_n == 3
    monkeypatch.setenv("FOREN_REVERSE_VERIFY_TOP", "0")
    assert reverse_verifier(None, tmp_path) is None
    monkeypatch.setenv("FOREN_REVERSE_VERIFY_TOP", "off")
    assert reverse_verifier(None, tmp_path) is None
    monkeypatch.setenv("FOREN_REVERSE_VERIFY_TOP", "5")
    monkeypatch.setenv("FOREN_THUMB_VERIFY_CACHE_DIR", "off")
    assert reverse_verifier(None, tmp_path).stats["cache_dir"] is None


def test_verification_only_covers_top_n(tmp_path):
    calls: list[str] = []
    routes = {"https://t.example/a.jpg": _jpeg(QUERY, 60, 120), "https://t.example/b.jpg": _jpeg(QUERY, 60, 120),
              "https://t.example/c.jpg": _jpeg(QUERY, 60, 120)}
    multi = MultiReverseImageProvider(_engines(), normalize=True,
                                      verifier=tv.ThumbnailVerifier(_thumb_server(routes, calls), top_n=1))
    batch = _search(multi, tmp_path)
    assert len(calls) == 1
    assert [c.extra["verified_match"] for c in batch.candidates] == ["same", "unchecked", "unchecked"]


# ---------------------------------------------------------------- 4. 日期


@pytest.mark.parametrize("text,expected", [
    ("3 years ago", "2023"), ("1 year ago", "2025"), ("a year ago", "2025"), ("2 months ago", "2026-08"),
    ("11 months ago", "2025-11"), ("3 weeks ago", "2026-09"), ("5 days ago", "2026-10-11"), ("2 hours ago", "2026-10-16"),
    ("yesterday", "2026-10-15"), ("Mar 12, 2021", None),
])
def test_relative_dates(text, expected):
    import datetime as dt

    assert rn.relative_date(text, dt.date(2026, 10, 16)) == expected


def test_date_kinds_and_sentences():
    lens_abs = rn.engine_dates("lens", _c("https://u.example/p", "l", 1, lens_date="Mar 12, 2021", fetched_on="2026-10-16"))
    assert lens_abs == [{"engine": "lens", "date": "2021-03-12", "date_kind": "page_date", "date_raw": "Mar 12, 2021"}]
    no_basis = rn.engine_dates("lens", _c("https://u.example/p", "l", 1, lens_date="3 years ago"))
    assert no_basis[0]["date_kind"] == "approx_relative" and no_basis[0]["date"] is None
    lists = {"tineye": [_c("https://old.example/p", "tineye", 1, crawl_date="2011-05-01", earliest_crawl_date="2009-02-03")],
             "lens": [_c("https://old.example/p", "google-lens", 1, match="exact", lens_backend="serpapi",
                         lens_date="Mar 12, 2021", source="Old News")]}
    c = rn.build_candidate(rn.merge_slots(lists, ["tineye", "lens"])[0])
    s = c.search_snippet
    assert "TinEye first crawled this page with the image on 2011-05-01 (a crawl date, not a publication date)." in s
    assert "Earliest TinEye crawl date among all copies: 2009-02-03 (a crawl date, not a publication date)." in s
    assert "Page date shown by Google Lens: Mar 12, 2021." in s and "Site: Old News." in s
    assert "Found by TinEye (same image), Google Lens (same image)." in s
    assert c.extra["date_kind"] == "crawl_date" and c.extra["crawl_date"] == "2011-05-01"
    assert [d["date_kind"] for d in c.extra["dates"]] == ["crawl_date", "page_date"]


def test_lens_disk_cache_entries_use_saved_at_for_relative_dates(tmp_path):
    import hashlib

    image = tmp_path / "q.jpg"
    image.write_bytes(QUERY_BYTES)
    key = hashlib.sha256(QUERY_BYTES).hexdigest()
    cache = tmp_path / "lens_cache" / key[:2]
    cache.mkdir(parents=True)
    (cache / f"{key}_en.json").write_text(json.dumps({
        "version": 1, "backend": "serper", "language": "en", "saved_at": "2023-05-01T10:00:00",
        "items": [{"url": "https://old.example/p", "title": "Old", "source": None, "image_url": None,
                   "thumbnail_url": None, "date": "3 years ago", "match": "visual", "video_id": None}]}), encoding="utf-8")

    def handler(request):
        raise AssertionError("cache hit must not call the network")

    provider = GoogleLensProvider(serper_key="s", cache_dir=tmp_path / "lens_cache",
                                  client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    batch = asyncio.run(provider.search(SearchQueryRequest("image", "E1", "text"), offset=0, limit=5, image_path=image))
    c = batch.candidates[0]
    assert c.extra["fetched_on"] == "2023-05-01"
    d = rn.engine_dates("lens", c)[0]
    assert d["date"] == "2020" and d["date_basis"] == "2023-05-01" and d["date_kind"] == "approx_relative"


def test_verification_respects_the_provider_time_budget(tmp_path):
    calls: list[str] = []
    verifier = tv.ThumbnailVerifier(_thumb_server({"https://t.example/b.jpg": _jpeg(QUERY, 60, 120)}, calls), top_n=10)
    multi = MultiReverseImageProvider(_engines(), normalize=True, verifier=verifier, time_budget_s=2.5)  # 留 2 秒余量后不到 1 秒
    batch = _search(multi, tmp_path)
    assert calls == [] and [c.extra["verified_match"] for c in batch.candidates] == ["unchecked"] * 3
    assert verifier.stats["unchecked_reasons"] == {"no_time_left": 3}
