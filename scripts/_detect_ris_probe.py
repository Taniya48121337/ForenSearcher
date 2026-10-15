"""关键帧以图搜图找原片：召回实测（10-12）。双击 scripts\\_detect_ris_probe.cmd。

直接用 SEARCH 的以图搜图 Provider（src/foren_searcher/search/providers/google_vision.py，run_cases --search-arm gcvimage
打开的就是它），所以这里量到的就是 agent 用 SEARCH image → video 时能看到的候选；另外用 SEARCH 里已有的免 key
Yandex Provider 同样跑一遍作对照。

问题：DEFAME / T²Agent / 多智能体核查这些论文都靠"以图搜图"找出处。我们约 70% 的 AI 样本原片从没进过候选池。
如果把视频关键帧拿去 Google Cloud Vision 的 Web Detection（DEFAME 用的同一个接口）搜，GT 原片能不能被搜出来？

不跑 agent、不跑模型，只量召回：
  - 样本：002–130 开发段每隔 6 个取 1 个（约 21 个），加上已读过的 18 个 AI 生成开发样本（插入 / 局部改动 / 换脸）
  - 每个视频取 4 帧（时长的 15% / 38% / 62% / 85%），缩到长边 1280，JPEG 发给 Web Detection（每帧 1 次调用，共约 160 次，
    在每月 1000 次免费额度内）
  - 命中 = 返回的任何 URL（匹配页面、完全 / 部分匹配图、相似图）里出现 GT 原片的 YouTube 编号
  - 另记：返回里有没有 youtube.com 页面、best guess 标签与网页实体（这些可以当文字查询再去搜）

Google 的 API key：放在 .local\\google_vision_key.txt（一行），或环境变量 GOOGLE_VISION_API_KEY，脚本不会打印 key。
没有 key 也能跑：只跑免 key 的 Yandex，Google 那几列写"未跑"；之后放好 key 再双击，Yandex 用缓存，只补 Google。
每帧的原始返回缓存在 outputs\\detect\\ris_probe\\cache\\，重跑不会重复计费。结果 outputs\\detect\\ris_probe\\summary.md。
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench"
VIDEOS = DATASET / "videos"
OUT = ROOT / "outputs" / "detect" / "ris_probe"
CACHE = OUT / "cache"
KEY_FILE = ROOT / ".local" / "google_vision_key.txt"
AI_DEV = {"152": "insert", "155": "insert", "160": "insert", "163": "insert", "167": "insert", "171": "insert",
          "201": "edit", "205": "edit", "210": "edit", "213": "edit", "216": "edit", "220": "edit",
          "175": "swap", "177": "swap", "178": "swap", "181": "swap", "184": "swap", "187": "swap"}
POSITIONS = (0.15, 0.38, 0.62, 0.85)
ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%m-%d %H:%M:%S')} {text}\n")


def api_key() -> str:
    key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return key


def samples() -> list[tuple[str, str, str]]:
    entries = json.loads((DATASET / "evid_bench.json").read_text(encoding="utf-8"))
    by_id = {str(e["id"]).removeprefix("video_"): e for e in entries}
    dev = [f"{i:03d}" for i in range(2, 131, 6)]
    out = []
    for s in dev + list(AI_DEV):
        e = by_id.get(s)
        if not e or not e.get("source"):
            log(f"  {s}：标注里没有原片编号，跳过")
            continue
        out.append((s, AI_DEV.get(s, "dev"), str(e["source"][0])))
    return out


def keyframes(path: Path) -> list[tuple[float, bytes]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for p in POSITIONS:
        idx = int(n * p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        k = min(1.0, 1280 / max(h, w))
        if k < 1:
            bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append((round(idx / fps, 1), buf.tobytes()))
    cap.release()
    return out


def web_detection(provider, jpeg: bytes, cache: Path) -> dict:
    """SEARCH 的 GoogleVisionWebProvider 发请求；原始 webDetection 存盘，重跑不重复计费。provider 为 None 时只读缓存。"""

    import asyncio

    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    if provider is None:
        raise RuntimeError("no Google key")
    web = asyncio.run(provider._annotate(jpeg))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(web, ensure_ascii=False), encoding="utf-8")
    return web


def yandex_rank(jpeg: bytes, source: str, cache: Path) -> dict:
    """SEARCH 里已有的 Yandex 反向图片 Provider（免 key），output_type=video，看原片排第几。"""

    import asyncio
    import tempfile

    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    from foren_searcher.search.contracts import SearchQueryRequest
    from foren_searcher.search.providers.yandex_reverse import YandexReverseImageProvider

    async def run():
        provider = YandexReverseImageProvider(30.0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "q.jpg"
                path.write_bytes(jpeg)
                batch = await provider.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=50, image_path=path)
            return [c.source_url for c in batch.candidates]
        finally:
            await provider.aclose()

    try:
        urls = asyncio.run(run())
        out = {"ok": True, "n": len(urls), "rank": next((i for i, u in enumerate(urls, 1) if source in u), None)}
    except Exception as exc:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(exc).__name__}: {str(getattr(exc, 'error', exc))[:160]}"}
    if out["ok"]:                        # 失败（网络、验证码）不缓存，重跑会再试
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    time.sleep(2)
    return out


def youtube_ids(text: str) -> set[str]:
    ids = set(re.findall(r"(?:v=|youtu\.be/|/shorts/|/embed/|/vi/|/vi_webp/|/live/)([A-Za-z0-9_-]{11})", text))
    return ids


def analyse(web: dict, source: str) -> dict:
    from foren_searcher.search.providers.google_vision import normalize_web_detection

    video_list = normalize_web_detection(web, "video")
    search_rank = next((i for i, item in enumerate(video_list, 1) if item.get("video_id") == source or source in item["url"]), None)
    groups = {"pages": [p.get("url", "") for p in web.get("pagesWithMatchingImages", [])],
              "full": [i.get("url", "") for i in web.get("fullMatchingImages", [])],
              "partial": [i.get("url", "") for i in web.get("partialMatchingImages", [])],
              "similar": [i.get("url", "") for i in web.get("visuallySimilarImages", [])]}
    # 匹配页面里也可能只在它附带的图片 URL 中出现编号
    page_imgs = [i.get("url", "") for p in web.get("pagesWithMatchingImages", [])
                 for i in (p.get("fullMatchingImages", []) + p.get("partialMatchingImages", []))]
    hit_in = []
    for name, urls in list(groups.items()) + [("page_images", page_imgs)]:
        for rank, url in enumerate(urls, 1):
            if source in youtube_ids(url) or f"/{source}" in url or f"={source}" in url:
                hit_in.append((name, rank))
                break
    yt_pages = [u for u in groups["pages"] if "youtube.com" in u or "youtu.be" in u]
    return {"hit": bool(hit_in), "hit_in": hit_in, "search_rank": search_rank, "search_videos": sum(
                _is_video(i["url"]) for i in video_list),
            "n": {k: len(v) for k, v in groups.items()},
            "youtube_pages": yt_pages[:10],
            "other_youtube_ids": sorted(set().union(*[youtube_ids(u) for u in sum(groups.values(), [])]) - {source})[:10],
            "best_guess": [g.get("label") for g in web.get("bestGuessLabels", [])],
            "entities": [e.get("description") for e in web.get("webEntities", [])[:8] if e.get("description")]}


def _is_video(url: str) -> bool:
    return any(h in url.lower() for h in ("youtube.com/", "youtu.be/", "vimeo.com/", "bilibili.com/video/", "tiktok.com/"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from foren_searcher.search.providers.google_vision import GoogleVisionWebProvider

    key = api_key()
    provider = GoogleVisionWebProvider(key) if key else None
    if provider is None:
        log(f"没有 Google Cloud Vision key（{KEY_FILE}），这次只跑免 key 的 Yandex；有 key 后再双击会补上 Google。")
    todo = samples()
    log(f"==== 以图搜图召回：{len(todo)} 个视频 × {len(POSITIONS)} 帧 ====")
    results, calls = [], 0
    for s, kind, source in todo:
        path = VIDEOS / f"video_{s}.mp4"
        if not path.is_file():
            log(f"  {s}：没有视频文件")
            continue
        frames = []
        for t, jpeg in keyframes(path):
            cache = CACHE / f"{s}_{t:.1f}.json"
            fresh = not cache.is_file()
            web, google_ok = {}, False
            if provider is not None or cache.is_file():
                try:
                    web = web_detection(provider, jpeg, cache)
                    google_ok = True
                    calls += fresh
                except Exception as exc:  # noqa: BLE001
                    log(f"!!! {s} t={t} Google：{getattr(exc, 'error', exc)}")
                    if "AUTH_ERROR" in repr(getattr(exc, "error", "")):
                        log("Google key 不能用（是否启用了 Cloud Vision API、key 是否限制了 API / IP）；后面只跑 Yandex。")
                        provider = None
            frames.append({"t": t, **analyse(web, source), "google_ok": google_ok,
                           "yandex": yandex_rank(jpeg, source, CACHE / f"{s}_{t:.1f}.yandex.json")})
        hit = any(f["hit"] for f in frames)
        results.append({"sample": s, "kind": kind, "source": source, "hit": hit, "frames": frames})
        yr = [(f.get("yandex") or {}).get("rank") for f in frames]
        log(f"  {s} {kind}：Google {'命中' if hit else '未命中' if any(f['google_ok'] for f in frames) else '未跑'}；"
            f"Yandex 名次 {yr}（失败 {sum(not (f.get('yandex') or {}).get('ok') for f in frames)} 帧）")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    def g_hit(f, k=None):
        return f["search_rank"] is not None and (k is None or f["search_rank"] <= k)

    def y_hit(f, k=None):
        y = f.get("yandex") or {}
        return y.get("rank") is not None and (k is None or y["rank"] <= k)

    lines = [f"# 关键帧以图搜图找原片：召回（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"每个视频 {len(POSITIONS)} 帧（时长 15/38/62/85%），每帧走一次 SEARCH 的 image → video。本次 Google 新调用 {calls} 次。",
             "- Google：SEARCH 新加的 Google Cloud Vision Provider（--search-arm gcvimage）。前 10 = SEARCH 给 agent 的候选里原片排在前 10；"
             "原始返回 = Google 返回的任何 URL 里有原片编号（不管 SEARCH 有没有把它排进视频候选）。",
             "- Yandex：SEARCH 里已有的免 key Provider，同一帧同样请求。", "",
             "| 组 | 视频 | Google 前 10（视频级） | Google 前 10（帧级） | Google 原始返回（视频级） | Yandex 前 10（视频级） | Yandex 可用帧 |",
             "|---|---|---|---|---|---|---|"]
    for label, kinds in (("开发段 002–130（真实素材）", ("dev",)), ("插入", ("insert",)), ("局部改动", ("edit",)),
                         ("换脸", ("swap",)), ("AI 生成合计", ("insert", "edit", "swap"))):
        rs = [r for r in results if r["kind"] in kinds]
        if not rs:
            continue
        fr = [f for r in rs for f in r["frames"]]
        if not any(f["google_ok"] for f in fr):
            lines.append(f"| {label} | {len(rs)} | 未跑 | 未跑 | 未跑 | "
                         f"{sum(any(y_hit(f, 10) for f in r['frames']) for r in rs)} | {sum((f.get('yandex') or {}).get('ok', False) for f in fr)}/{len(fr)} |")
            continue
        lines.append(f"| {label} | {len(rs)} | {sum(any(g_hit(f, 10) for f in r['frames']) for r in rs)} | "
                     f"{sum(g_hit(f, 10) for f in fr)}/{len(fr)} | {sum(r['hit'] for r in rs)} | "
                     f"{sum(any(y_hit(f, 10) for f in r['frames']) for r in rs)} | {sum((f.get('yandex') or {}).get('ok', False) for f in fr)}/{len(fr)} |")
    lines += ["", "## 逐个", "", "| 样本 | 类型 | 原片 | Google 名次（每帧） | 原始返回命中位置 | Yandex 名次（每帧） | best guess | 网页实体 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        g = ", ".join(str(f["search_rank"] or "—") for f in r["frames"])
        y = ", ".join(str((f.get("yandex") or {}).get("rank") or ("—" if (f.get("yandex") or {}).get("ok") else "错")) for f in r["frames"])
        where = "; ".join(f"t={f['t']}: {f['hit_in']}" for f in r["frames"] if f["hit"]) or "—"
        guess = "; ".join(sorted({x for f in r["frames"] for x in f["best_guess"] if x}))[:80]
        ents = ", ".join(list(dict.fromkeys(e for f in r["frames"] for e in f["entities"]))[:6])[:120]
        lines.append(f"| {r['sample']} | {r['kind']} | {r['source']} | {g} | {where} | {y} | {guess} | {ents} |")
    errs = [f["yandex"]["error"] for r in results for f in r["frames"] if not (f.get("yandex") or {}).get("ok", True)]
    if errs:
        lines += ["", f"Yandex 失败 {len(errs)} 次，例：{errs[0]}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"完成：{OUT / 'summary.md'}；新调用 {calls} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
