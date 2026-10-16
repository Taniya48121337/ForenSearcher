"""10-16 召回实测（Yandex 单引擎，不用 Serper、不用图床）：30 张图，统一写法打开，前 10 条缩略图核对。

原样存档的是当时跑的脚本；结果在 yandex/results.json。EVID 帧取视频 38% 处，存于 query_frames/。
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Codex\workspaces\ChatGPT\ForenSearcher")
OUT = Path(__file__).resolve().parent / "yandex_rerun"   # 存档；重跑会重新搜 Yandex（免费）并重新下载缩略图
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest  # noqa: E402
from foren_searcher.search.openers.http import SafeHttpClient  # noqa: E402
from foren_searcher.search.policy import SearchPolicy  # noqa: E402
from foren_searcher.search.providers.google_vision import youtube_id  # noqa: E402
from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider  # noqa: E402
from foren_searcher.search.providers.thumb_verify import ThumbnailVerifier  # noqa: E402
from foren_searcher.search.providers.yandex_reverse import YandexReverseImageProvider  # noqa: E402

EL = "e107 e297 e260 e505 e248 e193 e372 e403 e459 e122 l341 l274 l623 l497 l703".split()
EVID = "002 009 022 042 049 056 069 076 083 096 103 110 117 015 063".split()


def norm(url: str) -> str:
    u = re.sub(r"^https?://", "", url.strip().lower())
    u = re.sub(r"^(www\.|m\.|amp\.)", "", u)
    return u.split("#", 1)[0].rstrip("/")


def jpeg(data: bytes) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    k = min(1.0, 1280 / max(im.size))
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def keyframe(path: Path, pos: float = 0.38) -> bytes:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * pos))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame from {path}")
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jpeg(buf.tobytes())


def items() -> list[dict]:
    rows = {r["case"]: r for r in json.loads((ROOT / "experiments/sft_data/unanswerable_r0b.json").read_text(encoding="utf-8"))["rows"]}
    out = []
    for c in EL:
        r = rows[f"averimatec_{c}"]
        case = json.loads((ROOT / f"experiments/sft_data/cases/averimatec_{c}.json").read_text(encoding="utf-8"))
        gt = norm(r["gt_url"].split("/http", 1)[-1] if "web.archive.org" in r["gt_url"] else r["gt_url"])
        gt = gt if not gt.startswith("s://") else gt[2:]
        out.append({"id": c, "family": r["family"], "output_type": "text",
                    "image": jpeg((ROOT / f"experiments/sft_data/media/d3/averimatec_{c}.jpg").read_bytes()),
                    "gt_norm": norm(r["gt_norm"]), "gt_full": gt, "gt_domain": norm(r["gt_norm"]).split("/", 1)[0],
                    "excluded": [norm(k) for k in (case.get("source_policy") or {}).get("excluded_keys") or []]})
    bench = {x["id"]: x for x in json.loads((ROOT / "plan-harness-1/datasets/EVID-Bench/evid_bench.json").read_text(encoding="utf-8"))}
    for v in EVID:
        out.append({"id": f"evid_{v}", "family": "EVID", "output_type": "video",
                    "image": keyframe(ROOT / f"plan-harness-1/datasets/EVID-Bench/videos/video_{v}.mp4"),
                    "gt_ids": list(bench[f"video_{v}"]["source"]), "excluded": []})
    return out


def gt_hit(item: dict, url: str) -> bool:
    if item["family"] == "EVID":
        return youtube_id(url) in item["gt_ids"]
    u = norm(url)
    if item["id"] == "l497":                          # gt_norm 丢了 blogId / logNo
        return "blog.naver.com" in u and "150165640240" in u and "iamryo" in u
    return u.startswith(item["gt_norm"]) or u.startswith(item["gt_full"])


async def main() -> int:
    OUT.mkdir(exist_ok=True)
    data = items()
    rows, fails = [], 0
    for item in data:
        (OUT / f"{item['id']}.jpg").write_bytes(item["image"])
        path = OUT / f"{item['id']}.jpg"
        verifier = ThumbnailVerifier(SafeHttpClient(SearchPolicy()), top_n=10, cache_dir=OUT / "verify_cache")
        multi = MultiReverseImageProvider([("yandex", YandexReverseImageProvider(30.0))], normalize=True,
                                          verifier=verifier, time_budget_s=90.0)
        t0 = time.monotonic()
        try:
            batch = await multi.search(SearchQueryRequest("image", "E1", item["output_type"]), offset=0, limit=30, image_path=path)
            cands = list(batch.candidates)
            error = None
            fails = 0
        except SearchFailure as exc:
            cands, error = [], f"{exc.error.code}: {exc.error.message}"[:200]
            fails += 1
        finally:
            await multi.aclose()
        results = [{"rank": i, "url": c.source_url, "verified": (c.extra or {}).get("verified_match"),
                    "scores": (c.extra or {}).get("verified_scores"), "reason": (c.extra or {}).get("verified_reason"),
                    "gt": gt_hit(item, c.source_url),
                    "gt_domain": item["family"] != "EVID" and norm(c.source_url).startswith(item["gt_domain"]),
                    "excluded": any(norm(c.source_url).startswith(k) for k in item["excluded"])}
                   for i, c in enumerate(cands, start=1)]
        row = {k: v for k, v in item.items() if k != "image"}
        row.update(error=error, n=len(results), seconds=round(time.monotonic() - t0, 1), results=results,
                   verify=verifier.stats)
        rows.append(row)
        gt = [r["rank"] for r in results if r["gt"]]
        print(f"{item['id']:9s} {item['family']:4s} n={len(results):2d} gt_rank={gt[:3] or '-'} "
              f"top10 verdicts={dict(verifier.stats['verdicts'])} err={error}", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        if fails >= 3:
            print("Yandex failed 3 times in a row; stopping.")
            break
        await asyncio.sleep(4)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
