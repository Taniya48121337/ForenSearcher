"""10-16 第二轮：Google Lens（经 Serper）召回实测，与第一轮 Yandex 同一批查询图。

  python experiments\\image_search_probe_20261016\\recall_lens.py e107 e248 ...

- Serper 硬上限 20 次：lens/ledger.json 记每次发出的 Lens 请求（跨进程累计）；每个 provider 的 max_searches =
  20 − 已用，到了 GoogleLensProvider 自己报 QUOTA_EXCEEDED 停下。硬盘缓存（.local/lens_cache）命中不计。
- 每次搜索前后读 Serper 余额（GET https://google.serper.dev/account，不计费），记实际扣了几个额度。
- 图床：只在本进程里设 FOREN_LENS_IMAGE_HOST=imgbb（litterbox 备用，公开保留 1 小时），不写全局环境、不写跑数脚本。
- 查询图字节与第一轮相同：E/L 由 d3 原图重新编码（同一函数），EVID 用 query_frames/ 里存下的帧。
- SerpAPI 不用（serpapi_key=None）。key 只打印前后 4 位。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

os.environ["FOREN_LENS_IMAGE_HOST"] = "imgbb"          # 只在本进程
os.environ.pop("FOREN_LENS_IMAGE_FALLBACK", None)      # 默认 litterbox 备用

import httpx  # noqa: E402

from recall_yandex import gt_hit, items, norm  # noqa: E402  第一轮的样本、编码与命中判定

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest  # noqa: E402
from foren_searcher.search.openers.http import SafeHttpClient  # noqa: E402
from foren_searcher.search.policy import SearchPolicy  # noqa: E402
from foren_searcher.search.providers.google_lens import GoogleLensProvider  # noqa: E402
from foren_searcher.search.providers.image_host import resolve_fallback, resolve_host  # noqa: E402
from foren_searcher.search.providers.keys import find_key  # noqa: E402
from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider  # noqa: E402
from foren_searcher.search.providers.thumb_verify import ThumbnailVerifier  # noqa: E402

CAP = 20
# 每次 Lens 允许扣的额度上限（负责人定；超过就停）。10-16 实测一次 Lens 扣 3 个额度，余额接口约 20 秒后才更新。
MAX_CREDITS = int(os.environ.get("PROBE_MAX_CREDITS", "2"))
BALANCE_LAG_S = 25
OUT = HERE / "lens"
LEDGER = OUT / "ledger.json"


def balance(key: str) -> int | None:
    try:
        return int(httpx.get("https://google.serper.dev/account", headers={"X-API-KEY": key}, timeout=30).json()["balance"])
    except Exception:  # noqa: BLE001
        return None


class Recorder:
    """包住 Lens：记下 Lens 自己的原始名次与候选（离线合并要用）。"""

    def __init__(self, inner):
        self.inner = inner
        self.batch = None

    async def search(self, request, *, offset, limit, image_path=None):
        self.batch = await self.inner.search(request, offset=offset, limit=limit, image_path=image_path)
        return self.batch

    def summary(self):
        return self.inner.summary()

    async def aclose(self):
        await self.inner.aclose()


async def main(ids: list[str]) -> int:
    OUT.mkdir(exist_ok=True)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {"requests": 0, "entries": []}
    serper = find_key("serper", ROOT)
    assert serper and not find_key("serpapi", ROOT), "需要 Serper key，且 SerpAPI 必须停用"
    print(f"Serper key {serper[:4]}…{serper[-4:]}；已用 {ledger['requests']}/{CAP}；余额 {balance(serper)}")
    _host, imgbb_key, ttl = resolve_host(ROOT)
    fallback = resolve_fallback()
    wanted = {i for i in ids}
    data = [it for it in items() if it["id"] in wanted or it["id"].removeprefix("evid_") in wanted]
    results_path = OUT / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    done = {r["id"] for r in rows}
    host_fail_streak = 0
    for item in data:
        if item["id"] in done:
            print(f"{item['id']}: 已跑过，跳过")
            continue
        remaining = CAP - ledger["requests"]
        frame = HERE / "query_frames" / f"{item['id']}.jpg"
        path = OUT / "query" / f"{item['id']}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(frame.read_bytes() if frame.is_file() else item["image"])
        lens = GoogleLensProvider(serpapi_key=None, serper_key=serper, timeout_s=60.0, cache_dir=ROOT / ".local" / "lens_cache",
                                  imgbb_key=imgbb_key, image_ttl_s=ttl, image_fallback=fallback, max_searches=max(0, remaining))
        rec = Recorder(lens)
        verifier = ThumbnailVerifier(SafeHttpClient(SearchPolicy()), top_n=10, cache_dir=OUT / "verify_cache")
        multi = MultiReverseImageProvider([("lens", rec)], normalize=True, verifier=verifier, engine_timeout_s=90.0)
        before = balance(serper)
        t0 = time.monotonic()
        try:
            batch = await multi.search(SearchQueryRequest("image", "E1", item["output_type"]), offset=0, limit=30, image_path=path)
            cands, error = list(batch.candidates), None
        except SearchFailure as exc:
            cands, error = [], f"{exc.error.code}: {exc.error.message}"[:300]
        finally:
            await multi.aclose()
        s = lens.summary()
        if s["search_requests"]:
            await asyncio.sleep(BALANCE_LAG_S)            # 余额接口有延迟，等它更新再读
        after = balance(serper)
        used = s["search_requests"]
        ledger["requests"] += used
        ledger["entries"].append({"id": item["id"], "requests": used, "billed": s["billed_by_backend"],
                                  "disk_hits": s["disk_hits"], "balance_before": before, "balance_after": after,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
        raw = [asdict(c) for c in (rec.batch.candidates if rec.batch else ())]
        results = [{"rank": i, "url": c.source_url, "title": c.search_title, "verified": (c.extra or {}).get("verified_match"),
                    "scores": (c.extra or {}).get("verified_scores"), "reason": (c.extra or {}).get("verified_reason"),
                    "date": (c.extra or {}).get("date"), "date_kind": (c.extra or {}).get("date_kind"),
                    "date_raw": (c.extra or {}).get("date_raw"), "date_basis": (c.extra or {}).get("date_basis"),
                    "gt": gt_hit(item, c.source_url),
                    "gt_domain": item["family"] != "EVID" and norm(c.source_url).startswith(item["gt_domain"]),
                    "excluded": any(norm(c.source_url).startswith(k) for k in item["excluded"]),
                    "snippet": c.search_snippet}
                   for i, c in enumerate(cands, start=1)]
        uploaded = s.get("host_uploads", 0)
        host_err = s.get("image_host_stats", {}).get("errors", 0)
        host_fail_streak = host_fail_streak + 1 if host_err and not uploaded else 0
        row = {k: v for k, v in item.items() if k != "image"}
        row.update(error=error, n=len(results), seconds=round(time.monotonic() - t0, 1), results=results, lens_raw=raw,
                   lens_summary=s, verify=verifier.stats, balance_before=before, balance_after=after,
                   credits=(before - after) if before is not None and after is not None else None)
        rows.append(row)
        results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        gt = [r["rank"] for r in results if r["gt"]]
        print(f"{item['id']:9s} {item['family']:4s} 结果 {len(results):2d} 真来源 {gt[:3] or '-'} 请求 {used} "
              f"扣额度 {row['credits']} 缓存命中 {s['disk_hits']} 图床 {s.get('image_host_used')} 上传 {uploaded} "
              f"(图床错误 {host_err}) 核对 {dict(verifier.stats['verdicts'])} 错误 {error}", flush=True)
        if row["credits"] is not None and row["credits"] > MAX_CREDITS:
            print(f"停：这一次扣的额度 {row['credits']} 大于 {MAX_CREDITS}"); break
        if host_fail_streak >= 2:
            print("停：图床连续失败"); break
        if ledger["requests"] >= CAP:
            print(f"停：已到 {CAP} 次上限"); break
    print(f"已用 {ledger['requests']}/{CAP}；余额 {balance(serper)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
