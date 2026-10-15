"""检查以图搜图各引擎的 API key 能不能用（10-12）：双击 scripts\\_check_image_keys.cmd。

只做不计费的检查（不消耗搜索次数）：
  SerpAPI   GET https://serpapi.com/account.json        → 账户、套餐、本月剩余搜索次数
  TinEye    GET https://api.tineye.com/rest/remaining_searches/ → 剩余次数
  Google Cloud Vision / Serper 没有免费的检查接口：只报告 key 有没有读到（加 --search 时各做一次真实搜索，会计费）。
加 --search：用一张测试图（YouTube 的公开缩略图）真的搜一次，看能不能返回结果。
不会打印 key 本身，只打印前 4 位和后 4 位。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.search.providers.keys import ENGINE_KEYS, find_key  # noqa: E402

TEST_IMAGE_URL = "https://i.ytimg.com/vi/jNQXAC9IVRw/hqdefault.jpg"     # "Me at the zoo"，YouTube 第一个视频的封面


def mask(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}（{len(key)} 位）" if len(key) > 8 else f"（{len(key)} 位）"


def check_serpapi(key: str) -> str:
    import httpx

    r = httpx.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=30)
    try:
        data = r.json()
    except ValueError:
        return f"失败：HTTP {r.status_code}，返回的不是 JSON"
    if r.status_code != 200 or data.get("error"):
        return f"失败：HTTP {r.status_code}，{data.get('error') or data}"
    return (f"可用：账户 {data.get('account_email', '?')}，套餐 {data.get('plan_name', '?')}，"
            f"本月剩余 {data.get('plan_searches_left', data.get('total_searches_left', '?'))} 次")


def check_tineye(key: str) -> str:
    import httpx

    r = httpx.get("https://api.tineye.com/rest/remaining_searches/", headers={"x-api-key": key}, timeout=30)
    try:
        data = r.json()
    except ValueError:
        return f"失败：HTTP {r.status_code}，返回的不是 JSON"
    if r.status_code != 200 or data.get("code") not in (200, None):
        return f"失败：HTTP {r.status_code}，{data.get('messages') or data}"
    return f"可用：{json.dumps(data.get('results', data), ensure_ascii=False)[:300]}"


async def real_search(names: list[str]) -> list[str]:
    import httpx

    from foren_searcher.datagen.run_cases import image_engine_providers
    from foren_searcher.search.contracts import SearchQueryRequest
    from foren_searcher.search.providers.base import QUERY_IMAGE_URL

    image = httpx.get(TEST_IMAGE_URL, timeout=30).content
    path = Path(tempfile.mkdtemp()) / "query.jpg"
    path.write_bytes(image)
    token = QUERY_IMAGE_URL.set(TEST_IMAGE_URL)
    out = []
    try:
        for name, provider in image_engine_providers(names, ROOT, 45.0):
            try:
                batch = await provider.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=path)
                top = [c.source_url for c in batch.candidates[:3]]
                hit = any("jNQXAC9IVRw" in u for u in (c.source_url for c in batch.candidates))
                out.append(f"{name}: {len(batch.candidates)} 条结果；前 3 条 {top}；" + ("找回了原视频" if hit else "前 5 条里没有原视频"))
            except Exception as exc:  # noqa: BLE001
                out.append(f"{name}: 失败 {type(exc).__name__}: {exc}"[:400])
            finally:
                close = getattr(provider, "aclose", None)
                if callable(close):
                    await close()
    finally:
        QUERY_IMAGE_URL.reset(token)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="每个有 key 的引擎真的搜一次测试图（会计费）")
    args = parser.parse_args()
    keys = {name: find_key(name, ROOT) for name in ENGINE_KEYS}
    print("以图搜图 key 检查（.local\\ 下的文件或环境变量）：")
    for name, (variable, filename) in ENGINE_KEYS.items():
        key = keys[name]
        print(f"  {name:8s} {'读到 ' + mask(key) if key else '没有（' + filename + ' 里没有 key，环境变量 ' + variable + ' 也没设）'}")
    print()
    if keys["serpapi"]:
        print("SerpAPI：", check_serpapi(keys["serpapi"]))
    if keys["tineye"]:
        print("TinEye：", check_tineye(keys["tineye"]))
    if keys["gcv"]:
        print("Google Cloud Vision：读到 key；没有免费的检查接口，加 --search 才会真的试一次")
    if keys["serper"]:
        print("Serper：读到 key；没有免费的检查接口，加 --search 才会真的试一次")
    if args.search:
        names = [n for n, ok in (("gcv", keys["gcv"]), ("lens", keys["serpapi"] or keys["serper"]), ("tineye", keys["tineye"]),
                                 ("yandex", True)) if ok]
        print("\n真实搜索一次（测试图 = YouTube 第一个视频的封面）：")
        for line in asyncio.run(real_search(names)):
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
