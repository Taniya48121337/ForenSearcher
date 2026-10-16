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
                if name == "lens":         # 10-16：实际用的后端（SerpAPI 用完会自动换 Serper）与硬盘缓存命中
                    s = provider.summary()
                    out.append(f"      lens 计费 {s.get('billed_by_backend')}，缓存命中 {s.get('disk_hits')}，"
                               f"SerpAPI 停用原因 {s.get('serpapi_disabled', '无')}，缓存目录 {s.get('cache_dir')}")
            except Exception as exc:  # noqa: BLE001
                out.append(f"{name}: 失败 {type(exc).__name__}: {exc}"[:400])
            finally:
                close = getattr(provider, "aclose", None)
                if callable(close):
                    await close()
    finally:
        QUERY_IMAGE_URL.reset(token)
    return out


async def lens_local_frame() -> str:
    """10-16：模拟本机帧（不给公开地址）走一次 Lens，看 imgbb 图床 + Serper 是否打通。会计费 1 次。"""

    import httpx

    from foren_searcher.datagen.run_cases import image_engine_providers
    from foren_searcher.search.contracts import SearchQueryRequest

    raw = httpx.get(TEST_IMAGE_URL, timeout=30).content
    path = Path(tempfile.mkdtemp()) / "query.jpg"
    try:            # 重新编码成另一张合法 JPEG（内容不同 → 避开硬盘缓存）；10-16 末尾硬加字节时 imgbb 回 400
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "JPEG", quality=88)
        path.write_bytes(buf.getvalue())
    except Exception:  # noqa: BLE001
        path.write_bytes(raw)
    (name, provider), = image_engine_providers(["lens"], ROOT, 45.0)
    try:
        batch = await provider.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=5, image_path=path)
        s = provider.summary()
        hit = any("jNQXAC9IVRw" in c.source_url for c in batch.candidates)
        return (f"{len(batch.candidates)} 条结果，" + ("找回了原视频" if hit else "前 5 条里没有原视频")
                + f"；图床 {s.get('image_host')}，实际用 {s.get('image_host_used') or '无'}，"
                f"上传 {s.get('host_uploads')} 次 {s.get('host_uploads_by_host')}，计费 {s.get('billed_by_backend')}，"
                f"缓存命中 {s.get('disk_hits')}")
    except Exception as exc:  # noqa: BLE001
        return f"失败 {type(exc).__name__}: {exc}"[:400]
    finally:
        await provider.aclose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="每个有 key 的引擎真的搜一次测试图（会计费）")
    parser.add_argument("--lens-local", action="store_true",
                        help="不给公开地址、按本机帧的方式走一次 Lens（SerpAPI 用完时会经 imgbb/litterbox + Serper，计费 1 次）")
    args = parser.parse_args()
    from foren_searcher.search.providers.image_host import LITTERBOX_TIME, find_imgbb_key, resolve_fallback, resolve_host

    imgbb = find_imgbb_key(ROOT)
    try:
        host, _k, ttl = resolve_host(ROOT)
        fallback = resolve_fallback()
        host_line = (f"imgbb，{ttl} 秒后自动删除；" + (f"失败时改用 litterbox（{LITTERBOX_TIME} 后删除）" if fallback
                                                   else "没有备用图床（FOREN_LENS_IMAGE_FALLBACK=off）")
                     if host else "关（FOREN_LENS_IMAGE_HOST 没设成 imgbb）")
    except ValueError as exc:
        host_line = f"设置有误：{exc}"
    keys = {name: find_key(name, ROOT) for name in ENGINE_KEYS}
    print("以图搜图 key 检查（.local\\ 下的文件或环境变量）：")
    for name, (variable, filename) in ENGINE_KEYS.items():
        key = keys[name]
        print(f"  {name:8s} {'读到 ' + mask(key) if key else '没有（' + filename + ' 里没有 key，环境变量 ' + variable + ' 也没设）'}")
    print(f"  {'imgbb':8s} {'读到 ' + mask(imgbb) if imgbb else '没有（imgbb_key.txt 里没有 key，环境变量 IMGBB_API_KEY 也没设）'}")
    print(f"  本机帧走 Serper 的图床：{host_line}")
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
    if args.lens_local:
        print("\n按本机帧的方式走一次 Google Lens（不给公开地址）：")
        print("  " + asyncio.run(lens_local_frame()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
