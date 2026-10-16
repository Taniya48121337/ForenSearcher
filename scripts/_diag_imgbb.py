"""imgbb 上传诊断（10-16）：双击 scripts\\_diag_imgbb.cmd。

本机实测 imgbb 对 file / base64 两种上传都回 400 "Internal upload error"。这里换几种组合各试一次，看是哪一项引起的：
  图片：YouTube "Me at the zoo" 公开封面原样 / 程序画的一张 64x64 PNG
  方式：file（multipart）/ base64 表单
  过期：expiration=600 / 不带 expiration（不带的那几张不会自动删除，但都是公开测试图）
  User-Agent：httpx 默认 / 浏览器
只传测试图，不传调查材料，不扣 Serper。key 只打印前后 4 位。
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from foren_searcher.search.providers.image_host import IMGBB_API, find_imgbb_key  # noqa: E402

TEST_IMAGE_URL = "https://i.ytimg.com/vi/jNQXAC9IVRw/hqdefault.jpg"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0 Safari/537.36")


def tiny_png() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 40, 40)).save(buf, "PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        # 1x1 红色 PNG
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")


def attempt(client: httpx.Client, key: str, image: bytes, name: str, form: str, expiration: bool, ua: bool) -> str:
    params = {"key": key, **({"expiration": "600"} if expiration else {})}
    headers = {"User-Agent": BROWSER_UA} if ua else {}
    kwargs = ({"files": {"image": (name, image)}} if form == "file"
              else {"data": {"image": base64.b64encode(image).decode("ascii")}})
    try:
        r = client.post(IMGBB_API, params=params, headers=headers, **kwargs)
    except httpx.HTTPError as exc:
        return f"网络错误 {type(exc).__name__}: {exc}"[:200]
    try:
        data = r.json()
    except ValueError:
        return f"HTTP {r.status_code}，不是 JSON：{r.text[:150]!r}"
    if r.status_code == 200 and isinstance(data.get("data"), dict):
        return f"成功 {data['data'].get('url')}"
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return f"HTTP {r.status_code}，code {err.get('code')}：{err.get('message')}"
    return f"HTTP {r.status_code}：{str(data)[:150]}"


def main() -> int:
    key = find_imgbb_key(ROOT)
    if not key:
        print("没有读到 imgbb key（.local\\imgbb_key.txt 或 IMGBB_API_KEY）")
        return 1
    print(f"imgbb key：{key[:4]}…{key[-4:]}（{len(key)} 位）\n")
    with httpx.Client(timeout=60) as client:
        try:
            jpg = client.get(TEST_IMAGE_URL).content
        except httpx.HTTPError as exc:
            print(f"下载测试图失败：{exc}")
            jpg = b""
        images = [("tiny.png", tiny_png())] + ([("zoo.jpg", jpg)] if jpg else [])
        rows = []
        for name, image in images:
            for form in ("file", "base64"):
                for expiration in (True, False):
                    rows.append((name, form, expiration, False))
        rows.append(("tiny.png", "file", True, True))
        for name, form, expiration, ua in rows:
            image = dict(images)[name]
            label = f"{name:9s} {form:6s} {'过期600s' if expiration else '不过期  '} {'浏览器UA' if ua else '默认UA  '}"
            print(f"  {label}  →  {attempt(client, key, image, name, form, expiration, ua)}")
    print("\n把上面这几行整段贴回对话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
