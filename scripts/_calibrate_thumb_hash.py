"""校准以图搜图自核的感知哈希阈值（10-16）：只读本机图片，不联网、不上传。

  python scripts\\_calibrate_thumb_hash.py [图片目录 ...]

默认用 experiments/sft_data/media/d3 的图片做"不同的图"，experiments/evid/runs/*/artifacts 里同一视频的相邻帧做难负例。
正例：同一张图做成搜索引擎缩略图的样子——缩到 96–200 px 宽、JPEG q30–75、四边裁 ≤6%、加上下黑边，及其组合；
另报"裁 10%"一档（超出"轻微裁切"，只看会不会被判成 same）。
输出每种变换 / 负例的 dHash、pHash 距离分布，以及当前阈值（thumb_verify.SAME_* / SIMILAR_*）下的判定计数。
"""

from __future__ import annotations

import io
import itertools
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from foren_searcher.search.providers import thumb_verify as tv  # noqa: E402


def jpeg(im: Image.Image, q: int) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def resize_w(im: Image.Image, w: int) -> Image.Image:
    return im.resize((w, max(8, round(im.height * w / im.width))), Image.Resampling.BILINEAR)


def crop(im: Image.Image, f: float, rng: random.Random) -> Image.Image:
    w, h = im.size
    l, t, r, b = (rng.uniform(0, f) for _ in range(4))
    return im.crop((int(w * l), int(h * t), w - int(w * r), h - int(h * b)))


def letterbox(im: Image.Image) -> Image.Image:
    """整张图放进 4:3 黑底画布（YouTube hqdefault 的样子：宽图上下黑边，竖图左右黑边）。"""

    w, h = im.size
    cw, ch = (w, round(w * 3 / 4)) if w / h >= 4 / 3 else (round(h * 4 / 3), h)
    canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
    canvas.paste(im.convert("RGB"), ((cw - w) // 2, (ch - h) // 2))
    return canvas


TRANSFORMS = {
    "resize120_q75": lambda im, r: jpeg(resize_w(im, 120), 75),
    "resize96_q50": lambda im, r: jpeg(resize_w(im, 96), 50),
    "resize200_q30": lambda im, r: jpeg(resize_w(im, 200), 30),
    "crop3_resize150_q60": lambda im, r: jpeg(resize_w(crop(im, 0.03, r), 150), 60),
    "crop6_resize120_q50": lambda im, r: jpeg(resize_w(crop(im, 0.06, r), 120), 50),
    "letterbox_resize160_q60": lambda im, r: jpeg(resize_w(letterbox(im), 160), 60),
    "crop10_resize120_q60(beyond slight)": lambda im, r: jpeg(resize_w(crop(im, 0.10, r), 120), 60),
}


def case_of(name: str) -> str:
    """金标集文件名 videosham_4126_split_intervals_r7_E17.jpg → videosham_4126；其他文件各算一案。"""

    if re.search(r"_E\d+\.jpe?g$", name):
        return "_".join(name.split("_")[:2])
    return name


def load(paths: list[Path], limit: int) -> list[tuple[str, Image.Image]]:
    out = []
    for p in paths[:limit]:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        if min(im.size) >= 120:
            out.append((p.name, im))
    return out


def q(values: list[float]) -> str:
    a = np.array(values)
    return (f"n={len(a):4d} min {a.min():.3f} p05 {np.percentile(a, 5):.3f} p50 {np.median(a):.3f} "
            f"p95 {np.percentile(a, 95):.3f} max {a.max():.3f}")


def main() -> int:
    rng = random.Random(16)
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "experiments" / "sft_data" / "media" / "d3"]
    paths = sorted(p for d in dirs for p in d.glob("*.jp*g"))
    images = load(paths, 160)
    print(f"图片 {len(images)} 张（{', '.join(str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d) for d in dirs)}）")
    queries = {name: tv.fingerprint(jpeg(im, 90)) for name, im in images}
    pos: dict[str, list[dict[str, float]]] = defaultdict(list)
    thumbs: dict[str, dict] = {}
    for name, im in images:
        for tname, fn in TRANSFORMS.items():
            fp = tv.fingerprint(fn(im, rng))
            pos[tname].append(tv.distances(queries[name], fp))
            if tname == "resize120_q75":
                thumbs[name] = fp
    small = {name: np.asarray(im.resize((64, 36)).convert("L"), dtype=np.float32) for name, im in images}
    neg, dup_pairs, wrong = [], 0, []
    for a, b in itertools.permutations(thumbs, 2):
        if case_of(a) == case_of(b):        # 同一案例的不同帧（常是同一静止镜头）不算负例
            continue
        if float(np.abs(small[a] - small[b]).mean()) < 4:     # 数据里同一张图存了两份：不算负例
            dup_pairs += 1
            continue
        d = tv.distances(queries[a], thumbs[b])
        neg.append(d)
        if tv.verdict(d) != "different":
            wrong.append((a, b, tv.verdict(d), d))
    # 难负例：同一视频的不同帧（相隔几秒）
    hard, hard_dups = [], 0
    for run in sorted((ROOT / "experiments" / "evid" / "runs").glob("*/artifacts"))[:40]:
        frames = load(sorted(run.glob("*.jp*g")), 12)
        fps = [(n, tv.fingerprint(jpeg(im, 90)), tv.fingerprint(jpeg(resize_w(im, 120), 75)),
                np.asarray(im.resize((64, 36)).convert("L"), dtype=np.float32)) for n, im in frames]
        for (a, qa, _ta, sa), (b, _qb, tb, sb) in itertools.permutations(fps, 2):
            if sa.shape == sb.shape and float(np.abs(sa - sb).mean()) < 4:   # 同一帧存了两份：不算负例
                hard_dups += 1
                continue
            d = tv.distances(qa, tb)
            hard.append(d)
            if tv.verdict(d) != "different":
                wrong.append((f"{run.parent.name}/{a[:8]}", b[:8], tv.verdict(d), d))
    print(f"排除的重复对（像素几乎相同，算同一张图）：不同图 {dup_pairs}，同视频帧 {hard_dups}")
    print(f"阈值：same 需 dHash ≤ {tv.SAME_DHASH} 且 pHash ≤ {tv.SAME_PHASH}；"
          f"similar 需 dHash ≤ {tv.SIMILAR_DHASH} 且 pHash ≤ {tv.SIMILAR_PHASH}\n")
    rows = [*((f"正例 {k}", v) for k, v in pos.items()), ("负例 不同的图", neg), ("难负例 同视频不同帧", hard)]
    for label, ds in rows:
        if not ds:
            continue
        verdicts = Counter(tv.verdict(d) for d in ds)
        print(f"{label}")
        print(f"   dHash {q([d['dhash'] for d in ds])}")
        print(f"   pHash {q([d['phash'] for d in ds])}")
        print(f"   判定 {dict(verdicts)}")
    print("\n负例里没判成 different 的（逐条看是不是其实是同一张图的不同裁切）：")
    for row in wrong:
        print("  ", *row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
