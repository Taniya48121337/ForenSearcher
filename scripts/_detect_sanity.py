"""检测器 / 以图搜图的阳性对照（10-12）：双击 scripts\\_detect_sanity.cmd。

负责人问：效果不好的那些，是不是也有代码问题？之前的实测都只在 EVID 上跑，没有一个检测器先在"它本来就该认得出"的样本上验过，
所以分不清是方法不行还是我们接错了。这里给每个工具一个阳性对照：

  [images] AI 生成图检测器（UnivFD / NPR / SAFE / CommFor，即 10-10 的 _detect_dev_probe 实现；DDA、Effort 即 _detect_lit_probe 实现）
           在公开数据集 julienlucas/midjourney-dalle-sd-dataset 的 test 集上（Midjourney / DALL-E / SD 生成图 vs 真实照片，各取 100 张）
           打分，原图一遍、再压成 JPEG q75 一遍（模拟视频压缩）。代码没问题的话，原图上应明显 > 0.5 的 AUROC。
  [trufor] TruFor 官方仓库自带的 2 张篡改图 + 2 张原图；再加官方的 CocoGlide（GLIDE 生成式局部重绘，正对应"局部 AI 改动"）
           取 60 张，算掩码 F1（阈值 0.5，论文约 0.52）与峰值落在掩码内的比例。
  [yandex] SEARCH 的 Yandex 以图搜图：拿 GT 原片自己的帧、以及原片在 YouTube 上的封面图去搜，看能不能搜回原片；
           同时比较 Provider 现在的 User-Agent（ForenSearcher/0.1）和浏览器 User-Agent。

Skyra 的阳性对照等 KV 缓存修好、冒烟通过后再加。换脸检测器（SBI / Xception / DFDC）没有可直接下载的公开阳性样本，这里不含。
结果：outputs\\detect\\sanity\\summary.md；只跑某几段：--stages images yandex
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
import _detect_lit_probe as lit  # noqa: E402

OUT = ROOT / "outputs" / "detect" / "sanity"
DATA = ROOT / ".models" / "lit" / "sanity"
IMG_REPO = "julienlucas/midjourney-dalle-sd-dataset"
IMG_FILE = "data/test-00000-of-00001.parquet"
COCOGLIDE_URL = "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")
N_PER_CLASS = 100


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%m-%d %H:%M:%S')} {text}\n")


def auroc(pos, neg):
    return lit.auroc(pos, neg)


def fmt(v, d=2):
    return lit.fmt(v, d)


def jpeg(rgb, quality: int):
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ============================================================================ images

def load_image_set() -> tuple[list, list, dict]:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    path = _hf_dataset_file()
    table = pq.read_table(str(path))
    meta = table.schema.metadata or {}
    names = None
    try:
        info = json.loads(meta.get(b"huggingface", b"{}"))
        names = info["info"]["features"]["label"]["names"]
    except Exception:  # noqa: BLE001
        pass
    labels = table.column("label").to_pylist()
    images = table.column("image").to_pylist()
    if names is None:
        names = sorted({str(x) for x in labels}) if isinstance(labels[0], str) else ["fake", "real"]
    as_name = (lambda v: v) if isinstance(labels[0], str) else (lambda v: names[int(v)])
    idx = {"fake": [], "real": []}
    for i, v in enumerate(labels):
        name = str(as_name(v)).lower()
        if name in idx:
            idx[name].append(i)
    rng = random.Random(20261012)
    picked = {k: rng.sample(v, min(N_PER_CLASS, len(v))) for k, v in idx.items()}

    def decode(i):
        item = images[i]
        data = item.get("bytes") if isinstance(item, dict) else item
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    fake = [decode(i) for i in picked["fake"]]
    real = [decode(i) for i in picked["real"]]
    sizes = [f.shape[:2] for f in fake + real]
    info = {"label_names": names, "n_total": {k: len(v) for k, v in idx.items()},
            "median_hw": [int(np.median([s[0] for s in sizes])), int(np.median([s[1] for s in sizes]))]}
    return fake, real, info


def _hf_dataset_file() -> Path:
    from huggingface_hub import hf_hub_download

    last = None
    for endpoint in lit.HF_ENDPOINTS:
        try:
            kwargs = {"endpoint": endpoint} if endpoint else {}
            return Path(hf_hub_download(IMG_REPO, IMG_FILE, repo_type="dataset", local_dir=str(DATA), **kwargs))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  HF {IMG_REPO} 经 {endpoint or '默认端点'} 失败：{type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(f"下载不到 {IMG_REPO}: {last}")


def stage_images(lines: list[str]) -> None:
    import torch

    import _detect_dev_probe as probe

    probe.ensure_packages()
    lit.use_pylibs()
    device = lit.ensure_gpu(8)
    fake, real, info = load_image_set()
    log(f"图片阳性对照：fake {len(fake)}，real {len(real)}；标签 {info}")
    detectors = []
    for cls in (probe.UnivFD, probe.NPR, probe.SAFE, probe.CommFor):
        try:
            det = cls(device)
            detectors.append((det.name, lambda ims, d=det: d.score(ims)))
        except Exception as exc:  # noqa: BLE001
            log(f"!!! {cls.__name__} 加载失败：{exc}")
    try:
        dda = lit.DDA(device)

        def dda_center(ims):
            out = []
            for im in ims:
                H, W = im.shape[:2]
                if min(H, W) >= 336:
                    crop = im[(H - 336) // 2:(H - 336) // 2 + 336, (W - 336) // 2:(W - 336) // 2 + 336]
                else:
                    crop = lit.tiles(im, 336)[0][2]
                out += dda.score_crops([crop])
            return out

        detectors.append(("dda", dda_center))
    except Exception as exc:  # noqa: BLE001
        log(f"!!! DDA 加载失败：{exc}")
    for name in ("genimage", "chameleon"):
        if lit.EFFORT[name][1].is_file():
            try:
                eff = lit.Effort(lit.EFFORT[name][1], device)
                detectors.append((f"effort_{name}", lambda ims, e=eff: e.score_images(ims)))
            except Exception as exc:  # noqa: BLE001
                log(f"!!! Effort {name} 加载失败：{exc}")
        else:
            log(f"Effort {name} 权重还没下载下来，跳过")
    rows = []
    for variant, f in (("原图", lambda x: x), ("JPEG q75", lambda x: jpeg(x, 75))):
        fk, rl = [f(x) for x in fake], [f(x) for x in real]
        for name, score in detectors:
            try:
                sf, sr = [], []
                for k in range(0, len(fk), 8):
                    sf += score(fk[k:k + 8])
                    sr += score(rl[k:k + 8])
            except Exception as exc:  # noqa: BLE001
                log(f"!!! {name} 打分失败：{exc}\n{traceback.format_exc()[-600:]}")
                continue
            acc = (sum(s > 0.5 for s in sf) + sum(s <= 0.5 for s in sr)) / (len(sf) + len(sr))
            rows.append((variant, name, auroc(sf, sr), acc, lit.median(sf), lit.median(sr)))
            log(f"  {variant} {name}：AUROC {fmt(auroc(sf, sr))}，acc@0.5 {acc:.2f}，中位 fake {lit.median(sf):.3f} / real {lit.median(sr):.3f}")
        torch.cuda.empty_cache()
    lines += ["## AI 生成图检测器：公开数据集上的阳性对照", "",
              f"数据：{IMG_REPO} 的 test 集（Midjourney / DALL-E / SD vs 真实照片），各随机 {N_PER_CLASS} 张；中位尺寸 {info['median_hw']}；"
              f"标签名 {info['label_names']}。", "",
              "| 输入 | 检测器 | AUROC | 准确率@0.5 | fake 中位分 | real 中位分 |", "|---|---|---|---|---|---|"]
    for variant, name, a, acc, mf, mr in rows:
        lines.append(f"| {variant} | {name} | {fmt(a)} | {acc:.2f} | {fmt(mf, 3)} | {fmt(mr, 3)} |")
    lines.append("")


# ============================================================================ trufor

def stage_trufor(lines: list[str]) -> None:
    import cv2
    import numpy as np
    import torch

    lit.use_pylibs()
    device = lit.ensure_gpu(6)
    sys.path.insert(0, str(lit.TRUFOR_CODE))
    from config import _C as config

    config.defrost()
    config.merge_from_file(str(lit.TRUFOR_CODE / "trufor.yaml"))
    config.TEST.MODEL_FILE = str(lit.TRUFOR_W)
    config.freeze()
    from models.cmx.builder_np_conf import myEncoderDecoder

    model = myEncoderDecoder(cfg=config)
    model.load_state_dict(torch.load(str(lit.TRUFOR_W), map_location="cpu", weights_only=False)["state_dict"])
    model = model.to(device).eval()

    def run(rgb):
        x = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float, device=device)[None] / 256.0
        with torch.no_grad():
            pred, conf, det, _ = model(x)
        heat = torch.softmax(pred[0], dim=0)[1].float().cpu().numpy()
        return float(torch.sigmoid(det).item()), heat

    lines += ["## TruFor：官方样例与 CocoGlide", "", "| 图 | score | 热力图最大值 | 热力图 > 0.5 的面积 |", "|---|---|---|---|"]
    sample_dir = lit.TRUFOR_CODE.parent / "images"
    for name in ("tampered1.png", "tampered2.png", "pristine1.jpg", "pristine2.jpg"):
        path = sample_dir / name
        if not path.is_file():
            lines.append(f"| {name} | 缺文件 | | |")
            continue
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        score, heat = run(rgb)
        cv2.imwrite(str(OUT / f"trufor_{path.stem}_heat.png"), (np.clip(heat, 0, 1) * 255).astype("uint8"))
        lines.append(f"| {name} | {score:.3f} | {heat.max():.3f} | {(heat > 0.5).mean():.3f} |")
        log(f"  TruFor {name}：score {score:.3f}")
    try:
        archive = lit.download(COCOGLIDE_URL, DATA / "CocoGlide.zip")
        lit.unzip_once(archive, DATA / "cocoglide", "CocoGlide")
        root = DATA / "cocoglide"
        files = [p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        masks = {p.stem.replace("_mask", "").replace("_gt", ""): p for p in files if "mask" in str(p).lower()}
        fakes = [p for p in files if "mask" not in str(p).lower() and p.stem in masks]
        log(f"  CocoGlide：{len(fakes)} 张有掩码的篡改图（共 {len(files)} 个图片文件）")
        rng = random.Random(1)
        f1s, hits, fracs, scores = [], [], [], []
        for p in rng.sample(fakes, min(60, len(fakes))):
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            m = cv2.imread(str(masks[p.stem]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            score, heat = run(rgb)
            pr = heat > 0.5
            tp = float((pr & m).sum())
            f1 = 2 * tp / max(1.0, float(pr.sum() + m.sum()))
            f1s.append(max(f1, 2 * float((~pr & m).sum()) / max(1.0, float((~pr).sum() + m.sum()))))   # 论文按 F1 取正反两向较大者
            yy, xx = np.unravel_index(int(heat.argmax()), heat.shape)
            hits.append(bool(m[yy, xx]))
            fracs.append(float(m.mean()))
            scores.append(score)
        lines += ["", f"- CocoGlide（{len(f1s)} 张）：掩码 F1 中位 {lit.median(f1s):.2f}、均值 {sum(f1s) / len(f1s):.2f}（论文 0.52 左右）；"
                  f"峰值落在掩码内 {sum(hits)}/{len(hits)}（掩码面积中位 {lit.median(fracs):.2f}）；score 中位 {lit.median(scores):.2f}", ""]
    except Exception as exc:  # noqa: BLE001
        lines += ["", f"- CocoGlide 没跑成：{type(exc).__name__}: {str(exc)[:200]}", ""]
        log(traceback.format_exc()[-800:])


# ============================================================================ yandex

def stage_yandex(lines: list[str]) -> None:
    import cv2
    import httpx

    from foren_searcher.search.contracts import SearchQueryRequest
    from foren_searcher.search.providers.yandex_reverse import YandexReverseImageProvider

    queries = []
    for s in ["152", "155", "160", "163", "167", "171", "201", "205", "210", "213", "216", "220",
              "175", "177", "178", "181", "184", "187"]:
        try:
            d = lit.probe(s)
        except FileNotFoundError:
            continue
        source = d["source"]
        orig = lit.ORIGINALS / (d.get("original") or {}).get("path", "")
        if orig.is_file():
            fps, n = lit.video_meta(orig)
            frames = lit.grab(orig, [int(n * p) for p in (0.25, 0.5, 0.75)])
            for idx, rgb in sorted(frames.items()):
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                queries.append((s, source, f"原片帧 {idx / fps:.0f}s", buf.tobytes()))
        try:
            req = urllib.request.Request(f"https://i.ytimg.com/vi/{source}/hqdefault.jpg", headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                queries.append((s, source, "YouTube 封面", r.read()))
        except Exception as exc:  # noqa: BLE001
            log(f"  {s} 封面下载失败：{exc}")
    log(f"Yandex 阳性对照：{len(queries)} 个查询")

    async def search(jpeg_bytes, ua):
        client = httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False, headers={"User-Agent": ua})
        provider = YandexReverseImageProvider(30.0, client=client)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "q.jpg"
                path.write_bytes(jpeg_bytes)
                batch = await provider.search(SearchQueryRequest("image", "E1", "video"), offset=0, limit=50, image_path=path)
            return [c.source_url for c in batch.candidates]
        finally:
            await client.aclose()

    rows = []
    for s, source, kind, data in queries:
        row = {"sample": s, "kind": kind}
        for label, ua in (("ForenSearcher UA", "ForenSearcher/0.1"), ("浏览器 UA", BROWSER_UA)):
            try:
                urls = asyncio.run(search(data, ua))
                rank = next((i for i, u in enumerate(urls, 1) if source in u), None)
                yt = sum("youtube.com" in u or "youtu.be" in u for u in urls)
                row[label] = {"n": len(urls), "rank": rank, "youtube": yt,
                              "top": [u.split("/")[2] if "//" in u else u for u in urls[:5]]}
            except Exception as exc:  # noqa: BLE001
                row[label] = {"error": f"{type(exc).__name__}: {str(getattr(exc, 'error', exc))[:120]}"}
            time.sleep(2)
        rows.append(row)
        log(f"  {s} {kind}：" + "；".join(f"{k} 名次 {v.get('rank')} / YouTube {v.get('youtube')} / 共 {v.get('n')}"
                                         if "error" not in v else f"{k} 失败 {v['error']}"
                                         for k, v in row.items() if isinstance(v, dict)))
    (OUT / "yandex.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines += ["## Yandex 以图搜图：用原片自己的帧和封面去搜", "",
              "| 查询 | UA | 次数 | 搜回原片（任一名次） | 原片进前 10 | 结果里平均几个 YouTube 链接 | 失败 |", "|---|---|---|---|---|---|---|"]
    for kind in ("原片帧", "YouTube 封面"):
        for label in ("ForenSearcher UA", "浏览器 UA"):
            sel = [r[label] for r in rows if r["kind"].startswith(kind)]
            ok = [v for v in sel if "error" not in v]
            lines.append(f"| {kind} | {label} | {len(sel)} | {sum(v['rank'] is not None for v in ok)} | "
                         f"{sum(v['rank'] is not None and v['rank'] <= 10 for v in ok)} | "
                         f"{(sum(v['youtube'] for v in ok) / len(ok)) if ok else 0:.1f} | {len(sel) - len(ok)} |")
    tops = [r["ForenSearcher UA"].get("top") for r in rows if "top" in r.get("ForenSearcher UA", {})][:5]
    lines += ["", f"- 结果前 5 的域名举例：{tops}", ""]


# ============================================================================ main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=["images", "trufor", "yandex"])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    log(f"==== 阳性对照：{args.stages} ====")
    lines = [f"# 检测器与以图搜图的阳性对照（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             "目的：在工具本来就该认得出的样本上验证我们的接法；这里过不了，说明是我们的代码 / 预处理有问题，而不是方法在 EVID 上不行。", ""]
    for stage, fn in (("images", stage_images), ("trufor", stage_trufor), ("yandex", stage_yandex)):
        if stage not in args.stages:
            continue
        log(f"[{stage}] 开始")
        try:
            fn(lines)
        except SystemExit as exc:
            lines += [f"## {stage}", "", f"没跑成：{exc}", ""]
        except Exception as exc:  # noqa: BLE001
            lines += [f"## {stage}", "", f"出错：{type(exc).__name__}: {exc}", ""]
            log(traceback.format_exc()[-1500:])
        (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"完成：{OUT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
