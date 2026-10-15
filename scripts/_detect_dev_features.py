"""DETECT 开发探针第二步（10-10）：给开发样本逐帧抽特征，在本会话里做"留一视频"的小分类器验证。双击 scripts\\_detect_dev_features.cmd。

第一步（_detect_dev_probe）的结论：现成的 AI 图像检测器（UnivFD / NPR / SAFE / CommFor）对 EVID 里插入的 AI 片段
几乎全打 0 分，不能直接用。这一步改问：同样的骨干网络特征，用开发样本里"对齐得到的标签"训练一个线性头，
换一个没见过的视频还能不能分出 AI 帧？（留一视频交叉验证在本会话里算，这里只抽特征。）

每个开发样本，按第一步 <样本>.json 里的帧号取帧，抽：
  clip    CLIP ViT-L/14 image_embeds（768）
  dino    DINOv2-base CLS（768）
  commfor Community Forensics ViT-S pre-logits（384）
  npr     NPR 截断 ResNet 池化特征（512）
  safe    SAFE 截断 ResNet 池化特征（512，6 块均值）
  hand    手工低层统计（频谱能量分段、噪声残差、块效应、饱和度等，32 维）
  gray    96×54 灰度小图（离线看台标 / 水印是否连续）
另外：全帧率的镜头切分（HSV 直方图相关 < 0.6 记一刀），离线看"插入片段是否恰好 5 / 10 秒"。
再从 EVID 002–130 里取 12 个真实拍摄视频、每个 60 帧，作额外负样本。

输出：outputs/detect/dev_features/<样本>.npz、cuts.json、log.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _detect_dev_probe as probe  # noqa: E402

OUT = ROOT / "outputs" / "detect" / "dev_features"
EXTRA_REAL = ["004", "012", "020", "030", "040", "050", "064", "076", "088", "098", "110", "124"]


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%H:%M:%S')} {text}\n")


class Extractors:
    def __init__(self, device: str):
        import torch
        from transformers import AutoModel

        self.torch, self.device = torch, device
        self.univfd = probe.UnivFD(device)
        self.commfor = probe.CommFor(device)
        self.npr = probe.NPR(device)
        self.safe = probe.SAFE(device)
        path = None
        for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = probe.hf_file("facebook/dinov2-base", f)
        self.dino = AutoModel.from_pretrained(str(path.parent)).to(device).eval()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def clip(self, frames):
        import cv2
        import numpy as np

        u = self.univfd
        batch = []
        for f in frames:
            h, w = f.shape[:2]
            k = 224 / min(h, w)
            r = cv2.resize(f, (max(224, round(w * k)), max(224, round(h * k))), interpolation=cv2.INTER_CUBIC)
            y, x = (r.shape[0] - 224) // 2, (r.shape[1] - 224) // 2
            batch.append(r[y:y + 224, x:x + 224])
        x = self.torch.tensor(np.stack(batch), device=self.device).permute(0, 3, 1, 2).float() / 255
        x = ((x - u.mean) / u.std).to(u.model.dtype)
        return u.model(pixel_values=x).image_embeds.float().cpu().numpy()

    def dinov2(self, frames):
        import cv2
        import numpy as np

        batch = []
        for f in frames:
            h, w = f.shape[:2]
            k = 256 / min(h, w)
            r = cv2.resize(f, (max(256, round(w * k)), max(256, round(h * k))), interpolation=cv2.INTER_CUBIC)
            y, x = (r.shape[0] - 224) // 2, (r.shape[1] - 224) // 2
            batch.append(r[y:y + 224, x:x + 224])
        x = self.torch.tensor(np.stack(batch), device=self.device).permute(0, 3, 1, 2).float() / 255
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.dino(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy()

    def commfor_feat(self, frames):
        import cv2
        import numpy as np

        batch = []
        for f in frames:
            h, w = f.shape[:2]
            k = 440 / min(h, w)
            r = cv2.resize(f, (max(440, round(w * k)), max(440, round(h * k))), interpolation=cv2.INTER_LINEAR)
            y, x = (r.shape[0] - 384) // 2, (r.shape[1] - 384) // 2
            batch.append(r[y:y + 384, x:x + 384])
        c = self.commfor
        x = self.torch.tensor(np.stack(batch), device=self.device).permute(0, 3, 1, 2).float() / 255
        x = (x - c.mean) / c.std
        return c.net.forward_head(c.net.forward_features(x), pre_logits=True).float().cpu().numpy()

    def npr_feat(self, frames):
        import numpy as np

        net, out = self.npr.net, []
        for f in frames:
            h, w = f.shape[:2]
            f = f[:h - h % 2, :w - w % 2]
            x = self.torch.tensor(f, device=self.device).permute(2, 0, 1)[None].float() / 255
            x = (x - self.npr.mean) / self.npr.std
            x = (x - net._interp(x, 0.5)) * 2.0 / 3.0
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            out.append(net.avgpool(net.layer2(net.layer1(x))).flatten(1)[0].float().cpu().numpy())
        return np.stack(out)

    def safe_feat(self, frames):
        import numpy as np
        import torch.nn.functional as F

        from _detect_dev_nets import _shim_pkg_resources

        _shim_pkg_resources()
        from pytorch_wavelets import DWTForward

        net, out = self.safe.net, []
        if net._dwt is None:
            net._dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3").to(self.device)
        for f in frames:
            h, w = f.shape[:2]
            ys = [max(0, (h - 256) // 4), max(0, 3 * (h - 256) // 4)]
            xs = [max(0, (w - 256) // 6), max(0, (w - 256) // 2), max(0, 5 * (w - 256) // 6)]
            crops = np.stack([f[y:y + 256, x:x + 256] for y in ys for x in xs])
            x = self.torch.tensor(crops, device=self.device).permute(0, 3, 1, 2).float() / 255
            _, yh = net._dwt(x)
            x = F.interpolate(yh[0][:, :, 2, :, :], size=(256, 256), mode="bilinear", antialias=True)
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            out.append(net.avgpool(net.layer2(net.layer1(x))).flatten(1).mean(0).float().cpu().numpy())
        return np.stack(out)

    def all(self, frames) -> dict:
        import numpy as np

        with self.torch.inference_mode():
            feats = {"clip": self.clip(frames), "dino": self.dinov2(frames), "commfor": self.commfor_feat(frames),
                     "npr": self.npr_feat(frames), "safe": self.safe_feat(frames)}
        feats["hand"] = np.stack([hand(f) for f in frames])
        feats["gray"] = np.stack([gray_small(f) for f in frames])
        return feats


def gray_small(f):
    import cv2

    return cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (96, 54), interpolation=cv2.INTER_AREA)


def hand(f):
    """32 维低层统计：在 720p 等高下算，避免分辨率本身成为特征。"""

    import cv2
    import numpy as np

    h, w = f.shape[:2]
    if h != 720:
        f = cv2.resize(f, (round(w * 720 / h), 720), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = []
    # 频谱能量：中心 512×512 块，径向 10 段占比（对数）
    H, W = g.shape
    c = g[(H - 512) // 2:(H - 512) // 2 + 512, (W - 512) // 2:(W - 512) // 2 + 512] if H >= 512 and W >= 512 else cv2.resize(g, (512, 512))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(c - c.mean()))) ** 2
    yy, xx = np.mgrid[-256:256, -256:256]
    rad = np.sqrt(yy ** 2 + xx ** 2)
    total = spec.sum() + 1e-6
    for lo, hi in zip(np.linspace(0, 256, 11)[:-1], np.linspace(0, 256, 11)[1:]):
        feats.append(np.log(spec[(rad >= lo) & (rad < hi)].sum() / total + 1e-9))
    # 噪声残差：拉普拉斯、3×3 中值残差的 std 与峰度
    lap = cv2.Laplacian(g, cv2.CV_32F)
    med = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    for r in (lap, med):
        s = r.std() + 1e-6
        feats += [np.log(s), float(((r / s) ** 4).mean())]
    # 平坦区噪声（局部方差最低 20% 的块里的残差 std）
    var = cv2.blur(g * g, (8, 8)) - cv2.blur(g, (8, 8)) ** 2
    flat = var < np.percentile(var, 20)
    feats.append(np.log(med[flat].std() + 1e-6))
    # 块效应：8 像素格线与非格线的梯度比
    dx = np.abs(np.diff(g, axis=1))
    feats.append(float(dx[:, 7::8].mean() / (dx.mean() + 1e-6)))
    dy = np.abs(np.diff(g, axis=0))
    feats.append(float(dy[7::8].mean() / (dy.mean() + 1e-6)))
    # 清晰度：梯度幅值分位
    gm = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    feats += [np.log(np.percentile(gm, q) + 1) for q in (50, 90, 99)]
    # 颜色：饱和度均值 / std，亮度均值 / std，色度噪声
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
    feats += [hsv[..., 1].mean() / 255, hsv[..., 1].std() / 255, g.mean() / 255, g.std() / 255]
    ycc = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    for ch in (1, 2):
        feats.append(np.log((ycc[..., ch] - cv2.blur(ycc[..., ch], (3, 3))).std() + 1e-6))
    # 剪切 / 黑边比例、过曝 / 欠曝比例
    feats += [float((g < 8).mean()), float((g > 247).mean())]
    feats += [0.0] * (32 - len(feats))
    return np.asarray(feats[:32], dtype=np.float32)


def cuts_full(path: Path) -> dict:
    """全帧率镜头切分：96×54 HSV 直方图相关 < 0.6 记一刀。"""

    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, cuts, corr, idx = None, [], [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        ok, bgr = cap.retrieve()
        if not ok:
            continue
        hsv = cv2.cvtColor(cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev is not None:
            c = float(cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL))
            corr.append(round(c, 3))
            if c < 0.6:
                cuts.append(round(idx / fps, 3))
        prev = hist
    cap.release()
    return {"fps": fps, "frames": idx + 1, "cuts": cuts, "corr": corr}


def frames_at(path: Path, indices: list[int]):
    import cv2

    want, out = set(indices), {}
    cap = cv2.VideoCapture(str(path))
    idx, last = -1, max(indices)
    while idx < last:
        if not cap.grab():
            break
        idx += 1
        if idx in want:
            ok, bgr = cap.retrieve()
            if ok:
                out[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def run_video(ex: Extractors, path: Path, indices: list[int], fps: float, extra: dict) -> dict:
    import numpy as np

    frames = frames_at(path, indices)
    keys = [i for i in indices if i in frames]
    feats: dict[str, list] = {}
    for k in range(0, len(keys), 8):
        part = ex.all([frames[i] for i in keys[k:k + 8]])
        for name, value in part.items():
            feats.setdefault(name, []).append(value)
    out = {name: np.concatenate(v) for name, v in feats.items()}
    out["idx"] = np.asarray(keys)
    out["t"] = np.asarray(keys) / fps
    out.update(extra)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    log("==== DETECT 开发特征抽取 ====")
    probe.ensure_packages()
    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ex = Extractors(device)
    log(f"设备 {device}；特征提取器已加载")
    cuts_path = OUT / "cuts.json"
    cuts = json.loads(cuts_path.read_text(encoding="utf-8")) if cuts_path.is_file() else {}
    jobs = []
    for p in sorted(probe.OUT.glob("[0-9][0-9][0-9].json")):
        if args.only and p.stem not in args.only:
            continue
        jobs.append(("dev", p.stem))
    if not args.only:
        jobs += [("real", s) for s in EXTRA_REAL]
    for kind, sample in jobs:
        target = OUT / f"{sample}.npz"
        if target.is_file() and not args.redo:
            continue
        started = time.monotonic()
        path = probe.VIDEOS / f"video_{sample}.mp4"
        try:
            fps, n, _, _ = probe.video_info(path)
            if kind == "dev":
                r = json.loads((probe.OUT / f"{sample}.json").read_text(encoding="utf-8"))
                indices = [row["idx"] for row in r["frames"]]
                ncc = np.asarray([row.get("ncc", -1) if row.get("ncc") is not None else -1 for row in r["frames"]], dtype=np.float32)
                area = np.asarray([(row.get("diff") or {}).get("area", -1) for row in r["frames"]], dtype=np.float32)
                extra = {"ncc": ncc, "area": area, "kind": np.asarray(r["kind"]), "has_orig": np.asarray(bool(r["original"]))}
            else:
                indices = sorted({int(i) for i in np.linspace(int(fps), n - int(fps), 60)})
                extra = {"kind": np.asarray("real")}
            data = run_video(ex, path, indices, fps, extra)
            np.savez_compressed(target, **data)
            if sample not in cuts and kind == "dev":
                cuts[sample] = cuts_full(path)
                cuts_path.write_text(json.dumps(cuts), encoding="utf-8")
            log(f"[{sample}] {kind}：{len(data['idx'])} 帧，{time.monotonic() - started:.0f}s")
        except Exception as exc:  # noqa: BLE001
            log(f"!!! {sample} 失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1000:]}")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
