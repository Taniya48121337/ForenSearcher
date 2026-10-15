"""DETECT 开发探针（10-10）：在 EVID 的 18 个 AI 生成样本上，离线量几种检测方法到底有没有信号。双击 scripts\\_detect_dev_probe.cmd。

负责人决定直接在 EVID 上做 DETECT（这 18 个样本降级为开发集，最终报数另报其余 53 个干净样本）。不跑 agent，只做三件事：
  1. 用 GT 给的 YouTube 原片做"标签"：伪造版每 0.5 s 一帧与原片逐帧匹配（灰度缩略图 NCC），
     匹配不上的 = 插入片段；匹配上的再做配准差分（局部差异 = 物体被改）和人脸身份比对（换人 = 换脸）。
     这一步同时就是"找到原片后怎么比对"这条方法本身的离线验收。
  2. 不看原片的检测器给伪造版每秒一帧打分，同时给对齐的原片帧打分（配对的真实参照）：
       univfd   CLIP ViT-L/14 + 线性头（Ojha CVPR'23，ProGAN 训练）
       npr      NPR ResNet（Tan CVPR'24，ForenSynths 4 类）
       safe     SAFE ResNet + DWT（Li KDD'25）
       commfor  Community Forensics ViT-S/384（Park CVPR'25，4800+ 生成器训练）
       dfdc_b7  DFDC 冠军换脸检测器，逐张人脸
  3. 汇总：插入帧 vs 原片里有的帧（同一视频内）、被改帧 vs 未改帧、换脸的人 vs 没换的人，各检测器的 AUROC。

输出：outputs/detect/dev_probe/{<样本>.json, timeline_<样本>.jpg, diff_<样本>.jpg, faces_<样本>.jpg, summary.md, log.txt}
权重：.models/aigi/（CLIP-L/14 与 CommFor 走 Hugging Face；NPR / SAFE / UnivFD 头随脚本放好），换脸用 .models/deepfake/。
断点续跑：已有 <样本>.json 的跳过；加 --redo 重算。--only 152 160 只跑指定样本。--no-detectors 只做对齐。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
DF_MODELS = ROOT / ".models" / "deepfake"
AIGI = ROOT / ".models" / "aigi"
PYLIB = DF_MODELS / "pylib"
ORIGINALS = DF_MODELS / "originals"
OUT = ROOT / "outputs" / "detect" / "dev_probe"
DATASET = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench"
VIDEOS = DATASET / "videos"

DEV = {"152": "insert", "155": "insert", "160": "insert", "163": "insert", "167": "insert", "171": "insert",
       "201": "edit", "205": "edit", "210": "edit", "213": "edit", "216": "edit", "220": "edit",
       "175": "swap", "177": "swap", "178": "swap", "181": "swap", "184": "swap", "187": "swap"}
THUMB_STEP = 0.5          # 对齐用缩略图步长（秒）
MAX_SCORED = 420          # 每个伪造视频最多打分帧数（约每秒一帧）
HF_ENDPOINTS = [None, "https://hf-mirror.com"]


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%H:%M:%S')} {text}\n")


# ----------------------------------------------------------------------------- setup

def ensure_packages() -> None:
    PYLIB.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PYLIB))
    need = []
    for module, package in (("facenet_pytorch", "facenet-pytorch==2.6.0"), ("timm", "timm==1.0.15"),
                            ("pywt", "PyWavelets"), ("pytorch_wavelets", "pytorch_wavelets==1.3.0")):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            need.append(package)
    if need:
        log(f"pip --target 安装（不动本机环境）：{need}")
        nodeps = [p for p in need if not p.startswith("PyWavelets")]
        if "PyWavelets" in need:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", str(PYLIB), "PyWavelets"])
        if nodeps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(PYLIB), *nodeps])


def hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    last = None
    for endpoint in HF_ENDPOINTS:
        try:
            kwargs = {"endpoint": endpoint} if endpoint else {}
            return Path(hf_hub_download(repo, filename, local_dir=str(AIGI / repo.replace("/", "__")), **kwargs))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  HF {repo}/{filename} 经 {endpoint or '默认端点'} 失败：{type(exc).__name__}: {str(exc)[:160]}")
    raise RuntimeError(f"下载不到 {repo}/{filename}: {last}")


def fetch_original(video_id: str) -> Path | None:
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    hits = [p for p in ORIGINALS.glob(f"{video_id}.*") if not p.name.endswith((".part", ".ytdl"))]
    if hits:
        return hits[0]
    import yt_dlp

    for client in (None, "tv", "web_safari", "mweb", "ios", "android"):
        options = {"outtmpl": str(ORIGINALS / "%(id)s.%(ext)s"), "quiet": True, "noprogress": True,
                   "format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b"}
        if client:
            options["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            break
        except Exception as exc:  # noqa: BLE001
            log(f"!!! 原片 {video_id} 下载失败（客户端 {client or '默认'}）：{type(exc).__name__}: {str(exc)[:160]}")
    else:
        return None
    hits = [p for p in ORIGINALS.glob(f"{video_id}.*") if not p.name.endswith((".part", ".ytdl"))]
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- frame-level AIGI detectors

class UnivFD:
    name = "univfd"

    def __init__(self, device: str):
        import torch
        from transformers import CLIPVisionModelWithProjection

        for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = hf_file("openai/clip-vit-large-patch14", f)
        self.model = CLIPVisionModelWithProjection.from_pretrained(str(path.parent), torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        head = torch.load(str(AIGI / "univfd_fc_weights.pth"), map_location="cpu")
        self.fc = torch.nn.Linear(768, 1)
        self.fc.load_state_dict(head)
        self.fc = self.fc.to(device).float().eval()
        self.device = device
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def score(self, frames):
        import cv2
        import numpy as np
        import torch

        batch = []
        for f in frames:
            h, w = f.shape[:2]
            k = 224 / min(h, w)
            r = cv2.resize(f, (max(224, round(w * k)), max(224, round(h * k))), interpolation=cv2.INTER_CUBIC)
            y, x = (r.shape[0] - 224) // 2, (r.shape[1] - 224) // 2
            batch.append(r[y:y + 224, x:x + 224])
        with torch.inference_mode():
            x = torch.tensor(np.stack(batch), device=self.device).permute(0, 3, 1, 2).float() / 255
            x = ((x - self.mean) / self.std).to(self.model.dtype)
            emb = self.model(pixel_values=x).image_embeds.float()
            return torch.sigmoid(self.fc(emb)).squeeze(1).cpu().numpy().tolist()


class NPR:
    name = "npr"

    def __init__(self, device: str):
        import torch
        from _detect_dev_nets import npr_resnet50

        self.net = npr_resnet50()
        state = torch.load(str(AIGI / "NPR.pth"), map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in state.get("model", state).items()}
        self.net.load_state_dict(state, strict=True)
        self.net = self.net.to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def score(self, frames):
        import torch

        out = []
        with torch.inference_mode():
            for f in frames:   # 原图不缩放不裁（官方 test.py 的 no_resize / no_crop）；宽高截成偶数
                h, w = f.shape[:2]
                f = f[:h - h % 2, :w - w % 2]
                x = torch.tensor(f, device=self.device).permute(2, 0, 1)[None].float() / 255
                out.append(float(torch.sigmoid(self.net((x - self.mean) / self.std)).item()))
        return out


class SAFE:
    name = "safe"

    def __init__(self, device: str):
        import torch
        from _detect_dev_nets import safe_resnet50

        self.net = safe_resnet50()
        state = torch.load(str(AIGI / "SAFE_checkpoint-best.pth"), map_location="cpu", weights_only=False)
        self.net.load_state_dict(state.get("model", state), strict=True)
        self.net = self.net.to(device).eval()
        self.device = device

    def score(self, frames):
        import numpy as np
        import torch

        out = []
        with torch.inference_mode():
            for f in frames:   # 官方 eval：原分辨率 CenterCrop 256；视频帧大，取 2×3 网格 6 个 256 块求均值
                h, w = f.shape[:2]
                ys = [max(0, (h - 256) // 4), max(0, 3 * (h - 256) // 4)]
                xs = [max(0, (w - 256) // 6), max(0, (w - 256) // 2), max(0, 5 * (w - 256) // 6)]
                crops = np.stack([f[y:y + 256, x:x + 256] for y in ys for x in xs])
                x = torch.tensor(crops, device=self.device).permute(0, 3, 1, 2).float() / 255
                out.append(float(self.net(x).softmax(1)[:, 1].mean().item()))
        return out


class CommFor:
    name = "commfor"

    def __init__(self, device: str):
        import timm
        import torch
        from safetensors.torch import load_file

        path = hf_file("OwensLab/commfor-model-384", "model.safetensors")
        self.net = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        self.net.head = torch.nn.Linear(384, 1)
        state = {k.removeprefix("vit."): v for k, v in load_file(str(path)).items()}
        self.net.load_state_dict(state, strict=True)
        self.net = self.net.to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def score(self, frames):
        import cv2
        import numpy as np
        import torch

        batch = []
        for f in frames:   # 官方 test：短边缩到 440，CenterCrop 384
            h, w = f.shape[:2]
            k = 440 / min(h, w)
            r = cv2.resize(f, (max(440, round(w * k)), max(440, round(h * k))), interpolation=cv2.INTER_LINEAR)
            y, x = (r.shape[0] - 384) // 2, (r.shape[1] - 384) // 2
            batch.append(r[y:y + 384, x:x + 384])
        with torch.inference_mode():
            x = torch.tensor(np.stack(batch), device=self.device).permute(0, 3, 1, 2).float() / 255
            return torch.sigmoid(self.net((x - self.mean) / self.std)).squeeze(1).cpu().numpy().tolist()


# ----------------------------------------------------------------------------- faces

class FaceKit:
    """MTCNN 找脸 + FaceNet(VGGFace2) 身份向量 + DFDC B7 换脸分。"""

    def __init__(self, device: str):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        from _deepfake_probe import DFDC_FILES, DFDCWinner

        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
        self.embed = InceptionResnetV1(pretrained="vggface2").to(device).eval()
        self.dfdc = DFDCWinner([DF_MODELS / "dfdc" / n for n in DFDC_FILES], device)
        self.device = device
        self.torch = torch

    def detect(self, frame):
        from PIL import Image

        boxes, probs = self.mtcnn.detect(Image.fromarray(frame))
        if boxes is None:
            return []
        return [tuple(float(v) for v in b) for b, p in zip(boxes, probs)
                if p is not None and p >= 0.9 and (b[2] - b[0]) >= 40]

    def identity(self, frame, boxes):
        import cv2
        import numpy as np

        if not boxes:
            return []
        crops = []
        H, W = frame.shape[:2]
        for x0, y0, x1, y1 in boxes:
            m = 0.15 * (x1 - x0)
            c = frame[max(0, int(y0 - m)):min(H, int(y1 + m)), max(0, int(x0 - m)):min(W, int(x1 + m))]
            crops.append(cv2.resize(c, (160, 160), interpolation=cv2.INTER_AREA))
        torch = self.torch
        with torch.inference_mode():
            x = (torch.tensor(np.stack(crops), device=self.device).permute(0, 3, 1, 2).float() - 127.5) / 128
            e = self.embed(x)
            e = e / e.norm(dim=1, keepdim=True)
        return e.cpu().numpy()

    def swap_scores(self, frame, boxes):
        crops = [self.dfdc.crop(frame, b) for b in boxes]
        ok = [i for i, c in enumerate(crops) if c.size]
        scores = self.dfdc.score([crops[i] for i in ok]) if ok else []
        out = [None] * len(boxes)
        for i, s in zip(ok, scores):
            out[i] = round(float(s), 4)
        return out


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------- decoding & alignment

def thumb(frame):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)], cv2.COLOR_RGB2GRAY)
    v = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype("float32").ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-3 else v * 0


def video_info(path: Path) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def decode_pass(path: Path, want_thumb: set[int], want_full: set[int], on_full=None):
    """顺序解码一遍；want_thumb 里的帧号出缩略图，want_full 里的帧号把 RGB 原图交给 on_full(idx, frame)。"""

    import cv2

    cap = cv2.VideoCapture(str(path))
    thumbs, smalls = {}, {}
    last = max(want_thumb | want_full) if (want_thumb or want_full) else -1
    idx = -1
    while idx < last:
        if not cap.grab():
            break
        idx += 1
        if idx not in want_thumb and idx not in want_full:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if idx in want_thumb:
            thumbs[idx] = thumb(rgb)
            smalls[idx] = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
        if idx in want_full and on_full is not None:
            on_full(idx, rgb)
    cap.release()
    return thumbs, smalls


def jpeg(frame, quality: int = 93) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def unjpeg(data: bytes):
    import cv2
    import numpy as np

    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def ncc_small(a, b) -> float:
    import cv2
    import numpy as np

    x = cv2.cvtColor(cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype("float32").ravel()
    y = cv2.cvtColor(cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype("float32").ravel()
    x -= x.mean()
    y -= y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / d) if d > 1e-3 else 0.0


def local_diff(fake, orig) -> dict:
    """把原片帧配准到伪造帧（ECC 仿射），算局部差异：面积占比与最大的几个差异框（归一化 x0,y0,x1,y1）。"""

    import cv2
    import numpy as np

    W = 640
    h, w = fake.shape[:2]
    H = round(h * W / w)
    a = cv2.resize(fake, (W, H), interpolation=cv2.INTER_AREA)
    b = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    registered = False
    try:
        _, warp = cv2.findTransformECC(cv2.GaussianBlur(ga, (5, 5), 0), cv2.GaussianBlur(gb, (5, 5), 0), warp,
                                       cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        registered = True
    except cv2.error:
        pass
    b = cv2.warpAffine(b, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    d = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)).max(axis=2)
    m = int(0.02 * min(W, H))
    d[:m], d[-m:], d[:, :m], d[:, -m:] = 0, 0, 0, 0
    noise = float(np.median(d))
    thr = float(min(60, max(24, 20 + 3 * noise)))
    mask = (d > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = sorted(((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)), reverse=True)
    boxes = []
    for area, i in comps[:3]:
        if area < 0.0008 * W * H:
            break
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        boxes.append([round(x / W, 3), round(y / H, 3), round((x + bw) / W, 3), round((y + bh) / H, 3), round(area / (W * H), 4)])
    return {"area": round(float(mask.mean()), 4), "noise": round(noise, 1), "thr": round(thr, 1),
            "boxes": boxes, "registered": registered, "scale": round(float(math.sqrt(abs(np.linalg.det(warp[:, :2])))), 3),
            "shift": [round(float(warp[0, 2]) / W, 3), round(float(warp[1, 2]) / H, 3)]}


# ----------------------------------------------------------------------------- per-sample

def process(sample: str, kind: str, source: str, gt: list[str], detectors: list, faces: FaceKit | None) -> dict:
    import numpy as np

    started = time.monotonic()
    fake_path = VIDEOS / f"video_{sample}.mp4"
    orig_path = fetch_original(source)
    f_fps, f_n, f_w, f_h = video_info(fake_path)
    f_dur = f_n / f_fps
    log(f"[{sample}] {kind}：伪造 {f_dur:.0f}s {f_w}x{f_h}；原片 {source} {'有' if orig_path else '无'}")

    # 伪造版：每 0.5 s 缩略图；每 ~1 s（最多 MAX_SCORED 帧）全图 → 检测器 + 存 JPEG
    t_thumb = np.arange(0, f_dur - 0.05, THUMB_STEP)
    f_thumb_idx = sorted({min(f_n - 1, int(round(t * f_fps))) for t in t_thumb})
    step = max(1.0, f_dur / MAX_SCORED)
    f_full_idx = sorted({min(f_n - 1, int(round(t * f_fps))) for t in np.arange(0.25, f_dur - 0.05, step)})
    fake_frames: dict[int, bytes] = {}
    fake_scores: dict[int, dict] = {}
    pending: list = []

    def flush(store_scores, items):
        if not items:
            return
        idxs, imgs = zip(*items)
        for det in detectors:
            try:
                vals = det.score(list(imgs))
            except Exception as exc:  # noqa: BLE001
                log(f"  !!! {det.name} 打分失败：{type(exc).__name__}: {exc}")
                vals = [None] * len(imgs)
            for i, v in zip(idxs, vals):
                store_scores.setdefault(i, {})[det.name] = None if v is None else round(float(v), 4)
        items.clear()

    def on_fake(idx, rgb):
        fake_frames[idx] = jpeg(rgb)
        pending.append((idx, rgb))
        if len(pending) >= 8:
            flush(fake_scores, pending)

    f_thumbs, f_small = decode_pass(fake_path, set(f_thumb_idx), set(f_full_idx), on_fake)
    flush(fake_scores, pending)
    log(f"  伪造版解码+打分 {len(f_full_idx)} 帧，{time.monotonic() - started:.0f}s")

    result = {"sample": sample, "kind": kind, "source": source, "gt": gt, "fake": {"fps": f_fps, "frames": f_n, "w": f_w, "h": f_h},
              "original": None, "thumbs": [], "frames": [], "detectors": [d.name for d in detectors]}
    o_match: dict[int, tuple[int, float]] = {}
    orig_best: dict[int, tuple[float, int, bytes]] = {}
    orig_scores: dict[int, dict] = {}
    if orig_path:
        o_fps, o_n, o_w, o_h = video_info(orig_path)
        result["original"] = {"path": orig_path.name, "fps": o_fps, "frames": o_n, "w": o_w, "h": o_h}
        o_thumb_idx = sorted({min(o_n - 1, int(round(t * o_fps))) for t in np.arange(0, o_n / o_fps - 0.05, THUMB_STEP / 2)})
        o_thumbs, _ = decode_pass(orig_path, set(o_thumb_idx), set())
        F = np.stack([f_thumbs[i] for i in f_thumb_idx if i in f_thumbs])
        f_keys = [i for i in f_thumb_idx if i in f_thumbs]
        o_keys = sorted(o_thumbs)
        O = np.stack([o_thumbs[i] for i in o_keys])
        S = F @ O.T
        best = S.argmax(1)
        for k, (fi, bi) in enumerate(zip(f_keys, best)):
            o_match[fi] = (o_keys[bi], float(S[k, bi]))
            result["thumbs"].append([round(fi / f_fps, 2), round(o_keys[bi] / o_fps, 2), round(float(S[k, bi]), 4)])
        log(f"  原片 {o_n / o_fps:.0f}s，缩略图对齐完成；NCC 中位 {float(np.median(S.max(1))):.3f}，{time.monotonic() - started:.0f}s")

        # 每个打分帧：取最近的缩略图匹配，在原片对应时刻 ±0.3 s 里找像素最像的一帧
        def nearest_thumb(fi):
            j = min(f_keys, key=lambda k: abs(k - fi))
            return o_match[j]

        windows: dict[int, list[int]] = {}
        for fi in f_full_idx:
            oi, s = nearest_thumb(fi)
            if s < 0.6:
                continue
            c = oi + int(round((fi - min(f_keys, key=lambda k: abs(k - fi))) / f_fps * o_fps))
            for oj in range(max(0, c - int(0.3 * o_fps)), min(o_n, c + int(0.3 * o_fps) + 1)):
                windows.setdefault(oj, []).append(fi)
        fake_cache = {}
        window_end: dict[int, int] = {}
        for oj, fis in windows.items():
            for fi in fis:
                window_end[fi] = max(window_end.get(fi, oj), oj)

        def on_orig(oj, rgb):
            for fi in windows.get(oj, []):
                if fi not in fake_cache:
                    fake_cache[fi] = unjpeg(fake_frames[fi])
                s = ncc_small(fake_cache[fi], rgb)
                if fi not in orig_best or s > orig_best[fi][0]:
                    orig_best[fi] = (s, oj, None if s < 0.5 else jpeg(rgb))
                if window_end[fi] <= oj:
                    fake_cache.pop(fi, None)

        decode_pass(orig_path, set(), set(windows), on_orig)
        for fi in [k for k, v in orig_best.items() if v[2] is None]:
            orig_best.pop(fi)
        items = [(fi, unjpeg(v[2])) for fi, v in sorted(orig_best.items())]
        for k in range(0, len(items), 8):
            flush(orig_scores, items[k:k + 8])
        log(f"  原片配对帧 {len(orig_best)}，{time.monotonic() - started:.0f}s")

    for fi in f_full_idx:
        row = {"t": round(fi / f_fps, 2), "idx": fi, "scores": fake_scores.get(fi, {})}
        if fi in orig_best:
            s, oj, data = orig_best[fi]
            row.update({"orig_t": round(oj / result["original"]["fps"], 2), "ncc": round(s, 4), "orig_scores": orig_scores.get(fi, {})})
            if s >= 0.75:
                row["diff"] = local_diff(unjpeg(fake_frames[fi]), unjpeg(data))
        result["frames"].append(row)

    if faces is not None:
        n_face = 0
        for row in result["frames"]:
            fi = row["idx"]
            if kind != "swap" and fi % 3:      # 非换脸样本只抽 1/3 的帧看脸（量误报用）
                continue
            fake = unjpeg(fake_frames[fi])
            fb = faces.detect(fake)
            if not fb:
                continue
            entry = {"boxes": [[round(v) for v in b] for b in fb], "dfdc": faces.swap_scores(fake, fb)}
            if fi in orig_best and row.get("ncc", 0) >= 0.75:
                orig = unjpeg(orig_best[fi][2])
                sx, sy = fake.shape[1] / orig.shape[1], fake.shape[0] / orig.shape[0]
                ob = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in faces.detect(orig)]
                ef, eo = faces.identity(fake, fb), faces.identity(orig, [(b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy) for b in ob])
                sims, odfdc = [], faces.swap_scores(orig, [(b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy) for b in ob]) if ob else []
                for k, b in enumerate(fb):
                    j = max(range(len(ob)), key=lambda j: iou(b, ob[j]), default=None)
                    if j is None or iou(b, ob[j]) < 0.3:
                        sims.append(None)
                        continue
                    sims.append({"sim": round(float(ef[k] @ eo[j]), 4), "orig_dfdc": odfdc[j] if j < len(odfdc) else None})
                entry["vs_orig"] = sims
            row["faces"] = entry
            n_face += 1
        log(f"  人脸 {n_face} 帧，{time.monotonic() - started:.0f}s")

    sheets(result, f_small, fake_frames, orig_best)
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def sheets(result: dict, f_small: dict, fake_frames: dict, orig_best: dict) -> None:
    """时间线缩略图（红框=原片里找不到，黄框=局部有差异）；差异最大的帧的 伪造|原片|差异 三联图。"""

    import cv2
    import numpy as np

    sample = result["sample"]
    fps = result["fake"]["fps"]
    by_t = {round(r[0] * 2) / 2: r for r in result["thumbs"]}
    rows = result["frames"]
    tiles = []
    stride = max(1, len(rows) // 240)
    for row in rows[::stride]:
        small = f_small.get(min(f_small, key=lambda k: abs(k - row["idx"]))) if f_small else None
        if small is None:
            continue
        tile = cv2.cvtColor(small, cv2.COLOR_RGB2BGR).copy()
        ncc = row.get("ncc")
        area = (row.get("diff") or {}).get("area", 0)
        color = (0, 0, 255) if (ncc is None or ncc < 0.75) else ((0, 220, 255) if area > 0.003 else (0, 160, 0))
        cv2.rectangle(tile, (0, 0), (159, 89), color, 3)
        label = f"{row['t']:.0f}s " + ("--" if ncc is None else f"{ncc:.2f}")
        cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 16:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 16]) for i in range(0, len(tiles), 16)])
        cv2.imwrite(str(OUT / f"timeline_{sample}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    picks, used = [], []
    for row in sorted((r for r in rows if r.get("diff") and 0.002 < r["diff"]["area"] < 0.4), key=lambda r: -r["diff"]["area"]):
        if any(abs(row["t"] - u) < 4 for u in used):
            continue
        used.append(row["t"])
        picks.append(row)
        if len(picks) >= 10:
            break
    strips = []
    for row in sorted(picks, key=lambda r: r["t"]):
        fake = unjpeg(fake_frames[row["idx"]])
        orig = unjpeg(orig_best[row["idx"]][2])
        W = 480
        H = round(fake.shape[0] * W / fake.shape[1])
        a = cv2.resize(fake, (W, H))
        b = cv2.resize(orig, (W, H))
        ov = a.copy()
        for x0, y0, x1, y1, _ in row["diff"]["boxes"]:
            cv2.rectangle(ov, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (255, 0, 0), 2)
        strip = np.hstack([a, b, ov])
        cv2.putText(strip, f"{row['t']:.1f}s area {row['diff']['area']:.3f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        strips.append(cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    if strips:
        cv2.imwrite(str(OUT / f"diff_{sample}.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80])
    face_tiles = []
    for row in rows:
        fa = row.get("faces")
        if not fa or not fa.get("vs_orig"):
            continue
        fake = unjpeg(fake_frames[row["idx"]])
        orig = unjpeg(orig_best[row["idx"]][2])
        sx, sy = orig.shape[1] / fake.shape[1], orig.shape[0] / fake.shape[0]
        for b, v, d in zip(fa["boxes"], fa["vs_orig"], fa["dfdc"]):
            if not v:
                continue
            x0, y0, x1, y1 = b
            cf = fake[max(0, y0):y1, max(0, x0):x1]
            co = orig[max(0, int(y0 * sy)):int(y1 * sy), max(0, int(x0 * sx)):int(x1 * sx)]
            if not cf.size or not co.size:
                continue
            t = np.hstack([cv2.resize(cf, (112, 112)), cv2.resize(co, (112, 112))])
            cv2.putText(t, f"{row['t']:.0f}s s{v['sim']:.2f} d{(d or 0):.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
            face_tiles.append(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
    if face_tiles:
        face_tiles = face_tiles[:: max(1, len(face_tiles) // 48)][:48]
        while len(face_tiles) % 6:
            face_tiles.append(np.zeros_like(face_tiles[0]))
        cv2.imwrite(str(OUT / f"faces_{sample}.jpg"), np.vstack([np.hstack(face_tiles[i:i + 6]) for i in range(0, len(face_tiles), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------- summary

def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def summarize(results: dict) -> str:
    import numpy as np

    lines = ["# DETECT 开发探针：各方法在 18 个 EVID 开发样本上的信号", "",
             "标签来自与 GT 原片的对齐（NCC < 0.75 = 原片里没有；局部差异面积 > 0.3% = 被改）。AUROC 是同一视频内比较。", ""]
    dets = sorted({d for r in results.values() for d in r["detectors"]})
    lines += ["## 插入片段 / 被改帧：同一视频内 AUROC（正 = 插入或被改的帧，负 = 原片里有且没改的帧）", "",
              "| 样本 | 类型 | 帧 | 正 | " + " | ".join(dets) + " | 配对：伪造帧 > 原片同一帧 |", "|---|---|---|---|" + "---|" * len(dets) + "---|"]
    for s, r in sorted(results.items()):
        if r["kind"] == "swap":
            continue
        rows = r["frames"]
        pos = [x for x in rows if (x.get("ncc") is None or x["ncc"] < 0.75) if r["original"]] if r["kind"] == "insert" else \
              [x for x in rows if x.get("diff") and x["diff"]["area"] > 0.003]
        neg = [x for x in rows if x.get("diff") and x["diff"]["area"] <= 0.003]
        cells = []
        for d in dets:
            a = auroc([x["scores"].get(d) for x in pos if x["scores"].get(d) is not None],
                      [x["scores"].get(d) for x in neg if x["scores"].get(d) is not None])
            cells.append("—" if a is None else f"{a:.2f}")
        pair = []
        for d in dets:
            pr = [(x["scores"].get(d), x["orig_scores"].get(d)) for x in rows if x.get("orig_scores") and x["scores"].get(d) is not None
                  and x["orig_scores"].get(d) is not None and x.get("diff") and x["diff"]["area"] > 0.003]
            pair.append(f"{d} {np.mean([a > b for a, b in pr]):.2f}" if pr else f"{d} —")
        lines.append(f"| {s} | {r['kind']} | {len(rows)} | {len(pos)} | " + " | ".join(cells) + " | " + "；".join(pair) + " |")
    lines += ["", "## 换脸：与原片同位置人脸的身份相似度（FaceNet）与 DFDC 分", "",
              "| 样本 | 有原片对照的脸 | 身份相似度 中位 | 相似度 < 0.5 的比例 | DFDC 伪造脸 中位 | DFDC 原片脸 中位 |", "|---|---|---|---|---|---|"]
    for s, r in sorted(results.items()):
        sims, d_f, d_o = [], [], []
        for x in r["frames"]:
            fa = x.get("faces") or {}
            for v, d in zip(fa.get("vs_orig") or [], fa.get("dfdc") or []):
                if v:
                    sims.append(v["sim"])
                    if d is not None:
                        d_f.append(d)
                    if v.get("orig_dfdc") is not None:
                        d_o.append(v["orig_dfdc"])
        if not sims:
            continue
        lines.append(f"| {s} | {r['kind']} | {len(sims)} | {np.median(sims):.2f} | {np.mean([v < 0.5 for v in sims]):.2f} | "
                     f"{np.median(d_f) if d_f else float('nan'):.2f} | {np.median(d_o) if d_o else float('nan'):.2f} |")
    lines += ["", "## 对齐概况", "", "| 样本 | 类型 | 伪造时长 | 原片 | 打分帧 | 原片里找不到 | 局部差异帧 | 用时 |", "|---|---|---|---|---|---|---|---|"]
    for s, r in sorted(results.items()):
        rows = r["frames"]
        un = sum(1 for x in rows if x.get("ncc") is None or x["ncc"] < 0.75) if r["original"] else None
        ed = sum(1 for x in rows if x.get("diff") and x["diff"]["area"] > 0.003)
        lines.append(f"| {s} | {r['kind']} | {r['fake']['frames'] / r['fake']['fps']:.0f}s | {'有' if r['original'] else '无'} | {len(rows)} | "
                     f"{'—' if un is None else un} | {ed} | {r.get('seconds', 0):.0f}s |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--no-detectors", action="store_true")
    parser.add_argument("--no-faces", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    log("==== DETECT 开发探针 ====")
    ensure_packages()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备 {device}；torch {torch.__version__}")
    entries = {e["id"].removeprefix("video_"): e for e in json.loads((DATASET / "evid_bench.json").read_text(encoding="utf-8"))}
    todo = [s for s in (args.only or DEV) if s in DEV]
    todo = [s for s in todo if args.redo or not (OUT / f"{s}.json").is_file()]
    log(f"待跑 {len(todo)} 个：{todo}")
    detectors = []
    if not args.no_detectors and todo:
        for cls in (CommFor, UnivFD, NPR, SAFE):
            try:
                detectors.append(cls(device))
                log(f"已加载 {cls.name}")
            except Exception as exc:  # noqa: BLE001
                log(f"!!! {cls.name} 加载失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")
    faces = None
    if not args.no_faces and todo:
        try:
            faces = FaceKit(device)
            log("已加载 MTCNN + FaceNet + DFDC B7")
        except Exception as exc:  # noqa: BLE001
            log(f"!!! 人脸模块加载失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}")
    for sample in todo:
        e = entries[f"{sample}"] if sample in entries else entries[f"{int(sample):03d}"]
        try:
            result = process(sample, DEV[sample], e["source"][0], e["groundtruth"], detectors, faces)
        except Exception as exc:  # noqa: BLE001
            log(f"!!! {sample} 失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")
            continue
        (OUT / f"{sample}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        log(f"[{sample}] 完成，{result['seconds']:.0f}s")
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("[0-9][0-9][0-9].json"))}
    (OUT / "summary.md").write_text(summarize(results), encoding="utf-8")
    log("完成：" + str(OUT / "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
