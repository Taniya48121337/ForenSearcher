"""开源换脸检测器在 EVID 换脸样本上实测（10-10）：双击 scripts\\_deepfake_probe.cmd。

问的问题：在 FaceForensics++ 上训练的公开换脸检测器，能不能认出 EVID 的 FaceFusion 换脸？
不改 agent，只离线给视频打分：
  正样本  EVID 换脸样本 video_175 / 177 / 178 / 181 / 184 / 187（测试段；178、181 在 10-10 逐案诊断里已读过，
          另 4 个这次新读，统一列入待降级）。scripts/_faceswap_probe.py 是同一想法的早先草稿（未运行），本脚本取代它。
  配对负样本  上面 5 个的 YouTube 原片（同一批人、同一场景、没换脸）——最干净的对照
  开发段负样本  EVID 002–130 里人脸最多的 5 个真实画面视频（112 / 026 / 056 / 018 / 060）

检测器（权重第一次运行时下载到 .models/deepfake/，依赖用 pip --target 装到同目录，不动本机 Python 环境）：
  SBI  Self-Blended Images（CVPR 2022），EfficientNet-B4，FF++ c23 与 FF++ raw 两份权重（Google Drive）
  DFDC  DFDC 冠军方案（selimsef），tf_efficientnet_b7_ns ×2（GitHub release），DFDC 训练
  Xception  DeepfakeBench v1.0.1 发布的 xception_best.pth（FF++ c23 训练）
人脸：MTCNN（facenet-pytorch 自带权重）；每个视频均匀取 32 帧，每帧所有不小于最大脸一半的人脸都打分，
帧分 = 帧内最高，视频分 = 帧分均值（SBI 论文的视频级协议）。

输出：outputs/detect/deepfake_probe/{results.json, summary.md, faces_<视频>.jpg}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / ".models" / "deepfake"
PYLIB = MODELS / "pylib"
OUT = ROOT / "outputs" / "detect" / "deepfake_probe"
VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
ORIGINALS = MODELS / "originals"

POSITIVES = {"video_175": "XiCGwoTWJGA", "video_177": "raEjyMMuJbc", "video_178": "7uUxIqpwWQA",
             "video_181": "PPydi-ipMb4", "video_184": "Epmvg5fTbB0", "video_187": "AliSNrQrW0Y"}
DEV_NEGATIVES = ["video_112", "video_026", "video_056", "video_018", "video_060"]
SBI_WEIGHTS = {"sbi_ffc23": "1X0-NYT8KPursLZZdxduRQju6E52hauV0", "sbi_ffraw": "12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV"}
DFDC_TAG = "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/"
DFDC_FILES = ["final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
              "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31"]
DFB_RELEASE = "https://api.github.com/repos/SCLBD/DeepfakeBench/releases/tags/v1.0.1"
DFB_ZIP = "https://codeload.github.com/SCLBD/DeepfakeBench/zip/refs/heads/main"
N_FRAMES = 32


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
    for module, package in (("facenet_pytorch", "facenet-pytorch==2.6.0"), ("efficientnet_pytorch", "efficientnet_pytorch==0.7.1"),
                            ("gdown", "gdown"), ("timm", "timm==1.0.15")):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            need.append(package)
    if need:
        log(f"pip --target 安装（--no-deps，不动本机环境）：{need}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(PYLIB), *need])


def download(url: str, path: Path) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"下载 {url} → {path}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(str(path) + ".part", "wb") as stream:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            stream.write(chunk)
    os.replace(str(path) + ".part", path)
    return path


def fetch_sbi(name: str, drive_id: str) -> Path | None:
    path = MODELS / "sbi" / f"{name}.tar"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown

        gdown.download(id=drive_id, output=str(path), quiet=False)
    except Exception as exc:  # noqa: BLE001
        log(f"!!! SBI 权重 {name} 下载失败：{type(exc).__name__}: {exc}")
        return None
    return path if path.is_file() else None


def fetch_deepfakebench() -> tuple[Path | None, Path | None]:
    code = MODELS / "DeepfakeBench-main"
    try:
        if not (code / "training" / "networks" / "xception.py").is_file():
            archive = download(DFB_ZIP, MODELS / "DeepfakeBench-main.zip")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(MODELS)
        with urllib.request.urlopen(urllib.request.Request(DFB_RELEASE, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
            assets = {a["name"]: a["browser_download_url"] for a in json.loads(r.read().decode())["assets"]}
        log(f"DeepfakeBench v1.0.1 发布资产：{sorted(assets)}")
        name = next((n for n in assets if n.lower().startswith("xception") and n.endswith(".pth")), None)
        if name is None:
            log("!!! v1.0.1 里没有 xception 权重")
            return code, None
        return code, download(assets[name], MODELS / "dfb" / name)
    except Exception as exc:  # noqa: BLE001
        log(f"!!! DeepfakeBench 下载失败：{type(exc).__name__}: {exc}")
        return (code if code.is_dir() else None), None


def fetch_original(video_id: str) -> Path | None:
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    hits = list(ORIGINALS.glob(f"{video_id}.*"))
    if hits:
        return hits[0]
    try:
        import yt_dlp

        options = {"outtmpl": str(ORIGINALS / "%(id)s.%(ext)s"), "quiet": True, "noprogress": True,
                   "format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]/b"}
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as exc:  # noqa: BLE001
        log(f"!!! 原片 {video_id} 下载失败：{type(exc).__name__}: {str(exc)[:200]}")
        return None
    hits = [p for p in ORIGINALS.glob(f"{video_id}.*") if not p.name.endswith(".part")]
    return hits[0] if hits else None


# ----------------------------------------------------------------------------- detectors

class SBI:
    def __init__(self, name: str, weights: Path, device: str):
        import torch
        from efficientnet_pytorch import EfficientNet

        net = EfficientNet.from_name("efficientnet-b4", num_classes=2)
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = state.get("model", state)
        state = {k.removeprefix("net."): v for k, v in state.items()}
        result = net.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"SBI 权重缺键 {len(result.missing_keys)}：{result.missing_keys[:5]}")
        self.name, self.net, self.device = name, net.to(device).eval(), device

    @staticmethod
    def crop(frame, box):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        H, W = frame.shape[:2]
        # SBI crop_face 的测试相位：四边各加 1/8 框宽/高
        x0, x1 = max(0, int(x0 - w / 8)), min(W, int(x1 + w / 8) + 1)
        y0, y1 = max(0, int(y0 - h / 8)), min(H, int(y1 + h / 8) + 1)
        return frame[y0:y1, x0:x1]

    def score(self, crops):
        import cv2
        import numpy as np
        import torch

        batch = np.stack([cv2.resize(c, (380, 380), interpolation=cv2.INTER_LINEAR) for c in crops]).transpose(0, 3, 1, 2)
        with torch.inference_mode():
            x = torch.tensor(batch, device=self.device).float() / 255
            return self.net(x).softmax(1)[:, 1].float().cpu().numpy().tolist()


class DFDCWinner:
    """DFDC 冠军方案（selimsef）：tf_efficientnet_b7_ns 两个模型取均值；MTCNN 框外扩 1/3，等比缩到 380 居中补黑，ImageNet 归一化，sigmoid。"""

    def __init__(self, paths: list[Path], device: str):
        import timm
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = timm.create_model("tf_efficientnet_b7", pretrained=False, drop_path_rate=0.2)
                self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = torch.nn.Dropout(0.0)
                self.fc = torch.nn.Linear(2560, 1)

            def forward(self, x):
                return self.fc(self.dropout(self.avg_pool(self.encoder.forward_features(x)).flatten(1)))

        self.nets = []
        for path in paths:
            net = Net()
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            state = {k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items()}
            missing, unexpected = net.load_state_dict(state, strict=False)
            bad = [k for k in missing if not k.startswith("encoder.classifier")]
            if bad or [k for k in unexpected if not k.startswith("encoder.classifier")]:
                raise RuntimeError(f"DFDC 权重键不匹配：missing={bad[:5]} unexpected={unexpected[:5]}")
            self.nets.append(net.to(device).eval())
        self.name, self.device = "dfdc_b7", device

    @staticmethod
    def crop(frame, box):
        import cv2
        import numpy as np

        x0, y0, x1, y1 = (int(v) for v in box)
        pw, ph = (x1 - x0) // 3, (y1 - y0) // 3
        c = frame[max(y0 - ph, 0):y1 + ph, max(x0 - pw, 0):x1 + pw]
        if not c.size:
            return c
        h, w = c.shape[:2]
        k = 380 / max(h, w)
        c = cv2.resize(c, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        out = np.zeros((380, 380, 3), np.uint8)
        hh, ww = c.shape[:2]
        out[(380 - hh) // 2:(380 - hh) // 2 + hh, (380 - ww) // 2:(380 - ww) // 2 + ww] = c
        return out

    def score(self, crops):
        import numpy as np
        import torch

        x = torch.tensor(np.stack(crops)).permute(0, 3, 1, 2).float().to(self.device) / 255
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            x = (x - mean) / std
            return np.mean([torch.sigmoid(n(x)).squeeze(1).float().cpu().numpy() for n in self.nets], axis=0).tolist()


class DFBXception:
    def __init__(self, code: Path, weights: Path, device: str):
        import torch

        sys.path.insert(0, str(code / "training"))
        from networks.xception import Xception  # type: ignore

        net = Xception({"mode": "original", "num_classes": 2, "inc": 3, "dropout": False})
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()
                 if not k.startswith("loss_func")}
        result = net.load_state_dict(state, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"Xception 权重缺键 {len(result.missing_keys)}：{result.missing_keys[:5]}")
        self.name, self.net, self.device = "dfb_xception", net.to(device).eval(), device

    @staticmethod
    def crop(frame, box):
        # DeepfakeBench 预处理：dlib 人脸框放大 1.3 倍的正方形；这里用 MTCNN 框近似
        x0, y0, x1, y1 = box
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.3
        H, W = frame.shape[:2]
        a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        return frame[b:int(min(H, cy + side / 2)), a:int(min(W, cx + side / 2))]

    def score(self, crops):
        import cv2
        import numpy as np
        import torch

        batch = np.stack([cv2.resize(c, (256, 256), interpolation=cv2.INTER_LINEAR) for c in crops]).astype("float32") / 255
        batch = (batch - 0.5) / 0.5
        with torch.inference_mode():
            x = torch.tensor(batch.transpose(0, 3, 1, 2), device=self.device)
            logits = self.net.classifier(self.net.features(x))
            return logits.softmax(1)[:, 1].float().cpu().numpy().tolist()


# ----------------------------------------------------------------------------- faces & scoring

class Faces:
    def __init__(self, device: str):
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

    def __call__(self, frame):
        from PIL import Image

        boxes, probs = self.mtcnn.detect(Image.fromarray(frame))
        if boxes is None:
            return []
        keep = [(b, p) for b, p in zip(boxes, probs) if p is not None and p >= 0.9]
        if not keep:
            return []
        biggest = max((b[2] - b[0]) * (b[3] - b[1]) for b, _ in keep)
        return [tuple(float(v) for v in b) for b, _ in keep if (b[2] - b[0]) * (b[3] - b[1]) >= biggest / 2
                and (b[2] - b[0]) >= 40]


def sample_frames(path: Path, n: int = N_FRAMES):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    for index in np.linspace(0, max(0, total - 1), n, endpoint=True, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append((round(index / fps, 2), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def score_video(path: Path, faces: Faces, detectors: list) -> dict:
    import numpy as np

    started = time.monotonic()
    per_face = []
    frames = sample_frames(path)
    for t, frame in frames:
        boxes = faces(frame)
        if not boxes:
            continue
        row = {"t": t, "boxes": [[round(v) for v in b] for b in boxes], "px": [round(b[2] - b[0]) for b in boxes]}
        for det in detectors:
            crops = [det.crop(frame, b) for b in boxes]
            crops = [c for c in crops if c.size]
            row[det.name] = [round(s, 4) for s in det.score(crops)] if crops else []
        per_face.append(row)
    summary = {"frames": len(frames), "frames_with_face": len(per_face), "seconds": round(time.monotonic() - started, 1)}
    for det in detectors:
        frame_scores = [max(r[det.name]) for r in per_face if r.get(det.name)]
        all_scores = [s for r in per_face for s in r.get(det.name, [])]
        summary[det.name] = {"video_score": round(float(np.mean(frame_scores)), 4) if frame_scores else None,
                             "frames_over_0.5": round(float(np.mean([s > 0.5 for s in frame_scores])), 3) if frame_scores else None,
                             "faces": len(all_scores),
                             "face_median": round(float(np.median(all_scores)), 4) if all_scores else None}
    return {"summary": summary, "faces": per_face}


def face_sheet(path: Path, result: dict, detector: str, out: Path) -> None:
    """取分数最高与最低的若干张脸拼图，人工核对检测器看的是不是脸。"""

    import cv2
    import numpy as np

    rows = [(r["t"], b, s) for r in result["faces"] for b, s in zip(r["boxes"], r.get(detector, []))]
    if not rows:
        return
    rows.sort(key=lambda x: -x[2])
    pick = rows[:8] + rows[-8:] if len(rows) > 16 else rows
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tiles = []
    for t, box, s in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        x0, y0, x1, y1 = box
        crop = frame[max(0, y0):y1, max(0, x0):x1]
        if not crop.size:
            continue
        tile = cv2.resize(crop, (160, 160))
        cv2.putText(tile, f"{s:.2f} @{t:.0f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(str(out), grid)


# ----------------------------------------------------------------------------- main

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("==== 开源换脸检测器实测 ====")
    ensure_packages()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备 {device}；torch {torch.__version__}")
    detectors = []
    for name, drive_id in SBI_WEIGHTS.items():
        weights = fetch_sbi(name, drive_id)
        if weights:
            try:
                detectors.append(SBI(name, weights, device))
                log(f"已加载 {name}")
            except Exception as exc:  # noqa: BLE001
                log(f"!!! {name} 加载失败：{exc}")
    try:
        paths = [download(DFDC_TAG + name, MODELS / "dfdc" / name) for name in DFDC_FILES]
        detectors.append(DFDCWinner(paths, device))
        log("已加载 dfdc_b7")
    except Exception as exc:  # noqa: BLE001
        log(f"!!! dfdc_b7 不可用：{type(exc).__name__}: {exc}")
    code, xception = fetch_deepfakebench()
    if code and xception:
        try:
            detectors.append(DFBXception(code, xception, device))
            log("已加载 dfb_xception")
        except Exception as exc:  # noqa: BLE001
            log(f"!!! dfb_xception 加载失败：{type(exc).__name__}: {exc}")
    if not detectors:
        log("!!! 一个检测器都没加载成功，停")
        return 1
    faces = Faces(device)
    jobs = [(v, "swap", VIDEOS / f"{v}.mp4") for v in POSITIVES]
    for v, yid in POSITIVES.items():
        path = fetch_original(yid)
        if path:
            jobs.append((f"orig_{yid}", f"original_of_{v}", path))
    jobs += [(v, "dev_real", VIDEOS / f"{v}.mp4") for v in DEV_NEGATIVES]
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    for key, kind, path in jobs:
        if key in results and all(d.name in results[key]["summary"] for d in detectors):
            continue
        log(f"打分 {key}（{kind}）")
        try:
            result = score_video(path, faces, detectors)
        except Exception as exc:  # noqa: BLE001
            log(f"!!! {key} 失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")
            continue
        results[key] = {"kind": kind, "path": str(path), **result}
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        face_sheet(path, result, detectors[0].name, OUT / f"faces_{key}.jpg")
        log(f"  {key}: " + json.dumps(result["summary"], ensure_ascii=False))
    names = [d.name for d in detectors]
    lines = ["# 开源换脸检测器在 EVID 换脸样本上的实测", "",
             "视频分 = 32 帧里每帧最高人脸分的均值（>0.5 偏换脸）；括号里是帧分 > 0.5 的比例。", "",
             "| 视频 | 类别 | 有脸帧 | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for key, item in results.items():
        cells = []
        for name in names:
            s = item["summary"].get(name) or {}
            cells.append("—" if s.get("video_score") is None else f"{s['video_score']:.3f}（{100 * s['frames_over_0.5']:.0f}%）")
        lines.append(f"| {key} | {item['kind']} | {item['summary']['frames_with_face']}/{item['summary']['frames']} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("完成：" + str(OUT / "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
