"""换脸检测器试测（10-10）：开源 SBI 与 DFDC 冠军方案，能不能认出 EVID 的 FaceFusion 换脸？

    python scripts/_faceswap_probe.py            # 下权重 → 下原片 → 逐视频打分 → outputs/detect/faceswap_probe/

检测器（只读官方权重，推理流程照官方仓库）：
  SBI   Self-Blended Images（CVPR 2022），EfficientNet-B4，FF++ c23 权重（Google Drive，gdown 下载）。
        官方推理：每段均匀取 32 帧，每帧取人脸框外扩 1/8，缩到 380×380，像素 /255，softmax 第 1 类 = fake；
        每帧取各人脸最大值，视频分 = 帧均值。官方用 RetinaFace 找脸，这里用 MTCNN（facenet-pytorch 自带权重）。
  DFDC  DFDC 冠军（selimsef），EfficientNet-B7（timm），GitHub release 权重 2 个。
        官方推理：MTCNN 找脸，外扩 1/3，等比缩放到 380 后居中补黑，ImageNet 归一化，sigmoid；
        视频分 = confident_strategy（t=0.8）。

样本：
  换脸（EVID identity_swap，测试段，读过后按规则降级）：175 177 178 181 184 187
  这 6 个的原片（YouTube，同一批人、同一场景，没换脸）：配对对照
  真实人脸（EVID 002–130 开发段，剪辑类、真实拍摄、人脸多）：112 026 056 110 060 018

输出：results.json（每段视频：两种检测器的视频分、逐脸分数）与 faces_<视频>.jpg（人脸裁块 + 分数，供人看是谁被判高）。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "detect" / "faceswap_probe"
WEIGHTS = ROOT / ".models" / "faceswap"
VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
ORIG = OUT / "originals"
SWAP = {"video_175": "XiCGwoTWJGA", "video_177": "raEjyMMuJbc", "video_178": "7uUxIqpwWQA",
        "video_181": "PPydi-ipMb4", "video_184": "Epmvg5fTbB0", "video_187": "AliSNrQrW0Y"}
REAL = ["video_112", "video_026", "video_056", "video_110", "video_060", "video_018"]
SBI_GDRIVE = "1X0-NYT8KPursLZZdxduRQju6E52hauV0"          # FF-c23
DFDC_TAG = "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/"
DFDC_FILES = ["final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
              "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31"]
N_FRAMES = 32


def say(text: str) -> None:
    print(text, flush=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def ensure_packages() -> None:
    import importlib

    try:
        importlib.import_module("torchvision")
    except ImportError as exc:     # 不自动装 torchvision：版本必须和本机 torch 配套
        raise SystemExit("本机没有 torchvision；请按 torch 版本装好再跑") from exc
    need = [pkg for mod, pkg in (("timm", "timm"), ("efficientnet_pytorch", "efficientnet_pytorch"),
                                 ("facenet_pytorch", "facenet-pytorch"), ("gdown", "gdown"))
            if importlib.util.find_spec(mod) is None]
    if need:
        say(f"安装：{need}（--no-deps，不动 torch）")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *need])
        if "gdown" in need:
            subprocess.call([sys.executable, "-m", "pip", "install", "filelock", "beautifulsoup4", "requests", "tqdm"])


def download(url: str, dest: Path) -> None:
    import urllib.request

    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return
    say(f"下载 {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_weights() -> None:
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    sbi = WEIGHTS / "sbi_FFc23.tar"
    if not sbi.is_file():
        import gdown

        say("下载 SBI 权重（Google Drive）")
        gdown.download(id=SBI_GDRIVE, output=str(sbi), quiet=False)
    for name in DFDC_FILES:
        download(DFDC_TAG + name, WEIGHTS / name)


def ensure_originals() -> dict[str, Path]:
    ORIG.mkdir(parents=True, exist_ok=True)
    found = {}
    for sample, yid in SWAP.items():
        target = ORIG / f"{yid}.mp4"
        if not target.is_file():
            say(f"下载原片 {yid}（{sample}）")
            code = subprocess.call([sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]",
                                    "--no-playlist", "-o", str(target), f"https://www.youtube.com/watch?v={yid}"])
            if code != 0 or not target.is_file():
                say(f"  原片 {yid} 下载失败（退出码 {code}），跳过")
                continue
        found[sample] = target
    return found


# ----------------------------------------------------------------------------- models

def load_models(device):
    import torch
    import timm
    from efficientnet_pytorch import EfficientNet

    class SBI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)

        def forward(self, x):
            return self.net(x)

    sbi = SBI()
    state = torch.load(WEIGHTS / "sbi_FFc23.tar", map_location="cpu", weights_only=False)["model"]
    sbi.load_state_dict(state)
    sbi = sbi.to(device).eval()

    class DFDC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model("tf_efficientnet_b7", pretrained=False, drop_path_rate=0.2)
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = torch.nn.Dropout(0.0)
            self.fc = torch.nn.Linear(2560, 1)

        def forward(self, x):
            x = self.encoder.forward_features(x)
            return self.fc(self.dropout(self.avg_pool(x).flatten(1)))

    dfdc = []
    for name in DFDC_FILES:
        model = DFDC()
        checkpoint = torch.load(WEIGHTS / name, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("encoder.classifier")]
        if bad or [k for k in unexpected if not k.startswith("encoder.classifier")]:
            raise RuntimeError(f"DFDC 权重键不匹配：missing={bad[:5]} unexpected={unexpected[:5]}")
        dfdc.append(model.to(device).eval())
    return sbi, dfdc


# ----------------------------------------------------------------------------- video → faces

def sample_frames(path: Path, n: int = N_FRAMES):
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs = set(int(i) for i in __import__("numpy").linspace(0, max(0, total - 1), n, endpoint=True))
    frames = []
    for idx in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def detect(mtcnn, rgb):
    import numpy as np
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = 0.5 if max(h, w) > 1000 else 1.0           # DFDC 方案：大图先缩一半找脸
    img = Image.fromarray(rgb).resize((int(w * scale), int(h * scale))) if scale != 1 else Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(img)
    out = []
    if boxes is None:
        return out
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.9:
            continue
        x0, y0, x1, y1 = (np.asarray(box) / scale).tolist()
        if min(x1 - x0, y1 - y0) < 40:
            continue
        out.append((x0, y0, x1, y1, float(prob)))
    return out


def crop_sbi(rgb, box):
    import cv2

    x0, y0, x1, y1 = box[:4]
    w, h = x1 - x0, y1 - y0
    mx, my = w / 8, h / 8                                   # crop_face(test)：外扩 1/4 再减半
    H, W = rgb.shape[:2]
    c = rgb[int(max(0, y0 - my)):int(min(H, y1 + my)), int(max(0, x0 - mx)):int(min(W, x1 + mx))]
    return cv2.resize(c, (380, 380), interpolation=cv2.INTER_LINEAR)


def crop_dfdc(rgb, box):
    import cv2
    import numpy as np

    x0, y0, x1, y1 = (int(v) for v in box[:4])
    pw, ph = (x1 - x0) // 3, (y1 - y0) // 3
    c = rgb[max(y0 - ph, 0):y1 + ph, max(x0 - pw, 0):x1 + pw]
    h, w = c.shape[:2]
    s = 380 / max(h, w)
    c = cv2.resize(c, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((380, 380, 3), np.uint8)
    hh, ww = c.shape[:2]
    out[(380 - hh) // 2:(380 - hh) // 2 + hh, (380 - ww) // 2:(380 - ww) // 2 + ww] = c
    return out


def confident_strategy(pred, t=0.8):
    import numpy as np

    pred = np.asarray(pred)
    sz = len(pred)
    if sz == 0:
        return 0.5
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return float(np.mean(pred[pred > t]))
    if np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return float(np.mean(pred[pred < 0.2]))
    return float(np.mean(pred))


def score_video(path: Path, label: str, models, mtcnn, device):
    import numpy as np
    import torch

    sbi, dfdc = models
    frames = sample_frames(path)
    faces = []
    for t, rgb in frames:
        for box in detect(mtcnn, rgb):
            faces.append({"t": round(t, 2), "box": [round(v) for v in box[:4]], "sbi_crop": crop_sbi(rgb, box),
                          "dfdc_crop": crop_dfdc(rgb, box)})
    if not faces:
        return {"label": label, "path": str(path), "frames": len(frames), "faces": 0}
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(faces), 16):
            chunk = faces[start:start + 16]
            x = torch.tensor(np.stack([f["sbi_crop"] for f in chunk])).permute(0, 3, 1, 2).float().to(device) / 255
            s = sbi(x).softmax(1)[:, 1].cpu().numpy()
            y = torch.tensor(np.stack([f["dfdc_crop"] for f in chunk])).permute(0, 3, 1, 2).float().to(device) / 255
            y = (y - mean) / std
            d = np.mean([torch.sigmoid(m(y)).squeeze(1).cpu().numpy() for m in dfdc], axis=0)
            for f, a, b in zip(chunk, s, d):
                f["sbi"], f["dfdc"] = float(a), float(b)
    by_frame: dict[float, list] = {}
    for f in faces:
        by_frame.setdefault(f["t"], []).append(f)
    sbi_video = float(np.mean([max(x["sbi"] for x in fs) for fs in by_frame.values()]))
    dfdc_video = confident_strategy([f["dfdc"] for f in faces])
    sheet(path.stem if label != "original" else f"orig_{path.stem}", faces)
    return {"label": label, "path": str(path), "frames": len(frames), "faces": len(faces),
            "frames_with_face": len(by_frame), "sbi_video": round(sbi_video, 4), "dfdc_video": round(dfdc_video, 4),
            "sbi_face_median": round(float(np.median([f["sbi"] for f in faces])), 4),
            "dfdc_face_median": round(float(np.median([f["dfdc"] for f in faces])), 4),
            "per_face": [{"t": f["t"], "box": f["box"], "sbi": round(f["sbi"], 4), "dfdc": round(f["dfdc"], 4)} for f in faces]}


def sheet(name: str, faces) -> None:
    import cv2
    import numpy as np

    tiles = []
    for f in faces[:96]:
        img = cv2.resize(f["sbi_crop"], (160, 160))
        cv2.putText(img, f"{f['t']:.0f}s", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(img, f"S{f['sbi']:.2f} D{f['dfdc']:.2f}", (3, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 1)
        tiles.append(img)
    cols = 12
    rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * 160, cols * 160, 3), np.uint8)
    for i, t in enumerate(tiles):
        canvas[(i // cols) * 160:(i // cols + 1) * 160, (i % cols) * 160:(i % cols + 1) * 160] = t
    cv2.imwrite(str(OUT / f"faces_{name}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONUTF8", "1")
    started = time.time()
    ensure_packages()
    import torch
    from facenet_pytorch import MTCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"设备 {device}；torch {torch.__version__}")
    ensure_weights()
    originals = ensure_originals()
    models = load_models(device)
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.7, 0.8, 0.8], margin=0)
    jobs = [(VIDEOS / f"{s}.mp4", "faceswap", s) for s in SWAP]
    jobs += [(p, "original", s) for s, p in originals.items()]
    jobs += [(VIDEOS / f"{s}.mp4", "real_dev", s) for s in REAL]
    results = []
    path_json = OUT / "results.json"
    for path, label, sample in jobs:
        t0 = time.time()
        try:
            row = score_video(path, label, models, mtcnn, device)
        except Exception as exc:  # noqa: BLE001
            row = {"label": label, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        row["sample"] = sample
        results.append(row)
        say(f"{label:9s} {sample} {path.name}: 人脸 {row.get('faces')}，SBI {row.get('sbi_video')}，DFDC {row.get('dfdc_video')}"
            f"（{time.time() - t0:.0f}s）{row.get('error', '')}")
        path_json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"完成，{time.time() - started:.0f}s；结果 {path_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
