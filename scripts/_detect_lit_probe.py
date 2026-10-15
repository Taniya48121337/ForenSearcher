"""文献里的检测方法在 EVID 开发样本上实测（10-12）：双击 scripts\\_detect_lit_probe.cmd。

负责人要求把"事实核查 + AIGC 检测"论文里用的方法一个个试一下。只跑离线打分，不跑 agent，只用已经读过的 18 个
AI 生成开发样本（插入 152/155/160/163/167/171、局部改动 201/205/210/213/216/220、换脸 175/177/178/181/184/187）
和 6 个 002–130 的真实开发视频（004/020/040/064/088/110），不碰其他测试样本。

标签沿用 10-10 的离线对齐结果（不重算）：
  - 插入：outputs/detect/compare_dev/<样本>.json 的 segments_not_in_b（与 GT 原片比对，原片里没有的片段）
  - 局部改动：outputs/detect/dev_probe/<样本>.json 里与原片配准后局部差异面积 > 0.3% 的帧，差异框 = 被改的位置
  - 换脸：dev_probe 里与原片同位置人脸的 FaceNet 相似度 < 0.45 = 被换的脸，≥ 0.6 = 没换

各阶段（每段一个子进程，结果各写一个 json，已有就跳过，--redo 重算）：
  prep    下载权重与代码（约 25GB，放 .models\\lit\\）：DDA、Effort 三个权重、TruFor、Skyra-RL、DINOv2 代码
  frames  逐帧 AI 生成图检测：DDA（NeurIPS'25，MM-VeriAgent 选用的工具）、Effort（ICML'25，GenImage / Chameleon 两个权重）
  trufor  篡改定位：TruFor（CVPR'23），看热力图峰值是否落在被改的位置
  faces   换脸：Effort（FF++ 权重）逐张人脸，对照 10-10 的 DFDC 分
  skyra   AI 视频检测：Skyra-RL（CVPR'26，Qwen2.5-VL-7B 微调，按镜头切成 ≤10 s 的片段逐段判）
  summary 汇总 outputs\\detect\\lit_probe\\summary.md

只跑某几段：--stages frames skyra。GPU 要约 22GB：开跑前会让 Ollama 卸载已加载的模型。日志 outputs\\detect\\lit_probe\\log.txt。
DFFreq（MM-VeriAgent 的另一个工具）没找到公开代码和权重，不在此列。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / ".models" / "lit"
PYLIB = LIT / "pylib"
DF_PYLIB = ROOT / ".models" / "deepfake" / "pylib"
CLIP_L = ROOT / ".models" / "aigi" / "openai__clip-vit-large-patch14"
ORIGINALS = ROOT / ".models" / "deepfake" / "originals"
OUT = ROOT / "outputs" / "detect" / "lit_probe"
VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
DEV_PROBE = ROOT / "outputs" / "detect" / "dev_probe"
COMPARE_DEV = ROOT / "outputs" / "detect" / "compare_dev"

INSERT = ["152", "155", "160", "163", "167", "171"]
EDIT = ["201", "205", "210", "213", "216", "220"]
SWAP = ["175", "177", "178", "181", "184", "187"]
REAL = ["004", "020", "040", "064", "088", "110"]
SKYRA_REAL = ["004", "040", "088", "110"]
KIND = {**{s: "insert" for s in INSERT}, **{s: "edit" for s in EDIT}, **{s: "swap" for s in SWAP}, **{s: "real" for s in REAL}}

DDA_CKPT = LIT / "dda" / "DDA_ckpt.pth"
DINOV2 = LIT / "dinov2-main"
EFFORT = {"ffpp": ("1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI", LIT / "effort" / "effort_clip_L14_ffpp.pth"),
          "genimage": ("1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d", LIT / "effort" / "effort_clip_L14_genimage_sdv14.pth"),
          "chameleon": ("1GlJ1y4xmTdqV0FfIcyBwNNU6cQird9DR", LIT / "effort" / "effort_clip_L14_chameleon_sdv14.pth")}
TRUFOR_CODE = LIT / "TruFor-main" / "test_docker" / "src"
TRUFOR_W = LIT / "trufor" / "weights" / "trufor.pth.tar"
SKYRA = LIT / "Skyra-RL"
HF_ENDPOINTS = [None, "https://hf-mirror.com"]
STAGES = ["prep", "frames", "trufor", "faces", "skyra", "summary"]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%m-%d %H:%M:%S')} {text}\n")


def use_pylibs() -> None:
    for p in (PYLIB, DF_PYLIB):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def write_json(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT / f"{name}.json")


# ============================================================================ prep

def pip_target(packages: list[str], *, deps: bool = False) -> None:
    PYLIB.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--target", str(PYLIB), "--upgrade"]
    if not deps:
        cmd.append("--no-deps")
    log(f"pip --target {PYLIB.name}：{packages}")
    subprocess.check_call(cmd + packages)


def download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"下载 {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as stream:
        shutil.copyfileobj(response, stream, 1 << 20)
    tmp.replace(dest)
    log(f"  完成 {dest.name} {dest.stat().st_size / 1e6:.0f}MB")
    return dest


def unzip_once(archive: Path, target: Path, marker: str) -> None:
    if (target / marker).exists():
        return
    log(f"解压 {archive.name} → {target}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)


def hf_file(repo: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    last = None
    for endpoint in HF_ENDPOINTS:
        try:
            kwargs = {"endpoint": endpoint} if endpoint else {}
            return Path(hf_hub_download(repo, filename, local_dir=str(local_dir), **kwargs))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  HF {repo}/{filename} 经 {endpoint or '默认端点'} 失败：{type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(f"下载不到 {repo}/{filename}: {last}")


def hf_snapshot(repo: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    last = None
    for endpoint in HF_ENDPOINTS:
        try:
            kwargs = {"endpoint": endpoint} if endpoint else {}
            return Path(snapshot_download(repo, local_dir=str(local_dir), **kwargs))
        except Exception as exc:  # noqa: BLE001
            last = exc
            log(f"  HF {repo} 经 {endpoint or '默认端点'} 失败：{type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError(f"下载不到 {repo}: {last}")


def stage_prep() -> None:
    free = shutil.disk_usage(ROOT).free / 1e9
    log(f"E 盘剩余 {free:.0f}GB（本次下载约 25GB）")
    if free < 30:
        raise SystemExit("磁盘不够 30GB，先清出空间再跑。")
    use_pylibs()
    for module, package in (("gdown", "gdown==5.2.0"), ("yacs", "yacs==0.1.8")):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            pip_target([package])
    for module in ("bs4", "filelock", "requests", "tqdm", "timm", "facenet_pytorch"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001
            log(f"!!! 缺 {module}：{exc}")
    import transformers

    log(f"transformers {transformers.__version__}")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log(f"!!! 本机 transformers 没有 Qwen2.5-VL（{exc}）；skyra 阶段会失败，需要 transformers>=4.49")

    failures = []

    def attempt(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            log(f"!!! {name} 准备失败：{type(exc).__name__}: {str(exc)[:300]}")

    attempt("DDA 权重", lambda: hf_file("Junwei-Xi/Dual-Data-Alignment", "DDA_ckpt.pth", DDA_CKPT.parent))

    def dinov2():
        archive = download("https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip", LIT / "_zips" / "dinov2-main.zip")
        unzip_once(archive, LIT, "dinov2-main/hubconf.py")

    attempt("DINOv2 代码", dinov2)

    def effort():
        import gdown

        for name, (file_id, dest) in EFFORT.items():
            if dest.is_file() and dest.stat().st_size > 1e8:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            log(f"Google Drive 下载 Effort {name}")
            gdown.download(id=file_id, output=str(dest), quiet=False)
            if not dest.is_file() or dest.stat().st_size < 1e8:
                raise RuntimeError(f"Effort {name} 下载不完整")

    attempt("Effort 权重", effort)

    def trufor():
        code = download("https://github.com/grip-unina/TruFor/archive/refs/heads/main.zip", LIT / "_zips" / "TruFor-main.zip")
        unzip_once(code, LIT, "TruFor-main/test_docker/src/trufor_test.py")
        weights = download("https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip", LIT / "_zips" / "TruFor_weights.zip")
        unzip_once(weights, LIT / "trufor", "weights/trufor.pth.tar")
        if not TRUFOR_W.is_file():
            found = list((LIT / "trufor").rglob("*.pth.tar"))
            raise RuntimeError(f"解压后没找到 weights/trufor.pth.tar：{found}")

    attempt("TruFor", trufor)
    attempt("Skyra-RL", lambda: hf_snapshot("JoeLeelyf/Skyra-RL", SKYRA))
    if failures:
        log(f"准备阶段有 {len(failures)} 项失败：{failures}；对应的检测阶段会跳过。")
    else:
        log("准备阶段全部完成。")


# ============================================================================ shared helpers

def ensure_gpu(need_gb: float):
    import torch

    if not torch.cuda.is_available():
        log("没有 CUDA，改用 CPU（会很慢）")
        return "cpu"
    free = torch.cuda.mem_get_info()[0] / 1e9
    if free < need_gb:
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as r:
                models = [m.get("name") for m in json.loads(r.read()).get("models", [])]
            for name in models:
                log(f"显存只剩 {free:.1f}GB，让 Ollama 卸载 {name}")
                body = json.dumps({"model": name, "keep_alive": 0}).encode()
                urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                                              headers={"Content-Type": "application/json"}), timeout=60).read()
            time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            log(f"  查询 / 卸载 Ollama 模型失败：{exc}")
        free = torch.cuda.mem_get_info()[0] / 1e9
    log(f"GPU 空闲显存 {free:.1f}GB（需要约 {need_gb}GB）")
    if free < need_gb:
        raise SystemExit(f"显存不够（{free:.1f}GB < {need_gb}GB）：可能有别的任务在用 GPU，等它结束再双击。")
    return "cuda"


def video_meta(path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def grab(path: Path, idxs) -> dict:
    """按帧号取 RGB 帧（排序后顺读，间隔大于 2 秒才 seek）。"""

    import cv2

    want = sorted(set(int(i) for i in idxs))
    out = {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pos = -1
    for i in want:
        if i < pos + 1 or i - pos > 2 * fps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            pos = i - 1
        while pos < i - 1:
            if not cap.grab():
                break
            pos += 1
        ok, bgr = cap.read()
        pos += 1
        if ok:
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def even(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def probe(sample: str) -> dict:
    return json.loads((DEV_PROBE / f"{sample}.json").read_text(encoding="utf-8"))


def insert_segments(sample: str) -> list[tuple[float, float]]:
    path = COMPARE_DEV / f"{sample}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(seg["a_range_s"]) for seg in data.get("segments_not_in_b", [])]


def frame_plan(sample: str, max_pos: int = 40, max_neg: int = 40) -> tuple[list, list, dict]:
    """返回（正例帧, 负例帧, dev_probe 结果）。帧是 dev_probe 的 frame 行。"""

    d = probe(sample)
    rows = d["frames"]
    has_ncc = any("ncc" in x for x in rows)
    if KIND[sample] == "insert":
        segs = insert_segments(sample)
        pos = [x for x in rows if any(a + 0.5 <= x["t"] <= b - 0.5 for a, b in segs)]
        neg = [x for x in rows if not any(a - 1.5 <= x["t"] <= b + 1.5 for a, b in segs)
               and (not has_ncc or x.get("ncc", 0) >= 0.9)]
    else:
        pos = [x for x in rows if x.get("diff") and x["diff"]["area"] > 0.003]
        neg = [x for x in rows if x.get("diff") and x["diff"]["area"] <= 0.001 and x.get("ncc", 0) >= 0.9]
    return even(pos, max_pos), even(neg, max_neg), d


def tiles(frame, size: int) -> list[tuple[int, int, object]]:
    H, W = frame.shape[:2]
    if H < size or W < size:
        import cv2

        k = size / min(H, W)
        frame = cv2.resize(frame, (max(size, round(W * k)), max(size, round(H * k))), interpolation=cv2.INTER_CUBIC)
        H, W = frame.shape[:2]
    nx, ny = max(1, math.ceil(W / size)), max(1, math.ceil(H / size))
    xs = [round(i * (W - size) / (nx - 1)) for i in range(nx)] if nx > 1 else [(W - size) // 2]
    ys = [round(i * (H - size) / (ny - 1)) for i in range(ny)] if ny > 1 else [(H - size) // 2]
    return [(x, y, frame[y:y + size, x:x + size]) for y in ys for x in xs]


def to_batch(images, device, size=None):
    import cv2
    import numpy as np
    import torch

    if size:
        images = [cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC) for im in images]
    x = torch.tensor(np.stack(images), device=device).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


# ============================================================================ detectors

def merge_dda_state(state: dict) -> tuple[dict, dict]:
    """把 rank 8、alpha 1 的 LoRA 并回 DINOv2 权重（与 src/foren_searcher/detect/backends.py 同一实现）。"""

    prefix = "base_model.model."
    backbone, head, consumed = {}, {}, set()
    for key, value in state.items():
        if key.startswith("base_model.fc."):
            head[key.removeprefix("base_model.fc.")] = value
            consumed.add(key)
        elif key.startswith(prefix) and ".lora." not in key:
            name = key.removeprefix(prefix)
            if name.endswith(".original_layer.weight"):
                stem = key.removesuffix(".original_layer.weight")
                a_key, b_key = stem + ".lora.lora_A", stem + ".lora.lora_B"
                a, b = state[a_key], state[b_key]
                if a.shape[0] != 8 or b.shape[1] != 8:
                    raise ValueError("DDA checkpoint must use rank 8, alpha 1")
                value = value.float() + b.float() @ a.float() / 8
                consumed.update((a_key, b_key))
            backbone[name.replace(".original_layer.", ".")] = value
            consumed.add(key)
    if consumed != set(state) or not head or not backbone:
        raise ValueError(f"DDA 权重结构与预期不符：未用到的键 {sorted(set(state) - consumed)[:5]}")
    return backbone, head


class DDA:
    """DINOv2-L/14 + LoRA（rank 8, alpha 1），官方推理：原分辨率 CenterCrop 336，CLIP 归一化，sigmoid。"""

    def __init__(self, device: str):
        import torch

        ck = torch.load(str(DDA_CKPT), map_location="cpu", weights_only=False)
        backbone, head = merge_dda_state(ck.get("model", ck))
        model = torch.hub.load(str(DINOV2), "dinov2_vitl14", source="local", pretrained=False)
        model.load_state_dict(backbone, strict=True)
        fc = torch.nn.Linear(1024, 1)
        fc.load_state_dict(head, strict=True)
        self.model, self.fc, self.device = model.to(device).eval(), fc.to(device).eval(), device

    def score_crops(self, crops) -> list[float]:
        import torch

        out = []
        with torch.inference_mode():
            for k in range(0, len(crops), 16):
                x = to_batch(crops[k:k + 16], self.device)
                feat = self.model.forward_features(x)["x_norm_clstoken"]
                out += torch.sigmoid(self.fc(feat)).squeeze(1).float().cpu().tolist()
        return out


class Effort:
    """CLIP ViT-L/14 + Effort（自注意力层的 SVD 残差），官方 demo：整图缩到 224，CLIP 归一化，softmax 第 1 类。"""

    def __init__(self, ckpt: Path, device: str):
        import torch
        from transformers import CLIPVisionModel

        vision = CLIPVisionModel.from_pretrained(str(CLIP_L), torch_dtype=torch.float32).vision_model
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = ck.get("state_dict", ck)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        merged, n_svd = {}, 0
        skip = (".U_residual", ".S_residual", ".V_residual", ".U_r", ".S_r", ".V_r", ".weight_original_fnorm", ".weight_main_fnorm")
        for k, v in backbone.items():
            if k.endswith(".weight_main"):
                stem = k[: -len(".weight_main")]
                w = v.float()
                u, s, vv = backbone.get(stem + ".U_residual"), backbone.get(stem + ".S_residual"), backbone.get(stem + ".V_residual")
                if u is not None and s is not None and vv is not None:
                    w = w + u.float() @ torch.diag(s.float()) @ vv.float()
                merged[stem + ".weight"] = w
                n_svd += 1
            elif not k.endswith(skip):
                merged[k] = v
        missing, unexpected = vision.load_state_dict(merged, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        log(f"  Effort {ckpt.name}：合并 SVD 层 {n_svd}，缺 {len(missing)} 个键，多 {len(unexpected)} 个键"
            + (f"（缺：{missing[:4]}）" if missing else "") + (f"（多：{unexpected[:4]}）" if unexpected else ""))
        if n_svd == 0 or len(missing) > 0:
            raise RuntimeError("Effort 权重结构与预期不符")
        head = torch.nn.Linear(1024, 2)
        head.load_state_dict({"weight": state["head.weight"], "bias": state["head.bias"]})
        self.model, self.head, self.device = vision.to(device).eval(), head.to(device).eval(), device

    def score_images(self, images) -> list[float]:
        import torch

        out = []
        with torch.inference_mode():
            for k in range(0, len(images), 16):
                x = to_batch(images[k:k + 16], self.device, size=224)
                feat = self.model(pixel_values=x).pooler_output
                out += torch.softmax(self.head(feat), dim=1)[:, 1].float().cpu().tolist()
        return out


def align_face(frame, lm5, res: int = 224, scale: float = 1.3):
    """DeepfakeBench / Effort 的 img_align_crop：五点相似变换到 112 模板，按 1.3 倍外扩，输出 res×res。"""

    import cv2
    import numpy as np

    dst = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]], np.float32)
    dst[:, 0] += 8.0
    dst *= res / 112
    margin = res * (scale - 1) / 2
    dst += margin
    dst *= res / (res + 2 * margin)
    M, _ = cv2.estimateAffinePartial2D(np.asarray(lm5, np.float32), dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (res, res), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def box_iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ============================================================================ stage: frames

def frame_jobs() -> list[dict]:
    """每个要打分的帧：sample / role（pos 插入或被改、neg 同视频未改、orig 被改帧在原片里的同一帧、real 真实开发视频）。"""

    jobs = []
    for s in INSERT + EDIT:
        try:
            pos, neg, d = frame_plan(s)
        except FileNotFoundError:
            log(f"!!! 缺 dev_probe/{s}.json，跳过")
            continue
        for role, rows in (("pos", pos), ("neg", neg)):
            for x in rows:
                job = {"sample": s, "kind": KIND[s], "role": role, "t": x["t"], "idx": x["idx"]}
                if role == "pos" and KIND[s] == "edit":
                    job["boxes"] = [b[:4] for b in x["diff"]["boxes"]]
                jobs.append(job)
                if role == "pos" and KIND[s] == "edit" and d.get("original") and x.get("orig_t") is not None:
                    jobs.append({"sample": s, "kind": KIND[s], "role": "orig", "t": x["orig_t"], "pair_t": x["t"],
                                 "orig_path": d["original"]["path"], "orig_fps": d["original"]["fps"]})
    for s in REAL:
        path = VIDEOS / f"video_{s}.mp4"
        if not path.is_file():
            log(f"!!! 没有 {path.name}")
            continue
        fps, n = video_meta(path)
        for k in range(20):
            idx = int(n * (0.05 + 0.9 * k / 19))
            jobs.append({"sample": s, "kind": "real", "role": "real", "t": round(idx / fps, 2), "idx": idx})
    return jobs


def load_frames(jobs: list[dict]):
    """按视频分组取帧，逐组 yield (job, frame)。"""

    groups: dict[tuple, list] = {}
    for job in jobs:
        if job["role"] == "orig":
            key = ("orig", job["orig_path"])
        else:
            key = ("video", job["sample"])
        groups.setdefault(key, []).append(job)
    for (kind, name), items in groups.items():
        if kind == "orig":
            path = ORIGINALS / name
            fps = items[0]["orig_fps"]
            for j in items:
                j["idx"] = int(round(j["t"] * fps))
        else:
            path = VIDEOS / f"video_{name}.mp4"
        if not path.is_file():
            log(f"!!! 缺视频 {path}")
            continue
        frames = grab(path, [j["idx"] for j in items])
        for j in items:
            if j["idx"] in frames:
                yield j, frames[j["idx"]]


def stage_frames() -> None:
    import torch

    use_pylibs()
    device = ensure_gpu(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    detectors = {}
    try:
        detectors["dda"] = DDA(device)
        log("已加载 DDA")
    except Exception as exc:  # noqa: BLE001
        log(f"!!! DDA 加载失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")
    for name in ("genimage", "chameleon"):
        try:
            detectors[f"effort_{name}"] = Effort(EFFORT[name][1], device)
            log(f"已加载 Effort {name}")
        except Exception as exc:  # noqa: BLE001
            log(f"!!! Effort {name} 加载失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}")
    if not detectors:
        raise SystemExit("frames 阶段一个检测器都没加载上")
    jobs = frame_jobs()
    log(f"frames：{len(jobs)} 帧，检测器 {list(detectors)}")
    results, started = [], time.monotonic()
    for n, (job, frame) in enumerate(load_frames(jobs), 1):
        rec = dict(job)
        rec["scores"], rec["tiles"] = {}, {}
        H, W = frame.shape[:2]
        if "dda" in detectors:
            t336 = tiles(frame, 336)
            center = frame[(H - 336) // 2:(H - 336) // 2 + 336, (W - 336) // 2:(W - 336) // 2 + 336] if min(H, W) >= 336 else t336[0][2]
            vals = detectors["dda"].score_crops([center] + [c for _, _, c in t336])
            rec["scores"]["dda_center"] = round(vals[0], 4)
            rec["scores"]["dda_max"] = round(max(vals[1:]), 4)
            rec["tiles"]["dda"] = [[x / W, y / H, (x + 336) / W, (y + 336) / H, round(v, 4)] for (x, y, _), v in zip(t336, vals[1:])]
        for name in ("effort_genimage", "effort_chameleon"):
            if name not in detectors:
                continue
            t = tiles(frame, 336)
            vals = detectors[name].score_images([frame] + [c for _, _, c in t])
            rec["scores"][f"{name}_whole"] = round(vals[0], 4)
            rec["scores"][f"{name}_max"] = round(max(vals[1:]), 4)
            rec["tiles"][name] = [[x / W, y / H, (x + 336) / W, (y + 336) / H, round(v, 4)] for (x, y, _), v in zip(t, vals[1:])]
        results.append(rec)
        if n % 100 == 0:
            log(f"  {n}/{len(jobs)} 帧，{time.monotonic() - started:.0f}s")
    write_json("frames", {"detectors": list(detectors), "frames": results})
    log(f"frames 完成：{len(results)} 帧，{time.monotonic() - started:.0f}s")


# ============================================================================ stage: trufor

def stage_trufor() -> None:
    import numpy as np
    import torch

    use_pylibs()
    device = ensure_gpu(6)
    sys.path.insert(0, str(TRUFOR_CODE))
    from config import _C as config  # TruFor test_docker/src/config.py

    config.defrost()
    config.merge_from_file(str(TRUFOR_CODE / "trufor.yaml"))
    config.TEST.MODEL_FILE = str(TRUFOR_W)
    config.freeze()
    from models.cmx.builder_np_conf import myEncoderDecoder

    model = myEncoderDecoder(cfg=config)
    ck = torch.load(str(TRUFOR_W), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    log("已加载 TruFor")
    jobs = [j for j in frame_jobs() if j["kind"] in ("edit", "real") or (j["kind"] == "insert")]
    # 插入样本每个只取 10 正 10 负，主要看局部改动
    keep, seen = [], {}
    for j in jobs:
        if j["kind"] == "insert":
            key = (j["sample"], j["role"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 10:
                continue
        if j["kind"] == "real":
            key = (j["sample"], "real")
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 10:
                continue
        keep.append(j)
    log(f"trufor：{len(keep)} 帧")
    results, started = [], time.monotonic()
    with torch.no_grad():
        for n, (job, frame) in enumerate(load_frames(keep), 1):
            x = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float, device=device)[None] / 256.0
            try:
                pred, conf, det, _ = model(x)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  {job['sample']} t={job['t']} 显存不够，跳过")
                continue
            heat = torch.softmax(pred[0], dim=0)[1].float().cpu().numpy()
            H, W = heat.shape
            rec = dict(job)
            rec["score"] = round(float(torch.sigmoid(det).item()), 4) if det is not None else None
            rec["map_max"] = round(float(heat.max()), 4)
            rec["map_p99"] = round(float(np.percentile(heat, 99)), 4)
            rec["map_mean"] = round(float(heat.mean()), 4)
            yy, xx = np.unravel_index(int(heat.argmax()), heat.shape)
            rec["peak"] = [round(xx / W, 4), round(yy / H, 4)]
            if job.get("boxes"):
                mask = np.zeros_like(heat, dtype=bool)
                for b in job["boxes"]:
                    x0, y0 = int(max(0, b[0] - 0.02) * W), int(max(0, b[1] - 0.02) * H)
                    x1, y1 = int(min(1, b[2] + 0.02) * W), int(min(1, b[3] + 0.02) * H)
                    mask[y0:y1, x0:x1] = True
                rec["box_frac"] = round(float(mask.mean()), 4)
                rec["peak_in_box"] = bool(mask[yy, xx])
                rec["mean_in_box"] = round(float(heat[mask].mean()), 4) if mask.any() else None
                rec["mean_out_box"] = round(float(heat[~mask].mean()), 4) if (~mask).any() else None
                # 热力图前 1% 的像素里有多少落在框内
                top = heat >= np.percentile(heat, 99)
                rec["top1_in_box"] = round(float((top & mask).sum() / max(1, top.sum())), 4)
            results.append(rec)
            if n % 50 == 0:
                log(f"  {n}/{len(keep)} 帧，{time.monotonic() - started:.0f}s")
    write_json("trufor", {"frames": results})
    log(f"trufor 完成：{len(results)} 帧，{time.monotonic() - started:.0f}s")


# ============================================================================ stage: faces

def stage_faces() -> None:
    import numpy as np
    import torch
    from PIL import Image

    use_pylibs()
    device = ensure_gpu(6)
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
    effort = Effort(EFFORT["ffpp"][1], device)
    log("已加载 MTCNN + Effort FF++")

    jobs = []   # 每帧：sample, idx, faces=[{box, label, dfdc}]
    for s in SWAP + INSERT + EDIT:
        try:
            d = probe(s)
        except FileNotFoundError:
            continue
        rows = []
        for x in d["frames"]:
            fa = x.get("faces") or {}
            faces = []
            for b, dfdc, v in zip(fa.get("boxes") or [], fa.get("dfdc") or [], fa.get("vs_orig") or [None] * 99):
                if not v:
                    continue
                if KIND[s] == "swap":
                    label = "swapped" if v["sim"] < 0.45 else ("kept" if v["sim"] >= 0.6 else None)
                else:
                    label = "real_other" if v["sim"] >= 0.6 else None
                if label:
                    faces.append({"box": b, "label": label, "dfdc": dfdc, "sim": v["sim"]})
            if faces:
                rows.append({"sample": s, "idx": x["idx"], "t": x["t"], "faces": faces})
        cap = 60 if KIND[s] == "swap" else 20
        jobs += even(rows, cap)
    for s in REAL:
        path = VIDEOS / f"video_{s}.mp4"
        if path.is_file():
            fps, n = video_meta(path)
            for k in range(30):
                idx = int(n * (0.03 + 0.94 * k / 29))
                jobs.append({"sample": s, "idx": idx, "t": round(idx / fps, 2), "faces": None})
    log(f"faces：{len(jobs)} 帧")
    by_sample: dict[str, list] = {}
    for j in jobs:
        by_sample.setdefault(j["sample"], []).append(j)
    results, started = [], time.monotonic()
    for s, items in by_sample.items():
        frames = grab(VIDEOS / f"video_{s}.mp4", [j["idx"] for j in items])
        for j in items:
            frame = frames.get(j["idx"])
            if frame is None:
                continue
            boxes, probs, lms = mtcnn.detect(Image.fromarray(frame), landmarks=True)
            if boxes is None:
                continue
            dets = [(b, lm) for b, p, lm in zip(boxes, probs, lms) if p is not None and p >= 0.9 and (b[2] - b[0]) >= 40]
            targets = j["faces"] if j["faces"] is not None else [{"box": [float(v) for v in b], "label": "real_video", "dfdc": None, "sim": None} for b, _ in dets]
            crops, keep = [], []
            for f in targets:
                best = max(dets, key=lambda d: box_iou(f["box"], d[0]), default=None)
                if best is None or box_iou(f["box"], best[0]) < 0.3:
                    continue
                crop = align_face(frame, best[1])
                if crop is None:
                    continue
                crops.append(crop)
                keep.append(f)
            if not crops:
                continue
            scores = effort.score_images(crops)
            for f, sc in zip(keep, scores):
                results.append({"sample": s, "kind": KIND[s], "t": j["t"], "label": f["label"], "effort_ffpp": round(sc, 4),
                                "dfdc": f["dfdc"], "sim": f["sim"], "box": [round(float(v)) for v in f["box"]]})
        log(f"  {s}：累计 {len(results)} 张脸，{time.monotonic() - started:.0f}s")
    write_json("faces", {"faces": results})
    log(f"faces 完成：{len(results)} 张脸")


# ============================================================================ stage: skyra

SKYRA_SYSTEM = """\nYou are an expert AI video analyst. Your primary task is to review a sequence of video frames and provide a step-by-step analysis of their authenticity.\n\nYou MUST output your entire analysis using the following structure:\n1.  A `<think>...</think>` block containing your detailed reasoning.\n2.  An `<answer>...</answer>` block containing the final, one-word verdict: 'Fake' or 'Real'.\n\nInside the `<think>` block, you MUST:\n1.  Start by briefly describing the overall content of the video frames.\n2.  Follow a detailed, step-by-step \"discovery\" or \"verification\" process.\n3.  When you identify an artifact (or clear a region), you MUST use a valid L3 Category Name from the \"Artifact Category Definitions\" provided below.\n4.  You MUST embed your finding using the following exact tag structure:\n    <type>L3 Category Name</type> in <t>[startTime, endTime]</t> at <bbox>[x1, y1, x2, y2]</bbox>\n5.  If multiple artifacts are present, you must find and tag all of them in temporal order.\n6.  Your entire reasoning process must be self-contained\n\n---\n## Artifact Category Definitions (Valid L3 Categories for the <type> tag)\n\n### 1. Low-Level Forgery\n* **1.1 Texture Anomaly**:\n    * <type>Structure Anomaly</type>\n    * <type>Texture Jittering</type>\n    * <type>Unnatural Blur</type>\n* **1.2 Color and Lighting Anomaly**:\n    * <type>Color Over-saturation</type>\n    * <type>Lighting Inconsistency</type>\n* **1.3 Move Forgery**:\n    * <type>Camera Motion Inconsistency</type>\n\n### 2. Violation of Laws\n* **2.1 Object Inconsistency**:\n    * <type>Abnormal Object Disappearance</type>\n    * <type>Abnormal Object Appearance</type>\n    * <type>Person Identity Inconsistency</type>\n    * <type>General Object Identity Inconsistency</type>\n    * <type>Shape Distortion</type>\n* **2.2 Interaction Inconsistency**:\n    * <type>Abnormal Rigid-Body Crossing</type>\n    * <type>Abnormal Multi-Object Merging</type>\n    * <type>Abnormal Object Splitting</type>\n    * <type>General Interaction Anomaly</type>\n* **2.3 Unnatural Movement**:\n    * <type>Unnatural Human Movement</type>\n    * <type>Unnatural Animal Movement</type>\n    * <type>Unnatural General Object Movement</type>\n* **2.4 Violation of Causality Law**:\n    * <type>Violation of Physical Law</type>\n    * <type>Violation of General Causality Law</type>\n* **2.5 Violation of Commonsense**:\n    * <type>Abnormal Human Body Structure</type>\n    * <type>Abnormal General Object Structure</type>\n    * <type>Text Distortion</type>\n"""


def shots(path: Path, step_s: float = 0.2, threshold: float = 0.6) -> tuple[list[tuple[float, float]], float]:
    """HSV 直方图相关 < 0.6 视为硬切（与 ShotScanner 同口径），返回镜头列表和时长。"""

    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(step_s * fps)))
    prev, cuts, idx = None, [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                hsv = cv2.cvtColor(cv2.resize(bgr, (160, 90)), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < threshold:
                    cuts.append(idx / fps)
                prev = hist
        idx += 1
    cap.release()
    dur = n / fps
    bounds = [0.0] + cuts + [dur]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 1.0], dur


def skyra_chunks(sample: str, cap: int = 40) -> list[dict]:
    path = VIDEOS / f"video_{sample}.mp4"
    shot_list, dur = shots(path)
    chunks = []
    for a, b in shot_list:
        k = max(1, math.ceil((b - a) / 10.0))
        for i in range(k):
            chunks.append({"start": round(a + (b - a) * i / k, 2), "end": round(a + (b - a) * (i + 1) / k, 2)})
    kind = KIND[sample]
    if kind == "insert":
        segs = insert_segments(sample)
        for c in chunks:
            ov = sum(max(0.0, min(c["end"], b) - max(c["start"], a)) for a, b in segs)
            frac = ov / max(1e-6, c["end"] - c["start"])
            c["label"] = "pos" if frac >= 0.5 else ("neg" if frac == 0 else "mixed")
    elif kind == "edit":
        rows = probe(sample)["frames"]
        for c in chunks:
            inside = [x for x in rows if c["start"] <= x["t"] < c["end"]]
            if any(x.get("diff") and x["diff"]["area"] > 0.003 for x in inside):
                c["label"] = "pos"
            elif inside and all(x.get("diff") and x["diff"]["area"] <= 0.001 for x in inside):
                c["label"] = "neg"
            else:
                c["label"] = "unknown"
    else:
        for c in chunks:
            c["label"] = "real"
    if len(chunks) > cap:
        pos = [c for c in chunks if c["label"] == "pos"]
        rest = even([c for c in chunks if c["label"] != "pos"], max(1, cap - len(pos)))
        chunks = sorted(pos + rest, key=lambda c: c["start"])
    return chunks


def stage_skyra() -> None:
    import cv2
    import torch
    from PIL import Image

    use_pylibs()
    smoke = os.environ.get("LIT_SKYRA_SMOKE") == "1"
    device = ensure_gpu(22)
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    log(f"transformers {transformers.__version__}，torch {torch.__version__}，{torch.cuda.get_device_name(0)}；"
        f"{'冒烟：只跑 152 的第一段' if smoke else '全量'}")

    started = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(SKYRA), torch_dtype=torch.bfloat16, device_map=device,
                                                               attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(str(SKYRA), min_pixels=224 * 224, max_pixels=1280 * 28 * 28)
    tok = proc.tokenizer
    fake_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Fake", " Fake", "fake")})
    real_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Real", " Real", "real")})
    # 训练框架（LLaMA-Factory / verl）存下的 config 常带 use_cache=False：生成时不用 KV 缓存，每个 token 都要把 16 张图和
    # 全部上文重算一遍（10-12 冒烟实测 16 个 token 29 s）。推理时强制打开。
    cache_flags = (getattr(model.config, "use_cache", None), getattr(model.generation_config, "use_cache", None))
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    model.generation_config.use_cache = True
    log(f"权重自带 use_cache（config, generation_config）= {cache_flags}，已改为 True")
    devices = sorted({str(p.device) for p in model.parameters()})
    log(f"已加载 Skyra-RL（{time.monotonic() - started:.0f}s，参数在 {devices}，dtype {model.dtype}）；Fake token {fake_ids}，Real token {real_ids}")
    partial = OUT / "skyra.partial.jsonl"
    done = set()
    if partial.is_file():
        for line in partial.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["sample"], r["start"]))
    samples = INSERT + EDIT + SKYRA_REAL
    for s in samples:
        path = VIDEOS / f"video_{s}.mp4"
        if not path.is_file():
            continue
        chunks = skyra_chunks(s)
        fps, n = video_meta(path)
        todo = [c for c in chunks if (s, c["start"]) not in done]
        log(f"[{s}] {KIND[s]}：{len(chunks)} 段（正例 {sum(c['label'] == 'pos' for c in chunks)}），待跑 {len(todo)}")
        if smoke:
            todo = todo[:1]
        for c in todo:
            t0 = time.monotonic()
            log(f"  [{s}] {c['start']}–{c['end']} s：取帧") if smoke else None
            a, b = int(c["start"] * fps), max(int(c["start"] * fps) + 1, int(c["end"] * fps) - 1)
            idxs = [int(round(a + (b - a) * i / 15)) for i in range(16)]
            frames = grab(path, idxs)
            imgs, ts = [], []
            for i in idxs:
                if i not in frames:
                    continue
                f = frames[i]
                h, w = f.shape[:2]
                k = 256 / min(h, w)
                imgs.append(Image.fromarray(cv2.resize(f, (round(w * k), round(h * k)), interpolation=cv2.INTER_LANCZOS4)))
                ts.append((i - a) / fps)
            if len(imgs) < 4:
                continue
            content = []
            text = "Here are the video frames and their corresponding timestamps:\n"
            for k, t in enumerate(ts):
                content.append({"type": "text", "text": text + f"[T={t:.2f}s] "})
                content.append({"type": "image"})
                text = "\n"
            content.append({"type": "text", "text": "\n\nPlease analyze the video frames, determine if the video is real or fake, and provide your reasoning."})
            messages = [{"role": "system", "content": SKYRA_SYSTEM}, {"role": "user", "content": content}]
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            t1 = time.monotonic()
            inputs = proc(text=[prompt], images=imgs, return_tensors="pt").to(device)
            n_in = int(inputs["input_ids"].shape[1])
            if smoke:
                log(f"    取帧 {t1 - t0:.1f}s；输入 {n_in} token，{len(imgs)} 张图，预处理 {time.monotonic() - t1:.1f}s；先试生成 16 个 token")
                t2 = time.monotonic()
                with torch.inference_mode():
                    model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
                torch.cuda.synchronize()
                log(f"    16 token 用了 {time.monotonic() - t2:.1f}s；显存占用 {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            t2 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
            n_out = int(out.shape[1]) - n_in
            gen = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
            gen_s = time.monotonic() - t2
            k = gen.find("<answer>")
            prefix = gen[:k + len("<answer>")] if k >= 0 else gen.split("</think>")[0] + "</think>\n<answer>"
            inputs2 = proc(text=[prompt + prefix], images=imgs, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**inputs2).logits[0, -1].float()
            lf = torch.logsumexp(logits[fake_ids], 0)
            lr = torch.logsumexp(logits[real_ids], 0)
            p_fake = float(torch.sigmoid(lf - lr).item())
            verdict = gen[k + len("<answer>"):].split("</answer>")[0].strip() if k >= 0 else ""
            rec = {"sample": s, "kind": KIND[s], **c, "p_fake": round(p_fake, 4), "verdict": verdict,
                   "artifacts": gen.count("<type>"), "answer_found": k >= 0, "seconds": round(time.monotonic() - t0, 1),
                   "text": gen[-1500:]}
            rec["gen_tokens"], rec["gen_seconds"] = n_out, round(gen_s, 1)
            log(f"    {s} {c['start']}–{c['end']} s（{c['label']}）：p_fake {p_fake:.3f}，结论 {verdict or '无'}，"
                f"输入 {n_in} / 生成 {n_out} token，生成 {gen_s:.0f}s，共 {time.monotonic() - t0:.0f}s")
            if smoke:
                log("    生成的文字（末尾 600 字）：" + gen[-600:].replace("\n", " "))
                return
            with partial.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            del inputs, inputs2, out
        log(f"  [{s}] 完成，累计 {time.monotonic() - started:.0f}s")
    rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    write_json("skyra", {"chunks": rows})
    log(f"skyra 完成：{len(rows)} 段，{time.monotonic() - started:.0f}s")


# ============================================================================ summary

def auroc(pos, neg):
    pos, neg = [p for p in pos if p is not None], [n for n in neg if n is not None]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def summary_frames(lines: list[str]) -> None:
    path = OUT / "frames.json"
    if not path.is_file():
        lines += ["## 逐帧 AI 生成图检测（DDA / Effort）", "", "没有结果（frames 阶段没跑成）。", ""]
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["frames"]
    keys = sorted({k for r in rows for k in r["scores"]})
    lines += ["## 逐帧 AI 生成图检测（DDA / Effort）", "",
              "正例 = 插入片段里的帧 / 被改的帧；负例 = 同一视频里原片就有、没改的帧；real = 6 个真实开发视频各 20 帧。",
              "AUROC 0.5 = 没信号。_max = 把帧切成 336 块取最高分；_center / _whole = 官方协议（中心 336 / 整图缩到 224）。", "",
              "### 同一视频内 AUROC（正例 vs 同视频负例）", "",
              "| 样本 | 类型 | 正 | 负 | " + " | ".join(keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    samples = sorted({r["sample"] for r in rows if r["kind"] in ("insert", "edit")})
    for s in samples:
        pos = [r for r in rows if r["sample"] == s and r["role"] == "pos"]
        neg = [r for r in rows if r["sample"] == s and r["role"] == "neg"]
        cells = [fmt(auroc([r["scores"].get(k) for r in pos], [r["scores"].get(k) for r in neg])) for k in keys]
        lines.append(f"| {s} | {KIND[s]} | {len(pos)} | {len(neg)} | " + " | ".join(cells) + " |")
    lines += ["", "### 合并（跨视频）", "", "| 指标 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for label, kinds in (("插入帧 vs 真实视频帧", ("insert",)), ("被改帧 vs 真实视频帧", ("edit",))):
        pos = [r for r in rows if r["kind"] in kinds and r["role"] == "pos"]
        real = [r for r in rows if r["role"] == "real"]
        lines.append(f"| {label} AUROC | " + " | ".join(fmt(auroc([r["scores"].get(k) for r in pos], [r["scores"].get(k) for r in real])) for k in keys) + " |")
    for label, role in (("插入帧 中位分", "pos_insert"), ("被改帧 中位分", "pos_edit"), ("同视频未改帧 中位分", "neg"), ("真实视频帧 中位分", "real"), ("原片同一帧 中位分", "orig")):
        if role.startswith("pos_"):
            sel = [r for r in rows if r["role"] == "pos" and r["kind"] == role[4:]]
        else:
            sel = [r for r in rows if r["role"] == role]
        lines.append(f"| {label}（{len(sel)}） | " + " | ".join(fmt(median([r["scores"].get(k) for r in sel]), 3) for k in keys) + " |")
    pairs = {}
    for r in rows:
        if r["role"] == "orig":
            pairs[(r["sample"], r["pair_t"])] = r
    cells = []
    for k in keys:
        wins = [(r["scores"].get(k), pairs[(r["sample"], r["t"])]["scores"].get(k)) for r in rows
                if r["role"] == "pos" and r["kind"] == "edit" and (r["sample"], r["t"]) in pairs]
        wins = [a > b for a, b in wins if a is not None and b is not None]
        cells.append(f"{sum(wins)}/{len(wins)}" if wins else "—")
    lines.append("| 被改帧 > 原片同一帧 | " + " | ".join(cells) + " |")
    # 定位：最高分的 336 块是否覆盖被改的位置
    lines += ["", "### 定位：被改帧里最高分的 336 块是否覆盖被改位置（对照：随机一块覆盖的概率）", "",
              "| 检测器 | 命中 | 随机期望 |", "|---|---|---|"]
    for det in sorted({d for r in rows for d in r.get("tiles", {})}):
        hit, chance, n = 0, 0.0, 0
        for r in rows:
            if r["role"] != "pos" or r["kind"] != "edit" or det not in r.get("tiles", {}) or not r.get("boxes"):
                continue
            ts = r["tiles"][det]
            cover = [any(box_iou(t[:4], b) > 0 for b in r["boxes"]) for t in ts]
            best = max(range(len(ts)), key=lambda i: ts[i][4])
            hit += cover[best]
            chance += sum(cover) / len(cover)
            n += 1
        if n:
            lines.append(f"| {det} | {hit}/{n} = {hit / n:.2f} | {chance / n:.2f} |")
    lines.append("")


def summary_trufor(lines: list[str]) -> None:
    path = OUT / "trufor.json"
    if not path.is_file():
        lines += ["## TruFor 篡改定位", "", "没有结果（trufor 阶段没跑成）。", ""]
        return
    rows = json.loads(path.read_text(encoding="utf-8"))["frames"]
    lines += ["## TruFor 篡改定位", "", "score = 整图完整性分（越高越像被篡改）；峰值命中 = 热力图最大值落在被改位置（框外扩 2%）。", "",
              "| 样本 | 类型 | 正 | 负 | score AUROC | map_p99 AUROC | 峰值命中 | 框面积占比 | 前 1% 热点落在框内 | 框内 / 框外均值 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted({r["sample"] for r in rows if r["kind"] in ("insert", "edit")}):
        pos = [r for r in rows if r["sample"] == s and r["role"] == "pos"]
        neg = [r for r in rows if r["sample"] == s and r["role"] == "neg"]
        boxed = [r for r in pos if "peak_in_box" in r]
        hit = f"{sum(r['peak_in_box'] for r in boxed)}/{len(boxed)}" if boxed else "—"
        frac = fmt(median([r["box_frac"] for r in boxed])) if boxed else "—"
        top = fmt(median([r["top1_in_box"] for r in boxed])) if boxed else "—"
        ratio = fmt(median([r["mean_in_box"] / max(1e-4, r["mean_out_box"]) for r in boxed
                            if r.get("mean_in_box") is not None and r.get("mean_out_box") is not None]))
        lines.append(f"| {s} | {KIND[s]} | {len(pos)} | {len(neg)} | {fmt(auroc([r['score'] for r in pos], [r['score'] for r in neg]))} | "
                     f"{fmt(auroc([r['map_p99'] for r in pos], [r['map_p99'] for r in neg]))} | {hit} | {frac} | {top} | {ratio} |")
    pos = [r for r in rows if r["kind"] == "edit" and r["role"] == "pos"]
    real = [r for r in rows if r["role"] == "real"]
    orig = [r for r in rows if r["role"] == "orig"]
    lines += ["", f"- 被改帧 vs 真实视频帧 score AUROC {fmt(auroc([r['score'] for r in pos], [r['score'] for r in real]))}；"
              f"中位 score：被改 {fmt(median([r['score'] for r in pos]), 3)}，原片同一帧 {fmt(median([r['score'] for r in orig]), 3)}，"
              f"真实视频 {fmt(median([r['score'] for r in real]), 3)}", ""]


def summary_faces(lines: list[str]) -> None:
    path = OUT / "faces.json"
    if not path.is_file():
        lines += ["## 换脸：Effort（FF++）", "", "没有结果（faces 阶段没跑成）。", ""]
        return
    rows = json.loads(path.read_text(encoding="utf-8"))["faces"]
    lines += ["## 换脸：Effort（FF++ 训练，CLIP-L/14）对照 DFDC 冠军 B7", "",
              "swapped = 与原片同位置身份相似度 < 0.45 的脸；kept = 换脸视频里没换的人（≥ 0.6）；"
              "real_other = 插入 / 局部改动视频里与原片一致的真人脸；real_video = 真实开发视频里的脸（没有 DFDC 分）。", "",
              "| 样本 | swapped 张数 | kept 张数 | Effort 中位 swapped / kept | Effort AUROC | DFDC 中位 swapped / kept | DFDC AUROC |",
              "|---|---|---|---|---|---|---|"]
    for s in SWAP:
        sw = [r for r in rows if r["sample"] == s and r["label"] == "swapped"]
        kp = [r for r in rows if r["sample"] == s and r["label"] == "kept"]
        lines.append(f"| {s} | {len(sw)} | {len(kp)} | {fmt(median([r['effort_ffpp'] for r in sw]))} / {fmt(median([r['effort_ffpp'] for r in kp]))} | "
                     f"{fmt(auroc([r['effort_ffpp'] for r in sw], [r['effort_ffpp'] for r in kp]))} | "
                     f"{fmt(median([r['dfdc'] for r in sw]))} / {fmt(median([r['dfdc'] for r in kp]))} | "
                     f"{fmt(auroc([r['dfdc'] for r in sw], [r['dfdc'] for r in kp]))} |")
    sw = [r for r in rows if r["label"] == "swapped"]
    lines += ["", "| 对照组 | 张数 | Effort AUROC（swapped vs 该组） | DFDC AUROC | Effort 中位 | 该组 > 0.5 的比例（Effort） |", "|---|---|---|---|---|---|"]
    for label in ("kept", "real_other", "real_video"):
        grp = [r for r in rows if r["label"] == label]
        lines.append(f"| {label} | {len(grp)} | {fmt(auroc([r['effort_ffpp'] for r in sw], [r['effort_ffpp'] for r in grp]))} | "
                     f"{fmt(auroc([r['dfdc'] for r in sw], [r['dfdc'] for r in grp]))} | {fmt(median([r['effort_ffpp'] for r in grp]), 3)} | "
                     f"{fmt(sum(r['effort_ffpp'] > 0.5 for r in grp) / len(grp) if grp else None)} |")
    lines.append(f"| swapped 自身 | {len(sw)} | — | — | {fmt(median([r['effort_ffpp'] for r in sw]), 3)} | "
                 f"{fmt(sum(r['effort_ffpp'] > 0.5 for r in sw) / len(sw) if sw else None)} |")
    lines.append("")


def summary_skyra(lines: list[str]) -> None:
    path = OUT / "skyra.json"
    if not path.is_file() and (OUT / "skyra.partial.jsonl").is_file():
        rows = [json.loads(line) for line in (OUT / "skyra.partial.jsonl").read_text(encoding="utf-8").splitlines()]
        note = "（未跑完，部分结果）"
    elif path.is_file():
        rows, note = json.loads(path.read_text(encoding="utf-8"))["chunks"], ""
    else:
        lines += ["## Skyra-RL（AI 视频检测）", "", "没有结果（skyra 阶段没跑成）。", ""]
        return
    lines += [f"## Skyra-RL（AI 视频检测，按镜头 ≤10 s 一段，每段 16 帧）{note}", "",
              "p_fake = 在 <answer> 处 Fake 对 Real 的概率；判 Fake = 模型自己的结论。", "",
              "| 样本 | 类型 | 段 | 正 | 同视频 AUROC | 正例判 Fake | 负例判 Fake | 正例在 p_fake 排名（前几） |", "|---|---|---|---|---|---|---|---|"]
    for s in INSERT + EDIT + SKYRA_REAL:
        rs = [r for r in rows if r["sample"] == s]
        if not rs:
            continue
        pos = [r for r in rs if r["label"] == "pos"]
        neg = [r for r in rs if r["label"] in ("neg", "real")]
        order = sorted(rs, key=lambda r: -r["p_fake"])
        ranks = sorted(order.index(r) + 1 for r in pos)
        fake = lambda grp: f"{sum('fake' in r['verdict'].lower() for r in grp)}/{len(grp)}" if grp else "—"  # noqa: E731
        lines.append(f"| {s} | {KIND[s]} | {len(rs)} | {len(pos)} | {fmt(auroc([r['p_fake'] for r in pos], [r['p_fake'] for r in neg]))} | "
                     f"{fake(pos)} | {fake(neg)} | {ranks[:6]} |")
    ins = [r for r in rows if r["kind"] == "insert" and r["label"] == "pos"]
    real = [r for r in rows if r["kind"] == "real"]
    neg_in = [r for r in rows if r["kind"] in ("insert", "edit") and r["label"] == "neg"]
    lines += ["", f"- 插入段 vs 真实视频段 AUROC {fmt(auroc([r['p_fake'] for r in ins], [r['p_fake'] for r in real]))}；"
              f"插入段判 Fake {sum('fake' in r['verdict'].lower() for r in ins)}/{len(ins)}；"
              f"真实视频段判 Fake {sum('fake' in r['verdict'].lower() for r in real)}/{len(real)}；"
              f"伪造视频里未改段判 Fake {sum('fake' in r['verdict'].lower() for r in neg_in)}/{len(neg_in)}",
              f"- 每段平均 {fmt(sum(r['seconds'] for r in rows) / max(1, len(rows)), 1)} s；没给出 <answer> 的 {sum(not r['answer_found'] for r in rows)} 段", ""]


def stage_summary() -> None:
    lines = [f"# 文献检测方法在 EVID 开发样本上的实测（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             "样本：已读过的 18 个 AI 生成开发样本 + 6 个 002–130 真实开发视频。标签来自 10-10 与 GT 原片的离线对齐。",
             "不跑 agent，只看检测器本身有没有信号。", ""]
    for fn in (summary_frames, summary_trufor, summary_faces, summary_skyra):
        try:
            fn(lines)
        except Exception as exc:  # noqa: BLE001
            lines += [f"!!! {fn.__name__} 出错：{type(exc).__name__}: {exc}", ""]
            log(traceback.format_exc())
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"汇总写在 {OUT / 'summary.md'}")


# ============================================================================ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="*", default=STAGES)
    parser.add_argument("--stage", help="内部用：在子进程里跑单个阶段")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage:
        fn = {"prep": stage_prep, "frames": stage_frames, "trufor": stage_trufor, "faces": stage_faces,
              "skyra": stage_skyra, "summary": stage_summary}[args.stage]
        fn()
        return 0
    log(f"==== 文献检测方法实测：{args.stages} ====")
    status = {}
    for stage in [s for s in STAGES if s in args.stages]:
        target = OUT / f"{stage}.json"
        if stage not in ("prep", "summary") and target.is_file() and not args.redo:
            log(f"[{stage}] 已有结果，跳过（--redo 重算）")
            status[stage] = "skipped"
            continue
        if args.redo and stage == "skyra":
            (OUT / "skyra.partial.jsonl").unlink(missing_ok=True)
        log(f"[{stage}] 开始")
        started = time.monotonic()
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--stage", stage], cwd=str(ROOT))
        status[stage] = "ok" if code == 0 else f"exit {code}"
        log(f"[{stage}] 结束：{status[stage]}，{(time.monotonic() - started) / 60:.1f} 分钟")
    log(f"全部结束：{status}；汇总 {OUT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
