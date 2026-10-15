"""Step 5: pre-flight test on the server. Every module the baselines use is exercised once, with a PASS / WARN / FAIL
line, timing and a fix hint. Ends with a one-sample end-to-end run per dataset (scored with the real judge).

    bash server/05_preflight.sh              # everything (about 25-40 minutes, mostly the end-to-end runs)
    bash server/05_preflight.sh --quick      # skip the end-to-end runs (about 5 minutes)
    bash server/05_preflight.sh --only net   # one section: env files tools llm net detect judge tests e2e

Report: server_logs/preflight_report.md (+ .json). A FAIL means the baselines would break or silently lose a tool;
fix it before step 6. A WARN means the run works but something is weaker than on the local PC.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LOG_DIR = Path(os.environ.get("FS_LOG_DIR") or ROOT / "server_logs")
ENDPOINT = os.environ.get("FS_ENDPOINT", "http://127.0.0.1:8000/v1")
SERVED = os.environ.get("FS_SERVED_NAME", "qwen38-27b")
RESULTS: list[dict] = []
COLORS = {"PASS": "\033[1;32m", "WARN": "\033[1;33m", "FAIL": "\033[1;31m", "INFO": "\033[1;34m"}


def report(section: str, name: str, status: str, detail: str = "", hint: str = "", seconds: float | None = None) -> None:
    RESULTS.append({"section": section, "check": name, "status": status, "detail": detail, "hint": hint, "seconds": seconds})
    took = f" ({seconds:.1f} s)" if seconds is not None else ""
    print(f"{COLORS.get(status, '')}[{status}]\033[0m {section} | {name}{took}: {detail}", flush=True)
    if hint and status in ("FAIL", "WARN"):
        print(f"        -> {hint}", flush=True)


def check(section: str, name: str, hint: str = ""):
    """Decorator: run the function, which returns (status, detail) or raises -> FAIL with the exception."""

    def wrap(fn):
        def run():
            t = time.time()
            try:
                status, detail = fn()
            except Exception as exc:  # noqa: BLE001
                status, detail = "FAIL", f"{type(exc).__name__}: {str(exc)[:400]}"
                (LOG_DIR / "preflight_tracebacks.log").open("a", encoding="utf-8").write(
                    f"\n==== {section} | {name}\n{traceback.format_exc()}")
            report(section, name, status, detail, hint, time.time() - t)
            return status
        run.section = section
        return run
    return wrap


def tool_gpu_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("FS_TOOL_GPUS", "6,7")
    return env


def make_png(color=(220, 30, 30), text_shape="circle") -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((100, 100, 348, 348), fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ======================================================================================== A. environment

@check("env", "python and packages", "rerun bash server/01_setup_env.sh; see server_logs/pip_agent.log")
def env_packages():
    import importlib

    names = ["httpx", "av", "cv2", "numpy", "PIL", "fitz", "trafilatura", "lxml", "yt_dlp", "ddgs", "faster_whisper",
             "torch", "torchvision", "transformers", "huggingface_hub", "pytest", "foren_searcher"]
    got, bad = [], []
    for n in names:
        try:
            m = importlib.import_module(n)
            got.append(f"{n} {getattr(m, '__version__', '')}".strip())
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{n} ({type(exc).__name__})")
    if sys.version_info < (3, 11):
        bad.append(f"python {sys.version.split()[0]} < 3.11")
    return ("FAIL" if bad else "PASS"), (f"missing: {', '.join(bad)}; " if bad else "") + "; ".join(got[:8]) + " ..."


@check("env", "GPUs for the tools", "set FS_TOOL_GPUS in server/env.sh to GPUs not used by vLLM")
def env_gpu():
    import torch

    if not torch.cuda.is_available():
        return "FAIL", f"torch sees no CUDA GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
    rows = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        rows.append(f"cuda:{i} {torch.cuda.get_device_name(i)} free {free / 2**30:.0f}/{total / 2**30:.0f} GB")
    x = torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")
    low = [r for r in rows if float(r.split("free ")[1].split("/")[0]) < 30]
    return ("WARN" if low else "PASS"), f"tool GPUs (physical {os.environ.get('CUDA_VISIBLE_DEVICES')}): " + "; ".join(rows) + \
        (" — under 30 GB free: 16 parallel runs need ~3 GB each" if low else "")


@check("env", "disk space", "free space: runs keep downloaded original videos (~1-2 GB per EVID run)")
def env_disk():
    free = shutil.disk_usage(ROOT).free / 2**30
    return ("PASS" if free >= 300 else "WARN" if free >= 120 else "FAIL"), f"{free:.0f} GB free at {ROOT}"


@check("env", "ffmpeg (optional)", "sudo apt install ffmpeg")
def env_ffmpeg():
    path = shutil.which("ffmpeg")
    return ("PASS", path) if path else ("WARN", "not found: YouTube video+audio are merged with PyAV instead (works, slower)")


# ======================================================================================== B. files

@check("files", "code, specs, configs", "unpack the whole package into FS_ROOT")
def files_code():
    need = ["specs/hi@0.7.0/spec.json", "specs/gia@0.8.0/spec.json", "specs/hi@0.8.0/spec.json", "specs/gia@0.9.0/spec.json",
            "configs/evid_server_27b.toml",
            ".models/deepfake/pylib/facenet_pytorch/__init__.py", "scripts/_evid_score.py", "scripts/_bench_common.py",
            "src/foren_searcher/datagen/run_cases.py"]
    missing = [p for p in need if not (ROOT / p).exists()]
    return ("FAIL" if missing else "PASS"), (f"missing {missing}" if missing else f"{len(need)} key files present")


@check("files", "EVID-Bench videos and cases", "bash server/03_download_data.sh")
def files_evid():
    root = ROOT / "plan-harness-1/datasets/EVID-Bench"
    entries = json.loads((root / "evid_bench.json").read_text(encoding="utf-8"))
    vids = [e["id"] for e in entries if (root / "videos" / f"{e['id']}.mp4").is_file()]
    cases = list((ROOT / "experiments/evid/cases").glob("evid_video_*.json"))
    ok = len(vids) == len(entries) == 222 and len(cases) == 222
    return ("PASS" if ok else "FAIL"), f"videos {len(vids)}/{len(entries)}, cases {len(cases)}/222"


@check("files", "GroundLie360 / FMNV videos", "copy the videos in (README step 3b); EVID can run without them")
def files_bench():
    import _bench_common as B

    parts, low = [], False
    for ds in ("g360", "fmnv"):
        ids = {s.sample_id for s in B.LOADERS[ds]()}
        n = len(ids & set(B.video_index(ds)))
        need = int(os.environ.get("FS_BENCH_N", 200))
        low |= n < need
        parts.append(f"{B.DIRS[ds]} {n} matched (run wants {need})")
    return ("WARN" if low else "PASS"), "; ".join(parts)


# ======================================================================================== C. tool models

def _inspect_dir(name: str) -> Path:
    return ROOT / ".models/inspect" / name


@check("tools", "INSPECT model operations enabled", "bash server/02_download_models.sh")
def tools_enabled():
    from foren_searcher.datagen.run_cases import resolve_inspect_operations

    ops, detail = resolve_inspect_operations("auto", ROOT / ".models")
    model_ops = detail.get("model_ops", {})
    off = {k: v.get("reason") for k, v in model_ops.items() if not v.get("enabled")}
    return ("FAIL" if off else "PASS"), (f"disabled: {off}" if off else f"enabled: {', '.join(model_ops)} (+ {len(ops) - len(model_ops)} base ops)")


@check("tools", "speech-to-text (faster-whisper on GPU)", "cuDNN/cuBLAS libraries: source server/env.sh puts torch's nvidia libs on LD_LIBRARY_PATH; "
       "or pip install nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 in .venv-agent")
def tools_asr():
    import numpy as np
    import wave
    from faster_whisper import WhisperModel

    tmp = Path(tempfile.mkdtemp()) / "tone.wav"
    sr = 16000
    sig = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr) * 32767).astype("int16")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(sig.tobytes())
    model = WhisperModel(str(_inspect_dir("faster-whisper-small")), device="cuda", compute_type="float16")
    segments, info = model.transcribe(str(tmp))
    list(segments)
    return "PASS", f"loaded on cuda, transcribed 2 s test audio (language guess {info.language})"


@check("tools", "frame search (CLIP ViT-B/32)", "bash server/02_download_models.sh")
def tools_clip():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    path = _inspect_dir("clip-vit-base-patch32")
    model = CLIPModel.from_pretrained(str(path)).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(str(path))
    img = Image.open(io.BytesIO(make_png()))
    inputs = proc(text=["a red circle", "a blue square"], images=img, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        probs = model(**inputs).logits_per_image.softmax(-1)[0].tolist()
    return ("PASS" if probs[0] > probs[1] else "WARN"), f"'a red circle' {probs[0]:.2f} vs 'a blue square' {probs[1]:.2f}"


@check("tools", "region search (GroundingDINO-tiny)", "bash server/02_download_models.sh; transformers >= 4.57")
def tools_gdino():
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    path = _inspect_dir("grounding-dino-tiny")
    proc = AutoProcessor.from_pretrained(str(path))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(path)).to("cuda").eval()
    img = Image.open(io.BytesIO(make_png()))
    inputs = proc(images=img, text="a red circle.", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model(**inputs)
    best = float(out.logits.sigmoid().max())
    return ("PASS" if best > 0.3 else "WARN"), f"best box score {best:.2f} for 'a red circle'"


@check("tools", "FaceNet (compare_media face comparison)", "bash server/02_download_models.sh (FaceNet weights step)")
def tools_facenet():
    from foren_searcher.datagen.run_cases import add_face_pylib

    add_face_pylib(ROOT / ".models")
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

    hub = Path(torch.hub.get_dir()) / "checkpoints" / "20180402-114759-vggface2.pt"
    if not hub.is_file():
        return "FAIL", f"weights not at {hub}"
    net = InceptionResnetV1(pretrained="vggface2").to("cuda").eval()
    MTCNN(keep_all=True, device="cuda")
    with torch.inference_mode():
        e = net(torch.zeros(1, 3, 160, 160, device="cuda"))
    return "PASS", f"MTCNN + InceptionResnetV1(vggface2) on cuda, embedding {tuple(e.shape)}; weights {hub.stat().st_size / 1e6:.0f} MB"


# ======================================================================================== D. model server

def _chat(body: dict, timeout: float = 300) -> dict:
    import httpx

    r = httpx.post(ENDPOINT + "/chat/completions", json={"model": SERVED, **body}, timeout=timeout, trust_env=False)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


@check("llm", "server lists the model", "bash server/04_start_vllm.sh; tail server_logs/vllm.log")
def llm_models():
    import httpx

    data = httpx.get(ENDPOINT + "/models", timeout=10, trust_env=False).json()
    names = [m["id"] for m in data.get("data", [])]
    ml = [m.get("max_model_len") for m in data.get("data", [])]
    return ("PASS" if SERVED in names else "FAIL"), f"{ENDPOINT}: {names} max_model_len {ml}"


@check("llm", "JSON answer with thinking off", "the agent sends chat_template_kwargs.enable_thinking=false; vLLM >= 0.17 needed")
def llm_text():
    t = time.time()
    m = _chat({"temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": 'Return exactly {"ok": true} and nothing else.'}]})["choices"][0]["message"]
    content = (m.get("content") or "").strip()
    think = m.get("reasoning_content") or m.get("reasoning") or ("<think>" in content)
    ok = "ok" in content and not think
    return ("PASS" if ok else "WARN"), f"{content!r} in {time.time() - t:.1f} s" + ("; THINKING TEXT PRESENT (slow)" if think else "")


@check("llm", "image input", "--limit-mm-per-prompt in 04_start_vllm.sh; the model must be the vision version")
def llm_image():
    img = "data:image/png;base64," + base64.b64encode(make_png()).decode()
    m = _chat({"temperature": 0, "max_tokens": 20, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [{"type": "text", "text": "What colour is the circle? One word."},
                                                         {"type": "image_url", "image_url": {"url": img}}]}]})
    ans = (m["choices"][0]["message"].get("content") or "").strip()
    return ("PASS" if "red" in ans.lower() else "FAIL"), f"answer {ans!r} (expected red)"


@check("llm", "8 images in one request (agent sends up to 8)", "raise image limit in --limit-mm-per-prompt")
def llm_many_images():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(make_png()).decode()}}
    t = time.time()
    m = _chat({"temperature": 0, "max_tokens": 10, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": [{"type": "text", "text": "How many images? Number only."}, *[img] * 8]}]})
    u = m.get("usage", {})
    return "PASS", f"{(m['choices'][0]['message'].get('content') or '').strip()!r}, prompt tokens {u.get('prompt_tokens')}, {time.time() - t:.1f} s"


@check("llm", "JSON-schema constrained output", "vLLM structured outputs; the agent falls back to json_object if refused")
def llm_schema():
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["Real", "Fake"]}}, "required": ["verdict"]}
    m = _chat({"temperature": 0, "max_tokens": 50, "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
               "messages": [{"role": "user", "content": "Is the sky green? Answer Real or Fake in JSON."}]})
    data = json.loads(m["choices"][0]["message"]["content"])
    return "PASS", f"{data}"


@check("llm", "16 requests in parallel (throughput)", "more vLLM replicas (FS_VLLM_GPUS) or fewer FS_JOBS")
def llm_parallel():
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        t = time.time()
        _chat({"temperature": 0.6, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": f"Write two sentences about video number {i}."}]})
        return time.time() - t

    t = time.time()
    with ThreadPoolExecutor(16) as pool:
        times = list(pool.map(one, range(16)))
    wall = time.time() - t
    return ("PASS" if wall < 60 else "WARN"), f"16 answers in {wall:.1f} s (slowest {max(times):.1f} s)"


# ======================================================================================== E. network

def _proxy_note() -> str:
    return f"proxy {os.environ.get('FOREN_HTTP_PROXY') or 'none'}"


@check("net", "web page fetch (agent's HTTP client)", "set FS_PROXY in server/env.sh if the server needs a proxy")
def net_http():
    from foren_searcher.search.openers.http import SafeHttpClient
    from foren_searcher.search.policy import SearchPolicy

    async def go():
        c = SafeHttpClient(SearchPolicy(source_timeout_s=20))
        try:
            out = []
            for url in ("https://www.example.com/", "https://en.wikipedia.org/wiki/Video"):
                r = await c.fetch(url, maximum_bytes=5_000_000)
                out.append(f"{url} -> {r.status_code}, {len(r.content) // 1024} KB")
            return out
        finally:
            await c.aclose()

    return "PASS", "; ".join(asyncio.run(go())) + f" ({_proxy_note()})"


@check("net", "YouTube search (yt-dlp, the agent's video search)", "YouTube must be reachable (FS_PROXY); without it EVID originals cannot be found")
def net_ytsearch():
    from foren_searcher.search.contracts import SearchQueryRequest
    from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider

    req = SearchQueryRequest(input_type="text", query="apollo 11 moon landing original footage", output_type="video")
    batch = asyncio.run(YtDlpSearchProvider().search(req, offset=0, limit=5))
    titles = [c.search_title for c in batch.candidates][:3]
    return ("PASS" if batch.candidates else "FAIL"), f"{len(batch.candidates)} results, e.g. {titles} ({_proxy_note()})"


@check("net", "YouTube download (original-video path)", "YouTube 403 / bot check: update yt-dlp (pip install -U yt-dlp in .venv-agent) or use a proxy")
def net_ytdownload():
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy
    import cv2

    out = []
    tmp = Path(tempfile.mkdtemp())
    for label, kw in (("full (INSPECT)", {}), ("360p video-only (SEARCH content match / INSPECT fallback)", {"video_only_height": 360})):
        t = time.time()
        got = ytdlp_download("https://www.youtube.com/watch?v=jNQXAC9IVRw", tmp / f"{len(out)}.media", InspectPolicy(), **kw)
        cap = cv2.VideoCapture(str(got.path)); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.release()
        out.append(f"{label}: {got.details.get('method')}, {got.path.stat().st_size // 1024} KB, {int(n)} frames, {time.time() - t:.0f} s")
    shutil.rmtree(tmp, ignore_errors=True)
    return "PASS", "; ".join(out)


@check("net", "reverse image search engines (SEARCH image input)",
       "put keys in .local/: google_vision_key.txt, serpapi_key.txt (or serper_key.txt), tineye_key.txt; Yandex needs none")
def net_image_engines():
    from foren_searcher.datagen.run_cases import resolve_image_engines

    engines = resolve_image_engines("auto", ROOT)
    missing = [e for e in ("gcv", "lens", "tineye") if e not in engines]
    return ("PASS" if not missing else "WARN"), (f"engines {engines}" + (f"; no key for {missing}" if missing else ""))


@check("net", "DuckDuckGo search (fallback engine)", "DuckDuckGo blocked: set FS_PROXY; the agent still has YouTube search")
def net_ddgs():
    from foren_searcher.search.providers.ddgs import DdgsProvider
    from foren_searcher.search.contracts import SearchQueryRequest

    res = []
    for out_type in ("text", "video"):
        req = SearchQueryRequest(input_type="text", query="NASA Apollo 11 landing", output_type=out_type)
        batch = asyncio.run(DdgsProvider().search(req, offset=0, limit=5))
        res.append(f"{out_type}: {len(batch.candidates)}")
    return ("PASS" if all(not r.endswith(": 0") for r in res) else "WARN"), ", ".join(res)


@check("net", "fact-check sites are blocked (benchmark rule)", "retrieval guard blocklist")
def net_guard():
    from foren_searcher.search import retrieval_guard as g

    text = Path(g.__file__).read_text(encoding="utf-8")
    ok = "snopes.com" in text and "politifact.com" in text
    return ("PASS" if ok else "FAIL"), "snopes.com, politifact.com ... on the blocklist (runs use --blocklist exclude)"


# ======================================================================================== F. compare (was DETECT)

@check("detect", "compare_media on a synthetic pair (GPU faces)", "see server_logs/preflight_tracebacks.log")
def detect_compare():
    import cv2
    import numpy as np
    from foren_searcher.datagen.run_cases import add_face_pylib
    from foren_searcher.inspect.source_compare import FaceIdentity, SourceComparer

    add_face_pylib(ROOT / ".models")
    tmp = Path(tempfile.mkdtemp())

    def scene(t):
        img = np.zeros((180, 320, 3), np.uint8); img[:] = (60 + (t // 8) % 60, 90, 120)
        for k in range(4):
            cv2.circle(img, (int((k * 77 + t * 2 * (k + 1)) % 320), int((k * 41 + t) % 180)), 14 + 4 * k, (250 - 40 * k, 40 * k, 120), -1)
        return img

    def write(path, frames):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
        for f in frames:
            w.write(f)
        w.release()

    orig = [scene(t) for t in range(25 * 24)]
    rng = np.random.default_rng(0)
    fake = orig[:200] + [(rng.random((180, 320, 3)) * 40).astype(np.uint8) for _ in range(125)] + orig[200:500]
    write(tmp / "a.mp4", fake); write(tmp / "b.mp4", orig)
    res, images = SourceComparer(FaceIdentity("cuda")).compare(tmp / "a.mp4", tmp / "b.mp4")
    shutil.rmtree(tmp, ignore_errors=True)
    ok = any("Not in B" in x for x in res["summary_lines"])        # compare_media's wording for footage only in A
    return ("PASS" if ok else "FAIL"), f"{res['summary_lines'][:2]}; {len(images)} images; {res['timings_s']}"


@check("detect", "image description through the model server",
       "compare_media names changed objects with a fixed VLM: FOREN_DESCRIBE_ENDPOINT / FOREN_DESCRIBE_MODEL, else the run's model")
def detect_describer():
    from foren_searcher.inspect.describe import EndpointDescriber

    d = EndpointDescriber.from_env() or EndpointDescriber(endpoint=ENDPOINT, model=SERVED, thinking=False)
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(make_png(), np.uint8), cv2.IMREAD_COLOR)
    ans = d.describe(cv2.imencode(".jpg", img)[1].tobytes(), "What object is in the image and what colour is it? One short line.")
    return ("PASS" if ans else "FAIL"), repr(ans[:120])


# ======================================================================================== G. judge

@check("judge", "Claude CLI judge (Sonnet)", "install Node 18+ and `npm i -g @anthropic-ai/claude-code`, then `claude` once to log in "
       "(or export ANTHROPIC_API_KEY). Without it, score on the PC: bash server/08_export_for_scoring.sh")
def judge_claude():
    from foren_searcher.datagen.verify.judges import build_judge

    judge = build_judge("claude-cli", model="sonnet")
    t = time.time()
    value = asyncio.run(judge.judge(system_prompt="Return JSON only.", payload={"Q": 'Reply with {"ok": true}'}, images=[],
                                    schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}))
    return ("PASS" if isinstance(value, dict) and value.get("ok") is True else "WARN"), f"{value} in {time.time() - t:.0f} s ({judge.name})"


# ======================================================================================== H. unit tests

@check("tests", "unit tests (compare_media, source match, benchmarks, downloads)", "see server_logs/preflight_pytest.log")
def unit_tests():
    tests = ["tests/test_inspect_compare_media_20261012.py", "tests/test_search_source_match_20261012.py",
             "tests/test_merged_catalog_20261012.py", "tests/test_bench_20261012.py", "tests/test_evidb7_smoke_fixes_20261005.py"]
    base = LOG_DIR / "pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                           f"--basetemp={base / str(os.getpid())}"], cwd=ROOT, capture_output=True, text=True, env=tool_gpu_env())
    (LOG_DIR / "preflight_pytest.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    last = (done.stdout.strip().splitlines() or ["?"])[-1]
    return ("PASS" if done.returncode == 0 else "FAIL"), last


# ======================================================================================== I. end to end

SMOKE_ROUND = "srvsmoke"


def _run_cases(cases_dir: Path, runs_dir: Path, ids: list[str], snapshot_root: Path, log: Path) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "foren_searcher.datagen.run_cases", "--cases", str(cases_dir), "--runs-dir", str(runs_dir),
           "--split", "test", "--round", SMOKE_ROUND, "--endpoint", ENDPOINT, "--model", SERVED, "--no-think",
           "--jobs", "1", "--instruction-mode", "student",
           "--retry-failed", "--blocklist", "exclude", "--hi-spec", "hi@0.8.0", "--gia-spec", "gia@0.9.0",
           "--search-mode", "replay_or_record", "--snapshot-root", str(snapshot_root), "--runtime-guards", "eval",
           "--state-renderer", "compact_16k", "--ids", *ids]
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, env=tool_gpu_env())


def _inspect_run(run: Path) -> tuple[str, str]:
    if not (run / "result.json").is_file():
        return "FAIL", f"no result.json in {run} (see the smoke log)"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    decision = (result.get("decision") or {})
    cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
    traj = (run / "trajectory.jsonl").read_text(encoding="utf-8", errors="replace") if (run / "trajectory.jsonl").is_file() else ""
    actions = {a: traj.count(f'"action": "{a}"') + traj.count(f'"action":"{a}"') for a in ("SEARCH", "INSPECT")}
    ops = {k: v.get("enabled") for k, v in ((cfg.get("inspect_operations_detail") or {}).get("model_ops") or {}).items()}
    det = cfg.get("inspect") or {}
    match = cfg.get("search_match") or {}
    snap = (cfg.get("search_snapshot") or {}).get("counts")
    status = "PASS" if decision.get("decision") == "FINISH" and str(decision.get("answer") or "").strip() else "WARN"
    return status, (f"decision {decision.get('decision')}, answer {len(str(decision.get('answer') or ''))} chars; actions {actions}; "
                    f"model ops {ops}; compare calls {det.get('calls')}, descriptions {(det.get('describe') or {}).get('calls')}; "
                    f"source matches {match.get('by_status')}; image search {(cfg.get('image_search') or {}).get('engine')}; "
                    f"search snapshot {snap}")


def e2e() -> None:
    import _bench_build_cases as BC
    import _bench_common as B
    from foren_searcher.datagen.run_cases import _safe

    work = LOG_DIR / "smoke"
    snaps = work / "snapshots"
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    t0 = time.time()
    # EVID: a dev-range sample (002-130 is not part of the 131-222 test runs)
    evid_runs = ROOT / "experiments/evid/runs"
    procs.append(("evid", "video_002", _run_cases(ROOT / "experiments/evid/cases", evid_runs, ["evid:video_002"], snaps,
                                                  work / "evid.log"), evid_runs / f"evid_video_002_{SMOKE_ROUND}_1"))
    for ds in ("g360", "fmnv"):
        samples = B.LOADERS[ds]()
        vids = B.video_index(ds)
        pick = next((s for s in samples if s.label == "Fake" and s.sample_id in vids), None)
        if pick is None:
            report("e2e", f"{B.DIRS[ds]} one sample", "WARN", "no video on disk, skipped")
            continue
        cdir = work / f"{ds}_cases"
        cdir.mkdir(exist_ok=True)
        case = BC.build(pick, vids[pick.sample_id], cdir)
        (cdir / f"{ds}_{pick.sample_id}.json").write_text(json.dumps(case.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        rdir = work / f"{ds}_runs"
        procs.append((ds, pick.sample_id, _run_cases(cdir, rdir, [pick.case_id], snaps, work / f"{ds}.log"),
                      rdir / f"{_safe(ds)}_{_safe(pick.sample_id)}_{SMOKE_ROUND}_1"))
    print(f"        end-to-end: {len(procs)} runs started in parallel (10-25 min); logs in {work}", flush=True)
    while any(p.poll() is None for _, _, p, _ in procs):
        time.sleep(30)
        print(f"        ... {int(time.time() - t0)} s, running: {[d for d, _, p, _ in procs if p.poll() is None]}", flush=True)
    for ds, sid, proc, run in procs:
        status, detail = _inspect_run(run)
        report("e2e", f"{ds} {sid} full run", status, detail, f"log {work / (ds + '.log')}; run dir {run}", time.time() - t0)
        if ds in ("g360", "fmnv") and (run / "result.json").is_file():
            ans = str((json.loads((run / "result.json").read_text(encoding="utf-8")).get("decision") or {}).get("answer") or "")
            p = B.parse_answer(ds, ans)
            report("e2e", f"{ds} answer lines parsed", "PASS" if p["parsed"] else "WARN",
                   f"verdict {p.get('verdict')}, {({k: v for k, v in p.items() if k not in ('parsed', 'verdict')})}",
                   "unparsed answers are read by the Sonnet extractor at scoring time")
    # scoring the EVID smoke run with the real judge
    t = time.time()
    done = subprocess.run([sys.executable, "scripts/_evid_score.py", "--round", SMOKE_ROUND, "--ids", "video_002",
                           "--judges", "claude-cli:sonnet"], cwd=ROOT, capture_output=True, text=True)
    (work / "score.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    rows = ROOT / "experiments/evid/scores" / f"{SMOKE_ROUND}.sonnet.subset.jsonl"
    if done.returncode == 0 and rows.is_file():
        r = json.loads(rows.read_text(encoding="utf-8").splitlines()[0])
        report("e2e", "EVID scoring with Sonnet", "PASS", f"video_002: {r['passed_gt']}/{r['n_gt']} GT points passed "
               f"(one dev sample; the number itself does not matter)", "", time.time() - t)
    else:
        report("e2e", "EVID scoring with Sonnet", "WARN", "scoring failed (see smoke/score.log)",
               "fix the judge check above, or score on the PC (08_export_for_scoring.sh)", time.time() - t)


SECTIONS = {
    "env": [env_packages, env_gpu, env_disk, env_ffmpeg],
    "files": [files_code, files_evid, files_bench],
    "tools": [tools_enabled, tools_asr, tools_clip, tools_gdino, tools_facenet],
    "llm": [llm_models, llm_text, llm_image, llm_many_images, llm_schema, llm_parallel],
    "net": [net_http, net_ytsearch, net_ytdownload, net_ddgs, net_image_engines, net_guard],
    "detect": [detect_compare, detect_describer],
    "judge": [judge_claude],
    "tests": [unit_tests],
}


def write_report() -> Path:
    counts = {s: sum(1 for r in RESULTS if r["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    lines = [f"# Pre-flight report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"PASS {counts['PASS']}, WARN {counts['WARN']}, FAIL {counts['FAIL']}", "",
             "| section | check | status | seconds | detail | fix |", "|---|---|---|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['section']} | {r['check']} | {r['status']} | {r['seconds'] or 0:.0f} | "
                     f"{r['detail'].replace('|', '/')[:300]} | {r['hint'].replace('|', '/') if r['status'] != 'PASS' else ''} |")
    path = LOG_DIR / "preflight_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOG_DIR / "preflight_report.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the end-to-end runs")
    parser.add_argument("--only", choices=[*SECTIONS, "e2e"], default=None)
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FS_TOOL_GPUS", "6,7")      # tools only on the tool GPUs
    sections = [args.only] if args.only else [*SECTIONS, *([] if args.quick else ["e2e"])]
    for name in sections:
        print(f"\n\033[1;34m==== {name}\033[0m", flush=True)
        if name == "e2e":
            if any(r["status"] == "FAIL" and r["section"] in ("llm", "files", "env") for r in RESULTS):
                report("e2e", "skipped", "FAIL", "earlier FAIL in env / files / llm; fix those first")
                continue
            e2e()
        else:
            for fn in SECTIONS[name]:
                fn()
    path = write_report()
    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    warns = [r for r in RESULTS if r["status"] == "WARN"]
    print(f"\n==== summary: PASS {len(RESULTS) - len(fails) - len(warns)}, WARN {len(warns)}, FAIL {len(fails)}  -> {path}")
    for r in fails:
        print(f"\033[1;31m  FAIL\033[0m {r['section']} | {r['check']}: {r['detail'][:160]}\n        -> {r['hint']}")
    if not fails:
        print("\033[1;32mReady for step 6 (bash server/06_run_baselines.sh).\033[0m" if not args.only else "")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
