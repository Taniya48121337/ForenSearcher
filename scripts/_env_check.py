"""环境检查：Python、关键包、GPU、Ollama、Codex、端口。结果写到项目根目录 _环境检查结果.txt。"""
import importlib, importlib.util, json, os, shutil, subprocess, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lines = []
def say(*a): lines.append(" ".join(str(x) for x in a))
def run(cmd, timeout=60):
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=isinstance(cmd, str))
        text = (out.stdout or b"") + (out.stderr or b"")
        for enc in ("utf-8", "utf-16-le", "gbk"):
            try:
                t = text.decode(enc)
                if "\x00" not in t:
                    return t.strip()
            except UnicodeDecodeError:
                pass
        return text.decode("utf-8", "replace").replace("\x00", "").strip()
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"

say("=== python", sys.executable, sys.version.split()[0])
for m in ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "httpx", "av", "cv2", "PIL", "ddgs", "pytest", "vllm"]:
    if importlib.util.find_spec(m) is None:
        say(f"  {m}: MISSING")
    else:
        try:
            say(f"  {m}: {getattr(importlib.import_module(m), '__version__', '?')}")
        except Exception as exc:
            say(f"  {m}: import error {exc}")
try:
    import torch
    say("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}GB" if torch.cuda.is_available() else "")
except Exception as exc:
    say("  torch cuda check failed", exc)
say("=== nvidia-smi"); say(run("nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv"))
say("=== ollama", shutil.which("ollama")); say(run("ollama --version")); say(run("ollama list"))
say("=== ollama api")
for url in ["http://127.0.0.1:11434/v1/models", "http://127.0.0.1:8000/v1/models", "http://127.0.0.1:1234/v1/models"]:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            say(" ", url, "->", r.read()[:500].decode("utf-8", "replace"))
    except Exception as exc:
        say(" ", url, "->", type(exc).__name__)
say("=== codex", shutil.which("codex")); say(run("codex --version"))
say("=== wsl"); say(run("wsl -l -v", timeout=20))
say("=== models dir"); say([p.name for p in (ROOT / ".models").iterdir()] if (ROOT / ".models").is_dir() else "none")
say("=== disk"); say(run("wmic logicaldisk get caption,freespace,size"))
say("=== FOREN env", {k: v for k, v in os.environ.items() if k.startswith("FOREN")})
(ROOT / "_环境检查结果.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
