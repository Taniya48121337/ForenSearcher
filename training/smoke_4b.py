"""本机 4B 训练链路冒烟的总驱动（清单 E1）。双击 scripts\\_训练链路冒烟_4B.cmd 即调用这里。

顺序：
  0. 显卡占用检查：nvidia-smi 上有别的计算进程（walker 重跑、评测、vLLM、ollama 加载的核验 / 教师模型）或显存已用
     超过阈值就不开跑，列出占用者；本机只有一张卡，冒烟不能和它们抢。--force 可跳过（不建议）。
  1. 权重：.models\\Qwen3.5-4B 不在就问一次是否下载（Hugging Face，失败再试 ModelScope），约 9–10 GB。
  2. 训练：training/sft_lora.py --config training/configs/hi_smoke_4b.json（20 行 HI、40 步、存 LoRA）。
     期间每 5 秒读一次 nvidia-smi，记整卡显存峰值（含 PyTorch 之外的占用）。
  3. 推理 + 核验：training/smoke_infer.py（重新加载底座 + LoRA，对没参与训练的真实状态推理一次，输出过程序核验；
     --compare-base 再关掉 adapter 生成一次作对照）。
  4. 汇总：outputs/lora/hi_smoke_4b/smoke_report.json，屏幕和 _训练链路冒烟_4B日志.txt 各一份。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "_训练链路冒烟_4B日志.txt"
BUSY_PATTERNS = ("walker_run", "run_cases", "_evid", "_bench_run", "_launch", "vllm", "sft_lora", "_merge_ab", "complete_verify",
                 "export_gate", "behavior_metrics", "traj_pilot")


def say(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def nvidia_smi(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def busy_processes() -> list[str]:
    """命令行里带 walker / 评测 / vLLM 字样的 python 进程（Windows 用 PowerShell 读命令行）。"""

    found: list[str] = []
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|vllm|ollama' } | "
               "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"]
        try:
            raw = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace").stdout.strip()
            items = json.loads(raw) if raw else []
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            items = []
        if isinstance(items, dict):
            items = [items]
        me = os.getpid()
        for it in items:
            line = str(it.get("CommandLine") or "")
            if int(it.get("ProcessId") or 0) == me or "smoke_4b.py" in line:
                continue
            if any(p in line for p in BUSY_PATTERNS):
                found.append(f"pid {it.get('ProcessId')}: {line[:200]}")
    else:
        try:
            out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=20).stdout
        except OSError:
            out = ""
        for line in out.splitlines():
            if "smoke_4b.py" not in line and any(p in line for p in BUSY_PATTERNS):
                found.append(line.strip()[:200])
    return found


def ollama_loaded() -> list[str]:
    exe = shutil.which("ollama")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "ps"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    lines = [l for l in out.splitlines()[1:] if l.strip()]
    return lines


def preflight(max_used_mib: int) -> tuple[bool, dict[str, Any]]:
    gpus = nvidia_smi("index,name,memory.used,memory.total")
    apps = nvidia_smi("pid,process_name,used_memory", kind="apps")
    info: dict[str, Any] = {"gpus": gpus, "compute_apps": apps, "busy_processes": busy_processes(), "ollama_loaded": ollama_loaded()}
    problems: list[str] = []
    if not gpus:
        problems.append("nvidia-smi 没有返回显卡信息")
    for row in gpus:
        if len(row) >= 3 and row[2].isdigit() and int(row[2]) > max_used_mib:
            problems.append(f"GPU {row[0]} 已用 {row[2]} MiB（阈值 {max_used_mib} MiB）")
    for row in apps:
        name = row[1].lower() if len(row) > 1 else ""
        if any(k in name for k in ("python", "vllm", "ollama")):
            problems.append(f"显卡上有计算进程：{', '.join(row)}")
    if info["busy_processes"]:
        problems.append("有 walker / 评测 / 训练进程在跑：" + "；".join(info["busy_processes"][:5]))
    if info["ollama_loaded"]:
        problems.append("ollama 里有加载着的模型（占显存）：" + "；".join(info["ollama_loaded"][:3]) + "。可先 ollama stop <模型>")
    info["problems"] = problems
    return not problems, info


def ensure_model(model_dir: Path, repo: str, assume_yes: bool) -> bool:
    if (model_dir / "config.json").is_file() and any(model_dir.glob("*.safetensors")):
        return True
    free_gb = shutil.disk_usage(model_dir.parent if model_dir.parent.exists() else ROOT).free / 1e9
    say(f"没找到 {model_dir}（需要 config.json 与 *.safetensors）。下载 {repo} 约 9–10 GB，当前磁盘剩余 {free_gb:.0f} GB。")
    if free_gb < 15:
        say("磁盘剩余不足 15 GB，先不下载。")
        return False
    if not assume_yes:
        answer = input("现在下载？输入 y 回车确认，其它任意键跳过：").strip().lower()
        if answer != "y":
            say("跳过下载；冒烟没法继续。")
            return False
    model_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "*.py"]
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=str(model_dir), allow_patterns=patterns)
        say(f"Hugging Face 下载完成：{model_dir}")
        return True
    except Exception as exc:                          # 网络 / 代理问题时退到 ModelScope
        say(f"Hugging Face 下载失败（{type(exc).__name__}: {str(exc)[:200]}），改试 ModelScope")
    try:
        from modelscope import snapshot_download as ms_download
        ms_download(repo, local_dir=str(model_dir), allow_patterns=patterns)
        say(f"ModelScope 下载完成：{model_dir}")
        return True
    except Exception as exc:
        say(f"ModelScope 也失败了（{type(exc).__name__}: {str(exc)[:200]}）。可以手动把 {repo} 放到 {model_dir}")
        return False


class GpuSampler(threading.Thread):
    """每 interval 秒读一次整卡已用显存，记峰值（nvidia-smi 口径，含驱动与其它进程）。"""

    def __init__(self, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self.samples = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            for row in nvidia_smi("memory.used"):
                if row and row[0].isdigit():
                    self.peak_mib = max(self.peak_mib, int(row[0]))
            self.samples += 1
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()


def stream(cmd: list[str]) -> tuple[int, float]:
    say("> " + " ".join(cmd))
    started = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        say(line.rstrip("\n"))
    return proc.wait(), round(time.time() - started, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="training/configs/hi_smoke_4b.json")
    parser.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-used-mib", type=int, default=6000, help="开跑前整卡已用显存的上限（桌面本身约 3 GB）")
    parser.add_argument("--force", action="store_true", help="显卡有占用也开跑（不建议）")
    parser.add_argument("--yes", action="store_true", help="缺权重时不问直接下载")
    parser.add_argument("--skip-train", action="store_true", help="只重跑推理 + 核验（adapter 已在）")
    args = parser.parse_args(argv)

    say(f"===== 训练链路冒烟 4B  {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_dir"]
    report: dict[str, Any] = {"config": args.config, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    ok, info = preflight(args.max_used_mib)
    report["preflight"] = info
    for row in info["gpus"]:
        say(f"GPU {row[0]} {row[1]}：已用 {row[2]} / {row[3]} MiB")
    if not ok:
        for p in info["problems"]:
            say(f"!! {p}")
        if not args.force:
            say("显卡被占用，冒烟不开跑。等 walker / 评测跑完再双击；确认没问题可加 --force。")
            report["result"] = "refused_gpu_busy"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return 2
        say("--force：忽略占用继续。")
    else:
        say("显卡空闲，没有 walker / 评测 / vLLM / ollama 占用。")

    if not ensure_model(ROOT / cfg["model"], args.repo, args.yes):
        report["result"] = "no_model"
        return 3

    sampler = GpuSampler()
    sampler.start()
    py = sys.executable
    if not args.skip_train:
        rc, secs = stream([py, "training/sft_lora.py", "--config", args.config])
        report["train"] = {"exit": rc, "wall_s": secs, "gpu_peak_used_mib_nvidia_smi": sampler.peak_mib}
        if rc != 0:
            sampler.stop()
            say(f"训练失败（退出码 {rc}），看上面的输出。")
            report["result"] = "train_failed"
            (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
    sampler.peak_mib = 0
    smoke = cfg.get("_smoke_infer") or {}
    rc, secs = stream([py, "training/smoke_infer.py", "--config", args.config, "--compare-base",
                       "--max-new-tokens", str(smoke.get("max_new_tokens", 2048))])
    sampler.stop()
    report["infer"] = {"exit": rc, "wall_s": secs, "gpu_peak_used_mib_nvidia_smi": sampler.peak_mib}

    summary_path, infer_path = out_dir / "train_summary.json", out_dir / "smoke_infer.json"
    train_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    infer = json.loads(infer_path.read_text(encoding="utf-8")) if infer_path.is_file() else {}
    hist = train_summary.get("history") or []
    report["numbers"] = {
        "rows": train_summary.get("rows_selected"), "steps": train_summary.get("steps"),
        "tokens": (train_summary.get("prescan") or {}).get("tokens"),
        "loss_first": hist[0]["loss_unweighted"] if hist else None, "loss_last": hist[-1]["loss_unweighted"] if hist else None,
        "train_s": (train_summary.get("time_s") or {}).get("train"), "s_per_step": (train_summary.get("time_s") or {}).get("per_step"),
        "train_peak_alloc_gb": train_summary.get("peak_gpu_mem_gb"), "train_peak_reserved_gb": train_summary.get("peak_gpu_reserved_gb"),
        "infer_load_s": infer.get("load_s"), "infer_prompt_tokens": infer.get("prompt_tokens"),
        "infer_new_tokens": (infer.get("generation") or {}).get("new_tokens"), "infer_gen_s": (infer.get("generation") or {}).get("seconds"),
        "infer_peak_alloc_gb": infer.get("peak_gpu_mem_gb"), "adapter_active": infer.get("adapter_active"),
        "output_kind": (infer.get("generation") or {}).get("kind"), "output_parse": (infer.get("generation") or {}).get("parse"),
        "check_unit": (infer.get("checks") or {}).get("unit"), "check_errors": (infer.get("checks") or {}).get("n_errors"),
        "check_by_severity": (infer.get("checks") or {}).get("by_severity"),
        "teacher_check_errors": (infer.get("teacher_checks") or {}).get("n_errors"),
        "base_output_kind": (infer.get("base_generation") or {}).get("kind"),
    }
    report["result"] = "pass" if infer.get("smoke_ok") else "fail"
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    say("----- 冒烟数字 -----")
    for k, v in report["numbers"].items():
        say(f"  {k}: {v}")
    say(f"结果：{report['result'].upper()}（{out_dir / 'smoke_report.json'}）")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
