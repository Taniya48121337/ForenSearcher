"""评测侧 vLLM 回放压测（只读已完成的 run；2026-10-20，《训练与评测提速方案_20261020》三节）。

把 E2 基线已录下的模型请求（requests/request_*/request.json + actor/image_*.png，与 EndpointTeacher._messages 同形）
原样重发给一个 vLLM 端点：N 个"虚拟 run"并发，每个虚拟 run 内按原顺序串行（与 run_cases 一样一次只有一个请求在飞）。
不调工具、不写任何 run 目录、不碰检索快照——只量服务端在真实输入分布下的吞吐，用来比 vLLM 参数 / 并发 / 投机解码。

    # GPU 空窗期，先按要比的参数起 serve（端口随意），再：
    PYTHONPATH=src python server/28_replay_bench.py --runs-dir experiments/eval/r0/runs --glob '*_r0base-s1_1' \
        --endpoint http://127.0.0.1:8002/v1 --model student --concurrency 12 --format none --out /workspace/bench/c12_base.json
    # 只核对能否重建请求（不发请求；本机没有 png 时报缺图数）：
    python server/28_replay_bench.py --runs-dir … --glob '*_r0base-s*_1' --dry-run

--format none（默认）：不带 response_format，max_tokens = min_tokens = 录下的输出 token 数、ignore_eos——每次调用的
  prefill / decode 长度与基线逐次相同，不同配置之间可以直接比。
--format schema：照评测发 json_schema（EndpointTeacher._output_schema；没有运行时 guard 状态，数组上限取默认），
  max_tokens 4096、不强制长度——量结构化解码本身的开销（与 none 对比）。
--think S：每次调用之间睡 S 秒，模拟工具时间（0 = 纯服务端吞吐上限）。
结果：每次调用的延迟与 usage、汇总（墙钟、调用 / s、prompt / gen token/s、延迟分位、每个虚拟 run 的墙钟），
以及压测前后 /metrics 里 vllm:* 直方图 _sum / _count 的差（排队 / prefill / decode / TTFT / 每 token 时间，名字以实际版本为准）。
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROMPT_PREFIX = ("Return exactly one JSON object conforming to the role's instructions. "
                 "Use only the supplied state and attached images; do not invent observations, evidence, or future results.\n\n")


def build_body(request_dir: Path, model: str, fmt: str, temperature: float, *, with_images: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 EndpointTeacher._messages / _image_message 的顺序重建一次请求：system；user = 状态文本 + [说明…] + (图注, 图)×n。"""

    request = json.loads((request_dir / "request.json").read_text(encoding="utf-8"))
    transport = json.loads((request_dir / "transport.json").read_text(encoding="utf-8"))
    state = request.get("rendered_payload")
    text = PROMPT_PREFIX + (json.dumps(state, ensure_ascii=False, separators=(",", ":")) if state is not None
                            else json.dumps(request["payload"], ensure_ascii=False))
    notes = list(request.get("image_notes") or [])
    n_images = len(request.get("image_evidence_ids") or [])
    missing = 0
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if n_images:
        header, captions = notes[: len(notes) - n_images], notes[len(notes) - n_images:]
        content += [{"type": "text", "text": h} for h in header]
        for i, caption in enumerate(captions, 1):
            content.append({"type": "text", "text": caption})
            png = request_dir / "actor" / f"image_{i:02d}.png"
            if png.is_file() and with_images:
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")}})
            else:                                       # 服务器上约 3% 的请求目录缺图（12,511 / 12,864）：去掉这张图，计数
                missing += 1
    user: Any = content if n_images else text
    usage = transport.get("usage") or {}
    out_tokens = int(usage.get("output_tokens") or 0)
    body: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": request["system_prompt"]}, {"role": "user", "content": user}],
                            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    if fmt == "none":
        body.update({"max_tokens": max(1, out_tokens), "min_tokens": max(1, out_tokens), "ignore_eos": True})
    else:
        body["max_tokens"] = 4096
        if fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        else:
            from foren_searcher.datagen.teacher import EndpointTeacher
            schema = EndpointTeacher._output_schema(request["payload"])
            body["response_format"] = ({"type": "json_schema", "json_schema": {"name": "harness_decision", "strict": True, "schema": schema}}
                                       if schema is not None else {"type": "json_object"})
    meta = {"request": f"{request_dir.parent.parent.name}/{request_dir.name}", "recorded_input_tokens": int(usage.get("input_tokens") or 0),
            "recorded_output_tokens": out_tokens, "recorded_elapsed_s": transport.get("elapsed_s"), "images": n_images, "missing_images": missing}
    return body, meta


def run_dirs(root: Path, pattern: str, limit: int) -> list[Path]:
    dirs = [d for d in sorted(root.glob(pattern)) if d.is_dir() and ".failed" not in d.name and (d / "requests").is_dir()]
    return dirs[:limit] if limit else dirs


def request_dirs(run_dir: Path) -> list[Path]:
    return [d for d in sorted((run_dir / "requests").glob("request_*")) if (d / "request.json").is_file() and (d / "transport.json").is_file()]


def scrape_metrics(endpoint: str) -> dict[str, float]:
    """vllm:* 直方图 / 计数器的 _sum、_count、_total（按名字聚合掉 label）。端点没有 /metrics 时返回空。"""

    import httpx

    base = re.sub(r"/v1/?$", "", endpoint)
    try:
        text = httpx.get(base + "/metrics", timeout=10, trust_env=False).text
    except httpx.HTTPError:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-zA-Z0-9_:]+?(?:_sum|_count|_total))(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="experiments/eval/r0/runs")
    ap.add_argument("--glob", default="*_r0base-s1_1", help="run 目录名的 glob（例：'*_r0base-s*_1' = 两个种子 48 个）")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="student", help="服务名；训后可写 LoRA 名")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--format", choices=["none", "json_object", "schema"], default="none")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think", type=float, default=0.0, help="同一虚拟 run 内两次调用之间的停顿（秒）")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只重建请求、报统计，不发")
    args = ap.parse_args()

    runs = run_dirs(Path(args.runs_dir), args.glob, args.max_runs)
    if not runs:
        raise SystemExit(f"{args.runs_dir} 下没有匹配 {args.glob} 的 run")
    plan = {r.name: request_dirs(r) for r in runs}
    if args.dry_run:
        metas = [build_body(d, args.model, "none", args.temperature, with_images=True)[1] for dirs in plan.values() for d in dirs]
        print(json.dumps({"runs": len(runs), "calls": len(metas), "recorded_input_tokens": sum(m["recorded_input_tokens"] for m in metas),
                          "recorded_output_tokens": sum(m["recorded_output_tokens"] for m in metas), "images": sum(m["images"] for m in metas),
                          "missing_images": sum(m["missing_images"] for m in metas)}, ensure_ascii=False, indent=1))
        return 0

    import httpx

    work: queue.Queue[str] = queue.Queue()
    for name in plan:
        work.put(name)
    calls: list[dict[str, Any]] = []
    run_walls: dict[str, float] = {}
    lock = threading.Lock()

    def worker() -> None:
        client = httpx.Client(timeout=httpx.Timeout(args.timeout, connect=10.0), trust_env=False)
        while True:
            try:
                name = work.get_nowait()
            except queue.Empty:
                break
            started = time.monotonic()
            for i, d in enumerate(plan[name]):
                if i and args.think:
                    time.sleep(args.think)
                body, meta = build_body(d, args.model, args.format, args.temperature)
                t0 = time.monotonic()
                try:
                    resp = client.post(args.endpoint + "/chat/completions", json=body)
                    ok, status = resp.status_code < 400, resp.status_code
                    usage = (resp.json().get("usage") or {}) if ok else {}
                    error = None if ok else resp.text[:300]
                except httpx.HTTPError as exc:
                    ok, status, usage, error = False, None, {}, f"{type(exc).__name__}: {exc}"[:300]
                rec = {**meta, "run": name, "ok": ok, "status": status, "latency_s": round(time.monotonic() - t0, 3),
                       "t_start": round(t0 - bench_start, 3), "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"), "error": error}
                with lock:
                    calls.append(rec)
            with lock:
                run_walls[name] = round(time.monotonic() - started, 1)
        client.close()

    before = scrape_metrics(args.endpoint)
    bench_start = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - bench_start
    after = scrape_metrics(args.endpoint)

    good = [c for c in calls if c["ok"]]
    lat = sorted(c["latency_s"] for c in good)
    q = (lambda p: lat[min(len(lat) - 1, int(p * (len(lat) - 1)))] if lat else None)
    prompt = sum(c["prompt_tokens"] or 0 for c in good)
    gen = sum(c["completion_tokens"] or 0 for c in good)
    delta = {k: round(after[k] - before.get(k, 0.0), 3) for k in sorted(after) if after[k] - before.get(k, 0.0)}
    hist = {}
    for k, v in delta.items():
        if k.endswith("_sum") and delta.get(k[:-4] + "_count"):
            hist[k[:-4] + "_mean"] = round(v / delta[k[:-4] + "_count"], 4)
    summary = {"glob": args.glob, "runs": len(runs), "concurrency": args.concurrency, "format": args.format, "think_s": args.think,
               "calls": len(calls), "failed": len(calls) - len(good), "wall_s": round(wall, 1), "calls_per_s": round(len(good) / wall, 3),
               "prompt_tok_per_s": round(prompt / wall, 1), "gen_tok_per_s": round(gen / wall, 1),
               "latency_s": {"mean": round(statistics.mean(lat), 2) if lat else None, "p50": q(.5), "p90": q(.9), "p99": q(.99), "max": q(1)},
               "run_wall_s": {"mean": round(statistics.mean(run_walls.values()), 1) if run_walls else None, "max": max(run_walls.values(), default=None)},
               "vllm_metrics_mean": hist, "vllm_metrics_delta": delta}
    print(json.dumps({k: v for k, v in summary.items() if k != "vllm_metrics_delta"}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "calls": calls, "run_walls": run_walls}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if len(good) == len(calls) else 1


if __name__ == "__main__":
    sys.exit(main())
