"""INSPECT 工具重跑方差：关掉缓存、对同一媒体同一操作重跑 N 次，量化输出有多不确定。

    python scripts/tool_cache_variance.py --media A.mp4 --operation ocr --runs 5
    python scripts/tool_cache_variance.py --media A.mp4 --operation asr --params '{"language": "en"}' --runs 3
    python scripts/tool_cache_variance.py --media A.mp4 B.mp4 --operation align_media --params '{"mode": "visual"}'
    python scripts/tool_cache_variance.py --media A.mp4 --operation get_frames --params '{"count": 8}' --fresh-backend

为什么要有：工具缓存（inspect/tool_cache.py）让不同臂共用同一份 OCR / ASR / 对齐结果，消除了"工具这次和上次
不一样"对对照的干扰；但论文里还得报告"不缓存时这些工具本身的重跑方差有多大"，说明缓存有必要（或没必要）。

指标（都相对第 1 次运行）：
- ocr / asr / visual_qa / audio_qa：文本相似度（difflib.SequenceMatcher 比例，1.0 = 逐字相同）。
- align_media：最佳匹配的时间锚点差（秒，锚点 time_b−time_a 中位数的差）和候选区间 IoU（A、B 两侧取平均）。
- get_frames / get_pages / crop：产出图片的像素哈希（解码后 RGB 像素的 sha256）是否与第 1 次一致。
结果输出成 markdown 表（--out 另存文件）。

后端：默认用本机 LocalInspectBackend（Windows 本机装了 rapidocr、faster-whisper 时就是真实后端）；测试里注入假 backend。
直接调 processor 层（与缓存所在层相同），不经过 Store、不分配 E 编号，也不读写缓存。
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import io
import json
import mimetypes
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foren_searcher.inspect.backends.base import BackendFailure, BackendOutcome, PreparedMedia  # noqa: E402
from foren_searcher.inspect.contracts import parse_inspect_call  # noqa: E402
from foren_searcher.inspect.policy import InspectPolicy  # noqa: E402
from foren_searcher.inspect.processors import create_processors  # noqa: E402
from foren_searcher.store.resolver import sniff_media_kind  # noqa: E402

TEXT_OPERATIONS = {"ocr", "asr", "visual_qa", "audio_qa"}
IMAGE_OPERATIONS = {"get_frames", "get_pages", "crop"}


# ---------------------------------------------------------------- 从结果里取可比较的东西

def outcome_text(operation: str, outcome: BackendOutcome) -> str:
    """把 OCR / ASR / 问答结果压成一段文本（保持顺序），用于相似度。"""

    raw = outcome.result
    if operation == "ocr":
        return "\n".join(" ".join(str(item["text"]) for item in row.get("items", [])) for row in raw.get("results", []))
    if operation == "asr":
        return "\n".join(str(segment["text"]) for segment in raw.get("segments", []))
    return str(raw.get("answer", ""))


def pixel_hashes(outcome: BackendOutcome) -> list[str]:
    """产出图片解码后的像素哈希：PNG 编码参数不同但像素相同，也算一致。"""

    from PIL import Image

    values = []
    for artifact in outcome.artifacts:
        try:
            with Image.open(io.BytesIO(artifact.data)) as image:
                rgb = image.convert("RGB")
                values.append(hashlib.sha256(f"{rgb.width}x{rgb.height}".encode() + rgb.tobytes()).hexdigest()[:16])
        except Exception:
            values.append("bytes:" + hashlib.sha256(artifact.data).hexdigest()[:16])
    return values


def best_alignment(outcome: BackendOutcome) -> dict[str, Any] | None:
    """最佳匹配（分数最高）的锚点偏移与区间。"""

    matches = list(outcome.result.get("matches") or [])
    if not matches:
        return None
    best = max(matches, key=lambda m: float(m.get("score", 0.0)))
    anchors = best.get("anchors") or []
    offsets = [float(a["time_b_s"]) - float(a["time_a_s"]) for a in anchors]
    offset = statistics.median(offsets) if offsets else float(best["range_b_s"][0]) - float(best["range_a_s"][0])
    return {"offset_s": offset, "range_a_s": [float(x) for x in best["range_a_s"]],
            "range_b_s": [float(x) for x in best["range_b_s"]], "score": float(best.get("score", 0.0))}


def interval_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return 1.0 if union <= 0 else inter / union


def text_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def compare(operation: str, outcomes: list[BackendOutcome | BaseException]) -> list[dict[str, Any]]:
    """每次运行一行，指标都相对第 1 次成功的运行。"""

    reference = next((o for o in outcomes if isinstance(o, BackendOutcome)), None)
    rows: list[dict[str, Any]] = []
    for index, outcome in enumerate(outcomes, 1):
        row: dict[str, Any] = {"run": index}
        if not isinstance(outcome, BackendOutcome) or reference is None:
            row["error"] = f"{type(outcome).__name__}: {outcome}" if isinstance(outcome, BaseException) else "no result"
            rows.append(row)
            continue
        if operation in TEXT_OPERATIONS:
            text = outcome_text(operation, outcome)
            row["chars"] = len(text)
            row["similarity"] = round(text_similarity(outcome_text(operation, reference), text), 4)
        elif operation == "align_media":
            mine, base = best_alignment(outcome), best_alignment(reference)
            if mine is None or base is None:
                row["anchor_diff_s"] = None if mine is None and base is None else float("inf")
                row["iou"] = None if mine is None and base is None else 0.0
            else:
                row["offset_s"] = round(mine["offset_s"], 3)
                row["anchor_diff_s"] = round(abs(mine["offset_s"] - base["offset_s"]), 3)
                row["iou"] = round((interval_iou(mine["range_a_s"], base["range_a_s"])
                                    + interval_iou(mine["range_b_s"], base["range_b_s"])) / 2, 4)
        elif operation in IMAGE_OPERATIONS:
            mine_hashes, base_hashes = pixel_hashes(outcome), pixel_hashes(reference)
            row["images"] = len(mine_hashes)
            row["pixel_identical"] = mine_hashes == base_hashes
            row["pixel_hash"] = hashlib.sha256("".join(mine_hashes).encode()).hexdigest()[:12]
        else:
            same = json.dumps(outcome.result, sort_keys=True, default=str) == json.dumps(reference.result, sort_keys=True, default=str)
            row["result_identical"] = same
        row["issues"] = len(outcome.issues)
        rows.append(row)
    return rows


def summarize(operation: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    summary: dict[str, Any] = {"operation": operation, "runs": len(rows), "failed_runs": len(rows) - len(ok)}
    if operation in TEXT_OPERATIONS and ok:
        values = [r["similarity"] for r in ok]
        summary |= {"min_similarity": min(values), "mean_similarity": round(statistics.fmean(values), 4)}
    elif operation == "align_media" and ok:
        diffs = [r["anchor_diff_s"] for r in ok if r.get("anchor_diff_s") is not None]
        ious = [r["iou"] for r in ok if r.get("iou") is not None]
        summary |= {"max_anchor_diff_s": max(diffs) if diffs else None, "min_iou": min(ious) if ious else None}
    elif operation in IMAGE_OPERATIONS and ok:
        summary |= {"pixel_identical_runs": sum(bool(r["pixel_identical"]) for r in ok)}
    elif ok:
        summary |= {"identical_runs": sum(bool(r.get("result_identical")) for r in ok)}
    return summary


def render_markdown(operation: str, media: list[str], params: dict[str, Any], rows: list[dict[str, Any]],
                    elapsed: list[float], backend: str) -> str:
    columns = [key for key in ("chars", "similarity", "offset_s", "anchor_diff_s", "iou", "images", "pixel_identical",
                               "pixel_hash", "result_identical", "issues", "error") if any(key in r for r in rows)]
    names = {"chars": "字符数", "similarity": "文本相似度", "offset_s": "锚点偏移(s)", "anchor_diff_s": "锚点差(s)",
             "iou": "区间 IoU", "images": "图片数", "pixel_identical": "像素一致", "pixel_hash": "像素哈希",
             "result_identical": "结果一致", "issues": "问题数", "error": "错误"}
    lines = [f"### {operation} 重跑方差（{len(rows)} 次，缓存关闭）", "",
             f"- 媒体：{', '.join(media)}", f"- 参数：`{json.dumps(params, ensure_ascii=False)}`", f"- 后端：{backend}", "",
             "| 次 | 耗时(s) | " + " | ".join(names[c] for c in columns) + " |",
             "|---:|---:|" + "---|" * len(columns)]
    for row, seconds in zip(rows, elapsed):
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append("是" if value is True else "否" if value is False else "—" if value is None else str(value))
        lines.append(f"| {row['run']} | {seconds:.2f} | " + " | ".join(cells) + " |")
    summary = summarize(operation, rows)
    lines += ["", "汇总：" + "，".join(f"{k}={v}" for k, v in summary.items() if k != "operation"), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 运行

async def prepare_media(backend: Any, paths: list[Path]) -> tuple[PreparedMedia, ...]:
    prepared = []
    for index, path in enumerate(paths, 1):
        kind = sniff_media_kind(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if kind == "unknown":
            raise SystemExit(f"认不出媒体类型：{path}")
        profile = await backend.probe(path, kind)
        prepared.append(PreparedMedia(f"E{index}", path, kind, profile))
    return tuple(prepared)


async def run_variance(backend_factory: Callable[[], Any], paths: list[Path], operation: str, params: dict[str, Any], *,
                       runs: int = 3, policy: InspectPolicy | None = None, fresh_backend: bool = False
                       ) -> tuple[list[BackendOutcome | BaseException], list[float]]:
    """重跑 N 次；fresh_backend=True 时每次新建后端（连模型加载的不确定性一起量）。"""

    policy = policy or InspectPolicy()
    call = parse_inspect_call({"targets": [f"E{i}" for i in range(1, len(paths) + 1)],
                               "request": {"operation": operation, "params": params}})
    backend = backend_factory()
    media = await prepare_media(backend, paths)
    outcomes: list[BackendOutcome | BaseException] = []
    elapsed: list[float] = []
    for index in range(runs):
        if fresh_backend and index:
            await _close(backend)
            backend = backend_factory()
        processor = create_processors(backend)[operation]      # 不包 CachingProcessor：这里就是要测未缓存的方差
        started = time.monotonic()
        try:
            outcomes.append(await processor.process(media, call.request.params, policy, None))
        except BackendFailure as exc:
            outcomes.append(exc)
        elapsed.append(time.monotonic() - started)
    await _close(backend)
    return outcomes, elapsed


async def _close(backend: Any) -> None:
    close = getattr(backend, "aclose", None)
    if callable(close):
        await close()


def main(argv: list[str] | None = None, *, backend_factory: Callable[[], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description="INSPECT 工具重跑方差（缓存关闭）")
    parser.add_argument("--media", nargs="+", required=True, help="输入媒体；align_media 给两个")
    parser.add_argument("--operation", required=True,
                        choices=["ocr", "asr", "align_media", "get_frames", "get_pages", "crop", "visual_qa", "audio_qa",
                                 "locate_frames", "locate_regions"])
    parser.add_argument("--params", default="{}", help="操作参数 JSON（与 INSPECT 的 request.params 相同）")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fresh-backend", action="store_true", help="每次重跑都新建后端（重新加载模型）")
    parser.add_argument("--out", default=None, help="markdown 另存到这个文件")
    args = parser.parse_args(argv)
    if args.runs < 2:
        raise SystemExit("--runs 至少 2 次才谈得上方差")
    params = json.loads(args.params)
    if backend_factory is None:
        from foren_searcher.inspect.backends.local import LocalInspectBackend

        backend_factory = LocalInspectBackend
    paths = [Path(p).resolve() for p in args.media]
    outcomes, elapsed = asyncio.run(run_variance(backend_factory, paths, args.operation, params, runs=args.runs,
                                                 fresh_backend=args.fresh_backend))
    rows = compare(args.operation, outcomes)
    backend_name = getattr(backend_factory, "__name__", type(backend_factory).__name__)
    text = render_markdown(args.operation, [str(p) for p in paths], params, rows, elapsed, backend_name)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
