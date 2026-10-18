"""分支可达探测：SEARCH image→video（A 族 A4_search_original_image，只用帧搜视频）。

一、程序可达性（静态，读 walker 源码，不联网）：A4 的条件是"还没有任何以图搜（image / text_image）且工具允许以图搜且视图里
有材料帧"；而 A3 第 2 条查询在"已有文字查询、还没有以图搜、允许以图搜、有帧"时就发成 text_image——同一组条件先被 A3 用掉，
text_image 记进 image_searches 后 A4 不再满足。出错的查询只剔除第 1 条（errored[:1]），第 2 次出错的 text_image 仍算以图搜。
=> 在现行 walker（walker-r1@2026-10-17e）里 A4 **加案例不可达**，只能改 walker（例如"text_image 的以图搜那一半没对上时再
发一次纯帧搜"），要先问负责人。本脚本把这几处条件从源码里抽出来核对，并数 r1 / g1 run 里 A4 实际出现次数。
二、（可选 --live --allow-yandex）假如 walker 改了，纯帧搜能不能找回原片：FMNV 公开帧走 Yandex（output_type=video），
看原片进不进前 10。Lens 关、Serper 0 点。

    python scripts/_branch_probe_image_video.py                       # 只做静态核对 + 计数
    python scripts/_branch_probe_image_video.py --live --allow-yandex --n 10
输出 claude/_longtail_work/probe_image_video.json
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re

from _branch_probe_common import OUT_DIR, ROOT, SFT, a_probe, now, write_json

WALKER = ROOT / "src" / "foren_searcher" / "datagen" / "walker.py"


def static_check() -> dict[str, object]:
    src = WALKER.read_text(encoding="utf-8")
    body = src[src.index("def _decide_find_original"):src.index("def _material_images")]
    checks = {
        "max_text_queries_is_2": bool(re.search(r"^_MAX_TEXT_QUERIES = 2\b", src, re.M)),
        "a3_pair_condition": "pair = bool(text_searches) and not image_searches and _image_search_allowed(state) and bool(frames)" in body,
        "a3_runs_while_queries_lt_max": "if len(queries) < _MAX_TEXT_QUERIES and state.remaining >= 2:" in body,
        "a4_requires_no_image_search": "if not image_searches and _image_search_allowed(state) and state.remaining >= 2:" in body,
        "a4_requires_frames": "if frames:" in body[body.index("# A4"):body.index('return not_found("A5_original_not_found")')],
        "errored_only_first_dropped": "errored = [s for s in queries if _provider_error(s)][:1]" in body,
        "image_searches_includes_text_image": 'in ("image", "text_image")]' in body,
    }
    unreachable = all(checks.values())
    return {"checks": checks, "a4_reachable_by_adding_cases": not unreachable,
            "why": ("A3 第 2 条查询与 A4 的条件相同（无以图搜 + 允许以图搜 + 有帧），A3 先发 text_image 后 image_searches 非空，"
                    "A4 永远不满足；帧不可见时两者都不发。" if unreachable else "源码与预期不符，需人工复核")}


def count_runs() -> dict[str, int]:
    out: dict[str, int] = {}
    for label, pattern in (("r1", SFT / "runs" / "*_r1_1"), ("g1", ROOT / ".wt_g1" / "experiments" / "sft_data" / "runs" / "*_r1g1_1")):
        n_runs = n_a4 = n_a3ti = 0
        for run in glob.glob(str(pattern)):
            if "fmnv_a_" not in run:
                continue
            n_runs += 1
            for t in glob.glob(run + "/requests/*/transport.json"):
                s = open(t, encoding="utf-8").read()
                n_a4 += s.count('"A4_search_original_image"') > 0
                n_a3ti += s.count('"A3_search_original_text_image"') > 0
        out.update({f"{label}_a_runs": n_runs, f"{label}_A4_requests": n_a4, f"{label}_A3_text_image_requests": n_a3ti})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="另做纯帧搜（假如 walker 改了能不能找回原片）")
    parser.add_argument("--allow-yandex", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    payload: dict[str, object] = {"generated": now(), "static": static_check(), "runs": count_runs()}
    if args.live:
        live = asyncio.run(a_probe(args.n, ["image"], args.allow_yandex, "probe_image_video_live.json", args.redo))
        payload["live"] = {k: live[k] for k in ("candidates", "summary", "reachable", "serper_balance_before", "serper_balance_after")}
    write_json(OUT_DIR / "probe_image_video.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
