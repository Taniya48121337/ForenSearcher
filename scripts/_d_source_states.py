"""来源 D 扩展准备（本机跑）：为指定 bridge run 找出"最后一个还能开新轮的 GIA 请求"，把状态摘要拷出来供起草换题。

    python scripts/_d_source_states.py [--runs 名字 ...] [--out experiments/sft_data/offline_gia/_sources]

每个 run 输出 <out>/<run>.json：source_run、source_request、task、runtime_constraints、history_index、latest_reports
（含 Finding 原文与限制）、current_materials 的 E 列表（编号 / label / 获取方式），以及 run_config 里的 inspect_operations /
search / source_policy。只读 run 目录，不改任何东西。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
DEFAULT_RUNS = ("true_4124903_r2", "true_17838107_r1", "true_20205316753904_validation", "true_20361105_visibility_validation",
                "true_39132300_validation_r3", "videosham_0069_r1", "videosham_1007_validation", "videosham_4126_split_intervals_r7")


def gia_requests(run: Path) -> list[tuple[str, dict]]:
    out = []
    for req in sorted((run / "requests").glob("request_*")):
        path = req / "request.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("payload") or {}
        if "global_view" in payload and not payload.get("workspace_id"):
            out.append((req.name, data))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--out", default=str(SFT / "offline_gia" / "_sources"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        run = SFT / "runs" / name
        if not run.is_dir():
            print(f"{name}: 目录不存在", flush=True)
            continue
        candidates = []
        for req_name, data in gia_requests(run):
            view = data["payload"]["global_view"]
            rc = view.get("runtime_constraints") or {}
            if rc.get("can_create_round") and not rc.get("round_in_progress"):
                candidates.append((req_name, data))
        if not candidates:
            print(f"{name}: 没有能开新轮的 GIA 请求", flush=True)
            continue
        req_name, data = candidates[-1]
        view = data["payload"]["global_view"]
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").is_file() else {}
        brief = {
            "source_run": name, "source_request": req_name, "n_gia_requests": len(gia_requests(run)),
            "task": view.get("task"), "runtime_constraints": view.get("runtime_constraints"),
            "history_index": view.get("history_index"), "latest_reports": view.get("latest_reports"),
            "current_materials": [{k: m.get(k) for k in ("evidence_id", "reference_id", "label", "acquired_by", "modalities", "source_ref")}
                                  for m in view.get("current_materials") or []],
            "visual_observation_summary": view.get("visual_observation_summary"),
            "run_config": {k: config.get(k) for k in ("dataset", "case", "inspect_operations", "search", "evidence_source_policy",
                                                     "source_policy", "gia_output_source")},
        }
        (out / f"{name}.json").write_text(json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {req_name}（共 {brief['n_gia_requests']} 个 GIA 请求，可开新轮的 {len(candidates)} 个）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
