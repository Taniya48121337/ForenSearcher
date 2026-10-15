"""Step 6: run the baselines on the server (called by server/06_run_baselines.sh).

Arms (run one after another; inside an arm FS_JOBS investigations run in parallel):
    evid_base    EVID-Bench 131-222 test samples, old tool catalog   round srvF27  (= local evidF27 setting)
    evid_merged  same samples, merged catalog (10-12)                round srvM27
    g360         GroundLie360, FS_BENCH_N samples, merged catalog    round gsrvM27
    fmnv         FMNV, FS_BENCH_N samples, merged catalog            round fsrvM27
    g360_base / fmnv_base (optional, --arms ...): the same with the old catalog
Merged catalog (10-12, DETECT folded into SEARCH and INSPECT): hi@0.8.0 / gia@0.9.0 — INSPECT get_frames / ocr / asr /
locate / video_structure / compare_media, SEARCH video results checked against the questioned video, reverse image
search on. Base arms: hi@0.7.0 / gia@0.8.0 with --image-search off, as the local evidF27 runs.
(evid_detect is kept as a name for evid_merged; DETECT v2.4 itself has been removed.)
Common: Qwen3.8-27B bf16 on vLLM (thinking off), 3 rounds, compact_16k, eval guards, fact-check sites blocked. Searches
are recorded once per dataset and replayed by the later arm, so base and merged arms see the same text/video results.

Resumable: stop with Ctrl-C (or lose the SSH session) and start again; finished runs are kept and skipped.
Progress every minute on screen and in server_logs/06_run_baselines.log; summary in server_logs/baselines_summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LOG_DIR = Path(os.environ.get("FS_LOG_DIR") or ROOT / "server_logs")
ENDPOINT = os.environ.get("FS_ENDPOINT", "http://127.0.0.1:8000/v1")
SERVED = os.environ.get("FS_SERVED_NAME", "qwen38-27b")
JOBS = os.environ.get("FS_JOBS", "16")
BENCH_N = int(os.environ.get("FS_BENCH_N", "200"))
JUDGE = "claude-cli:sonnet"

ARMS = {
    "evid_base":   {"dataset": "evid", "round": "srvF27", "merged": False},
    "evid_merged": {"dataset": "evid", "round": "srvM27", "merged": True},
    "g360":        {"dataset": "g360", "round": "gsrvM27", "merged": True},
    "fmnv":        {"dataset": "fmnv", "round": "fsrvM27", "merged": True},
    "g360_base":   {"dataset": "g360", "round": "gsrvF27", "merged": False},
    "fmnv_base":   {"dataset": "fmnv", "round": "fsrvF27", "merged": False},
}
ARMS["evid_detect"] = ARMS["evid_merged"]          # old name (10-12: DETECT folded into SEARCH / INSPECT)
DEFAULT_ARMS = ("evid_base", "evid_merged", "g360", "fmnv")
SPECS = {True: ("hi@0.8.0", "gia@0.9.0"), False: ("hi@0.7.0", "gia@0.8.0")}
# EVID samples read during DETECT development on the PC (used_for_revision): reported separately as "clean subset"
TOUCHED = {f"video_{i}" for i in (152, 155, 160, 163, 167, 171, 175, 177, 178, 179, 181, 184, 187, 201, 205, 210, 213, 216,
                                  220, 154, 159, 164, 168, 169, 174, 180, 186, 189, 194, 199, 215,
                                  138, 156, 172, 198, 208, 209)}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (LOG_DIR / "06_run_baselines.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ samples and run directories

def evid_ids() -> list[str]:
    from _evid_score import revised_ids

    entries = json.loads((ROOT / "plan-harness-1/datasets/EVID-Bench/evid_bench.json").read_text(encoding="utf-8"))
    revised = revised_ids(ROOT / "experiments/evid/cases")
    return sorted(e["id"] for e in entries if 131 <= int(e["id"].split("_")[-1]) <= 222 and e["id"] not in revised)


def bench_ids(ds: str) -> list[str]:
    sel = ROOT / "experiments/bench" / ds / f"selection_n{BENCH_N}.json"
    if not sel.is_file():
        done = subprocess.run([sys.executable, "scripts/_bench_build_cases.py", "--dataset", ds, "--n", str(BENCH_N)],
                              cwd=ROOT, capture_output=True, text=True)
        log(done.stdout.strip().replace("\n", " | ") or done.stderr.strip()[-400:])
        if done.returncode != 0 or not sel.is_file():
            return []
    return json.loads(sel.read_text(encoding="utf-8"))


def run_dir(ds: str, sid: str, rnd: str) -> Path:
    from foren_searcher.datagen.run_cases import _safe

    if ds == "evid":
        return ROOT / "experiments/evid/runs" / f"evid_{_safe(sid)}_{_safe(rnd)}_1"
    return ROOT / "experiments/bench" / ds / "runs" / f"{_safe(ds)}_{_safe(sid)}_{_safe(rnd)}_1"


def is_done(path: Path) -> bool:
    return (path / "result.json").is_file() and (path / "execution_links.json").is_file()


def run_command(arm: dict, ids: list[str]) -> list[str]:
    ds = arm["dataset"]
    if ds == "evid":
        cases, runs, snaps = "experiments/evid/cases", "experiments/evid/runs", "experiments/evid/search_snapshots_server"
        case_ids = [f"evid:{i}" for i in ids]
    else:
        base = f"experiments/bench/{ds}"
        cases, runs, snaps = f"{base}/cases", f"{base}/runs", f"{base}/search_snapshots"
        case_ids = [f"{ds}:{i}" for i in ids]
    cmd = [sys.executable, "-m", "foren_searcher.datagen.run_cases", "--cases", cases, "--runs-dir", runs, "--split", "test",
           "--round", arm["round"], "--endpoint", ENDPOINT, "--model", SERVED, "--no-think", "--jobs", JOBS,
           "--instruction-mode", "student", "--retry-failed", "--blocklist", "exclude",
           "--hi-spec", SPECS[arm["merged"]][0], "--gia-spec", SPECS[arm["merged"]][1],
           "--search-mode", "replay_or_record", "--snapshot-root", snaps,
           "--runtime-guards", "eval", "--state-renderer", "compact_16k"]
    if not arm["merged"]:
        cmd += ["--image-search", "off"]      # = the local evidF27 setting (reverse image search was off before 10-12)
    return cmd + ["--ids", *case_ids]


# ------------------------------------------------------------------ running with progress

def run_with_progress(name: str, arm: dict, ids: list[str], label: str) -> None:
    todo = [i for i in ids if not is_done(run_dir(arm["dataset"], i, arm["round"]))]
    if not todo:
        log(f"{name}: {label}: nothing to do ({len(ids)}/{len(ids)} done)")
        return
    out = LOG_DIR / f"{name}_{arm['round']}.runlog"
    log(f"{name}: {label}: {len(todo)} runs ({len(ids) - len(todo)} already done), {JOBS} in parallel; "
        f"detailed output: {out}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("FS_TOOL_GPUS", "6,7")
    t0 = time.time()
    with out.open("a", encoding="utf-8") as stream:
        proc = subprocess.Popen(run_command(arm, todo), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        last = -1
        while proc.poll() is None:
            time.sleep(60)
            done = sum(is_done(run_dir(arm["dataset"], i, arm["round"])) for i in ids)
            new = done - (len(ids) - len(todo))
            elapsed = (time.time() - t0) / 60
            eta = (elapsed / new * (len(todo) - new)) if new else float("nan")
            if done != last or int(elapsed) % 10 == 0:
                log(f"{name}: {done}/{len(ids)} done, {elapsed:.0f} min elapsed, ETA {eta:.0f} min "
                    f"| vLLM {'up' if vllm_up() else 'DOWN'} | free disk {shutil.disk_usage(ROOT).free / 2**30:.0f} GB")
                last = done
    done = sum(is_done(run_dir(arm["dataset"], i, arm["round"])) for i in ids)
    log(f"{name}: {label} finished with exit code {proc.returncode}: {done}/{len(ids)} done in {(time.time() - t0) / 60:.0f} min")


def vllm_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(ENDPOINT + "/models", timeout=5) as r:
            return SERVED in r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return False


def judge_available() -> bool:
    try:
        from foren_searcher.datagen.verify.judges import find_claude_cli

        return bool(find_claude_cli())
    except Exception:  # noqa: BLE001
        return False


def score(name: str, arm: dict, ids: list[str]) -> None:
    if not judge_available():
        log(f"{name}: scoring skipped: Claude CLI not found. Score on the PC: bash server/08_export_for_scoring.sh")
        return
    if arm["dataset"] == "evid":
        cmd = [sys.executable, "scripts/_evid_score.py", "--round", arm["round"], "--ids", *ids, "--judges", JUDGE]
    else:
        cmd = [sys.executable, "scripts/_bench_score.py", "--dataset", arm["dataset"], "--round", arm["round"], "--n", str(BENCH_N)]
    out = LOG_DIR / f"{name}_{arm['round']}.scorelog"
    log(f"{name}: scoring with Sonnet ({len(ids)} samples) -> {out}")
    with out.open("a", encoding="utf-8") as stream:
        code = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    log(f"{name}: scoring exit code {code}")


# ------------------------------------------------------------------ summary

def summary(arms: list[str]) -> Path:
    lines = [f"# Server baselines ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Qwen3.8-27B bf16 (vLLM, thinking off), 3 rounds, compact_16k, {JOBS} parallel runs; base arms hi@0.7.0 / gia@0.8.0, "
             "other arms the merged catalog hi@0.8.0 / gia@0.9.0.", ""]
    evid_rows = []
    for name in arms:
        arm = ARMS[name]
        if arm["dataset"] != "evid":
            continue
        path = ROOT / "experiments/evid/scores" / f"{arm['round']}.sonnet.subset.jsonl"
        if not path.is_file():
            evid_rows.append(f"| {name} | {arm['round']} | not scored yet | | | |")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

        def acc(sub):
            n = sum(r["n_gt"] for r in sub)
            p = sum(r["passed_gt"] for r in sub)
            return f"{p}/{n} = {100 * p / n:.1f}%" if n else "—"
        full = sum(1 for r in rows if r["n_gt"] and r["passed_gt"] == r["n_gt"])
        clean = [r for r in rows if r["sample_id"] not in TOUCHED]
        ai = [r for r in rows if r.get("category") == "ai_generation"]
        evid_rows.append(f"| {name} | {arm['round']} | {acc(rows)} | {100 * full / len(rows):.1f}% | {acc(clean)} | {acc(ai)} |")
    if evid_rows:
        lines += ["## EVID-Bench 131-222 (Sonnet judge, point level)", "",
                  "| arm | round | all | video level | not read during development | AI generation |", "|---|---|---|---|---|---|",
                  *evid_rows, "", "PC reference (Q4 Ollama, 2 parallel): evidF27 15.9% / clean 21.4%; evidC27G 33.6% / clean 33.0%.", ""]
    for name in arms:
        arm = ARMS[name]
        if arm["dataset"] == "evid":
            continue
        path = ROOT / "experiments/bench" / arm["dataset"] / "scores" / f"{arm['round']}.metrics.json"
        if not path.is_file():
            lines.append(f"- {name} ({arm['round']}): not scored yet")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        b = m["binary"]
        extra = (f"type macro-F1 {100 * m['types_macro_f1']:.1f}" if arm["dataset"] == "g360"
                 else f"5-class acc {100 * m['class5_acc']:.1f}, macro-F1 {100 * m['class5_macro_f1']:.1f}")
        lines.append(f"- **{name}** ({arm['round']}, n={m['n']}): Acc {100 * b['acc']:.1f}, Fake F1 {100 * b['fake_f1']:.1f}, "
                     f"macro-F1 {100 * b['macro_f1']:.1f}, verdict given {100 * b['parsed']:.0f}%; {extra}")
    lines += ["", "GroundLie360 paper reference: zero-shot FakeMark Acc 62.90 / F1 59.38 / type macro-F1 29.23."]
    path = LOG_DIR / "baselines_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if args.summary_only:
        print((summary(args.arms)).read_text(encoding="utf-8"))
        return 0
    if not vllm_up():
        log(f"model server not answering at {ENDPOINT}: run bash server/04_start_vllm.sh first")
        return 1
    log(f"==== baselines: arms {args.arms}; jobs {JOBS}; bench N {BENCH_N}; endpoint {ENDPOINT}; proxy "
        f"{os.environ.get('FOREN_HTTP_PROXY') or 'none'}")
    for name in args.arms:
        arm = ARMS[name]
        ids = evid_ids() if arm["dataset"] == "evid" else bench_ids(arm["dataset"])
        if not ids:
            log(f"{name}: no samples (videos missing?) -> skipped")
            continue
        log(f"---- {name}: {len(ids)} samples, round {arm['round']}, catalog {'merged' if arm['merged'] else 'old'}")
        run_with_progress(name, arm, ids, "first pass")
        failed = [i for i in ids if not is_done(run_dir(arm["dataset"], i, arm["round"]))]
        if failed:
            log(f"{name}: {len(failed)} runs did not finish: {failed[:10]}{' ...' if len(failed) > 10 else ''} -> one retry")
            run_with_progress(name, arm, ids, "retry")
        if not args.no_score:
            score(name, arm, ids)
        log(f"summary so far: {summary(args.arms)}")
    path = summary(args.arms)
    log(f"==== all arms done. Summary: {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
