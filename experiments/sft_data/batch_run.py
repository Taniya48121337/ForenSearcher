"""Run explicitly selected independent cases serially and preserve every outcome."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("selection")
    args = parser.parse_args()
    selection = Path(args.selection)
    for entry in json.loads(selection.read_text(encoding="utf-8")):
        run = ROOT / "runs" / entry["run_name"]
        if run.exists():
            print(json.dumps({"run": entry["run_name"], "status": "already_exists_not_restarted"}), flush=True)
            continue
        command = [sys.executable, str(ROOT / "run_case.py"), "--dataset", entry["dataset"],
                   "--case", str(entry["case"]), "--mode", entry["mode"], "--run-name", entry["run_name"]]
        if "question_index" in entry:
            command.extend(["--question-index", str(entry["question_index"])])
        if entry.get("question_images"):
            command.append("--question-images")
        if entry.get("exclude_original_article"):
            command.append("--exclude-original-article")
        if entry.get("source_bound_search"):
            command.append("--source-bound-search")
        if entry.get("source_hint"):
            command.extend(["--source-hint", entry["source_hint"]])
        if "usage" in entry:
            command.extend(["--usage", entry["usage"]])
        if "instruction_file" in entry:
            command.extend(["--instruction-file", str(ROOT / entry["instruction_file"])])
        logs = ROOT / "batch_logs"
        logs.mkdir(exist_ok=True)
        with (logs / (entry["run_name"] + ".log")).open("w", encoding="utf-8") as stream:
            print(json.dumps({"run": entry["run_name"], "status": "started"}), flush=True)
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        outcomes, reports = [], 0
        trajectory = run / "trajectory.jsonl"
        if trajectory.exists():
            for line in trajectory.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["event_type"] == "workspace_outcome":
                    outcomes.append(event["payload"])
                if event["event_type"] == "workspace_report":
                    reports += 1
        result = {"run": entry["run_name"], "returncode": completed.returncode,
                  "committed_reports": reports, "outcomes": outcomes}
        with (ROOT / "batch_outcomes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
