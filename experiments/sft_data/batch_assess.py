"""Assess an explicit list of completed runs and export candidate targets."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    logs = ROOT / "batch_logs"
    logs.mkdir(exist_ok=True)
    for name in args.runs:
        run = ROOT / "runs" / name
        assessment = run / "assessment_r2"
        if assessment.exists():
            print(json.dumps({"run": name, "status": "assessment_exists_not_restarted"}), flush=True)
            continue
        with (logs / (name + "_assessment.log")).open("w", encoding="utf-8") as stream:
            print(json.dumps({"run": name, "status": "assessment_started"}), flush=True)
            status = 0
            for script in ("audit_run.py", "judge_run.py", "export_sft.py"):
                result = subprocess.run([sys.executable, str(ROOT / script), name], stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    status = result.returncode
                    break
        print(json.dumps({"run": name, "assessment_returncode": status}), flush=True)


if __name__ == "__main__":
    main()
