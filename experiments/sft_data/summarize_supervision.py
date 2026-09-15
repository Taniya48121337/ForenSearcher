"""Count role decisions and report conditions without treating rows as independent cases."""
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    summaries = []
    for entry in manifest["files"]:
        counts, cases, reports = Counter(), defaultdict(set), Counter()
        for line in (corpus / entry["file"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            target = json.loads(row["messages"][-1]["content"])
            kind = target.get("action", target.get("decision", "REPORT" if "findings" in target else "UNKNOWN"))
            metadata = row["metadata"]
            counts[kind] += 1
            cases[kind].add(metadata["case_family"])
            if kind != "REPORT":
                continue
            content = row["messages"][1]["content"]
            marker = "<image>\n" * len(row["images"])
            payload = json.loads(content.removeprefix(marker))["payload"]
            remaining = payload.get("model_view", {}).get("runtime_constraints", {}).get("remaining_turns")
            if metadata["origin"] == "hi_terminal_replay":
                condition = "report_regenerated_from_terminal_input"
            elif remaining == 0:
                condition = "last_available_turn"
            elif isinstance(remaining, int) and remaining > 0:
                condition = "turns_remaining"
            else:
                condition = "turn_budget_unknown"
            reports[(target.get("status", "UNKNOWN"), condition)] += 1
        summaries.append({"file": entry["file"], "rows": sum(counts.values()),
                          "decisions": [{"kind": kind, "rows": count, "case_families": len(cases[kind])}
                                        for kind, count in sorted(counts.items())],
                          "hi_report_conditions": [{"status": status, "condition": condition, "rows": count}
                                                   for (status, condition), count in sorted(reports.items())]})
    return {"files": summaries, "interpretation": [
        "A row is one supervised decision, not one independent task.",
        "A report at the last available turn trains report quality and completion status; it does not by itself demonstrate autonomous stopping.",
        "Remaining turns do not guarantee unlimited time or prove that stopping was discretionary; inspect the actual input and deadline.",
        "The distribution describes retained candidates, not task success or trained-student performance."
    ]}


if __name__ == "__main__":
    corpus = ROOT / "corpus"
    summary = summarize(corpus)
    (corpus / "supervision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
