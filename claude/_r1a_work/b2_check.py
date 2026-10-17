"""B2 抽查的程序比对：每条抽查行里被标的词，是否出现在该行学生可见的输入（user / system 消息）里。"""
import json, re, sys
from pathlib import Path
exp = Path(sys.argv[1])
rows = {}
for b in ("accepted", "held", "rejected"):
    for role in ("hi", "gia"):
        p = exp / f"{role}_{b}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); m = r["metadata"]
            rows[(m.get("run"), m.get("request"))] = r
table = [l for l in (exp / "gate_b2_review.md").read_text(encoding="utf-8").splitlines() if re.match(r"\| \d+ \|", l)]
out = []
for line in table:
    cells = [c.strip() for c in line.strip("|").split("|")]
    n, run, req, unit, words = cells[0], cells[1], cells[2], cells[3], [w.strip() for w in cells[4].split(",")]
    r = rows.get((run, req))
    if r is None:
        out.append((n, run, req, words, "row not found")); continue
    text = ""
    for msg in r["messages"]:
        if msg["role"] != "assistant":
            c = msg["content"]
            text += c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    low = text.lower()
    vis = {w: (w.lower() in low or w.lower().replace(",", "") in low.replace(",", "")) for w in words}
    out.append((n, run, req, words, vis, r["metadata"].get("bucket"), unit))
for o in out:
    print(" | ".join(str(x) for x in o))
