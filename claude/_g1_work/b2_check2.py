"""B2 抽查：被标的词是否真在放行出处（E 的正文 / 更早的观察）里；仍扣的那条看学生输入里有没有。"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "src")
from foren_searcher.datagen.runlog import load_run
exp = Path(sys.argv[1])
rows = {}
for b in ("accepted", "held", "rejected"):
    for role in ("hi", "gia"):
        for line in (exp / f"{role}_{b}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line); m = r["metadata"]; rows[(m.get("run"), m.get("request"))] = r
table = [l for l in (exp / "gate_b2_review.md").read_text(encoding="utf-8").splitlines() if re.match(r"\| \d+ \|", l)]
logs = {}
for line in table:
    cells = [c.strip() for c in line.strip("|").split("|")]
    n, run, req, unit, words_cell, src_cell = cells[:6]
    try:
        src = json.loads(src_cell)
    except ValueError:
        src = {}
    r = rows[(run, req)]
    student = "".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
                      for m in r["messages"] if m["role"] != "assistant")
    if run not in logs:
        logs[run] = load_run(Path("experiments/sft_data/runs") / run)
    log = logs[run]
    res = {}
    for w, where in (src.items() if src else [(words_cell, None)]):
        if where and where.startswith("E"):
            try:
                content = log.lookup.get(where).content or ""
            except Exception as exc:  # noqa: BLE001
                content = ""
            from foren_searcher.datagen.export import _token_in
            res[w] = f"{where}:{_token_in(w, content)}"
        elif where:
            res[w] = f"{where}:(observation)"
        else:
            res[w] = f"student_input:{w in student or w.replace(',', '') in student}"
    print(n, run, req, unit, res)
for l in logs.values():
    l.close()
