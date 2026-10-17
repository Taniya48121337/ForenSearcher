"""对话 3 补：Opus D2 通读标的 3 处 hidden_knowledge / GIA 泄漏，按 request.json 程序核对学生（这一步模型）实际看到的输入。
    python claude/_r1a_work/leak_check.py
"""
import json, re, sys
from pathlib import Path

RUNS = Path("experiments/sft_data/runs")


def req(run, rid):
    d = RUNS / run / "requests" / rid
    return (json.loads((d / "request.json").read_text(encoding="utf-8")),
            json.loads((d / "response.json").read_text(encoding="utf-8")) if (d / "response.json").is_file() else {},
            json.loads((d / "transport.json").read_text(encoding="utf-8")) if (d / "transport.json").is_file() else {})


def show(run, rid, needles):
    q, r, t = req(run, rid)
    payload = q.get("payload") or {}
    rendered = q.get("rendered_payload")
    raw = json.dumps(payload, ensure_ascii=False)
    ren = rendered if isinstance(rendered, str) else json.dumps(rendered, ensure_ascii=False)
    out = json.dumps(r, ensure_ascii=False)
    print(f"== {run} {rid}  rule={(t.get('oracle') or {}).get('rule')}  ws={payload.get('workspace_id') or 'GIA'}")
    print("   output:", out[:400])
    for n in needles:
        for name, text in (("payload", raw), ("rendered_payload", ren), ("output", out)):
            hits = [m.start() for m in re.finditer(re.escape(n), text, re.I)]
            ctx = text[max(0, hits[0] - 160): hits[0] + 120].replace("\n", " ") if hits else ""
            print(f"   [{n!r}] {name}: {len(hits)} 处 {ctx!r}")
    return q, r, t


if __name__ == "__main__":
    for run, rid, needles in (("fmnv_c_RXR0B1SxiU0_r1_1", "request_0003", []),
                              ("groundlie360_v1959500_r1_1", "request_0003", ["living worms", "worm"]),
                              ("groundlie360_v1959500_r1_1", "request_0014", ["dissolve in hot water", "dissolve", "E20"])):
        show(run, rid, needles)
