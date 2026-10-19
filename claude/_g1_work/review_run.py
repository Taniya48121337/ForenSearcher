"""g1 冒烟自审：把一个 run 的动作序列、S1 / S3 选择、S4 报告、S5 答案、循环内检查、GT 核对打成一页文字。"""

import json
import sys
from pathlib import Path


def short(x, n=300):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s[:n].replace("\n", " ")


def main(run: str) -> None:
    rd = Path(run)
    cfg = json.loads((rd / "run_config.json").read_text(encoding="utf-8"))
    print("TASK:", short(cfg.get("task"), 400))
    case = json.loads(Path(cfg["case_file"]).read_text(encoding="utf-8"))
    print("GT:", short(case.get("checks"), 300))
    for req in sorted((rd / "requests").glob("request_*")):
        t = json.loads((req / "transport.json").read_text(encoding="utf-8")) if (req / "transport.json").is_file() else {}
        o = t.get("oracle") or {}
        slots = [s["name"] for s in t.get("slots") or []]
        line = f"{req.name[-4:]} {o.get('rule')} | {o.get('branch')} | {slots}"
        for name in ("slot_S1_query_choice.json", "slot_S3_next_choice.json"):
            if (req / name).is_file():
                line += f"\n      {name[5:7]}: " + short(json.loads((req / name).read_text(encoding="utf-8")), 260)
        resp = req / "response.json"
        if resp.is_file():
            r = json.loads(resp.read_text(encoding="utf-8"))
            out = r.get("output") or r
            if isinstance(out, dict) and out.get("action") in ("SEARCH",):
                line += "\n      args: " + short(out.get("arguments"), 220)
            if isinstance(out, dict) and (out.get("status") or out.get("findings")):
                line += "\n      REPORT " + str(out.get("status")) + " reason=" + short(out.get("status_reason"), 200)
                for f in out.get("findings") or []:
                    line += "\n        - " + short(f.get("statement") or f, 240)
            if isinstance(out, dict) and out.get("decision") == "FINISH":
                line += "\n      ANSWER: " + short(out.get("answer"), 600)
        if (req / "slot_S5_gt_check.json").is_file():
            line += "\n      GT_CHECK: " + short(json.loads((req / "slot_S5_gt_check.json").read_text(encoding="utf-8")), 300)
        print(line)
    inl = rd / "inloop_sufficiency.json"
    if inl.is_file():
        for c in json.loads(inl.read_text(encoding="utf-8"))["checks"]:
            print("INLOOP:", c.get("workspace"), c.get("status_in"), "->", c.get("status_out"), "flagged", c.get("flagged"),
                  "err", c.get("error"), "missing", short(c.get("missing"), 300))


if __name__ == "__main__":
    main(sys.argv[1])
