"""按合好的 sft0 目录重算两份训练配置的 export_dir / warmup（每遍更新次数 = ceil(行 / grad_accum)，warmup = 总更新 × 5% 四舍五入）。"""
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1]); mid = d.name
for role, f in (("hi", "training/configs/hi_sft0_27b.json"), ("gia", "training/configs/gia_sft0_27b.json")):
    p = Path(f); c = json.loads(p.read_text(encoding="utf-8"))
    rows = sum(1 for l in (d / f"{role}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    per = math.ceil(rows / c["grad_accum"]); total = per * c["epochs"]; warm = round(total * 0.05)
    c["export_dir"] = f"dist/{mid}"; c["warmup_steps"] = warm
    c["_what"] = c["_what"].split("（循环第 0 轮训练集")[0] + f"（循环第 0 轮训练集 = r1 + g1 + g2 + g3，{mid}）" + c["_what"].split("）", 1)[1] if "（循环第 0 轮训练集" in c["_what"] else c["_what"]
    c["_why"]["warmup_steps"] = f"{warm}：{role.upper()} {rows} 行（{mid}，10-19 重合）、grad_accum {c['grad_accum']} → 每遍 {per} 次更新，{c['epochs']} 遍 {total} 次，5% 取整。"
    p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(role, rows, per, total, warm)
