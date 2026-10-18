"""分支可达探测：SEARCH image→text（覆盖表"期望 ≥ 1"、族 E L；r1 accepted 0 行）。

r1 起 E 族 E2_reverse_image、L 族 L4_reverse_image 都发 output_type=image（结果是载图页 + 图，分支名 SEARCH image→image；
r1 accepted 62 行），A 族的以图搜是 →video。本脚本静态核对：walker 里所有 input_type 为 image / text_image 的 SEARCH
决定各自的 output_type；再数 r0g / r1 / g1 run 里实际发出的 image→text 次数。全程不联网、Serper 0 点。
结论若是"没有任何规则发 image→text"，这一行是覆盖表（scripts/walker_coverage.py MATRIX）的口径问题，不是数据缺口：
要么把期望改到 SEARCH image→image（族 E L、最少 4），要么把 image→text 挪进 ALLOWED_ZERO——由负责人 / 门对话定，本对话不改。

    python scripts/_branch_probe_image_text.py
输出 claude/_longtail_work/probe_image_text.json
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter

from _branch_probe_common import OUT_DIR, ROOT, SFT, now, write_json

WALKER = ROOT / "src" / "foren_searcher" / "datagen" / "walker.py"
_RULE = re.compile(r'Decision\(\s*"(?:action|query_slot)",\s*(?:"([A-Z][\w]*)"|\(?\s*"([A-Z][\w]*)"\s+if)', re.S)


def static_rules() -> list[dict[str, str]]:
    """walker 源码里 input_type 是 image / text_image 的 SEARCH：取所在规则名与 output_type（字面量或 slot 默认）。"""

    src = WALKER.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r'"input_type":\s*"(image|text_image)"', src):
        window = src[m.start():m.start() + 400]
        out_m = re.search(r'"output_type":\s*(?:"(\w+)"|([^,}\n]+))', window)
        before = src[max(0, m.start() - 1500):m.start()]
        rules = [a or b for a, b in _RULE.findall(before)]
        out.append({"line": src.count("\n", 0, m.start()) + 1, "input_type": m.group(1),
                    "output_type": (out_m.group(1) or out_m.group(2).strip()) if out_m else "?",
                    "rule": rules[-1] if rules else "?"})
    # S1 槽（A3）：input_type 来自 slot，output_type 默认 video（替换上面按字面量抓到的同一行）
    for m in re.finditer(r'"input_type":\s*"text_image" if pair else "text",\s*\n?\s*"output_type":\s*"(\w+)"', src):
        line = src.count("\n", 0, m.start()) + 1
        out = [r for r in out if r["line"] != line]
        out.append({"line": line, "input_type": "text_image (S1 slot)", "output_type": m.group(1),
                    "rule": "A3_search_original_text_image"})
    for r in out:
        if r["rule"] == "?" and "slot.get" in r["output_type"]:
            r["rule"] = "S1 slot handler（A3 的实际 SEARCH，output_type 取槽里的值，默认 video）"
    return out


def count_runs() -> dict[str, dict[str, int]]:
    from foren_searcher.datagen.export import decision_branch

    out: dict[str, dict[str, int]] = {}
    for label, pattern in (("r0g", SFT / "runs" / "*_r0g_1"), ("r1", SFT / "runs" / "*_r1_1"),
                           ("g1", ROOT / ".wt_g1" / "experiments" / "sft_data" / "runs" / "*_r1g1_1")):
        c: Counter[str] = Counter()
        for req in glob.glob(str(pattern) + "/requests/*/"):
            try:
                payload = (json.load(open(req + "request.json", encoding="utf-8")) or {}).get("payload") or {}
                output = json.load(open(req + "response.json", encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(output, dict) or output.get("action") != "SEARCH":
                continue
            name = decision_branch(output, payload) or ""
            if name.startswith("SEARCH image") or name.startswith("SEARCH text_image"):
                c[name] += 1
        out[label] = dict(c)
    return out


def main() -> int:
    rules = static_rules()
    runs = count_runs()
    emits = [r for r in rules if r["output_type"] == "text"]
    payload = {"generated": now(), "image_search_rules": rules, "rules_emitting_image_to_text": emits, "run_counts": runs,
               "reachable_by_adding_cases": bool(emits),
               "conclusion": ("walker 没有任何规则发 image→text；E / L 的以图搜是 image→image（r1 起），A 是 →video。"
                              "覆盖表口径问题，不是数据缺口。" if not emits else "有规则发 image→text，见 rules_emitting_image_to_text")}
    write_json(OUT_DIR / "probe_image_text.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
