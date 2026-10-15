"""看图复核（image_confirm）离线评估：r0c 替换测试的 76 个图片对（替换后 / 原图），以及金标批次里含图片的条目。

    FOREN_IMAGE_CONFIRM=1 python scripts/_image_confirm_eval.py <输出目录>
"""
import asyncio, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, "src")
from foren_searcher.datagen.verify.calibrate_precision import load_units, all_pairs, cli_judge
from foren_searcher.datagen.verify.anchored import judge_pair
from foren_searcher.datagen.verify.image_confirm import IMAGE_CONFIRM, confirm_image_judgment
IMAGE_CONFIRM["enabled"] = True

SFT = Path("experiments/sft_data")
OUT = Path(sys.argv[1])
units, runs = load_units(SFT / "verdicts_r0c_judged.claude-cli-claude-sonnet-5-5.11runs-c8436db3.v034.jsonl",
                         SFT / "verdicts_units_r0c_judged.claude-cli-claude-sonnet-5-5.11runs-c8436db3.v034.jsonl", SFT / "runs")
pairs = {p.key: p for p in all_pairs(units)}
rows = json.load(open(SFT / "precision_r0c_v034.rows.json", encoding="utf-8"))["swap"]
img = [r for r in rows if r["modality"] == "image"]
pair_judge = cli_judge("sonnet", SFT / "verify_cache.precision.sonnet.jsonl")
conf_judge = cli_judge("sonnet", OUT / "confirm_cache.jsonl")
strata = {s["unit"]: s for s in json.load(open(SFT / "gold_set/v1p2_hard/strata.json", encoding="utf-8"))["strata"]}

async def main():
    out = {"swap": [], "orig": [], "gold": []}
    async def swap_one(r):
        p = pairs.get(r["key"])
        if p is None or not Path(r["donor_image"]).is_file():
            return None
        t = replace(p.text, image_path=Path(r["donor_image"]), media_note="")
        j = await judge_pair(pair_judge, p.unit.finding_id, p.unit.statement, t, task=p.unit.task)
        j2 = await confirm_image_judgment(conf_judge, j, t, p.unit.statement, task=p.unit.task)
        return {"key": r["key"], "before": j.verdict, "after": j2.verdict, "kind": r.get("stamp_kind"),
                "info": j2.raw.get("image_confirm"), "parts": [x.part for x in j.supported]}
    async def orig_one(p):
        j = await judge_pair(pair_judge, p.unit.finding_id, p.unit.statement, p.text, task=p.unit.task)
        j2 = await confirm_image_judgment(conf_judge, j, p.text, p.unit.statement, task=p.unit.task)
        return {"key": p.key, "before": j.verdict, "after": j2.verdict, "info": j2.raw.get("image_confirm"),
                "parts": [x.part for x in j.supported], "statement": p.unit.statement[:160],
                "gold": (strata.get(p.unit.key) or {}).get("id")}
    sw = [x for x in await asyncio.gather(*(swap_one(r) for r in img)) if x]
    orig_keys = [r["key"] for r in img if r["key"] in pairs]
    gold_keys = [k for k, p in pairs.items() if p.unit.key in strata and p.text.is_image and p.text.image_path]
    keys = list(dict.fromkeys(orig_keys + gold_keys))
    og = await asyncio.gather(*(orig_one(pairs[k]) for k in keys))
    out["swap"] = sw
    out["orig"] = og
    json.dump(out, open(OUT / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("swap pairs", len(sw), "stamped before", sum(x["before"] == "supported" for x in sw), "after", sum(x["after"] == "supported" for x in sw))
    sup = [x for x in og if x["before"] == "supported"]
    print("orig image pairs", len(og), "supported before", len(sup), "downgraded", sum(x["after"] != "supported" for x in sup),
          "parts dropped in", sum(1 for x in sup if x["info"] and x["info"]["dropped"]))
    print(pair_judge.summary()); print(conf_judge.summary())
asyncio.run(main())
