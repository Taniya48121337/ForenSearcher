"""按 family 一次性分配五份划分（2026-09-28，案卷扩到 33 个后）。

    python scripts/_assign_splits.py            # 预览，不改文件
    python scripts/_assign_splits.py --apply    # 写回 cases/*.json 的 split，并写 experiments/sft_data/splits_manifest.json（不能放进 cases/，load_cases 会把它当案卷读）

规矩：
- 旧 11 个案卷（true 6 + videosham 5）固定 dev_train：它们已经用来改过核验器、规格和来源 D 的题，
  不能再进参与决策的集合，更不能进 dev_eval。
- 新 22 个案卷（20 个 family）按数据集分层配额分配。同一数据集内按 sha1(seed + family_key) 排序依次填配额，
  可以复现；换 seed 就能得到另一种分法。
- test 这里不分：按设计它是 EVID-Bench 与 TRUE test，锁定使用，不在当前案例池里。
- 同一 family 只进一个划分，最后跑 check_split_leakage 确认。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CASES = ROOT / "experiments" / "sft_data" / "cases"

from foren_searcher.datagen.cases import Case, check_split_leakage, split_summary  # noqa: E402

SEED = "forensearcher-splits-20260928"
FIXED_TRAIN = {"true", "videosham"}
# 每个数据集的 family 配额，顺序 = 填充顺序；各类任务在 dev_select / dev_eval 里都至少有一个
QUOTAS = {
    "averitec":     [("dev_select", 2), ("dev_confirm", 1), ("dev_eval", 1), ("dev_train", 99)],
    "averimatec":   [("dev_select", 2), ("dev_confirm", 1), ("dev_eval", 1), ("dev_train", 99)],
    "qvhighlights": [("dev_select", 1), ("dev_confirm", 1), ("dev_eval", 1), ("dev_train", 99)],
    "vcsl":         [("dev_eval", 1), ("dev_train", 99)],
}


def order_key(family: str) -> str:
    return hashlib.sha1(f"{SEED}|{family}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    paths = sorted(CASES.glob("*.json"))
    cases = {p: Case.load(p) for p in paths}
    families: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path, case in cases.items():
        families[case.dataset][case.family_key].append(path)
    plan: dict[Path, str] = {}
    for dataset, fams in families.items():
        if dataset in FIXED_TRAIN:
            for fam_paths in fams.values():
                for p in fam_paths:
                    plan[p] = "dev_train"
            continue
        if dataset not in QUOTAS:
            raise SystemExit(f"数据集 {dataset} 没有配额，先在 QUOTAS 里写明")
        slots = [split for split, n in QUOTAS[dataset] for _ in range(n)]
        # vcsl 的雪人歌曲 family 有 3 个案例，放 dev_eval 会让评测集偏向同一素材：多案例 family 排到最后
        ordered = sorted(fams, key=lambda f: (len(fams[f]) > 1, order_key(f)))
        for fam, split in zip(ordered, slots):
            for p in fams[fam]:
                plan[p] = split
    new_cases = []
    for path, case in cases.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("effective_split", None)
        data["split"] = plan[path]
        new_cases.append((path, data, Case.from_dict(data)))
    leaks = check_split_leakage([c for _, _, c in new_cases])
    if leaks:
        raise SystemExit("划分泄漏：\n" + "\n".join(leaks))
    summary = split_summary([c for _, _, c in new_cases])
    rows = sorted(((c.effective_split, c.dataset, c.case_id, c.family_key) for _, _, c in new_cases))
    lines = [f"seed={SEED}", json.dumps(summary, ensure_ascii=False)]
    lines += [f"  {s:12s} {d:13s} {cid:28s} {fam}" for s, d, cid, fam in rows]
    print("\n".join(lines))
    if not args.apply:
        return 0
    for path, data, _ in new_cases:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"seed": SEED, "created": "2026-09-28", "rule": __doc__.strip(), "quotas": QUOTAS,
                "fixed_dev_train_datasets": sorted(FIXED_TRAIN), "summary": summary,
                "cases": [{"case_id": cid, "dataset": d, "family_key": fam, "split": s} for s, d, cid, fam in rows]}
    (CASES.parent / "splits_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写回 {len(new_cases)} 个案卷与 splits_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
