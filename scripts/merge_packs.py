"""sft0：把已校验的数据包（r1、g1、g2）合成循环第 0 轮的训练集（2026-10-18；口径 = 《长尾分支补量方案与三包合一前核对》六节定案）。

    python scripts/merge_packs.py --pack r1 experiments/sft_data/r1_pack/r1 dist/ForenSearcher_r1-….tar.gz \
        --pack g1 <g1 包目录> <g1 tar> --pack g2 <g2 包目录> <g2 tar> \
        --unanswerable experiments/sft_data/unanswerable_r1.json --hold experiments/sft_data/sft0_holds/hold_premature_match.json \
        --out-dir dist
    python scripts/merge_packs.py --verify dist/sft0-<日期>-<hash>

产物 dist/sft0-<日期>-<hash>/（hash = 合集 data 文件摘要的前 12 位）：
    hi.jsonl / gia.jsonl   各包 data/<role>_accepted.jsonl 的行（包的顺序 = 命令行顺序 = 流水线版本从低到高），三步筛 / 改：
      1. 定案 ②：同一 case_id 出现在几个包里时只留**最高版本**（命令行里最后出现的那个包）的行；
      2. --hold 清单里的行（pack + run + request + role）跳过，manifest 记 bucket_reason（例：rationale_premature_match）；
      3. 定案 ③ 降权（只 HI，一行不删）：报告行重标前固定 1、不受下面两个上限（10-19 裁决 3，实现 B）；动作行（unit=step）权重
         先截到 --step-cap（2）→ 每案例动作行权重和 > --case-cap（10）该案例动作行按同一系数缩 → 每分支权重和 > --branch-cap（60）该分支按同一系数缩 → **每个角色全局乘一个系数使
         平均权重 = 1**（sft_lora 按行乘权重、不归一，不重标梯度会小约 5 倍）。上限比较都在重标前的单位里做；重标只乘同一个
         系数，各分支 / 各族 / 各案例之间的比例不变。原权重留在 metadata.sample_weight_orig，各步系数在 metadata.weight_factors。
      每行 metadata 另加：pack（来源批次）、teacher_source（S0–S3 槽的教师：qwen / codex，按该包 provenance/runs.jsonl 的逐槽
      slot_models 判，不只看 S0；S0–S3 不一致记 mixed）、teacher_slots（S0–S5 模型）、family（A/B/C/E/L）、
      answerable（answerable / unanswerable / not_probed）、row_kind（unit:action）。messages、images 不改。
    images/                被留下的行引用的图（路径原样；三个包的 run 名带轮次后缀，撞名就报错）
    manifest.json          每包：pack_id、tar sha256、freeze（基础冻结文件 / 教师覆盖或 g1_freeze）、三个 hash、行数；三个 hash 跨包一致性；
                           去重 / 跳行 / 降权的明细；行数与权重按 族 × row_kind × 教师来源；族占比按行与按权重（门 C2 的补充列）
    SHA256SUMS

合并前对每个包先跑 pack_r0 的 --verify（不过就不合）。r0 包（hi@0.10.0 / 旧工具层 / verifier 0.3.6）三个 hash 与 r1 不同，
合不进来——定案 ②：r0 包只训基线、不进 sft0。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QWEN_HINT, CODEX_HINT = "qwen", "gpt-"


def _pack_module():
    spec = importlib.util.spec_from_file_location("pack_r0", ROOT / "scripts" / "pack_r0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PACK = _pack_module()


def family(case_id: str) -> str:
    ds, _, sid = str(case_id).partition(":")
    if ds in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if ds == "averimatec":
        return "E" if sid.startswith("e") else "L"
    if ds == "evid" or (ds == "fmnv" and sid.startswith("a_")):
        return "A"
    return "C"


def source_of(model: str | None) -> str:
    m = str(model or "").lower()
    return "qwen" if QWEN_HINT in m else "codex" if m.startswith(CODEX_HINT) else ("none" if not m else "other")


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def hashes(ident: dict[str, Any]) -> dict[str, Any]:
    return {"spec_hash": ident.get("spec_hash"), "tool_layer_hash": ident.get("tool_layer_hash"),
            "verifier_code_hash": ident.get("verifier_code_hash"), "verifier": ident.get("verifier"),
            "state_renderer": ident.get("state_renderer")}


def is_report(md: dict[str, Any]) -> bool:
    return md.get("unit") == "report" or str(md.get("branch") or "").startswith("report_")


def downweight(rows: list[dict[str, Any]], step_cap: float, case_cap: float | None, branch_cap: float | None) -> dict[str, Any]:
    """定案 ③ + 10-19 裁决 3（实现 B）（HI）：报告行重标前权重固定 1、不受每案例 / 每分支上限；动作行截顶 → 每案例 → 每分支
    （案例 / 分支的权重和只算动作行）。返回重标前的统计。就地写 metadata.sample_weight / weight_factors。"""

    w = []
    for r in rows:
        md = r["metadata"]
        x = float(md["sample_weight_orig"])
        f: dict[str, Any] = {}
        if is_report(md):
            f["report_fixed"] = 1.0                      # 原权重留在 sample_weight_orig
            x = 1.0
        elif md.get("unit") == "step" and x > step_cap:
            f["step_cap"] = round(step_cap / x, 6)
            x = step_cap
        w.append(x)
        md["weight_factors"] = f
    steps = [i for i, r in enumerate(rows) if not is_report(r["metadata"])]
    if case_cap:
        by_case: dict[str, list[int]] = defaultdict(list)
        for i in steps:
            by_case[rows[i]["metadata"]["case_id"]].append(i)
        for idx in by_case.values():
            tot = sum(w[i] for i in idx)
            if tot > case_cap:
                k = case_cap / tot
                for i in idx:
                    w[i] *= k
                    rows[i]["metadata"]["weight_factors"]["case_cap"] = round(k, 6)
    if branch_cap:
        by_branch: dict[str, list[int]] = defaultdict(list)
        for i in steps:
            by_branch[str(rows[i]["metadata"].get("branch") or "")].append(i)
        for idx in by_branch.values():
            tot = sum(w[i] for i in idx)
            if tot > branch_cap:
                k = branch_cap / tot
                for i in idx:
                    w[i] *= k
                    rows[i]["metadata"]["weight_factors"]["branch_cap"] = round(k, 6)
    per_case, per_branch = Counter(), Counter()
    for i in steps:
        per_case[rows[i]["metadata"]["case_id"]] += w[i]
        per_branch[rows[i]["metadata"].get("branch")] += w[i]
    for r, x in zip(rows, w):
        r["metadata"]["sample_weight"] = x
    reports = [x for r, x in zip(rows, w) if is_report(r["metadata"])]
    return {"weight_sum_before_rescale": round(sum(w), 3), "step_weight_sum_before_rescale": round(sum(w[i] for i in steps), 3),
            "report_rows": len(reports), "report_weight_sum_before_rescale": round(sum(reports), 3),
            "max_case_step_weight": round(max(per_case.values(), default=0), 4),
            "max_branch_weight": round(max(per_branch.values(), default=0), 4),
            "max_step_weight": round(max((w[i] for i in steps), default=0), 4)}


def rescale(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    k = len(rows) / sum(float(r["metadata"]["sample_weight"]) for r in rows)
    for r in rows:
        r["metadata"]["sample_weight"] = round(float(r["metadata"]["sample_weight"]) * k, 6)
        r["metadata"]["weight_factors"]["rescale"] = round(k, 6)
    return k


def shares(rows: list[dict[str, Any]], key: str, weighted: bool) -> dict[str, float]:
    c: Counter = Counter()
    for r in rows:
        c[r["metadata"][key]] += float(r["metadata"]["sample_weight"]) if weighted else 1
    tot = sum(c.values()) or 1
    return {k: round(v / tot, 4) for k, v in c.most_common()}


def tail_share(rows: list[dict[str, Any]]) -> float:
    n, cases, w = Counter(), defaultdict(set), Counter()
    for r in rows:
        b = r["metadata"].get("branch")
        n[b] += 1
        cases[b].add(r["metadata"]["case_id"])
        w[b] += float(r["metadata"]["sample_weight"])
    tail = {b for b in n if n[b] < 15 or len(cases[b]) < 5}
    return round(sum(w[b] for b in tail) / (sum(w.values()) or 1), 4)


def merge(args: argparse.Namespace) -> int:
    unans: set[str] = set()
    for p in args.unanswerable:
        unans |= set(json.loads(resolve(p).read_text(encoding="utf-8")).get("unanswerable_case_ids") or [])
    holds: dict[tuple[str, str, str, str], str] = {}
    hold_files = []
    for p in args.hold:
        h = json.loads(resolve(p).read_text(encoding="utf-8"))
        hold_files.append({"file": str(p), "sha256": PACK.sha256_file(resolve(p)), "bucket_reason": h["bucket_reason"], "n": len(h["rows"])})
        for r in h["rows"]:
            holds[(r["pack"], r["role"], r["run"], r["request"])] = h["bucket_reason"]

    packs = []
    for name, pack_dir, tar in args.pack:
        pdir = resolve(pack_dir)
        print(f"[{name}] 校验 {pdir}")
        if PACK.verify(pdir) != 0:
            raise SystemExit(f"{name}：pack_r0 --verify 不过，不合并")
        ident_file = sorted(pdir.glob("*_pack.json"))[0]
        identity = json.loads(ident_file.read_text(encoding="utf-8"))
        runs = {}
        for line in (pdir / "provenance" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                runs[p["run"]] = p
        tar_path = resolve(tar)
        if not tar_path.is_file():
            raise SystemExit(f"{name}：tar 不在 {tar_path}")
        print(f"[{name}] tar sha256 …")
        packs.append({"name": name, "dir": pdir, "identity": identity, "identity_file": ident_file.name, "runs": runs,
                      "tar": tar_path, "tar_sha256": PACK.sha256_file(tar_path),
                      "freeze_files": sorted(p.name for p in (pdir / "freeze").glob("*.json"))})
    base = hashes(packs[0]["identity"])
    mismatch = {p["name"]: hashes(p["identity"]) for p in packs if hashes(p["identity"]) != base}
    if mismatch:
        raise SystemExit(f"三个 hash 跨包不一致：基准 {packs[0]['name']} = {base}；不一致 {mismatch}")

    # 读行；定案 ②：同一 case_id 只留最高版本（命令行最后）的包
    loaded = []
    for i, p in enumerate(packs):
        for role in ("hi", "gia"):
            for line in (p["dir"] / "data" / f"{role}_accepted.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    loaded.append((i, role, json.loads(line)))
    owner: dict[str, int] = {}
    for i, _role, row in loaded:
        owner[str(row["metadata"]["case_id"])] = i
    kept: dict[str, list[dict[str, Any]]] = {"hi": [], "gia": []}
    dropped_dedup: Counter = Counter()
    dropped_hold: Counter = Counter()
    hold_rows_seen = set()
    for i, role, row in loaded:
        p = packs[i]
        md = row["metadata"]
        case = str(md["case_id"])
        if owner[case] != i:
            dropped_dedup[f"{p['name']}→{packs[owner[case]]['name']}"] += 1
            continue
        key = (p["name"], role, str(md.get("run")), str(md.get("request")))
        if key in holds:
            dropped_hold[f"{p['name']}:{holds[key]}"] += 1
            hold_rows_seen.add(key)
            continue
        slots = (p["runs"].get(str(md.get("run"))) or {}).get("slot_models") or {}
        s03 = {source_of(slots.get(k)) for k in ("S0", "S1", "S2", "S3")}
        fam = family(case)
        md.update({"pack": p["name"], "teacher_source": s03.pop() if len(s03) == 1 else "mixed", "teacher_slots": slots,
                   "family": fam, "answerable": ("unanswerable" if case in unans else "answerable") if fam in ("E", "L") else "not_probed",
                   "row_kind": f"{md.get('unit')}:{md.get('action') or ''}", "sample_weight_orig": float(md.get("sample_weight") or 0)})
        kept[role].append(row)
    unmatched_holds = sorted(set(holds) - hold_rows_seen)
    if unmatched_holds:
        raise SystemExit(f"--hold 清单里有 {len(unmatched_holds)} 行在包里找不到（例：{unmatched_holds[:3]}）")

    before = {"family_rows": shares(kept["hi"], "family", False), "family_weight": shares(kept["hi"], "family", True),
              "tail_share": tail_share(kept["hi"])}
    dw = downweight(kept["hi"], args.step_cap, args.case_cap, args.branch_cap)
    for r in kept["gia"]:
        r["metadata"]["weight_factors"] = {}
    dw["tail_share_after"] = tail_share(kept["hi"])
    factors = {role: round(rescale(rows), 6) for role, rows in kept.items()}

    tmp = resolve(args.out_dir) / f"sft0-tmp-{time.strftime('%H%M%S')}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images").mkdir(parents=True)
    image_owner: dict[str, str] = {}
    by_cell: Counter = Counter()
    by_cell_w: Counter = Counter()
    for role, rows in kept.items():
        lines = []
        for row in rows:
            md = row["metadata"]
            p = next(x for x in packs if x["name"] == md["pack"])
            for rel in row.get("images") or []:
                owner_name = image_owner.get(rel)
                if owner_name and owner_name != p["name"]:
                    raise SystemExit(f"图路径撞名：{rel}（{owner_name} 与 {p['name']}）")
                if owner_name is None:
                    target = tmp / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p["dir"] / "data" / rel, target)
                    image_owner[rel] = p["name"]
            by_cell[(role, md["family"], md["row_kind"], md["teacher_source"])] += 1
            by_cell_w[(role, md["family"], md["row_kind"], md["teacher_source"])] += float(md["sample_weight"])
            lines.append(json.dumps(row, ensure_ascii=False))
        (tmp / f"{role}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_files = [tmp / "hi.jsonl", tmp / "gia.jsonl", *sorted((tmp / "images").rglob("*"))]
    digest = PACK.data_digest({f.relative_to(tmp).as_posix(): PACK.sha256_file(f) for f in data_files if f.is_file()})
    merged_id = f"sft0-{time.strftime('%Y%m%d')}-{digest[:12]}"
    manifest = {
        "merged_id": merged_id, "training_set": "sft0", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data_digest": digest,
        "made_by": "scripts/merge_packs.py", "git": PACK.git_info(),
        "decisions": "《长尾分支补量方案与三包合一前核对_20261018》六节 ①②③⑥；hold 清单（负责人 10-18 晚）",
        "packs": [{"name": p["name"], "pack_id": p["identity"].get("pack_id"), "status": p["identity"].get("status"),
                   "pack_dir": str(p["dir"]), "identity_file": p["identity_file"], "tar": str(p["tar"]), "tar_sha256": p["tar_sha256"],
                   "data_digest": p["identity"].get("data_digest"), "freeze_files": p["freeze_files"],
                   "freeze_id": p["identity"].get("freeze_id"), "teacher_override": p["identity"].get("teacher_override"),
                   "git_head": (p["identity"].get("git") or {}).get("head"), **hashes(p["identity"]),
                   "rows_in_pack": (p["identity"].get("counts") or {}),
                   "rows_kept": {r: sum(1 for x in kept[r] if x["metadata"]["pack"] == p["name"]) for r in ("hi", "gia")},
                   "teacher_source_rows": dict(Counter(x["metadata"]["teacher_source"] for r in kept for x in kept[r] if x["metadata"]["pack"] == p["name"]))}
                  for p in packs],
        "hashes_consistent": True, "hashes": base,
        "dedup": {"rule": "同一 case_id 只留最高版本（命令行最后出现）的包的行（定案 ②）", "rows_dropped": dict(dropped_dedup)},
        "holds": {"files": hold_files, "rows_skipped": dict(dropped_hold)},
        "weights": {"rule": "定案 ③ + 10-19 裁决 3（实现 B）：只 HI；报告行重标前固定 1、不受每案例 / 每分支上限；动作行先截到 step_cap → "
                            "每案例（只算动作行）≤ case_cap → 每分支 ≤ branch_cap；上限都在重标前的单位里比；之后每个角色乘同一系数使平均权重 = 1",
                    "step_cap": args.step_cap, "case_cap": args.case_cap, "branch_cap": args.branch_cap,
                    "hi_before": before, "hi_downweight": dw, "rescale_factor": factors,
                    "mean_weight": {r: round(sum(float(x["metadata"]["sample_weight"]) for x in v) / max(1, len(v)), 6) for r, v in kept.items()},
                    "hi_report_weight_share": round(sum(float(x["metadata"]["sample_weight"]) for x in kept["hi"] if is_report(x["metadata"]))
                                                    / max(1e-9, sum(float(x["metadata"]["sample_weight"]) for x in kept["hi"])), 4),
                    "hi_family_share_rows": shares(kept["hi"], "family", False),
                    "hi_family_share_weight": shares(kept["hi"], "family", True)},
        "rows": {r: len(v) for r, v in kept.items()}, "images": len(image_owner),
        "rows_by_role_family_kind_teacher": [{"role": k[0], "family": k[1], "row_kind": k[2], "teacher_source": k[3], "rows": v,
                                             "weight": round(by_cell_w[k], 3)} for k, v in sorted(by_cell.items())],
        "row_fields_added": ["pack", "teacher_source", "teacher_slots", "family", "answerable", "row_kind", "sample_weight_orig", "weight_factors"],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sums = [f"{PACK.sha256_file(f)}  {f.relative_to(tmp).as_posix()}" for f in sorted(tmp.rglob("*")) if f.is_file() and f.name != "SHA256SUMS"]
    (tmp / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    final = tmp.parent / merged_id
    if final.exists():
        if not args.overwrite:
            raise SystemExit(f"{final} 已存在；要覆盖加 --overwrite（临时目录留在 {tmp}）")
        shutil.rmtree(final)
    tmp.rename(final)
    print(json.dumps({"merged": str(final), "merged_id": merged_id, "rows": manifest["rows"], "images": manifest["images"],
                      "dedup": manifest["dedup"]["rows_dropped"], "holds": manifest["holds"]["rows_skipped"],
                      "weights": {k: manifest["weights"][k] for k in ("rescale_factor", "mean_weight", "hi_family_share_rows", "hi_family_share_weight")}},
                     ensure_ascii=False, indent=2))
    return 0


def verify(path: Path) -> int:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    listed = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d, rel = line.split("  ", 1)
            listed[rel] = d
    bad = [rel for rel, d in listed.items() if not (path / rel).is_file() or PACK.sha256_file(path / rel) != d]
    extra = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
             and p.relative_to(path).as_posix() not in listed]
    data_hashes = {rel: d for rel, d in listed.items() if rel in ("hi.jsonl", "gia.jsonl") or rel.startswith("images/")}
    digest_ok = PACK.data_digest(data_hashes) == manifest.get("data_digest") and path.name == manifest.get("merged_id")
    w = manifest.get("weights") or {}
    problems, rows, means = [], {}, {}
    case_pack: dict[str, str] = {}
    pre_case, pre_branch = Counter(), Counter()
    need = ("pack", "teacher_source", "teacher_slots", "family", "answerable", "row_kind", "sample_weight_orig", "weight_factors")
    for role in ("hi", "gia"):
        ws = []
        for i, line in enumerate((path / f"{role}.jsonl").read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            md = row.get("metadata") or {}
            ws.append(float(md.get("sample_weight") or 0))
            if md.get("role") != role or not all(k in md for k in need):
                problems.append(f"{role}#{i}: metadata 字段不全")
            if ws[-1] <= 0:
                problems.append(f"{role}#{i}: sample_weight={md.get('sample_weight')}")
            if (row.get("messages") or [{}])[-1].get("role") != "assistant":
                problems.append(f"{role}#{i}: 最后一条不是 assistant")
            for rel in row.get("images") or []:
                if not (path / rel).is_file():
                    problems.append(f"{role}#{i}: 图不在 {rel}")
            c = str(md.get("case_id"))
            if case_pack.setdefault(c, md.get("pack")) != md.get("pack"):
                problems.append(f"{role}#{i}: case {c} 跨包（{case_pack[c]} / {md.get('pack')}），违反去重")
            if role == "hi":
                k = (md.get("weight_factors") or {}).get("rescale", 1.0)
                pre = ws[-1] / k
                if is_report(md):
                    if abs(pre - 1.0) > 1e-6:
                        problems.append(f"hi#{i}: 报告行重标前权重 {pre:.4f} ≠ 1（裁决 3 实现 B）")
                else:
                    pre_branch[md.get("branch")] += pre
                    pre_case[c] += pre
                if md.get("unit") == "step" and pre > float(w.get("step_cap") or 1e9) + 1e-6:
                    problems.append(f"hi#{i}: 动作行重标前权重 {pre:.4f} > step_cap")
        rows[role] = len(ws)
        means[role] = sum(ws) / max(1, len(ws))
    eps = 1e-3
    checks = {
        "mean_weight_is_1": all(abs(m - 1) < eps for m in means.values()),
        "case_cap": max(pre_case.values(), default=0) <= float(w.get("case_cap") or 1e9) + eps,
        "branch_cap": max(pre_branch.values(), default=0) <= float(w.get("branch_cap") or 1e9) + eps,
        "rows_match_manifest": rows == manifest.get("rows"), "hashes_consistent": bool(manifest.get("hashes_consistent")),
        "no_cross_pack_case": not any("跨包" in p for p in problems),
    }
    ok = not bad and not extra and digest_ok and not problems and all(checks.values())
    print(json.dumps({"dir": str(path), "merged_id": manifest.get("merged_id"), "files": len(listed), "changed_or_missing": bad[:20],
                      "extra": extra[:20], "data_digest_ok": digest_ok, "rows": rows, "mean_weight": {k: round(v, 6) for k, v in means.items()},
                      "max_case_step_weight_pre_rescale": round(max(pre_case.values(), default=0), 4),
                      "max_branch_weight_pre_rescale": round(max(pre_branch.values(), default=0), 4),
                      "checks": checks, "problems": problems[:20], "ok": ok}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", nargs=3, action="append", metavar=("NAME", "PACK_DIR", "TAR"), help="按流水线版本从低到高给")
    ap.add_argument("--unanswerable", nargs="*", default=[])
    ap.add_argument("--hold", nargs="*", default=[], help="跳行清单（scripts/sft0_holds.py 的产物）")
    ap.add_argument("--step-cap", type=float, default=2.0)
    ap.add_argument("--case-cap", type=float, default=10.0)
    ap.add_argument("--branch-cap", type=float, default=60.0)
    ap.add_argument("--out-dir", default="dist")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", default=None)
    args = ap.parse_args(argv)
    if args.verify:
        return verify(resolve(args.verify))
    if not args.pack:
        ap.error("需要 --pack 或 --verify")
    return merge(args)


if __name__ == "__main__":
    sys.exit(main())
