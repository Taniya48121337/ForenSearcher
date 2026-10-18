"""r0 训练集定稿：把几个已校验的数据包（r1、g1、g2）合成一个训练目录（2026-10-19）。

    python scripts/merge_packs.py --pack r1 experiments/sft_data/r1_pack/r1 dist/ForenSearcher_r1-….tar.gz \
        --pack g1 <g1 包目录> <g1 tar> --pack g2 <g2 包目录> <g2 tar> \
        --unanswerable experiments/sft_data/unanswerable_r1.json --out-dir dist
    python scripts/merge_packs.py --verify dist/r0train-<日期>-<hash>

产物 dist/r0train-<日期>-<hash>/（hash = 合集 data 文件摘要的前 12 位）：
    hi.jsonl / gia.jsonl   各包 data/<role>_accepted.jsonl 原样拼接（包的顺序 = 命令行顺序），每行 metadata 另加
                           pack、teacher_source（S0–S3 的教师：qwen / codex）、family（A / B / C / E / L）、
                           answerable（answerable / unanswerable / not_probed）、row_kind（unit:action）；
                           sample_weight、messages、images 都不改
    images/                各包被行引用的图（路径原样；三个包的 run 名带轮次后缀，不会撞名——撞了就报错）
    manifest.json          每包：pack_id、tar 的 sha256、身份文件里的 spec_hash / tool_layer_hash / verifier_code_hash、data_digest、行数；
                           三个 hash 跨包一致性；合集行数按 族 × row_kind × 教师来源；跨包同一 case_family 的重叠
    SHA256SUMS             目录里除它以外每个文件的 sha256

合并前对每个包先跑 pack_r0 的 --verify（不过就不合）；不改任何行的入选与权重（那是 export 的事）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


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


def teacher_source(identity: dict[str, Any]) -> str:
    """S0–S3 的教师：包身份文件里有 teacher_override（g1_teacher.json：S0–S3 = Codex）→ codex；否则看冻结的 qwen 后端 → qwen。"""

    if identity.get("teacher_override"):
        return "codex"
    models = " ".join(str(k) for k in (identity.get("teacher_model") or {}))
    return "qwen" if "qwen" in models.lower() else "other"


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def merge(args: argparse.Namespace) -> int:
    unans: set[str] = set()
    for p in args.unanswerable:
        unans |= set(json.loads(resolve(p).read_text(encoding="utf-8")).get("unanswerable_case_ids") or [])
    probed_families = {"E", "L"}

    packs = []
    for name, pack_dir, tar in args.pack:
        pdir = resolve(pack_dir)
        print(f"[{name}] 校验 {pdir}")
        if PACK.verify(pdir) != 0:
            raise SystemExit(f"{name}：pack_r0 --verify 不过，不合并")
        ident_files = sorted(pdir.glob("*_pack.json"))
        identity = json.loads(ident_files[0].read_text(encoding="utf-8"))
        tar_path = resolve(tar)
        if not tar_path.is_file():
            raise SystemExit(f"{name}：tar 不在 {tar_path}")
        print(f"[{name}] tar sha256 …")
        packs.append({"name": name, "dir": pdir, "identity": identity, "identity_file": ident_files[0].name, "tar": tar_path,
                      "tar_sha256": PACK.sha256_file(tar_path), "teacher_source": teacher_source(identity)})

    # 三个 hash 跨包一致
    def hashes(ident: dict[str, Any]) -> dict[str, Any]:
        return {"spec_hash": ident.get("spec_hash"), "tool_layer_hash": ident.get("tool_layer_hash"),
                "verifier_code_hash": ident.get("verifier_code_hash"), "verifier": ident.get("verifier"),
                "state_renderer": ident.get("state_renderer")}
    base = hashes(packs[0]["identity"])
    mismatch = {p["name"]: hashes(p["identity"]) for p in packs if hashes(p["identity"]) != base}
    if mismatch and not args.allow_hash_mismatch:
        raise SystemExit(f"三个 hash 跨包不一致：基准 {packs[0]['name']} = {base}；不一致 {mismatch}")

    tmp = resolve(args.out_dir) / f"r0train-tmp-{time.strftime('%H%M%S')}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images").mkdir(parents=True)
    out_rows: dict[str, list[str]] = {"hi": [], "gia": []}
    counts: Counter = Counter()
    by_fam_kind_teacher: Counter = Counter()
    weights: Counter = Counter()
    families_by_pack: dict[str, set[str]] = defaultdict(set)
    image_owner: dict[str, str] = {}
    for p in packs:
        data = p["dir"] / "data"
        for role in ("hi", "gia"):
            for line in (data / f"{role}_accepted.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                md = row.setdefault("metadata", {})
                case = str(md.get("case_id"))
                fam = family(case)
                action = md.get("action") or ""
                md.update({"pack": p["name"], "teacher_source": p["teacher_source"], "family": fam,
                           "answerable": ("unanswerable" if case in unans else "answerable") if fam in probed_families else "not_probed",
                           "row_kind": f"{md.get('unit')}:{action}"})
                for rel in row.get("images") or []:
                    owner = image_owner.get(rel)
                    if owner and owner != p["name"]:
                        raise SystemExit(f"图路径撞名：{rel}（{owner} 与 {p['name']}）")
                    if owner is None:
                        target = tmp / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(data / rel, target)
                        image_owner[rel] = p["name"]
                out_rows[role].append(json.dumps(row, ensure_ascii=False))
                counts[(p["name"], role)] += 1
                by_fam_kind_teacher[(role, fam, md["row_kind"], p["teacher_source"])] += 1
                weights[(p["name"], role)] += float(md.get("sample_weight") or 0)
                families_by_pack[p["name"]].add(str(md.get("case_family") or case))
    for role, lines in out_rows.items():
        (tmp / f"{role}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    overlap = {}
    names = [p["name"] for p in packs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(families_by_pack[a] & families_by_pack[b])
            if shared:
                overlap[f"{a}&{b}"] = shared[:50]

    data_files = [tmp / "hi.jsonl", tmp / "gia.jsonl", *sorted((tmp / "images").rglob("*"))]
    data_hashes = {f.relative_to(tmp).as_posix(): PACK.sha256_file(f) for f in data_files if f.is_file()}
    digest = PACK.data_digest(data_hashes)
    merged_id = f"r0train-{time.strftime('%Y%m%d')}-{digest[:12]}"
    manifest = {
        "merged_id": merged_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data_digest": digest,
        "made_by": "scripts/merge_packs.py", "git": PACK.git_info(),
        "packs": [{"name": p["name"], "pack_id": p["identity"].get("pack_id"), "status": p["identity"].get("status"),
                   "pack_dir": str(p["dir"]), "identity_file": p["identity_file"], "tar": str(p["tar"]), "tar_sha256": p["tar_sha256"],
                   "data_digest": p["identity"].get("data_digest"), "teacher_source": p["teacher_source"],
                   "teacher_override": p["identity"].get("teacher_override"), **hashes(p["identity"]),
                   "rows": {r: counts[(p["name"], r)] for r in ("hi", "gia")},
                   "sample_weight_sum": {r: round(weights[(p["name"], r)], 3) for r in ("hi", "gia")}} for p in packs],
        "hashes_consistent": not mismatch, "hash_mismatch": mismatch or None, "hashes": base,
        "rows": {r: len(v) for r, v in out_rows.items()}, "images": len(image_owner),
        "rows_by_role_family_kind_teacher": [{"role": k[0], "family": k[1], "row_kind": k[2], "teacher_source": k[3], "rows": v}
                                             for k, v in sorted(by_fam_kind_teacher.items())],
        "case_family_overlap_between_packs": overlap,
        "row_fields_added": ["pack", "teacher_source", "family", "answerable", "row_kind"],
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
                      "hashes_consistent": manifest["hashes_consistent"], "overlap_pairs": {k: len(v) for k, v in overlap.items()}},
                     ensure_ascii=False, indent=2))
    return 0


def verify(path: Path) -> int:
    sums = path / "SHA256SUMS"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d, rel = line.split("  ", 1)
            listed[rel] = d
    bad = [rel for rel, d in listed.items() if not (path / rel).is_file() or PACK.sha256_file(path / rel) != d]
    extra = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
             and p.relative_to(path).as_posix() not in listed]
    data_hashes = {rel: d for rel, d in listed.items() if rel in ("hi.jsonl", "gia.jsonl") or rel.startswith("images/")}
    digest_ok = PACK.data_digest(data_hashes) == manifest.get("data_digest") and path.name == manifest.get("merged_id")
    problems = []
    rows = {}
    for role in ("hi", "gia"):
        n = 0
        for i, line in enumerate((path / f"{role}.jsonl").read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            md = row.get("metadata") or {}
            if md.get("role") != role or not all(k in md for k in ("pack", "teacher_source", "family", "answerable", "row_kind")):
                problems.append(f"{role}#{i}: metadata 字段不全")
            if float(md.get("sample_weight") or 0) <= 0:
                problems.append(f"{role}#{i}: sample_weight={md.get('sample_weight')}")
            if (row.get("messages") or [{}])[-1].get("role") != "assistant":
                problems.append(f"{role}#{i}: 最后一条不是 assistant")
            for rel in row.get("images") or []:
                if not (path / rel).is_file():
                    problems.append(f"{role}#{i}: 图不在 {rel}")
        rows[role] = n
    rows_ok = rows == manifest.get("rows")
    ok = not bad and not extra and digest_ok and not problems and rows_ok and manifest.get("hashes_consistent")
    print(json.dumps({"dir": str(path), "merged_id": manifest.get("merged_id"), "files": len(listed), "changed_or_missing": bad[:20],
                      "extra": extra[:20], "data_digest_ok": digest_ok, "rows": rows, "rows_match_manifest": rows_ok,
                      "hashes_consistent": manifest.get("hashes_consistent"), "problems": problems[:20], "ok": ok},
                     ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", nargs=3, action="append", metavar=("NAME", "PACK_DIR", "TAR"))
    ap.add_argument("--unanswerable", nargs="*", default=[])
    ap.add_argument("--out-dir", default="dist")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--verify", default=None)
    args = ap.parse_args(argv)
    if args.verify:
        return verify(resolve(args.verify))
    if not args.pack:
        ap.error("需要 --pack 或 --verify")
    return merge(args)


if __name__ == "__main__":
    sys.exit(main())
