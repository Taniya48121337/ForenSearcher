#!/usr/bin/env bash
# Step 3: datasets.
#   EVID-Bench   : Hugging Face Kirito-Lab/EVID-Bench (gated: accept the terms on the website once and log in, see README)
#   GroundLie360 : videos come from OneDrive / Baidu Netdisk -> copy them to the server (README step 3b); annotations
#                  are already in the package
#   FMNV         : videos come from Baidu Netdisk -> same; annotations already in the package
# Rerun any time: it only downloads what is missing and then re-checks all three datasets.
set -uo pipefail
source "$(dirname "$0")/env.sh"
LOG="$FS_LOG_DIR/03_download_data.log"
exec > >(tee "$LOG") 2>&1
say "Step 3: datasets  (log: $LOG)"
PY="$FS_ENV_AGENT/bin/python"
DS="$FS_ROOT/plan-harness-1/datasets"

say "--- EVID-Bench (222 videos, ~10 GB)"
"$PY" - "$DS/EVID-Bench" <<'PY'
import json, os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
have = len(list((root / "videos").glob("video_*.mp4"))) if (root / "videos").is_dir() else 0
if have >= 222:
    print(f"[ OK ] already have {have} videos")
else:
    print(f"[....] downloading (have {have}/222); token: {'HF_TOKEN set' if os.environ.get('HF_TOKEN') else 'from hf auth login'}", flush=True)
    for attempt in range(1, 4):
        try:
            snapshot_download("Kirito-Lab/EVID-Bench", repo_type="dataset", local_dir=str(root / "_hf"),
                              allow_patterns=["videos/*", "evid_bench.json", "README.md"], max_workers=8)
            break
        except GatedRepoError:
            print("[FAIL] the dataset is gated: open https://huggingface.co/datasets/Kirito-Lab/EVID-Bench while logged in, "
                  "accept the terms, then run `hf auth login` (or export HF_TOKEN=...) on the server and rerun this step.")
            sys.exit(1)
        except (HfHubHTTPError, OSError) as exc:
            print(f"[WARN] attempt {attempt}: {type(exc).__name__}: {str(exc)[:300]}", flush=True); time.sleep(15 * attempt)
    else:
        print("[FAIL] EVID-Bench download failed (network: FS_PROXY / FS_HF_ENDPOINT in server/env.sh)"); sys.exit(1)
    (root / "videos").mkdir(exist_ok=True)
    for f in (root / "_hf" / "videos").glob("*.mp4"):
        dst = root / "videos" / f.name
        if not dst.exists():
            os.replace(f, dst)
ours = {e["id"] for e in json.loads((root / "evid_bench.json").read_text(encoding="utf-8"))}
hf_manifest = root / "_hf" / "evid_bench.json"
if hf_manifest.is_file():
    theirs = {e["id"] for e in json.loads(hf_manifest.read_text(encoding="utf-8"))}
    print(("[ OK ] " if theirs == ours else "[WARN] ") + f"manifest ids: package {len(ours)}, Hugging Face {len(theirs)}, same={theirs == ours}")
missing = sorted(i for i in ours if not (root / "videos" / f"{i}.mp4").is_file())
print(("[ OK ] " if not missing else "[FAIL] ") + f"EVID videos present {len(ours) - len(missing)}/{len(ours)}" + (f"; missing {missing[:10]}" if missing else ""))
sys.exit(1 if missing else 0)
PY
evid=$?

say "--- GroundLie360 / FMNV (videos must be copied in; see README step 3b)"
"$PY" - "$FS_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
import _bench_common as B
bad = 0
for ds, need in (("g360", 2044), ("fmnv", 2393)):
    samples = B.LOADERS[ds]()
    vids = B.video_index(ds)
    ids = {s.sample_id for s in samples}
    matched = ids & set(vids)
    real = sum(1 for s in samples if s.sample_id in matched and s.label == "Real")
    fake = len(matched) - real
    folder = B.DATASETS / B.DIRS[ds]
    if not vids:
        print(f"[FAIL] {B.DIRS[ds]}: no video files under {folder}"); bad += 1; continue
    unmatched = [p.name for k, p in vids.items() if k not in ids][:5]
    print(("[ OK ] " if len(matched) >= 200 else "[WARN] ") + f"{B.DIRS[ds]}: {len(vids)} video files, {len(matched)}/{len(ids)} match an annotation "
          f"(Real {real}, Fake {fake})" + (f"; unmatched examples {unmatched}" if unmatched else ""))
    if len(matched) < 200:
        print(f"       fewer than 200 matched videos: the run samples only from these (FS_BENCH_N per dataset)")
    # decodability spot check
    import cv2
    broken = []
    for sid in sorted(matched)[:20]:
        cap = cv2.VideoCapture(str(vids[sid])); ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0; cap.release()
        if not ok: broken.append(sid)
    print(("[ OK ] " if not broken else "[WARN] ") + f"{B.DIRS[ds]}: decodable {20 - len(broken)}/20 checked" + (f" broken {broken}" if broken else ""))
sys.exit(1 if bad else 0)
PY
bench=$?
echo
[ $evid -eq 0 ] && ok "EVID-Bench ready" || fail "EVID-Bench not ready (see above)"
[ $bench -eq 0 ] && ok "GroundLie360 + FMNV videos found" || warn "GroundLie360 / FMNV videos missing: the EVID runs can start without them; copy them in (README 3b) and rerun this step"
[ $evid -eq 0 ] && ok "Step 3 done. Next: bash server/04_start_vllm.sh"
