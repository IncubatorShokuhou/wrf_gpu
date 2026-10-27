#!/usr/bin/env bash
# w44-rc v0.4.4: CPU-only PREPARE step (no GPU, no JAX/gpuwrf import). Run ONCE when the manager pins the release
# candidate; writes the frozen manifest every GPU entry reads.
#
# usage: prepare.sh PIN [--allow-src-diverge]                      (real release prepare)
#        prepare.sh PIN --standin --source-tree DIR                 (CPU launch smoke only)
#   real: PIN must carry __version__ = "0.4.4" AND pyproject version 0.4.4; refuses an existing manifest, an existing
#     primed/sealed bundle dir, a dirty tree and <DATA_ROOT> free - 3 GiB < 365 GiB. Writes (disk forecast ~3 GiB):
#       bundle P   <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.4   P/src = THE ONE detached worktree of PIN (+ data/wrf_pristine
#                  link; 2.4 GB) = also the manifest "snapshot" for the CPU tools (E41 PYTHONPATH, scorer); jax/ seeded with
#                  the v0.4.3 per-fusion autotune results (1227 files, 5 MB); tools/ + bundle.env
#       mini       <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/cuda_mini_<pin12>  CUDA-mini bundle (same layout; src = P/src)
#       cases      <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/cases  own SHA-verified copy of the w43 staged WN3 cases (0.5 GB)
#       manifest   <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/PIN_MANIFEST.json (+ PREREG.json, autotune seed list)
#     (DISK-STOP 2026-10-24: one worktree instead of snapshot + bundle source; the snapshot path IS P/src.)
#   --standin: CPU launch smoke only. PIN = HEAD of an EXISTING clean-src tree given by --source-tree (no worktree is
#     created); cases are read in place from the w43 staged copy; the two bundles go below
#     <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/standin_<pin12>/ and the manifest to
#     <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/launch_smoke/standin_<pin12>/PIN_MANIFEST.json with "standin": true.
#     Every GPU entry REFUSES a standin manifest.
#   --allow-src-diverge: record that PIN:src may differ from main:src at GPU time (default: the entries refuse when
#     main:src moved away from the pinned src tree = source freeze broken, as in v0.4.3).
set -euo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/lib.sh
umask 022
REPO=<USER_HOME>/src/wrf_gpu2
P43=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
P43_BUNDLE_SHA=d2f4f197b8affdecc0c297faae3593eb9d4df658cc3a58fc24f6523518aeee70
W43_CASES=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
W43_CASES_SUMS=<USER_HOME>/wrf_gpu2_lanes/w43-release/cases_sha256.txt
SWISS=<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
PRISTINE=<USER_HOME>/src/wrf_pristine
(( $# >= 1 )) || { sed -n 2,25p "$0"; exit 2; }
PIN_IN=$1; shift
STANDIN=false ALLOW=false SRCTREE=
while (( $# )); do case $1 in --standin) STANDIN=true;; --allow-src-diverge) ALLOW=true;;
  --source-tree) SRCTREE=$2; shift;; *) echo "unknown $1"; exit 2;; esac; shift; done
if w44_quiet; then echo QUIET; exit 3; fi
PIN=$(git -C "$REPO" rev-parse "$PIN_IN^{commit}")
P12=${PIN:0:12}
TREE=$(git -C "$REPO" rev-parse "$PIN:src")
GTREE=$(git -C "$REPO" rev-parse "$PIN:src/gpuwrf")
VER=$(git -C "$REPO" show "$PIN:src/gpuwrf/__init__.py" | sed -n 's/^__version__ = "\(.*\)"$/\1/p')
PVER=$(git -C "$REPO" show "$PIN:pyproject.toml" | sed -n 's/^version = "\(.*\)"$/\1/p')
MAIN_TREE=$(git -C "$REPO" rev-parse main:src)
if [[ $STANDIN == true ]]; then
  [[ -n $SRCTREE && -d $SRCTREE ]] || { echo "REFUSED: --standin needs --source-tree DIR"; exit 2; }
  ROOT=$W44R_SCRATCH/standin_${P12}_v5
  SRC=$(cd "$SRCTREE" && pwd); P=$ROOT/bundle; MINI=$ROOT/cuda_mini; C=$W43_CASES
  MDIR=$W44R_SCRATCH/smoke_v5; FORECAST=1
else
  [[ -z $SRCTREE ]] || { echo "REFUSED: --source-tree is for --standin only"; exit 2; }
  [[ $VER == 0.4.4 && $PVER == 0.4.4 ]] || { echo "REFUSED: PIN version __init__=$VER pyproject=$PVER (need 0.4.4: bump BEFORE the pin)"; exit 4; }
  [[ $ALLOW == true || $MAIN_TREE == "$TREE" ]] || { echo "REFUSED: main:src $MAIN_TREE != PIN:src $TREE (use --allow-src-diverge only on a manager decision)"; exit 4; }
  P=$W44R_SCRATCH/rc_bundle_$P12; SRC=$P/src; MINI=$W44R_SCRATCH/cuda_mini_$P12; C=$W44R_SCRATCH/cases
  MDIR=$W44R_TOOLS; FORECAST=3
fi
MAN=$MDIR/PIN_MANIFEST.json
[ ! -e "$MAN" ] || { echo "REFUSED: $MAN exists (the manifest is write-once)"; exit 4; }
for b in "$P" "$MINI"; do
  [ ! -e "$b/jax/aot" ] && [ ! -e "$b/sealed" ] || { echo "REFUSED: $b already primed/sealed"; exit 4; }
done
w44_disk_ok "$FORECAST" || { echo "REFUSED: <DATA_ROOT> free $(w44_free_gib <DATA_ROOT>) GiB - forecast $FORECAST GiB < $W44R_DATA_FLOOR_GIB GiB"; exit 95; }
mkdir -p "$MDIR" "$W44R_SCRATCH"
LOG=$MDIR/prepare.log; W44R_LOG=$LOG
w44_log "PREPARE pin $PIN src $TREE gpuwrf $GTREE version $VER/$PVER main:src $MAIN_TREE standin=$STANDIN source $SRC"

# 1. cases: real = own copy verified against the w43 staged list (48 files, 6 cases), read-only; standin = w43 copy in place
if [[ $STANDIN == false && ! -e $C ]]; then
  (cd "$W43_CASES" && sha256sum -c --quiet "$W43_CASES_SUMS") || { w44_log "REFUSED: w43 staged cases differ from $W43_CASES_SUMS"; exit 5; }
  mkdir "$C"
  for d in wn3_0227 wn3_0227_48h wn3_0227_icloud0 wn3_0502 wn3_0502_icloud0 wn3_0614; do
    mkdir "$C/$d"; cp -p "$W43_CASES/$d"/* "$C/$d/"; chmod a-w "$C/$d"/*
  done
fi
(cd "$C" && sha256sum -c --quiet "$W43_CASES_SUMS") || { w44_log "REFUSED: case copy $C differs from the w43 list"; exit 5; }
[ "$(cd "$C" && find . -type f | wc -l)" = 48 ] || { w44_log "REFUSED: case copy $C has extra/missing files"; exit 5; }
w44_log "cases OK: $C (48 files == $W43_CASES_SUMS)"

# 2. source tree: real = ONE detached worktree P/src of PIN (E44: never checkout in the main checkout); standin = existing tree
if [[ $STANDIN == false ]]; then
  mkdir -p "$P"
  [ -e "$SRC/.git" ] || git -C "$REPO" worktree add -q --detach "$SRC" "$PIN"
  ! git -C "$SRC" symbolic-ref -q HEAD >/dev/null || { w44_log "REFUSED: $SRC not detached"; exit 4; }
  [ -e "$SRC/data/wrf_pristine" ] || ln -s "$PRISTINE" "$SRC/data/wrf_pristine"
  [ -z "$(git -C "$SRC" status --porcelain --untracked-files=all)" ] || { w44_log "REFUSED: $SRC dirty"; exit 4; }
fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$PIN" && "$(git -C "$SRC" rev-parse HEAD:src)" == "$TREE" ]] || { w44_log "REFUSED: $SRC not at PIN"; exit 4; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=all -- src)" ] || { w44_log "REFUSED: $SRC/src dirty"; exit 4; }
[ -f "$SRC/data/wrf_pristine/WRF/run/MPTABLE.TBL" ] || { w44_log "REFUSED: $SRC/data/wrf_pristine incomplete"; exit 4; }
w44_log "source $SRC at $PIN, src clean"

# 3. bundle layouts (both bundles: src = $SRC) + autotune seed from the v0.4.3 bundle (content-hash list, no inode use)
SEED_SRC=$P43/jax/xla_gpu_per_fusion_autotune_cache_dir
(cd "$SEED_SRC" && find . -type f | sort | xargs sha256sum) > "$MDIR/autotune_seed_v043.sha256"
[ "$(grep -c . "$MDIR/autotune_seed_v043.sha256")" -ge 1000 ] || { w44_log "REFUSED: v0.4.3 autotune seed too small"; exit 5; }
mklayout() {  # mklayout BUNDLE
  local b=$1
  mkdir -p "$b/jax" "$b/autotune" "$b/triton" "$b/cuda" "$b/meta" "$b/tools" "$b/receipts"
  cat > "$b/bundle.env" <<EOF
# v0.4.4 sealed-bundle environment (w44-rc prepare, pin $PIN): source this file unchanged in priming AND in every
# consumer arm (release defaults: set no other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT)
export PYTHONPATH=$SRC/src
export GPUWRF_JAX_CACHE_DIR=$b/jax JAX_COMPILATION_CACHE_DIR=$b/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$b/autotune
export TRITON_CACHE_DIR=$b/triton CUDA_CACHE_PATH=$b/cuda
EOF
  cp "$b/bundle.env" "$b/meta/bundle.env"
  local f; for f in compile_bundle.py bundle_consumer.sh bundle_consumer.py; do install -m 0644 "$SRC/scripts/release/$f" "$b/tools/$f"; done
  chmod 0755 "$b/tools/bundle_consumer.sh"
  (cd "$b/tools" && sha256sum compile_bundle.py bundle_consumer.sh bundle_consumer.py > SHA256SUMS)
  if [ ! -e "$b/jax/xla_gpu_per_fusion_autotune_cache_dir" ]; then cp -a "$SEED_SRC" "$b/jax/xla_gpu_per_fusion_autotune_cache_dir"; fi
  (cd "$b/jax/xla_gpu_per_fusion_autotune_cache_dir" && find . -type f | sort | xargs sha256sum) | cmp -s - "$MDIR/autotune_seed_v043.sha256" \
    || { w44_log "REFUSED: $b autotune seed != v0.4.3 list"; exit 5; }
}
mklayout "$P"
mklayout "$MINI"
w44_log "bundle layouts OK: $P, $MINI (src $SRC); autotune seed $(grep -c . "$MDIR/autotune_seed_v043.sha256") v0.4.3 files"

# 4. manifest (write-once) + real-world-gate/RW3T pre-registration
SUMS_SHA=$(w44_sha "$W44R_TOOLS/SHA256SUMS")
"$W44R_PY" -I - "$MAN" <<PY
import json, sys, datetime, hashlib
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
man = dict(schema="w44-rc.pin-manifest.v1", created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  standin=("$STANDIN" == "true"), pin="$PIN", src_tree="$TREE", gpuwrf_tree="$GTREE", version="$VER", pyproject_version="$PVER",
  main_src_tree_at_prepare="$MAIN_TREE", allow_src_diverge=("$ALLOW" == "true"), repo="$REPO", snapshot="$SRC",
  bundles=dict(prod=dict(dir="$P", src="$SRC"), mini=dict(dir="$MINI", src="$SRC")),
  old_bundle=dict(dir="$P43", bundle_sha256="$P43_BUNDLE_SHA", manifest_sha256=sha("$P43/sealed/BUNDLE_MANIFEST.json")),
  cases=dict(dir="$C", sums_file="$W43_CASES_SUMS", sums_sha256=sha("$W43_CASES_SUMS")),
  inputs=dict(SWISS="$SWISS", PROD="$PROD", WN3_0227="$C/wn3_0227", WN3_0227_48H="$C/wn3_0227_48h",
              WN3_0227_ICLOUD0="$C/wn3_0227_icloud0", WN3_0502="$C/wn3_0502", WN3_0502_ICLOUD0="$C/wn3_0502_icloud0",
              WN3_0614="$C/wn3_0614"),
  input_sha256={k: {f: sha(f"{d}/{f}") for f in ("namelist.input", "wrfinput_d01")} for k, d in (("SWISS", "$SWISS"), ("PROD", "$PROD"))},
  autotune_seed=dict(source="$SEED_SRC", list="$MDIR/autotune_seed_v043.sha256", list_sha256=sha("$MDIR/autotune_seed_v043.sha256")),
  tools_dir="$W44R_TOOLS", tools_sha256sums_sha256="$SUMS_SHA")
open(sys.argv[1], "x").write(json.dumps(man, indent=1) + "\n")
print("MANIFEST", sys.argv[1], sha(sys.argv[1]))
PY
"$W44R_PY" -I "$W44R_TOOLS/prereg_v044.py" "$MAN" "$MDIR/PREREG.json"
for b in "$P" "$MINI"; do cp "$MAN" "$MDIR/PREREG.json" "$b/meta/"; done
w44_log "PREPARE_OK manifest $MAN sha $(w44_sha "$MAN") prereg $(w44_sha "$MDIR/PREREG.json") <DATA_ROOT> free $(w44_free_gib <DATA_ROOT>) GiB"
