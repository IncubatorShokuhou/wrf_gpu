#!/usr/bin/env bash
# w43-release CPU setup of the v0.4.3 bundle directory P (no GPU, no JAX) once the manager posts 'BUNDLE PIN <sha>'.
# usage: setup.sh FROZEN_TOOLS_DIR PIN ICLOUD0(0|1)
#   - FROZEN_TOOLS_DIR = an exact copy of this tools dir (the session runs from it); gets pin.env, bundle.env.expected,
#     cases_sha256.txt (+ icloud0 case dirs when ICLOUD0=1) and a final SHA256SUMS over every file
#   - P/src = clean detached checkout of PIN (git worktree of the main repo) + data/wrf_pristine link
#   - P layout, P/bundle.env, P/tools/{compile_bundle.py,bundle_consumer.sh,bundle_consumer.py} + SHA256SUMS, P/meta
# Refuses when P/jax already has content, when PIN's src tree differs from main:src, or when anything is dirty.
set -euo pipefail
umask 022
F=$1 PIN=$2 ICLOUD0=$3
REPO=<USER_HOME>/src/wrf_gpu2
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
P42=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
C=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
[[ $ICLOUD0 == 0 || $ICLOUD0 == 1 ]]
PIN=$(git -C "$REPO" rev-parse "$PIN^{commit}")
TREE=$(git -C "$REPO" rev-parse "$PIN:src")
[[ "$(git -C "$REPO" rev-parse main:src)" == "$TREE" ]] || { echo "main:src != PIN src"; exit 4; }
[ ! -e "$P/jax/aot" ] || { echo "P/jax already primed"; exit 4; }
mkdir -p "$P"
if [ ! -e "$P/src/.git" ]; then
  git -C "$REPO" worktree add --detach "$P/src" "$PIN"
fi
[[ "$(git -C "$P/src" rev-parse HEAD)" == "$PIN" && "$(git -C "$P/src" rev-parse HEAD:src)" == "$TREE" ]]
[ -e "$P/src/data/wrf_pristine" ] || ln -s <USER_HOME>/src/wrf_pristine "$P/src/data/wrf_pristine"
[[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src)" ]]
! git -C "$P/src" symbolic-ref -q HEAD >/dev/null
mkdir -p "$P/jax" "$P/autotune" "$P/triton" "$P/cuda" "$P/meta" "$P/tools"
cat > "$F/bundle.env.expected" <<EOF
# v0.4.3 sealed-bundle environment: source this file unchanged in priming AND in every consumer arm (release defaults:
# set no other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT)
export PYTHONPATH=$P/src/src
export GPUWRF_JAX_CACHE_DIR=$P/jax JAX_COMPILATION_CACHE_DIR=$P/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$P/autotune
export TRITON_CACHE_DIR=$P/triton CUDA_CACHE_PATH=$P/cuda
EOF
if [ -e "$P/bundle.env" ]; then cmp "$P/bundle.env" "$F/bundle.env.expected"; else cp "$F/bundle.env.expected" "$P/bundle.env"; fi
cp "$P/bundle.env" "$P/meta/bundle.env"
for f in compile_bundle.py bundle_consumer.sh bundle_consumer.py; do install -m 0644 "$P/src/scripts/release/$f" "$P/tools/$f"; done
chmod 0755 "$P/tools/bundle_consumer.sh"
(cd "$P/tools" && sha256sum compile_bundle.py bundle_consumer.sh bundle_consumer.py > SHA256SUMS)
cp "$P/tools/compile_bundle.py" "$P/meta/compile_bundle.py"
need() { "$@" || { echo "SETUP REFUSED: $*"; exit 5; }; }
# icloud=0 arms (A4 prime 0227 24 h, B8 0502 3 h replay) = the v0.4.2 bundle's icloud=0 arms on byte-identical inputs:
# cases/wn3_0227_icloud0 = cases/wn3_0227 model inputs + the w42-icloud0 spec namelist (reconstructed, sha 2c15e10b),
# cases/wn3_0502_icloud0 = cases/wn3_0502 with only icloud 1 -> 0. No CPU twin here (deleted after v0.4.2): icloud=0
# qualification carries forward only through C3 byte identity vs P42/receipts/A4 + B8.
if [[ $ICLOUD0 == 1 ]]; then
  need test "$(sha256sum < "$C/wn3_0227_icloud0/namelist.input" | cut -d' ' -f1)" = 2c15e10b51d73654869ca6f604545eea17e666a15e57be34c8be03e34a893f80
  for f in wrfbdy_d01 wrfinput_d01 wrfinput_d02 wrfinput_d03 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do
    need cmp -s "$C/wn3_0227_icloud0/$f" "$C/wn3_0227/$f"; need cmp -s "$C/wn3_0502_icloud0/$f" "$C/wn3_0502/$f"; done
  [ "$(diff "$C/wn3_0502/namelist.input" "$C/wn3_0502_icloud0/namelist.input" | /usr/bin/grep -c '^[<>]')" = 2 ] || { echo "icloud edit not exactly one line"; exit 5; }
fi
# C3 references: the v0.4.2 bundle receipts (whole-file wrfout SHA per arm)
for a in A1_swiss A2_prod A3_wn3_0227_72h A4_wn3_0227_icloud0 B2_wn3_0502_3h B3_wn3_0614_48h B4_wn3_0227in48h_3h B5_swiss_3h B6_prod_3h B8_wn3_0502_icloud0_3h; do
  need test -s "$P42/receipts/$a/wrfout_sha.txt"; done
# autotune seed (section 1 of RELEASE_PLAN.md): exact copy of the v0.4.2 per-fusion autotune results into P/jax BEFORE priming
SEED_SRC=$P42/jax/xla_gpu_per_fusion_autotune_cache_dir; SEED_DST=$P/jax/xla_gpu_per_fusion_autotune_cache_dir
need test -d "$SEED_SRC"
if [ ! -e "$SEED_DST" ]; then cp -a "$SEED_SRC" "$SEED_DST"; fi
(cd "$SEED_SRC" && find . -type f | sort | xargs sha256sum) > "$F/autotune_seed_v042.sha256"
(cd "$SEED_DST" && find . -type f | sort | xargs sha256sum) | cmp -s - "$F/autotune_seed_v042.sha256" || { echo "autotune seed != v0.4.2 source"; exit 5; }
echo "seeded $(wc -l < "$F/autotune_seed_v042.sha256") autotune entries from $SEED_SRC"
(cd "$C" && sha256sum */* > "$F/cases_sha256.txt")
printf 'PIN=%s\nTREE=%s\nBS_ICLOUD0=%s\n' "$PIN" "$TREE" "$ICLOUD0" > "$F/pin.env"
[[ $ICLOUD0 == 1 ]] && printf 'ICLOUD0_INPUT=%s\nICLOUD0_HOURS=24\n' "$C/wn3_0227_icloud0" >> "$F/pin.env"
cat > "$P/meta/arms.txt" <<EOF
# name|input|product CLI args after --output-dir (release defaults; no GPUWRF_* model flags) — PRIMING arms of this bundle
A1_swiss|<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input|--domains-from-namelist --hours 3
A2_prod|<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725|--domains-from-namelist --hours 3
A3_wn3_0227_72h|<guarded copy of WN3 20260227_18z_a1 inputs (72 h: wrfbdy 12, wrflowinp 13)>|--domains-from-namelist --emit-initial-history --hours 72
EOF
[[ $ICLOUD0 == 1 ]] && echo "A4_wn3_0227_icloud0|<WN3 0227 inputs + namelist icloud=0, run_hours 24, restart_interval 100000 (sha 2c15e10b)>|--domains-from-namelist --emit-initial-history --hours 24" >> "$P/meta/arms.txt"
python3 "$F/prereg.py" "$F"   # RW3T pre-registration (immutable; tool SHAs incl. pin.env)
for f in gpu_session.sh session.sh rw3t.sh arm.sh setup.sh pin.env PREREG.json; do cp "$F/$f" "$P/meta/$f"; done
(cd "$F" && rm -f SHA256SUMS && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "SETUP_OK P=$P pin=$PIN src=$TREE icloud0=$ICLOUD0 tools: $(cut -c1-12 "$P/tools/SHA256SUMS" | tr '\n' ' ') frozen=$F"
