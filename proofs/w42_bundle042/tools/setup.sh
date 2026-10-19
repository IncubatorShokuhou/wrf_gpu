#!/usr/bin/env bash
# w42-bundle042 CPU setup of the v0.4.2 bundle directory P (no GPU, no JAX) once the manager posts 'BUNDLE PIN <sha>'.
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
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
C=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cases
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
# v0.4.2 sealed-bundle environment: source this file unchanged in priming AND in every consumer arm (release defaults:
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
IC_CASE=${IC_CASE:-<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/wn3_icloud0}; IC_COMMIT=${IC_COMMIT:-e0b40d3732}
need() { "$@" || { echo "SETUP REFUSED: $*"; exit 5; }; }
if [[ $ICLOUD0 == 1 ]]; then
  # A4 = w42-icloud0 BUNDLE_ARM_SPEC (lane/w42-icloud0 e0b40d3732) exactly: input = its CASE/input (7 model inputs byte-identical
  # to cases/wn3_0227; namelist differs only in run_hours 24, restart_interval, icloud 0), CPU-WRF icloud=0 twin = CASE/cpu
  need test "$(sha256sum < "$IC_CASE/input/namelist.input" | cut -d' ' -f1)" = 2c15e10b51d73654869ca6f604545eea17e666a15e57be34c8be03e34a893f80
  need test "$(sha256sum < "$IC_CASE/CPU_OUTPUTS.json" | cut -d' ' -f1)" = 41f257b1d882de69bcba75ba66ceb9c1ff86108c2985c0cd1bfbffc3319de05b
  need test "$(sha256sum < "$IC_CASE/cpu_run.json" | cut -d' ' -f1)" = 31534bec2c79286e146c5cb8b9aa8b2ec7c5410447697b7281b5b7ee0e485cb7
  need python3 -c "import json,sys; assert json.load(open(sys.argv[1]))['verdict'] == 'OK'" "$IC_CASE/STAGE.json"
  for f in wrfbdy_d01 wrfinput_d01 wrfinput_d02 wrfinput_d03 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do
    need cmp -s "$IC_CASE/input/$f" "$C/wn3_0227/$f"; done
  sha256sum "$IC_CASE"/input/* "$IC_CASE/CPU_OUTPUTS.json" "$IC_CASE/cpu_run.json" > "$F/ic_sha256.txt"
  # D6 tools of the spec (icloud0_report.py, d6_summary.py, fixtures): from the pin when merged, else a frozen copy
  if git -C "$REPO" cat-file -e "$PIN:proofs/w42_icloud0/icloud0_report.py" 2>/dev/null; then IC_TOOLS=$P/src/proofs/w42_icloud0
  else git -C "$REPO" archive "$IC_COMMIT" proofs/w42_icloud0 | tar -x -C "$F"; IC_TOOLS=$F/proofs/w42_icloud0; fi
  need test -s "$IC_TOOLS/icloud0_report.py" -a -s "$IC_TOOLS/d6_summary.py" -a -s "$IC_TOOLS/fixtures/d6_summary_morning.json"
  # B8 cross-date replay case: WN3 0502 with ONLY icloud = 1 -> 0
  for d in 0502; do
    src=$C/wn3_$d; dst=$C/wn3_${d}_icloud0
    if [ ! -e "$dst" ]; then
      mkdir "$dst"; cp -a "$src"/wrf* "$dst"/
      sed 's/^ icloud                              = 1,$/ icloud                              = 0,/' "$src/namelist.input" > "$dst/namelist.input"
      [ "$(diff "$src/namelist.input" "$dst/namelist.input" | /usr/bin/grep -c '^[<>]')" = 2 ] || { echo "icloud edit not exactly one line"; exit 5; }
      chmod -R a-w "$dst"/*
    fi
  done
fi
(cd "$C" && sha256sum */* > "$F/cases_sha256.txt")
printf 'PIN=%s\nTREE=%s\nBS_ICLOUD0=%s\n' "$PIN" "$TREE" "$ICLOUD0" > "$F/pin.env"
[[ $ICLOUD0 == 1 ]] && printf 'ICLOUD0_CASE=%s\nICLOUD0_INPUT=%s\nICLOUD0_HOURS=24\nICLOUD0_TOOLS=%s\nICLOUD0_SPEC=%s\n' \
  "$IC_CASE" "$IC_CASE/input" "$IC_TOOLS" "$IC_COMMIT:proofs/w42_icloud0/BUNDLE_ARM_SPEC.md" >> "$F/pin.env"
cat > "$P/meta/arms.txt" <<EOF
# name|input|product CLI args after --output-dir (release defaults; no GPUWRF_* model flags) — PRIMING arms of this bundle
A1_swiss|<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input|--domains-from-namelist --hours 3
A2_prod|<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725|--domains-from-namelist --hours 3
A3_wn3_0227_72h|<guarded copy of WN3 20260227_18z_a1 inputs (72 h: wrfbdy 12, wrflowinp 13)>|--domains-from-namelist --emit-initial-history --hours 72
EOF
[[ $ICLOUD0 == 1 ]] && echo "A4_wn3_0227_icloud0|$IC_CASE/input (w42-icloud0 BUNDLE_ARM_SPEC; WN3 0227 inputs, namelist icloud=0)|--domains-from-namelist --emit-initial-history --hours 24" >> "$P/meta/arms.txt"
python3 "$F/prereg.py" "$F"   # RW3T pre-registration (immutable; tool SHAs incl. pin.env)
for f in gpu_session.sh session.sh rw3t.sh arm.sh setup.sh pin.env PREREG.json; do cp "$F/$f" "$P/meta/$f"; done
(cd "$F" && rm -f SHA256SUMS && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
echo "SETUP_OK P=$P pin=$PIN src=$TREE icloud0=$ICLOUD0 tools: $(cut -c1-12 "$P/tools/SHA256SUMS" | tr '\n' ' ') frozen=$F"
