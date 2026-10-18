#!/usr/bin/env bash
# w41-bundle041 CPU setup of the v0.4.1 bundle directory P (no GPU, no JAX): layout, bundle.env, shipped tools + SHA256SUMS,
# arms list, usage note. P/src must already be the clean detached BUNDLE PIN checkout with data/wrf_pristine.
# usage: setup.sh            (idempotent for the files it writes; never touches P/jax content)
set -euo pipefail
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=104c97de98b45b790058d3cdad6e0ff1e88a9f82; TREE=319848e91a5ecaf5516d0a783bad11447c349049
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1
[[ "$(git -C "$P/src" rev-parse HEAD)" == "$PIN" && "$(git -C "$P/src" rev-parse HEAD:src)" == "$TREE" ]]
[[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src)" ]]
! git -C "$P/src" symbolic-ref -q HEAD >/dev/null
test -e "$P/src/data/wrf_pristine"
mkdir -p "$P/jax" "$P/autotune" "$P/triton" "$P/cuda" "$P/meta" "$P/tools"
cat > "$HERE/bundle.env.expected" <<EOF
# v0.4.1 sealed-bundle environment: source this file unchanged in priming AND in every consumer arm (release defaults:
# set no other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT)
export PYTHONPATH=$P/src/src
export GPUWRF_JAX_CACHE_DIR=$P/jax JAX_COMPILATION_CACHE_DIR=$P/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$P/autotune
export TRITON_CACHE_DIR=$P/triton CUDA_CACHE_PATH=$P/cuda
EOF
if [ -e "$P/bundle.env" ]; then cmp "$P/bundle.env" "$HERE/bundle.env.expected"; else cp "$HERE/bundle.env.expected" "$P/bundle.env"; fi
cp "$P/bundle.env" "$P/meta/bundle.env"
for f in compile_bundle.py bundle_consumer.sh bundle_consumer.py; do
  install -m 0644 "$P/src/scripts/release/$f" "$P/tools/$f"
done
chmod 0755 "$P/tools/bundle_consumer.sh"
(cd "$P/tools" && sha256sum compile_bundle.py bundle_consumer.sh bundle_consumer.py > SHA256SUMS)
cp "$P/tools/compile_bundle.py" "$P/meta/compile_bundle.py"
cat > "$P/meta/arms.txt" <<'EOF'
# name|input|product CLI args after --output-dir (release defaults; no GPUWRF_* model flags) — PRIMING arms of this bundle
A1_swiss|<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input|--domains-from-namelist --hours 3
A2_prod|<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725|--domains-from-namelist --hours 3
A3_wn3_0227_72h|<guarded copy of WN3 20260227_18z_a1 inputs (72 h: wrfbdy 12, wrflowinp 13)>|--domains-from-namelist --emit-initial-history --hours 72
EOF
for f in session.sh arm.sh setup.sh; do cp "$HERE/$f" "$P/meta/$f"; done
echo "SETUP_OK P=$P pin=$PIN tools: $(cut -c1-12 "$P/tools/SHA256SUMS" | tr '\n' ' ')"
