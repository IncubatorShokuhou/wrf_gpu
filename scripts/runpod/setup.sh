#!/usr/bin/env bash
# RunPod (or any multi-GPU Linux box) setup for the wrf_gpu multi-GPU tests (lane o2-multigpu).
#   bash scripts/runpod/setup.sh [WORKDIR]          (run from a checkout of the branch under test)
# Installs into a venv: JAX 0.10 CUDA 13 (same pins as the release workstation), this checkout (-e), WRF v4.7.1
# runtime tables (GPUWRF_WRF_ROOT), and records the hardware/topology facts the tests depend on.
# Writes WORKDIR/env.sh (source it before every test) and WORKDIR/facts/*.txt. No GPU work beyond a JAX probe.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=${1:-$repo/../wrfgpu_runpod}
mkdir -p "$work/facts"
cd "$work"

echo "== driver / GPUs"
nvidia-smi | tee facts/nvidia-smi.txt
nvidia-smi --query-gpu=index,uuid,name,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max,driver_version \
  --format=csv | tee facts/gpus.csv
nvidia-smi topo -m | tee facts/topo.txt || true
nvidia-smi topo -p2p r 2>/dev/null | tee facts/p2p_read.txt || true   # GeForce RTX 4090: expect NS (not supported)
nproc | tee facts/nproc.txt; free -g | tee facts/free.txt; df -h . | tee facts/df.txt
python3 --version | tee facts/python.txt

echo "== venv + JAX (CUDA 13 wheels need a driver >= 580; for an older driver use jax[cuda12])"
python3 -m venv "$work/venv"
# shellcheck disable=SC1091
source "$work/venv/bin/activate"
pip install --upgrade pip >/dev/null
if [ "${JAX_CUDA:-13}" = 12 ]; then pip install "jax[cuda12]==0.10.*"; else pip install "jax[cuda13]==0.10.*"; fi
pip install -e "$repo"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|jax-cuda|nvidia-nccl|numpy|netcdf4|h5netcdf) " | tee facts/pip.txt

echo "== WRF v4.7.1 runtime tables (Noah-MP/RRTMG tables; ~200 MB shallow clone)"
if [ ! -d "$work/WRF/run" ]; then
  git clone --depth 1 --branch v4.7.1 https://github.com/wrf-model/WRF.git "$work/WRF"
fi
bash "$repo/scripts/runpod/prepare_wrf_tables.sh" "$work/WRF" | tee facts/wrf_tables.txt

cat > "$work/env.sh" <<EOF
source "$work/venv/bin/activate"
export GPUWRF_WRF_ROOT="$work/WRF"
export GPUWRF_JAX_CACHE_DIR="$work/jax_cache"
export JAX_COMPILATION_CACHE_DIR="$work/jax_cache"
export PYTHONPATH="$repo/src\${PYTHONPATH:+:\$PYTHONPATH}"
export WRFGPU_REPO="$repo"
EOF
# shellcheck disable=SC1091
source "$work/env.sh"

echo "== JAX device probe"
python3 - <<'PY' | tee facts/jax_devices.txt
import jax, jax.numpy as jnp
ds = jax.devices()
print("jax", jax.__version__, "devices", ds, [d.device_kind for d in ds])
assert len(ds) >= 2 and all(d.platform == "gpu" for d in ds), "setup requires at least two actual GPUs"
x = jax.device_put(jnp.arange(1 << 20, dtype=jnp.float32), ds[0])
for d in ds[1:]:
    y = jax.device_put(x, d)
    assert float(y.sum()) == float(x.sum()), "cross-device copy mismatch"
print("cross-device copy OK")
PY
echo "setup done: source $work/env.sh ; then scripts/runpod/test_matrix.sh $work/results"
