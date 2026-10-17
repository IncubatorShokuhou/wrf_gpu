#!/usr/bin/env bash
# wz1080_setup.sh — REMOTE-side, idempotent setup of the wz1080 test box (user space, no sudo needed).
# Runs from ~/src/wrf_gpu2_env/tools/ (pushed by wz1080_bootstrap.sh / wz1080_sync.sh).
#   inventory   OS, CPU flags, RAM, disks, NVIDIA driver/GPU, toolchains, suspend policy -> env/inventory.txt
#   python      uv-managed CPython 3.13 + venv at ~/src/wrf_gpu2_env/venv
#   jax-cpu     CPU layer (requirements-wz1080-cpu.txt) + CPU import/jit check
#   jax-cuda    CUDA-12 JAX plugin layer (requirements-wz1080-cuda.txt) + freeze.txt
#   activate    writes ~/src/wrf_gpu2_env/activate.sh, installs ~/.local/bin/wz-gpu-lock
#   cpuwrf-env  micromamba env 'wrf-build' from wrf-build.explicit.txt = the Swiss reference's CPU-WRF runtime (OpenMPI, netCDF, gfortran 14.3)
set -euo pipefail
export LC_ALL=C
ENV="$HOME/src/wrf_gpu2_env"; TOOLS="$ENV/tools"; VENV="$ENV/venv"
UV="$HOME/.local/bin/uv"
export UV_CACHE_DIR="$HOME/.cache/uv" UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
mkdir -p "$ENV" "$HOME/.local/bin" "$HOME/wrf_gpu2_runs" "$HOME/wrf_gpu2_cache/jax" "$HOME/wrf_gpu2_cache/autotune" \
         "$HOME/src/wrf_gpu2_data/overlay" "$HOME/src/wrf_gpu2_data/refs"

phase_inventory() {
  {
    set +e   # inventory is best-effort: a missing tool or a non-mount path must not abort it
    echo "== date"; date -u
    echo "== os"; cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION_ID)='; uname -r
    echo "== shell"; echo "$SHELL"; getent passwd "$USER" | cut -d: -f7
    echo "== cpu"; lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\)|CPU max MHz)'
    echo "cpu_flags: $(grep -m1 -o -w -E 'avx|avx2|fma|avx512f|f16c' /proc/cpuinfo | sort -u | tr '\n' ' ')"
    echo "== mem"; free -g
    echo "== disk"; df -hT "$HOME" /tmp / 2>/dev/null; findmnt -no FSTYPE "$HOME" 2>/dev/null
    echo "== gpu"; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.used,pcie.link.gen.max,pcie.link.width.current,power.limit --format=csv 2>&1 | head -5
    nvidia-smi 2>&1 | sed -n '1,4p'
    echo "gpu apps: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
    echo "== nvidia kmod"; modinfo -F version nvidia 2>/dev/null; rpm -qa 2>/dev/null | grep -i -E '^(akmod-nvidia|xorg-x11-drv-nvidia|nvidia-driver|kmod-nvidia)' | sort | head
    echo "== cuda toolkits"; ls -d /usr/local/cuda* 2>/dev/null; command -v nvcc ptxas 2>/dev/null
    echo "== python"; for p in python3 python3.11 python3.12 python3.13 python3.14; do command -v $p >/dev/null && echo "$p $($p -V 2>&1)"; done
    echo "== tools"; for t in git rsync gcc gfortran make cmake flock taskset systemd-inhibit; do printf '%s: %s\n' "$t" "$(command -v $t || echo MISSING)"; done
    git --version 2>/dev/null; gcc --version 2>/dev/null | head -1; gfortran --version 2>/dev/null | head -1
    echo "== sudo"; if sudo -n true 2>/dev/null; then echo "passwordless sudo: yes"; else echo "passwordless sudo: no"; fi
    echo "== suspend policy"; systemctl is-enabled sleep.target suspend.target 2>/dev/null | tr '\n' ' '; echo
    echo "dconf sleep-inactive-ac-type=$(dconf read /org/gnome/settings-daemon/plugins/power/sleep-inactive-ac-type 2>/dev/null) timeout=$(dconf read /org/gnome/settings-daemon/plugins/power/sleep-inactive-ac-timeout 2>/dev/null) (empty = GNOME default)"
    loginctl list-sessions --no-legend 2>/dev/null
    echo "== network"; (timeout 10 curl -sS -o /dev/null -w 'pypi http %{http_code} %{time_total}s\n' https://pypi.org/simple/jax/ 2>&1 || echo "pypi: unreachable")
  } 2>&1 | tee "$ENV/inventory.txt"
}

phase_python() {
  [[ -x "$UV" ]] || { echo "uv missing at $UV (bootstrap copies it)" >&2; exit 3; }
  "$UV" python install 3.13
  [[ -x "$VENV/bin/python" ]] || "$UV" venv --python 3.13 "$VENV"
  "$VENV/bin/python" -V
}

phase_jax_cpu() {
  VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" -r "$TOOLS/requirements-wz1080-cpu.txt"
  JAX_PLATFORMS=cpu "$VENV/bin/python" -c "import jax, jax.numpy as jnp; print(jax.__version__, jax.devices()); \
print('jit ok', float(jax.jit(lambda x: (x*x).sum())(jnp.arange(10.))))"
}

phase_jax_cuda() {
  VIRTUAL_ENV="$VENV" "$UV" pip install --python "$VENV/bin/python" -r "$TOOLS/requirements-wz1080-cuda.txt"
  VIRTUAL_ENV="$VENV" "$UV" pip freeze --python "$VENV/bin/python" > "$ENV/freeze.txt"
  echo "frozen $(wc -l < "$ENV/freeze.txt") packages -> $ENV/freeze.txt"
}

phase_activate() {
  install -m 755 "$TOOLS/wz1080_gpu_lock.sh" "$HOME/.local/bin/wz-gpu-lock"
  cat > "$ENV/activate.sh" <<'EOF'
# source ~/src/wrf_gpu2_env/activate.sh [TREE]   (wz1080 test box; written by wz1080_setup.sh)
_wz_tree="${1:-$PWD}"
source "$HOME/src/wrf_gpu2_env/venv/bin/activate"
# tools/shim/sitecustomize.py is inert unless WZ_PALLAS_PORTABLE=1 (box-only Pallas interpret experiment).
export PYTHONPATH="$_wz_tree/src:$HOME/src/wrf_gpu2_env/tools/shim${PYTHONPATH:+:$PYTHONPATH}"
export GPUWRF_WRF_ROOT="${GPUWRF_WRF_ROOT:-$HOME/src/wrf_pristine/WRF}"
# E23/E53/E68: only these two select the JAX cache; E19/E32: autotune dir is separate and part of the cheap key.
export GPUWRF_JAX_CACHE_DIR="${GPUWRF_JAX_CACHE_DIR:-$HOME/wrf_gpu2_cache/jax}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$GPUWRF_JAX_CACHE_DIR}"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="${GPUWRF_XLA_AUTOTUNE_CACHE_DIR:-$HOME/wrf_gpu2_cache/autotune}"
# 11 GB card: the product's conservative fallback (max(24 GiB, 0.5 x VRAM)) can never pass here.
export GPUWRF_MIN_FREE_VRAM_GIB="${GPUWRF_MIN_FREE_VRAM_GIB:-4}"
# Pristine-WRF oracle drivers (E129) when the optional toolchain exists.
if [ -x "$HOME/micromamba/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran" ]; then
  export FC="${FC:-$HOME/micromamba/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran}"
fi
# Core plan (inventory 2026-10-15: Ryzen 5 5500, 6C/12T, SMT siblings N and N+6): the GPU-lock holder gets 3 cores
# (0-2 + 6-8: host dispatch + XLA compile threads), lane CPU jobs of wz1080_run.sh the other 3 (3-5 + 9-11).
# Timed whole-box baselines override with --env WZ_CPU_CPUS= / WZ_GPU_CPUS= (empty = unpinned).
export WZ_GPU_CPUS="${WZ_GPU_CPUS-0-2,6-8}" WZ_CPU_CPUS="${WZ_CPU_CPUS-3-5,9-11}"
# This box IS pre-Ampere: admit it past the product's sm_80 gate (runtime/gpu_preflight.py) with its "unsupported,
# slow" warning. GPU runs still need the box-only shim for Pallas kernels: --env WZ_PALLAS_PORTABLE=1 (+ legacy path).
export GPUWRF_ALLOW_UNSUPPORTED_GPU="${GPUWRF_ALLOW_UNSUPPORTED_GPU-1}"
export WZ1080=1
unset _wz_tree
EOF
  echo "wrote $ENV/activate.sh; installed ~/.local/bin/wz-gpu-lock"
}

phase_cpuwrf_env() {  # exact runtime of the CPU-WRF that produced the Swiss reference (also gfortran 14.3 for oracles)
  local mm="$HOME/.local/bin/micromamba"
  if [[ ! -x "$mm" ]]; then
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
  fi
  export MAMBA_ROOT_PREFIX="$HOME/micromamba"
  [[ -d "$MAMBA_ROOT_PREFIX/envs/wrf-build" ]] || "$mm" create -y -p "$MAMBA_ROOT_PREFIX/envs/wrf-build" --file "$TOOLS/wrf-build.explicit.txt"
  "$MAMBA_ROOT_PREFIX/envs/wrf-build/bin/mpirun" --version | head -1
  "$MAMBA_ROOT_PREFIX/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran" --version | head -1
  local exe="$HOME/src/wrf_gpu2_data/cpuwrf/wrf.exe"
  if [[ -x "$exe" ]]; then echo "wrf.exe unresolved libs: $(LD_LIBRARY_PATH=$MAMBA_ROOT_PREFIX/envs/wrf-build/lib ldd "$exe" | grep -c 'not found')"; fi
}

[[ $# -gt 0 ]] || { sed -n '2,10p' "$0"; exit 2; }
for ph in "$@"; do
  echo "######## phase $ph ($(date -u +%H:%M:%SZ))"
  case "$ph" in
    inventory) phase_inventory;; python) phase_python;; jax-cpu) phase_jax_cpu;; jax-cuda) phase_jax_cuda;;
    activate) phase_activate;; cpuwrf-env) phase_cpuwrf_env;;
    *) echo "unknown phase $ph" >&2; exit 2;;
  esac
done
