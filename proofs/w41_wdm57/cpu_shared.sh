#!/usr/bin/env bash
# Full unchanged required selection in tests/conftest.py's CPU FAST0 recipe.
set -euo pipefail
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/final-candidate
repo=<USER_HOME>/src/wrf_gpu2_wt/w41-wdm57
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p "$lane/cpu-shared-work"/{tmp,jax,triton,cuda} "$lane/cpu_shared_tools"
git -C "$repo" show "$1:proofs/w41_wdm57/bounded_consumers.py" > "$lane/cpu_shared_tools/bounded_consumers.py"
chmod 444 "$lane/cpu_shared_tools/bounded_consumers.py"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 GPUWRF_CENSUS=0
export JAX_ENABLE_X64=1 WDM57_BASELINE_X64=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$lane/cpu-shared-work/tmp JAX_COMPILATION_CACHE_DIR=$lane/cpu-shared-work/jax
export TRITON_CACHE_DIR=$lane/cpu-shared-work/triton CUDA_CACHE_PATH=$lane/cpu-shared-work/cuda
export PYTHONPATH=$tree/src:$lane/cpu_shared_tools
export WDM57_TEST_EVENTS=$lane/consumers-cpu-default.events.jsonl
cd "$tree"
[[ $(git rev-parse HEAD) == 4474617f2585ef9d577fc360a7688e2de1efcd15 ]]
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
timeout 3600 taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python -m pytest \
  tests/test_v013_operational_smoke.py tests/test_v060_physics_dispatch.py \
  tests/test_cli.py tests/test_scheme_catalog_fail_closed.py tests/test_v034_implemented_notes.py \
  tests/test_namelist_binding.py tests/test_m7_restart_checkpoint_roundtrip.py tests/test_namelist_check.py \
  tests/test_netcdf_process_lock.py tests/test_v018_mp_family_fail_closed.py tests/contracts/ -q \
  -p bounded_consumers --basetemp "$lane/pytest" -p no:cacheprovider \
  --junitxml="$lane/consumers-cpu-default.xml" > "$lane/consumers-cpu-default.log" 2>&1
echo 'SHARED CPU DEFAULT PASS (FAST0/x64 baseline; original assertions)'
