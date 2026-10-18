#!/usr/bin/env bash
# Only unpaid gates: preserve the completed68 component checks/mutant2.
# Official RAM12G is needed: original full shared batch hit its6G hard cap.
set -euo pipefail
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/final-candidate
repo=<USER_HOME>/src/wrf_gpu2_wt/w41-wdm57
proof_commit=$1
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p "$lane/cpu-resume-work"/{tmp,jax,triton,cuda} "$lane/cpu_resume_tools"
git -C "$repo" show "$proof_commit:proofs/w41_wdm57/bounded_consumers.py" > "$lane/cpu_resume_tools/bounded_consumers.py"
chmod 444 "$lane/cpu_resume_tools/bounded_consumers.py"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=0
export TMPDIR=$lane/cpu-resume-work/tmp JAX_COMPILATION_CACHE_DIR=$lane/cpu-resume-work/jax
export TRITON_CACHE_DIR=$lane/cpu-resume-work/triton CUDA_CACHE_PATH=$lane/cpu-resume-work/cuda
export PYTHONPATH=$tree/src:$lane/cpu_resume_tools
export WDM57_TEST_EVENTS=$lane/consumers-resume1.events.jsonl
cd "$tree"
[[ $(git rev-parse HEAD) == 4474617f2585ef9d577fc360a7688e2de1efcd15 ]]
cpu_py() { if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout "$1" env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python "${@:2}"; }
echo 'STAGE identity-Swiss'
cpu_py 600 "$tree/proofs/w41_wdm57/identity.py" "$tree" "$lane/identity-swiss.json" > "$lane/identity-swiss.log" 2>&1
echo 'STAGE identity-PROD'
cpu_py 600 "$tree/proofs/w41_wdm57/identity.py" "$tree" "$lane/identity-prod.json" <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 > "$lane/identity-prod.log" 2>&1
cpu_py 30 - "$lane" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for case,sha in {'swiss':'3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57',
                 'prod':'196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e'}.items():
    d=json.loads((p/f'identity-{case}.json').read_text());assert d['arms']['release']['sha']==sha,(case,d)
print('DEFAULT IDENTITIES EXACT')
PY
echo 'STAGE shared-consumers-memory-bounded'
cpu_py 3600 -m pytest tests/test_v013_operational_smoke.py tests/test_v060_physics_dispatch.py \
  tests/test_cli.py tests/test_scheme_catalog_fail_closed.py tests/test_v034_implemented_notes.py \
  tests/test_namelist_binding.py tests/test_m7_restart_checkpoint_roundtrip.py tests/test_namelist_check.py \
  tests/test_netcdf_process_lock.py tests/test_v018_mp_family_fail_closed.py tests/contracts/ -q \
  -p bounded_consumers --basetemp "$lane/pytest" -p no:cacheprovider \
  --junitxml="$lane/consumers-resume1.xml" > "$lane/consumers-resume1.log" 2>&1
cpu_py 30 - "$lane" <<'PY'
import sys,xml.etree.ElementTree as E
from pathlib import Path
p=Path(sys.argv[1]);root=E.parse(p/'consumers-resume1.xml')
assert not root.findall('.//failure') and not root.findall('.//error')
print('CPU RESUME PASS; full shared consumers clean; both default identities EXACT')
PY
