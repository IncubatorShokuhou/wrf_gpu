#!/usr/bin/env bash
# One RAM slot, one CPU JAX process at a time; no duplicate model validation.
set -euo pipefail
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$(cd "$(dirname "$0")/../.." && pwd)
tools=$tree/proofs/w41_wdm57
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p "$lane/cpu-work"/{tmp,jax,triton,cuda}
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export TMPDIR=$lane/cpu-work/tmp JAX_COMPILATION_CACHE_DIR=$lane/cpu-work/jax
export TRITON_CACHE_DIR=$lane/cpu-work/triton CUDA_CACHE_PATH=$lane/cpu-work/cuda
export PYTHONPATH=$tree/src:$tree/proofs/w41_wdm57
cd "$tree"
cpu_py() { if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout "$1" env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python "${@:2}"; }
echo 'STAGE radii'
cpu_py 600 -m pytest tests/test_w41_wdm57_firewall.py -k cpu_outputs_bitwise_equal -q \
  --basetemp "$lane/pytest" -p no:cacheprovider --junitxml="$lane/radii.xml" > "$lane/radii.log" 2>&1
echo 'STAGE deletion-control'
mutant_rc=0
cpu_py 180 -m pytest tests/test_w41_wdm57_firewall.py -k only_nn_column_view -q \
  --basetemp "$lane/pytest" -p no:cacheprovider -p no_firewall_mutant --junitxml="$lane/mutant.xml" \
  > "$lane/mutant.log" 2>&1 || mutant_rc=$?
[[ $mutant_rc == 1 ]]
cpu_py 30 - "$lane" <<'PY'
import json,sys,xml.etree.ElementTree as E
from pathlib import Path
p=Path(sys.argv[1]);tests=E.parse(p/'mutant.xml').findall('.//testcase')
assert len(tests)==2 and not E.parse(p/'mutant.xml').findall('.//error')
assert all(t.find('failure') is not None and 'assert 0 == 1' in ''.join(t.find('failure').itertext()) for t in tests)
(p/'mutant_verified.json').write_text(json.dumps({'genuine_assertion_kills':2,'setup_errors':0,'mutant':'small_grid_firewall returns its input'})+'\n')
PY
echo 'STAGE shared-consumers'
cpu_py 900 -m pytest tests/test_v013_operational_smoke.py tests/test_v060_physics_dispatch.py \
  tests/test_cli.py tests/test_scheme_catalog_fail_closed.py tests/test_v034_implemented_notes.py \
  tests/test_namelist_binding.py tests/test_m7_restart_checkpoint_roundtrip.py tests/test_namelist_check.py \
  tests/test_netcdf_process_lock.py tests/test_v018_mp_family_fail_closed.py tests/contracts/ -q \
  --basetemp "$lane/pytest" -p no:cacheprovider --junitxml="$lane/consumers.xml" > "$lane/consumers.log" 2>&1
echo 'STAGE identity-Swiss'
GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=0 cpu_py 600 "$tools/identity.py" "$tree" "$lane/identity-swiss.json" \
  > "$lane/identity-swiss.log" 2>&1
echo 'STAGE identity-PROD'
GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=0 cpu_py 600 "$tools/identity.py" "$tree" "$lane/identity-prod.json" \
  <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 > "$lane/identity-prod.log" 2>&1
cpu_py 30 - "$lane" <<'PY'
import json,sys,xml.etree.ElementTree as E
from pathlib import Path
p=Path(sys.argv[1])
expected={'swiss':'3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57',
          'prod':'196aed9f9973499c5610fdf9965223503a61944156b8cd41357fd8c6e911911e'}
for case,sha in expected.items():
 d=json.loads((p/f'identity-{case}.json').read_text());assert d['arms']['release']['sha']==sha,(case,d)
for name in ('radii','consumers'):
 root=E.parse(p/f'{name}.xml');assert not root.findall('.//failure') and not root.findall('.//error')
assert len(E.parse(p/'radii.xml').findall('.//testcase'))==8
print('CPU FINISH PASS; default identities EXACT; fp32wet8PASS; deletion2ASSERT/0ERROR; consumersPASS')
PY
