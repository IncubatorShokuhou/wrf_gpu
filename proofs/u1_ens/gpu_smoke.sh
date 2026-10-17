#!/usr/bin/env bash
# PREPARED ONLY. Manager-approved daytime GPU window required; <=30min total.
# Two same-physics release replicas exercise workflow/isolation, not calibration.
# CASE must be a small prepared real case (e.g. Swiss42x42x44 RD11); originals read-only.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case_dir=${1:?usage: gpu_smoke.sh PREPARED_CASE NEW_LANE_ARTIFACT_DIR}
out=${2:?usage: gpu_smoke.sh PREPARED_CASE NEW_LANE_ARTIFACT_DIR}
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
for tool in cubin_scan.py hlo_scan.py; do
  if [ ! -f "<USER_HOME>/wrf_gpu2_lanes/opus-a19/tools/$tool" ]; then
    echo "NEED-GPU prerequisite: manager must restore E41 $tool before this cold arm" >&2
    exit 2
  fi
done
# Preparation is CPU-only, contains no JAX backend initialization.
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 9 nice -n 19 timeout 120 python - "$case_dir" "$out" <<'PY'
from pathlib import Path
import sys
from gpuwrf.ens.workflows import prepare
case, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
namelists = [out/'replica-0.input', out/'replica-1.input']
for path in namelists:
    path.write_bytes((case/'namelist.input').read_bytes())
prepare('physics', out/'ensemble', case=case, namelists=namelists, hours=3,
        max_parallel=2, max_per_gpu=2, pool_gib=4)
PY
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
export GPUWRF_JAX_CACHE_DIR="$out/cache"
export GPUWRF_JAX_CACHE=1
export PYTHONPATH="$repo/proofs/u1_ens/smoke_backend:$PYTHONPATH"
# with_gpu_lock sets exclusive CPU affinity. Native CLI preflight is authoritative.
timeout 1800 "$repo/scripts/with_gpu_lock.sh" --label u1-ens -- \
  env JAX_PLATFORMS=cuda python -m gpuwrf.cli ens launch --plan "$out/ensemble/plan.json"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 9 nice -n 19 timeout 300 \
  python "$repo/scripts/wave_cubin_gate.py" "$out/cache" "$out/cubin_gate"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 9 nice -n 19 timeout 120 python - "$out" <<'PY'
import json
from pathlib import Path
import sys
from gpuwrf.ens.wrfout import write_products
out = Path(sys.argv[1])
members = []
for member in sorted((out/'ensemble/runs').glob('member-*')):
    assert json.loads((member/'receipt.json').read_text())['rc'] == 0
    assert json.loads((member/'wrfout/ensemble_smoke_backend.json').read_text())['asserted_gpu'] is True
    files = sorted(member.glob('wrfout/wrfout_d01_*'))
    assert files
    members.append(dict(id=member.name, files=list(map(str, files))))
assert len(members) == 2
inventory = out/'members.json'
inventory.write_text(json.dumps(dict(members=members), indent=2))
write_products(inventory, 'T2', out/'ensemble_T2.nc', thresholds=[273.15,303.15])
PY
# Receipts are hardware-bound smoke evidence only. Compare replicas as
# regression/determinism, never WRF fidelity. Manager removes this exact arm's
# cache after the native gate and every dependent evidence consumer completes.
