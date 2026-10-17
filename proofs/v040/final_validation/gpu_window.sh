#!/usr/bin/env bash
# Authorized v-prep execution, from an immutable measured source snapshot.
set -euo pipefail
umask 077
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python "$(dirname "$0")/gpu_authorization.py" || exit 3
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if (( $# < 2 )); then echo 'usage: gpu_window.sh <absolute UTC window end> c00 [c01 ...]'; exit 2; fi
deadline=$(date -u -d "$1" +%s)
shift
now=$(date -u +%s)
cutoff=$(date -u -d "2026-10-15 18:45:00 UTC" +%s)
last_start=$(date -u -d "2026-10-15 18:15:00 UTC" +%s)
# Manager Oct15 priority queue: hard end18:45, last24h start<=18:15.
(( now <= last_start )) || exit 3
(( deadline > now && deadline - now <= 5400 )) || exit 3
(( deadline <= cutoff )) || exit 3
export FINALVAL_DEADLINE_EPOCH=$deadline
repo=$(JAX_PLATFORMS=cpu git rev-parse --show-toplevel)
if JAX_PLATFORMS=cpu git -C "$repo" symbolic-ref --quiet HEAD >/dev/null; then echo 'Detached RC required'; exit 2; fi
test -z "$(JAX_PLATFORMS=cpu git -C "$repo" status --porcelain)"
source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
export PYTHONPATH="$repo/src"
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 0 nice -n 19 python - "$@" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads((Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/name/'inputs.json').read_text()) for name in sys.argv[1:]]
assert all(r.get('gpu_eligible',True) and not r['admission_errors'] for r in rows)
assert len(rows)==1 or not any(len(r['domains'])>1 for r in rows),'Nested arm must run alone'
assert len(rows)==1 or not any(r['options']['ra_lw_physics']==3 for r in rows),'CAM larger-VRAM arm runs alone initially'
assert len(rows)==1 or not any(r['options']['mp_physics']==38 or r['geography'].startswith('authentic-WN3') for r in rows),'mp38/Atlantic larger-VRAM arms run alone initially'
PY
# The enclosing shell has no CUDA context. Both children inherit the official
# FD9 lease/lock; no Python parent holds a pool when starting a GPU child.
# Manager Oct15 core fix also applies to an older, still-pinned wrapper.
export GPUWRF_GPU_ARM_CPUS=3,19,13,29
lock_wrapper=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
expected=$(awk '{print $1}' <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sha256)
if [[ $(sha256sum "$lock_wrapper" | awk '{print $1}') != "$expected" ]]; then
  # Explicitly authorized fallback: the official snapshot wrapper must match
  # MAIN's live pin exactly. Its own admission/lease/digest checks still run.
  lock_wrapper="${GPUWRF_PINNED_LOCK_WRAPPER:-$repo/scripts/with_gpu_lock.sh}"
  [[ $(sha256sum "$lock_wrapper" | awk '{print $1}') = "$expected" ]] || exit 126
fi
echo "FINALVAL official wrapper=$lock_wrapper pin=$expected GPU_ARM_CPUS=$GPUWRF_GPU_ARM_CPUS"
"$lock_wrapper" --timeout "$((deadline - now))" --label v-prep -- \
  timeout --signal=TERM --kill-after=15 "$((deadline - now))" \
  bash "$(dirname "$0")/gpu_inside.sh" "$@"
