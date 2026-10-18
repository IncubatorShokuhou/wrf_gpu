#!/usr/bin/env bash
# CPU-only, under a normal RAM slot after the corresponding GPU lock exits.
set -u
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
case=$1
[[ $case == c10 || $case == c18 ]] || exit 9
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/final-candidate
export PYTHONPATH=$tree/src
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
timeout 240 taskset -c 8,9,12,13,24,25,28,29 nice -n 19 \
  python "$tree/scripts/wave_cubin_gate.py" "$lane/gpu-$case/cache" "$lane/gpu-$case/e41" \
  > "$lane/gpu-$case-e41.log" 2>&1
rc=$?
[[ $rc == 0 || $rc == 2 ]] || exit "$rc"
[[ -f "$lane/gpu-$case/e41/wave_gate.json" ]] || exit 9
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
case "$case" in
  c10) taskset -c 8,9,12,13,24,25,28,29 nice -n 19 rm -rf <USER_HOME>/wrf_gpu2_lanes/w41-wdm57/gpu-c10/cache;;
  c18) taskset -c 8,9,12,13,24,25,28,29 nice -n 19 rm -rf <USER_HOME>/wrf_gpu2_lanes/w41-wdm57/gpu-c18/cache;;
esac
echo "$case E41scanner=$rc complete; executable cache cleaned"
