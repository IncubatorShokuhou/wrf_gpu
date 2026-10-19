#!/usr/bin/env bash
# w42-bundle042: run stage_cpu72.py (guarded facade_stage of the 219 original CPU-WRF WN3 0227 72 h frames) at idle IO
# priority, SIGSTOP/SIGCONT of its process group across foreign QUIET windows (no IO during timed arms).
# usage: stage_run.sh CPU     (STAGE_ROOT = <DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cpu72, receipts lane/cpu72_receipt)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042; CPU=$1
[ ! -e "$S/cpu72" ] || { echo "stage root exists"; exit 2; }
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 10; done
echo "$(date -u +%FT%TZ) START stage_cpu72 (core $CPU, ionice idle)"
rm -f "$L/stage_cpu72.pid"
setsid bash -c "echo \$\$ > '$L/stage_cpu72.pid'; exec env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c '$CPU' nice -n 19 ionice -c3 python3 '$HERE/stage_cpu72.py' '$S/cpu72' '$L/cpu72_receipt'" \
  > "$L/stage_cpu72.log" 2>&1 < /dev/null &   # E131: the pid file holds the session leader = process group id
for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$L/stage_cpu72.pid" ] && break; sleep 0.5; done
pid=$(cat "$L/stage_cpu72.pid"); state=run
while kill -0 "$pid" 2>/dev/null; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then
    [ $state = run ] && { kill -STOP -- "-$pid"; state=stopped; echo "$(date -u +%FT%TZ) STOP ($(cat /tmp/wrf_gpu2_quiet 2>/dev/null))"; }
  else
    [ $state = stopped ] && { kill -CONT -- "-$pid"; state=run; echo "$(date -u +%FT%TZ) CONT"; }
  fi
  sleep 5
done
rc=$(grep -q '"sealed24_checked": 75, "sha_mismatch": \[\]' "$L/stage_cpu72.log" && echo 0 || echo 1)
echo "$(date -u +%FT%TZ) END stage_cpu72 rc=$rc $(tail -1 "$L/stage_cpu72.log" | cut -c1-300)"
