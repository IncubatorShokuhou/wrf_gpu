#!/usr/bin/env bash
# w44-rc v0.4.4: stage the ORIGINAL CPU-WRF WN3 0227 72 h frames (219 files, ~22 GB) read-only through the ALISIOS guard
# (v0.4.2 stage_run.sh recipe: idle IO priority, SIGSTOP/SIGCONT across foreign QUIET windows, E131 pid file).
# PRECONDITIONS (manager): ALISIOS admission for the 20260227 reads + DISK-PLAN w44-rc ~22 GB
#   <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/cpu72_<pin12> until followup_rw3.sh has scored N72 (it deletes the stage).
# usage: stage_cpu72.sh MANIFEST
set -u
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=${1:?manifest}
eval "$("$W44R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"])' "$MAN")"
S=$W44R_SCRATCH/cpu72_${PIN:0:12}; RCPT=$W44R_LANE/receipts/cpu72_${PIN:0:12}
[ ! -e "$S" ] || { echo "stage root $S exists"; exit 2; }
w44_disk_ok 23 || { echo "REFUSED: <DATA_ROOT> free $(w44_free_gib <DATA_ROOT>) GiB - 23 GiB < $W44R_DATA_FLOOR_GIB GiB (DISK-PLAN needed)"; exit 95; }
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 10; done
mkdir -p "$RCPT"
echo "$(date -u +%FT%TZ) START stage_cpu72 (core 25, ionice idle)"
rm -f "$RCPT/stage_cpu72.pid"
setsid bash -c "echo \$\$ > '$RCPT/stage_cpu72.pid'; exec env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 \
  taskset -c 25 nice -n 19 ionice -c3 '$W44R_PY' '$W44R_TOOLS/followup/stage_cpu72.py' '$S' '$RCPT' '$SNAP'" \
  > "$RCPT/stage_cpu72.log" 2>&1 < /dev/null &
for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$RCPT/stage_cpu72.pid" ] && break; sleep 0.5; done
pid=$(cat "$RCPT/stage_cpu72.pid"); state=run
while kill -0 "$pid" 2>/dev/null; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then
    [ $state = run ] && { kill -STOP -- "-$pid"; state=stopped; echo "$(date -u +%FT%TZ) STOP ($(cat /tmp/wrf_gpu2_quiet 2>/dev/null))"; }
  else
    [ $state = stopped ] && { kill -CONT -- "-$pid"; state=run; echo "$(date -u +%FT%TZ) CONT"; }
  fi
  sleep 5
done
rc=$(grep -q '"sealed24_checked": 75, "sha_mismatch": \[\]' "$RCPT/stage_cpu72.log" && echo 0 || echo 1)
echo "$(date -u +%FT%TZ) END stage_cpu72 rc=$rc $(tail -1 "$RCPT/stage_cpu72.log" | cut -c1-300)"
exit "$rc"
