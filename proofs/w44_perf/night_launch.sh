#!/usr/bin/env bash
# Manager sequencer entry: no waiter, retries or interactive input.
# Public wrappers bind PHASE. CPU dry-run exercises admission without touching
# the official lock, NVML, JAX or any GPU command.
set -uo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
PY=<USER_HOME>/miniconda3/bin/python
MODE=outer
if [[ "${1:-}" == --locked ]]; then MODE=locked; shift; fi
PHASE=${1:?phase}; shift
DRY=0
SMOKE=0
if [[ "${1:-}" == --cpu-dry ]]; then DRY=1; shift; fi
if [[ "${1:-}" == --launch-smoke ]]; then SMOKE=1; shift; fi
[[ $# == 0 ]] || exit 2
unset W44_LAUNCH_SMOKE W44_SMOKE_ID
case "$PHASE" in
  prime32|prime40) CAP=1500; TARGET=$L/tools_2d9959fbab/proofs/w44_perf/session.sh; ARGS=("$PHASE") ;;
  compare) CAP=1800; TARGET=$L/tools_2d9959fbab/proofs/w44_perf/session.sh; ARGS=(compare) ;;
  repeatcompare) CAP=600; TARGET=$HERE/repeat_session.sh; ARGS=() ;;
  hostprofile) CAP=180; TARGET=$HERE/host_session.sh; ARGS=() ;;
  radiation) CAP=180; TARGET=$HERE/radiation_session.sh; ARGS=() ;;
  wn3sds) CAP=300; TARGET=$HERE/wn3_sds_gpu.sh; ARGS=() ;;
  *) exit 2 ;;
esac
export JAX_PLATFORMS=cpu
REAL=$HERE/real_utc.sh
now() {
  if [[ "$DRY" == 1 && -n "${W44_DRY_NOW:-}" ]]; then printf '%s\n' "$W44_DRY_NOW"; else bash "$REAL"; fi
}
epoch() {
  if [[ "$1" =~ ^[0-9]+$ ]]; then printf '%s\n' "$1"; else date -u -d "$1" +%s; fi
}
START=$(epoch "${GA_START_UTC:?required}") || exit 2
END=$(epoch "${GA_END_UTC:?required}") || exit 2
NOW=$(now)
[[ "$NOW" =~ ^[0-9]+$ && "$START" =~ ^[0-9]+$ && "$END" =~ ^[0-9]+$ ]] || exit 2
if [[ "$MODE" == outer ]]; then
  RID="${PHASE}_${NOW}_$$"
  export W44_NIGHT_RECEIPT="$L/night_receipts/$RID"
  mkdir -p "$W44_NIGHT_RECEIPT"
  export W44_PHASE_DEADLINE=$((NOW + CAP))
fi
[[ -n "${W44_NIGHT_RECEIPT:-}" && -n "${W44_PHASE_DEADLINE:-}" ]] || exit 2
if [[ "$SMOKE" == 1 ]]; then
  [[ "$MODE" == outer && "$PHASE" =~ ^(repeatcompare|hostprofile)$ ]] || exit 2
  [[ "${GPUWRF_GPU_LOCK_HELD:-0}" != 1 ]] || exit 2
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  export W44_LAUNCH_SMOKE=1
  export W44_SMOKE_ID="${PHASE}_${NOW}_$$"
  timeout --signal=TERM --kill-after=30 "$CAP" bash "$TARGET" "${ARGS[@]}"
  exit $?
fi
receipt() {  # stdlib only; never emit environment or raw profiler metadata
  JAX_PLATFORMS=cpu "$PY" - "$W44_NIGHT_RECEIPT" "$PHASE" "$1" "$2" "$NOW" "$START" "$END" "$CAP" "$DRY" "$TARGET" "$LOCK" "$0" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,phase,status,rc,now,start,end,cap,dry,target,lock,launcher=sys.argv[1:]
rec=dict(phase=phase,status=status,rc=int(rc),real_now_epoch=int(now),start_epoch=int(start),end_epoch=int(end),
         cap_s=int(cap),cpu_dry=bool(int(dry)),gpu_started=dry=='0' and (status=='LOCKED_START' or Path(out,'LOCKED_START.json').exists()),
         target=target,target_sha256=hashlib.sha256(Path(target).read_bytes()).hexdigest(),
         official_lock=lock,official_lock_sha256=hashlib.sha256(Path(lock).read_bytes()).hexdigest(),
         launcher_sha256=hashlib.sha256(Path(launcher).read_bytes()).hexdigest())
Path(out,status+'.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec))
PY
}
reject() { receipt "$1" "$2"; exit "$2"; }
(( NOW >= START )) || reject BEFORE_WINDOW 4
NEED=$CAP
if [[ "$MODE" == locked ]]; then NEED=$((W44_PHASE_DEADLINE - NOW)); fi
# Reserve teardown time before the manager's hard window end.
(( END - NOW >= NEED + 60 && NEED > 0 )) || reject INSUFFICIENT_WINDOW 6
if [ -e /tmp/wrf_gpu2_quiet ]; then reject QUIET 3; fi
[[ -s "$TARGET" && -s "$LOCK" ]] || reject MISSING_TOOL 2
if [[ "$MODE" == locked ]]; then
  [[ "$DRY" == 0 && "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ]] || reject NO_OFFICIAL_LOCK 2
  BUDGET=$((W44_PHASE_DEADLINE - NOW))
  (( BUDGET > 0 )) || reject CAP_EXPIRED 6
  receipt LOCKED_START 0
  timeout --signal=TERM --kill-after=30 "$BUDGET" bash "$TARGET" "${ARGS[@]}"
  exit $?
fi
if [[ "$DRY" == 1 ]]; then receipt CPU_DRY_PASS 0; exit 0; fi
# Mandatory global floors; estimates/accounting uncertainty are not extra gates.
ROOT_FREE=$(df --output=avail -B1 <USER_HOME>/wrf_gpu2_lanes/w44-perf | tail -1 | tr -d ' ')
DATA_FREE=$(df --output=avail -B1 <DATA_ROOT> | tail -1 | tr -d ' ')
(( ROOT_FREE >= 20 * 1024**3 && DATA_FREE >= 200 * 1024**3 )) || reject DISK_FLOOR 95
receipt ADMITTED 0
CHILD=
on_signal() {
  [[ -z "$CHILD" ]] || { kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD"; }
  NOW=$(now); receipt SIGNAL 143; exit 143
}
trap on_signal TERM INT HUP
# One-second lock wait only; the manager owns all sequencing. A timeout owns
# this child tree and the official wrapper retains FD9 until CUDA work exits.
timeout --signal=TERM --kill-after=60 "$CAP" bash "$LOCK" --timeout 1 --label w44-perf -- \
  bash "$0" --locked "$PHASE" > "$W44_NIGHT_RECEIPT/gpu.log" 2>&1 & CHILD=$!
wait "$CHILD"; RC=$?; CHILD=
NOW=$(now)
receipt FINISHED "$RC"
exit "$RC"
