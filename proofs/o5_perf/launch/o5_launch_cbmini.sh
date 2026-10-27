#!/usr/bin/env bash
# o5-perf manager-sequencer entry for phase cbmini (pattern of w44-perf night_launch.sh): no waiter, no retries, no interactive input.
# usage: GA_START_UTC=.. GA_END_UTC=.. o5_launch.sh PHASE [--cpu-dry | --launch-smoke]
# Takes the OFFICIAL lock itself with --timeout 1 (label o5-perf; needs the manager's o5-perf lease). Refuses before the
# window, when the window is shorter than CAP+60 s, under QUIET / GPU-window RAM rules, below the disk floors.
# --cpu-dry: admission only (no lock, no JAX). --launch-smoke: runs the phase payload on CPU (O5_LAUNCH_SMOKE=1, no lock).
set -uo pipefail
umask 022
HERE=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
LOCK=<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh
MODE=outer
if [[ "${1:-}" == --locked ]]; then MODE=locked; shift; fi
PHASE=${1:?phase}; shift
DRY=0 SMOKE=0
if [[ "${1:-}" == --cpu-dry ]]; then DRY=1; shift; fi
if [[ "${1:-}" == --launch-smoke ]]; then SMOKE=1; shift; fi
[[ $# == 0 ]] || exit 2
case "$PHASE" in
  cbmini) CAP=300; TARGET=$HERE/cbmini_session.sh ;;
  *) exit 2 ;;
esac
epoch() { if [[ "$1" =~ ^[0-9]+$ ]]; then printf '%s\n' "$1"; else date -u -d "$1" +%s; fi; }
now() { if [[ "$DRY" == 1 && -n "${O5_DRY_NOW:-}" ]]; then printf '%s\n' "$O5_DRY_NOW"; else date -u +%s; fi; }
START=$(epoch "${GA_START_UTC:?required}") || exit 2
END=$(epoch "${GA_END_UTC:?required}") || exit 2
NOW=$(now)
if [[ "$MODE" == outer ]]; then
  export O5_RUN_TAG="${PHASE}_${NOW}_$$"
  export O5_RECEIPT=<USER_HOME>/wrf_gpu2_lanes/o5-perf/receipts/$O5_RUN_TAG
  mkdir -p "$O5_RECEIPT"
  export O5_PHASE_DEADLINE=$((NOW + CAP))
fi
[[ -n "${O5_RECEIPT:-}" && -n "${O5_PHASE_DEADLINE:-}" ]] || exit 2
receipt() {
  python3 -I - "$O5_RECEIPT" "$PHASE" "$1" "$2" "$NOW" "$START" "$END" "$CAP" "$DRY" "$SMOKE" "$TARGET" "$LOCK" "$0" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, phase, status, rc, now, start, end, cap, dry, smoke, target, lock, launcher = sys.argv[1:]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = dict(phase=phase, status=status, rc=int(rc), now_epoch=int(now), start_epoch=int(start), end_epoch=int(end),
           cap_s=int(cap), cpu_dry=dry == '1', launch_smoke=smoke == '1', target=target, target_sha256=h(target),
           official_lock=lock, official_lock_sha256=h(lock), launcher_sha256=h(launcher),
           payload_sha256={p.name: h(p) for p in sorted(Path(target).parent.glob('*')) if p.is_file()})
Path(out, status + '.json').write_text(json.dumps(rec, indent=2) + '\n')
print(json.dumps({k: rec[k] for k in ('phase', 'status', 'rc', 'cap_s')}))
PY
}
reject() { receipt "$1" "$2"; exit "$2"; }
if [[ "$SMOKE" == 1 ]]; then
  [[ "$MODE" == outer && "${GPUWRF_GPU_LOCK_HELD:-0}" != 1 ]] || exit 2
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  export O5_LAUNCH_SMOKE=1
  receipt SMOKE_START 0
  timeout --signal=TERM --kill-after=30 "$CAP" bash "$TARGET" > "$O5_RECEIPT/smoke.log" 2>&1; RC=$?
  receipt SMOKE_FINISHED "$RC"; exit "$RC"
fi
(( NOW >= START )) || reject BEFORE_WINDOW 4
NEED=$CAP
[[ "$MODE" == locked ]] && NEED=$((O5_PHASE_DEADLINE - NOW))
(( END - NOW >= NEED + 60 && NEED > 0 )) || reject INSUFFICIENT_WINDOW 6
if [ -e /tmp/wrf_gpu2_quiet ]; then reject QUIET 3; fi
[[ -s "$TARGET" && -s "$LOCK" ]] || reject MISSING_TOOL 2
if [[ "$MODE" == locked ]]; then
  [[ "$DRY" == 0 && "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == o5-perf ]] || reject NO_OFFICIAL_LOCK 2
  BUDGET=$((O5_PHASE_DEADLINE - NOW)); (( BUDGET > 0 )) || reject CAP_EXPIRED 6
  receipt LOCKED_START 0
  timeout --signal=TERM --kill-after=30 "$BUDGET" bash "$TARGET"
  exit $?
fi
DATA_FREE=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' ')
ROOT_FREE=$(df --output=avail -B1G <USER_HOME>/wrf_gpu2_lanes | tail -1 | tr -d ' ')
(( DATA_FREE >= 365 && ROOT_FREE >= 10 )) || reject DISK_FLOOR 95  # manager 2026-10-24: abort below 365 GiB
if [[ "$DRY" == 1 ]]; then receipt CPU_DRY_PASS 0; exit 0; fi
receipt ADMITTED 0
CHILD=
on_signal() { [[ -z "$CHILD" ]] || { kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD"; }; NOW=$(now); receipt SIGNAL 143; exit 143; }
trap on_signal TERM INT HUP
timeout --signal=TERM --kill-after=60 "$CAP" bash "$LOCK" --timeout 1 --label o5-perf -- \
  bash "$0" --locked "$PHASE" > "$O5_RECEIPT/gpu.log" 2>&1 & CHILD=$!
wait "$CHILD"; RC=$?; CHILD=
NOW=$(now)
receipt FINISHED "$RC"
exit "$RC"
