#!/usr/bin/env bash
# Manual entry after an explicit manager -> w44-perf GO GPU window.
# Invoke under scripts/with_gpu_lock.sh --label w44-perf -- timeout 1800 bash ...
# Pre-register: ONE compare session, B43_1 A32_1 C40_1 B43_2 C40_2 A32_2 B43_3;
# all measurements retained. Per-version max-min/mean <=5% for a version wall
# verdict; otherwise INCONCLUSIVE (no selective deletion or favorable retry).
# Prime phases are separate cold-setup arms and never counted as warm speed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
PHASE=${1:?prime32|prime40|compare|dry}
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
REAL=$HERE/real_utc.sh
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [[ "$PHASE" == dry ]]; then
  for spec in 'v032 D32 prime' 'v040 D40 warm' 'v043 D43 warm' 'v043 DG graph' 'v032 DN nodes' 'v043 DA auto'; do
    read -r v a k <<< "$spec"
    bash "$HERE/arm.sh" "$v" "$a" "$k" --dry-run
  done
  echo 'CPU dry lifecycle PASS; no GPU/NVML/consumer imports'
  exit 0
fi
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ]] || exit 2
[[ -n "$(grep -E 'manager -> w44-perf: GO GPU ' "$MAILBOX" | tail -1)" ]] || { echo 'No manager GO GPU line'; exit 2; }
[[ "$PHASE" =~ ^(prime32|prime40|compare)$ ]] || exit 2
mkdir -p "$L/gpu"
[[ ! -e "$L/gpu/$PHASE.started" ]] || { echo 'immutable phase already started'; exit 4; }
JAX_PLATFORMS=cpu "$REAL" --iso > "$L/gpu/$PHASE.started"
post() { printf '%s w44-perf: %s\n' "$(JAX_PLATFORMS=cpu "$REAL" --iso)" "$*" >> "$MAILBOX"; }
cleanup() {
  rc=$?
  if [[ -n "${QUIET_TOKEN:-}" && -e /tmp/wrf_gpu2_quiet ]] && [[ "$(cat /tmp/wrf_gpu2_quiet)" == "$QUIET_TOKEN" ]]; then rm -f /tmp/wrf_gpu2_quiet; fi
  post "GPU END phase=$PHASE rc=$rc (no gate verdict until CPU receipt/E41 analysis)"
}
trap cleanup EXIT
post "GPU START phase=$PHASE fixed explicit cuda_async .25 pool, immutable sources; manager GO read manually before launch"
case "$PHASE" in
  prime32) bash "$HERE/arm.sh" v032 P32 prime ;;
  prime40) bash "$HERE/arm.sh" v040 P40 prime ;;
  compare)
    QUIET_TOKEN="w44-perf compare $$"
    (set -o noclobber; printf '%s\n' "$QUIET_TOKEN" > /tmp/wrf_gpu2_quiet)
    # One warm-up per source: lazy init/output caches populated before timing.
    bash "$HERE/arm.sh" v032 W32 warm
    bash "$HERE/arm.sh" v040 W40 warm
    bash "$HERE/arm.sh" v043 W43 warm
    for spec in 'v043 B43_1' 'v032 A32_1' 'v040 C40_1' 'v043 B43_2' 'v040 C40_2' 'v032 A32_2' 'v043 B43_3'; do
      read -r v a <<< "$spec"
      bash "$HERE/arm.sh" "$v" "$a" warm
    done
    # Host-idle/output timeline: graph tracing, no node inflation (E185).
    bash "$HERE/arm.sh" v032 G32 graph
    bash "$HERE/arm.sh" v043 G43 graph
    # Source-family ranking only; node-traced host gaps are never used as wall evidence.
    bash "$HERE/arm.sh" v032 N32 nodes
    bash "$HERE/arm.sh" v043 N43 nodes
    # Pool-policy bridge on the same current source, separated from version ratios.
    bash "$HERE/arm.sh" v043 AUTO43 auto
    ;;
esac
