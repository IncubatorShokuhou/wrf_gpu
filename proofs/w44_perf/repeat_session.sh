#!/usr/bin/env bash
# Manager-launched new session; original Oct23 evidence stays intact.
set -euo pipefail
umask 077
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w44-perf
ARM=$HERE/arm.sh
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
PY=<USER_HOME>/miniconda3/bin/python
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [[ "${W44_LAUNCH_SMOKE:-0}" == 1 && "${W44_CPU_PROBE:-0}" != 1 ]]; then
  for spec in 'v032 R2_W32' 'v040 R2_W40' 'v043 R2_W43'; do
    read -r version label <<< "$spec"
    bash "$ARM" "$version" "$label" warm
  done
  exit 0
fi
if [[ "${1:-}" == --cpu-dry ]]; then
  bash "$ARM" v032 R2_D32 warm --dry-run
  bash "$ARM" v040 R2_D40 warm --dry-run
  bash "$ARM" v043 R2_D43 warm --dry-run
  exit 0
fi
[[ "${W44_CPU_PROBE:-0}" == 1 || ( "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ) ]] || exit 2
if [[ "${W44_CPU_PROBE:-0}" != 1 ]]; then grep -q 'manager -> w44-perf: GO GPU ' "$MAILBOX" || exit 2; fi
RUN_ROOT=${W44_RUN_ROOT:-$L/gpu}
[[ ! -e "$RUN_ROOT/repeatcompare.started" ]] || exit 4
JAX_PLATFORMS=cpu bash "$HERE/real_utc.sh" --iso > "$RUN_ROOT/repeatcompare.started"
QUIET_TOKEN="w44-perf repeatcompare $$"
if [[ "${W44_CPU_PROBE:-0}" != 1 && "${W44_SIZE:-prod}" == prod ]]; then
  (set -o noclobber; printf '%s\n' "$QUIET_TOKEN" > /tmp/wrf_gpu2_quiet)
  printf '%s\n' "$QUIET_TOKEN" > "$RUN_ROOT/quiet_token"
fi
post() { printf '%s w44-perf: %s\n' "$(JAX_PLATFORMS=cpu bash "$HERE/real_utc.sh" --iso)" "$*" >> "$MAILBOX"; }
cleanup() {
  rc=$?
  if [[ -e /tmp/wrf_gpu2_quiet && "$(cat /tmp/wrf_gpu2_quiet)" == "$QUIET_TOKEN" ]]; then rm -f /tmp/wrf_gpu2_quiet; fi
  if [[ "${W44_CPU_PROBE:-0}" != 1 ]]; then post "GPU END phase=repeatcompare rc=$rc; all arms retained, fresh-session A/A required"; fi
}
trap cleanup EXIT
if [[ "${W44_CPU_PROBE:-0}" != 1 ]]; then post 'GPU START phase=repeatcompare; immutable protocol; no prior-session pooling'; fi
for spec in 'v032 R2_W32' 'v040 R2_W40' 'v043 R2_W43' 'v043 R2_B43_1' 'v032 R2_A32_1' 'v040 R2_C40_1' 'v043 R2_B43_2' 'v040 R2_C40_2' 'v032 R2_A32_2' 'v043 R2_B43_3'; do
  read -r version label <<< "$spec"
  bash "$ARM" "$version" "$label" warm
done
# No profiling or JAX import. Summary is also evaluated after release of QUIET.
JAX_PLATFORMS=cpu "$PY" "$HERE/summarize_gpu.py" --plan "$HERE/REPEAT_PLAN.json" --runs "$RUN_ROOT/runs" --case "${W44_INPUT_CASE:-<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725}" --out "$RUN_ROOT/BENCH.json"
