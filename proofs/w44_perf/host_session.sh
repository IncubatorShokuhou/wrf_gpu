#!/usr/bin/env bash
# Supplemental diagnostic only, after the matched timings. No speed verdict.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[[ "${W44_LAUNCH_SMOKE:-0}" == 1 || ( "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ) ]] || exit 2
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
exec bash "$HERE/arm.sh" v043 H43 host
