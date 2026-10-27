#!/usr/bin/env bash
# Supplemental capture only: run after the unchanged d0935fec compare session
# if the manager's existing window has >=180s remaining. Needs explicit GO.
# Roots100/101/102 include d01 radiation at100, d02 at ownstep301 (root101),
# and one ordinary root. Uses the same input, allocator and current bundle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && "${GPUWRF_GPU_LOCK_LABEL:-}" == w44-perf ]] || exit 2
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
exec bash "$HERE/arm.sh" v043 R43 radnodes
