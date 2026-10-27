#!/usr/bin/env bash
# Private overlay of the lane's old-version cache; absolute child paths unchanged.
set -euo pipefail
LAYER=${1:?layer}; shift
[[ $LAYER == <USER_HOME>/wrf_gpu2_lanes/w44-perf/launch_v5_runs/*/layer_v032 ]] || exit 2
mkdir -p "$LAYER/upper" "$LAYER/work"
exec bwrap --dev-bind / / --die-with-parent \
  --overlay-src <USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/cache_v032 \
  --overlay "$LAYER/upper" "$LAYER/work" <USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/cache_v032 -- "$@"
