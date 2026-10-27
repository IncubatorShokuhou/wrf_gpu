#!/usr/bin/env bash
# Immutable night entry point. Manager sequencer only; cap 30 min incl. E41.
# Required environment: GA_START_UTC and GA_END_UTC (explicit UTC timestamps).
set -euo pipefail
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
TOOL=<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_5367b7fda6/night_supervisor.py
EXPECTED=e86766677bb962af73233e78a8f6ea03a61ef8ba33a59f17e2f672e6431e60f1
if [[ $# != 0 || ! -f "$TOOL" ]]; then
  echo '{"rc":126,"verdict":"REFUSED_LAUNCHER_INPUT_OR_MISSING_TOOL"}' >&2
  exit 126
fi
ACTUAL=$(sha256sum "$TOOL" | awk '{print $1}')
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo '{"rc":126,"verdict":"REFUSED_SUPERVISOR_HASH"}' >&2
  exit 126
fi
exec python3 -u "$TOOL"
