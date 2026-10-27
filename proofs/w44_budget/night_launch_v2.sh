#!/usr/bin/env bash
# Immutable v2; manager sequencer only. 60 min = cold setup/replay + E41.
set -euo pipefail
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
TOOLS=<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_v2_d485e554d1
check() { [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || exit 126; }
[[ $# == 0 ]] || exit 126
check "$TOOLS/night_supervisor_v2.py" 1fcda1a39bc32a5162251b95ca8bc174f01c253e024f2a619e3f534d8d631d75
check "$TOOLS/night_probe_v2.py" 049b8bd9711d9e3c1be2f6590a3152f270d029c27cdec3426f3ffcfa0ae6f753
check <USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_5367b7fda6/night_supervisor.py e86766677bb962af73233e78a8f6ea03a61ef8ba33a59f17e2f672e6431e60f1
exec python3 -u "$TOOLS/night_supervisor_v2.py"
