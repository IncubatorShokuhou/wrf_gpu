#!/usr/bin/env bash
# Immutable design-B CUDA launcher; manager sequencer only.
set -euo pipefail
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
[[ $# == 0 ]] || exit 126
TOOLS=<USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_v3c_71b8edf1e6
check() { [[ $(sha256sum "$1" | awk '{print $1}') == "$2" ]] || exit 126; }
check "$TOOLS/manifest.json" 111059beb11b0fa90338a943c2e2358da0aa6ada18093d7bed824139486b86be
check "$TOOLS/twin_probe.py" 5f5881973a248aff0be8643d227973802ecf92a1d3940832ba4ce103a004414a
check "$TOOLS/twin_oracle.py" 075aaa0fb786601db38814a74b5c7894861ed90e18a1b4c607380a8cbf405b6b
check "$TOOLS/history_gate.py" 1599cdc360e1ba5978a17709771ab61fa3b7ebd62532ecd52bf77aa2f8c3c2a4
check "$TOOLS/pristine_conditioning.py" de667750f053d3db1320f255eca82bcb72c8c4817f9a12fdc4966ff8541386e1
check "$TOOLS/night_probe_design_b.py" f0ed1306a04119fa1ca363bbe79c1d7ff3d2cc40cdb52834bbcb6e509d40f635
check "$TOOLS/night_supervisor_design_b.py" ba289b3a96b96032f36196f42877154333b57c1bdf4fb472ec80cc6fd2922e7b
check "$TOOLS/budget_cubin_gate.py" 69dcc2dad6b8d634882502de27b07e4e5cbd3c3b2b591b673c41c688ecd98e15
check "$TOOLS/budget_post_cpu.py" cd463b13eda87fcd2c31b6036d3e58c6a6bc3a3882fb1bf539ddc900a947fa39
check "$TOOLS/test_night_design_b.py" f9b9694e7bd3c3ea4039e3d3dc9ba790ba5dddb60b08bc5e845e84372dd5ec25
check <USER_HOME>/wrf_gpu2_lanes/w44-budget/night_tools_5367b7fda6/night_supervisor.py e86766677bb962af73233e78a8f6ea03a61ef8ba33a59f17e2f672e6431e60f1
exec python3 -u "$TOOLS/night_supervisor_design_b.py"
