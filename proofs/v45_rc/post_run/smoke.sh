#!/usr/bin/env bash
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
HERE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/post_run
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
for file in control_post.py control_workflow.py control_catalog.py control_service.py control_reference_phase.py control_own_queued.py control_recovery_threshold.py control_completeness.py control_plan_pin.py; do
 <USER_HOME>/miniconda3/bin/python3 -B "$HERE/$file"
done
