#!/usr/bin/env bash
# Synchronous CPU post-GPU phase after all model children are reaped; v9 flag remains intact.
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
HERE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/post_run
POST_SUMS_SHA=3eb263961b62207ff671746eabfeeedfe601ba058cd191e77955be188296dde2
[[ $(sha256sum "$HERE/SHA256SUMS" | cut -d' ' -f1) == "$POST_SUMS_SHA" ]] || exit 9
(cd "$HERE" && sha256sum -c --quiet SHA256SUMS) || exit 9
if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 ]]; then
 exec env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 taskset -c 12,28 nice -n 19 bash "$HERE/smoke.sh"
fi
exec bash "$HERE/launch.sh" "${1:?actual PIN manifest}" "${GA_START_UTC:?}" 2026-10-29T05:30:00Z
