#!/usr/bin/env bash
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
exec env JAX_PLATFORMS=cpu taskset -c 12,28 nice -n 19 <USER_HOME>/miniconda3/bin/python3 "$(dirname "$0")/prepare.py" "$@"
