#!/usr/bin/env bash
cd <USER_HOME>/wrf_gpu2_lanes/w5-portplan
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export TRITON_CACHE_DIR=<USER_HOME>/wrf_gpu2_lanes/w5-portplan/triton_cache JAX_PLATFORMS=cpu W5_WORKERS=1 W5_MAX_CHARS=400000 W5_TIMEOUT=900
exec systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -- taskset -c 12,13,28,29 nice -n 19 \
  python tools/w5compile.py W5P01/rep W5P01/compile_rep_c.jsonl "$1"
