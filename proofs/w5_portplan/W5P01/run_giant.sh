#!/usr/bin/env bash
# Re-run one giant TTIR compile with a long timeout and CPU-time accounting.
cd <USER_HOME>/wrf_gpu2_lanes/w5-portplan
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
f=$1; be=$2; arch=$3; nw=$4; ns=$5
TRITON_CACHE_DIR=<USER_HOME>/wrf_gpu2_lanes/w5-portplan/triton_cache JAX_PLATFORMS=cpu systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -- taskset -c 12,13,28,29 nice -n 19 \
  /usr/bin/time -f '{"cpu_user_s": %U, "cpu_sys_s": %S, "wall_s": %e, "maxrss_kb": %M}' -o W5P01/giant_$(basename $f .ttir)_$arch.time \
  timeout 3600 python tools/w5compile_one.py $f $be $arch $nw $ns >> W5P01/compile_giant.jsonl
