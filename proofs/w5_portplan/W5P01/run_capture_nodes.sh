#!/usr/bin/env bash
# Per-node shapes-only capture, each capped at 1 GB (cgroup), sequential, CPU lane cores, nice 19.
cd <USER_HOME>/src/wrf_gpu2_wt/w5-portplan
OUT=${W5_OUT:-<USER_HOME>/wrf_gpu2_lanes/w5-portplan/W5P01/capture_nodes}
mkdir -p $OUT
while read -r node; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET wait $(date -u +%H:%M:%S)"; while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; fi
  tag=$(echo "$node" | md5sum | cut -c1-8)
  echo "START $(date -u +%H:%M:%S) $tag $node"
  W5_CAPTURE_DIR=$OUT/$tag W5_CAPTURE_FAST=1 JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=<USER_HOME>/wrf_gpu2_lanes/w5-portplan/tools \
  systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 -- taskset -c 12,13,28,29 nice -n 19 timeout 600 \
    python -m pytest -q -p w5capture "$node" --basetemp <USER_HOME>/wrf_gpu2_lanes/w5-portplan/pytest/n_$tag -p no:cacheprovider > $OUT/$tag.log 2>&1
  echo "END $(date -u +%H:%M:%S) $tag rc=$? $(tail -1 $OUT/$tag.log | cut -c1-120)"
  rm -rf <USER_HOME>/wrf_gpu2_lanes/w5-portplan/pytest/n_$tag
done < "${1:-<USER_HOME>/wrf_gpu2_lanes/w5-portplan/W5P01/nodes_prio.txt}"
