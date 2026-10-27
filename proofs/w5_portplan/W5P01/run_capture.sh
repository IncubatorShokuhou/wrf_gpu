#!/usr/bin/env bash
cd <USER_HOME>/src/wrf_gpu2_wt/w5-portplan
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export W5_CAPTURE_DIR=<USER_HOME>/wrf_gpu2_lanes/w5-portplan/W5P01/capture W5_CAPTURE_FAST=1 JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
  PYTHONPATH=<USER_HOME>/wrf_gpu2_lanes/w5-portplan/tools
exec scripts/with_ram_slot.sh --label w5-portplan --max 6G --timeout 5400 -- taskset -c 12,13,28,29 nice -n 19 timeout 5000 \
  python -m pytest -q -p w5capture $(cat <USER_HOME>/wrf_gpu2_lanes/w5-portplan/W5P01/nodes.txt) \
  --basetemp <USER_HOME>/wrf_gpu2_lanes/w5-portplan/pytest/capture -p no:cacheprovider
