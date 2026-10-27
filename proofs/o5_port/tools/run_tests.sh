#!/usr/bin/env bash
# usage: run_tests.sh LABEL TREE test_file...   per-file pytest processes on CPU (lane o5-port)
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi
avail=$(df --output=avail -BG <DATA_ROOT> | tail -1 | tr -dc 0-9); if [ "$avail" -lt 365 ]; then echo "DISK_LOW ${avail}G"; exit 3; fi
LABEL=$1; TREE=$2; shift 2
L=<USER_HOME>/wrf_gpu2_lanes/o5-port
OUT=$L/$LABEL.log; : > $OUT
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 CUDA_VISIBLE_DEVICES=
cd "$TREE" || exit 2
echo "tree $(git rev-parse HEAD) dirty=$(git status --porcelain -- src tests | wc -l)" >> $OUT
"$TREE/scripts/with_ram_slot.sh" --label o5-port --max 12G --timeout 5400 -- bash -c '
for f in "$@"; do
  r=$(taskset -c 8,9,12,24,25,28 nice -n 19 timeout 1500 python -m pytest -q -p no:cacheprovider "$f" \
      --basetemp '"$L"'/pytest/$(basename $f .py) 2>&1 | tail -1)
  echo "$f :: $r"
done' _ "$@" >> $OUT 2>&1
echo "rc=$?" >> $OUT
