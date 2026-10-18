#!/usr/bin/env bash
# w41-bundle041 RW4 CPU prediction: gpuwrf aot-key pairs on the pinned tree (lane worktree, src == pin 104c97de98 src 319848e9), sequential.
set -uo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/w41-bundle041; C=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041/cases
SRC=<USER_HOME>/src/wrf_gpu2_wt/w41-bundle041
test "$(git -C $SRC rev-parse HEAD:src)" = 319848e91a5ecaf5516d0a783bad11447c349049 || exit 2
test -z "$(git -C $SRC status --porcelain --untracked-files=all -- src)" || exit 2
mkdir -p $L/aotkey; cd $L/aotkey
run() { # name case compare hours
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  echo "$(date -u +%FT%TZ) START $1"
  env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SRC/src TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041/tmp \
    /usr/bin/time -v python3 -m gpuwrf.cli aot-key --input-dir $C/$2 --compare-input-dir $C/$3 --hours $4 --json $L/aotkey/$1.json \
    > $L/aotkey/$1.log 2>&1
  echo "$(date -u +%FT%TZ) END $1 rc=$?"; grep -E "cheap_key=|SHARED|UNAVAILABLE|Maximum resident" $L/aotkey/$1.log
}
mkdir -p <DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041/tmp
run k0227_v_0502_h72 wn3_0227 wn3_0502 72
run k0227_v_0614_h48 wn3_0227 wn3_0614 48
run k0227_v_0227in48_h3 wn3_0227 wn3_0227_48h 3
echo ALL_DONE
