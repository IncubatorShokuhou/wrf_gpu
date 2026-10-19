#!/usr/bin/env bash
# w42-bundle042: launch RC-suite workers (rc_suite.py worker) on the plan in OUT, one RAM scope + one core each.
# usage: rc_launch.sh TREE OUT SPEC...   SPEC = NAME:CORE:MEMMAX:TIERS   e.g. A:8:6G:6G  H:12:20G:20G,12G,6G
# Each worker: scripts/with_ram_slot.sh (FIFO RAM slot, MemoryMax, no swap) -> taskset CORE nice 19 -> JAX_PLATFORMS=cpu.
# Detached (setsid); worker logs OUT/worker_NAME.log, launcher logs OUT/launch_NAME.log, pid OUT/launch_NAME.pid.
set -u
TOOLS=$(cd "$(dirname "$0")" && pwd)
REPO=<DATA_ROOT>/wrf_gpu2/worktrees/w42-bundle042
TREE=$1 OUT=$2; shift 2
[ -s "$OUT/plan.json" ] || { echo "no plan in $OUT"; exit 2; }
for spec in "$@"; do
  IFS=: read -r name core max tiers <<< "$spec"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  setsid bash -c "echo \$\$ > '$OUT/launch_$name.pid'; exec '$REPO/scripts/with_ram_slot.sh' --label 'w42-bundle042-rc$name' \
    --max '$max' --timeout 28800 -- env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c '$core' nice -n 19 \
    python3 '$TOOLS/rc_suite.py' worker --tree '$TREE' --out '$OUT' --name '$name' --tiers '$tiers'" \
    > "$OUT/launch_$name.log" 2>&1 < /dev/null &
  echo "launched $name core $core max $max tiers $tiers"
done
