#!/usr/bin/env bash
# Wait for w41-rw3add's exact 'RW3ADD GPU END' MAILBOX line (appended after arming), then run arms.sh in one RAM slot.
set -uo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/w42-aotkey; M=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
S=<DATA_ROOT>/wrf_gpu2/scratch/w42-aotkey; OUT=$L/real_$(date -u +%Y%m%dT%H%MZ)
start=$(wc -l < $M); echo "$(date -u +%FT%TZ) armed at MAILBOX line $start" > $L/launch.log
until tail -n +$((start + 1)) $M | /usr/bin/grep -qE '^[0-9TZ:.-]+ w41-rw3add: RW3ADD GPU END'; do sleep 60; done
echo "$(date -u +%FT%TZ) trigger seen" >> $L/launch.log
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
mkdir -p $S/cases/wn3_0227
for f in <DATA_ROOT>/wrf_gpu2/scratch/w41-rw3add/cases/wn3_0227/*; do ln -f "$f" $S/cases/wn3_0227/; done
echo "$(date -u +%FT%TZ) case linked; OUT=$OUT" >> $L/launch.log
cd <USER_HOME>/src/wrf_gpu2_wt/w42-aotkey && scripts/with_ram_slot.sh --label w42-aotkey --max 10G --timeout 7200 -- \
  bash $L/tools/arms.sh $OUT >> $L/launch.log 2>&1
echo "$(date -u +%FT%TZ) arms rc=$?" >> $L/launch.log
