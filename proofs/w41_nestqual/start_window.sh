#!/bin/bash
# Start the ALISIOS CPU window queue at exactly 2026-10-18T09:00:00Z. Grant (alisios_mgr core correction 2026-10-18,
# cross TO_WRF_GPU2_TWINS_CORE_CORRECTION_20261018.md, manager 03:45:17Z/03:45:26Z): physical 0,1,2 + SMT 16,17,18,
# NOT 10 (ALISIOS G2 GPU arms), free by 18:30Z hard -> run_twin watchdog deadline 18:27Z, twins RAM <= 24 GB.
# Plan: 3 ranks, one per physical core 0,1,2, SMT siblings idle (shared siblings measured ~2.5x slower per run),
# C2 YSU -> C3 Shin-Hong -> C4 ACM2 sequential; C4 dropped + reported if its predicted end is past 18:27Z.
# C7 (sealed 01:11Z) rides along for GPU staging + scoring only. Touch <lane>/QUEUE_STOP to cancel before the start.
set -u
HERE=$(dirname "$(readlink -f "$0")"); LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual
START=$(date -u -d 2026-10-18T09:00:00Z +%s)
while (( $(date +%s) < START )); do [ -e "$LANE/QUEUE_STOP" ] && { echo "$(date -u +%FT%TZ) window start cancelled (QUEUE_STOP)" >> "$LANE/twin_status.log"; exit 3; }; sleep 20; done
[ -e "$LANE/QUEUE_STOP" ] && exit 3
exec "$HERE/run_queue.sh" 0,1,2 2026-10-18T18:27:00Z live C2 C3 C4 C7
