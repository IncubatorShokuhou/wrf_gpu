#!/usr/bin/env bash
# w43-teacher: start the E-T reference (20240218_18z, 2x2) on 1,4,5,7 once the E-C twin launcher PID exits (wait on PID, E77/E141).
set -u
TW=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/cpu/era5_20240302_twin_2x2
C=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/cpu/era5_20240218_ref_2x2
P=$(cat "$TW.pid")
while kill -0 "$P" 2>/dev/null; do sleep 60; done
grep -q " END " "$TW/run_cpu.log" 2>/dev/null || echo "$(date -u +%FT%TZ) chain: twin launcher exited without END line" >> "$C.chain.log"
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
echo $$ > "$C.pid"
exec bash <DATA_ROOT>/wrf_gpu2/worktrees/w43-teacher/analysis/cpu/run_cpu.sh "$C" 1,4,5,7 2026-10-22T07:30:00Z 24G era5_0218_ref
