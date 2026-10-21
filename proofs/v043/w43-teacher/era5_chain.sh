#!/usr/bin/env bash
# w43-teacher ERA5 scoring chain (new file, E163): after the batch1 post_batch_v2 driver exits (one heavy process per lane),
#  E-C 20240302: wait for the CPU ref (3x1) to complete -> twin C (SCORE_ONLY, twin 2x2 vs ref 3x1) -> post_batch_v2 with C;
#  E-T 20240218: wait for the CPU ref (2x2) to complete -> post_batch_v2 without twin (PREREG v3: no twin -> tau25-72 FAIL rule).
# Then CPU teacher extracts (+ readback) of both refs in the same alisios.teacher_extract.v2 format (POINTS 20260102: grid
# XLAT/XLONG/HGT identical to WN3 on d01-d03, checked 12:3xZ). A ref that ends rc!=0 or misses DEADLINE -> INCOMPLETE, posted.
# usage: bash analysis/era5_chain.sh WAIT_PID DEADLINE_UTC     (run from the worktree root)
set -u
WAIT_PID=$1; DL=$(date -u -d "$2" +%s)
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher; L=<USER_HOME>/wrf_gpu2_lanes/w43-teacher; RUNS=$S/gpu_b2/runs
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
POINTS=$L/te_stage/20260102/root2/state/manager/teacher_extract/20260102_18z_a1/POINTS.parquet
post() { printf '%s w43-teacher: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MBX"; }
ok_run() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['rc']==0 and r['success_complete'] and r['wrfout_files']==219 else 1)" "$1/cpu_run.json" 2>/dev/null; }
wait_ref() {  # $1 run dir; 0 = complete, 1 = failed / deadline
  while ! [ -f "$1/cpu_run.json" ]; do (( $("$REAL") < DL )) || return 1; sleep 120; done
  ok_run "$1"
}
cpu_extract() {  # $1 issue, $2 ref cpu dir
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w43-teacher-extract --max 2G --timeout 36000 -- \
    taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu timeout 1800 python3 analysis/teacher_extract.py --wrfout-dir "$2" \
    --issue "${1}_18z" --attempt 1 --points "$POINTS" --out "$L/te_cpu/${1}_cpu" \
    --producer "wrf_gpu2 w43-teacher teacher_extract.py (CPU-WRF 4.7.1 reference $(basename "$(dirname "$2")"))" > "$L/te_cpu/${1}_cpu.log" 2>&1
  taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu timeout 600 python3 analysis/teacher_extract.py --readback "$L/te_cpu/${1}_cpu" > /dev/null 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['status'], {k:(v['frames'],v['allowed_nan_total'],v['foreign_nonfinite_total']) for k,v in d['domains'].items()})" "$L/te_cpu/${1}_cpu/READBACK.json" 2>&1 | cut -c1-200
}
mkdir -p "$L/te_cpu"
while [ -e "/proc/$WAIT_PID" ]; do sleep 60; done
# E-C 20240302
RC=$S/cpu/era5_20240302_ref_3x1; TW=$S/cpu/era5_20240302_twin_2x2
if wait_ref "$RC" && ok_run "$TW"; then
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  SCORE_ONLY=1 RAM_WAIT=36000 bash analysis/score_case_v2.sh C_era5_20240302 "$RC/cpu" "$TW/cpu" "$L/score/C_era5_20240302" - > "$L/score/C_era5_20240302.log" 2>&1
  if [ -d "$L/score/C_era5_20240302/x" ]; then
    post "era5_chain: twin C_era5_20240302 (twin 2x2 vs ref 3x1) scored: $(tail -1 "$L/score/C_era5_20240302.log" | cut -c1-200)"
    bash analysis/post_batch_v2.sh "$RUNS" "$2" "20240302:$L/score/C_era5_20240302/x"
  else
    post "era5_chain: twin C_era5_20240302 FAILED to score ($(tail -1 "$L/score/C_era5_20240302.log" | cut -c1-200)) -> scoring era5_20240302 as no-twin"
    bash analysis/post_batch_v2.sh "$RUNS" "$2" 20240302
  fi
  post "era5_chain: CPU extract era5_20240302 ref readback $(cpu_extract 20240302 "$RC/cpu")"
else
  post "era5_chain: era5_20240302 CPU ref/twin not complete by $(date -u -d @$DL +%FT%TZ) -> E-C row INCOMPLETE"
fi
# E-T 20240218
RT=$S/cpu/era5_20240218_ref_2x2
if wait_ref "$RT"; then
  bash analysis/post_batch_v2.sh "$RUNS" "$2" 20240218
  post "era5_chain: CPU extract era5_20240218 ref readback $(cpu_extract 20240218 "$RT/cpu")"
else
  post "era5_chain: era5_20240218 CPU ref not complete by $(date -u -d @$DL +%FT%TZ) -> E-T row INCOMPLETE"
fi
post "era5_chain: done"
