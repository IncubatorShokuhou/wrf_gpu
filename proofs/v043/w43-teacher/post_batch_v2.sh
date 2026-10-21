#!/usr/bin/env bash
# v2 (2026-10-21, E163 new file): scores with score_case_v2.sh (G-INT import root redirected, tool error -> INCOMPLETE); a case that
# already has score/<name>/VERDICT.json gets INTEG_ONLY=1 (G-INT + verdict rerun, D6 rows kept); an existing PASS extract readback is reused.
# w43-teacher: after each GPU case of a batch ends (receipt rc set), score it (score_case.sh, PREREG v3) and write the GPU teacher
# extract + readback, one case at a time (one heavy process per lane). Raw outputs are NOT deleted here (after the 0227 d6_N72 check).
# usage: bash analysis/post_batch.sh RUNS_ROOT DEADLINE_UTC ISSUE[:C_DIR] ...   (run from the worktree root)
set -u
RUNS=$1; DL=$(date -u -d "$2" +%s); shift 2
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher; L=<USER_HOME>/wrf_gpu2_lanes/w43-teacher
TE=$L/te_stage; MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
mkdir -p "$L/score" "$L/te_gpu"
post() { printf '%s w43-teacher: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MBX"; }
rc_of() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('-' if r.get('end_utc') is None or r.get('rc') is None else r['rc'])" "$1" 2>/dev/null || echo -; }
for spec in "$@"; do
  i=${spec%%:*}; C=-; [[ $spec == *:* ]] && C=${spec#*:}
  name=wn3_$i; [ -d "$S/cases/era5_$i" ] && name=era5_$i
  R=$RUNS/$name/receipt.json
  while [ "$(rc_of "$R")" = - ] && (( $("$REAL") < DL )); do sleep 60; done
  rc=$(rc_of "$R")
  if [ "$rc" != 0 ]; then post "post_batch $name: GPU case rc=$rc (or no receipt by $(date -u -d @$DL +%TZ)) -> INCOMPLETE, not scored"; continue; fi
  GPU=$RUNS/$name/wrfout; REF=$S/ref${i:4:4}_view; [[ $name == era5_* ]] && REF=$(ls -d $S/cpu/era5_${i}_ref_*/cpu | head -1)
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  if [ -f "$L/score/$name/VERDICT.json" ]; then
    INTEG_ONLY=1 RAM_WAIT=36000 bash analysis/score_case_v2.sh "$name" "$REF" "$GPU" "$L/score/$name" "$C" > "$L/score/$name.integ_rerun.log" 2>&1
    v="(G-INT rerun) $(grep '^INTEGRITY' "$L/score/$name.integ_rerun.log" | cut -c1-150) $(tail -1 "$L/score/$name.integ_rerun.log" | cut -c1-300)"
  else
    RAM_WAIT=36000 bash analysis/score_case_v2.sh "$name" "$REF" "$GPU" "$L/score/$name" "$C" > "$L/score/$name.log" 2>&1
    v="$(grep '^INTEGRITY' "$L/score/$name.log" | cut -c1-150) $(tail -1 "$L/score/$name.log" | cut -c1-300)"
  fi
  P=$TE/$i/root2/state/manager/teacher_extract/${i}_18z_a1/POINTS.parquet
  [ -f "$P" ] || P=$TE/20260102/root2/state/manager/teacher_extract/20260102_18z_a1/POINTS.parquet
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['status'] == 'PASS' else 1)" "$L/te_gpu/${i}_gpu/READBACK.json" 2>/dev/null; then
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w43-teacher-extract --max 2G --timeout 36000 -- \
    taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu timeout 1800 python3 analysis/teacher_extract.py --wrfout-dir "$GPU" \
    --issue "${i}_18z" --attempt 1 --points "$P" --out "$L/te_gpu/${i}_gpu" \
    --producer "wrf_gpu2 w43-teacher teacher_extract.py (GPU wrf_gpu v0.4.2 bundle bfa2dc90, $name)" > "$L/te_gpu/${i}_gpu.log" 2>&1
  taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu timeout 600 python3 analysis/teacher_extract.py --readback "$L/te_gpu/${i}_gpu" > /dev/null 2>&1
  fi
  rb=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['status'], {k:(v['frames'],v['allowed_nan_total'],v['foreign_nonfinite_total']) for k,v in d['domains'].items()})" "$L/te_gpu/${i}_gpu/READBACK.json" 2>&1 | cut -c1-200)
  post "post_batch $name scored: $v | GPU extract readback $rb"
done
