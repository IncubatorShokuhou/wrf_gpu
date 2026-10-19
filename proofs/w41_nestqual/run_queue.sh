#!/bin/bash
# CPU-twin queue for the ALISIOS CPU window (lane w41-nestqual).
#  1. builds every arm's case (staged inputs only)
#  2. per arm, in the background on lane cores: waits (metadata only: dir names/permissions) until rd-s6's post_case
#     has made the kept G2 raw wrfout read-only, then stages frames 0-24 h + receipts under the guard (stage_gpu.py)
#  3. runs the CPU twins SEQUENTIALLY on the given physical cores (one rank per core, ranks = number of cores given;
#     SMT siblings left idle: N/N+16 sharing measured ~2.5x slower per run, no throughput gain), in contract order.
#     Before each start: predicted end = now + rate*24*1.03 + 240 s (rate = last sealed wall s/sim-h, default 480)
#     must be <= deadline, else SKIP (logged; live mode also reports the drop in MAILBOX)
#  4. after each sealed twin, in the background on lane cores: score (release scorer, 3 domains) once GPU_STAGE exists
# usage: run_queue.sh <cores c0,c1,c2,c3> <deadline ISO-UTC> <live|dry> <arm> [arm ...]
set -u
CORES=$1; DEADLINE_ISO=$2; MODE=$3; shift 3
HERE=$(dirname "$(readlink -f "$0")"); TREE=$(readlink -f "$HERE/../..")
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; SCR=<DATA_ROOT>/wrf_gpu2/scratch/w41-nestqual
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; NR=$(awk -F, '{print NF}' <<< "$CORES")
STAGE=$LANE/guard/stage_20260707_18z/STAGE.json; CASES=${TWIN_CASES:-$SCR/cases}; HOURS=24; ISSUE=20260707_18z
G2RUNS=<DATA_ROOT>/alisios/scratch/rd-s6-wrf-cloud-attribution/g2/runs/$ISSUE
DEADLINE=$(date -u -d "$DEADLINE_ISO" +%s) || exit 2
taskset -pc "$CORES" $$ > /dev/null      # launcher + helpers stay on the granted cores (never OS cores)
log(){ echo "$(date -u +%FT%TZ) queue $*" | tee -a "$LANE/twin_status.log"; }
PYENV="JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1"
for ARM in "$@"; do
  CASE=$CASES/${ISSUE}_$ARM
  [ -s "$CASE/PREPARE.json" ] || python3 "$HERE/build_twin.py" "$STAGE" "$CASES" "$ARM" --hours $HOURS \
    --tables "$SCR/tables" --ranks "$NR" > "$LANE/build_$ARM.log" 2>&1 || log "build $ARM FAILED"
  [ -s "$CASE/GPU_STAGE.json" ] && continue
  ( taskset -c 8,9,12 bash -c "
      W=$G2RUNS/$ARM/v040/wrfout
      until [ -d \$W ] && [ ! -w \$W ] && [ \$(ls \$W | grep -c '^wrfout_d03_') -ge $((HOURS+1)) ]; do
        (( \$(date +%s) < $DEADLINE + 10800 )) || { echo GPU_OUTPUT_NEVER_READY; exit 1; }; sleep 60; done
      while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
      env $PYENV nice -n 19 python $HERE/stage_gpu.py $CASE $SCR/stage_gpu_${ISSUE}_$ARM $LANE/guard/gpu_${ISSUE}_$ARM $HOURS" \
      > "$LANE/stage_gpu_$ARM.log" 2>&1
    echo "$(date -u +%FT%TZ) queue GPU staged $ARM rc=$? $(tail -1 $LANE/stage_gpu_$ARM.log | cut -c1-300)" >> "$LANE/twin_status.log" ) &
done
RATE=480
for ARM in "$@"; do
  CASE=$CASES/${ISSUE}_$ARM
  [ -s "$CASE/PREPARE.json" ] || { log "$ARM has no case, skipped"; continue; }
  if [ -s "$CASE/cpu_run.json" ]; then log "$ARM already sealed"
  else
    NOW=$(date +%s); EST=$(python3 -c "print(int($RATE*$HOURS*1.03+240))")
    if (( NOW + EST > DEADLINE )); then
      log "SKIP $ARM: predicted end $(date -u -d @$((NOW+EST)) +%TZ) > deadline $DEADLINE_ISO (rate ${RATE} s/sim-h)"
      [ "$MODE" = live ] && echo "$(date -u +%FT%TZ) w41-nestqual: CPU twin $ARM DROPPED (not started): predicted end $(date -u -d @$((NOW+EST)) +%TZ) > deadline $DEADLINE_ISO at the measured ${RATE} s/sim-h; $ARM stays nested-experimental (no D6)" >> "$MAILBOX"
      continue
    fi
    log "start $ARM (predicted end $(date -u -d @$((NOW+EST)) +%TZ), rate ${RATE} s/sim-h)"
    "$HERE/run_twin.sh" "$CASE" "$CORES" "$DEADLINE_ISO" "$MODE" $HOURS 10G > "$LANE/run_$ARM.out" 2>&1
    RC=$?
    if [ $RC -ne 0 ] || [ ! -s "$CASE/cpu_run.json" ]; then log "$ARM not sealed (rc=$RC); next arm"; continue; fi
    RATE=$(python3 -c "import json;t=json.load(open('$CASE/cpu_run.json'))['timing'];print(int(t['wall_s']/$HOURS))")
    log "$ARM sealed; measured wall rate ${RATE} s/sim-h (incl. init)"
  fi
  ( taskset -c 8,9,12 bash -c "
      until [ -s $CASE/GPU_STAGE.json ]; do (( \$(date +%s) < $DEADLINE + 10800 )) || exit 1; sleep 60; done
      while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
      $HERE/score_twin.sh $CASE" > "$LANE/score_$ARM.log" 2>&1 ) &
done
wait
log "queue done"
