#!/bin/bash
# GPU staging + D6 nest gate for the sealed CPU twins (lane w41-nestqual), one arm after the other, on manager-granted
# cores only. Starts after rd-s6's cross note TO_WRF_GPU2_G2_0707_OUTPUTS_READY_<date>.md (G2 rerun 19.10 on gpuwrf
# v0.4.1, run dirs g2/runs/<issue>/<arm>/<tag>/). Per arm: stage_gpu.py (one sealed guarded_read session: frames 0-24 h,
# run receipts, live in/namelist.input, rd-s6 SHA manifest) into a FRESH stage dir; an identity/rc failure stops that
# arm before scoring (reported, never scored as a pass); else score_twin.sh (release scorer, per domain, RAM slot 10G)
# + gate_summary.py -> NEST_GATE.json. Every step under QUIET guard, nice 19, inside <cores>; one MAILBOX line per arm
# (ALISIOS reads + verdict).
# usage: stage_and_score.sh <cores> <tag> <arm ...>      e.g. stage_and_score.sh 4,5,6 v041 C2 C3 C4 C7
set -u
CORES=$1; TAG=$2; shift 2
HERE=$(dirname "$(readlink -f "$0")"); TREE=$(readlink -f "$HERE/../.."); ISSUE=20260707_18z; HOURS=24
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; SCR=<DATA_ROOT>/wrf_gpu2/scratch/w41-nestqual; CASES=$SCR/cases
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
taskset -pc "$CORES" $$ > /dev/null || exit 2
log(){ echo "$(date -u +%FT%TZ) stage_and_score $*" | tee -a "$LANE/twin_status.log"; }
quiet(){ local n=0; while [ -e /tmp/wrf_gpu2_quiet ]; do (( n++ == 0 )) && log "QUIET set, waiting"; sleep 30; done; }
log "start cores=$CORES tag=$TAG arms=$*"
for ARM in "$@"; do
  CASE=$CASES/${ISSUE}_$ARM
  [ -s "$CASE/cpu_run.json" ] || { log "$ARM: no sealed CPU twin, skipped"; continue; }
  [ -s "$CASE/NEST_GATE.json" ] && { log "$ARM: NEST_GATE.json exists, skipped"; continue; }
  if [ ! -s "$CASE/GPU_STAGE.json" ]; then
    quiet
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1 \
      timeout 3600 taskset -c "$CORES" nice -n 19 python3 "$HERE/stage_gpu.py" "$CASE" \
      "$SCR/stage_gpu_${ISSUE}_${ARM}_$TAG" "$LANE/guard/gpu_${ISSUE}_${ARM}_$TAG" $HOURS "$TAG" \
      > "$LANE/stage_gpu_${ARM}_$TAG.log" 2>&1
    R=$?; log "$ARM GPU staged rc=$R $(tail -1 "$LANE/stage_gpu_${ARM}_$TAG.log" | cut -c1-300)"
    if [ $R -ne 0 ]; then
      echo "$(date -u +%FT%TZ) w41-nestqual: ALISIOS READS G2 $ISSUE $ARM/$TAG staging rc=$R (identity or GPU rc not OK, NOT scored; log $LANE/stage_gpu_${ARM}_$TAG.log, receipts $LANE/guard/gpu_${ISSUE}_${ARM}_$TAG)" >> "$MAILBOX"
      continue
    fi
  fi
  quiet
  SCORE_CORES=$CORES "$HERE/score_twin.sh" "$CASE" "$TREE" > "$LANE/score_${ARM}_$TAG.log" 2>&1
  R=$?
  V=$(python3 -c "import json,sys;g=json.load(open(sys.argv[1]));print(g['verdict'], {d:v.get('verdict') for d,v in g['domains'].items()}, 'identity', g['input_identity'])" "$CASE/NEST_GATE.json" 2>/dev/null || echo "NO NEST_GATE")
  log "$ARM scored rc=$R $V"
  echo "$(date -u +%FT%TZ) w41-nestqual: ALISIOS READS G2 $ISSUE $ARM/$TAG (wrfout d01-d03 0-24 h + run receipts + in/namelist.input + rd-s6 manifest, one guarded_read session, receipts $LANE/guard/gpu_${ISSUE}_${ARM}_$TAG); D6 vs CPU twin: $V" >> "$MAILBOX"
done
log "done"
