#!/bin/bash
# One G2 arm of the D6 nest gate (lane w41-nestqual), run by launch.sh inside its own systemd --user scope, a RAM slot,
# nice 19 and one granted core: (1) stage_gpu.py copies the arm's GPU frames 0-24 h + run receipts + live namelist +
# rd-s6 manifest in ONE sealed guarded_read session (identity/rc failure -> stop, not scored); (2) the merged bounded
# scorer scripts/wn3_score.py --sequential-domains (frozen manifest, d01-d03, 25 frames) under guard_run.py;
# (3) gate_summary.py -> NEST_GATE.json; one twin_status line + one MAILBOX line (ALISIOS reads + verdict).
# usage: worker.sh <tree> <arm> <tag>      (tree = immutable export of the lane commit; cores/scope set by launch.sh)
set -u
TREE=$1; ARM=$2; TAG=$3; ISSUE=20260707_18z; HOURS=24
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; SCR=<DATA_ROOT>/wrf_gpu2/scratch/w41-nestqual; CASE=$SCR/cases/${ISSUE}_$ARM
MAILBOX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; T=$TREE/proofs/w41_nestqual
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$TREE/src PYTHONDONTWRITEBYTECODE=1 NESTQUAL_MANIFEST_SHA=cfe5648d3d61ddd7
log(){ echo "$(date -u +%FT%TZ) worker $ARM $*" | tee -a "$LANE/twin_status.log"; }
mb(){ echo "$(date -u +%FT%TZ) w41-nestqual: $*" >> "$MAILBOX"; }
quiet(){ while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
log "start core=$(taskset -pc $$ | awk -F': ' '{print $2}') tree=$TREE tag=$TAG pid=$$"
[ -s "$CASE/cpu_run.json" ] || { log "no sealed CPU twin"; exit 2; }
[ -s "$CASE/NEST_GATE.json" ] && { log "NEST_GATE.json exists, nothing to do"; exit 0; }
RD=$LANE/guard/gpu_${ISSUE}_${ARM}_$TAG
if [ ! -s "$CASE/GPU_STAGE.json" ]; then
  quiet; if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  timeout 3600 python3 "$T/stage_gpu.py" "$CASE" "$SCR/stage_gpu_${ISSUE}_${ARM}_$TAG" "$RD" $HOURS "$TAG" \
    > "$LANE/stage_gpu_${ARM}_$TAG.log" 2>&1
  R=$?; log "GPU staged rc=$R $(tail -1 "$LANE/stage_gpu_${ARM}_$TAG.log" | cut -c1-400)"
  if [ $R -ne 0 ]; then
    mb "ALISIOS READS G2 $ISSUE $ARM/$TAG (one guarded_read session, receipts $RD): staging rc=$R -> identity or GPU rc NOT OK, arm NOT scored (log $LANE/stage_gpu_${ARM}_$TAG.log)"
    exit 1
  fi
fi
OUT=$CASE/wn3_$TAG
[ -e "$OUT" ] && { log "score dir $OUT exists (fresh dir per attempt required)"; exit 2; }
quiet; if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
timeout 5400 python3 "$T/guard_run.py" "$LANE/guard/score_${ARM}_$TAG" "$TREE/scripts/wn3_score.py" \
  --cpu-dir "$CASE/cpu" --gpu-dir "$CASE/gpu" --out "$OUT" --hours $HOURS --domains d01 d02 d03 --sequential-domains \
  > "$LANE/score_${ARM}_$TAG.log" 2>&1
R=$?
python3 "$T/gate_summary.py" "$CASE" "$OUT" > "$CASE/gate_summary_$TAG.log" 2>&1
G=$(tail -1 "$CASE/gate_summary_$TAG.log" | cut -c1-400)
log "scored wn3_score rc=$R gate: $G"
S=$(python3 -c "import json,sys;s=json.load(open(sys.argv[1]));o=s['input_identity']['output_manifest'];print(f\"{len(s['files'])} files, frames {s['frames']}, manifest matched {o['matched'] if o else None}/{sum(1 for n in s['files'] if n.startswith('wrfout_'))} sha {o['sha256'][:12] if o else None}, spy refused {len(s['spy_refused'])}\")" "$CASE/GPU_STAGE.json" 2>&1 | cut -c1-300)
mb "ALISIOS READS G2 $ISSUE $ARM/$TAG: one guarded_read session ($S; receipts $RD) + scorer guard session (frozen twin-threshold metadata hashed only; receipts $LANE/guard/score_${ARM}_$TAG). D6 24 h vs CPU twin (wn3_score rc=$R): $G"
