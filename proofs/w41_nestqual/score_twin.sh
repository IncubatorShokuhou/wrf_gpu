#!/bin/bash
# D6 nest gate of one G2 arm (lane w41-nestqual), CPU only, OUTSIDE any GPU lock: the v0.4.0 release scorer
# proofs/v040/final_validation/score.py (frozen manifest 9df4fab8, every CPU field/frame, header audit on EVERY
# paired frame, raw finite audit of BOTH arms; E63/E184) on d01 d02 d03 over 0-24 h, original CPU-WRF twin vs the
# G2 GPU output staged by stage_gpu.py; then gate_summary.py writes <case>/NEST_GATE.json.
# usage: score_twin.sh <case_dir> [source_tree=this worktree]   (cores: SCORE_CORES, default 16,17,18 = the lane's SMT
# siblings of the twin cores 0,1,2 under the manager CORE PLAN 2026-10-18T08:26:28Z; 8,9,12 are no longer ours)
set -u
CASE=$(readlink -f "$1"); HERE=$(dirname "$(readlink -f "$0")"); SRC=$(readlink -f "${2:-$HERE/../..}")
LANE=<USER_HOME>/wrf_gpu2_lanes/w41-nestqual; SC=${SCORE_CORES:-16,17,18}
taskset -pc "$SC" $$ > /dev/null
[ -s "$CASE/cpu_run.json" ] && [ -s "$CASE/GPU_STAGE.json" ] || { echo "case not ready (cpu_run.json + GPU_STAGE.json)"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$SRC/src PYTHONDONTWRITEBYTECODE=1
cd "$CASE" || exit 2
RC=0
for D in d01 d02 d03; do   # one process per domain (a 3-domain process was OOM-killed at the slot cap, topowind 13:35Z)
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  "$SRC/scripts/with_ram_slot.sh" --label w41-nestqual --max 10G --timeout 3600 -- \
    timeout 3600 taskset -c "$SC" nice -n 19 python "$SRC/proofs/v040/final_validation/score.py" "$SRC" "$CASE" \
    --domains $D --hours 24 > "$CASE/score_$D.log" 2>&1
  R=$?; (( R > RC )) && RC=$R
done
python3 "$HERE/gate_summary.py" "$CASE" > "$CASE/gate_summary.log" 2>&1
echo "$(date -u +%FT%TZ) $(basename "$CASE") score rc=$RC $(tail -1 "$CASE/gate_summary.log")" | tee -a "$LANE/twin_status.log"
