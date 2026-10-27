#!/usr/bin/env bash
# w44-rc v0.4.4 CPU follow-up of the RW3 entry: RW3 72 h D6 + output integrity of N72 vs the ORIGINAL CPU-WRF (staged
# read-only through the ALISIOS guard by stage_cpu72.sh), frozen-A3 floor reading, run-length prefix bytes N72 vs A3, final
# RW1-RW4 table; then deletes the receipted N72 wrfouts (content-verified) and the CPU-WRF stage by exact path.
# usage: followup_rw3.sh MANIFEST SHORT_RUN_DIR RW3_RUN_DIR    (ONE RAM slot <= 12G; v0.4.2 measured D6 72 h: 22.4 min,
#        2.17 GB RSS [M w42-bundle042 d6_N72.time]); needs $L/receipts/cpu72_<pin12>/STAGED_INDEX.json (stage_cpu72.sh).
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=${1:?manifest} SRUN=${2:?short run} RRUN=${3:?rw3 run}
F=$W44R_TOOLS/followup
if [[ -z ${W44R_FU_INNER:-} ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; then echo "WAIT: QUIET or GPU window"; exit 3; fi
  exec env W44R_FU_INNER=1 taskset -c 8,9,12 nice -n 19 bash "$W44R_RAM_SLOT" --label w44-rc-fu-rw3 --max 12G --timeout 3600 -- bash "$0" "$MAN" "$SRUN" "$RRUN"
fi
eval "$("$W44R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"])' "$MAN")"
OUT=$W44R_LANE/receipts/followup_rw3_${PIN:0:12}; mkdir -p "$OUT"; W44R_LOG=$OUT/followup.log
IDX=$W44R_LANE/receipts/cpu72_${PIN:0:12}/STAGED_INDEX.json
STAGE=$W44R_SCRATCH/cpu72_${PIN:0:12}
N72=$RRUN/runs/N72
w44_log "START followup_rw3 pin $PIN short $SRUN rw3 $RRUN"
"$W44R_PY" -I "$F/bytes_check.py" "$OUT/BYTES_RW3.json" \
  "PREFIX N72 vs A3=prefix:$N72/wrfout_sha.txt:$SRUN/runs/A3_wn3_0227_3h/wrfout_sha.txt" | tee -a "$W44R_LOG"
if [ ! -s "$OUT/d6_N72/summary.json" ] && [ "$(cat "$N72/rc.txt" 2>/dev/null)" = 0 ] && [ -d "$N72/wrfout" ]; then
  [ -s "$IDX" ] || { w44_log "MISSING $IDX: run stage_cpu72.sh (ALISIOS admission + DISK-PLAN) first"; exit 4; }
  VIEW=$OUT/cpu72_view; mkdir -p "$VIEW"
  "$W44R_PY" -I - "$IDX" "$VIEW" <<'PY' | tee -a "$W44R_LOG"
import json, sys
from pathlib import Path
index, view = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
assert not index["sha_mismatch_vs_f20"] and index["sealed24_checked"] == 75
n = 0
for dom, files in index["index"].items():
    for name, rec in files.items():
        if not (view / name).exists():
            (view / name).symlink_to(rec["staged"])
        n += 1
print("cpu72 view", n, "links")
PY
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  w44_log "D6 N72 start (wn3_score.py --hours 72, sequential domains)"
  (cd "$SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v -o "$OUT/d6_N72.time" \
     "$W44R_PY" scripts/wn3_score.py --cpu-dir "$VIEW" --gpu-dir "$N72/wrfout" --out "$OUT/d6_N72" --hours 72 --sequential-domains) \
     > "$OUT/d6_N72.log" 2>&1
  w44_log "D6 N72 rc=$? $(tail -1 "$OUT/d6_N72.log" | cut -c1-300)"
fi
[ -s "$OUT/d6_N72/summary.json" ] && "$W44R_PY" -I "$W44R_TOOLS/floor72_new.py" "$OUT/d6_N72" "$OUT/FLOOR72_N72.json" > "$OUT/floor72.log" 2>&1
PRE=$(dirname "$MAN")/PREREG.json
RT=$(ls -d "$W44R_LANE"/receipts/rw3_*/rw3t 2>/dev/null | tail -1)
"$W44R_PY" -I "$F/rw_verdict_v044.py" "$OUT/RW_VERDICT_v044.json" "$W44R_LANE/receipts/followup_short_${PIN:0:12}/SHORT_VERDICT.json" \
  "$RT/RW3T_VERDICT.json" "$RT/RW3T_DETAIL24.json" "$RT/RW3T_DETAIL72.json" "$OUT/FLOOR72_N72.json" "$OUT/d6_N72" \
  "$OUT/BYTES_RW3.json" "$PRE" | tee -a "$W44R_LOG"
if [ -s "$OUT/d6_N72/summary.json" ] && [ -s "$OUT/FLOOR72_N72.json" ]; then
  [ -d "$N72/wrfout" ] && "$W44R_PY" -I "$F/delete_listed.py" "$RRUN" "$N72" | tee -a "$W44R_LOG"
  [ -d "$OUT/cpu72_view" ] && find "$OUT/cpu72_view" -maxdepth 1 -type l -name 'wrfout_d0*' -delete && rmdir "$OUT/cpu72_view"
  w44_log "CPU-WRF stage retained: $STAGE (reference cleanup belongs to the manager)"
fi
w44_log "DONE followup_rw3; <DATA_ROOT> free $(w44_free_gib <DATA_ROOT>) GiB"
