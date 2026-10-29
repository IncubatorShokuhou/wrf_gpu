#!/usr/bin/env bash
# Full frozen-floor scoring via the carried guarded original-CPU adapter. Retain N72 for review.
# Usage: bash followup_rw3.sh MAN SRUN RRUN (no receipt environment required).
# rw3_summary.sh supplies RRUN/rw3t/ENTRY_RECEIPTS.json from the actual entry environment.
set -euo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
if [ -e /tmp/wrf_gpu2_gpu_window ]; then echo GPU_WINDOW; exit 3; fi
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/lib.sh
MAN=${1:?manifest}; SRUN=${2:?short run}; RRUN=${3:?rw3 run}
# Resolve all post-run paths BEFORE opening/scoring the original reference.
RT=$RRUN/rw3t
W45R_RECEIPTS=$("$W45R_PY" -I - "$RT/ENTRY_RECEIPTS.json" "$RRUN" "$MAN" <<'BINDING'
import json,sys
from pathlib import Path
binding,run,manifest=map(Path,sys.argv[1:]);j=json.loads(binding.read_text())
assert j['schema']=='v45-rc.rw3-receipts.v1'
assert Path(j['run_dir']).resolve()==run.resolve(), 'RW3 receipt belongs to another run'
assert Path(j['manifest']).resolve()==manifest.resolve(), 'RW3 receipt belongs to another manifest'
receipts=Path(j['receipts_dir']);assert receipts.is_absolute() and receipts.is_dir()
print(receipts)
BINDING
) || exit 4
export W45R_RECEIPTS
[[ -d $W45R_RECEIPTS ]] || exit 4
for post in RW3T_VERDICT.json RW3T_DETAIL24.json RW3T_DETAIL72.json; do
  [[ -s $RT/$post ]] || { echo "missing RW3 post receipt: $RT/$post"; exit 4; }
done
if [[ ${W45R_FU_INNER:-0} != 1 ]]; then
 exec env JAX_PLATFORMS=cpu W45R_FU_INNER=1 PYTHONDONTWRITEBYTECODE=1 systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- taskset -c 12,28 nice -n 19 ionice -c3 timeout 7200 bash "$0" "$MAN" "$SRUN" "$RRUN"
fi
PIN=$("$W45R_PY" -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["pin"])' "$MAN")
OUT=$W45R_LANE/receipts/followup_rw3_${PIN:0:12}; mkdir -p "$OUT"
IDX=$W45R_LANE/receipts/cpu72_${PIN:0:12}/STAGED_INDEX.json
[[ -s $IDX ]] || { echo "missing guarded CPU index; run followup/stage_cpu72.sh first"; exit 4; }
N72=$RRUN/runs/N72
"$W45R_PY" -I "$W45R_TOOLS/followup/bytes_check.py" "$OUT/BYTES_RW3.json" "PREFIX N72 vs A3=prefix:$N72/wrfout_sha.txt:$SRUN/runs/A3_wn3_0227_3h/wrfout_sha.txt"
set +e
/usr/bin/time -v -o "$OUT/d6_N72.time" "$W45R_PY" -B "$W45R_TOOLS/reference_io.py" score --execute --manifest "$MAN" --index "$IDX" --gpu-dir "$N72/wrfout" --out "$OUT/d6_N72" > "$OUT/d6_N72.log" 2>&1
SCORE_RC=$?
set -e
printf '%s\n' "$SCORE_RC" > "$OUT/d6_N72.rc"
[[ -s $OUT/d6_N72/FLOOR72.json && -s $OUT/d6_N72/summary.json ]] || exit 4
"$W45R_PY" -I "$W45R_TOOLS/followup/rw_verdict_v044.py" "$OUT/RW_VERDICT_v045.json" "$W45R_LANE/receipts/followup_short_${PIN:0:12}/SHORT_VERDICT.json" "$RT/RW3T_VERDICT.json" "$RT/RW3T_DETAIL24.json" "$RT/RW3T_DETAIL72.json" "$OUT/d6_N72/FLOOR72.json" "$OUT/d6_N72" "$OUT/BYTES_RW3.json" "$(dirname "$MAN")/PREREG.json"
echo "N72 retained; frozen-floor raw result and integrity remain explicit; no automatic exception"
