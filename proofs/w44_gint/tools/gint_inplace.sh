#!/usr/bin/env bash
# w44-gint: frozen all-frame G-INT of a candidate wrfout dir (ours) against an ALISIOS WT2 reference read IN PLACE:
# reference read only inside guarded sessions (tools/guarded_run.py, admission A-O5-W44-GINT) into a small read cache, candidate
# cached outside, frozen tools run unchanged on the caches (tools/gint_cached.py, equivalence GI09). No staging copy.
# usage: gint_inplace.sh NAME ISSUE CAND_DIR OUT [DOMAINS=d01,d02,d03]
set -uo pipefail
NAME=$1; ISSUE=$2; CAND=$(readlink -f "$3"); OUT=$4; DOMS=${5:-d01,d02,d03}
HERE=$(cd "$(dirname "$0")" && pwd)
REF=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_${ISSUE}_18z_a1/run/run
A=$(awk '{sub(/^A=/,"",$1); print $1}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt); S=$(awk '{print $2}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt)
C=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/cache/$NAME
P="taskset -c ${GINT_CPUS:-12,13} nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
R4=ECAN,EVC,CANLIQ,CANWAT,FWET,RAINNC,RAINC
q(){ while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done; }
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 2; }
[ -e "$C" ] && { echo "refusing existing $C"; exit 2; }
mkdir -p "$C" "$OUT"; tag=$(date -u +%H%M%S)
refp=; candp=
for d in ${DOMS//,/ }; do
  q; $P "$HERE/guarded_run.py" "$A" "$S" "GC_${NAME}_${d}_$tag" "wg_${ISSUE}_18z_a1/run/run/wrfout_${d}_" -- \
    "$HERE/gint_cached.py" cache "$REF" "$d" "$C/ref_$d.pkl" --full "$R4" || { echo "ref cache $d failed"; exit 5; }
  q; $P "$HERE/gint_cached.py" cache "$CAND" "$d" "$C/cand_$d.pkl" --full "$R4" || { echo "cand cache $d failed"; exit 5; }
  refp=${refp:+$refp,}$C/ref_$d.pkl; candp=${candp:+$candp,}$C/cand_$d.pkl
done
$P "$HERE/gint_cached.py" run "$REF" "$CAND" "$refp" "$candp" "$OUT/integ" "$DOMS" || { echo "run failed"; exit 6; }
if [ -s "$OUT/integ/annex_items.json" ] && [ "$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$OUT/integ/annex_items.json")" != 0 ]; then
  q; $P "$HERE/guarded_run.py" "$A" "$S" "GC_${NAME}_items_$tag" "wg_${ISSUE}_18z_a1/run/run/wrfout_" -- \
    "$HERE/gint_cached.py" cache "$REF" "$DOMS" "$C/ref_items.pkl" --frames-json "$OUT/integ/annex_items.json" || { echo "ref item cache failed"; exit 5; }
  $P "$HERE/gint_cached.py" cache "$CAND" "$DOMS" "$C/cand_items.pkl" --frames-json "$OUT/integ/annex_items.json"
  refp=$refp,$C/ref_items.pkl; candp=$candp,$C/cand_items.pkl
fi
$P "$HERE/gint_cached.py" annex "$REF" "$CAND" "$refp" "$candp" "$OUT/integ" > "$OUT/annex.log" 2>&1; echo "annex rc=$?"
cat "$OUT/integ/classify.log" | cut -c1-600; cat "$OUT/integ/annex_postpass.txt"
du -sh "$C"
