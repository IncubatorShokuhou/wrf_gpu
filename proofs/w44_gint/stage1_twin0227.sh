#!/usr/bin/env bash
# w44-gint stage 1 (manager 15:46Z OK, 16 GB plan): stage the WT2 CPU-WRF 0227 reference (219 wrfout, ~5.2 GB) through the
# sealed guarded facade (one GuardedRead session per domain, tools/stage.py), frozen all-frame G-INT of the CPU twin0227 (4x1)
# against it (= CPU-vs-CPU G-INT floor), item anatomy, then delete the staged copy (exact literal root, after receipts).
# usage: bash stage1_twin0227.sh ADMISSION_JSON ADMISSION_SHA256
set -uo pipefail
ADM=$1; ASHA=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ST=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/stage/ref0227
VIEW=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/stage/ref0227_view
TWIN=<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu
OUT=<USER_HOME>/wrf_gpu2_lanes/w44-gint/gint
PY="taskset -c ${GINT_CPUS:-8,9,12,13} nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
q(){ while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done; }
free_g(){ df --output=avail -B1G <DATA_ROOT> | tail -1; }
(( $(free_g) >= 371 )) || { echo "refuse: <DATA_ROOT> free $(free_g) GiB < 365 + 6"; exit 4; }
[ -e "$ST" ] && { echo "refusing existing $ST"; exit 2; }
mkdir -p "$OUT"
for d in d01 d02 d03; do
  q; (cd "$HERE/../.." && $PY proofs/w44_gint/tools/stage.py "$ADM" "$ASHA" "$ST" "S1_ref0227_$d" "wg_20260227_18z_a1/run/run/wrfout_${d}_") || { echo "stage $d failed"; exit 5; }
done
mkdir -p "$VIEW"
python3 - "$ST" "$VIEW" <<'EOF'
import json, glob, os, sys
st, view = sys.argv[1:3]
n = 0
for f in glob.glob(f"{st}/STAGE_S1_ref0227_d0*.json"):
    for src, rec in json.load(open(f))["files"].items():
        assert rec["sha256"] == rec["admitted_sha256"] or rec["admitted_sha256"] is None, src
        os.symlink(rec["staged"], os.path.join(view, os.path.basename(src))); n += 1
print("view", n); assert n == 219
EOF
q; bash "$HERE/tools/score_gint.sh" twin0227_vs_ref "$VIEW" "$TWIN" "$OUT/twin0227" > "$OUT/twin0227.log" 2>&1; echo "score rc=$?"; tail -12 "$OUT/twin0227.log"
q; $PY "$HERE/tools/item_maps.py" "$OUT/twin0227_items.json" "$VIEW" twin="$TWIN" -- d01:QNICE:2026-03-01_13 d03:CANICE:2026-03-01_11 \
  d03:QGRAUP:2026-03-02_23 d03:QGRAUP:2026-03-03_00 > "$OUT/twin0227_items.log" 2>&1; echo "items rc=$?"
echo "stage1 done; staged copy kept until the receipts are checked: $ST"
