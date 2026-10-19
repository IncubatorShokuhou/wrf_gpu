#!/usr/bin/env bash
# CPU post-processing of the icloud=0 arm (w42-icloud0), after gpu_arm.sh AND cpu_ref.sh are done:
# (a) frozen-D6 score.py per domain vs the ORIGINAL CPU-WRF icloud=0 reference, one domain at a time (V042 lesson:
#     sequential scoring) in a main RAM slot, (b) icloud0_report.py (CLDFRA = 0 in both arms, cloud-free radiation,
#     unbounded radiation fields: disclosure), (c) small receipts copied to the lane dir (the 9 GB case goes after the
#     critic).
# usage: TREE=<the GPU arm snapshot> bash post.sh [CASE]
set -uo pipefail
TREE=$(realpath "${TREE:?the GPU arm snapshot}")
CASE=$(realpath "${1:-<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/wn3_icloud0}")
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0
[ "$(cat "$L/gpu_arm/source_commit.txt")" = "$(git -C "$TREE" rev-parse HEAD)" ] || { echo "TREE is not the GPU arm tree"; exit 2; }
for f in "$CASE/cpu_run.json" "$CASE/CPU_OUTPUTS.json" "$CASE/gpu_proof/nested_pipeline_run.json"; do
  [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
RS=<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh
PY="env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$TREE/src taskset -c 8,9,12 nice -n 19"
for d in d01 d02 d03; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  $RS --label w42-icloud0-d6 --max 12G --timeout 21600 -- $PY timeout 5400 \
      python "$TREE/proofs/v040/final_validation/score.py" "$TREE" "$CASE" --domains "$d" > "$CASE/score_$d.log" 2>&1
  echo "score $d rc=$? $(tail -1 "$CASE/score_$d.log")"
done
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
$RS --label w42-icloud0-report --max 6G --timeout 21600 -- $PY timeout 1800 \
    python "$HERE/icloud0_report.py" "$CASE" "$CASE/ICLOUD0_report.json" > "$CASE/icloud0_report.log" 2>&1
echo "report rc=$?"; cat "$CASE/icloud0_report.log"
R=$L/wn3_icloud0_receipts
mkdir -p "$R/input" "$R/gpu_proof"
cp "$CASE"/{STAGE.json,CPU_OUTPUTS.json,cpu_run.json,ICLOUD0_report.json,icloud0_report.log} "$R/" 2>/dev/null
cp "$CASE"/D6_d0?.json "$CASE"/compare_d0?.json "$CASE"/score_d0?.log "$R/" 2>/dev/null
cp "$CASE/input/namelist.input" "$R/input/"; cp "$CASE/gpu_proof/nested_pipeline_run.json" "$R/gpu_proof/"
cp "$CASE/cpu/rsl.error.0000" "$R/cpu_rsl.error.0000"
( cd "$CASE/gpu" && sha256sum wrfout_d0* ) > "$R/gpu_wrfout_sha256.txt"
python3 - "$CASE" <<'EOF'
import json, sys
from pathlib import Path
case = Path(sys.argv[1])
print(json.dumps({d: json.loads((case / f"D6_{d}.json").read_text())["verdict"] for d in ("d01", "d02", "d03")
                  if (case / f"D6_{d}.json").exists()}))
EOF
