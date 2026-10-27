#!/usr/bin/env bash
# w44-gint: 0218 ERA5 3-nest CPU-WRF 18 h, original 2x2 (= the deleted teacher reference config; byte-identity checked against
# the teacher PRECONDITIONS cpu_sha) then 4x1 (pure decomposition twin); frozen all-frame G-INT 4x1 vs 2x2 (both ours, plain
# chain) + item series for the 0218 items. Launch ONLY a frozen copy (E163).  usage: chain_3n0218.sh CORES
set -uo pipefail
CORES=$1; HERE=$(cd "$(dirname "$0")" && pwd); T=$HERE/../tools; C=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/cpu; O=<USER_HOME>/wrf_gpu2_lanes/w44-gint/gint
P="taskset -c 12,13 nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
real(){ <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh; }
runc(){ local case=$1 secs=$2; local dl; dl=$(date -u -d "@$(( $(real) + secs ))" +%FT%TZ); bash "$HERE/run_cpu.sh" "$C/$case" "$CORES" "$dl" 8G "$case"; }
echo "=== r3n0218 $(date -u +%FT%TZ)"
runc e0218_3n_2x2 14400; echo "run rc=$?"
python3 - "$C/e0218_3n_2x2/cpu" <<'PY' | tee "$O/r3n0218_sha_vs_teacher.txt"
import hashlib, json, sys, glob, os
want = json.load(open("<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score/era5_20240218/PRECONDITIONS.json"))["cpu_sha"]
ok, bad = [], []
for p in sorted(glob.glob(sys.argv[1] + "/wrfout_d0*")):
    n = os.path.basename(p); h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    (ok if want.get(n) == h else bad).append(n)
print(json.dumps({"files": len(ok) + len(bad), "sha_equal_teacher_ref": len(ok), "differ": bad[:6]}))
PY
echo "=== t3n0218 $(date -u +%FT%TZ)"
runc e0218_3n_4x1 14400; echo "run rc=$?"
DISK_FLOOR_G=365 bash "$T/score_gint.sh" t3n0218 "$C/e0218_3n_2x2/cpu" "$C/e0218_3n_4x1/cpu" "$O/t3n0218" > "$O/t3n0218.log" 2>&1; tail -14 "$O/t3n0218.log"
$P "$T/item_maps.py" "$O/t3n0218_items.json" "$C/e0218_3n_2x2/cpu" twin="$C/e0218_3n_4x1/cpu" -- d01:QRAIN:2024-02-19_06 d01:QNRAIN:2024-02-19_06 d01:RAINNC:2024-02-19_06 d03:QCLOUD:2024-02-19_16 > "$O/t3n0218_items.log" 2>&1; tail -4 "$O/t3n0218_items.log" | cut -c1-500
echo "chain3n done $(date -u +%FT%TZ)"
