#!/usr/bin/env bash
# w44-gint root-only CPU-WRF chain (sequential, one 4-rank run at a time on CORES; run_cpu.sh guards: E219 env, exe sha,
# libcuda per rank, deadline, RAM slot, <DATA_ROOT> floor 365). Launch ONLY a frozen copy (E163).
# usage: chain_d01.sh CORES STEP [STEP ...]   STEP in: t0502 | r0218 | t0218 | r0502
#   t0502: c0502_d01_4x1 72 h  -> frozen G-INT d01 (in place, cached) vs WT2 ref 0502 + digests vs ref
#   r0218: e0218_d01_2x2 18 h  -> sha vs teacher PRECONDITIONS cpu_sha (deleted ERA5 ref) + digests
#   t0218: e0218_d01_4x1 18 h  -> frozen G-INT d01 vs r0218 (both ours, plain chain) + digests vs r0218
#   r0502: c0502_d01_2x2 72 h  -> digests vs WT2 ref 0502 d01 (in place)
set -uo pipefail
CORES=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd); T=$HERE/../tools; C=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/cpu; O=<USER_HOME>/wrf_gpu2_lanes/w44-gint/gint
A=$(awk '{sub(/^A=/,"",$1); print $1}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt); S=$(awk '{print $2}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt)
P="taskset -c 12,13 nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
real(){ <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh; }
runc(){ local case=$1 secs=$2; local dl; dl=$(date -u -d "@$(( $(real) + secs ))" +%FT%TZ)
  bash "$HERE/run_cpu.sh" "$C/$case" "$CORES" "$dl" 6G "$case"; }
digest_ref(){ local issue=$1 out=$2; $P "$T/guarded_run.py" "$A" "$S" "DG_${issue}_$(date -u +%H%M%S)" "wg_${issue}_18z_a1/run/run/wrfout_d01_" -- \
  "$T/digests.py" "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_${issue}_18z_a1/run/run" d01 "$out"; }
for st in "$@"; do
  echo "=== $st $(date -u +%FT%TZ)"
  case $st in
    t0502) runc c0502_d01_4x1 5400 || { echo "run failed"; continue; }
      bash "$T/gint_inplace.sh" t0502d01 20260502 "$C/c0502_d01_4x1/cpu" "$O/t0502d01" d01 > "$O/t0502d01.log" 2>&1; tail -12 "$O/t0502d01.log"
      [ -s "$O/ref0502_d01_digest.json" ] || digest_ref 20260502 "$O/ref0502_d01_digest.json"
      $P "$T/digests.py" "$C/c0502_d01_4x1/cpu" d01 "$O/t0502d01_digest.json"
      $P "$T/digests.py" --compare "$O/ref0502_d01_digest.json" "$O/t0502d01_digest.json" | tee "$O/t0502d01_vs_ref.txt";;
    r0218) runc e0218_d01_2x2 3600 || { echo "run failed"; continue; }
      python3 - "$C/e0218_d01_2x2/cpu" <<'EOF' | tee "$O/r0218_sha_vs_teacher.txt"
import hashlib, json, sys, glob, os
want = json.load(open("<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score/era5_20240218/PRECONDITIONS.json"))["cpu_sha"]
ok = bad = 0; badl = []
for p in sorted(glob.glob(sys.argv[1] + "/wrfout_d01_*")):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest(); n = os.path.basename(p)
    if want.get(n) == h: ok += 1
    else: bad += 1; badl.append(n)
print(json.dumps({"d01_files": ok + bad, "sha_equal_teacher_ref": ok, "differ": badl[:5]}))
EOF
      $P "$T/digests.py" "$C/e0218_d01_2x2/cpu" d01 "$O/r0218_digest.json";;
    t0218) runc e0218_d01_4x1 3600 || { echo "run failed"; continue; }
      mkdir -p "$O/t0218d01"
      $P "$T/../../v043/w43-teacher/frozen/integrity_all/scripts/wn3_integrity.py" --cpu-dir "$C/e0218_d01_2x2/cpu" --gpu-dir "$C/e0218_d01_4x1/cpu" \
        --out "$O/t0218d01/d01.json" --domains d01 --all-frames | tee "$O/t0218d01/d01.log"
      $P "$T/item_maps.py" "$O/t0218d01/items.json" "$C/e0218_d01_2x2/cpu" twin="$C/e0218_d01_4x1/cpu" -- d01:QRAIN:2024-02-19_06 d01:QNRAIN:2024-02-19_06 d01:RAINNC:2024-02-19_06 > "$O/t0218d01/items.log" 2>&1; tail -3 "$O/t0218d01/items.log" | cut -c1-500
      $P "$T/digests.py" "$C/e0218_d01_4x1/cpu" d01 "$O/t0218d01_digest.json"
      $P "$T/digests.py" --compare "$O/r0218_digest.json" "$O/t0218d01_digest.json" | tee "$O/t0218d01_vs_r0218.txt";;
    r0502) runc c0502_d01_2x2 5400 || { echo "run failed"; continue; }
      [ -s "$O/ref0502_d01_digest.json" ] || digest_ref 20260502 "$O/ref0502_d01_digest.json"
      $P "$T/digests.py" "$C/c0502_d01_2x2/cpu" d01 "$O/r0502d01_digest.json"
      $P "$T/digests.py" --compare "$O/ref0502_d01_digest.json" "$O/r0502d01_digest.json" | tee "$O/r0502d01_vs_ref.txt";;
    *) echo "unknown step $st";;
  esac
done
echo "chain done $(date -u +%FT%TZ)"
