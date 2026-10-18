#!/usr/bin/env bash
# Original CPU-WRF (same executable/namelist/launch as the Swiss RD11 reference) on a run dir whose MPTABLE.TBL is
# either the original (reference) or edited like a Track P control set (spec_T).
# usage: run_cpuwrf_T.sh <run_dir> <cores e.g. 25,28> [limit_s]   (one rank per listed core)
# PRTE puts the ranks in their own process groups under ONE new session (D30): the timeout signals the session.
set -u
RUN=$1; CORES=$2; LIMIT=${3:-3600}
NP=$(echo "$CORES" | tr ',' '\n' | grep -c .)
cd "$RUN" || exit 2
[ -e /tmp/wrf_gpu2_quiet ] && { echo QUIET; exit 3; }
start=$(date -u +%s.%N)
rm -f "$RUN/launcher.sid"
# E131: read the PID from inside the setsid'd shell (setsid may fork; $! is then not the job)
setsid bash -c "echo \$\$ > $RUN/launcher.sid; exec taskset -c $CORES nice -n 19 <DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun --use-hwthread-cpus --bind-to none -np $NP ./wrf.exe" > "$RUN/mpirun.log" 2>&1 &
for _ in $(seq 50); do [ -s "$RUN/launcher.sid" ] && break; sleep 0.2; done
sid=$(cat "$RUN/launcher.sid")
while kill -0 "$sid" 2>/dev/null; do
  if [ "$(date -u +%s)" -gt "$(( ${start%.*} + LIMIT ))" ]; then echo "TIMEOUT: signalling session $sid"; pkill -TERM -s "$sid"; sleep 20; pkill -KILL -s "$sid"; fi
  sleep 5
done
pgrep -s "$sid" > /dev/null && { echo "leftover processes in session $sid -> TERM"; pkill -TERM -s "$sid"; }
rc=0; grep -q "SUCCESS COMPLETE WRF" "$RUN/rsl.out.0000" 2>/dev/null || rc=1
end=$(date -u +%s.%N)
python3 - "$RUN" "$rc" "$start" "$end" "$CORES" <<'PY'
import json, sys, hashlib
from pathlib import Path
run, rc, t0, t1, cores = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
frames = sorted(p.name for p in Path(run).glob("wrfout_d01_*"))
ok = "SUCCESS COMPLETE WRF" in (Path(run) / "rsl.out.0000").read_text(errors="replace") if (Path(run) / "rsl.out.0000").exists() else False
mpt = Path(run) / "MPTABLE.TBL"
json.dump({"rc": rc, "wrf_success": ok, "wall_s": t1 - t0, "frames": frames, "cores": cores,
           "mptable_sha256": hashlib.sha256(mpt.read_bytes()).hexdigest()},
          open(Path(run) / "receipt.json", "w"), indent=1)
print(json.dumps({"rc": rc, "wrf_success": ok, "frames": len(frames), "wall_s": round(t1 - t0, 1)}))
PY
