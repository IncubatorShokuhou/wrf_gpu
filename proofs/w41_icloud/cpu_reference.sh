#!/usr/bin/env bash
# Original CPU-WRF 4.7.1 reference for the WN3 0227 3-nest icloud composite arm (w41-icloud).
# Executable: gen2 dmpar build (the WN3 reference executable; its cal_cldfra3/RSLF/RSIF/caller
# source is byte-identical to the pristine oracle source). Inputs + namelist: CASE/input (model
# input files only, no observations). Output: CASE/cpu, sealed CASE/CPU_OUTPUTS.json and
# CASE/cpu_run.json in the layout proofs/v040/final_validation/score.py reads.
# 4 MPI ranks, one per granted physical core (CORES, default lane cores 8,9,12,13), nice 19.
# Wrap in scripts/with_ram_slot.sh (5.4 GB).
# usage: CORES=0,1,2,8 bash cpu_reference.sh CASE
set -uo pipefail
CASE=$(realpath "${1:?case root}")
BUILD=<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar
WENV=<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build
RUN=$CASE/cpu
CORES=${CORES:-8,9,12,13}
# Wave rule (manager 17:15Z, alisios_mgr C1745): ALISIOS model inputs are opened only inside alisios_guard; refuse any
# input whose realpath is under an ALISIOS root (stage a guarded copy outside the roots first, alisios_guard.facade_stage).
for f in "$CASE"/input/*; do
  case "$(realpath "$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/src/alisios/*|<DATA_ROOT>/server/work/alisios/*)
    echo "refusing ALISIOS-root input $f -> $(realpath "$f"): stage it via alisios_guard first"; exit 2;; esac
done
NR=$(echo "$CORES" | tr ',' '\n' | wc -l)
[ -e "$RUN" ] && { echo "refusing existing $RUN"; exit 2; }
mkdir -p "$RUN"
for f in "$BUILD"/run/*; do ln -s "$f" "$RUN/$(basename "$f")"; done
rm -f "$RUN/namelist.input" "$RUN/wrf.exe" "$RUN"/wrfinput_d0* "$RUN"/wrfbdy_d01 "$RUN"/wrflowinp_d0*
for f in "$CASE"/input/*; do cp -P "$f" "$RUN/$(basename "$f")"; done
grep -qiE "^ *icloud *= *[123]," "$RUN/namelist.input" || { echo "namelist icloud missing"; exit 2; }
export LD_LIBRARY_PATH="$WENV/lib:${LD_LIBRARY_PATH:-}" PATH="$WENV/bin:$PATH" OMP_NUM_THREADS=1
T0=$(date +%s)
# --bind-to core binds a rank to a whole PHYSICAL core (both SMT siblings, e.g. 0+16), so the ranks
# could roam onto another lane's sibling cores: pin each rank to exactly one granted logical CPU.
# (--cpu-set conflicts with --bind-to none in this PRRTE: "PE=<list> mapping ... cannot be combined" -> confine
# prterun + ranks to the granted CPUs with taskset instead, as w41-nestqual run_twin.sh; then one CPU per rank below.)
( cd "$RUN" && exec taskset -c "$CORES" nice -n 19 "$WENV/bin/mpirun" --use-hwthread-cpus --bind-to none -np "$NR" \
    "$BUILD/bin/wrf" > wrf.log 2>&1 ) &
MPI=$!
IFS=, read -ra CPUS <<< "$CORES"
# Ranks = processes whose EXECUTABLE is the wrf binary (prterun's argv also ends in .../bin/wrf and shares the cwd).
WRFEXE=$(readlink -f "$BUILD/bin/wrf")
for _ in $(seq 120); do
  mapfile -t RANKS < <(for p in $(pgrep -f "$BUILD/bin/wrf\$"); do
    [ "$(readlink /proc/$p/cwd)" = "$RUN" ] && [ "$(readlink /proc/$p/exe)" = "$WRFEXE" ] && echo "$p"; done | sort -n)
  [ "${#RANKS[@]}" -eq "$NR" ] && break; sleep 1
done
if [ "${#RANKS[@]}" -eq "$NR" ]; then
  for i in "${!RANKS[@]}"; do taskset -a -cp "${CPUS[$i]}" "${RANKS[$i]}" > /dev/null; done
else  # never die here: that would orphan the ranks outside their RAM slot without a receipt
  echo "WARNING: found ${#RANKS[@]} ranks, expected $NR: per-rank pinning skipped (ranks stay inside $CORES)"
fi
for p in "${RANKS[@]}"; do echo "rank pid $p cpus $(taskset -cp "$p" | awk -F': ' '{print $2}')"; done | tee "$RUN/rank_affinity.txt"
wait "$MPI"
RC=$?
T1=$(date +%s)
tail -2 "$RUN/rsl.error.0000"
NR=$NR CORES=$CORES JAX_PLATFORMS=cpu python3 - "$CASE" "$BUILD/bin/wrf" "$RC" "$((T1-T0))" <<'EOF'
import hashlib, json, sys
from pathlib import Path
from netCDF4 import Dataset
case, exe, rc, wall = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = []
for f in sorted((case / "cpu").glob("wrfout_d0*")):
    with Dataset(f) as ds:
        times = [b"".join(t).decode() for t in ds["Times"][:]]
    rows.append({"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f), "times": times})
(case / "CPU_OUTPUTS.json").write_text(json.dumps(rows, indent=1) + "\n")
inputs = {p.name: sha(p) for p in sorted((case / "input").iterdir())}
receipt = {"schema": "w41-icloud-original-cpu-reference-v1", "rc": rc, "wall_s": wall, "wrfout_files": len(rows),
           "executable": str(exe), "executable_sha256": sha(exe), "ranks": int(__import__("os").environ["NR"]), "cores": __import__("os").environ["CORES"],
           "input_sha256": inputs, "output_manifest_sha256": sha(case / "CPU_OUTPUTS.json")}
(case / "cpu_run.json").write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps({k: receipt[k] for k in ("rc", "wall_s", "wrfout_files")}))
EOF
