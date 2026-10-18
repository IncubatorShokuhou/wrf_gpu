#!/usr/bin/env bash
# CPU proof 2 (lane w41-da-cycle): the one-process loop with all seams/arms, Swiss d01 (examples/switzerland_d01),
# 2 members (m00 = example IC, m01 = interior T noise) x 2 windows x 3 h, `gpuwrf da cycle` on a gpuwrf.da.v1 file:
#   zero_restart  zero update, restart seam (first restart arm: runs the shared window 0)
#   letkf_file    LETKF of a synthetic sonde (valid 03Z, T +1 K), file seam
#   letkf_restart the same LETKF analysis, restart seam (own checkpointed window 0)
# Order: the first-window cases are written up front by the loop's own first-window step (initial_cases), the
# continuous separate run cold-compiles the step into the private cache, then the loop and the c01 separate run
# load that blob (a cold first member in-process plus a blob-loaded second member keeps two CPU executables
# resident: proof-2 attempt 12:21Z was OOM-killed at 12.5 GB in its 12G slot).
# Gates: (1) zero_restart stream of m00 == continuous 6 h separate `gpuwrf run` of its case, all 7 frames
# byte-identical (= window 0 in-process == separate, and the zero-increment restart seam == continuous);
# (2) letkf_file window 1 of m01 in-process == separate `gpuwrf run` of that case (byte-identical);
# (3) zero injections change no cell, LETKF injections change u/v/theta/qv; every member run PIPELINE_GREEN.
# Usage: cpu_proof2.sh MEMBER_CASES_DIR OUT_DIR   (MEMBER_CASES_DIR/m0{0,1}/wrfinput_d01)
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TREE=$(cd -- "$HERE/../.." && pwd)
CASES=$(realpath -- "$1"); OUT=$2
[[ -e $OUT ]] && { echo "REFUSED: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(realpath -- "$OUT")
export JAX_PLATFORMS=cpu PYTHONPATH="$TREE/src" PYTHONDONTWRITEBYTECODE=1
# One private compile cache for the whole proof (as the GPU arm's session cache): the first member's cold compile
# writes the step blob, later members/windows and the separate runs load it. Deleted at the end.
export GPUWRF_JAX_CACHE=1 GPUWRF_JAX_CACHE_DIR="$OUT/jaxcache" JAX_COMPILATION_CACHE_DIR="$OUT/jaxcache"
SWISS=$TREE/examples/switzerland_d01
python3 "$HERE/make_sonde.py" "$SWISS/wrfinput_d01" 2023-01-15_03:00:00 1.0 "$OUT/sonde_03z.txt"
cat > "$OUT/da.json" <<JSON
{"schema": "gpuwrf.da.v1", "description": "w41-da-cycle CPU proof 2 (synthetic sonde, Swiss d01)",
 "t_a": "2023-01-15T03:00:00Z", "template_wrfinput": "$SWISS/wrfinput_d01",
 "ensemble": {"kind": "members", "member_glob": "$OUT/unused/*/wrfout_d01_*"},
 "obs": {"sources": [{"format": "igra2", "path": "$OUT/sonde_03z.txt", "station": "SWZ00000001"}],
         "sonde_superob": false},
 "params": {"workers": 2},
 "experiments": [{"name": "A", "assimilate": ["sonde"]}],
 "cycle": {"source_case": "$SWISS", "t0": "2023-01-15T00:00:00Z", "window_hours": 3, "cycles": 2,
   "arms": {"zero_restart": {"kind": "zero_update", "seam": "restart"},
            "letkf_file": {"kind": "letkf"},
            "letkf_restart": {"kind": "letkf", "seam": "restart"}},
   "initial_cases": {"m00": "$OUT/cases0/m00", "m01": "$OUT/cases0/m01"}, "forcing_hours": 6}}
JSON
cat > "$OUT/cases0_spec.json" <<JSON
{"source_case": "$SWISS", "t_a": "2023-01-15_00:00:00", "hours": 6, "out": "$OUT/cases0",
 "members": {"m00": "$CASES/m00/wrfinput_d01", "m01": "$CASES/m01/wrfinput_d01"}}
JSON
python3 -m gpuwrf.assimilation.letkf.cycle_loop first-window "$OUT/cases0_spec.json" > "$OUT/cases0.log" 2>&1 || exit 2
python3 -m gpuwrf da cycle --config "$OUT/da.json" --out "$OUT/loop" --dry-run > "$OUT/plan.json" || exit 2
separate() {  # tag case_rel hours
  local s=$SECONDS
  /usr/bin/time -v python3 -m gpuwrf run --input-dir "$OUT/$2" --output-dir "$OUT/separate/$1" \
    --scratch-dir "$OUT/separate/$1.scratch" --domain d01 --max-dom 1 --hours "$3" \
    > "$OUT/separate_$1.stdout" 2> "$OUT/separate_$1.stderr"
  echo "separate $1 rc=$? wall=$((SECONDS-s))s" | tee -a "$OUT/separate.rc"
}
separate continuous_m00 cases0/m00 6
start=$SECONDS
/usr/bin/time -v python3 -m gpuwrf da cycle --config "$OUT/da.json" --out "$OUT/loop" \
  > "$OUT/loop.stdout" 2> "$OUT/loop.stderr"
echo "loop rc=$? wall=$((SECONDS-start))s" | tee "$OUT/loop.rc"
separate c01_letkf_file_m01 loop/c01/letkf_file/analysis_work/analysis/members_A/cases/m01 3
python3 - "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
out = Path(sys.argv[1]); loop = out / "loop"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def pair(a_dir, b_dir, n_expected):
    b = sorted(p.name for p in b_dir.glob("wrfout_d01_*"))
    files = {n: {"in_process": sha(a_dir / n) if (a_dir / n).is_file() else None, "separate": sha(b_dir / n)} for n in b}
    return {"identical": len(b) == n_expected and all(v["in_process"] == v["separate"] for v in files.values()),
            "files": files}
pairs = {"zero_restart_m00_vs_continuous": pair(loop / "streams/zero_restart/m00", out / "separate/continuous_m00", 7),
         "letkf_file_c01_m01_vs_separate": pair(loop / "c01/letkf_file/fc/m01", out / "separate/c01_letkf_file_m01", 4)}
inj = {}
for arm in ("zero_restart", "letkf_restart"):
    rec = json.loads((loop / "c01" / arm / "injection_receipt.json").read_text())
    inj[arm] = {m: {leaf: v["changed_cells"] for leaf, v in r["leaves"].items()} for m, r in rec.items()}
zero_ok = all(c == 0 for m in inj["zero_restart"].values() for c in m.values())
letkf_ok = all(m["theta"] > 0 and m["u"] > 0 and m["v"] > 0 and m["qv"] > 0 for m in inj["letkf_restart"].values())
receipt = json.loads((loop / "CYCLE_RECEIPT.json").read_text())
green = all(r["verdict"] == "PIPELINE_GREEN" for r in receipt["driver"]["runs"])
def rms(a, b, name, t):
    with Dataset(loop / a / f"wrfout_d01_{t}") as da, Dataset(loop / b / f"wrfout_d01_{t}") as db:
        return float(np.sqrt(np.mean((np.asarray(da[name][:], float) - np.asarray(db[name][:], float)) ** 2)))
desc = {}
for t in ("2023-01-15_04:00:00", "2023-01-15_06:00:00"):
    for name in ("T2", "PSFC", "U10", "T"):
        desc[f"{name}@{t[11:13]}Z m00 restart-vs-file seam (same LETKF analysis)"] = rms(
            "streams/letkf_restart/m00", "c01/letkf_file/fc/m00", name, t)
        desc[f"{name}@{t[11:13]}Z m00 letkf-vs-zero (restart seam: DA response)"] = rms(
            "streams/letkf_restart/m00", "streams/zero_restart/m00", name, t)
runs = receipt["driver"]["runs"]
summary = {"verdict": "PASS" if all(p["identical"] for p in pairs.values()) and zero_ok and letkf_ok and green else "FAIL",
           "pairs_identical": {k: v["identical"] for k, v in pairs.items()}, "all_member_runs_green": green,
           "zero_injection_changes_nothing": zero_ok, "letkf_injection_changes_analysed_leaves": letkf_ok,
           "injection_changed_cells": inj, "descriptive_rms": desc, "member_runs": len(runs),
           "member_walls_s": [round(r["wall_seconds"], 1) for r in runs], "loop_wall_s": receipt["wall_seconds"],
           "aot_sources": [sorted({d.get("source") for d in r["aot"].values()}) for r in runs], "pairs": pairs}
(out / "CPU_PROOF2.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=1))
PY
du -sh "$OUT/jaxcache" > "$OUT/jaxcache.size" 2>/dev/null
case $OUT in <USER_HOME>/wrf_gpu2_lanes/w41-da-cycle/*|<DATA_ROOT>/wrf_gpu2/scratch/w41-da-cycle/*) rm -rf -- "$OUT/jaxcache";; esac
