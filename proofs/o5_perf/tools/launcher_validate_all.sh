#!/usr/bin/env bash
# E225 check of EVERY o5-perf GPU launcher invocation (manager 21:25Z/21:28Z): exact production CLI argv through the real
# `gpuwrf.cli run --dry-run` + explicit dt/step/history/bdy alignment (launcher_validate.py, product planning functions)
# + the independent o5-coast check_alignment.py (exact rationals). CPU only, no GPU, no lock.
# Deviation from the arms, recorded per receipt: JAX/XLA cache env -> private scratch (the validator imports JAX on CPU and
# must never write into a sealed bundle cache, E68); output/scratch/proof paths -> validate/<id>/ (fresh, deleted after).
set -u
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
T=<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/proofs/o5_perf/tools
OUTR=<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf/proofs/o5_perf/receipts/launcher_validate
V=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/validate
mkdir -p "$OUTR" "$V/cache"
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
C=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
fails=0
ALIGN=<DATA_ROOT>/wrf_gpu2/scratch/o5-coast/tools_10db38419c6aaa0b851bb017075200a38de0ecf6/proofs/o5-coast/tools/check_alignment.py
sha256sum "$ALIGN" > "$OUTR/check_alignment.sha256"
check() {  # id src_root(extra env...) -- argv
  local id=$1 src=$2; shift 2; local envs=(); while [[ $1 != -- ]]; do envs+=("$1"); shift; done; shift
  env -i PATH="$PATH" HOME="$HOME" LC_ALL=C JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$src/src" \
    GPUWRF_JAX_CACHE=0 GPUWRF_JAX_CACHE_DIR="$V/cache/jax" JAX_COMPILATION_CACHE_DIR="$V/cache/jax" \
    GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$V/cache/autotune" TRITON_CACHE_DIR="$V/cache/triton" "${envs[@]}" \
    taskset -c 24,25,28 nice -n 19 python3 -P "$T/launcher_validate.py" "$OUTR/$id.json" -- "$@" 2>"$OUTR/$id.stderr.txt"
  local rc=$?
  # second, independent explicit guard (manager 21:28Z): o5-coast check_alignment.py on the case namelist + --hours
  local args=("$@") case="" hours="" i; for ((i = 0; i < ${#args[@]}; i++)); do
    [[ ${args[i]} == --input-dir ]] && case=${args[i+1]}; [[ ${args[i]} == --hours ]] && hours=${args[i+1]}; done
  python3 -I "$ALIGN" "$case/namelist.input" "$hours" > "$OUTR/$id.check_alignment.txt" 2>&1
  local arc=$?; echo "$id validate_rc=$rc check_alignment_rc=$arc $(cat "$OUTR/$id.check_alignment.txt")"
  (( rc == 0 && arc == 0 )) || fails=$((fails + 1))
}
B43=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src
B41=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1/src
SNAP=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/snap_cbfix
# cbmini + cbab (o5_s0_arm.sh -> s0_nested_harness.py cli_argv, PROD --hours 3, --max-dom 2): bundle OFF, snapshot OFF/V1/V2
# (snapshot = lane/o5-perf-cbab, SNAP_COMMIT of the launch dir)
h() { echo run --input-dir "$PROD" --output-dir "$V/$1/run/wrfout" --scratch-dir "$V/$1/run/scratch" --max-dom 2 --hours 3; }
check s0_bundle_OFF "$B43" -- $(h s0_bundle_OFF)
check s0_snap_OFF "$SNAP" -- $(h s0_snap_OFF)
check s0_snap_V1 "$SNAP" GPUWRF_STEP_CMDBUF=v1 -- $(h s0_snap_V1)
check s0_snap_V2 "$SNAP" GPUWRF_STEP_CMDBUF=v2 -- $(h s0_snap_V2)
# bisect + cbprobe PROD arm (o5_arm.sh): CLI run ... --domains-from-namelist --hours 3, bundles v0.4.1 / v0.4.3
a() { echo run --input-dir "$PROD" --output-dir "$V/$1/wrfout" --proof-dir "$V/$1/proof" --scratch-dir "$V/$1/scratch" --domains-from-namelist --hours 3; }
check cli_prod_v043 "$B43" -- $(a cli_prod_v043)
check cli_prod_v041 "$B41" -- $(a cli_prod_v041)
# par3 per-case argv (parallel_cases.py: <cli> --input-dir CASE --output-dir OUT/<name>/wrfout <passthrough>)
p() { echo run --input-dir "$C/$2" --output-dir "$V/$1/cases/$2/wrfout" --domains-from-namelist --emit-initial-history --hours "$3"; }
for cs in wn3_0227 wn3_0502 wn3_0614; do
  check "bisect_par3_v043_$cs" "$B43" -- $(p "bisect_par3_v043" "$cs" 3)
  check "bisect_par3_v041_$cs" "$B41" -- $(p "bisect_par3_v041" "$cs" 3)
  check "mps_par3_v043_$cs" "$B43" -- $(p "mps_par3_v043" "$cs" 6)
done
# cbprobe WN3 arm (o5_arm.sh, superseded launcher, still checked)
check cbprobe_wn3_0502 "$B43" -- run --input-dir "$C/wn3_0502" --output-dir "$V/cbprobe/wrfout" --proof-dir "$V/cbprobe/proof" \
  --scratch-dir "$V/cbprobe/scratch" --domains-from-namelist --emit-initial-history --hours 3
echo "FAILS=$fails"
