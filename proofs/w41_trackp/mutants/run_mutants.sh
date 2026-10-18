#!/usr/bin/env bash
# Run every Track P mutant against tests/w41_trackp; a kill = >= 1 failed test with an assertion/DID-NOT-RAISE
# message and 0 INFRASTRUCTURE errors (E202: missing file/module/permission/fixture). Errors raised by product
# code under the mutant (e.g. the basis self-check ValueError inside a fixture) are reported, not disqualifying.
# usage: run_mutants.sh <src_root> <out_dir>      re-score existing logs: run_mutants.sh --classify <out_dir>
set -u
classify() {  # <mutant> <log>
  local m=$1 log=$2 failed errors genuine infra verdict
  failed=$(grep -Eo '[0-9]+ failed' "$log" | head -1 | cut -d' ' -f1); errors=$(grep -Eo '[0-9]+ error' "$log" | head -1 | cut -d' ' -f1)
  genuine=$(grep -cE '^E +(AssertionError|Failed: DID NOT RAISE|assert )' "$log")
  infra=$(grep -cE '^E +(FileNotFoundError|ModuleNotFoundError|ImportError|PermissionError|OSError)|fixture .* not found' "$log")
  verdict=SURVIVED; [ "${failed:-0}" -gt 0 ] && [ "$infra" -eq 0 ] && [ "$genuine" -gt 0 ] && verdict=KILLED
  echo "$m failed=${failed:-0} errors=${errors:-0} infra_errors=$infra genuine_lines=$genuine $verdict"
}
if [ "$1" = --classify ]; then for log in "$2"/M*.log; do classify "$(basename "$log" .log)" "$log"; done; exit 0; fi
ROOT=$(cd "$1" && pwd); OUT=$2; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
ALL="M1_apply_drops_energy M2_phenology_ignores_lai_tables M3_restart_check_noop M4_aux_hashes_values
     M5_eki_innovation_sign M6_clock_base_drops_controls M7_bounds_unchecked M8_expand_rounds_twice
     M9_basis_preflight_noop M10_protected_ignores_symlinks M11_session_covers_observations_only
     M12_basis_open_backstop_removed"
for m in ${MUTANTS:-$ALL}; do  # MUTANTS="..." runs a subset
  ( cd "$ROOT" && TRACKP_MUTANT=$m PYTHONPATH="$HERE" GPUWRF_NOAHMP_NATIVE_REAL=1 JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    python -m pytest -q -p trackp_mutant_plugin tests/w41_trackp -k "not twin" --basetemp "$OUT/pt_$m" -p no:cacheprovider \
    > "$OUT/$m.log" 2>&1 )
  classify "$m" "$OUT/$m.log"
  rm -rf "$OUT/pt_$m"
done
