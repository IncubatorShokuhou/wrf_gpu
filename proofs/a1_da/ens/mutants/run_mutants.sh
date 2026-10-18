#!/bin/bash
# Genuine-kill mutant run (E202): a mutant is KILLED only if pytest exits 1 with an AssertionError/assert failure
# and ZERO errors (setup/collection); the unmutated control must pass. Usage: run_mutants.sh OUT.json
cd "$(dirname "$0")/../../../.." || exit 2
OUT=$1; B=<USER_HOME>/wrf_gpu2_lanes/w41-da-ens/pytest_mut; L=<USER_HOME>/wrf_gpu2_lanes/w41-da-ens/logs/mutants
mkdir -p "$L"
T="tests/a1_da/test_adaptive_inflation.py tests/a1_da/test_a1_perturb.py tests/da/test_ensemble_design.py tests/a1_da/test_osse_gate.py tests/a1_da/test_a1_ensemble.py"
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$PWD/proofs/a1_da/ens/mutants${PYTHONPATH:+:$PYTHONPATH}"
echo "{" > "$OUT"
first=1
for m in CONTROL M1_no_rebalance M2_paths_passthrough M3_band_mix M4_no_coherent M5_levels_as_samples M6_gate_no_completeness M7_config_no_base M8_gate_no_time_check M9_gate_first_row_base M10_inflation_default_width M11_ic_default_width; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  rm -rf "$B"
  if [ "$m" = CONTROL ]; then env -u W41_MUTANT timeout 900 taskset -c 13,29 nice -n 19 python -m pytest -q -p w41mut $T --basetemp "$B" -p no:cacheprovider > "$L/$m.log" 2>&1
  else W41_MUTANT=$m timeout 900 taskset -c 13,29 nice -n 19 python -m pytest -q -p w41mut $T --basetemp "$B" -p no:cacheprovider > "$L/$m.log" 2>&1; fi
  rc=$?
  summary=$(grep -E "passed|failed" "$L/$m.log" | tail -n 1)
  failed_tests=$(grep -E "^FAILED " "$L/$m.log" | sed 's/ - .*//' | tr '\n' ';')
  asserts=$(grep -cE "AssertionError|^E +assert|Mismatched elements" "$L/$m.log")
  errors=$(echo "$summary" | grep -oE "[0-9]+ error" | grep -oE "[0-9]+" || echo 0)
  if [ "$m" = CONTROL ]; then verdict=$([ $rc -eq 0 ] && echo PASS || echo CONTROL_FAILED)
  else verdict=$([ $rc -eq 1 ] && [ "$asserts" -gt 0 ] && [ "${errors:-0}" = 0 ] && echo KILLED || echo SURVIVED_OR_ERROR); fi
  [ $first = 1 ] || echo "," >> "$OUT"; first=0
  printf '"%s": {"rc": %d, "verdict": "%s", "summary": "%s", "assertion_lines": %d, "failed": "%s"}' \
    "$m" "$rc" "$verdict" "$summary" "$asserts" "$failed_tests" >> "$OUT"
  echo "$m rc=$rc $verdict | $summary"
done
echo "}" >> "$OUT"; rm -rf "$B"
