#!/usr/bin/env bash
# usage: run_files.sh <tree> <tag> -- one pytest per file, each in its own 6G RAM slot; summary to <tag>.txt
set -u
TREE=$1; TAG=$2; OUT=<USER_HOME>/wrf_gpu2_lanes/t-calleraudit/nsslmock
FILES="tests/v040/test_nssl_one_moment_wiring.py tests/v040/test_nssl_nohail22.py tests/v040/test_nssl_nohail22_history.py tests/v040/test_nssl_legacy17.py tests/v040/test_nssl_one_moment.py tests/v040/test_nssl_one_moment_radar.py tests/v040/test_nssl_one_moment_aot_keys.py tests/v040/nssl_cumulus/test_caller.py tests/v034/nssl2mom/test_radar_handoff.py tests/v034/nssl2mom/test_radius_handoff.py tests/v034/nssl2mom/test_wiring.py tests/v040/test_t_calleraudit_land_precip.py"
cd "$TREE" || exit 2
: > "$OUT/$TAG.txt"
for f in $FILES; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET >> "$OUT/$TAG.txt"; exit 3; fi
  log="$OUT/$TAG.$(basename "$f" .py).log"
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 0,1,2,8,9,12 nice -n 19 timeout 2400 \
    "$TREE/scripts/with_ram_slot.sh" --label t-calleraudit --max 6G -- \
    python -m pytest -q -rfE -p no:cacheprovider --basetemp <USER_HOME>/wrf_gpu2_lanes/t-calleraudit/pytest_nssl "$f" > "$log" 2>&1
  rc=$?
  echo "$f rc=$rc $(grep -E '[0-9]+ (passed|failed)' "$log" | tail -1)" >> "$OUT/$TAG.txt"
  grep -E '^(FAILED|ERROR)' "$log" >> "$OUT/$TAG.txt"
done
echo DONE >> "$OUT/$TAG.txt"
