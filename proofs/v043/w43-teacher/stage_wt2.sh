#!/usr/bin/env bash
# w43-teacher: stage the WT2 original CPU-WRF 72 h references (219 wrfout) for the batch1 WN3 issues, one GuardedRead session per
# domain, up to 3 tries per domain in a fresh stage dir (CIFS file_identity_changed refusals, G03/G04), then a flat symlink view per case.
# usage: bash analysis/stage_wt2.sh ISSUE...   (run from the worktree root)
set -u
S=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher
PY="taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 timeout 3600 python3"
for i in "$@"; do
  declare -A OKDIR=()
  for d in d01 d02 d03; do
    for t in 1 2 3; do
      while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
      R=$S/stage_ref${i:4:4}/${d}_try$t
      [ -e "$R" ] && continue
      if $PY analysis/stage_ref.py "$i" "$R" "WT2_${i}_try$t" "$d" > "$S/stage_${i}_${d}_try$t.log" 2>&1; then OKDIR[$d]=$R; echo "$i $d try$t OK"; break; fi
      echo "$i $d try$t FAILED: $(tail -1 "$S/stage_${i}_${d}_try$t.log" | cut -c1-200)"
    done
    [ -n "${OKDIR[$d]:-}" ] || { echo "$i $d GAVE UP"; continue 2; }
  done
  V=$S/ref${i:4:4}_view; mkdir -p "$V"
  for d in d01 d02 d03; do
    find "${OKDIR[$d]}" -name "wrfout_${d}_*" -type f | while read -r f; do ln -s "$f" "$V/$(basename "$f")"; done
  done
  echo "$i view $(ls "$V" | wc -l) files"
done
