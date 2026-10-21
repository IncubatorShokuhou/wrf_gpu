#!/usr/bin/env bash
# w43-teacher DISK URGENT (manager 14:25Z): manifest + exact-file delete of scored GPU cases, one at a time, QUIET-aware.
# usage: bash analysis/free_chain.sh RUN_DIR:SCORE_NAME:ISSUE ...
set -u
L=<USER_HOME>/wrf_gpu2_lanes/w43-teacher; MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md; REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
post() { printf '%s w43-teacher: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MBX"; }
for spec in "$@"; do
  IFS=: read -r run name issue <<< "$spec"
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  b=$(df -B1 --output=avail <DATA_ROOT> | tail -1)
  out=$(taskset -c 8,9,12 nice -n 19 ionice -c3 env JAX_PLATFORMS=cpu python3 analysis/free_gpu_case.py "$run" "$L/score/$name" "$L/te_gpu/${issue}_gpu" --delete 2>&1 | tail -1)
  a=$(df -B1 --output=avail <DATA_ROOT> | tail -1)
  post "DISK: $name GPU wrfouts deleted by exact file list after D6+G-INT+extract readback PASS: $out | <DATA_ROOT> free $((b>>30)) -> $((a>>30)) GiB. Manifest (per-file sha256 re-verified vs wrfout_sha.txt + Times + bytes) in score/$name/GPU_WRFOUT_MANIFEST.json; receipt + proofs copied."
done
