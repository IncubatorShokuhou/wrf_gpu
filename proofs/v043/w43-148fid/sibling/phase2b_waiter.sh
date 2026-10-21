#!/usr/bin/env bash
# One-shot waiter for gpu_phase2b.sh (lease 2026-10-21 11:12Z real): waits for REAL UTC >= 15:20:05Z, then runs the frozen launcher once.
export LC_ALL=C
W=<DATA_ROOT>/wrf_gpu2/worktrees/w43-148fid
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
T=$(date -u -d "2026-10-21T15:20:05Z" +%s)
while :; do
  now=$($RU)
  case "$now" in ''|*[!0-9]*) sleep 20; continue;; esac
  [ "$now" -ge "$T" ] && break
  sleep 20
done
echo "waiter fire real $($RU --iso) sys $(date -u +%FT%TZ)"
cd "$W" && bash analysis/sibling/gpu_phase2b.sh
echo "launcher rc=$? real $($RU --iso)"
