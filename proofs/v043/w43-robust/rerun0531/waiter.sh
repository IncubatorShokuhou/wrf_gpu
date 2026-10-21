#!/usr/bin/env bash
# One waiter for the w43-robust GPU slot (manager GPU SLOTS 2026-10-20T21:30:44Z): REAL UTC 2026-10-21 08:00:00Z–08:28:00Z.
# Waits on real_utc.sh, then runs the WHOLE locked chain (lock wait ≤ LOCK_BY, gpu_rerun0531.sh with its own slot backstop,
# with_gpu_lock teardown) under one cap ending 10 s before the slot end (x0.9 against the system-clock slew). TERM to
# with_gpu_lock is forwarded to the launcher, whose trap posts GPU END; the lock is released when the chain has exited.
set -uo pipefail
RU=${GA_REAL_UTC:-<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh}
LOCK=${GA_LOCK:-<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh}
LAUNCH=${GA_LAUNCH:-<DATA_ROOT>/wrf_gpu2/worktrees/w43-robust/analysis/rerun0531/gpu_rerun0531.sh}
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
START=${GA_START_UTC:-2026-10-21T08:00:00Z}; END=${GA_END_UTC:-2026-10-21T08:28:00Z}; LOCK_BY=${GA_LOCK_BY:-2026-10-21T08:03:00Z}
S=$(date -u -d "$START" +%s); E=$(date -u -d "$END" +%s); L=$(date -u -d "$LOCK_BY" +%s)
real() { local e; e=$("$RU" 2>/dev/null); [[ $e =~ ^[0-9]+$ ]] && echo "$e" || echo 0; }
post() { printf '%s w43-robust: %s (real UTC)\n' "$(date -u -d @"$(real)" +%FT%TZ)" "$*" >> "$MAILBOX"; }
until [ "$(real)" -ge "$((S + 5))" ]; do sleep 20; done
NOW=$(real); WAIT=$(( L - NOW )); CAP=$(( (E - NOW - 10) * 9 / 10 ))
(( NOW > 0 && WAIT >= 1 && CAP > 0 )) || { post "GPU NOT STARTED: lock deadline $LOCK_BY passed (real now $(date -u -d @"$NOW" +%TZ))"; exit 6; }
timeout --signal=TERM --kill-after=15 "$CAP" "$LOCK" --label w43-robust --timeout "$WAIT" -- \
  env GA_START_UTC="$START" GA_END_UTC="$END" bash "$LAUNCH"; r=$?
(( r == 124 || r == 137 )) && post "GPU chain stopped by waiter cap rc=$r (cap ${CAP} s)"
exit "$r"
