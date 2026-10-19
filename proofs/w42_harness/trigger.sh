#!/usr/bin/env bash
# w42-harness S1H trigger (manager 2026-10-19T00:37:56Z: SLOT 1, queue on with_gpu_lock not before 13:00Z, hard end 13:45Z,
# lease to 14:00Z, next slot triggers on the exact line 'w42-harness: S1H GPU END').
# usage: trigger.sh TOOLS_DIR TRIG_DIR      (lane core 28, nice 19, in the background; never prompts anyone)
# Waits until now >= H_NOT_BEFORE (default 2026-10-19T13:00:00Z), the lease /tmp/wrf_gpu2_run_approvals/w42-harness.lease
# has expires-utc >= H_END_UTC (default 2026-10-19T13:45:00Z) and no QUIET file exists; then
#   scripts/with_gpu_lock.sh --label w42-harness --timeout <until H_LATEST_START> -- env H_END_UTC=.. bash TOOLS/session.sh
# (the session posts 'S1H GPU START' and on every exit the exact 'S1H GPU END'). Session rc 3 = QUIET seen at its start,
# root untouched -> retry after 60 s. Gives up at H_LATEST_START (default 13:33Z: W0 + one 24 h pair still fit) or on a
# manager line '^<ts> manager -> w42-harness: S1H (HOLD|STOP)' after arming, and then posts 'S1H NOT STARTED <why>'
# followed by the exact 'S1H GPU END' so the next slot is never blocked.
# Dry (CPU): H_TRIGGER_DRY=1 H_MAILBOX=<file> H_LEASE=<file> H_DRY... -> runs the session directly (no lock) with the
# caller's H_* dry variables.
set -uo pipefail
TOOLS=$(readlink -f "$1"); TRIG=$(readlink -f "$2"); mkdir -p "$TRIG"
REPO=<USER_HOME>/src/wrf_gpu2_wt/w42-harness
MAILBOX=${H_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
LEASE=${H_LEASE:-/tmp/wrf_gpu2_run_approvals/w42-harness.lease}
END_UTC=${H_END_UTC:-2026-10-19T13:45:00Z}
NB=$(date -u -d "${H_NOT_BEFORE:-2026-10-19T13:00:00Z}" +%s); NA=$(date -u -d "${H_LATEST_START:-2026-10-19T13:33:00Z}" +%s)
END=$(date -u -d "$END_UTC" +%s)
log() { echo "$(date -u +%FT%TZ) $*" >> "$TRIG/trigger.log"; }
post() { printf '%s w42-harness: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
give_up() { log "give up: $1"; post "S1H NOT STARTED ($1)"
  tail -n +$((FROM + 1)) "$MAILBOX" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ w42-harness: S1H GPU END$' || post "S1H GPU END"; exit 0; }
FROM=$(wc -l < "$MAILBOX")
[ -x "$REPO/scripts/with_gpu_lock.sh" ] && [ -x "$TOOLS/session.sh" ] || { log "missing tools"; give_up "trigger tools missing"; }
log "armed: mailbox line $FROM, not before $(date -u -d @$NB +%FT%TZ), latest start $(date -u -d @$NA +%FT%TZ), hard end $END_UTC, tools $TOOLS"
post "S1H TRIGGER ARMED (slot 1, manager 00:37:56Z): queues $TOOLS/session.sh on with_gpu_lock not before $(date -u -d @$NB +%TZ) (lease + no QUIET), hard end $(date -u -d @$END +%TZ), latest start $(date -u -d @$NA +%TZ); posts exactly 'w42-harness: S1H GPU END' on every exit; '<ts> manager -> w42-harness: S1H HOLD' cancels."
while :; do
  now=$(date +%s)
  if tail -n +$((FROM + 1)) "$MAILBOX" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> w42-harness: S1H (HOLD|STOP)\b'; then
    give_up "manager HOLD/STOP"; fi
  (( now >= NA )) && give_up "no lease/QUIET-free start before $(date -u -d @$NA +%TZ)"
  exp=$(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
  lease_ok=; [ -n "$exp" ] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) >= END )) && lease_ok=1
  if (( now >= NB )) && [ -n "$lease_ok" ] && [ ! -e /tmp/wrf_gpu2_quiet ]; then
    log "conditions met (lease to $exp): queue on with_gpu_lock"
    out="$TRIG/session.$(date -u +%H%M%S).out"
    if [ -n "${H_TRIGGER_DRY:-}" ]; then
      H_END_UTC="$END_UTC" bash "$TOOLS/session.sh" > "$out" 2>&1
    else
      "$REPO/scripts/with_gpu_lock.sh" --label w42-harness --timeout $(( NA - now > 60 ? NA - now : 60 )) -- \
        env H_END_UTC="$END_UTC" bash "$TOOLS/session.sh" > "$out" 2>&1
    fi
    rc=$?; log "session rc=$rc"
    if (( rc == 3 )); then sleep 60; continue; fi
    (( rc == 124 )) && give_up "GPU lock not acquired before $(date -u -d @$NA +%TZ)"
    /usr/bin/grep -q 'S1H GPU START' <(tail -n +$((FROM + 1)) "$MAILBOX") || give_up "session refused before START (rc $rc, $out)"
    exit "$rc"
  fi
  sleep 30
done
