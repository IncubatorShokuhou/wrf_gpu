#!/usr/bin/env bash
# w42-bundle042 GPU trigger (v0.4.2 common rule: exact event phrase anchored on the sender at line start, only lines
# appended after arming, /usr/bin/grep). Starts ONE scripts/with_gpu_lock.sh session running gpu_session.sh from THIS
# frozen dir on the first MAILBOX line after line FROM matching
#   ^<UTC ts> manager -> w42-bundle042: <GO>    (GO = env TRIGGER_GO at arming, default 'GPU GO'; exact word: 'GPU GO' does
#   not match 'GPU GO2' and vice versa -- manager 13:29:16Z re-pin: 'GPU GO2')
# but not before NOT_BEFORE_UTC (default 13:00Z) and only while the lease is valid, no QUIET exists and >= 30 min remain
# before 18:30Z. No GO by GIVE_UP_UTC (default 17:00Z) -> posts 'w42-bundle042: GPU NOT STARTED (...)' and exits.
# usage: setsid bash trigger.sh FROM_LINE [NOT_BEFORE_UTC] [GIVE_UP_UTC]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=<USER_HOME>/src/wrf_gpu2_wt/w42-bundle042
MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
FROM=$1; NB=$(date -u -d "${2:-2026-10-19T13:00:00Z}" +%s); GU=$(date -u -d "${3:-2026-10-19T17:00:00Z}" +%s)
LEASE=/tmp/wrf_gpu2_run_approvals/w42-bundle042.lease
GO=${TRIGGER_GO:-GPU GO}
post() { printf '%s w42-bundle042: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MB"; }
echo "$(date -u +%FT%TZ) ARMED on 'manager -> w42-bundle042: $GO' from line $FROM, not before $(date -u -d @$NB +%TZ), give up $(date -u -d @$GU +%TZ), tools $HERE"
while :; do
  now=$(date +%s)
  # grep reads all input (no -q): pipefail + an early grep exit would SIGPIPE tail and hide the match
  if tail -n +$((FROM + 1)) "$MB" | /usr/bin/grep -E "^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w42-bundle042: ${GO}([^0-9A-Za-z]|\$)" > /dev/null; then
    if (( now >= NB )); then break; fi
  fi
  if (( now >= GU )); then post "GPU NOT STARTED (no 'manager -> w42-bundle042: $GO' after line $FROM by $(date -u -d @$GU +%TZ))"; exit 4; fi
  sleep 20
done
exp=$(awk '/^expires-utc:/{print $2}' "$LEASE" 2>/dev/null); [ -n "$exp" ] && (( $(date -u -d "$exp" +%s) > $(date +%s) + 1800 )) || {
  post "GPU NOT STARTED (lease $LEASE missing or expiring: '$exp')"; exit 4; }
while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 10; done
echo "$(date -u +%FT%TZ) GO seen -> with_gpu_lock (lock wait <= 3600 s)"
cd "$REPO" && exec scripts/with_gpu_lock.sh --label w42-bundle042 --timeout 3600 -- bash "$HERE/gpu_session.sh"
