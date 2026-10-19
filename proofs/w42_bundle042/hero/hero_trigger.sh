#!/usr/bin/env bash
# w42-bundle042 HERO trigger (manager 2026-10-19T12:21:49Z option B; v0.4.2 common trigger rule: exact event phrase anchored
# on the sender at line start, only MAILBOX lines appended after arming, /usr/bin/grep).
# usage: setsid bash hero_trigger.sh FROM_LINE      (detached, lane core, nice 19; log $L/hero_trigger.log via the caller)
# Starts ONE scripts/with_gpu_lock.sh --label w42-bundle042 session running hero_session.sh from THIS frozen dir on the first
# MAILBOX line after FROM_LINE matching
#   ^<UTC ts> manager -> w42-bundle042: HERO GO
# and only when ALL hold: the main bundle session is not running (no tools_session_*/gpu_session.sh process of this lane),
# P is sealed (P/sealed/BUNDLE_MANIFEST.json), the lease is valid until HT_END_UTC, no QUIET file (waits), the WN3 inputs
# match hero_cases_sha256.txt. Latest start HT_LATEST_START (default 17:45Z, incl. the lock wait: the session needs ~22 of
# its 30 min and must END by HT_END_UTC 18:15Z) -> else posts 'w42-bundle042: HERO NOT STARTED (...)' (the hero then uses
# RW3 only, manager option A). A manager line '^<ts> manager -> w42-bundle042: HERO (HOLD|STOP)' -> NOT STARTED.
# Session rc 3 (QUIET seen at its start, root untouched) -> retry after 60 s. After the session (lock released): CPU
# hero_summary.py (nice 19, waits while QUIET) -> posts 'w42-bundle042: HERO RESULT <HERO_* key=value ...>'.
# Dry (CPU): HT_DRY=1 + HT_MAILBOX/HT_LEASE + time overrides -> session directly (no lock) with the caller's H_DRY* vars.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=<USER_HOME>/src/wrf_gpu2_wt/w42-bundle042
L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042
P=${H_P:-<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2}
MB=${HT_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
LEASE=${HT_LEASE:-/tmp/wrf_gpu2_run_approvals/w42-bundle042.lease}
FROM=$1
ts() { date -u -d "$1" +%s; }
NB=$(ts "${HT_NOT_BEFORE:-2026-10-19T13:00:00Z}"); NA=$(ts "${HT_LATEST_START:-2026-10-19T17:45:00Z}")
END_UTC=${HT_END_UTC:-2026-10-19T18:15:00Z}; END=$(ts "$END_UTC")
SUM=${HT_SUMMARY_CPU:-12}
post() { printf '%s w42-bundle042: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MB"; }
after() { tail -n +$((FROM + 1)) "$MB"; }
not_started() { echo "$(date -u +%FT%TZ) NOT STARTED: $1"; post "HERO NOT STARTED ($1) -> the hero uses RW3 only (A)"; exit 0; }
[ -x "$REPO/scripts/with_gpu_lock.sh" ] && [ -s "$HERE/hero_session.sh" ] || not_started "trigger tools missing"
(( $(wc -l < "$MB") >= FROM )) || not_started "mailbox shorter than the anchor line $FROM"
echo "$(date -u +%FT%TZ) ARMED from line $FROM, not before $(date -u -d @$NB +%TZ), latest start $(date -u -d @$NA +%TZ), end $END_UTC, tools $HERE ($(sha256sum "$0" | cut -c1-12))"
while :; do
  now=$(date +%s)
  # grep reads all input (no -q): pipefail + an early grep exit would SIGPIPE tail and hide the match
  after | /usr/bin/grep -E '^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w42-bundle042: HERO (HOLD|STOP)\b' > /dev/null \
    && not_started "manager HERO HOLD/STOP"
  (( now >= NA )) && not_started "no start before the latest start $(date -u -d @$NA +%TZ)"
  go=; after | /usr/bin/grep -E '^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z manager -> w42-bundle042: HERO GO' > /dev/null && go=1
  main=$(pgrep -f 'tools_session_[0-9a-f]+/(gpu_session|session|rw3t)\.sh' | tr '\n' ' ')
  if [ -n "$go" ] && (( now >= NB )) && [ -z "$main" ] && [ ! -e /tmp/wrf_gpu2_quiet ]; then
    [ -s "$P/sealed/BUNDLE_MANIFEST.json" ] || not_started "P not sealed (main session incomplete)"
    exp=$(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
    [ -n "$exp" ] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) >= END )) || not_started "lease $LEASE missing or expiring before $END_UTC: '$exp'"
    (cd <DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cases && sha256sum -c --quiet "$HERE/hero_cases_sha256.txt") \
      || not_started "WN3 case inputs changed (hero_cases_sha256.txt)"
    out="$L/hero_session.$(date -u +%H%M%S).out"
    echo "$(date -u +%FT%TZ) GO seen (main session not running, P sealed, lease to $exp) -> with_gpu_lock, lock wait <= $(( NA - now )) s"
    if [ -n "${HT_DRY:-}" ]; then
      H_END_UTC="$END_UTC" bash "$HERE/hero_session.sh" > "$out" 2>&1
    else
      cd "$REPO" && "$REPO/scripts/with_gpu_lock.sh" --label w42-bundle042 --timeout $(( NA - now > 60 ? NA - now : 60 )) -- \
        env H_END_UTC="$END_UTC" bash "$HERE/hero_session.sh" > "$out" 2>&1
    fi
    rc=$?; echo "$(date -u +%FT%TZ) session rc=$rc ($out)"
    if (( rc == 3 )); then sleep 60; continue; fi
    (( rc == 124 )) && not_started "GPU lock not acquired before $(date -u -d @$NA +%TZ)"
    after | /usr/bin/grep -E '^20[0-9T:.+Z-]+ w42-bundle042: HERO START' > /dev/null || not_started "session refused before START (rc $rc, $out)"
    root=${H_ROOT:-<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/hero}; so=${HT_SUMMARY_OUT:-$L/hero_summary}
    while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 10; done
    env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 taskset -c "$SUM" nice -n 19 python3 "$HERE/hero_summary.py" "$root" "$so" \
      > "$so.log" 2>&1
    src=$?; line=$(/usr/bin/grep -m1 '^HERO_LINE ' "$so.log" | cut -d' ' -f2-)
    echo "$(date -u +%FT%TZ) summary rc=$src: $line"
    post "HERO RESULT (session rc $rc, summary rc $src; pre-registered rule, $so/HERO_RESULT.json): ${line:-no HERO_LINE}"
    exit "$rc"
  fi
  sleep 15
done
