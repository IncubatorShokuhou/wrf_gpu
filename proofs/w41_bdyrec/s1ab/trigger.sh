#!/usr/bin/env bash
# S1AB trigger (manager 2026-10-18T11:47:21Z / 11:47:42Z). usage: trigger.sh OUT     (OUT must not exist yet)
# Queue: after the first MAILBOX line after S1AB_MAILBOX_FROM matching '^<ts> w41-swddirscreen: GPU END' (their B' session
# end; their refusal paths post the same prefix), then a manager lease for S1AB (lease file written after this trigger was
# armed, or naming S1AB; expiry > now + 30 min), no foreign QUIET, P = v0.4.0 bundle BUNDLE_OK (P's own checker) -> GPU START
# post -> ONE with_gpu_lock session (session.sh) -> exactly 'w41-bdyrec: S1AB GPU END ...' -> CPU: P re-check (BUNDLE_OK
# after) + summary.py on the OS/agents cores (JSON only, seconds) -> result line. No Bp end / lease / clean P by
# S1AB_LATEST_UTC (16:30Z), or a refusal before the lock -> exactly 'w41-bdyrec: S1AB NOT STARTED ...'. The C 0125 twin of
# w41-swddirscreen on 8,9,12,13 waits for one of these two lines. A kill posts nothing. Wall-clock waits (E60/E180).
set -uo pipefail
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
L=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec
M=${S1AB_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}     # S1AB_MAILBOX/LEASE/DRY_ARM/QUIET_FILE: dry test only
LEASE=${S1AB_LEASE:-/tmp/wrf_gpu2_run_approvals/w41-bdyrec.lease}
N0=${S1AB_MAILBOX_FROM:?MAILBOX line after which the w41-swddirscreen GPU END line counts}
LATEST=$(date -u -d "${S1AB_LATEST_UTC:-2026-10-18T16:30:00Z}" +%s)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
TLOG=${S1AB_TRIGGER_LOG:-$L/s1ab_trigger.log}
BP_RE='^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z w41-swddirscreen: GPU END'
ARMED=$(date +%s)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$TLOG"; }
post() { echo "$(date -u +%FT%TZ) w41-bdyrec: $*" >> "$M"; log "posted: $*"; }
not_started() { post "S1AB NOT STARTED: $1"; exit 1; }
bundle_ok() { JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 nice -n 19 python3 "$P/tools/compile_bundle.py" check --cache "$P/jax" \
                --manifest "$P/sealed/BUNDLE_MANIFEST.json" 2>&1 | tail -1; }
lease_ok() { local exp; exp=$(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
  [[ -n "$exp" ]] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) > $(date +%s) + 1800 )) \
    && { (( $(stat -c %Y "$LEASE") > ARMED )) || /usr/bin/grep -q 'S1AB' "$LEASE"; }; }
said=
say_once() { [[ "$said" == "$1" ]] && return 0; said=$1; log "$2"; }

[ -e "$OUT" ] && { log "refuse: $OUT exists (fresh dir per attempt, E42)"; exit 2; }
log "armed (tools $HERE): OUT $OUT; after MAILBOX line $N0 /$BP_RE/, lease $LEASE (new or naming S1AB), latest start $(date -u -d @$LATEST +%FT%TZ)"
until tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -qE "$BP_RE"; do
  (( $(date +%s) < LATEST )) || not_started "no w41-swddirscreen GPU END line by $(date -u -d @$LATEST +%TZ); no GPU used."
  say_once bp "waiting for the w41-swddirscreen B' GPU END line"; sleep 30
done
log "Bp end seen: $(tail -n +$((N0 + 1)) "$M" | /usr/bin/grep -E "$BP_RE" | head -1 | cut -c1-200)"
sleep 20
while ! lease_ok || [ -e /tmp/wrf_gpu2_quiet ]; do
  (( $(date +%s) < LATEST )) || not_started "no S1AB lease / QUIET still set by $(date -u -d @$LATEST +%TZ); no GPU used."
  if ! lease_ok; then say_once lease "waiting for the S1AB lease ($LEASE: $(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null))"
  else say_once quiet "foreign QUIET: $(cat /tmp/wrf_gpu2_quiet 2>/dev/null)"; fi
  sleep 30
done
pre=$(bundle_ok); log "P check before: $pre"
[[ "$pre" == "BUNDLE_OK "*"new=0"* ]] || not_started "v0.4.0 bundle check before the lock: '$pre'; no GPU used."
mkdir -p "$OUT"
WAIT=$(( LATEST + 1800 - $(date +%s) ))
post "GPU START S1AB matched PROD 24 h whole-run A/B (A = v0.4.0 sealed bundle consumer, B = v0.4.1 B1 pin bcd156d19a; A0 prime + A0w, then A1 B1 A2 B2; one lock session <= 25 GPU-min; P before: $pre)"
LOCK=(<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label w41-bdyrec --timeout "$WAIT" --)
[ -n "${S1AB_DRY_ARM:-}" ] && LOCK=()
"${LOCK[@]}" bash "$HERE/session.sh" "$OUT"
rc=$?
[ -e "$OUT/session.log" ] || not_started "the GPU lock was not acquired / session refused before start (rc=$rc); no GPU used."
MIN=$(/usr/bin/grep -oE 'END( \(signal\))?( GPU part:)? [0-9]+ GPU-min' "$OUT/session.log" 2>/dev/null | /usr/bin/grep -oE '[0-9]+' | tail -1)
post "S1AB GPU END rc=$rc, ${MIN:-?} GPU-min (lock released); CPU summary on 14,15,30,31 next."
post_check=$(bundle_ok); log "P check after: $post_check"
taskset -c 14,15,30,31 nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3 "$HERE/summary.py" "$OUT" "$pre" "$post_check" \
  > "$OUT/summary.log" 2>&1
log "summary rc=$? -> $OUT/AB.json: $(tail -1 "$OUT/summary.log" | cut -c1-1500)"
post "S1AB RESULT: $(tail -1 "$OUT/summary.log" | cut -c1-900) (AB $OUT/AB.json)"
