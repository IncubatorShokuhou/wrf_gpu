#!/usr/bin/env bash
# w42-harness S1H trigger 2 (manager 2026-10-19T04:17:26Z: ALISIOS GPU window now..07:30Z HARD, SLOT 1 of 3).
# usage: trigger2.sh TOOLS_DIR TRIG_DIR      (lane core 28, nice 19, detached; never prompts anyone)
# Reads only MAILBOX lines AFTER line H_ANCHOR (default 14228), /usr/bin/grep -E anchored on the sender at line start.
#   start = the exact line '^2026-10-19T<ts> w42-icloud0: CPU REF END' (twin frees core 13), not before H_NOT_BEFORE (04:45Z);
#   no such line by H_FALLBACK_AT (05:15Z): start at once if NO w42-icloud0 process uses core 13 (pinned to a CPU list that
#     contains 13, or running on 13 now; full-affinity sleeping wrappers do not count), else post 'S1H NOT STARTED (...)';
#   every start also needs the lease (expires-utc >= H_END_UTC) and no QUIET file; latest start H_LATEST_START (05:40Z,
#     incl. the with_gpu_lock wait) -> else 'S1H NOT STARTED (...)' (slot falls back to the 13:00Z window, same order);
#   a manager line '^<ts> manager -> w42-harness: S1H (HOLD|STOP)' -> 'S1H NOT STARTED (manager HOLD/STOP)'.
# Session: TOOLS_DIR/session.sh with H_END_UTC (default 06:25Z hard end); it posts exactly 'S1H GPU END' on every exit
# after its traps are set; a session that stops before posting 'S1H GPU START' -> this trigger posts 'S1H NOT STARTED'.
# Session rc 3 (QUIET seen at its start, root untouched) -> retry after 60 s. Slot 2 (w42-icloud0) starts on either line.
# Watcher core (session H_WATCH_CORE: core_watch/sched/disk watchers, keycheck, bundle checks) is read from the data file
# TRIG_DIR/watch_core at start time (one integer; default 28 = lane core; a manager core loan goes there, not into a script).
# Dry (CPU): H_TRIGGER_DRY=1 + H_MAILBOX/H_LEASE/H_ANCHOR and the time overrides -> session directly (no lock) with the
# caller's H_DRY* variables; H_CORE13_PATTERN overrides the process pattern of the core-13 check.
set -uo pipefail
TOOLS=$(readlink -f "$1"); TRIG=$(readlink -f "$2"); mkdir -p "$TRIG"
REPO=<USER_HOME>/src/wrf_gpu2_wt/w42-harness
MAILBOX=${H_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
LEASE=${H_LEASE:-/tmp/wrf_gpu2_run_approvals/w42-harness.lease}
ANCHOR=${H_ANCHOR:-14228}
END_UTC=${H_END_UTC:-2026-10-19T06:25:00Z}
ts() { date -u -d "$1" +%s; }
NB=$(ts "${H_NOT_BEFORE:-2026-10-19T04:45:00Z}"); FB=$(ts "${H_FALLBACK_AT:-2026-10-19T05:15:00Z}")
NA=$(ts "${H_LATEST_START:-2026-10-19T05:40:00Z}"); END=$(ts "$END_UTC")
PAT=${H_CORE13_PATTERN:-w42-icloud0}
log() { echo "$(date -u +%FT%TZ) $*" >> "$TRIG/trigger.log"; }
post() { printf '%s w42-harness: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
after() { tail -n +$((ANCHOR + 1)) "$MAILBOX"; }
not_started() { log "NOT STARTED: $1"; post "S1H NOT STARTED ($1)"; exit 0; }
core13_users() {  # PIDs of PAT processes pinned to a CPU list containing 13 (not the whole machine) or running on 13
  python3 - "$PAT" <<'PY'
import os, sys
pat, out = sys.argv[1], []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        cwd = os.readlink(f"/proc/{pid}/cwd") if pat not in cmd else ""
        if pat not in cmd + " " + cwd:
            continue
        aff = os.sched_getaffinity(int(pid))
        stat = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        state, psr = stat[0], int(stat[36])
    except (OSError, ValueError, IndexError):
        continue
    if 13 in aff and len(aff) < os.cpu_count():
        out.append(f"{pid}:aff={','.join(map(str, sorted(aff)))}")
    elif state == "R" and psr == 13:
        out.append(f"{pid}:running_on_13")
print(" ".join(out))
PY
}
[ -x "$REPO/scripts/with_gpu_lock.sh" ] && [ -x "$TOOLS/session.sh" ] || not_started "trigger tools missing"
(( $(wc -l < "$MAILBOX") >= ANCHOR )) || not_started "mailbox shorter than the anchor line $ANCHOR"
log "armed: anchor line $ANCHOR, not before $(date -u -d @$NB +%TZ), fallback $(date -u -d @$FB +%TZ), latest start $(date -u -d @$NA +%TZ), hard end $END_UTC, tools $TOOLS ($(sha256sum "$0" | cut -c1-12))"
fallback=
while :; do
  now=$(date +%s)
  after | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> w42-harness: S1H (HOLD|STOP)\b' && not_started "manager HOLD/STOP"
  (( now >= NA )) && not_started "no start before the latest start $(date -u -d @$NA +%TZ); slot falls back to 13:00Z"
  ref=; after | /usr/bin/grep -q -E '^2026-10-19T[0-9:.+Z-]+ w42-icloud0: CPU REF END\b' && ref=1
  if [ -z "$ref" ] && [ -z "$fallback" ] && (( now >= FB )); then
    users=$(core13_users)
    [ -n "$users" ] && not_started "no 'w42-icloud0: CPU REF END' by $(date -u -d @$FB +%TZ) and w42-icloud0 still on core 13: $users"
    fallback=1; log "fallback at $(date -u +%TZ): no CPU REF END line, core 13 free of $PAT"
  fi
  exp=$(/usr/bin/grep -m1 '^expires-utc:' "$LEASE" 2>/dev/null | cut -d' ' -f2)
  lease_ok=; [ -n "$exp" ] && (( $(date -u -d "$exp" +%s 2>/dev/null || echo 0) >= END )) && lease_ok=1
  if (( now >= NB )) && [[ -n "$ref$fallback" ]] && [ -n "$lease_ok" ] && [ ! -e /tmp/wrf_gpu2_quiet ]; then
    log "start (ref=${ref:-0} fallback=${fallback:-0}, lease to $exp): queue on with_gpu_lock"
    out="$TRIG/session.$(date -u +%H%M%S).out"
    wc=$(tr -dc '0-9' < "$TRIG/watch_core" 2>/dev/null); [[ $wc =~ ^[0-9]+$ ]] && (( wc < $(nproc --all) )) || wc=28
    log "watcher core $wc"
    if [ -n "${H_TRIGGER_DRY:-}" ]; then
      H_END_UTC="$END_UTC" H_WATCH_CORE="$wc" bash "$TOOLS/session.sh" > "$out" 2>&1
    else
      "$REPO/scripts/with_gpu_lock.sh" --label w42-harness --timeout $(( NA - now > 60 ? NA - now : 60 )) -- \
        env H_END_UTC="$END_UTC" H_WATCH_CORE="$wc" bash "$TOOLS/session.sh" > "$out" 2>&1
    fi
    rc=$?; log "session rc=$rc"
    if (( rc == 3 )); then sleep 60; continue; fi
    (( rc == 124 )) && not_started "GPU lock not acquired before $(date -u -d @$NA +%TZ)"
    after | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ w42-harness: S1H GPU START' || not_started "session refused before START (rc $rc, $out)"
    exit "$rc"
  fi
  sleep 15
done
