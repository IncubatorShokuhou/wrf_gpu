#!/usr/bin/env bash
# w43-release TOP-UP (copy of gpu_session.sh, 2026-10-21 ~15:30Z): RW3T phase 2 ONLY on a NEW immutable root
# ($S/rw3t_topup) with slots O72 N1 O1 N2 O2 N3 O3 (closes the MISSING O72 byte control + same-session N/O speed).
# Original header follows.
# w43-release: the ONE v0.4.3 GPU lock session (contract lanes/w43-release.md; manager slot, REAL UTC). Phase 1 session.sh =
# bundle prime (autotune seeded from v0.4.2) + seal + RW1/RW2/RW4 replay arms; phase 2 rw3t.sh = RW3 interleaved timing
# N (v0.4.3) / O (v0.4.2) 24 h + N72/O72. One deadline for both = min(start + GS_BUDGET_S, GA_END_UTC real).
# usage: scripts/with_gpu_lock.sh --label w43-release --timeout <lock-wait s> -- bash gpu_session.sh
#        env GA_START_UTC / GA_END_UTC (REQUIRED, real UTC via real_utc.sh: refuse before the start, refuse with < GA_MIN_S
#        left), GS_BUDGET_S (default 7200), GS_SKIP_RW3T=1 to stop after phase 1, GS_MAILBOX (dry runs).
# Posts exactly 'w43-release: GPU START ...' and 'w43-release: GPU END rc=<rc> <min> GPU-min ...' to MAILBOX; records the
# kernel hibernation journal before/after (E60).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MAILBOX=${GS_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
REAL=${GS_REAL_UTC:-<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh}
post() { printf '%s w43-release: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "${BS_DRY_ARM:-}" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
source "$HERE/pin.env"
rnow() { local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric"; exit 7; }; echo "$e"; }
T0=$(date +%s); R0=$(rnow) || exit 7
START=$(date -u -d "${GA_START_UTC:?GA_START_UTC required}" +%s); END=$(date -u -d "${GA_END_UTC:?GA_END_UTC required}" +%s)
(( R0 >= START )) || { echo "too early (window starts $GA_START_UTC real UTC)"; exit 4; }
(( END - R0 >= ${GA_MIN_S:-1800} )) || { post "GPU NOT STARTED: only $(( (END - R0) / 60 )) min left before $GA_END_UTC"; exit 6; }
DEADLINE=$(( T0 + (END - R0 - 60) * 9 / 10 ))   # system-clock deadline, x0.9 against slew
(( T0 + ${GS_BUDGET_S:-7200} < DEADLINE )) && DEADLINE=$(( T0 + ${GS_BUDGET_S:-7200} ))
FREE_G=$(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) ))   # alisios floor 120 GiB + ~40 GiB session output
(( FREE_G >= 240 )) || { post "GPU NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < 240 (ALISIOS floor 200 + O72)"; exit 4; }
LOGD=${BS_S:-<DATA_ROOT>/wrf_gpu2/scratch/w43-release/gpu_topup}; mkdir -p "$LOGD"
journalctl -k --since "-6h" 2>/dev/null | /usr/bin/grep -i hibernat > "$LOGD/hibernate_before.txt"
echo "system $(date -u +%FT%TZ) real $("$REAL" --iso) T0=$T0 R0=$R0 START=$START END=$END DEADLINE=$DEADLINE" > "$LOGD/clocks.txt"
child=
rc=0
export AT_ROOT=${AT_ROOT:-<DATA_ROOT>/wrf_gpu2/scratch/w43-release/rw3t_topup}
export AT_SLOTS=${AT_SLOTS:-O72:O:72 N1:N:24 O1:O:24 N2:N:24 O2:O:24 N3:N:24 O3:O:24}
LOGS="$AT_ROOT/session.log"
finish() { journalctl -k --since "@$T0" 2>/dev/null | /usr/bin/grep -i hibernat > "$LOGD/hibernate_during.txt"
  post "GPU END rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); hibernation lines during: $(wc -l < "$LOGD/hibernate_during.txt"); logs $LOGS; CPU follow-up (O72 bytes, RW3T verdict top-up) next"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child"; rc=143; finish; exit 143; }
trap on_signal TERM INT HUP
trap finish EXIT
post "GPU START RW3T top-up [$AT_SLOTS] v0.4.3 (N) vs v0.4.2 (O) sealed bundles, root $AT_ROOT; cap $(( (DEADLINE - T0) / 60 )) GPU-min, ends by $(date -u -d @$DEADLINE +%TZ) system"
for i in $(seq 1 24); do [ -e /tmp/wrf_gpu2_quiet ] || break; sleep 5; done   # a foreign QUIET <= 120 s
if [ -e /tmp/wrf_gpu2_quiet ]; then post "RW3T top-up NOT STARTED: foreign QUIET $(cat /tmp/wrf_gpu2_quiet)"; rc=3; exit 3; fi
AT_WINDOW_END_EPOCH=$DEADLINE bash "$HERE/rw3t.sh" & child=$!; wait "$child"; rc=$?; child=
exit "$rc"
