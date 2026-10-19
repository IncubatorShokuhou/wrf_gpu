#!/usr/bin/env bash
# w42-bundle042: the ONE v0.4.2 GPU lock session (contract lanes/w42-bundle042.md; GPU slot from 13:00Z, <= 120 GPU-min,
# hard end 18:30Z). Phase 1 session.sh = bundle prime + seal + RW1/RW2/RW4 replay arms; phase 2 rw3t.sh = RW3 interleaved
# timing N (v0.4.2) / O (v0.4.1) 24 h + O72. One deadline for both = min(start + BUDGET, END_UTC).
# usage: scripts/with_gpu_lock.sh --label w42-bundle042 --timeout <lock-wait s> -- bash gpu_session.sh
#        env GS_BUDGET_S (default 7200), GS_END_UTC (default 2026-10-19T18:30:00Z), GS_SKIP_RW3T=1 to stop after phase 1.
# Posts exactly 'w42-bundle042: GPU START ...' and 'w42-bundle042: GPU END rc=<rc> <min> GPU-min ...' to MAILBOX.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MAILBOX=${GS_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
post() { printf '%s w42-bundle042: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "${BS_DRY_ARM:-}" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
source "$HERE/pin.env"
T0=$(date +%s)
END=$(date -u -d "${GS_END_UTC:-2026-10-19T18:30:00Z}" +%s)
DEADLINE=$(( T0 + ${GS_BUDGET_S:-7200} )); (( END < DEADLINE )) && DEADLINE=$END
(( DEADLINE - T0 >= 1800 )) || { post "GPU NOT STARTED: only $(( (DEADLINE - T0) / 60 )) min left before $(date -u -d @$END +%TZ)"; exit 4; }
FREE_G=$(( $(df --output=avail -B1G <DATA_ROOT> | tail -1) ))   # alisios floor 120 GiB + ~40 GiB session output
(( FREE_G >= 160 )) || { post "GPU NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < 160"; exit 4; }
child=
rc=0
LOGS="${BS_S:-<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/gpu}/session.log ${AT_ROOT:-<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/rw3t}/session.log"
finish() { post "GPU END rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); logs $LOGS; CPU follow-up (E41, RW3 D6 72 h + integrity, aot-key, verdict) next"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child"; rc=143; finish; exit 143; }
trap on_signal TERM INT HUP
trap finish EXIT
post "GPU START v0.4.2 bundle prime+seal + real-world gate RW1-RW4 (pin ${PIN:0:12}, src ${TREE:0:12}, icloud0=${BS_ICLOUD0:-0}); ONE lock session, cap $(( (DEADLINE - T0) / 60 )) GPU-min, ends by $(date -u -d @$DEADLINE +%TZ)"
BS_DEADLINE_EPOCH=$DEADLINE bash "$HERE/session.sh" & child=$!; wait "$child"; rc=$?; child=
(( rc == 0 )) || exit "$rc"
[ -n "${GS_SKIP_RW3T:-}" ] && exit 0
for i in $(seq 1 24); do [ -e /tmp/wrf_gpu2_quiet ] || break; sleep 5; done   # a foreign QUIET (not ours: traps removed it) <= 120 s
if [ -e /tmp/wrf_gpu2_quiet ]; then post "RW3T NOT STARTED: foreign QUIET $(cat /tmp/wrf_gpu2_quiet) after phase 1"; rc=3; exit 3; fi
AT_WINDOW_END_EPOCH=$DEADLINE bash "$HERE/rw3t.sh" & child=$!; wait "$child"; rc=$?; child=
exit "$rc"
