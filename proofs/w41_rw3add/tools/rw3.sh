#!/usr/bin/env bash
# w41-rw3add: clean RW3 timing addendum for v0.4.1 (contract kernel 081f17f070). Derived from the frozen w41-bundle041
# ATTRIB2 session (tools_c9be5bcd32/attrib2.sh d583a56c; same abort rules, same arm.sh/consumer/bundle checks):
# ONE lock session, slots N1 O1 N2 O2 N3 O3 = WN3 0227 3-nest 24 h warm, then N72 O72 = one 72 h pair if the window allows;
# N = v0.4.1 sealed bundle, O = v0.4.0 sealed bundle, both through P/tools/bundle_consumer.sh (== the v0.4.0 consumer;
# v0.4.0 ships no consumer of its own) with a fresh layer outside both bundles; P / P0 compile_bundle check BUNDLE_OK
# new=0 changed=0 before and after every arm, else stop (ALISIOS uses P0).
# DURING every arm core_watch.py (frozen a4938233) samples all threads each second (pinned to core 24, off the arm cores):
# rule P = one foreign process > 20 % of a core on 3,13,19,29 for > 10 s, rule A = foreign sum > 50 % for > 10 s -> the arm
# is stopped, its dir is kept as <slot>.abortedK, and the slot is repeated (fresh dir + layer; at most 3 tries per slot).
# Disclosure samplers beside it (core 25): disk_watch.py (<DATA_ROOT> await) and sched_watch.py (the arm's own cgroup CPU
# pressure + CLI main-thread runqueue wait).
# Changes vs attrib2.sh (abort rules, arm, consumer, checks unchanged): lane paths/labels; slot list with hours (72 h budgets);
# pre-gate = wait for a quiet stretch (AT_QGATE_S, default 60 s, see pre_gate) up to AT_GATE_MAX_S (default 2700 s) per slot
# instead of 4 x 5 s samples, then start anyway (disclosed BUSY_START); a 72 h pair starts only if the window holds both
# arms; samplers on the lane cores 24/25; a manager HOLD/STOP line to w41-rw3add stops before the next arm; abort thresholds
# AT_RULE_P_PCT / AT_RULE_A_PCT (default = contract 20 / 50) - when changed, a never-aborting shadow watcher at 20/50 records
# what the contract rules would have said (core_watch_strict_summary.json). Optional post-arm contention limits
# AT_MAX_CLI_WAIT (CLI main-thread runqueue-wait share), AT_MAX_PSI_S (arm cgroup cpu.pressure some, s), AT_MAX_AGG_MEAN
# (foreign agg mean %, core_watch.jsonl), AT_MAX_DISK_AWAIT_MS (arm-window <DATA_ROOT> await): an arm over any limit is kept as
# <slot>.abortedK (ABORT_CONTENTION) and repeated. The pre-gate also waits while updatedb runs; nvidia-smi dmon per arm.
# Window: ends at AT_WINDOW_END_UTC or start + AT_WINDOW_S (default 1560 s); an arm starts only if its NEED remains.
# QUIET 'w41-rw3add RW3ADD <slot>' during each arm. Posts 'RW3ADD GPU START/END'.
# Dry lifecycle (CPU, no GPU): AT_DRY=<s> AT_ROOT=<dir> AT_QUIET_FILE=<file> AT_MAILBOX=<file> AT_CORES=<cores> -> arms sleep.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1; P0=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
DRY=${AT_DRY:-}
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-rw3add; A=${AT_ROOT:-$S/rw3}; R=$A/runs; LAY=$A/layers; LOG=$A/session.log
CASE=$S/cases/wn3_0227; CORES=${AT_CORES:-3,13,19,29}; SLOTS=${AT_SLOTS:-N1:N:24 O1:O:24 N2:N:24 O2:O:24 N3:N:24 O3:O:24 N72:N:72 O72:O:72}
QGATE_S=${AT_QGATE_S:-60}; GATE_MAX_S=${AT_GATE_MAX_S:-2700}
RULE_P=${AT_RULE_P_PCT:-20}; RULE_A=${AT_RULE_A_PCT:-50}   # abort thresholds (% of one core; > 10 s cumulative); contract 20/50
MAX_CLI_WAIT=${AT_MAX_CLI_WAIT:-}; MAX_PSI=${AT_MAX_PSI_S:-}; MAX_AGG_MEAN=${AT_MAX_AGG_MEAN:-}   # post-arm contention limits (off if empty)
MAX_DISK=${AT_MAX_DISK_AWAIT_MS:-}   # post-arm <DATA_ROOT> (nvme3n1) arm-window await limit, ms (off if empty)
MAILBOX=${AT_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; QUIET_FILE=${AT_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || ( "$QUIET_FILE" != /tmp/wrf_gpu2_quiet && -n "${AT_MAILBOX:-}" && -n "${AT_ROOT:-}" && "$CORES" != 3,13,19,29 ) ]] \
  || { echo "dry needs AT_ROOT AT_QUIET_FILE AT_MAILBOX AT_CORES"; exit 2; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
[ -e "$A/session.log" ] && { echo "$A already used: immutable sessions need a fresh root"; exit 2; }
[ -n "$DRY" ] || [ -s "$CASE/wrfbdy_d01" ] || { echo "case $CASE missing"; exit 2; }
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
mkdir -p "$R" "$LAY"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
post() { printf '%s w41-rw3add: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
T0=$(date +%s)
if [ -n "${AT_WINDOW_END_UTC:-}" ]; then DEADLINE=$(date -u -d "$AT_WINDOW_END_UTC" +%s); else DEADLINE=$(( T0 + ${AT_WINDOW_S:-1560} )); fi
need_of() { if [[ $1 == 72 ]]; then echo "${AT_NEED72_S:-650}"; else echo "${AT_NEED_S:-270}"; fi; }
log "START RW3ADD: rules P>$RULE_P %/A>$RULE_A % >10 s, post-arm limits cli_wait<=${MAX_CLI_WAIT:-off} psi<=${MAX_PSI:-off} agg_mean<=${MAX_AGG_MEAN:-off} disk<=${MAX_DISK:-off}ms, quiet gate ${QGATE_S}s; N=$P O=$P0 slots [$SLOTS] window end $(date -u -d @$DEADLINE +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') watch $CORES"
log "tools $(cd "$HERE" && sha256sum rw3.sh arm.sh core_watch.py cores_free.py disk_watch.py sched_watch.py | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
post "RW3ADD GPU START (lease to 07:45Z; one lock session, window end $(date -u -d @$DEADLINE +%TZ)): [$SLOTS] WN3 0227 warm, v0.4.1 vs v0.4.0 sealed bundles, continuous core watch $CORES, abort+repeat rule P>$RULE_P %/A>$RULE_A % >10 s (max 3 tries/slot)"
PREFIX="w41rw3add-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() { local u n=0; while [ -n "$(session_scopes)" ]; do for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && sleep 10; done; }
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-rw3add ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
DISKPID=; disk_off() { [ -n "$DISKPID" ] && { kill -TERM "$DISKPID" 2>/dev/null; wait "$DISKPID" 2>/dev/null; DISKPID=; }; }
child=
done_post() { post "RW3ADD GPU END rc=$1 $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); log $LOG; table follows (CPU)"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; stop_scopes; samplers_off; disk_off; quiet_off
  log "END (signal)"; done_post 143; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; disk_off; quiet_off; log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"; done_post "$rc"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
taskset -c 25 nice -n 5 python3 "$HERE/disk_watch.py" sample "$A/disk.jsonl" > "$A/disk_watch.log" 2>&1 &
DISKPID=$!
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {  # samplers_on RUN_DIR UNIT: VRAM + the core watcher on core 24 + the arm's own scheduler view on core 25
  if [ -z "$DRY" ]; then
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
    SAMPLERS+=($!)
    taskset -c 25 nvidia-smi dmon -s puc -o T > "$1/gpu_dmon.txt" 2>/dev/null &   # disclosure: power, util, clocks per second
    SAMPLERS+=($!)
  fi
  taskset -c 24 nice -n 5 python3 "$HERE/core_watch.py" "$CORES" "$RULE_P" 10 "$RULE_A" "$$" "$1/core_watch.jsonl" \
    "$1/core_watch_summary.json" "$1/ABORT" 1 > "$1/core_watch.log" 2>&1 &
  SAMPLERS+=($!)
  if [[ $RULE_P != 20 || $RULE_A != 50 ]]; then   # shadow watcher at the contract 20/50 % (never aborts): disclosure
    taskset -c 25 nice -n 5 python3 "$HERE/core_watch.py" "$CORES" 20 1e9 50 "$$" "$1/core_watch_strict.jsonl" \
      "$1/core_watch_strict_summary.json" "$1/ABORT_STRICT_NEVER" 1 > "$1/core_watch_strict.log" 2>&1 &
    SAMPLERS+=($!)
  fi
  taskset -c 25 nice -n 5 python3 "$HERE/sched_watch.py" "$2" "$1/sched_watch.jsonl" "$1/sched_watch_summary.json" 1 \
    > "$1/sched_watch.log" 2>&1 &
  SAMPLERS+=($!)
}
pre_gate() {  # pre_gate SLOT: wait for a QUIET STRETCH, then start; after GATE_MAX_S start anyway (disclosed BUSY_START)
  # quiet stretch = the frozen core_watch.py at 1 s for QGATE_S seconds with the arm's rules tightened to 3 s
  # (no foreign process > RULE_P % for >= 3 s, foreign sum > RULE_A % in < 3 s); a busy stretch is re-checked at once.
  local i=0 t_end=$(( $(date +%s) + GATE_MAX_S )) g
  while :; do
    i=$((i + 1)); g="$A/gate/$1.$i"; mkdir -p "$g"
    taskset -c 24 nice -n 5 python3 "$HERE/core_watch.py" "$CORES" "$RULE_P" 3 "$RULE_A" "$$" "$g/w.jsonl" "$g/s.json" "$g/ABORT" 1 \
      > "$g/w.log" 2>&1 &
    child=$!; sleep "$QGATE_S" & local sl=$!; wait "$sl"; kill -TERM "$child" 2>/dev/null; wait "$child"; child=
    if pgrep -f '^/usr/(s)?bin/updatedb' > /dev/null 2>&1; then   # anchored: comm is cut to 15 chars, never matches our shells
      echo '{"rule": "UPDATEDB", "note": "updatedb running (system cron, <DATA_ROOT> IO)"}' > "$g/ABORT"; fi
    if [ ! -s "$g/ABORT" ]; then
      printf '%s\t%s\tfree\tquiet-stretch %s s, check %s\n' "$1" "$(date -u +%FT%TZ)" "$QGATE_S" "$i" >> "$A/core_gate.tsv"; return 0; fi
    (( i % 10 == 1 )) && log "pre-gate $1 check $i busy ($(cut -c1-160 "$g/ABORT"))"
    hold_stop && return 0
    (( $(date +%s) >= t_end )) && break
    (( DEADLINE - $(date +%s) >= $(need_of "$2") )) || break
  done
  printf '%s\t%s\tBUSY_START\tafter %s checks: %s\n' "$1" "$(date -u +%FT%TZ)" "$i" "$(cut -c1-200 "$g/ABORT")" >> "$A/core_gate.tsv"
}
MB_FROM=$(wc -l < "$MAILBOX")
hold_stop() {  # a manager HOLD/STOP line to w41-rw3add after the session start cancels the remaining slots
  tail -n +$((MB_FROM + 1)) "$MAILBOX" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> [^:]*w41-rw3add[^:]*:.*\b(HOLD|STOP)\b'; }
chk() {  # chk BUNDLE OUTBASE
  [ -n "$DRY" ] && return 0
  JAX_PLATFORMS=cpu python3 "$1/tools/compile_bundle.py" check --cache "$1/jax" --manifest "$1/sealed/BUNDLE_MANIFEST.json" \
    --out "$2.json" > "$2.txt" 2>&1; /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$2.txt"; }
W="--domains-from-namelist --emit-initial-history"
run_slot() {  # run_slot SLOT KIND(N|O) TRY HOURS -> 0 accepted, 9 no time, 8 aborted (repeat), 99 bundle check failed
  local name=$1 kind=$2 try=$3 hours=$4 b=$P; [[ $kind == O ]] && b=$P0
  local need budget rmax; need=$(need_of "$hours")
  if [[ $hours == 72 ]]; then budget=760; rmax=860; else budget=330; rmax=420; fi
  (( DEADLINE - $(date +%s) >= need )) || { log "SKIP $name: window too short"; return 9; }
  mkdir -p "$R/$name"
  chk "$b" "$R/$name/check_before" || { log "$name: $b check before FAILED: stop"; return 99; }
  pre_gate "$name" "$hours"
  hold_stop && { log "manager HOLD/STOP: stop before $name"; return 99; }
  (( DEADLINE - $(date +%s) >= need )) || { log "SKIP $name: window too short after the pre-gate"; return 9; }
  printf '%s %s\n' "w41-rw3add RW3ADD $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  local unit="$PREFIX-$name-$try"
  samplers_on "$R/$name" "$unit.scope"; host_load "before $name try $try"
  log "ARM $name kind $kind try $try $hours h"
  if [ -n "$DRY" ]; then
    systemd-run --user --scope --quiet --unit="$unit" -p RuntimeMaxSec="$rmax" -- sleep "$DRY" > "$R/$name.scope.log" 2>&1 &
  else
    systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec="$rmax" -- \
      bash "$P/tools/bundle_consumer.sh" --bundle "$b" --layer "$LAY/${name}_t$try" -- \
      bash "$HERE/arm.sh" "$b/bundle.env" "$name" "$CASE" "$R/$name" "$budget" -- $W --hours "$hours" > "$R/$name.scope.log" 2>&1 &
  fi
  child=$!
  local aborted=
  while kill -0 "$child" 2>/dev/null; do
    if [ -s "$R/$name/ABORT" ] && [ -z "$aborted" ]; then
      aborted=$(cat "$R/$name/ABORT"); log "ABORT $name try $try: $aborted"; systemctl --user stop "$unit.scope" 2>/dev/null
    fi
    sleep 1
  done
  wait "$child"; local rc=$?; child=
  samplers_off; quiet_off; host_load "after $name try $try"
  chk "$b" "$R/$name/check_after" || { log "$name: $b NOT OK after: stop"; return 99; }
  if [ -z "$aborted" ] && [ -n "$MAX_CLI_WAIT$MAX_PSI$MAX_AGG_MEAN$MAX_DISK" ]; then   # the arm's own measured contention
    aborted=$(python3 - "$R/$name" "$MAX_CLI_WAIT" "$MAX_PSI" "$MAX_AGG_MEAN" "$MAX_DISK" "$A/disk.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
d, w, p, a, k, disk = Path(sys.argv[1]), *sys.argv[2:7]
s = json.loads((d / "sched_watch_summary.json").read_text())
cw, psi = s.get("cli_main_wait_share"), ((s.get("last") or {}).get("pressure_some_total_s"))
agg = statistics.mean(json.loads(x)["foreign_agg"] for x in (d / "core_watch.jsonl").read_text().splitlines() if x.strip())
bad = []
if w and (cw is None or cw > float(w)): bad.append(f"cli_main_wait_share {cw} > {w}")
if p and (psi is None or psi > float(p)): bad.append(f"cgroup cpu pressure {psi} s > {p}")
if a and agg > float(a): bad.append(f"foreign agg mean {agg:.1f} % > {a}")
if k:   # interval-weighted arm-window await on nvme3n1, as disk_watch.py summarize
    try:
        t0, t1 = float((d / "t0_unix").read_text()), float((d / "t_end_unix").read_text())
        win = [r for r in (json.loads(x) for x in open(disk) if x.strip()) if t0 <= r["t"] - r["dt"] and r["t"] <= t1 + 1e-6]
        ios = sum(r["nvme3n1"]["ios"] for r in win); ms = sum(r["nvme3n1"]["ms"] for r in win)
        aw = ms / ios if ios else 0.0
        if not win or aw > float(k): bad.append(f"<DATA_ROOT> await {aw:.1f} ms > {k} ({len(win)} s)")
    except (OSError, ValueError) as e:
        bad.append(f"disk check failed: {e}")
print(json.dumps({"rule": "CONTENTION", "violations": bad}) if bad else "")
PY
)
    [ -n "$aborted" ] && { echo "$aborted" > "$R/$name/ABORT_CONTENTION"; log "REJECT $name try $try (post-arm contention): $aborted"; }
  fi
  if [ -n "$aborted" ]; then
    mv "$R/$name" "$R/$name.aborted$try"; mv "$R/$name.scope.log" "$R/$name.aborted$try.scope.log"
    printf '%s\t%s\tABORTED\t%s\n' "$name" "$(date -u +%FT%TZ)" "$aborted" >> "$A/core_gate.tsv"
    log "ARM $name aborted try $try rc=$rc (kept as $name.aborted$try)"; return 8
  fi
  log "ARM $name rc=$rc try $try watch $(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('clean' if s['clean'] else 'within-rule' if s['within_rule'] else 'BREACH', 'maxproc', s['max_proc_seconds'], 'agg', s['agg_seconds'])" "$R/$name/core_watch_summary.json" 2>/dev/null) sched $(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('cli_wait_share', s['cli_main_wait_share'], 'psi_s', round((s['last'] or {}).get('pressure_some_total_s', -1), 1))" "$R/$name/sched_watch_summary.json" 2>/dev/null) $(/usr/bin/grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1 | cut -c1-120)"
  return 0
}
for spec in $SLOTS; do
  IFS=: read -r name kind hours <<< "$spec"
  if [[ $hours == 72 && $kind == N ]] && (( DEADLINE - $(date +%s) < 2 * $(need_of 72) )); then
    log "SKIP 72 h pair: window does not hold both arms"; break; fi
  for try in 1 2 3; do
    run_slot "$name" "$kind" "$try" "$hours"; rc=$?
    (( rc == 99 )) && exit 7
    (( rc == 8 )) && continue
    break
  done
  (( rc == 9 )) && break
done
log "DONE"
