#!/usr/bin/env bash
# w41-bundle041 RW3 decisive re-run (manager 2026-10-18T14:17:09Z; GPU only after the manager's GO, clean window with the
# e2 jobs paused): ONE lock session, slots N1 O1 N2 O2 N3 O3 = WN3 0227 3-nest 24 h warm, N = v0.4.1 sealed bundle,
# O = v0.4.0 sealed bundle, both through P/tools/bundle_consumer.sh with a fresh layer outside both bundles; P / P0
# compile_bundle check BUNDLE_OK new=0 changed=0 before and after every arm, else stop (ALISIOS uses P0).
# DURING every arm core_watch.py samples all threads each second (pinned to core 28): rule P = one foreign process
# > 20 % of a core on 3,13,19,29 for > 10 s, rule A = foreign sum > 50 % for > 10 s -> the arm is stopped, its dir is
# kept as <slot>.abortedK, and the slot is repeated (fresh dir + layer; at most 2 repeats per slot) while the window allows.
# Pre-arm gate: cores_free.py 5 s; busy -> retry every 15 s, at most 4 tries, then start (the watcher governs).
# Window: ends at AT_WINDOW_END_UTC (from the GO) or start + AT_WINDOW_S (default 1560 s); an arm starts only if
# AT_ARM_NEED_S (default 270 s) remain. QUIET 'w41-bundle041 ATTRIB2 <slot>' during each arm. Posts 'ATTRIB2 GPU END'.
# Dry lifecycle (CPU, no GPU): AT_DRY=<s> AT_ROOT=<dir> AT_QUIET_FILE=<file> AT_MAILBOX=<file> AT_CORES=28 -> arms sleep.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1; P0=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
DRY=${AT_DRY:-}
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; A=${AT_ROOT:-$S/attrib2}; R=$A/runs; LAY=$A/layers; LOG=$A/session.log
CASE=$S/cases/wn3_0227; CORES=${AT_CORES:-3,13,19,29}
MAILBOX=${AT_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; QUIET_FILE=${AT_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || ( "$QUIET_FILE" != /tmp/wrf_gpu2_quiet && -n "${AT_MAILBOX:-}" && -n "${AT_ROOT:-}" && "$CORES" != 3,13,19,29 ) ]] \
  || { echo "dry needs AT_ROOT AT_QUIET_FILE AT_MAILBOX AT_CORES"; exit 2; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
[ -e "$A/session.log" ] && { echo "$A already used: immutable sessions need a fresh root"; exit 2; }
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
mkdir -p "$R" "$LAY"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
post() { printf '%s w41-bundle041: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
T0=$(date +%s)
if [ -n "${AT_WINDOW_END_UTC:-}" ]; then DEADLINE=$(date -u -d "$AT_WINDOW_END_UTC" +%s); else DEADLINE=$(( T0 + ${AT_WINDOW_S:-1560} )); fi
NEED=${AT_ARM_NEED_S:-270}
log "START ATTRIB2: N=$P O=$P0 slots N1 O1 N2 O2 N3 O3 window end $(date -u -d @$DEADLINE +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') watch $CORES"
log "tools $(cd "$HERE" && sha256sum attrib2.sh core_watch.py cores_free.py arm.sh | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
post "ATTRIB2 GPU START (manager GO; one lock session, window end $(date -u -d @$DEADLINE +%TZ)): N1 O1 N2 O2 N3 O3 warm 24 h WN3 0227, continuous core watch $CORES, abort+repeat rule P/A"
PREFIX="w41bundle041a2-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() { local u n=0; while [ -n "$(session_scopes)" ]; do for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && sleep 10; done; }
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-bundle041 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
child=
done_post() { post "ATTRIB2 GPU END rc=$1 $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); log $LOG; table follows (CPU)"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; stop_scopes; samplers_off; quiet_off
  log "END (signal)"; done_post 143; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; quiet_off; log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"; done_post "$rc"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {  # samplers_on RUN_DIR: VRAM (as in ATTRIB) + the core watcher on core 28 (off the arm cores)
  if [ -z "$DRY" ]; then
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
    SAMPLERS+=($!)
  fi
  taskset -c 28 nice -n 5 python3 "$HERE/core_watch.py" "$CORES" 20 10 50 "$$" "$1/core_watch.jsonl" "$1/core_watch_summary.json" \
    "$1/ABORT" 1 > "$1/core_watch.log" 2>&1 &
  SAMPLERS+=($!)
}
pre_gate() {  # pre_gate SLOT: 5 s sample, retry every 15 s, at most 4 tries, then start anyway (disclosed)
  local i
  for i in 1 2 3 4; do
    if python3 "$HERE/cores_free.py" "$CORES" 20 "$A/core_checks.jsonl" "$$" 5 >> "$A/core_gate.log" 2>&1; then
      printf '%s\t%s\tfree\tgate-try %s\n' "$1" "$(date -u +%FT%TZ)" "$i" >> "$A/core_gate.tsv"; return 0; fi
    log "pre-gate $1 try $i busy ($(tail -1 "$A/core_gate.log" | cut -c1-160))"
    (( i < 4 )) && { sleep 15 & child=$!; wait "$child"; child=; }
  done
  printf '%s\t%s\tBUSY_START\t%s\n' "$1" "$(date -u +%FT%TZ)" "$(tail -1 "$A/core_gate.log" | cut -c1-200)" >> "$A/core_gate.tsv"
}
chk() {  # chk BUNDLE OUTBASE
  [ -n "$DRY" ] && return 0
  JAX_PLATFORMS=cpu python3 "$1/tools/compile_bundle.py" check --cache "$1/jax" --manifest "$1/sealed/BUNDLE_MANIFEST.json" \
    --out "$2.json" > "$2.txt" 2>&1; /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$2.txt"; }
W="--domains-from-namelist --emit-initial-history"
run_slot() {  # run_slot SLOT KIND(N|O) -> 0 accepted, 9 no time, 8 aborted (repeat), 99 bundle check failed
  local name=$1 kind=$2 try=$3 b=$P; [[ $kind == O ]] && b=$P0
  (( DEADLINE - $(date +%s) >= NEED )) || { log "SKIP $name: window too short"; return 9; }
  mkdir -p "$R/$name"
  chk "$b" "$R/$name/check_before" || { log "$name: $b check before FAILED: stop"; return 99; }
  pre_gate "$name"
  printf '%s %s\n' "w41-bundle041 ATTRIB2 $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  samplers_on "$R/$name"; host_load "before $name try $try"
  log "ARM $name kind $kind try $try 24 h"
  if [ -n "$DRY" ]; then
    systemd-run --user --scope --quiet --unit="$PREFIX-$name-$try" -p RuntimeMaxSec=420 -- sleep "$DRY" > "$R/$name.scope.log" 2>&1 &
  else
    systemd-run --user --scope --quiet --unit="$PREFIX-$name-$try" -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec=420 -- \
      bash "$P/tools/bundle_consumer.sh" --bundle "$b" --layer "$LAY/${name}_t$try" -- \
      bash "$HERE/arm.sh" "$b/bundle.env" "$name" "$CASE" "$R/$name" 330 -- $W --hours 24 > "$R/$name.scope.log" 2>&1 &
  fi
  child=$!
  local aborted=
  while kill -0 "$child" 2>/dev/null; do
    if [ -s "$R/$name/ABORT" ] && [ -z "$aborted" ]; then
      aborted=$(cat "$R/$name/ABORT"); log "ABORT $name try $try: $aborted"; systemctl --user stop "$PREFIX-$name-$try.scope" 2>/dev/null
    fi
    sleep 1
  done
  wait "$child"; local rc=$?; child=
  samplers_off; quiet_off; host_load "after $name try $try"
  chk "$b" "$R/$name/check_after" || { log "$name: $b NOT OK after: stop"; return 99; }
  if [ -n "$aborted" ]; then
    mv "$R/$name" "$R/$name.aborted$try"; mv "$R/$name.scope.log" "$R/$name.aborted$try.scope.log"
    printf '%s\t%s\tABORTED\t%s\n' "$name" "$(date -u +%FT%TZ)" "$aborted" >> "$A/core_gate.tsv"
    log "ARM $name aborted try $try rc=$rc (kept as $name.aborted$try)"; return 8
  fi
  log "ARM $name rc=$rc try $try watch $(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('clean' if s['clean'] else 'within-rule' if s['within_rule'] else 'BREACH', 'maxproc', s['max_proc_seconds'], 'agg', s['agg_seconds'])" "$R/$name/core_watch_summary.json" 2>/dev/null) $(/usr/bin/grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1 | cut -c1-120)"
  return 0
}
for spec in "N1 N" "O1 O" "N2 N" "O2 O" "N3 N" "O3 O"; do
  set -- $spec
  for try in 1 2 3; do
    run_slot "$1" "$2" "$try"; rc=$?
    (( rc == 99 )) && exit 7
    (( rc == 8 )) && continue
    break
  done
  (( rc == 9 )) && break
done
log "DONE"
