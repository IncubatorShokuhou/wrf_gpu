#!/usr/bin/env bash
# w41-bundle041 RW3 attribution session (manager GO 12:39:43Z, <= 45 GPU-min): is the +7.6 % whole-run / +10.1 % stepping
# v0.4.1-vs-v0.4.0 delta of the first B1/B7 pair real, and which change carries it? ONE lock session, WN3 0227 3-nest.
#   C0  cold 3 h of snapshot 06ab6c9156 (= v0.4.1 minus the fused-namelist-arguments change; has record-window + swddir)
#       into a private cache (untimed compile)
#   then interleaved warm 24 h arms, each behind B1's core gate on 3,13,19,29 (cores_free.py, 20 %, retry 60 s, after 180 s
#   run + DISCLOSE) and under QUIET:  N1 O1 C1 N2 O2 C2
#     N = v0.4.1 sealed bundle via P/tools/bundle_consumer.sh (fresh layer), O = v0.4.0 sealed bundle via the same consumer
#     (layer outside v0.4.0 P; P0 BUNDLE_OK before/after else stop), C = snapshot 06ab6c, private warm cache, direct.
# Budget: deadline = min(start + 2700 s, 17:30Z); an arm runs only if 1.3 x expected fits. Posts 'ATTRIB GPU END'.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1; P0=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
DRY=${AT_DRY:-}   # AT_DRY=<s>: CPU lifecycle proof (needs AT_ROOT, AT_QUIET_FILE, AT_MAILBOX): arms sleep, checks skipped
S=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041; A=${AT_ROOT:-$S/attrib}; R=$A/runs; LAY=$A/layers; LOG=$A/session.log
SN=$S/snap_06ab6c; SNTREE=54ef1c991d780c264065b8a86d72f9c1a2514043; CASE=$S/cases/wn3_0227
MAILBOX=${AT_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; QUIET_FILE=${AT_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || ( "$QUIET_FILE" != /tmp/wrf_gpu2_quiet && -n "${AT_MAILBOX:-}" && -n "${AT_ROOT:-}" ) ]] || { echo "dry needs AT_*"; exit 2; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
mkdir -p "$R" "$LAY" "$A/cache/jax" "$A/cache/autotune" "$A/cache/triton" "$A/cache/cuda"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
post() { printf '%s w41-bundle041: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
[[ "$(git -C "$SN" rev-parse HEAD:src)" == "$SNTREE" && -z "$(git -C "$SN" status --porcelain --untracked-files=all -- src)" ]] \
  || { log "snapshot moved/dirty"; exit 4; }
cat > "$A/c.env" <<EOF
export PYTHONPATH=$SN/src
export GPUWRF_JAX_CACHE_DIR=$A/cache/jax JAX_COMPILATION_CACHE_DIR=$A/cache/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$A/cache/autotune
export TRITON_CACHE_DIR=$A/cache/triton CUDA_CACHE_PATH=$A/cache/cuda
EOF
T0=$(date +%s); END_UTC=$(date -u -d "${AT_END_UTC:-2026-10-18T17:30:00Z}" +%s)
DEADLINE=$(( T0 + ${AT_BUDGET_S:-2700} )); (( END_UTC < DEADLINE )) && DEADLINE=$END_UTC
log "START attribution: N=$P O=$P0 C=$SN (src $SNTREE) deadline $(date -u -d @$DEADLINE +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}')"
post "ATTRIB GPU START (manager GO 12:39:43Z; one lock session <= $(( (DEADLINE - T0) / 60 )) GPU-min, ends by $(date -u -d @$DEADLINE +%TZ)): C0 cold 06ab6c 3 h, then N1 O1 C1 N2 O2 C2 warm 24 h WN3 0227 behind the core gate"
PREFIX="w41bundle041at-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() { local u n=0; while [ -n "$(session_scopes)" ]; do for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && sleep 10; done; }
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-bundle041 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill "$p" 2>/dev/null; done; SAMPLERS=(); }
child=
done_post() { post "ATTRIB GPU END rc=$1 $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); log $LOG; B/A table follows (CPU)"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; stop_scopes; samplers_off; quiet_off
  log "END (signal)"; done_post 143; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; quiet_off; log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"; done_post "$rc"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() { "$@" & child=$!; wait "$child"; local rc=$?; child=; return $rc; }
cap_for() { local left=$(( DEADLINE - $(date +%s) )); (( left >= $1 * 13 / 10 )) || return 1; (( left < $2 )) && echo "$left" || echo "$2"; }
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {
  [ -n "$DRY" ] && return 0
  nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
  SAMPLERS+=($!)
}
core_gate() {  # core_gate ARM: free -> run; still busy after 180 s -> run + DISCLOSE (manager 12:39:43Z)
  local t_start; t_start=$(date +%s)
  while :; do
    if python3 "$HERE/cores_free.py" 3,13,19,29 20 "$A/core_checks.jsonl" "$$" 10 >> "$A/core_gate.log" 2>&1; then
      printf '%s\t%s\tfree\n' "$1" "$(date -u +%FT%TZ)" >> "$A/core_gate.tsv"; log "core gate $1: free"; return 0; fi
    if (( $(date +%s) - t_start >= 180 )); then
      printf '%s\t%s\tBUSY_RUN_DISCLOSED\t%s\n' "$1" "$(date -u +%FT%TZ)" "$(tail -1 "$A/core_gate.log" | cut -c1-200)" >> "$A/core_gate.tsv"
      log "core gate $1: BUSY after 180 s -> RUN + DISCLOSE ($(tail -1 "$A/core_gate.log" | cut -c1-200))"; return 0; fi
    log "core gate $1: busy ($(tail -1 "$A/core_gate.log" | cut -c1-160)); retry 60 s"
    sleep 60 & child=$!; wait "$child"; child=
  done
}
check0() { [ -n "$DRY" ] && return 0; JAX_PLATFORMS=cpu python3 "$P0/tools/compile_bundle.py" check --cache "$P0/jax" --manifest "$P0/sealed/BUNDLE_MANIFEST.json" \
    --out "$1.json" > "$1.txt" 2>&1; cat "$1.txt"; /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$1.txt"; }
check1() { [ -n "$DRY" ] && return 0; JAX_PLATFORMS=cpu python3 "$P/tools/compile_bundle.py" check --cache "$P/jax" --manifest "$P/sealed/BUNDLE_MANIFEST.json" \
    --out "$1.json" > "$1.txt" 2>&1; cat "$1.txt"; /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$1.txt"; }
W="--domains-from-namelist --emit-initial-history"
run_arm() {  # run_arm NAME KIND(N|O|C|C0) HOURS EXPECTED CAP
  local name=$1 kind=$2 hours=$3 cap; cap=$(cap_for "$4" "$5") || { log "SKIP $name: not enough session time"; return 9; }
  mkdir -p "$R/$name"
  case $kind in
    N) check1 "$R/$name/check_before" > /dev/null || { log "$name: P check before FAILED"; return 99; } ;;
    O) check0 "$R/$name/check_before" > /dev/null || { log "$name: v0.4.0 P check before FAILED: stop"; return 99; } ;;
  esac
  [[ $kind != C0 ]] && { core_gate "$name"; printf '%s %s\n' "w41-bundle041 ATTRIB $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"; }
  samplers_on "$R/$name"; host_load "before $name"
  log "ARM $name kind $kind ${hours} h cap ${cap}s"
  local k=$kind; [ -n "$DRY" ] && k=DRY
  case $k in
    DRY) in_bg systemd-run --user --scope --quiet --unit="$PREFIX-$name" -p RuntimeMaxSec="$cap" -- sleep "$DRY" > "$R/$name.scope.log" 2>&1 ;;
    N|O) local b=$P; [[ $kind == O ]] && b=$P0
      in_bg systemd-run --user --scope --quiet --unit="$PREFIX-$name" -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec="$cap" -- \
        bash "$P/tools/bundle_consumer.sh" --bundle "$b" --layer "$LAY/$name" -- \
        bash "$HERE/arm.sh" "$b/bundle.env" "$name" "$CASE" "$R/$name" $((cap - 90)) -- $W --hours "$hours" > "$R/$name.scope.log" 2>&1 ;;
    C|C0)
      in_bg systemd-run --user --scope --quiet --unit="$PREFIX-$name" -p MemoryMax=20G -p MemorySwapMax=0 -p RuntimeMaxSec="$cap" -- \
        bash "$HERE/arm.sh" "$A/c.env" "$name" "$CASE" "$R/$name" $((cap - 30)) -- $W --hours "$hours" > "$R/$name.scope.log" 2>&1 ;;
  esac
  local rc=$?; samplers_off; quiet_off; host_load "after $name"
  case $kind in
    N) check1 "$R/$name/check_after" > /dev/null || { log "$name: P NOT OK after: stop"; return 99; } ;;
    O) check0 "$R/$name/check_after" > /dev/null || { log "$name: v0.4.0 P NOT OK after: stop"; return 99; } ;;
  esac
  log "ARM $name rc=$rc $(/usr/bin/grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1 | cut -c1-120)"
  return $rc
}
run_arm C0 C0 3 900 1500 || { log "C0 failed: no attribution arms"; exit 5; }
for spec in "N1 N" "O1 O" "C1 C" "N2 N" "O2 O" "C2 C"; do
  set -- $spec
  run_arm "$1" "$2" 24 200 600; rc=$?
  (( rc == 99 )) && exit 7
done
log "DONE"
