#!/usr/bin/env bash
# w41-bundle041: prime + seal the v0.4.1 compile bundle P and run the real-world gate RW1-RW4 in ONE GPU lock session.
# usage: scripts/with_gpu_lock.sh --label w41-bundle041 -- bash session.sh      (setup.sh staged P, S and the cases)
# Program = clean detached P/src at the manager's BUNDLE PIN (source freeze); main:src must still equal it. Release
# defaults: arm.sh refuses any GPUWRF_* model flag. Order (manager plan 11:15Z, approved 11:15:41Z incl. B7):
#   PRIME (cold, writes P/jax): A1 Swiss d01 3 h, A2 PROD 2-dom 3 h, A3 WN3 0227 3-nest 72 h --emit-initial-history (= G2 C0
#         config: C0 namelist == WN3 namelist except dates); WN3 last -> its keys are the last_cheap_key prefetch hints
#   SEAL  (CPU, in the lock): P/src/scripts/release/compile_bundle.py seal -> P/sealed; check -> BUNDLE_OK new=0
#   REPLAY via P/tools/bundle_consumer.sh (fresh layer per arm, P never written), compile_bundle check before/after:
#         B2 0502 3 h, B4 0227 48 h-INPUT 3 h, B5 Swiss 3 h, B6 PROD 3 h, B3 0614 48 h (G2 length),
#         B1 0227 72 h (RW1 bytes == A3, RW3 timing, QUIET), B7 v0.4.0 bundle 0227 72 h through the same consumer with a
#         layer outside v0.4.0 P (P0 BUNDLE_OK missing=0 changed=0 new=0 before AND after, else abort B7; QUIET)
# Budget: DEADLINE = min(start + 150 GPU-min, 18:43Z); an arm runs only if 1.3 x its expected wall fits, its
# RuntimeMaxSec is cut to the deadline. Each arm in its own systemd scope (MemoryMax 20G, no swap); TERM/INT/HUP or any
# exit stops every scope of this session; the session-owned QUIET file is removed by the same traps.
# BS_DRY_ARM=<s>: CPU lifecycle proof (arms sleep, no GPU, no seal); needs BS_QUIET_FILE.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=104c97de98b45b790058d3cdad6e0ff1e88a9f82; TREE=319848e91a5ecaf5516d0a783bad11447c349049
P=${BS_P:-<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1}; P0=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
S=${BS_S:-<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041}; C=<DATA_ROOT>/wrf_gpu2/scratch/w41-bundle041/cases
R=$S/runs; LAY=$S/layers; LOG=$S/session.log
SWISS=<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input; PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
DRY=${BS_DRY_ARM:-}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
QUIET_FILE=${BS_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || "$QUIET_FILE" != /tmp/wrf_gpu2_quiet ]] || { echo "dry run needs BS_QUIET_FILE"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
mkdir -p "$R" "$LAY"
# release defaults and the bundle's AOT target: no inherited model flag, XLA_FLAGS, allocator knob or CUDA_ROOT (E16, E107)
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
MAILBOX=${BS_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
post() { [ -n "$DRY" ] && [ -z "${BS_MAILBOX:-}" ] && return 0; printf '%s w41-bundle041: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
if [ -z "$DRY" ]; then
  [[ "$(git -C "$P/src" rev-parse HEAD)" == "$PIN" && "$(git -C "$P/src" rev-parse HEAD:src)" == "$TREE" ]] || { log "P/src moved"; exit 4; }
  [[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src)" ]] || { log "P/src dirty"; exit 4; }
  [[ "$(git -C <USER_HOME>/src/wrf_gpu2 rev-parse main:src)" == "$TREE" ]] || { log "main:src != pin src (source freeze broken): refuse"; exit 4; }
  test -e "$P/src/data/wrf_pristine" || { log "missing P/src/data/wrf_pristine"; exit 4; }
  cmp -s "$P/bundle.env" "$HERE/bundle.env.expected" || { log "P/bundle.env != staged copy"; exit 4; }
fi
PHASES=" ${BS_PHASES:-PRIME SEAL REPLAY} "
T0=$(date +%s)
END_UTC=$(date -u -d "${BS_END_UTC:-2026-10-18T18:43:00Z}" +%s)
DEADLINE=$(( T0 + ${BS_BUDGET_S:-9000} )); (( END_UTC < DEADLINE )) && DEADLINE=$END_UTC
log "START pin $PIN src $TREE phases[$PHASES] deadline $(date -u -d @$DEADLINE +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') dry='$DRY'"
post "GPU START v0.4.1 bundle prime+seal + real-world gate RW1-RW4 (pin ${PIN:0:12}, src ${TREE:0:12}; one lock session, cap $(( (DEADLINE - T0) / 60 )) GPU-min, ends by $(date -u -d @$DEADLINE +%TZ))"
[ -n "$DRY" ] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader | sed 's/^/  gpu free,used: /' | tee -a "$LOG"

PREFIX="w41bundle041-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() {
  local u n=0
  while [ -n "$(session_scopes)" ]; do
    for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && { log "scopes still alive: $(session_scopes | tr '\n' ' ')"; sleep 10; }
  done
}
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-bundle041 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=()
samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill "$p" 2>/dev/null; done; SAMPLERS=(); }
child=
on_signal() { trap - EXIT; log "signal: stopping session scopes"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  stop_scopes; samplers_off; quiet_off; log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"
  post "GPU END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min; session log $LOG"; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; quiet_off
  log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"
  post "GPU END rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); session log $LOG; CPU follow-up (E41, RW3 D6/integrity, verdict table) next"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() { "$@" & child=$!; wait "$child"; local rc=$?; child=; return $rc; }
cap_for() {  # cap_for EXPECTED_S CAP_S -> RuntimeMaxSec, or fail if 1.3 x EXPECTED does not fit before the deadline
  local left=$(( DEADLINE - $(date +%s) ))
  (( left >= $1 * 13 / 10 )) || return 1
  (( left < $2 )) && echo "$left" || echo "$2"
}
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -20 | cut -c1-180; } >> "$S/HOST_LOAD.txt"; }
samplers_on() {  # per-process and framebuffer VRAM, 0.5 s
  [ -n "$DRY" ] && return 0
  nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
  SAMPLERS+=($!)
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$1/vram_fb.csv" 2>/dev/null &
  SAMPLERS+=($!)
}
scope() {  # scope NAME CAP CMD... (output to $R/NAME.scope.log)
  local name=$1 cap=$2; shift 2
  in_bg systemd-run --user --scope --quiet --unit="$PREFIX-$name" -p MemoryMax=20G -p MemorySwapMax=0 \
    -p RuntimeMaxSec="$cap" -- "$@" > "$R/$name.scope.log" 2>&1
}
arm_cmd() {  # arm_cmd ENVFILE NAME INPUT BUDGET ARGS... -> the arm.sh command (or sleep when dry)
  local envf=$1 name=$2 input=$3 budget=$4; shift 4
  if [ -n "$DRY" ]; then echo sleep "$DRY"; else echo bash "$HERE/arm.sh" "$envf" "$name" "$input" "$R/$name" "$budget" -- "$@"; fi
}
check() {  # check BUNDLE OUT_PREFIX -> its stdout line; 0 iff BUNDLE_OK missing=0 changed=0 new=0
  [ -n "$DRY" ] && { echo "BUNDLE_OK dry"; return 0; }
  JAX_PLATFORMS=cpu python3 "$1/tools/compile_bundle.py" check --cache "$1/jax" --manifest "$1/sealed/BUNDLE_MANIFEST.json" \
    --out "$2.json" > "$2.txt" 2>&1
  cat "$2.txt"; /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$2.txt"
}

prime_arm() {  # prime_arm NAME INPUT EXPECTED_S CAP_S -- CLI_ARGS...
  local name=$1 input=$2 cap; cap=$(cap_for "$3" "$4") || { log "SKIP prime $name: not enough session time"; return 9; }
  shift 5
  [ -e "$R/$name" ] && { log "prime $name: $R/$name exists (E42): refuse"; return 8; }
  mkdir -p "$R/$name"; samplers_on "$R/$name"; host_load "before $name"
  log "PRIME $name cap ${cap}s args: $*"
  # shellcheck disable=SC2046
  scope "$name" "$cap" $(arm_cmd "$P/bundle.env" "$name" "$input" $((cap - 30)) "$@")
  local rc=$?; samplers_off; host_load "after $name"; log "PRIME $name rc=$rc"; return $rc
}
replay_arm() {  # replay_arm NAME BUNDLE INPUT EXPECTED_S CAP_S QUIET_LABEL -- CLI_ARGS...
  local name=$1 bundle=$2 input=$3 quiet=$6 cap; cap=$(cap_for "$4" "$5") || { log "SKIP replay $name: not enough session time"; return 9; }
  shift 7
  [ -e "$R/$name" ] || [ -e "$LAY/$name" ] && { log "replay $name: run dir or layer exists: refuse"; return 8; }
  mkdir -p "$R/$name"
  local before; before=$(check "$bundle" "$R/$name/check_before") || { log "replay $name: check before FAILED ($before): skip"; return 7; }
  log "REPLAY $name bundle $bundle check before: $before"
  if [ -e /tmp/wrf_gpu2_quiet ] && ! /usr/bin/grep -q '^w41-bundle041 ' /tmp/wrf_gpu2_quiet; then log "foreign QUIET during session: $(cat /tmp/wrf_gpu2_quiet)"; fi
  [ -n "$quiet" ] && [ ! -e "$QUIET_FILE" ] && printf '%s %s\n' "$quiet" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  samplers_on "$R/$name"; host_load "before $name"
  log "REPLAY $name cap ${cap}s quiet='$quiet' args: $*"
  # shellcheck disable=SC2046
  if [ -n "$DRY" ]; then scope "$name" "$cap" sleep "$DRY"; else
  scope "$name" "$cap" bash "$P/tools/bundle_consumer.sh" --bundle "$bundle" --layer "$LAY/$name" -- \
    $(arm_cmd "$bundle/bundle.env" "$name" "$input" $((cap - 90)) "$@"); fi
  local rc=$?; samplers_off; quiet_off; host_load "after $name"
  local after; after=$(check "$bundle" "$R/$name/check_after"); local crc=$?
  log "REPLAY $name rc=$rc consumer: $(/usr/bin/grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1) check after: $after"
  (( crc == 0 )) || { log "replay $name: BUNDLE $bundle NOT OK after the arm: stop session"; return 99; }
  return $rc
}

PRIME_OK=1
if [[ $PHASES == *" PRIME "* ]]; then
  [ -n "$DRY" ] || [ ! -e "$P/jax/aot" ] || { log "P/jax already primed: refuse PRIME"; exit 4; }
  prime_arm A1_swiss "$SWISS" 400 900 -- --domains-from-namelist --hours 3 || PRIME_OK=
  [ -n "$PRIME_OK" ] && { prime_arm A2_prod "$PROD" 800 1500 -- --domains-from-namelist --hours 3 || PRIME_OK=; }
  [ -n "$PRIME_OK" ] && { prime_arm A3_wn3_0227_72h "$C/wn3_0227" 1500 2700 -- --domains-from-namelist --emit-initial-history --hours 72 || PRIME_OK=; }
  [ -n "$PRIME_OK" ] || { log "PRIME incomplete: no seal, no replay"; exit 5; }
fi
if [[ $PHASES == *" SEAL "* ]]; then
  if [ -n "$DRY" ]; then log "dry: seal skipped"; else
    [ ! -e "$P/sealed" ] || { log "P/sealed exists: refuse"; exit 4; }
    log "SEAL start"
    JAX_PLATFORMS=cpu python3 "$P/src/scripts/release/compile_bundle.py" seal --cache "$P/jax" --commit "$PIN" --source "$P/src" \
      --input "$SWISS" --input "$PROD" --input "$C/wn3_0227" --out "$P/sealed" > "$P/meta/seal.log" 2>&1
    src=$?; log "SEAL rc=$src $(tail -1 "$P/meta/seal.log")"; (( src == 0 )) || exit 6
    out=$(check "$P" "$P/meta/check_after_seal") || { log "check after seal FAILED: $out"; exit 6; }
    log "check after seal: $out"
  fi
fi
if [[ $PHASES == *" REPLAY "* ]]; then
  [ -n "$DRY" ] || test -s "$P/sealed/BUNDLE_MANIFEST.json" || { log "no sealed manifest"; exit 6; }
  W="--domains-from-namelist --emit-initial-history"
  # shellcheck disable=SC2086
  for spec in "B2_wn3_0502_3h|$C/wn3_0502|90|600||$W --hours 3" \
              "B4_wn3_0227in48h_3h|$C/wn3_0227_48h|90|600||$W --hours 3" \
              "B5_swiss_3h|$SWISS|60|400||--domains-from-namelist --hours 3" \
              "B6_prod_3h|$PROD|120|600||--domains-from-namelist --hours 3" \
              "B3_wn3_0614_48h|$C/wn3_0614|500|1200||$W --hours 48" \
              "B1_wn3_0227_72h|$C/wn3_0227|750|1500|w41-bundle041 RW3 timed B1 (v0.4.1 bundle, WN3 0227 72 h)|$W --hours 72"; do
    IFS='|' read -r name input exp cap quiet args <<< "$spec"
    replay_arm "$name" "$P" "$input" "$exp" "$cap" "$quiet" -- $args; rc=$?
    (( rc == 99 )) && exit 7
  done
  # B7: last-release baseline for RW3 (v0.4.0 bundle READ-ONLY through the consumer; manager 11:15:41Z)
  # shellcheck disable=SC2086
  replay_arm B7_v040_wn3_0227_72h "$P0" "$C/wn3_0227" 750 1500 "w41-bundle041 RW3 timed B7 (v0.4.0 bundle, WN3 0227 72 h)" \
    -- $W --hours 72; rc=$?
  (( rc == 99 )) && exit 7
  final=$(check "$P" "$S/final_check_P"); log "final check P: $final"
  final0=$(check "$P0" "$S/final_check_P0"); log "final check v0.4.0 P: $final0"
fi
log "DONE $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"
