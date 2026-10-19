#!/usr/bin/env bash
# w42-bundle042: prime + seal the v0.4.2 compile bundle P and run the real-world gate arms RW1/RW2/RW4 (phase 1 of the ONE
# GPU lock session; gpu_session.sh runs rw3t.sh = RW3 interleaved timing after it). Derived from the frozen v0.4.1
# w41-bundle041 session.sh (11:36Z run: 10/10 arms rc 0, 61 GPU-min); changes: v0.4.2 paths/labels, PIN/TREE from
# pin.env (manager BUNDLE PIN), optional icloud=0 arms (A4 prime, B8 cross-date replay; BS_ICLOUD0=1 in pin.env when
# w42-icloud0 is in the pin), no B1/B7 (rw3t.sh: N72 = the 72 h warm replay, N/O interleaved + O72), deadline from the
# wrapper (BS_DEADLINE_EPOCH), GPU START/END posts by the wrapper.
# usage (inside gpu_session.sh): BS_DEADLINE_EPOCH=<s> bash session.sh
# Order: PRIME (cold, writes P/jax): A1 Swiss d01 3 h, A2 PROD 2-dom 3 h, [A4 WN3 0227 icloud=0 24 h = D6 arm,] A3 WN3 0227 3-nest
#        72 h --emit-initial-history (= G2 C0 config superset; WN3 default last -> its keys are the last_cheap_key hints)
#        SEAL (CPU, in the lock): P/src/scripts/release/compile_bundle.py seal -> P/sealed; check -> BUNDLE_OK new=0
#        REPLAY via P/tools/bundle_consumer.sh (fresh layer per arm, P never written), compile_bundle check before/after:
#        B2 0502 3 h, B4 0227 48 h-INPUT 3 h, B5 Swiss 3 h, B6 PROD 3 h, [B8 0502 icloud=0 3 h,] B3 0614 48 h (G2 length).
#        The 0227 72 h warm replay (RW1 bytes == A3, RW3 D6 input and 72 h timing) is rw3t.sh's watched slot N72.
# An arm runs only if 1.3 x its expected wall fits before the deadline; its RuntimeMaxSec is cut to the deadline. Each arm in
# its own systemd scope (MemoryMax 20G, no swap); TERM/INT/HUP or any exit stops every scope of this session; the
# session-owned QUIET file is removed by the same traps.
# BS_DRY_ARM=<s>: CPU lifecycle proof (arms sleep, no GPU, no seal); needs BS_QUIET_FILE, BS_P, BS_S.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/pin.env"   # PIN TREE BS_ICLOUD0 (written once by setup when the manager posts BUNDLE PIN)
P=${BS_P:-<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2}
S=${BS_S:-<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/gpu}; C=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/cases
R=$S/runs; LAY=$S/layers; LOG=$S/session.log
SWISS=<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input; PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
DRY=${BS_DRY_ARM:-}
ICLOUD0=${BS_ICLOUD0:-0}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
QUIET_FILE=${BS_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || "$QUIET_FILE" != /tmp/wrf_gpu2_quiet ]] || { echo "dry run needs BS_QUIET_FILE"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
[ -n "${BS_DEADLINE_EPOCH:-}" ] || { echo "BS_DEADLINE_EPOCH (from gpu_session.sh) required"; exit 2; }
mkdir -p "$R" "$LAY"
# release defaults and the bundle's AOT target: no inherited model flag, XLA_FLAGS, allocator knob or CUDA_ROOT (E16, E107)
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if [ -z "$DRY" ]; then
  [[ "$(git -C "$P/src" rev-parse HEAD)" == "$PIN" && "$(git -C "$P/src" rev-parse HEAD:src)" == "$TREE" ]] || { log "P/src moved"; exit 4; }
  [[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src)" ]] || { log "P/src dirty"; exit 4; }
  [[ "$(git -C <USER_HOME>/src/wrf_gpu2 rev-parse main:src)" == "$TREE" ]] || { log "main:src != pin src (source freeze broken): refuse"; exit 4; }
  test -e "$P/src/data/wrf_pristine" || { log "missing P/src/data/wrf_pristine"; exit 4; }
  cmp -s "$P/bundle.env" "$HERE/bundle.env.expected" || { log "P/bundle.env != staged copy"; exit 4; }
  (cd "$C" && sha256sum -c --quiet "$HERE/cases_sha256.txt") || { log "case inputs changed (cases_sha256.txt)"; exit 4; }
  if [[ $ICLOUD0 == 1 ]]; then   # w42-icloud0 BUNDLE_ARM_SPEC inputs (absolute paths, written by setup.sh)
    sha256sum -c --quiet "$HERE/ic_sha256.txt" || { log "icloud0 spec inputs changed (ic_sha256.txt)"; exit 4; }
  fi
fi
# icloud=0 WN3 arm (manager 06:05:46Z: final-candidate repeat of w42-icloud0's D6 vs its CPU-WRF icloud=0 twin; spec
# proofs/w42_icloud0/BUNDLE_ARM_SPEC.md via pin.env: ICLOUD0_INPUT = its CASE/input, ICLOUD0_HOURS 24) - primes the
# icloud=0 programs too; the spec's CLI args are exactly W below + --hours 24 (release defaults, bundle env recorded)
IC_IN=${ICLOUD0_INPUT:-$C/wn3_0227_icloud0}; IC_H=${ICLOUD0_HOURS:-24}
PHASES=" ${BS_PHASES:-PRIME SEAL REPLAY} "
T0=$(date +%s)
DEADLINE=$BS_DEADLINE_EPOCH
log "START pin $PIN src $TREE icloud0=$ICLOUD0 phases[$PHASES] deadline $(date -u -d @$DEADLINE +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') dry='$DRY'"
[ -n "$DRY" ] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader | sed 's/^/  gpu free,used: /' | tee -a "$LOG"

PREFIX="w42bundle042-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() {
  local u n=0
  while [ -n "$(session_scopes)" ]; do
    for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && { log "scopes still alive: $(session_scopes | tr '\n' ' ')"; sleep 10; }
  done
}
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w42-bundle042 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=()
samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill "$p" 2>/dev/null; done; SAMPLERS=(); }
child=
on_signal() { trap - EXIT; log "signal: stopping session scopes"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  stop_scopes; samplers_off; quiet_off; log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; quiet_off
  log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"; }
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
  log "PRIME $name cap ${cap}s input $input args: $*"
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
  if [ -e /tmp/wrf_gpu2_quiet ] && ! /usr/bin/grep -q '^w42-bundle042 ' /tmp/wrf_gpu2_quiet; then log "foreign QUIET during session: $(cat /tmp/wrf_gpu2_quiet)"; fi
  [ -n "$quiet" ] && [ ! -e "$QUIET_FILE" ] && printf '%s %s\n' "$quiet" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  samplers_on "$R/$name"; host_load "before $name"
  log "REPLAY $name cap ${cap}s input $input quiet='$quiet' args: $*"
  # shellcheck disable=SC2046
  if [ -n "$DRY" ]; then scope "$name" "$cap" sleep "$DRY"; else
  scope "$name" "$cap" bash "$bundle/tools/bundle_consumer.sh" --bundle "$bundle" --layer "$LAY/$name" -- \
    $(arm_cmd "$bundle/bundle.env" "$name" "$input" $((cap - 90)) "$@"); fi
  local rc=$?; samplers_off; quiet_off; host_load "after $name"
  local after; after=$(check "$bundle" "$R/$name/check_after"); local crc=$?
  log "REPLAY $name rc=$rc consumer: $(/usr/bin/grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1) check after: $after"
  (( crc == 0 )) || { log "replay $name: BUNDLE $bundle NOT OK after the arm: stop session"; return 99; }
  return $rc
}

W="--domains-from-namelist --emit-initial-history"
PRIME_OK=1
if [[ $PHASES == *" PRIME "* ]]; then
  [ -n "$DRY" ] || [ ! -e "$P/jax/aot" ] || { log "P/jax already primed: refuse PRIME"; exit 4; }
  prime_arm A1_swiss "$SWISS" 400 900 -- --domains-from-namelist --hours 3 || PRIME_OK=
  [ -n "$PRIME_OK" ] && { prime_arm A2_prod "$PROD" 800 1500 -- --domains-from-namelist --hours 3 || PRIME_OK=; }
  # shellcheck disable=SC2086
  [ -n "$PRIME_OK" ] && [[ $ICLOUD0 == 1 ]] && { prime_arm A4_wn3_0227_icloud0 "$IC_IN" $(( 600 + IC_H * 12 )) $(( 1500 + IC_H * 20 )) -- $W --hours "$IC_H" || PRIME_OK=; }
  # shellcheck disable=SC2086
  [ -n "$PRIME_OK" ] && { prime_arm A3_wn3_0227_72h "$C/wn3_0227" 1500 2700 -- $W --hours 72 || PRIME_OK=; }
  [ -n "$PRIME_OK" ] || { log "PRIME incomplete: no seal, no replay"; exit 5; }
fi
if [[ $PHASES == *" SEAL "* ]]; then
  if [ -n "$DRY" ]; then log "dry: seal skipped"; else
    [ ! -e "$P/sealed" ] || { log "P/sealed exists: refuse"; exit 4; }
    INPUTS=(--input "$SWISS" --input "$PROD" --input "$C/wn3_0227")
    [[ $ICLOUD0 == 1 ]] && INPUTS+=(--input "$IC_IN")
    log "SEAL start"
    JAX_PLATFORMS=cpu python3 "$P/src/scripts/release/compile_bundle.py" seal --cache "$P/jax" --commit "$PIN" --source "$P/src" \
      "${INPUTS[@]}" --out "$P/sealed" > "$P/meta/seal.log" 2>&1
    src=$?; log "SEAL rc=$src $(tail -1 "$P/meta/seal.log")"; (( src == 0 )) || exit 6
    out=$(check "$P" "$P/meta/check_after_seal") || { log "check after seal FAILED: $out"; exit 6; }
    log "check after seal: $out"
  fi
fi
if [[ $PHASES == *" REPLAY "* ]]; then
  [ -n "$DRY" ] || test -s "$P/sealed/BUNDLE_MANIFEST.json" || { log "no sealed manifest"; exit 6; }
  SPECS=("B2_wn3_0502_3h|$C/wn3_0502|90|600||$W --hours 3"
         "B4_wn3_0227in48h_3h|$C/wn3_0227_48h|90|600||$W --hours 3"
         "B5_swiss_3h|$SWISS|60|400||--domains-from-namelist --hours 3"
         "B6_prod_3h|$PROD|120|600||--domains-from-namelist --hours 3")
  [[ $ICLOUD0 == 1 ]] && SPECS+=("B8_wn3_0502_icloud0_3h|$C/wn3_0502_icloud0|90|600||$W --hours 3")
  SPECS+=("B3_wn3_0614_48h|$C/wn3_0614|500|1200||$W --hours 48")
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r name input exp cap quiet args <<< "$spec"
    # shellcheck disable=SC2086
    replay_arm "$name" "$P" "$input" "$exp" "$cap" "$quiet" -- $args; rc=$?
    (( rc == 99 )) && exit 7
  done
  final=$(check "$P" "$S/final_check_P"); log "final check P: $final"
fi
log "DONE phase 1 $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"
