#!/usr/bin/env bash
# w44-rc v0.4.4 SHORT session payload (entry_rw_short.sh / cuda_mini_short.sh; inside with_gpu_lock.sh on the GPU):
# PRIME the bundle P (cold compiles, autotune seeded from v0.4.3) -> SEAL (compile_bundle seal + check) -> REPLAY arms through
# P/tools/bundle_consumer.sh (fresh layer per arm, P never written, compile_bundle check before/after) -> final check.
# Derived from the v0.4.3 session.sh (w43-release, 2026-10-21: 10/10 arms rc 0, 80 GPU-min). Changes: arm table file
# (W44R_ARMS: grid/length), no systemd scopes and no GNU timeout (arm_timer.py caps each arm inside the entry's single
# setsid group and the entry's RAM-slot cgroup), per-arm <DATA_ROOT> floor check, receipts + wrfout deletion per arm
# (arm_receipt.py), WN3 primed with 3 h (the 72 h is the separate RW3 entry), no icloud=0 D6 (CPU twin deleted).
# Env from the entry: W44R_RUN W44R_ENVFILE W44R_ARMS W44R_SESSION_DEADLINE W44R_RECEIPTS W44R_MODE(gpu|smoke)
#   smoke only: W44R_SMOKE_DURATION W44R_SMOKE_REPLAY_BUNDLE (used only when the CPU seal refuses)
# Arm table rows: PHASE|NAME|INPUT_KEY|EXPECTED_S|CAP_S|OUT_GIB|KEEP_WRFOUT|SMOKE_MODE|CLI ARGS   (PHASE PRIME|SEAL|REPLAY)
set -uo pipefail
exec < /dev/null   # no child may read the arm table or a terminal
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/lib.sh
: "${W44R_RUN:?}" "${W44R_ENVFILE:?}" "${W44R_ARMS:?}" "${W44R_SESSION_DEADLINE:?}" "${W44R_RECEIPTS:?}" "${W44R_MODE:?}"
source "$W44R_ENVFILE"
export W44R_CPU_SWISS_INPUT=$W44R_IN_SWISS
export W44R_RUN=$W44R_SCRATCH/session_${W44R_PIN:0:12}_${W44R_MODE}_v5
mkdir -p "$W44R_RUN"
STAGE=warm
[[ $W44R_ENTRY_NAME == prime_* ]] && STAGE=prime
SMOKE=0; [[ $W44R_MODE == smoke ]] && SMOKE=1
[[ $SMOKE == 1 || ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if w44_quiet; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
P=$W44R_P; R=$W44R_RUN/runs; LAY=$W44R_RUN/layers; W44R_LOG=$W44R_RUN/session.log
export W44R_TMP_ROOT=$W44R_RUN/tmp
mkdir -p "$R" "$LAY" "$W44R_TMP_ROOT" "$W44R_RECEIPTS"
# release defaults: no inherited model flag, XLA_FLAGS, allocator knob or CUDA_ROOT (E16, E107); lock vars stay
for v in $(compgen -e | grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
export JAX_PLATFORMS=cpu
DEADLINE=$W44R_SESSION_DEADLINE
T0=$(date +%s)
w44_log "START session_short pin $W44R_PIN bundle $P mode $W44R_MODE arms $W44R_ARMS deadline $(date -u -d @"$DEADLINE" +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}')"
[[ $SMOKE == 1 ]] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader | sed 's/^/  gpu free,used: /' | tee -a "$W44R_LOG"

child=
SAMPLERS=()
samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
on_signal() { trap - EXIT; w44_log "signal: stopping the current arm"; [ -n "$child" ] && { kill -TERM "$child" 2>/dev/null; wait "$child"; }
  samplers_off; w44_log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) min"; exit 143; }
on_exit() { local rc=$?; samplers_off; w44_log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) min"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() { "$@" & child=$!; wait "$child"; local rc=$?; child=; return $rc; }
cap_for() {  # cap_for EXPECTED_S CAP_S -> cap, or fail if 1.3 x EXPECTED does not fit before the deadline (CPU smoke: x factor)
  local left=$(( DEADLINE - $(date +%s) )) f=1 exp cap
  [[ $SMOKE == 1 ]] && f=${W44R_SMOKE_CAP_FACTOR:?}
  exp=$(( $1 * f )); cap=$(( $2 * f ))
  (( left >= exp * 13 / 10 )) || return 1
  (( left < cap )) && echo "$left" || echo "$cap"
}
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg) free_gib_data $(w44_free_gib <DATA_ROOT>)"
    ps -eo psr,pcpu,rss,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-180; } >> "$W44R_RUN/HOST_LOAD.txt"; }
samplers_on() {  # per-process + framebuffer VRAM every 0.5 s (GPU only; a CPU smoke never calls nvidia-smi)
  [[ $SMOKE == 1 ]] && { echo "smoke: no nvidia-smi sampler" > "$1/vram_apps.csv"; return 0; }
  nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
  SAMPLERS+=($!)
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$1/vram_fb.csv" 2>/dev/null &
  SAMPLERS+=($!)
}
check() {  # check BUNDLE OUT_PREFIX -> stdout line; 0 iff BUNDLE_OK missing=0 changed=0 new=0
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 "$W44R_PY" -I "$1/tools/compile_bundle.py" check --cache "$1/jax" \
    --manifest "$1/sealed/BUNDLE_MANIFEST.json" --out "$2.json" > "$2.txt" 2>&1
  local crc=$?; (( crc == 0 )) || return "$crc"
  tail -1 "$2.txt"; grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$2.txt"
}
receipt() {  # receipt NAME KEEP
  local del=(); [[ $2 == 0 && "$(cat "$R/$1/rc.txt" 2>/dev/null)" == 0 ]] && del=(--delete-wrfout)
  JAX_PLATFORMS=cpu "$W44R_PY" -I "$W44R_TOOLS/arm_receipt.py" "$W44R_RUN" "$R/$1" "$R/$1/timer.json" "$W44R_RECEIPTS" "${del[@]}" \
    >> "$W44R_LOG" 2>&1
  local rrc=$?; (( rrc == 0 )) || w44_log "receipt $1 FAILED rc=$rrc"
  return "$rrc"
}
run_arm() {  # run_arm NAME CAP SMODE -- CMD... (arm_timer, output to R/NAME.scope.log; same file name as v0.4.3)
  local name=$1 cap=$2 smode=$3; shift 4
  if [[ $SMOKE == 1 ]]; then export W44R_SMOKE_ARM_MODE=$smode; fi
  in_bg "$W44R_PY" -I "$W44R_TOOLS/arm_timer.py" --seconds "$cap" --receipt "$R/$name/timer.json" -- "$@" > "$R/$name.scope.log" 2>&1
}
input_of() { local v="W44R_IN_$1"; echo "${!v:?unknown input key $1}"; }

prime_arm() {  # prime_arm NAME KEY EXP CAP GIB KEEP SMODE ARGS...
  local name=$1 input cap; input=$(input_of "$2") || return 8
  cap=$(cap_for "$3" "$4") || { w44_log "SKIP prime $name: not enough session time"; return 9; }
  w44_disk_ok "$5" || { w44_log "SKIP prime $name: <DATA_ROOT> would fall below the floor"; return 95; }
  [ -e "$R/$name" ] && { w44_log "prime $name: $R/$name exists (E42): refuse"; return 8; }
  local keep=$6 smode=$7; shift 7
  mkdir -p "$R/$name"; samplers_on "$R/$name"; host_load "before $name"
  w44_log "PRIME $name cap ${cap}s input $input args: $*"
  run_arm "$name" "$cap" "$smode" -- bash "$W44R_TOOLS/arm.sh" "$P/bundle.env" "$name" "$input" "$R/$name" -- "$@"
  local rc=$?; samplers_off; host_load "after $name"; receipt "$name" "$keep"; local rrc=$?
  (( rc == 0 && rrc != 0 )) && rc=$rrc
  if (( rc == 0 )); then
    env JAX_PLATFORMS=cpu "$W44R_PY" "$W44R_TOOLS/prime_record.py" "$P" "$R/$name" "$W44R_PIN" "$W44R_MODE" "$input" || return 7
  fi
  w44_log "PRIME $name rc=$rc"; return $rc
}
replay_arm() {  # replay_arm NAME KEY EXP CAP GIB KEEP SMODE ARGS...   (bundle = $RB)
  local name=$1 input cap; input=$(input_of "$2") || return 8
  cap=$(cap_for "$3" "$4") || { w44_log "SKIP replay $name: not enough session time"; return 9; }
  w44_disk_ok "$5" || { w44_log "SKIP replay $name: <DATA_ROOT> would fall below the floor"; return 95; }
  [ -e "$R/$name" ] || [ -e "$LAY/$name" ] && { w44_log "replay $name: run dir or layer exists: refuse"; return 8; }
  local keep=$6 smode=$7; shift 7
  mkdir -p "$R/$name"
  local before; before=$(check "$RB" "$R/$name/check_before") || { w44_log "replay $name: check before FAILED ($before): skip"; return 7; }
  samplers_on "$R/$name"; host_load "before $name"
  w44_log "REPLAY $name bundle $RB cap ${cap}s input $input check before: $before args: $*"
  run_arm "$name" "$cap" "$smode" -- bash "$RB/tools/bundle_consumer.sh" --bundle "$RB" --layer "$LAY/$name" -- \
    bash "$W44R_TOOLS/arm.sh" "$RB/bundle.env" "$name" "$input" "$R/$name" -- "$@"
  local rc=$?; samplers_off; host_load "after $name"
  local after; after=$(check "$RB" "$R/$name/check_after"); local crc=$?
  receipt "$name" "$keep"; local rrc=$?
  (( rc == 0 && rrc != 0 )) && rc=$rrc
  w44_log "REPLAY $name rc=$rc consumer: $(grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1 | cut -c1-200) check after: $after"
  (( crc == 0 )) || { w44_log "replay $name: BUNDLE $RB NOT OK after the arm: stop session"; return 99; }
  return $rc
}

PRIME_OK=1; SEALED=0; PRIME_INPUTS=()
RB=$P
if [[ $W44R_ENTRY_NAME == prime_swiss ]]; then
  [ -n "$(ls -A "$P/jax/aot" 2>/dev/null)" ] && { w44_log "P/jax/aot already primed: refuse"; exit 4; }
fi
if [[ $STAGE == warm ]]; then
  for key in SWISS PROD WN3_0227 WN3_0227_ICLOUD0; do PRIME_INPUTS+=(--input "$(input_of "$key")"); done
fi
(cd "$P/jax/xla_gpu_per_fusion_autotune_cache_dir" && find . -type f | sort | xargs sha256sum) > "$W44R_RUN/autotune_before_prime.sha256"
w44_log "autotune before PRIME: $(grep -c . "$W44R_RUN/autotune_before_prime.sha256") files (v0.4.3 seed verified by manifest_check)"
while IFS='|' read -r -u 3 phase name key exp cap gib keep smode args; do
  [[ -z $phase || $phase == \#* ]] && continue
  case $phase in
    PRIME)
      [[ $PRIME_OK == 1 ]] || continue
      # shellcheck disable=SC2086
      prime_arm "$name" "$key" "$exp" "$cap" "$gib" "$keep" "$smode" $args || PRIME_OK=0
      PRIME_INPUTS+=(--input "$(input_of "$key")") ;;
    SEAL)
      [[ $PRIME_OK == 1 ]] || { w44_log "PRIME incomplete: no seal, no replay"; exit 5; }
      [ ! -e "$P/sealed" ] || { w44_log "P/sealed exists: refuse"; exit 4; }
      w44_disk_ok "$gib" || { w44_log "SEAL refused: <DATA_ROOT> free - ${gib} GiB < $W44R_DATA_FLOOR_GIB GiB"; exit 95; }
      w44_log "SEAL start (source $W44R_PSRC)"
      in_bg env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 "$W44R_PY" -I "$W44R_PSRC/scripts/release/compile_bundle.py" seal --cache "$P/jax" \
        --commit "$W44R_PIN" --source "$W44R_PSRC" "${PRIME_INPUTS[@]}" --out "$P/sealed" > "$W44R_RUN/seal.log" 2>&1
      src=$?; cp "$W44R_RUN/seal.log" "$P/meta/seal.log"; w44_log "SEAL rc=$src $(tail -1 "$W44R_RUN/seal.log" | cut -c1-300)"
      if (( src == 0 )); then
        out=$(check "$P" "$W44R_RUN/check_after_seal") || { w44_log "check after seal FAILED: $out"; exit 6; }
        cp "$W44R_RUN/check_after_seal".{json,txt} "$P/meta/"; SEALED=1; w44_log "check after seal: $out"
      elif [[ $SMOKE == 1 && -n ${W44R_SMOKE_REPLAY_BUNDLE:-} ]]; then
        RB=$W44R_SMOKE_REPLAY_BUNDLE; w44_log "SMOKE: seal refused on the CPU cache; replays exercise the consumer builder on $RB (read-only)"
      else exit 6; fi ;;
    REPLAY|PROFILE)
      [[ $phase == PROFILE ]] && export W44R_PROFILE=1 || unset W44R_PROFILE
      [[ $SEALED == 1 || $RB != "$P" ]] || { w44_log "no sealed manifest: skip $name"; continue; }
      # shellcheck disable=SC2086
      replay_arm "$name" "$key" "$exp" "$cap" "$gib" "$keep" "$smode" $args; rc=$?
      (( rc == 0 )) || exit "$rc" ;;
    *) w44_log "bad arm table row: $phase"; exit 2 ;;
  esac
done 3< "$W44R_ARMS"
if [[ $SEALED == 1 ]]; then
  final=$(check "$P" "$W44R_RUN/final_check_P"); frc=$?; w44_log "final check P rc=$frc: $final"
  (( frc == 0 )) || exit "$frc"
fi
w44_log "DONE session_short $(( ($(date +%s) - T0 + 59) / 60 )) min prime_ok=$PRIME_OK sealed=$SEALED"
[[ $PRIME_OK == 1 && ( $SEALED == 1 || $SMOKE == 1 || $STAGE == prime ) ]] || exit 5
exit 0
