#!/usr/bin/env bash
# w44-rc v0.4.4 RW3 session payload (entry_rw3.sh / cuda_mini_rw3.sh; inside with_gpu_lock.sh on the GPU):
# interleaved warm timing of WN3 0227 3-nest, N = the sealed v0.4.4 bundle vs O = the sealed v0.4.3 bundle (last release),
# both READ-ONLY through their own tools/bundle_consumer.sh with a fresh layer per arm, compile_bundle check before/after
# every arm. N72 first (RW3 D6 72 h input vs the original CPU-WRF + RW1 72 h warm replay), then N1 O1 N2 O2 N3 O3 (24 h),
# then O72 (the matched 72 h timing pair) if the window holds it.
# Port of the frozen v0.4.3 rw3t.sh (abort/repeat rule, pre-gate, core watch, shadow watcher, samplers unchanged; A/B rule =
# v0.4.3 common rule). Changes: slot table file (W44R_SLOTS: case + slots with cap/size/keep), arm_timer.py instead of systemd
# scopes, sched_watch on the entry's RAM-slot cgroup (per-arm deltas), disk device resolved from df (<DATA_ROOT> = nvme1n1 now),
# per-slot <DATA_ROOT> floor check, receipts + wrfout deletion per arm (only N72 keeps its wrfouts for the CPU D6 follow-up).
# Env from the entry: W44R_RUN W44R_ENVFILE W44R_SLOTS W44R_SESSION_DEADLINE W44R_RECEIPTS W44R_MODE(gpu|smoke)
#   smoke only: W44R_SMOKE_DURATION W44R_SMOKE_REPLAY_BUNDLE W44R_SMOKE_QUIET_FILE W44R_SMOKE_MAILBOX W44R_SMOKE_CORES
# Slot table: 'CASE|INPUT_KEY|BASE CLI ARGS' then rows NAME|KIND(N|O)|HOURS|CAP_S|OUT_GIB|KEEP_WRFOUT
set -uo pipefail
exec < /dev/null   # no child may read the arm table or a terminal
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/lib.sh
: "${W44R_RUN:?}" "${W44R_ENVFILE:?}" "${W44R_SLOTS:?}" "${W44R_SESSION_DEADLINE:?}" "${W44R_RECEIPTS:?}" "${W44R_MODE:?}"
source "$W44R_ENVFILE"
export W44R_CPU_SWISS_INPUT=$W44R_IN_SWISS
SMOKE=0; [[ $W44R_MODE == smoke ]] && SMOKE=1
[[ $SMOKE == 1 || ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
A=$W44R_RUN; R=$A/runs; LAY=$A/layers; W44R_LOG=$A/session.log
export W44R_TMP_ROOT=$A/tmp
mkdir -p "$R" "$LAY" "$W44R_TMP_ROOT" "$W44R_RECEIPTS"
for v in $(compgen -e | grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
export JAX_PLATFORMS=cpu
P=$W44R_P; P0=$W44R_P_OLD
if [ ! -s "$P/sealed/BUNDLE_MANIFEST.json" ]; then
  [[ $SMOKE == 1 && -n ${W44R_SMOKE_REPLAY_BUNDLE:-} ]] || { echo "N bundle $P not sealed: run the short entry first"; exit 4; }
  P=$W44R_SMOKE_REPLAY_BUNDLE   # CPU smoke only: the stand-in bundle could not be sealed on CPU (documented delta)
fi
IFS='|' read -r tag CASE_KEY BASE_ARGS < <(grep -m1 '^CASE|' "$W44R_SLOTS")
[[ $tag == CASE ]] || { echo "slot table without CASE row"; exit 2; }
v="W44R_IN_$CASE_KEY"; CASE=${!v:?unknown case key}
CORES=3,13,19,29; WATCH_CPU=8; SAMPLE_CPU=12; QGATE_S=60; GATE_MAX_S=180; RULE_P=100; RULE_A=200
QUIET_FILE=/tmp/wrf_gpu2_quiet; MAILBOX=$W44R_MAILBOX
if [[ $SMOKE == 1 ]]; then   # CPU smoke: never write the global QUIET flag, never read the real MAILBOX, watch the smoke cores
  QUIET_FILE=${W44R_SMOKE_QUIET_FILE:?}; MAILBOX=${W44R_SMOKE_MAILBOX:?}; CORES=${W44R_SMOKE_CORES:?}; RULE_P=1000000; RULE_A=1000000
fi
DATA_DEV=$(lsblk -no PKNAME "$(df --output=source <DATA_ROOT> | tail -1)" 2>/dev/null | head -1)
ROOT_DEV=$(lsblk -no PKNAME "$(df --output=source / | tail -1)" 2>/dev/null | head -1)
[[ -n $DATA_DEV && -n $ROOT_DEV && -e /sys/block/$DATA_DEV ]] || { echo "cannot resolve the <DATA_ROOT> disk"; exit 2; }
CG=/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
DEADLINE=$W44R_SESSION_DEADLINE
T0=$(date +%s)
log() { w44_log "$@"; }
post() { [[ $SMOKE == 1 ]] && { printf '%s w44-rc: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$A/smoke_posts.txt"; return; }
  printf '%s w44-rc: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
log "START RW3T: N=$P O=$P0 case $CASE ($CASE_KEY) base args '$BASE_ARGS' rules P>$RULE_P %/A>$RULE_A % >10 s, quiet gate ${QGATE_S}s, watch $CORES, disk data=$DATA_DEV root=$ROOT_DEV, cgroup $CG, deadline $(date -u -d @"$DEADLINE" +%FT%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') mode $W44R_MODE"
log "tools $(cd "$W44R_TOOLS" && sha256sum session_rw3.sh arm.sh arm_timer.py core_watch.py sched_watch.py disk_watch.py | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
post "RW3T START (inside the w44-rc lock session; window end $(date -u -d @"$DEADLINE" +%TZ)): WN3 warm, v0.4.4 (N) vs v0.4.3 (O) sealed bundles, slots $(grep -v -E '^(#|CASE)' "$W44R_SLOTS" | cut -d'|' -f1 | tr '\n' ' ')"
quiet_off() { if [ -e "$QUIET_FILE" ] && grep -q '^w44-rc ' "$QUIET_FILE"; then rm -f "${QUIET_FILE:?}"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
DISKPID=; disk_off() { [ -n "$DISKPID" ] && { kill -TERM "$DISKPID" 2>/dev/null; wait "$DISKPID" 2>/dev/null; DISKPID=; }; }
child=
done_post() { post "RW3T END rc=$1 $(( ($(date +%s) - T0 + 59) / 60 )) min; log $W44R_LOG"; }
on_signal() { trap - EXIT; [ -n "$child" ] && { kill -TERM "$child" 2>/dev/null; wait "$child"; }; samplers_off; disk_off; quiet_off
  log "END (signal)"; done_post 143; exit 143; }
on_exit() { local rc=$?; samplers_off; disk_off; quiet_off; log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) min"; done_post "$rc"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
taskset -c "$SAMPLE_CPU" nice -n 5 "$W44R_PY" -I "$W44R_TOOLS/disk_watch.py" sample "$A/disk.jsonl" "data=$DATA_DEV" "root=$ROOT_DEV" \
  > "$A/disk_watch.log" 2>&1 &
DISKPID=$!
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg) free_gib_data $(w44_free_gib <DATA_ROOT>)"
    ps -eo psr,pcpu,rss,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {  # samplers_on RUN_DIR: VRAM/dmon/iostat (GPU only) + core watcher + shadow watcher + scheduler view
  if [[ $SMOKE == 0 ]]; then
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
    SAMPLERS+=($!)
    taskset -c "$SAMPLE_CPU" nvidia-smi dmon -s puc -o T > "$1/gpu_dmon.txt" 2>/dev/null &
    SAMPLERS+=($!)
  else echo "smoke: no nvidia-smi sampler" > "$1/vram_apps.csv"; fi
  S_TIME_FORMAT=ISO taskset -c "$SAMPLE_CPU" iostat -x -d -t "$DATA_DEV" 1 > "$1/iostat_x.txt" 2>&1 &
  SAMPLERS+=($!)
  taskset -c "$WATCH_CPU" nice -n 5 "$W44R_PY" -I "$W44R_TOOLS/core_watch.py" "$CORES" "$RULE_P" 10 "$RULE_A" "$$" "$1/core_watch.jsonl" \
    "$1/core_watch_summary.json" "$1/ABORT" 1 > "$1/core_watch.log" 2>&1 &
  SAMPLERS+=($!)
  taskset -c "$SAMPLE_CPU" nice -n 5 "$W44R_PY" -I "$W44R_TOOLS/core_watch.py" "$CORES" 20 1e9 50 "$$" "$1/core_watch_strict.jsonl" \
    "$1/core_watch_strict_summary.json" "$1/ABORT_STRICT_NEVER" 1 > "$1/core_watch_strict.log" 2>&1 &
  SAMPLERS+=($!)
  taskset -c "$SAMPLE_CPU" nice -n 5 "$W44R_PY" -I "$W44R_TOOLS/sched_watch.py" "$CG" "$1/sched_watch.jsonl" "$1/sched_watch_summary.json" 1 \
    > "$1/sched_watch.log" 2>&1 &
  SAMPLERS+=($!)
}
pre_gate() {  # pre_gate SLOT CAP: wait for a quiet stretch (frozen core_watch at 3 s), else start after GATE_MAX_S (BUSY_START)
  local i=0 t_end=$(( $(date +%s) + GATE_MAX_S )) g
  while :; do
    i=$((i + 1)); g="$A/gate/$1.$i"; mkdir -p "$g"
    taskset -c "$WATCH_CPU" nice -n 5 "$W44R_PY" -I "$W44R_TOOLS/core_watch.py" "$CORES" "$RULE_P" 3 "$RULE_A" "$$" "$g/w.jsonl" "$g/s.json" "$g/ABORT" 1 \
      > "$g/w.log" 2>&1 &
    child=$!; sleep "$QGATE_S" & local sl=$!; wait "$sl"; kill -TERM "$child" 2>/dev/null; wait "$child"; child=
    if pgrep -f '^/usr/(s)?bin/updatedb' > /dev/null 2>&1; then
      echo '{"rule": "UPDATEDB", "note": "updatedb running (system cron, <DATA_ROOT> IO)"}' > "$g/ABORT"; fi
    if [ ! -s "$g/ABORT" ]; then
      printf '%s\t%s\tfree\tquiet-stretch %s s, check %s\n' "$1" "$(date -u +%FT%TZ)" "$QGATE_S" "$i" >> "$A/core_gate.tsv"; return 0; fi
    (( i % 10 == 1 )) && log "pre-gate $1 check $i busy ($(cut -c1-160 "$g/ABORT"))"
    hold_stop && return 0
    (( $(date +%s) >= t_end )) && break
    (( DEADLINE - $(date +%s) >= $2 + QGATE_S )) || break
  done
  printf '%s\t%s\tBUSY_START\tafter %s checks: %s\n' "$1" "$(date -u +%FT%TZ)" "$i" "$(cut -c1-200 "$g/ABORT")" >> "$A/core_gate.tsv"
}
MB_FROM=$(wc -l < "$MAILBOX")
hold_stop() {  # a manager HOLD/STOP line to w44-rc after the session start cancels the remaining slots
  tail -n +$((MB_FROM + 1)) "$MAILBOX" | grep -E '^20[0-9T:.+Z-]+ manager -> [^:]*w44-rc[^:]*:.*\b(HOLD|STOP)\b' > /dev/null; }
chk() {  # chk BUNDLE OUTBASE
  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 "$W44R_PY" -I "$1/tools/compile_bundle.py" check --cache "$1/jax" \
    --manifest "$1/sealed/BUNDLE_MANIFEST.json" --out "$2.json" > "$2.txt" 2>&1
  local crc=$?; (( crc == 0 )) || return "$crc"
  grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$2.txt"; }
run_slot() {  # run_slot SLOT KIND TRY HOURS CAP GIB KEEP -> 0 accepted, 9 no time/disk, 8 aborted (repeat), 99 bundle check failed
  local name=$1 kind=$2 try=$3 hours=$4 cap=$5 gib=$6 keep=$7 b=$P; [[ $kind == O ]] && b=$P0
  [[ $SMOKE == 1 ]] && cap=$(( cap * ${W44R_SMOKE_CAP_FACTOR:?} ))   # CPU smoke only (see entry_common.sh)
  (( DEADLINE - $(date +%s) >= cap )) || { log "SKIP $name: window too short"; return 9; }
  w44_disk_ok "$gib" || { log "SKIP $name: <DATA_ROOT> would fall below the floor ($(w44_free_gib <DATA_ROOT>) GiB free, need $gib)"; return 9; }
  mkdir -p "$R/$name"
  chk "$b" "$R/$name/check_before" || { log "$name: $b check before FAILED: stop"; return 99; }
  pre_gate "$name" "$cap"
  hold_stop && { log "manager HOLD/STOP: stop before $name"; return 99; }
  (( DEADLINE - $(date +%s) >= cap )) || { log "SKIP $name: window too short after the pre-gate"; return 9; }
  printf '%s %s\n' "w44-rc RW3T $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  samplers_on "$R/$name"; host_load "before $name try $try"
  log "ARM $name kind $kind try $try $hours h cap ${cap}s bundle $b"
  # CPU smoke: the model runs only on the mini grid (Swiss); WN3 slots run the exact CLI args with --dry-run (E187: 1 h is
  # not a valid WN3 window, and a 3-nest CPU compile does not fit the 12G smoke slot)
  if [[ $SMOKE == 1 ]]; then export W44R_SMOKE_ARM_MODE=model; fi
  # shellcheck disable=SC2086
  "$W44R_PY" -I "$W44R_TOOLS/arm_timer.py" --seconds "$cap" --receipt "$R/$name/timer.json" -- \
    bash "$b/tools/bundle_consumer.sh" --bundle "$b" --layer "$LAY/${name}_t$try" -- \
    bash "$W44R_TOOLS/arm.sh" "$b/bundle.env" "$name" "$CASE" "$R/$name" -- $BASE_ARGS --hours "$hours" > "$R/$name.scope.log" 2>&1 &
  child=$!
  local aborted=
  while kill -0 "$child" 2>/dev/null; do
    if [ -s "$R/$name/ABORT" ] && [ -z "$aborted" ]; then
      aborted=$(cat "$R/$name/ABORT"); log "ABORT $name try $try: $aborted"; kill -TERM "$child" 2>/dev/null
    fi
    sleep 1
  done
  wait "$child"; local rc=$?; child=
  samplers_off; quiet_off; host_load "after $name try $try"
  chk "$b" "$R/$name/check_after" || { log "$name: $b NOT OK after: stop"; return 99; }
  if [ -n "$aborted" ]; then
    JAX_PLATFORMS=cpu "$W44R_PY" -I "$W44R_TOOLS/arm_receipt.py" "$A" "$R/$name" "$R/$name/timer.json" "$W44R_RECEIPTS/aborted_$try" --delete-wrfout >> "$W44R_LOG" 2>&1
    local rrc=$?
    mv "$R/$name" "$R/$name.aborted$try"; mv "$R/$name.scope.log" "$R/$name.aborted$try.scope.log"
    printf '%s\t%s\tABORTED\t%s\n' "$name" "$(date -u +%FT%TZ)" "$aborted" >> "$A/core_gate.tsv"
    log "ARM $name aborted try $try rc=$rc (kept as $name.aborted$try, wrfouts deleted after their SHA list)"; (( rrc == 0 )) || return "$rrc"; return 8
  fi
  local del=(); [[ $keep == 0 && "$(cat "$R/$name/rc.txt" 2>/dev/null)" == 0 ]] && del=(--delete-wrfout)
  JAX_PLATFORMS=cpu "$W44R_PY" -I "$W44R_TOOLS/arm_receipt.py" "$A" "$R/$name" "$R/$name/timer.json" "$W44R_RECEIPTS" "${del[@]}" >> "$W44R_LOG" 2>&1
  local rrc=$?; (( rrc == 0 )) || log "receipt $name FAILED rc=$rrc"
  (( rc == 0 && rrc != 0 )) && rc=$rrc
  log "ARM $name rc=$rc try $try watch $("$W44R_PY" -I -c "import json,sys; s=json.load(open(sys.argv[1])); print('clean' if s['clean'] else 'within-rule' if s['within_rule'] else 'BREACH', 'maxproc', s['max_proc_seconds'], 'agg', s['agg_seconds'])" "$R/$name/core_watch_summary.json" 2>/dev/null) $(grep -E '^BUNDLE_CONSUMER rc=' "$R/$name.scope.log" | tail -1 | cut -c1-140)"
  return "$rc"
}
while IFS='|' read -r -u 3 name kind hours cap gib keep; do
  [[ -z $name || $name == \#* || $name == CASE ]] && continue
  rc=0
  for try in 1 2 3; do
    run_slot "$name" "$kind" "$try" "$hours" "$cap" "$gib" "$keep"; rc=$?
    (( rc == 99 )) && exit 7
    (( rc == 0 || rc == 8 || rc == 9 )) || exit "$rc"
    (( rc == 8 )) && continue
    break
  done
  (( rc == 8 )) && exit 8  # retries exhausted: no successful mandatory arm
  if (( rc == 9 )); then [[ $name == N72 ]] && { log "N72 not run: stop (mandatory D6 input)"; exit 9; }; break; fi
done 3< "$W44R_SLOTS"
chk "$P" "$A/final_check_N"; frc=$?; (( frc == 0 )) || exit "$frc"
chk "$P0" "$A/final_check_O"; frc=$?; (( frc == 0 )) || exit "$frc"
log "DONE"
