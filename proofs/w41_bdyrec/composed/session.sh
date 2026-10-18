#!/usr/bin/env bash
# Composed default requal, GPU part: ONE lock session (run inside scripts/with_gpu_lock.sh --label w41-bdyrec).
# usage: session.sh ROOT        (ROOT staged by setup.py; PLAN.json)
#  A  cold 24 h CLI arms on the candidate snapshot, release defaults (no GPUWRF_* model flag), sequential:
#     WN3-0227 (3-nest, --emit-initial-history), PROD (d01+d02, --emit-initial-history), Swiss (d01)
#     -> D6 vs original CPU-WRF + E41 + identities on CPU afterwards (followup.py)
#  S  S1 proxy, PROD 3 h, bench method (scripts/bench/run_arm.sh + s0_nested_harness.py of the snapshot, the
#     bench_prod.sh arm environment): prime (fills the harness cheap key; shares the PROD case cache path) +
#     warm1 + warm2 (timed; QUIET label like bench --quiet-label) -> bench_summary.py seg 2-3 in followup.py
# Every arm runs in its own systemd scope (MemoryMax=20G, no swap, RuntimeMaxSec = cap; no nested timeout: a
# uutils timeout escapes the session group, 15:10Z incident). The lock holder is pinned to the GPU-arm cores
# 3,19,13,29 by with_gpu_lock.sh. Hard end: every arm's RuntimeMaxSec is cut to the session deadline
# (min(07:30Z - 2 min, start + 75 GPU-min lease)); an arm that cannot get its expected wall is skipped, never started.
# Never returns (=> with_gpu_lock never releases) while a scope of THIS session is alive: scopes are named
# w41bdyrec-req-<session pid>-*, arms run in the background under an interruptible wait, and TERM/INT/HUP, any exit
# (error) and the normal end stop every session scope and wait until none is left (gpu_session2.sh protection,
# 15:10Z incident). REQUAL_PHASES (default 'A S') selects the phases (an S1-only re-run: 'S').
# The S1 arms source ROOT/s1_flags.env after s0_env.sh (run_arm.sh GPUWRF_BENCH_FLAGS_ENV): 'unset GPUWRF_WRF_ROOT',
# because s0_env.sh still points at the deleted tenerife_b4 tree (01:15Z prime: MPTABLE.TBL missing) and the release
# CLI arms resolve the pristine WRF tables themselves. REQUAL_DRY_ARM=<seconds> (CPU lifecycle proof only): each arm scope runs 'sleep <seconds>'
# instead of the model, no GPU query; REQUAL_DRY_EXIT_AFTER_START=1 exits 7 right after the first scope started.
set -uo pipefail
ROOT=$(readlink -f "$1")
DRY=${REQUAL_DRY_ARM:-}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
plan() { python3 -c "import json,sys; d=json.load(open('$ROOT/PLAN.json')); print(eval(sys.argv[1]))" "$1"; }
SNAP=$(plan "d['candidate']"); HEAD=$(plan "d['candidate_head']"); TREE=$(plan "d['candidate_src_tree']")
[[ "$(git -C "$SNAP" rev-parse HEAD)" == "$HEAD" && "$(git -C "$SNAP" rev-parse HEAD:src/gpuwrf)" == "$TREE" ]] \
  || { echo "snapshot moved"; exit 4; }
[[ -z "$(git -C "$SNAP" status --porcelain --untracked-files=all -- src)" ]] || { echo "snapshot dirty"; exit 4; }
T0=$(date +%s)
END_UTC=$(date -u -d "${REQUAL_END_UTC:-2026-10-18T07:28:00Z}" +%s)
DEADLINE=$(( T0 + ${REQUAL_BUDGET_S:-4500} )); (( END_UTC < DEADLINE )) && DEADLINE=$END_UTC
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/session.log"; }
log "START candidate $HEAD (src $TREE) deadline $(date -u -d @$DEADLINE +%FT%TZ) pin $(taskset -cp $$ | awk -F': ' '{print $2}')"
[ -n "$DRY" ] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv | tee -a "$ROOT/session.log"

PREFIX="w41bdyrec-req-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() {  # stop every scope of this session and wait until none is left (systemd: TERM, then KILL)
  local u n=0
  while [ -n "$(session_scopes)" ]; do
    for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && { log "scopes still alive: $(session_scopes | tr '\n' ' ')"; sleep 10; }
  done
}
child=
on_signal() { trap - EXIT; log "signal: stopping session scopes"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  stop_scopes; log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"; exit 143; }
on_exit() { local rc=$?; stop_scopes; log "exit rc=$rc, scopes left: $(session_scopes | wc -l)"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() {  # run "$@" in the background and wait (a trapped signal interrupts wait; a foreground child would defer it)
  "$@" & child=$!
  [ -n "${REQUAL_DRY_EXIT_AFTER_START:-}" ] && { sleep 3; log "dry: simulated session error after the first scope started"; exit 7; }
  wait "$child"; local rc=$?; child=; return $rc
}

cap_for() {  # cap_for EXPECTED_S CAP_S -> RuntimeMaxSec to use, or nothing if the arm does not fit
  local left=$(( DEADLINE - $(date +%s) ))
  (( left >= $1 * 13 / 10 )) || return 1
  (( left < $2 )) && echo "$left" || echo "$2"
}

in_dir() { cd "$1" || exit 1; shift; exec "$@"; }   # background helper: the scope's processes start in DIR

cli_arm() {  # cli_arm CASE OUT HOURS EXPECTED_S CAP_S
  local case=$ROOT/$1 out=$2 hours=$3 cap; cap=$(cap_for "$4" "$5") || { log "SKIP $1/$out: not enough session time"; return 9; }
  local cache=$case/cache dom; dom=$(python3 -c "import json;print(len(json.load(open('$case/inputs.json'))['domains']))")
  local extra=(); (( dom > 1 )) && extra=(--emit-initial-history)
  mkdir -p "$case/$out/wrfout" "$case/$out/proof" "$case/$out/scratch"
  local cmd=(python3 -m gpuwrf.cli run --input-dir "$case/input" --output-dir "$case/$out/wrfout"
             --proof-dir "$case/$out/proof" --scratch-dir "$case/$out/scratch" --hours "$hours" --domains-from-namelist
             "${extra[@]}")
  [ -n "$DRY" ] && cmd=(sleep "$DRY")
  log "arm $1/$out ${hours} h cap ${cap}s"
  in_bg in_dir "$case/$out" env JAX_PLATFORMS=cuda PYTHONPATH="$SNAP/src" \
      GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/autotune" TRITON_CACHE_DIR="$cache/triton" \
      CUDA_CACHE_PATH="$cache/cuda" TMPDIR="$case/tmp" \
    systemd-run --user --scope --quiet --unit="$PREFIX-$1-$out" -p MemoryMax=20G -p MemorySwapMax=0 \
      -p RuntimeMaxSec="$cap" -- /usr/bin/time -v "${cmd[@]}" > "$case/$out/run.log" 2>&1
  local rc=$?
  echo "arm rc=$rc" >> "$case/$out/run.log"
  (cd "$case/$out/wrfout" && sha256sum wrfout_* > ../wrfout.sha256 2>/dev/null)
  log "arm $1/$out rc=$rc"; return $rc
}

S1=$(plan "d['s1']['root']"); PCACHE=$(plan "d['s1']['cache']"); S1CASE=$(plan "d['s1']['case_input']")
S0ROOT=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/requal_s0/$(basename "$ROOT")   # s0_guard refuses <DATA_ROOT> roots
ENVFILE=$SNAP/.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools/s0_env.sh
s1_arm() {  # s1_arm NAME EXPECTED_S CAP_S [QUIET_LABEL]
  local name=$1 cap quiet=${4:-}; cap=$(cap_for "$2" "$3") || { log "SKIP S1/$name: not enough session time"; return 9; }
  mkdir -p "$S1/$name" "$S0ROOT"
  local cmd=(nice -n 10 bash "$SNAP/scripts/bench/run_arm.sh" "$ENVFILE" "$SNAP"
             "$SNAP/.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools" "" "$quiet" 0 --
             <USER_HOME>/miniconda3/bin/python3 "$SNAP/scripts/bench/s0_nested_harness.py"
             --input-dir "$S1CASE" --out "$S1/$name/run" --hours 3 --tag "composed_${name}")
  [ -n "$DRY" ] && cmd=(sleep "$DRY")
  log "S1 arm $name cap ${cap}s quiet='${quiet}'"
  in_bg env S0_ROOT="$S0ROOT" S0_CPUS="3,13,19,29" S0_MIN_FREE_MIB=26000 S0_MIN_FREE_DISK_GB="${S0_MIN_FREE_DISK_GB:-8}" \
      OMP_NUM_THREADS=4 GPUWRF_CENSUS=0 BENCH_ARM_DIR="$S1/$name" GPUWRF_BENCH_FLAGS_ENV="$ROOT/s1_flags.env" \
      GPUWRF_JAX_CACHE_DIR="$PCACHE/jax" JAX_COMPILATION_CACHE_DIR="$PCACHE/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$PCACHE/autotune" TRITON_CACHE_DIR="$PCACHE/triton" CUDA_CACHE_PATH="$PCACHE/cuda" \
    systemd-run --user --scope --quiet --unit="$PREFIX-s1-$name" -p MemoryMax=20G -p MemorySwapMax=0 \
      -p RuntimeMaxSec="$cap" -- "${cmd[@]}" > "$S1/$name/stdout.log" 2> "$S1/$name/stderr.log"
  local rc=$?; log "S1 arm $name rc=$rc"; return $rc
}

if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop"; exit 3; fi
PHASES=" ${REQUAL_PHASES:-A S} "
if [[ $PHASES == *" A "* ]]; then
  cli_arm WN3-0227 run24 24 1160 2400
  cli_arm PROD run24 24 750 1800
  cli_arm Swiss run24 24 410 1200
fi
for c in WN3-0227 PROD Swiss; do [ -d "$ROOT/$c/run24/wrfout" ] && ln -sfn run24/wrfout "$ROOT/$c/gpu"; done  # scorer reads case/gpu
if [[ $PHASES == *" S "* ]]; then
  printf 'unset GPUWRF_WRF_ROOT\n' > "$ROOT/s1_flags.env"
  s1_arm prime 300 1200 && s1_arm warm1 120 600 "w41-bdyrec composed S1 warm1" && s1_arm warm2 120 600 "w41-bdyrec composed S1 warm2"
fi
stop_scopes
log "END GPU part: $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l); CPU follow-up: followup.py $ROOT"
