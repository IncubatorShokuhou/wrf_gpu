#!/usr/bin/env bash
# B1 v0.4.1 release benchmark, GPU part: ONE lock session (run inside scripts/with_gpu_lock.sh --label w41-bdyrec).
# usage: session.sh ROOT        (ROOT staged by setup.py: PLAN.json, snap, wn3/<case>, empty cache_prod / cache_wn3)
# Program = immutable detached snapshot of the manager's source pin (B1_PIN file, manager 2026-10-18T08:23:19Z; release
# defaults, no GPUWRF_* model flag). Refuses when the pin file is missing or names another commit than the snapshot,
# when the snapshot moved/is dirty, or when main's src tree differs from the snapshot's (later main commits = docs only).
# Order: the UNTIMED cold compiles first, then the TIMED arms, each behind the core gate; the WN3 arms come first because
# par24 is also the nested D6 gate of the pinned fix (manager 08:23:19Z):
#   P1 prime   PROD 3 h s0 harness, empty cache_prod  = cold compile (bench 'cold' arm; not QUIET, not headline)
#   W1 cold    WN3-0227 3 h, product parallel launcher (1 case), empty cache_wn3: cold compile + C-auto plan
#   W1 warm    WN3-0227 3 h warm, QUIET: records the WARM host peak (launcher admission of W2) + solo warm reference
#   W2 par24   WN3 0227/0502/0614 --max-parallel 3, 24 h warm, QUIET -> s per case-sim-hour vs CPU-WRF 3x4-core 123.385;
#              its 75 frames per case are the nested D6 gate (follow-up, original CPU-WRF)
#   P2 warm1 / P3 warm2   PROD 3 h harness, QUIET     -> S1 proxy = bench_summary seg 2-3 mean, A/A
#   P4 whole24 PROD 24 h harness, QUIET               -> whole-run S1 = last wrfout mtime - receipt t0_wall_utc, / 24
#                                                         (E162: mtimes captured right after the arm, inside the window)
# Core gate (manager 2026-10-18T05:31:27Z): before every TIMED arm cores_free.py samples all threads; a foreign thread
# > 20 % of a core on 3,13,19,29 => REFUSE (list -> core_checks.jsonl), retry every B1_CORE_RETRY_S (300 s) until
# B1_CORE_UNTIL_UTC (15:00Z); after that the arm runs and the load is DISCLOSED (core_gate.tsv). Host load snapshots
# around every arm -> HOST_LOAD.txt.
# s0_guard free-VRAM floor 22000 MiB (bench default 26000; desktop held 5.3 GB at 05:56Z, PROD per-PID ~8.8 GiB A58).
# Budget: DEADLINE = min(B1_END_UTC (19:00Z) - 2 min, start + B1_BUDGET_S (150 GPU-min)); every arm's RuntimeMaxSec is
# cut to it and an arm whose expected wall does not fit (1.3x) is skipped, never started. Arms run in their own systemd
# scope (MemoryMax 20G; W2: B1_W2_MEM, 3 cases), no swap, no nested timeout. Lifecycle as the approved composed session:
# scopes w41bdyrec-b1-<pid>-*, arms in the background under an interruptible wait, TERM/INT/HUP + any exit stop every
# session scope and wait until none is left; the session-owned QUIET file is removed by the same traps.
# B1_DRY_ARM=<s> (CPU lifecycle proof only): each arm runs 'sleep <s>', no GPU query, QUIET -> B1_QUIET_FILE;
# B1_DRY_EXIT_AFTER_START=1 exits 7 right after the first scope started.
set -uo pipefail
ROOT=$(readlink -f "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
DRY=${B1_DRY_ARM:-}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
QUIET_FILE=${B1_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || "$QUIET_FILE" != /tmp/wrf_gpu2_quiet ]] || { echo "dry run needs B1_QUIET_FILE"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
plan() { python3 -c "import json,sys; d=json.load(open('$ROOT/PLAN.json')); print(eval(sys.argv[1]))" "$1"; }
SNAP=$(plan "d['snapshot']"); PROGRAM=$(plan "d['program']"); TREE=$(plan "d['src_tree']")
MAIN_REPO=${B1_MAIN_REPO:-<USER_HOME>/src/wrf_gpu2}
[[ "$(git -C "$SNAP" rev-parse HEAD)" == "$PROGRAM" && "$(git -C "$SNAP" rev-parse HEAD:src)" == "$TREE" ]] \
  || { echo "snapshot moved"; exit 4; }
[[ -z "$(git -C "$SNAP" status --porcelain --untracked-files=all -- src scripts)" ]] || { echo "snapshot dirty"; exit 4; }
MAIN_TREE=$(git -C "$MAIN_REPO" rev-parse "${B1_MAIN_REF:-main}:src")   # B1_MAIN_REF: refusal test only
[[ "$MAIN_TREE" == "$TREE" ]] || { echo "main src tree $MAIN_TREE != program $TREE: refuse"; exit 4; }
PIN_FILE=${B1_PIN_FILE:-<USER_HOME>/wrf_gpu2_lanes/manager/B1_PIN}         # B1_PIN_FILE: dry test only
PIN=$( { tr -d '[:space:]' < "$PIN_FILE"; } 2>/dev/null )
[[ "$PIN" == "$PROGRAM" ]] || { echo "pin file $PIN_FILE ('$PIN') != program $PROGRAM: refuse"; exit 4; }
T0=$(date +%s)
END_UTC=$(( $(date -u -d "${B1_END_UTC:-2026-10-18T19:00:00Z}" +%s) - 120 ))
DEADLINE=$(( T0 + ${B1_BUDGET_S:-9000} )); (( END_UTC < DEADLINE )) && DEADLINE=$END_UTC
CORE_UNTIL=$(date -u -d "${B1_CORE_UNTIL_UTC:-2026-10-18T15:00:00Z}" +%s)
RETRY=${B1_CORE_RETRY_S:-300}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/session.log"; }
log "START program $PROGRAM (== pin file, src $TREE == main:src) deadline $(date -u -d @$DEADLINE +%FT%TZ) pin $(taskset -cp $$ | awk -F': ' '{print $2}')"
[ -n "$DRY" ] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv | tee -a "$ROOT/session.log"

PREFIX="w41bdyrec-b1-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() {  # stop every scope of this session and wait until none is left (systemd: TERM, then KILL)
  local u n=0
  while [ -n "$(session_scopes)" ]; do
    for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && { log "scopes still alive: $(session_scopes | tr '\n' ' ')"; sleep 10; }
  done
}
quiet_off() {  # remove the QUIET file only while it carries a B1 label of this session (also run_arm.sh's, if its trap died)
  if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-bdyrec B1 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
child=
on_signal() { trap - EXIT; log "signal: stopping session scopes"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  stop_scopes; quiet_off; log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"; exit 143; }
on_exit() { local rc=$?; stop_scopes; quiet_off; log "exit rc=$rc, scopes left: $(session_scopes | wc -l)"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() {  # run "$@" in the background and wait (a trapped signal interrupts wait; a foreground child would defer it)
  "$@" & child=$!
  [ -n "${B1_DRY_EXIT_AFTER_START:-}" ] && { sleep 3; log "dry: simulated session error after the first scope started"; exit 7; }
  wait "$child"; local rc=$?; child=; return $rc
}
pause() { sleep "$1" & child=$!; wait "$child"; child=; }

cap_for() {  # cap_for EXPECTED_S CAP_S -> RuntimeMaxSec to use, or nothing if the arm does not fit
  local left=$(( DEADLINE - $(date +%s) ))
  (( left >= $1 * 13 / 10 )) || return 1
  (( left < $2 )) && echo "$left" || echo "$2"
}
host_load() {
  { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -25 | cut -c1-200; } >> "$ROOT/HOST_LOAD.txt"
}
core_gate() {  # core_gate ARM -> always returns 0 to run; records 'free' or the disclosed foreign load in core_gate.tsv
  local tries=0
  while :; do
    tries=$((tries + 1))
    if python3 "$HERE/cores_free.py" 3,13,19,29 20 "$ROOT/core_checks.jsonl" "$$" 10 >> "$ROOT/core_gate.log" 2>&1; then
      printf '%s\t%s\tfree\t%s\n' "$1" "$(date -u +%FT%TZ)" "$tries" >> "$ROOT/core_gate.tsv"; log "core gate $1: free"; return 0
    fi
    if (( $(date +%s) >= CORE_UNTIL )); then
      printf '%s\t%s\tBUSY_RUN_DISCLOSED\t%s\n' "$1" "$(date -u +%FT%TZ)" "$tries" >> "$ROOT/core_gate.tsv"
      log "core gate $1: foreign load on 3,13,19,29 after ${B1_CORE_UNTIL_UTC:-15:00Z} -> RUN + DISCLOSE ($(tail -1 "$ROOT/core_gate.log"))"
      return 0
    fi
    log "core gate $1: REFUSE (foreign load: $(tail -1 "$ROOT/core_gate.log" | cut -c1-200)); retry in ${RETRY}s"
    pause "$RETRY"
  done
}

S1=$ROOT/S1; PCACHE=$ROOT/cache_prod; PROD_INPUT=$(plan "d['prod_input']")
S0ROOT=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/b1_s0/$(basename "$ROOT")   # s0_guard refuses <DATA_ROOT> roots
ENVFILE=$SNAP/.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools/s0_env.sh
printf 'unset GPUWRF_WRF_ROOT\n' > "$ROOT/s1_flags.env"   # s0_env.sh still exports the deleted tenerife_b4 tree (E: 01:15Z)
s1_arm() {  # s1_arm NAME HOURS EXPECTED_S CAP_S [QUIET_LABEL]
  local name=$1 hours=$2 cap quiet=${5:-}; cap=$(cap_for "$3" "$4") || { log "SKIP S1/$name: not enough session time"; return 9; }
  mkdir -p "$S1/$name" "$S0ROOT"
  local cmd=(nice -n 10 bash "$SNAP/scripts/bench/run_arm.sh" "$ENVFILE" "$SNAP"
             "$SNAP/.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools" "" "$quiet" 0 --
             <USER_HOME>/miniconda3/bin/python3 "$SNAP/scripts/bench/s0_nested_harness.py"
             --input-dir "$PROD_INPUT" --out "$S1/$name/run" --hours "$hours" --tag "w41bdyrec_b1_${name}")
  [ -n "$DRY" ] && cmd=(sleep "$DRY")
  host_load "before S1/$name"
  log "S1 arm $name ${hours} h cap ${cap}s quiet='${quiet}'"
  date +%s.%N > "$S1/$name/launch_unix"
  in_bg env S0_ROOT="$S0ROOT" S0_CPUS="3,13,19,29" S0_MIN_FREE_MIB="${B1_S0_MIN_FREE_MIB:-22000}" S0_MIN_FREE_DISK_GB="${S0_MIN_FREE_DISK_GB:-8}" \
      OMP_NUM_THREADS=4 GPUWRF_CENSUS=0 BENCH_ARM_DIR="$S1/$name" GPUWRF_BENCH_FLAGS_ENV="$ROOT/s1_flags.env" \
      BENCH_QUIET_FILE="$QUIET_FILE" \
      GPUWRF_JAX_CACHE_DIR="$PCACHE/jax" JAX_COMPILATION_CACHE_DIR="$PCACHE/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$PCACHE/autotune" TRITON_CACHE_DIR="$PCACHE/triton" CUDA_CACHE_PATH="$PCACHE/cuda" \
    systemd-run --user --scope --quiet --unit="$PREFIX-s1-$name" -p MemoryMax=20G -p MemorySwapMax=0 \
      -p RuntimeMaxSec="$cap" -- "${cmd[@]}" > "$S1/$name/stdout.log" 2> "$S1/$name/stderr.log"
  local rc=$?; host_load "after S1/$name"; log "S1 arm $name rc=$rc"; return $rc
}
endpoint() {  # endpoint NAME HOURS: E162 publication receipt (original wrfout mtimes inside the completed run window)
  [ -n "$DRY" ] && return 0
  python3 - "$S1/$1/run/receipt.json" "$1" "$2" "$TREE" "$S1/$1/launch_unix" > "$S1/$1/endpoint.json" <<'EOF' || log "endpoint $1 FAILED"
import json, sys
from pathlib import Path
receipt, arm, hours, tree, launch = sys.argv[1:6]
r = json.loads(Path(receipt).read_text())
args = r["cli_argv"]
out = Path(args[args.index("--output-dir") + 1])
files = sorted(p for p in out.glob("wrfout_d0*"))
writes = {p.name: p.stat().st_mtime for p in files}
t0 = float(r["t0_wall_utc"])
ok = (r.get("end_reason") == "completed" and len(files) == len(r["outputs"]) and bool(writes)
      and all(t0 <= t <= t0 + float(r["t_end"]) + 0.1 for t in writes.values()))
print(json.dumps({"benchmark_receipt": receipt, "src_tree": r.get("src_tree"), "program_src_tree": tree, "arm": arm,
                  "hours": int(hours), "t0_wall_utc": t0, "launch_unix": float(Path(launch).read_text()),
                  "file_write_unix": writes, "n_outputs_receipt": len(r["outputs"]), "valid": ok,
                  "wall_s": max(writes.values()) - t0 if writes else None,
                  "method": "original output file mtimes, captured right after the arm inside the completed run window (E162)"},
                 indent=1))
EOF
}

WCACHE=$ROOT/cache_wn3; WRUNS=$ROOT/wn3_runs
wn3_arm() {  # wn3_arm NAME HOURS EXPECTED_S CAP_S MEMMAX QUIET_LABEL CASE...
  local name=$1 hours=$2 mem=$5 quiet=$6 cap; cap=$(cap_for "$3" "$4") || { log "SKIP WN3/$name: not enough session time"; return 9; }
  shift 6
  local cases=() c; for c in "$@"; do cases+=("$ROOT/wn3/$c"); done
  local out=$WRUNS/$name
  [ -e "$out" ] && { log "WN3/$name: $out exists (E42 no-overwrite): refuse"; return 8; }
  local cmd=(bash "$SNAP/scripts/run_parallel_cases.sh" --max-parallel 3 --out-root "$out" "${cases[@]}" --
             --domains-from-namelist --hours "$hours" --emit-initial-history)
  [ -n "$DRY" ] && { cmd=(sleep "$DRY"); mkdir -p "$out"; }
  host_load "before WN3/$name"
  if [ -n "$quiet" ]; then printf '%s %s\n' "$quiet" "$(date -u +%FT%TZ)" > "$QUIET_FILE"; fi
  log "WN3 arm $name ${hours} h cases $* cap ${cap}s mem $mem quiet='${quiet}'"
  in_bg env JAX_PLATFORMS=cuda PYTHONPATH="$SNAP/src" \
      GPUWRF_JAX_CACHE_DIR="$WCACHE/jax" JAX_COMPILATION_CACHE_DIR="$WCACHE/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$WCACHE/autotune" TRITON_CACHE_DIR="$WCACHE/triton" CUDA_CACHE_PATH="$WCACHE/cuda" \
      TMPDIR="$ROOT/tmp" \
    systemd-run --user --scope --quiet --unit="$PREFIX-wn3-$name" -p MemoryMax="$mem" -p MemorySwapMax=0 \
      -p RuntimeMaxSec="$cap" -- "${cmd[@]}" > "$WRUNS/$name.log" 2>&1
  local rc=$?
  quiet_off
  [ -n "$DRY" ] || python3 - "$out" > "$WRUNS/$name.mtimes.json" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(json.dumps({c.name: {p.name: p.stat().st_mtime for p in sorted((c / "wrfout").glob("wrfout_d0*"))}
                  for c in sorted(out.iterdir()) if (c / "wrfout").is_dir()}, indent=1))
EOF
  host_load "after WN3/$name"; log "WN3 arm $name rc=$rc"; return $rc
}

if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop"; exit 3; fi
PHASES=" ${B1_PHASES:-P W} "
# --- untimed: cold compiles (disclosed core state, no gate) ---
if [[ $PHASES == *" P "* ]]; then
  s1_arm prime 3 900 1800 && P_OK=1 || P_OK=
fi
if [[ $PHASES == *" W "* ]]; then
  wn3_arm cold3 3 1000 2400 20G "" wn3_0227 && W_OK=1 || W_OK=
fi
# --- timed (core gate + QUIET); WN3 first: par24 is also the nested D6 gate ---
if [[ $PHASES == *" W "* && -n "${W_OK:-}" ]]; then
  core_gate wn3_warm3 && wn3_arm warm3 3 240 900 20G "w41-bdyrec B1 WN3 warm3" wn3_0227 \
    && core_gate wn3_par24 && wn3_arm par24 24 1100 2700 "${B1_W2_MEM:-40G}" "w41-bdyrec B1 WN3 par24" wn3_0227 wn3_0502 wn3_0614
fi
if [[ $PHASES == *" P "* && -n "${P_OK:-}" ]]; then
  core_gate warm1 && s1_arm warm1 3 120 600 "w41-bdyrec B1 S1 warm1" \
    && core_gate warm2 && s1_arm warm2 3 120 600 "w41-bdyrec B1 S1 warm2"
  core_gate whole24 && s1_arm whole24 24 240 900 "w41-bdyrec B1 S1 whole24" && endpoint whole24 24
fi
stop_scopes
log "END GPU part: $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l); CPU follow-up: followup.py $ROOT"
