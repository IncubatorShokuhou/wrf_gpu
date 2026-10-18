#!/usr/bin/env bash
# S1AB: matched PROD 24 h warm WHOLE-RUN A/B v0.4.0 vs v0.4.1 (manager 2026-10-18T11:47:21Z), ONE lock session.
# usage: session.sh OUT        (inside scripts/with_gpu_lock.sh --label w41-bdyrec; trigger.sh does that)
# A = v0.4.0 sealed bundle P (P/src = tag v0.4.0 7bb5ae669b) run as a CONSUMER: snapshot scripts/release/bundle_consumer.sh
#     --bundle P --layer OUT/layer_v040 (P is never written: every write lands in the layer; the consumer re-checks P with
#     P's own checker after each run, last line BUNDLE_CONSUMER ... compiled_new=N). P seals WN3-0227 + Swiss only, so the
#     PROD programs are compiled once into the layer: A0 = 3 h cold prime, A0w = 3 h warm (first warm run populates the
#     eager output programs, E71); both untimed.
# B = v0.4.1 = the B1 pin bcd156d19a snapshot + its warm cache_prod (B1: prime + warm1 + warm2 + whole24).
# Timed: A1 B1 A2 B2 = PROD 24 h warm, the SAME harness (scripts/bench/run_arm.sh + s0_nested_harness.py + s0_env.sh are
# byte-identical in both trees), same env (s1_flags.env, OMP 4, census off, S0_CPUS 3,13,19,29, free-VRAM floor 22000),
# QUIET, a core gate before each (foreign thread > 20 % on 3,13,19,29 -> retry every 60 s up to S1AB_GATE_MAX_S, then run
# and DISCLOSE in core_gate.tsv), E162 endpoint (original wrfout mtimes) right after each arm.
# Budget: DEADLINE = start + S1AB_BUDGET_S (1500 s = 25 GPU-min); an arm whose expected wall (x1.3) does not fit is skipped.
# Lifecycle as B1: one systemd scope per arm (MemoryMax 20G, no swap, RuntimeMaxSec), arms in the background under an
# interruptible wait, TERM/INT/HUP + any exit stop every session scope and remove the session-owned QUIET file.
# S1AB_DRY_ARM=<s> (CPU lifecycle proof only): each arm = 'sleep <s>' (A arms still through the real consumer), no GPU
# query, QUIET -> S1AB_QUIET_FILE.
set -uo pipefail
OUT=$(readlink -f "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
DRY=${S1AB_DRY_ARM:-}
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
QUIET_FILE=${S1AB_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
[[ -z "$DRY" || "$QUIET_FILE" != /tmp/wrf_gpu2_quiet ]] || { echo "dry run needs S1AB_QUIET_FILE"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0
B1=<DATA_ROOT>/wrf_gpu2/scratch/w41-bdyrec/b1_bcd156d19a0b
SNAP=$B1/snap; BCACHE=$B1/cache_prod
A_HEAD=7bb5ae669b2e4b863d6bafdd659a03fea026004f; B_HEAD=bcd156d19a0babb376f0f2976833e76f0f4c470a
[[ "$(git -C "$P/src" rev-parse HEAD)" == "$A_HEAD" && -z "$(git -C "$P/src" status --porcelain --untracked-files=no -- src scripts)" ]] \
  || { echo "A source $P/src is not a clean v0.4.0 checkout: refuse"; exit 4; }
[[ "$(git -C "$SNAP" rev-parse HEAD)" == "$B_HEAD" && -z "$(git -C "$SNAP" status --porcelain --untracked-files=all -- src scripts)" ]] \
  || { echo "B snapshot moved/dirty: refuse"; exit 4; }
for f in scripts/bench/run_arm.sh scripts/bench/s0_nested_harness.py .agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools/s0_env.sh; do
  cmp -s "$P/src/$f" "$SNAP/$f" || { echo "harness file $f differs between A and B: refuse"; exit 4; }
done
PROD_INPUT=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
TOOLREL=.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools
T0=$(date +%s)
DEADLINE=$(( T0 + ${S1AB_BUDGET_S:-1500} ))
GATE_MAX=${S1AB_GATE_MAX_S:-180}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/session.log"; }
log "START A=$A_HEAD ($P consumer) B=$B_HEAD ($SNAP + $BCACHE) deadline $(date -u -d @$DEADLINE +%FT%TZ) pin $(taskset -cp $$ | awk -F': ' '{print $2}')"
[ -n "$DRY" ] || nvidia-smi --query-gpu=memory.free,memory.used --format=csv | tee -a "$OUT/session.log"

PREFIX="w41bdyrec-s1ab-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() {
  local u n=0
  while [ -n "$(session_scopes)" ]; do
    for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && { log "scopes still alive: $(session_scopes | tr '\n' ' ')"; sleep 10; }
  done
}
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w41-bdyrec S1AB ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
child=
on_signal() { trap - EXIT; log "signal: stopping session scopes"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
  stop_scopes; quiet_off; log "END (signal) $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"; exit 143; }
on_exit() { local rc=$?; stop_scopes; quiet_off; log "exit rc=$rc, scopes left: $(session_scopes | wc -l)"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
in_bg() { "$@" & child=$!; wait "$child"; local rc=$?; child=; return $rc; }
pause() { sleep "$1" & child=$!; wait "$child"; child=; }
cap_for() { local left=$(( DEADLINE - $(date +%s) )); (( left >= $1 * 13 / 10 )) || return 1; (( left < $2 )) && echo "$left" || echo "$2"; }
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -25 | cut -c1-200; } >> "$OUT/HOST_LOAD.txt"; }
core_gate() {  # core_gate ARM: free -> run; foreign load -> retry every 60 s up to GATE_MAX, then run + DISCLOSE
  local t=$(date +%s) tries=0
  while :; do
    tries=$((tries + 1))
    if python3 "$HERE/cores_free.py" 3,13,19,29 20 "$OUT/core_checks.jsonl" "$$" 10 >> "$OUT/core_gate.log" 2>&1; then
      printf '%s\t%s\tfree\t%s\n' "$1" "$(date -u +%FT%TZ)" "$tries" >> "$OUT/core_gate.tsv"; log "core gate $1: free"; return 0
    fi
    if (( $(date +%s) - t >= GATE_MAX )); then
      printf '%s\t%s\tBUSY_RUN_DISCLOSED\t%s\n' "$1" "$(date -u +%FT%TZ)" "$tries" >> "$OUT/core_gate.tsv"
      log "core gate $1: foreign load after ${GATE_MAX}s -> RUN + DISCLOSE ($(tail -1 "$OUT/core_gate.log" | cut -c1-200))"; return 0
    fi
    log "core gate $1: REFUSE ($(tail -1 "$OUT/core_gate.log" | cut -c1-200)); retry in 60s"
    pause 60
  done
}
printf 'unset GPUWRF_WRF_ROOT\n' > "$OUT/s1_flags.env"   # as B1: s0_env.sh exports a deleted tenerife_b4 tree
S0ROOT=<USER_HOME>/wrf_gpu2_lanes/w41-bdyrec/s1ab_s0   # s0_guard refuses <DATA_ROOT> roots
arm() {  # arm NAME A|B HOURS EXPECTED_S CAP_S [QUIET_LABEL]
  local name=$1 side=$2 hours=$3 quiet=${6:-} cap src cache wrap=()
  cap=$(cap_for "$4" "$5") || { log "SKIP $name: not enough session time"; return 9; }
  if [[ $side == A ]]; then
    src=$P/src; cache=$P
    wrap=(bash "$SNAP/scripts/release/bundle_consumer.sh" --bundle "$P" --layer "$OUT/layer_v040" --)
  else
    src=$SNAP; cache=$BCACHE
  fi
  mkdir -p "$OUT/$name" "$S0ROOT"
  local cmd=(nice -n 10 bash "$src/scripts/bench/run_arm.sh" "$src/$TOOLREL/s0_env.sh" "$src" "$src/$TOOLREL" "" "$quiet" 0 --
             <USER_HOME>/miniconda3/bin/python3 "$src/scripts/bench/s0_nested_harness.py"
             --input-dir "$PROD_INPUT" --out "$OUT/$name/run" --hours "$hours" --tag "w41bdyrec_s1ab_${name}")
  [ -n "$DRY" ] && cmd=(sleep "$DRY")
  host_load "before $name"
  log "arm $name side $side ${hours} h cap ${cap}s quiet='${quiet}' src $(git -C "$src" rev-parse --short=12 HEAD) cache $cache"
  date +%s.%N > "$OUT/$name/launch_unix"
  in_bg env S0_ROOT="$S0ROOT" S0_CPUS="3,13,19,29" S0_MIN_FREE_MIB="${S1AB_S0_MIN_FREE_MIB:-22000}" S0_MIN_FREE_DISK_GB=8 \
      OMP_NUM_THREADS=4 GPUWRF_CENSUS=0 BENCH_ARM_DIR="$OUT/$name" GPUWRF_BENCH_FLAGS_ENV="$OUT/s1_flags.env" \
      BENCH_QUIET_FILE="$QUIET_FILE" \
      GPUWRF_JAX_CACHE_DIR="$cache/jax" JAX_COMPILATION_CACHE_DIR="$cache/jax" \
      GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$cache/autotune" TRITON_CACHE_DIR="$cache/triton" CUDA_CACHE_PATH="$cache/cuda" \
    systemd-run --user --scope --quiet --unit="$PREFIX-$name" -p MemoryMax=20G -p MemorySwapMax=0 \
      -p RuntimeMaxSec="$cap" -- "${wrap[@]}" "${cmd[@]}" > "$OUT/$name/stdout.log" 2> "$OUT/$name/stderr.log"
  local rc=$?; quiet_off; host_load "after $name"
  [[ $side == A ]] && log "arm $name consumer: $(/usr/bin/grep -E '^BUNDLE_CONSUMER ' "$OUT/$name/stdout.log" | tail -1)"
  log "arm $name rc=$rc"; return $rc
}
endpoint() {  # endpoint NAME HOURS: E162 publication receipt (original wrfout mtimes inside the completed run window)
  [ -n "$DRY" ] && return 0
  python3 - "$OUT/$1/run/receipt.json" "$1" "$2" "$OUT/$1/launch_unix" > "$OUT/$1/endpoint.json" <<'EOF' || log "endpoint $1 FAILED"
import json, sys
from pathlib import Path
receipt, arm, hours, launch = sys.argv[1:5]
r = json.loads(Path(receipt).read_text())
args = r["cli_argv"]
out = Path(args[args.index("--output-dir") + 1])
files = sorted(p for p in out.glob("wrfout_d0*"))
writes = {p.name: p.stat().st_mtime for p in files}
t0 = float(r["t0_wall_utc"])
ok = (r.get("end_reason") == "completed" and len(files) == len(r["outputs"]) and bool(writes)
      and all(t0 <= t <= t0 + float(r["t_end"]) + 0.1 for t in writes.values()))
print(json.dumps({"benchmark_receipt": receipt, "git_head": r.get("git_head"), "src_tree": r.get("src_tree"), "arm": arm,
                  "hours": int(hours), "t0_wall_utc": t0, "launch_unix": float(Path(launch).read_text()),
                  "file_write_unix": writes, "n_outputs_receipt": len(r["outputs"]), "valid": ok,
                  "wall_s": max(writes.values()) - t0 if writes else None,
                  "s_per_fc_h": (max(writes.values()) - t0) / int(hours) if writes else None,
                  "method": "original output file mtimes, captured right after the arm inside the completed run window (E162)"},
                 indent=1))
EOF
}

if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET: stop"; exit 3; fi
# --- untimed: v0.4.0 PROD programs into the private layer (P untouched) ---
arm A0 A 3 660 1200 && arm A0w A 3 60 300 && A_OK=1 || A_OK=
# --- timed, interleaved A B A B ---
if [[ -n "$A_OK" ]]; then
  for spec in A1:A B1:B A2:A B2:B; do
    name=${spec%%:*}; side=${spec##*:}
    core_gate "$name" && arm "$name" "$side" 24 115 420 "w41-bdyrec S1AB $name" && endpoint "$name" 24
  done
fi
stop_scopes
log "END GPU part: $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, scopes left: $(session_scopes | wc -l)"
