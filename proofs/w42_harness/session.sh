#!/usr/bin/env bash
# w42-harness S1H: v0.4.2 S1 re-baseline with the fixed bench harness (contract lanes/w42-harness.md), ONE lock session.
# usage: session.sh            (inside scripts/with_gpu_lock.sh --label w42-harness; frozen copy in tools_<sha12>/)
# Program = v0.4.1 sealed bundle P (release defaults) through P/tools/bundle_consumer.sh (P never written; PROD arms share
# layer L, the WN3 arm its own layer). Slots (pre-registered order, H_SLOTS):
#   W0          PROD 3 h, NEW harness, untimed warm-up of the shared layer (first run in a layer)
#   S_O1 S_N1 P_O1 P_N1 S_O2 S_N2 P_O2 P_N2   timed, interleaved: S_* = PROD 24 h (whole-run S1), P_* = PROD 3 h (proxy
#               seg 2-3); O = OLD harness (P/src/scripts/bench/s0_nested_harness.py, v0.4.1: device touch before the CLI
#               C-auto probe -> demand pool), N = NEW harness (this dir, --require-c-auto: refuses rc 4 off C-auto)
#   WN3         WN3 0227/0502/0614 3-nest 24 h, product launcher run_parallel_cases.sh --max-parallel 3 (C-auto by product)
# Every PROD arm: P/src/scripts/bench/run_arm.sh (s0_env.sh + flags env RELEASE_ENV: unset GPUWRF_NEST_PERF_TIMERS /
# GPUWRF_CENSUS / GPUWRF_WRF_ROOT, JAX_PLATFORMS=cuda -> the release env the bundle was primed with, so both harnesses hit
# the sealed programs and the sealed C-auto plan), env -u GPUWRF_LOCK_MIN_FREE_MIB (keeps the run_arm drain floor out of
# the trace-env hash), nice 10, systemd scope MemoryMax 20G. CPU keycheck.py gate first: measured env == bundle env on
# trace-env hash + plan key, plan present (else stop before any GPU work).
# DURING every arm (watchers on core $H_WATCH_CORE, lane core): core_watch.py (frozen copy of w41-rw3add a4938233) on
# 3,13,19,29 with the sweep-class abort (manager GO(b) 2026-10-18T21:49Z / V042 common): rule P one foreign process > 100 %
# of a core > 10 s, rule A foreign sum > 200 % > 10 s -> stop + repeat (max 2 tries); a never-aborting shadow watcher at the
# contract 20/50 %, sched_watch.py (arm cgroup PSI + CLI main-thread runqueue wait) and disk_watch.py (<DATA_ROOT> await)
# are DISCLOSURE. Pre-gate: quiet stretch H_QGATE_S (20 s) under the sweep rules (3 s), at most H_GATE_MAX_S (120 s), then
# start anyway (BUSY_START, disclosed); waits while updatedb runs. QUIET 'w42-harness S1H <slot>' during each arm.
# compile_bundle check BUNDLE_OK before, after the PROD block and after WN3 (else stop). Consumer last line per arm logged
# (compiled_new must be 0 on timed arms; disclosed otherwise). Endpoint per arm: original wrfout mtimes (E162).
# Window: ends at min(start + H_WINDOW_S (default 2700 s), H_END_UTC if set: the manager's hard end); an arm starts only
# if its NEED remains and its RuntimeMaxSec is cut to the remaining window. A manager HOLD/STOP line to w42-harness stops
# before the next arm. Posts 'S1H GPU START ...' and, on every exit, the exact line '<ts> w42-harness: S1H GPU END' (the
# next slot triggers on it) followed by a separate details line.
# Dry lifecycle (CPU, no GPU): H_DRY=<s> H_ROOT=<dir> H_QUIET_FILE=<file> H_MAILBOX=<file> H_CORES=<cores> -> each arm
# = sleep <s> inside the same scope/watchers (keycheck + bundle checks still run). H_DRY_PROBE=1 additionally runs the REAL
# arm command chains: PROD with probe_env.py in place of the harness (records the harness process env; run_arm's lock
# guard satisfied by a probe-only GPUWRF_GPU_LOCK_HELD=1, nothing imports jax), WN3 with the launcher's own --dry-run.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1
DRY=${H_DRY:-}
SCR=<DATA_ROOT>/wrf_gpu2/scratch/w42-harness
A=${H_ROOT:-$SCR/s1h}; R=$A/runs; LOG=$A/session.log; LAY=$A/layer; LAYW=$A/layer_wn3
CASES=${H_CASES:-$SCR/cases}; PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
CORES=${H_CORES:-3,13,19,29}; WCORE=${H_WATCH_CORE:-28}
SLOTS=${H_SLOTS:-W0:N:3 S_O1:O:24 S_N1:N:24 P_O1:O:3 P_N1:N:3 S_O2:O:24 S_N2:N:24 P_O2:O:3 P_N2:N:3 WN3:W:24}
RULE_P=${H_RULE_P_PCT:-100}; RULE_A=${H_RULE_A_PCT:-200}
QGATE_S=${H_QGATE_S:-20}; GATE_MAX_S=${H_GATE_MAX_S:-120}
MAILBOX=${H_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; QUIET_FILE=${H_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
PY=<USER_HOME>/miniconda3/bin/python3
TOOLREL=.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools
OLD_H=$P/src/scripts/bench/s0_nested_harness.py; NEW_H=$HERE/s0_nested_harness.py
S0ROOT=<USER_HOME>/wrf_gpu2_lanes/w42-harness/s0root/$(basename "$A")   # s0_guard refuses <DATA_ROOT> roots
[[ -z "$DRY" || ( "$QUIET_FILE" != /tmp/wrf_gpu2_quiet && -n "${H_MAILBOX:-}" && -n "${H_ROOT:-}" && "$CORES" != 3,13,19,29 ) ]] \
  || { echo "dry needs H_ROOT H_QUIET_FILE H_MAILBOX H_CORES"; exit 2; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
[ -e "$A/session.log" ] && { echo "$A already used: immutable sessions need a fresh root"; exit 2; }
for f in "$NEW_H" "$OLD_H" "$HERE/keycheck.py" "$HERE/probe_env.py" "$HERE/core_watch.py" "$HERE/sched_watch.py" "$HERE/disk_watch.py" "$PROD/wrfbdy_d01" \
         "$CASES/wn3_0227/wrfbdy_d01" "$CASES/wn3_0502/wrfbdy_d01" "$CASES/wn3_0614/wrfbdy_d01"; do
  [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
mkdir -p "$R" "$S0ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
post() { printf '%s w42-harness: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
T0=$(date +%s); DEADLINE=$(( T0 + ${H_WINDOW_S:-2700} ))
if [ -n "${H_END_UTC:-}" ]; then E=$(date -u -d "$H_END_UTC" +%s) || exit 2; (( E < DEADLINE )) && DEADLINE=$E; fi
need_of() { case $1 in 24) echo 300;; 3) echo 120;; W) echo 900;; *) echo 200;; esac; }
log "START S1H slots [$SLOTS] P=$P window end $(date -u -d @$DEADLINE +%TZ) pin $(taskset -cp $$ | awk -F': ' '{print $2}')"
log "tools $(cd "$HERE" && sha256sum session.sh s0_nested_harness.py keycheck.py core_watch.py sched_watch.py disk_watch.py summarize.py \
  | awk '{printf "%s:%s ", $2, substr($1,1,12)}') old_harness:$(sha256sum "$OLD_H" | cut -c1-12) P/src $(git -C "$P/src" rev-parse --short=12 HEAD)"

# --- release env as the bundle was primed (P/meta/arm.sh) and the measured harness env (run_arm.sh steps) ---
RELEASE_ENV=$A/release_env.sh
printf 'unset GPUWRF_WRF_ROOT GPUWRF_NEST_PERF_TIMERS GPUWRF_CENSUS\nexport JAX_PLATFORMS=cuda\n' > "$RELEASE_ENV"
FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
[[ "$FREE0" =~ ^[0-9]+$ ]] || FREE0=24000
DRAIN=$(( FREE0 > 9500 ? FREE0 - 1500 : 8000 ))   # run_arm drain floor = session-start free - 1.5 GiB (desktop varies)
CACHE_ENV=(GPUWRF_JAX_CACHE_DIR="$P/jax" JAX_COMPILATION_CACHE_DIR="$P/jax" GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$P/autotune"
           TRITON_CACHE_DIR="$P/triton" CUDA_CACHE_PATH="$P/cuda")
ARM_ENV=(S0_ROOT="$S0ROOT" S0_CPUS="$CORES" S0_MIN_FREE_MIB=16000 S0_MIN_FREE_DISK_GB=8 OMP_NUM_THREADS=4
         GPUWRF_BENCH_FLAGS_ENV="$RELEASE_ENV" GPUWRF_LOCK_MIN_FREE_MIB="$DRAIN" "${CACHE_ENV[@]}")
measured_env() {  # what the harness process sees: run_arm.sh steps (s0_env.sh, PYTHONPATH/T/BENCH_SRC_REPO, flags env) + env -u
  ( set +u; export "${ARM_ENV[@]}"; source "$P/src/$TOOLREL/s0_env.sh"
    export PYTHONPATH="$P/src/src" T="$P/src/$TOOLREL" BENCH_SRC_REPO="$P/src"; source "$GPUWRF_BENCH_FLAGS_ENV"
    env -u GPUWRF_LOCK_MIN_FREE_MIB env -0 > "$1" )
}
bundle_env() { ( set +u; source "$P/bundle.env"; export JAX_PLATFORMS=cuda; env -0 > "$1" ); }
mkdir -p "$A/keycheck"
measured_env "$A/keycheck/measured.env0"; bundle_env "$A/keycheck/bundle.env0"
KA=(run --input-dir "$PROD" --output-dir /nonexistent/w --scratch-dir /nonexistent/s --max-dom 2 --hours 3)
KB=(run --input-dir "$PROD" --output-dir /nonexistent/w --domains-from-namelist --hours 3)
j() { "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"; }
kcheck() {  # kcheck NAME SPEC...: keycheck.py real mode = the product launcher's backend-dark sizing child per env dump
  local n=$1; shift
  (cd "$A" && taskset -c "$WCORE" nice -n 5 timeout 900 \
    "$PY" "$HERE/keycheck.py" "$P/jax/cli_memory_plans" "$A/keycheck/$n.json" "$@" > "$A/keycheck/$n.log" 2>&1); }
if ! kcheck PROD "prod:harness=$A/keycheck/measured.env0=$(j "${KA[@]}")" "prod:bundle_A2=$A/keycheck/bundle.env0=$(j "${KB[@]}")"; then
  log "KEYCHECK PROD FAILED (harness env does not hit the sealed bundle): $(tail -c 600 "$A/keycheck/PROD.log")"; exit 6; fi
log "keycheck PROD OK: $(tail -1 "$A/keycheck/PROD.log" | cut -c1-400)"
wspec=(); for c in 0227 0502 0614; do
  wspec+=("wn3:$c=$A/keycheck/bundle.env0=$(j run --input-dir "$CASES/wn3_$c" --output-dir /nonexistent/w --domains-from-namelist --hours 24 --emit-initial-history)"); done
kcheck WN3 "${wspec[@]}" && log "keycheck WN3 OK: $(tail -1 "$A/keycheck/WN3.log" | cut -c1-300)" \
  || log "keycheck WN3 NOT OK (disclosed; the launcher records a missing plan by a solo first case): $(tail -c 400 "$A/keycheck/WN3.log")"

PREFIX="w42harness-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() { local u n=0; while [ -n "$(session_scopes)" ]; do for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && sleep 10; done; }
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w42-harness ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
DISKPID=; disk_off() { [ -n "$DISKPID" ] && { kill -TERM "$DISKPID" 2>/dev/null; wait "$DISKPID" 2>/dev/null; DISKPID=; }; }
child=
done_post() { post "S1H GPU END"
  post "S1H details: rc=$1, $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min, lock released as the session exits; log $LOG; table follows (CPU)"; }
on_signal() { trap - EXIT; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; stop_scopes; samplers_off; disk_off; quiet_off
  log "END (signal)"; done_post 143; exit 143; }
on_exit() { local rc=$?; stop_scopes; samplers_off; disk_off; quiet_off; log "exit rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min"; done_post "$rc"; }
trap on_signal TERM INT HUP
trap on_exit EXIT
chk() {  # chk OUTBASE: P must stay BUNDLE_OK (consumer view writes only the layers)
  JAX_PLATFORMS=cpu taskset -c "$WCORE" nice -n 5 "$PY" "$P/tools/compile_bundle.py" check --cache "$P/jax" \
    --manifest "$P/sealed/BUNDLE_MANIFEST.json" --out "$1.json" > "$1.txt" 2>&1
  /usr/bin/grep -q "^BUNDLE_OK missing=0 changed=0 new=0 " "$1.txt"; }
chk "$A/check_before" || { log "P check before FAILED: stop"; exit 7; }
post "S1H GPU START (one lock session, window end $(date -u -d @$DEADLINE +%TZ)): [$SLOTS] PROD old vs new harness interleaved + WN3 par3 24 h, v0.4.1 sealed bundle via consumer, sweep-class abort P>$RULE_P %/A>$RULE_A % >10 s on $CORES (20/50 shadow disclosed), QUIET per arm"
taskset -c "$WCORE" nice -n 5 "$PY" "$HERE/disk_watch.py" sample "$A/disk.jsonl" > "$A/disk_watch.log" 2>&1 &
DISKPID=$!
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {  # samplers_on RUN_DIR UNIT
  if [ -z "$DRY" ]; then
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
    SAMPLERS+=($!)
    taskset -c "$WCORE" nvidia-smi dmon -s pucm -o T > "$1/gpu_dmon.txt" 2>/dev/null &
    SAMPLERS+=($!)
  fi
  taskset -c "$WCORE" nice -n 5 "$PY" "$HERE/core_watch.py" "$CORES" "$RULE_P" 10 "$RULE_A" "$$" "$1/core_watch.jsonl" \
    "$1/core_watch_summary.json" "$1/ABORT" 1 > "$1/core_watch.log" 2>&1 &
  SAMPLERS+=($!)
  taskset -c "$WCORE" nice -n 5 "$PY" "$HERE/core_watch.py" "$CORES" 20 1e9 50 "$$" "$1/core_watch_strict.jsonl" \
    "$1/core_watch_strict_summary.json" "$1/ABORT_STRICT_NEVER" 1 > "$1/core_watch_strict.log" 2>&1 &
  SAMPLERS+=($!)
  taskset -c "$WCORE" nice -n 5 "$PY" "$HERE/sched_watch.py" "$2" "$1/sched_watch.jsonl" "$1/sched_watch_summary.json" 1 \
    > "$1/sched_watch.log" 2>&1 &
  SAMPLERS+=($!)
}
MB_FROM=$(wc -l < "$MAILBOX")
hold_stop() { tail -n +$((MB_FROM + 1)) "$MAILBOX" | /usr/bin/grep -q -E '^20[0-9T:.+Z-]+ manager -> [^:]*w42-harness[^:]*:.*\b(HOLD|STOP)\b'; }
pre_gate() {  # pre_gate SLOT KINDHOURS
  local i=0 t_end=$(( $(date +%s) + GATE_MAX_S )) g
  while :; do
    i=$((i + 1)); g="$A/gate/$1.$i"; mkdir -p "$g"
    taskset -c "$WCORE" nice -n 5 "$PY" "$HERE/core_watch.py" "$CORES" "$RULE_P" 3 "$RULE_A" "$$" "$g/w.jsonl" "$g/s.json" "$g/ABORT" 1 \
      > "$g/w.log" 2>&1 &
    child=$!; sleep "$QGATE_S" & local sl=$!; wait "$sl"; kill -TERM "$child" 2>/dev/null; wait "$child"; child=
    if pgrep -f '^/usr/(s)?bin/updatedb' > /dev/null 2>&1; then
      echo '{"rule": "UPDATEDB", "note": "updatedb running (system cron, <DATA_ROOT> IO)"}' > "$g/ABORT"; fi
    if [ ! -s "$g/ABORT" ]; then
      printf '%s\t%s\tfree\tquiet-stretch %s s, check %s\n' "$1" "$(date -u +%FT%TZ)" "$QGATE_S" "$i" >> "$A/core_gate.tsv"; return 0; fi
    hold_stop && return 0
    (( $(date +%s) >= t_end )) && break
    (( DEADLINE - $(date +%s) >= $(need_of "$2") )) || break
  done
  printf '%s\t%s\tBUSY_START\tafter %s checks: %s\n' "$1" "$(date -u +%FT%TZ)" "$i" "$(cut -c1-200 "$g/ABORT")" >> "$A/core_gate.tsv"
}
endpoint() {  # endpoint DIR: E162 publication receipt (original wrfout mtimes inside the completed run window)
  [ -n "$DRY" ] && return 0
  "$PY" - "$1" > "$1/endpoint.json" <<'EOF' || log "endpoint $1 FAILED"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
r = json.loads((d / "run" / "receipt.json").read_text())
args = r["cli_argv"]
out = Path(args[args.index("--output-dir") + 1])
files = sorted(out.glob("wrfout_d0*"))
writes = {p.name: p.stat().st_mtime for p in files}
t0 = float(r["t0_wall_utc"])
launch = float((d / "launch_unix").read_text())
ok = (r.get("end_reason") == "completed" and r.get("rc") == 0 and len(files) == len(r["outputs"]) and bool(writes)
      and all(t0 <= t <= t0 + float(r["t_end"]) + 0.1 for t in writes.values()))
last = max(writes.values()) if writes else None
print(json.dumps({"receipt": str(d / "run" / "receipt.json"), "git_head": r.get("git_head"), "src_tree": r.get("src_tree"),
                  "t0_wall_utc": t0, "harness_launch_wall_utc": r.get("launch_wall_utc"), "session_launch_unix": launch,
                  "allocator": r.get("allocator"), "pre_exec": r.get("pre_exec"), "file_write_unix": writes,
                  "n_outputs_receipt": len(r["outputs"]), "valid": ok,
                  "wall_from_session_launch_s": last - launch if last else None, "wall_from_receipt_t0_s": last - t0 if last else None,
                  "method": "original output file mtimes, captured right after the arm inside the completed run window (E162)"},
                 indent=1))
EOF
}
run_slot() {  # run_slot SLOT KIND(O|N|W) TRY HOURS -> 0 done, 9 no time, 8 aborted (repeat), 4 new harness off C-auto, 99 stop
  local name=$1 kind=$2 try=$3 hours=$4 need rmax d cmd=() lay=$LAY mem=20G
  need=$(need_of "$([[ $kind == W ]] && echo W || echo "$hours")")
  case $kind:$hours in W:*) rmax=1200; mem=40G; lay=$LAYW;; *:24) rmax=480;; *) rmax=300;; esac
  [[ $name == W0 ]] && rmax=420
  (( DEADLINE - $(date +%s) >= need )) || { log "SKIP $name: window too short"; return 9; }
  d=$R/$name; mkdir -p "$d"
  pre_gate "$name" "$([[ $kind == W ]] && echo W || echo "$hours")"
  hold_stop && { log "manager HOLD/STOP: stop before $name"; return 99; }
  (( DEADLINE - $(date +%s) >= need )) || { log "SKIP $name: window too short after the pre-gate"; return 9; }
  if [[ $kind == W ]]; then
    local probe=(); [ -n "$DRY" ] && [ -n "${H_DRY_PROBE:-}" ] && probe=(--dry-run)
    cmd=(env "${CACHE_ENV[@]}" PYTHONPATH="$P/src/src" JAX_PLATFORMS=cuda TMPDIR="$A/tmp/$name"
         bash "$P/src/scripts/run_parallel_cases.sh" "${probe[@]}" --max-parallel 3 --out-root "$d/out"
         "$CASES/wn3_0227" "$CASES/wn3_0502" "$CASES/wn3_0614" -- --domains-from-namelist --hours "$hours" --emit-initial-history)
    mkdir -p "$A/tmp/$name"
  else
    local h=$NEW_H extra=(--require-c-auto); [[ $kind == O ]] && { h=$OLD_H; extra=(); }
    cmd=(env "${ARM_ENV[@]}" BENCH_ARM_DIR="$d" BENCH_QUIET_FILE="$QUIET_FILE"
         nice -n 10 bash "$P/src/scripts/bench/run_arm.sh" "$P/src/$TOOLREL/s0_env.sh" "$P/src" "$P/src/$TOOLREL" "" "" 0 --
         env -u GPUWRF_LOCK_MIN_FREE_MIB "$PY" "$h" --input-dir "$PROD" --out "$d/run" --hours "$hours" --tag "w42h_${name}" "${extra[@]}")
  fi
  if [ -n "$DRY" ] && [ -n "${H_DRY_PROBE:-}" ]; then
    [[ $kind == W ]] || cmd=(env GPUWRF_GPU_LOCK_HELD=1 "${cmd[@]/#$h/$HERE/probe_env.py}")
  elif [ -n "$DRY" ]; then cmd=(sleep "$DRY"); fi
  printf '%s %s\n' "w42-harness S1H $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  local unit="$PREFIX-$name-$try" left=$(( DEADLINE - $(date +%s) ))
  (( left < rmax )) && rmax=$left   # never run past the window / the manager's hard end
  samplers_on "$d" "$unit.scope"; host_load "before $name try $try"
  log "ARM $name kind $kind try $try $hours h cap ${rmax}s"
  date +%s.%N > "$d/launch_unix"
  systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$mem" -p MemorySwapMax=0 -p RuntimeMaxSec="$rmax" -- \
    bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$lay" -- "${cmd[@]}" > "$d/stdout.log" 2> "$d/stderr.log" &
  child=$!
  local aborted=
  while kill -0 "$child" 2>/dev/null; do
    if [ -s "$d/ABORT" ] && [ -z "$aborted" ]; then
      aborted=$(cat "$d/ABORT"); log "ABORT $name try $try: $aborted"; systemctl --user stop "$unit.scope" 2>/dev/null
    fi
    sleep 1
  done
  wait "$child"; local rc=$?; child=
  date +%s.%N > "$d/t_end_unix"
  samplers_off; quiet_off; host_load "after $name try $try"
  log "ARM $name rc=$rc consumer: $(/usr/bin/grep -E '^BUNDLE_CONSUMER ' "$d/stdout.log" | tail -1)"
  if [ -n "$aborted" ]; then
    mv "$d" "$d.aborted$try"; printf '%s\t%s\tABORTED\t%s\n' "$name" "$(date -u +%FT%TZ)" "$aborted" >> "$A/core_gate.tsv"
    log "ARM $name aborted try $try (kept as $name.aborted$try)"; return 8
  fi
  if [[ $kind == W ]]; then
    [ -n "$DRY" ] || "$PY" - "$d/out" > "$d/mtimes.json" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(json.dumps({c.name: {p.name: p.stat().st_mtime for p in sorted((c / "wrfout").glob("wrfout_d0*"))}
                  for c in sorted(out.iterdir()) if (c / "wrfout").is_dir()}, indent=1))
EOF
  else
    endpoint "$d"
    if [ -z "$DRY" ]; then
      log "ARM $name allocator: $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); a=r.get("allocator") or {}; p=r.get("pre_exec") or {}
print(a.get("mode","(old harness: unattested)"), a.get("mem_fraction"), a.get("jax_bytes_limit"), "probe_dark", (p.get("nvml_probe") or {}).get("backend_dark"))' "$d/run/receipt.json" 2>&1) | CLI: $(/usr/bin/grep -h -o -E 'gpuwrf: C-auto[^;]{0,120}' "$d/stderr.log" 2>/dev/null | head -1)"
    fi
    [[ $kind == N && $rc == 4 ]] && { log "NEW harness refused off C-auto in $name: stop the PROD block"; return 4; }
  fi
  return 0
}
PROD_DONE=
for spec in $SLOTS; do
  IFS=: read -r name kind hours <<< "$spec"
  if [[ $kind == W && -z "$PROD_DONE" ]]; then
    PROD_DONE=1; chk "$A/check_after_prod" || { log "P NOT OK after the PROD block: stop"; exit 7; }; fi
  [[ $kind != W && -n "${PROD_STOP:-}" ]] && continue
  for try in 1 2; do
    run_slot "$name" "$kind" "$try" "$hours"; rc=$?
    (( rc == 99 )) && exit 7
    (( rc == 8 )) && continue
    break
  done
  (( rc == 4 )) && PROD_STOP=1
  [[ $name == W0 && -z "$DRY" ]] && ! /usr/bin/grep -q '^BUNDLE_CONSUMER rc=0 ' "$R/W0/stdout.log" 2>/dev/null \
    && { PROD_STOP=1; log "W0 warm-up failed: PROD block skipped"; }
  (( rc == 9 )) && [[ $kind != W ]] && log "window: remaining PROD slots skipped"
done
[ -n "$PROD_DONE" ] || chk "$A/check_after_prod" || { log "P NOT OK after the PROD block: stop"; exit 7; }
chk "$A/check_after" || { log "P NOT OK at the end"; exit 7; }
stop_scopes
log "DONE $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min; CPU follow-up: summarize.py $A"
