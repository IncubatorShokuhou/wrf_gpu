#!/usr/bin/env bash
# w42-bundle042 HERO session (manager 2026-10-19T12:21:49Z, option B): the README hero numbers on the SEALED v0.4.2 bundle,
# warm, FIXED bench harness (product CLI C-auto pool attested per receipt), in ONE separate GPU lock session after the
# main bundle session's GPU END (hero_trigger.sh starts it on '<ts> manager -> w42-bundle042: HERO GO'); own cap 30 GPU-min,
# must END by 18:15Z. Derived from the critic-approved w42-harness S1H session (pin 8c92c3c5d7:proofs/w42_harness/session.sh,
# MERGE 2026-10-19T06:10:34Z). Changes: P = the v0.4.2 sealed bundle; harness = P/src/scripts/bench/s0_nested_harness.py
# (the merged fixed harness, always --require-c-auto; no old-harness arms); WN3 inputs = this lane's staged cases
# (sha-checked by the trigger and here); pre-registered slots + hero rule below; iostat -x per arm; watchers / samplers on
# lane cores; a keycheck failure skips only its kind (PROD or WN3); launcher --host-gb-per-case 8.
# usage: session.sh            (inside scripts/with_gpu_lock.sh --label w42-bundle042; frozen copy in $L/tools_hero_<sha12>/)
# Slots (pre-registered order, H_SLOTS):
#   W0      PROD 3 h, fixed harness, untimed warm-up of the shared PROD layer (first run in a layer)
#   H1 H2   PROD 24 h whole-run, fixed harness --require-c-auto, shared PROD layer               -> HERO_PROD_*
#   WN3     WN3 0227/0502/0614 3-nest 24 h --emit-initial-history, product launcher run_parallel_cases.sh --max-parallel 3
#           --host-gb-per-case 8 (S1H warm VmHWM 6.21-6.35 GB x 1.1 = 7.0; the 16 GB default would make the 3rd start
#           depend on the host's MemAvailable >= 56 GB), fresh layer, C-auto by product            -> HERO_PAR3_*
#   H3      PROD 24 h, ONLY if no CLEAN PROD 24 h arm exists after H1/H2 (clean = accepted, rc 0, arm-window <DATA_ROOT>
#           await <= 50 ms) and the window still holds it; runs after WN3
# Hero rule (pre-registered; hero_summary.py computes it): HERO_PROD_S_PER_FCH = mean whole-run s/fc-h (process start =
#   harness launch -> last original wrfout mtime, E162; the basis of the released 4.600) over the CLEAN PROD 24 h arms;
#   none clean -> mean over all accepted 24 h arms, labelled DISTURBED. HERO_PROD_SECONDS = that wall (x 24). Session-clock
#   and process-t0 bases and stepping (seg2+ median) disclosed. HERO_PAR3_S_PER_CASE_H = WN3 batch wall (session launch ->
#   last wrfout of the 3 cases) / 72 case-hours; valid iff all 3 started together (start events <= 5 s), rc 0 x 3, 75 wrfout
#   each, C-auto plan + pool line in every case receipt, AOT-warm each, consumer compiled_new=0; await > 50 ms = DISTURBED.
# Every PROD arm: P/src/scripts/bench/run_arm.sh (s0_env.sh + flags env RELEASE_ENV: unset GPUWRF_NEST_PERF_TIMERS /
# GPUWRF_CENSUS / GPUWRF_WRF_ROOT, JAX_PLATFORMS=cuda -> the release env the bundle was primed with), env -u
# GPUWRF_LOCK_MIN_FREE_MIB, nice 10, systemd scope MemoryMax 20G (WN3 40G). CPU keycheck.py gate first (measured env ==
# bundle env on trace-env hash + C-auto plan key, plan present; env dumps carry PYTHONDONTWRITEBYTECODE=1 so the check
# never writes P/src; no key reads it).
# DURING every arm: core_watch.py (frozen copy of w41-rw3add a4938233) on 3,13,19,29 with the sweep-class abort (one foreign
# process > 100 % of a core > 10 s, or foreign sum > 200 % > 10 s -> stop + repeat, max 2 tries); never-aborting shadow
# watcher at 20/50 %, sched_watch.py (arm cgroup PSI + CLI main-thread runqueue wait), disk_watch.py (session-wide 1 s
# <DATA_ROOT> await), per-arm `iostat -x -d -t nvme3n1 1`, nvidia-smi dmon + per-PID VRAM = DISCLOSURE. Pre-gate: quiet
# stretch H_QGATE_S (20 s) under the sweep rules (3 s), at most H_GATE_MAX_S (120 s), then start anyway (BUSY_START,
# disclosed); waits while updatedb runs. QUIET 'w42-bundle042 HERO <slot>' during each arm. compile_bundle check BUNDLE_OK
# before, after the PROD block, after WN3 and at the end (else stop). Endpoint per arm: original wrfout mtimes (E162).
# Window: ends at min(start + H_WINDOW_S (default 1800 s = the 30 GPU-min cap), H_END_UTC (default 2026-10-19T18:15:00Z));
# an arm starts only if its NEED remains, its RuntimeMaxSec is cut to the window. A manager HOLD/STOP line to w42-bundle042
# ('HERO HOLD' / 'HERO STOP') stops before the next arm. Posts 'HERO START ...' and, on every exit, 'HERO END rc=<rc> <n> GPU-min ...'.
# Dry lifecycle (CPU, no GPU): H_DRY=<s> H_ROOT=<dir> H_QUIET_FILE=<file> H_MAILBOX=<file> H_CORES=<cores> [H_P=<bundle>]
# -> each arm = sleep <s> inside the same scope/watchers (keycheck --emulate-cpu, bundle checks still run). H_DRY_PROBE=1
# additionally runs the REAL arm command chains: PROD with probe_env.py in place of the harness (records the harness
# process env; run_arm's lock guard satisfied by a probe-only GPUWRF_GPU_LOCK_HELD=1, nothing imports jax), WN3 with the
# launcher's own --dry-run (admission plan JSON).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DRY=${H_DRY:-}
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
[ -n "${H_P:-}" ] && { [ -n "$DRY" ] || { echo "H_P only in dry runs"; exit 2; }; P=$H_P; }
L=<USER_HOME>/wrf_gpu2_lanes/w42-bundle042; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042
A=${H_ROOT:-$S/hero}; R=$A/runs; LOG=$A/session.log; LAY=$A/layer; LAYW=$A/layer_wn3
CASES=${H_CASES:-$S/cases}; PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
CORES=${H_CORES:-3,13,19,29}; WCORE=${H_WATCH_CORE:-8}; SCORE=${H_SAMPLE_CORE:-9}
SLOTS=${H_SLOTS:-W0:N:3 H1:N:24 H2:N:24 WN3:W:24}
RULE_P=${H_RULE_P_PCT:-100}; RULE_A=${H_RULE_A_PCT:-200}
QGATE_S=${H_QGATE_S:-20}; GATE_MAX_S=${H_GATE_MAX_S:-120}; DISK_LIMIT=${H_DISK_LIMIT_MS:-50}
MAILBOX=${H_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}; QUIET_FILE=${H_QUIET_FILE:-/tmp/wrf_gpu2_quiet}
PY=<USER_HOME>/miniconda3/bin/python3
TOOLREL=.agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools
NEW_H=$P/src/scripts/bench/s0_nested_harness.py
S0ROOT=$L/s0root/$(basename "$A")   # s0_guard refuses <DATA_ROOT> roots
[[ -z "$DRY" || ( "$QUIET_FILE" != /tmp/wrf_gpu2_quiet && -n "${H_MAILBOX:-}" && -n "${H_ROOT:-}" && "$CORES" != 3,13,19,29 ) ]] \
  || { echo "dry needs H_ROOT H_QUIET_FILE H_MAILBOX H_CORES"; exit 2; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "$DRY" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET: $(cat /tmp/wrf_gpu2_quiet)"; exit 3; fi
[ -e "$A/session.log" ] && { echo "$A already used: immutable sessions need a fresh root"; exit 2; }
for f in "$NEW_H" "$HERE/keycheck.py" "$HERE/probe_env.py" "$HERE/core_watch.py" "$HERE/sched_watch.py" "$HERE/disk_watch.py" \
         "$HERE/hero_disk.py" "$HERE/hero.env" "$PROD/wrfbdy_d01" "$CASES/wn3_0227/wrfbdy_d01" "$CASES/wn3_0502/wrfbdy_d01" \
         "$CASES/wn3_0614/wrfbdy_d01"; do
  [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
source "$HERE/hero.env"   # PIN TREE (the main session's pin.env, copied at freeze time)
if [ -z "$DRY" ]; then   # the measured program is exactly the sealed pin bundle
  [[ "$(git -C "$P/src" rev-parse HEAD)" == "$PIN" && "$(git -C "$P/src" rev-parse HEAD:src)" == "$TREE" ]] || { echo "P/src != pin"; exit 4; }
  [[ -z "$(git -C "$P/src" status --porcelain --untracked-files=all -- src scripts)" ]] || { echo "P/src dirty"; exit 4; }
  [ -s "$P/sealed/BUNDLE_MANIFEST.json" ] || { echo "P not sealed"; exit 4; }
fi
(cd "$CASES" && sha256sum -c --quiet "$HERE/hero_cases_sha256.txt") || { echo "WN3 case inputs changed (hero_cases_sha256.txt)"; exit 4; }
for v in $(compgen -e | /usr/bin/grep -E '^(GPUWRF_|XLA_|JAX_|CUDA_ROOT$|TF_)' | /usr/bin/grep -v -E '^GPUWRF_(GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)$'); do
  unset "$v"; done
mkdir -p "$R" "$S0ROOT"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
post() { printf '%s w42-bundle042: %s\n' "$(date -u +%FT%TZ)" "$*" >> "$MAILBOX"; }
T0=$(date +%s); DEADLINE=$(( T0 + ${H_WINDOW_S:-1800} ))
E=$(date -u -d "${H_END_UTC:-2026-10-19T18:15:00Z}" +%s) || exit 2; (( E < DEADLINE )) && DEADLINE=$E
need_of() { case $1 in 24) echo 300;; 3) echo 120;; W) echo 900;; *) echo 200;; esac; }
log "START HERO slots [$SLOTS] (+H3 if no clean PROD 24 h arm) P=$P pin ${PIN:0:12} window end $(date -u -d @$DEADLINE +%TZ) cpus $(taskset -cp $$ | awk -F': ' '{print $2}') dry='$DRY'"
log "tools $(cd "$HERE" && sha256sum hero_session.sh keycheck.py probe_env.py core_watch.py sched_watch.py disk_watch.py hero_disk.py hero_summary.py \
  | awk '{printf "%s:%s ", $2, substr($1,1,12)}') harness:$(sha256sum "$NEW_H" | cut -c1-12) P/src $(git -C "$P/src" rev-parse --short=12 HEAD)"

# --- release env as the bundle was primed (arm.sh: P/bundle.env, release defaults) and the measured harness env (run_arm.sh steps) ---
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
    env -u GPUWRF_LOCK_MIN_FREE_MIB PYTHONDONTWRITEBYTECODE=1 env -0 > "$1" )
}
bundle_env() { ( set +u; source "$P/bundle.env"; export JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1; env -0 > "$1" ); }
mkdir -p "$A/keycheck"
measured_env "$A/keycheck/measured.env0"; bundle_env "$A/keycheck/bundle.env0"
KA=(run --input-dir "$PROD" --output-dir /nonexistent/w --scratch-dir /nonexistent/s --max-dom 2 --hours 24)
KB=(run --input-dir "$PROD" --output-dir /nonexistent/w --domains-from-namelist --hours 3)   # = the bundle's A2 prime arm
j() { "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"; }
EMU=(); [ -n "$DRY" ] && EMU=(--emulate-cpu)   # dry: no lock -> CPU emulation (group agreement + darkness only)
kcheck() {  # kcheck NAME SPEC...: keycheck.py real mode = the product launcher's backend-dark sizing child per env dump
  local n=$1; shift
  (cd "$A" && taskset -c "$WCORE" nice -n 5 timeout 900 \
    "$PY" "$HERE/keycheck.py" "${EMU[@]}" "$P/jax/cli_memory_plans" "$A/keycheck/$n.json" "$@" > "$A/keycheck/$n.log" 2>&1); }
PROD_STOP=; WN3_STOP=
if kcheck PROD "prod:harness=$A/keycheck/measured.env0=$(j "${KA[@]}")" "prod:bundle_A2=$A/keycheck/bundle.env0=$(j "${KB[@]}")"; then
  log "keycheck PROD OK: $(tail -1 "$A/keycheck/PROD.log" | cut -c1-400)"
else
  PROD_STOP=1; log "KEYCHECK PROD FAILED (harness env does not hit the sealed bundle): PROD arms skipped: $(tail -c 600 "$A/keycheck/PROD.log")"
fi
wspec=(); for c in 0227 0502 0614; do
  wspec+=("wn3:$c=$A/keycheck/bundle.env0=$(j run --input-dir "$CASES/wn3_$c" --output-dir /nonexistent/w --domains-from-namelist --hours 24 --emit-initial-history)"); done
if kcheck WN3 "${wspec[@]}"; then log "keycheck WN3 OK: $(tail -1 "$A/keycheck/WN3.log" | cut -c1-300)"
else WN3_STOP=1; log "KEYCHECK WN3 FAILED (no sealed C-auto plan for the par3 args -> a solo sizing run, not par3): WN3 arm skipped: $(tail -c 400 "$A/keycheck/WN3.log")"; fi
[[ -n "$PROD_STOP" && -n "$WN3_STOP" ]] && { log "both keychecks failed: nothing to measure"; exit 6; }

PREFIX="w42hero-$$"
session_scopes() { systemctl --user list-units --type=scope --state=active,activating,deactivating --no-legend --plain \
    "$PREFIX-*" | awk '{print $1}'; }
stop_scopes() { local u n=0; while [ -n "$(session_scopes)" ]; do for u in $(session_scopes); do systemctl --user stop "$u"; done
    n=$((n + 1)); (( n > 30 )) && sleep 10; done; }
quiet_off() { if [ -e "$QUIET_FILE" ] && /usr/bin/grep -q '^w42-bundle042 ' "$QUIET_FILE"; then rm -f "$QUIET_FILE"; fi; }
SAMPLERS=(); samplers_off() { local p; for p in "${SAMPLERS[@]}"; do kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; SAMPLERS=(); }
DISKPID=; disk_off() { [ -n "$DISKPID" ] && { kill -TERM "$DISKPID" 2>/dev/null; wait "$DISKPID" 2>/dev/null; DISKPID=; }; }
child=
done_post() { post "HERO END rc=$1 $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released as the session exits); log $LOG; numbers follow (CPU summary)"; }
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
post "HERO START (separate lock session after GPU END, cap $(( (DEADLINE - T0) / 60 )) GPU-min, ends by $(date -u -d @$DEADLINE +%TZ)): [$SLOTS (+H3 only if no clean PROD 24 h arm)] PROD 24 h fixed harness --require-c-auto x2 + WN3 0227/0502/0614 par3 24 h product launcher, v0.4.2 sealed bundle via consumer, sweep-class abort P>$RULE_P %/A>$RULE_A % >10 s on $CORES (20/50 shadow disclosed), iostat + disk await per arm, QUIET per arm${PROD_STOP:+; PROD SKIPPED (keycheck)}${WN3_STOP:+; WN3 SKIPPED (keycheck)}"
taskset -c "$SCORE" nice -n 5 "$PY" "$HERE/disk_watch.py" sample "$A/disk.jsonl" > "$A/disk_watch.log" 2>&1 &
DISKPID=$!
host_load() { { echo "== $1 $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
    ps -eo psr,pcpu,pid,etimes,args --sort=-pcpu | head -15 | cut -c1-160; } >> "$A/HOST_LOAD.txt"; }
samplers_on() {  # samplers_on RUN_DIR UNIT
  if [ -z "$DRY" ]; then
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > "$1/vram_apps.csv" 2>/dev/null &
    SAMPLERS+=($!)
    taskset -c "$SCORE" nvidia-smi dmon -s pucm -o T > "$1/gpu_dmon.txt" 2>/dev/null &
    SAMPLERS+=($!)
  fi
  S_TIME_FORMAT=ISO taskset -c "$SCORE" iostat -x -d -t nvme3n1 1 > "$1/iostat_x.txt" 2>&1 &   # <DATA_ROOT> await per arm (manager 05:59:13Z)
  SAMPLERS+=($!)
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
hold_stop() {  # 'HERO HOLD' / 'HERO STOP' in a manager line to w42-bundle042 after the session start (hero-specific phrase:
  # main-session wording such as '>10 % = STOP' must not cancel the hero); grep reads ALL input (no -q, pipefail SIGPIPE)
  tail -n +$((MB_FROM + 1)) "$MAILBOX" | /usr/bin/grep -E '^20[0-9T:.+Z-]+ manager -> [^:]*w42-bundle042[^:]*:.*\bHERO (HOLD|STOP)\b' > /dev/null; }
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
disk_of() {  # disk_of DIR: arm-window <DATA_ROOT> await (launch_unix .. t_end_unix) -> DIR/disk_await.json; prints it
  "$PY" "$HERE/hero_disk.py" "$A/disk.jsonl" "$(cat "$1/launch_unix")" "$(cat "$1/t_end_unix")" "$DISK_LIMIT" > "$1/disk_await.json" 2>&1
  cat "$1/disk_await.json"; }
run_slot() {  # run_slot SLOT KIND(N|W) TRY HOURS -> 0 done, 9 no time, 8 aborted (repeat), 4 harness off C-auto, 99 stop
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
         bash "$P/src/scripts/run_parallel_cases.sh" "${probe[@]}" --max-parallel 3 --host-gb-per-case 8 --out-root "$d/out"
         "$CASES/wn3_0227" "$CASES/wn3_0502" "$CASES/wn3_0614" -- --domains-from-namelist --hours "$hours" --emit-initial-history)
    mkdir -p "$A/tmp/$name"
  else
    local h=$NEW_H
    cmd=(env "${ARM_ENV[@]}" BENCH_ARM_DIR="$d" BENCH_QUIET_FILE="$QUIET_FILE"
         nice -n 10 bash "$P/src/scripts/bench/run_arm.sh" "$P/src/$TOOLREL/s0_env.sh" "$P/src" "$P/src/$TOOLREL" "" "" 0 --
         env -u GPUWRF_LOCK_MIN_FREE_MIB "$PY" "$h" --input-dir "$PROD" --out "$d/run" --hours "$hours" --tag "w42hero_${name}" --require-c-auto)
  fi
  if [ -n "$DRY" ] && [ -n "${H_DRY_PROBE:-}" ]; then
    [[ $kind == W ]] || cmd=(env GPUWRF_GPU_LOCK_HELD=1 "${cmd[@]/#$h/$HERE/probe_env.py}")
  elif [ -n "$DRY" ]; then cmd=(sleep "$DRY"); fi
  printf '%s %s\n' "w42-bundle042 HERO $name" "$(date -u +%FT%TZ)" > "$QUIET_FILE"
  local unit="$PREFIX-$name-$try" left=$(( DEADLINE - $(date +%s) ))
  (( left < rmax )) && rmax=$left   # never run past the window / the hard end
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
  log "ARM $name disk: $(disk_of "$d")"
  if [[ $kind == W ]]; then
    [ -n "$DRY" ] || "$PY" - "$d/out" > "$d/mtimes.json" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print(json.dumps({c.name: {p.name: p.stat().st_mtime for p in sorted((c / "wrfout").glob("wrfout_d0*"))}
                  for c in sorted(out.iterdir()) if (c / "wrfout").is_dir()}, indent=1))
EOF
    [ -n "$DRY" ] || log "ARM $name cases: $("$PY" -c 'import json,sys,pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("*/receipt.json")):
    r=json.loads(p.read_text()); s=r.get("sizing") or {}
    print(r["name"], "rc", r.get("rc"), "start", r.get("start_utc"), "sizing", s.get("source"), "warm", r.get("aot_cache_warm"), "|", ";".join(r.get("allocator") or [])[:120])' "$d/out" 2>&1 | tr '\n' ' ')"
  else
    endpoint "$d"
    if [ -z "$DRY" ]; then
      log "ARM $name allocator: $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); a=r.get("allocator") or {}; p=r.get("pre_exec") or {}
print(a.get("mode","(unattested)"), a.get("mem_fraction"), a.get("jax_bytes_limit"), "probe_dark", (p.get("nvml_probe") or {}).get("backend_dark"))' "$d/run/receipt.json" 2>&1) | CLI: $(/usr/bin/grep -h -o -E 'gpuwrf: C-auto[^;]{0,120}' "$d/stderr.log" 2>/dev/null | head -1)"
    fi
    [[ $kind == N && $rc == 4 ]] && { log "harness refused off C-auto in $name: stop the PROD block"; return 4; }
  fi
  return 0
}
clean_prod() {  # number of CLEAN PROD 24 h arms so far: accepted dir, rc 0 (endpoint valid), arm-window await <= limit
  local n=0 x
  for x in H1 H2 H3; do
    [ -s "$R/$x/disk_await.json" ] || continue
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[2])); e={"valid": True} if sys.argv[1] == "-" else json.load(open(sys.argv[1]))
sys.exit(0 if e.get("valid") and d.get("disturbed") is False else 1)' "$([ -n "$DRY" ] && echo - || echo "$R/$x/endpoint.json")" \
      "$R/$x/disk_await.json" 2>/dev/null && n=$((n + 1))
  done
  echo "$n"; }
PROD_DONE=
for spec in $SLOTS; do
  IFS=: read -r name kind hours <<< "$spec"
  if [[ $kind == W && -z "$PROD_DONE" ]]; then
    PROD_DONE=1; chk "$A/check_after_prod" || { log "P NOT OK after the PROD block: stop"; exit 7; }; fi
  [[ $kind != W && -n "$PROD_STOP" ]] && continue
  [[ $kind == W && -n "$WN3_STOP" ]] && continue
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
if [ -z "$PROD_STOP" ] && [ "${H_H3:-1}" = 1 ]; then
  nc=$(clean_prod)
  if (( nc == 0 )); then
    log "no clean PROD 24 h arm (pre-registered H3 rule): H3"
    for try in 1 2; do run_slot H3 N "$try" 24; rc=$?; (( rc == 99 )) && exit 7; (( rc == 8 )) && continue; break; done
  else log "clean PROD 24 h arms: $nc -> no H3"; fi
fi
chk "$A/check_after" || { log "P NOT OK at the end"; exit 7; }
stop_scopes
log "DONE $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min; CPU follow-up: hero_summary.py $A"
