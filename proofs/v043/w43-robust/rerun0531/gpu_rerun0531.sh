#!/usr/bin/env bash
# w43-robust item 1: ONE GPU rerun of ALISIOS E2 case 20250531_18z_ERA5_r1 (slow outlier 1180 s, SIGKILLed by the E2 wall cap) with the
# SAME v0.4.2 sealed bundle and the SAME CLI call as the E2 harness (gpu.log of attempt a1), on a quiet <DATA_ROOT>, GPU-arm process pinned to
# ONE host core like E2 (core 4 there, core 3 here). Records iostat (nvme3n1 = <DATA_ROOT>) + pidstat -d/-u of the run every 5 s.
# Decision [pre-registered]: wall <= 600 s and 73 frames per domain -> the outlier was environmental (disk contention);
# wall > 900 s on a quiet disk (nvme3 util < 50 %) -> case-intrinsic slowness -> profile next.
# usage: scripts/with_gpu_lock.sh --label w43-robust --timeout <lock-wait s> -- bash gpu_rerun0531.sh
# Probes/monitors run with the GPU-lock fd 9 closed so an orphaned hung probe can never keep the lock (cr-w43-robust).
# Env: GA_START_UTC (refuse before), GA_END_UTC (hard end; refuse to start if < GA_MIN_S left, default 1500), GA_MAILBOX.
# All window times are REAL UTC (manager 2026-10-20 22:18Z: system clock ~+1 h fast since the wake-up) -> real_utc.sh, never raw date;
# both clocks are recorded in clocks.txt. Run wall is measured on CLOCK_MONOTONIC (python) and is immune to the skew.
set -uo pipefail
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
CASE=${GA_CASE:-<DATA_ROOT>/wrf_gpu2/scratch/w43-robust/case0531/input}
OUT=${GA_OUT:-<DATA_ROOT>/wrf_gpu2/scratch/w43-robust/rerun0531}
LAYER=<DATA_ROOT>/wrf_gpu2/scratch/w43-robust/layer
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
CORE=${GA_CORE:-3}
REAL_UTC=${GA_REAL_UTC:-<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh}
now() { local e; e=$("$REAL_UTC" 2>/dev/null); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric: '$e'" >&2; exit 7; }; echo "$e"; }
post() { printf '%s w43-robust: %s (real UTC; system clock %s)\n' "$("$REAL_UTC" --iso)" "$*" "$(date -u +%TZ)" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET"; exit 3; fi
if [[ "${GA_INNER:-0}" != 1 ]]; then
  # Slot-budget backstop (cr-w43-robust 21:33:38Z): validation, probes, model and finish together can never outlive GA_END_UTC.
  # uutils timeout signals the whole process group; the inner TERM trap still posts GPU END.
  E=$(now) || exit 7; ENDX=$(date -u -d "${GA_END_UTC:?}" +%s)
  BUDGET=$(( (ENDX - E - 20) * 9 / 10 ))
  (( BUDGET > 0 )) || { echo "slot already over (${GA_END_UTC})"; exit 6; }
  timeout --signal=TERM --kill-after=20 "$BUDGET" env GA_INNER=1 bash "$0" "$@"; r=$?
  (( r == 124 || r == 137 )) && post "GPU END by slot backstop rc=$r (budget ${BUDGET} s exhausted)"
  exit "$r"
fi
T0=$(now) || exit 7; SYS_T0=$(date +%s)
START=$(date -u -d "${GA_START_UTC:?}" +%s); END=$(date -u -d "${GA_END_UTC:?}" +%s)
(( T0 >= START )) || { echo "too early (window starts ${GA_START_UTC})"; exit 4; }
(( END - T0 >= ${GA_MIN_S:-1500} )) || { post "GPU NOT STARTED: only $(( (END - T0) / 60 )) min left before ${GA_END_UTC}"; exit 6; }
for f in namelist.input wrfbdy_d01 wrfinput_d01 wrfinput_d02 wrfinput_d03 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do
  [ -s "$CASE/$f" ] || { echo "missing $CASE/$f"; exit 2; }; done
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1)
(( FREE_G >= 150 )) || { post "GPU NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < 150"; exit 4; }
[ -e "$OUT" ] && { echo "refusing existing $OUT (fresh output root per attempt)"; exit 2; }
mkdir -p "$OUT/proof" "$LAYER"
printf 'real_t0=%s system_t0=%s chrony=%s\n' "$T0" "$SYS_T0" "$(chronyc tracking 2>/dev/null | grep '^System time' | tr -s ' ')" > "$OUT/clocks.txt"
timeout -k 5 30 journalctl -k --since "-6h" 2>/dev/null 9>&- | grep -i hibernat > "$OUT/hibernate_before.txt"
(cd "$P" && timeout -k 10 180 sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_before.txt" 2>&1 9>&-; echo "rc=$?" >> "$OUT/bundle_check_before.txt")
grep -qx "rc=0" "$OUT/bundle_check_before.txt" || { post "GPU NOT STARTED: bundle SHA256SUMS check failed/timed out before the run"; exit 5; }
{ timeout -k 5 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 9>&- &&
  timeout -k 5 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv 9>&-; } > "$OUT/nvidia_before.csv" 2>&1 ||
  { post "GPU NOT STARTED: nvidia-smi probe failed or hung > 20 s (GPU suspended/unavailable?)"; exit 5; }
(cd "$CASE" && sha256sum $(ls) > "$OUT/inputs.sha256") || { echo "input hashing failed"; exit 2; }
NOW=$(now) || exit 7
REM=$(( (END - NOW - 60) * 9 / 10 ))   # coreutils timeout runs on the slewing system clock: x0.9 margin
(( REM >= 300 )) || { post "GPU NOT STARTED: only ${REM} s model budget left before ${GA_END_UTC}"; exit 6; }
LC_ALL=C iostat -x -t -y nvme3n1 5 > "$OUT/iostat_nvme3.txt" 2>&1 9>&- & IOS=$!
rc=0
finish() { kill "$IOS" "${PDS:-}" 2>/dev/null; [ -n "${RUNPID:-}" ] && kill -TERM "$RUNPID" 2>/dev/null
  timeout -k 5 30 journalctl -k --since "@$SYS_T0" 2>/dev/null 9>&- | grep -i hibernat > "$OUT/hibernate_during.txt"
  W=$(( $(now) - T0 ))
  N=$(for d in 1 2 3; do printf 'd0%s=%s ' $d "$(ls "$OUT/raw" 2>/dev/null | grep -c "^wrfout_d0$d")"; done)
  post "GPU END rc=$rc wall ${W} s frames ${N}(lock released next); out $OUT"; }
trap finish EXIT
trap 'rc=124; exit 124' TERM
post "GPU START rerun E2 20250531_18z_ERA5_r1 (v0.4.2 bundle, 72 h, 3-nest, GPU arm on core $CORE), cap $(( (END - T0) / 60 )) min, ends by $(date -u -d @$END +%TZ) real"
# shellcheck disable=SC1091
source "$P/bundle.env"
export JAX_PLATFORMS=cuda TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w43-robust/tmp
mkdir -p "$TMPDIR"
RUN0=$(python3 -c 'import time;print(time.monotonic())')
taskset -c "$CORE" timeout --signal=TERM --kill-after=60 "$REM" bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$LAYER" -- \
  <USER_HOME>/miniconda3/bin/python3 -m gpuwrf.cli run --input-dir "$CASE" --output-dir "$OUT/raw" --proof-dir "$OUT/proof" --scratch-dir "$OUT/engine" \
  --domains-from-namelist --emit-initial-history --hours 72 > "$OUT/gpu.log" 2>&1 &
RUNPID=$!
LC_ALL=C pidstat -d -u -h -p "$RUNPID" -T ALL 5 > "$OUT/pidstat.txt" 2>&1 9>&- & PDS=$!
wait "$RUNPID"; rc=$?
RUN1=$(python3 -c 'import time;print(time.monotonic())')
python3 -c "import json,sys; json.dump({'run_wall_s': float(sys.argv[2])-float(sys.argv[1]), 'rc': int(sys.argv[3]), 'core': sys.argv[4]}, open(sys.argv[5],'w'))" \
  "$RUN0" "$RUN1" "$rc" "$CORE" "$OUT/run_wall.json"
(cd "$P" && timeout -k 10 180 sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_after.txt" 2>&1 9>&-; echo "rc=$?" >> "$OUT/bundle_check_after.txt")
(cd "$OUT/raw" 2>/dev/null && ls | sort | xargs -r stat -c '%n %s %Y' > "$OUT/wrfout_listing.txt")
exit "$rc"
