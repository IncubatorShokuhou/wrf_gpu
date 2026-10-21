#!/usr/bin/env bash
# w43-coastwind Q-B GPU arm (v2, critic 23:14Z: no sha256sum -L, explicit hash refusal, bounded diagnostics): v0.4.2 sealed bundle (release defaults), SAME inputs W9N 20260316_12z input42 (guarded stage S02),
# namelist identical except max_dom 3 vs 9 (feedback 0), 3 h each, sequential (m3 first: warm date-blind 3-nest key;
# m9: one cold 9-nest compile into the private layer). Then d01/d02/d03 frames are compared bitwise on CPU (cmp_maxdom.py).
# Mirrors the CPU-WRF test analysis/maxdom/run_maxdom.sh (same inputs).
# usage: scripts/with_gpu_lock.sh --label w43-coastwind --timeout <lock-wait s> -- bash gpu_maxdom_v2.sh
# Slot bound: each arm's consumer timeout = (END - real_now - 360) x 0.9, leaving >= 300 s for the bounded teardown (2 x 120 s).
# Env: GA_START_UTC (refuse before), GA_END_UTC (hard end; refuse to start if < GA_MIN_S left, default 1800), GA_MAILBOX.
# CLOCK SKEW (manager 2026-10-20): every window decision uses REAL UTC from <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
# (system clock ~+1 h, slewing); system-clock durations (coreutils timeout) are scaled x0.9 against the chrony slew.
set -uo pipefail
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
G=<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/gpucases
OUT=${GA_OUT:-<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/gpu_maxdom}
LAYER=<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/layer
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
RU=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
real_now() { local e; e=$("$RU" 2>/dev/null); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric: '$e'" >&2; exit 7; }; echo "$e"; }
post() { printf '%s w43-coastwind: [real %s] %s\n' "$(date -u +%FT%TZ)" "$(date -u -d @"$(real_now)" +%FT%TZ)" "$*" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET"; exit 3; fi
T0=$(real_now) || exit 7
T0SYS=$(date +%s)
START=$(date -u -d "${GA_START_UTC:?}" +%s); END=$(date -u -d "${GA_END_UTC:?}" +%s)
(( T0 >= START )) || { echo "too early (window starts ${GA_START_UTC})"; exit 4; }
(( END - T0 >= ${GA_MIN_S:-1800} )) || { post "GPU NOT STARTED: only $(( (END - T0) / 60 )) min left before ${GA_END_UTC}"; exit 6; }
for m in m3 m9; do [ -s "$G/$m/namelist.input" ] || { echo "missing case $G/$m"; exit 2; }; done
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1)
(( FREE_G >= 140 )) || { post "GPU NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < 140"; exit 4; }
[ -e "$OUT" ] && { echo "refusing existing $OUT (fresh output root per attempt)"; exit 2; }
mkdir -p "$OUT" "$LAYER"
timeout 20 journalctl -k --since "-6h" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_before.txt"
(cd "$P" && timeout 300 sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_before.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_before.txt")
grep -qx 'rc=0' "$OUT/bundle_check_before.txt" || { post "GPU NOT STARTED: bundle check before failed"; exit 8; }
timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$OUT/nvidia_before.csv" 2>&1
timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/nvidia_before.csv" 2>&1
for m in m3 m9; do
  files=(); for f in "$G/$m"/*; do files+=("$(readlink -f "$f")"); done
  timeout 300 sha256sum "${files[@]}" > "$OUT/inputs_$m.sha256"; hr=$?
  n=$(grep -c . "$OUT/inputs_$m.sha256")
  (( hr == 0 && n == ${#files[@]} && n > 0 )) || { post "GPU NOT STARTED: input hashing failed for $m (rc $hr, $n/${#files[@]})"; exit 8; }
done
rc=0
finish() { timeout 20 journalctl -k --since "@$T0SYS" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_during.txt";
  post "GPU END rc=$rc $(( ($(real_now) - T0 + 59) / 60 )) GPU-min (lock released next); out $OUT; $(grep -h BUNDLE_CONSUMER "$OUT"/consumer_*.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; }
trap finish EXIT
post "GPU START Q-B max_dom 3 vs 9 (v0.4.2 bundle, W9N 0316 inputs, 3 h each, sequential), cap $(( (END - T0) / 60 )) min, ends by $(date -u -d @$END +%TZ)"
# shellcheck disable=SC1091
source "$P/bundle.env"
export JAX_PLATFORMS=cuda TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/tmp
mkdir -p "$TMPDIR"
for m in m3 m9; do
  NOW=$(real_now) || exit 7
  REM=$(( END - NOW - 360 ))
  (( REM >= 600 )) || { echo "skip $m: $REM s left" >> "$OUT/skipped.txt"; rc=7; break; }
  timeout --signal=TERM --kill-after=60 "$(( REM * 9 / 10 ))" bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$LAYER" -- \
    bash "$P/src/scripts/run_parallel_cases.sh" --out-root "$OUT/runs_$m" --max-parallel 1 --host-gb-per-case 12 "$G/$m" -- \
    --domains-from-namelist --emit-initial-history --hours 3 > "$OUT/consumer_$m.log" 2>&1
  r=$?; echo "$m rc=$r" >> "$OUT/rc.txt"; (( r != 0 )) && rc=$r
done
(cd "$P" && timeout 120 sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_after.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_after.txt")
for d in "$OUT"/runs_*/; do [ -d "$d" ] && (cd "$d" && find . -name 'wrfout_d0*' | sort | timeout 120 xargs -r sha256sum > wrfout_sha.txt); done
exit "$rc"
