#!/usr/bin/env bash
# w44-gint GPU session = w43-teacher gpu_arms.sh (the launcher of the v0.4.2 teacher runs) with ONLY these changes:
# lane paths/labels; cases come in BATCHES of equal --hours (PLAN lines "HOURS CASE_DIR"), each batch one bundle-consumer
# call of run_parallel_cases.sh (max 3 parallel), batches sequential; <DATA_ROOT> floor check per batch (205 GiB + 23 GiB x
# hours/72 per case). Same sealed v0.4.2 bundle bfa2dc90, release defaults, product CLI, --domains-from-namelist
# --emit-initial-history. Purpose: perturbed-IC member reruns (1-ulp T, perturb_t.py) of the G-INT FAIL cases + unperturbed
# control m0 (must reproduce the teacher GPU_WRFOUT_MANIFEST sha256 for its frames).
# usage: scripts/with_gpu_lock.sh --label w44-gint --timeout <lock-wait s> -- bash gpu_members.sh PLAN
#   Env: GA_START_UTC (refuse before), GA_END_UTC (hard end; refuse a batch if < GA_MIN_S left), GA_OUT (output root), GA_MAILBOX.
set -uo pipefail
PLAN=$(readlink -f "$1")
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
OUT=${GA_OUT:-<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/gpu/session}
LAYER=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/layer
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
post() { printf '%s w44-gint: [real %s] %s\n' "$(date -u +%FT%TZ)" "$("$REAL" --iso)" "$*" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 || -n "${GA_DRY:-}" ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET"; exit 3; fi
rnow() { local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric"; exit 7; }; echo "$e"; }
T0=$(date +%s); R0=$(rnow) || exit 7
START=$(date -u -d "${GA_START_UTC:?}" +%s); END=$(date -u -d "${GA_END_UTC:?}" +%s)
(( R0 >= START )) || { echo "too early (window starts ${GA_START_UTC} real UTC)"; exit 4; }
mapfile -t ROWS < <(grep -v '^\s*\(#.*\)\?$' "$PLAN")
(( ${#ROWS[@]} >= 1 )) || exit 2
mapfile -t HOURS < <(printf '%s\n' "${ROWS[@]}" | awk '{print $1}' | awk '!s[$0]++')
for r in "${ROWS[@]}"; do
  read -r h c <<< "$r"; [[ $h =~ ^[0-9]+$ ]] && (( h % 3 == 0 && h > 0 && h <= 72 )) || { echo "bad hours $h (E187: multiple of 3)"; exit 2; }
  case "$(readlink -f "$c")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/*) echo "ALISIOS-root case $c"; exit 2;; esac
  for f in namelist.input wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfbdy_d01 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do
    [ -s "$c/$f" ] || { echo "missing $c/$f"; exit 2; }; done
done
[ -e "$OUT" ] && { echo "refusing existing $OUT (fresh output root per attempt)"; exit 2; }
[ -n "${GA_DRY:-}" ] && { echo "DRY OK: ${#ROWS[@]} cases in ${#HOURS[@]} batches (${HOURS[*]} h)"; exit 0; }
mkdir -p "$OUT" "$LAYER"
journalctl -k --since "-6h" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_before.txt"
(cd "$P" && sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_before.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_before.txt")
timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$OUT/nvidia_before.csv" 2>&1
timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/nvidia_before.csv" 2>&1
cp "$PLAN" "$OUT/plan.txt"
echo "system $(date -u +%FT%TZ) real $("$REAL" --iso) T0=$T0 R0=$R0 END=$END" > "$OUT/clocks.txt"
rc=0; done_b=0
finish() { journalctl -k --since "@$T0" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_during.txt";
  (cd "$P" && sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_after.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_after.txt")
  post "GPU END rc=$rc batches $done_b/${#HOURS[@]} $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); out $OUT; $(grep -h BUNDLE_CONSUMER "$OUT"/consumer_*.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; }
trap finish EXIT
post "GPU START v0.4.2 bundle G-INT member reruns: ${#ROWS[@]} cases in ${#HOURS[@]} batches (${HOURS[*]} h, max 3 parallel), ends by $(date -u -d @$END +%TZ) real"
# shellcheck disable=SC1091
source "$P/bundle.env"
export JAX_PLATFORMS=cuda TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w44-gint/tmp
mkdir -p "$TMPDIR"
for h in "${HOURS[@]}"; do
  mapfile -t CASES < <(printf '%s\n' "${ROWS[@]}" | awk -v h="$h" '$1==h{print $2}')
  REM=$(( (END - $(rnow) - 60) * 9 / 10 ))   # system-clock timeout, x0.9 against slew
  NEED_S=$(( 300 + 1500 * h / 72 ))          # measured: 3 parallel 72 h WN3 = 1206-1395 s (teacher), + startup
  (( REM >= NEED_S )) || { post "batch ${h} h NOT STARTED: ${REM} s left < ${NEED_S}"; rc=6; break; }
  FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1); NEED_G=$(( 205 + 23 * ${#CASES[@]} * h / 72 + 1 ))
  (( FREE_G >= NEED_G )) || { post "batch ${h} h NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < ${NEED_G}"; rc=4; break; }
  timeout --signal=TERM --kill-after=60 "$REM" bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$LAYER" -- \
    bash "$P/src/scripts/run_parallel_cases.sh" --out-root "$OUT/runs_${h}h" --max-parallel 3 --host-gb-per-case 8 "${CASES[@]}" -- \
    --domains-from-namelist --emit-initial-history --hours "$h" > "$OUT/consumer_${h}h.log" 2>&1
  rc=$?
  for d in "$OUT"/runs_${h}h/*/; do [ -d "$d" ] && (cd "$d" && find . -name 'wrfout_d0*' | sort | xargs -r sha256sum > wrfout_sha.txt); done
  post "batch ${h} h rc=$rc ${#CASES[@]} cases; $(tail -1 "$OUT/consumer_${h}h.log" | cut -c1-200)"
  (( rc == 0 )) || break
  done_b=$(( done_b + 1 ))
done
exit "$rc"
