#!/usr/bin/env bash
# w43-teacher GPU session: v0.4.2 sealed bundle, release defaults, 3-nest 72 h teacher-qualification arms (WN3 + ERA5 inputs),
# up to 3 cases in parallel on ONE GPU through the bundle consumer (P never written; compiled_new reported).
# usage: scripts/with_gpu_lock.sh --label w43-teacher --timeout <lock-wait s> -- bash gpu_arms.sh CASES_FILE
#   CASES_FILE: one staged case input dir per line (namelist.input + wrfinput_d01..03 + wrfbdy_d01 + wrflowinp_d01..03,
#   all outside the ALISIOS roots). Env: GA_START_UTC (refuse before), GA_END_UTC (hard end; refuse to start if
#   < GA_MIN_S left), GA_OUT (output root), GA_MAILBOX.
# Posts 'w43-teacher: GPU START ...' / 'w43-teacher: GPU END rc=.. <min> GPU-min ...'. Records the hibernation journal around it.
set -uo pipefail
CASES_FILE=$(readlink -f "$1")
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2
OUT=${GA_OUT:-<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/gpu}
LAYER=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/layer
MAILBOX=${GA_MAILBOX:-<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md}
post() { printf '%s w43-teacher: [real %s] %s\n' "$(date -u +%FT%TZ)" "$(<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh --iso)" "$*" >> "$MAILBOX"; }
[[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 ]] || { echo "run inside scripts/with_gpu_lock.sh"; exit 2; }
if [ -e /tmp/wrf_gpu2_quiet ]; then echo "QUIET"; exit 3; fi
# CLOCK SKEW (manager 2026-10-20 22:18 system time): windows are REAL UTC via real_utc.sh; durations on the system clock x0.9
REAL=<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh
rnow() { local e; e=$("$REAL"); [[ $e =~ ^[0-9]+$ ]] || { echo "real_utc.sh non-numeric"; exit 7; }; echo "$e"; }
T0=$(date +%s); R0=$(rnow) || exit 7
START=$(date -u -d "${GA_START_UTC:?}" +%s); END=$(date -u -d "${GA_END_UTC:?}" +%s)
(( R0 >= START )) || { echo "too early (window starts ${GA_START_UTC} real UTC)"; exit 4; }
(( END - R0 >= ${GA_MIN_S:-2700} )) || { post "[real $("$REAL" --iso)] GPU NOT STARTED: only $(( (END - R0) / 60 )) min left before ${GA_END_UTC}"; exit 6; }
mapfile -t CASES < <(grep -v '^\s*$' "$CASES_FILE")
(( ${#CASES[@]} >= 1 )) || exit 2
for c in "${CASES[@]}"; do
  case "$(readlink -f "$c")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/*) echo "ALISIOS-root case $c"; exit 2;; esac
  for f in namelist.input wrfinput_d01 wrfinput_d02 wrfinput_d03 wrfbdy_d01 wrflowinp_d01 wrflowinp_d02 wrflowinp_d03; do
    [ -s "$c/$f" ] || { echo "missing $c/$f"; exit 2; }; done
done
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1)
NEED_G=$(( 120 + 23 * ${#CASES[@]} ))
(( FREE_G >= NEED_G )) || { post "GPU NOT STARTED: <DATA_ROOT> free ${FREE_G} GiB < ${NEED_G}"; exit 4; }
[ -e "$OUT" ] && { echo "refusing existing $OUT (fresh output root per attempt)"; exit 2; }
mkdir -p "$OUT" "$LAYER"
journalctl -k --since "-6h" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_before.txt"
(cd "$P" && sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_before.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_before.txt")
timeout 20 nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$OUT/nvidia_before.csv" 2>&1
timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> "$OUT/nvidia_before.csv" 2>&1
printf '%s\n' "${CASES[@]}" > "$OUT/cases.txt"
REM=$(( (END - $(rnow) - 60) * 9 / 10 ))   # system-clock timeout, x0.9 against slew
echo "system $(date -u +%FT%TZ) real $("$REAL" --iso) T0=$T0 R0=$R0 END=$END REM=$REM" > "$OUT/clocks.txt"
rc=0
finish() { journalctl -k --since "@$T0" 2>/dev/null | grep -i hibernat > "$OUT/hibernate_during.txt";
  post "GPU END rc=$rc $(( ($(date +%s) - T0 + 59) / 60 )) GPU-min (lock released next); out $OUT; $(tail -1 "$OUT/consumer.log" 2>/dev/null | cut -c1-200)"; }
trap finish EXIT
post "GPU START v0.4.2 bundle teacher arms: ${#CASES[@]} cases x 72 h (max 3 parallel), cap $(( REM / 60 )) GPU-min, ends by $(date -u -d @$END +%TZ) real"
# shellcheck disable=SC1091
source "$P/bundle.env"
export JAX_PLATFORMS=cuda TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/w43-teacher/tmp
mkdir -p "$TMPDIR"
timeout --signal=TERM --kill-after=60 "$REM" bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$LAYER" -- \
  bash "$P/src/scripts/run_parallel_cases.sh" --out-root "$OUT/runs" --max-parallel 3 --host-gb-per-case 8 "${CASES[@]}" -- \
  --domains-from-namelist --emit-initial-history --hours 72 > "$OUT/consumer.log" 2>&1
rc=$?
(cd "$P" && sha256sum -c --quiet SHA256SUMS > "$OUT/bundle_check_after.txt" 2>&1; echo "rc=$?" >> "$OUT/bundle_check_after.txt")
for d in "$OUT"/runs/*/; do [ -d "$d" ] && (cd "$d" && find . -name 'wrfout_d0*' | sort | xargs -r sha256sum > wrfout_sha.txt); done
exit "$rc"
