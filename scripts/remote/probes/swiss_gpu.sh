#!/usr/bin/env bash
# swiss_gpu.sh <tag> [hours]  — gpuwrf CLI on the bundled Swiss d01 case (run inside wz1080_run.sh --gpu, tree = cwd).
# Path selection via env from the caller: release defaults (none) | WZ_PALLAS_PORTABLE=1 | GPUWRF_FAST_DEFAULTS=0.
set -uo pipefail
tag="$1"; hours="${2:-3}"
o="$WZ_RUN_DIR/${tag}_$(date -u +%Y%m%dT%H%M%SZ)"
echo "OUT=$o"; echo "env: FAST_DEFAULTS=${GPUWRF_FAST_DEFAULTS:-<release>} PORTABLE=${WZ_PALLAS_PORTABLE:-0} cache=$GPUWRF_JAX_CACHE_DIR"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$o.vram.log" 2>/dev/null & smi=$!
t0=$(date +%s.%N)
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $o.time.txt"
$TIMEV python -m gpuwrf.cli run --input-dir examples/switzerland_d01 --output-dir "$o" --domain d01 --hours "$hours" \
  --scratch-dir "$o.scratch" > "$o.log" 2>&1
rc=$?
t1=$(date +%s.%N); kill "$smi" 2>/dev/null
echo "rc=$rc process_wall_s=$(python -c "print(round($t1-$t0,3))") vram_peak_MiB=$(sort -n "$o.vram.log" | tail -1)"
[ -f "$o.time.txt" ] && grep -E "Maximum resident|Elapsed" "$o.time.txt"
python "$HOME/src/wrf_gpu2_env/tools/probes/wrfout_timing.py" "$o" "$t0" --out "$o.timing.json" | grep TIMING_SUMMARY
if [ "$rc" -ne 0 ]; then echo "---- log tail"; grep -v "^\s*$" "$o.log" | tail -40; fi
exit "$rc"
