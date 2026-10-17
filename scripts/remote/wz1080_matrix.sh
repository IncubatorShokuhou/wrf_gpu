#!/usr/bin/env bash
# wz1080_matrix.sh — capability-matrix steps on the wz1080 box (GTX 1080 Ti, sm_61); results land locally in
# $WZ_LOCAL_LANE/matrix/<step>.{out,err}. Run steps one by one and read them; nothing here is a release claim.
# Usage: scripts/remote/wz1080_matrix.sh [--tree SHA|main] <step>...
#   gpu-env gpu-bench gpu-tridiag gpu-pallas gpu-cmdbuf      micro probes (probes/gpu_caps.py)
#   swiss-release  Swiss d01 ${WZ_HOURS:-3} h gpuwrf CLI on the GPU, release defaults (1st call cold, repeat = warm)
#   swiss-portable same + WZ_PALLAS_PORTABLE=1 (box-only shim: Pallas interpret on sm_61)
#   swiss-legacy   same with GPUWRF_FAST_DEFAULTS=0 (XLA legacy paths)
#   swiss-legacy-portable  legacy XLA path + shim for the few Pallas kernels the legacy path still uses (largest runnable path)
#   cpuwrf         same case with the reference's own CPU-WRF binary on the box CPU (4 ranks like the reference; WZ_NP)
#   swiss-score    D6 of the newest output of every kind vs the original CPU-WRF reference (CPU)
#   report         fetch timing/D6/VRAM/log files from the box and print the GPU-vs-CPU table (wz1080_report.py)
#   cpu-quick      CPU pytest: contracts + fast-defaults + cli (foreground)
#   cpu-full       CPU pytest: whole tests/ in the background (junit xml in the run dir)
#   gpu-tests      GPU pytest (legacy defaults) of the State-building suites that skip on CPU, background
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
# shellcheck source=wz1080_common.sh
source "$HERE/wz1080_common.sh"
RUN="$HERE/wz1080_run.sh"
TREE=main
if [[ "${1:-}" == --tree ]]; then TREE="$2"; shift 2; fi
OUT="$WZ_LOCAL_LANE/matrix"; mkdir -p "$OUT"
L=wz1080
PROBE='python "$HOME/src/wrf_gpu2_env/tools/probes/gpu_caps.py"'
CMDBUF_FLAGS="--xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE --xla_enable_command_buffers_during_profiling=true"
HOURS="${WZ_HOURS:-3}"
TOOL='exec bash "$HOME/src/wrf_gpu2_env/tools/probes/$0" "$@"'   # expanded on the box: bash -c "$TOOL" <script> args
SWISS='o="$WZ_RUN_DIR/$0_$(date -u +%Y%m%dT%H%M%SZ)"; echo "OUT=$o"; $([ -x /usr/bin/time ] && echo /usr/bin/time -v) python -m gpuwrf.cli run \
  --input-dir examples/switzerland_d01 --output-dir "$o" --domain d01 --hours 1 --scratch-dir "$o.scratch"; rc=$?; \
  ls -la "$o" | head -20; exit $rc'
PYT_BASE='--basetemp "$WZ_RUN_DIR/pytest_$0" -p no:cacheprovider -q -rfEs --continue-on-collection-errors'

go() {  # go <step-name> <run.sh args...>
  local name="$1"; shift
  echo "######## $name ($(date -u +%H:%M:%SZ))"
  set +e
  "$RUN" --tree "$TREE" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
  local rc=$?
  set -e
  echo "rc=$rc" >> "$OUT/$name.out"
  tail -n 25 "$OUT/$name.out"; echo "--- stderr tail"; tail -n 15 "$OUT/$name.err"
}

for step in "$@"; do
  case "$step" in
    gpu-env|gpu-bench|gpu-tridiag|gpu-pallas)
      go "$step" --gpu --label $L --timeout 900 --env JAX_PALLAS_USE_MOSAIC_GPU=false -- bash -c "$PROBE --section ${step#gpu-}";;
    gpu-cmdbuf)
      go gpu-cmdbuf-off --gpu --label $L --timeout 600 -- bash -c "$PROBE --section cmdbuf"
      go gpu-cmdbuf-on --gpu --label $L --timeout 600 --env "XLA_FLAGS=$CMDBUF_FLAGS" -- bash -c "$PROBE --section cmdbuf";;
    swiss-release)
      go swiss-release-$(date -u +%H%M) --gpu --label $L --timeout 7200 -- bash -c "$TOOL" swiss_gpu.sh swiss_release "$HOURS";;
    swiss-portable)
      go swiss-portable-$(date -u +%H%M) --gpu --label $L --timeout 10800 --env WZ_PALLAS_PORTABLE=1 -- bash -c "$TOOL" swiss_gpu.sh swiss_portable "$HOURS";;
    swiss-legacy)
      go swiss-legacy-$(date -u +%H%M) --gpu --label $L --timeout 7200 --env GPUWRF_FAST_DEFAULTS=0 -- bash -c "$TOOL" swiss_gpu.sh swiss_legacy "$HOURS";;
    swiss-legacy-portable)
      go swiss-legacy-portable-$(date -u +%H%M) --gpu --label $L --timeout 7200 --env GPUWRF_FAST_DEFAULTS=0 --env WZ_PALLAS_PORTABLE=1 \
        -- bash -c "$TOOL" swiss_gpu.sh swiss_legacyportable "$HOURS";;
    cpuwrf)
      go cpuwrf-np${WZ_NP:-4}-$(date -u +%H%M) --cpu --no-nice --label $L --timeout 7200 -- bash -c "$TOOL" swiss_cpuwrf.sh "$HOURS" "${WZ_NP:-4}";;
    swiss-score)
      for kind in swiss_release swiss_portable swiss_legacy swiss_legacyportable cpuwrf_np4 cpuwrf_np6; do
        go "score-$kind" --cpu --label $L --timeout 1800 -- bash -c 'd=$(ls -1dt "$WZ_RUN_DIR"/$0_*Z/ 2>/dev/null | head -1); \
          [ -n "$d" ] || { echo "no $0 output"; exit 0; }; ls "$d"wrfout_d01_* >/dev/null 2>&1 || { echo "no wrfout in $d"; exit 3; }; \
          python scripts/compare_wrfout_grid.py --cpu-dir "$HOME/src/wrf_gpu2_data/refs/swiss_rd11_cpu" --gpu-dir "$d" \
          --domain d01 --init 2023-01-15T00:00:00 --max-lead "$1" --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
          --out-json "${d%/}.d6.json" --out-md "${d%/}.d6.md" > "${d%/}.d6.log" 2>&1; rc=$?; sed -n "1,60p" "${d%/}.d6.md"; exit $rc' "$kind" "$HOURS"
      done;;
    report)
      mkdir -p "$OUT/box"
      wz_rsync -a --include='*.timing.json' --include='*.d6.json' --include='*.d6.md' --include='*.vram.log' \
        --include='*.log' --include='*.time.txt' --exclude='*/' --exclude='*' "$WZ_HOST:$WZ_R_RUNS/$L/" "$OUT/box/"
      python3 "$HERE/wz1080_report.py" "$OUT/box" --out "$OUT/REPORT.md";;
    cpu-quick)
      go cpu-quick --cpu --label $L --timeout 3600 -- bash -c "python -m pytest tests/contracts tests/test_fast_defaults.py tests/test_cli.py $PYT_BASE" cpuquick;;
    cpu-full)
      go cpu-full --cpu --bg --label wz1080-cpufull --timeout 43200 -- bash -c \
        "python -m pytest tests $PYT_BASE --junitxml=\"\$WZ_RUN_DIR/cpu_full_junit.xml\"" cpufull;;
    gpu-tests)
      go gpu-tests --gpu --bg --label wz1080-gputests --timeout 21600 --env GPUWRF_FAST_DEFAULTS=0 -- bash -c \
        "python -m pytest tests/idealized tests/dynamics tests/regression tests/savepoint $PYT_BASE --junitxml=\"\$WZ_RUN_DIR/gpu_tests_junit.xml\"" gputests;;
    *) echo "unknown step $step" >&2; exit 2;;
  esac
done
