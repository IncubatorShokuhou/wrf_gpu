#!/usr/bin/env bash
# o5-perf ONE S0-harness PROD 3 h arm (HERO harness = sealed v0.4.3 bundle scripts/bench/s0_nested_harness.py; per-call host_s
# and per-segment call_s receipts). usage: o5_s0_arm.sh SRC VARIANT NAME OUT_DIR BUDGET_S KIND [vlog]
#   SRC bundle: sealed v0.4.3 source + bundle caches (run INSIDE bundle_consumer, P never written); VARIANT must be OFF
#   SRC snap  : o5-perf lane snapshot (GPUWRF_STEP_CMDBUF prototype) + private cache seeded with the bundle's tunes
#   VARIANT OFF|V1|V2 -> GPUWRF_STEP_CMDBUF unset|v1|v2 (per-jit options on the stepping programs only)
#   KIND warm|nsys (nsys = graph-level trace of segment 2, structure only, E185)
set -uo pipefail
SRC=$1 VAR=$2 NAME=$3 OUT=$4 BUDGET=$5 KIND=$6 VLOG=${7:-}
case "$OUT" in <DATA_ROOT>/wrf_gpu2/scratch/o5-perf/*) ;; *) echo "refuse: OUT outside o5-perf scratch"; exit 2;; esac
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
SNAP=${O5_SNAP:-}
HARNESS=$P/src/scripts/bench/s0_nested_harness.py
if env | grep -E "^GPUWRF_" | grep -v -E "^GPUWRF_(JAX_CACHE_DIR|XLA_AUTOTUNE_CACHE_DIR|GPU_LOCK_[A-Z_]+|GPU_ARM_CPUS)=" > /dev/null; then echo "refuse: GPUWRF_* model flags"; exit 2; fi
unset XLA_FLAGS CUDA_ROOT GPUWRF_STEP_CMDBUF
case "$SRC" in
  bundle) [[ "$VAR" == OFF ]] || exit 2; source "$P/bundle.env"; export BENCH_SRC_REPO=$P/src ;;
  snap) [[ -n "$SNAP" && -d "$SNAP/src/gpuwrf" ]] || { echo "refuse: no snapshot"; exit 2; }
        C=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/cache_cbfix
        export BENCH_SRC_REPO=$SNAP PYTHONPATH=$SNAP/src GPUWRF_JAX_CACHE_DIR=$C/jax JAX_COMPILATION_CACHE_DIR=$C/jax \
               GPUWRF_XLA_AUTOTUNE_CACHE_DIR=$C/autotune TRITON_CACHE_DIR=$C/triton CUDA_CACHE_PATH=$C/cuda
        case "$VAR" in OFF) ;; V1) export GPUWRF_STEP_CMDBUF=v1 ;; V2) export GPUWRF_STEP_CMDBUF=v2 ;; *) exit 2 ;; esac ;;
  *) exit 2 ;;
esac
# S0 env (s0_env.sh of the harness) minus its XLA_FLAGS unset (E16) and minus GPUWRF_NEST_PERF_TIMERS (cheap key, E32)
export TMPDIR=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/tmp/$NAME XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async OMP_NUM_THREADS=4 LC_ALL=C
if [[ "$VLOG" == vlog ]]; then
  export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=command_buffer_conversion_pass=3,command_buffer_thunk=2
fi
if [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]]; then export JAX_PLATFORMS=cpu; else export JAX_PLATFORMS=cuda; fi
mkdir -p "$TMPDIR" "$OUT"; cd "$OUT" || exit 2
env | grep -E "^(GPUWRF|JAX|XLA|CUDA|TRITON|PYTHONPATH|BENCH|TF_CPP)" | grep -v TOKEN | sort > env.txt
[[ -n "${BENCH_SRC_REPO:-}" ]] && git -C "$BENCH_SRC_REPO" rev-parse HEAD > source_head.txt 2>/dev/null
H=(python3 "$HARNESS" --input-dir <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 --out "$OUT/run" --hours 3 --tag "o5_$NAME" --require-c-auto)
NSYS=()
if [[ "$KIND" == nsys ]]; then
  H+=(--profile-segment 2)
  NSYS=(nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=graph
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$OUT/profile")
  [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]] && NSYS=()
fi
date +%s.%N > t0_unix
timeout --signal=TERM --kill-after=30 "$BUDGET" "${NSYS[@]}" "${H[@]}" > run.log 2>&1
rc=$?; echo "$rc" > rc.txt; date +%s.%N > t_end_unix
(cd "$OUT/run/wrfout" 2>/dev/null && sha256sum wrfout_d0* > "$OUT/wrfout_sha.txt")
[[ -d "$OUT/run/wrfout" ]] && find "$OUT/run/wrfout" -maxdepth 1 -type f -name 'wrfout_d0*' -delete
echo "s0 arm $NAME $SRC $VAR $KIND rc=$rc"; exit "$rc"
