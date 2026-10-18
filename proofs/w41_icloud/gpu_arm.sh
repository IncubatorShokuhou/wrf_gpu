#!/usr/bin/env bash
# WN3 0227 3-nest icloud=3 24 h GPU arm for the manager's GPU window (lease w41-icloud required).
# TREE = immutable detached snapshot of lane/w41-icloud (data/wrf_pristine symlinked), CASE = arm root
# (input/ from proofs/w41_icloud, cpu/ from cpu_reference.sh). Writes CASE/gpu, CASE/gpu_proof, the
# E41 scan and removes the private caches afterwards. Scoring runs on CPU afterwards:
#   JAX_PLATFORMS=cpu python proofs/v040/final_validation/score.py TREE CASE --domains d01 d02 d03
set -uo pipefail
TREE=$(realpath "${TREE:?immutable snapshot}")
CASE=$(realpath "${CASE:-<DATA_ROOT>/wrf_gpu2/scratch/w41-icloud/wn3_icloud3}")
L=<USER_HOME>/wrf_gpu2_lanes/w41-icloud/gpu_arm
HERE=$(cd "$(dirname "$0")" && pwd)
# Wave rule (manager 17:15Z, alisios_mgr C1745): ALISIOS model inputs are opened only inside alisios_guard; refuse any
# input whose realpath is under an ALISIOS root (stage a guarded copy outside the roots first, alisios_guard.facade_stage).
for f in "$CASE"/input/*; do
  case "$(realpath "$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/src/alisios/*|<DATA_ROOT>/server/work/alisios/*)
    echo "refusing ALISIOS-root input $f -> $(realpath "$f"): stage it via alisios_guard first"; exit 2;; esac
done
[ -e "$CASE/gpu" ] && { echo "refusing existing $CASE/gpu"; exit 2; }
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || { echo "dirty tree"; exit 2; }
mkdir -p "$L"
git -C "$TREE" rev-parse HEAD > "$L/source_commit.txt"
mkdir -p "$L/cache/jax" "$L/cache/autotune" "$L/cache/triton" "$L/cache/cuda" "$L/tmp"
# Same env as the v0.4.0 WN3 composite arms (proofs/v040/final_validation/wn3_window.sh), census OFF.
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_RRTMG_MP_RE=1 GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 \
       GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_JAX_CACHE=1 \
       GPUWRF_JAX_CACHE_DIR="$L/cache/jax" JAX_COMPILATION_CACHE_DIR="$L/cache/jax" \
       GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$L/cache/autotune" TRITON_CACHE_DIR="$L/cache/triton" \
       CUDA_CACHE_PATH="$L/cache/cuda" TMPDIR="$L/tmp" \
       XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30 \
       GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=4 PYTHONPATH="$TREE/src"
cd "$TREE"
T0=$(date +%s)
# One lock session: the 24 h forecast, then the cal_cldfra3 per-call cost probe (< 1 min).
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
# Manager rule 11:37Z: no RAM slot around/inside the GPU lock; the GPU process gets its own memcg cap
# (WN3 3-nest ~13 GB host RSS). CPU post-processing (E41 scan) afterwards in a normal RAM slot.
scripts/with_gpu_lock.sh --label w41-icloud -- systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 \
  bash -c '
    s=$(date +%s); echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud: GPU START WN3 0227 3-nest icloud=3 24 h cold arm + cal_cldfra3 cost probe (one lock session, <= 30 min)" >> "$4"
    timeout --signal=TERM --kill-after=15 1700 python -m gpuwrf run \
        --input-dir "$1/input" --output-dir "$1/gpu" --proof-dir "$1/gpu_proof" \
        --scratch-dir "$1/gpu_scratch" --hours 24 --domains-from-namelist --emit-initial-history \
        > "$2/gpu_run.log" 2>&1
    rc=$?
    timeout 240 python "$3/gpu_cost.py" "$1" "$2/gpu_cost.json" > "$2/gpu_cost.log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud: GPU END rc=$rc, $(( ($(date +%s) - s + 59) / 60 )) GPU-min (lock released next; E41 scan on CPU follows)" >> "$4"
    exit $rc' w41-icloud-arm "$CASE" "$L" "$HERE" "$MBX"
RC=$?
echo "{\"rc\": $RC, \"wall_s\": $(( $(date +%s) - T0 ))}" > "$L/gpu_run.json"
# E41 on the actual GPU executables BEFORE cache cleanup (E166/E177/E201).
scripts/with_ram_slot.sh --label w41-icloud-e41 --max 6G --timeout 21600 -- env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python scripts/wave_cubin_gate.py "$L/cache/jax" "$L/e41.json" \
    --run-proof "$CASE/gpu_proof/nested_pipeline_run.json" > "$L/e41.log" 2>&1
E41=$?
echo "e41 rc=$E41" >> "$L/gpu_run.json.log"
# Keep the executables if the scan did not produce its receipt (E166/E177: the cache is its only input).
if [ -s "$L/e41.json" ]; then rm -rf "$L/cache" "$L/tmp" "$CASE/gpu_scratch"; else echo "e41 receipt missing: cache kept" >> "$L/gpu_run.json.log"; fi
cat "$L/gpu_run.json"
