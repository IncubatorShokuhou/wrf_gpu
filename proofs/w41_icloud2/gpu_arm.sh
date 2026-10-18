#!/usr/bin/env bash
# WN3 0227 3-nest icloud=2 24 h GPU arm for the manager's GPU window (lease w41-icloud2 required).
# Same recipe as the merged icloud=3 twin (proofs/w41_icloud/gpu_arm.sh = v0.4.0 WN3 composite env, census OFF,
# rich AOT modules so the E41 scan sees the kernels); only the namelist icloud value differs (stage_inputs.py).
# TREE = immutable detached snapshot of lane/w41-icloud2 (data/wrf_pristine symlinked), CASE = arm root
# (input/ from stage_inputs.py). Writes CASE/gpu, CASE/gpu_proof, the E41 scan on the actual executables BEFORE the
# private caches are removed (E166/E177/E201). D6 scoring afterwards on CPU: post.sh.
# Budget: icloud=3 twin 28 GPU-min (forecast wall 1623 s incl. ~530 s cold compile); hard cap 2100 s + 240 s probe.
# usage: TREE=<snapshot> bash gpu_arm.sh [CASE]
set -uo pipefail
TREE=$(realpath "${TREE:?immutable snapshot}")
CASE=$(realpath "${1:-<DATA_ROOT>/wrf_gpu2/scratch/w41-icloud2/wn3_icloud2}")
L=<USER_HOME>/wrf_gpu2_lanes/w41-icloud2/gpu_arm
HERE=$(cd "$(dirname "$0")" && pwd)
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
# GPU queue (manager 22:32Z): the next lane starts on our GPU END line -> a refusal before the lock must post one too.
refuse() { echo "$1"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: GPU END rc=$2, 0 GPU-min (refused before the lock: $1; next lane may go)" >> "$MBX"; exit "$2"; }
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['verdict']=='OK' else 1)" \
  <USER_HOME>/wrf_gpu2_lanes/w41-icloud2/stage/STAGE.json || refuse "STAGE.json missing or not OK" 2
[ "$(/usr/bin/grep -cE '^ *icloud *= *2,' "$CASE/input/namelist.input")" = 1 ] || refuse "namelist is not icloud=2" 2
# Wave rule (alisios_mgr C1745): never open an ALISIOS-root input outside alisios_guard.
for f in "$CASE"/input/*; do
  case "$(realpath "$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/src/alisios/*|<DATA_ROOT>/server/work/alisios/*)
    refuse "ALISIOS-root input $f" 2;; esac
done
[ -e "$CASE/gpu" ] && refuse "existing $CASE/gpu" 2
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || refuse "dirty tree" 2
[ -e "$TREE/data/wrf_pristine" ] || refuse "snapshot lacks data/wrf_pristine" 2
[ -e /tmp/wrf_gpu2_quiet ] && refuse QUIET 3
mkdir -p "$L/cache/jax" "$L/cache/autotune" "$L/cache/triton" "$L/cache/cuda" "$L/tmp"
git -C "$TREE" rev-parse HEAD > "$L/source_commit.txt"
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_RRTMG_MP_RE=1 GPUWRF_CENSUS=0 GPUWRF_CENSUS_SLIM=0 \
       GPUWRF_AOT_SLIM_MODULE=0 GPUWRF_JAX_CACHE=1 \
       GPUWRF_JAX_CACHE_DIR="$L/cache/jax" JAX_COMPILATION_CACHE_DIR="$L/cache/jax" \
       GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$L/cache/autotune" TRITON_CACHE_DIR="$L/cache/triton" \
       CUDA_CACHE_PATH="$L/cache/cuda" TMPDIR="$L/tmp" \
       XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.30 \
       GPUWRF_MIN_FREE_VRAM_GIB=1.0 OMP_NUM_THREADS=4 PYTHONPATH="$TREE/src"
cd "$TREE"
T0=$(date +%s)
rm -f "$L/inner.started"
# No RAM slot around/inside the GPU lock (manager 11:37Z); the GPU process gets its own 20G memcg cap.
scripts/with_gpu_lock.sh --label w41-icloud2 -- systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 \
  bash -c '
    : > "$2/inner.started"; s=$(date +%s); echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: GPU START WN3 0227 3-nest icloud=2 24 h cold arm + cal_cldfra2 cost probe (one lock session, ~28 min, cap 39)" >> "$4"
    timeout --signal=TERM --kill-after=15 2100 python -m gpuwrf run \
        --input-dir "$1/input" --output-dir "$1/gpu" --proof-dir "$1/gpu_proof" \
        --scratch-dir "$1/gpu_scratch" --hours 24 --domains-from-namelist --emit-initial-history \
        > "$2/gpu_run.log" 2>&1
    rc=$?
    timeout 240 python "$3/gpu_cost2.py" "$1" "$2/gpu_cost.json" > "$2/gpu_cost.log" 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: GPU END rc=$rc, $(( ($(date +%s) - s + 59) / 60 )) GPU-min (lock released next; E41 scan on CPU follows)" >> "$4"
    exit $rc' w41-icloud2-arm "$CASE" "$L" "$HERE" "$MBX"
RC=$?
# The inner command never ran (lock timeout 124 / approval 126): still release the queue.
[ -e "$L/inner.started" ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w41-icloud2: GPU END rc=$RC, 0 GPU-min (with_gpu_lock refused before the run; next lane may go)" >> "$MBX"
echo "{\"rc\": $RC, \"wall_s\": $(( $(date +%s) - T0 ))}" > "$L/gpu_run.json"
# E41 on the actual GPU executables BEFORE cache cleanup (E166/E177/E201); main's RAM-slot wrapper.
<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-icloud2-e41 --max 6G --timeout 21600 -- env JAX_PLATFORMS=cpu \
    GPUWRF_JAX_CACHE=0 PYTHONPATH="$TREE/src" taskset -c 10,26,27 nice -n 19 timeout 3600 python scripts/wave_cubin_gate.py \
    "$L/cache/jax" "$L/e41.json" --run-proof "$CASE/gpu_proof/nested_pipeline_run.json" > "$L/e41.log" 2>&1
E41=$?
V=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['e41_gate'])" "$L/e41.json/wave_gate.json" 2>/dev/null || echo MISSING)
echo "{\"e41_rc\": $E41, \"e41\": \"$V\"}" >> "$L/gpu_run.json.log"
# Keep the executables unless the scan wrote its verdict (E166/E177: the cache is its only input).
if [ "$V" != MISSING ]; then rm -rf -- "$L/cache" "$L/tmp" "$CASE/gpu_scratch"; else echo "e41 verdict missing: cache kept" >> "$L/gpu_run.json.log"; fi
cat "$L/gpu_run.json" "$L/gpu_run.json.log"
