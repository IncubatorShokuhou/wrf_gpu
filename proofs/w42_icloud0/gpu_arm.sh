#!/usr/bin/env bash
# WN3 0227 3-nest icloud=0 24 h GPU arm (w42-icloud0) for the manager's 2026-10-19 13:00-19:00Z window (lease required).
# Same recipe as the merged icloud=3 / icloud=2 twins (proofs/w41_icloud2/gpu_arm.sh: release defaults, census OFF, rich
# AOT modules so the E41 scan sees the kernels, own 20G memcg); only the namelist icloud value differs (stage_inputs.py).
# TREE = immutable detached snapshot of lane/w42-icloud0 (data/wrf_pristine symlinked), CASE = arm root.
# Records that TREE carries the v0.4.1 SWDDIR fix (rad_sw_band_sums unscaled beam 'dn * bn' x2; ALISIOS asks whether
# icloud=0 is qualified TOGETHER with it). Writes CASE/gpu, CASE/gpu_proof and the E41 scan on the actual executables
# BEFORE the private caches are removed (E166/E177/E201). D6 scoring afterwards on CPU: post.sh.
# Budget: icloud=2 twin 24-26 GPU-min (forecast wall 1092-1210 s incl. cold compile); hard cap 2100 s.
# Dry test (no GPU, no lock): DRY=1 TREE=<snapshot> bash gpu_arm.sh
# usage: TREE=<snapshot> bash gpu_arm.sh [CASE]
set -uo pipefail
TREE=$(realpath "${TREE:?immutable snapshot}")
CASE=$(realpath "${1:-<DATA_ROOT>/wrf_gpu2/scratch/w42-icloud0/wn3_icloud0}")
L=<USER_HOME>/wrf_gpu2_lanes/w42-icloud0/gpu_arm
MBX=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
[ "${DRY:-0}" = 1 ] && MBX=/dev/null
# GPU queue: the next lane starts on our GPU END line -> a refusal before the lock must post one too.
refuse() { echo "$1"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=$2, 0 GPU-min (refused before the lock: $1; next lane may go)" >> "$MBX"; exit "$2"; }
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['verdict']=='OK' else 1)" "$CASE/STAGE.json" \
  || refuse "STAGE.json missing or not OK" 2
[ "$(/usr/bin/grep -cE '^ *icloud *= *0,' "$CASE/input/namelist.input")" = 1 ] || refuse "namelist is not icloud=0" 2
# Wave rule (alisios_mgr C1745): never open an ALISIOS-root input outside alisios_guard.
for f in "$CASE"/input/*; do
  case "$(realpath "$f")" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/src/alisios/*|<DATA_ROOT>/server/work/alisios/*)
    refuse "ALISIOS-root input $f" 2;; esac
done
[ -e "$CASE/gpu" ] && refuse "existing $CASE/gpu" 2
[ -z "$(git -C "$TREE" status --porcelain --untracked-files=no)" ] || refuse "dirty tree" 2
[ -e "$TREE/data/wrf_pristine" ] || refuse "snapshot lacks data/wrf_pristine" 2
[ "$(/usr/bin/grep -c 'dn \* bn' "$TREE/src/gpuwrf/kernels/rad_sw_band_sums.py")" = 2 ] || refuse "tree lacks the SWDDIR fix" 2
/usr/bin/grep -q "if icloud not in (0, 2, 3)" "$TREE/src/gpuwrf/io/namelist_binding.py" || refuse "tree does not bind icloud=0" 2
if [ -e /tmp/wrf_gpu2_quiet ]; then refuse QUIET 3; fi
if [ "${DRY:-0}" = 1 ]; then
  [ -x "$TREE/scripts/with_gpu_lock.sh" ] && [ -f "$TREE/scripts/wave_cubin_gate.py" ] || { echo "DRY: tools missing"; exit 2; }
  echo "DRY OK: TREE $(git -C "$TREE" rev-parse HEAD) CASE $CASE (SWDDIR fix present, icloud=0 bound, inputs guarded)"; exit 0
fi
mkdir -p "$L/cache/jax" "$L/cache/autotune" "$L/cache/triton" "$L/cache/cuda" "$L/tmp"
git -C "$TREE" rev-parse HEAD > "$L/source_commit.txt"
echo "swddir_fix_markers=2 (rad_sw_band_sums 'dn * bn', w41-swddir v0.4.1 default)" > "$L/swddir_fix.txt"
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
# No RAM slot around/inside the GPU lock; the GPU process gets its own 20G memcg cap.
scripts/with_gpu_lock.sh --label w42-icloud0 -- systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 \
  bash -c '
    : > "$2/inner.started"; s=$(date +%s); echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU START WN3 0227 3-nest icloud=0 24 h cold arm (one lock session, ~26 min, cap 36)" >> "$3"
    timeout --signal=TERM --kill-after=15 2100 python -m gpuwrf run \
        --input-dir "$1/input" --output-dir "$1/gpu" --proof-dir "$1/gpu_proof" \
        --scratch-dir "$1/gpu_scratch" --hours 24 --domains-from-namelist --emit-initial-history \
        > "$2/gpu_run.log" 2>&1
    rc=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=$rc, $(( ($(date +%s) - s + 59) / 60 )) GPU-min (lock released next; E41 scan on CPU follows)" >> "$3"
    exit $rc' w42-icloud0-arm "$CASE" "$L" "$MBX"
RC=$?
# The inner command never ran (lock timeout 124 / approval 126): still release the queue.
[ -e "$L/inner.started" ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: GPU END rc=$RC, 0 GPU-min (with_gpu_lock refused before the run; next lane may go)" >> "$MBX"
echo "{\"rc\": $RC, \"wall_s\": $(( $(date +%s) - T0 ))}" > "$L/gpu_run.json"
# E41 on the actual GPU executables BEFORE cache cleanup (E166/E177/E201); main's RAM-slot wrapper, lane cores.
<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w42-icloud0-e41 --max 6G --timeout 21600 -- env JAX_PLATFORMS=cpu \
    GPUWRF_JAX_CACHE=0 PYTHONPATH="$TREE/src" taskset -c 8,9,12 nice -n 19 timeout 3600 python scripts/wave_cubin_gate.py \
    "$L/cache/jax" "$L/e41.json" --run-proof "$CASE/gpu_proof/nested_pipeline_run.json" > "$L/e41.log" 2>&1
E41=$?
V=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['e41_gate'])" "$L/e41.json/wave_gate.json" 2>/dev/null || echo MISSING)
echo "{\"e41_rc\": $E41, \"e41\": \"$V\"}" >> "$L/gpu_run.json.log"
# Keep the executables unless the scan wrote its verdict (E166/E177: the cache is its only input).
if [ "$V" != MISSING ]; then rm -rf -- "$L/cache" "$L/tmp" "$CASE/gpu_scratch"; else echo "e41 verdict missing: cache kept" >> "$L/gpu_run.json.log"; fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) w42-icloud0: E41 $V (rc $E41); D6 scoring next on CPU" >> "$MBX"
cat "$L/gpu_run.json" "$L/gpu_run.json.log"
