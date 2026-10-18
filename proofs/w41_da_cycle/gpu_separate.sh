#!/usr/bin/env bash
# GPU arm, second lock session (lane w41-da-cycle): ONLY the 2 separate fresh `gpuwrf run` processes of
# gpu_arm.sh, for a session whose loop finished its member runs but hit the bound before them. Same session dir
# and cache (warm: the autotune cache path is part of the cheap key, E32/E143), same environment as the
# gpu_arm.sh lock section, model source = the measured snapshot TREE.
#   gpu_separate.sh TREE SESSION     (TREE: the clean detached snapshot gpu_arm.sh ran from)
# The GPU processes run under their own memory cap inside the lock (manager 11:37Z: no RAM slot near the lock).
set -uo pipefail
TREE=$(realpath -- "${1:?snapshot}"); SESSION=$(realpath -- "${2:?session}")
LEASE=/tmp/wrf_gpu2_run_approvals/w41-da-cycle.lease
if [[ ${3:-} != --locked ]]; then
  [[ -s $LEASE ]] || { echo "REFUSED: $LEASE missing" >&2; exit 126; }
  if git -C "$TREE" symbolic-ref -q HEAD >/dev/null; then echo 'REFUSED: model source must be a detached snapshot' >&2; exit 2; fi
  [[ -z $(git -C "$TREE" status --porcelain --untracked-files=no) ]] || { echo 'REFUSED: dirty snapshot' >&2; exit 2; }
  [[ -d $SESSION/loop && -d $SESSION/cache && ! -e $SESSION/separate ]] || { echo "REFUSED: $SESSION needs loop/ + cache/, no separate/" >&2; exit 2; }
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  rc=0
  "$TREE/scripts/with_gpu_lock.sh" --label w41-da-cycle --timeout "${W41_LOCK_WAIT_S:-3600}" -- \
    systemd-run --user --scope --quiet -p MemoryMax="${W41_MEMORY_MAX:-12G}" -p MemorySwapMax=0 -- \
    timeout --signal=TERM --kill-after=15s "${W41_GPU_BOUND_S:-600}s" bash "$0" "$TREE" "$SESSION" --locked || rc=$?
  echo "$rc" > "$SESSION/gpu_separate_rc.txt"
  echo "GPU_RC=$rc session=$SESSION"
  exit "$rc"
fi

[[ ${GPUWRF_GPU_LOCK_HELD:-0} == 1 ]] || { echo 'REFUSED: official GPU lock absent' >&2; exit 126; }
export JAX_PLATFORMS=cuda GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${W41_POOL_FRACTION:-0.25}
export GPUWRF_JAX_CACHE_DIR="$SESSION/cache" JAX_COMPILATION_CACHE_DIR="$SESSION/cache"
export GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$SESSION/cache/xla_gpu_per_fusion_autotune_cache_dir"
export TMPDIR="$SESSION/tmp" TRITON_CACHE_DIR="$SESSION/triton_cache" CUDA_CACHE_PATH="$SESSION/cuda_cache"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$TREE/src"
cd -- "$TREE"
for pair in "cont:cases0/m01:12" "c01:loop/c01/zero_file/prep/cases/m01:6"; do
  IFS=: read -r tag rel hours <<< "$pair"; case_dir=$SESSION/$rel
  [[ -d $case_dir ]] || { echo "missing $case_dir" >> "$SESSION/separate.rc"; continue; }
  s=$SECONDS
  /usr/bin/time -v python3 -m gpuwrf run --input-dir "$case_dir" --output-dir "$SESSION/separate/$tag" \
    --scratch-dir "$SESSION/tmp/sep_$tag" --domain d01 --max-dom 1 --hours "$hours" \
    > "$SESSION/separate_$tag.stdout" 2> "$SESSION/separate_$tag.stderr"
  echo "separate $tag rc=$? wall=$((SECONDS - s))s" | tee -a "$SESSION/separate.rc"
done
exit 0
