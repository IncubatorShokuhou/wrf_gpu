#!/usr/bin/env bash
# One official GPU session,20G child cap, no RAM-slot nesting.
set -euo pipefail
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/final-candidate
repo=<USER_HOME>/src/wrf_gpu2_wt/w41-wdm57
proof=$1
arm=$2
case "$arm" in c10-replay) case=c10; seconds=240; work=c10-replay;; c10-replay2) case=c10; seconds=300; work=c10-replay;; c18) case=c18; seconds=480; work=c18;; *) exit 9;; esac
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
git -C "$repo" show "$proof:proofs/w41_wdm57/gpu_confirm.py" > "$lane/gpu_confirm_$arm.py"
chmod 444 "$lane/gpu_confirm_$arm.py"
mkdir -p "$lane/gpu-$work-work"/{tmp,jax,triton,cuda}
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0
exec "$repo/scripts/with_gpu_lock.sh" --label w41-wdm57 --timeout 14400 -- bash -c '
  lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
  tree=$lane/final-candidate
  arm=$1; case=$2; seconds=$3; work=$4
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  [[ $(( $(date +%s) + seconds + 5 )) -le $(date -u -d 2026-10-17T18:45:00Z +%s) ]] || exit 9
  extra=()
  [[ $case != c10 ]] || extra=("$lane/replay-c10/d01.xlaexec")
  started=$(date +%s)
  echo "$(date -u +%FT%TZ) w41-wdm57: GPU START $arm cap${seconds}s; actual200steps/rad33; properproductionbufferownership" >> <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
  rc=0
  timeout --kill-after=5 "$seconds" systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 \
    env -u CUDA_VISIBLE_DEVICES JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=0 GPUWRF_CARRY_DONATE=1 \
    GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 \
    GPUWRF_RRTMG_MP_RE=1 GPUWRF_XLA_AUTOTUNE_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 TMPDIR="$lane/gpu-$work-work/tmp" \
    JAX_COMPILATION_CACHE_DIR="$lane/gpu-$work-work/jax" TRITON_CACHE_DIR="$lane/gpu-$work-work/triton" \
    CUDA_CACHE_PATH="$lane/gpu-$work-work/cuda" PYTHONPATH="$tree/src" \
    python "$lane/gpu_confirm_$arm.py" "$tree" "<USER_HOME>/wrf_gpu2_lanes/v-prep/$case" \
      "$lane/gpu-$arm" "${extra[@]}" || rc=$?
  elapsed=$(( $(date +%s) - started ))
  echo "$(date -u +%FT%TZ) w41-wdm57: GPU END $arm rc=$rc seconds=$elapsed minutes=$(( (elapsed+59)/60 ))" >> <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
  echo "$arm rc=$rc seconds=$elapsed"
  exit "$rc"
' -- "$arm" "$case" "$seconds" "$work"
