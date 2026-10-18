#!/usr/bin/env bash
# No RAM slot surrounds a GPU lock. Each GPU child has its own20G scope;
# independent CPU postprocessing queues only AFTER the corresponding lock exits.
set -u
lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
tree=$lane/final-candidate
repo=<USER_HOME>/src/wrf_gpu2_wt/w41-wdm57
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1
proof_commit=$1
post_pids=()
overall=0
[[ $(git -C "$tree" rev-parse HEAD) == 4474617f2585ef9d577fc360a7688e2de1efcd15 ]] || exit 9
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
# Freeze the receipt-only helper from the same immutable proof commit as this
# coordinator; the measured model stays at snapshot4474617f25.
git -C "$repo" show "$proof_commit:proofs/w41_wdm57/gpu_confirm.py" > "$lane/gpu_confirm_sealed.py" || exit 9
chmod 444 "$lane/gpu_confirm_sealed.py"
for case in c10 c18; do
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  mkdir -p "$lane/gpu-$case-work"/{tmp,jax,triton,cuda}
  "$repo/scripts/with_gpu_lock.sh" --label w41-wdm57 --timeout 7200 -- bash -c '
    lane=<USER_HOME>/wrf_gpu2_lanes/w41-wdm57
    tree=$lane/final-candidate
    case=$1
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
    [[ $(( $(date +%s) + 440 )) -le $(date -u -d 2026-10-17T18:45:00Z +%s) ]] || exit 9
    started=$(date +%s)
    echo "$(date -u +%FT%TZ) w41-wdm57: GPU START $case actual 200-step segment; <=440s, total lease15min" >> <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
    timeout 440 systemd-run --user --scope --quiet -p MemoryMax=20G -p MemorySwapMax=0 \
      env -u CUDA_VISIBLE_DEVICES JAX_PLATFORMS=cuda GPUWRF_JAX_CACHE=0 \
      GPUWRF_FAST_DEFAULTS=1 GPUWRF_CENSUS=1 GPUWRF_CENSUS_SLIM=0 GPUWRF_AOT_SLIM_MODULE=0 \
      GPUWRF_RRTMG_MP_RE=1 GPUWRF_XLA_AUTOTUNE_CACHE=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
      TMPDIR="$lane/gpu-$case-work/tmp" JAX_COMPILATION_CACHE_DIR="$lane/gpu-$case-work/jax" \
      TRITON_CACHE_DIR="$lane/gpu-$case-work/triton" CUDA_CACHE_PATH="$lane/gpu-$case-work/cuda" \
      PYTHONPATH="$tree/src" python "$lane/gpu_confirm_sealed.py" \
      "$tree" "<USER_HOME>/wrf_gpu2_lanes/v-prep/$case" "$lane/gpu-$case"
    rc=$?
    elapsed=$(( $(date +%s) - started ))
    echo "$(date -u +%FT%TZ) w41-wdm57: GPU END $case rc=$rc seconds=$elapsed (minutes=$(( (elapsed+59)/60 )))" >> <USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
    echo "$case GPU rc=$rc seconds=$elapsed"
    exit "$rc"
  ' -- "$case" > "$lane/gpu-$case.log" 2>&1
  gpu_rc=$?
  [[ $gpu_rc == 0 ]] || overall=1
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
  # Disposable compiler caches are no longer needed; retain the serialized
  # executable until its queued CPU scanner consumes it.
  case "$case" in
    c10) taskset -c 8,9,12,13,24,25,28,29 nice -n 19 rm -rf <USER_HOME>/wrf_gpu2_lanes/w41-wdm57/gpu-c10-work;;
    c18) taskset -c 8,9,12,13,24,25,28,29 nice -n 19 rm -rf <USER_HOME>/wrf_gpu2_lanes/w41-wdm57/gpu-c18-work;;
  esac
  "$repo/scripts/with_ram_slot.sh" --label "w41-wdm57-e41-$case" --max 6G --timeout 14400 -- \
    timeout 300 bash -c 'set -o pipefail; git -C "$1" show "$2:proofs/w41_wdm57/gpu_post.sh" | bash -s -- "$3"' \
    -- "$repo" "$proof_commit" "$case" > "$lane/gpu-$case-post.log" 2>&1 &
  post_pids+=("$!")
  [[ $gpu_rc == 0 ]] || break
done
for pid in "${post_pids[@]}"; do wait "$pid" || overall=1; done
exit "$overall"
