#!/usr/bin/env bash
# CPU consumers: run after the GPU group exits/releases its lock.
set -uo pipefail
repo=$(JAX_PLATFORMS=cpu git rev-parse --show-toplevel)
export JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$repo/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
result=0
for id in "$@"; do
  [[ $id =~ ^c[0-9][0-9]$ ]] || exit 2
  case_dir=<USER_HOME>/wrf_gpu2_lanes/v-prep/$id
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 0 nice -n 19 timeout 900 python scripts/wave_cubin_gate.py \
    "$case_dir/cache/jax" "$case_dir/e41" --run-proof "$case_dir/gpu_proof/nested_pipeline_run.json" \
    > "$case_dir/e41.log" 2>&1 || result=1
  domains=$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["domains"]))' "$case_dir/inputs.json")
  # Domain names come from our bounded prepared manifest, never arbitrary shell text.
  read -r -a domain_names <<< "$domains"
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 1 nice -n 19 timeout 1800 python proofs/v040/final_validation/score.py \
    "$repo" "$case_dir" --domains "${domain_names[@]}" > "$case_dir/score.log" 2>&1 || result=1
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; taskset -c 0 nice -n 19 timeout 180 python proofs/v040/final_validation/pin_cleanup.py "$id" \
    > "$case_dir/cleanup.log" 2>&1 || result=1
done
exit "$result"
