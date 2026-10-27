#!/usr/bin/env bash
# Runs INSIDE one bundle_consumer view of the sealed v0.4.3 bundle P (writes land in the private layer, P never written).
# 1) M_OFF: S0 harness PROD 3 h, sealed programs, conversion-pass VLOG. 2) copy the pre-built patched PROD d01/d02 AOT entries
# (only stored DebugOption 208 xla_gpu_graph_min_graph_size 5->1; native records 0/1/2 byte-identical, metas re-hashed; receipt
# cbmini_patched/PATCH_RECEIPT.json) over their paths IN THIS VIEW. 3) M_V1P: same arm on the patched executables.
set -uo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
R=${1:?run dir}; BUD=${2:-240}
PATCHED=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf/cbmini_patched
bash "$L/o5_s0_arm.sh" bundle OFF M_OFF "$R/M_OFF" "$BUD" warm vlog; echo "M_OFF rc=$?"
n=0
while IFS= read -r -d '' f; do
  rel=${f#"$PATCHED"/}; cp -f "$f" "$P/$rel" || { echo "COPY_FAILED $rel"; exit 7; }; n=$((n + 1))
done < <(find "$PATCHED/jax" -type f \( -name '*.xlaexec' -o -name '*.meta' \) -print0)
echo "patched entries placed in view: $n"
bash "$L/o5_s0_arm.sh" bundle OFF M_V1P "$R/M_V1P" "$BUD" warm vlog; echo "M_V1P rc=$?"
