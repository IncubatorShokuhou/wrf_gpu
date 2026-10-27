#!/usr/bin/env bash
# o5-perf P1b H-CB A/B (manager 16:36Z): OFF vs V1 (GPUWRF_STEP_CMDBUF=v1: xla_gpu_graph_min_graph_size=1 per-jit on the
# stepping programs) vs V2 (=v2: capture set without WHILE per-jit). OFF = SEALED v0.4.3 bundle (consumer, warm, 0 compiles);
# V1/V2 = o5-perf prototype snapshot (O5_SNAP, sparse src/scripts/data of the frozen commit) + private cache seeded with the
# bundle's 1,228 per-fusion tunes (E99: same autotune picks). Deviceless gate before this phase: OFF/V1/V2 identical
# optimized HLO + native records (PROD d02/d01). Here: PROD 3 h, S0 harness (HERO), all arms retained.
# Order (one lock): W0 OFF warm-up + conversion-pass VLOG (runtime cause confirmation, not scored); P1 V1 cold; P2 V2 cold;
# measured fresh processes OFF_1 V1_1 V2_1 V2_2 V1_2 OFF_2 OFF_3 V1_3 V2_3; graph-level nsys N_OFF N_V1 N_V2 (structure only).
# Gate (pre-registered, cbab plan in proofs/o5_perf/REPORT.md §6): wrfout SHA of every arm == W0 (sealed OFF).
set -uo pipefail
umask 022
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf
TAG=${O5_RUN_TAG:?run tag}
R=$S/cbab/$TAG
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
export O5_SNAP=${O5_SNAP:-$S/snap_cbfix}
SMOKE=${O5_LAUNCH_SMOKE:-0}
FLOOR=365; [[ "$SMOKE" == 1 ]] && FLOOR=200
mkdir -p "$R"
[[ "$(git -C "$P/src" rev-parse --short=10 HEAD)" == 888804310b ]] || { echo PIN_MISMATCH; exit 4; }
[[ "$(git -C "$O5_SNAP" rev-parse HEAD 2>/dev/null)" == "$(cat "$L/SNAP_COMMIT")" ]] || { echo SNAP_MISMATCH; exit 4; }
C=$S/cache_cbfix; mkdir -p "$C/jax" "$C/autotune" "$C/triton" "$C/cuda"
if [[ ! -d "$C/jax/xla_gpu_per_fusion_autotune_cache_dir" ]]; then
  cp -r "$P/jax/xla_gpu_per_fusion_autotune_cache_dir" "$C/jax/"   # 5 MB, bundle tunes (read-only copy)
fi
LAYER=$S/layer_v043_cb_OFF; [[ "$SMOKE" == 1 ]] && LAYER=${LAYER}_smoke
arm() {  # src variant name budget kind [vlog]
  local src=$1 v=$2 name=$3 budget=$4 kind=$5 vl=${6:-} free
  [[ "$SMOKE" == 1 ]] && budget=60
  free=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' '); (( free >= FLOOR )) || { echo "DISK_FLOOR ${free}"; exit 95; }
  if [[ "$src" == bundle ]]; then
    bash "$P/tools/bundle_consumer.sh" --no-verify --bundle "$P" --layer "$LAYER" -- \
      bash "$L/o5_s0_arm.sh" bundle "$v" "$name" "$R/$name" "$budget" "$kind" "$vl" > "$R/$name.consumer.log" 2>&1
  else
    bash "$L/o5_s0_arm.sh" snap "$v" "$name" "$R/$name" "$budget" "$kind" "$vl" > "$R/$name.consumer.log" 2>&1
  fi
  echo "arm $name $src $v $kind rc=$? $(tail -1 "$R/$name.consumer.log")"
}
if [[ "$SMOKE" == 1 ]]; then arm bundle OFF S_OFF 60 warm vlog; arm snap V1 S_V1 60 warm; arm snap V2 S_V2 60 nsys; echo "cbab smoke done $TAG"; exit 0; fi
arm bundle OFF W0 300 warm vlog
arm snap V1 P1 1500 warm; arm snap V2 P2 1500 warm
for spec in "bundle OFF OFF_1" "snap V1 V1_1" "snap V2 V2_1" "snap V2 V2_2" "snap V1 V1_2" "bundle OFF OFF_2" "bundle OFF OFF_3" "snap V1 V1_3" "snap V2 V2_3"; do
  read -r s v a <<< "$spec"; arm "$s" "$v" "$a" 300 warm; done
arm bundle OFF N_OFF 420 nsys; arm snap V1 N_V1 420 nsys; arm snap V2 N_V2 420 nsys
python3 -I "$L/cbab_summary.py" "$R"
echo "cbab session done $TAG"
