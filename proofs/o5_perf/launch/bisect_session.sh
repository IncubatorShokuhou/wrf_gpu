#!/usr/bin/env bash
# o5-perf P2 version bisection v0.4.1 vs v0.4.3 (the split w44-perf's v0.3.2/v0.4.0/v0.4.3 repeat does not cover), ONE lock
# session, product CLI / product parallel runner of each SEALED bundle in consumer mode (P never written), release defaults,
# each version's own allocator policy (disclosed per arm, env.txt + receipts). Interleaved, all arms retained.
# Pre-registered: proofs/o5_perf/BISECT_PLAN.json (A/A per version (max-min)/mean <= 5 % per metric; failure => INCONCLUSIVE).
# Workloads: PROD s0_case_20260725 3 h solo; WN3 0227/0502/0614 3-nest 3 h, 3 cases concurrent.
# Smoke (O5_LAUNCH_SMOKE=1): every arm command once on CPU, private smoke layers, 75 s per arm (must reach model init).
set -uo pipefail
umask 022
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf
TAG=${O5_RUN_TAG:?run tag}
R=$S/bisect/$TAG
mkdir -p "$R"
declare -A P=([v041]=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1 [v043]=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3)
declare -A PIN=([v041]=104c97de98 [v043]=888804310b)
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
C=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
WN3=("$C/wn3_0227" "$C/wn3_0502" "$C/wn3_0614")
SMOKE=${O5_LAUNCH_SMOKE:-0}
FLOOR=365; [[ "$SMOKE" == 1 ]] && FLOOR=200
for v in v041 v043; do
  [[ "$(git -C "${P[$v]}/src" rev-parse --short=10 HEAD)" == "${PIN[$v]}" ]] || { echo "PIN_MISMATCH $v"; exit 4; }
done
for d in "$PROD" "${WN3[@]}"; do [[ -s "$d/namelist.input" ]] || { echo "MISSING_INPUT $d"; exit 4; }; done
sha256sum "$PROD"/namelist.input "${WN3[@]/%//namelist.input}" > "$R/input_namelist_sha.txt"
layer() { if [[ "$SMOKE" == 1 ]]; then echo "$S/layer_${1}_smoke"; else echo "$S/layer_${1}"; fi; }
disk_ok() { local free; free=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' '); (( free >= FLOOR )) || { echo "DISK_FLOOR ${free} < ${FLOOR}"; return 1; }; }
prod() {  # version arm
  local v=$1 arm=$2 budget=240; [[ "$SMOKE" == 1 ]] && budget=75
  disk_ok || exit 95
  bash "${P[$v]}/tools/bundle_consumer.sh" --no-verify --bundle "${P[$v]}" --layer "$(layer "$v")" -- \
    bash "$L/o5_arm.sh" "${P[$v]}" "o5_$arm" "$PROD" "$R/$arm" "$budget" "$L/nolog.env" -- --domains-from-namelist --hours 3 \
    > "$R/$arm.consumer.log" 2>&1
  echo "arm $arm $v rc=$? $(tail -1 "$R/$arm.consumer.log")"
  [[ -d "$R/$arm/wrfout" ]] && find "$R/$arm/wrfout" -maxdepth 1 -type f -name 'wrfout_d0*' -delete
}
par3() {  # version arm
  local v=$1 arm=$2 budget=420; [[ "$SMOKE" == 1 ]] && budget=75
  disk_ok || exit 95
  bash "${P[$v]}/tools/bundle_consumer.sh" --no-verify --bundle "${P[$v]}" --layer "$(layer "$v")" -- \
    bash "$L/o5_par3_arm.sh" "${P[$v]}" "o5_$arm" "$R/$arm" "$budget" 3 "${WN3[@]}" > "$R/$arm.consumer.log" 2>&1
  echo "arm $arm $v rc=$? $(tail -1 "$R/$arm.consumer.log")"
}
if [[ "$SMOKE" == 1 ]]; then
  prod v041 S_P41; prod v043 S_P43; par3 v041 S_W41; par3 v043 S_W43
  echo "bisect smoke done $TAG"; exit 0
fi
# Warm-ups (retained, not scored), then interleaved measured arms (BISECT_PLAN.json order).
prod v041 P41_W; prod v043 P43_W
for spec in "v043 P43_1" "v041 P41_1" "v041 P41_2" "v043 P43_2" "v043 P43_3" "v041 P41_3"; do read -r v a <<< "$spec"; prod "$v" "$a"; done
par3 v041 W41_W; par3 v043 W43_W
for spec in "v043 W43_1" "v041 W41_1" "v041 W41_2" "v043 W43_2"; do read -r v a <<< "$spec"; par3 "$v" "$a"; done
echo "bisect session done $TAG"
