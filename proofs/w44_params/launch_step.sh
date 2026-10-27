#!/usr/bin/env bash
# Controller stays outside all GPU and CPU/RAM scopes. No date-of-day guard.
# V3 (2026-10-24): E225 step-alignment guard before anything; first-entry gate lives in step_body.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CASE=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
HOURS=3  # cold_prepare.py / root_gpu.py build NestedPipelineConfig(hours=3); the CLI dry-run does not check this (LV01)
family=${1:?family}; stage=${2:?stage}; shift 2
[[ $family == diffusion || $family == mynn ]]
[[ $stage == prepare || $stage == warm ]]
grep -q "hours=$HOURS, max_dom=2" "$HERE/cold_prepare.py" && grep -q "hours=$HOURS, max_dom=2" "$HERE/root_gpu.py" \
  || { echo "launcher hours drifted from HOURS=$HOURS"; exit 6; }
JAX_PLATFORMS=cpu python "$HERE/check_alignment.py" "$CASE/namelist.input" "$HOURS" || { echo 'E225 alignment refused'; exit 6; }
cap=960; [[ $stage != warm ]] || cap=1800
smoke=0
for arg in "$@"; do [[ $arg != --cpu-smoke ]] || smoke=1; done
if [[ $smoke == 1 ]]; then
  smoke_cap=600; [[ $stage != warm ]] || smoke_cap=1500  # warm rehearsal also runs checker + E41 donor scan
  exec env JAX_PLATFORMS=cpu python "$HERE/supervise.py" --whole-arm --seconds "$smoke_cap" -- \
    bash "$HERE/step_body.sh" "$family" "$stage" "$@"
fi
now=$(JAX_PLATFORMS=cpu <USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh)
start=$(date -u -d "${GA_START_UTC:?manager window required}" +%s)
end=$(date -u -d "${GA_END_UTC:?manager window required}" +%s)
(( now >= start && end-now >= 30 )) || { echo 'outside authorized window'; exit 5; }
remaining=$((end-now)); (( remaining >= cap )) || cap=$remaining
exec env JAX_PLATFORMS=cpu python "$HERE/supervise.py" --whole-arm --seconds "$cap" --deadline-epoch "$end" -- \
  bash "$HERE/step_body.sh" "$family" "$stage" "$@"
