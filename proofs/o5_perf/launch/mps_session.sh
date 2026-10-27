#!/usr/bin/env bash
# o5-perf P3: does CUDA MPS raise WN3 par3 throughput (release benchmark workload)? Same sealed v0.4.3 programs, same product
# parallel runner, same cases; ONLY difference = the 3 case processes connect to a private MPS daemon (spatial sharing instead
# of time-sliced contexts). Numerics must be unchanged: wrfout SHA of every MPS arm == every non-MPS arm (determinism check).
# Order (one lock): N0 warm-up (no MPS, not scored), M1, N1, M2, N2 — WN3 0227/0502/0614, 6 h, --max-parallel 3.
# MPS daemon: private CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY under o5-perf scratch (other users/processes unaffected),
# started right before and stopped ('quit') right after each MPS arm, also on any exit (trap).
# Smoke (O5_LAUNCH_SMOKE=1): commands on CPU, 75 s, NO MPS daemon start (GPU-adjacent), private smoke layer.
set -uo pipefail
umask 022
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf
TAG=${O5_RUN_TAG:?run tag}
R=$S/mps/$TAG
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
C=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases
WN3=("$C/wn3_0227" "$C/wn3_0502" "$C/wn3_0614")
SMOKE=${O5_LAUNCH_SMOKE:-0}
LAYER=$S/layer_v043_mps; [[ "$SMOKE" == 1 ]] && LAYER=$S/layer_v043_mps_smoke
FLOOR=365; [[ "$SMOKE" == 1 ]] && FLOOR=200
HOURS=6; BUDGET=600; [[ "$SMOKE" == 1 ]] && BUDGET=75
mkdir -p "$R"
[[ "$(git -C "$P/src" rev-parse --short=10 HEAD)" == 888804310b ]] || { echo PIN_MISMATCH; exit 4; }
for d in "${WN3[@]}"; do [[ -s "$d/namelist.input" ]] || { echo "MISSING_INPUT $d"; exit 4; }; done
MPS_PIPE=$S/mps_pipe MPS_LOG=$S/mps_log
mps_stop() {
  if [[ -S "$MPS_PIPE/control" || -e "$MPS_PIPE/control" ]]; then
    echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG nvidia-cuda-mps-control >> "$R/mps_control.log" 2>&1
    for _ in $(seq 1 30); do pgrep -u "$(id -u)" -f "nvidia-cuda-mps-(control|server)" >/dev/null || break; sleep 1; done
  fi
  pgrep -u "$(id -u)" -a -f "nvidia-cuda-mps-(control|server)" >> "$R/mps_control.log" 2>&1 && echo "MPS_STILL_RUNNING" >> "$R/mps_control.log"
  return 0
}
trap mps_stop EXIT
mps_start() {
  pgrep -u "$(id -u)" -f "nvidia-cuda-mps-(control|server)" >/dev/null && { echo "MPS_ALREADY_RUNNING (foreign?)"; return 1; }
  mkdir -p "$MPS_PIPE" "$MPS_LOG"
  CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG nvidia-cuda-mps-control -d >> "$R/mps_control.log" 2>&1
}
arm() {  # name mps(0|1)
  local name=$1 mps=$2 free
  free=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' '); (( free >= FLOOR )) || { echo "DISK_FLOOR ${free}"; exit 95; }
  local extra=()
  if [[ "$mps" == 1 && "$SMOKE" != 1 ]]; then mps_start || { echo "arm $name MPS_START_FAILED"; return 0; }
    extra=(env CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG); fi
  "${extra[@]}" bash "$P/tools/bundle_consumer.sh" --no-verify --bundle "$P" --layer "$LAYER" -- \
    bash "$L/o5_par3_arm.sh" "$P" "o5_$name" "$R/$name" "$BUDGET" "$HOURS" "${WN3[@]}" > "$R/$name.consumer.log" 2>&1
  echo "arm $name mps=$mps rc=$? $(tail -1 "$R/$name.consumer.log")"
  [[ "$mps" == 1 && "$SMOKE" != 1 ]] && mps_stop
  return 0
}
if [[ "$SMOKE" == 1 ]]; then arm S_N 0; arm S_M 1; echo "mps smoke done $TAG"; exit 0; fi
arm N0 0; arm M1 1; arm N1 0; arm M2 1; arm N2 0
# Determinism: every arm's wrfout SHA list must be identical (paths relative to its cases/ dir).
python3 -I - "$R" <<'PY'
import json, sys
from pathlib import Path
R = Path(sys.argv[1]); sha = {}
for a in ('N0', 'M1', 'N1', 'M2', 'N2'):
    f = R / a / 'wrfout_sha.txt'
    sha[a] = sorted(l.split()[0] + ' ' + l.split()[1] for l in f.read_text().splitlines()) if f.exists() else None
ref = sha['N0']
out = {'n_files': {a: len(v) if v else 0 for a, v in sha.items()}, 'identical_to_N0': {a: v == ref and v is not None for a, v in sha.items()}}
(R / 'MPS_DETERMINISM.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))
PY
echo "mps session done $TAG"
