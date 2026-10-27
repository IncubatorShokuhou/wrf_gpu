#!/usr/bin/env bash
# o5-perf P1 (H-CB): why does the PROD d02 stepping program run outside CUDA graphs on v0.4.3? XLA VLOG of the
# command-buffer conversion pass at executable load, on the SEALED v0.4.3 bundle in consumer mode (P never written).
# Arms (sequential, warm sealed programs, release defaults, logging env only):
#   A  PROD s0_case_20260725  --domains-from-namelist --hours 3                               (bundle arm B6 recipe)
#   B  WN3 0502               --domains-from-namelist --emit-initial-history --hours 3        (bundle arm B2 recipe)
# Measured caps: B6 55.5 s, B2 98.1 s wall in the sealed bundle receipts (+ device assert + consumer check).
# Smoke (O5_LAUNCH_SMOKE=1): same commands on CPU, private smoke layer, 75 s per arm: the real CLI must reach model init
# ('gpuwrf: init mode =') with XLA INFO logging active; rc 124 (smoke timeout) is expected. The CLI does NOT refuse on CPU.
set -uo pipefail
umask 022
P=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3
L=<USER_HOME>/wrf_gpu2_lanes/o5-perf/launch
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-perf
TAG=${O5_RUN_TAG:?run tag}
R=$S/cbprobe/$TAG
if [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]]; then LAYER=$S/layer_v043_smoke; SMOKE_BUDGET=75; else LAYER=$S/layer_v043; SMOKE_BUDGET=; fi
mkdir -p "$R"
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
WN3=<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0502
# Disk: real run writes <= 2 GB (wrfouts deleted after hashing, logs kept); smoke writes < 50 MB.
FLOOR=365; [[ "${O5_LAUNCH_SMOKE:-0}" == 1 ]] && FLOOR=200
free=$(df --output=avail -B1G <DATA_ROOT> | tail -1 | tr -d ' ')
(( free >= FLOOR )) || { echo "DISK_FLOOR <DATA_ROOT> ${free} GiB < ${FLOOR}"; exit 95; }
run_arm() {  # name input budget extra -- args
  local name=$1 in=$2 budget=$3 extra=$4; shift 5
  [[ -d "$in" ]] || { echo "arm $name SKIPPED_MISSING_INPUT $in"; echo "SKIPPED_MISSING_INPUT" > "$R/$name.skipped"; return 0; }
  [[ -n "$SMOKE_BUDGET" ]] && budget=$SMOKE_BUDGET
  bash "$P/tools/bundle_consumer.sh" --bundle "$P" --layer "$LAYER" -- \
    bash "$L/o5_arm.sh" "$P" "o5_$name" "$in" "$R/$name" "$budget" "$extra" -- "$@" > "$R/$name.consumer.log" 2>&1
  local rc=$?
  echo "arm $name consumer_rc=$rc $(tail -1 "$R/$name.consumer.log")"
  # wrfouts are not needed by this probe: keep sha/mtimes receipts, delete the files (exact dir, fixed name pattern).
  [[ -d "$R/$name/wrfout" ]] && find "$R/$name/wrfout" -maxdepth 1 -type f -name 'wrfout_d0*' -delete
  return 0
}
run_arm A_prod_vlog "$PROD" 240 "$L/vlog.env" -- --domains-from-namelist --hours 3
run_arm B_wn3_vlog "$WN3" 300 "$L/vlog.env" -- --domains-from-namelist --emit-initial-history --hours 3
# CPU summary (stdlib only): conversion-pass decisions per log; no IR/env dumped.
JAX_PLATFORMS=cpu python3 - "$R" <<'PY'
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1]); out = {}
pats = {'created_cmdbuf_thunk': r'Creating command buffer thunk', 'min_graph_size_skip': r'is not lowered to command buffer because',
        'warmup_iteration': r'Executing warm up iteration of command buffer thunk', 'profiling_fallback': r'as a regular thunk sequence',
        'evict': r'Evict command buffer', 'record_create': r'Record create', 'record_update': r'Record update',
        'cuda_graph_created': r'Create CUDA command buffer', 'unsupported': r'[Uu]nsupported'}
for arm in sorted(p for p in R.iterdir() if p.is_dir()):
    log = arm / 'run.log'
    if not log.exists(): continue
    text = log.read_text(errors='replace')
    rec = {k: len(re.findall(v, text)) for k, v in pats.items()}
    rec['rc'] = (arm / 'rc.txt').read_text().strip() if (arm / 'rc.txt').exists() else None
    rec['log_bytes'] = log.stat().st_size
    rec['cli_init_started'] = 'gpuwrf: init mode =' in text
    rec['tf_cpp_info_lines'] = len(re.findall(r'^I\d{4} ', text, re.M))
    rec['cuda_backend'] = bool(re.search(r'Addressable PjRt-IFRT device: CudaDevice', text))
    rec['conversion_lines_head'] = [l[:300] for l in text.splitlines() if 'command_buffer_conversion_pass' in l][:40]
    out[arm.name] = rec
(R / 'CBPROBE_SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: {kk: v[kk] for kk in v if kk != 'conversion_lines_head'} for k, v in out.items()}))
PY
echo "cbprobe session done $TAG"
