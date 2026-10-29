#!/usr/bin/env bash
# v45-rc v0.4.5 release GPU entry: 2/2 RW3 (long slot) = WN3 0227 3-nest 72 h warm N72 (v0.4.5, D6 input) + interleaved 24 h N1 O1 N2 O2 N3 O3 + O72 (v0.4.4) timing/VRAM/RSS
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label v45-rc (lane lease required). Reads the frozen
# PIN manifest tools_v6/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/entry_rw3.sh
ENTRY_NAME=rw3_missing3
ROLE=rw3
SESSION=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/session_rw3.sh
TABLE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/slots_missing3.txt
CAP_MIN=45
START_FORECAST_GIB=8
POST_RESERVE_S=120
POST=rw3summary
TOOLS_SUMS_SHA=833d3a1438070417d0d02db78664d692554de2af45eac1d858a00f18b250644c
if [[ ${1:-outer} == outer ]]; then
  if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 ]]; then
    JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 -B <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/resource_admission.py --smoke || exit 95
  else
    JAX_PLATFORMS=cpu <USER_HOME>/miniconda3/bin/python3 -B <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/resource_admission.py || exit 95
  fi
fi
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/entry_common.sh
