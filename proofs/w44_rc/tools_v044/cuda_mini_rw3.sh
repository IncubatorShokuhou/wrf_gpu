#!/usr/bin/env bash
# w44-rc v0.4.4 release GPU entry: CUDA MINI of entry_rw3.sh (same builder/env/phases incl. RAM slot, lock, pre-gate, core watch, N + O consumers, RW3T summary; mini grid = Swiss 43x43x45, 1 h, slots N72 O1; N = the CUDA-mini bundle, run cuda_mini_short.sh first)
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label w44-rc (lane lease required). Reads the frozen
# PIN manifest tools_v044/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/cuda_mini_rw3.sh
ENTRY_NAME=mini_rw3
ROLE=mini_rw3
SESSION=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/session_rw3.sh
TABLE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/slots_rw3_mini.txt
CAP_MIN=12
START_FORECAST_GIB=1
POST_RESERVE_S=60
POST=rw3summary
TOOLS_SUMS_SHA=03507a848b9e4e2edef68b3abe6f3c9942febd6754bc267d17fd35e5918c18c2
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_common.sh
