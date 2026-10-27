#!/usr/bin/env bash
# w44-rc v0.4.4 release GPU entry: CUDA MINI of entry_rw_short.sh (same builder/env/phases incl. RAM slot, lock, prime, seal, consumer replay, E41 + checker; mini grid = Swiss 43x43x45, 1 h; own mini bundle)
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label w44-rc (lane lease required). Reads the frozen
# PIN manifest tools_v044/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/cuda_mini_short.sh
ENTRY_NAME=mini_short
ROLE=mini_short
SESSION=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/session_short.sh
TABLE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/arms_short_mini.txt
CAP_MIN=18
START_FORECAST_GIB=2
POST_RESERVE_S=600
POST=e41
TOOLS_SUMS_SHA=03507a848b9e4e2edef68b3abe6f3c9942febd6754bc267d17fd35e5918c18c2
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_common.sh
