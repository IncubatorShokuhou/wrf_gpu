#!/usr/bin/env bash
# v45-rc v0.4.5 release GPU entry: CUDA MINI of entry_rw_short.sh (same builder/env/phases incl. RAM slot, lock, prime, seal, consumer replay, E41 + checker; mini grid = Swiss 43x43x45, 1 h; own mini bundle)
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label v45-rc (lane lease required). Reads the frozen
# PIN manifest tools_v6/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/cuda_mini_short.sh
ENTRY_NAME=mini_short
ROLE=mini_short
SESSION=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/session_short.sh
TABLE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/arms_short_mini.txt
CAP_MIN=18
START_FORECAST_GIB=2
POST_RESERVE_S=600
POST=e41
TOOLS_SUMS_SHA=f455aa3817a2a6fc94a72a3d01627d2b2e69aad18aeb03fc1551f2893fc594a8
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/entry_common.sh
