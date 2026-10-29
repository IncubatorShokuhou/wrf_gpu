#!/usr/bin/env bash
# v45-rc v0.4.5 release GPU entry: 1/2 SHORT = bundle prime (Swiss 3 h, PROD 3 h, WN3 0227 3 h, WN3 0227 icloud=0 3 h; fresh empty autotune caches) + seal + RW1/RW2/RW4 replays (0227, 0502, 0227-48h-input, Swiss, PROD, 0502 icloud=0, 0614) + E41/checker
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label v45-rc (lane lease required). Reads the frozen
# PIN manifest tools_v6/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/entry_rw_short.sh
ENTRY_NAME=rw_short
ROLE=short
SESSION=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/session_short.sh
TABLE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/arms_short_prod.txt
CAP_MIN=95
START_FORECAST_GIB=12
POST_RESERVE_S=600
POST=e41
TOOLS_SUMS_SHA=72cd6f8352d9000946b1b465787e815786afa822e7c0e1172b249b6e6742149f
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/entry_common.sh
