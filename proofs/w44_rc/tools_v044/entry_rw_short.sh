#!/usr/bin/env bash
# w44-rc v0.4.4 release GPU entry: 1/2 SHORT = bundle prime (Swiss 3 h, PROD 3 h, WN3 0227 3 h, WN3 0227 icloud=0 3 h; autotune seeded from v0.4.3) + seal + RW1/RW2/RW4 replays (0227, 0502, 0227-48h-input, Swiss, PROD, 0502 icloud=0, 0614) + E41/checker
# Manager sequencer ONLY (night_seq_v9.sh: `bash <this file>`, own process group, GA_START_UTC/GA_END_UTC exported, group
# killed at cap+1 min). No arguments. GA-window-only: refuses unless the whole cap fits before GA_END_UTC (real UTC); no
# hour/date guards. GPU only through scripts/with_gpu_lock.sh --label w44-rc (lane lease required). Reads the frozen
# PIN manifest tools_v044/PIN_MANIFEST.json written by prepare.sh. Body: entry_common.sh (in SHA256SUMS, pinned below).
ENTRY_PATH=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_rw_short.sh
ENTRY_NAME=rw_short
ROLE=short
SESSION=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/session_short.sh
TABLE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/arms_short_prod.txt
CAP_MIN=95
START_FORECAST_GIB=12
POST_RESERVE_S=600
POST=e41
TOOLS_SUMS_SHA=03507a848b9e4e2edef68b3abe6f3c9942febd6754bc267d17fd35e5918c18c2
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_common.sh
