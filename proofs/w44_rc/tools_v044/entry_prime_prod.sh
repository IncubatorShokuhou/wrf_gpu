#!/usr/bin/env bash
# Explicit cold setup / warm replay, manager ruling 2026-10-26T01:19:21Z.
# Only manager night_seq_v9.sh may schedule the GPU session. CPU SEQ_SMOKE uses the same entry.
ENTRY_PATH=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_prime_prod.sh
ENTRY_NAME=prime_prod
ROLE=prime_prod
SESSION=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/session_short.sh
TABLE=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/arms_prime_prod.txt
CAP_MIN=30
START_FORECAST_GIB=5
POST_RESERVE_S=30
POST=none
TOOLS_SUMS_SHA=03507a848b9e4e2edef68b3abe6f3c9942febd6754bc267d17fd35e5918c18c2
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/entry_common.sh
