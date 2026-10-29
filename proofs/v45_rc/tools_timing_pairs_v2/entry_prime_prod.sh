#!/usr/bin/env bash
# Explicit cold setup / warm replay, manager ruling 2026-10-26T01:19:21Z.
# Only manager night_seq_v9.sh may schedule the GPU session. CPU SEQ_SMOKE uses the same entry.
ENTRY_PATH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/entry_prime_prod.sh
ENTRY_NAME=prime_prod
ROLE=prime_prod
SESSION=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/session_short.sh
TABLE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/arms_prime_prod.txt
CAP_MIN=30
START_FORECAST_GIB=5
POST_RESERVE_S=30
POST=none
TOOLS_SUMS_SHA=72cd6f8352d9000946b1b465787e815786afa822e7c0e1172b249b6e6742149f
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_pairs_v2/entry_common.sh
