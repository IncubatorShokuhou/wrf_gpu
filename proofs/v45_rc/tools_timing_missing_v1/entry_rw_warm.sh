#!/usr/bin/env bash
# Explicit cold setup / warm replay, manager ruling 2026-10-26T01:19:21Z.
# Only manager night_seq_v9.sh may schedule the GPU session. CPU SEQ_SMOKE uses the same entry.
ENTRY_PATH=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/entry_rw_warm.sh
ENTRY_NAME=rw_warm
ROLE=warm
SESSION=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/session_short.sh
TABLE=<USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/arms_warm.txt
CAP_MIN=45
START_FORECAST_GIB=12
POST_RESERVE_S=600
POST=profile
TOOLS_SUMS_SHA=833d3a1438070417d0d02db78664d692554de2af45eac1d858a00f18b250644c
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_timing_missing_v1/entry_common.sh
