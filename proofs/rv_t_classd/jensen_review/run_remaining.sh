#!/usr/bin/env bash
set -u
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True GPUWRF_FAST_DEFAULTS=0 GPUWRF_JAX_CACHE=0
export JAX_ENABLE_COMPILATION_CACHE=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/snapshot/src:<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/snapshot/tests
cd <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/snapshot || exit 2
precheck() {
 rg -i 'jensen|mp55|8b844|radar.*cadence' <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md > "<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/$1.precheck.log"
}
precheck RV34
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 1200 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/oracle_rebuild.py > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV34_oracle.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV34.rc
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; rm -rf <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/oracle_build
[ "$rc" -eq 0 ] || exit "$rc"
precheck RV35
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 600 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/cpu_cost.py > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV35_CPU_COST.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV35.rc
[ "$rc" -eq 0 ] || exit "$rc"
precheck RV36
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 240 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/mutant.py > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV36_mutant.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV36.rc
[ "$rc" -eq 1 ] || exit 2
precheck RV37
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; GPUWRF_FAST_DEFAULTS=1 timeout 300 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/release_identity_private.py <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/snapshot <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_swiss.json > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_swiss.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_swiss.rc
[ "$rc" -eq 0 ] || exit "$rc"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; GPUWRF_FAST_DEFAULTS=1 timeout 300 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/release_identity_private.py <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/snapshot <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_prod.json <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_prod.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/jensen_review/RV37_prod.rc
exit "$rc"
