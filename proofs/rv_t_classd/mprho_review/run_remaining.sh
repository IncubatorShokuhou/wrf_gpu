#!/usr/bin/env bash
set -u
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_COMPILATION_CACHE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GPUWRF_FAST_DEFAULTS=0
export PYTHONPATH=<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/snapshot/src:<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/snapshot/tests
cd <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/snapshot || exit 2
precheck() {
    rg -i 'mprho|MP.rho|moist_prep|dry density|DZ8W' <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md > "<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/$1.precheck.log"
}
precheck RV27b
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 180 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/compiled_probe.py > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV27b_compiled_probe.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV27b.rc
[ "$rc" -eq 0 ] || exit "$rc"
for mutation in moist dzclamp; do
    precheck "RV28_$mutation"
    if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; timeout 180 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/mutants.py "$mutation" > "<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV28_$mutation.log" 2>&1
    rc=$?; echo "$rc" > "<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV28_$mutation.rc"
    [ "$rc" -eq 1 ] || exit 2
done
precheck RV29
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; GPUWRF_FAST_DEFAULTS=1 timeout 300 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/release_identity_private.py <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/snapshot <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_swiss_identity.json > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_swiss_identity.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_swiss.rc
[ "$rc" -eq 0 ] || exit "$rc"
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; GPUWRF_FAST_DEFAULTS=1 timeout 300 python <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/release_identity_private.py <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/snapshot <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_prod_identity.json <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_prod_identity.log 2>&1
rc=$?; echo "$rc" > <USER_HOME>/wrf_gpu2_lanes/rv-t-classd/mprho_review/RV29_prod.rc
exit "$rc"
