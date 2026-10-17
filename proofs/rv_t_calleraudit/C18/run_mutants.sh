#!/usr/bin/env bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu timeout 90 python <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/mutant.py species > <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-species-mutant.log 2>&1
rp_species_rc=$?
if [ "$rp_species_rc" -ne 0 ]; then tail -n 8 <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-species-mutant.log; exit "$rp_species_rc"; fi
tail -n 2 <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-species-mutant.log
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
JAX_PLATFORMS=cpu timeout 90 python <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/mutant.py traceback > <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-traceback-mutant.log 2>&1
rp_trace_rc=$?
if [ "$rp_trace_rc" -ne 0 ]; then tail -n 8 <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-traceback-mutant.log; exit "$rp_trace_rc"; fi
tail -n 2 <USER_HOME>/wrf_gpu2_lanes/rv-t-calleraudit/C18/RP04-traceback-mutant.log
