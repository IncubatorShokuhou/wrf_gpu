#!/usr/bin/env bash
# After arms: M4 product mutant + aot-key test file + fused-cascade consumer test (CPU, core 24, nice 19).
set -uo pipefail
L=<USER_HOME>/wrf_gpu2_lanes/w42-aotkey; W=<USER_HOME>/src/wrf_gpu2_wt/w42-aotkey; S=/tmp/claude-1000/-home-user-src-wrf-gpu2-wt-w42-aotkey/35a7c144-5f58-4e1d-81c5-4ee9336409b8/scratchpad
OUT=$L/post; mkdir -p $OUT; cd $W
PT=(env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 taskset -c 24 nice -n 19 timeout 1800 python -m pytest -q -p no:cacheprovider --basetemp $L/pytest)
q() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
cp src/gpuwrf/runtime/aot_key_cli.py $OUT/aot_key_cli.py.orig
q; python3 $S/mutants.py M4_no_libm $OUT/aot_key_cli.py.orig && "${PT[@]}" tests/test_aot_key_cli.py > $OUT/M4.log 2>&1; echo "M4 rc=$?" >> $OUT/post.log
cp $OUT/aot_key_cli.py.orig src/gpuwrf/runtime/aot_key_cli.py; cmp -s $OUT/aot_key_cli.py.orig src/gpuwrf/runtime/aot_key_cli.py && echo restored >> $OUT/post.log
git status --porcelain -- src >> $OUT/post.log
q; "${PT[@]}" tests/test_aot_key_cli.py tests/test_fused_cascade_case_blind.py > $OUT/tests.log 2>&1; echo "tests rc=$?" >> $OUT/post.log
grep -E "passed|failed|^FAILED|^ERROR" $OUT/M4.log $OUT/tests.log >> $OUT/post.log
echo POST_DONE >> $OUT/post.log
