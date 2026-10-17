#!/usr/bin/env bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
git -C <USER_HOME>/src/wrf_gpu2_wt/t-p3fix worktree remove <USER_HOME>/wrf_gpu2_lanes/t-p3fix/tp16-tree || exit $?
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
git -C <USER_HOME>/src/wrf_gpu2_wt/t-p3fix worktree remove <USER_HOME>/wrf_gpu2_lanes/t-p3fix/tp02-tree || exit $?
rg -i 'TP43|P348|native.*FMA' <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md | tail -n 15 > <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP43.prior.log
timeout --kill-after=5 340 python <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP43_fma.py > <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP43.log 2>&1 || exit $?
for case_id in c23 c24; do
 if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
 rg -i "TP35.*${case_id}|zero.*original.*spread" <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md | tail -n 15 > "<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP35_${case_id}.prior.log"
 timeout --kill-after=5 500 python <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP33_score.py --case "$case_id" --kind spread > "<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP35_${case_id}.log" 2>&1 || exit $?
done
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
rg -i 'TP24|TP40|MP52.*coalesce' <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md | tail -n 15 > <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24.prior.log
timeout --kill-after=5 160 bash <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24_run.sh > <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP24.log 2>&1
exit $?
