#!/usr/bin/env bash
for pair in 'c22 gpu' 'c21 spread' 'c22 spread'; do
 if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
 read -r case_id score_kind <<< "$pair"
 rg -i "TP33.*${case_id}|TP35.*${case_id}|zero.*original.*spread" <USER_HOME>/src/wrf_gpu2/.agent/kernel/FINDINGS.md <USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv <USER_HOME>/src/wrf_gpu2/.agent/kernel/INBOX.md | tail -n 15 > "<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP38_${case_id}_${score_kind}.prior.log"
 timeout --kill-after=5 500 python <USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP33_score.py --case "$case_id" --kind "$score_kind" > "<USER_HOME>/wrf_gpu2_lanes/t-p3fix/TP38_${case_id}_${score_kind}.log" 2>&1 || exit $?
done
