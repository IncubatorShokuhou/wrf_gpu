#!/usr/bin/env bash
# Summarize DL_<tag>_* deviceless receipts + control-flow counts. Usage: summ_c7b.sh TAG
for d in <USER_HOME>/wrf_gpu2_lanes/t-cfix-mynn/e41/DL_${1}_*/; do
  python3 -c "
import json,sys;d=json.load(open(sys.argv[1]+'/receipt.json'));print(sys.argv[2], d.get('phase'), (d.get('exception') or '')[:200], 'tf', d.get('transpose_fusions'), 'over500', d.get('any_transpose_over500'), 'stack', d.get('max_stack_B'), 'kernels', d.get('kernels'), 'gate', d.get('native_gate'), 'compile_s', round(d.get('compile_s',0),1), 'top', d.get('top_transpose', [])[:2])" "$d" "$(basename $d)"
  f=$d/optimized.hlo; [ -f "$f" ] && echo "   cond=$(grep -c ' conditional(' $f) while=$(grep -c ' while(' $f) cc=$(grep -c ' custom-call(' $f)"
done
