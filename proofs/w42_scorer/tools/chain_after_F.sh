#!/usr/bin/env bash
# After the F 72 h arm: (1) final-tree 24 h identity, (2) wn3_integrity consumer smoke old vs new, (3) A/B/A/B CPU time
# of the d02 24 h comparator (A = main d96c268a5e held handles, B = candidate). usage: chain_after_F.sh FPID
set -u
FPID=${1:?}; L=<USER_HOME>/wrf_gpu2_lanes/w42-scorer; S=<DATA_ROOT>/wrf_gpu2/scratch/w42-scorer/d0227_24h
REF=$L/ref_main_d96c268a5e; NEW=$L/cand_c98cffc56c; SLOT=<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh
while kill -0 "$FPID" 2>/dev/null; do sleep 10; done
run() {  # label cmd...
  local label=$1; shift
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 2; done
  if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; $SLOT --label w42-scorer --max 12G --timeout 14400 -- \
    env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19 python3 $L/tools/measure.py $L/$label.measure.json \
    $L/tools/pause_on_quiet.sh "$@" > $L/$label.log 2>&1
  echo "$(date -u +%FT%TZ) $label rc=$? $(grep -E '^\[measure\]' $L/$label.log)"
}
run final24p python3 $NEW/scripts/wn3_score.py --cpu-dir $S/cpu --gpu-dir $S/gpu --out $L/final24p --hours 24
python3 $L/tools/compare_outputs.py $L/ref24 $L/final24p --alias $REF=$NEW --out $L/identity_24h_final_c98cffc56c.json
mkdir -p $L/consumers
run integ_ref python3 $REF/scripts/wn3_integrity.py --cpu-dir $S/cpu --gpu-dir $S/gpu --out $L/consumers/integrity_ref.json
run integ_new python3 $NEW/scripts/wn3_integrity.py --cpu-dir $S/cpu --gpu-dir $S/gpu --out $L/consumers/integrity_new.json
cmp $L/consumers/integrity_ref.json $L/consumers/integrity_new.json && echo "wn3_integrity outputs IDENTICAL"
for rep in 1 2; do
  for arm in A B; do
    T=$REF; [ $arm = B ] && T=$NEW
    run ab_d02_${arm}$rep python3 $T/scripts/wn3_fast_compare.py --cpu-dir $S/cpu --gpu-dir $S/gpu --domain d02 \
      --init 2026-02-28T00:00:00Z --min-lead 0 --max-lead 24 --tolerance-json $T/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
      --progress 25 --out-json $L/ab/d02_${arm}$rep.json --out-md $L/ab/d02_${arm}$rep.md
  done
done
echo "$(date -u +%FT%TZ) chain done"
