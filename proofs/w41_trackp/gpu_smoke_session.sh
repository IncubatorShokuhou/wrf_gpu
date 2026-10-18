#!/usr/bin/env bash
# Track P smoke driver (holds NO RAM slot itself; manager rule 2026-10-17T10:41Z):
#   1. CPU prep in its own RAM slot (PROD basis + control files), released before any GPU wait
#   2. arms in with_gpu_lock sessions outside any RAM slot (ARM_GROUPS env, default one arm per session)
#   3. post-processing in a new RAM slot: E41 gate on both private caches (before cleanup; caches removed only on
#      a passing hard stack gate), smoke verdicts, then the cheap-key suite (it rewrites a source file, E125, so
#      it never runs while an arm imports this tree)
# usage: gpu_smoke_session.sh <src_root> <work_dir> <lane_cores>
# env: ARM_GROUPS (lock sessions), ARMS_ONLY=1 (skip step 3), POST_ONLY=1 (only step 3) — one RAM ticket per lane at a time
set -u
SRC=$(cd "$1" && pwd); W=$(cd "$2" && pwd); CORES=$3
L=$(dirname "$W"); MB=<USER_HOME>/src/wrf_gpu2/.agent/kernel/MAILBOX.md
LAST_START=$(date -u -d "2026-10-17T18:15:00Z" +%s)   # lease ends 18:45Z; no arm may start after 18:15Z
cd "$SRC"
# the MAIN copy of with_ram_slot.sh carries the current admission rule (manager 14:35Z)
slot() { local max=$1; shift; <USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w41-trackp --max "$max" --timeout 25000 -- "$@"; }
quiet_wait() { while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 60; done; }
CPUENV="env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=$SRC/src taskset -c $CORES nice -n 19"

quiet_wait
if [ "${POST_ONLY:-0}" != 1 ]; then
if [ ! -e "$W/controls/MANIFEST.json" ]; then
  slot 6G bash -c "cd $SRC
    [ -e $W/basis_swiss.npz ] || $CPUENV timeout 2400 python -m gpuwrf surface-params basis --input-dir $W/case/swiss \
        --output $W/basis_swiss.npz > $W/basis_swiss2.log 2>&1
    [ -e $W/basis_prod.npz ] || $CPUENV timeout 2400 python -m gpuwrf surface-params basis \
        --input-dir <DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725 --max-dom 2 --output $W/basis_prod.npz > $W/basis_prod.log 2>&1
    $CPUENV timeout 900 python proofs/w41_trackp/prep_smoke_controls.py $W proofs/w41_trackp/spec_T.json > $W/prep.log 2>&1"
fi
[ -e "$W/controls/MANIFEST.json" ] || { echo "control prep failed"; tail -n 20 "$W/prep.log" "$W/basis_prod.log"; exit 4; }

started=0
# ARM_GROUPS (not GROUPS: a read-only bash array of the user's group ids): space-separated lock sessions, each a comma list of arms (default: one arm per session)
for group in ${ARM_GROUPS:-S1 S2 S0 P0 P1 P2}; do
  quiet_wait
  [ "$(date -u +%s)" -lt "$LAST_START" ] || { echo "DEADLINE: arms $group not started"; break; }
  if [ $started = 0 ]; then echo "$(date -u +%FT%TZ) w41-trackp: GPU queue entered, Track P smoke sessions: ${ARM_GROUPS:-one arm each}" >> "$MB"; started=1; fi
  scripts/with_gpu_lock.sh --label w41-trackp --timeout 14400 -- bash -c 'rc=0; for a in ${1//,/ }; do
      proofs/w41_trackp/gpu_smoke.sh "$2" "$3" "$a" || rc=$?; done; exit $rc' _ "$group" "$SRC" "$W"
  echo "arm $group rc=$?"
done
if [ $started = 1 ]; then
  mins=$(python3 -c "import json;print(round(sum(json.loads(l).get('wall_s',0) for l in open('$W/arms.jsonl') if l.startswith('{\"arm\"'))/60,1))")
  echo "$(date -u +%FT%TZ) w41-trackp: GPU END Track P smoke, $mins GPU-min summed over arms ($W/arms.jsonl); E41 next in a CPU slot" >> "$MB"
fi

fi  # POST_ONLY
[ "${ARMS_ONLY:-0}" = 1 ] && { echo ARMS_DONE; exit 0; }
quiet_wait
slot 6G bash -c "cd $SRC
  for c in swiss prod; do
    [ -d $W/cache_\$c ] || continue
    $CPUENV timeout 1800 python scripts/wave_cubin_gate.py $W/cache_\$c $W/e41_\$c.json > $W/e41_\$c.log 2>&1; echo \"e41 \$c rc=\$?\"
    if python3 -c \"import json,sys; g=json.load(open('$W/e41_\$c.json')); sys.exit(0 if g.get('hard_stack_gate')=='PASS' else 1)\"; then
      rm -rf $W/cache_\$c; echo \"cache_\$c removed (hard stack gate PASS)\"
    else echo \"cache_\$c KEPT (hard stack gate not PASS)\"; fi
  done
  $CPUENV python proofs/w41_trackp/smoke_compare.py $W $W/smoke_verdicts.json > $W/smoke_compare.log 2>&1; echo \"compare rc=\$?\"
  $CPUENV timeout 3600 python -m pytest -q tests/test_aot_cheap_key.py --basetemp $L/pt_aotkey -p no:cacheprovider > $L/queue/aotkey.log 2>&1
  echo \"aotkey rc=\$? \$(tail -n 1 $L/queue/aotkey.log)\" >> $L/queue/summary.txt; rm -rf $L/pt_aotkey; git status --short src"
echo SESSION_DONE
