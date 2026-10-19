#!/usr/bin/env bash
# w42-aotkey real-case key + bundle-integrity arms (CPU only, core 24, nice 19; runs inside ONE with_ram_slot scope).
# usage: arms.sh OUTDIR   (immutable once launched; tree pinned below)
set -uo pipefail
export LC_ALL=C
OUT=$1; L=<USER_HOME>/wrf_gpu2_lanes/w42-aotkey; T=$L/tools; W=<USER_HOME>/src/wrf_gpu2_wt/w42-aotkey
S=<DATA_ROOT>/wrf_gpu2/scratch/w42-aotkey; PY=<USER_HOME>/miniconda3/bin/python
WN3=$S/cases/wn3_0227; PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
FIX=64dfe6746f
test "$(git -C $W rev-parse HEAD:src)" = "$(git -C $W rev-parse $FIX:src)" || { echo "tree moved"; exit 2; }
test -z "$(git -C $W status --porcelain --untracked-files=all -- src)" || { echo "dirty src"; exit 2; }
mkdir -p $OUT $S/tmp
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/arms.log; }
quiet_wait() { while [ -e /tmp/wrf_gpu2_quiet ]; do log "QUIET wait ($(cat /tmp/wrf_gpu2_quiet))"; sleep 30; done; }
BASE=(env -i HOME=$HOME USER=$USER PATH=<USER_HOME>/miniconda3/bin:/usr/bin:/bin LANG=C.UTF-8 JAX_PLATFORMS=cpu
      PYTHONDONTWRITEBYTECODE=1 TMPDIR=$S/tmp)
RUN=(taskset -c 24 nice -n 19 /usr/bin/time -v)
# arm NAME TOOL(fixed|v041) CASE HOURS ENV(recorded|bundle:<dir>)
arm() {
  local name=$1 tool=$2 case=$3 hours=$4 envmode=$5 cmd pre="" rc
  if [ $tool = fixed ]; then cmd="$PY -m gpuwrf.cli"; else cmd="$PY $T/runner_v041.py"; fi
  if [[ $envmode == bundle:* ]]; then pre="source ${envmode#bundle:}/bundle.env && "; fi
  quiet_wait
  log "START $name tool=$tool case=$case hours=$hours env=$envmode"
  if [ -e /tmp/wrf_gpu2_quiet ]; then log "QUIET appeared, skip $name"; return 3; fi; "${BASE[@]}" PYTHONPATH=$W/src GPUWRF_JAX_CACHE=$([ $envmode = recorded ] && echo 0 || echo "") \
    "${RUN[@]}" bash -c "${pre}[ -n \"\$GPUWRF_JAX_CACHE\" ] || unset GPUWRF_JAX_CACHE; exec $cmd aot-key --input-dir $case --hours $hours --json $OUT/$name.json" \
    > $OUT/$name.log 2>&1; rc=$?
  log "END $name rc=$rc $(grep -E 'Elapsed|Maximum resident' $OUT/$name.log | tr -s ' \t' ' ' | tr '\n' ' ')"
  grep -E "cheap_key=|WARNING|UNAVAILABLE" $OUT/$name.log | tee -a $OUT/arms.log
  ls -A $S/tmp | sed "s/^/  leftover tmp: /" | tee -a $OUT/arms.log
  return $rc
}
bundle_state() { # dir label
  $PY $T/tree.py $1 $OUT/tree_$2.json | sed "s/^/  tree $2: /" | tee -a $OUT/arms.log
  env -i PATH=/usr/bin:/bin JAX_PLATFORMS=cpu $PY $W/scripts/release/compile_bundle.py check --cache $1/jax \
    --manifest $1/sealed/BUNDLE_MANIFEST.json --out $OUT/check_$2.json | sed "s/^/  check $2: /" | tee -a $OUT/arms.log
}
log "arms.sh tree $(git -C $W rev-parse HEAD) src $(git -C $W rev-parse HEAD:src)"
# A1 WN3 0227, fixed tool, sealed-bundle env without the keyed autotune var -> keys must equal the v0.4.1 records
rm -rf -- $S/b1; $PY $T/mkbundle.py $S/b1 $W/src | tee -a $OUT/arms.log; bundle_state $S/b1 b1_before
arm A1_wn3_fixed_bundle fixed $WN3 72 bundle:$S/b1; bundle_state $S/b1 b1_after
# A2/A3 WN3 0227, the incident's env shape (bundle.env incl. GPUWRF_XLA_AUTOTUNE_CACHE_DIR), fixed then v0.4.1 tool, SAME bundle path
rm -rf -- $S/b2; $PY $T/mkbundle.py $S/b2 $W/src --autotune | tee -a $OUT/arms.log; bundle_state $S/b2 b2_before
arm A2_wn3_fixed_incidentenv fixed $WN3 72 bundle:$S/b2; bundle_state $S/b2 b2_after_fixed
arm A3_wn3_v041_incidentenv v041 $WN3 72 bundle:$S/b2; bundle_state $S/b2 b2_after_v041
# A4/A5 PROD (2 domains): fixed in a sealed-bundle env vs v0.4.1 tool in the recorded env (GPUWRF_JAX_CACHE=0)
rm -rf -- $S/b4; $PY $T/mkbundle.py $S/b4 $W/src | tee -a $OUT/arms.log; bundle_state $S/b4 b4_before
arm A4_prod_fixed_bundle fixed $PROD 3 bundle:$S/b4; bundle_state $S/b4 b4_after
arm A5_prod_v041_recorded v041 $PROD 3 recorded
# release identities (tooling-only change: must stay EXACT)
for c in swiss prod; do
  quiet_wait; log "START identity $c"
  arg=""; [ $c = prod ] && arg=$PROD; [ $c = swiss ] && arg=<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu
  (cd $W && "${BASE[@]}" GPUWRF_JAX_CACHE=0 "${RUN[@]}" timeout 1800 $PY proofs/s1_port/release_identity.py $W $OUT/identity_$c.json $arg > $OUT/identity_$c.log 2>&1)
  log "END identity $c rc=$? sha=$($PY -c "import json;d=json.load(open('$OUT/identity_$c.json'));r=d['arms']['release'];print(r.get('sha'),r.get('case'))" 2>&1)"
done
log ALL_DONE
