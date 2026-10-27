#!/usr/bin/env bash
# w44-rc v0.4.4 CPU step BEFORE the GPU session (after prepare.sh): OFF identity of the pinned source vs the v0.4.3
# bundle source with the official o5-port schema-2 probe (copy idtools/schema2, source lane/o5-port 446807db75; FINDINGS
# E222: release_identity.py is BLIND to constant values and child programs). Decides the pre-registered Swiss branch:
#   Swiss root (with canon): off_identity_compare exit 0 (IDENTICAL incl. raw_hlo_full_sha256 + canon_consts) -> branch (a)
#   (native + bytes identity vs the v0.4.3 bundle); exit 1 (DIFFERENT) -> branch (b) (Swiss 24 h D6 vs RD11/cpu);
#   exit 2 (INVALID_INPUT / BLIND / ERROR) -> no branch, STOP.
# Recorded beside it (expected: PROD changed by topo only): PROD --nested and PROD topo_shading=0 --nested (expected IDENTICAL).
# release_identity.py stays the second (canonical) convention: Swiss + PROD shas of the candidate (old relid recipe).
# Rules: base = <DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src, candidate = the manifest snapshot, both behind ONE alias
# path (this run's own symlink, re-pointed between the sides, never in parallel); JAX_PLATFORMS=cpu, PYTHONDONTWRITEBYTECODE=1,
# GPUWRF_JAX_CACHE=0, CUDA_VISIBLE_DEVICES=''; one RAM slot <= 12G per probe, cores 24,25,28, nice 19; a probe whose receipt
# already exists with verdict COMPLETE is not re-run (resume; base receipts may be copied in from an identical earlier run).
# usage: identity_pre.sh MANIFEST [--dry]     (--dry: every check + the topo0 case, prints the planned commands, no probe)
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=${1:?manifest} DRY=${2:-}
ID=<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools
PROBE_SHA=5bbd796f76d88ae671f90f6c62bb3299d2c2e9c64499346cdd8bfcf5fa887454
CMP_SHA=0e56d33cd40f651da0e013cfbf1daf48cfcac8f3abfe3f06de036a79498b3799
BASE_TREE=<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src
BASE_HEAD=888804310bb6bfc0237177c0d5e5452a29d5abf6
SWISS_CASE=<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu   # inputs byte-identical to F20 Swiss [M 2026-10-24]
PROD=<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725
SWISS_V043=b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a
PROD_V043=321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b
eval "$("$W44R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"], "TREE="+m["src_tree"])' "$MAN")"
OUT=$W44R_LANE/receipts/identity_${PIN:0:12}; mkdir -p "$OUT"; W44R_LOG=$OUT/identity_pre.log
ALIAS=$OUT/alias; T0=$OUT/prod_topo0; S=$W44R_SCRATCH/idpre_${PIN:0:12}
fail() { w44_log "REFUSED: $*"; exit 4; }
[[ $(w44_sha "$ID/off_identity_probe.py") == "$PROBE_SHA" && $(w44_sha "$ID/off_identity_compare.py") == "$CMP_SHA" ]] || fail "schema-2 probe/compare SHA != pinned"
[[ $(git -C "$BASE_TREE" rev-parse HEAD) == "$BASE_HEAD" && -z $(git -C "$BASE_TREE" status --porcelain -- src) ]] || fail "base tree $BASE_TREE not clean v0.4.3 $BASE_HEAD"
[[ $(git -C "$SNAP" rev-parse HEAD) == "$PIN" && $(git -C "$SNAP" rev-parse HEAD:src) == "$TREE" && -z $(git -C "$SNAP" status --porcelain -- src) ]] || fail "candidate $SNAP not clean at $PIN"
for f in wrfinput_d01 wrfbdy_d01 namelist.input; do
  cmp -s "$SWISS_CASE/$f" "<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input/$f" || fail "Swiss case $f != F20 Swiss input"
done
# PROD topo_shading=0: namelist-only edit (every domain value of topo_shading -> 0), all other inputs symlinked to PROD
if [ ! -e "$T0/namelist.input" ]; then
  mkdir -p "$T0"
  sed -E 's/^( topo_shading +=) *1, *1,/\1 0, 0,/' "$PROD/namelist.input" > "$T0/namelist.input"
  for f in "$PROD"/wrf*; do ln -sfn "$f" "$T0/$(basename "$f")"; done
fi
[ "$(diff "$PROD/namelist.input" "$T0/namelist.input" | grep -c '^[<>]')" = 2 ] && grep -q '^ topo_shading *= *0, *0,' "$T0/namelist.input" \
  || fail "prod_topo0 namelist is not exactly the one-line topo_shading edit"
for f in "$PROD"/wrf*; do [[ $(readlink "$T0/$(basename "$f")") == "$f" ]] || fail "prod_topo0 $(basename "$f") not a symlink to PROD"; done
w44_log "identity_pre pin $PIN snapshot $SNAP base $BASE_TREE alias $ALIAS dry='$DRY'"
probe() {  # probe SIDE LABEL CASE [args...]
  local side=$1 label=$2 case=$3; shift 3
  local out=$OUT/${side}_${label}.json
  if [ -s "$out" ] && grep -q '"verdict": "COMPLETE"' "$out"; then w44_log "skip $side $label (COMPLETE receipt exists)"; return 0; fi
  while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done
  w44_disk_ok 1 || { w44_log "REFUSED $side $label: <DATA_ROOT> below the floor"; return 95; }
  local cmd=(bash "$W44R_RAM_SLOT" --label w44-rc-idpre --max 12G --timeout 5400 -- env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0
             CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR="$S/triton" taskset -c 8,9,12 nice -n 19 timeout 2400
             "$W44R_PY" "$ID/off_identity_probe.py" --alias "$ALIAS" --case "$case" --out "$out" "$@")
  w44_log "PROBE $side $label alias -> $(readlink "$ALIAS"): ${cmd[*]}"
  [[ -n $DRY ]] && return 0
  mkdir -p "$S"; "${cmd[@]}" > "$OUT/${side}_${label}.log" 2>&1; local rc=$?
  w44_log "PROBE $side $label rc=$rc $(tail -1 "$OUT/${side}_${label}.log" | cut -c1-160)"; return $rc
}
for side in base cand; do
  tree=$BASE_TREE; [[ $side == cand ]] && tree=$SNAP
  ln -sfn "$tree" "$ALIAS"; [[ $(readlink "$ALIAS") == "$tree" ]] || fail "alias re-point"
  probe "$side" swiss "$SWISS_CASE"
  probe "$side" prod_nested "$PROD" --nested --no-canon
  probe "$side" prodtopo0_nested "$T0" --nested --no-canon
done
[[ -n $DRY ]] && { w44_log "DRY: checks passed, 6 probes + 3 compares + relid planned"; exit 0; }
declare -A CMP
for c in swiss prod_nested prodtopo0_nested; do
  "$W44R_PY" "$ID/off_identity_compare.py" "$OUT/base_$c.json" "$OUT/cand_$c.json" "$OUT/CMP_$c.json" > "$OUT/CMP_$c.txt" 2>&1
  CMP[$c]=$?; w44_log "COMPARE $c rc=${CMP[$c]} $(tail -1 "$OUT/CMP_$c.txt" | cut -c1-200)"
done
# second (canonical) convention: release_identity.py on the candidate (the v0.4.3 relid recipe)
cat > "$OUT/relid.sh" <<EOF
#!/usr/bin/env bash
cd "$SNAP" || exit 2
for arm in swiss prod; do
  while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done
  if [ \$arm = swiss ]; then extra=""; else extra=$PROD; fi
  /usr/bin/time -v env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 "$W44R_PY" proofs/s1_port/release_identity.py \
    "$SNAP" "$OUT/release_identity_\$arm.json" \$extra > "$OUT/relid_\$arm.log" 2>&1
  echo "\$arm rc=\$?" >> "$OUT/relid_rc.txt"
done
EOF
while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done
taskset -c 8,9,12 nice -n 19 bash "$W44R_RAM_SLOT" --label w44-rc-relid --max 12G --timeout 5400 -- bash "$OUT/relid.sh"
"$W44R_PY" -I "$W44R_TOOLS/followup/identity_branch.py" "$OUT" "$PIN" "${CMP[swiss]}" "${CMP[prod_nested]}" "${CMP[prodtopo0_nested]}" \
  "$SWISS_V043" "$PROD_V043" | tee -a "$W44R_LOG"
[ -d "$S" ] && rm -rf "${S:?}"
