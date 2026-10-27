#!/usr/bin/env bash
# w44-rc v0.4.4 CPU follow-up of the SHORT entry (after entry_rw_short.sh; identity_pre.sh and aotkey_cpu.sh done).
# usage: followup_short.sh MANIFEST SHORT_RUN_DIR      (wraps itself in ONE RAM slot <= 12G, cores 24,25,28, nice 19)
# Order: bytes (RW1 binding + new prefix checks + Swiss C3) -> Swiss branch (a: swiss_native C1/C2 | b: Swiss 24 h D6 vs
# RD11/cpu) -> short_verdict.py (RW1/RW2/RW4/icloud0/E41/compile) -> delete the receipted A2/A3/A4 wrfouts (content-
# verified); A1 + B9 Swiss wrfouts are deleted only after the Swiss branch verdict exists. E41 already ran in the entry.
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044/lib.sh
MAN=${1:?manifest} RUN=${2:?short run dir}
F=$W44R_TOOLS/followup
if [[ -z ${W44R_FU_INNER:-} ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; then echo "WAIT: QUIET or GPU window"; exit 3; fi
  exec env W44R_FU_INNER=1 taskset -c 8,9,12 nice -n 19 bash "$W44R_RAM_SLOT" --label w44-rc-fu-short --max 12G --timeout 3600 -- bash "$0" "$MAN" "$RUN"
fi
eval "$("$W44R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"], "P="+shlex.quote(m["bundles"]["prod"]["dir"]), "P0="+shlex.quote(m["old_bundle"]["dir"]))' "$MAN")"
OUT=$W44R_LANE/receipts/followup_short_${PIN:0:12}; mkdir -p "$OUT"; W44R_LOG=$OUT/followup.log
R=$RUN/runs; P0R=$P0/receipts
ID=$W44R_LANE/receipts/identity_${PIN:0:12}/IDENTITY_BRANCH.json
[ -s "$ID" ] || { w44_log "MISSING $ID: run identity_pre.sh first"; exit 4; }
BRANCH=$("$W44R_PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["swiss_branch"])' "$ID")
w44_log "START followup_short pin $PIN run $RUN swiss branch $BRANCH"
"$W44R_PY" -I "$F/bytes_check.py" "$OUT/BYTES.json" \
  "RW1 B1_wn3_0227_3h=full:$R/B1_wn3_0227_3h/wrfout_sha.txt:$R/A3_wn3_0227_3h/wrfout_sha.txt" \
  "RW1 B5_swiss_3h=full:$R/B5_swiss_3h/wrfout_sha.txt:$R/A1_swiss/wrfout_sha.txt" \
  "RW1 B6_prod_3h=full:$R/B6_prod_3h/wrfout_sha.txt:$R/A2_prod/wrfout_sha.txt" \
  "PREFIX B9_swiss_24h vs A1=prefix:$R/B9_swiss_24h/wrfout_sha.txt:$R/A1_swiss/wrfout_sha.txt" \
  "C3 A1_swiss=full:$R/A1_swiss/wrfout_sha.txt:$P0R/A1_swiss/wrfout_sha.txt" \
  "C3 B5_swiss_3h=full:$R/B5_swiss_3h/wrfout_sha.txt:$P0R/B5_swiss_3h/wrfout_sha.txt" | tee -a "$W44R_LOG"
EXTRA=()
if [[ $BRANCH == a_* ]]; then
  (cd "$SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 "$W44R_PY" "$F/swiss_native.py" "$SNAP" \
     "$R/B5_swiss_3h/run.log" "$P0R/B5_swiss_3h/run.log" "$OUT/SWISS_NATIVE.json" "$P/src" "$P0/src") 2>&1 | tail -2 | tee -a "$W44R_LOG"
  EXTRA=(--swiss-native "$OUT/SWISS_NATIVE.json")
elif [[ $BRANCH == b_* ]]; then
  w44_log "Swiss D6 24 h vs RD11/cpu start"
  (cd "$SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v -o "$OUT/d6_swiss.time" \
     "$W44R_PY" scripts/wn3_score.py --cpu-dir <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu --gpu-dir "$R/B9_swiss_24h/wrfout" \
     --out "$OUT/d6_swiss" --hours 24 --domains d01) > "$OUT/d6_swiss.log" 2>&1
  w44_log "Swiss D6 rc=$? $(tail -1 "$OUT/d6_swiss.log" | cut -c1-200)"
  EXTRA=(--swiss-d6 "$OUT/d6_swiss")
fi
"$W44R_PY" -I "$F/short_verdict.py" "$RUN" "$P" "$OUT/SHORT_VERDICT.json" --bytes "$OUT/BYTES.json" --identity "$ID" \
  --aotkey "$W44R_LANE/receipts/aotkey_${PIN:0:12}" --aotkey-x "$W44R_LANE/receipts/aotkey_x_${PIN:0:12}" "${EXTRA[@]}" | tee -a "$W44R_LOG"
for a in A2_prod A3_wn3_0227_3h A4_wn3_0227_icloud0_3h; do
  [ -d "$R/$a/wrfout" ] && "$W44R_PY" -I "$F/delete_listed.py" "$RUN" "$R/$a" | tee -a "$W44R_LOG"
done
if grep -q '"verdict": "\(IDENTICAL\|PASS\)"' <("$W44R_PY" -I -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["swiss_identity"]))' "$OUT/SHORT_VERDICT.json"); then
  for a in A1_swiss B9_swiss_24h; do [ -d "$R/$a/wrfout" ] && "$W44R_PY" -I "$F/delete_listed.py" "$RUN" "$R/$a" | tee -a "$W44R_LOG"; done
fi
w44_log "DONE followup_short; <DATA_ROOT> free $(w44_free_gib <DATA_ROOT>) GiB"
