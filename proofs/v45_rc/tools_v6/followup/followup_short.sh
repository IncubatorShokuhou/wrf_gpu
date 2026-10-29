#!/usr/bin/env bash
# v45-rc v0.4.5 CPU follow-up of the SHORT entry (after entry_rw_short.sh; identity_pre.sh and aotkey_cpu.sh done).
# usage: followup_short.sh MANIFEST SHORT_RUN_DIR      (wraps itself in ONE RAM slot <= 8G, cores 12,28, nice 19)
# Order: bytes (RW1 binding + new prefix checks + Swiss C3) -> Swiss branch (a: swiss_native C1/C2 | b: Swiss 24 h D6 vs
# RD11/cpu) -> short_verdict.py (RW1/RW2/RW4/icloud0/E41/compile) -> delete the receipted A2/A3/A4 wrfouts (content-
# verified); A1 + B9 Swiss wrfouts are deleted only after the Swiss branch verdict exists. E41 already ran in the entry.
set -uo pipefail
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
source <USER_HOME>/wrf_gpu2_lanes/v45-rc/tools_v6/lib.sh
MAN=${1:?manifest} RUN=${2:?short run dir}
F=$W45R_TOOLS/followup
if [[ -z ${W45R_FU_INNER:-} ]]; then
  if [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; then echo "WAIT: QUIET or GPU window"; exit 3; fi
  exec env W45R_FU_INNER=1 taskset -c 12,28 nice -n 19 bash "$W45R_RAM_SLOT" --label v45-rc-fu-short --max 8G --timeout 3600 -- bash "$0" "$MAN" "$RUN"
fi
eval "$("$W45R_PY" -I -c 'import json,sys,shlex; m=json.load(open(sys.argv[1])); print("SNAP="+shlex.quote(m["snapshot"]), "PIN="+m["pin"], "P="+shlex.quote(m["bundles"]["prod"]["dir"]), "P0="+shlex.quote(m["old_bundle"]["dir"]))' "$MAN")"
OUT=$W45R_LANE/receipts/followup_short_${PIN:0:12}; mkdir -p "$OUT"; W45R_LOG=$OUT/followup.log
R=$RUN/runs; P0R=$P0/receipts
ID=$W45R_LANE/receipts/identity_${PIN:0:12}/IDENTITY_BRANCH.json
[ -s "$ID" ] || { w45_log "MISSING $ID: run identity_pre.sh first"; exit 4; }
BRANCH=$("$W45R_PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["swiss_branch"])' "$ID")
w45_log "START followup_short pin $PIN run $RUN swiss branch $BRANCH"
"$W45R_PY" -I "$F/bytes_check.py" "$OUT/BYTES.json" \
  "RW1 B1_wn3_0227_3h=full:$R/B1_wn3_0227_3h/wrfout_sha.txt:$R/A3_wn3_0227_3h/wrfout_sha.txt" \
  "RW1 B5_swiss_3h=full:$R/B5_swiss_3h/wrfout_sha.txt:$R/A1_swiss/wrfout_sha.txt" \
  "RW1 B6_prod_3h=full:$R/B6_prod_3h/wrfout_sha.txt:$R/A2_prod/wrfout_sha.txt" \
  "PREFIX B9_swiss_24h vs A1=prefix:$R/B9_swiss_24h/wrfout_sha.txt:$R/A1_swiss/wrfout_sha.txt" \
  "C3 A1_swiss=full:$R/A1_swiss/wrfout_sha.txt:$P0R/A1_swiss/wrfout_sha.txt" \
  "C3 B5_swiss_3h=full:$R/B5_swiss_3h/wrfout_sha.txt:$P0R/B5_swiss_3h/wrfout_sha.txt" | tee -a "$W45R_LOG"
EXTRA=()
if [[ $BRANCH == a_* ]]; then
  (cd "$SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 "$W45R_PY" "$F/swiss_native.py" "$SNAP" \
     "$R/B5_swiss_3h/run.log" "$P0R/B5_swiss_3h/run.log" "$OUT/SWISS_NATIVE.json" "$P/src" "$P0/src") 2>&1 | tail -2 | tee -a "$W45R_LOG"
  EXTRA=(--swiss-native "$OUT/SWISS_NATIVE.json")
elif [[ $BRANCH == b_* ]]; then
  w45_log "Swiss D6 24 h vs RD11/cpu start"
  (cd "$SNAP" && env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v -o "$OUT/d6_swiss.time" \
     "$W45R_PY" scripts/wn3_score.py --cpu-dir <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu --gpu-dir "$R/B9_swiss_24h/wrfout" \
     --out "$OUT/d6_swiss" --hours 24 --domains d01) > "$OUT/d6_swiss.log" 2>&1
  w45_log "Swiss D6 rc=$? $(tail -1 "$OUT/d6_swiss.log" | cut -c1-200)"
  EXTRA=(--swiss-d6 "$OUT/d6_swiss")
fi
"$W45R_PY" -I "$F/short_verdict.py" "$RUN" "$P" "$OUT/SHORT_VERDICT.json" --bytes "$OUT/BYTES.json" --identity "$ID" \
  --aotkey "$W45R_LANE/receipts/aotkey_${PIN:0:12}" --aotkey-x "$W45R_LANE/receipts/aotkey_x_${PIN:0:12}" "${EXTRA[@]}" | tee -a "$W45R_LOG"
w45_log "DONE followup_short; retained raw outputs until consolidated verdict/review"
