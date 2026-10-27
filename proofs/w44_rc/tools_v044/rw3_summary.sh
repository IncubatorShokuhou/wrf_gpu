#!/usr/bin/env bash
# w44-rc v0.4.4: RW3T post (CPU, file reads only, seconds) after the RW3 lock session: disk summary, pre-registered
# acceptance, per-arm anatomy (24 h and 72 h), timing verdict. The 72 h D6 vs the original CPU-WRF is NOT run here
# (heavy; followup_rw3.sh with the staged CPU-WRF frames).
# env from the entry: W44R_RUN W44R_RECEIPTS
set -uo pipefail
source <DATA_ROOT>/wrf_gpu2/scratch/w44-rc/tools_v044_v5/lib.sh
: "${W44R_RUN:?}" "${W44R_RECEIPTS:?}"
export JAX_PLATFORMS=cpu
A=$W44R_RUN; O=$W44R_RECEIPTS/rw3t; mkdir -p "$O"
W44R_LOG=$A/post.log
PRE=$(dirname "${W44R_MANIFEST_PATH:?}")/PREREG.json
"$W44R_PY" -I "$W44R_TOOLS/disk_watch.py" summarize "$A/disk.jsonl" "$A" "$O/RW3T_DISK.json" > "$O/rw3t_disk.log" 2>&1
DISK_RC=$?
"$W44R_PY" -I "$W44R_TOOLS/accept.py" "$A" "$O/RW3T_ACCEPT.json" "$O/RW3T_DISK.json" > "$O/rw3t_accept.log" 2>&1
ACCEPT_RC=$?
w44_log "RW3T accept rc=$ACCEPT_RC $(tail -1 "$O/rw3t_accept.log" | cut -c1-200)"
N24=$(cd "$A/runs" && ls -d N[0-9] O[0-9] 2>/dev/null | tr '\n' ' ')
"$W44R_PY" -I "$W44R_TOOLS/attrib_detail.py" "$A" "$O/RW3T_DETAIL24.json" ${N24:-N1 O1} > "$O/rw3t_detail24.log" 2>&1
DETAIL24_RC=$?
"$W44R_PY" -I "$W44R_TOOLS/attrib_detail.py" "$A" "$O/RW3T_DETAIL72.json" N72 O72 > "$O/rw3t_detail72.log" 2>&1
DETAIL72_RC=$?
echo '{"note": "bytes: RW1 N72 vs A3 first 3 h is checked by followup_short/followup_rw3"}' > "$O/RW3T_BYTES.json"
"$W44R_PY" -I "$W44R_TOOLS/rw3t_verdict.py" "$O/RW3T_ACCEPT.json" "$O/RW3T_DETAIL24.json" "$O/RW3T_DETAIL72.json" "$O/RW3T_BYTES.json" \
  "$O/RW3T_VERDICT.json" "$PRE" "$O/RW3T_DISK.json" > "$O/rw3t_verdict.log" 2>&1
rc=$?
w44_log "RW3T verdict rc=$rc $(grep -m1 '"verdict"' "$O/rw3t_verdict.log" | cut -c1-120)"
for f in session.log core_gate.tsv HOST_LOAD.txt; do [ -s "$A/$f" ] && cp "$A/$f" "$O/"; done
"$W44R_PY" -I - "$O/RW3T_CHILDREN.json" "$DISK_RC" "$ACCEPT_RC" "$DETAIL24_RC" "$DETAIL72_RC" "$rc" <<'CHILDREN'
import json,sys
names=['disk','accept','detail24','detail72','verdict']
json.dump(dict(zip(names,map(int,sys.argv[2:]))),open(sys.argv[1],'w'),indent=2)
CHILDREN
for crc in "$DISK_RC" "$ACCEPT_RC" "$DETAIL24_RC" "$DETAIL72_RC" "$rc"; do
  (( crc == 0 )) || exit "$crc"
done
exit 0
