#!/usr/bin/env bash
# w44-gint: the FROZEN teacher G-INT chain (tools/score_gint.sh = w43-teacher score_case_v2.sh lines 70-80) with the ALISIOS
# reference read IN PLACE through guarded sessions (tools/guarded_run.py; manager 15:54Z: no staging copies). Steps and tools are
# the frozen ones in the frozen order: wn3_integrity --all-frames per domain, merge, r4_eval --integrity, integ_classes.classify,
# annex_postpass. The ONLY difference to classify_all.sh: the reference-reading steps run in-process inside one GuardedRead session
# each (same interpreter, same arguments) instead of a plain subprocess; the merge and classify heredocs are copied verbatim.
# usage: score_gint_guarded.sh NAME ISSUE(YYYYMMDD) CAND_DIR OUT
set -uo pipefail
NAME=$1; ISSUE=$2; GPU=$(readlink -f "$3"); OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd); FZ=$(cd "$HERE/../../v043/w43-teacher/frozen" && pwd)
CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_${ISSUE}_18z_a1/run/run
ADM=$(awk '{sub(/^A=/,"",$1); print $1}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt)
ASHA=$(awk '{print $2}' <USER_HOME>/wrf_gpu2_lanes/w44-gint/ADMISSION.txt)
PY="taskset -c ${GINT_CPUS:-8,9,12,13} nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
RAM="<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w44-gint --max 12G --timeout ${RAM_WAIT:-7200} --"
G="$HERE/guarded_run.py $ADM $ASHA"
q(){ while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done; }
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 2; }
want="413299de211072b0a38c integrity_all/scripts/wn3_integrity.py
cbc95179c95dd08d53ad integrity_all/scripts/wn3_score.py
d3cfb72e21ef014adc41 gate/integ_classes.py
73deaf41563304b40b36 gate/r4_eval.py
3b03e7cb151bd75e24e2 integrity_all/annex_postpass.py"
while read -r h f; do [ "$(sha256sum "$FZ/$f" | cut -c1-20)" = "$h" ] || { echo "frozen tool drift $f"; exit 3; }; done <<< "$want"
I=$OUT/integ; mkdir -p "$I"; tag=$(date -u +%H%M%S)
for d in d01 d02 d03; do q; $RAM $PY $G "GI_${NAME}_${d}_$tag" "wg_${ISSUE}_18z_a1/run/run/wrfout_${d}_" -- \
  "$FZ/integrity_all/scripts/wn3_integrity.py" --cpu-dir "$CPU" --gpu-dir "$GPU" --out "$I/$d.json" --domains $d --all-frames > "$I/$d.log" 2>&1
  echo "integrity $d rc=$?"; done
nice -n 19 env JAX_PLATFORMS=cpu python3 - "$I" <<'Y'
import json, sys
from pathlib import Path
O = Path(sys.argv[1]); doms = {}
for d in ("d01", "d02", "d03"):
    doms[d] = json.load(open(O / f"{d}.json"))["domains"][d]
first = json.load(open(O / "d01.json")); first["domains"] = doms; first["pass"] = all(v["pass"] for v in doms.values())
(O / "integrity_all_72h.json").write_text(json.dumps(first, indent=1) + "\n")
Y
q; $RAM $PY $G "GI_${NAME}_r4_$tag" "wg_${ISSUE}_18z_a1/run/run/wrfout_" -- "$FZ/gate/r4_eval.py" --cpu-dir "$CPU" --gpu-dir "$GPU" \
  --out "$I/r4_watch_all_72h.json" --integrity "$I/integrity_all_72h.json" > "$I/r4_all.log" 2>&1; echo "r4 rc=$?"
nice -n 19 env JAX_PLATFORMS=cpu python3 - "$I" "$FZ/gate" > "$I/classify.log" 2>&1 <<'Y'
import json, sys
sys.path.insert(0, sys.argv[2])
from integ_classes import classify, compact
O = sys.argv[1]; v = classify(json.load(open(f"{O}/integrity_all_72h.json")), json.load(open(f"{O}/r4_watch_all_72h.json")))
json.dump(v, open(f"{O}/classes_all_72h.json", "w"), indent=1, default=str); print(compact(v))
Y
q; $RAM $PY $G "GI_${NAME}_annex_$tag" "wg_${ISSUE}_18z_a1/run/run/wrfout_" -- "$FZ/integrity_all/annex_postpass.py" \
  "$I/classes_all_72h.json" "$CPU" "$GPU" > "$I/annex_postpass.txt" 2> "$I/annex.err"; echo "annex rc=$?"
echo "$NAME $(tail -1 "$I/classify.log" | cut -c1-300)"; grep -v "^{" "$I/annex_postpass.txt"
