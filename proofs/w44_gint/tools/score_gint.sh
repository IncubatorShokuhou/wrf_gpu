#!/usr/bin/env bash
# w44-gint: the FROZEN teacher G-INT chain (score_case_v2.sh lines 70-80 of w43-teacher, unchanged tools from
# proofs/v043/w43-teacher/frozen, sha-checked against REGISTRY_ALLFRAMES.md) for ONE pair of wrfout dirs:
#   "CPU" = reference side, "GPU" = candidate side (for the CPU-vs-CPU control: CPU = original ref, GPU = CPU twin;
#   for GPU members: CPU = original ref, GPU = member). CPU only. No D6.
# usage: score_gint.sh NAME REF_DIR CAND_DIR OUT
set -uo pipefail
NAME=$1; CPU=$(readlink -f "$2"); GPU=$(readlink -f "$3"); OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd); FZ=$(cd "$HERE/../../v043/w43-teacher/frozen" && pwd)
PY="taskset -c ${GINT_CPUS:-8,9,12,13} nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
RAM="<USER_HOME>/src/wrf_gpu2/scripts/with_ram_slot.sh --label w44-gint --max 12G --timeout ${RAM_WAIT:-7200} --"
q(){ while [ -e /tmp/wrf_gpu2_quiet ] || [ -e /tmp/wrf_gpu2_gpu_window ]; do sleep 30; done; }
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 2; }
FREE_G=$(df --output=avail -B1G <DATA_ROOT> | tail -1); (( FREE_G >= ${DISK_FLOOR_G:-365} )) || { echo "refuse: <DATA_ROOT> free ${FREE_G} GiB < ${DISK_FLOOR_G:-365}"; exit 4; }
want="413299de211072b0a38c integrity_all/scripts/wn3_integrity.py
cbc95179c95dd08d53ad integrity_all/scripts/wn3_score.py
d3cfb72e21ef014adc41 gate/integ_classes.py
73deaf41563304b40b36 gate/r4_eval.py
738be63ce5bdeb7222a6 integrity_all/classify_all.sh
3b03e7cb151bd75e24e2 integrity_all/annex_postpass.py"
while read -r h f; do [ "$(sha256sum "$FZ/$f" | cut -c1-20)" = "$h" ] || { echo "frozen tool drift $f"; exit 3; }; done <<< "$want"
I=$OUT/integ; mkdir -p "$I"
for d in d01 d02 d03; do q; $RAM $PY "$FZ/integrity_all/scripts/wn3_integrity.py" --cpu-dir "$CPU" --gpu-dir "$GPU" --out "$I/$d.json" \
  --domains $d --all-frames > "$I/$d.log" 2>&1; done
sed -e "s#G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate;#G=$FZ/gate;#" -e "s#CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_\$base/run/run#CPU=$CPU#" \
  -e "s#sys.path.insert(0, \"<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate\")#sys.path.insert(0, \"$FZ/gate\")#" \
  "$FZ/integrity_all/classify_all.sh" > "$I/classify_all.redirected.sh"
grep -n "<USER_HOME>/wrf_gpu2_lanes/wn3\|<DATA_ROOT>/server" "$I/classify_all.redirected.sh" && { echo "redirected classify still names a non-frozen root"; exit 5; }
diff "$FZ/integrity_all/classify_all.sh" "$I/classify_all.redirected.sh" > "$I/classify_all.diff"
q; taskset -c ${GINT_CPUS:-8,9,12,13} bash "$I/classify_all.redirected.sh" "$NAME" "$GPU" "$I" > "$I/classify.log" 2>&1
q; $PY "$FZ/integrity_all/annex_postpass.py" "$I/classes_all_72h.json" "$CPU" "$GPU" > "$I/annex_postpass.txt" 2>&1
echo "$NAME $(tail -1 "$I/classify.log" | cut -c1-200)"; cat "$I/annex_postpass.txt"
