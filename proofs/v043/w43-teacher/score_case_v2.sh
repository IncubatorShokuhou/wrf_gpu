#!/usr/bin/env bash
# v2 (2026-10-21 tool-error fix, E163: new file, score_case.sh untouched): classify_all.sh's integ_classes import root is
# redirected too (the original <USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate no longer holds integ_classes.py -> ModuleNotFoundError on 0227);
# a redirected copy that still names a non-frozen root is refused; a tool error gives INTEGRITY pass=null (INCOMPLETE, never FAIL);
# INTEG_ONLY=1 reruns only the G-INT part + verdict on an existing OUT (old integ/ and VERDICT.json kept as *.toolerror).
# Score ONE case under TEACHER_PREREG v2 (CPU only): preconditions (219 exact τ0-72 names per side, sha256 lists), frozen D6 scorer
# 72 h, frozen all-frame G-INT (integrity copy --all-frames, classify_all with I/O roots redirected, annex post-pass), verdict.py.
# usage: score_case.sh NAME CPU_DIR GPU_DIR OUT [C_DIR|-]     (C_DIR = wn3_score output of twin vs reference, or '-')
# For the twin itself (C production): score_case.sh NAME CPU_DIR TWIN_DIR OUT - with SCORE_ONLY=1 (D6 rows + preconditions only).
set -uo pipefail
NAME=$1; CPU=$(readlink -f "$2"); GPU=$(readlink -f "$3"); OUT=$4; CDIR=${5:--}
HERE=$(cd "$(dirname "$0")" && pwd); MAIN=<USER_HOME>/src/wrf_gpu2; FZ=$HERE/frozen
PY="taskset -c 8,9,12 nice -n 19 env JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python3"
RAM="$MAIN/scripts/with_ram_slot.sh --label w43-teacher-score --max 12G --timeout ${RAM_WAIT:-7200} --"
q(){ while [ -e /tmp/wrf_gpu2_quiet ]; do sleep 30; done; }
if [ "${INTEG_ONLY:-0}" = 1 ]; then
  [ -d "$OUT/x" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$OUT/PRECONDITIONS.json" \
    || { echo "INTEG_ONLY needs $OUT with x/ and passing PRECONDITIONS.json"; exit 2; }
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  [ -e "$OUT/integ" ] && mv "$OUT/integ" "$OUT/integ.toolerror_$ts"
  [ -e "$OUT/VERDICT.json" ] && mv "$OUT/VERDICT.json" "$OUT/VERDICT.toolerror_$ts.json"
else
[ -e "$OUT" ] && { echo "refusing existing $OUT"; exit 2; }
mkdir -p "$OUT"
for d in "$CPU" "$GPU"; do case "$d" in <DATA_ROOT>/alisios/*|<DATA_ROOT>/server/work/*) echo "ALISIOS root $d: stage first"; exit 2;; esac; done
[ "$(sha256sum "$MAIN/scripts/wn3_score.py" | cut -c1-8)" = 7a5b51d6 ] && [ "$(sha256sum "$MAIN/scripts/wn3_fast_compare.py" | cut -c1-8)" = 905fc13b ] \
  && [ "$(sha256sum "$MAIN/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json" | cut -c1-8)" = 9df4fab8 ] || { echo "scorer sha drift"; exit 3; }
# preconditions: exact names tau0..72 on both sides, Times inside each file, sha lists
q; $PY - "$CPU" "$GPU" "$OUT/PRECONDITIONS.json" <<'EOF' || { echo "PRECONDITIONS FAIL"; exit 4; }
import hashlib, json, sys
from datetime import datetime, timedelta
from pathlib import Path
from netCDF4 import Dataset
cpu, gpu, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def names(d): return sorted(p.name for p in d.glob("wrfout_d0*") if not p.name.endswith(".tmp"))
cn, gn = names(cpu), names(gpu)
t0 = min(datetime.strptime(n[11:30], "%Y-%m-%d_%H:%M:%S") for n in cn if n.startswith("wrfout_d01"))
# native WRF history alarm names (d01 dt 54 s: h1 +18 s, h2 +36 s, ...; w43-twin0227 FINDING): lead h <-> the CPU file whose stamp is
# within one root step after t0 + h; GPU names must equal the CPU names exactly
want = []
for k in (1, 2, 3):
    dn = sorted(n for n in cn if n.startswith(f"wrfout_d0{k}_"))
    for h in range(73):
        hit = [n for n in dn if 0 <= (datetime.strptime(n[11:30], "%Y-%m-%d_%H:%M:%S") - t0 - timedelta(hours=h)).total_seconds() < 54]
        want.append(hit[0] if len(hit) == 1 else f"MISSING_d0{k}_h{h}")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
res = {"t0": t0.isoformat(), "cpu_exact": sorted(cn) == sorted(want), "gpu_exact": sorted(gn) == sorted(want), "times_ok": True,
       "cpu_sha": {}, "gpu_sha": {}}
for side, d in (("cpu", cpu), ("gpu", gpu)):
    for n in want:
        p = d / n
        if not p.exists(): continue
        with Dataset(p) as ds:
            t = [b"".join(x).decode() for x in ds["Times"][:]]
        res["times_ok"] &= t == [n[11:30]]
        res[f"{side}_sha"][n] = sha(p)
res["pass"] = res["cpu_exact"] and res["gpu_exact"] and res["times_ok"]
out.write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps({k: res[k] for k in ("cpu_exact", "gpu_exact", "times_ok", "pass")}))
sys.exit(0 if res["pass"] else 1)
EOF
q; (cd "$MAIN" && $RAM $PY scripts/wn3_score.py --cpu-dir "$CPU" --gpu-dir "$GPU" --out "$OUT/x" --hours 72) > "$OUT/score.log" 2>&1
echo "D6 rc=$? $(tail -1 "$OUT/score.log" | cut -c1-300)"
[ "${SCORE_ONLY:-0}" = 1 ] && exit 0
fi
# frozen all-frame G-INT; classify_all.sh executed as a copy with ONLY its two I/O roots substituted (diff recorded)
I=$OUT/integ; mkdir -p "$I"
for d in d01 d02 d03; do q; $RAM $PY "$FZ/integrity_all/scripts/wn3_integrity.py" --cpu-dir "$CPU" --gpu-dir "$GPU" --out "$I/$d.json" \
  --domains $d --all-frames > "$I/$d.log" 2>&1; done
sed -e "s#G=<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate;#G=$FZ/gate;#" -e "s#CPU=<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_\$base/run/run#CPU=$CPU#" \
  -e "s#sys.path.insert(0, \"<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate\")#sys.path.insert(0, \"$FZ/gate\")#" \
  "$FZ/integrity_all/classify_all.sh" > "$I/classify_all.redirected.sh"
grep -n "<USER_HOME>/wrf_gpu2_lanes/wn3\|<DATA_ROOT>/server" "$I/classify_all.redirected.sh" && { echo "redirected classify still names a non-frozen root"; exit 5; }
diff "$FZ/integrity_all/classify_all.sh" "$I/classify_all.redirected.sh" > "$I/classify_all.diff"
q; taskset -c 8,9,12 bash "$I/classify_all.redirected.sh" "$NAME" "$GPU" "$I" > "$I/classify.log" 2>&1
q; $PY "$FZ/integrity_all/annex_postpass.py" "$I/classes_all_72h.json" "$CPU" "$GPU" > "$I/annex_postpass.txt" 2>&1
$PY - "$I" <<'EOF'
import json, re, sys
from pathlib import Path
I = Path(sys.argv[1])
try:
    c = json.loads((I / "classes_all_72h.json").read_text())
except Exception as e:
    c = {"verdict": f"ERROR {e}"}
ann = [l for l in (I / "annex_postpass.txt").read_text().splitlines() if l.startswith("ANNEX ")] if (I / "annex_postpass.txt").exists() else []
annex_pass = bool(ann) and ann[-1].startswith("ANNEX PASS")
reg = c.get("verdict")
ok = reg in ("PASS", "PASS-with-registered-exceptions") or (reg == "FAIL-new-class" and annex_pass)
if reg not in ("PASS", "PASS-with-registered-exceptions", "FAIL-new-class"):
    ok = None   # tool error / unknown registry verdict -> INCOMPLETE (PREREG section 1), never FAIL
(I / "INTEGRITY.json").write_text(json.dumps({"pass": ok, "registry_verdict": reg, "annex": ann[-1] if ann else None,
                                              "new": c.get("new")}, indent=1) + "\n")
print("INTEGRITY", ok, reg, ann[-1] if ann else "no annex line")
EOF
$PY "$HERE/verdict.py" "$NAME" "$OUT/x" "$CDIR" "$I/INTEGRITY.json" "$OUT/VERDICT.json"
